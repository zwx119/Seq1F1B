# Copyright (c) 2024, Seq1F1B-DeltaNet Contributors.
# Adapted from flash-linear-attention (fla) library by Songlin Yang, Yu Zhang.
# Integrated into Megatron-LM Seq1F1B pipeline parallelism framework.

"""
DeltaNet Linear Attention for Megatron Seq1F1B.

This module implements DeltaNet as a drop-in replacement for softmax attention
in the Megatron transformer, with full support for Seq1F1B sequence-level
pipeline parallelism.

Key design decisions:
  - DeltaNet uses a recurrent hidden state S ∈ R^{d_k × d_v} instead of KV cache.
  - For Seq1F1B (pipe_sp_splits > 1), when micro_sp_idx == 0 we start with fresh
    state; for subsequent splits we pass the recurrent state from the previous
    split as initial_state to chunk_delta_rule.
  - Short convolutions (from fla) replace positional encoding (RoPE is not used).
  - Tensor parallelism is handled by splitting heads across TP ranks, similar
    to how Megatron splits attention heads.
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron import get_args, core
from megatron.core import mpu, tensor_parallel
from megatron.model.module import MegatronModule

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    warnings.warn(
        "flash-linear-attention (fla) is not installed. "
        "DeltaNet attention will not be available. "
        "Install with: pip install flash-linear-attention"
    )

try:
    from fla.modules import RMSNorm as FLARMSNorm
    from fla.modules import ShortConvolution
    HAS_FLA_MODULES = True
except ImportError:
    HAS_FLA_MODULES = False


class DeltaNetAttention(MegatronModule):
    """
    DeltaNet linear attention layer for Megatron, supporting Seq1F1B.

    This replaces ParallelAttention in the transformer layer. It takes input
    of shape [s, b, h] (Megatron convention: seq_len first) and returns
    output of the same shape.

    For Seq1F1B sequence splitting:
      - Each DeltaNet layer maintains a per-layer recurrent state dict.
      - When micro_sp_idx == 0: state is cleared (new microbatch).
      - When micro_sp_idx > 0: previous recurrent state is passed as
        initial_state to chunk_delta_rule, enabling correct state continuity
        across sequence splits.

    Tensor parallelism:
      - Q, K, V projections use ColumnParallelLinear (split output across TP ranks).
      - Output projection uses RowParallelLinear (gather across TP ranks).
      - num_heads is divided by TP world size.
      - Beta projection is NOT split (each TP rank computes all heads' beta,
        then indexes its own partition). This is because beta has shape [b, s, H]
        which is small and doesn't benefit from splitting.

    Args:
        config: Megatron transformer config
        layer_number: Layer index (1-based)
    """

    def __init__(self, config, layer_number):
        super(DeltaNetAttention, self).__init__()
        assert HAS_FLA, "flash-linear-attention (fla) is required for DeltaNet"
        assert HAS_FLA_MODULES, "fla.modules is required for DeltaNet"
        assert rearrange is not None, "einops is required for DeltaNet"

        args = get_args()
        self.layer_number = max(1, layer_number)
        self.params_dtype = config.params_dtype
        self.sequence_parallel = config.sequence_parallel

        # Head configuration with tensor parallelism
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_attention_heads_per_partition = core.utils.divide(
            config.num_attention_heads, world_size
        )
        self.hidden_size_per_attention_head = core.utils.divide(
            config.hidden_size, config.num_attention_heads
        )
        self.head_dim = self.hidden_size_per_attention_head

        # Key/value dimensions (expand_k=1.0, expand_v=1.0 by default)
        # Total key_dim = hidden_size, split across TP ranks by heads
        self.key_dim_per_partition = self.num_attention_heads_per_partition * self.head_dim
        self.value_dim_per_partition = self.num_attention_heads_per_partition * self.head_dim

        # DeltaNet-specific config from args
        self.use_short_conv = getattr(args, 'deltanet_use_short_conv', True)
        self.conv_size = getattr(args, 'deltanet_conv_size', 4)
        self.use_beta = getattr(args, 'deltanet_use_beta', True)
        self.use_gate = getattr(args, 'deltanet_use_output_gate', False)
        self.qk_activation = getattr(args, 'deltanet_qk_activation', 'silu')
        self.qk_norm = getattr(args, 'deltanet_qk_norm', 'l2')
        self.deltanet_mode = getattr(args, 'deltanet_mode', 'chunk')

        # Q, K, V projections with tensor parallelism
        # ColumnParallelLinear splits the output dimension across TP ranks
        self.q_proj = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,  # = num_heads * head_dim
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,  # Keep split across TP ranks
        )
        self.k_proj = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )
        self.v_proj = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )

        # Beta projection: projects to num_heads_per_partition
        # We use a simple nn.Linear here because beta is small (output = num_heads)
        # and we only need our partition's heads.
        if self.use_beta:
            self.b_proj = nn.Linear(
                self.hidden_size,
                self.num_attention_heads_per_partition,
                bias=False,
            )

        # Short convolutions (per TP partition)
        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim_per_partition,
                kernel_size=self.conv_size,
                bias=False,
                activation='silu' if self.qk_activation == 'silu' else None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim_per_partition,
                kernel_size=self.conv_size,
                bias=False,
                activation='silu' if self.qk_activation == 'silu' else None,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim_per_partition,
                kernel_size=self.conv_size,
                bias=False,
                activation='silu',
            )

        # Output normalization
        if self.use_gate:
            self.g_proj = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=False,
            )
            # FLA's FusedRMSNormGated for gated output
            from fla.modules import FusedRMSNormGated
            self.o_norm = FusedRMSNormGated(self.head_dim, eps=1e-5)
        else:
            self.o_norm = FLARMSNorm(self.head_dim, eps=1e-5, dtype=torch.float32)

        # Output projection with tensor parallelism
        # RowParallelLinear gathers from TP ranks
        self.o_proj = tensor_parallel.RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=args.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
        )

        # === Seq1F1B state management ===
        # recurrent_state: the DeltaNet hidden state S ∈ R^{H, d_k, d_v}
        # conv_state_*: short convolution cache for each of q, k, v
        self.recurrent_state = None
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None

    def _clear_states(self):
        """Clear all cached states. Called when micro_sp_idx == 0."""
        self.recurrent_state = None
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None

    def forward(
        self,
        hidden_states,      # [s, b, h] (Megatron convention)
        attention_mask,      # unused for DeltaNet (causal by default)
        encoder_output=None, # unused
        inference_params=None, # unused for now
        rotary_pos_emb=None, # unused (DeltaNet doesn't use RoPE)
        micro_sp_idx=None,   # Seq1F1B: which sequence split (0, 1, ..., pipe_sp_splits-1)
    ):
        """
        Forward pass of DeltaNet attention.

        Input:  [s, b, h]  (seq_len, batch, hidden)
        Output: ([s, b, h], bias)

        For Seq1F1B:
          - micro_sp_idx == 0: clear states, start fresh
          - micro_sp_idx > 0: use cached recurrent_state and conv_state
        """
        args = get_args()

        # Handle micro_sp_idx
        if micro_sp_idx is not None:
            if torch.is_tensor(micro_sp_idx):
                micro_sp_idx = micro_sp_idx.item()
            if micro_sp_idx == 0:
                self._clear_states()

        # Megatron uses [s, b, h], but DeltaNet/fla expects [b, s, h]
        # Transpose: [s, b, h] -> [b, s, h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        batch_size, seq_len, _ = hidden_states.shape

        # ============ Projections ============
        # ColumnParallelLinear returns (output, bias)
        q, _ = self.q_proj(hidden_states)  # [b, s, key_dim_per_partition]
        k, _ = self.k_proj(hidden_states)  # [b, s, key_dim_per_partition]
        v, _ = self.v_proj(hidden_states)  # [b, s, value_dim_per_partition]

        # ============ Short Convolutions ============
        if self.use_short_conv:
            # Determine if we need to use conv state (Seq1F1B continuation)
            use_conv_cache = (args.pipe_sp_splits > 1 and micro_sp_idx is not None
                              and micro_sp_idx > 0)

            q, conv_state_q = self.q_conv1d(
                x=q,
                cache=self.conv_state_q if use_conv_cache else None,
                output_final_state=(args.pipe_sp_splits > 1),
            )
            k, conv_state_k = self.k_conv1d(
                x=k,
                cache=self.conv_state_k if use_conv_cache else None,
                output_final_state=(args.pipe_sp_splits > 1),
            )
            v, conv_state_v = self.v_conv1d(
                x=v,
                cache=self.conv_state_v if use_conv_cache else None,
                output_final_state=(args.pipe_sp_splits > 1),
            )
            # Detach conv states to prevent backward through previous span's graph
            self.conv_state_q = conv_state_q.detach() if conv_state_q is not None else None
            self.conv_state_k = conv_state_k.detach() if conv_state_k is not None else None
            self.conv_state_v = conv_state_v.detach() if conv_state_v is not None else None
        else:
            # Without short conv, apply activation directly
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            v = F.silu(v)

        # ============ Reshape to multi-head ============
        # [b, s, num_heads_per_partition * head_dim] -> [b, s, num_heads_per_partition, head_dim]
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
        k = rearrange(k, 'b s (h d) -> b s h d', d=self.head_dim)
        v = rearrange(v, 'b s (h d) -> b s h d', d=self.head_dim)

        # ============ QK activation (if not handled by conv) ============
        if self.use_short_conv and self.qk_activation != 'silu':
            # Short conv already applied silu for 'silu' mode
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q = (F.elu(q, 1., False) + 1.).to(q)
                k = (F.elu(k, 1., False) + 1.).to(k)

        # ============ Beta ============
        if self.use_beta:
            # hidden_states is [b, s, h], b_proj outputs [b, s, num_heads_per_partition]
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones(
                batch_size, seq_len, self.num_attention_heads_per_partition,
                device=hidden_states.device, dtype=hidden_states.dtype
            )

        # ============ DeltaNet core computation ============
        # Choose mode based on sequence length
        mode = 'fused_recurrent' if seq_len <= 64 else self.deltanet_mode

        # For Seq1F1B: pass recurrent state from previous split
        initial_state = self.recurrent_state

        # Determine if we need to output final state (for next split or not)
        output_final_state = (args.pipe_sp_splits > 1)

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q, k=k, v=v, beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=(self.qk_norm == 'l2'),
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_delta_rule(
                q=q, k=k, v=v, beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=(self.qk_norm == 'l2'),
            )
        else:
            raise NotImplementedError(f"DeltaNet mode `{mode}` not supported.")

        # Cache state for next Seq1F1B split
        if output_final_state and recurrent_state is not None:
            self.recurrent_state = recurrent_state.detach()

        # ============ Output normalization & gate ============
        if self.use_gate:
            g, _ = self.g_proj(hidden_states)
            g = rearrange(g, 'b s (h d) -> b s h d', d=self.head_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        # ============ Reshape and output projection ============
        # [b, s, num_heads_per_partition, head_dim] -> [b, s, hidden_per_partition]
        o = rearrange(o, 'b s h d -> b s (h d)')

        # RowParallelLinear: gathers across TP ranks, returns (output, bias)
        output, bias = self.o_proj(o)

        # Transpose back to Megatron convention: [b, s, h] -> [s, b, h]
        output = output.transpose(0, 1).contiguous()
        if bias is not None:
            # bias is [h], needs to be compatible with [s, b, h]
            pass

        return output, bias
