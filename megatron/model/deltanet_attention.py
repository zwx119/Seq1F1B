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
    from fla.ops.delta_rule.chunk import chunk_delta_rule_fwd, chunk_delta_rule_bwd
    from fla.modules.conv.triton.ops import causal_conv1d_fwd, causal_conv1d_bwd
    from fla.modules.l2norm import l2norm_fwd, l2norm_bwd
    from fla.ops.utils.index import prepare_chunk_indices
    from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
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


# ============================================================================
# ShortConvChunkFunc: depthwise causal conv1d with Seq1F1B cross-chunk
# gradient relay (same pattern as DeltaNetChunkFunc).
#
# Short convolution cache semantics (from fla.modules.conv.short_conv):
#   - `cache` fed into ShortConvolution.forward is the last (W-1) input
#     tokens of the previous chunk, used as prefix so that the first (W-1)
#     positions of the current chunk have valid causal history.
#   - Plain `.detach()` on the cache (the pre-existing implementation) breaks
#     the gradient path, losing (W-1) positions worth of gradient per chunk
#     boundary, per layer.
#
# This Function manages the cache via explicit dicts (conv_cache / conv_grad)
# instead of tensor-arg autograd edges, mirroring DeltaNetChunkFunc and
# FlashAttnVarlenFunc's kv_cache design so the Seq1F1B scheduler's
# chunk-by-chunk backward never tries to traverse back through a previous
# chunk's autograd graph.
#
# Forward  (chunk N):  state := conv_cache[name]
#                      y, final_state := FLA Triton causal_conv1d(x, state)
#                      conv_cache[name] := final_state
# Backward (chunk N):  d_final_state := conv_grad[name]    (written by N+1)
#                      dx, dw, db, d_state := FLA Triton causal_conv1d_bwd
#                      conv_grad[name] := d_state           (read by N-1)
# ============================================================================

class ShortConvChunkFunc(torch.autograd.Function):
    """FLA causal conv1d with Seq1F1B cross-chunk gradient relay.

    Parameters
    ----------
    x      : [B, T, D] chunk input
    weight : [D, 1, W] depthwise conv weight (from fla.ShortConvolution.weight)
    bias   : [D] or None
    cache_dict, grad_dict : per-layer shared dicts, keyed by `name` ('q','k','v')
    name       : unique key within the layer
    activation : 'silu' | 'swish' | None
    """

    @staticmethod
    def forward(ctx, x, weight, bias, cache_dict, grad_dict, name, activation):
        # FLA stores causal-conv state as [B, D, W]. The returned final_state
        # is produced inside this custom forward, so autograd cannot see an
        # edge to the previous chunk; gradients are relayed manually below.
        weight_dw = rearrange(weight, "d 1 w -> d w")
        initial_state = cache_dict.get(name, None)
        had_state = initial_state is not None
        y, final_state = causal_conv1d_fwd(
            x=x,
            weight=weight_dw,
            bias=bias,
            initial_state=initial_state,
            output_final_state=True,
            activation=activation,
        )
        cache_dict[name] = final_state

        ctx.save_for_backward(x, weight_dw, bias, initial_state)
        ctx.activation = activation
        ctx.grad_dict = grad_dict
        ctx.name = name
        ctx.had_state = had_state
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight_dw, bias, initial_state = ctx.saved_tensors
        dht = ctx.grad_dict.pop(ctx.name, None)

        dx, dw, db, _, dh0 = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=dht,
            weight=weight_dw,
            bias=bias,
            initial_state=initial_state,
            activation=ctx.activation,
        )

        # Publish gradient wrt this chunk's initial state for the previous
        # chunk. Chunk 0 has no previous chunk; _clear_states() clears any
        # stale entry before the next microbatch starts.
        if ctx.had_state and dh0 is not None:
            ctx.grad_dict[ctx.name] = dh0

        # grads for (x, weight, bias, cache_dict, grad_dict, name, activation)
        return dx, rearrange(dw, "d w -> d 1 w"), db, None, None, None, None


# ============================================================================
# DeltaNetChunkFunc: custom autograd Function with backward state gradient relay
# Analogous to FlashAttnVarlenFunc for softmax attention in Seq1F1B.
#
# In Seq1F1B, each sequence is split into chunks processed sequentially.
# Forward:  chunk_0 -> state_0 -> chunk_1 -> state_1 -> ... -> chunk_{N-1}
# Backward: chunk_{N-1} -> dh0_{N-1} -> chunk_{N-2} -> dh0_{N-2} -> ... -> chunk_0
#
# Since DeltaNetChunkFunc is a custom autograd.Function, autograd does NOT
# penetrate into its forward (same as FlashAttnVarlenFunc with kv_cache).
# So final_state needs no .detach() — it has no grad_fn from the inner graph.
# The gradient relay is handled manually: each chunk's backward stores dh0
# into a shared state_cache dict, and the previous chunk reads it as dht.
# ============================================================================

class DeltaNetChunkFunc(torch.autograd.Function):
    """Wrap chunk_delta_rule_fwd/bwd with manual state gradient relay.

    State is passed between chunks via the shared `state_cache` dict,
    NOT as tensor arguments to .apply(). This is critical because:
      - If initial_state were a tensor arg to .apply(), it would have a grad_fn
        from the previous chunk's .apply() output, creating an autograd edge.
      - When chunk_N's backward runs, autograd would traverse that edge into
        chunk_{N-1}'s backward, freeing its saved tensors prematurely.
      - Then when the scheduler explicitly calls chunk_{N-1}'s backward,
        it fails with "backward through graph a second time".

    By reading/writing state through a Python dict (like FlashAttnVarlenFunc
    does with kv_cache), autograd has no visibility into the state passing.
    """

    @staticmethod
    def forward(
        ctx,
        q, k, v, beta,
        scale,
        state_cache,         # shared dict: holds 'recurrent_state' and 'd_state'
        use_qk_l2norm_in_kernel,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        # Read initial_state from shared cache (NOT a tensor arg to .apply())
        initial_state = state_cache.get('recurrent_state', None)

        o, A, final_state = chunk_delta_rule_fwd(
            q=q, k=k, v=v, beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=None,
            chunk_indices=None,
        )

        # Store final_state in cache for next chunk's forward
        state_cache['recurrent_state'] = final_state

        # Save for backward. initial_state is a plain tensor (no grad_fn)
        # because it was created inside a previous DeltaNetChunkFunc.forward
        # where autograd doesn't track operations.
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A)
        ctx._initial_state = initial_state  # may be None, can't use save_for_backward
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx._state_cache = state_cache

        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, do):
        q, q_rstd, k, k_rstd, v, beta, A = ctx.saved_tensors
        initial_state = ctx._initial_state
        state_cache = ctx._state_cache

        # Retrieve dht from state_cache (stored by the next chunk's backward).
        # If absent, this is the last chunk (first to run backward), dht=None.
        dht = state_cache.pop('d_state', None)

        dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(
            q=q, k=k, v=v, beta=beta, A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=None,
            chunk_indices=None,
        )

        # Store dh0 for the previous chunk's backward
        if dh0 is not None:
            state_cache['d_state'] = dh0

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # grad for: q, k, v, beta, scale, state_cache, use_qk_l2norm
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), \
               None, None, None


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

        # Q, K, V projections with tensor parallelism. When output gating is
        # enabled, fuse G into the same ColumnParallelLinear so SP chunks do
        # one larger GEMM instead of q/k/v plus a separate g projection. This
        # is mathematically equivalent to independent linear projections; the
        # local output layout is [q_local, k_local, v_local, g_local].
        if self.use_gate:
            self.qkvg_proj = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                4 * self.hidden_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=False,
                stride=4,
            )
        else:
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
        # DeltaNet recurrent state + d_state are relayed via self.state_cache.
        # Short-conv prefix tokens + their gradients are relayed via two
        # per-layer dicts managed by ShortConvChunkFunc (keyed by 'q'/'k'/'v').
        self.state_cache = {}
        self.conv_cache_dict = {}   # fwd: previous chunk's tail input tokens
        self.conv_grad_dict = {}    # bwd: gradient wrt the previous chunk's tail
        # Legacy placeholders (kept so external state_dict code does not break,
        # but are no longer read/written by the forward path).
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None

    def _clear_states(self):
        """Clear all cached states. Called when micro_sp_idx == 0."""
        self.state_cache = {}
        self.conv_cache_dict = {}
        self.conv_grad_dict = {}
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None

    def _stateful_middle(self, q, k, v, hidden_states_bsh,
                          use_conv_cache, output_final_state):
        """Stateful middle section: short conv -> reshape -> beta -> delta rule.

        Runs on a single (possibly chunked) slice of the sequence. All the
        state passing (conv caches, recurrent_state) happens through
        self.conv_state_{q,k,v} and self.state_cache, which are read when
        use_conv_cache / initial_state is active and written when
        output_final_state is True.

        Inputs:
          q, k, v:             [b, s, key/value_dim_per_partition] (post-projection, post-transpose)
          hidden_states_bsh:   [b, s, h]                            (for b_proj)
          use_conv_cache:      bool — read self.conv_state_{q,k,v} as initial conv state
          output_final_state:  bool — update self.conv_state_{q,k,v} and self.state_cache

        Output:
          o: [b, s, num_heads_per_partition, head_dim]
        """
        args = get_args()
        batch_size, seq_len, _ = hidden_states_bsh.shape

        # ============ Short Convolutions ============
        if self.use_short_conv:
            # Activation name per conv (q/k driven by qk_activation, v always silu)
            qk_act = 'silu' if self.qk_activation == 'silu' else None
            if output_final_state:
                # Chunked / stateful Seq1F1B path: use ShortConvChunkFunc so
                # gradient correctly relays across chunk boundaries (no detach).
                q = ShortConvChunkFunc.apply(
                    q, self.q_conv1d.weight, self.q_conv1d.bias,
                    self.conv_cache_dict, self.conv_grad_dict, 'q', qk_act,
                )
                k = ShortConvChunkFunc.apply(
                    k, self.k_conv1d.weight, self.k_conv1d.bias,
                    self.conv_cache_dict, self.conv_grad_dict, 'k', qk_act,
                )
                v = ShortConvChunkFunc.apply(
                    v, self.v_conv1d.weight, self.v_conv1d.bias,
                    self.conv_cache_dict, self.conv_grad_dict, 'v', 'silu',
                )
            else:
                # Non-chunked (SP=1 + no force-seq-chunks) — use fla's fused
                # kernel for speed; there is no cross-chunk gradient to relay.
                q, _ = self.q_conv1d(x=q, cache=None, output_final_state=False)
                k, _ = self.k_conv1d(x=k, cache=None, output_final_state=False)
                v, _ = self.v_conv1d(x=v, cache=None, output_final_state=False)
        else:
            # Without short conv, apply activation directly
            if self.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            v = F.silu(v)

        # ============ Reshape to multi-head ============
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
        k = rearrange(k, 'b s (h d) -> b s h d', d=self.head_dim)
        v = rearrange(v, 'b s (h d) -> b s h d', d=self.head_dim)

        # ============ QK activation (if not handled by conv) ============
        if self.use_short_conv and self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q = (F.elu(q, 1., False) + 1.).to(q)
                k = (F.elu(k, 1., False) + 1.).to(k)

        # ============ Beta ============
        if self.use_beta:
            beta = self.b_proj(hidden_states_bsh).sigmoid()
        else:
            beta = torch.ones(
                batch_size, seq_len, self.num_attention_heads_per_partition,
                device=hidden_states_bsh.device, dtype=hidden_states_bsh.dtype
            )

        # ============ DeltaNet core computation ============
        mode = 'fused_recurrent' if seq_len <= 64 else self.deltanet_mode
        initial_state = self.state_cache.get('recurrent_state', None)

        if output_final_state and mode == 'chunk':
            # Seq1F1B path: use DeltaNetChunkFunc with backward state gradient relay.
            orig_dtype = q.dtype
            needs_cast = (orig_dtype == torch.float32)
            if needs_cast:
                q_kv_dtype = torch.bfloat16
                q = q.to(q_kv_dtype)
                k = k.to(q_kv_dtype)
                v = v.to(q_kv_dtype)
                beta = beta.to(q_kv_dtype)

            o = DeltaNetChunkFunc.apply(
                q, k, v, beta,
                self.head_dim ** -0.5,
                self.state_cache,
                (self.qk_norm == 'l2'),
            )
            if needs_cast:
                o = o.to(orig_dtype)
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q, k=k, v=v, beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=(self.qk_norm == 'l2'),
            )
            if output_final_state and recurrent_state is not None:
                self.state_cache['recurrent_state'] = recurrent_state.detach()
        elif mode == 'chunk':
            orig_dtype = q.dtype
            needs_cast = (orig_dtype == torch.float32)
            if needs_cast:
                q_kv_dtype = torch.bfloat16
                q = q.to(q_kv_dtype)
                k = k.to(q_kv_dtype)
                v = v.to(q_kv_dtype)
                beta = beta.to(q_kv_dtype)

            o, recurrent_state = chunk_delta_rule(
                q=q, k=k, v=v, beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=(self.qk_norm == 'l2'),
            )
            if needs_cast:
                o = o.to(orig_dtype)
                if recurrent_state is not None:
                    try:
                        recurrent_state = recurrent_state.to(orig_dtype)
                    except Exception:
                        pass
        else:
            raise NotImplementedError(f"DeltaNet mode `{mode}` not supported.")

        return o

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

        # hidden_states: [s, b, h] (Megatron convention)
        # When sequence_parallel is True and TP>1, shape is [s/tp, b, h].
        # ColumnParallelLinear expects [s, b, h] format and internally does
        # all-gather along dim-0 (the seq dimension). So we must NOT
        # transpose before calling the projections.

        # ============ Projections (keep [s, b, h] for ColumnParallelLinear) ============
        # ColumnParallelLinear: [s(/tp), b, h] -> all-gather dim-0 -> [s, b, h] -> matmul -> [s, b, h/tp]
        if self.use_gate:
            qkvg, _ = self.qkvg_proj(hidden_states)
            q, k, v, g = torch.chunk(qkvg, 4, dim=-1)
        else:
            q, _ = self.q_proj(hidden_states)  # [s, b, key_dim_per_partition]
            k, _ = self.k_proj(hidden_states)  # [s, b, key_dim_per_partition]
            v, _ = self.v_proj(hidden_states)  # [s, b, value_dim_per_partition]
            g = None

        # Now transpose to DeltaNet convention: [s, b, h/tp] -> [b, s, h/tp]
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        if g is not None:
            g = g.transpose(0, 1).contiguous()

        # Also prepare hidden_states for b_proj (nn.Linear, no internal all-gather).
        # When sequence_parallel=True, hidden_states is [s/tp, b, h], but
        # b_proj needs full seq_len to match q/k/v (which were all-gathered by ColumnParallelLinear).
        if self.use_beta and self.sequence_parallel and mpu.get_tensor_model_parallel_world_size() > 1:
            # All-gather hidden_states along dim-0 to get full seq for b_proj
            hidden_states_full = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=True
            )
            # [s, b, h] -> [b, s, h]
            hidden_states_bsh = hidden_states_full.transpose(0, 1).contiguous()
        else:
            # [s, b, h] -> [b, s, h]
            hidden_states_bsh = hidden_states.transpose(0, 1).contiguous()
        batch_size, seq_len, _ = hidden_states_bsh.shape

        # ============ Optional force-seq-chunks mode (for verification) ============
        # When --force-seq-chunks N > 1 and pipe_sp_splits == 1, internally run
        # the stateful middle N times with state passing — replicating exactly
        # the DeltaNet kernel-call sequence that pipe_sp_splits=N would use,
        # while keeping everything else (projections, norms, FFN, PP schedule)
        # on the full sequence. This isolates the chunked DeltaNet computation
        # from all other system-level sources of fp noise, so that a bit-level
        # (fp32) diff between SP=1 and SP=1+force-seq-chunks=N proves Seq1F1B's
        # DeltaNet integration is algorithmically correct.
        fsc = getattr(args, 'force_seq_chunks', 1)
        use_force_chunks = (
            args.pipe_sp_splits == 1
            and fsc > 1
            and (micro_sp_idx is None or micro_sp_idx == 0)
        )
        if use_force_chunks:
            assert seq_len % fsc == 0, (
                f"seq_len {seq_len} must be divisible by "
                f"--force-seq-chunks {fsc}")
            # This forward owns the full sequence; always start fresh and
            # rebuild state via the in-loop chain.
            self._clear_states()
            chunk = seq_len // fsc
            outs = []
            for i in range(fsc):
                sl = slice(i * chunk, (i + 1) * chunk)
                o_i = self._stateful_middle(
                    q[:, sl].contiguous(),
                    k[:, sl].contiguous(),
                    v[:, sl].contiguous(),
                    hidden_states_bsh[:, sl].contiguous(),
                    use_conv_cache=(i > 0),
                    output_final_state=True,
                )
                outs.append(o_i)
            # Clear transient cross-chunk state so that subsequent forward
            # calls in the same training iteration (e.g. microbatch 2, 3, …)
            # start fresh — matching the SP=1 baseline behaviour.
            self._clear_states()
            o = torch.cat(outs, dim=1)
        else:
            # Original single-call path.
            use_conv_cache = (args.pipe_sp_splits > 1 and micro_sp_idx is not None
                              and micro_sp_idx > 0)
            output_final_state = (args.pipe_sp_splits > 1)
            o = self._stateful_middle(
                q, k, v, hidden_states_bsh,
                use_conv_cache=use_conv_cache,
                output_final_state=output_final_state,
            )

        # ============ Output normalization & gate ============
        if self.use_gate:
            g = rearrange(g, 'b s (h d) -> b s h d', d=self.head_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        # ============ Reshape and output projection ============
        # [b, s, num_heads_per_partition, head_dim] -> [b, s, hidden_per_partition]
        o = rearrange(o, 'b s h d -> b s (h d)')

        # RowParallelLinear expects [s, b, h/tp] format for correct reduce-scatter along dim-0
        o = o.transpose(0, 1).contiguous()  # [b, s, h/tp] -> [s, b, h/tp]

        # RowParallelLinear: matmul + reduce-scatter -> [s/tp, b, h] (when sequence_parallel)
        # or matmul + all-reduce -> [s, b, h] (when not sequence_parallel)
        output, bias = self.o_proj(o)
        if bias is not None:
            # bias is [h], needs to be compatible with [s, b, h]
            pass

        return output, bias
