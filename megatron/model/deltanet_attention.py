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

import contextlib
import math
import os
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron import get_args, core
from megatron.core import mpu, tensor_parallel
from megatron.model.module import MegatronModule

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
    from fla.ops.delta_rule.chunk import (
        chunk_delta_rule_bwd,
        chunk_delta_rule_fwd,
    )
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


_DELTANET_TIMING_STATS = {}


if HAS_TRITON:
    @triton.jit
    def _qkvg_sbh_to_bsh_fwd_kernel(
        qkvg,
        q,
        k,
        v,
        g,
        N: tl.constexpr,
        S: tl.constexpr,
        B: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        i_d = offsets % D
        tmp = offsets // D
        i_s = tmp % S
        i_b = tmp // S

        in_base = (i_s * B + i_b) * (4 * D) + i_d
        out = (i_b * S + i_s) * D + i_d

        q_val = tl.load(qkvg + in_base, mask=mask)
        k_val = tl.load(qkvg + in_base + D, mask=mask)
        v_val = tl.load(qkvg + in_base + 2 * D, mask=mask)
        g_val = tl.load(qkvg + in_base + 3 * D, mask=mask)

        tl.store(q + out, q_val, mask=mask)
        tl.store(k + out, k_val, mask=mask)
        tl.store(v + out, v_val, mask=mask)
        tl.store(g + out, g_val, mask=mask)


    @triton.jit
    def _qkvg_sbh_to_bsh_bwd_kernel(
        dq,
        dk,
        dv,
        dg,
        dqkvg,
        N: tl.constexpr,
        S: tl.constexpr,
        B: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        i_d = offsets % D
        tmp = offsets // D
        i_s = tmp % S
        i_b = tmp // S

        in_out = (i_b * S + i_s) * D + i_d
        grad_base = (i_s * B + i_b) * (4 * D) + i_d

        tl.store(dqkvg + grad_base, tl.load(dq + in_out, mask=mask), mask=mask)
        tl.store(dqkvg + grad_base + D, tl.load(dk + in_out, mask=mask), mask=mask)
        tl.store(dqkvg + grad_base + 2 * D, tl.load(dv + in_out, mask=mask), mask=mask)
        tl.store(dqkvg + grad_base + 3 * D, tl.load(dg + in_out, mask=mask), mask=mask)


class _QKVGLayoutFunc(torch.autograd.Function):
    """Split [S, B, 4D] qkvg into four contiguous [B, S, D] tensors."""

    @staticmethod
    def forward(ctx, qkvg):
        S, B, four_d = qkvg.shape
        D = four_d // 4
        q = torch.empty((B, S, D), device=qkvg.device, dtype=qkvg.dtype)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        g = torch.empty_like(q)

        n_elements = B * S * D
        block = 256
        grid = (triton.cdiv(n_elements, block),)
        _qkvg_sbh_to_bsh_fwd_kernel[grid](
            qkvg, q, k, v, g,
            n_elements, S, B, D,
            BLOCK=block,
        )
        ctx.shape = (S, B, four_d)
        return q, k, v, g

    @staticmethod
    def backward(ctx, dq, dk, dv, dg):
        S, B, four_d = ctx.shape
        D = four_d // 4

        grads = [dq, dk, dv, dg]
        ref = next((x for x in grads if x is not None), None)
        if ref is None:
            return None
        grads = [
            x.contiguous() if x is not None else torch.zeros((B, S, D), device=ref.device, dtype=ref.dtype)
            for x in grads
        ]

        dqkvg = torch.empty((S, B, four_d), device=ref.device, dtype=ref.dtype)
        n_elements = B * S * D
        block = 256
        grid = (triton.cdiv(n_elements, block),)
        _qkvg_sbh_to_bsh_bwd_kernel[grid](
            grads[0], grads[1], grads[2], grads[3], dqkvg,
            n_elements, S, B, D,
            BLOCK=block,
        )
        return dqkvg


def _can_fuse_qkvg_layout(qkvg):
    return (
        HAS_TRITON
        and qkvg.is_cuda
        and qkvg.is_contiguous()
        and qkvg.dim() == 3
        and qkvg.shape[-1] % 4 == 0
        and os.environ.get("DELTANET_FUSED_QKVG_LAYOUT", "1") != "0"
    )


def _layout_qkvg_to_bsh(qkvg):
    if _can_fuse_qkvg_layout(qkvg):
        return _QKVGLayoutFunc.apply(qkvg)
    q, k, v, g = torch.chunk(qkvg, 4, dim=-1)
    return (
        q.transpose(0, 1).contiguous(),
        k.transpose(0, 1).contiguous(),
        v.transpose(0, 1).contiguous(),
        g.transpose(0, 1).contiguous(),
    )


@contextlib.contextmanager
def _temporarily_disable_wu_h_pipeline(enabled: bool):
    if not enabled:
        yield
        return
    old_value = os.environ.get("FLA_DELTA_WU_H_PIPELINE")
    os.environ["FLA_DELTA_WU_H_PIPELINE"] = "0"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("FLA_DELTA_WU_H_PIPELINE", None)
        else:
            os.environ["FLA_DELTA_WU_H_PIPELINE"] = old_value


def _deltanet_profile_enabled():
    return (
        os.environ.get("DELTANET_ATTN_NVTX", "0") != "0"
        or os.environ.get("DELTANET_ATTN_CUDA_TIMING", "0") != "0"
    ) and torch.cuda.is_available()


def _deltanet_profile_start(name):
    if not _deltanet_profile_enabled():
        return None
    use_nvtx = os.environ.get("DELTANET_ATTN_NVTX", "0") != "0"
    use_timing = os.environ.get("DELTANET_ATTN_CUDA_TIMING", "0") != "0"
    if use_nvtx:
        torch.cuda.nvtx.range_push(name)
    start_event = end_event = None
    if use_timing:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    return name, use_nvtx, start_event, end_event


def _deltanet_profile_end(token):
    if token is None:
        return
    name, use_nvtx, start_event, end_event = token
    if end_event is not None:
        end_event.record()
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        total_ms, count, window_ms, window_count = _DELTANET_TIMING_STATS.get(
            name, (0.0, 0, 0.0, 0)
        )
        total_ms += elapsed_ms
        count += 1
        window_ms += elapsed_ms
        window_count += 1
        interval = int(os.environ.get("DELTANET_ATTN_TIMING_INTERVAL", "100"))
        if interval > 0 and count % interval == 0:
            try:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            except Exception:
                rank = 0
            print(
                f"[rank {rank}] {name}: last={elapsed_ms:.3f} ms "
                f"window_avg={window_ms / max(window_count, 1):.3f} ms "
                f"global_avg={total_ms / count:.3f} ms count={count}",
                flush=True,
            )
            window_ms = 0.0
            window_count = 0
        _DELTANET_TIMING_STATS[name] = (total_ms, count, window_ms, window_count)
    if use_nvtx:
        torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def _deltanet_profile(name):
    token = _deltanet_profile_start(name)
    try:
        yield
    finally:
        _deltanet_profile_end(token)


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
            residual=None,
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
        use_ho_pipeline=False,
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
            use_ho_pipeline=use_ho_pipeline,
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
        state_cache = ctx._state_cache

        # Retrieve dht from state_cache (stored by the next chunk's backward).
        # If absent, this is the last chunk (first to run backward), dht=None.
        dht = state_cache.pop('d_state', None)

        q, q_rstd, k, k_rstd, v, beta, A = ctx.saved_tensors
        initial_state = ctx._initial_state

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

        # grad for: q, k, v, beta, scale, state_cache, use_qk_l2norm,
        # use_ho_pipeline
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), \
               None, None, None, None


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
        self._qkvg_lookahead_stream = None

    def _clear_states(self):
        """Clear all cached states. Called when micro_sp_idx == 0."""
        self.state_cache = {}
        self.conv_cache_dict = {}
        self.conv_grad_dict = {}
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None

    def _should_use_ho_pipeline(self):
        args = get_args()
        return (
            getattr(args, 'deltanet_fused_h_o_pipeline', False)
            and torch.cuda.is_available()
        )

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
        profile_token = _deltanet_profile_start("DeltaNetAttention.stateful_middle")
        args = get_args()
        batch_size, seq_len, _ = hidden_states_bsh.shape

        # ============ Short Convolutions ============
        with _deltanet_profile("DeltaNetAttention.short_conv_or_activation"):
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

        # ============ DeltaNet core mode ============
        mode = 'fused_recurrent' if seq_len <= 64 else self.deltanet_mode

        use_ho_pipeline = (
            mode == 'chunk'
            and self._should_use_ho_pipeline()
        )

        # ============ Beta ============
        with _deltanet_profile("DeltaNetAttention.beta"):
            if self.use_beta:
                beta = self.b_proj(hidden_states_bsh).sigmoid()
            else:
                beta = torch.ones(
                    batch_size, seq_len, self.num_attention_heads_per_partition,
                    device=hidden_states_bsh.device, dtype=hidden_states_bsh.dtype
                )

        # ============ DeltaNet core computation ============
        initial_state = self.state_cache.get('recurrent_state', None)

        with _deltanet_profile("DeltaNetAttention.delta_rule_core"):
            if output_final_state and mode == 'chunk':
                # Seq1F1B path: use DeltaNetChunkFunc with backward state gradient relay.
                orig_dtype = q.dtype
                needs_cast = (orig_dtype == torch.float32)
                if needs_cast:
                    q_kv_dtype = torch.bfloat16
                    q = q.to(q_kv_dtype)
                    k = k.to(q_kv_dtype)
                    v = v.to(q_kv_dtype)
                    if beta is not None:
                        beta = beta.to(q_kv_dtype)

                o = DeltaNetChunkFunc.apply(
                    q, k, v, beta,
                    self.head_dim ** -0.5,
                    self.state_cache,
                    (self.qk_norm == 'l2'),
                    use_ho_pipeline,
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
                    use_ho_pipeline=use_ho_pipeline,
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

        _deltanet_profile_end(profile_token)
        return o

    def _qkvg_flat_pipeline_forward(self, hidden_states, hidden_states_bsh):
        """Two-stage diagnostic schedule with one global split point.

        This is a forward-only profiling path for the single-rank attention
        microbench. It keeps qkvg/layout/conv/beta/WY/H/O on the same split
        boundary so the timeline does not become an outer x inner split.
        """
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla.ops.common.chunk_o import chunk_fwd_o_range_into
        from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd_range_into
        from fla.ops.utils.solve_tril import solve_tril_range_into

        assert self.use_gate, "qkvg flat pipeline requires fused qkvg projection."
        B, T, _ = hidden_states_bsh.shape
        BT = 64
        NT = math.ceil(T / BT)
        if NT < 2:
            raise RuntimeError("DELTANET_QKVG_PIPELINE requires at least two chunks")
        if "FLA_DELTA_WU_H_SPLIT_NT" in os.environ:
            split_nt = int(os.environ["FLA_DELTA_WU_H_SPLIT_NT"])
        else:
            split_frac = float(os.environ.get("FLA_DELTA_WU_H_SPLIT_FRAC", "0.5"))
            split_nt = int(round(NT * split_frac))
        split_nt = max(1, min(NT - 1, split_nt))
        split_t = min(T, split_nt * BT)

        if T % BT != 0:
            raise RuntimeError("qkvg flat pipeline currently expects seq_len to be a multiple of 64")

        qk_act = 'silu' if self.qk_activation == 'silu' else None
        scale = self.head_dim ** -0.5
        use_l2 = self.qk_norm == 'l2'
        output_final_state = False
        initial_state = self.state_cache.get('recurrent_state', None)

        def run_conv(q_flat, k_flat, v_flat, initial_conv_states):
            if self.use_short_conv:
                q_state, k_state, v_state = initial_conv_states
                q_out, q_final = causal_conv1d_fwd(
                    x=q_flat,
                    weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
                    bias=self.q_conv1d.bias,
                    residual=None,
                    initial_state=q_state,
                    output_final_state=True,
                    activation=qk_act,
                )
                k_out, k_final = causal_conv1d_fwd(
                    x=k_flat,
                    weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
                    bias=self.k_conv1d.bias,
                    residual=None,
                    initial_state=k_state,
                    output_final_state=True,
                    activation=qk_act,
                )
                v_out, v_final = causal_conv1d_fwd(
                    x=v_flat,
                    weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                    bias=self.v_conv1d.bias,
                    residual=None,
                    initial_state=v_state,
                    output_final_state=True,
                    activation='silu',
                )
                conv_states = (q_final, k_final, v_final)
            else:
                if self.qk_activation == 'silu':
                    q_out, k_out = F.silu(q_flat), F.silu(k_flat)
                else:
                    q_out, k_out = q_flat, k_flat
                v_out = F.silu(v_flat)
                conv_states = (None, None, None)

            q_out = rearrange(q_out, 'b s (h d) -> b s h d', d=self.head_dim)
            k_out = rearrange(k_out, 'b s (h d) -> b s h d', d=self.head_dim)
            v_out = rearrange(v_out, 'b s (h d) -> b s h d', d=self.head_dim)

            if self.use_short_conv and self.qk_activation != 'silu':
                if self.qk_activation == 'relu':
                    q_out, k_out = q_out.relu(), k_out.relu()
                elif self.qk_activation == 'elu':
                    q_out = (F.elu(q_out, 1., False) + 1.).to(q_out)
                    k_out = (F.elu(k_out, 1., False) + 1.).to(k_out)

            if use_l2:
                q_out, _ = l2norm_fwd(q_out)
                k_out, _ = l2norm_fwd(k_out)

            return q_out, k_out, v_out, conv_states

        def prepare_half(tag, hidden_sbh, hidden_bsh_i, initial_conv_states):
            with _deltanet_profile(f"DeltaNetAttention.qkv_proj.flat_{tag}"):
                qkvg_i, _ = self.qkvg_proj(hidden_sbh)
            with _deltanet_profile(f"DeltaNetAttention.input_layout.flat_{tag}"):
                q_flat, k_flat, v_flat, g_flat = _layout_qkvg_to_bsh(qkvg_i)
            with _deltanet_profile(f"DeltaNetAttention.short_conv.flat_{tag}"):
                q_i, k_i, v_i, conv_states = run_conv(
                    q_flat,
                    k_flat,
                    v_flat,
                    initial_conv_states,
                )
            with _deltanet_profile(f"DeltaNetAttention.beta.flat_{tag}"):
                if self.use_beta:
                    beta_i = self.b_proj(hidden_bsh_i).sigmoid()
                else:
                    beta_i = torch.ones(
                        B,
                        hidden_bsh_i.shape[1],
                        self.num_attention_heads_per_partition,
                        device=hidden_bsh_i.device,
                        dtype=hidden_bsh_i.dtype,
                    )
            with _deltanet_profile(f"DeltaNetCore.WY.A.flat_{tag}"):
                raw_A_i = chunk_scaled_dot_kkt_fwd(
                    k=k_i,
                    beta=beta_i,
                    cu_seqlens=None,
                    chunk_size=BT,
                    output_dtype=torch.float32,
                    chunk_indices=None,
                )
            A_i = torch.zeros_like(raw_A_i, dtype=k_i.dtype)
            w_i = torch.empty_like(k_i)
            u_i = torch.empty_like(v_i)
            nt_i = math.ceil(hidden_bsh_i.shape[1] / BT)
            with _deltanet_profile(f"DeltaNetCore.WY.solve_inverse.flat_{tag}"):
                solve_tril_range_into(
                    A=raw_A_i,
                    Ai=A_i,
                    t_block_start=0,
                    t_block_count=nt_i,
                    cu_seqlens=None,
                    chunk_indices=None,
                )
            with _deltanet_profile(f"DeltaNetCore.recompute_wu.flat_{tag}"):
                recompute_w_u_fwd_range_into(
                    k=k_i,
                    v=v_i,
                    beta=beta_i,
                    A=A_i,
                    w=w_i,
                    u=u_i,
                    t_block_start=0,
                    t_block_count=nt_i,
                    cu_seqlens=None,
                )
            return {
                "q": q_i,
                "k": k_i,
                "v": v_i,
                "g": g_flat,
                "A": A_i,
                "w": w_i,
                "u": u_i,
                "conv_states": conv_states,
                "nt": nt_i,
                "raw_A": raw_A_i,
                "beta": beta_i,
            }

        hidden1 = hidden_states[:split_t].contiguous()
        hidden2 = hidden_states[split_t:].contiguous()
        hidden_bsh1 = hidden_states_bsh[:, :split_t].contiguous()
        hidden_bsh2 = hidden_states_bsh[:, split_t:].contiguous()

        prep1 = prepare_half("first", hidden1, hidden_bsh1, (None, None, None))

        if self._qkvg_lookahead_stream is None:
            self._qkvg_lookahead_stream = torch.cuda.Stream()
        prep_stream = self._qkvg_lookahead_stream
        o_stream = torch.cuda.Stream(device=hidden_states.device)
        current_stream = torch.cuda.current_stream(hidden_states.device)
        prep_stream.wait_stream(current_stream)
        hidden2.record_stream(prep_stream)
        hidden_bsh2.record_stream(prep_stream)
        for state in prep1["conv_states"]:
            if state is not None:
                state.record_stream(prep_stream)

        prep2_holder = {}
        prep2_event = torch.cuda.Event()
        with torch.cuda.stream(prep_stream):
            prep2_holder["value"] = prepare_half(
                "second",
                hidden2,
                hidden_bsh2,
                prep1["conv_states"],
            )
            prep2_event.record(prep_stream)

        h1, v_new1, mid_state = chunk_gated_delta_rule_fwd_h(
            k=prep1["k"],
            w=prep1["w"],
            u=prep1["u"],
            g=None,
            initial_state=initial_state,
            output_final_state=True,
            t_block_start=0,
            t_block_count=prep1["nt"],
        )

        o1 = torch.empty_like(prep1["v"])
        o2 = None
        o_stream.wait_stream(current_stream)
        with torch.cuda.stream(o_stream):
            with _deltanet_profile("DeltaNetCore.O.flat_first"):
                chunk_fwd_o_range_into(
                    q=prep1["q"],
                    k=prep1["k"],
                    v=v_new1,
                    h=h1,
                    o=o1,
                    g=None,
                    scale=scale,
                    cu_seqlens=None,
                    chunk_indices=None,
                    t_block_start=0,
                    t_block_count=prep1["nt"],
                )

        current_stream.wait_event(prep2_event)
        prep2 = prep2_holder["value"]
        for tensor in (
            prep2["q"], prep2["k"], prep2["v"], prep2["g"], prep2["A"],
            prep2["w"], prep2["u"], prep2["raw_A"], prep2["beta"],
        ):
            tensor.record_stream(current_stream)

        h2, v_new2, final_state = chunk_gated_delta_rule_fwd_h(
            k=prep2["k"],
            w=prep2["w"],
            u=prep2["u"],
            g=None,
            initial_state=mid_state,
            output_final_state=output_final_state,
            t_block_start=0,
            t_block_count=prep2["nt"],
        )

        o2 = torch.empty_like(prep2["v"])
        with _deltanet_profile("DeltaNetCore.O.flat_second"):
            chunk_fwd_o_range_into(
                q=prep2["q"],
                k=prep2["k"],
                v=v_new2,
                h=h2,
                o=o2,
                g=None,
                scale=scale,
                cu_seqlens=None,
                chunk_indices=None,
                t_block_start=0,
                t_block_count=prep2["nt"],
            )
        current_stream.wait_stream(o_stream)

        g = torch.cat((prep1["g"], prep2["g"]), dim=1)
        o = torch.cat((o1, o2), dim=1)

        for tensor in (h1, h2, v_new1, v_new2, mid_state, final_state, o1, o2, g, o):
            if tensor is not None:
                tensor.record_stream(current_stream)
        return o, g

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
        profile_token = _deltanet_profile_start("DeltaNetAttention.forward")
        args = get_args()
        fsc = getattr(args, 'force_seq_chunks', 1)
        qkvg_pipeline_requested = os.environ.get("DELTANET_QKVG_PIPELINE", "0") != "0"
        qkvg_pipeline_mode = os.environ.get("DELTANET_QKVG_PIPELINE_MODE", "flat").lower()
        qkvg_pipeline_flat = (
            qkvg_pipeline_requested
            and qkvg_pipeline_mode == "flat"
            and args.pipe_sp_splits == 1
            and (micro_sp_idx is None or micro_sp_idx == 0)
            and self.use_gate
            and torch.cuda.is_available()
            and not torch.is_grad_enabled()
            and not (
                self.sequence_parallel
                and mpu.get_tensor_model_parallel_world_size() > 1
            )
        )
        qkvg_pipeline_outer = (
            qkvg_pipeline_requested
            and qkvg_pipeline_mode == "outer"
        )
        if (
            qkvg_pipeline_outer
            and args.pipe_sp_splits == 1
            and (micro_sp_idx is None or micro_sp_idx == 0)
        ):
            fsc = max(fsc, int(os.environ.get("DELTANET_QKVG_PIPELINE_CHUNKS", "2")))
        use_force_chunks = (
            args.pipe_sp_splits == 1
            and fsc > 1
            and (micro_sp_idx is None or micro_sp_idx == 0)
        )

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

        qkvg_chunked = (
            use_force_chunks
            and self.use_gate
            and torch.cuda.is_available()
            and (
                os.environ.get("DELTANET_QKVG_CHUNKED", "0") != "0"
                or os.environ.get("DELTANET_QKVG_LOOKAHEAD", "0") != "0"
                or qkvg_pipeline_outer
            )
            # The first implementation is a diagnostic for the single-rank
            # force-chunk path. Real SP+TP lookahead needs a schedule-level
            # cache because the next chunk's per-layer hidden state is not
            # available inside this attention call.
            and not (
                self.sequence_parallel
                and mpu.get_tensor_model_parallel_world_size() > 1
            )
        )
        qkvg_lookahead = (
            qkvg_chunked
            and (
                os.environ.get("DELTANET_QKVG_LOOKAHEAD", "0") != "0"
                or qkvg_pipeline_outer
            )
        )

        if not qkvg_chunked and not qkvg_pipeline_flat:
            # ============ Projections (keep [s, b, h] for ColumnParallelLinear) ============
            # ColumnParallelLinear: [s(/tp), b, h] -> all-gather dim-0 -> [s, b, h] -> matmul -> [s, b, h/tp]
            with _deltanet_profile("DeltaNetAttention.qkv_proj"):
                if self.use_gate:
                    qkvg, _ = self.qkvg_proj(hidden_states)
                elif not self.use_gate:
                    q, _ = self.q_proj(hidden_states)  # [s, b, key_dim_per_partition]
                    k, _ = self.k_proj(hidden_states)  # [s, b, key_dim_per_partition]
                    v, _ = self.v_proj(hidden_states)  # [s, b, value_dim_per_partition]
                    g = None

            # Now transpose to DeltaNet convention: [s, b, h/tp] -> [b, s, h/tp]
            with _deltanet_profile("DeltaNetAttention.input_layout"):
                if self.use_gate:
                    q, k, v, g = _layout_qkvg_to_bsh(qkvg)
                else:
                    q = q.transpose(0, 1).contiguous()
                    k = k.transpose(0, 1).contiguous()
                    v = v.transpose(0, 1).contiguous()
        else:
            q = k = v = g = None

        # Also prepare hidden_states for b_proj (nn.Linear, no internal all-gather).
        # When sequence_parallel=True, hidden_states is [s/tp, b, h], but
        # b_proj needs full seq_len to match q/k/v (which were all-gathered by ColumnParallelLinear).
        with _deltanet_profile("DeltaNetAttention.beta_input_layout"):
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
        if qkvg_pipeline_flat:
            # This path intentionally owns the whole single-span sequence and
            # applies one shared split to qkvg/conv/KKT/WU/H/O. It is
            # forward-only because it bypasses the autograd wrapper used by
            # the normal training path.
            self._clear_states()
            with _deltanet_profile("DeltaNetAttention.qkvg_flat_pipeline"):
                o, g = self._qkvg_flat_pipeline_forward(hidden_states, hidden_states_bsh)
            self._clear_states()
        elif use_force_chunks:
            assert seq_len % fsc == 0, (
                f"seq_len {seq_len} must be divisible by "
                f"--force-seq-chunks {fsc}")
            # This forward owns the full sequence; always start fresh and
            # rebuild state via the in-loop chain.
            self._clear_states()
            chunk = seq_len // fsc
            outs = []

            if qkvg_lookahead:
                if self._qkvg_lookahead_stream is None:
                    self._qkvg_lookahead_stream = torch.cuda.Stream()
                qkvg_stream = self._qkvg_lookahead_stream
                current_stream = torch.cuda.current_stream()
                qkvg_stream.wait_stream(current_stream)
                qkvg_cache = {}
                qkvg_events = {}
                g_chunks = []
                use_nvtx = os.environ.get("DELTANET_ATTN_NVTX", "0") != "0"
                disable_inner_wuh = (
                    qkvg_pipeline_outer
                    and os.environ.get("DELTANET_QKVG_PIPELINE_INNER_WUH", "0") == "0"
                )

                def launch_qkvg(chunk_idx):
                    sl_i = slice(chunk_idx * chunk, (chunk_idx + 1) * chunk)
                    hidden_i = hidden_states[sl_i]
                    hidden_i.record_stream(qkvg_stream)
                    with torch.cuda.stream(qkvg_stream):
                        if use_nvtx:
                            torch.cuda.nvtx.range_push(
                                f"DeltaNetAttention.qkv_proj.lookahead_{chunk_idx}"
                            )
                        try:
                            qkvg_i, _ = self.qkvg_proj(hidden_i)
                        finally:
                            if use_nvtx:
                                torch.cuda.nvtx.range_pop()
                        event_i = torch.cuda.Event()
                        event_i.record(qkvg_stream)
                    qkvg_cache[chunk_idx] = qkvg_i
                    qkvg_events[chunk_idx] = event_i

                launch_qkvg(0)
                for i in range(fsc):
                    sl = slice(i * chunk, (i + 1) * chunk)
                    current_stream.wait_event(qkvg_events.pop(i))
                    qkvg_i = qkvg_cache.pop(i)
                    qkvg_i.record_stream(current_stream)
                    with _deltanet_profile("DeltaNetAttention.input_layout"):
                        q_i, k_i, v_i, g_i = _layout_qkvg_to_bsh(qkvg_i)
                    if i + 1 < fsc:
                        launch_qkvg(i + 1)
                    with _temporarily_disable_wu_h_pipeline(disable_inner_wuh):
                        o_i = self._stateful_middle(
                            q_i,
                            k_i,
                            v_i,
                            hidden_states_bsh[:, sl].contiguous(),
                            use_conv_cache=(i > 0),
                            output_final_state=True,
                        )
                    outs.append(o_i)
                    g_chunks.append(g_i)
                g = torch.cat(g_chunks, dim=1)
            elif qkvg_chunked:
                g_chunks = []
                disable_inner_wuh = (
                    qkvg_pipeline_outer
                    and os.environ.get("DELTANET_QKVG_PIPELINE_INNER_WUH", "0") == "0"
                )
                for i in range(fsc):
                    sl = slice(i * chunk, (i + 1) * chunk)
                    with _deltanet_profile("DeltaNetAttention.qkv_proj"):
                        qkvg_i, _ = self.qkvg_proj(hidden_states[sl])
                    with _deltanet_profile("DeltaNetAttention.input_layout"):
                        q_i, k_i, v_i, g_i = _layout_qkvg_to_bsh(qkvg_i)
                    with _temporarily_disable_wu_h_pipeline(disable_inner_wuh):
                        o_i = self._stateful_middle(
                            q_i,
                            k_i,
                            v_i,
                            hidden_states_bsh[:, sl].contiguous(),
                            use_conv_cache=(i > 0),
                            output_final_state=True,
                        )
                    outs.append(o_i)
                    g_chunks.append(g_i)
                g = torch.cat(g_chunks, dim=1)
            else:
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
        with _deltanet_profile("DeltaNetAttention.output_norm"):
            if self.use_gate:
                g = rearrange(g, 'b s (h d) -> b s h d', d=self.head_dim)
                o = self.o_norm(o, g)
            else:
                o = self.o_norm(o)

        # ============ Reshape and output projection ============
        # [b, s, num_heads_per_partition, head_dim] -> [b, s, hidden_per_partition]
        with _deltanet_profile("DeltaNetAttention.output_proj"):
            o = rearrange(o, 'b s h d -> b s (h d)')

            # RowParallelLinear expects [s, b, h/tp] format for correct reduce-scatter along dim-0
            o = o.transpose(0, 1).contiguous()  # [b, s, h/tp] -> [s, b, h/tp]

            # RowParallelLinear: matmul + reduce-scatter -> [s/tp, b, h] (when sequence_parallel)
            # or matmul + all-reduce -> [s, b, h] (when not sequence_parallel)
            output, bias = self.o_proj(o)
        if bias is not None:
            # bias is [h], needs to be compatible with [s, b, h]
            pass

        _deltanet_profile_end(profile_token)
        return output, bias
