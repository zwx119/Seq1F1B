# Seq1F1B × DeltaNet: Linear Attention for Pipeline Parallelism

## Overview

This PR integrates **DeltaNet linear attention** into the Seq1F1B pipeline
parallelism framework, providing a drop-in replacement for softmax (FlashAttention)
in the Megatron-LM GPT model. DeltaNet uses a recurrent state instead of a KV
cache, which naturally supports Seq1F1B's sequence-splitting mechanism.

### Why DeltaNet + Seq1F1B?

In standard softmax attention, Seq1F1B must concatenate KV caches across sequence
splits and replay them through `FlashAttnVarlenFunc` with a custom backward. This
is complex and memory-intensive. DeltaNet replaces this with:

$$
S_t = (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top
$$

$$
o_t = q_t^\top S_t
$$

The recurrent state $S \in \mathbb{R}^{d_k \times d_v}$ is constant-size regardless
of sequence length, making it **O(1) memory** for cross-split state transfer (vs
O(s·d) for KV cache). For Seq1F1B, we simply:
- `micro_sp_idx == 0`: clear state (new microbatch)
- `micro_sp_idx > 0`: pass cached `recurrent_state` as `initial_state`

---

## Files Changed

### New Files

| File | Lines | Description |
|------|-------|-------------|
| `megatron/model/deltanet_attention.py` | ~370 | Core DeltaNet attention module |
| `megatron/model/deltanet_layer.py` | ~200 | Transformer layer using DeltaNet |
| `run_deltanet.sh` | ~120 | Launch script with DeltaNet args |
| `docs/deltanet.md` | (this file) | Documentation |

### Modified Files

| File | Description |
|------|-------------|
| `megatron/arguments.py` | Added `--use-deltanet` and 10 DeltaNet config args |
| `megatron/model/transformer.py` | Modified `build_layer()` to use DeltaNetTransformerLayer |

---

## New Command-Line Arguments

```
--use-deltanet               Enable DeltaNet (disables RoPE + FlashAttn)
--deltanet-mode              chunk | fused_recurrent (default: chunk)
--deltanet-use-short-conv    Use causal 1-D convolution (default: True)
--no-deltanet-short-conv     Disable short convolutions
--deltanet-conv-size         Conv kernel size (default: 4)
--deltanet-use-beta          Learned per-token step size (default: True)
--no-deltanet-beta           Fixed beta=1
--deltanet-use-output-gate   Output gating mechanism (default: True)
--no-deltanet-output-gate    Disable output gate
--deltanet-qk-activation     silu|relu|elu|identity|none (default: silu)
--deltanet-qk-norm           l2|none (default: l2)
--deltanet-head-dim          Per-head dimension (default: from kv_channels)
```

When `--use-deltanet` is set, the following are **automatically** applied:
- `--position-embedding-type` is forced to `none` (no RoPE, no learned PE)
- `--use-flash-attn` is rejected (mutually exclusive)

---

## Architecture Details

### `deltanet_attention.py` — DeltaNetAttention

The core attention module, replacing `ParallelAttention`.

**Projections** (with Tensor Parallelism):
- Q, K, V: `ColumnParallelLinear(H, H, gather_output=False)` — splits heads across TP ranks.
- Output: `RowParallelLinear(H, H, input_is_parallel=True, skip_bias_add=True)`.
- Beta: `nn.Linear(H, num_heads_per_partition)` — small, no TP split needed.

**Short Convolutions**:
- Uses `fla.modules.ShortConvolution` (depthwise causal Conv1d, kernel=4 by default).
- Applied to Q, K, V before the delta-rule computation.
- Replaces RoPE for positional information.
- For Seq1F1B: conv state is cached and passed between splits.

**Seq1F1B State Management**:
```python
# In forward():
if micro_sp_idx == 0:
    self._clear_states()   # New microbatch → fresh start

# DeltaNet computation with state continuity
o, recurrent_state = chunk_delta_rule(
    q, k, v, beta,
    initial_state=self.recurrent_state,  # None for first split, cached for others
    output_final_state=True,
)
self.recurrent_state = recurrent_state.detach()  # Cache for next split
```

**Data Flow**:
```
Input [s,b,h] → transpose → [b,s,h]
  → Q,K,V projections (ColumnParallelLinear)
  → Short convolutions (with state caching)
  → Reshape to multi-head [b,s,H,d]
  → chunk_delta_rule / fused_recurrent_delta_rule
  → Output normalization (FLARMSNorm or FusedRMSNormGated)
  → Output projection (RowParallelLinear)
  → transpose → [s,b,h]
```

### `deltanet_layer.py` — DeltaNetTransformerLayer

Drop-in replacement for `ParallelTransformerLayer`. Structure:

```
hidden_states
  │
  ├─ input_layernorm
  ├─ DeltaNetAttention (instead of ParallelAttention)
  ├─ bias_dropout_add + residual
  ├─ post_attention_layernorm
  ├─ ParallelMLP (unchanged — SwiGLU or GELU)
  ├─ bias_dropout_add + residual
  │
  └─ output
```

Everything except the attention module is identical to the original layer.

### `arguments.py` — Argument Modifications

Two changes:
1. **New `_add_deltanet_args()` function**: Adds all `--deltanet-*` arguments.
2. **Validation in `parse_args()`**: When `--use-deltanet` is set, forces
   `position_embedding_type='none'` and validates no conflicting args.
3. **Extended `position_embedding_type` choices**: Added `'none'` option.

### `transformer.py` — build_layer() Modification

```python
def build_layer(layer_number):
    if args.transformer_impl == 'local':
        current_layer_type = _get_layer_type(...)
        if getattr(args, 'use_deltanet', False):          # ← NEW
            from megatron.model.deltanet_layer import DeltaNetTransformerLayer
            return DeltaNetTransformerLayer(config, layer_number, ...)
        return ParallelTransformerLayer(config, layer_number, ...)
```

Lazy import ensures no overhead when DeltaNet is not used.

---

## Dependencies

- **flash-linear-attention (fla)**: Provides `chunk_delta_rule`, `fused_recurrent_delta_rule`,
  `ShortConvolution`, `RMSNorm`. Install with `pip install flash-linear-attention`.
- **einops**: For `rearrange()` tensor reshaping.
- **triton**: Required by fla's Triton kernels.

---

## Quick Start

```bash
# Single-node, 8 GPU example
export GPUS_PER_NODE=8 WORLD_SIZE=1
export MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/path/to/preprocessed/data
export PP_SIZE=4 TP_SIZE=2 PP_SP=4
export NUM_LAYERS=24 HIDDEN=2048 NUM_ATTN_HEADS=16
export SEQ_LENGTH=8192 MICRO_BATCH=1 GLOBAL_BATCH=8

bash run_deltanet.sh
```

---

## Design Decisions & Trade-offs

1. **Beta projection is not TP-split**: Beta has shape `[b, s, num_heads]` which is
   small (no d dimension). Splitting would add communication overhead for negligible
   memory savings. Each TP rank has its own `nn.Linear(H, heads_per_partition)`.

2. **Short conv state caching**: `ShortConvolution` returns `(output, final_state)`.
   We cache `final_state` as `self.conv_state_q/k/v` and pass it as `cache=` on the
   next split. This maintains causal conv continuity across Seq1F1B splits.

3. **`recurrent_state.detach()`**: We detach the cached state to prevent gradient
   flow across Seq1F1B splits. Each split computes gradients independently.
   This matches how Seq1F1B handles KV cache in the FlashAttention path.

4. **Fallback to fused_recurrent for short sequences**: When `seq_len ≤ 64`, we
   use `fused_recurrent_delta_rule` instead of `chunk_delta_rule` because the
   chunk overhead exceeds the benefit for very short sequences.

5. **No cross-attention / encoder-decoder**: DeltaNet layer only supports
   decoder-only (GPT-style) models. The layer_type is accepted for interface
   compatibility but cross-attention is not implemented.
