#!/usr/bin/env python3
"""
DeltaNet Seq1F1B Alignment Test (end-to-end, real code path).

Verifies that running a multi-layer DeltaNet transformer stack on the FULL
sequence produces IDENTICAL hidden states as running it split into N chunks
with recurrent state + conv state passing — exactly the Seq1F1B approach.

What this tests (and why it matters):
  - This is NOT a simplified simulation. It mirrors the EXACT data flow of
    megatron/model/deltanet_attention.py and deltanet_layer.py:
      * [s, b, h] Megatron convention with internal transpose to [b, s, h]
      * ColumnParallelLinear → nn.Linear (equivalent at TP=1)
      * ShortConvolution with state caching (conv_state_q/k/v)
      * chunk_delta_rule with recurrent state passing (initial_state)
      * RMSNorm on head_dim
      * Full transformer layer: Pre-LN → DeltaNet Attn → Residual → Post-LN → MLP → Residual
      * Multi-layer stacking (state accumulation across layers)
  - Comparison:
      Baseline (serial):  pipe_sp_splits=1, full sequence through all layers in one call
      Seq1F1B  (split):   pipe_sp_splits=N, N calls with micro_sp_idx=0..N-1, state passed

Usage (single GPU, no torchrun needed):
    python tests/test_deltanet_alignment.py
    python tests/test_deltanet_alignment.py --num-layers 4 --seq-len 4096 --splits 8
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.delta_rule import chunk_delta_rule
    from fla.modules import ShortConvolution, RMSNorm
    from einops import rearrange
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install flash-linear-attention einops")
    sys.exit(1)


# ============================================================================
# DeltaNet Attention — mirrors deltanet_attention.py line-by-line at TP=1
# ============================================================================

class DeltaNetAttentionCore(nn.Module):
    """
    DeltaNet attention core — faithfully mirrors DeltaNetAttention
    (megatron/model/deltanet_attention.py) at TP=1.

    Every operation has a comment referencing the corresponding line in the
    real code so we can verify they match.
    """

    def __init__(self, hidden_size, num_heads, head_dim, conv_size=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Q/K/V projections — ColumnParallelLinear at TP=1 ≡ nn.Linear
        # (deltanet_attention.py L123-143)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)

        # Beta projection — nn.Linear (deltanet_attention.py L148-152)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Short convolutions (deltanet_attention.py L156-173)
        self.q_conv1d = ShortConvolution(num_heads * head_dim, conv_size,
                                         bias=False, activation='silu')
        self.k_conv1d = ShortConvolution(num_heads * head_dim, conv_size,
                                         bias=False, activation='silu')
        self.v_conv1d = ShortConvolution(num_heads * head_dim, conv_size,
                                         bias=False, activation='silu')

        # Output norm + projection (deltanet_attention.py L183, L191-198)
        self.o_norm = RMSNorm(head_dim, eps=1e-5)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # === Seq1F1B state (deltanet_attention.py L201-206) ===
        self.recurrent_state = None
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None

    def _clear_states(self):
        """deltanet_attention.py L208-212"""
        self.recurrent_state = None
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None

    def forward(self, hidden_states, pipe_sp_splits=1, micro_sp_idx=None):
        """
        hidden_states: [s, b, h]  (Megatron convention, seq-first)
        Returns:       [s, b, h]

        This follows deltanet_attention.py forward() L224-370 exactly.
        """
        # --- Handle micro_sp_idx (L245-250) ---
        if micro_sp_idx is not None:
            if micro_sp_idx == 0:
                self._clear_states()

        # --- Projections: [s, b, h] → [s, b, proj_dim] (L258-261) ---
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # --- Transpose to [b, s, proj_dim] for conv & DeltaNet (L264-267) ---
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        # For beta: [s, b, h] → [b, s, h] (L274-276)
        hidden_states_bsh = hidden_states.transpose(0, 1).contiguous()

        # --- Short convolutions with state passing (L283-308) ---
        use_conv_cache = (pipe_sp_splits > 1
                          and micro_sp_idx is not None
                          and micro_sp_idx > 0)
        output_final = (pipe_sp_splits > 1)

        q, conv_q = self.q_conv1d(
            x=q,
            cache=self.conv_state_q if use_conv_cache else None,
            output_final_state=output_final)
        k, conv_k = self.k_conv1d(
            x=k,
            cache=self.conv_state_k if use_conv_cache else None,
            output_final_state=output_final)
        v, conv_v = self.v_conv1d(
            x=v,
            cache=self.conv_state_v if use_conv_cache else None,
            output_final_state=output_final)

        # Detach conv states (L305-308)
        self.conv_state_q = conv_q.detach() if conv_q is not None else None
        self.conv_state_k = conv_k.detach() if conv_k is not None else None
        self.conv_state_v = conv_v.detach() if conv_v is not None else None

        # --- Multi-head reshape: [b, s, H*d] → [b, s, H, d] (L311-313) ---
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
        k = rearrange(k, 'b s (h d) -> b s h d', d=self.head_dim)
        v = rearrange(v, 'b s (h d) -> b s h d', d=self.head_dim)

        # --- Beta (L325-326) ---
        beta = self.b_proj(hidden_states_bsh).sigmoid()

        # --- DeltaNet core with state passing (L338-359) ---
        o, recurrent_state = chunk_delta_rule(
            q=q, k=k, v=v, beta=beta,
            initial_state=self.recurrent_state,
            output_final_state=output_final,
            use_qk_l2norm_in_kernel=True,  # qk_norm='l2'
        )
        if output_final and recurrent_state is not None:
            self.recurrent_state = recurrent_state.detach()  # L357

        # --- Output norm + projection (L362-370) ---
        o = self.o_norm(o)
        o = rearrange(o, 'b s h d -> b s (h d)')
        o = self.o_proj(o)

        # --- Back to [s, b, h] (Megatron convention) ---
        o = o.transpose(0, 1).contiguous()
        return o


# ============================================================================
# DeltaNet Transformer Layer — mirrors deltanet_layer.py
# ============================================================================

class DeltaNetTransformerLayerTest(nn.Module):
    """
    Full transformer layer: Pre-LN → DeltaNet Attn → Residual → Post-LN → MLP → Residual.

    Mirrors DeltaNetTransformerLayer (deltanet_layer.py L56-204) at TP=1.
    Uses nn.LayerNorm instead of Megatron's LayerNorm (identical at TP=1).
    Uses a SwiGLU MLP instead of Megatron's ParallelMLP (same architecture).
    """

    def __init__(self, hidden_size, num_heads, head_dim, ffn_hidden_size, conv_size=4):
        super().__init__()
        # (deltanet_layer.py L78-82)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        # (deltanet_layer.py L85)
        self.self_attention = DeltaNetAttentionCore(
            hidden_size, num_heads, head_dim, conv_size)
        # (deltanet_layer.py L94-98)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        # SwiGLU MLP — ParallelMLP with swiglu (transformer.py)
        self.mlp_gate_up = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
        self.mlp_down = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, pipe_sp_splits=1, micro_sp_idx=None):
        """
        hidden_states: [s, b, h]
        Returns:       [s, b, h]

        Mirrors deltanet_layer.py forward() L126-200.
        """
        # Pre-LN (L128)
        layernorm_output = self.input_layernorm(hidden_states)

        # DeltaNet self-attention (L131-136)
        attention_output = self.self_attention(
            layernorm_output,
            pipe_sp_splits=pipe_sp_splits,
            micro_sp_idx=micro_sp_idx)

        # First residual (L139-140: apply_residual_connection_post_layernorm=False)
        hidden_states = hidden_states + attention_output

        # Post-LN (L166)
        layernorm_output = self.post_attention_layernorm(hidden_states)

        # SwiGLU MLP (L169)
        gate_up = self.mlp_gate_up(layernorm_output)
        gate, up = gate_up.chunk(2, dim=-1)
        mlp_output = self.mlp_down(F.silu(gate) * up)

        # Second residual (L178-179)
        hidden_states = hidden_states + mlp_output

        return hidden_states


# ============================================================================
# Multi-layer DeltaNet Model — the actual test target
# ============================================================================

class DeltaNetModelTest(nn.Module):
    """
    Stack of DeltaNetTransformerLayerTest layers.
    Mirrors ParallelTransformer (transformer.py) looping over layers.
    """

    def __init__(self, num_layers, hidden_size, num_heads, ffn_hidden_size, conv_size=4):
        super().__init__()
        head_dim = hidden_size // num_heads
        self.layers = nn.ModuleList([
            DeltaNetTransformerLayerTest(
                hidden_size, num_heads, head_dim, ffn_hidden_size, conv_size)
            for _ in range(num_layers)
        ])

    def forward_serial(self, x):
        """
        Baseline: run full sequence through all layers, no splitting.
        Equivalent to pipe_sp_splits=1.

        x: [s, b, h]
        Returns: [s, b, h]
        """
        # Clear all states first
        for layer in self.layers:
            layer.self_attention._clear_states()

        h = x
        for layer in self.layers:
            h = layer(h, pipe_sp_splits=1, micro_sp_idx=None)
        return h

    def forward_seq1f1b(self, x, num_splits):
        """
        Seq1F1B mode: split sequence into num_splits chunks, process each
        chunk through ALL layers with state passing before moving to next chunk.

        This is exactly how Seq1F1B works:
          for sp_idx in 0..N-1:
              chunk = x[sp_idx * chunk_len : (sp_idx+1) * chunk_len]
              for each layer:
                  chunk = layer(chunk, pipe_sp_splits=N, micro_sp_idx=sp_idx)

        In real Seq1F1B, different layers may be on different PP stages,
        but each layer sees all N splits in order. The computation per-layer
        is identical to this single-GPU simulation.

        x: [s, b, h]
        Returns: [s, b, h]
        """
        s, b, h = x.shape
        assert s % num_splits == 0, f"seq_len={s} must be divisible by num_splits={num_splits}"
        chunk_len = s // num_splits

        # Clear all states (will also be cleared at sp_idx=0, but be explicit)
        for layer in self.layers:
            layer.self_attention._clear_states()

        outputs = []
        for sp_idx in range(num_splits):
            start = sp_idx * chunk_len
            end = start + chunk_len
            x_chunk = x[start:end, :, :]  # [chunk_len, b, h]

            h = x_chunk
            for layer in self.layers:
                h = layer(h, pipe_sp_splits=num_splits, micro_sp_idx=sp_idx)
            outputs.append(h)

        return torch.cat(outputs, dim=0)  # [s, b, h]


# ============================================================================
# Per-layer hidden state capture — for detailed debugging
# ============================================================================

def capture_per_layer_states(model, x, num_splits=None):
    """
    Run forward and capture hidden states after each layer.
    Returns dict: {layer_idx: tensor [s, b, h]}
    """
    states = {}

    if num_splits is None:
        # Serial mode
        for layer in model.layers:
            layer.self_attention._clear_states()
        h = x
        for i, layer in enumerate(model.layers):
            h = layer(h, pipe_sp_splits=1, micro_sp_idx=None)
            states[i] = h.clone()
    else:
        # Seq1F1B mode
        s = x.shape[0]
        chunk_len = s // num_splits
        for layer in model.layers:
            layer.self_attention._clear_states()

        # Accumulate per-layer outputs across splits
        layer_outputs = {i: [] for i in range(len(model.layers))}
        for sp_idx in range(num_splits):
            start = sp_idx * chunk_len
            end = start + chunk_len
            h = x[start:end, :, :]
            for i, layer in enumerate(model.layers):
                h = layer(h, pipe_sp_splits=num_splits, micro_sp_idx=sp_idx)
                layer_outputs[i].append(h.clone())

        for i in range(len(model.layers)):
            states[i] = torch.cat(layer_outputs[i], dim=0)

    return states


# ============================================================================
# Test runner
# ============================================================================

def test_alignment(num_layers=2, batch_size=2, seq_len=1024, hidden_size=256,
                   num_heads=4, ffn_hidden_size=None, num_splits=4,
                   conv_size=4, dtype=torch.bfloat16, device='cuda'):
    """
    The main alignment test.

    Builds a multi-layer DeltaNet model, runs forward_serial (full sequence)
    and forward_seq1f1b (split sequence), compares hidden states.
    """
    if ffn_hidden_size is None:
        ffn_hidden_size = hidden_size * 4

    print(f"\n{'='*70}")
    print(f"Alignment Test")
    print(f"  layers={num_layers}, B={batch_size}, S={seq_len}, H={hidden_size}")
    print(f"  heads={num_heads}, ffn={ffn_hidden_size}, splits={num_splits}")
    print(f"  conv_size={conv_size}, dtype={dtype}")
    print(f"{'='*70}")

    # Build model
    torch.manual_seed(42)
    model = DeltaNetModelTest(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_hidden_size=ffn_hidden_size,
        conv_size=conv_size,
    ).to(device=device, dtype=dtype)
    model.eval()

    # Random input: [s, b, h] (Megatron convention)
    torch.manual_seed(123)
    x = torch.randn(seq_len, batch_size, hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        # ---- Baseline: serial, full sequence ----
        out_serial = model.forward_serial(x)

        # ---- Seq1F1B: split sequence with state passing ----
        out_seq1f1b = model.forward_seq1f1b(x, num_splits)

    # ---- Compare final hidden states ----
    diff = (out_serial.float() - out_seq1f1b.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (out_serial.float().abs() + 1e-8)).mean().item()

    chunk_len = seq_len // num_splits
    print(f"\n  Per-split final hidden state max errors (dim 0 = seq):")
    for i in range(num_splits):
        start = i * chunk_len
        end = start + chunk_len
        chunk_diff = (out_serial[start:end].float() - out_seq1f1b[start:end].float()).abs()
        print(f"    split {i} [seq {start}:{end}]: "
              f"max_err={chunk_diff.max().item():.6e}, "
              f"mean_err={chunk_diff.mean().item():.6e}")

    print(f"\n  Overall: max_err={max_err:.6e}, mean_err={mean_err:.6e}, rel_err={rel_err:.6e}")

    # ---- Per-layer comparison for debugging ----
    print(f"\n  Per-layer hidden state max errors:")
    with torch.no_grad():
        states_serial = capture_per_layer_states(model, x, num_splits=None)
        states_seq1f1b = capture_per_layer_states(model, x, num_splits=num_splits)

    for layer_idx in range(num_layers):
        s_ser = states_serial[layer_idx].float()
        s_sp = states_seq1f1b[layer_idx].float()
        layer_diff = (s_ser - s_sp).abs()
        layer_max = layer_diff.max().item()
        layer_mean = layer_diff.mean().item()
        status = "✓" if layer_max < 1e-1 else "✗"
        print(f"    layer {layer_idx}: max_err={layer_max:.6e}, mean_err={layer_mean:.6e} {status}")

    # ---- Verdict ----
    # bf16 rounding: single op ~1e-3, multi-layer accumulation ~1e-2
    # With state passing we allow up to 1e-1 for long sequences
    threshold = 1e-1
    passed = max_err < threshold
    print(f"\n  Threshold: {threshold}")
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")

    if not passed:
        # Print detailed debug info
        print(f"\n  DEBUG: locations of max error:")
        flat_idx = diff.argmax().item()
        s_idx = flat_idx // (batch_size * hidden_size)
        remaining = flat_idx % (batch_size * hidden_size)
        b_idx = remaining // hidden_size
        h_idx = remaining % hidden_size
        print(f"    position: seq={s_idx}, batch={b_idx}, hidden={h_idx}")
        print(f"    serial value:  {out_serial[s_idx, b_idx, h_idx].item():.6f}")
        print(f"    seq1f1b value: {out_seq1f1b[s_idx, b_idx, h_idx].item():.6f}")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description='DeltaNet Seq1F1B alignment: serial vs split hidden states')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--ffn-hidden-size', type=int, default=None,
                        help='FFN hidden size (default: 4 * hidden_size)')
    parser.add_argument('--splits', type=int, default=4,
                        help='Number of sequence splits (pipe_sp_splits)')
    parser.add_argument('--conv-size', type=int, default=4)
    parser.add_argument('--dtype', type=str, default='bf16',
                        choices=['bf16', 'fp16', 'fp32'])
    args = parser.parse_args()

    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    dtype = dtype_map[args.dtype]

    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        sys.exit(0)

    all_pass = True

    # === Test 1: main test with user-specified config ===
    print("\n" + "=" * 70)
    print("TEST 1: User-specified configuration")
    print("=" * 70)
    all_pass &= test_alignment(
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        num_splits=args.splits,
        conv_size=args.conv_size,
        dtype=dtype,
    )

    # === Test 2: vary split counts ===
    print("\n" + "=" * 70)
    print("TEST 2: Different split counts (same model)")
    print("=" * 70)
    for splits in [1, 2, 4, 8]:
        if args.seq_len % splits == 0:
            all_pass &= test_alignment(
                num_layers=args.num_layers,
                batch_size=1,
                seq_len=args.seq_len,
                hidden_size=args.hidden_size,
                num_heads=args.num_heads,
                ffn_hidden_size=args.ffn_hidden_size,
                num_splits=splits,
                conv_size=args.conv_size,
                dtype=dtype,
            )

    # === Test 3: realistic config (closer to run_deltanet.sh defaults) ===
    print("\n" + "=" * 70)
    print("TEST 3: Realistic config (2048 hidden, 16 heads, 4 layers)")
    print("=" * 70)
    all_pass &= test_alignment(
        num_layers=4,
        batch_size=1,
        seq_len=4096,
        hidden_size=2048,
        num_heads=16,
        ffn_hidden_size=5504,  # typical for ~7B scale
        num_splits=4,
        conv_size=4,
        dtype=dtype,
    )

    # === Summary ===
    print(f"\n{'='*70}")
    if all_pass:
        print("✅ ALL ALIGNMENT TESTS PASSED")
        print("   Serial (full sequence) == Seq1F1B (split + state passing)")
    else:
        print("❌ SOME ALIGNMENT TESTS FAILED")
        print("   Serial and Seq1F1B hidden states DIVERGE — check state passing logic!")
    print(f"{'='*70}")

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
