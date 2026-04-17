#!/usr/bin/env python3
"""
DeltaNet Alignment Test: verify that Seq1F1B sequence splitting produces
the same hidden states as running the full sequence in one shot.

This test does NOT launch Megatron distributed training. Instead, it:
  1. Directly constructs a DeltaNetAttention module (TP=1, single GPU)
  2. Runs the full sequence through chunk_delta_rule in one call
  3. Runs the same sequence split into N chunks with state passing
  4. Compares the outputs

Usage:
    python tests/test_deltanet_alignment.py
    python tests/test_deltanet_alignment.py --splits 4 --seq-len 2048
"""

import argparse
import sys
import torch
import torch.nn as nn

# We bypass Megatron's full initialization and directly test fla's core logic
# plus the state-passing mechanism from DeltaNetAttention.

try:
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
    from fla.modules import ShortConvolution, RMSNorm
    from einops import rearrange
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install flash-linear-attention einops")
    sys.exit(1)


class SimpleDeltaNet(nn.Module):
    """
    Minimal DeltaNet attention (no TP, no Megatron dependency).
    Mirrors the logic in megatron/model/deltanet_attention.py exactly,
    but without ColumnParallelLinear / RowParallelLinear.
    """

    def __init__(self, hidden_size=256, num_heads=4, head_dim=64,
                 conv_size=4, use_beta=True, qk_norm='l2'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_norm = qk_norm

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Beta
        self.use_beta = use_beta
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Short convolution (matches DeltaNetAttention)
        self.q_conv1d = ShortConvolution(num_heads * head_dim, conv_size, bias=False, activation='silu')
        self.k_conv1d = ShortConvolution(num_heads * head_dim, conv_size, bias=False, activation='silu')
        self.v_conv1d = ShortConvolution(num_heads * head_dim, conv_size, bias=False, activation='silu')

        # Output norm
        self.o_norm = RMSNorm(head_dim, eps=1e-5)

    def forward_full(self, x):
        """
        Run the full sequence in one shot (ground truth).
        x: [B, S, H]
        Returns: [B, S, H]
        """
        B, S, H = x.shape

        q = self.q_proj(x)  # [B, S, num_heads * head_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Short conv (no cache, full sequence)
        q, _ = self.q_conv1d(q, cache=None, output_final_state=False)
        k, _ = self.k_conv1d(k, cache=None, output_final_state=False)
        v, _ = self.v_conv1d(v, cache=None, output_final_state=False)

        # Reshape to multi-head
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
        k = rearrange(k, 'b s (h d) -> b s h d', d=self.head_dim)
        v = rearrange(v, 'b s (h d) -> b s h d', d=self.head_dim)

        # Beta
        if self.use_beta:
            beta = self.b_proj(x).sigmoid()  # [B, S, num_heads]
        else:
            beta = torch.ones(B, S, self.num_heads, device=x.device, dtype=x.dtype)

        # DeltaNet core — full sequence, no initial state
        o, _ = chunk_delta_rule(
            q=q, k=k, v=v, beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=(self.qk_norm == 'l2'),
        )

        # Output
        o = self.o_norm(o)
        o = rearrange(o, 'b s h d -> b s (h d)')
        o = self.o_proj(o)
        return o

    def forward_split(self, x, num_splits):
        """
        Run the sequence split into `num_splits` chunks with state passing.
        This mimics exactly what Seq1F1B does with micro_sp_idx.
        x: [B, S, H]
        Returns: [B, S, H]
        """
        B, S, H = x.shape
        assert S % num_splits == 0, f"S={S} must be divisible by num_splits={num_splits}"
        chunk_len = S // num_splits

        outputs = []
        recurrent_state = None
        conv_state_q = None
        conv_state_k = None
        conv_state_v = None

        for sp_idx in range(num_splits):
            start = sp_idx * chunk_len
            end = start + chunk_len
            x_chunk = x[:, start:end, :]  # [B, chunk_len, H]

            q = self.q_proj(x_chunk)
            k = self.k_proj(x_chunk)
            v = self.v_proj(x_chunk)

            # Short conv with state passing (exactly like DeltaNetAttention)
            use_cache = (sp_idx > 0)
            q, new_conv_q = self.q_conv1d(
                x=q,
                cache=conv_state_q if use_cache else None,
                output_final_state=True,
            )
            k, new_conv_k = self.k_conv1d(
                x=k,
                cache=conv_state_k if use_cache else None,
                output_final_state=True,
            )
            v, new_conv_v = self.v_conv1d(
                x=v,
                cache=conv_state_v if use_cache else None,
                output_final_state=True,
            )

            # Detach conv states (matches DeltaNetAttention behavior)
            conv_state_q = new_conv_q.detach() if new_conv_q is not None else None
            conv_state_k = new_conv_k.detach() if new_conv_k is not None else None
            conv_state_v = new_conv_v.detach() if new_conv_v is not None else None

            # Reshape to multi-head
            q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
            k = rearrange(k, 'b s (h d) -> b s h d', d=self.head_dim)
            v = rearrange(v, 'b s (h d) -> b s h d', d=self.head_dim)

            # Beta
            if self.use_beta:
                beta = self.b_proj(x_chunk).sigmoid()
            else:
                beta = torch.ones(B, chunk_len, self.num_heads, device=x.device, dtype=x.dtype)

            # DeltaNet core — with state passing
            o, new_recurrent_state = chunk_delta_rule(
                q=q, k=k, v=v, beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=(self.qk_norm == 'l2'),
            )
            recurrent_state = new_recurrent_state.detach()

            # Output
            o = self.o_norm(o)
            o = rearrange(o, 'b s h d -> b s (h d)')
            o = self.o_proj(o)

            outputs.append(o)

        # Concatenate all chunks
        return torch.cat(outputs, dim=1)  # [B, S, H]


def test_alignment(batch_size=2, seq_len=1024, hidden_size=256, num_heads=4,
                   num_splits=4, dtype=torch.bfloat16, device='cuda'):
    """Test that split forward matches full forward."""
    head_dim = hidden_size // num_heads

    print(f"\n{'='*70}")
    print(f"Alignment Test: B={batch_size}, S={seq_len}, H={hidden_size}, "
          f"heads={num_heads}, splits={num_splits}, dtype={dtype}")
    print(f"{'='*70}")

    torch.manual_seed(42)
    model = SimpleDeltaNet(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_size=4,
        use_beta=True,
        qk_norm='l2',
    ).to(device=device, dtype=dtype)
    model.eval()

    # Random input
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        # Ground truth: full sequence in one shot
        out_full = model.forward_full(x)

        # Test: split into chunks with state passing
        out_split = model.forward_split(x, num_splits)

    # Compare
    diff = (out_full.float() - out_split.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (out_full.float().abs() + 1e-8)).mean().item()

    # Per-chunk breakdown
    chunk_len = seq_len // num_splits
    print(f"\n  Per-chunk max errors:")
    for i in range(num_splits):
        start = i * chunk_len
        end = start + chunk_len
        chunk_diff = (out_full[:, start:end].float() - out_split[:, start:end].float()).abs()
        print(f"    chunk {i} [{start}:{end}]: max_err={chunk_diff.max().item():.6e}, "
              f"mean_err={chunk_diff.mean().item():.6e}")

    print(f"\n  Overall: max_err={max_err:.6e}, mean_err={mean_err:.6e}, rel_err={rel_err:.6e}")

    # Tolerance: bf16 has ~1e-2 precision, but with recurrent state accumulation
    # we allow up to 1e-1 for long sequences
    threshold = 1e-1
    passed = max_err < threshold
    print(f"  Threshold: {threshold}")
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")

    return passed


def main():
    parser = argparse.ArgumentParser(description='DeltaNet Seq1F1B alignment test')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--splits', type=int, default=4)
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    args = parser.parse_args()

    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    dtype = dtype_map[args.dtype]

    all_pass = True

    # Test 1: basic alignment
    all_pass &= test_alignment(
        batch_size=args.batch_size, seq_len=args.seq_len,
        hidden_size=args.hidden_size, num_heads=args.num_heads,
        num_splits=args.splits, dtype=dtype,
    )

    # Test 2: different split counts
    for splits in [2, 4, 8]:
        if args.seq_len % splits == 0:
            all_pass &= test_alignment(
                batch_size=1, seq_len=args.seq_len,
                hidden_size=args.hidden_size, num_heads=args.num_heads,
                num_splits=splits, dtype=dtype,
            )

    # Test 3: longer sequence
    if args.seq_len <= 2048:
        all_pass &= test_alignment(
            batch_size=1, seq_len=4096,
            hidden_size=args.hidden_size, num_heads=args.num_heads,
            num_splits=args.splits, dtype=dtype,
        )

    # Test 4: split=1 should be identical to full
    all_pass &= test_alignment(
        batch_size=1, seq_len=args.seq_len,
        hidden_size=args.hidden_size, num_heads=args.num_heads,
        num_splits=1, dtype=dtype,
    )

    print(f"\n{'='*70}")
    print(f"{'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
