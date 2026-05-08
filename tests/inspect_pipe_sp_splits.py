#!/usr/bin/env python3
"""Inspect Seq1F1B sequence split lengths and estimated FLOPs balance.

This mirrors the formulas in `megatron/core/pipeline_parallel/sp_utils.py` and
`split_solver.py` without importing Megatron runtime state. It is intended as a
quick audit tool for `--pipe-sp-strategy hybrid_comp`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class SeqTFlops:
    num_layers: int
    hidden_size: int
    ffn_size: int
    num_heads: int
    dim_head: int
    vocab_size: int
    softmax_layers: int | None = None
    deltanet_layers: int = 0
    deltanet_head_dim: int | None = None
    deltanet_chunk_size: int = 64
    deltanet_conv_size: int = 4
    deltanet_use_beta: bool = True
    deltanet_use_short_conv: bool = True
    deltanet_use_output_gate: bool = False

    def get_quadratic_layers(self) -> int:
        return self.num_layers if self.softmax_layers is None else self.softmax_layers

    def get_ffn_tflops(self, seqlen: int) -> float:
        return 4 * seqlen * self.hidden_size * self.ffn_size

    def get_deltanet_core_tflops(self, seqlen: int) -> float:
        if self.deltanet_layers <= 0:
            return 0.0

        head_dim = self.deltanet_head_dim or self.dim_head
        key_dim = head_dim
        value_dim = head_dim
        block = self.deltanet_chunk_size

        kkt = 2 * seqlen * block * self.num_heads * key_dim
        wy = 2 * seqlen * block * self.num_heads * (key_dim + value_dim)
        state_and_output = 4 * seqlen * self.num_heads * key_dim * value_dim

        short_conv = 0
        if self.deltanet_use_short_conv:
            short_conv = (
                2 * seqlen * self.deltanet_conv_size * self.num_heads
                * (2 * key_dim + value_dim)
            )

        beta = 0
        if self.deltanet_use_beta:
            beta = 2 * seqlen * self.hidden_size * self.num_heads

        gate = 0
        if self.deltanet_use_output_gate:
            gate = 2 * seqlen * self.hidden_size * self.hidden_size

        return kkt + wy + state_and_output + short_conv + beta + gate

    def get_deltanet_core_tflops_per_token(self) -> float:
        return self.get_deltanet_core_tflops(1)

    def get_emb_tflops(self, seqlen: int) -> tuple[float, float]:
        embed_tflops = 2 * seqlen * self.hidden_size * self.vocab_size
        embed_proj_tflops = 2 * seqlen * self.hidden_size * self.vocab_size
        return embed_tflops, embed_proj_tflops

    def get_seq_tflops(self, seqlen: int, causal: bool = True) -> float:
        scale = 0.5 if causal else 1
        embed_tflops, embed_proj_tflops = self.get_emb_tflops(seqlen)
        ffn_tflops = self.get_ffn_tflops(seqlen)
        attn_proj_tflops = 2 * seqlen * 3 * self.hidden_size * (self.dim_head * self.num_heads)
        attn_qk_tflops = 2 * seqlen * seqlen * self.dim_head * self.num_heads * scale
        attn_softmax_tflops = (
            3 * seqlen * seqlen * self.num_heads
            + 2 * seqlen * seqlen * self.num_heads * self.dim_head
        ) * scale
        attn_o_proj_tflops = 2 * seqlen * self.hidden_size * (self.dim_head * self.num_heads)
        attn_linear = attn_proj_tflops + attn_o_proj_tflops
        attn_quadratic = attn_qk_tflops + attn_softmax_tflops
        total = (
            embed_tflops
            + self.num_layers * (attn_linear + ffn_tflops)
            + self.get_quadratic_layers() * attn_quadratic
            + self.deltanet_layers * self.get_deltanet_core_tflops(seqlen)
            + embed_proj_tflops
        )
        return total / 1e12

    def get_prefix_tflops(self, seqlen: int, prefix: int) -> float:
        attn_quadratic = (
            seqlen * prefix * (self.dim_head * 4 + 3) * self.num_heads
            - seqlen**2 * (4 * self.dim_head + 3) * self.num_heads / 2
        )
        attn_linear = seqlen * 8 * self.hidden_size * self.num_heads * self.dim_head
        ffn_tflops = self.get_ffn_tflops(seqlen)
        embed_tflops, embed_proj_tflops = self.get_emb_tflops(seqlen)
        tf = (
            embed_tflops
            + self.num_layers * (attn_linear + ffn_tflops)
            + self.get_quadratic_layers() * attn_quadratic
            + self.deltanet_layers * self.get_deltanet_core_tflops(seqlen)
            + embed_proj_tflops
        )
        return tf / 1e12


def round_down(x: int, mod: int) -> int:
    return x // mod * mod


def solve_partition(total_seqlen: int, num_splits: int, config: SeqTFlops, mod: int) -> list[int]:
    """Solve the same aligned imbalance minimization used by split_solver.py."""
    total = config.get_seq_tflops(total_seqlen, causal=True)
    target = total / num_splits
    if mod <= 1 or total_seqlen % mod != 0:
        raise ValueError("--mod must divide every inspected sequence length")

    units = total_seqlen // mod
    if units < num_splits:
        raise ValueError(
            f"Cannot split sequence length {total_seqlen} into {num_splits} "
            f"positive chunks aligned to {mod}."
        )

    inf = float("inf")
    dp = [[inf] * (units + 1) for _ in range(num_splits + 1)]
    prev = [[-1] * (units + 1) for _ in range(num_splits + 1)]
    dp[0][0] = 0.0

    for k in range(1, num_splits + 1):
        min_end = k
        max_end = units - (num_splits - k)
        for end in range(min_end, max_end + 1):
            for start in range(k - 1, end):
                if dp[k - 1][start] == inf:
                    continue
                length = (end - start) * mod
                prefix = end * mod
                cost = config.get_prefix_tflops(length, prefix)
                score = max(dp[k - 1][start], abs(cost - target))
                if score < dp[k][end]:
                    dp[k][end] = score
                    prev[k][end] = start

    splits: list[int] = []
    end = units
    for k in range(num_splits, 0, -1):
        start = prev[k][end]
        if start < 0:
            raise RuntimeError("Failed to solve aligned sequence partition.")
        splits.insert(0, (end - start) * mod)
        end = start
    return splits


def parse_layers(spec: str) -> set[int]:
    layers: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            layers.update(range(int(start_s), int(end_s) + 1))
        else:
            layers.add(int(item))
    return layers


def softmax_layer_count(args: argparse.Namespace) -> int | None:
    if args.pattern == "all":
        return None
    if args.pattern == "global2":
        layers = parse_layers(args.global_layers)
    elif args.pattern == "period4":
        layers = {i for i in range(1, args.num_layers + 1) if (i - args.period_offset) % args.period == 0}
    else:
        raise ValueError(args.pattern)
    return len([i for i in layers if 1 <= i <= args.num_layers])


def chunk_costs(config: SeqTFlops, splits: list[int]) -> list[float]:
    costs = []
    prefix = 0
    for length in splits:
        prefix += length
        costs.append(config.get_prefix_tflops(length, prefix))
    return costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-list", default="16384,24576,32768")
    parser.add_argument("--sp-list", default="4,8")
    parser.add_argument("--pattern", choices=["global2", "period4", "all"], default="global2")
    parser.add_argument("--global-layers", default="2,15")
    parser.add_argument("--period", type=int, default=4)
    parser.add_argument("--period-offset", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--ffn-hidden-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--deltanet-head-dim", type=int, default=None)
    parser.add_argument("--deltanet-chunk-size", type=int, default=64)
    parser.add_argument("--deltanet-conv-size", type=int, default=4)
    parser.add_argument("--no-deltanet-core", action="store_true")
    parser.add_argument("--no-deltanet-beta", action="store_true")
    parser.add_argument("--no-deltanet-short-conv", action="store_true")
    parser.add_argument("--no-deltanet-output-gate", action="store_true")
    parser.add_argument("--mod", type=int, default=128, help="Round split lengths down to a multiple of this value")
    args = parser.parse_args()

    ffn_hidden_size = args.ffn_hidden_size or 4 * args.hidden_size
    softmax_layers = softmax_layer_count(args)
    deltanet_layers = 0 if args.no_deltanet_core else args.num_layers - (softmax_layers or args.num_layers)
    config = SeqTFlops(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_size=ffn_hidden_size,
        num_heads=args.num_heads,
        dim_head=args.hidden_size // args.num_heads,
        vocab_size=args.vocab_size,
        softmax_layers=softmax_layers,
        deltanet_layers=max(deltanet_layers, 0),
        deltanet_head_dim=args.deltanet_head_dim or args.hidden_size // args.num_heads,
        deltanet_chunk_size=args.deltanet_chunk_size,
        deltanet_conv_size=args.deltanet_conv_size,
        deltanet_use_beta=not args.no_deltanet_beta,
        deltanet_use_short_conv=not args.no_deltanet_short_conv,
        deltanet_use_output_gate=not args.no_deltanet_output_gate,
    )

    print("Seq1F1B split formula audit")
    print(f"  pattern={args.pattern} softmax_layers={config.get_quadratic_layers()} / {args.num_layers}")
    print(f"  deltanet_layers={config.deltanet_layers} include_core={not args.no_deltanet_core}")
    print(f"  model=L{args.num_layers} H{args.hidden_size} heads={args.num_heads} ffn={ffn_hidden_size}")
    print(f"  align={args.mod}")
    print("  note=cost model balances common linear FLOPs, DeltaNet linear core, and causal quadratic softmax FLOPs")

    for seq in [int(x) for x in args.seq_list.split(",") if x.strip()]:
        for sp in [int(x) for x in args.sp_list.split(",") if x.strip()]:
            avg = [seq // sp] * sp
            hcomp = solve_partition(seq, sp, config, args.mod)
            for name, splits in (("average", avg), ("hybrid_comp", hcomp)):
                costs = chunk_costs(config, splits)
                total = sum(costs)
                full = config.get_seq_tflops(seq, causal=True)
                target = full / sp
                max_imbalance = max(abs(c - target) / target for c in costs)
                print(
                    f"seq={seq:<5} sp={sp:<2} {name:<11} splits={splits} "
                    f"sum={sum(splits)} cost_sum/full={total / full:.6f} "
                    f"max_imbalance={max_imbalance * 100:.3f}%"
                )


if __name__ == "__main__":
    main()
