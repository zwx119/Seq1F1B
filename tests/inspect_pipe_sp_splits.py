#!/usr/bin/env python3
"""Inspect Seq1F1B sequence split lengths and estimated FLOPs balance.

This mirrors the formulas in `megatron/core/pipeline_parallel/sp_utils.py` and
`split_solver.py` without importing Megatron runtime state. It is intended as a
quick audit tool for `--pipe-sp-strategy hybrid_comp`.
"""

from __future__ import annotations

import argparse
import math
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

    def get_quadratic_layers(self) -> int:
        return self.num_layers if self.softmax_layers is None else self.softmax_layers

    def get_ffn_tflops(self, seqlen: int) -> float:
        return 4 * seqlen * self.hidden_size * self.ffn_size

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
            + embed_proj_tflops
        )
        return tf / 1e12


def round_down(x: int, mod: int) -> int:
    return x // mod * mod


def solve_partition(total_seqlen: int, num_splits: int, config: SeqTFlops, mod: int) -> list[int]:
    """Solve the same monotonic quadratic used by split_solver.py.

    The original code uses sympy. Here we use the closed-form root so this tool
    works even in minimal environments.
    """
    total = config.get_seq_tflops(total_seqlen, causal=True)
    target = total / num_splits
    res: list[int] = []
    prefix = total_seqlen

    linear_per_token = (
        4 * config.hidden_size * config.vocab_size
        + config.num_layers * (
            8 * config.hidden_size * config.num_heads * config.dim_head
            + 4 * config.hidden_size * config.ffn_size
        )
    ) / 1e12
    quad_coeff = config.get_quadratic_layers() * (4 * config.dim_head + 3) * config.num_heads / 1e12

    for _ in range(1, num_splits):
        # cost(x, prefix) = linear*x + quad*(prefix*x - x^2/2) = target
        # 0.5*quad*x^2 - (linear + quad*prefix)*x + target = 0
        a = 0.5 * quad_coeff
        b = -(linear_per_token + quad_coeff * prefix)
        c = target
        if abs(a) < 1e-30:
            root = c / -b
        else:
            disc = max(b * b - 4 * a * c, 0.0)
            r1 = (-b - math.sqrt(disc)) / (2 * a)
            r2 = (-b + math.sqrt(disc)) / (2 * a)
            candidates = [r for r in (r1, r2) if 0 <= r <= prefix]
            root = min(candidates) if candidates else min(r1, r2)
        part = round_down(int(root), mod)
        res.insert(0, part)
        prefix -= part

    res.insert(0, prefix)
    return res


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
    parser.add_argument("--mod", type=int, default=1, help="Round split lengths down to a multiple of this value")
    args = parser.parse_args()

    ffn_hidden_size = args.ffn_hidden_size or 4 * args.hidden_size
    softmax_layers = softmax_layer_count(args)
    config = SeqTFlops(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_size=ffn_hidden_size,
        num_heads=args.num_heads,
        dim_head=args.hidden_size // args.num_heads,
        vocab_size=args.vocab_size,
        softmax_layers=softmax_layers,
    )

    print("Seq1F1B split formula audit")
    print(f"  pattern={args.pattern} softmax_layers={config.get_quadratic_layers()} / {args.num_layers}")
    print(f"  model=L{args.num_layers} H{args.hidden_size} heads={args.num_heads} ffn={ffn_hidden_size}")
    print("  note=cost model balances linear all-layer FLOPs plus causal quadratic softmax-layer FLOPs")

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
