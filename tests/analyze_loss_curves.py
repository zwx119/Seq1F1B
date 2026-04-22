#!/usr/bin/env python3
"""Parse two Megatron training logs' loss lines and diff the loss curves.

Input: two `loss_{tag}.txt` files, each containing lines like:
    iteration      100/    2000 | ... | lm loss: 1.234E+01 | ...

Output: per-iter loss table + summary statistics:
    - abs_diff = |loss_a - loss_b|
    - rel_diff = |loss_a - loss_b| / loss_a
    - running rel_diff (windowed mean) to see if divergence grows

Fails (exit != 0) if the mean rel_diff over the last 10% of iters exceeds
--fail-threshold (default 2%).
"""
import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

ITER_RE = re.compile(r"iteration\s+(\d+)\s*/\s*\d+")
LOSS_RE = re.compile(r"lm loss:\s*([0-9.eE+\-]+)")


def parse_loss_file(path: str) -> Dict[int, float]:
    result: Dict[int, float] = {}
    if not os.path.exists(path):
        print(f"[WARN] {path} does not exist")
        return result
    with open(path, "r", errors="replace") as f:
        for line in f:
            m_iter = ITER_RE.search(line)
            m_loss = LOSS_RE.search(line)
            if m_iter and m_loss:
                try:
                    result[int(m_iter.group(1))] = float(m_loss.group(1))
                except ValueError:
                    continue
    return result


def summary(name: str, losses: Dict[int, float]) -> None:
    if not losses:
        print(f"  {name}: <empty>")
        return
    iters = sorted(losses.keys())
    print(f"  {name}: {len(iters)} iters  "
          f"[{iters[0]}..{iters[-1]}]  "
          f"loss[{iters[0]}]={losses[iters[0]]:.4f}  "
          f"loss[{iters[-1]}]={losses[iters[-1]]:.4f}")


def compare(
    a: Dict[int, float],
    b: Dict[int, float],
    label_a: str,
    label_b: str,
    output_path: str,
    sample_n: int = 30,
) -> Tuple[float, float]:
    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        print("[ERROR] no common iters between the two runs")
        return float("nan"), float("nan")

    rel_diffs: List[float] = []
    abs_diffs: List[float] = []
    rows = []
    for it in common:
        la, lb = a[it], b[it]
        abs_d = abs(la - lb)
        rel_d = abs_d / max(abs(la), 1e-12)
        abs_diffs.append(abs_d)
        rel_diffs.append(rel_d)
        rows.append((it, la, lb, abs_d, rel_d))

    # Sampled print
    print(f"\n  iter      {label_a:>10}  {label_b:>10}  {'abs_diff':>10}  {'rel_diff':>10}")
    print("  " + "-" * 62)
    step = max(1, len(common) // sample_n)
    for it, la, lb, ad, rd in rows[::step]:
        print(f"  {it:>6}  {la:>10.4f}  {lb:>10.4f}  {ad:>10.4e}  {rd:>10.4e}")
    # Always show the last 3 iters
    for it, la, lb, ad, rd in rows[-3:]:
        print(f"  {it:>6}  {la:>10.4f}  {lb:>10.4f}  {ad:>10.4e}  {rd:>10.4e}")

    # Summary statistics
    n = len(rows)
    mean_rel = sum(rel_diffs) / n
    max_rel = max(rel_diffs)
    last10 = rel_diffs[max(0, int(n * 0.9)):]
    tail_rel = sum(last10) / len(last10)
    first10 = rel_diffs[: max(1, int(n * 0.1))]
    head_rel = sum(first10) / len(first10)

    print("\n  ── Summary ──────────────────────────────────────────────")
    print(f"  common iters           : {n}")
    print(f"  mean rel_diff          : {mean_rel:.4e}")
    print(f"  max  rel_diff          : {max_rel:.4e}")
    print(f"  rel_diff first 10%     : {head_rel:.4e}")
    print(f"  rel_diff last  10%     : {tail_rel:.4e}")
    ratio = tail_rel / max(head_rel, 1e-12)
    print(f"  tail / head ratio      : {ratio:.2f}x  "
          f"({'GROWING' if ratio > 2.0 else 'stable'})")

    # Save full table
    if output_path:
        with open(output_path, "w") as f:
            f.write(f"# {label_a} vs {label_b}\n")
            f.write(f"# iter, {label_a}, {label_b}, abs_diff, rel_diff\n")
            for it, la, lb, ad, rd in rows:
                f.write(f"{it}, {la:.6f}, {lb:.6f}, {ad:.6e}, {rd:.6e}\n")
        print(f"\n  Full table written to {output_path}")

    return mean_rel, tail_rel


def try_plot(a: Dict[int, float], b: Dict[int, float], label_a: str,
             label_b: str, plot_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not installed, skipping plot")
        return

    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        return
    la = [a[it] for it in common]
    lb = [b[it] for it in common]
    rel = [abs(la[i] - lb[i]) / max(abs(la[i]), 1e-12) for i in range(len(common))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(common, la, label=label_a, linewidth=1.2)
    ax1.plot(common, lb, label=label_b, linewidth=1.2, linestyle="--")
    ax1.set_ylabel("lm loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title(f"Loss curve: {label_a} vs {label_b}")

    ax2.plot(common, rel, color="red", linewidth=1.0)
    ax2.set_ylabel("rel_diff")
    ax2.set_xlabel("iteration")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print(f"  [plot] saved to {plot_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loss-a", required=True)
    p.add_argument("--loss-b", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--output", default="")
    p.add_argument("--plot", default="")
    p.add_argument("--fail-threshold", type=float, default=0.02,
                   help="Exit != 0 if tail mean rel_diff exceeds this")
    args = p.parse_args()

    print(f"Loss-A: {args.loss_a}")
    print(f"Loss-B: {args.loss_b}")
    a = parse_loss_file(args.loss_a)
    b = parse_loss_file(args.loss_b)

    print("\n── Raw file summary ──────────────────────────────────────")
    summary(args.label_a, a)
    summary(args.label_b, b)

    mean_rel, tail_rel = compare(a, b, args.label_a, args.label_b, args.output)

    if args.plot:
        try_plot(a, b, args.label_a, args.label_b, args.plot)

    # Verdict
    print("\n══════════════════════════════════════════════════════════")
    if tail_rel != tail_rel:  # NaN
        print("  VERDICT: could not compute (no common iters)")
        sys.exit(2)
    if tail_rel < args.fail_threshold:
        print(f"  VERDICT: ✓ loss curves aligned "
              f"(tail rel_diff {tail_rel:.2e} < threshold {args.fail_threshold})")
        sys.exit(0)
    else:
        print(f"  VERDICT: ✗ loss curves diverged "
              f"(tail rel_diff {tail_rel:.2e} ≥ threshold {args.fail_threshold})")
        sys.exit(1)


if __name__ == "__main__":
    main()
