#!/usr/bin/env python3
"""
Compare two runs (e.g. sp1_noconv vs sp4_noconv) per PP stage / layer to
locate the first layer where forward or backward diverges.

Inputs (default dir: tests/alignment_outputs):
  hs_{tag}_stage{rank}.pt          full encoder output hidden states
  layer_outs_{tag}_stage{rank}.pt  per-layer outputs: iter -> layer -> [s,b,h]
  layer_stats_{tag}_stage{rank}.pt per-layer stats: {'forward': ..., 'grad': {iter: {layer: [norms]}}}

Usage:
  python3 tests/analyze_alignment_outputs.py \
      --tag-a sp1_noconv --tag-b sp4_noconv \
      --iters 1 10 --cos-threshold 0.999 --topk 5
"""
import argparse
import math
import os

import torch


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return float('nan')
    return (a @ b).item() / denom


def max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def mean_abs_diff(a, b):
    return (a.float() - b.float()).abs().mean().item()


def find_stage_files(d, pattern):
    out = {}
    for rank in range(32):
        p = os.path.join(d, pattern.replace('{rank}', str(rank)))
        if os.path.exists(p):
            out[rank] = p
    return out


def compare_hidden_states(d, tag_a, tag_b, iters_of_interest, cos_threshold):
    print('\n' + '=' * 70)
    print("=== Encoder output (hs_*): {}  vs  {}".format(tag_a, tag_b))
    print('=' * 70)
    stages_a = find_stage_files(d, 'hs_{}_stage{{rank}}.pt'.format(tag_a))
    stages_b = find_stage_files(d, 'hs_{}_stage{{rank}}.pt'.format(tag_b))
    stages = sorted(set(stages_a) & set(stages_b))
    if not stages:
        print("  (No hs files found for tags {} / {})".format(tag_a, tag_b))
        return
    for s in stages:
        ha = torch.load(stages_a[s], map_location='cpu')
        hb = torch.load(stages_b[s], map_location='cpu')
        common = sorted(set(ha.keys()) & set(hb.keys()))
        print("\n  PP Stage {}: common iters = {}".format(s, common))
        for it in common:
            if iters_of_interest and it not in iters_of_interest:
                continue
            a, b = ha[it], hb[it]
            if a.shape != b.shape:
                print("    iter {}: SHAPE MISMATCH {} vs {}".format(it, a.shape, b.shape))
                continue
            cs = cos_sim(a, b)
            md = max_abs_diff(a, b)
            mn = mean_abs_diff(a, b)
            status = 'OK' if cs > cos_threshold else 'DIFF'
            print("    iter {}: cos_sim={:.8f}  max_diff={:.3e}  mean_diff={:.3e}  [{}]".format(it, cs, md, mn, status))


def compare_layer_outs(d, tag_a, tag_b, iters_of_interest, cos_threshold):
    print('\n' + '=' * 70)
    print("=== Per-layer forward outputs (layer_outs_*): {}  vs  {}".format(tag_a, tag_b))
    print('=' * 70)
    stages_a = find_stage_files(d, 'layer_outs_{}_stage{{rank}}.pt'.format(tag_a))
    stages_b = find_stage_files(d, 'layer_outs_{}_stage{{rank}}.pt'.format(tag_b))
    stages = sorted(set(stages_a) & set(stages_b))
    if not stages:
        print("  (No layer_outs files. Re-run with --dump-layer-stats to generate them.)")
        return

    first_div_global = None
    for s in stages:
        la = torch.load(stages_a[s], map_location='cpu')
        lb = torch.load(stages_b[s], map_location='cpu')
        common_iters = sorted(set(la.keys()) & set(lb.keys()))
        if not common_iters:
            continue
        print("\n  PP Stage {}: iters = {}".format(s, common_iters))

        for it in common_iters:
            if iters_of_interest and it not in iters_of_interest:
                continue
            pla = la[it]
            plb = lb[it]
            common_layers = sorted(set(pla.keys()) & set(plb.keys()))
            print("    --- iter {}: {} layers ---".format(it, len(common_layers)))
            first_bad_layer = None
            for li in common_layers:
                a, b = pla[li], plb[li]
                if a.shape != b.shape:
                    print("      layer {:>3}: SHAPE MISMATCH {} vs {}".format(li, tuple(a.shape), tuple(b.shape)))
                    continue
                cs = cos_sim(a, b)
                md = max_abs_diff(a, b)
                mn = mean_abs_diff(a, b)
                is_bad = (not math.isnan(cs)) and cs < cos_threshold
                st = 'DIFF' if is_bad else 'OK'
                print("      layer {:>3}: cos_sim={:.8f}  max_diff={:.3e}  mean_diff={:.3e}  [{}]".format(li, cs, md, mn, st))
                if is_bad and first_bad_layer is None:
                    first_bad_layer = li

            if first_bad_layer is not None:
                print("    >> Stage {}, iter {}: first diverging layer = layer {}".format(s, it, first_bad_layer))
                if first_div_global is None:
                    first_div_global = (s, first_bad_layer, it)

    if first_div_global is not None:
        s, li, it = first_div_global
        print("\n*** Global first divergence: stage={}, layer={}, iter={} ***".format(s, li, it))
    else:
        print("\n*** All layers cos_sim > {}; no forward divergence detected ***".format(cos_threshold))


def compare_grad_norms(d, tag_a, tag_b, iters_of_interest, topk):
    print('\n' + '=' * 70)
    print("=== Per-layer grad norm ratio (layer_stats_*): {}  vs  {}".format(tag_a, tag_b))
    print('=' * 70)
    stages_a = find_stage_files(d, 'layer_stats_{}_stage{{rank}}.pt'.format(tag_a))
    stages_b = find_stage_files(d, 'layer_stats_{}_stage{{rank}}.pt'.format(tag_b))
    stages = sorted(set(stages_a) & set(stages_b))
    if not stages:
        print("  (No layer_stats files.)")
        return

    for s in stages:
        sa = torch.load(stages_a[s], map_location='cpu')
        sb = torch.load(stages_b[s], map_location='cpu')
        ga = sa.get('grad', {}) if isinstance(sa, dict) else {}
        gb = sb.get('grad', {}) if isinstance(sb, dict) else {}
        common_iters = sorted(set(ga.keys()) & set(gb.keys()))
        if not common_iters:
            print("\n  PP Stage {}: no grad records".format(s))
            continue
        print("\n  PP Stage {}:".format(s))
        for it in common_iters:
            if iters_of_interest and it not in iters_of_interest:
                continue
            la = ga[it]
            lb = gb[it]
            common_layers = sorted(set(la.keys()) & set(lb.keys()))
            rows = []
            for li in common_layers:
                va = la[li][0] if la[li] else None
                vb = lb[li][0] if lb[li] else None
                if va is None or vb is None or va == 0:
                    continue
                ratio = vb / va
                rows.append((li, ratio, va, vb))
            rows.sort(key=lambda x: abs(math.log(abs(x[1]) + 1e-12)), reverse=True)
            print("    iter {}: top-{} layers by |log(ratio_b/a)|".format(it, topk))
            for li, ratio, va, vb in rows[:topk]:
                print("      layer {:>3}: ratio(b/a)={:.4f}  grad_a={:.4e}  grad_b={:.4e}".format(li, ratio, va, vb))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alignment_outputs'))
    p.add_argument('--tag-a', default='sp1_noconv', help='reference run tag')
    p.add_argument('--tag-b', default='sp4_noconv', help='comparison run tag')
    p.add_argument('--iters', type=int, nargs='*', default=[1, 2],
                   help='only analyze these iters (default 1 10)')
    p.add_argument('--cos-threshold', type=float, default=0.999)
    p.add_argument('--topk', type=int, default=5)
    args = p.parse_args()

    print("Analysis dir: {}".format(args.dir))
    print("tag-a (reference) = {}".format(args.tag_a))
    print("tag-b (compared)  = {}".format(args.tag_b))
    print("iters of interest = {}".format(args.iters))

    iters_set = set(args.iters) if args.iters else None

    compare_hidden_states(args.dir, args.tag_a, args.tag_b, iters_set, args.cos_threshold)
    compare_layer_outs(args.dir, args.tag_a, args.tag_b, iters_set, args.cos_threshold)
    compare_grad_norms(args.dir, args.tag_a, args.tag_b, iters_set, args.topk)


if __name__ == '__main__':
    main()
