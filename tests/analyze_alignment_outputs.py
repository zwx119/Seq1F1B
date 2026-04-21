#!/usr/bin/env python3
"""
分析 alignment_outputs 目录：
- 对比 hs_sp1_stageX.pt vs hs_sp4_stageX.pt 以找到每个 PP stage 最早发散的迭代（cos_sim < threshold）
- 尝试加载 layer_stats_{tag}_stage{rank}.pt 并报告 iter1->iter2 输出梯度放大最多的层

用法示例：
  python3 tests/analyze_alignment_outputs.py --dir tests/alignment_outputs --threshold 0.999 --topk 10

输出为终端友好报告，便于下一步定位可疑层。
"""
import argparse
import glob
import os
import torch
import math
from collections import defaultdict


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return float('nan')
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def find_hs_files(d):
    hs_files = defaultdict(dict)  # hs_files[tag][stage]=path
    for p in glob.glob(os.path.join(d, 'hs_*_stage*.pt')):
        base = os.path.basename(p)
        # hs_{tag}_stage{rank}.pt
        parts = base.split('_')
        if len(parts) < 3:
            continue
        tag = '_'.join(parts[1:-1])
        stage_part = parts[-1]
        if not stage_part.startswith('stage'):
            continue
        try:
            stage = int(stage_part.replace('stage', '').replace('.pt',''))
        except Exception:
            continue
        hs_files[tag][stage] = p
    return hs_files


def analyze_hidden_states(d, threshold=0.999):
    hs_files = find_hs_files(d)
    # We expect tags like sp1, sp4, nostate, sp1_noconv, sp4_noconv
    report = {}
    for stage in range(0, 32):
        # find pair sp1 vs sp4 for this stage
        a = hs_files.get('sp1', {}).get(stage)
        b = hs_files.get('sp4', {}).get(stage)
        if not a or not b:
            continue
        ha = torch.load(a, map_location='cpu')
        hb = torch.load(b, map_location='cpu')
        common = sorted(set(ha.keys()) & set(hb.keys()))
        if not common:
            continue
        first_bad = None
        sims = {}
        for it in common:
            aa = ha[it].float()
            bb = hb[it].float()
            if aa.numel() != bb.numel():
                sims[it] = float('nan')
                continue
            sims[it] = cos_sim(aa, bb)
            if it > 1 and first_bad is None and (not math.isnan(sims[it])) and sims[it] < threshold:
                first_bad = it
        report[stage] = {'common_iters': common, 'cos_sims': sims, 'first_bad': first_bad, 'path_sp1': a, 'path_sp4': b}
    return report


def try_extract_grad_stats(obj):
    """
    尝试从加载的 layer_stats 对象中提取每层在各 iter 的输出梯度范数信息。
    支持多种常见命名：layer_grad_stats, layer_grad_norms, grad_stats 等。
    返回 {layer_idx: {iter: value}}
    """
    # If obj is a tensor/dict directly mapping, try to detect structure
    if isinstance(obj, dict):
        # common key names
        for key in ['layer_grad_stats', 'layer_grad_norms', 'layer_grad', 'layer_stats']:
            if key in obj:
                candidate = obj[key]
                # expect candidate to be dict-like
                if isinstance(candidate, dict):
                    return candidate
        # fallback: if keys look like 'layer_0', 'layer_1'
        layer_map = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith('layer'):
                layer_map[k] = v
        if layer_map:
            return layer_map
    # give up
    return None


def analyze_layer_stats(d, tag='sp4', stage=0, topk=10):
    pattern = os.path.join(d, f'layer_stats_{tag}_stage{stage}.pt')
    if not os.path.exists(pattern):
        return None
    obj = torch.load(pattern, map_location='cpu')
    extracted = try_extract_grad_stats(obj)
    if extracted is None:
        return {'raw_keys': list(obj.keys())}
    # normalize to {layer_idx: {iter: val}}
    normalized = {}
    for lk, lv in extracted.items():
        # layer key may be int or '0' or 'layer_0'
        try:
            lid = int(lk) if not isinstance(lk, int) else lk
        except Exception:
            # try to parse digits
            import re
            m = re.search(r'(\d+)', str(lk))
            if m:
                lid = int(m.group(1))
            else:
                lid = str(lk)
        # lv might be dict of iters or a list
        if isinstance(lv, dict):
            normalized[lid] = {int(it): float(val) for it, val in lv.items()}
        elif isinstance(lv, (list, tuple)):
            normalized[lid] = {i+1: float(v) for i, v in enumerate(lv)}
        elif torch.is_tensor(lv):
            arr = lv.cpu().numpy()
            normalized[lid] = {i+1: float(x) for i, x in enumerate(arr)}
        else:
            # single number
            try:
                normalized[lid] = {1: float(lv)}
            except Exception:
                normalized[lid] = { 'raw': lv }

    # compute ratio iter2/iter1 where possible
    ratios = []
    for lid, series in normalized.items():
        if 1 in series and 2 in series and series[1] != 0:
            ratio = series[2] / series[1]
            ratios.append((lid, ratio, series[1], series[2]))
    ratios.sort(key=lambda x: abs(x[1]), reverse=True)
    return {'normalized': normalized, 'ratios_topk': ratios[:topk]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='tests/alignment_outputs')
    p.add_argument('--threshold', type=float, default=0.999)
    p.add_argument('--topk', type=int, default=10)
    args = p.parse_args()

    d = args.dir
    print(f'分析目录: {d}')
    hs_report = analyze_hidden_states(d, threshold=args.threshold)
    if not hs_report:
        print('找不到 hs_* 文件（请确认 alignment_outputs 中存在 hs_sp1_stageX.pt 与 hs_sp4_stageX.pt）')
        return
    print('\n=== Hidden-state divergence summary per PP stage ===')
    for stage, info in sorted(hs_report.items()):
        print(f'PP Stage {stage}: common iters={info["common_iters"]}  first_bad_iter={info["first_bad"]}')
        # print cos sim table compact
        sims = info['cos_sims']
        for it in sorted(sims.keys()):
            print(f'  iter {it}: cos={sims[it]:.8f}')
        # if diverged at iter t, try to analyze layer stats around that stage
        fb = info['first_bad']
        if fb is not None and fb >= 2:
            print(f'  -> Detected divergence at iter {fb}. 尝试加载 layer_stats 做进一步分析...')
            ls = analyze_layer_stats(d, tag='sp4', stage=stage, topk=args.topk)
            if ls is None:
                print('     未找到 layer_stats 文件。')
            elif 'raw_keys' in ls:
                print(f'     layer_stats 文件存在，但无法识别内部结构，文件 keys: {ls["raw_keys"]}')
            else:
                print('     top layer grad ratio (iter2/iter1):')
                for lid, ratio, v1, v2 in ls['ratios_topk']:
                    print(f'       layer {lid}: ratio={ratio:.3f}  grad1={v1:.6e} grad2={v2:.6e}')
        print('')

    print('\n分析完成。下步建议：对报告中 earliest bad iter 所在的 stage，定位 layer_stats 中 ratio 最大的几层，开启更细粒度的 dump（按 head 或按 time-step）来继续调试。')


if __name__ == '__main__':
    main()
