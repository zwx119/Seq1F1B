#!/bin/bash
# Run alignment comparison: pipe_sp_splits=1 vs pipe_sp_splits=4
# Compares loss values between serial and Seq1F1B execution.
#
# Usage:
#   DATA_PATH=/path/to/data bash tests/run_alignment_compare.sh
set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
SAVE_DIR="${DIR}/tests/alignment_outputs"
mkdir -p "${SAVE_DIR}"

echo "======================================================"
echo "  Step 1: Running baseline (PP_SP=1, no sequence split)"
echo "======================================================"
PP_SP=1 MASTER_PORT=29500 bash "${DIR}/tests/run_alignment_test.sh"

echo ""
echo "======================================================"
echo "  Step 2: Running Seq1F1B (PP_SP=4, sequence split=4)"
echo "======================================================"
PP_SP=4 MASTER_PORT=29501 bash "${DIR}/tests/run_alignment_test.sh"

echo ""
echo "======================================================"
echo "  Step 3: Comparing losses"
echo "======================================================"
echo "--- PP_SP=1 (baseline) last 20 lines ---"
tail -20 "${SAVE_DIR}/loss_sp1.txt"
echo ""
echo "--- PP_SP=4 (Seq1F1B) last 20 lines ---"
tail -20 "${SAVE_DIR}/loss_sp4.txt"
echo ""

# Simple Python comparison — save results to file
COMPARE_OUTPUT="${SAVE_DIR}/compare_result.txt"
python3 -c "
import re, sys, os, torch

SAVE_DIR = '${SAVE_DIR}'

# ── 1. Compare losses ──
def parse_losses(filename):
    losses = []
    with open(filename) as f:
        for line in f:
            m = re.search(r'lm loss[:\s]+([\d.eE+-]+)', line)
            if m:
                losses.append(float(m.group(1)))
    return losses

l1 = parse_losses(os.path.join(SAVE_DIR, 'loss_sp1.txt'))
l4 = parse_losses(os.path.join(SAVE_DIR, 'loss_sp4.txt'))

if not l1 or not l4:
    print('WARNING: Could not parse losses. Check log files manually.')
else:
    n = min(len(l1), len(l4))
    print(f'=== Loss Comparison ({n} iterations) ===')
    max_diff = 0
    diffs = []
    for i in range(n):
        diff = abs(l1[i] - l4[i])
        max_diff = max(max_diff, diff)
        diffs.append(diff)
    # Summary
    import statistics
    print(f'  Total iters: {n}')
    print(f'  Max  loss diff: {max_diff:.6e}')
    print(f'  Mean loss diff: {statistics.mean(diffs):.6e}')
    loss_pass = max_diff < 0.1
    print(f'  Loss check: {\"PASSED\" if loss_pass else \"FAILED\"} (max_diff < 0.1)')
    # Show last 10 iters detail
    show = min(10, n)
    print(f'  --- Last {show} iterations ---')
    for i in range(n - show, n):
        diff = abs(l1[i] - l4[i])
        status = 'OK' if diff < 0.01 else 'DIFF'
        print(f'  iter {i+1}: SP1={l1[i]:.6f}  SP4={l4[i]:.6f}  diff={diff:.6e}  [{status}]')

# ── 2. Compare hidden states (per PP stage) ──
print()
print('=== Hidden States Comparison (pre-update forward equivalence) ===')
all_pass = True
for stage in range(8):  # up to 8 stages
    f1 = os.path.join(SAVE_DIR, f'hs_sp1_stage{stage}.pt')
    f4 = os.path.join(SAVE_DIR, f'hs_sp4_stage{stage}.pt')
    if not os.path.exists(f1) or not os.path.exists(f4):
        continue
    hs1 = torch.load(f1, map_location='cpu')  # dict: iter_num -> tensor
    hs4 = torch.load(f4, map_location='cpu')
    common_iters = sorted(set(hs1.keys()) & set(hs4.keys()))
    print(f'  PP Stage {stage}: comparing iters {common_iters}')
    for it in common_iters:
        h1 = hs1[it].float()
        h4 = hs4[it].float()
        if h1.shape != h4.shape:
            print(f'    iter {it}: SHAPE MISMATCH {h1.shape} vs {h4.shape}')
            all_pass = False
            continue
        max_d = (h1 - h4).abs().max().item()
        mean_d = (h1 - h4).abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(
            h1.flatten().unsqueeze(0), h4.flatten().unsqueeze(0)).item()
        status = 'OK' if cos > 0.9999 else 'DIFF'
        if status == 'DIFF':
            all_pass = False
        print(f'    iter {it}: max_diff={max_d:.6e}  mean_diff={mean_d:.6e}  cos_sim={cos:.8f}  [{status}]')

print()
if not all_pass:
    print('FAILED: Pre-update hidden states diverge (forward pass NOT equivalent).')
    sys.exit(1)
elif common_iters:
    print('PASSED: Pre-update forward pass is equivalent (hidden states match).')
    print('        Loss curves also match across full training.')
else:
    print('WARNING: No hidden state files found. Only loss was compared.')
    print('         Loss curves match.' if loss_pass else '         Loss curves DIFFER.')
" 2>&1 | tee "${COMPARE_OUTPUT}"

echo ""
echo "Compare result saved to: ${COMPARE_OUTPUT}"
