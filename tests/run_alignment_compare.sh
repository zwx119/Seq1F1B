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
echo "--- PP_SP=1 (baseline) ---"
cat "${SAVE_DIR}/loss_sp1.txt"
echo ""
echo "--- PP_SP=4 (Seq1F1B) ---"
cat "${SAVE_DIR}/loss_sp4.txt"
echo ""

# Simple Python comparison
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
    for i in range(n):
        diff = abs(l1[i] - l4[i])
        max_diff = max(max_diff, diff)
        status = 'OK' if diff < 0.01 else 'DIFF'
        print(f'  iter {i+1}: SP1={l1[i]:.6f}  SP4={l4[i]:.6f}  diff={diff:.6e}  [{status}]')
    print(f'  Max loss diff: {max_diff:.6e}')

# ── 2. Compare hidden states (per PP stage) ──
print()
print('=== Hidden States Comparison ===')
all_pass = True
# Find all stage files
for stage in range(8):  # up to 8 stages
    f1 = os.path.join(SAVE_DIR, f'hs_sp1_stage{stage}.pt')
    f4 = os.path.join(SAVE_DIR, f'hs_sp4_stage{stage}.pt')
    if not os.path.exists(f1) or not os.path.exists(f4):
        continue
    hs1 = torch.load(f1, map_location='cpu')
    hs4 = torch.load(f4, map_location='cpu')
    n = min(len(hs1), len(hs4))
    print(f'  PP Stage {stage}: {n} iterations, shape={hs1[0].shape if hs1 else \"?\"} vs {hs4[0].shape if hs4 else \"?\"}')
    for i in range(n):
        h1 = hs1[i].float()
        h4 = hs4[i].float()
        if h1.shape != h4.shape:
            print(f'    iter {i+1}: SHAPE MISMATCH {h1.shape} vs {h4.shape}')
            all_pass = False
            continue
        max_d = (h1 - h4).abs().max().item()
        mean_d = (h1 - h4).abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(
            h1.flatten().unsqueeze(0), h4.flatten().unsqueeze(0)).item()
        status = 'OK' if max_d < 0.05 and cos > 0.999 else 'DIFF'
        if status == 'DIFF':
            all_pass = False
        print(f'    iter {i+1}: max_diff={max_d:.6e}  mean_diff={mean_d:.6e}  cos_sim={cos:.8f}  [{status}]')

print()
if all_pass:
    print('PASSED: Hidden states match across all PP stages.')
else:
    print('FAILED: Hidden states diverge on some stages.')
    sys.exit(1)
"
