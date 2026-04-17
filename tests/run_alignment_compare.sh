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
import re, sys

def parse_losses(filename):
    losses = []
    with open(filename) as f:
        for line in f:
            m = re.search(r'lm loss[:\s]+([\d.eE+-]+)', line)
            if m:
                losses.append(float(m.group(1)))
    return losses

l1 = parse_losses('${SAVE_DIR}/loss_sp1.txt')
l4 = parse_losses('${SAVE_DIR}/loss_sp4.txt')

if not l1 or not l4:
    print('WARNING: Could not parse losses. Check log files manually.')
    sys.exit(0)

n = min(len(l1), len(l4))
print(f'Comparing {n} iterations:')
max_diff = 0
for i in range(n):
    diff = abs(l1[i] - l4[i])
    max_diff = max(max_diff, diff)
    status = 'OK' if diff < 0.01 else 'DIFF'
    print(f'  iter {i+1}: SP1={l1[i]:.6f}  SP4={l4[i]:.6f}  diff={diff:.6e}  [{status}]')
print(f'Max loss diff: {max_diff:.6e}')
if max_diff < 0.01:
    print('PASSED: Losses match within tolerance.')
else:
    print('FAILED: Losses diverge.')
    sys.exit(1)
"
