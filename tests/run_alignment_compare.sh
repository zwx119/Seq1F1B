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

# Default to a short debug run (5 iters) unless TRAIN_ITER is explicitly set
TRAIN_ITER=${TRAIN_ITER:-5}

echo "======================================================"
echo "  Step 1: Running baseline (PP_SP=1, no sequence split)"
echo "======================================================"
PP_SP=1 MASTER_PORT=29500 TRAIN_ITER=${TRAIN_ITER} bash "${DIR}/tests/run_alignment_test.sh" "$@"

echo ""
echo "======================================================"
echo "  Step 2: Running Seq1F1B (PP_SP=4, sequence split=4)"
echo "======================================================"
PP_SP=4 MASTER_PORT=29501 TRAIN_ITER=${TRAIN_ITER} bash "${DIR}/tests/run_alignment_test.sh" "$@"

echo ""
echo "======================================================"
echo "  Step 3: Running ablation (PP_SP=4, NO state passing)"
echo "======================================================"
PP_SP=4 DISABLE_STATE_PASSING=1 MASTER_PORT=29502 TRAIN_ITER=${TRAIN_ITER} bash "${DIR}/tests/run_alignment_test.sh" "$@"

echo ""
echo "======================================================"
echo "  Step 4: Running no-conv baseline (PP_SP=1, no short conv)"
echo "======================================================"
PP_SP=1 NO_SHORT_CONV=1 MASTER_PORT=29503 TRAIN_ITER=${TRAIN_ITER} bash "${DIR}/tests/run_alignment_test.sh" "$@"

echo ""
echo "======================================================"
echo "  Step 5: Running no-conv Seq1F1B (PP_SP=4, no short conv)"
echo "======================================================"
PP_SP=4 NO_SHORT_CONV=1 MASTER_PORT=29504 TRAIN_ITER=${TRAIN_ITER} bash "${DIR}/tests/run_alignment_test.sh" "$@"

echo ""
echo "======================================================"
echo "  Step 6: Comparing results"
echo "======================================================"
echo "--- PP_SP=1 (baseline) last 20 lines ---"
tail -20 "${SAVE_DIR}/loss_sp1.txt"
echo ""
echo "--- PP_SP=4 (Seq1F1B) last 20 lines ---"
tail -20 "${SAVE_DIR}/loss_sp4.txt"
echo ""
echo "--- PP_SP=4 no-state (ablation) last 20 lines ---"
tail -20 "${SAVE_DIR}/loss_nostate.txt"
echo ""
echo "--- PP_SP=1 no-conv (baseline) last 20 lines ---"
tail -20 "${SAVE_DIR}/loss_sp1_noconv.txt" 2>/dev/null || echo "(not found)"
echo ""
echo "--- PP_SP=4 no-conv (Seq1F1B) last 20 lines ---"
tail -20 "${SAVE_DIR}/loss_sp4_noconv.txt" 2>/dev/null || echo "(not found)"
echo ""

# Simple Python comparison — save results to file
COMPARE_OUTPUT="${SAVE_DIR}/compare_result.txt"
python3 -c "
import re, sys, os, torch, statistics

SAVE_DIR = '${SAVE_DIR}'

# ── Helper ──
def parse_losses(filename):
    losses = []
    if not os.path.exists(filename):
        return losses
    with open(filename) as f:
        for line in f:
            m = re.search(r'lm loss[:\s]+([\d.eE+-]+)', line)
            if m:
                losses.append(float(m.group(1)))
    return losses

def compare_losses(name_a, la, name_b, lb):
    n = min(len(la), len(lb))
    if n == 0:
        print(f'  WARNING: no losses to compare for {name_a} vs {name_b}')
        return
    diffs = [abs(la[i] - lb[i]) for i in range(n)]
    max_d = max(diffs)
    mean_d = statistics.mean(diffs)
    status = 'PASSED' if max_d < 0.1 else 'FAILED'
    print(f'  {name_a} vs {name_b}: {n} iters, max_diff={max_d:.6e}, mean_diff={mean_d:.6e} [{status}]')
    # Last 5 iters
    for i in range(max(0, n-5), n):
        d = abs(la[i] - lb[i])
        print(f'    iter {i+1}: {name_a}={la[i]:.6f}  {name_b}={lb[i]:.6f}  diff={d:.6e}')
    return max_d

def compare_hidden_states(tag_a, tag_b, label, threshold=0.999):
    print(f'\n--- {label} (threshold={threshold}) ---')
    all_pass = True
    found = False
    iter1_cos_sims = []  # collect iter-1 cos_sims across stages
    for stage in range(8):
        fa = os.path.join(SAVE_DIR, f'hs_{tag_a}_stage{stage}.pt')
        fb = os.path.join(SAVE_DIR, f'hs_{tag_b}_stage{stage}.pt')
        if not os.path.exists(fa) or not os.path.exists(fb):
            continue
        found = True
        ha = torch.load(fa, map_location='cpu')
        hb = torch.load(fb, map_location='cpu')
        common = sorted(set(ha.keys()) & set(hb.keys()))
        print(f'  PP Stage {stage}: iters {common}')
        for it in common:
            a = ha[it].float()
            b = hb[it].float()
            if a.shape != b.shape:
                print(f'    iter {it}: SHAPE MISMATCH {a.shape} vs {b.shape}')
                all_pass = False
                continue
            max_d = (a - b).abs().max().item()
            mean_d = (a - b).abs().mean().item()
            cos = torch.nn.functional.cosine_similarity(
                a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
            # Only check threshold for iter 1 (pre-update)
            if it == 1:
                status = 'OK' if cos > threshold else 'DIFF'
                iter1_cos_sims.append(cos)
                if status == 'DIFF':
                    all_pass = False
            else:
                status = 'info'  # later iters are expected to diverge
            print(f'    iter {it}: max_diff={max_d:.6e}  mean_diff={mean_d:.6e}  cos_sim={cos:.8f}  [{status}]')
    if not found:
        print('  No hidden state files found.')
    if iter1_cos_sims:
        print(f'  iter 1 summary: min_cos={min(iter1_cos_sims):.8f}  mean_cos={statistics.mean(iter1_cos_sims):.8f}')
    return all_pass

# ── 1. Loss comparisons ──
l1 = parse_losses(os.path.join(SAVE_DIR, 'loss_sp1.txt'))
l4 = parse_losses(os.path.join(SAVE_DIR, 'loss_sp4.txt'))
lns = parse_losses(os.path.join(SAVE_DIR, 'loss_nostate.txt'))
l1nc = parse_losses(os.path.join(SAVE_DIR, 'loss_sp1_noconv.txt'))
l4nc = parse_losses(os.path.join(SAVE_DIR, 'loss_sp4_noconv.txt'))

print('=' * 60)
print('=== Loss Comparison ===')
print()
print('[A] SP1 (baseline) vs SP4 (Seq1F1B with state passing):')
compare_losses('SP1', l1, 'SP4', l4)
print()
print('[B] SP1 (baseline) vs SP4-nostate (NO state passing):')
compare_losses('SP1', l1, 'nostate', lns)
print()
print('[C] SP4 (with state) vs SP4-nostate (without state):')
compare_losses('SP4', l4, 'nostate', lns)
print()
print('[D] SP1-noconv vs SP4-noconv (isolate conv state effect):')
compare_losses('SP1nc', l1nc, 'SP4nc', l4nc)

# ── 2. Hidden states comparisons ──
print()
print('=' * 60)
print('=== Hidden States Comparison ===')

pass_a = compare_hidden_states('sp1', 'sp4', 'SP1 vs SP4 (with state passing)', threshold=0.999)
pass_b = compare_hidden_states('sp1', 'nostate', 'SP1 vs SP4-nostate (NO state passing)', threshold=0.999)
pass_c = compare_hidden_states('sp4', 'nostate', 'SP4 vs SP4-nostate', threshold=0.999)
pass_d = compare_hidden_states('sp1_noconv', 'sp4_noconv', 'SP1-noconv vs SP4-noconv (no conv state effect)', threshold=0.999)

# ── 3. Verdict ──
print()
print('=' * 60)
print('=== Verdict ===')
if pass_a:
    print('PASSED: SP1 vs SP4 hidden states match → Seq1F1B forward is equivalent.')
else:
    print('FAILED: SP1 vs SP4 hidden states diverge.')
if not pass_b:
    print('EXPECTED: SP1 vs nostate hidden states differ → state passing IS necessary.')
else:
    print('UNEXPECTED: SP1 vs nostate match → state passing has no effect?!')
if pass_d:
    print('PASSED: SP1-noconv vs SP4-noconv match → conv state detach confirmed as divergence source.')
else:
    print('INFO: SP1-noconv vs SP4-noconv also diverge → other factors at play.')
print()
print('Key insight: If [D] passes but [A] fails, conv state .detach() is the sole remaining issue.')
" 2>&1 | tee "${COMPARE_OUTPUT}"

echo ""
echo "Compare result saved to: ${COMPARE_OUTPUT}"
