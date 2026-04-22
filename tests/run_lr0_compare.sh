#!/bin/bash
# lr=0 experiment: freeze weights, diff SP=1 vs SP=4 over many iterations.
#
# Why this works:
#   - optimizer.step() becomes a no-op (lr=0, weight_decay=0, constant schedule)
#   - forward + backward still run normally, all hooks fire normally
#   - weights stay at W0 for the entire run → no chaotic amplification
#
# What we expect:
#   - If Seq1F1B is correct:  cos_sim stays ≈ 1 - ε (bf16) or ≈ 1 (fp32)
#                             across all iters, modulo per-iter kernel noise
#   - If Seq1F1B has a bug:   cos_sim still drifts (bug adds per-iter error
#                             to the output regardless of weight freeze)
#
# Usage:
#   DATA_PATH=/path/to/data TRAIN_ITER=500 bash tests/run_lr0_compare.sh
#   DATA_PATH=/path/to/data TRAIN_ITER=500 USE_FP32=1 bash tests/run_lr0_compare.sh

set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
SAVE_DIR="${DIR}/tests/alignment_outputs"
mkdir -p "${SAVE_DIR}"

TRAIN_ITER=${TRAIN_ITER:-500}
USE_FP32=${USE_FP32:-0}
SEQ_LEN=${SEQ_LEN:-8192}

# Capture several iters spread across the run so we can see whether
# cos_sim stays flat or drifts.
EXTRA_ITERS_DEFAULT="1,10,50,100,200,300,400,498,499,500"
if [ "${TRAIN_ITER}" != "500" ]; then
    EXTRA_ITERS_DEFAULT="1,$((TRAIN_ITER/10)),$((TRAIN_ITER/2)),$((TRAIN_ITER-2)),$((TRAIN_ITER-1)),${TRAIN_ITER}"
fi
export SAVE_EXTRA_ITERS=${SAVE_EXTRA_ITERS:-${EXTRA_ITERS_DEFAULT}}

EXTRA_ARGS="--dump-layer-stats"
if [ "${USE_FP32}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --fp32"
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "  lr=0 experiment: freeze weights, check cos_sim stays flat"
echo "  TRAIN_ITER=${TRAIN_ITER}  SEQ_LEN=${SEQ_LEN}  fp32=${USE_FP32}"
echo "  capturing iters: ${SAVE_EXTRA_ITERS}"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo "  (A) SP=1  lr=0"
echo "───────────────────────────────────────────────────────────────────────"
LR0=1 PP_SP=1 TRAIN_ITER=${TRAIN_ITER} SEQ_LEN=${SEQ_LEN} \
    MASTER_PORT=29520 \
    bash "${DIR}/tests/run_alignment_test.sh" ${EXTRA_ARGS}

echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo "  (B) SP=4  lr=0"
echo "───────────────────────────────────────────────────────────────────────"
LR0=1 PP_SP=4 TRAIN_ITER=${TRAIN_ITER} SEQ_LEN=${SEQ_LEN} \
    MASTER_PORT=29521 \
    bash "${DIR}/tests/run_alignment_test.sh" ${EXTRA_ARGS}

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Analysis:  sp1_lr0  vs  sp4_lr0"
echo "═══════════════════════════════════════════════════════════════════════"

# Parse SAVE_EXTRA_ITERS "1,10,50,..." -> space-separated for --iters arg.
ITERS_SPACED=$(echo "${SAVE_EXTRA_ITERS}" | tr ',' ' ')

python3 "${DIR}/tests/analyze_alignment_outputs.py" \
    --tag-a sp1_lr0 --tag-b sp4_lr0 \
    --iters ${ITERS_SPACED} \
    --cos-threshold 0.999 \
    --dir "${SAVE_DIR}" || true

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Interpretation:"
echo "    cos_sim flat  across iters 1..${TRAIN_ITER}   → Seq1F1B IS CORRECT"
echo "                                                     (long-run drift"
echo "                                                      in lr>0 runs is"
echo "                                                      weight-update"
echo "                                                      chaos only)"
echo "    cos_sim drifts across iters 1..${TRAIN_ITER}   → Per-iter bug"
echo "                                                     (needs debugging)"
echo "═══════════════════════════════════════════════════════════════════════"
