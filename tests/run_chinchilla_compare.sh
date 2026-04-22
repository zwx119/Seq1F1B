#!/bin/bash
# Chinchilla-style pretraining loss-curve alignment test: SP=1 vs SP=4.
#
# Unlike the lr=0 / hidden-state tests (which check numerical equivalence
# at the tensor level), this test checks *training dynamics* — does SP=4
# produce the same loss trajectory as SP=1 over thousands of iterations
# with a real LR schedule (warmup + cosine decay), gradient clipping,
# Adam, weight decay, etc.?
#
# If the per-iter loss curves track each other within FP noise, Seq1F1B
# is training-dynamics-equivalent to vanilla PP. If they drift, there is
# a systematic optimizer/grad bug that only shows up under weight update.
#
# Usage:
#   DATA_PATH=/path/to/data TRAIN_ITER=2000 bash tests/run_chinchilla_compare.sh
#   DATA_PATH=/path/to/data TRAIN_ITER=5000 GLOBAL_BATCH=32 bash tests/run_chinchilla_compare.sh

set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
SAVE_DIR="${DIR}/tests/alignment_outputs"
mkdir -p "${SAVE_DIR}"

# ── Training hyperparameters (Chinchilla-flavored defaults) ────────────
# Model: reuse the existing 24L/1024H/16H default (~350M params w/ embed)
# Chinchilla 20× tokens would be ~7B tokens → 107k iters at 65K tok/iter;
# too long for a quick sanity check. We default to 2000 iters (~130M
# tokens, ~0.4 Chinchilla-multiple) which is enough to see the warmup
# peak and settle into cosine decay.
export TRAIN_ITER=${TRAIN_ITER:-2000}
export SEQ_LEN=${SEQ_LEN:-8192}
export GLOBAL_BATCH=${GLOBAL_BATCH:-16}
export MICRO_BATCH=${MICRO_BATCH:-1}
export NUM_LAYERS=${NUM_LAYERS:-24}
export HIDDEN=${HIDDEN:-1024}
export NUM_HEADS=${NUM_HEADS:-16}

# Proper LR schedule: warmup then cosine decay.
export LR=${LR:-3.0e-4}
export MIN_LR=${MIN_LR:-3.0e-5}
export LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}

# Warmup iters: forward to the python script via EXTRA_ARGS.
WARMUP_ITERS=${WARMUP_ITERS:-$((TRAIN_ITER / 20))}  # 5% warmup
EXTRA_ARGS="--lr-warmup-iters ${WARMUP_ITERS} --lr-decay-iters ${TRAIN_ITER}"

# Differentiate chinchilla runs so their loss/log files don't clobber
# the lr=0 / sp1_vs_sp1 outputs.
export TAG_SUFFIX=${TAG_SUFFIX:-chinchilla}

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Chinchilla-style pretraining loss alignment: SP=1 vs SP=4"
echo "  TRAIN_ITER=${TRAIN_ITER}  SEQ_LEN=${SEQ_LEN}  GLOBAL_BATCH=${GLOBAL_BATCH}"
echo "  Tokens/iter = ${SEQ_LEN} × ${GLOBAL_BATCH} = $((SEQ_LEN * GLOBAL_BATCH))"
echo "  Total tokens ≈ $((SEQ_LEN * GLOBAL_BATCH * TRAIN_ITER)) (~$(awk -vN=$((SEQ_LEN*GLOBAL_BATCH*TRAIN_ITER)) 'BEGIN{printf "%.1f M", N/1e6}'))"
echo "  lr=${LR} → ${MIN_LR}  warmup=${WARMUP_ITERS}  decay=cosine"
echo "  Precision: bf16  Seed: 42"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo "  (A) SP=1  chinchilla"
echo "───────────────────────────────────────────────────────────────────────"
PP_SP=1 MASTER_PORT=29540 \
    bash "${DIR}/tests/run_alignment_test.sh" ${EXTRA_ARGS}

echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo "  (B) SP=4  chinchilla"
echo "───────────────────────────────────────────────────────────────────────"
PP_SP=4 MASTER_PORT=29541 \
    bash "${DIR}/tests/run_alignment_test.sh" ${EXTRA_ARGS}

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Analysis: sp1_${TAG_SUFFIX}  vs  sp4_${TAG_SUFFIX}"
echo "═══════════════════════════════════════════════════════════════════════"

python3 "${DIR}/tests/analyze_loss_curves.py" \
    --loss-a "${SAVE_DIR}/loss_sp1_${TAG_SUFFIX}.txt" \
    --loss-b "${SAVE_DIR}/loss_sp4_${TAG_SUFFIX}.txt" \
    --label-a "SP=1" --label-b "SP=4" \
    --output "${SAVE_DIR}/loss_compare_${TAG_SUFFIX}.txt" \
    --plot "${SAVE_DIR}/loss_compare_${TAG_SUFFIX}.png" || true

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Interpretation:"
echo "    two loss curves track each other (rel_diff < 1%)"
echo "                                 → training dynamics equivalent ✓"
echo "    curves diverge monotonically → systematic optimizer/grad bug ✗"
echo ""
echo "  Note: small per-iter noise is expected (bf16 kernel accumulation"
echo "  order differs between SP=1 and SP=4), but it should NOT grow."
echo "═══════════════════════════════════════════════════════════════════════"
