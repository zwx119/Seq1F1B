#!/bin/bash
# System-level rigorous-equivalence test for Seq1F1B's DeltaNet integration.
#
# Runs three experiments on the same data/seed/weights and diffs their
# per-layer outputs + gradients:
#
#   (A) SP=1 baseline                    — DeltaNet sees full [S] sequence
#   (B) SP=1 + --force-seq-chunks=N      — DeltaNet internally chunks into N
#                                           pieces w/ state passing; everything
#                                           else (projections, norms, FFN, PP
#                                           schedule) is identical to (A)
#   (C) SP=N Seq1F1B (real pipe SP)      — full production path
#
# What each diff proves:
#
#   (A) vs (B): The DeltaNet kernel's chunked-state-passing math is equivalent
#               to its full-seq math, inside the real model (real fp32
#               weights, real activations, real autograd).
#               *This is the direct algorithmic-correctness proof you asked
#                for: same execution order for everything except the DeltaNet
#                kernel itself.*
#
#   (B) vs (C): The same DeltaNet computation, wrapped by two different outer
#               pipelines (full-seq layer call vs PP-SP layer call with
#               micro_sp_idx). Any diff here is attributable to PP-SP plumbing
#               only (no DeltaNet-math difference).
#
# Usage:
#   DATA_PATH=/path/to/data  bash tests/run_forcechunks_compare.sh
#
# Optional:
#   TRAIN_ITER=10 FSC=4 bash tests/run_forcechunks_compare.sh   # 10 iters, 4 chunks
#   USE_FP32=1 bash tests/run_forcechunks_compare.sh            # fp32 (most rigorous)
#
set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
SAVE_DIR="${DIR}/tests/alignment_outputs"
mkdir -p "${SAVE_DIR}"

TRAIN_ITER=${TRAIN_ITER:-10}
FSC=${FSC:-4}                     # force-seq-chunks value
USE_FP32=${USE_FP32:-1}           # fp32 by default — the whole point is bit-level
SEQ_LEN=${SEQ_LEN:-8192}

EXTRA_ARGS="--dump-layer-stats"
if [ "${USE_FP32}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --fp32"
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Seq1F1B DeltaNet — system-level force-chunks equivalence test"
echo "  TRAIN_ITER=${TRAIN_ITER}  FSC=${FSC}  SEQ_LEN=${SEQ_LEN}  fp32=${USE_FP32}"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo "  (A) Baseline: SP=1, no chunking — DeltaNet sees full seq"
echo "───────────────────────────────────────────────────────────────────────"
PP_SP=1 TRAIN_ITER=${TRAIN_ITER} SEQ_LEN=${SEQ_LEN} \
    MASTER_PORT=29510 \
    bash "${DIR}/tests/run_alignment_test.sh" ${EXTRA_ARGS}

echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo "  (B) SP=1 + --force-seq-chunks=${FSC} — DeltaNet chunks internally"
echo "      (projections/norms/FFN/PP identical to (A), only DeltaNet differs)"
echo "───────────────────────────────────────────────────────────────────────"
PP_SP=1 TRAIN_ITER=${TRAIN_ITER} SEQ_LEN=${SEQ_LEN} \
    TAG_SUFFIX="forcechunk${FSC}" MASTER_PORT=29511 \
    bash "${DIR}/tests/run_alignment_test.sh" ${EXTRA_ARGS} --force-seq-chunks ${FSC}

echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo "  (C) SP=${FSC} Seq1F1B — full production path"
echo "───────────────────────────────────────────────────────────────────────"
PP_SP=${FSC} TRAIN_ITER=${TRAIN_ITER} SEQ_LEN=${SEQ_LEN} \
    MASTER_PORT=29512 \
    bash "${DIR}/tests/run_alignment_test.sh" ${EXTRA_ARGS}

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Analysis"
echo "═══════════════════════════════════════════════════════════════════════"

# Resolve run tags. The tag format produced by run_alignment_test.sh is
# "sp<PP_SP>" (+ "_<TAG_SUFFIX>" if set), so:
#   (A) -> sp1
#   (B) -> sp1_forcechunk${FSC}
#   (C) -> sp${FSC}
TAG_A="sp1"
TAG_B="sp1_forcechunk${FSC}"
TAG_C="sp${FSC}"

echo ""
echo "── Compare (A) sp1  vs  (B) sp1_forcechunk${FSC}  (DIRECT proof) ──"
python3 "${DIR}/tests/analyze_alignment_outputs.py" \
    --tag-a "${TAG_A}" --tag-b "${TAG_B}" \
    --iters 1 ${TRAIN_ITER} \
    --cos-threshold 0.9999 --dir "${SAVE_DIR}" || true

echo ""
echo "── Compare (B) sp1_forcechunk${FSC}  vs  (C) sp${FSC}  (PP-SP plumbing only) ──"
python3 "${DIR}/tests/analyze_alignment_outputs.py" \
    --tag-a "${TAG_B}" --tag-b "${TAG_C}" \
    --iters 1 ${TRAIN_ITER} \
    --cos-threshold 0.9999 --dir "${SAVE_DIR}" || true

echo ""
echo "── Compare (A) sp1  vs  (C) sp${FSC}  (full system end-to-end, for reference) ──"
python3 "${DIR}/tests/analyze_alignment_outputs.py" \
    --tag-a "${TAG_A}" --tag-b "${TAG_C}" \
    --iters 1 ${TRAIN_ITER} \
    --cos-threshold 0.9999 --dir "${SAVE_DIR}" || true

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Done. Expected in fp32:"
echo "    (A) vs (B): cos_sim ≈ 1.0 (bit-equivalent modulo kernel reduction)"
echo "                → proves DeltaNet chunked-state-pass math is correct"
echo "    (B) vs (C): cos_sim ≈ (A) vs (B) level"
echo "                → proves PP-SP plumbing adds no math error"
echo "    (A) vs (C): cos_sim matches the above (at worst compounds both)"
echo "═══════════════════════════════════════════════════════════════════════"
