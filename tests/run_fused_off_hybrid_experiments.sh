#!/usr/bin/env bash
# Fused-off hybrid DeltaNet sweep for the main Seq1F1B result table.
#
# Defaults:
#   - L32/H2560/32 heads, PP8/TP1, GBS16, seq 16K/24K/32K
#   - hybrid patterns: global2 and period4
#   - split policies: average and hybrid_comp
#   - SP values: 4 and 8

set -uo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
RUNNER="${DIR}/tests/run_fineweb_long.sh"

export FLA_USE_FUSED_SOLVE_WU="${FLA_USE_FUSED_SOLVE_WU:-0}"
export FLA_DIR="${FLA_DIR:-${DIR}/flash-linear-attention}"
export PYTHONPATH="${FLA_DIR}:${DIR}:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCHRUN_STANDALONE="${TORCHRUN_STANDALONE:-1}"

OUT_ROOT="${OUT_ROOT:-${DIR}/tests/fused_off_hybrid_main}"
SEQ_LIST="${SEQ_LIST:-16384 24576 32768}"
SP_LIST="${SP_LIST:-4 8}"
STRATEGY_LIST="${STRATEGY_LIST:-average hybrid_comp}"
HYBRID_LIST="${HYBRID_LIST:-global2 period4}"

NUM_LAYERS="${NUM_LAYERS:-32}"
HIDDEN="${HIDDEN:-2560}"
NUM_HEADS="${NUM_HEADS:-32}"
GLOBAL_BATCH="${GLOBAL_BATCH:-16}"
MICRO_BATCH="${MICRO_BATCH:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-30}"
WARMUP_ITERS="${WARMUP_ITERS:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
EVAL_ITERS="${EVAL_ITERS:-0}"
NO_SAVE="${NO_SAVE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PP_SIZE="${PP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"

mkdir -p "${OUT_ROOT}"

echo "Fused-off hybrid Seq1F1B sweep"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "  FLA_USE_FUSED_SOLVE_WU=${FLA_USE_FUSED_SOLVE_WU}"
echo "  HYBRID_LIST=${HYBRID_LIST}"
echo "  STRATEGY_LIST=${STRATEGY_LIST}"
echo "  SEQ_LIST=${SEQ_LIST}"
echo "  SP_LIST=${SP_LIST}"

declare -a RUN_STATUSES=()

hybrid_args() {
    case "$1" in
        global2)
            echo "--deltanet-hybrid-attention-layers 2,15 --use-flash-attn --timing-log-level 2"
            ;;
        period4)
            echo "--deltanet-hybrid-attention-period 4 --use-flash-attn --timing-log-level 2"
            ;;
        *)
            echo "ERROR: unknown hybrid pattern '$1'" >&2
            return 1
            ;;
    esac
}

for hybrid in ${HYBRID_LIST}; do
    extra_args=$(hybrid_args "${hybrid}") || exit 1
    for seq in ${SEQ_LIST}; do
        for sp in ${SP_LIST}; do
            for strategy in ${STRATEGY_LIST}; do
                out_dir="${OUT_ROOT}/${hybrid}_seq${seq}_gbs${GLOBAL_BATCH}_sp${sp}_${strategy}"
                echo ""
                echo "======================================================================"
                echo "Running ${hybrid} seq=${seq} sp=${sp} strategy=${strategy}"
                echo "  out=${out_dir}"
                echo "======================================================================"

                OUT_DIR="${out_dir}" \
                NUM_LAYERS="${NUM_LAYERS}" HIDDEN="${HIDDEN}" NUM_HEADS="${NUM_HEADS}" \
                SEQ_LEN="${seq}" GLOBAL_BATCH="${GLOBAL_BATCH}" MICRO_BATCH="${MICRO_BATCH}" \
                TRAIN_ITERS="${TRAIN_ITERS}" WARMUP_ITERS="${WARMUP_ITERS}" \
                GPUS_PER_NODE_USER=1 GPUS_PER_NODE="${GPUS_PER_NODE}" \
                PP_SIZE="${PP_SIZE}" TP_SIZE="${TP_SIZE}" \
                SEQ1F1B_SP="${sp}" ONLY=seq \
                NO_SAVE="${NO_SAVE}" LOG_INTERVAL="${LOG_INTERVAL}" EVAL_ITERS="${EVAL_ITERS}" \
                PIPE_SP_STRATEGY="${strategy}" EXTRA_ARGS="${extra_args}" \
                bash "${RUNNER}"
                status=$?
                RUN_STATUSES+=("${hybrid}:seq${seq}:sp${sp}:${strategy}:${status}")
                if [ "${status}" -ne 0 ]; then
                    echo "FAILED/OOM: ${hybrid} seq=${seq} sp=${sp} strategy=${strategy}"
                fi
            done
        done
    done
done

echo ""
echo "Run statuses:"
for s in "${RUN_STATUSES[@]}"; do
    echo "  ${s}"
done

echo ""
echo "Summary command:"
echo "  python3 tests/summarize_fineweb_logs.py --root ${OUT_ROOT}"
