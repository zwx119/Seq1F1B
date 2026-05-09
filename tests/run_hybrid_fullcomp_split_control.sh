#!/usr/bin/env bash
# Hybrid-attention split-policy control using the all-full-attention cost split.
#
# Question:
#   Does the very non-uniform split that is optimal for all-full-attention also
#   help the period4 hybrid DeltaNet model, or is it specific to the all-softmax
#   control?
#
# Default run:
#   - model: L32/H2560/A32, seq=32768, GBS=16, PP=8
#   - hybrid: period4, every 4 layers has 1 full-attention layer
#   - Seq1F1B splits: sp=4
#   - split: full_comp from the all-full-attention cost model

set -uo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
RUNNER="${DIR}/tests/run_fineweb_long.sh"

export FLA_USE_FUSED_SOLVE_WU="${FLA_USE_FUSED_SOLVE_WU:-0}"
export FLA_DIR="${FLA_DIR:-${DIR}/flash-linear-attention}"
export PYTHONPATH="${FLA_DIR}:${DIR}:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCHRUN_STANDALONE="${TORCHRUN_STANDALONE:-1}"

OUT_ROOT="${OUT_ROOT:-${DIR}/tests/hybrid_fullcomp_split_control}"
SEQ_LIST="${SEQ_LIST:-32768}"
SP_LIST="${SP_LIST:-4}"
SPLIT_LIST="${SPLIT_LIST:-full_comp}"
HYBRID_LIST="${HYBRID_LIST:-period4}"

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

manual_splits() {
    local seq=$1
    local sp=$2
    local name=$3
    if [ "${seq}" = "32768" ] && [ "${sp}" = "4" ]; then
        case "${name}" in
            # Full-attention cost-model optimum from run_full_attention_split_control.sh.
            full_comp) echo "11904,8320,6784,5760" ;;
            very_mild) echo "8320,8192,8192,8064" ;;
            mild)      echo "8448,8320,8064,7936" ;;
            medium)    echo "8704,8320,8064,7680" ;;
            hcomp)     echo "9472,8448,7680,7168" ;;
            *)
                echo "ERROR: unknown manual split '${name}' for seq=${seq} sp=${sp}" >&2
                return 1
                ;;
        esac
        return 0
    fi
    if [ "${seq}" = "65536" ] && [ "${sp}" = "4" ]; then
        case "${name}" in
            # 64K counterparts of the 32K split-control table above.
            full_comp) echo "23808,16640,13568,11520" ;;
            very_mild) echo "16640,16384,16384,16128" ;;
            mild)      echo "16896,16640,16128,15872" ;;
            medium)    echo "17408,16640,16128,15360" ;;
            hcomp)     echo "18944,16896,15360,14336" ;;
            *)
                echo "ERROR: unknown manual split '${name}' for seq=${seq} sp=${sp}" >&2
                return 1
                ;;
        esac
        return 0
    fi
    echo "ERROR: no manual split table for seq=${seq} sp=${sp} name=${name}" >&2
    return 1
}

echo "Hybrid Seq1F1B all-full-attention split-control sweep"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "  FLA_USE_FUSED_SOLVE_WU=${FLA_USE_FUSED_SOLVE_WU}"
echo "  HYBRID_LIST=${HYBRID_LIST}"
echo "  SPLIT_LIST=${SPLIT_LIST}"
echo "  SEQ_LIST=${SEQ_LIST}"
echo "  SP_LIST=${SP_LIST}"
echo "  model=L${NUM_LAYERS}/H${HIDDEN}/A${NUM_HEADS} GBS=${GLOBAL_BATCH}"

declare -a RUN_STATUSES=()

for hybrid in ${HYBRID_LIST}; do
    extra_args=$(hybrid_args "${hybrid}") || exit 1
    for seq in ${SEQ_LIST}; do
        for sp in ${SP_LIST}; do
            for split_name in ${SPLIT_LIST}; do
                strategy=""
                manual=""
                case "${split_name}" in
                    average)
                        strategy="average"
                        ;;
                    hybrid_comp)
                        strategy="hybrid_comp"
                        ;;
                    full_comp|very_mild|mild|medium|hcomp)
                        strategy="manual"
                        manual=$(manual_splits "${seq}" "${sp}" "${split_name}") || exit 1
                        ;;
                    *)
                        echo "ERROR: unknown split_name='${split_name}'"
                        exit 1
                        ;;
                esac

                out_dir="${OUT_ROOT}/${hybrid}_seq${seq}_gbs${GLOBAL_BATCH}_sp${sp}_${split_name}"
                echo ""
                echo "======================================================================"
                echo "Running ${hybrid} seq=${seq} sp=${sp} split=${split_name}"
                if [ -n "${manual}" ]; then
                    echo "  manual=${manual}"
                fi
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
                PIPE_SP_STRATEGY="${strategy}" PIPE_SP_MANUAL_SPLITS="${manual}" \
                EXTRA_ARGS="${extra_args}" \
                bash "${RUNNER}"
                status=$?
                RUN_STATUSES+=("${hybrid}:seq${seq}:sp${sp}:${split_name}:${status}")
                if [ "${status}" -ne 0 ]; then
                    echo "FAILED/OOM: ${hybrid} seq=${seq} sp=${sp} split=${split_name}"
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
