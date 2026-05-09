#!/usr/bin/env bash
# Full-attention split-policy control for the hybrid-attention story.
#
# Question:
#   If every layer is softmax/full attention, do non-uniform Seq1F1B sequence
#   splits improve throughput too?
#
# This controls for the "causal attention gets more expensive later in the
# sequence" effect.  The default shape mirrors the period4 hybrid runs, but
# disables DeltaNet entirely and uses FlashAttention for all attention layers.

set -uo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
RUNNER="${DIR}/tests/run_fineweb_long.sh"

export FLA_DIR="${FLA_DIR:-${DIR}/flash-linear-attention}"
export PYTHONPATH="${FLA_DIR}:${DIR}:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCHRUN_STANDALONE="${TORCHRUN_STANDALONE:-1}"

OUT_ROOT="${OUT_ROOT:-${DIR}/tests/full_attention_split_control}"
SEQ_LIST="${SEQ_LIST:-32768}"
SP_LIST="${SP_LIST:-4}"
SPLIT_LIST="${SPLIT_LIST:-average full_comp very_mild mild medium hcomp}"

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

# Match the hybrid full-attention layers as closely as possible: FlashAttention
# on, no learned/rotary position embedding, same untied embedding/head setting
# from the common runner.
FULL_ATTN_EXTRA_ARGS="${FULL_ATTN_EXTRA_ARGS:---use-flash-attn --position-embedding-type none --timing-log-level 2}"

mkdir -p "${OUT_ROOT}"

manual_splits() {
    local seq=$1
    local sp=$2
    local name=$3
    if [ "${seq}" = "32768" ] && [ "${sp}" = "4" ]; then
        case "${name}" in
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
    echo "ERROR: no manual split table for seq=${seq} sp=${sp} name=${name}" >&2
    return 1
}

echo "Full-attention Seq1F1B split-control sweep"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "  SEQ_LIST=${SEQ_LIST}"
echo "  SP_LIST=${SP_LIST}"
echo "  SPLIT_LIST=${SPLIT_LIST}"
echo "  model=L${NUM_LAYERS}/H${HIDDEN}/A${NUM_HEADS} GBS=${GLOBAL_BATCH}"
echo "  extra=${FULL_ATTN_EXTRA_ARGS}"

declare -a RUN_STATUSES=()

for seq in ${SEQ_LIST}; do
    for sp in ${SP_LIST}; do
        for split_name in ${SPLIT_LIST}; do
            strategy=""
            manual=""
            case "${split_name}" in
                average)
                    strategy="average"
                    ;;
                full_comp)
                    # With USE_DELTANET=0, hybrid_comp degenerates to the
                    # all-softmax/full-attention cost model.
                    strategy="hybrid_comp"
                    ;;
                very_mild|mild|medium|hcomp)
                    strategy="manual"
                    manual=$(manual_splits "${seq}" "${sp}" "${split_name}") || exit 1
                    ;;
                *)
                    echo "ERROR: unknown split_name='${split_name}'"
                    exit 1
                    ;;
            esac

            out_dir="${OUT_ROOT}/fullattn_seq${seq}_gbs${GLOBAL_BATCH}_sp${sp}_${split_name}"
            echo ""
            echo "======================================================================"
            echo "Running full-attn seq=${seq} sp=${sp} split=${split_name}"
            if [ -n "${manual}" ]; then
                echo "  manual=${manual}"
            fi
            echo "  out=${out_dir}"
            echo "======================================================================"

            OUT_DIR="${out_dir}" \
            USE_DELTANET=0 \
            NUM_LAYERS="${NUM_LAYERS}" HIDDEN="${HIDDEN}" NUM_HEADS="${NUM_HEADS}" \
            SEQ_LEN="${seq}" GLOBAL_BATCH="${GLOBAL_BATCH}" MICRO_BATCH="${MICRO_BATCH}" \
            TRAIN_ITERS="${TRAIN_ITERS}" WARMUP_ITERS="${WARMUP_ITERS}" \
            GPUS_PER_NODE_USER=1 GPUS_PER_NODE="${GPUS_PER_NODE}" \
            PP_SIZE="${PP_SIZE}" TP_SIZE="${TP_SIZE}" \
            SEQ1F1B_SP="${sp}" ONLY=seq \
            NO_SAVE="${NO_SAVE}" LOG_INTERVAL="${LOG_INTERVAL}" EVAL_ITERS="${EVAL_ITERS}" \
            PIPE_SP_STRATEGY="${strategy}" PIPE_SP_MANUAL_SPLITS="${manual}" \
            EXTRA_ARGS="${FULL_ATTN_EXTRA_ARGS}" \
            bash "${RUNNER}"
            status=$?
            RUN_STATUSES+=("seq${seq}:sp${sp}:${split_name}:${status}")
            if [ "${status}" -ne 0 ]; then
                echo "FAILED/OOM: seq=${seq} sp=${sp} split=${split_name}"
            fi
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
