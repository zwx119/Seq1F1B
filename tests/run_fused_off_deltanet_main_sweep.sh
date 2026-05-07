#!/usr/bin/env bash
# Fused-off pure DeltaNet main-result sweep.
#
# This reruns the original DeltaNet (non-hybrid) Seq1F1B table with the
# upstream FLA WY path:
#   chunk_scaled_dot_kkt -> solve_tril -> recompute_w_u
#
# MODEL_SPECS format:
#   name:num_layers:hidden_size:num_heads

set -uo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
RUNNER="${DIR}/tests/run_fineweb_long.sh"

export FLA_USE_FUSED_SOLVE_WU="${FLA_USE_FUSED_SOLVE_WU:-0}"
export FLA_DIR="${FLA_DIR:-${DIR}/flash-linear-attention}"
export PYTHONPATH="${FLA_DIR}:${DIR}:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCHRUN_STANDALONE="${TORCHRUN_STANDALONE:-1}"

OUT_ROOT="${OUT_ROOT:-${DIR}/tests/fused_off_deltanet_main}"
MODEL_SPECS="${MODEL_SPECS:-m2p0b:24:2560:32 m2p7b:32:2560:32}"
SEQ_LIST="${SEQ_LIST:-16384 24576 32768}"
SP_LIST="${SP_LIST:-1 2 4 8}"

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

echo "Fused-off pure DeltaNet main sweep"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "  MODEL_SPECS=${MODEL_SPECS}"
echo "  SEQ_LIST=${SEQ_LIST}"
echo "  SP_LIST=${SP_LIST}"
echo "  FLA_USE_FUSED_SOLVE_WU=${FLA_USE_FUSED_SOLVE_WU}"

declare -a RUN_STATUSES=()

for spec in ${MODEL_SPECS}; do
    IFS=: read -r model_name num_layers hidden heads extra <<< "${spec}"
    if [ -z "${model_name}" ] || [ -z "${num_layers}" ] || [ -z "${hidden}" ] || [ -z "${heads}" ] || [ -n "${extra:-}" ]; then
        echo "ERROR: bad MODEL_SPECS entry '${spec}'"
        echo "Expected format: name:num_layers:hidden_size:num_heads"
        exit 1
    fi

    for seq in ${SEQ_LIST}; do
        for sp in ${SP_LIST}; do
            if [ "${sp}" = "1" ]; then
                only="sp1"
                runner_sp="2"
                tag_sp="sp1"
            else
                only="seq"
                runner_sp="${sp}"
                tag_sp="sp${sp}"
            fi

            out_dir="${OUT_ROOT}/${model_name}_L${num_layers}_H${hidden}_A${heads}_seq${seq}_gbs${GLOBAL_BATCH}_${tag_sp}_average"
            echo ""
            echo "======================================================================"
            echo "Running ${model_name} seq=${seq} ${tag_sp} average"
            echo "  out=${out_dir}"
            echo "======================================================================"

            OUT_DIR="${out_dir}" \
            NUM_LAYERS="${num_layers}" HIDDEN="${hidden}" NUM_HEADS="${heads}" \
            SEQ_LEN="${seq}" GLOBAL_BATCH="${GLOBAL_BATCH}" MICRO_BATCH="${MICRO_BATCH}" \
            TRAIN_ITERS="${TRAIN_ITERS}" WARMUP_ITERS="${WARMUP_ITERS}" \
            GPUS_PER_NODE_USER=1 GPUS_PER_NODE="${GPUS_PER_NODE}" \
            PP_SIZE="${PP_SIZE}" TP_SIZE="${TP_SIZE}" \
            SEQ1F1B_SP="${runner_sp}" ONLY="${only}" \
            NO_SAVE="${NO_SAVE}" LOG_INTERVAL="${LOG_INTERVAL}" EVAL_ITERS="${EVAL_ITERS}" \
            PIPE_SP_STRATEGY=average EXTRA_ARGS="--timing-log-level 2" \
            bash "${RUNNER}"
            status=$?
            RUN_STATUSES+=("${model_name}:seq${seq}:${tag_sp}:average:${status}")
            if [ "${status}" -ne 0 ]; then
                echo "FAILED/OOM: ${model_name} seq=${seq} ${tag_sp}"
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
