#!/bin/bash
# Launch alignment test on 4 GPUs with PP=4.
# This saves the per-iteration loss to a file for later comparison.
#
# Usage:
#   PP_SP=1 bash tests/run_alignment_test.sh   # baseline (no SP)
#   PP_SP=4 bash tests/run_alignment_test.sh   # Seq1F1B
set -euo pipefail

# Default to the deterministic single-connection setting, but let callers
# override it (e.g. the sp1-vs-sp1 sanity check uses =8 on the "alt" run to
# perturb floating-point accumulation order without changing the algorithm).
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

GPUS=${GPUS_PER_NODE:-8}
PP_SP=${PP_SP:-1}
PP_SP_STR=${PP_SP_STR:-average}
SEQ_LEN=${SEQ_LEN:-8192}
NUM_LAYERS=${NUM_LAYERS:-24}
HIDDEN=${HIDDEN:-1024}
NUM_HEADS=${NUM_HEADS:-16}
MICRO_BATCH=${MICRO_BATCH:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-4}
TRAIN_ITER=${TRAIN_ITER:-500}
LR=${LR:-6.0e-4}
MIN_LR=${MIN_LR:-6.0e-5}

# LR0=1 ⇒ set lr=0, min_lr=0, weight_decay=0, constant schedule.
# Effect: forward + backward run normally (so all cos_sim / grad hooks
# still work), but optimizer.step() becomes a no-op, so weights stay at
# W0 for the entire run. This isolates per-iteration kernel noise from
# the multi-iter chaotic amplification of weight drift.
LR0=${LR0:-0}
if [ "${LR0}" = "1" ]; then
    LR=0.0
    MIN_LR=0.0
    LR_DECAY_STYLE=constant
    WEIGHT_DECAY=0.0
else
    LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
    WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
fi

DATA_PATH=${DATA_PATH:-""}

DIR=$(cd "$(dirname "$0")/.." && pwd)
SAVE_DIR="${DIR}/tests/alignment_outputs"
mkdir -p "${SAVE_DIR}"

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS} \
                   --nnodes 1 \
                   --rdzv_id=1 \
                   --rdzv_backend=c10d \
                   --rdzv_endpoint=localhost:${MASTER_PORT:-29500} \
                   --max_restarts=0"

options=" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size ${GPUS} \
    --pipe-sp-strategy ${PP_SP_STR} \
    --pipe-sp-splits ${PP_SP} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MICRO_BATCH} \
    --global-batch-size ${GLOBAL_BATCH} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style ${LR_DECAY_STYLE} \
    --train-iters ${TRAIN_ITER} \
    --log-interval 1 \
    --eval-iters 0 \
    --eval-interval 10000 \
    --use-deltanet \
    --deltanet-mode chunk \
    --deltanet-conv-size 4 \
    --deltanet-qk-activation silu \
    --deltanet-qk-norm l2 \
    --deltanet-use-beta \
    --untie-embeddings-and-output-weights \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --no-async-tensor-model-parallel-allreduce \
    --seed 42 \
    --no-save-optim \
    --no-save-rng \
    --clip-grad 1.0 \
    --weight-decay ${WEIGHT_DECAY} \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --initial-loss-scale 65536 \
    --use-distributed-optimizer \
"

# Data path (required: prefix for .bin/.idx data files AND vocab/merge)
if [ -z "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH must be set (directory containing data/ with vocab.json, merges.txt and *_document.bin/.idx)"
    exit 1
fi
options="${options} \
    --data-path ${DATA_PATH}/data/codeparrot_content_document_text_document \
    --vocab-file ${DATA_PATH}/data/vocab.json \
    --merge-file ${DATA_PATH}/data/merges.txt \
    --split 98,2,0 \
"

DISABLE_STATE_PASSING=${DISABLE_STATE_PASSING:-0}
NO_SHORT_CONV=${NO_SHORT_CONV:-0}

# Short conv flag
if [ "${NO_SHORT_CONV}" = "1" ]; then
    options="${options} --no-deltanet-short-conv"
else
    options="${options} --deltanet-use-short-conv"
fi

# Tag for output files
if [ "${DISABLE_STATE_PASSING}" = "1" ]; then
    TAG="nostate"
elif [ "${NO_SHORT_CONV}" = "1" ]; then
    TAG="sp${PP_SP}_noconv"
else
    TAG="sp${PP_SP}"
fi
if [ "${LR0}" = "1" ]; then
    TAG="${TAG}_lr0"
fi
# Keep log/loss filename in sync with the python-side TAG_SUFFIX if set.
if [ -n "${TAG_SUFFIX:-}" ]; then
    TAG="${TAG}_${TAG_SUFFIX}"
fi

OUTPUT_FILE="${SAVE_DIR}/loss_${TAG}.txt"

# Forward any extra CLI args (e.g. --fp32, --dump-layer-stats) from the
# caller (run_noconv_compare.sh / run_alignment_compare.sh) to the python
# script. Without this, flags injected by the outer script never reach
# test_deltanet_alignment.py.
EXTRA_PY_ARGS="$*"

# Precision: default bf16, but if caller passes --fp32 we must NOT also
# append --bf16 (they are mutually exclusive and --bf16 wins at arg-parse
# time, silently turning an "fp32" run back into bf16).
USE_FP32=0
for a in "$@"; do
    if [ "$a" = "--fp32" ]; then
        USE_FP32=1
        break
    fi
done
if [ "${USE_FP32}" = "0" ]; then
    options="${options} --bf16"
fi

run_cmd="DISABLE_STATE_PASSING=${DISABLE_STATE_PASSING} torchrun ${DISTRIBUTED_ARGS} ${DIR}/tests/test_deltanet_alignment.py ${options} ${EXTRA_PY_ARGS}"
echo "====== Alignment Test (PP_SP=${PP_SP}, TAG=${TAG}) ======"
echo "${run_cmd}"
echo "============================================="
eval ${run_cmd} 2>&1 | tee "${SAVE_DIR}/log_${TAG}.txt"

# Extract loss from log
grep "lm loss" "${SAVE_DIR}/log_${TAG}.txt" > "${OUTPUT_FILE}" || true
echo "Loss saved to ${OUTPUT_FILE}"
