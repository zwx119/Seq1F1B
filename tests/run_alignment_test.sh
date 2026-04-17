#!/bin/bash
# Launch alignment test on 4 GPUs with PP=4.
# This saves the per-iteration loss to a file for later comparison.
#
# Usage:
#   PP_SP=1 bash tests/run_alignment_test.sh   # baseline (no SP)
#   PP_SP=4 bash tests/run_alignment_test.sh   # Seq1F1B
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

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
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --train-iters ${TRAIN_ITER} \
    --log-interval 1 \
    --eval-iters 0 \
    --eval-interval 10000 \
    --use-deltanet \
    --deltanet-mode chunk \
    --deltanet-conv-size 4 \
    --deltanet-qk-activation silu \
    --deltanet-qk-norm l2 \
    --deltanet-use-short-conv \
    --deltanet-use-beta \
    --bf16 \
    --untie-embeddings-and-output-weights \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --no-async-tensor-model-parallel-allreduce \
    --seed 42 \
    --no-save-optim \
    --no-save-rng \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --initial-loss-scale 65536 \
    --use-distributed-optimizer \
"

# Data path (required for tokenizer vocab/merge files)
if [ -z "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH must be set (need vocab.json and merges.txt)"
    exit 1
fi
options="${options} \
    --vocab-file ${DATA_PATH}/data/vocab.json \
    --merge-file ${DATA_PATH}/data/merges.txt \
"

OUTPUT_FILE="${SAVE_DIR}/loss_sp${PP_SP}.txt"

run_cmd="torchrun ${DISTRIBUTED_ARGS} ${DIR}/tests/test_deltanet_alignment.py ${options}"
echo "====== Alignment Test (PP_SP=${PP_SP}) ======"
echo "${run_cmd}"
echo "============================================="
eval ${run_cmd} 2>&1 | tee "${SAVE_DIR}/log_sp${PP_SP}.txt"

# Extract loss from log
grep "lm loss" "${SAVE_DIR}/log_sp${PP_SP}.txt" > "${OUTPUT_FILE}" || true
echo "Loss saved to ${OUTPUT_FILE}"
