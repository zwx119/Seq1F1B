#!/bin/bash
# Run script for Seq1F1B with DeltaNet linear attention.
#
# Usage:
#   export GPUS_PER_NODE=8 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345
#   export DATA_PATH=/path/to/data
#   bash run_deltanet.sh
#
# This script mirrors run.sh but replaces FlashAttention with DeltaNet
# and disables RoPE (DeltaNet uses short convolutions for position info).

set -euo pipefail
MAX_RESTARTS=0
export NCCL_IB_QPS_PER_CONNECTION=8
export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE:-8} \
                   --nnodes ${WORLD_SIZE:-1} \
                   --rdzv_id=1 \
                   --rdzv_backend=c10d \
                   --rdzv_endpoint=${MASTER_ADDR:-localhost}:${MASTER_PORT:-12345} \
                   --max_restarts=${MAX_RESTARTS}"

DIR=$(pwd)
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
mkdir -p "${DIR}/logs"

# ──────── Model Configuration (override via env) ────────
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-2}
PP_SP=${PP_SP:-4}
PP_SP_STR=${PP_SP_STR:-uniform_comp}
VPP_SIZE=${VPP_SIZE:-1}
NUM_LAYERS=${NUM_LAYERS:-24}
HIDDEN=${HIDDEN:-2048}
NUM_ATTN_HEADS=${NUM_ATTN_HEADS:-16}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MICRO_BATCH=${MICRO_BATCH:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-8}
TRAIN_ITER=${TRAIN_ITER:-10}

# ──────── DeltaNet Configuration (override via env) ────────
DELTANET_MODE=${DELTANET_MODE:-chunk}
DELTANET_CONV_SIZE=${DELTANET_CONV_SIZE:-4}
DELTANET_QK_ACTIVATION=${DELTANET_QK_ACTIVATION:-silu}
DELTANET_QK_NORM=${DELTANET_QK_NORM:-l2}

# ──────── VPP ────────
if [ "${VPP_SIZE}" -eq 1 ]; then
    VPP_STR=""
else
    NUM_LAYERS_PER_VSTAGE=$((NUM_LAYERS / PP_SIZE / VPP_SIZE))
    VPP_STR="--num-layers-per-virtual-pipeline-stage ${NUM_LAYERS_PER_VSTAGE}"
fi

options=" \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --timing-log-level 2 \
    --pipe-sp-strategy ${PP_SP_STR} \
    --pipe-sp-splits ${PP_SP} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --micro-batch-size ${MICRO_BATCH} \
    --global-batch-size ${GLOBAL_BATCH} \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --train-iters ${TRAIN_ITER} \
    --log-interval 1 \
    --eval-iters 0 \
    --eval-interval 1000 \
    --use-deltanet \
    --deltanet-mode ${DELTANET_MODE} \
    --deltanet-conv-size ${DELTANET_CONV_SIZE} \
    --deltanet-qk-activation ${DELTANET_QK_ACTIVATION} \
    --deltanet-qk-norm ${DELTANET_QK_NORM} \
    --deltanet-use-short-conv \
    --deltanet-use-beta \
    --deltanet-use-output-gate \
    --data-path ${DATA_PATH}/data/codeparrot_content_document_text_document \
    --vocab-file ${DATA_PATH}/data/vocab.json \
    --merge-file ${DATA_PATH}/data/merges.txt \
    --initial-loss-scale 65536 \
    --save-interval 1000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --bf16 \
    --untie-embeddings-and-output-weights \
    --use-distributed-optimizer \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --sequence-parallel \
    --no-async-tensor-model-parallel-allreduce \
    ${VPP_STR}
"

# ──────── Profile (optional) ────────
if [ "${PROFILE:-false}" = "true" ]; then
    options="${options} \
    --profile \
    --profile-step-start 3 \
    --profile-step-end 5 \
    --profile-ranks 0
    "
fi

# ──────── Activation checkpointing (optional) ────────
if [ "${RECOMPUTE:-0}" -eq 1 ]; then
    options="${options} \
    --recompute-method uniform \
    --recompute-granularity full
    "
fi

run_cmd="torchrun ${DISTRIBUTED_ARGS} ${DIR}/pretrain_gpt.py ${options}"
echo "====== DeltaNet Seq1F1B Launch Command ======"
echo "${run_cmd}"
echo "============================================="
exec ${run_cmd}
