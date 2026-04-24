#!/bin/bash
# Long training run on FineWeb-Edu for Seq1F1B × DeltaNet × ShortConv.
#
# Designed for:
#   - 4 × 80GB GPU (A100 or H100), PP=4, TP=1, DP=1
#   - ~12h wall-clock budget (SP=1 + Seq1F1B sequentially)
#   - ~300M non-embed params, seq_len=16384, ~800M training tokens
#
# Usage:
#     # default: run both SP=1 baseline and SP=<PP_SIZE> Seq1F1B sequentially
#     DATA_PREFIX=/path/to/fineweb_edu_sample_10BT_text_document \
#     VOCAB=/path/to/gpt2/gpt2-vocab.json \
#     MERGE=/path/to/gpt2/gpt2-merges.txt \
#     bash tests/run_fineweb_long.sh
#
#     # just one config:
#     ONLY=seq DATA_PREFIX=... bash tests/run_fineweb_long.sh
#     ONLY=sp1 DATA_PREFIX=... bash tests/run_fineweb_long.sh
#     SEQ1F1B_SP=2 ONLY=seq DATA_PREFIX=... bash tests/run_fineweb_long.sh
#
#     # override iters / seq_len / model size:
#     TRAIN_ITERS=1500 SEQ_LEN=8192 bash tests/run_fineweb_long.sh
#
#     # resume from existing checkpoints explicitly:
#     RESUME=1 bash tests/run_fineweb_long.sh
#
#     # disable checkpoint saving entirely:
#     NO_SAVE=1 bash tests/run_fineweb_long.sh
#
# Output structure:
#     tests/fineweb_outputs/
#         log_fineweb_sp1.txt          # full stdout (grep "validation loss" / "lm loss" here)
#         log_fineweb_sp${SEQ1F1B_SP}.txt
#         ckpt_fineweb_sp1/            # checkpoints (one every SAVE_INTERVAL)
#         ckpt_fineweb_sp${SEQ1F1B_SP}/
#         tb/fineweb_sp1/              # tensorboard (lm loss, val loss, lr, grad_norm, ...)
#         tb/fineweb_sp${SEQ1F1B_SP}/
#
# Monitoring during training (separate terminal):
#     tensorboard --logdir tests/fineweb_outputs/tb --port 6006

set -euo pipefail

# ============================================================================
# Config (all env-overridable)
# ============================================================================
DIR=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR="${OUT_DIR:-${DIR}/tests/fineweb_outputs}"
mkdir -p "${OUT_DIR}"

# --- which configs to run ---
ONLY=${ONLY:-both}       # {sp1, seq, both}

# --- data ---
# Default: expect `bash tools/preprocess_fineweb_edu.sh` output.
DEFAULT_PREFIX="${DIR}/data/fineweb_edu_sample_10BT_text_document"
DATA_PREFIX=${DATA_PREFIX:-${DEFAULT_PREFIX}}
VOCAB=${VOCAB:-${DIR}/data/gpt2/gpt2-vocab.json}
MERGE=${MERGE:-${DIR}/data/gpt2/gpt2-merges.txt}

# --- model ---
NUM_LAYERS=${NUM_LAYERS:-24}
HIDDEN=${HIDDEN:-1024}
NUM_HEADS=${NUM_HEADS:-16}

# --- training ---
SEQ_LEN=${SEQ_LEN:-16384}
MICRO_BATCH=${MICRO_BATCH:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
TRAIN_ITERS=${TRAIN_ITERS:-3000}
LR=${LR:-3.0e-4}
MIN_LR=${MIN_LR:-3.0e-5}
WARMUP_ITERS=${WARMUP_ITERS:-200}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}
RESUME=${RESUME:-0}

# Megatron requires lr_warmup_iters < train_iters when using iteration-based
# scheduling. Keep the long-run default for real experiments, but auto-shrink
# warmup for short smoke tests such as TRAIN_ITERS=20.
if [ "${TRAIN_ITERS}" -le 1 ]; then
    EFFECTIVE_WARMUP_ITERS=0
elif [ "${WARMUP_ITERS}" -ge "${TRAIN_ITERS}" ]; then
    EFFECTIVE_WARMUP_ITERS=$((TRAIN_ITERS - 1))
else
    EFFECTIVE_WARMUP_ITERS=${WARMUP_ITERS}
fi

# --- logging / eval / ckpt ---
LOG_INTERVAL=${LOG_INTERVAL:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
EVAL_ITERS=${EVAL_ITERS:-20}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
NO_SAVE=${NO_SAVE:-0}

# --- distributed ---
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
if command -v nvidia-smi > /dev/null 2>&1 && [ -z "${GPUS_PER_NODE_USER:-}" ]; then
    DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${DETECTED}" -gt 0 ]; then
        GPUS_PER_NODE=${DETECTED}
    fi
fi
PP_SIZE=${PP_SIZE:-${GPUS_PER_NODE}}
TP_SIZE=${TP_SIZE:-1}
SEQ1F1B_SP=${SEQ1F1B_SP:-${PP_SIZE}}
SEQ_TAG="fineweb_sp${SEQ1F1B_SP}"

if [ "${SEQ1F1B_SP}" -lt 2 ]; then
    echo "ERROR: SEQ1F1B_SP must be >= 2, got '${SEQ1F1B_SP}'"
    exit 1
fi

if [ "${SEQ1F1B_SP}" -gt "${SEQ_LEN:-16384}" ]; then
    echo "ERROR: SEQ1F1B_SP (${SEQ1F1B_SP}) must be <= SEQ_LEN (${SEQ_LEN})"
    exit 1
fi

# CUDA_DEVICE_MAX_CONNECTIONS=1 is the Megatron-recommended deterministic
# setting (overrideable).
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# ============================================================================
# Sanity checks
# ============================================================================
for f in "${DATA_PREFIX}.bin" "${DATA_PREFIX}.idx" "${VOCAB}" "${MERGE}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required file missing: ${f}"
        echo ""
        echo "Expected pipeline (run once before this script):"
        echo "    bash tools/download_fineweb_edu.sh    # download parquet"
        echo "    bash tools/preprocess_fineweb_edu.sh  # parquet -> bin/idx"
        echo ""
        echo "Or override paths:"
        echo "    DATA_PREFIX=... VOCAB=... MERGE=... bash tests/run_fineweb_long.sh"
        exit 1
    fi
done

BIN_BYTES=$(stat -c%s "${DATA_PREFIX}.bin" 2>/dev/null || stat -f%z "${DATA_PREFIX}.bin" 2>/dev/null || echo 0)
BIN_HUMAN=$(du -h "${DATA_PREFIX}.bin" | awk '{print $1}')
TOKENS_PER_STEP=$((GLOBAL_BATCH * SEQ_LEN))
TRAIN_TOKENS=$((TOKENS_PER_STEP * TRAIN_ITERS))

echo "═══════════════════════════════════════════════════════════════════════"
echo "  FineWeb-Edu long training"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  data prefix  = ${DATA_PREFIX}  (bin=${BIN_HUMAN})"
echo "  vocab/merge  = ${VOCAB} / ${MERGE}"
echo "  GPUs         = ${GPUS_PER_NODE}  (TP=${TP_SIZE}, PP=${PP_SIZE})"
echo "  model        = L=${NUM_LAYERS} H=${HIDDEN} heads=${NUM_HEADS}"
echo "                 (~300M non-embed for 24/1024/16 defaults)"
echo "  seq_len      = ${SEQ_LEN}"
echo "  batch        = micro=${MICRO_BATCH}  global=${GLOBAL_BATCH}"
echo "  tokens/step  = ${TOKENS_PER_STEP}"
echo "  train_iters  = ${TRAIN_ITERS}"
echo "  total tokens = ${TRAIN_TOKENS}  (~$(echo ${TRAIN_TOKENS} | awk '{printf "%.2fB", $1/1e9}'))"
echo "  lr           = ${LR} → ${MIN_LR}  (cosine, warmup ${EFFECTIVE_WARMUP_ITERS})"
echo "  weight_decay = ${WEIGHT_DECAY}"
echo "  eval         = every ${EVAL_INTERVAL} iter × ${EVAL_ITERS} batches"
if [ "${NO_SAVE}" = "1" ]; then
    echo "  save         = disabled"
else
    echo "  save         = every ${SAVE_INTERVAL} iter"
fi
echo "  OUT_DIR      = ${OUT_DIR}"
echo "  ONLY         = ${ONLY}"
echo "  seq1f1b_sp   = ${SEQ1F1B_SP}"
echo "  resume       = ${RESUME}"
if [ "${EFFECTIVE_WARMUP_ITERS}" -ne "${WARMUP_ITERS}" ]; then
    echo "  note         = warmup clipped from ${WARMUP_ITERS} to ${EFFECTIVE_WARMUP_ITERS} for short run"
fi
echo "═══════════════════════════════════════════════════════════════════════"

# ============================================================================
# run_one(pp_sp, tag, master_port)
#   pp_sp       : 1 (baseline 1F1B) or ${SEQ1F1B_SP} (Seq1F1B chunks)
#   tag         : string used in log/ckpt/tb paths
#   master_port : to avoid collision if two runs ever overlap
# ============================================================================
run_one() {
    local PP_SP=$1
    local TAG=$2
    local MASTER_PORT=$3

    local CKPT_DIR="${OUT_DIR}/ckpt_${TAG}"
    local TB_DIR="${OUT_DIR}/tb/${TAG}"
    local LOG_FILE="${OUT_DIR}/log_${TAG}.txt"
    mkdir -p "${TB_DIR}"
    if [ "${NO_SAVE}" != "1" ]; then
        mkdir -p "${CKPT_DIR}"
    fi
    local LOAD_ARGS=""
    if [ "${RESUME}" = "1" ]; then
        if [ "${NO_SAVE}" = "1" ]; then
            echo "WARNING: RESUME=1 ignored because NO_SAVE=1 disables checkpoint paths."
        else
            LOAD_ARGS="--load ${CKPT_DIR}"
        fi
    fi
    local SAVE_ARGS=""
    if [ "${NO_SAVE}" != "1" ]; then
        SAVE_ARGS="--save-interval ${SAVE_INTERVAL} --save ${CKPT_DIR}"
    fi

    local DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} \
                            --nnodes 1 \
                            --rdzv_id=1 \
                            --rdzv_backend=c10d \
                            --rdzv_endpoint=localhost:${MASTER_PORT} \
                            --max_restarts=0"

    local options=" \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
        --pipe-sp-strategy average \
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
        --lr-decay-style cosine \
        --lr-warmup-iters ${EFFECTIVE_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --weight-decay ${WEIGHT_DECAY} \
        --clip-grad ${CLIP_GRAD} \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.006 \
        --initial-loss-scale 65536 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        ${SAVE_ARGS} \
        --tensorboard-dir ${TB_DIR} \
        --tensorboard-queue-size 10 \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --use-deltanet \
        --deltanet-mode chunk \
        --deltanet-conv-size 4 \
        --deltanet-qk-activation silu \
        --deltanet-qk-norm l2 \
        --deltanet-use-beta \
        --deltanet-use-short-conv \
        --untie-embeddings-and-output-weights \
        --hidden-dropout 0 \
        --attention-dropout 0 \
        --no-async-tensor-model-parallel-allreduce \
        --seed 42 \
        --no-save-optim \
        --no-save-rng \
        --use-distributed-optimizer \
        --bf16 \
        --data-path ${DATA_PREFIX} \
        --vocab-file ${VOCAB} \
        --merge-file ${MERGE} \
        --split 98,2,0 \
        --tokenizer-type GPT2BPETokenizer \
        ${LOAD_ARGS} \
    "

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ Starting: ${TAG}  (PP_SP=${PP_SP}, port=${MASTER_PORT})"
    echo "    log        → ${LOG_FILE}"
    if [ "${NO_SAVE}" = "1" ]; then
        echo "    ckpt       → disabled"
    else
        echo "    ckpt       → ${CKPT_DIR}"
    fi
    echo "    tensorboard→ ${TB_DIR}"
    echo "    started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local run_cmd="torchrun ${DISTRIBUTED_ARGS} ${DIR}/pretrain_gpt.py ${options}"

    # tee to log; keep pipefail so we catch real failures. Also echo the command.
    echo "${run_cmd}" | tee "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
    eval ${run_cmd} 2>&1 | tee -a "${LOG_FILE}"
    local status=${PIPESTATUS[0]}

    echo ""
    echo "  ✓ ${TAG} finished at $(date '+%Y-%m-%d %H:%M:%S')  (exit=${status})"
    return ${status}
}

# ============================================================================
# Dispatch
# ============================================================================
case "${ONLY}" in
    sp1)
        run_one 1 "fineweb_sp1" 29600
        ;;
    seq|sp${SEQ1F1B_SP})
        run_one "${SEQ1F1B_SP}" "${SEQ_TAG}" 29601
        ;;
    both)
        # Run the Seq1F1B variant first so user sees the comparison curve sooner,
        # then SP=1 baseline. If Seq1F1B fails, SP=1 is skipped.
        run_one "${SEQ1F1B_SP}" "${SEQ_TAG}" 29601
        run_one 1 "fineweb_sp1" 29600
        ;;
    *)
        echo "ERROR: ONLY must be one of {sp1, seq, both, sp${SEQ1F1B_SP}}, got '${ONLY}'"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  ✓ All runs complete"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Inspect logs:"
echo "    grep 'validation loss' ${OUT_DIR}/log_fineweb_sp*.txt"
echo "    grep 'lm loss'         ${OUT_DIR}/log_fineweb_sp*.txt | head -50"
echo "  Tensorboard:"
echo "    tensorboard --logdir ${OUT_DIR}/tb --port 6006"
echo "═══════════════════════════════════════════════════════════════════════"
