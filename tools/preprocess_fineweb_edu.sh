#!/bin/bash
# Preprocess FineWeb-Edu parquet -> jsonl -> Megatron bin/idx.
#
# Usage:
#   bash tools/preprocess_fineweb_edu.sh                  # default SUBSET=sample-10BT
#   SUBSET=sample-100BT bash tools/preprocess_fineweb_edu.sh
#
# Produces:
#   data/fineweb_edu_<SUBSET>_text_document.bin
#   data/fineweb_edu_<SUBSET>_text_document.idx
# which can be fed into Megatron via:
#   --data-path data/fineweb_edu_<SUBSET>_text_document

set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
SUBSET=${SUBSET:-sample-10BT}
IN_DIR=${IN_DIR:-${DIR}/data/fineweb-edu-${SUBSET}}
JSONL_PATH=${JSONL_PATH:-${DIR}/data/fineweb-edu-${SUBSET}.jsonl}
OUT_PREFIX=${OUT_PREFIX:-${DIR}/data/fineweb_edu_${SUBSET//-/_}}
VOCAB=${VOCAB:-${DIR}/data/gpt2/gpt2-vocab.json}
MERGE=${MERGE:-${DIR}/data/gpt2/gpt2-merges.txt}
WORKERS=${WORKERS:-64}

echo "═══════════════════════════════════════════════════════════════════════"
echo "  FineWeb-Edu preprocessing  (SUBSET=${SUBSET})"
echo "    IN_DIR     = ${IN_DIR}"
echo "    JSONL_PATH = ${JSONL_PATH}"
echo "    OUT_PREFIX = ${OUT_PREFIX}"
echo "    WORKERS    = ${WORKERS}"
echo "═══════════════════════════════════════════════════════════════════════"

# -------- 1. Fetch GPT-2 vocab/merges if missing --------
if [ ! -f "${VOCAB}" ] || [ ! -f "${MERGE}" ]; then
    echo "Fetching GPT-2 vocab/merges..."
    mkdir -p "$(dirname "${VOCAB}")"
    # Try HF first, then official S3 fallback
    BASE="${HF_ENDPOINT:-https://huggingface.co}/gpt2/resolve/main"
    wget -q -O "${VOCAB}" "${BASE}/vocab.json" || \
        wget -q -O "${VOCAB}" "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json"
    wget -q -O "${MERGE}" "${BASE}/merges.txt" || \
        wget -q -O "${MERGE}" "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt"
    echo "✓ vocab/merges at ${VOCAB} / ${MERGE}"
fi

# -------- 2. parquet -> jsonl (streaming, resumable) --------
if [ -f "${JSONL_PATH}.done" ]; then
    echo "✓ JSONL already built: ${JSONL_PATH}"
else
    echo ""
    echo "Step 1/2: converting parquet → jsonl..."
    python3 "${DIR}/tools/parquet_to_jsonl.py" "${IN_DIR}" "${JSONL_PATH}"
fi

# -------- 3. jsonl -> bin/idx --------
if [ -f "${OUT_PREFIX}_text_document.bin" ] && [ -f "${OUT_PREFIX}_text_document.idx" ]; then
    echo ""
    echo "✓ bin/idx already built: ${OUT_PREFIX}_text_document.{bin,idx}"
else
    echo ""
    echo "Step 2/2: tokenizing jsonl → Megatron bin/idx (workers=${WORKERS})..."
    python3 "${DIR}/tools/preprocess_data.py" \
        --input "${JSONL_PATH}" \
        --output-prefix "${OUT_PREFIX}" \
        --vocab-file "${VOCAB}" \
        --merge-file "${MERGE}" \
        --tokenizer-type GPT2BPETokenizer \
        --append-eod \
        --workers "${WORKERS}" \
        --chunk-size 1000
fi

BIN_SIZE=$(du -h "${OUT_PREFIX}_text_document.bin" 2>/dev/null | awk '{print $1}')
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  ✓ Preprocessing complete"
echo "    bin size = ${BIN_SIZE}"
echo "    use in Megatron with:"
echo "      --data-path ${OUT_PREFIX}_text_document"
echo "      --vocab-file ${VOCAB}"
echo "      --merge-file ${MERGE}"
echo "═══════════════════════════════════════════════════════════════════════"
