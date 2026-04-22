#!/bin/bash
# Download FineWeb-Edu `sample-10BT` subset (~25 GB) with fault-tolerance.
#
# Features:
#   - Auto-falls back to hf-mirror.com if the official endpoint is slow
#   - Uses hf_transfer for multi-connection acceleration (if available)
#   - Resumable via huggingface-cli's built-in retry/resume
#   - Verifies at the end that every *.parquet is non-empty and readable
#
# Usage:
#   bash tools/download_fineweb_edu.sh                # default = sample-10BT
#   SUBSET=sample-100BT bash tools/download_fineweb_edu.sh
#   USE_MIRROR=1 bash tools/download_fineweb_edu.sh   # force hf-mirror
#   USE_MIRROR=0 bash tools/download_fineweb_edu.sh   # force official
#
# Output directory:
#   data/fineweb-edu-<SUBSET>/sample/<SUBSET>/*.parquet
#   (matches HF repo layout so resume works on retry)

set -euo pipefail

SUBSET=${SUBSET:-sample-10BT}        # sample-10BT | sample-100BT | sample-350BT
OUT_DIR=${OUT_DIR:-data/fineweb-edu-${SUBSET}}
# Auto-detect mirror: if USE_MIRROR not set, probe both endpoints.
USE_MIRROR=${USE_MIRROR:-auto}
WORKERS=${WORKERS:-16}

mkdir -p "${OUT_DIR}"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  FineWeb-Edu downloader"
echo "    SUBSET   = ${SUBSET}"
echo "    OUT_DIR  = ${OUT_DIR}"
echo "    WORKERS  = ${WORKERS}"
echo "═══════════════════════════════════════════════════════════════════════"

# -------- 1. pick endpoint --------
probe() {
    local url="$1"
    local speed
    speed=$(curl -o /dev/null -w "%{speed_download}" -s --max-time 15 "${url}" 2>/dev/null || echo 0)
    # trim decimal
    speed=${speed%.*}
    echo "${speed:-0}"
}

if [ "${USE_MIRROR}" = "auto" ]; then
    echo "Probing endpoints..."
    TEST_FILE="datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet"
    S_OFFICIAL=$(probe "https://huggingface.co/${TEST_FILE}")
    S_MIRROR=$(probe   "https://hf-mirror.com/${TEST_FILE}")
    echo "    huggingface.co : ${S_OFFICIAL} B/s"
    echo "    hf-mirror.com  : ${S_MIRROR} B/s"
    if [ "${S_MIRROR}" -gt "${S_OFFICIAL}" ]; then
        USE_MIRROR=1
    else
        USE_MIRROR=0
    fi
fi

if [ "${USE_MIRROR}" = "1" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    echo "→ Using hf-mirror.com"
else
    unset HF_ENDPOINT || true
    echo "→ Using huggingface.co (official)"
fi

# -------- 2. install deps if missing --------
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "Installing huggingface_hub + hf_transfer..."
    # IMPORTANT: pin huggingface-hub<1.0 to avoid breaking `transformers` (fla dep).
    # Using --no-deps on hf_transfer to keep other packages untouched.
    pip install -q "huggingface_hub[cli]>=0.34.0,<1.0"
    pip install -q --no-deps hf_transfer || true
fi
# hf_transfer: multi-connection downloader, 3-5x faster on fat pipes.
# If hf_transfer import fails at runtime, HF cli will fall back gracefully.
export HF_HUB_ENABLE_HF_TRANSFER=1

# -------- 3. download with resume (idempotent) --------
echo ""
echo "Downloading ${SUBSET} (resumable; safe to re-run if interrupted)..."
# Map SUBSET → include pattern
# sample-10BT  => sample/10BT/*
# sample-100BT => sample/100BT/*
# sample-350BT => sample/350BT/*
INCLUDE_PAT="sample/${SUBSET#sample-}/*"

# Retry loop — if the download dies mid-way (broken pipe, TLS reset, ...)
# just re-invoke; HF cli will skip already-complete files.
MAX_RETRIES=${MAX_RETRIES:-10}
for attempt in $(seq 1 ${MAX_RETRIES}); do
    echo "── attempt ${attempt}/${MAX_RETRIES} ──"
    if huggingface-cli download \
         HuggingFaceFW/fineweb-edu \
         --repo-type dataset \
         --include "${INCLUDE_PAT}" \
         --local-dir "${OUT_DIR}" \
         --max-workers "${WORKERS}" ; then
        echo "✓ download completed"
        break
    else
        echo "✗ attempt ${attempt} failed, retrying in 5s..."
        sleep 5
    fi
    if [ "${attempt}" = "${MAX_RETRIES}" ]; then
        echo "!! reached MAX_RETRIES, giving up"
        exit 1
    fi
done

# -------- 4. verify --------
echo ""
echo "Verifying parquet files..."
python3 - <<PYEOF
import os, glob, sys
import pyarrow.parquet as pq
root = "${OUT_DIR}"
files = sorted(glob.glob(os.path.join(root, "**/*.parquet"), recursive=True))
print(f"found {len(files)} parquet files under {root}")
bad = []
for i, f in enumerate(files):
    try:
        pq.ParquetFile(f).metadata.num_rows
    except Exception as e:
        bad.append((f, repr(e)))
    if i % 50 == 0:
        print(f"  verified {i+1}/{len(files)}", flush=True)
if bad:
    print("!! corrupted files:")
    for f, err in bad[:10]:
        print(f"   {f}  {err}")
    print(f"   ... total {len(bad)} bad files")
    sys.exit(1)
print(f"✓ all {len(files)} files OK")
PYEOF

TOTAL_SIZE=$(du -sh "${OUT_DIR}" 2>/dev/null | awk '{print $1}')
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  ✓ FineWeb-Edu ${SUBSET} ready at ${OUT_DIR}  (${TOTAL_SIZE})"
echo "  Next: bash tools/preprocess_fineweb_edu.sh ${SUBSET}"
echo "═══════════════════════════════════════════════════════════════════════"
