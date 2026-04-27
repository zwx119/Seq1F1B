#!/bin/bash
# One-command DeltaNet model-size sweep with the FineWeb-Edu runner.
#
# Default goal:
#   Compare ordinary PP (sp1) vs Seq1F1B (sp8 by default) on several model
#   sizes. Each run is independent: if one config OOMs, the script records the
#   failure and continues to the next config.
#
# Example:
#   bash tests/run_model_size_sweep.sh
#
#   SP_LIST="4 8 16" bash tests/run_model_size_sweep.sh
#
#   MODEL_SPECS="m1p3b:24:2048:16 m1p7b:32:2048:16" \
#   SP_LIST="8" TRAIN_ITERS=200 bash tests/run_model_size_sweep.sh
#
# MODEL_SPECS format:
#   name:num_layers:hidden_size:num_heads

set -uo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
RUNNER="${DIR}/tests/run_fineweb_long.sh"

OUT_ROOT="${OUT_ROOT:-${DIR}/tests/exp_model_size_sweep}"
MODEL_SPECS="${MODEL_SPECS:-m0p7b:24:1536:12 m2p6b:32:2560:20 m4b:36:3072:24}"
SP_LIST="${SP_LIST:-8}"
RUN_SP1="${RUN_SP1:-1}"

# Shared training defaults. Override from env if needed.
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PP_SIZE="${PP_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-32768}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH:-4}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
EVAL_ITERS="${EVAL_ITERS:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Use the local FLA fork when present. This avoids accidentally importing a
# broken site-packages namespace package on the training machine.
FLA_DIR="${FLA_DIR:-${DIR}/flash-linear-attention}"
if [ -d "${FLA_DIR}/fla" ]; then
    export PYTHONPATH="${FLA_DIR}:${PYTHONPATH:-}"
fi
export FLA_USE_FUSED_SOLVE_WU="${FLA_USE_FUSED_SOLVE_WU:-1}"

mkdir -p "${OUT_ROOT}"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  DeltaNet model-size sweep"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  OUT_ROOT             = ${OUT_ROOT}"
echo "  MODEL_SPECS          = ${MODEL_SPECS}"
echo "  RUN_SP1              = ${RUN_SP1}"
echo "  SP_LIST              = ${SP_LIST}"
echo "  GPUS/PP              = ${GPUS_PER_NODE}/${PP_SIZE}"
echo "  seq/micro/global     = ${SEQ_LEN}/${MICRO_BATCH}/${GLOBAL_BATCH}"
echo "  iters/warmup         = ${TRAIN_ITERS}/${WARMUP_ITERS}"
echo "  FLA_DIR              = ${FLA_DIR}"
echo "  FLA_USE_FUSED_SOLVE_WU = ${FLA_USE_FUSED_SOLVE_WU}"
if [ -n "${EXTRA_ARGS}" ]; then
    echo "  EXTRA_ARGS           = ${EXTRA_ARGS}"
fi
echo "═══════════════════════════════════════════════════════════════════════"

declare -a RUN_STATUSES=()

run_config() {
    local model_name=$1
    local num_layers=$2
    local hidden=$3
    local heads=$4
    local only=$5
    local sp=$6

    local out_dir="${OUT_ROOT}/${model_name}_L${num_layers}_H${hidden}_A${heads}_seq${SEQ_LEN}_gbs${GLOBAL_BATCH}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ ${model_name}: L=${num_layers}, H=${hidden}, heads=${heads}, ONLY=${only}, SP=${sp}"
    echo "    out → ${out_dir}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    OUT_DIR="${out_dir}" \
    GPUS_PER_NODE="${GPUS_PER_NODE}" \
    PP_SIZE="${PP_SIZE}" \
    SEQ1F1B_SP="${sp}" \
    NUM_LAYERS="${num_layers}" \
    HIDDEN="${hidden}" \
    NUM_HEADS="${heads}" \
    SEQ_LEN="${SEQ_LEN}" \
    MICRO_BATCH="${MICRO_BATCH}" \
    GLOBAL_BATCH="${GLOBAL_BATCH}" \
    TRAIN_ITERS="${TRAIN_ITERS}" \
    WARMUP_ITERS="${WARMUP_ITERS}" \
    EVAL_ITERS="${EVAL_ITERS}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    LOG_INTERVAL="${LOG_INTERVAL}" \
    EXTRA_ARGS="${EXTRA_ARGS}" \
    ONLY="${only}" \
    bash "${RUNNER}"

    local status=$?
    RUN_STATUSES+=("${model_name}:${only}:sp${sp}:${status}")
    if [ "${status}" -ne 0 ]; then
        echo ""
        echo "  ⚠ ${model_name} ${only}/sp${sp} exited with status ${status}; continuing."
    fi
    return 0
}

for spec in ${MODEL_SPECS}; do
    IFS=: read -r model_name num_layers hidden heads extra <<< "${spec}"
    if [ -z "${model_name}" ] || [ -z "${num_layers}" ] || [ -z "${hidden}" ] || [ -z "${heads}" ] || [ -n "${extra:-}" ]; then
        echo "ERROR: bad MODEL_SPECS entry '${spec}'"
        echo "Expected format: name:num_layers:hidden_size:num_heads"
        exit 1
    fi

    # Run sp1 once per model. SEQ1F1B_SP still needs to be >= 2 because the
    # underlying runner validates it globally, so use the first SP_LIST value.
    first_sp=$(echo "${SP_LIST}" | awk '{print $1}')
    if [ "${RUN_SP1}" = "1" ]; then
        run_config "${model_name}" "${num_layers}" "${hidden}" "${heads}" "sp1" "${first_sp}"
    fi

    for sp in ${SP_LIST}; do
        run_config "${model_name}" "${num_layers}" "${hidden}" "${heads}" "seq" "${sp}"
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Sweep finished. Summary"
echo "═══════════════════════════════════════════════════════════════════════"

echo "  Run statuses:"
for s in "${RUN_STATUSES[@]}"; do
    echo "    ${s}"
done
echo ""

python - "${OUT_ROOT}" <<'PY'
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
line_re = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([0-9.]+).*?"
    r"toks/s:\s*([0-9.]+).*?TFlops/s:\s*([0-9.]+).*?mem_each_stage:\s*([^|]+)\|\s*lm loss:\s*([0-9.E+-]+)"
)

rows = []
for log in sorted(root.glob("**/log_fineweb_sp*.txt")):
    vals = []
    for line in log.read_text(errors="ignore").splitlines():
        m = line_re.search(line)
        if not m:
            continue
        iteration = int(m.group(1))
        vals.append(
            {
                "iter": iteration,
                "ms": float(m.group(3)),
                "toks": float(m.group(4)),
                "tflops": float(m.group(5)),
                "mem": m.group(6).strip(),
                "loss": float(m.group(7)),
            }
        )
    if not vals:
        continue
    # Drop the first logged point because it includes initialization/warmup.
    steady = vals[1:] if len(vals) > 1 else vals
    tail = steady[-5:] if len(steady) >= 5 else steady
    model = log.parent.name
    run = log.stem.replace("log_fineweb_", "")
    rows.append(
        (
            model,
            run,
            statistics.mean(v["toks"] for v in tail),
            statistics.mean(v["tflops"] for v in tail),
            statistics.mean(v["ms"] for v in tail),
            max(max(float(x) for x in v["mem"].split(",")) for v in tail),
            vals[-1]["loss"],
        )
    )

if not rows:
    print("No training metrics found yet.")
    sys.exit(0)

print(f"{'model':55s} {'run':8s} {'tok/s':>10s} {'TF/s':>8s} {'ms/iter':>9s} {'peakGB':>8s} {'last_loss':>10s}")
for model, run, toks, tflops, ms, peak_mem, loss in rows:
    print(f"{model:55s} {run:8s} {toks:10.1f} {tflops:8.2f} {ms:9.1f} {peak_mem:8.1f} {loss:10.4f}")
PY

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Raw logs:"
echo "    find ${OUT_ROOT} -name 'log_fineweb_sp*.txt' -print"
echo "═══════════════════════════════════════════════════════════════════════"
