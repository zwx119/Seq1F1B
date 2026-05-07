#!/usr/bin/env bash
# Run/profiling wrapper for the DeltaNet solve_wu H100 CC/TC experiment.
#
# Default:
#   bash tests/run_h100_cc_tc_experiment.sh
#
# Nsight Systems with NVTX ranges:
#   PROFILE=nsys bash tests/run_h100_cc_tc_experiment.sh
#
# Nsight Compute for the fused kernel:
#   PROFILE=ncu MODE=fused bash tests/run_h100_cc_tc_experiment.sh
#
# Useful overrides:
#   T=16384 N_ITER=100 REPEATS=5 DTYPE=bf16 bash tests/run_h100_cc_tc_experiment.sh
#   NORMALIZE_K=0 bash tests/run_h100_cc_tc_experiment.sh  # old raw-K stress input

set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
FLA_DIR=${FLA_DIR:-"${DIR}/flash-linear-attention"}
OUT_DIR=${OUT_DIR:-"${DIR}/tests/h100_cc_tc_outputs"}
mkdir -p "${OUT_DIR}"

export PYTHONPATH="${FLA_DIR}:${DIR}:${PYTHONPATH:-}"

MODE=${MODE:-both}          # original, fused, both
PROFILE=${PROFILE:-none}    # none, nsys, ncu

B=${B:-1}
T=${T:-8192}
H=${H:-32}
K=${K:-80}
V=${V:-80}
DTYPE=${DTYPE:-bf16}
WARMUP=${WARMUP:-10}
N_ITER=${N_ITER:-50}
REPEATS=${REPEATS:-3}
NORMALIZE_K=${NORMALIZE_K:-1}
BETA_SCALE=${BETA_SCALE:-0.1}

BENCH="${FLA_DIR}/tests/bench_h100_cc_tc_solve_wu.py"

COMMON_ARGS=(
  --mode "${MODE}"
  --batch "${B}"
  --seq "${T}"
  --heads "${H}"
  --key-dim "${K}"
  --value-dim "${V}"
  --dtype "${DTYPE}"
  --warmup "${WARMUP}"
  --n-iter "${N_ITER}"
  --repeats "${REPEATS}"
  --beta-scale "${BETA_SCALE}"
)

case "${NORMALIZE_K}" in
  1|true|TRUE|yes|YES|on|ON)
    COMMON_ARGS+=(--normalize-k)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    COMMON_ARGS+=(--no-normalize-k)
    ;;
  *)
    echo "ERROR: NORMALIZE_K must be 0/1, true/false, yes/no, or on/off; got ${NORMALIZE_K}" >&2
    exit 1
    ;;
esac

echo "H100 CC/TC solve_wu experiment"
echo "  FLA_DIR=${FLA_DIR}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  mode=${MODE} profile=${PROFILE}"
echo "  shape=B${B}_T${T}_H${H}_K${K}_V${V}_${DTYPE}"
echo "  normalize_k=${NORMALIZE_K} beta_scale=${BETA_SCALE}"

case "${PROFILE}" in
  none)
    python3 "${BENCH}" "${COMMON_ARGS[@]}"
    ;;

  nsys)
    if ! command -v nsys >/dev/null 2>&1; then
      echo "ERROR: nsys not found in PATH" >&2
      exit 1
    fi
    nsys profile \
      -t cuda,nvtx \
      -o "${OUT_DIR}/nsys_solve_wu_${MODE}_T${T}" \
      --force-overwrite true \
      python3 "${BENCH}" "${COMMON_ARGS[@]}" --profile
    ;;

  ncu)
    if ! command -v ncu >/dev/null 2>&1; then
      echo "ERROR: ncu not found in PATH" >&2
      exit 1
    fi
    if [ "${MODE}" = "both" ]; then
      echo "ERROR: use MODE=original or MODE=fused with PROFILE=ncu to keep reports clean" >&2
      exit 1
    fi
    if [ "${MODE}" = "fused" ]; then
      KERNEL_REGEX="regex:.*fused_solve_wu_fwd_kernel.*"
    else
      KERNEL_REGEX="regex:.*(solve_tril|recompute_w_u_fwd_kernel).*"
    fi
    ncu \
      --target-processes all \
      --kernel-name-base function \
      --kernel-name "${KERNEL_REGEX}" \
      --set full \
      --force-overwrite \
      --export "${OUT_DIR}/ncu_solve_wu_${MODE}_T${T}" \
      python3 "${BENCH}" "${COMMON_ARGS[@]}" --profile --skip-correctness
    ;;

  *)
    echo "ERROR: PROFILE must be one of none, nsys, ncu; got ${PROFILE}" >&2
    exit 1
    ;;
esac
