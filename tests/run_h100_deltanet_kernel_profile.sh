#!/usr/bin/env bash
# Profile the full DeltaNet chunk forward operator on H100.
#
# This is broader than run_h100_cc_tc_experiment.sh: it runs chunk_delta_rule()
# and can profile the complete 1-5 forward kernel chain.
#
# Examples:
#   bash tests/run_h100_deltanet_kernel_profile.sh
#   PROFILE=ncu_all SOLVE_WU_IMPL=original N_ITER=1 WARMUP=0 bash tests/run_h100_deltanet_kernel_profile.sh
#   PROFILE=nsys SOLVE_WU_IMPL=original bash tests/run_h100_deltanet_kernel_profile.sh

set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
FLA_DIR=${FLA_DIR:-"${DIR}/flash-linear-attention"}
OUT_DIR=${OUT_DIR:-"${DIR}/tests/h100_cc_tc_outputs"}
mkdir -p "${OUT_DIR}"

export PYTHONPATH="${FLA_DIR}:${DIR}:${PYTHONPATH:-}"

PROFILE=${PROFILE:-none}              # none, nsys, ncu_all
SOLVE_WU_IMPL=${SOLVE_WU_IMPL:-original}  # original, fused, hopper, overlap

B=${B:-1}
T=${T:-32768}
H=${H:-32}
K=${K:-80}
V=${V:-80}
DTYPE=${DTYPE:-bf16}
WARMUP=${WARMUP:-1}
N_ITER=${N_ITER:-3}
REPEATS=${REPEATS:-1}
NORMALIZE_QK=${NORMALIZE_QK:-1}
BETA_SCALE=${BETA_SCALE:-0.1}
NCU_SET=${NCU_SET:-full}
WARM_CACHE=${WARM_CACHE:-1}

BENCH="${FLA_DIR}/tests/bench_h100_deltanet_kernels.py"

COMMON_ARGS=(
  --solve-wu-impl "${SOLVE_WU_IMPL}"
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

case "${NORMALIZE_QK}" in
  1|true|TRUE|yes|YES|on|ON)
    COMMON_ARGS+=(--normalize-qk)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    COMMON_ARGS+=(--no-normalize-qk)
    ;;
  *)
    echo "ERROR: NORMALIZE_QK must be 0/1, true/false, yes/no, or on/off; got ${NORMALIZE_QK}" >&2
    exit 1
    ;;
esac

echo "H100 full DeltaNet kernel profile"
echo "  FLA_DIR=${FLA_DIR}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  solve_wu_impl=${SOLVE_WU_IMPL} profile=${PROFILE}"
echo "  shape=B${B}_T${T}_H${H}_K${K}_V${V}_${DTYPE}"
echo "  normalize_qk=${NORMALIZE_QK} beta_scale=${BETA_SCALE}"
echo "  warmup/n_iter/repeats=${WARMUP}/${N_ITER}/${REPEATS}"

KERNEL_REGEX="regex:.*(chunk_scaled_dot_kkt_fwd_kernel|merge_16x16_to_64x64_inverse_kernel|solve_tril_16x16_kernel|merge_16x16_to_32x32_inverse_kernel|recompute_w_u_fwd_kernel|chunk_gated_delta_rule_fwd_kernel_h_blockdim64|chunk_fwd_kernel_o|fused_solve_wu_fwd_kernel|hopper_solve_tril_64x64_kernel|overlap_solve_tril_64x64_kernel).*"

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
      -o "${OUT_DIR}/nsys_deltanet_${SOLVE_WU_IMPL}_T${T}" \
      --force-overwrite true \
      python3 "${BENCH}" "${COMMON_ARGS[@]}" --profile
    ;;

  ncu_all)
    if ! command -v ncu >/dev/null 2>&1; then
      echo "ERROR: ncu not found in PATH" >&2
      exit 1
    fi
    case "${WARM_CACHE}" in
      1|true|TRUE|yes|YES|on|ON)
        echo "  warming Triton/CUDA cache before ncu..."
        python3 "${BENCH}" "${COMMON_ARGS[@]}" --warmup 1 --n-iter 1 --repeats 1 >/dev/null
        ;;
      0|false|FALSE|no|NO|off|OFF)
        ;;
      *)
        echo "ERROR: WARM_CACHE must be 0/1, true/false, yes/no, or on/off; got ${WARM_CACHE}" >&2
        exit 1
        ;;
    esac
    ncu \
      --target-processes all \
      --kernel-name-base function \
      --kernel-name "${KERNEL_REGEX}" \
      --set "${NCU_SET}" \
      --force-overwrite \
      --export "${OUT_DIR}/ncu_deltanet_${SOLVE_WU_IMPL}_T${T}" \
      python3 "${BENCH}" "${COMMON_ARGS[@]}" --warmup 0 --repeats 1 --profile
    ;;

  *)
    echo "ERROR: PROFILE must be one of none, nsys, ncu_all; got ${PROFILE}" >&2
    exit 1
    ;;
esac
