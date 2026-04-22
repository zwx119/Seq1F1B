#!/bin/bash
# Sanity check: run the SAME config (PP_SP=1, no-short-conv) twice, with only
# the floating-point reduction order perturbed between the two runs. Then
# compare, exactly as we compare SP1 vs SP4.
#
# The two runs are algorithmically identical (same code path, same kernels,
# same data, same seed). Any divergence we see is pure floating-point
# accumulation noise. If long-run cos_sim here collapses to ~0.7 by iter 500,
# that proves the SP1-vs-SP4 late-iter "divergence" is NOT a Seq1F1B bug but
# inherent non-associativity of float arithmetic compounded by SGD chaos.
#
# What we change between the two runs (none of these affect the algorithm):
#   run_a (ref): CUDA_DEVICE_MAX_CONNECTIONS=1     (default, deterministic)
#   run_b (alt): CUDA_DEVICE_MAX_CONNECTIONS=8     (more launch concurrency,
#                CUBLAS_WORKSPACE_CONFIG=:16:8      different GEMM algo choices)
#
# Usage (bf16, default):
#   DATA_PATH=/path bash tests/run_sp1_vs_sp1.sh
#
# Usage (fp32):
#   DATA_PATH=/path bash tests/run_sp1_vs_sp1.sh --fp32
#
# Long run to reproduce the iter-500 "divergence":
#   DATA_PATH=/path TRAIN_ITER=500 SAVE_EXTRA_ITERS=10,100,498,499,500 \
#       bash tests/run_sp1_vs_sp1.sh
set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
SAVE_DIR="${DIR}/tests/alignment_outputs"
mkdir -p "${SAVE_DIR}"

TRAIN_ITER=${TRAIN_ITER:-10}
SAVE_EXTRA_ITERS=${SAVE_EXTRA_ITERS:-"5,10"}
export SAVE_EXTRA_ITERS

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Forward user args (e.g. --fp32); always enable per-layer dump for analyzer.
EXTRA_ARGS=("$@")
HAS_DUMP=0
for a in "${EXTRA_ARGS[@]:-}"; do
    [ "$a" = "--dump-layer-stats" ] && HAS_DUMP=1
done
[ "$HAS_DUMP" = "0" ] && EXTRA_ARGS+=("--dump-layer-stats")

ANALYZER_ITERS="1 2 $(echo ${SAVE_EXTRA_ITERS} | tr ',' ' ')"

echo "======================================================"
echo "  Step 1: SP1 no-short-conv, REF run"
echo "          CUDA_DEVICE_MAX_CONNECTIONS=1"
echo "          TRAIN_ITER=${TRAIN_ITER}  SAVE_EXTRA_ITERS=${SAVE_EXTRA_ITERS}"
echo "======================================================"
PP_SP=1 NO_SHORT_CONV=1 MASTER_PORT=29520 \
    TRAIN_ITER=${TRAIN_ITER} \
    SAVE_EXTRA_ITERS=${SAVE_EXTRA_ITERS} \
    TAG_SUFFIX=ref \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    bash "${DIR}/tests/run_alignment_test.sh" "${EXTRA_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Step 2: SP1 no-short-conv, ALT run"
echo "          CUDA_DEVICE_MAX_CONNECTIONS=8"
echo "          CUBLAS_WORKSPACE_CONFIG=:16:8"
echo "  (same code path, same seed, only FP accumulation order differs)"
echo "======================================================"
PP_SP=1 NO_SHORT_CONV=1 MASTER_PORT=29521 \
    TRAIN_ITER=${TRAIN_ITER} \
    SAVE_EXTRA_ITERS=${SAVE_EXTRA_ITERS} \
    TAG_SUFFIX=alt \
    CUDA_DEVICE_MAX_CONNECTIONS=8 \
    CUBLAS_WORKSPACE_CONFIG=:16:8 \
    bash "${DIR}/tests/run_alignment_test.sh" "${EXTRA_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Step 3: Analyze sp1_noconv_ref vs sp1_noconv_alt"
echo "  iters of interest: ${ANALYZER_ITERS}"
echo "======================================================"
python3 "${DIR}/tests/analyze_alignment_outputs.py" \
    --dir "${SAVE_DIR}" \
    --tag-a sp1_noconv_ref \
    --tag-b sp1_noconv_alt \
    --iters ${ANALYZER_ITERS} \
    2>&1 | tee "${SAVE_DIR}/compare_sp1_vs_sp1_result.txt"

echo ""
echo "结果保存在: ${SAVE_DIR}/compare_sp1_vs_sp1_result.txt"
echo ""
echo "解读提示："
echo "  - iter 1 max_diff 很小、cos_sim ~1.0 -> 两次 run 算法相同（废话，但验证基线）"
echo "  - iter 10 cos_sim 0.999x -> 正常浮点噪声"
echo "  - iter 500 如果也掉到 ~0.7 -> 证明 SP1-vs-SP4 在 500 步的发散是浮点混沌，"
echo "    不是 Seq1F1B 的实现 bug。"
