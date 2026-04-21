#!/bin/bash
# 只跑 no-short-conv 的 SP1 与 SP4 两个实验，默认抓多个 checkpoint iter 看累积发散。
# 用法:
#   DATA_PATH=/path/to/data bash tests/run_noconv_compare.sh
#   DATA_PATH=/path/to/data TRAIN_ITER=500 SAVE_EXTRA_ITERS=10,50,100,200,500 \
#       bash tests/run_noconv_compare.sh
#
# 跑完后用 tests/analyze_alignment_outputs.py 分析：
#   python3 tests/analyze_alignment_outputs.py --tag-a sp1_noconv --tag-b sp4_noconv \
#       --iters 1 2 10 50 100 200 500
set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
SAVE_DIR="${DIR}/tests/alignment_outputs"
mkdir -p "${SAVE_DIR}"

# Default to a longer run so we can see cumulative drift
TRAIN_ITER=${TRAIN_ITER:-500}
# Iters at which we dump full hidden states + per-layer outputs + grad norms.
# iter 1 and iter 2 are always included by the python side.
SAVE_EXTRA_ITERS=${SAVE_EXTRA_ITERS:-"10,50,100,200,500"}
export SAVE_EXTRA_ITERS

# 把用户传入的额外参数（例如 --fp32）转发到内层脚本；并默认加 --dump-layer-stats
EXTRA_ARGS=("$@")
HAS_DUMP=0
for a in "${EXTRA_ARGS[@]:-}"; do
    if [ "$a" = "--dump-layer-stats" ]; then
        HAS_DUMP=1
    fi
done
if [ "$HAS_DUMP" = "0" ]; then
    EXTRA_ARGS+=("--dump-layer-stats")
fi

# Convert comma-separated SAVE_EXTRA_ITERS to space-separated for analyzer
ANALYZER_ITERS="1 2 $(echo ${SAVE_EXTRA_ITERS} | tr ',' ' ')"

echo "======================================================"
echo "  Step 1: SP1 no-short-conv (baseline, 串行全序列)"
echo "  TRAIN_ITER=${TRAIN_ITER}  SAVE_EXTRA_ITERS=${SAVE_EXTRA_ITERS}"
echo "======================================================"
PP_SP=1 NO_SHORT_CONV=1 MASTER_PORT=29510 TRAIN_ITER=${TRAIN_ITER} \
    SAVE_EXTRA_ITERS=${SAVE_EXTRA_ITERS} \
    bash "${DIR}/tests/run_alignment_test.sh" "${EXTRA_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Step 2: SP4 no-short-conv (Seq1F1B 分4段)"
echo "======================================================"
PP_SP=4 NO_SHORT_CONV=1 MASTER_PORT=29511 TRAIN_ITER=${TRAIN_ITER} \
    SAVE_EXTRA_ITERS=${SAVE_EXTRA_ITERS} \
    bash "${DIR}/tests/run_alignment_test.sh" "${EXTRA_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Step 3: 分析 sp1_noconv vs sp4_noconv"
echo "  iters of interest: ${ANALYZER_ITERS}"
echo "======================================================"
python3 "${DIR}/tests/analyze_alignment_outputs.py" \
    --dir "${SAVE_DIR}" \
    --tag-a sp1_noconv \
    --tag-b sp4_noconv \
    --iters ${ANALYZER_ITERS} \
    2>&1 | tee "${SAVE_DIR}/compare_noconv_result.txt"

echo ""
echo "结果保存在: ${SAVE_DIR}/compare_noconv_result.txt"
