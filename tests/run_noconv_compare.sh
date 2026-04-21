#!/bin/bash
# 只跑 no-short-conv 的 SP1 与 SP4 两个实验，默认 5 iters，并开启 --dump-layer-stats
# 用法:
#   DATA_PATH=/path/to/data bash tests/run_noconv_compare.sh
#   DATA_PATH=/path/to/data bash tests/run_noconv_compare.sh --fp32
#
# 跑完后用 tests/analyze_alignment_outputs.py 分析：
#   python3 tests/analyze_alignment_outputs.py --tag-a sp1_noconv --tag-b sp4_noconv
set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
SAVE_DIR="${DIR}/tests/alignment_outputs"
mkdir -p "${SAVE_DIR}"

TRAIN_ITER=${TRAIN_ITER:-5}

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

echo "======================================================"
echo "  Step 1: SP1 no-short-conv (baseline, 串行全序列)"
echo "======================================================"
PP_SP=1 NO_SHORT_CONV=1 MASTER_PORT=29510 TRAIN_ITER=${TRAIN_ITER} \
    bash "${DIR}/tests/run_alignment_test.sh" "${EXTRA_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Step 2: SP4 no-short-conv (Seq1F1B 分4段)"
echo "======================================================"
PP_SP=4 NO_SHORT_CONV=1 MASTER_PORT=29511 TRAIN_ITER=${TRAIN_ITER} \
    bash "${DIR}/tests/run_alignment_test.sh" "${EXTRA_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Step 3: 分析 sp1_noconv vs sp4_noconv，定位首个发散层"
echo "======================================================"
python3 "${DIR}/tests/analyze_alignment_outputs.py" \
    --dir "${SAVE_DIR}" \
    --tag-a sp1_noconv \
    --tag-b sp4_noconv \
    --iters 1 2 \
    2>&1 | tee "${SAVE_DIR}/compare_noconv_result.txt"

echo ""
echo "结果保存在: ${SAVE_DIR}/compare_noconv_result.txt"
