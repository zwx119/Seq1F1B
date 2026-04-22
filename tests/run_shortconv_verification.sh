#!/bin/bash
# End-to-end verification of ShortConvChunkFunc (short-conv cross-chunk grad relay).
#
# Chain A: pure-pytorch unit test (no distributed). Verifies the Function itself
#          — forward & backward are exactly equivalent to a single-call F.conv1d
#          when invoked chunk-by-chunk with grad_dict relay in reverse order.
#
# Chain B: lr=0 SP=1 vs SP=4 with short-conv ENABLED.
#          hidden-state cos_sim should stay ≥ 0.9998 across iters (same bar
#          used for the DeltaNet recurrent-state relay test).
#
# Chain C: piggy-backs on Chain B — analyze_alignment_outputs.py also prints
#          per-layer gradient ratios; q_conv1d / k_conv1d / v_conv1d weight
#          grads must have ratio ≈ 1.00.
#
# Usage:
#   DATA_PATH=/path/to/data bash tests/run_shortconv_verification.sh
#   DATA_PATH=/path/to/data TRAIN_ITER=100 bash tests/run_shortconv_verification.sh
#
# Optional: USE_FP32=1 to run B/C in fp32 for bit-exact signal.

set -euo pipefail

DIR=$(cd "$(dirname "$0")/.." && pwd)
TRAIN_ITER=${TRAIN_ITER:-100}
USE_FP32=${USE_FP32:-0}

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Chain A: ShortConvChunkFunc pure-pytorch unit test"
echo "═══════════════════════════════════════════════════════════════════════"
python3 "${DIR}/tests/test_shortconv_chunk.py"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Chain B & C: lr=0 SP=1 vs SP=4, short-conv ENABLED"
echo "    (hs cos_sim must ≥ 0.9998, conv.weight.grad ratio must ≈ 1.00)"
echo "═══════════════════════════════════════════════════════════════════════"
TRAIN_ITER=${TRAIN_ITER} USE_FP32=${USE_FP32} \
    bash "${DIR}/tests/run_lr0_compare.sh"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Verification done. Check the 'Analysis' section above:"
echo "    • hs cos_sim iter 1 / mid / last  — all ≥ 0.9998  ⇒ Chain B ✓"
echo "    • q_conv1d.weight / k_conv1d.weight / v_conv1d.weight grad ratios"
echo "      printed with |ratio - 1| ≤ 5e-3                ⇒ Chain C ✓"
echo "═══════════════════════════════════════════════════════════════════════"
