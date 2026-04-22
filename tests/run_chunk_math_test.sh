#!/bin/bash
# Run the isolated DeltaNet chunk state-passing math-equivalence test.
#
# This is the DECISIVE proof that Seq1F1B's DeltaNet integration is
# algorithmically correct: it compares a full-sequence chunk_delta_rule
# call against N state-passed chunked calls on the same random inputs,
# bit-for-bit (forward + backward).
#
# No Megatron, no distributed, no NCCL — pure math on a single GPU.
#
# Expected: all cos_sim ≥ 0.9999 (bf16) or ≥ 1 - 1e-7 (fp16).
#
# Usage (on the cluster):
#   bash tests/run_chunk_math_test.sh
# or with custom shape:
#   bash tests/run_chunk_math_test.sh --seq 16384 --n_chunks 8

set -e

cd "$(dirname "$0")/.."

# Use any single visible GPU.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python tests/test_deltanet_chunk_math.py \
    --batch   ${BATCH:-2} \
    --heads   ${HEADS:-8} \
    --seq     ${SEQ:-8192} \
    --dim     ${DIM:-64} \
    --n_chunks ${N_CHUNKS:-4} \
    --dtype   ${DTYPE:-bf16} \
    --seeds   ${SEEDS:-0 1 2 3} \
    --cos_sim_thresh ${THRESH:-0.9999} \
    "$@"
