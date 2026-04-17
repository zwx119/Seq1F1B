#!/usr/bin/env python
# Copyright (c) 2024, Seq1F1B-DeltaNet Contributors.
"""
Alignment test: verify that DeltaNet Seq1F1B splitting produces identical
hidden states compared to serial (full-sequence) execution.

Uses the REAL DeltaNetTransformerLayer from megatron/model/deltanet_layer.py
with a minimal single-GPU Megatron mock environment.

Usage:
    python tests/test_deltanet_alignment.py [--num-layers 4] [--splits 4] [--seq-len 1024]
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# 0. Parse test-level CLI args BEFORE Megatron touches sys.argv
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument("--num-layers", type=int, default=4)
_parser.add_argument("--splits", type=int, default=4)
_parser.add_argument("--seq-len", type=int, default=1024)
_parser.add_argument("--hidden-size", type=int, default=256)
_parser.add_argument("--num-heads", type=int, default=4)
_parser.add_argument("--batch-size", type=int, default=2)
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--atol", type=float, default=1e-2)
_parser.add_argument("--rtol", type=float, default=1e-2)
_test_args = _parser.parse_args()

# Wipe sys.argv so Megatron's argument parser doesn't choke
sys.argv = [sys.argv[0]]

# ---------------------------------------------------------------------------
# 1. Minimal torch.distributed init (single GPU, required by TP layers)
# ---------------------------------------------------------------------------
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

if not dist.is_initialized():
    dist.init_process_group(backend="nccl", world_size=1, rank=0)

# ---------------------------------------------------------------------------
# 2. Set up fake Megatron args via global_vars
# ---------------------------------------------------------------------------
from megatron.global_vars import set_args

_fake_args = SimpleNamespace(
    # Pipeline / Seq1F1B
    pipe_sp_splits=1,  # will be changed per test
    # Linear layer bias
    add_bias_linear=False,
    # LayerNorm
    no_persist_layer_norm=True,
    apply_layernorm_1p=False,
    # MLP
    num_experts=None,
    swiglu=False,
    openai_gelu=False,
    onnx_safe=False,
    squared_relu=False,
    bias_gelu_fusion=False,
    # DeltaNet flags
    use_deltanet=True,
    deltanet_use_short_conv=True,
    deltanet_conv_size=4,
    deltanet_use_beta=True,
    deltanet_use_output_gate=False,
    deltanet_qk_activation="silu",
    deltanet_qk_norm="l2",
    deltanet_mode="chunk",
    # Misc required by various code paths
    retro_add_retriever=False,
    sequence_parallel=False,
    params_dtype=torch.bfloat16,
    rank=0,
    # For MegatronModule / grad sync
    DDP_impl="local",
    use_contiguous_buffers_in_local_ddp=False,
    # Transformer config fields read in some paths
    hidden_dropout=0.0,
    attention_dropout=0.0,
    apply_residual_connection_post_layernorm=False,
    fp32_residual_connection=False,
    bias_dropout_fusion=False,
    masked_softmax_fusion=False,
    gradient_accumulation_fusion=False,
    async_tensor_model_parallel_allreduce=False,
)
set_args(_fake_args)

# ---------------------------------------------------------------------------
# 3. Initialise Megatron model-parallel (TP=1, PP=1)
# ---------------------------------------------------------------------------
from megatron.core import mpu
mpu.initialize_model_parallel(
    tensor_model_parallel_size_=1,
    pipeline_model_parallel_size_=1,
)

# ---------------------------------------------------------------------------
# 4. Build TransformerConfig
# ---------------------------------------------------------------------------
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    num_layers=_test_args.num_layers,
    hidden_size=_test_args.hidden_size,
    num_attention_heads=_test_args.num_heads,
    bf16=True,
    params_dtype=torch.bfloat16,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    add_bias_linear=False,
    bias_dropout_fusion=False,
    persist_layer_norm=False,
    sequence_parallel=False,
    apply_residual_connection_post_layernorm=False,
    fp32_residual_connection=False,
)

# ---------------------------------------------------------------------------
# 5. Build layers (real DeltaNetTransformerLayer)
# ---------------------------------------------------------------------------
from megatron.model.deltanet_layer import DeltaNetTransformerLayer

device = torch.device("cuda")

layers = torch.nn.ModuleList([
    DeltaNetTransformerLayer(config, layer_number=i + 1)
    for i in range(_test_args.num_layers)
]).to(device).to(torch.bfloat16)

# ---------------------------------------------------------------------------
# 6. Helper: run all layers in serial or split mode
# ---------------------------------------------------------------------------

def _clear_states(layers):
    """Reset recurrent / conv states on every DeltaNetAttention."""
    for layer in layers:
        attn = layer.self_attention
        attn.recurrent_state = None
        attn.conv_state_q = None
        attn.conv_state_k = None
        attn.conv_state_v = None


def run_serial(layers, x):
    """Full-sequence forward: pipe_sp_splits=1, micro_sp_idx=None."""
    _fake_args.pipe_sp_splits = 1
    _clear_states(layers)
    h = x.clone()
    with torch.no_grad():
        for layer in layers:
            h = layer(h, attention_mask=None, micro_sp_idx=None)
    return h


def run_split(layers, x, num_splits):
    """Split-sequence forward: pipe_sp_splits=num_splits."""
    _fake_args.pipe_sp_splits = num_splits
    _clear_states(layers)
    S = x.size(0)
    chunk_size = S // num_splits
    assert S % num_splits == 0, f"seq_len {S} not divisible by splits {num_splits}"

    h_chunks = []
    with torch.no_grad():
        for sp_idx in range(num_splits):
            chunk = x[sp_idx * chunk_size : (sp_idx + 1) * chunk_size].clone()
            h = chunk
            for layer in layers:
                h = layer(h, attention_mask=None, micro_sp_idx=sp_idx)
            h_chunks.append(h)

    return torch.cat(h_chunks, dim=0)


# ---------------------------------------------------------------------------
# 7. Run the test
# ---------------------------------------------------------------------------
def main():
    S = _test_args.seq_len
    B = _test_args.batch_size
    H = _test_args.hidden_size
    N = _test_args.splits

    torch.manual_seed(_test_args.seed)
    torch.cuda.manual_seed_all(_test_args.seed)

    x = torch.randn(S, B, H, device=device, dtype=torch.bfloat16)

    layers.eval()

    # --- Serial (baseline) ---
    out_serial = run_serial(layers, x)

    # --- Split (Seq1F1B style) ---
    out_split = run_split(layers, x, N)

    # --- Compare ---
    max_diff = (out_serial - out_split).abs().max().item()
    mean_diff = (out_serial - out_split).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_serial.flatten().unsqueeze(0).float(),
        out_split.flatten().unsqueeze(0).float(),
    ).item()

    print(f"{'='*60}")
    print(f"DeltaNet Seq1F1B Alignment Test")
    print(f"  layers={_test_args.num_layers}, hidden={H}, heads={_test_args.num_heads}")
    print(f"  seq_len={S}, batch={B}, splits={N}")
    print(f"  dtype=bf16")
    print(f"{'='*60}")
    print(f"  max  |serial - split| = {max_diff:.6e}")
    print(f"  mean |serial - split| = {mean_diff:.6e}")
    print(f"  cosine similarity     = {cos_sim:.10f}")
    print(f"{'='*60}")

    close = torch.allclose(out_serial, out_split, atol=_test_args.atol, rtol=_test_args.rtol)
    if close and cos_sim > 0.999:
        print("PASSED: serial and split outputs are aligned.")
    else:
        print("FAILED: outputs diverge beyond tolerance.")
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
