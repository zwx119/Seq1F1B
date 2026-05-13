#!/usr/bin/env python3
"""Single-GPU DeltaNetAttention forward microbenchmark with NVTX ranges.

This is intended for Nsight Systems profiling of DeltaNet attention forward
only. It avoids torchrun, datasets, pipeline parallelism, and optimizer work.
Run base and H/O-grid modes as separate nsys captures, then compare the NVTX
range durations for DeltaNetAttention.forward and delta_rule_core.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "flash-linear-attention"))

from megatron.core import mpu
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.global_vars import set_args
from megatron.model.deltanet_attention import DeltaNetAttention


def build_args(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        add_bias_linear=False,
        deltanet_conv_size=cli.conv_size,
        deltanet_fused_h_o_pipeline=(cli.mode == "grid"),
        deltanet_mode="chunk",
        deltanet_qk_activation="silu",
        deltanet_qk_norm="l2",
        deltanet_use_beta=True,
        deltanet_use_output_gate=cli.use_output_gate,
        deltanet_use_short_conv=True,
        force_seq_chunks=1,
        pipe_sp_splits=cli.pipe_sp_splits,
    )


def build_config(cli: argparse.Namespace) -> TransformerConfig:
    dtype = torch.bfloat16 if cli.dtype == "bf16" else torch.float16
    return TransformerConfig(
        num_layers=1,
        hidden_size=cli.hidden,
        num_attention_heads=cli.heads,
        init_method_std=cli.init_std,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=(cli.dtype == "bf16"),
        fp16=(cli.dtype == "fp16"),
        sequence_parallel=False,
        perform_initialization=False,
        async_tensor_model_parallel_allreduce=False,
    )


def init_weights(module: torch.nn.Module, std: float) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "bias" in name:
                param.zero_()
            else:
                torch.nn.init.normal_(param, mean=0.0, std=std)


def run(cli: argparse.Namespace) -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    torch.cuda.set_device(cli.device)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed(cli.seed)

    # Single-process model-parallel setup. Tensor-parallel collectives bypass
    # distributed groups when TP world size is one.
    mpu.set_tensor_model_parallel_world_size(1)
    mpu.set_tensor_model_parallel_rank(0)

    os.environ.setdefault("DELTANET_ATTN_NVTX", "1")
    os.environ.pop("DELTANET_ATTN_CUDA_TIMING", None)
    if cli.mode == "grid":
        os.environ.setdefault("FLA_HO_PIPELINE_CONSUMER", "grid")

    set_args(build_args(cli))
    config = build_config(cli)
    dtype = torch.bfloat16 if cli.dtype == "bf16" else torch.float16
    model = DeltaNetAttention(config, layer_number=1).cuda()
    init_weights(model, cli.init_std)
    model.to(dtype=dtype)
    model.eval()

    hidden = torch.randn(
        cli.seq_len,
        cli.batch_size,
        cli.hidden,
        device="cuda",
        dtype=dtype,
    )

    def one_forward(iter_idx: int) -> None:
        # micro_sp_idx=0 makes the module clear state and output final state,
        # matching the Seq1F1B per-span code path used in the 8-GPU run.
        torch.cuda.nvtx.range_push(f"bench.{cli.mode}.iter_{iter_idx}")
        try:
            out, _ = model(hidden, None, micro_sp_idx=0)
            if cli.check_finite and not torch.isfinite(out.float()).all():
                raise RuntimeError("non-finite DeltaNetAttention output")
        finally:
            torch.cuda.nvtx.range_pop()

    with torch.inference_mode():
        for i in range(cli.warmup):
            one_forward(-cli.warmup + i)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if cli.cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(f"bench.{cli.mode}.measured")
        start.record()
        for i in range(cli.iters):
            one_forward(i)
        end.record()
        end.synchronize()
        torch.cuda.nvtx.range_pop()
        if cli.cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()

    avg_ms = start.elapsed_time(end) / cli.iters
    print(
        f"mode={cli.mode} seq_len={cli.seq_len} batch={cli.batch_size} "
        f"hidden={cli.hidden} heads={cli.heads} "
        f"use_output_gate={cli.use_output_gate} "
        f"avg_forward_wall={avg_ms:.3f} ms",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "grid"], required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--use-output-gate", action="store_true")
    parser.add_argument("--pipe-sp-splits", type=int, default=4)
    parser.add_argument("--conv-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--init-std", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--check-finite", action="store_true")
    parser.add_argument("--cuda-profiler-range", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
