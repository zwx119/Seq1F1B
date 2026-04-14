"""
profile_triton_bench.py — 用 triton.testing.do_bench 精确测量每个 kernel

不依赖 ncu，直接得到:
  - 精确耗时 (μs)
  - FLOPS 计算 (手动)
  - 有效带宽 (手动)

也可以用 TRITON_PRINT_AUTOTUNING=1 环境变量查看 autotune 选择的配置。

用法:
  # 基本测量
  python profile_triton_bench.py --T 8192 --B 1

  # 查看 autotune 选中的配置
  TRITON_PRINT_AUTOTUNING=1 python profile_triton_bench.py --T 8192 --B 1

  # 查看 Triton IR（含寄存器数）
  MLIR_ENABLE_DUMP=1 python profile_triton_bench.py --T 8192 --B 1 --target 4 2>&1 | grep "register"
"""

import argparse
import torch
import torch.nn.functional as F
import triton

parser = argparse.ArgumentParser()
parser.add_argument("--B", type=int, default=1)
parser.add_argument("--T", type=int, default=8192)
parser.add_argument("--H", type=int, default=32)
parser.add_argument("--D", type=int, default=128)
args = parser.parse_args()

B, T, H, D = args.B, args.T, args.H, args.D
BT = 64
NT = T // BT
device = "cuda"
dtype = torch.bfloat16

print(f"Config: B={B}, T={T}, H={H}, D={D}, BT={BT}, NT={NT}")
print(f"GPU: {torch.cuda.get_device_name()}")
print()

# ============================================================
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o

# ============================================================
torch.manual_seed(42)
q = torch.randn(B, T, H, D, device=device, dtype=dtype)
k = F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), p=2, dim=-1)
v = torch.randn(B, T, H, D, device=device, dtype=dtype)
beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
scale = D ** -0.5

# ============================================================
# Warmup — autotune
# ============================================================
print("Warmup...")
with torch.no_grad():
    for _ in range(3):
        A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=BT, output_dtype=torch.float32)
        A_inv = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=None, initial_state=None,
            output_final_state=True, cu_seqlens=None,
        )
        o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None)
torch.cuda.synchronize()
print("Warmup done.\n")

# 预计算中间结果
with torch.no_grad():
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=BT, output_dtype=torch.float32)
    A_inv = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=None, initial_state=None,
        output_final_state=True, cu_seqlens=None,
    )
torch.cuda.synchronize()


# ============================================================
# Benchmark 每个 kernel
# ============================================================
def bench(fn, name, flops=None, hbm_bytes=None, warmup=100, rep=500):
    """用 triton.testing.do_bench 精确测量"""
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    line = f"  {name:45s}  {ms*1000:8.1f} μs"
    if flops:
        tflops = flops / (ms * 1e-3) / 1e12
        line += f"  | {tflops:6.2f} TFLOPS"
    if hbm_bytes:
        gbps = hbm_bytes / (ms * 1e-3) / 1e9
        line += f"  | {gbps:7.1f} GB/s"
    print(line)
    return ms


print("=" * 80)
print("Per-kernel benchmark (triton.testing.do_bench, median of 500 runs)")
print("=" * 80)

# --- 🔥1 ---
# FLOPs: NT * (2 * BT * K * BT) = NT * 2*64*128*64
flops_1 = B * H * NT * (2 * BT * D * BT)
# HBM: read k [B,T,H,D] (shared), write A [B,T,H,BT]
hbm_1 = B * T * H * D * 2 + B * T * H * BT * 4  # read bf16 + write fp32
ms1 = bench(
    lambda: chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=BT, output_dtype=torch.float32),
    "🔥1 chunk_scaled_dot_kkt", flops=flops_1, hbm_bytes=hbm_1,
)

# --- 🔥2 ---
# FLOPs very small: ~NT * 133K (see kernel_flops_analysis.md)
flops_2 = B * H * NT * 133_000
hbm_2 = B * T * H * BT * 4 * 2  # read A_raw fp32 + write A_inv fp32
ms2 = bench(
    lambda: solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype),
    "🔥2 solve_tril", flops=flops_2, hbm_bytes=hbm_2,
)

# --- 🔥3 ---
# FLOPs: NT * 2*BT*BT*(K+V) = NT * 2*64*64*(128+128)
flops_3 = B * H * NT * (2 * BT * BT * (D + D))
hbm_3 = (B * T * H * BT * 4  # read A_inv
          + B * T * H * D * 2 * 2  # read k,v bf16
          + B * T * H * D * 2 * 2  # write w,u bf16
          + B * T * H * 2)  # read beta bf16
ms3 = bench(
    lambda: recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None),
    "🔥3 recompute_w_u", flops=flops_3, hbm_bytes=hbm_3,
)

# --- 🔥4 ---
# FLOPs: B*H * NT * (2*2*64*BT*BV) where BV~64, plus v_new correction
# Per chunk: 2 * (2*64*BT*BV) for w@h and k^T@v
flops_4 = B * H * NT * (4 * 64 * BT * D)  # approx, K-tiles * BT * BV
hbm_4 = (B * T * H * D * 2 * 2  # read w,k bf16
          + B * T * H * D * 2  # read u/v bf16
          + B * NT * H * D * D * 2  # write h bf16 (big!)
          + B * T * H * D * 2)  # write v_new bf16
ms4 = bench(
    lambda: chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=None, initial_state=None,
        output_final_state=True, cu_seqlens=None,
    ),
    "🔥4 chunk_fwd_h", flops=flops_4, hbm_bytes=hbm_4,
)

# --- 🔥5 ---
# FLOPs: NT * B*H * (2*BT*K*(BV+BT) + 2*BT*BT*BV) per V-tile, cdiv(V,BV) tiles
# Simplified: NT * B*H * (2*BT*K*BV + 2*BT*K*BT + 2*BT*BT*BV)
flops_5 = B * H * NT * (2*BT*D*D + 2*BT*D*BT + 2*BT*BT*D)
hbm_5 = (B * T * H * D * 2 * 2  # read q,k bf16
          + B * NT * H * D * D * 2  # read h bf16
          + B * T * H * D * 2  # read v_new bf16
          + B * T * H * D * 2)  # write o bf16
ms5 = bench(
    lambda: chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None),
    "🔥5 chunk_fwd_o", flops=flops_5, hbm_bytes=hbm_5,
)

# --- Total ---
print("-" * 80)
total_ms = ms1 + ms2 + ms3 + ms4 + ms5
total_flops = flops_1 + flops_2 + flops_3 + flops_4 + flops_5
print(f"  {'Total':45s}  {total_ms*1000:8.1f} μs  | {total_flops/(total_ms*1e-3)/1e12:6.2f} TFLOPS")
print()

# --- Breakdown ---
print("=" * 80)
print("Breakdown (% of total)")
print("=" * 80)
for name, ms in [("🔥1 kkt", ms1), ("🔥2 solve", ms2), ("🔥3 w_u", ms3),
                  ("🔥4 fwd_h", ms4), ("🔥5 fwd_o", ms5)]:
    pct = ms / total_ms * 100
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"  {name:12s} {pct:5.1f}% {bar} {ms*1000:.0f}μs")

print()

# --- Roofline hints ---
print("=" * 80)
print("Roofline analysis")
print("=" * 80)
peak_bf16_tflops = 312  # A100
peak_hbm_gbps = 2039    # A100
for name, ms, flops, hbm in [
    ("🔥1 kkt", ms1, flops_1, hbm_1),
    ("🔥2 solve", ms2, flops_2, hbm_2),
    ("🔥3 w_u", ms3, flops_3, hbm_3),
    ("🔥4 fwd_h", ms4, flops_4, hbm_4),
    ("🔥5 fwd_o", ms5, flops_5, hbm_5),
]:
    achieved_tflops = flops / (ms * 1e-3) / 1e12
    achieved_gbps = hbm / (ms * 1e-3) / 1e9
    ai = flops / hbm if hbm > 0 else 0  # arithmetic intensity (FLOP/Byte)
    ridge_point = peak_bf16_tflops * 1e12 / (peak_hbm_gbps * 1e9)  # ~153 FLOP/Byte

    if ai < ridge_point:
        bound = "MEM-BOUND"
    else:
        bound = "COMPUTE-BOUND"

    pct_compute = achieved_tflops / peak_bf16_tflops * 100
    pct_mem = achieved_gbps / peak_hbm_gbps * 100

    print(f"  {name:12s}  AI={ai:6.1f} FLOP/B  |  {achieved_tflops:5.1f}/{peak_bf16_tflops} TFLOPS ({pct_compute:4.1f}%)  "
          f"|  {achieved_gbps:6.0f}/{peak_hbm_gbps} GB/s ({pct_mem:4.1f}%)  |  {bound}")

print()
print("Done!")
