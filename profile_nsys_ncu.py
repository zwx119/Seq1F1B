"""
profile_nsys_ncu.py — 适配 nsys / ncu 的轻量 profiling 脚本

用法：
  # 1) nsys — 全局 timeline (推荐先跑这个)
  nsys profile --trace=cuda,nvtx --output=deltanet_nsys \
    python profile_nsys_ncu.py --mode nsys

  # 2) ncu — 分析最慢的 kernel (🔥4 chunk_fwd_h)
  #    用 --kernel-name 过滤只分析目标 kernel，否则太慢
  ncu --set full --kernel-name "chunk_gated_delta" --launch-skip 3 --launch-count 1 \
    -o deltanet_ncu python profile_nsys_ncu.py --mode ncu

  # 3) 纯 torch.profiler (不需要 nsys/ncu)
  python profile_nsys_ncu.py --mode torch

查看结果：
  nsys: nsys-ui deltanet_nsys.nsys-rep   (或导出 .sqlite 用 nsys stats)
  ncu:  ncu-ui deltanet_ncu.ncu-rep
  torch: 终端直接打印 + deltanet_trace.json (chrome://tracing)
"""

import argparse
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["nsys", "ncu", "torch"], default="torch")
parser.add_argument("--B", type=int, default=2)
parser.add_argument("--T", type=int, default=8192)
parser.add_argument("--H", type=int, default=32)
parser.add_argument("--D", type=int, default=128)
parser.add_argument("--iters", type=int, default=5, help="Number of measured iterations")
parser.add_argument("--backward", action="store_true", help="Also profile backward pass")
args = parser.parse_args()

B, T, H, D = args.B, args.T, args.H, args.D
device = "cuda"
dtype = torch.bfloat16

print(f"Config: B={B}, T={T}, H={H}, D={D}, BT=64, NT={T//64}")
print(f"Mode: {args.mode}, iters={args.iters}, backward={args.backward}")

# ============================================================
# 导入 fla 各子函数
# ============================================================
from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.delta_rule.wy_fast import prepare_wy_repr_fwd, recompute_w_u_fwd
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o

# ============================================================
# 准备数据
# ============================================================
torch.manual_seed(42)
q = torch.randn(B, T, H, D, device=device, dtype=dtype)
k = F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), p=2, dim=-1)
v = torch.randn(B, T, H, D, device=device, dtype=dtype)
beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()

if args.backward:
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    beta.requires_grad_(True)

# ============================================================
# 带 NVTX 标注的 forward（nsys 会显示这些区间）
# ============================================================
def forward_with_nvtx(q, k, v, beta):
    scale = D ** -0.5

    nvtx.range_push("🔥1 chunk_scaled_dot_kkt")
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
    nvtx.range_pop()

    nvtx.range_push("🔥2 solve_tril")
    A = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    nvtx.range_pop()

    nvtx.range_push("🔥3 recompute_w_u")
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=None)
    nvtx.range_pop()

    nvtx.range_push("🔥4 chunk_fwd_h")
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=None, initial_state=None,
        output_final_state=True, cu_seqlens=None,
    )
    nvtx.range_pop()

    nvtx.range_push("🔥5 chunk_fwd_o")
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None)
    nvtx.range_pop()

    return o, final_state

# ============================================================
# Warmup (让 Triton autotune 完成)
# ============================================================
print("Warmup...")
for _ in range(3):
    with torch.no_grad():
        forward_with_nvtx(q, k, v, beta)
torch.cuda.synchronize()
print("Warmup done.\n")

# ============================================================
# 实际 profile
# ============================================================
if args.mode == "nsys":
    # nsys 自带 profiling，这里只要跑代码 + NVTX 标注
    # nsys 会自动捕获所有 CUDA kernel 和 NVTX 区间
    torch.cuda.cudart().cudaProfilerStart()

    for i in range(args.iters):
        nvtx.range_push(f"iter_{i}")

        nvtx.range_push("forward")
        if args.backward:
            o, _ = forward_with_nvtx(q, k, v, beta)
        else:
            with torch.no_grad():
                o, _ = forward_with_nvtx(q, k, v, beta)
        nvtx.range_pop()  # forward

        if args.backward:
            nvtx.range_push("backward")
            loss = o.sum()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None
            beta.grad = None
            nvtx.range_pop()  # backward

        torch.cuda.synchronize()
        nvtx.range_pop()  # iter_i

    torch.cuda.cudart().cudaProfilerStop()
    print(f"nsys profile done. {args.iters} iterations recorded.")
    print("查看: nsys-ui deltanet_nsys.nsys-rep")

elif args.mode == "ncu":
    # ncu 外部启动
    # 关键: 用 cudaProfilerStart/Stop 精确控制 ncu 只捕获目标迭代
    # ncu 需要加 --replay-mode application --target-processes all
    # 或者更简单地: 只跑 1 次 forward, warmup 已经在上面完成
    print("Starting ncu-targeted forward (warmup already done)...")
    print("NOTE: Use ncu with --launch-skip and --launch-count to target specific kernels")
    print(f"  Autotune caches should be warm. Expected launches per fwd:")
    print(f"    🔥1 kkt:       {(T//64) * 1 * H} (NT*B*H)")
    print(f"    🔥2 solve_tril: {(T//64) * 1 * H}")
    print(f"    🔥3 recompute:  {(T//64) * 1 * H}")
    print(f"    🔥4 fwd_h:      {(128 // 32) * 1 * H}..{(128 // 64) * 1 * H} (ceil(V/BV)*B*H)")
    print(f"    🔥5 fwd_o:      variable (ceil(V/BV)*NT*B*H)")

    # 用 cudaProfiler API 标记 ncu 捕获范围
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.iters):
        with torch.no_grad():
            forward_with_nvtx(q, k, v, beta)
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"ncu profile done. {args.iters} iterations.")
    print("查看: ncu-ui deltanet_ncu.ncu-rep")

elif args.mode == "torch":
    # torch.profiler + record_function
    from torch.profiler import profile, ProfilerActivity

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(args.iters):
            if args.backward:
                o, _ = forward_with_nvtx(q, k, v, beta)
                loss = o.sum()
                loss.backward()
                q.grad = None; k.grad = None; v.grad = None; beta.grad = None
            else:
                with torch.no_grad():
                    o, _ = forward_with_nvtx(q, k, v, beta)
            torch.cuda.synchronize()

    print("\nTop-25 CUDA kernels by total GPU time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    trace_path = "deltanet_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace: {trace_path}")
    print("可用 chrome://tracing 或 https://ui.perfetto.dev 打开")

    # 额外：按 sequence length scaling
    print("\n--- Forward latency scaling ---")
    import time
    for tT in [1024, 2048, 4096, 8192, 16384, 32768]:
        try:
            qt = torch.randn(1, tT, H, D, device=device, dtype=dtype)
            kt = F.normalize(torch.randn(1, tT, H, D, device=device, dtype=dtype), p=2, dim=-1)
            vt = torch.randn(1, tT, H, D, device=device, dtype=dtype)
            bt = torch.rand(1, tT, H, device=device, dtype=dtype).sigmoid()
            for _ in range(3):
                with torch.no_grad():
                    chunk_delta_rule(qt, kt, vt, bt, output_final_state=True)
            torch.cuda.synchronize()
            N = 20
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                with torch.no_grad():
                    chunk_delta_rule(qt, kt, vt, bt, output_final_state=True)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / N * 1000
            print(f"  T={tT:6d} (NT={tT//64:4d}) | fwd = {ms:.2f} ms")
            del qt, kt, vt, bt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  T={tT:6d} | FAILED: {e}")

print("\nDone!")
