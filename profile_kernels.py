"""
profile_kernels.py — 用 torch.profiler 实测 DeltaNet chunk_delta_rule 各 kernel 耗时
在 GPU 机器上运行: python profile_kernels.py

输出:
  1. 终端打印各 kernel 耗时排序 (top-20)
  2. 生成 chrome trace 文件 (deltanet_profile.json)，可用 chrome://tracing 或 Perfetto 查看
"""

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

# ============================================================
# 配置 — 模拟实际训练的维度
# ============================================================
# 1.3B DeltaNet: H=32, D=128
# 可在此调整序列长度来测试不同场景
B, T, H, D = 2, 8192, 32, 128

print(f"Profile config: B={B}, T={T}, H={H}, D={D}")
print(f"  BT=64 → NT={T//64} chunks per sequence")
print(f"  Total tokens = {B*T}")
print()

device = 'cuda'
dtype = torch.bfloat16

# ============================================================
# 准备数据
# ============================================================
torch.manual_seed(42)
q = torch.randn(B, T, H, D, device=device, dtype=dtype)
k = F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), p=2, dim=-1)
v = torch.randn(B, T, H, D, device=device, dtype=dtype)
beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()

# 需要 grad 来 profile backward
q.requires_grad_(True)
k.requires_grad_(True)
v.requires_grad_(True)
beta.requires_grad_(True)

# ============================================================
# Warmup — 让 Triton autotune 完成，避免编译时间影响 profile
# ============================================================
print("Warmup (Triton autotune)...")
from fla.ops.delta_rule import chunk_delta_rule

for _ in range(3):
    o, final_state = chunk_delta_rule(q, k, v, beta, output_final_state=True)
    loss = o.sum()
    loss.backward()
    q.grad = None
    k.grad = None
    v.grad = None
    beta.grad = None

torch.cuda.synchronize()
print("Warmup done.\n")

# ============================================================
# Profile 1: Forward only
# ============================================================
print("=" * 70)
print("Profile: Forward Only")
print("=" * 70)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    profile_memory=False,
) as prof_fwd:
    for _ in range(5):
        with torch.no_grad():
            o, final_state = chunk_delta_rule(q, k, v, beta, output_final_state=True)
        torch.cuda.synchronize()

print("\n[Forward] Top-20 CUDA kernels by total GPU time:")
print(prof_fwd.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20,
    top_level_events_only=False,
))

# ============================================================
# Profile 2: Forward + Backward
# ============================================================
print("=" * 70)
print("Profile: Forward + Backward")
print("=" * 70)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    profile_memory=False,
) as prof_fwdbwd:
    for _ in range(5):
        o, final_state = chunk_delta_rule(q, k, v, beta, output_final_state=True)
        loss = o.sum()
        loss.backward()
        q.grad = None
        k.grad = None
        v.grad = None
        beta.grad = None
        torch.cuda.synchronize()

print("\n[Fwd+Bwd] Top-30 CUDA kernels by total GPU time:")
print(prof_fwdbwd.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=30,
    top_level_events_only=False,
))

# ============================================================
# Profile 3: 用 record_function 标注每个子步骤
# ============================================================
print("=" * 70)
print("Profile: Annotated sub-steps (manual instrumentation)")
print("=" * 70)

from fla.ops.delta_rule.wy_fast import prepare_wy_repr_fwd, recompute_w_u_fwd
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.ops.utils.index import prepare_chunk_indices

def annotated_forward(q, k, v, beta, scale=None, initial_state=None, output_final_state=True):
    """手动拆解 chunk_delta_rule_fwd，加 record_function 标注"""
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Step 0: L2 norm (如果需要)
    # 这里假设 k 已经 L2 归一化

    # Step 1a: chunk_scaled_dot_kkt (🔥1)
    with torch.profiler.record_function("🔥1_chunk_scaled_dot_kkt"):
        A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)

    # Step 1b: solve_tril (🔥2)
    with torch.profiler.record_function("🔥2_solve_tril"):
        A = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)

    # Step 1c: recompute_w_u (🔥3)
    with torch.profiler.record_function("🔥3_recompute_w_u"):
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=None)

    # Step 2: chunk_fwd_h (🔥4)
    with torch.profiler.record_function("🔥4_chunk_fwd_h"):
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=None,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=None,
        )

    # Step 3: chunk_fwd_o (🔥5)
    with torch.profiler.record_function("🔥5_chunk_fwd_o"):
        o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None)

    return o, final_state

# Warmup annotated version
for _ in range(2):
    with torch.no_grad():
        annotated_forward(q, k, v, beta)
    torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    profile_memory=False,
) as prof_annotated:
    for _ in range(10):
        with torch.no_grad():
            annotated_forward(q, k, v, beta)
        torch.cuda.synchronize()

print("\n[Annotated Forward] Sub-step breakdown:")
print(prof_annotated.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=30,
    top_level_events_only=False,
))

# ============================================================
# 导出 chrome trace
# ============================================================

# 再跑一次用于导出 trace
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    profile_memory=False,
) as prof_trace:
    with torch.no_grad():
        annotated_forward(q, k, v, beta)
    torch.cuda.synchronize()

trace_path = "deltanet_profile.json"
prof_trace.export_chrome_trace(trace_path)
print(f"\nChrome trace exported to: {trace_path}")
print("可用 chrome://tracing 或 https://ui.perfetto.dev/ 打开查看时间线")

# ============================================================
# 额外：不同序列长度的 kernel 耗时对比
# ============================================================
print()
print("=" * 70)
print("Scaling test: Forward latency vs sequence length")
print("=" * 70)

import time

for test_T in [2048, 4096, 8192, 16384, 32768]:
    if test_T > 32768:
        continue
    try:
        q_t = torch.randn(1, test_T, H, D, device=device, dtype=dtype)
        k_t = F.normalize(torch.randn(1, test_T, H, D, device=device, dtype=dtype), p=2, dim=-1)
        v_t = torch.randn(1, test_T, H, D, device=device, dtype=dtype)
        beta_t = torch.rand(1, test_T, H, device=device, dtype=dtype).sigmoid()

        # warmup
        with torch.no_grad():
            for _ in range(3):
                chunk_delta_rule(q_t, k_t, v_t, beta_t, output_final_state=True)
        torch.cuda.synchronize()

        # measure
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N_ITERS = 20
        with torch.no_grad():
            for _ in range(N_ITERS):
                chunk_delta_rule(q_t, k_t, v_t, beta_t, output_final_state=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) / N_ITERS * 1000
        print(f"  T={test_T:6d} (NT={test_T//64:4d}) | fwd latency = {ms:.2f} ms")

        del q_t, k_t, v_t, beta_t
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  T={test_T:6d} | FAILED: {e}")

print("\nDone!")
