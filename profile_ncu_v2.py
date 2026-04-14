"""
profile_ncu_v2.py — 两阶段 ncu profiling，绕过 Triton JIT 卡死问题

问题: ncu 拦截 cuModuleLoadData，导致 Triton JIT 编译卡死。
方案: 
  阶段1: 裸跑一次，让 Triton 把所有 kernel 编译+缓存好
  阶段2: ncu 跑时，Triton 从缓存加载，不触发 JIT

用法:
  # 阶段 1: 预热编译缓存（不加 ncu，几秒搞定）
  python profile_ncu_v2.py --warmup-only --T 8192 --B 1

  # 阶段 2a: 一次抓全部 5 个 kernel（推荐）
  ncu --replay-mode kernel --set basic \
    --launch-count 100 -o ncu_all \
    python profile_ncu_v2.py --target all --T 8192 --B 1

  # 阶段 2b: 也可以只抓某个 kernel
  ncu --replay-mode kernel --set basic \
    --kernel-name-base "chunk_gated_delta_rule_fwd_kernel_h" \
    --launch-skip 0 --launch-count 1 \
    -o ncu_fire4 python profile_ncu_v2.py --target 4 --T 8192 --B 1
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--B", type=int, default=1)
parser.add_argument("--T", type=int, default=8192)
parser.add_argument("--H", type=int, default=32)
parser.add_argument("--D", type=int, default=128)
parser.add_argument("--target", type=str, default="4",
                    help="Which kernel to profile: 1,2,3,4,5,all")
parser.add_argument("--warmup-only", action="store_true",
                    help="Only do warmup (compile+cache), then exit. Run this first without ncu.")
args = parser.parse_args()

B, T, H, D = args.B, args.T, args.H, args.D
device = "cuda"
dtype = torch.bfloat16

print(f"Config: B={B}, T={T}, H={H}, D={D}, BT=64, NT={T//64}")

# ============================================================
# 导入
# ============================================================
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd
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
scale = D ** -0.5

# ============================================================
# 跑一次完整 forward，触发所有 kernel 的 JIT 编译 + autotune
# ============================================================
print("Running full forward to trigger JIT compile + autotune cache...")
with torch.no_grad():
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
    A_inv = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=None, initial_state=None,
        output_final_state=True, cu_seqlens=None,
    )
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None)
torch.cuda.synchronize()
print("All kernels compiled and cached.\n")

if args.warmup_only:
    print("--warmup-only: exiting. Triton cache is now warm.")
    print("Next step: run again WITH ncu (see docstring for commands).")
    sys.exit(0)

# ============================================================
# 阶段 2: 只跑目标 kernel — ncu 只会抓这一个
# 此时 Triton 从 ~/.triton/cache 加载，不触发 JIT 编译
# ============================================================

# 预计算中间结果（这些不是目标 kernel，ncu 通过 --kernel-name 过滤掉）
print("Pre-computing intermediate results...")
with torch.no_grad():
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
    A_inv = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=None, initial_state=None,
        output_final_state=True, cu_seqlens=None,
    )
torch.cuda.synchronize()
print("Pre-computation done.\n")

# 跑目标 kernel
target = args.target
print(f"Running target 🔥{target} for ncu capture...")

with torch.no_grad():
    if target == "all":
        # 按依赖顺序跑全部 5 个 kernel，ncu 一次抓齐
        print("  🔥1 chunk_scaled_dot_kkt_fwd ...")
        A2 = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
        torch.cuda.synchronize()
        print("  🔥2 solve_tril ...")
        A_inv2 = solve_tril(A=A2, cu_seqlens=None, output_dtype=k.dtype)
        torch.cuda.synchronize()
        print("  🔥3 recompute_w_u_fwd ...")
        w2, u2 = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv2, cu_seqlens=None)
        torch.cuda.synchronize()
        print("  🔥4 chunk_gated_delta_rule_fwd_h ...")
        h2, v_new2, _ = chunk_gated_delta_rule_fwd_h(
            k=k, w=w2, u=u2, g=None, initial_state=None,
            output_final_state=True, cu_seqlens=None,
        )
        torch.cuda.synchronize()
        print("  🔥5 chunk_fwd_o ...")
        o2 = chunk_fwd_o(q=q, k=k, v=v_new2, h=h2, g=None, scale=scale, cu_seqlens=None)
        torch.cuda.synchronize()
    elif target == "1":
        chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
    elif target == "2":
        solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    elif target == "3":
        recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None)
    elif target == "4":
        chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=None, initial_state=None,
            output_final_state=True, cu_seqlens=None,
        )
    elif target == "5":
        chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None)
    else:
        print(f"Unknown target: {target}")
        sys.exit(1)

torch.cuda.synchronize()
print("Done!")
