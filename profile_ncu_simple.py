"""
profile_ncu_simple.py — 专门给 ncu 的极简脚本

设计原则:
  1. 先 warmup 让 autotune 完成 + kernel 编译好
  2. 用 torch.cuda.cudart().cudaProfilerStart/Stop 精确标记
  3. 只在标记区间内跑 1 次目标 kernel

用法:
  # 🔥4 (串行递推, 最慢的单次 kernel)
  ncu --set basic --kernel-name "chunk_gated_delta" --launch-count 1 \
    -o ncu_fire4 python profile_ncu_simple.py --target 4

  # 🔥2 (延迟瓶颈)
  ncu --set basic --kernel-name "merge_16x16_to_64x64" --launch-count 1 \
    -o ncu_fire2 python profile_ncu_simple.py --target 2

  # 🔥1
  ncu --set basic --kernel-name "chunk_scaled_dot_kkt" --launch-count 1 \
    -o ncu_fire1 python profile_ncu_simple.py --target 1

  # 🔥3
  ncu --set basic --kernel-name "recompute_w_u" --launch-count 1 \
    -o ncu_fire3 python profile_ncu_simple.py --target 3

  # 🔥5
  ncu --set basic --kernel-name "chunk_fwd_kernel_o" --launch-count 1 \
    -o ncu_fire5 python profile_ncu_simple.py --target 5

  # 全部 5 个 kernel 各 1 次 (不过滤 kernel name)
  ncu --set basic --launch-count 5 \
    -o ncu_all python profile_ncu_simple.py --target all
"""

import argparse
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--B", type=int, default=1)
parser.add_argument("--T", type=int, default=8192)
parser.add_argument("--H", type=int, default=32)
parser.add_argument("--D", type=int, default=128)
parser.add_argument("--target", type=str, default="4",
                    help="Which kernel to profile: 1,2,3,4,5,all")
args = parser.parse_args()

B, T, H, D = args.B, args.T, args.H, args.D
device = "cuda"
dtype = torch.bfloat16

print(f"Config: B={B}, T={T}, H={H}, D={D}, BT=64, NT={T//64}")
print(f"Target kernel: 🔥{args.target}")

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

# ============================================================
# Warmup — 让 Triton autotune 完成，所有 kernel 编译好
# ============================================================
print("Warmup (autotune + JIT compile)...")
with torch.no_grad():
    for _ in range(3):
        scale = D ** -0.5
        A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
        A_inv = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None)
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=None, initial_state=None,
            output_final_state=True, cu_seqlens=None,
        )
        o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None)
torch.cuda.synchronize()
print("Warmup done. Autotune caches warm.\n")

# ============================================================
# 预计算中间结果（给 target=4,5 用）
# ============================================================
with torch.no_grad():
    scale = D ** -0.5
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
    A_inv = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=None, initial_state=None,
        output_final_state=True, cu_seqlens=None,
    )
torch.cuda.synchronize()

# ============================================================
# 目标 kernel — ncu 只会捕获这里的 launch
# ============================================================
print("Running target kernel for ncu capture...")

# NOTE: ncu 不需要 cudaProfilerStart/Stop — 我们靠 --launch-skip 0 --launch-count 1
#       因为 warmup 和预计算的 kernel 名字和 target 不同
#       但如果 target=all，则 warmup 的 kernel 也会被捕获
#       所以 target=all 时需要更大的 --launch-skip

with torch.no_grad():
    target = args.target

    if target in ("1", "all"):
        print("  🔥1 chunk_scaled_dot_kkt_fwd ...")
        A_ncu = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
        torch.cuda.synchronize()

    if target in ("2", "all"):
        print("  🔥2 solve_tril ...")
        A_inv_ncu = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        torch.cuda.synchronize()

    if target in ("3", "all"):
        print("  🔥3 recompute_w_u ...")
        w_ncu, u_ncu = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None)
        torch.cuda.synchronize()

    if target in ("4", "all"):
        print("  🔥4 chunk_gated_delta_rule_fwd_h ...")
        h_ncu, v_new_ncu, _ = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=None, initial_state=None,
            output_final_state=True, cu_seqlens=None,
        )
        torch.cuda.synchronize()

    if target in ("5", "all"):
        print("  🔥5 chunk_fwd_o ...")
        o_ncu = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None)
        torch.cuda.synchronize()

print("Done!")
