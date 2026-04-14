"""
profile_kernel_meta.py — 从 Triton 编译器直接获取 kernel 元数据
不需要 ncu，不需要 root 权限，不需要 profiling permission

获取: register count, shared memory, grid size, block size, 理论 occupancy
"""
import torch
import torch.nn.functional as F
import argparse
import math

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
print(f"{'='*80}")

# ============================================================
# 导入 kernel 函数
# ============================================================
# 准备数据（需要一次 forward 触发 JIT + autotune）
torch.manual_seed(42)
q = torch.randn(B, T, H, D, device=device, dtype=dtype)
k = F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), p=2, dim=-1)
v = torch.randn(B, T, H, D, device=device, dtype=dtype)
beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
scale = D ** -0.5

# 触发编译
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o

print("Running forward to trigger JIT compilation...")
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
print("Compilation done.\n")

# ============================================================
# A100 specs
# ============================================================
SM_COUNT = 108
MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
MAX_REGS_PER_SM = 65536
MAX_SHARED_PER_SM = 164 * 1024  # 164KB configurable

# ============================================================
# 方法1: 用 torch.profiler 获取 grid/block size
# ============================================================
print("Capturing kernel launches with torch.profiler...")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        A2 = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
        A_inv2 = solve_tril(A=A2, cu_seqlens=None, output_dtype=k.dtype)
        w2, u2 = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv2, cu_seqlens=None)
        h2, v_new2, _ = chunk_gated_delta_rule_fwd_h(
            k=k, w=w2, u=u2, g=None, initial_state=None,
            output_final_state=True, cu_seqlens=None,
        )
        o2 = chunk_fwd_o(q=q, k=k, v=v_new2, h=h2, g=None, scale=scale, cu_seqlens=None)
    torch.cuda.synchronize()

# 解析 kernel launch 信息
kernel_names_map = {
    "chunk_scaled_dot_kkt_fwd_kernel": "🔥1 kkt",
    "merge_16x16_to_64x64_inverse_kernel": "🔥2 solve_tril",
    "recompute_w_u_fwd_kernel": "🔥3 recompute_w_u",
    "chunk_gated_delta_rule_fwd_kernel_h": "🔥4 chunk_h",
    "chunk_fwd_kernel_o": "🔥5 chunk_o",
}

print(f"\n{'='*80}")
print(f"{'Kernel':<25} {'CUDA Time':>12}")
print(f"{'='*80}")

for evt in prof.events():
    if evt.device_type == torch.autograd.DeviceType.CUDA:
        for kname, label in kernel_names_map.items():
            if kname in evt.name:
                cuda_time = evt.cuda_time / 1000  # us -> ms
                print(f"{label:<25} {cuda_time:>10.3f} ms")
                break

# ============================================================
# 方法2: 直接从 Triton kernel 对象获取编译元数据
# ============================================================
print(f"\n{'='*80}")
print("Triton Kernel Compilation Metadata")
print(f"{'='*80}")

# 尝试获取每个 kernel 的 cache 信息
import triton

# Import kernel functions by correct names from modules
import fla.ops.common.chunk_scaled_dot_kkt as _mod_kkt
import fla.ops.utils.solve_tril as _mod_solve
import fla.ops.delta_rule.wy_fast as _mod_wy
import fla.ops.common.chunk_delta_h as _mod_h
import fla.ops.common.chunk_o as _mod_o

kernels_info = [
    ("🔥1 kkt", getattr(_mod_kkt, 'chunk_scaled_dot_kkt_fwd_kernel', None)),
    ("🔥2 solve_tril", getattr(_mod_solve, 'merge_16x16_to_64x64_inverse_kernel', None)),
    ("🔥3 recompute_w_u", getattr(_mod_wy, 'recompute_w_u_fwd_kernel', None)),
    ("🔥4 chunk_h", getattr(_mod_h, 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64', None)),
    ("🔥5 chunk_o", getattr(_mod_o, 'chunk_fwd_kernel_o', None)),
]

for name, kernel_fn in kernels_info:
    print(f"\n--- {name} ---")
    
    # Unwrap Heuristics/Autotuner to get the actual JitFunction
    actual_fn = kernel_fn
    while actual_fn is not None and hasattr(actual_fn, 'fn'):
        actual_fn = actual_fn.fn
    
    if actual_fn is None:
        print(f"  Kernel not found (None)")
        continue
    
    print(f"  Unwrapped type: {type(actual_fn).__name__}")
    print(f"  Attributes: {[a for a in dir(actual_fn) if not a.startswith('_') and a not in ('run',)]}")
    
    # Try .cache (Triton JitFunction stores compiled kernels here)
    cache = getattr(actual_fn, 'cache', None)
    if cache:
        print(f"  Cache entries: {len(cache)}")
        for key, compiled in cache.items():
            print(f"\n  Cache key: {key}")
            # Print all attributes of compiled object to find metadata
            comp_attrs = [a for a in dir(compiled) if not a.startswith('_')]
            print(f"  Compiled type: {type(compiled).__name__}")
            print(f"  Compiled attrs: {comp_attrs}")
            
            # Try various ways to get metadata
            meta = getattr(compiled, 'metadata', None)
            if meta:
                print(f"  Metadata type: {type(meta).__name__}")
                meta_attrs = {a: getattr(meta, a, '?') for a in dir(meta) if not a.startswith('_')}
                for k, v in sorted(meta_attrs.items()):
                    if not callable(v):
                        print(f"    {k} = {v}")
            
            # Try .asm dict
            asm = getattr(compiled, 'asm', None)
            if asm and isinstance(asm, dict):
                print(f"  ASM keys: {list(asm.keys())}")
                if 'metadata' in asm:
                    print(f"  ASM metadata: {asm['metadata']}")
            
            # Try n_regs, n_spills, shared directly
            for attr in ['n_regs', 'num_gprs', 'n_spills', 'shared', 'n_shared_bytes', 'num_warps', 'num_stages']:
                val = getattr(compiled, attr, None)
                if val is not None:
                    print(f"  {attr} = {val}")
            
            # Only process first cache entry to keep output manageable
            break
    else:
        print(f"  No cache found on unwrapped fn")
        # Maybe the cache is elsewhere
        for attr_name in ['cache', 'kernel_cache', 'compiled_cache']:
            c = getattr(actual_fn, attr_name, None)
            if c:
                print(f"  Found {attr_name}: {type(c)}")
    
    # Also check if there's a .best_config from autotuner
    if hasattr(kernel_fn, 'best_config'):
        print(f"  Best config: {kernel_fn.best_config}")
    elif hasattr(kernel_fn, 'configs'):
        print(f"  Configs: {kernel_fn.configs}")

# ============================================================
# 方法3: 计算理论 Grid Size
# ============================================================
print(f"\n{'='*80}")
print("Theoretical Grid Sizes")
print(f"{'='*80}")

# BV for 🔥4 and 🔥5
from fla.utils import check_shared_mem
BV_candidates = [32, 64] if check_shared_mem() else [32]
print(f"BV candidates on this GPU: {BV_candidates}")

grids = {
    "🔥1 kkt":           f"({NT}, {B*H}) = {NT * B * H} CTAs",
    "🔥2 solve_tril":    f"({NT}, {B*H}) = {NT * B * H} CTAs",
    "🔥3 recompute_w_u": f"({NT}, {B*H}) = {NT * B * H} CTAs",
}
print(f"\n🔥1 kkt:           Grid = {grids['🔥1 kkt']}")
print(f"🔥2 solve_tril:    Grid = {grids['🔥2 solve_tril']}")
print(f"🔥3 recompute_w_u: Grid = {grids['🔥3 recompute_w_u']}")

for BV in BV_candidates:
    nv = math.ceil(D / BV)
    grid4 = f"({nv}, {B * H}) = {nv * B * H} CTAs"  # 🔥4: NT serial inside CTA
    grid5 = f"({nv}, {NT}, {B * H}) = {nv * NT * B * H} CTAs"  # 🔥5: fully parallel
    print(f"\n  With BV={BV}:")
    print(f"  🔥4 chunk_h:     Grid = {grid4}")
    print(f"  🔥5 chunk_o:     Grid = {grid5}")
    print(f"  🔥4 CTAs/SM:     {nv * B * H / SM_COUNT:.1f}")
    print(f"  🔥5 CTAs/SM:     {nv * NT * B * H / SM_COUNT:.1f}")

# ============================================================
# 方法4: 使用 do_bench 测量每个 kernel 的延迟
# ============================================================
print(f"\n{'='*80}")
print("Kernel Latency (triton.testing.do_bench)")
print(f"{'='*80}")

from triton.testing import do_bench

with torch.no_grad():
    # 🔥1
    t1 = do_bench(lambda: chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32))
    print(f"🔥1 kkt:           {t1:.3f} ms")
    
    # 🔥2
    t2 = do_bench(lambda: solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype))
    print(f"🔥2 solve_tril:    {t2:.3f} ms")
    
    # 🔥3
    t3 = do_bench(lambda: recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None))
    print(f"🔥3 recompute_w_u: {t3:.3f} ms")
    
    # 🔥4
    t4 = do_bench(lambda: chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None, initial_state=None, output_final_state=True, cu_seqlens=None))
    print(f"🔥4 chunk_h:       {t4:.3f} ms")
    
    # 🔥5
    t5 = do_bench(lambda: chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None))
    print(f"🔥5 chunk_o:       {t5:.3f} ms")
    
    total = t1 + t2 + t3 + t4 + t5
    print(f"\nTotal:             {total:.3f} ms")
    print(f"\nBreakdown:")
    for name, t in [("🔥1 kkt", t1), ("🔥2 solve_tril", t2), ("🔥3 recompute_w_u", t3), ("🔥4 chunk_h", t4), ("🔥5 chunk_o", t5)]:
        print(f"  {name}: {t/total*100:.1f}%")

print(f"\n{'='*80}")
print("Done!")
