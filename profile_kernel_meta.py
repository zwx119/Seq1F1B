"""
profile_kernel_meta.py — DeltaNet 5 个 Triton kernel 的完整分析
不需要 ncu，不需要 root 权限，不需要 profiling permission

分析内容:
  1. 模型大小 / H 对 CTA 数的影响（为什么 🔥4 CTA 少）
  2. 🔥2 solve_tril 的计算 bound 类型（CC 串行 vs TC）
  3. 每个 kernel 的 FLOPs 和 MFU（硬件利用率）
  4. Triton 编译元数据（寄存器/shared memory/occupancy）
  5. do_bench 延迟测量
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

# ============================================================
# A100-SXM4-80GB 硬件规格
# ============================================================
SM_COUNT = 108                    # SM 数量
MAX_THREADS_PER_SM = 2048         # 每 SM 最大线程
MAX_BLOCKS_PER_SM = 32            # 每 SM 最大 CTA
MAX_REGS_PER_SM = 65536           # 每 SM 寄存器总数
MAX_SHARED_PER_SM = 164 * 1024    # 每 SM 可配置 shared memory (bytes)
# A100 峰值算力
A100_BF16_TFLOPS = 312.0          # BF16 Tensor Core
A100_FP32_TFLOPS = 19.5           # FP32 CUDA Core
A100_TF32_TFLOPS = 156.0          # TF32 Tensor Core
A100_HBM_BW_GB = 2039.0           # HBM 带宽 GB/s

print(f"{'='*80}")
print(f"DeltaNet Forward Kernel 完整分析")
print(f"{'='*80}")
print(f"Config: B={B}, T={T}, H={H}, D={D}, BT={BT}, NT={NT}")
print(f"GPU: {torch.cuda.get_device_name()}")

# ============================================================
# 问题 1: 模型大小分析 — H=32 对应什么规模？为什么 🔥4 CTA 少？
# ============================================================
print(f"\n{'='*80}")
print("问题 1: 模型大小与 🔥4 CTA 数分析")
print(f"{'='*80}")

print(f"""
当前 profiling 参数: H={H}, D={D}
  -> hidden_size = H * D = {H} * {D} = {H*D}

对应模型规模参考 (Megatron-LM exp.sh 配置):
  1.3B: hidden=2048, num_heads=16  -> H=16, D=128
  2.7B: hidden=2560, num_heads=32  -> H=32, D=80
  7B:   hidden=4096, num_heads=32  -> H=32, D=128  <-- 当前配置最接近
  13B:  hidden=5120, num_heads=40  -> H=40, D=128
  30B:  hidden=6144, num_heads=64  -> H=64, D=96

注意: DeltaNet GatedDeltaNetConfig 默认 num_heads=6, head_dim=256
      实际 H 和 D 取决于你的模型配置。当前 H={H},D={D} 是测试配置。

🔥4 CTA 数为什么少？
--------------------
关键: 🔥4 的 grid = (ceil(V/BV), N*H)，没有 NT 维度！
  NT 是在 CTA 内部串行循环的: for i_t in range(NT)

对比其他 kernel:
  🔥1,🔥2,🔥3: grid = (NT, B*H)        <- NT 在 grid 上，全并行
  🔥5:         grid = (V/BV, NT, B*H)   <- NT 也在 grid 上
  🔥4:         grid = (V/BV, B*H)       <- 没有 NT! 串行!
""")

from fla.utils import check_shared_mem
BV_candidates = [32, 64] if check_shared_mem('ada') else [32]

print(f"BV 候选值 (A100): {BV_candidates}")
print(f"  {'BV':>4} | {'ceil(D/BV)':>10} | {'CTA总数':>10} | {'CTA/SM':>8} | 状态")
print(f"  {'----':>4}-+-{'----------':>10}-+-{'----------':>10}-+-{'--------':>8}-+-{'----'}")
for BV in BV_candidates:
    nv = math.ceil(D / BV)
    total_ctas = nv * B * H
    ctas_per_sm = total_ctas / SM_COUNT
    status = "OK" if total_ctas >= SM_COUNT else "CTA不足，SM空闲!"
    print(f"  {BV:>4} | {nv:>10} | {total_ctas:>10} | {ctas_per_sm:>8.1f} | {status}")

print(f"""
对比不同模型规模 🔥4 CTA 数 (B=1):
  模型   |  H  |  D  | BV=32 CTA | BV=64 CTA | vs 108 SM
  -------+-----+-----+-----------+-----------+----------""")
models = [
    ("1.3B", 16, 128), ("2.7B", 32, 80), ("7B", 32, 128),
    ("13B", 40, 128), ("30B", 64, 96),
]
for mname, mh, md in models:
    c32 = math.ceil(md/32) * mh
    c64 = math.ceil(md/64) * mh
    s = "OK" if c64 >= SM_COUNT else "CTA<SM!"
    print(f"  {mname:>6} | {mh:>3} | {md:>3} | {c32:>9} | {c64:>9} | {s}")

print(f"""
结论:
  H={H}, D={D}, B={B} 时 🔥4 CTA 只有 {math.ceil(D/64)*B*H}~{math.ceil(D/32)*B*H} 个
  108 个 SM 大量空闲! 这就是 🔥4 慢的原因之一。
  🔥1/🔥2/🔥3 的 CTA={NT*B*H}={NT*B*H}，是 🔥4 的 {NT*B*H // max(math.ceil(D/64)*B*H, 1)} 倍
""")

# ============================================================
# 问题 2: 🔥2 solve_tril 的计算 bound — CC 还是 TC？
# ============================================================
print(f"{'='*80}")
print("问题 2: 🔥2 solve_tril 的计算类型 — CC (CUDA Core) 还是 TC (Tensor Core)？")
print(f"{'='*80}")

print(f"""
🔥2 merge_16x16_to_64x64_inverse_kernel 分两个阶段:

Phase 1: 4 个 16x16 对角块的前向替代法 (Forward Substitution)
  代码:
    for i in range(2, 16):
        b_a = -tl.load(...)                         # 标量 load
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)       # [16] 向量和 [16,16] 矩阵做 elementwise 乘 + reduce
        b_Ai = tl.where(mask, b_a, b_Ai)            # 条件写入

  这部分是纯 CC (CUDA Core):
    - tl.sum(a * B, axis) 不是 tl.dot, 不会调 Tensor Core
    - 是 elementwise 乘法 + reduce sum, 全在 CUDA Core 上跑
    - 4 个 16x16 块各 14 步串行 -> 共 56 步串行
    - 每步有数据依赖 (第 i 行依赖前 i-1 行), 无法并行
    - FLOPs: 4 x 14 x ~31 = 1,736 FLOPs (极少!)

Phase 2: 用 tl.dot 合并非对角块
  代码:
    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21), b_Ai_11)   # 16x16 dot
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32), b_Ai_22)   # 16x16 dot
    ... 共约 16 次 16x16 的 tl.dot

  这部分可以用 TC (Tensor Core):
    - tl.dot 会被编译为 mma.sync 指令
    - 但 16x16 是 TC 的最小 tile, 效率不高
    - FLOPs: ~16 x 2 x 16^3 = 131K FLOPs

总结:
  Phase 1 (CC 串行): ~1.7K FLOPs, 但决定了延迟下限 (56 步数据依赖)
  Phase 2 (TC 小 dot): ~131K FLOPs, 计算量虽大但 16x16 dot 很快

  瓶颈类型: Latency Bound (延迟受限)
    - 不是 compute bound (计算量太少)
    - 不是 memory bound (数据量也很少)
    - 是 Phase 1 的 56 步串行数据依赖决定了延迟
    - 每个 CTA 内部串行, 但不同 chunk 的 CTA 之间是全并行的
    
  DOT_PRECISION 参数控制 Phase 2 dot 精度:
    'ieee' = FP32 CUDA Core (精确), 'tf32' = TF32 Tensor Core
""")

# ============================================================
# 准备数据并触发编译
# ============================================================
print(f"{'='*80}")
print("准备数据并触发 JIT 编译...")
print(f"{'='*80}")

torch.manual_seed(42)
q = torch.randn(B, T, H, D, device=device, dtype=dtype)
k = F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), p=2, dim=-1)
v = torch.randn(B, T, H, D, device=device, dtype=dtype)
beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
scale = D ** -0.5

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o

print("第一次 forward (触发 autotune + JIT)...")
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
print("编译完成。\n")

# ============================================================
# do_bench 延迟测量
# ============================================================
print(f"{'='*80}")
print("Kernel 延迟 (triton.testing.do_bench)")
print(f"{'='*80}")

from triton.testing import do_bench

with torch.no_grad():
    t1 = do_bench(lambda: chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32))
    t2 = do_bench(lambda: solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype))
    t3 = do_bench(lambda: recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None))
    t4 = do_bench(lambda: chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None, initial_state=None, output_final_state=True, cu_seqlens=None))
    t5 = do_bench(lambda: chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None))

total = t1 + t2 + t3 + t4 + t5
times = [("🔥1 kkt", t1), ("🔥2 solve_tril", t2), ("🔥3 recompute_w_u", t3),
         ("🔥4 chunk_h", t4), ("🔥5 chunk_o", t5)]

print(f"\n  {'Kernel':<25} {'延迟':>8} {'占比':>6}")
print(f"  {'─'*25} {'─'*8} {'─'*6}")
for name, t in times:
    print(f"  {name:<25} {t:>7.3f}ms {t/total*100:>5.1f}%")
print(f"  {'─'*25} {'─'*8} {'─'*6}")
print(f"  {'总计':<25} {total:>7.3f}ms")

# ============================================================
# 问题 3+4: FLOPs 和 MFU 计算
# ============================================================
print(f"\n{'='*80}")
print("问题 3+4: 每个 Kernel 的 FLOPs 和 MFU")
print(f"{'='*80}")

K = D
V = D
N_chunks = B * NT * H  # 🔥1/🔥2/🔥3 的 CTA 数 (并行)

# 🔥1 FLOPs: 每chunk = 2*BT*K*BT (一个大 dot), 全部chunk并行
flops_1_total = 2 * BT * K * BT * N_chunks

# 🔥2 FLOPs: 极少
flops_2_cc = 4 * 14 * 31 * N_chunks           # Phase 1 CC
flops_2_tc = 16 * 2 * 16**3 * N_chunks        # Phase 2 TC
flops_2_total = flops_2_cc + flops_2_tc

# 🔥3 FLOPs: dot [BT,BT]@[BT,BV] * cdiv(V,BV)次 + dot [BT,BT]@[BT,BK] * cdiv(K,BK)次
#   注意: allow_tf32=False -> 用 FP32 CUDA Core!
BV3 = 64  # autotune typical
BK3 = 64
flops_3_total = (2 * BT * BT * V + 2 * BT * BT * K) * N_chunks

# 🔥4 FLOPs: 每步(每chunk) w@h + k^T@v, K 拆成 ceil(K/64) 块
#   w@h:   [BT,64]@[64,BV] = 2*BT*64*BV, ceil(K/64)次
#   k^T@v: [64,BT]@[BT,BV] = 2*64*BT*BV, ceil(K/64)次
n_k_blocks = math.ceil(K / 64)
BV4 = 64  # autotune typical
flops_4_per_step = n_k_blocks * (2 * BT * 64 * BV4 + 2 * 64 * BT * BV4)
nv4 = math.ceil(V / BV4)
flops_4_total = flops_4_per_step * NT * B * H  # 每个 BV slice 独立

# 🔥5 FLOPs: q@h + q@k^T + A@v
BK5 = 64
BV5 = 64
n_k5 = math.ceil(K / BK5)
nv5 = math.ceil(V / BV5)
flops_5_per_cta = n_k5 * (2*BT*BK5*BV5 + 2*BT*BK5*BT) + 2*BT*BT*BV5
flops_5_total = flops_5_per_cta * nv5 * NT * B * H

print(f"""
  MFU = 实际FLOPs / (延迟 x 峰值算力) x 100%
  A100 BF16 TC: {A100_BF16_TFLOPS} TFLOPS | FP32 CC: {A100_FP32_TFLOPS} TFLOPS | HBM: {A100_HBM_BW_GB} GB/s
""")

kernel_mfu = [
    ("🔥1 kkt",           flops_1_total, "BF16 TC", A100_BF16_TFLOPS, t1),
    ("🔥2 solve_tril",    flops_2_total, "FP32 CC", A100_FP32_TFLOPS, t2),
    ("🔥3 recompute_w_u", flops_3_total, "FP32 CC", A100_FP32_TFLOPS, t3),
    ("🔥4 chunk_h",       flops_4_total, "BF16 TC", A100_BF16_TFLOPS, t4),
    ("🔥5 chunk_o",       flops_5_total, "BF16 TC", A100_BF16_TFLOPS, t5),
]

print(f"  {'Kernel':<22} {'GFLOPs':>8} {'延迟ms':>7} {'计算类型':<18} {'达到TFLOPS':>10} {'MFU':>6}")
print(f"  {'─'*22} {'─'*8} {'─'*7} {'─'*18} {'─'*10} {'─'*6}")
for name, flops, comp_type, peak_tflops, lat_ms in kernel_mfu:
    gflops = flops / 1e9
    achieved = gflops / lat_ms  # GFLOPs/ms = TFLOPS
    mfu = achieved / peak_tflops * 100
    print(f"  {name:<22} {gflops:>8.3f} {lat_ms:>7.3f} {comp_type:<18} {achieved:>10.2f} {mfu:>5.1f}%")

print(f"""
MFU 解读:
  🔥1: TC dot, 但每个 dot 只有 64x128x64, tile 较小 -> MFU 不高
  🔥2: FLOPs 极少 ({flops_2_total/1e6:.1f}M), MFU 极低是正常的
       -> 这不是 compute bound, 是 latency bound (56 步串行数据依赖)
  🔥3: allow_tf32=False 强制用 FP32 CC (19.5T 峰值而非 312T)
       如果改为 tf32: 理论峰值从 19.5T -> 156T, 延迟可降 ~8x
  🔥4: TC dot 但 CTA 不足 + NT 步串行 -> SM 大量空闲
  🔥5: 全并行 TC, 应该是 MFU 最高的 kernel
""")

# ============================================================
# 问题 5: Triton 编译元数据 (寄存器/shared memory/occupancy)
# ============================================================
print(f"{'='*80}")
print("问题 5: Triton 编译元数据 — 寄存器、Shared Memory、Occupancy")
print(f"{'='*80}")

print("""
原理说明:
  Triton JIT 编译后, kernel 对象结构是:
    Heuristics (最外层, 运行时决定参数如 IS_VARLEN)
      -> fn: Autotuner (搜索最优 num_warps/num_stages/BV 等)
           -> fn: JITFunction (真正的编译后 kernel)
                -> cache: dict
                     key = 编译参数组合 (constexpr 的值)
                     value = CompiledKernel 对象
                         .metadata -> 寄存器数, shared memory, warps 等

  要从 Heuristics 一路沿 .fn 找到 JITFunction 的 .cache
""")

import triton
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

    if kernel_fn is None:
        print(f"  找不到 kernel 对象 (None)")
        continue

    # 逐层解包: Heuristics -> Autotuner -> JITFunction
    fn = kernel_fn
    chain = [type(fn).__name__]
    while hasattr(fn, 'fn'):
        fn = fn.fn
        chain.append(type(fn).__name__)

    print(f"  解包链: {' -> '.join(chain)}")
    print(f"  最内层类型: {type(fn).__name__}")

    # 读取 autotune 最优配置
    autotuner = kernel_fn
    while hasattr(autotuner, 'fn'):
        if hasattr(autotuner, 'best_config'):
            print(f"  Autotune 最优配置: {autotuner.best_config}")
            break
        autotuner = autotuner.fn

    # 找编译缓存 cache
    cache = getattr(fn, 'cache', None)
    if not cache:
        fn2 = kernel_fn
        while hasattr(fn2, 'fn'):
            if hasattr(fn2, 'cache') and fn2.cache:
                cache = fn2.cache
                print(f"  (cache 在 {type(fn2).__name__} 层找到)")
                break
            fn2 = fn2.fn

    if cache:
        print(f"  编译缓存条目数: {len(cache)}")
        for i, (key, compiled) in enumerate(cache.items()):
            if i > 0:
                break  # 只看第一个

            print(f"  缓存 key: {key}")
            print(f"  CompiledKernel 类型: {type(compiled).__name__}")
            attrs = [a for a in dir(compiled) if not a.startswith('_')]
            print(f"  CompiledKernel 属性: {attrs}")

            # 方式 1: 直接属性
            for attr in ['n_regs', 'num_gprs', 'n_spills', 'shared', 'n_shared_bytes',
                         'num_warps', 'num_stages', 'num_ctas', 'cluster_dims']:
                val = getattr(compiled, attr, None)
                if val is not None:
                    print(f"  {attr} = {val}")

            # 方式 2: metadata 对象
            meta = getattr(compiled, 'metadata', None)
            if meta:
                print(f"  metadata 类型: {type(meta).__name__}")
                for a in sorted(dir(meta)):
                    if not a.startswith('_'):
                        try:
                            v = getattr(meta, a)
                            if not callable(v):
                                print(f"    {a} = {v}")
                        except Exception:
                            pass

            # 方式 3: asm dict
            asm = getattr(compiled, 'asm', None)
            if asm and isinstance(asm, dict):
                print(f"  ASM keys: {list(asm.keys())}")

            # === Occupancy 计算 ===
            n_regs = None
            shared_bytes = None
            num_warps_val = None

            for src in [compiled, meta]:
                if src is None:
                    continue
                if n_regs is None:
                    n_regs = getattr(src, 'num_gprs', getattr(src, 'n_regs', None))
                if shared_bytes is None:
                    shared_bytes = getattr(src, 'shared', getattr(src, 'n_shared_bytes', None))
                if num_warps_val is None:
                    num_warps_val = getattr(src, 'num_warps', None)

            if n_regs and shared_bytes is not None and num_warps_val:
                threads = num_warps_val * 32
                regs_total = n_regs * threads
                blk_by_reg = MAX_REGS_PER_SM // regs_total if regs_total > 0 else MAX_BLOCKS_PER_SM
                blk_by_smem = MAX_SHARED_PER_SM // shared_bytes if shared_bytes > 0 else MAX_BLOCKS_PER_SM
                blk_by_thrd = MAX_THREADS_PER_SM // threads
                blk_per_sm = min(blk_by_reg, blk_by_smem, blk_by_thrd, MAX_BLOCKS_PER_SM)
                occ = (blk_per_sm * threads) / MAX_THREADS_PER_SM * 100

                if blk_per_sm == blk_by_reg:
                    limiter = "寄存器"
                elif blk_per_sm == blk_by_smem:
                    limiter = "shared memory"
                elif blk_per_sm == blk_by_thrd:
                    limiter = "线程数"
                else:
                    limiter = "CTA上限"

                print(f"""
  === Occupancy 分析 ===
  寄存器/线程:      {n_regs}
  Shared Memory:    {shared_bytes} bytes ({shared_bytes/1024:.1f} KB)
  Warps/Block:      {num_warps_val} ({threads} 线程/block)

  寄存器限制:       {blk_by_reg} blocks/SM
  Shared Mem 限制:  {blk_by_smem} blocks/SM
  线程数限制:       {blk_by_thrd} blocks/SM

  -> 实际 blocks/SM:    {blk_per_sm}  (受限于: {limiter})
  -> 理论 Occupancy:    {occ:.1f}%""")
            else:
                print(f"  无法计算 occupancy: regs={n_regs}, shared={shared_bytes}, warps={num_warps_val}")
    else:
        print(f"  没有找到编译缓存 (.cache)")
        print(f"  内层类型: {type(fn).__name__}")
        print(f"  内层属性: {[a for a in dir(fn) if not a.startswith('_')]}")

# ============================================================
# Grid Size 总结
# ============================================================
print(f"\n{'='*80}")
print("Grid Size 总结")
print(f"{'='*80}")

nv32 = math.ceil(D / 32)
nv64 = math.ceil(D / 64)

print(f"""
  Kernel                 Grid                      CTA总数  CTA/SM  并行度
  ──────────────────── ──────────────────────── ──────── ─────── ──────
  🔥1 kkt              ({NT}, {B*H})                  {NT*B*H:>6}  {NT*B*H/SM_COUNT:>6.1f}  全并行
  🔥2 solve_tril        ({NT}, {B*H})                  {NT*B*H:>6}  {NT*B*H/SM_COUNT:>6.1f}  全并行
  🔥3 recompute_w_u     ({NT}, {B*H})                  {NT*B*H:>6}  {NT*B*H/SM_COUNT:>6.1f}  全并行
  🔥4 chunk_h BV=32    ({nv32}, {B*H})                  {nv32*B*H:>6}  {nv32*B*H/SM_COUNT:>6.1f}  NT串行!
  🔥4 chunk_h BV=64    ({nv64}, {B*H})                  {nv64*B*H:>6}  {nv64*B*H/SM_COUNT:>6.1f}  NT串行!
  🔥5 chunk_o BV=64 ({nv64},{NT},{B*H})          {nv64*NT*B*H:>6}  {nv64*NT*B*H/SM_COUNT:>6.1f}  全并行
""")

print(f"{'='*80}")
print("分析完成!")
print(f"{'='*80}")
