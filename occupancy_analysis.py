"""
occupancy_analysis.py — 用已知的 shared memory 数据 + cuobjdump 提取物理寄存器数
精确计算 A100 上每个 kernel 的 occupancy

如果 cuobjdump 不可用，则用 Triton 的 triton.compiler API 获取物理寄存器数
"""
import torch
import torch.nn.functional as F
import triton
import os, glob, json, subprocess, re, sys

B, T, H, D = 1, 8192, 32, 128
BT = 64
device = "cuda"
dtype = torch.bfloat16

print(f"Triton version: {triton.__version__}")
print(f"GPU: {torch.cuda.get_device_name()}")
print()

# === 触发编译 ===
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

with torch.no_grad():
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
    A_inv = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A_inv, cu_seqlens=None)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None, initial_state=None, output_final_state=True, cu_seqlens=None)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=None)
torch.cuda.synchronize()
print("编译完成\n")

# ================================================================
# 搜索 JSON 缓存，找到获胜配置的目录
# ================================================================
KERNEL_CONFIGS = {
    'chunk_scaled_dot_kkt_fwd_kernel': {
        'label': '🔥1 kkt',
        'win': {'num_warps': 8, 'num_stages': 3},  # BK=128
    },
    'merge_16x16_to_64x64_inverse_kernel': {
        'label': '🔥2 solve_tril',
        'win': None,  # 我们不确定获胜配置，收集所有
    },
    'recompute_w_u_fwd_kernel': {
        'label': '🔥3 recompute_w_u',
        'win': {'num_warps': 2, 'num_stages': 3},
    },
    'chunk_gated_delta_rule_fwd_kernel_h_blockdim64': {
        'label': '🔥4 chunk_h',
        'win': {'num_warps': 4, 'num_stages': 4},  # BV=64
    },
    'chunk_fwd_kernel_o': {
        'label': '🔥5 chunk_o',
        'win': {'num_warps': 8, 'num_stages': 3},  # BK=128, BV=128
    },
}

cache_dir = os.environ.get('TRITON_CACHE_DIR', os.path.expanduser('~/.triton/cache'))
json_files = glob.glob(os.path.join(cache_dir, '**', '*.json'), recursive=True)
print(f"总 JSON 文件数: {len(json_files)}")

# 按 kernel 名 + 获胜配置筛选
kernel_dirs = {}  # kernel_name -> [(dir_path, json_data), ...]
for jf in json_files:
    try:
        with open(jf) as f:
            data = json.load(f)
        name = data.get('name', '')
        if name in KERNEL_CONFIGS:
            cfg = KERNEL_CONFIGS[name]
            win = cfg['win']
            if win is None or all(data.get(wk) == wv for wk, wv in win.items()):
                parent_dir = os.path.dirname(jf)
                if name not in kernel_dirs:
                    kernel_dirs[name] = []
                kernel_dirs[name].append((parent_dir, data))
    except:
        pass

# ================================================================
# 用 cuobjdump 从 .cubin 提取物理寄存器数
# ================================================================
print("\n" + "=" * 70)
print("用 cuobjdump 提取物理寄存器数和 shared memory")
print("=" * 70)

# 检查 cuobjdump 是否可用
cuobjdump_available = False
try:
    result = subprocess.run(['cuobjdump', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        cuobjdump_available = True
        print(f"cuobjdump 可用: {result.stdout.strip().split(chr(10))[0]}")
except FileNotFoundError:
    pass

if not cuobjdump_available:
    # 尝试在常见路径找
    for cuda_path in ['/usr/local/cuda/bin/cuobjdump', '/usr/bin/cuobjdump']:
        if os.path.exists(cuda_path):
            cuobjdump_available = True
            cuobjdump_cmd = cuda_path
            print(f"cuobjdump 找到: {cuda_path}")
            break
    else:
        cuobjdump_cmd = 'cuobjdump'
        print("cuobjdump 未找到，尝试其他方法...")
else:
    cuobjdump_cmd = 'cuobjdump'

results = {}

for kname, cfg in KERNEL_CONFIGS.items():
    label = cfg['label']
    entries = kernel_dirs.get(kname, [])
    if not entries:
        print(f"\n{label}: 未在缓存中找到")
        continue

    print(f"\n{label} ({kname}):")
    print(f"  找到 {len(entries)} 个匹配的编译版本")

    best_result = None
    for parent_dir, data in entries:
        shared = data.get('shared', '?')
        num_warps = data.get('num_warps', '?')
        num_stages = data.get('num_stages', '?')

        # 找 cubin 文件
        cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
        if not cubin_files:
            continue

        for cf in cubin_files:
            if cuobjdump_available:
                try:
                    result = subprocess.run(
                        [cuobjdump_cmd, '--resource-usage', cf],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        output = result.stdout
                        # 解析 REG, SMEM, STACK, CMEM
                        regs = None
                        smem = None
                        stack = None
                        spill_loads = None
                        spill_stores = None
                        for line in output.split('\n'):
                            line_s = line.strip()
                            # 典型输出: "REG:128 STACK:0 SHARED:65536 LOCAL:0"
                            # 或分行: "  Registers: 128"
                            reg_match = re.search(r'REG[:\s]+(\d+)', line_s, re.IGNORECASE)
                            if reg_match:
                                regs = int(reg_match.group(1))
                            smem_match = re.search(r'SHARED[:\s]+(\d+)|SMEM[:\s]+(\d+)', line_s, re.IGNORECASE)
                            if smem_match:
                                smem = int(smem_match.group(1) or smem_match.group(2))
                            stack_match = re.search(r'STACK[:\s]+(\d+)', line_s, re.IGNORECASE)
                            if stack_match:
                                stack = int(stack_match.group(1))
                            spill_load_match = re.search(r'spill.*load[:\s]+(\d+)', line_s, re.IGNORECASE)
                            if spill_load_match:
                                spill_loads = int(spill_load_match.group(1))
                            spill_store_match = re.search(r'spill.*store[:\s]+(\d+)', line_s, re.IGNORECASE)
                            if spill_store_match:
                                spill_stores = int(spill_store_match.group(1))

                        if regs is not None:
                            print(f"  ✅ warps={num_warps}, stages={num_stages}: "
                                  f"REG={regs}, SHARED={shared}, "
                                  f"STACK={stack}, SPILL_LD={spill_loads}, SPILL_ST={spill_stores}")
                            if best_result is None:
                                best_result = {
                                    'regs': regs, 'shared': shared,
                                    'num_warps': num_warps, 'num_stages': num_stages,
                                    'stack': stack, 'spill_loads': spill_loads, 'spill_stores': spill_stores
                                }
                            # 打印完整 cuobjdump 输出（仅第一个）
                            if best_result and best_result['regs'] == regs:
                                for line in output.split('\n'):
                                    if line.strip():
                                        print(f"    | {line.strip()}")
                        else:
                            # 可能格式不同，打印原始输出看看
                            print(f"  ⚠️ warps={num_warps}, stages={num_stages}: "
                                  f"shared={shared}, cuobjdump 输出格式未解析:")
                            for line in output.split('\n')[:15]:
                                if line.strip():
                                    print(f"    | {line.strip()}")
                except Exception as e:
                    print(f"  ❌ cuobjdump error: {e}")
            break  # 只看第一个 cubin
        
        if best_result:
            break  # 找到一个就够了

    if best_result:
        results[label] = best_result

# ================================================================
# 如果 cuobjdump 失败，尝试用 ptxas 编译 PTX 获取物理寄存器数
# ================================================================
if not results:
    print("\n" + "=" * 70)
    print("cuobjdump 未成功，尝试用 ptxas --print-reg-usage 从 PTX 获取")
    print("=" * 70)

    ptxas_cmd = None
    for p in ['ptxas', '/usr/local/cuda/bin/ptxas']:
        try:
            result = subprocess.run([p, '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                ptxas_cmd = p
                print(f"ptxas 可用: {result.stdout.strip().split(chr(10))[0]}")
                break
        except FileNotFoundError:
            pass

    if ptxas_cmd:
        for kname, cfg in KERNEL_CONFIGS.items():
            label = cfg['label']
            entries = kernel_dirs.get(kname, [])
            for parent_dir, data in entries:
                ptx_files = glob.glob(os.path.join(parent_dir, '*.ptx'))
                if not ptx_files:
                    continue
                pf = ptx_files[0]
                shared = data.get('shared', '?')
                num_warps = data.get('num_warps', '?')
                num_stages = data.get('num_stages', '?')
                try:
                    result = subprocess.run(
                        [ptxas_cmd, '--print-reg-usage', '--gpu-name=sm_80', pf],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        print(f"\n{label} (warps={num_warps}, stages={num_stages}, shared={shared}):")
                        for line in result.stdout.split('\n'):
                            if line.strip():
                                print(f"  {line.strip()}")
                        for line in result.stderr.split('\n'):
                            if line.strip() and ('reg' in line.lower() or 'smem' in line.lower()):
                                print(f"  {line.strip()}")
                        # 解析
                        for line in (result.stdout + result.stderr).split('\n'):
                            reg_match = re.search(r'(\d+)\s+reg', line, re.IGNORECASE)
                            if reg_match:
                                regs = int(reg_match.group(1))
                                results[label] = {
                                    'regs': regs, 'shared': shared,
                                    'num_warps': num_warps, 'num_stages': num_stages,
                                }
                                break
                    else:
                        print(f"\n{label}: ptxas 错误: {result.stderr[:200]}")
                except Exception as e:
                    print(f"\n{label}: {e}")
                break  # 只处理第一个

# ================================================================
# 兜底方案: 从 cubin 二进制头部解析 (ELF format)
# ================================================================
if not results:
    print("\n" + "=" * 70)
    print("兜底: 直接读取 cubin ELF 头部的 .nv.info section")
    print("=" * 70)
    
    for kname, cfg in KERNEL_CONFIGS.items():
        label = cfg['label']
        entries = kernel_dirs.get(kname, [])
        for parent_dir, data in entries:
            cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
            if not cubin_files:
                continue
            cf = cubin_files[0]
            shared = data.get('shared', '?')
            num_warps = data.get('num_warps', '?')
            
            # 尝试用 readelf
            try:
                result = subprocess.run(
                    ['readelf', '-S', cf],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    print(f"\n{label} (warps={num_warps}, shared={shared}):")
                    for line in result.stdout.split('\n'):
                        if '.nv.info' in line or '.text.' in line:
                            print(f"  {line.strip()}")
            except:
                pass
            
            # 尝试用 nvdisasm 
            try:
                result = subprocess.run(
                    ['nvdisasm', '--print-code', cf],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout:
                    # 只看前几行
                    lines = result.stdout.split('\n')[:5]
                    for line in lines:
                        print(f"  nvdisasm: {line.strip()}")
            except:
                pass
            break

# ================================================================
# 用已知数据计算 Occupancy
# ================================================================
print("\n" + "=" * 70)
print("Occupancy 分析 (A100-SXM4-80GB)")
print("=" * 70)

# A100 SM 资源
SM_COUNT = 108
REGS_PER_SM = 65536
SHARED_PER_SM = 163840  # 160 KB (configurable, max 164KB with reduced L1)
MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
REG_ALLOC_UNIT = 256     # A100 分配粒度: 256 regs
SHARED_ALLOC_UNIT = 1024 # shared memory 分配粒度: 1KB (Ampere)

# 如果 cuobjdump 成功了，用真实数据
# 否则用 JSON 里的 shared memory + 估算寄存器
if not results:
    print("\n物理寄存器数未获取到 (cuobjdump/ptxas 不可用)")
    print("使用 JSON 中的 shared memory 数据 + 寄存器估算值\n")

    # 从 JSON shared memory 数据（这些是精确的）
    kernel_data = {
        '🔥1 kkt': {
            'shared': None, 'num_warps': 8, 'num_stages': 3,
            'threads': 256, 'regs_est': 64,
        },
        '🔥2 solve_tril': {
            'shared': None, 'num_warps': None, 'num_stages': None,
            'threads': None, 'regs_est': None,
        },
        '🔥3 recompute_w_u': {
            'shared': None, 'num_warps': 2, 'num_stages': 3,
            'threads': 64, 'regs_est': 128,
        },
        '🔥4 chunk_h': {
            'shared': None, 'num_warps': 4, 'num_stages': 4,
            'threads': 128, 'regs_est': 128,
        },
        '🔥5 chunk_o': {
            'shared': 65536, 'num_warps': 8, 'num_stages': 3,
            'threads': 256, 'regs_est': 64,
        },
    }

    # 从刚才的 JSON 搜索结果中填入精确 shared memory
    for kname, cfg in KERNEL_CONFIGS.items():
        label = cfg['label']
        entries = kernel_dirs.get(kname, [])
        for parent_dir, data in entries:
            shared = data.get('shared', None)
            if shared is not None and label in kernel_data:
                kernel_data[label]['shared'] = shared
                break

    results = {}
    for label, kd in kernel_data.items():
        if kd['shared'] is not None:
            results[label] = {
                'regs': kd['regs_est'],
                'shared': kd['shared'],
                'num_warps': kd['num_warps'],
                'threads': kd['threads'],
                'regs_source': 'estimated',
            }

# 打印 occupancy 分析
print(f"\n{'Kernel':<25} {'warps':>5} {'threads':>7} {'regs/t':>6} {'shared':>8} "
      f"{'max_CTA_reg':>11} {'max_CTA_shmem':>13} {'max_CTA_thrd':>12} "
      f"{'eff_CTA/SM':>10} {'occ%':>5}")
print("-" * 120)

for label in ['🔥1 kkt', '🔥2 solve_tril', '🔥3 recompute_w_u', '🔥4 chunk_h', '🔥5 chunk_o']:
    r = results.get(label)
    if not r:
        print(f"{label:<25} — 数据不足 —")
        continue

    regs = r['regs']
    shared = r['shared']
    num_warps = r['num_warps']
    threads = num_warps * 32

    # Register-limited CTAs per SM
    if regs and regs > 0:
        regs_per_cta = threads * regs
        # 向上对齐到分配粒度
        regs_per_cta_aligned = ((regs_per_cta + REG_ALLOC_UNIT - 1) // REG_ALLOC_UNIT) * REG_ALLOC_UNIT
        max_cta_reg = REGS_PER_SM // regs_per_cta_aligned
    else:
        max_cta_reg = MAX_BLOCKS_PER_SM
        regs = '?'

    # Shared memory-limited CTAs per SM
    if shared and shared > 0:
        shared_aligned = ((shared + SHARED_ALLOC_UNIT - 1) // SHARED_ALLOC_UNIT) * SHARED_ALLOC_UNIT
        max_cta_shmem = SHARED_PER_SM // shared_aligned
    else:
        max_cta_shmem = MAX_BLOCKS_PER_SM

    # Thread-limited CTAs per SM
    max_cta_thread = MAX_THREADS_PER_SM // threads

    # Effective CTAs per SM (minimum of all limits)
    eff_cta = min(max_cta_reg, max_cta_shmem, max_cta_thread, MAX_BLOCKS_PER_SM)

    # Occupancy
    active_threads = eff_cta * threads
    occupancy = active_threads / MAX_THREADS_PER_SM * 100

    regs_src = r.get('regs_source', 'cuobjdump')
    regs_str = f"{regs}{'*' if regs_src == 'estimated' else ''}"

    print(f"{label:<25} {num_warps:>5} {threads:>7} {regs_str:>6} {shared:>8} "
          f"{max_cta_reg:>11} {max_cta_shmem:>13} {max_cta_thread:>12} "
          f"{eff_cta:>10} {occupancy:>5.1f}")

print()
print("注: regs 列带 * 表示估算值，不带 * 表示 cuobjdump 实测值")
print(f"A100 SM 资源: {REGS_PER_SM} regs, {SHARED_PER_SM}B shared, {MAX_THREADS_PER_SM} threads, {MAX_BLOCKS_PER_SM} blocks/SM")

# ================================================================
# 额外: 如果有 cuobjdump，对所有 autotune 配置做 occupancy 对比
# ================================================================
if cuobjdump_available:
    print("\n" + "=" * 70)
    print("🔥4 chunk_h: 所有 autotune 配置的 occupancy 对比")
    print("=" * 70)

    kname = 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64'
    all_entries = []
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            if data.get('name') == kname:
                parent_dir = os.path.dirname(jf)
                all_entries.append((parent_dir, data))
        except:
            pass

    print(f"找到 {len(all_entries)} 个编译版本\n")
    print(f"{'warps':>5} {'stages':>6} {'shared':>8} {'regs':>5} {'max_CTA_reg':>11} "
          f"{'max_CTA_shm':>11} {'eff_CTA/SM':>10} {'occ%':>5}")
    print("-" * 80)

    seen = set()
    for parent_dir, data in all_entries:
        shared = data.get('shared', 0)
        num_warps = data.get('num_warps', 0)
        num_stages = data.get('num_stages', 0)
        key = (num_warps, num_stages, shared)
        if key in seen:
            continue
        seen.add(key)

        threads = num_warps * 32
        cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
        regs = None
        if cubin_files:
            try:
                result = subprocess.run(
                    [cuobjdump_cmd, '--resource-usage', cubin_files[0]],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    reg_match = re.search(r'REG[:\s]+(\d+)', result.stdout, re.IGNORECASE)
                    if reg_match:
                        regs = int(reg_match.group(1))
            except:
                pass

        if regs:
            regs_per_cta = ((threads * regs + REG_ALLOC_UNIT - 1) // REG_ALLOC_UNIT) * REG_ALLOC_UNIT
            max_cta_reg = REGS_PER_SM // regs_per_cta
        else:
            max_cta_reg = MAX_BLOCKS_PER_SM
            regs = '?'

        shared_aligned = ((shared + SHARED_ALLOC_UNIT - 1) // SHARED_ALLOC_UNIT) * SHARED_ALLOC_UNIT
        max_cta_shmem = SHARED_PER_SM // shared_aligned if shared > 0 else MAX_BLOCKS_PER_SM
        max_cta_thread = MAX_THREADS_PER_SM // threads
        eff_cta = min(max_cta_reg, max_cta_shmem, max_cta_thread, MAX_BLOCKS_PER_SM)
        occ = eff_cta * threads / MAX_THREADS_PER_SM * 100

        print(f"{num_warps:>5} {num_stages:>6} {shared:>8} {str(regs):>5} "
              f"{max_cta_reg:>11} {max_cta_shmem:>11} {eff_cta:>10} {occ:>5.1f}")

print("\n完成!")
