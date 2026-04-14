"""
occupancy_analysis_v2.py — 从 cubin 提取物理寄存器数 + occupancy 精确计算
修复:
  1. cuobjdump --resource-usage 输出格式解析 (打印原始输出以调试)
  2. ptxas 选项修复 (--print-reg-usage)  
  3. 🔥2 None warps 崩溃修复
  4. 新增: 从 .nv.info section 二进制解析寄存器数
"""
import torch
import torch.nn.functional as F
import triton
import os, glob, json, subprocess, re, struct

B, T, H, D = 1, 8192, 32, 128
BT = 64
device = "cuda"
dtype = torch.bfloat16

print(f"Triton version: {triton.__version__}")
print(f"GPU: {torch.cuda.get_device_name()}")

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
# 搜索 JSON 缓存，找获胜配置
# ================================================================
KERNEL_CONFIGS = {
    'chunk_scaled_dot_kkt_fwd_kernel': {
        'label': '🔥1 kkt',
        'win': {'num_warps': 8, 'num_stages': 3},
    },
    'merge_16x16_to_64x64_inverse_kernel': {
        'label': '🔥2 solve_tril',
        'win': None,
    },
    'recompute_w_u_fwd_kernel': {
        'label': '🔥3 recompute_w_u',
        'win': {'num_warps': 2, 'num_stages': 3},
    },
    'chunk_gated_delta_rule_fwd_kernel_h_blockdim64': {
        'label': '🔥4 chunk_h',
        'win': {'num_warps': 4, 'num_stages': 4},
    },
    'chunk_fwd_kernel_o': {
        'label': '🔥5 chunk_o',
        'win': {'num_warps': 8, 'num_stages': 3},
    },
}

cache_dir = os.environ.get('TRITON_CACHE_DIR', os.path.expanduser('~/.triton/cache'))
json_files = glob.glob(os.path.join(cache_dir, '**', '*.json'), recursive=True)
print(f"总 JSON 文件数: {len(json_files)}")

kernel_dirs = {}
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
# 方法 1: cuobjdump — 先打印原始输出看格式
# ================================================================
print("\n" + "=" * 70)
print("方法 1: cuobjdump --resource-usage 原始输出")
print("=" * 70)

results = {}

for kname, cfg in KERNEL_CONFIGS.items():
    label = cfg['label']
    entries = kernel_dirs.get(kname, [])
    if not entries:
        print(f"\n{label}: 未在缓存中找到")
        continue

    parent_dir, data = entries[0]
    shared = data.get('shared', '?')
    num_warps = data.get('num_warps', '?')
    num_stages = data.get('num_stages', '?')

    cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
    if not cubin_files:
        print(f"\n{label}: 无 cubin 文件")
        continue

    cf = cubin_files[0]
    print(f"\n{label} (warps={num_warps}, stages={num_stages}, shared={shared}):")
    print(f"  cubin: {cf}")

    # 尝试多种 cuobjdump 模式
    for flag in ['--resource-usage', '-res-usage']:
        try:
            result = subprocess.run(
                ['cuobjdump', flag, cf],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"  cuobjdump {flag}:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"    {line}")
                # 尝试多种格式解析
                full_output = result.stdout
                regs = None
                # 格式1: "REG:128"
                m = re.search(r'REG\s*[:=]\s*(\d+)', full_output, re.IGNORECASE)
                if m: regs = int(m.group(1))
                # 格式2: "Registers: 128"  
                if not regs:
                    m = re.search(r'[Rr]egisters?\s*[:=]\s*(\d+)', full_output)
                    if m: regs = int(m.group(1))
                # 格式3: "128 registers"
                if not regs:
                    m = re.search(r'(\d+)\s+registers?', full_output, re.IGNORECASE)
                    if m: regs = int(m.group(1))
                # 格式4: cuobjdump 可能不输出 resource usage
                if regs:
                    print(f"  → 解析到 regs = {regs}")
                    results[label] = {
                        'regs': regs, 'shared': shared,
                        'num_warps': num_warps, 'num_stages': num_stages,
                    }
                break
            elif result.stderr.strip():
                print(f"  cuobjdump {flag} stderr: {result.stderr.strip()[:200]}")
        except Exception as e:
            print(f"  cuobjdump {flag}: {e}")

    # 如果 cuobjdump 没解析到，尝试 nvdisasm --print-code (有时显示资源)
    if label not in results:
        try:
            result = subprocess.run(
                ['nvdisasm', '-gi', cf],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # nvdisasm -gi 输出 kernel info 包含寄存器数
                for line in result.stdout.split('\n')[:20]:
                    if line.strip():
                        print(f"  nvdisasm -gi: {line.strip()}")
                    m = re.search(r'(\d+)\s+reg', line, re.IGNORECASE)
                    if m:
                        regs = int(m.group(1))
                        results[label] = {
                            'regs': regs, 'shared': shared,
                            'num_warps': num_warps, 'num_stages': num_stages,
                        }
        except:
            pass

# ================================================================
# 方法 2: 从 cubin 的 ELF .nv.info section 直接解析寄存器数
# ================================================================
print("\n" + "=" * 70)
print("方法 2: 从 cubin ELF .nv.info 二进制解析寄存器数")
print("=" * 70)

def parse_nv_info_regs(cubin_path):
    """从 cubin ELF 的 .nv.info section 解析寄存器数
    
    .nv.info section 包含 EIATTR 条目:
      EIATTR_REGCOUNT (0x2f04): 4 bytes, 寄存器数
      EIATTR_MAX_THREADS (0x0504): 4 bytes, 最大线程数
      EIATTR_MIN_STACK_SIZE: stack size
    """
    try:
        # 用 readelf 获取 .nv.info section 的偏移和大小
        result = subprocess.run(
            ['readelf', '-S', '-W', cubin_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        
        # 找 .nv.info. 开头的 section (kernel-specific info)
        nv_info_sections = []
        for line in result.stdout.split('\n'):
            if '.nv.info.' in line:
                # 解析 section header
                # 格式: [Nr] Name Type Addr Off Size ...
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.startswith('.nv.info.'):
                        # 找 offset 和 size (十六进制)
                        # readelf -W 格式不固定，用正则
                        hex_vals = re.findall(r'[0-9a-fA-F]{8,16}', line)
                        if len(hex_vals) >= 3:
                            offset = int(hex_vals[1], 16)  # 通常第2个是offset
                            size = int(hex_vals[2], 16)    # 第3个是size
                            nv_info_sections.append((p, offset, size))
                        break
        
        if not nv_info_sections:
            return None
        
        # 读取 cubin 二进制
        with open(cubin_path, 'rb') as f:
            cubin_data = f.read()
        
        for sec_name, offset, size in nv_info_sections:
            section_data = cubin_data[offset:offset+size]
            # EIATTR 格式: 每个条目 = format(2B) + attr(2B) + size(4B if needed) + value
            # EIATTR_REGCOUNT: format=0x04, attr=0x2f (or combined 0x2f04)
            # 实际格式可能是: attr(2B) + size(2B) + value(size bytes)
            
            # 简单搜索: 找 0x2f04 pattern
            pos = 0
            while pos < len(section_data) - 8:
                # 尝试不同的字节序
                b0, b1 = section_data[pos], section_data[pos+1]
                
                # EIATTR_REGCOUNT = 0x2f, format = 0x04  
                if (b0 == 0x04 and b1 == 0x2f) or (b0 == 0x2f and b1 == 0x04):
                    # 后面跟着寄存器数 (通常是 4 bytes LE)
                    try:
                        # 尝试不同偏移
                        for delta in [2, 4]:
                            val = struct.unpack_from('<I', section_data, pos + delta)[0]
                            if 8 <= val <= 255:  # 合理的寄存器范围
                                return val
                    except:
                        pass
                pos += 1
                
        return None
    except Exception as e:
        print(f"  parse error: {e}")
        return None


for kname, cfg in KERNEL_CONFIGS.items():
    label = cfg['label']
    if label in results:
        continue  # 已经有数据了
    
    entries = kernel_dirs.get(kname, [])
    if not entries:
        continue
    
    parent_dir, data = entries[0]
    shared = data.get('shared', '?')
    num_warps = data.get('num_warps', '?')
    
    cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
    if not cubin_files:
        continue
    
    regs = parse_nv_info_regs(cubin_files[0])
    if regs:
        print(f"  {label}: regs={regs} (from .nv.info)")
        results[label] = {
            'regs': regs, 'shared': shared,
            'num_warps': num_warps, 'num_stages': data.get('num_stages', '?'),
        }
    else:
        print(f"  {label}: .nv.info 解析失败")

# ================================================================
# 方法 3: 用 ptxas 重新编译 PTX (修复选项名)
# ================================================================
if len(results) < 5:
    print("\n" + "=" * 70)
    print("方法 3: 用 ptxas 重新编译 PTX 获取物理寄存器数")
    print("=" * 70)

    for kname, cfg in KERNEL_CONFIGS.items():
        label = cfg['label']
        if label in results:
            continue
        
        entries = kernel_dirs.get(kname, [])
        if not entries:
            continue
        
        parent_dir, data = entries[0]
        shared = data.get('shared', '?')
        num_warps = data.get('num_warps', '?')
        
        ptx_files = glob.glob(os.path.join(parent_dir, '*.ptx'))
        if not ptx_files:
            # 尝试用 cuobjdump 从 cubin 提取 ptx
            cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
            if cubin_files:
                try:
                    result = subprocess.run(
                        ['cuobjdump', '--dump-ptx', cubin_files[0]],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0 and result.stdout:
                        ptx_tmp = '/tmp/_triton_tmp.ptx'
                        # 提取 PTX 内容 (去掉 cuobjdump header)
                        ptx_content = result.stdout
                        # 找 .version 开头
                        idx = ptx_content.find('.version')
                        if idx >= 0:
                            with open(ptx_tmp, 'w') as f:
                                f.write(ptx_content[idx:])
                            ptx_files = [ptx_tmp]
                except:
                    pass
            if not ptx_files:
                continue
        
        pf = ptx_files[0]
        print(f"\n  {label} (warps={num_warps}, shared={shared}):")
        
        # ptxas --print-reg-usage (双横线!)
        try:
            result = subprocess.run(
                ['ptxas', '--print-reg-usage', '--gpu-name=sm_80', '-o', '/dev/null', pf],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout + '\n' + result.stderr
            print(f"    ptxas output:")
            for line in output.split('\n'):
                if line.strip():
                    print(f"      {line.strip()}")
            
            # 解析: "Used 128 registers, 65536 bytes smem, ..."
            m = re.search(r'[Uu]sed\s+(\d+)\s+reg', output)
            if m:
                regs = int(m.group(1))
                print(f"    → regs = {regs}")
                results[label] = {
                    'regs': regs, 'shared': shared,
                    'num_warps': num_warps, 'num_stages': data.get('num_stages', '?'),
                }
            # 尝试另一种格式: "N registers"
            if label not in results:
                m = re.search(r'(\d+)\s+reg', output)
                if m:
                    regs = int(m.group(1))
                    results[label] = {
                        'regs': regs, 'shared': shared,
                        'num_warps': num_warps, 'num_stages': data.get('num_stages', '?'),
                    }
        except Exception as e:
            print(f"    ptxas error: {e}")

# ================================================================
# 方法 4: 用 CUfunction API 获取寄存器数 (通过 pycuda 或 cupy)
# ================================================================
if len(results) < 5:
    print("\n" + "=" * 70)
    print("方法 4: 尝试用 cupy/pycuda 加载 cubin 获取寄存器数")
    print("=" * 70)
    
    try:
        import cupy
        for kname, cfg in KERNEL_CONFIGS.items():
            label = cfg['label']
            if label in results:
                continue
            entries = kernel_dirs.get(kname, [])
            if not entries:
                continue
            parent_dir, data = entries[0]
            cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
            if not cubin_files:
                continue
            with open(cubin_files[0], 'rb') as f:
                cubin_data = f.read()
            mod = cupy.cuda.Module()
            mod.load(cubin_data)
            # 获取 function
            func_name = kname
            try:
                func = mod.get_function(func_name)
                regs = func.num_regs
                print(f"  {label}: regs={regs} (from cupy)")
                results[label] = {
                    'regs': regs, 'shared': data.get('shared', '?'),
                    'num_warps': data.get('num_warps', '?'),
                    'num_stages': data.get('num_stages', '?'),
                }
            except Exception as e:
                print(f"  {label}: {e}")
    except ImportError:
        print("  cupy 未安装")

    if len(results) < 5:
        try:
            import pycuda.driver as drv
            drv.init()
            dev = drv.Device(0)
            ctx = dev.make_context()
            for kname, cfg in KERNEL_CONFIGS.items():
                label = cfg['label']
                if label in results:
                    continue
                entries = kernel_dirs.get(kname, [])
                if not entries:
                    continue
                parent_dir, data = entries[0]
                cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
                if not cubin_files:
                    continue
                mod = drv.module_from_file(cubin_files[0])
                func = mod.get_function(kname)
                regs = func.num_regs
                print(f"  {label}: regs={regs} (from pycuda)")
                results[label] = {
                    'regs': regs, 'shared': data.get('shared', '?'),
                    'num_warps': data.get('num_warps', '?'),
                    'num_stages': data.get('num_stages', '?'),
                }
            ctx.pop()
        except ImportError:
            print("  pycuda 未安装")
        except Exception as e:
            print(f"  pycuda error: {e}")

# ================================================================
# 方法 5: 用 ctypes 直接调用 CUDA Driver API
# ================================================================
if len(results) < 5:
    print("\n" + "=" * 70)
    print("方法 5: 用 ctypes 调用 cuModuleLoad + cuFuncGetAttribute")
    print("=" * 70)
    
    import ctypes
    try:
        libcuda = ctypes.CDLL('libcuda.so.1')
        
        # CU_FUNC_ATTRIBUTE_NUM_REGS = 0
        # CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
        CU_FUNC_ATTRIBUTE_NUM_REGS = 0
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
        
        for kname, cfg in KERNEL_CONFIGS.items():
            label = cfg['label']
            if label in results:
                continue
            entries = kernel_dirs.get(kname, [])
            if not entries:
                continue
            parent_dir, data = entries[0]
            cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
            if not cubin_files:
                continue
            
            cf = cubin_files[0]
            print(f"\n  {label}: loading {os.path.basename(cf)}")
            
            # cuModuleLoad
            module = ctypes.c_void_p()
            ret = libcuda.cuModuleLoad(ctypes.byref(module), cf.encode())
            if ret != 0:
                print(f"    cuModuleLoad failed: {ret}")
                continue
            
            # cuModuleGetFunction
            func = ctypes.c_void_p()
            ret = libcuda.cuModuleGetFunction(ctypes.byref(func), module, kname.encode())
            if ret != 0:
                print(f"    cuModuleGetFunction failed: {ret}")
                libcuda.cuModuleUnload(module)
                continue
            
            # cuFuncGetAttribute — NUM_REGS
            num_regs = ctypes.c_int()
            ret = libcuda.cuFuncGetAttribute(
                ctypes.byref(num_regs), CU_FUNC_ATTRIBUTE_NUM_REGS, func
            )
            if ret != 0:
                print(f"    cuFuncGetAttribute(NUM_REGS) failed: {ret}")
                libcuda.cuModuleUnload(module)
                continue
            
            # cuFuncGetAttribute — SHARED_SIZE_BYTES
            shared_bytes = ctypes.c_int()
            ret = libcuda.cuFuncGetAttribute(
                ctypes.byref(shared_bytes), CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func
            )
            
            regs = num_regs.value
            shmem = shared_bytes.value if ret == 0 else data.get('shared', '?')
            
            print(f"    ✅ regs={regs}, shared_static={shmem}")
            results[label] = {
                'regs': regs, 'shared': data.get('shared', '?'),  # 用 JSON 的 shared (包含动态分配)
                'num_warps': data.get('num_warps', '?'),
                'num_stages': data.get('num_stages', '?'),
            }
            
            libcuda.cuModuleUnload(module)
            
    except Exception as e:
        print(f"  ctypes error: {e}")

# ================================================================
# Occupancy 计算
# ================================================================
print("\n" + "=" * 70)
print("Occupancy 分析 (A100-SXM4-80GB)")
print("=" * 70)

SM_COUNT = 108
REGS_PER_SM = 65536
SHARED_PER_SM = 166912  # 163 KB (A100 max configurable)
MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
REG_ALLOC_UNIT = 256
SHARED_ALLOC_UNIT = 128  # A100: 128 bytes

print(f"\n已获取到 {len(results)}/{len(KERNEL_CONFIGS)} 个 kernel 的物理寄存器数\n")

# 如果某些 kernel 还是没有寄存器数据，用估算
for label in ['🔥1 kkt', '🔥2 solve_tril', '🔥3 recompute_w_u', '🔥4 chunk_h', '🔥5 chunk_o']:
    if label not in results:
        # 从 JSON 获取 shared memory
        for kname, cfg in KERNEL_CONFIGS.items():
            if cfg['label'] == label:
                entries = kernel_dirs.get(kname, [])
                if entries:
                    _, data = entries[0]
                    results[label] = {
                        'regs': None,
                        'shared': data.get('shared', 0),
                        'num_warps': data.get('num_warps', 4),
                        'num_stages': data.get('num_stages', 3),
                    }
                break

# 打印
latency_ms = {'🔥1 kkt': 0.090, '🔥2 solve_tril': 0.316, '🔥3 recompute_w_u': 0.200, 
              '🔥4 chunk_h': 0.332, '🔥5 chunk_o': 0.298}
grid_ctas = {'🔥1 kkt': 4096, '🔥2 solve_tril': 4096, '🔥3 recompute_w_u': 4096,
             '🔥4 chunk_h': 64, '🔥5 chunk_o': 4096}

print(f"{'Kernel':<22} {'warps':>5} {'thr':>4} {'regs':>5} {'shared_KB':>9} "
      f"{'lim_reg':>7} {'lim_shm':>7} {'lim_thr':>7} "
      f"{'CTA/SM':>6} {'occ%':>5} {'grid':>6} {'lat_ms':>6}")
print("-" * 110)

for label in ['🔥1 kkt', '🔥2 solve_tril', '🔥3 recompute_w_u', '🔥4 chunk_h', '🔥5 chunk_o']:
    r = results.get(label, {})
    regs = r.get('regs')
    shared = r.get('shared', 0)
    num_warps = r.get('num_warps', 4)
    
    if num_warps is None:
        num_warps = 4  # fallback
    threads = num_warps * 32

    # Register limit
    if regs and regs > 0:
        regs_per_cta = threads * regs
        regs_aligned = ((regs_per_cta + REG_ALLOC_UNIT - 1) // REG_ALLOC_UNIT) * REG_ALLOC_UNIT
        lim_reg = REGS_PER_SM // regs_aligned
    else:
        lim_reg = MAX_BLOCKS_PER_SM

    # Shared memory limit
    if shared and shared > 0:
        shared_aligned = ((shared + SHARED_ALLOC_UNIT - 1) // SHARED_ALLOC_UNIT) * SHARED_ALLOC_UNIT
        lim_shm = SHARED_PER_SM // shared_aligned
    else:
        lim_shm = MAX_BLOCKS_PER_SM

    # Thread limit
    lim_thr = MAX_THREADS_PER_SM // threads

    # Effective
    eff_cta = min(lim_reg, lim_shm, lim_thr, MAX_BLOCKS_PER_SM)
    occupancy = eff_cta * threads / MAX_THREADS_PER_SM * 100

    regs_str = f"{regs}" if regs else "?"
    shared_kb = f"{shared/1024:.1f}" if shared else "?"
    lat = latency_ms.get(label, 0)
    grid = grid_ctas.get(label, 0)

    print(f"{label:<22} {num_warps:>5} {threads:>4} {regs_str:>5} {shared_kb:>9} "
          f"{lim_reg:>7} {lim_shm:>7} {lim_thr:>7} "
          f"{eff_cta:>6} {occupancy:>5.1f} {grid:>6} {lat:>6.3f}")

print()
print(f"A100 resources/SM: regs={REGS_PER_SM}, shared={SHARED_PER_SM}B, "
      f"threads={MAX_THREADS_PER_SM}, blocks={MAX_BLOCKS_PER_SM}")
print("lim_reg/lim_shm/lim_thr = 寄存器/shared/线程 各自允许的最大 CTA/SM")
print("CTA/SM = min(lim_reg, lim_shm, lim_thr, 32)")
print("occ% = CTA/SM × threads / 2048 × 100")

print("\n完成!")
