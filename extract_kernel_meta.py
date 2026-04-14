"""
extract_kernel_meta.py — 从 Triton 缓存中提取 5 个 DeltaNet kernel 的
寄存器数、shared memory、spill 等编译元数据

方法:
  1. 在 ~/.triton/cache 的 JSON 文件中按 kernel 名匹配
  2. 深入 JITFunction.cache[0] 的 dict，查找 CompiledKernel
  3. 查找 .cubin/.ttgir/.ptx 文件获取寄存器信息
"""
import torch
import torch.nn.functional as F
import triton
import os, glob, json, re

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
# 方法 1: 搜索 Triton 缓存 JSON — 按 kernel 名匹配
# ================================================================
print("=" * 70)
print("方法 1: 在 Triton 缓存 JSON 中搜索 DeltaNet kernel")
print("=" * 70)

KERNEL_NAMES = [
    'chunk_scaled_dot_kkt_fwd_kernel',        # 🔥1
    'solve_tril_16x16_kernel',                # 🔥2a
    'merge_16x16_to_32x32_inverse_kernel',    # 🔥2b
    'merge_16x16_to_64x64_inverse_kernel',    # 🔥2c
    'recompute_w_u_fwd_kernel',               # 🔥3
    'chunk_gated_delta_rule_fwd_kernel_h_blockdim64',  # 🔥4
    'chunk_fwd_kernel_o',                     # 🔥5
]

cache_dir = os.environ.get('TRITON_CACHE_DIR', os.path.expanduser('~/.triton/cache'))
json_files = glob.glob(os.path.join(cache_dir, '**', '*.json'), recursive=True)
print(f"总 JSON 文件数: {len(json_files)}")

found_kernels = {}
for jf in json_files:
    try:
        with open(jf) as f:
            data = json.load(f)
        name = data.get('name', '')
        if name in KERNEL_NAMES:
            if name not in found_kernels:
                found_kernels[name] = []
            found_kernels[name].append((jf, data))
    except:
        pass

for kname in KERNEL_NAMES:
    entries = found_kernels.get(kname, [])
    if not entries:
        print(f"\n  ❌ {kname}: 未找到")
        continue
    print(f"\n  ✅ {kname}: 找到 {len(entries)} 个编译版本")
    for jf, data in entries:
        parent_dir = os.path.dirname(jf)
        shared = data.get('shared', '?')
        num_warps = data.get('num_warps', '?')
        num_stages = data.get('num_stages', '?')
        num_ctas = data.get('num_ctas', '?')
        maxnreg = data.get('maxnreg', '?')
        print(f"    shared={shared}, num_warps={num_warps}, num_stages={num_stages}, "
              f"num_ctas={num_ctas}, maxnreg={maxnreg}")

        # 在同目录下找 .ptx 文件 — PTX 里有寄存器信息
        ptx_files = glob.glob(os.path.join(parent_dir, '*.ptx'))
        for pf in ptx_files:
            try:
                with open(pf) as f:
                    ptx = f.read()
                # 查找 .reqntid 和 .maxnreg
                for line in ptx.split('\n'):
                    if '.reqntid' in line or '.maxnreg' in line or '.reg ' in line:
                        print(f"      PTX: {line.strip()}")
                        break
            except:
                pass

        # 在同目录下找 .cubin 文件 — 用 cuobjdump 可以看
        cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
        for cf in cubin_files:
            print(f"      cubin: {cf} ({os.path.getsize(cf)} bytes)")

        # 在同目录下找其他文件
        all_files = os.listdir(parent_dir)
        print(f"      目录内文件: {all_files}")

# ================================================================
# 方法 2: 深入 JITFunction.cache[0] dict
# ================================================================
print("\n" + "=" * 70)
print("方法 2: 深入 JITFunction.cache[0] 的 dict 内容")
print("=" * 70)

import fla.ops.common.chunk_scaled_dot_kkt as _mod_kkt
import fla.ops.delta_rule.wy_fast as _mod_wy
import fla.ops.common.chunk_delta_h as _mod_h
import fla.ops.common.chunk_o as _mod_o

kernels = [
    ("🔥1 kkt", getattr(_mod_kkt, 'chunk_scaled_dot_kkt_fwd_kernel', None)),
    ("🔥3 recompute_w_u", getattr(_mod_wy, 'recompute_w_u_fwd_kernel', None)),
    ("🔥4 chunk_h", getattr(_mod_h, 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64', None)),
    ("🔥5 chunk_o", getattr(_mod_o, 'chunk_fwd_kernel_o', None)),
]

for name, kfn in kernels:
    print(f"\n--- {name} ---")
    if kfn is None:
        print("  (None)")
        continue

    # 找到最内层 JITFunction
    fn = kfn
    while hasattr(fn, 'fn'):
        fn = fn.fn

    if not hasattr(fn, 'cache'):
        print(f"  最内层类型: {type(fn).__name__}, 无 cache")
        continue

    print(f"  最内层类型: {type(fn).__name__}")
    print(f"  cache keys: {list(fn.cache.keys())}")

    for ck, cv in fn.cache.items():
        print(f"  cache[{ck}] type: {type(cv).__name__}")
        if isinstance(cv, dict):
            print(f"  cache[{ck}] 有 {len(cv)} 个 sub-keys:")
            for sk, sv in cv.items():
                sv_type = type(sv).__name__
                print(f"    [{repr(sk)[:80]}] -> {sv_type}")
                # 深入 CompiledKernel 或类似对象
                if sv_type not in ('int', 'float', 'str', 'bool', 'NoneType', 'list', 'tuple', 'dict'):
                    sv_attrs = [a for a in dir(sv) if not a.startswith('_')]
                    print(f"      attrs: {sv_attrs}")
                    for attr in sv_attrs:
                        try:
                            val = getattr(sv, attr)
                            if not callable(val):
                                val_str = repr(val)
                                if len(val_str) > 300:
                                    val_str = val_str[:300] + '...'
                                print(f"      .{attr} = {val_str}")
                        except Exception as e:
                            print(f"      .{attr} -> ERROR: {e}")
            if len(cv) > 5:
                print(f"    ... (showing first 5 of {len(cv)})")
                break

# ================================================================
# 方法 3: 用 cuobjdump 从 cubin 提取寄存器信息
# ================================================================
print("\n" + "=" * 70)
print("方法 3: 用 cuobjdump 从 cubin 提取寄存器信息")
print("=" * 70)

import subprocess
for kname in KERNEL_NAMES:
    entries = found_kernels.get(kname, [])
    for jf, data in entries:
        parent_dir = os.path.dirname(jf)
        cubin_files = glob.glob(os.path.join(parent_dir, '*.cubin'))
        for cf in cubin_files:
            try:
                result = subprocess.run(
                    ['cuobjdump', '--resource-usage', cf],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    output = result.stdout
                    # 找 REG, SMEM, STACK 等
                    print(f"\n  {kname}:")
                    for line in output.split('\n'):
                        line_s = line.strip()
                        if line_s and any(w in line_s.upper() for w in ['REG', 'SMEM', 'SHARED', 'STACK', 'SPILL', 'CMEM', 'FUNCTION']):
                            print(f"    {line_s}")
                else:
                    print(f"\n  {kname}: cuobjdump error: {result.stderr[:200]}")
            except FileNotFoundError:
                print(f"\n  cuobjdump 未安装或不在 PATH 中")
                break
            except Exception as e:
                print(f"\n  {kname}: {e}")

# ================================================================
# 方法 4: 直接从 PTX 解析寄存器信息
# ================================================================
print("\n" + "=" * 70)
print("方法 4: 从 PTX 文件解析寄存器声明")
print("=" * 70)

for kname in KERNEL_NAMES:
    entries = found_kernels.get(kname, [])
    for jf, data in entries:
        parent_dir = os.path.dirname(jf)
        ptx_files = glob.glob(os.path.join(parent_dir, '*.ptx'))
        for pf in ptx_files:
            try:
                with open(pf) as f:
                    ptx = f.read()

                shared = data.get('shared', '?')
                num_warps = data.get('num_warps', '?')
                num_stages = data.get('num_stages', '?')

                # 解析 .reg 声明
                reg_decls = re.findall(r'\.reg\s+\.(\w+)\s+%(\w+)<(\d+)>', ptx)
                total_regs = {}
                for rtype, rname, rcount in reg_decls:
                    rcount = int(rcount)
                    if rtype not in total_regs:
                        total_regs[rtype] = 0
                    total_regs[rtype] += rcount

                print(f"\n  {kname} (warps={num_warps}, stages={num_stages}, shared={shared}):")
                print(f"    PTX 寄存器声明:")
                for rtype in sorted(total_regs.keys()):
                    print(f"      .{rtype}: {total_regs[rtype]} 个")

                # 找 .maxnreg 指令
                maxnreg_match = re.findall(r'\.maxnreg\s+(\d+)', ptx)
                if maxnreg_match:
                    print(f"    .maxnreg: {maxnreg_match}")

                # 找 .reqntid 指令
                reqntid_match = re.findall(r'\.reqntid\s+(\d+)', ptx)
                if reqntid_match:
                    print(f"    .reqntid: {reqntid_match}")

            except Exception as e:
                print(f"  {kname}: PTX 读取失败: {e}")

print("\n" + "=" * 70)
print("完成!")
print("=" * 70)
