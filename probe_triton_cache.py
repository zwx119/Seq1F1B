"""
probe_triton_cache.py — 探测 Triton 编译缓存的真实结构
找到寄存器数、shared memory 等元数据的确切位置
"""
import torch
import torch.nn.functional as F
import triton
import os, glob, json

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

# === 探测 1: solve_tril 模块里所有可用名字 ===
import fla.ops.utils.solve_tril as _mod_solve
print("=" * 60)
print("solve_tril 模块所有公开属性:")
for name in dir(_mod_solve):
    if not name.startswith('_'):
        obj = getattr(_mod_solve, name)
        print(f"  {name}: {type(obj).__name__}")
print()

# === 探测 2: 查看 kernel 对象每一层的结构 ===
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
    print("=" * 60)
    print(f"{name}")
    if kfn is None:
        print("  (None)")
        continue
    
    # 逐层展开
    layer = 0
    fn = kfn
    while fn is not None:
        tname = type(fn).__name__
        attrs = [a for a in dir(fn) if not a.startswith('_')]
        print(f"  第{layer}层: {tname}")
        print(f"    属性: {attrs}")
        
        # 检查 cache
        if hasattr(fn, 'cache') and fn.cache:
            print(f"    cache 有 {len(fn.cache)} 条:")
            for ck, cv in fn.cache.items():
                print(f"      key = {ck}")
                print(f"      value type = {type(cv).__name__}")
                cv_attrs = [a for a in dir(cv) if not a.startswith('_')]
                print(f"      value attrs = {cv_attrs}")
                # 深入看 value 的每个属性
                for va in cv_attrs:
                    try:
                        vv = getattr(cv, va)
                        if not callable(vv):
                            print(f"        {va} = {repr(vv)[:200]}")
                    except:
                        pass
                break  # 只看第一个
        
        # 检查 kernel_cache (有些 Triton 版本用这个名字)
        for cache_name in ['kernel_cache', '_cache', 'compiled_cache', 'kernels']:
            c = getattr(fn, cache_name, None)
            if c:
                print(f"    发现 {cache_name}: {type(c).__name__}, len={len(c) if hasattr(c, '__len__') else '?'}")
        
        # 进入下一层
        if hasattr(fn, 'fn'):
            fn = fn.fn
            layer += 1
        else:
            break
    print()

# === 探测 3: 从 Triton 缓存目录读取编译元数据 ===
print("=" * 60)
print("Triton 缓存目录:")
cache_dir = os.environ.get('TRITON_CACHE_DIR', os.path.expanduser('~/.triton/cache'))
print(f"  TRITON_CACHE_DIR = {cache_dir}")
if os.path.exists(cache_dir):
    # 找所有 .json 元数据文件
    json_files = glob.glob(os.path.join(cache_dir, '**', '*.json'), recursive=True)
    print(f"  JSON 文件数: {len(json_files)}")
    
    # 找最近修改的（刚编译的 kernel 的元数据）
    if json_files:
        json_files.sort(key=os.path.getmtime, reverse=True)
        for jf in json_files[:10]:
            try:
                with open(jf) as f:
                    data = json.load(f)
                # 看看包含什么 key
                fname = os.path.basename(os.path.dirname(jf)) + "/" + os.path.basename(jf)
                relevant_keys = [k for k in data.keys() if any(w in k.lower() for w in ['reg', 'shared', 'warp', 'spill', 'name', 'kernel'])]
                if relevant_keys or 'name' in data:
                    print(f"\n  {fname}:")
                    for k in sorted(data.keys()):
                        v = data[k]
                        if not isinstance(v, (dict, list)) or len(str(v)) < 200:
                            print(f"    {k} = {v}")
            except:
                pass
else:
    print(f"  目录不存在!")

# === 探测 4: 用 CUDA runtime 查询 kernel 的寄存器数 ===
print("\n" + "=" * 60)
print("尝试通过 CUDA Runtime API 查询 kernel 属性:")
print("(cuFuncGetAttribute — 寄存器数, shared memory)")

try:
    # 尝试用 pycuda 或 ctypes 调用 cuFuncGetAttribute
    import ctypes
    libcuda = ctypes.CDLL('libcuda.so.1')
    
    # CU_FUNC_ATTRIBUTE_NUM_REGS = 0
    # CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    print("  libcuda.so.1 loaded")
    print("  (需要 CUfunction handle, 但 Triton 不直接暴露)")
except Exception as e:
    print(f"  {e}")

# === 探测 5: 用 Triton compiler API 直接编译一个小 kernel 看 metadata ===
print("\n" + "=" * 60)
print("用 Triton 编译一个最小 kernel 来验证 metadata 获取方式:")

@triton.jit
def _test_kernel(X, Y, N: triton.language.constexpr):
    pid = triton.language.program_id(0)
    offs = pid * N + triton.language.arange(0, N)
    x = triton.language.load(X + offs)
    triton.language.store(Y + offs, x * 2.0)

x = torch.randn(1024, device='cuda')
y = torch.empty_like(x)
grid = (1024 // 256,)
_test_kernel[grid](x, y, 256)
torch.cuda.synchronize()

print(f"  _test_kernel type: {type(_test_kernel).__name__}")
print(f"  _test_kernel attrs: {[a for a in dir(_test_kernel) if not a.startswith('_')]}")

if hasattr(_test_kernel, 'cache'):
    print(f"  cache len: {len(_test_kernel.cache)}")
    for ck, cv in _test_kernel.cache.items():
        print(f"    key: {ck}")
        print(f"    value type: {type(cv).__name__}")
        cv_attrs = [a for a in dir(cv) if not a.startswith('_')]
        print(f"    value attrs: {cv_attrs}")
        for va in cv_attrs:
            try:
                vv = getattr(cv, va)
                if not callable(vv):
                    print(f"      {va} = {repr(vv)[:300]}")
            except:
                pass
        break

print("\n完成!")
