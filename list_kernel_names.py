"""
list_kernel_names.py — 列出所有实际 CUDA kernel 名字
用 torch.profiler 捕获真实 kernel name，给 ncu --kernel-name 用
"""
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

B, T, H, D = 1, 1024, 32, 128  # 小一点跑快
device = "cuda"
dtype = torch.bfloat16

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o

torch.manual_seed(42)
q = torch.randn(B, T, H, D, device=device, dtype=dtype)
k = F.normalize(torch.randn(B, T, H, D, device=device, dtype=dtype), p=2, dim=-1)
v = torch.randn(B, T, H, D, device=device, dtype=dtype)
beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
scale = D ** -0.5

# warmup
with torch.no_grad():
    for _ in range(3):
        A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
        Ai = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, cu_seqlens=None)
        h, vn, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None, initial_state=None, output_final_state=True, cu_seqlens=None)
        o = chunk_fwd_o(q=q, k=k, v=vn, h=h, g=None, scale=scale, cu_seqlens=None)
torch.cuda.synchronize()

# profile
with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    with torch.no_grad():
        A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32)
        Ai = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, cu_seqlens=None)
        h, vn, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None, initial_state=None, output_final_state=True, cu_seqlens=None)
        o = chunk_fwd_o(q=q, k=k, v=vn, h=h, g=None, scale=scale, cu_seqlens=None)
    torch.cuda.synchronize()

print("\n" + "=" * 80)
print("CUDA Kernel Names (for ncu --kernel-name)")
print("=" * 80)
seen = set()
for evt in prof.key_averages():
    if evt.device_type == torch.autograd.DeviceType.CUDA and evt.key not in seen:
        seen.add(evt.key)
        print(f"  {evt.key}")
