"""
Minimal test: can ncu profile a Triton kernel at all?

Usage:
  # Step 1: warm up Triton cache
  python test_ncu_triton.py

  # Step 2: run under ncu
  ncu --replay-mode kernel --set basic --launch-count 1 \
      python test_ncu_triton.py
"""
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 4096
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
out = torch.empty(n, device='cuda')

# Warmup (triggers JIT)
add_kernel[(n // 1024,)](x, y, out, n)
torch.cuda.synchronize()
print("Warmup done.")

# Target launch
add_kernel[(n // 1024,)](x, y, out, n)
torch.cuda.synchronize()
print(f"Result check: {torch.allclose(out, x + y)}")
print("Done!")
