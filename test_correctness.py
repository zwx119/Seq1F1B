"""
test_correctness.py — 验证 DeltaNet 分 span 计算的数值一致性
新版 fla API: [B, T, H, D] 格式 (非 head-first)，beta 为 [B, T, H]
"""
import torch
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

print("=" * 70)
print("验证 1：chunk_delta_rule 分 span vs 完整序列")
print("=" * 70)

torch.manual_seed(42)
B, T, H, D = 2, 8192, 16, 128

q = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)
beta = torch.rand(B, T, H, device='cuda', dtype=torch.bfloat16).sigmoid()

# 方法 A：一次性处理完整序列
o_full, state_full = chunk_delta_rule(q, k, v, beta, output_final_state=True)
print(f"Full output shape: {o_full.shape}")    # [B, T, H, D]
print(f"Full state shape:  {state_full.shape}") # [B, H, D, D]

# 方法 B：分成 N 个 span，逐个传递 state
all_pass = True
for num_spans in [2, 4, 8, 16]:
    span_len = T // num_spans
    o_spans, state = [], None
    for i in range(num_spans):
        s, e = i * span_len, (i + 1) * span_len
        o_span, state = chunk_delta_rule(
            q[:, s:e], k[:, s:e], v[:, s:e], beta[:, s:e],
            initial_state=state, output_final_state=True
        )
        o_spans.append(o_span)
    o_split = torch.cat(o_spans, dim=1)  # 沿 T 维度拼接

    out_diff = (o_full - o_split).abs().max().item()
    state_diff = (state_full - state).abs().max().item()
    passed = out_diff < 0.01
    all_pass = all_pass and passed
    print(f"  spans={num_spans:2d} | output max_diff={out_diff:.2e} | "
          f"state max_diff={state_diff:.2e} | {'PASS ✅' if passed else 'FAIL ❌'}")

print()
print("=" * 70)
print("验证 2：chunk vs fused_recurrent state 一致性")
print("=" * 70)

torch.manual_seed(42)
B2, T2, H2, D2 = 1, 2048, 4, 64

q2 = torch.randn(B2, T2, H2, D2, device='cuda', dtype=torch.bfloat16)
k2 = torch.randn(B2, T2, H2, D2, device='cuda', dtype=torch.bfloat16)
v2 = torch.randn(B2, T2, H2, D2, device='cuda', dtype=torch.bfloat16)
beta2 = torch.rand(B2, T2, H2, device='cuda', dtype=torch.bfloat16).sigmoid()

# chunk_delta_rule 的 final state
_, state_chunk = chunk_delta_rule(q2, k2, v2, beta2, output_final_state=True)
# fused_recurrent_delta_rule 的 final state
_, state_recurrent = fused_recurrent_delta_rule(q2, k2, v2, beta2, output_final_state=True)

diff = (state_chunk - state_recurrent).abs().max().item()
passed2 = diff < 0.05
all_pass = all_pass and passed2
print(f"  chunk vs recurrent state max_diff = {diff:.2e} | "
      f"{'PASS ✅' if passed2 else 'FAIL ❌'}")
print(f"  State shape: {state_chunk.shape}")
print(f"  State non-zero ratio: {(state_chunk.abs() > 1e-6).float().mean():.2%}")

# 分半验证
mid = T2 // 2
_, state_half_recurrent = fused_recurrent_delta_rule(
    q2[:, :mid], k2[:, :mid], v2[:, :mid], beta2[:, :mid],
    output_final_state=True
)
_, state_half_chunk = chunk_delta_rule(
    q2[:, :mid], k2[:, :mid], v2[:, :mid], beta2[:, :mid],
    output_final_state=True
)
diff_half = (state_half_recurrent - state_half_chunk).abs().max().item()
passed3 = diff_half < 0.05
all_pass = all_pass and passed3
print(f"  half-seq chunk vs recurrent state max_diff = {diff_half:.2e} | "
      f"{'PASS ✅' if passed3 else 'FAIL ❌'}")

print()
print("=" * 70)
if all_pass:
    print("所有验证通过 ✅ — DeltaNet 分 span 计算与完整序列数值一致")
else:
    print("存在验证失败 ❌ — 请检查具体项目")
print("=" * 70)
