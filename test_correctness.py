"""
test_correctness.py — 验证 DeltaNet 分 span 计算的数值一致性
新版 fla API: [B, T, H, D] 格式 (非 head-first)，beta 为 [B, T, H]

注意：DeltaNet 要求 k 做 L2 归一化，否则 state 会数值爆炸 → NaN。
实际训练中通过 use_qk_l2norm_in_kernel=True 在 kernel 内部完成。
测试时需要手动归一化或开启该选项。
"""
import torch
import torch.nn.functional as F
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

print("=" * 70)
print("验证 1：chunk_delta_rule 分 span vs 完整序列")
print("=" * 70)

torch.manual_seed(42)
B, T, H, D = 2, 8192, 16, 128

q = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)
# k 必须 L2 归一化，否则 state 更新会数值爆炸
k = F.normalize(torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16), p=2, dim=-1)
v = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)
beta = torch.rand(B, T, H, device='cuda', dtype=torch.bfloat16).sigmoid()

# 方法 A：一次性处理完整序列
o_full, state_full = chunk_delta_rule(q, k, v, beta, output_final_state=True)
print(f"Full output shape: {o_full.shape}")    # [B, T, H, D]
print(f"Full state shape:  {state_full.shape}") # [B, H, D, D]
print(f"Full output has NaN: {o_full.isnan().any().item()}")
print(f"Full state has NaN:  {state_full.isnan().any().item()}")

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
k2 = F.normalize(torch.randn(B2, T2, H2, D2, device='cuda', dtype=torch.bfloat16), p=2, dim=-1)
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
print("验证 3：use_qk_l2norm_in_kernel=True（与实际训练一致）")
print("=" * 70)

torch.manual_seed(123)
B3, T3, H3, D3 = 2, 4096, 8, 128

# 不手动归一化 k，让 kernel 内部做 L2 norm（与实际训练行为一致）
q3 = torch.randn(B3, T3, H3, D3, device='cuda', dtype=torch.bfloat16)
k3 = torch.randn(B3, T3, H3, D3, device='cuda', dtype=torch.bfloat16)  # 未归一化
v3 = torch.randn(B3, T3, H3, D3, device='cuda', dtype=torch.bfloat16)
beta3 = torch.rand(B3, T3, H3, device='cuda', dtype=torch.bfloat16).sigmoid()

o3_full, state3_full = chunk_delta_rule(
    q3, k3, v3, beta3, output_final_state=True,
    use_qk_l2norm_in_kernel=True  # kernel 内部 L2 归一化
)
has_nan = o3_full.isnan().any().item()
print(f"  use_qk_l2norm_in_kernel=True, output has NaN: {has_nan}")

# 分 span
for num_spans in [4, 8]:
    span_len = T3 // num_spans
    o_spans3, state3 = [], None
    for i in range(num_spans):
        s, e = i * span_len, (i + 1) * span_len
        o_span3, state3 = chunk_delta_rule(
            q3[:, s:e], k3[:, s:e], v3[:, s:e], beta3[:, s:e],
            initial_state=state3, output_final_state=True,
            use_qk_l2norm_in_kernel=True
        )
        o_spans3.append(o_span3)
    o3_split = torch.cat(o_spans3, dim=1)

    out_diff3 = (o3_full - o3_split).abs().max().item()
    state_diff3 = (state3_full - state3).abs().max().item()
    passed4 = out_diff3 < 0.01
    all_pass = all_pass and passed4
    print(f"  spans={num_spans:2d} | output max_diff={out_diff3:.2e} | "
          f"state max_diff={state_diff3:.2e} | {'PASS ✅' if passed4 else 'FAIL ❌'}")

print()
print("=" * 70)
if all_pass:
    print("所有验证通过 ✅ — DeltaNet 分 span 计算与完整序列数值一致")
else:
    print("存在验证失败 ❌ — 请检查具体项目")
print("=" * 70)
