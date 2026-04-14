# DeltaNet Kernel 优化分析

> Phase 2: 内层算子优化 — 精细调整 DeltaNet 每部分计算，让 TC/CC 更并行

## 1. Forward 流水线总览

```
chunk_delta_rule_fwd(q, k, v, beta):
  ┌─────────────────────────────────────────────────────┐
  │ Step 1: prepare_wy_repr_fwd(k, v, beta)            │
  │   🔥1 chunk_scaled_dot_kkt  → A_raw [BT×BT]       │  TC + CC
  │   🔥2 solve_tril            → A = (I+A_raw)^{-1}   │  CC dominant
  │   🔥3 recompute_w_u         → w [BT×K], u [BT×V]  │  TC + CC
  ├─────────────────────────────────────────────────────┤
  │ Step 2: chunk_gated_delta_rule_fwd_h(k, w, u)      │
  │   🔥4 chunk_fwd_h   → h [K×V], v_new [BT×V]       │  TC dominant, 串行
  ├─────────────────────────────────────────────────────┤
  │ Step 3: chunk_fwd_o(q, k, v_new, h)                │
  │   🔥5 chunk_fwd_o   → o [BT×V]                     │  TC dominant
  └─────────────────────────────────────────────────────┘
```

---

## 2. 逐 Kernel 深度分析

### 🔥1 `chunk_scaled_dot_kkt` — β·(K·K^T) 的下三角

**文件**: `fla/ops/common/chunk_scaled_dot_kkt.py`

**计算**:
```
for i_k in range(cdiv(K, BK)):
    b_A += tl.dot(b_k, tl.trans(b_k))    # [BT,BK]@[BK,BT] → [BT,BT]  ← TC
b_A *= b_b[:, None]                       # elementwise scale by β        ← CC
b_A = where(lower_tri, b_A, 0)           # mask                          ← CC
```

**维度** (1.3B, D=128, BT=64):
- K=128, BK∈{32,64,128} (autotune)
- 计算: `cdiv(128,BK)` 次 `tl.dot` = 64×BK × BK×64, 每次 2×64×BK×64 FLOPs
- 总 FLOPs/chunk: 2×64×128×64 = 1,048,576 ≈ 1M FLOPs
- HBM: 读 k [BT×K] = 64×128×2B = 16KB, 写 A [BT×BT] = 64×64×4B = 16KB

**Grid**: `(NT, B×H)` — 全部 chunks × heads 并行

**瓶颈**: 计算量小 (1M FLOPs/block)，TC 利用率低。β 的 elementwise 乘和 mask 是 CC。问题不大因为量小。

**优化价值**: ⭐ 低 — 计算量太小，单独优化收益有限

---

### 🔥2 `solve_tril` — (I+A)^{-1} 三角求逆

**文件**: `fla/ops/utils/solve_tril.py`

**计算** (BT=64 → `merge_16x16_to_64x64_inverse_kernel`):
```
# Phase 1: 4 个 16×16 对角块各自标量 forward substitution
for i in range(2, 16):                    # 串行 14 步
    b_a = -tl.load(...)                    # scalar load    ← CC
    b_a += tl.sum(b_a[:, None] * b_Ai, 0) # [16] reduction ← CC
    b_Ai = tl.where(mask, b_a, b_Ai)      # update         ← CC

# Phase 2: 通过 tl.dot 合并 off-diagonal blocks
b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21), b_Ai_11)  # [16,16]@[16,16]@[16,16] ← TC
b_Ai_31 = -tl.dot(b_Ai_33, dot(b_A_31, b_Ai_11) + dot(b_A_32, b_Ai_21))  ← TC
... (共 10 个 16×16 块的 dot)
```

**Grid**: `(NT, B×H)` — 全部 chunks × heads 并行

**特点**:
- Phase 1 是纯 CC，14-step 串行 for-loop，是 **最大的串行瓶颈**
- Phase 2 有约 10 次 16×16 的 `tl.dot`，但 16×16 太小无法充分利用 TC (A100 TC 最小 tile = 16×8×16)
- 总 FLOPs 小 (约 10 × 2×16^3 ≈ 80K FLOPs)

**瓶颈**: **串行数据依赖** — 每一行依赖前面所有行的结果。for-loop 无法并行化。

**优化价值**: ⭐⭐ 中低 — 绝对计算量小，但串行 latency 可能成为 pipeline bubble

---

### 🔥3 `recompute_w_u` — 用 A 重建 w, u

**文件**: `fla/ops/delta_rule/wy_fast.py`

**计算**:
```
b_A [BT×BT], b_beta [BT]

for i_v in range(cdiv(V, BV)):     # V 维度循环
    b_v = load(v, [BT, BV])
    b_vb = b_v * b_beta[:, None]   # elementwise ← CC
    b_u = tl.dot(b_A, b_vb)       # [BT,BT]@[BT,BV] → [BT,BV] ← TC
    store(u, b_u)

for i_k in range(cdiv(K, BK)):     # K 维度循环
    b_k = load(k, [BT, BK])
    b_kb = b_k * b_beta[:, None]   # elementwise ← CC
    b_w = tl.dot(b_A, b_kb)       # [BT,BT]@[BT,BK] → [BT,BK] ← TC
    store(w, b_w)
```

**维度** (D=128, BV=BK=64):
- V 循环: cdiv(128,64)=2 次 dot, 每次 2×64×64×64 = 524K FLOPs
- K 循环: cdiv(128,64)=2 次 dot, 每次 524K FLOPs
- 总: 4 × 524K ≈ 2.1M FLOPs/chunk

**Grid**: `(NT, B×H)` — 全部 chunks × heads 并行

**特点**: TC 和 CC 交替 — 先 elementwise 乘 β，再 dot。`allow_tf32=False` 强制 FP32 精度。

**瓶颈**: `allow_tf32=False` 使 TC 降速 (FP32 dot on A100 ≈ 19.5 TFLOPS vs BF16 312 TFLOPS)

**优化价值**: ⭐⭐ 中 — `allow_tf32=False` 是精度需求，但可考虑混合精度策略

---

### 🔥4 `chunk_fwd_h` — 跨 chunk 的状态递推 S_t

**文件**: `fla/ops/common/chunk_delta_h.py`

**计算**:
```
b_h [64×BV] = initial_state           # 状态 h ∈ R^{K×V}

for i_t in range(NT):                 # ⚠️ 串行 over chunks!
    store(h[i_t], b_h)                # 先存当前状态

    # v_new = v - w @ h (intra-chunk correction)
    for k_block in range(cdiv(K, 64)):
        b_w = load(w, [BT, 64])
        b_v -= tl.dot(b_w, b_h)      # [BT,64]@[64,BV] → [BT,BV] ← TC

    # h += k^T @ v_new (state update)
    for k_block in range(cdiv(K, 64)):
        b_k = load(k, [64, BT])
        b_h += tl.dot(b_k, b_v)      # [64,BT]@[BT,BV] → [64,BV] ← TC
```

**维度** (D=128, BV∈{32,64}):
- 每个 chunk: 2×(2×64×BT×BV) for w@h + 2×(2×64×BT×BV) for k^T@v = 4×524K ≈ 2.1M FLOPs (BV=64)
- NT 个 chunk 串行

**Grid**: `(cdiv(V, BV), N×H)` — 只沿 V 维度和 batch×head 并行

**特点**:
- **这是整个 pipeline 唯一串行的部分** — for i_t in range(NT) 必须顺序执行
- 状态 b_h [64×BV] 在寄存器中累积，跨 chunk 传递
- 对于 seq=32K, BT=64: NT=512，串行 512 个 chunk
- TC 利用率好 (64×64×BV 或 64×BT×BV 的 dot)

**瓶颈**: 
1. **串行依赖** — NT 步递推无法并行
2. **寄存器压力** — b_h 占 K×BV×4B 寄存器 (64×64×4B = 16KB per block if BV=64, 但 K 被拆成多个 64 块，每个 CTA 只处理一块)
3. **HBM 带宽** — 每个 chunk 需读 w [BT×K] + k [K×BT] + v [BT×BV]，写 h [K×BV] + v_new [BT×BV]

**优化价值**: ⭐⭐⭐⭐⭐ 最高 — Amdahl 定律，串行部分是最大瓶颈

---

### 🔥5 `chunk_fwd_o` — 输出计算 o = q·h + causal(q·k^T)·v

**文件**: `fla/ops/common/chunk_o.py`

**计算**:
```
b_o = zeros([BT, BV])
b_A = zeros([BT, BT])

for i_k in range(cdiv(K, BK)):
    b_q = load(q, [BT, BK])
    b_k = load(k, [BK, BT])
    b_h = load(h, [BK, BV])
    
    b_o += tl.dot(b_q, b_h)    # [BT,BK]@[BK,BV] → [BT,BV]  ← TC (inter-chunk)
    b_A += tl.dot(b_q, b_k)    # [BT,BK]@[BK,BT] → [BT,BT]  ← TC (intra-chunk attn)

b_A = where(causal_mask, b_A, 0)
b_o = b_o * scale + tl.dot(b_A, b_v) * scale   # [BT,BT]@[BT,BV] ← TC + CC
```

**维度** (D=128, BK=BV∈{64,128}):
- K 循环: cdiv(128,BK) 次，每次 2 个 dot
  - q@h: 2×BT×BK×BV FLOPs
  - q@k^T: 2×BT×BK×BT FLOPs
- 最后 A@v: 2×BT×BT×BV FLOPs
- BK=BV=128 时总: 2×64×128×128 + 2×64×128×64 + 2×64×64×128 ≈ 3.1M FLOPs

**Grid**: `(cdiv(V, BV), NT, B×H)` — 三维全并行

**特点**:
- 完全并行化 — 每个 (chunk, head, V_block) 独立
- 几乎纯 TC 计算
- 依赖 🔥4 的输出 h

**优化价值**: ⭐⭐ 中 — 已经高度并行，autotune 效果好

---

## 3. 实测耗时数据

### 3.1 实测结果 (B=2, T=8192, H=32, D=128, A100)

```
python profile_nsys_ncu.py --mode torch --T 8192
```

| Kernel | CUDA 耗时 | 占比 | 每次调用 |
|--------|----------|------|---------|
| 🔥4 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 3.260ms | **27.93%** | 652μs |
| 🔥2 `merge_16x16_to_64x64_inverse_kernel` | 2.840ms | **24.33%** | 568μs |
| 🔥5 `chunk_fwd_kernel_o` | 2.837ms | **24.30%** | 567μs |
| 🔥3 `recompute_w_u_fwd_kernel` | 1.796ms | **15.39%** | 359μs |
| 🔥1 `chunk_scaled_dot_kkt_fwd_kernel` | 0.747ms | **6.40%** | 149μs |
| 其他 (zeros/fill) | 0.384ms | 1.65% | — |
| **总计** | **11.673ms** | | |

### 3.2 Latency Scaling (B=1, H=32, D=128)

```
T= 1024 (NT= 16) | fwd = 0.62 ms
T= 2048 (NT= 32) | fwd = 0.63 ms  ← GPU 未跑满
T= 4096 (NT= 64) | fwd = 0.69 ms
T= 8192 (NT=128) | fwd = 1.32 ms  ← 开始线性增长
T=16384 (NT=256) | fwd = 2.58 ms  ← ~2×
T=32768 (NT=512) | fwd = 5.16 ms  ← ~2×, 线性确认
```

从 T=8192 开始，每翻倍 T → latency 翻倍，确认 🔥4 串行递推是随 T 线性增长的主导项。

### 3.3 关键发现

**🔥2 solve_tril 占了 24%，远超理论预估 (~8%)！**

原因分析：
- FLOPs 极小 (~80K/chunk)，但 14-step 串行 for-loop 每步有数据依赖
- 16×16 的 `tl.dot` 太小，无法充分利用 TC (A100 TC 最小高效 tile = 16×8×16)
- 每个 chunk 的 solve 虽然互相并行 (Grid = NT×B×H)，但单 CTA 内部是 **latency bound**
- `merge_16x16_to_64x64` 需要 4 个 16×16 块各自串行求逆，再用 10 次 dot 合并

**前三大瓶颈几乎三足鼎立：🔥4 (28%) ≈ 🔥2 (24%) ≈ 🔥5 (24%)**

> 注：在 T=32K 时，🔥4 占比会上升（因为串行步数 NT=512），🔥2/🔥5 占比相对下降。
> 需要在 T=32K 条件下再做一次 profile 确认。

---

## 4. 优化策略建议

### 优先级 P0: 🔥2 solve_tril + 🔥4 chunk_fwd_h (共占 52%)

#### P0a: 🔥2 solve_tril (24%) — 意外的大瓶颈

**问题**: FLOPs 很小但耗时很高，纯 latency bound。
- 4 个 16×16 对角块各自 14-step 串行 for-loop
- 合并阶段 10 次 16×16 dot，太小无法充分利用 TC
- 每次调用 568μs，占总耗时 24%

**策略**:
1. **融合到 🔥1+🔥3 中** — 避免单独 kernel launch + HBM 读写 A
2. **考虑近似替代** — 截断 Neumann 级数 $(I+A)^{-1} \approx I - A + A^2 - A^3$
   只需几次矩阵乘，全部在 TC 上跑，但会引入近似误差
3. **增大子块** — 从 16×16 改为 32×32 基础块，减少合并层数

#### P0b: 🔥4 chunk_fwd_h (28%) — 串行递推

**问题**: NT=512 步串行，每步做 4 个 `tl.dot`，受寄存器大小限制只能在 1 个 CTA 上跑。
在 T=32K 时占比会进一步上升。

**策略 A: 增大 BT (chunk size)**
- 当前 BT=64 → 可尝试 BT=128 或 BT=256
- NT 减半/减四，串行步数减少
- 代价: 🔥2 solve_tril 复杂度从 O(BT^2) 变 O(BT^3)，寄存器压力增大
- **可行性**: ⭐⭐⭐ 需修改多处 BT 硬编码

**策略 B: Chunk 级并行递推 (Parallel Prefix Sum / Scan)**
- 将 h_t = decay(h_{t-1}) + k_t^T @ u_t 改写为 parallel scan
- DeltaNet 的 decay 矩阵是 (I - βkk^T)，不是简单标量 decay，parallel scan 需要矩阵乘法
- 状态 S ∈ R^{K×V}，scan 操作 = 矩阵乘，复杂度 O(K^2×V) per step
- **可行性**: ⭐⭐ 理论可行但工程难度大

**策略 C: 流水线化 w@h 和 k^T@v 的 TC 操作**
- 当前同一 CTA 内顺序做 `b_v -= dot(w, h)` 然后 `b_h += dot(k, v)`
- 利用 Triton 的 `num_stages` 和 software pipelining 重叠内存访问
- **可行性**: ⭐⭐⭐⭐ 通过 autotune num_stages 已部分实现

**策略 D: Warp Specialization (CUTLASS-style)**
- 将 w@h (修正) 和 k^T@v (更新) 分配给不同 warp
- 需要 warp-level 同步 (shared memory 传递 b_v)
- Triton 暂不支持显式 warp specialization
- **可行性**: ⭐ 需要手写 CUDA kernel

### 优先级 P1: 🔥1+🔥2+🔥3 融合 (共占 46%)

**当前**: 3 个独立 kernel launch，中间通过 HBM 传递 A [B,T,H,BT]
**实测**: 🔥1(6%) + 🔥2(24%) + 🔥3(15%) = 46%，接近一半！

**优化**: 将 kkt → solve → w_u 融合为 1 个 kernel
- kkt 生成 A 后直接在 SRAM 中做 solve，再直接做 w_u
- 省掉 A 写回 HBM 和重新读取的带宽开销
- A 的大小: B×T×H×BT×4B = 1×32K×32×64×4 ≈ 256MB (BT=64, fp32)

**实现思路**:
```python
@triton.jit
def fused_wy_repr_kernel(k, v, beta, w, u, ...):
    # Phase 1: compute A = β·kk^T (🔥1)
    b_A = zeros([BT, BT])
    for i_k in range(cdiv(K, BK)):
        b_k = load(k, [BT, BK])
        b_A += tl.dot(b_k, tl.trans(b_k))
    b_A *= b_beta[:, None]
    b_A = where(lower_tri, b_A, 0)
    
    # Phase 2: solve (I+A)^{-1} (🔥2) — 直接在寄存器/SRAM 中
    # ... forward substitution in registers ...
    
    # Phase 3: compute w, u (🔥3) — 用寄存器中的 A
    for i_v in range(cdiv(V, BV)):
        b_vb = load(v) * b_beta[:, None]
        b_u = tl.dot(b_A, b_vb)
        store(u, b_u)
    for i_k in range(cdiv(K, BK)):
        b_kb = load(k) * b_beta[:, None]
        b_w = tl.dot(b_A, b_kb)
        store(w, b_w)
```

**收益**: 
- 省掉 A 的 2× HBM 读写 (256MB × 2 = 512MB bandwidth)
- 减少 2 次 kernel launch overhead (约 5-10μs each)
- A [BT×BT] = 64×64×4B = 16KB 可以放在 shared memory

**可行性**: ⭐⭐⭐⭐ 高 — 推荐第一个动手

### 优先级 P2: 🔥4+🔥5 融合 (Inter-chunk overlap)

**思路**: chunk_fwd_h 产出 h[i_t] 后，chunk_fwd_o 可以立即使用 h[i_t] 计算 o[i_t]。

当前是分开的两个 kernel，🔥4 全部完成后才启动 🔥5。

**Producer-Consumer 模式**:
- 🔥4 的一个 CTA 串行生成 h[0], h[1], ..., h[NT-1]
- 🔥5 的多个 CTA 并行消费 h[i_t]
- 用 atomicAdd 或 global memory flag 做同步

**实现**:
```python
# 🔥4 CTA: 每生成一个 h[i_t]，signal flag
for i_t in range(NT):
    store(h[i_t], b_h)
    tl.atomic_xchg(ready_flag + i_t, 1)  # signal 🔥5
    # continue to compute h[i_t+1]...

# 🔥5 CTA: spin-wait on flag
while tl.atomic_cas(ready_flag + i_t, 1, 1) != 1:
    pass  # spin
b_h = load(h[i_t])
# compute output...
```

**挑战**: Triton 对 global memory synchronization 支持有限
**可行性**: ⭐⭐ 需要 CUDA 层面实现

### 优先级 P3: 精度策略优化

**🔥3 recompute_w_u 中 `allow_tf32=False`**:
- 当前强制 FP32 精度，TC 效率只有 BF16 的 1/16
- 考虑: 在 A 已经是 FP32 的情况下，dot(A, v_beta) 的精度需求是否真的需要 IEEE FP32?
- 替代: 用 `allow_tf32=True` 配合 FP32 accumulator (Triton 默认)
- 或者: 将 A cast 到 BF16 后再 dot (如果数值稳定)

---

## 5. 推荐行动路线 (基于实测数据修正)

```
Phase 2.1: ✅ Profile (已完成)
  └─ 实测 T=8192: 🔥4(28%) ≈ 🔥2(24%) ≈ 🔥5(24%) > 🔥3(15%) > 🔥1(6%)

Phase 2.1b: 补充 Profile — T=32K (1 天)
  └─ 在 T=32K 下重新 profile，确认 🔥4 在长序列下的占比上升趋势

Phase 2.2: 🔥1+🔥2+🔥3 Fusion (3-5 天) ← 最推荐先做
  ├─ 实现 fused_wy_repr_kernel (kkt → solve → w_u 融合)
  ├─ A [64×64] 留在 SRAM，省掉 HBM 读写
  ├─ 三者合占 46%，融合后预期省 30-50% 的这部分开销
  └─ 单元测试 + 端到端性能对比

Phase 2.3: 🔥4 加速 (1-2 周)
  ├─ 方案 A: 增大 BT (BT=128) → 减少串行步数 NT
  ├─ 方案 C: 优化 num_stages/num_warps autotune 范围
  └─ 方案 B: 研究 parallel scan 的可行性

Phase 2.4: Precision Tuning (2-3 天)
  ├─ 测试 🔥3 allow_tf32=True 对收敛的影响
  └─ 混合精度策略
```

---

## 6. 附录：A100 硬件参数快速参考

| 资源 | 数值 |
|------|------|
| SMs | 108 |
| Tensor Cores / SM | 4 (3rd gen) |
| CUDA Cores / SM | 64 (FP32) |
| BF16 TC 峰值 | 312 TFLOPS |
| FP32 TC (TF32) | 156 TFLOPS |
| FP32 CUDA Core | 19.5 TFLOPS |
| HBM2e 带宽 | 2039 GB/s |
| L2 Cache | 40 MB |
| Shared Memory / SM | 164 KB (最大配置) |
| 寄存器文件 / SM | 256 KB |
