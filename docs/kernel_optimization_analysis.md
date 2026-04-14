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

## 4. 优化策略（按优先级排列）

> 目标硬件：A100-80GB (108 SMs)、H100 (132 SMs, warp specialization, TMA)、H20
> 实测占比 (T=8K): 🔥4(28%) ≈ 🔥2(24%) ≈ 🔥5(24%) > 🔥3(15%) > 🔥1(6%)

### P0：🔥1+🔥2+🔥3 融合 — 最大收益 (合占 46%)

**现状**: 3 个独立 kernel，中间通过 HBM 传递 A `[B,T,H,BT]`

```
🔥1: k,β → A_raw [BT×BT]  ──HBM写──→  🔥2: A_raw → A_inv [BT×BT]  ──HBM写──→  🔥3: A_inv,k,v,β → w,u
                            ←──HBM读──                               ←──HBM读──
```

**融合后**: A 始终在寄存器/SRAM 中，省掉所有中间 HBM 往返

```
fused_wy_repr_kernel: k,v,β → w,u   (A 在寄存器中生产+消费，不写 HBM)
```

**收益分析**:
- **省掉 A 的 4× HBM 传输**: A 大小 = B×NT×H×BT×BT×4B
  - T=8K, B=2, H=32: 2×128×32×64×64×4 = **2 GB** 总带宽节省
  - T=32K, B=1, H=32: 1×512×32×64×64×4 = **4 GB** 总带宽节省
- **减少 2 次 kernel launch overhead**: ~5-10μs each
- **🔥2 的 CC 串行延迟被隐藏**: 编译器可在 🔥2 串行求逆期间 prefetch 🔥3 需要的 v, k 数据
- A `[64×64]` FP32 = 16KB，放寄存器可行（4096 个 float32 寄存器）

**预期提升**: 当前 46% 的部分节省 30-50% → **总体加速 14-23%**

**实现思路**:
```python
@triton.jit
def fused_wy_repr_kernel(k, v, beta, w, u, ...):
    # ═══ Phase 1: 🔥1 — 计算 A_raw = β·(k·k^T) 的下三角 ═══
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        b_k = tl.load(p_k, ...)                   # [BT, BK] from HBM
        b_A += tl.dot(b_k, tl.trans(b_k))         # TC
    b_A *= b_beta[:, None]                         # CC
    b_A = tl.where(lower_tri_mask, b_A, 0)         # CC
    # 🔑 A_raw 在寄存器中，不写回 HBM！

    # ═══ Phase 2: 🔥2 — 原地求逆 (I+A)^{-1} ═══
    # 4 个 16×16 对角块各自 forward substitution (CC, 14-step 串行)
    # 然后用 ~10 次 16×16 dot 合并成 64×64 (TC, 小 tile)
    # 🔑 A_inv 还是在寄存器中！

    # ═══ Phase 3: 🔥3 — 计算 w, u ═══
    for i_v in range(tl.cdiv(V, BV)):
        b_v = tl.load(p_v, ...)                    # [BT, BV] from HBM
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)  # CC
        b_u = tl.dot(b_A_inv, b_vb, allow_tf32=False) # TC/CC
        tl.store(p_u, b_u, ...)                     # to HBM
    for i_k in range(tl.cdiv(K, BK)):
        b_k = tl.load(p_k, ...)                    # [BT, BK] from HBM
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)  # CC
        b_w = tl.dot(b_A_inv, b_kb, allow_tf32=False) # TC/CC
        tl.store(p_w, b_w, ...)                     # to HBM
```

**Grid**: `(NT, B×H)` — 与原 🔥1/🔥2/🔥3 相同

**寄存器压力**:
- b_A / b_A_inv: [64×64] FP32 = 4096 reg (或拆成 4 个 [16×16] 对角块 + 6 个 off-diagonal)
- b_k tile: [64×BK] = 最多 4096 reg
- b_v tile: [64×BV] = 最多 4096 reg
- A100: 65536 regs/SM, 4 warps(128 threads) → 512 regs/thread，需谨慎调配

**实施步骤**:
1. 先实现融合 kernel，`allow_tf32` 设置与原 🔥3 保持一致 (`False`)
2. 用 `prepare_wy_repr_fwd` 的原始输出做 correctness 对比
3. ncu profile 检查 register spill 和 SM occupancy
4. autotune num_warps, num_stages, BK

---

### P1：🔥3 精度优化 `allow_tf32=True`

**现状**: `recompute_w_u_fwd_kernel` 中 `tl.dot(b_A, b_vb, allow_tf32=False)`
- 强制 FP32 精度（CUDA Core），A100 上仅 19.5 TFLOPS
- 如果改为 `allow_tf32=True`，可用 TF32 Tensor Core (156 TFLOPS)，**8× 理论加速**

**风险**: TF32 有效尾数 10 bit (vs FP32 的 23 bit)
- A 矩阵是 `(I + β·KK^T)^{-1}` 的逆，数值敏感
- 需要端到端训练测试验证收敛性

**实施**: 改一行代码 + 跑收敛测试
- 如果 🔥1+🔥2+🔥3 已融合，直接在融合 kernel 中测试
- 可作为 autotune 参数之一：`DOT_PRECISION: tl.constexpr` → 让 Triton 自动选

---

### P2：🔥4 chunk_fwd_h 提升占用率 (28%)

**问题**:
- Grid = `(ceil(V/BV), B×H)` — CTA 数不含 NT，NT 是 CTA 内串行 for-loop
- 小模型 (H=16) CTA 数严重不足：

| 模型 | H | V=D | BV=32 CTA数 | BV=64 CTA数 | vs 108 SMs |
|------|---|-----|------------|------------|------------|
| 1.3B | 16 | 128 | 64 | 32 | ❌ 都不够 |
| 2.7B | 32 | 80 | 96 | 64 | ❌ 都不够 |
| 7B | 32 | 128 | 128 | 64 | BV=32 ✅ |
| 13B | 40 | 128 | 160 | 80 | BV=32 ✅ |
| 30B | 64 | 96 | 192 | 128 | 两者 ✅ |

> BV 在 A100 上由 `@triton.autotune` 从 [32, 64] 中自动选择
> （`check_shared_mem('ada')` = True on A100，所以 BV 候选 = [32, 64]）
> BV 设置位置: `fla/ops/common/chunk_delta_h.py` 第 29 行

**策略 A: 增大 BT (64→128)**
- NT 减半，串行步数减少
- 代价: 🔥2 solve_tril 从 O(BT²) → O(BT³)，寄存器压力 ×4
- 可行性: ⭐⭐⭐ 需修改多处 BT=64 硬编码

**策略 B: split-K**
- 把 K 维度拆到 Grid 上: Grid = `(ceil(V/BV), ceil(K/64), B×H)`
- 每个 CTA 只负责一个 K-block 的累积
- 需要 atomic 或 reduction 合并结果
- 可行性: ⭐⭐ 增加同步开销

**策略 C: 🔥4+🔥5 Producer-Consumer 融合** (见下文 P3)

---

### P3：🔥4+🔥5 融合 — Producer-Consumer 模式 (合占 52%)

**动机**: 🔥4 串行产出 h[0], h[1], ..., h[NT-1]，🔥5 等 🔥4 全部完成才开始。
实际上 🔥5 处理 chunk i_t 只需要 `h[i_t]` 和 `v_new[i_t]`，不需要等后面的 chunk。

**数据依赖分析**:
```
🔥4 CTA (串行):  h[0] → h[1] → h[2] → ... → h[NT-1]
                   ↓      ↓      ↓              ↓
🔥5 CTA (并行):  o[0]   o[1]   o[2]   ...    o[NT-1]
```

🔥5 处理 chunk t 需要: `q[t]`, `k[t]`, `v_new[t]`, `h[t]` — 前三个已知，`h[t]` 由🔥4 逐步产出。

**实现方案 1: Global Memory Flag (Triton 可行)**
```python
# 🔥4 CTA: 产出 h[i_t] 后 signal
for i_t in range(NT):
    tl.store(h[i_t], b_h)
    tl.store(v_new[i_t], b_v_new)
    tl.debug_barrier()                         # 确保写入完成
    tl.atomic_xchg(ready_flag + i_t, 1)        # signal: h[i_t] ready

# 🔥5 CTA: spin-wait on flag
while tl.load(ready_flag + my_i_t) == 0:
    pass  # spin
b_h = tl.load(h[my_i_t])
# ... 计算 o[my_i_t] ...
```

**实现方案 2: Warp Specialization (H100/H20, CUDA)**
```
CTA 内部:
  Warp Group 0 (producer): 运行🔥4 递推，产出 h → shared memory
  Warp Group 1 (consumer): 运行🔥5 输出计算，从 shared memory 读 h

  用 shared memory barrier / arrive-wait 同步
```

H100 的 warp specialization 支持让 producer/consumer warp 在同一 CTA 内高效协作。
**这是 H100 相比 A100 的核心架构优势之一。**

**收益**:
- 🔥4 和 🔥5 的延迟完全重叠（理想情况下 max(🔥4, 🔥5) 而非 🔥4 + 🔥5）
- 减少 1 次 kernel launch
- h 不需要写 HBM → 直接在 SRAM 中传递（省 h 的 HBM 带宽）
  - h 大小 = B×NT×H×K×V×2B = 2×128×32×128×128×2 ≈ **4 GB** for T=8K

**挑战**:
- 方案 1: GPU 上 spin-wait 浪费算力，且 Triton 的 atomic 支持有限
- 方案 2: 需要手写 CUDA kernel（CUTLASS 3.x 风格），工程量大
- 🔥5 的 CTA 远多于 🔥4（🔥5 有 NT 维度在 Grid 中），调度不匹配

**可行性**: ⭐⭐ (A100) / ⭐⭐⭐⭐ (H100/H20 with warp specialization)

---

### P4：🔥2 算法改进（如果不走融合路线的备选）

**Neumann 级数近似**:
$(I+A)^{-1} \approx I - A + A^2 - A^3 + ...$
- 只需几次 [64×64] 矩阵乘，全部跑 TC
- 截断到 k 阶: k 次 `tl.dot`，复杂度 O(k×BT³)
- 但 A 的谱半径可能 >1，级数不保证收敛 → **数值风险**

**分块 LU 分解**:
- 将 64×64 分成 2 个 32×32 块，递归求逆
- 减少串行步数（14→6），但增加矩阵乘次数
- 可行性: ⭐⭐⭐ 如果融合方案寄存器不够，这是降级方案

---

### P5：全局 BT (chunk_size) 调优

当前 BT=64 硬编码在多处。增大 BT 的影响:

| BT | NT (T=32K) | 🔥4 串行步数 | 🔥2 复杂度 | A 的大小 | 寄存器压力 |
|----|-----------|------------|----------|---------|----------|
| 64 | 512 | 512 | O(64³) ≈ 262K | 16 KB | 适中 |
| 128 | 256 | 256 (↓50%) | O(128³) ≈ 2M (↑8×) | 64 KB | 极高 |
| 256 | 128 | 128 (↓75%) | O(256³) ≈ 16M (↑64×) | 256 KB | 不可行 |

BT=128 是潜在 sweet spot：🔥4 加速 2× 但 🔥2 增 8×。
**如果 🔥1+🔥2+🔥3 已融合且 🔥2 延迟被隐藏，BT=128 更可行。**

---

## 5. 推荐行动路线

```
Phase 2.1: ✅ Profile (已完成)
  └─ 实测数据: 🔥4(28%) ≈ 🔥2(24%) ≈ 🔥5(24%) > 🔥3(15%) > 🔥1(6%)
  └─ T=32K 补充 profile 已完成: 占比稳定，全部线性 scaling

Phase 2.2: 🔥1+🔥2+🔥3 Fusion (3-5 天) ← 最推荐先做，最大收益
  ├─ 实现 fused_wy_repr_kernel (kkt → solve → w_u 单 kernel)
  ├─ A [64×64] 留在寄存器/SRAM，省掉 HBM 读写
  ├─ 三者合占 46%，融合后预期省 30-50% → 总体加速 14-23%
  ├─ 在融合 kernel 中同步测试 allow_tf32=True (P1)
  └─ 正确性测试 + ncu profile

Phase 2.3: 🔥4 提升占用率 (1 周)
  ├─ 优化 BV autotune 策略（强制 BV=32 for 小 H 模型）
  ├─ 测试 BT=128 对端到端的影响
  └─ A100 + H100/H20 上分别测试

Phase 2.4: 🔥4+🔥5 Producer-Consumer 融合 (2-3 周)
  ├─ 先在 Triton 中用 global memory flag 做原型
  ├─ 如果效果好，在 H100 上用 warp specialization 做高效实现
  └─ 省掉 h 的 HBM 往返 (4GB @T=8K)

Phase 2.5: 端到端训练验证 (持续)
  ├─ 确认融合 kernel 的数值精度
  ├─ allow_tf32 收敛性验证
  └─ 在 Seq1F1B pipeline 中做 throughput 测试
```

---

## 6. 附录：硬件参数快速参考

### A100-80GB SXM

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
| 寄存器文件 / SM | 256 KB (65536 × 32-bit) |
| Warp Specialization | ❌ 不支持 |
| TMA | ❌ 不支持 |

### H100-80GB SXM

| 资源 | 数值 |
|------|------|
| SMs | 132 |
| Tensor Cores / SM | 4 (4th gen) |
| BF16 TC 峰值 | 990 TFLOPS |
| FP32 TC (TF32) | 495 TFLOPS |
| FP32 CUDA Core | 67 TFLOPS |
| HBM3 带宽 | 3350 GB/s |
| L2 Cache | 50 MB |
| Shared Memory / SM | 228 KB (最大配置) |
| 寄存器文件 / SM | 256 KB (65536 × 32-bit) |
| Warp Specialization | ✅ **支持** — producer/consumer warp 协作 |
| TMA (Tensor Memory Accelerator) | ✅ **支持** — 异步块拷贝 |

### H20

| 资源 | 数值 |
|------|------|
| SMs | 132 |
| BF16 TC 峰值 | 148 TFLOPS |
| FP32 TC (TF32) | 74 TFLOPS |
| HBM3 带宽 | 4000 GB/s |
| L2 Cache | 60 MB |
| Shared Memory / SM | 228 KB |
| Warp Specialization | ✅ 支持 (Hopper 架构) |
| TMA | ✅ 支持 |
| 特点 | 算力弱但带宽极高 → **更受 compute bound 影响** |

### 🔥4+🔥5 融合在不同硬件上的策略选择

| 方案 | A100 | H100 | H20 |
|------|------|------|-----|
| Global Memory Flag (Triton) | ⭐⭐ spin-wait 浪费 SM | ⭐⭐ 同上 | ⭐⭐ 同上 |
| Warp Specialization (CUDA) | ❌ 不支持 | ⭐⭐⭐⭐ **最佳** | ⭐⭐⭐⭐ 支持 |
| Persistent Kernel + TMA | ❌ | ⭐⭐⭐⭐ TMA 异步加载 | ⭐⭐⭐⭐ |
| 分开执行 (现状) | 🔵 baseline | 🔵 baseline | 🔵 baseline |
