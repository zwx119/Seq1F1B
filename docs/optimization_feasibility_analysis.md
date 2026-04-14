# DeltaNet 内核优化可行性分析

> 分析人: zwx119 | GPU: A100-SXM4-80GB (108 SMs)
> 配置: B=1, T=8192, H=32, K=D=128, V=128, BT=64

---

## 目录
1. [问题1: 🔥4 SM 用不满时能否提前插入🔥5？](#q1)
2. [问题1b: BV=32 是否更好？BV 受寄存器/SRAM 限制吗？](#q1b)
3. [问题2: 🔥2 的 CC 串行能否和 TC 运算并行？](#q2)
4. [总结与优先级排序](#summary)

---

<a id="q1"></a>
## 问题1: 🔥4 SM 用不满时能否提前插入🔥5 的计算？

### 1.1 数据依赖分析

**🔥4 (chunk_delta_h)** 的输出:
- `h[B, NT, H, K, V]` — 每个 chunk 的隐状态，写入 HBM
- `v_new[B, T, H, V]` — delta 修正后的 value，写入 HBM

**🔥5 (chunk_o)** 的输入:
- `h[B, NT, H, K, V]` — **直接来自 🔥4 的输出**
- `v` — 原始 value（不是 v_new，chunk_o 中直接读原始 v 用于 `A @ v`）
- `q, k` — 原始 query/key

**关键依赖链**:
```
🔥4 writes h[i_t] ──→ 🔥5 reads h[i_t]   (严格 RAW 依赖)
```

但注意 🔥4 的 `for i_t in range(NT)` 是**串行循环**，每步都会 `tl.store(p_h, b_h)` 写出该 chunk 的 h。而 🔥5 的 grid 是 `(ceil(V/BV), NT, B*H)` — NT 维度是**完全并行**的。

### 1.2 理论上的 Overlap 方案

**方案 A: CUDA Stream Overlap（不同 kernel 并行）**
```
Stream 1: 🔥4 整体运行 (64 CTAs, 占 ~0.6 CTA/SM)
Stream 2: 🔥5 在 🔥4 完成后立即启动
```
- ❌ 问题: 🔥4 必须**全部完成**（所有 NT=128 步循环跑完）才能保证所有 h[i_t] 都写入 HBM
- 🔥4 只有 64 个 CTA 但**每个 CTA 串行处理 128 个 chunk**，所以不是 "SM 空闲"——是 "SM 数量少，但每个 SM 上的 CTA 在忙"

**方案 B: Producer-Consumer Fusion（fused kernel）**
```
单个 kernel 内:
  CTA group 1 (producer): 计算 h[i_t]，写入 shared/global memory
  CTA group 2 (consumer): 等待 h[i_t] 就绪后计算 o[i_t]
```
- ⚠️ 需要 **Cooperative Groups** 或 **TMA + barrier** 跨 CTA 同步
- Triton 目前（2024）**不原生支持** 跨 CTA 同步和 persistent kernel 模式
- 需要手写 CUDA kernel 或用 Triton descriptor + 自定义同步

**方案 C: Pipeline Overlap via chunk 粒度**
```
🔥4 内部每完成一个 chunk i_t 就 store h[i_t] → 
🔥5 的 CTA (i_v, i_t, i_bh) 只要 h[i_t] ready 就能开始
```
- 理论最优，但实现上要求:
  1. 🔥4 和 🔥5 在同一个 persistent kernel 中
  2. 🔥4 CTA 写完 h[i_t] 后通过 flag/semaphore 通知 🔥5 CTA
  3. 🔥5 CTA spin-wait 直到 flag 变为 ready

### 1.3 为什么目前无法简单 overlap？

| 因素 | 说明 |
|------|------|
| **🔥4 的 64 个 CTA 不是 "空闲"** | 每个 CTA 内部串行跑 NT=128 步，全程在计算 dot product + 访存，SM 不空闲，只是**CTA 数量少**导致部分 SM 没分配到 CTA |
| **CUDA stream 无法利用** | 两个 kernel 在不同 stream 上并行需要 GPU 有空余 SM。🔥4 只用了 ~64 个 CTA (4 warps/CTA = 128 线程/CTA)，最多占 64 个 SM（每 SM 1 CTA），剩 44 个 SM 空闲——理论上可以塞 🔥5 |
| **但 HBM 依赖阻塞** | 🔥5 的 CTA(i_v, i_t=0, i_bh) 需要 h[i_t=0]，而 h[i_t=0] 只在 🔥4 的 CTA 跑第 0 步循环时写出。时间点上 🔥4 第 0 步几乎立即完成 → 🔥5 的 i_t=0 CTA 可以很快开始，但需要同步机制 |

### 1.4 Stream-Level Overlap 的可行性估算

```
🔥4: 64 CTAs, 4 warps/CTA = 128 threads/CTA
     每 SM 最多放 2048/128 = 16 CTAs (线程数上限)
     但寄存器压力: BV=64, K=128 → 2×[64,64]×4B = 32KB 状态
     假设每 CTA 需要 ~128 regs/thread → 128×128 = 16384 regs/CTA
     每 SM 65536 regs → 最多 4 CTAs/SM
     64 CTAs 分配到 108 SMs → ~0.6 CTA/SM → 大部分 SM 只有 0-1 CTA
```

如果 🔥4 平均每 SM 分 0.6 CTA，意味着：
- ~44 个 SM 可能完全空闲（0 个 🔥4 CTA）
- ~64 个 SM 各有 1 个 🔥4 CTA

**结论**: 用 CUDA stream 并行启动 🔥5 到空闲 SM 上是**理论可行**的，但：
1. 需要用 `cudaStreamCreate` + 异步 launch，Triton 的 autotuner launch 不容易控制 stream
2. 🔥5 有 4096 CTAs，会立刻涌入所有 108 SMs，和 🔥4 的 CTA 抢资源
3. 真正的 bottleneck 是 🔥4 第 i_t 步完成 → 🔥5 的 (*, i_t, *) CTA 才能开始的**数据依赖**

### 1.5 结论

| 方案 | 可行性 | 难度 | 预期收益 |
|------|--------|------|----------|
| Stream overlap（两个独立 kernel） | ⚠️ 理论可行但依赖同步困难 | 中 | 最多 ~10-15%（受 h 依赖限制） |
| Producer-consumer fusion | ✅ 最优但需 CUDA | 极高 | ~20-30%（消除 🔥4 的 HBM write + 🔥5 的 HBM read of h） |
| Triton persistent kernel | ❌ Triton 不支持 | - | - |

**建议**: 短期内不推荐。优先做 🔥1+🔥2+🔥3 fusion（三者都是 grid=(128,32)，可以 fuse）。

---

<a id="q1b"></a>
## 问题1b: BV=32 vs BV=64 的 Trade-off

### 2.1 BV 对 🔥4 的直接影响

从 `chunk_delta_h.py` 的 autotune 配置：
```python
triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
for num_warps in [2, 4]
for num_stages in [2, 3, 4]
for BV in [32, 64]  # 在 Ampere 上
```

| 指标 | BV=32 | BV=64 |
|------|-------|-------|
| **Grid dim 0** | ceil(128/32) = 4 | ceil(128/64) = 2 |
| **总 CTA 数** | 4 × 32 = 128 | 2 × 32 = 64 |
| **CTA/SM** | 128/108 ≈ 1.19 | 64/108 ≈ 0.59 |
| **状态寄存器** (K=128) | 2 × [64,32] × 4B = 16 KB | 2 × [64,64] × 4B = 32 KB |
| **状态寄存器/thread** (4 warps) | 16KB / 128 threads ≈ 128B ≈ 32 regs | 32KB / 128 threads ≈ 256B ≈ 64 regs |
| **dot 效率** | [64,32] dot → 64×32 TC tiles | [64,64] dot → 64×64 TC tiles |

### 2.2 BV 受什么限制？

**2.2.1 寄存器限制**

🔥4 kernel 的寄存器消耗来源:
1. **状态变量 b_h1, b_h2**: 各 [64, BV] × FP32
   - BV=64: 2 × 64 × 64 × 4B = 32KB → 32768/128 threads = 256B/thread = **64 regs/thread** (仅状态)
   - BV=32: 2 × 64 × 32 × 4B = 16KB → 16384/128 threads = 128B/thread = **32 regs/thread** (仅状态)

2. **工作变量** (b_w, b_v, b_k, b_g 等): 估计 ~30-50 regs/thread

3. **总计** (粗估):
   - BV=64: ~64 + 50 = ~114 regs/thread
   - BV=32: ~32 + 50 = ~82 regs/thread

A100 限制: 每 SM 65536 regs, 每 thread 最多 255 regs

| 配置 | regs/thread (估) | threads/CTA | regs/CTA | max CTAs/SM (regs) | max CTAs/SM (threads: 2048) |
|------|------------------|-------------|----------|--------------------|-----------------------------|
| BV=64, w=4 | ~114 | 128 | 14592 | 4 | 16 |
| BV=32, w=4 | ~82 | 128 | 10496 | 6 | 16 |
| BV=32, w=2 | ~82 | 64 | 5248 | 12 | 32 |

**2.2.2 Shared Memory 限制**

🔥4 的 shared memory 主要用于 software pipeline stages (`num_stages`):
- 每个 stage 需要缓存 `b_w[BT, 64]` + `b_v[BT, BV]` + `b_k[64, BT]` 的一份
- BT=64, K=128:
  - b_w: [64, 64] × 2B = 8KB/stage
  - b_v: [64, BV] × 2B = BV=64 → 8KB/stage, BV=32 → 4KB/stage
  - b_k: [64, 64] × 2B = 8KB/stage
  - 每 stage 总计: BV=64 → 24KB, BV=32 → 20KB

| 配置 | stages | shared/CTA | A100 限制 (164KB/SM) | max CTAs/SM (shared) |
|------|--------|------------|---------------------|----------------------|
| BV=64, s=4 | 4 | 96 KB | 164 KB | 1 |
| BV=64, s=3 | 3 | 72 KB | 164 KB | 2 |
| BV=64, s=2 | 2 | 48 KB | 164 KB | 3 |
| BV=32, s=4 | 4 | 80 KB | 164 KB | 2 |
| BV=32, s=3 | 3 | 60 KB | 164 KB | 2 |
| BV=32, s=2 | 2 | 40 KB | 164 KB | 4 |

**⚠️ 这解释了为什么 BV=64, num_stages=4 获胜但 CTA/SM 极低！**

BV=64, s=4 时:
- shared memory ≈ 96KB/CTA → 每 SM 最多 1 CTA (164KB ÷ 96KB ≈ 1.7 → 下取整 1)
- 64 CTAs ÷ 108 SMs = **很多 SM 完全空闲**

### 2.3 BV=32 真的更好吗？

Autotune 选择了 BV=64 而不是 BV=32，这意味着**尽管 CTA 更少，BV=64 每个 CTA 的效率更高**：

| 维度 | BV=32 | BV=64 | 分析 |
|------|-------|-------|------|
| **每 CTA 工作量** | 处理 V 的 32 列 | 处理 V 的 64 列 | BV=64 每 CTA 做 2× 工作 |
| **dot product 尺寸** | w[64,64] @ h[64,32] = 2048 MADs | w[64,64] @ h[64,64] = 4096 MADs | BV=64 的 TC utilization 更好 (64×64 对齐) |
| **循环次数** | 同样 NT=128 步 | 同样 NT=128 步 | 相同 |
| **总 FLOPs** | 相同 | 相同 | 总计算量不变 |
| **总 CTA** | 128 (1.19/SM) | 64 (0.59/SM) | BV=32 多一倍 CTA |
| **HBM store h** | 128 CTAs 各写 [K,32] | 64 CTAs 各写 [K,64] | 总写量相同 |
| **Launch overhead** | 128 CTAs 调度开销大 | 64 CTAs 调度开销小 | BV=64 略优 |

**为什么 autotune 选 BV=64**:
1. **TC tile 效率**: A100 的 mma.m16n8k16 指令，64×64 矩阵乘天然对齐多个 TC tile，利用率高于 64×32
2. **寄存器重用**: 状态 b_h 常驻寄存器，BV=64 时每次 dot 复用更多状态数据
3. **shared memory pipeline**: BV=64,s=4 虽然每 SM 只放 1 CTA，但 4 stage pipeline 让内存延迟完全隐藏

**BV=32 什么时候更好？**
- 当 V 很大（如 V=512）：BV=32 给 16 CTAs, BV=64 给 8 CTAs，差距扩大
- 当 kernel 受 occupancy 限制而非 compute bound 时
- 当 K 更大导致寄存器压力超标时

### 2.4 BV 结论

> **BV=64 是 autotune 的正确选择**（在当前 K=128, V=128 配置下）。
> BV 主要受 **shared memory** 限制（BV=64, s=4 → 96KB/CTA → 每 SM 仅 1 CTA）。
> 寄存器也有影响但不是第一瓶颈。
> 🔥4 的根本问题不是 BV，而是**串行循环 `for i_t in range(NT=128)`** 导致 CTA 总数仅为 ceil(V/BV) × N×H = 2×32 = 64，远少于 SM 数量 108。

---

<a id="q2"></a>
## 问题2: 🔥2 的 CC 串行能否和 TC 运算并行？

### 3.1 🔥2 算法结构回顾

🔥2 (`merge_16x16_to_64x64_inverse_kernel`) 的完整结构:

```
Phase 1 — CC 串行求逆 (4 个独立的 16×16 块):

  for i = 2..16:                    ← 串行 loop (CC 计算)
    b_a_11 = -load(A行) 
    b_a_11 += sum(b_a_11 * b_Ai_11)   ← 标量/向量乘加
    b_Ai_11[row i] = b_a_11

  同理 b_Ai_22, b_Ai_33, b_Ai_44     ← 4 个 for 循环，互相独立

Phase 2 — TC 矩阵乘合并 (依赖 Phase 1 结果):

  b_Ai_21 = -dot(dot(Ai_22, A_21), Ai_11)          ← 2 个 16×16 dot
  b_Ai_32 = -dot(dot(Ai_33, A_32), Ai_22)          ← 2 个 16×16 dot
  b_Ai_43 = -dot(dot(Ai_44, A_43), Ai_33)          ← 2 个 16×16 dot

  b_Ai_31 = -dot(Ai_33, dot(A_31,Ai_11) + dot(A_32,Ai_21))  ← 3 个 dot
  b_Ai_42 = -dot(Ai_44, dot(A_42,Ai_22) + dot(A_43,Ai_32))  ← 3 个 dot
  b_Ai_41 = -dot(Ai_44, dot(A_41,Ai_11) + dot(A_42,Ai_21) + dot(A_43,Ai_31))  ← 4 个 dot

  共计: 2+2+2+3+3+4 = 16 个 16×16 dot products
```

### 3.2 依赖图分析

```
Phase 1 (CC, 互相独立):
  Loop_11 ──→ Ai_11
  Loop_22 ──→ Ai_22
  Loop_33 ──→ Ai_33
  Loop_44 ──→ Ai_44

Phase 2 (TC, 有严格依赖链):
  Level 1 (仅需 Phase 1 输出):
    Ai_21 = f(Ai_22, A_21, Ai_11)  ← 依赖 Ai_22, Ai_11
    Ai_32 = f(Ai_33, A_32, Ai_22)  ← 依赖 Ai_33, Ai_22
    Ai_43 = f(Ai_44, A_43, Ai_33)  ← 依赖 Ai_44, Ai_33

  Level 2 (依赖 Level 1):
    Ai_31 = f(Ai_33, A_31, Ai_11, A_32, Ai_21)  ← 依赖 Ai_21
    Ai_42 = f(Ai_44, A_42, Ai_22, A_43, Ai_32)  ← 依赖 Ai_32

  Level 3 (依赖 Level 2):
    Ai_41 = f(Ai_44, A_41, Ai_11, A_42, Ai_21, A_43, Ai_31)  ← 依赖 Ai_31
```

### 3.3 CC 和 TC 并行的可行性

**3.3.1 CTA 内部的指令级并行 (ILP)**

在单个 CTA 内，CUDA SM 的 warp scheduler 可以同时发射:
- FP32 CUDA Core 指令（CC）
- Tensor Core 指令（TC/mma）

到**不同的执行单元**。这是 A100 SM 的固有能力。

**但前提是：两条指令流之间没有数据依赖。**

当前 🔥2 的情况:
- Phase 1 的 4 个 for-loop 产出 `Ai_11, Ai_22, Ai_33, Ai_44`
- Phase 2 的 dot 计算需要这些结果作为输入
- **Phase 1 → Phase 2: 严格 RAW 依赖** ❌

**3.3.2 Phase 1 内部能否并行？**

4 个 for-loop (`Loop_11`, `Loop_22`, `Loop_33`, `Loop_44`) 之间**完全独立**！

理论方案: **Warp Specialization**
```
Warp 0: 计算 Loop_11 → Ai_11
Warp 1: 计算 Loop_22 → Ai_22
Warp 2: 计算 Loop_33 → Ai_33  
Warp 3: 计算 Loop_44 → Ai_44
同步 barrier
Warp 0-3: 一起做 Phase 2 的 TC dot products
```

但问题:
1. Triton 不支持 warp specialization — 所有 warp 执行相同代码
2. 当前 Phase 1 的每个 loop 操作的是 `tl.arange(0, 16)` 大小的数据，本身就用所有 warp 并行处理
3. 如果分成 4 个 warp 各自处理一个 16×16 块，每个 warp 只有 32 线程处理 16×16 = 256 元素，效率反而降低

**3.3.3 Phase 2 的 dot 和 Phase 1 的 loop 能否 pipeline？**

理想情况:
```
时间 →
Warp 0-3: [Loop_11 完成] ──→ [开始 dot(Ai_22, A_21)] ──→ ...
                              ↑ 此时 Loop_22 可能还在跑
```

只要 `Ai_11` ready，`dot(Ai_22, A_21, precision) @ Ai_11` 的第一个 dot `dot(Ai_22, A_21)` 只需要 `Ai_22`，不需要 `Ai_11`。所以:

```
可以重叠:
  dot(Ai_22, A_21)  和  Loop_33、Loop_44 并行  ← 只要 Ai_22 ready
  dot(Ai_33, A_32)  和  Loop_44 并行            ← 只要 Ai_33 ready
```

**但**这需要:
1. 手写 CUDA kernel 或用 CUTLASS warp-specialized API
2. Triton 编译器**可能**自动做一些指令重排（ILP），但无法跨 for-loop 做深度 pipeline
3. 实际效果需要看编译后的 SASS

### 3.4 更根本的优化: 消除 🔥2

从 profiling 数据看:
- 🔥2 = 0.316ms (25.5%)
- 🔥2 的 FLOPs = 0.5G (16 个 16×16 dot = 16 × 2 × 16³ = 131K FLOPs × 4096 CTAs)
- MFU = 8.8% (极低)

**🔥2 是 memory-bound** — 大量时间花在:
1. Phase 1 的标量 for-loop（纯串行，CC 利用率极低）
2. 16×16 的 dot product（太小，TC tile utilization 差）
3. 4096 个 CTA 各自独立做很少的计算

**更好的方案**:
1. **fuse 🔥1+🔥2+🔥3**: 三者都是 grid=(NT, B*H)=(128,32)，可以 fuse 成一个 kernel
   - 🔥1 产出 A = k@k^T → 🔥2 消费 A → 🔥3 消费 Ai
   - Fuse 后 A 和 Ai 完全在 shared memory/寄存器中传递，不经过 HBM
   - 预期收益: 节省 🔥1 写 A 到 HBM + 🔥2 读 A + 🔥2 写 Ai + 🔥3 读 Ai = 大量 HBM 带宽
2. **Block inverse algorithm**: 直接用 32×32 或 64×64 的 block-Gauss 消元代替层次 merge
3. **Approximate inverse**: 用 Neumann 级数 $(I+A)^{-1} \approx I - A + A^2 - A^3$ (A 是下三角严格矩阵，级数有限项即精确)

### 3.5 🔥2 CC+TC 并行结论

| 方案 | 可行性 | 预期收益 |
|------|--------|----------|
| Phase 1 CC 和 Phase 2 TC 直接并行 | ❌ 严格 RAW 依赖 | 0 |
| Phase 1 内部 4 个 loop warp specialization | ⚠️ 理论可行, Triton 不支持 | ~15-20%（减少 Phase 1 时间到 1/4） |
| Phase 2 Level 1 dot 和 Phase 1 剩余 loop pipeline | ⚠️ 理论可行, 需要 CUDA | ~10%（部分重叠） |
| **fuse 🔥1+🔥2+🔥3** | ✅ 推荐 | **40%+**（消除 HBM 往返） |

---

<a id="summary"></a>
## 总结与优先级排序

### 核心发现

1. **🔥4 SM 用不满的根因**: 不是 BV 的问题，而是**算法本身是串行递推** (`for i_t in range(NT)`)。CTA 数 = ceil(V/BV) × N×H = 2×32 = 64，远少于 108 SMs。加上 BV=64, stages=4 导致每 CTA 占 ~96KB shared memory → 每 SM 只能放 1 CTA。

2. **🔥4+🔥5 overlap**: 理论可行（44 个 SM 空闲），但被 h[i_t] 的 RAW 依赖阻塞。需要 producer-consumer fusion（persistent kernel + 跨 CTA 同步），Triton 目前不支持。

3. **BV=32 vs BV=64**: Autotune 选 BV=64 是正确的。BV=64 的 TC tile 利用率更高，4-stage pipeline 隐藏了延迟。BV=32 虽然 CTA 翻倍 (128)，但每个 dot 计算效率下降。

4. **🔥2 CC/TC 并行**: Phase 1→Phase 2 有严格依赖，不能直接并行。Phase 1 内部 4 个 loop 理论上可以 warp specialize 但 Triton 不支持。**更根本的方案是 fuse 🔥1+🔥2+🔥3。**

### 优化路线图（更新版）

| 优先级 | 方案 | 目标 kernel | 预期收益 | 难度 | 工具 |
|--------|------|------------|----------|------|------|
| **P0** | 🔥1+🔥2+🔥3 fusion | 46% 总延迟 | **40%+ 总延迟** | 高 | Triton (手写 fused kernel) |
| **P1** | 🔥3 启用 tf32 | 🔥3 (16.2%) | ~8% 总延迟 | 低 | 改 `allow_tf32=True` |
| **P2** | 🔥4 减少 stages | 🔥4 (26.9%) | ~5% 总延迟 | 低 | autotune config 调优 |
| **P3** | 🔥4+🔥5 fusion | 51% 总延迟 | ~20% 总延迟 | 极高 | CUDA persistent kernel |
| **P4** | 🔥2 Neumann 级数 | 🔥2 (25.5%) | ~15% 总延迟 | 中 | Triton 重写 |
| **P5** | BT 全局调优 | 全部 | 待测 | 低 | 环境变量/参数扫描 |

### 速查卡片

```
🔥4 的问题:
  ├── CTA 太少: 64 vs 108 SMs = 40% SM 空闲
  ├── 根因: for i_t in range(NT=128) 串行递推
  ├── BV=64 正确: TC 效率 > CTA 数量
  └── BV 限制源: shared memory (96KB/CTA @s=4) > 寄存器 (114 regs/thread 估)

🔥2 的问题:
  ├── MFU 8.8% (极低) — memory bound
  ├── CC 串行 for-loop: 14 步 × 4 块 = 56 步标量循环
  ├── TC dot: 16 个 16×16 太小，tile utilization 差
  ├── CC 和 TC 不能直接并行 (RAW 依赖)
  └── 最优方案: fuse 🔥1+🔥2+🔥3，消除 HBM 往返
```
