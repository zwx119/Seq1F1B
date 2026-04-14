# DeltaNet Forward Kernel 维度与复杂度分析

## 全局参数

| 符号 | 含义 | 1.3B 实际值 |
|------|------|-----------|
| $B$ | batch size | 1~2 |
| $T$ | 序列长度 | 8192~32768 |
| $H$ | head 数 | 32 |
| $K$ | key/query head dim | 128 |
| $V$ | value head dim | 128 |
| $BT$ | chunk size | 64 |
| $NT$ | chunk 数 = $T/BT$ | 128~512 |
| $BK$ | K 方向 tile size | autotune (32/64/128) |
| $BV$ | V 方向 tile size | autotune (32/64) |

---

## 🔥1 `chunk_scaled_dot_kkt` — 计算 chunk 内注意力矩阵

### 数学公式

对每个 chunk $c$ (包含 $BT$ 个 token)：

$$A_c = \text{tril}\Big(\beta_c \odot (K_c \cdot K_c^\top)\Big)$$

其中 $\odot$ 表示按行广播乘 $\beta$。

### 输入/输出维度

| 张量 | 维度 | 说明 |
|------|------|------|
| **输入** $k$ | $[B, T, H, K]$ | key，全局 |
| **输入** $\beta$ | $[B, T, H]$ | gate 标量 |
| **输出** $A$ | $[B, T, H, BT]$ | 每个 token 存一行长 $BT$ 的下三角 |

### 单个 CTA 的计算（处理 1 个 chunk 的 1 个 head）

```python
b_A = zeros([BT, BT])                      # 累加器，fp32
for i_k in range(ceil(K / BK)):            # 循环 ceil(128/BK) 次
    b_k = load(k, [BT, BK])               # 读 1 个 tile
    b_A += dot(b_k, trans(b_k))            # [BT, BK] @ [BK, BT] → [BT, BT]
b_A *= b_b[:, None]                        # [BT] broadcast mul → [BT, BT]
b_A = where(lower_tri_strict, b_A, 0)     # 严格下三角 mask
store(A, b_A)                              # 写出
```

### FLOPs 精确计算

矩阵乘 `[BT, BK] @ [BK, BT]` 的 FLOPs = $2 \times BT \times BK \times BT$

循环 $\lceil K/BK \rceil$ 次，总 FLOPs：

$$\text{FLOPs}_1 = \lceil K/BK \rceil \times 2 \times BT \times BK \times BT = 2 \times BT^2 \times K$$

代入 $BT=64, K=128$：

$$\text{FLOPs}_1 = 2 \times 64^2 \times 128 = 1,048,576 \approx 1.05\text{M}$$

> 解释：不管 BK 取多少（32/64/128），循环次数 × BK 总是 = K，所以 BK 约掉了。

### HBM 读写

- 读：$k$ 的一个 chunk = $BT \times K \times 2\text{B}$ = $64 \times 128 \times 2 = 16\text{KB}$
- 读：$\beta$ = $BT \times 2\text{B}$ = $128\text{B}$
- 写：$A$ = $BT \times BT \times 4\text{B}$ = $64 \times 64 \times 4 = 16\text{KB}$ (fp32)

### Grid 和并行度

$$\text{Grid}_1 = (NT, B \times H)$$

CTA 总数 = $NT \times B \times H$，每个 CTA 处理 1 个 chunk × 1 个 head。

**所有 CTA 互相独立，完全并行。**

### 总 FLOPs 和 Latency

$$\text{Total FLOPs}_1 = NT \times B \times H \times 2 \times BT^2 \times K = \frac{T}{BT} \times B \times H \times 2 \times BT^2 \times K = 2 \times B \times H \times T \times BT \times K$$

$$\boxed{\text{FLOPs}_1 = 2 \cdot B \cdot H \cdot T \cdot BT \cdot K \propto T}$$

$$\text{Latency}_1 = \left\lceil \frac{NT \times B \times H}{\text{SMs}} \right\rceil \times t_1 \propto T$$

B=1, H=32, T=32K → CTA 数 = 512×32 = 16384，A100 有 108 SMs → 每 SM 跑 ~152 个 CTA。

---

## 🔥2 `solve_tril` — 三角矩阵求逆 $(I + A)^{-1}$

### 数学公式

$$A_c \leftarrow (I + A_c)^{-1}$$

$A_c$ 是 $BT \times BT$ 的严格下三角矩阵。

### 输入/输出维度

| 张量 | 维度 | 说明 |
|------|------|------|
| **输入** $A$ | $[B, T, H, BT]$ | 🔥1 的输出，fp32 |
| **输出** $A_i$ | $[B, T, H, BT]$ | 求逆结果，转为 bf16 |

### 算法结构（BT=64 → `merge_16x16_to_64x64_inverse_kernel`）

分两个阶段：

**Phase 1: 4 个 16×16 对角块，各自串行求逆**

```python
# 对每个 16×16 对角块 (共 4 个: [0:16], [16:32], [32:48], [48:64])
b_Ai = -where(strict_lower, b_A_diag, 0)   # [16, 16]
for i in range(2, 16):                       # 14 步串行 ← CC dominant
    b_a = -load(A, row=i, [16])             # 读一行
    b_a = where(col < i, b_a, 0)
    b_a += sum(b_a[:, None] * b_Ai, dim=0)  # [16] × [16,16] → reduce ← CC
    b_Ai = where(row==i, b_a, b_Ai)         # 更新一行
b_Ai += I                                   # 加单位矩阵
```

**Phase 2: 合并 off-diagonal 块**（用 `tl.dot`）

```python
# 6 个 off-diagonal 16×16 块:
b_Ai_21 = -dot(dot(b_Ai_22, b_A_21), b_Ai_11)     # [16,16]@[16,16]@[16,16] ← TC
b_Ai_32 = -dot(dot(b_Ai_33, b_A_32), b_Ai_22)     # 同上
b_Ai_43 = -dot(dot(b_Ai_44, b_A_43), b_Ai_33)
b_Ai_31 = -dot(b_Ai_33, dot(b_A_31,b_Ai_11) + dot(b_A_32,b_Ai_21))
b_Ai_42 = -dot(b_Ai_44, dot(b_A_42,b_Ai_22) + dot(b_A_43,b_Ai_32))
b_Ai_41 = -dot(b_Ai_44, dot(b_A_41,b_Ai_11) + dot(b_A_42,b_Ai_21) + dot(b_A_43,b_Ai_31))
```

### FLOPs 精确计算

**Phase 1**（每个 16×16 块）：

每步：`sum(b_a[:, None] * b_Ai)` = 16 次乘 + 15 次加 ≈ $2 \times 16 = 32$ FLOPs  
14 步 × 4 块 = $14 \times 4 \times 32 = 1,792$ FLOPs

**Phase 2**：

每个 `dot([16,16], [16,16])` = $2 \times 16^3 = 8,192$ FLOPs

统计代码中的 dot 调用次数：
- 3 个两层 dot（各 2 次 dot）= 6 次
- 2 个三层 dot（各 3 次 dot）= 6 次  
- 1 个四层 dot（4 次 dot）= 4 次
- 合计约 **16 次 dot**

$$\text{FLOPs}_2 \approx 1,792 + 16 \times 8,192 = 1,792 + 131,072 \approx 133\text{K}$$

### HBM 读写

- 读：$A$ 的一个 chunk = $BT \times BT \times 4\text{B}$ = $16\text{KB}$
- 写：$A_i$ 的一个 chunk = $BT \times BT \times 2\text{B}$ = $8\text{KB}$

### Grid 和并行度

$$\text{Grid}_2 = (NT, B \times H)$$

与 🔥1 相同，完全并行。

### 为什么 FLOPs 这么小却耗时这么长？

FLOPs 只有 133K（是 🔥1 的 1/8），但耗时却是 🔥1 的 **3.6 倍** (5.676ms vs 1.580ms)！

原因：**Phase 1 的 for-loop 是 latency bound**
- 14-step 串行，每步都依赖上一步结果
- 每步的计算量极小（32 FLOPs），但 GPU 一个时钟周期只能执行 1 步
- Phase 2 的 16×16 dot 也太小，TC 的最小高效粒度需要更大的矩阵

$$\boxed{\text{FLOPs}_2 \approx 133\text{K/chunk} \ll \text{其他 kernel，但受 latency bound 限制}}$$

---

## 🔥3 `recompute_w_u` — 重建 w 和 u

### 数学公式

$$u_c = A_c^{-1} \cdot (\beta_c \odot v_c), \quad w_c = A_c^{-1} \cdot (\beta_c \odot k_c)$$

其中 $A_c^{-1}$ 就是 🔥2 的输出。

### 输入/输出维度

| 张量 | 维度 | 说明 |
|------|------|------|
| **输入** $A$ | $[B, T, H, BT]$ | 🔥2 的输出 |
| **输入** $k$ | $[B, T, H, K]$ | key |
| **输入** $v$ | $[B, T, H, V]$ | value |
| **输入** $\beta$ | $[B, T, H]$ | gate |
| **输出** $w$ | $[B, T, H, K]$ | 修正后的 key |
| **输出** $u$ | $[B, T, H, V]$ | 修正后的 value |

### 单个 CTA 的计算

```python
b_A = load(A, [BT, BT])          # 读 A^{-1}
b_beta = load(beta, [BT])        # 读 β

# 计算 u = A^{-1} @ (v * β)
for i_v in range(ceil(V / BV)):           # ceil(128/64) = 2 次
    b_v = load(v, [BT, BV])               # 读 v 的一个 tile
    b_vb = b_v * b_beta[:, None]           # elementwise, [BT,BV] ← CC
    b_u = dot(b_A, b_vb, allow_tf32=False) # [BT,BT] @ [BT,BV] → [BT,BV] ← TC(FP32)
    store(u, b_u)

# 计算 w = A^{-1} @ (k * β)
for i_k in range(ceil(K / BK)):           # ceil(128/64) = 2 次
    b_k = load(k, [BT, BK])
    b_kb = b_k * b_beta[:, None]           # elementwise ← CC
    b_w = dot(b_A, b_kb, allow_tf32=False) # [BT,BT] @ [BT,BK] → [BT,BK] ← TC(FP32)
    store(w, b_w)
```

### FLOPs 精确计算

每个 `dot([BT,BT], [BT,BX])` = $2 \times BT^2 \times BX$ FLOPs

$$\text{FLOPs}_3 = \underbrace{\lceil V/BV \rceil \times 2 \times BT^2 \times BV}_{u \text{ 部分}} + \underbrace{\lceil K/BK \rceil \times 2 \times BT^2 \times BK}_{w \text{ 部分}}$$

$$= 2 \times BT^2 \times V + 2 \times BT^2 \times K = 2 \times BT^2 \times (K + V)$$

代入 $BT=64, K=V=128$：

$$\text{FLOPs}_3 = 2 \times 64^2 \times 256 = 2,097,152 \approx 2.10\text{M}$$

elementwise `v * β` 的 FLOPs = $BT \times V + BT \times K = BT \times (K+V)$ ≈ 16K（可忽略）

### HBM 读写

- 读：$A$ = 16KB, $k$ = 16KB, $v$ = 16KB, $\beta$ = 128B
- 写：$w$ = 16KB, $u$ = 16KB
- 总: ~80KB

### Grid 和并行度

$$\text{Grid}_3 = (NT, B \times H)$$

完全并行。

### 注意 `allow_tf32=False`

这个 flag 强制用 **IEEE FP32** 做矩阵乘，而非 TF32。在 A100 上：
- BF16 TC: 312 TFLOPS
- TF32 TC: 156 TFLOPS  
- FP32 (IEEE, 无 TC): 19.5 TFLOPS

`allow_tf32=False` 意味着 **不能用 Tensor Core**，只能用 CUDA Core 做 FP32 乘加，理论峰值降 16 倍。

$$\boxed{\text{FLOPs}_3 = 2 \cdot BT^2 \cdot (K+V) \approx 2.10\text{M/chunk, FP32 precision}}$$

---

## 🔥4 `chunk_fwd_h` — 跨 chunk 状态递推

### 数学公式

$$S_0 = \text{initial\_state}$$

对每个 chunk $c = 0, 1, ..., NT-1$：

$$\tilde{v}_c = u_c - w_c \cdot S_c \quad \text{(intra-chunk correction)}$$

$$S_{c+1} = S_c + k_c^\top \cdot \tilde{v}_c \quad \text{(state update)}$$

输出所有 $S_c$ 和 $\tilde{v}_c$。

### 输入/输出维度

| 张量 | 维度 | 说明 |
|------|------|------|
| **输入** $k$ | $[B, T, H, K]$ | key |
| **输入** $w$ | $[B, T, H, K]$ | 🔥3 的输出 |
| **输入** $u$ | $[B, T, H, V]$ | 🔥3 的输出 |
| **输入** $S_0$ | $[B, H, K, V]$ 或 None | 初始状态 |
| **输出** $h$ | $[B, NT, H, K, V]$ | 每个 chunk 的状态快照 |
| **输出** $v\_new$ | $[B, T, H, V]$ | 修正后的 value |
| **输出** $S_T$ | $[B, H, K, V]$ 或 None | 最终状态 |

### 单个 CTA 的计算

每个 CTA 处理**一个 head 的一个 V-block**，遍历所有 NT 个 chunk：

```python
b_h = zeros([64, BV])    # 状态，K 被拆成多个 64-block，每个 CTA 处理一个

for i_t in range(NT):                       # ← ⚠️ 串行！NT 步
    store(h[i_t], b_h)                       # 存当前状态快照

    # Step A: v_correction = v - w @ h  (intra-chunk correction)
    b_v = load(v, [BT, BV])                  # 读 u
    for k_block in range(ceil(K / 64)):      # K=128 → 2 次
        b_w = load(w, [BT, 64])              # 读 w 的一个 tile
        b_v -= dot(b_w, b_h)                 # [BT,64] @ [64,BV] → [BT,BV] ← TC

    store(v_new[i_t], b_v)                   # 写修正后的 v

    # Step B: h += k^T @ v_corrected  (state update)
    for k_block in range(ceil(K / 64)):      # K=128 → 2 次
        b_k = load(k, [64, BT])             # 读 k 转置
        b_h += dot(b_k, b_v)                 # [64,BT] @ [BT,BV] → [64,BV] ← TC
```

### FLOPs 精确计算

**每个 chunk (每步循环)：**

Step A: $\lceil K/64 \rceil$ 次 `dot([BT,64], [64,BV])` = $\lceil K/64 \rceil \times 2 \times BT \times 64 \times BV$

Step B: $\lceil K/64 \rceil$ 次 `dot([64,BT], [BT,BV])` = $\lceil K/64 \rceil \times 2 \times 64 \times BT \times BV$

两者相同！每步总 FLOPs:

$$\text{FLOPs per step} = 2 \times \lceil K/64 \rceil \times 2 \times BT \times 64 \times BV = 4 \times BT \times K \times BV$$

> 注意：🔥4 中 K 的 tile 硬编码为 64（不是 autotune 的 BK），代码中直接 `(BT, 64)` 和 `(64, BV)`

代入 $BT=64, K=128, BV=32$：

$$\text{FLOPs per step} = 4 \times 64 \times 128 \times 32 = 1,048,576 \approx 1.05\text{M}$$

**一个 CTA 的总 FLOPs**（遍历所有 NT 个 chunk）：

$$\text{FLOPs per CTA} = NT \times 4 \times BT \times K \times BV$$

### HBM 读写（每步）

- 读：$w$ = $BT \times K \times 2\text{B}$ = 16KB, $u$ = $BT \times BV \times 2\text{B}$ = 4KB, $k$ = $K \times BT \times 2\text{B}$ = 16KB
- 写：$h$ = $K \times BV \times 2\text{B}$ = 8KB（但只写当前 CTA 的 V-block，即 $64 \times BV$），$v\_new$ = 4KB
- 每步约 ~48KB，NT 步总 ~NT × 48KB

### Grid 和并行度

$$\text{Grid}_4 = (\lceil V/BV \rceil,\ B \times H)$$

⚠️ **NT 不在 Grid 中！** NT 是 CTA 内的 for-loop。

CTA 数 = $\lceil V/BV \rceil \times B \times H$

代入 $V=128, BV=32, B=1, H=32$：

$$\text{CTA 数} = 4 \times 32 = 128 > 108 \text{ SMs}$$

如果 $BV=64$：CTA 数 = $2 \times 32 = 64 < 108$ → **41% SM 空闲！**

### 总 FLOPs

$$\text{Total FLOPs}_4 = \lceil V/BV \rceil \times B \times H \times NT \times 4 \times BT \times K \times BV$$

$$= B \times H \times \frac{V}{BV} \times \frac{T}{BT} \times 4 \times BT \times K \times BV = 4 \times B \times H \times T \times K \times V$$

$$\boxed{\text{FLOPs}_4 = 4 \cdot B \cdot H \cdot T \cdot K \cdot V \propto T}$$

$$\boxed{\text{Latency}_4 = \left\lceil \frac{\lceil V/BV \rceil \cdot B \cdot H}{108} \right\rceil \times NT \times t_4 \propto T}$$

---

## 🔥5 `chunk_fwd_o` — 输出计算

### 数学公式

对每个 chunk $c$：

$$o_c = \underbrace{q_c \cdot S_c}_{\text{inter-chunk}} \cdot \text{scale} + \underbrace{\text{causal}(q_c \cdot k_c^\top) \cdot \tilde{v}_c}_{\text{intra-chunk}} \cdot \text{scale}$$

### 输入/输出维度

| 张量 | 维度 | 说明 |
|------|------|------|
| **输入** $q$ | $[B, T, H, K]$ | query |
| **输入** $k$ | $[B, T, H, K]$ | key |
| **输入** $v\_new$ | $[B, T, H, V]$ | 🔥4 修正后的 value |
| **输入** $h$ | $[B, NT, H, K, V]$ | 🔥4 输出的状态快照 |
| **输出** $o$ | $[B, T, H, V]$ | 最终输出 |

### 单个 CTA 的计算

每个 CTA 处理 1 个 chunk × 1 个 V-block：

```python
b_o = zeros([BT, BV])                      # 输出累加器
b_A = zeros([BT, BT])                      # 注意力矩阵

for i_k in range(ceil(K / BK)):            # K=128, BK=128 → 1 次 (或 BK=64 → 2 次)
    b_q = load(q, [BT, BK])               # 读 query tile
    b_k = load(k, [BK, BT])               # 读 key tile (转置)
    b_h = load(h, [BK, BV])               # 读状态 tile

    b_o += dot(b_q, b_h)                   # [BT,BK] @ [BK,BV] → [BT,BV] ← TC (inter-chunk)
    b_A += dot(b_q, b_k)                   # [BT,BK] @ [BK,BT] → [BT,BT] ← TC (intra-chunk attn)

b_A = where(causal_mask, b_A, 0)          # 因果 mask ← CC
b_v = load(v_new, [BT, BV])               # 读修正后的 v

b_o = b_o * scale + dot(b_A, b_v) * scale # [BT,BT] @ [BT,BV] → [BT,BV] ← TC
store(o, b_o)
```

### FLOPs 精确计算

K 循环中每次：
- `dot(q, h)`: $[BT, BK] \times [BK, BV]$ → $2 \times BT \times BK \times BV$ FLOPs
- `dot(q, k^T)`: $[BT, BK] \times [BK, BT]$ → $2 \times BT \times BK \times BT$ FLOPs

循环后：
- `dot(A, v)`: $[BT, BT] \times [BT, BV]$ → $2 \times BT^2 \times BV$ FLOPs

总 FLOPs per CTA:

$$\text{FLOPs}_5 = \lceil K/BK \rceil \times (2 \times BT \times BK \times BV + 2 \times BT \times BK \times BT) + 2 \times BT^2 \times BV$$

$$= 2 \times BT \times K \times BV + 2 \times BT \times K \times BT + 2 \times BT^2 \times BV$$

$$= 2 \times BT \times K \times (BV + BT) + 2 \times BT^2 \times BV$$

代入 $BT=64, K=128, BV=128$ (autotune 可能选 BK=BV=128):

$$= 2 \times 64 \times 128 \times (128 + 64) + 2 \times 64^2 \times 128$$

$$= 2 \times 64 \times 128 \times 192 + 2 \times 4096 \times 128$$

$$= 3,145,728 + 1,048,576 = 4,194,304 \approx 4.19\text{M}$$

### Grid 和并行度

$$\text{Grid}_5 = (\lceil V/BV \rceil,\ NT,\ B \times H)$$

**三维全并行** — 并行度最高。

CTA 数 = $\lceil V/BV \rceil \times NT \times B \times H$

代入 $V=128, BV=128, NT=512, B=1, H=32$：

$$\text{CTA 数} = 1 \times 512 \times 32 = 16384 \gg 108$$

### 总 FLOPs

$$\text{Total FLOPs}_5 = \lceil V/BV \rceil \times NT \times B \times H \times \text{FLOPs per CTA}$$

由于 $\lceil V/BV \rceil \times BV = V$：

$$= B \times H \times NT \times (2 \times BT \times K \times V + 2 \times BT \times K \times BT + 2 \times BT^2 \times V)$$

$$= B \times H \times \frac{T}{BT} \times 2 \times BT \times (K \times V + K \times BT + BT \times V)$$

$$= 2 \times B \times H \times T \times (KV + K \cdot BT + BT \cdot V)$$

$$\boxed{\text{FLOPs}_5 = 2 \cdot B \cdot H \cdot T \cdot (KV + K \cdot BT + BT \cdot V) \propto T}$$

代入 $K=V=128, BT=64$：

$$= 2 \times B \times H \times T \times (16384 + 8192 + 8192) = 2 \times B \times H \times T \times 32768$$

---

## 总结对比

### FLOPs per chunk (单个 CTA)

| Kernel | FLOPs 公式 | 代入值 (BT=64,K=V=128) | 计算单元 |
|--------|-----------|----------------------|---------|
| 🔥1 kkt | $2 \cdot BT^2 \cdot K$ | 1.05M | TC (BF16) |
| 🔥2 solve | $\sim 133\text{K}$ | 0.13M | CC (for-loop) + TC (16×16 dot) |
| 🔥3 w_u | $2 \cdot BT^2 \cdot (K+V)$ | 2.10M | TC (FP32, allow_tf32=False) |
| 🔥4 h_fwd | $4 \cdot BT \cdot K \cdot BV$ (×NT步) | 1.05M ×NT步 | TC (BF16) |
| 🔥5 o_fwd | $2 \cdot BT \cdot K \cdot (BV+BT) + 2 \cdot BT^2 \cdot BV$ | 4.19M | TC (BF16) |

### Grid 和并行模式

| Kernel | Grid | CTA 数 (B=1,H=32,T=32K) | 串行 |
|--------|------|------------------------|------|
| 🔥1 | $(NT, BH)$ | 16,384 | 无 |
| 🔥2 | $(NT, BH)$ | 16,384 | CTA 内 14-step for-loop |
| 🔥3 | $(NT, BH)$ | 16,384 | 无 |
| 🔥4 | $(\lceil V/BV \rceil, BH)$ | 128 (BV=32) | **CTA 内 NT-step for-loop** |
| 🔥5 | $(\lceil V/BV \rceil, NT, BH)$ | 16,384 (BV=128) | 无 |

### 总 FLOPs

| Kernel | 总 FLOPs 公式 | 代入值 (B=1,H=32,T=32K) |
|--------|-------------|----------------------|
| 🔥1 | $2 \cdot BHT \cdot BT \cdot K$ | 34.4G |
| 🔥2 | $\sim 133\text{K} \times NT \times BH$ | 2.2G |
| 🔥3 | $2 \cdot BHT \cdot BT \cdot (K+V)$ | 68.7G |
| 🔥4 | $4 \cdot BHT \cdot K \cdot V$ | 68.7G |
| 🔥5 | $2 \cdot BHT \cdot (KV + K \cdot BT + BT \cdot V)$ | 68.7G |

### 为什么实测耗时和 FLOPs 不成正比？

| Kernel | 总 FLOPs | 实测占比 | FLOPs/实测 比 | 原因 |
|--------|---------|---------|-------------|------|
| 🔥1 | 34.4G | 6.7% | ✓ 符合 | BF16 TC, 高并行 |
| 🔥2 | 2.2G | **24.0%** | **11× 超支** | latency bound, for-loop 串行 |
| 🔥3 | 68.7G | 16.5% | 偏低 | allow_tf32=False, FP32 速度慢但还行 |
| 🔥4 | 68.7G | 27.2% | ✓ 符合 | BF16 TC, CTA 内 NT 步串行 |
| 🔥5 | 68.7G | 24.0% | ✓ 符合 | BF16 TC, 高并行 |

**🔥2 是硬件效率最差的 kernel — FLOPs 只占 1%，耗时却占 24%，效率差了 24 倍。这是最值得优化的目标。**
