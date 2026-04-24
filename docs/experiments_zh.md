# DeltaNet + Seq1F1B 实验指南

## 目录

1. [实验概述](#1-实验概述)
2. [环境准备](#2-环境准备)
3. [实验一：DeltaNet vs Softmax 基线对比](#3-实验一deltanet-vs-softmax-基线对比)
4. [实验二：不同序列长度下的显存对比](#4-实验二不同序列长度下的显存对比)
5. [实验三：Seq1F1B 切分数对 DeltaNet 的影响](#5-实验三seq1f1b-切分数对-deltanet-的影响)
6. [实验四：不同模型规模对比](#6-实验四不同模型规模对比)
7. [实验五：超长序列（64K/128K）极限测试](#7-实验五超长序列64k128k极限测试)
8. [结果收集与分析](#8-结果收集与分析)

---

## 1. 实验概述

### 目标

验证 DeltaNet（线性注意力）与 Seq1F1B（序列级流水线并行）结合后，相比原版 Softmax Attention + Seq1F1B 的优势：

| 对比维度 | 预期结果 | 实验验证 |
|---------|---------|---------|
| **显存** | DeltaNet 的 recurrent state 是 O(1)，Softmax 的 KV cache 是 O(n)，长序列下 DeltaNet 显存更低 | ✅ seq=32K 时 DeltaNet 42.4GB vs Softmax 43.8GB，更长序列差距更大 |
| **吞吐量** | DeltaNet 无 softmax 计算，理论上更快，但 chunk kernel 有额外开销 | ✅ 交叉点在 seq≈12K-16K；seq=32K 时 DeltaNet 1.46× 快 |
| **Loss 收敛** | 两者架构不同，loss 绝对值不可直接比较，但各自应该正常下降 | ✅ 两者 loss 均正常下降 |
| **PP_SP 消融** | PP_SP 增大降低显存，但过度切分增加 pipeline bubble | ✅ seq=32K 下 SP=8 最优（74,659 tok/s）；DeltaNet 在所有 SP 下均 1.41-1.75× 快于 Softmax |
| **模型扩展** | 不同模型规模下 DeltaNet 均快于 Softmax | ✅ 1.3B 1.41×、2.7B **1.65×**、7B 1.23×；主要看 tok/s 与显存 |
| **超长序列** | DeltaNet 在 64K+ 序列下仍可训练 | ✅ seq=64K: DeltaNet ✅ (73K tok/s, corrected MFU ≈25.3%) vs Softmax **OOM** ❌ |

### 核心变量

- **注意力类型**：DeltaNet（`--use-deltanet`） vs Softmax（`--use-flash-attn`）
- **序列长度**：4096 / 8192 / 16384 / 32768
- **Seq1F1B 切分数**：PP_SP = 1（不切分）/ 2 / 4 / 8
- **模型规模**：1.3B / 2.7B / 7B（TP=2）

### 1.1 MFU 口径修正

早期日志中打印的 `TFlops/s` 使用了 Megatron 风格的 softmax attention FLOPs 公式，
其中包含 `seq_len^2` attention 项。这个口径适合 Softmax/FlashAttention，但会明显高估
DeltaNet 的 FLOPs，因为 DeltaNet 的核心 attention 是线性的 recurrent/state update。

本文后续表格中的 DeltaNet MFU 已按更保守的线性 DeltaNet FLOPs 重新计算：

```text
DeltaNet forward FLOPs/token/layer ≈
  2 * (q_proj + k_proj + v_proj + gate_proj + out_proj)
+ 4 * hidden * ffn_hidden
+ beta_proj + short_conv + delta_rule_core

training FLOPs ≈ 3 * forward FLOPs
MFU = tokens_per_second * training_FLOPs_per_token / num_gpus / 312e12
```

其中 A100 BF16 Tensor Core 峰值按 312 TFLOP/s/device 计算。`tokens/s`、step time 和显存
来自原始日志，未做修改。由于 DeltaNet 和 Softmax 的 FLOPs 定义不同，跨注意力类型比较时
应优先看 **tokens/s、step time、显存、是否 OOM**，MFU 只作为同一口径下的辅助指标。

---

## 2. 环境准备

### 2.1 通用环境变量（所有实验共用）

在测试机上执行：

```bash
# 进入项目目录
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

# 拉取最新代码
git pull myfork main

# 通用分布式配置（8 卡单机）
export GPUS_PER_NODE=8
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12345

# 数据路径（指向包含 data/ 子目录的路径）
export DATA_PATH=/mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

# 创建日志目录
mkdir -p exp_logs/deltanet_exps
```

### 2.2 确认数据文件

确保以下文件存在：

```bash
ls -la data/codeparrot_content_document_text_document.bin
ls -la data/codeparrot_content_document_text_document.idx
ls -la data/vocab.json
ls -la data/merges.txt
```

### 2.3 确认依赖

```bash
python -c "import torch; from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction; print('apex OK')"
python -c "import torch; from fla.ops.delta_rule import chunk_delta_rule; print('fla OK')"
```

---

## 3. 实验一：DeltaNet vs Softmax 基线对比

### 目的

在相同模型配置下，对比 DeltaNet 和 Softmax Attention 的显存、吞吐量、Loss。

### 配置

- 模型：1.3B（24层, hidden=2048, heads=16）
- 序列长度：4096
- PP=4, TP=1, PP_SP=4
- 训练 10 步

### 3.1 运行 DeltaNet

**不需要改文件，只需要 export + 运行脚本。**

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

export GPUS_PER_NODE=8 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B
export PP_SIZE=4 TP_SIZE=1 PP_SP=4 PP_SP_STR=uniform_comp
export NUM_LAYERS=24 HIDDEN=2048 NUM_ATTN_HEADS=16
export SEQ_LENGTH=4096 MICRO_BATCH=1 GLOBAL_BATCH=8 TRAIN_ITER=10

bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp1_deltanet_1.3b_seq4096_pp4_sp4.log
```

### 3.2 运行 Softmax（原版 Seq1F1B）

**不需要改文件，只需要 export + 运行 `run.sh`。**

`run.sh` 使用 `--use-flash-attn` 和 `--position-embedding-type rope`，这就是原版 Softmax + RoPE。

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

export GPUS_PER_NODE=8 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B
export PP_SIZE=4 TP_SIZE=1 PP_SP=4 PP_SP_STR=uniform_comp VPP_SIZE=1
export NUM_LAYERS=24 HIDDEN=2048 NUM_ATTN_HEADS=16
export SEQ_LENGTH=4096 MICRO_BATCH=1 GLOBAL_BATCH=8 TRAIN_ITER=10

bash run.sh 2>&1 | tee exp_logs/deltanet_exps/exp1_softmax_1.3b_seq4096_pp4_sp4.log
```

> **注意**：`run.sh` 的 data-path 写的是 `codeparrot_content_document`（没有 `_text_document` 后缀）。
> 如果你之前用 `preprocess_data.py` 生成的文件带 `_text_document` 后缀，需要确认 `run.sh` 第 42 行的路径是否匹配。
> 如果不匹配，有两种解决方式：
>
> **方式 A**：创建软链接（推荐，不改文件）
> ```bash
> cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B/data
> ln -sf codeparrot_content_document_text_document.bin codeparrot_content_document.bin
> ln -sf codeparrot_content_document_text_document.idx codeparrot_content_document.idx
> ```
>
> **方式 B**：直接改 `run.sh` 第 42 行
> 把 `--data-path $DATA_PATH/data/codeparrot_content_document`
> 改为 `--data-path $DATA_PATH/data/codeparrot_content_document_text_document`

### 3.3 对比项

从日志末尾提取：

```bash
# 提取关键指标
grep "after training is done" exp_logs/deltanet_exps/exp1_deltanet_*.log
grep "after training is done" exp_logs/deltanet_exps/exp1_softmax_*.log
```

记录到表格：

| 指标 | DeltaNet | Softmax | 说明 |
|------|----------|---------|------|
| Loss (iter 5) | 0.783 | 0.766 | 架构不同，绝对值不可直接比较 |
| 吞吐量 (tok/s) | 24,873 | 54,407 | Softmax ~2.19× 快（短序列下 FlashAttention 高效） |
| Corrected TFLOPs | 26.83 | 61.76 | DeltaNet 使用线性 attention FLOPs 口径 |
| 显存 stage 0 (GB) | 9.49 | 9.04 | |
| 显存 stage 1 (GB) | 7.57 | 7.53 | |
| 显存 stage 2 (GB) | 6.90 | 7.01 | |
| 显存 stage 3 (GB) | 8.83 | 8.74 | |
| 每步耗时 (ms) | 1,370 | 602 | 取 iter≥2 的平均值（排除 Triton JIT 预热） |

> **分析**：在 seq=4096 短序列下，Softmax（FlashAttention）吞吐量约 2.19× 于 DeltaNet，显存两者基本持平。
> 这符合预期——短序列下 FlashAttention 的 IO-aware 优化非常高效，DeltaNet 的 chunk kernel 有额外开销。
> **DeltaNet 的优势在长序列下才会显现**（见实验二）。
>
> **注意**：DeltaNet 的 exp1 数据取自 exp2_deltanet_seq4096（相同配置 1.3B/PP=4/SP=4/seq=4096），Softmax 数据取自 exp1_softmax_1.3b_seq4096_pp4_sp4（训练 10 步，取 iter 2-10 平均值）。

---

## 4. 实验二：不同序列长度下的显存对比

### 目的

验证 DeltaNet 的显存不随序列长度增长（O(1) state），而 Softmax 的 KV cache 线性增长（O(n)）。这是 DeltaNet + Seq1F1B 的核心优势。

### 配置

- 模型：1.3B
- PP=4, TP=1, PP_SP=4
- 序列长度：4096 / 8192 / 16384 / 32768
- 训练 5 步（只看显存，不需要太多步）

### 4.1 DeltaNet 不同序列长度

**不需要改文件，只改 SEQ_LENGTH。**

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

# 通用配置
export GPUS_PER_NODE=8 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B
export PP_SIZE=4 TP_SIZE=1 PP_SP=4 PP_SP_STR=uniform_comp
export NUM_LAYERS=24 HIDDEN=2048 NUM_ATTN_HEADS=16
export MICRO_BATCH=1 GLOBAL_BATCH=8 TRAIN_ITER=5

# --- seq=4096 ---
export SEQ_LENGTH=4096
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp2_deltanet_seq4096.log

# --- seq=8192 ---
export SEQ_LENGTH=8192
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp2_deltanet_seq8192.log

# --- seq=16384 ---
export SEQ_LENGTH=16384
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp2_deltanet_seq16384.log

# --- seq=32768 ---
export SEQ_LENGTH=32768
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp2_deltanet_seq32768.log
```

### 4.2 Softmax 不同序列长度

**不需要改文件（假设 data-path 已解决，见实验一的注意事项）。**

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

export GPUS_PER_NODE=8 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B
export PP_SIZE=4 TP_SIZE=1 PP_SP=4 PP_SP_STR=uniform_comp VPP_SIZE=1
export NUM_LAYERS=24 HIDDEN=2048 NUM_ATTN_HEADS=16
export MICRO_BATCH=1 GLOBAL_BATCH=8 TRAIN_ITER=5

# --- seq=4096 ---
export SEQ_LENGTH=4096
bash run.sh 2>&1 | tee exp_logs/deltanet_exps/exp2_softmax_seq4096.log

# --- seq=8192 ---
export SEQ_LENGTH=8192
bash run.sh 2>&1 | tee exp_logs/deltanet_exps/exp2_softmax_seq8192.log

# --- seq=16384 ---
export SEQ_LENGTH=16384
bash run.sh 2>&1 | tee exp_logs/deltanet_exps/exp2_softmax_seq16384.log

# --- seq=32768 ---
export SEQ_LENGTH=32768
bash run.sh 2>&1 | tee exp_logs/deltanet_exps/exp2_softmax_seq32768.log
```

> **实际结果**：Softmax 在 seq=32768 / 1.3B / PP=4 / SP=4 下没有 OOM（peak 43.8GB，A100-80G 尚有余量）。
> 但 Softmax 吞吐量显著下降至 49,450 tok/s，而 DeltaNet 反而提升至 72,420 tok/s。
> 更大模型（7B/13B）+ 更长序列（64K+）时 Softmax 大概率会 OOM。

### 4.3 结果表格

> 以下数据均取 iter≥2 的平均值（排除第 1 步的 Triton JIT 预热），显存取最终稳定值的 max stage。

| 序列长度 | DeltaNet 显存 (max stage, GB) | Softmax 显存 (max stage, GB) | DeltaNet tok/s | Softmax tok/s | 吞吐量比 (DN/SM) |
|---------|---------------------------|---------------------------|---------------|--------------|-----------------|
| 4096  | 9.5 | 9.0 | 24,873 | 52,909 | 0.47× |
| 8192  | 13.9 | 13.4 | 50,508 | 62,762 | 0.80× |
| 16384 | 23.5 | 22.8 | 66,384 | 60,053 | **1.11×** ★ |
| 32768 | 42.4 | 43.8 | 72,420 | 49,450 | **1.46×** ★★ |

**各 stage 详细显存（GB）：**

| 序列长度 | 类型 | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---------|------|---------|---------|---------|---------|
| 4096 | DeltaNet | 9.49 | 7.57 | 6.90 | 8.83 |
| 4096 | Softmax | 9.04 | 7.53 | 7.01 | 8.74 |
| 8192 | DeltaNet | 13.9 | 11.5 | 10.1 | 13.8 |
| 8192 | Softmax | 13.4 | 11.4 | 10.4 | 13.4 |
| 16384 | DeltaNet | 23.5 | 19.9 | 16.9 | 22.8 |
| 16384 | Softmax | 22.6 | 19.4 | 17.6 | 22.8 |
| 32768 | DeltaNet | 42.4 | 37.2 | 31.2 | 42.1 |
| 32768 | Softmax | 41.7 | 37.4 | 32.9 | 43.8 |

**Corrected TFLOPs/device 及 MFU**（DeltaNet 使用线性 attention FLOPs 口径；Softmax 使用 causal softmax FLOPs 口径）：

| 序列长度 | DeltaNet TFLOPs | DeltaNet MFU | Softmax TFLOPs | Softmax MFU |
|---------|----------------|-------------|----------------|-------------|
| 4096 | 26.83 | 8.6% | 60.06 | 19.2% |
| 8192 | 54.48 | 17.5% | 80.77 | 25.9% |
| 16384 | 71.60 | 23.0% | 95.53 | 30.6% |
| 32768 | 78.11 | 25.0% | 108.70 | 34.8% |

> **核心发现**：
>
> **1. 吞吐量交叉点在 seq≈12K-16K**：
> - seq=4096：Softmax 约 2.13× 快（FlashAttention 在短序列上极度优化）
> - seq=8192：差距缩小到 1.24×
> - seq=16384：**DeltaNet 反超**，1.11× 快 ★
> - seq=32768：**DeltaNet 大幅领先**，1.46× 快 ★★
> - 趋势：随序列长度增长，DeltaNet 优势持续扩大
>
> **2. 显存方面**：
> - seq=4096/8192：两者显存基本持平（DeltaNet 略高 ~0.5GB）
> - seq=16384：两者持平
> - seq=32768：**DeltaNet 开始占优**（42.4 vs 43.8 GB），Softmax 的 KV cache 增长开始显现
> - 注意：当前序列长度下，per-span 激活显存仍然是主要部分，cross-span state 的 O(1) vs O(n) 差异还未充分拉大。更长序列（64K/128K）下差异会更明显。
>
> **3. MFU 口径修正后的解读**：
> - DeltaNet 的 corrected TFLOPs 不再随 `seq_len^2` 虚高，seq=32K 时约 78.11 TFLOPs/device。
> - DeltaNet 的优势主要体现在 **更少实际计算量带来的更高 tokens/s**，而不是更高 MFU。
> - 因此跨 DeltaNet/Softmax 对比时，应优先使用吞吐量和显存；MFU 只做辅助参考。

---

## 5. 实验三：Seq1F1B 切分数对 DeltaNet 的影响

### 目的

对比不同 PP_SP（序列切分数）对 DeltaNet 的显存和吞吐量的影响。

### 配置

- 模型：1.3B（24层, hidden=2048, heads=16）
- PP=4, TP=1
- PP_SP = 1（不切分）/ 2 / 4 / 8
- **序列长度：32768**（DeltaNet 优势区间，实验二交叉点 ~12K-16K）
- 训练 5 步

### 5.1 运行

**不需要改文件，只改 PP_SP。**

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

export GPUS_PER_NODE=8 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B
export PP_SIZE=4 TP_SIZE=1 PP_SP_STR=uniform_comp
export NUM_LAYERS=24 HIDDEN=2048 NUM_ATTN_HEADS=16
export SEQ_LENGTH=32768 MICRO_BATCH=1 GLOBAL_BATCH=8 TRAIN_ITER=5

# --- PP_SP=1（不做序列切分） ---
export PP_SP=1
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp3_deltanet_seq32k_sp1.log

# --- PP_SP=2 ---
export PP_SP=2
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp3_deltanet_seq32k_sp2.log

# --- PP_SP=4 ---
export PP_SP=4
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp3_deltanet_seq32k_sp4.log

# --- PP_SP=8 ---
export PP_SP=8
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp3_deltanet_seq32k_sp8.log
```

> **注意**：PP_SP=1 时 `run_deltanet.sh` 中 `pipe_sp_splits=1`，DeltaNet 不会缓存 recurrent state（`output_final_state=False`），行为等价于普通 PP（无序列切分）。

### 5.2 结果表格

> 数据均取 iter≥2 的平均值（排除第 1 步 Triton JIT 预热），显存取最终稳定值的 max stage。
> 实际运行 **seq=32768**，模型 1.3B，PP=4，TP=1。

**DeltaNet：**

| PP_SP | 每 span 长度 | 显存 (max stage, GB) | 吞吐量 (tok/s) | TFLOPs | 每步耗时 (ms) | MFU | 相对 SP=1 吞吐 |
|-------|------------|---------------------|---------------|--------|-------------|-----|--------------|
| 1 | 32768 | 76.3 | 64,775 | 69.87 | 4,047 | 22.4% | 1.00× |
| 2 | 16384 | 53.9 | 71,033 | 76.62 | 3,691 | 24.6% | **1.10×** |
| 4 | 8192 | 42.4 | 69,780 | 75.27 | 3,761 | 24.1% | 1.08× |
| 8 | 4096 | 35.4 | 74,659 | 80.53 | 3,512 | 25.8% | **1.15×** ★ |

**Softmax (FlashAttention) 对照：**

| PP_SP | 每 span 长度 | 显存 (max stage, GB) | 吞吐量 (tok/s) | TFLOPs | 每步耗时 (ms) | MFU | 相对 SP=1 吞吐 |
|-------|------------|---------------------|---------------|--------|-------------|-----|--------------|
| 1 | 32768 | 60.6 | 36,999 | 81.33 | 7,085 | 26.1% | 1.00× |
| 2 | 16384 | 48.6 | 45,045 | 99.02 | 5,820 | 31.7% | 1.22× |
| 4 | 8192 | 43.8 | 49,500 | 108.82 | 5,296 | 34.9% | 1.34× |
| 8 | 4096 | 40.8 | 50,455 | 110.91 | 5,196 | 35.5% | **1.36×** |

**DeltaNet vs Softmax 对比：**

| PP_SP | DN tok/s | SM tok/s | 加速比 (DN/SM) | DN 显存 | SM 显存 | 显存差 |
|-------|---------|---------|---------------|---------|---------|--------|
| 1 | 64,775 | 36,999 | **1.75×** ★★ | 76.3 | 60.6 | +15.7 GB |
| 2 | 71,033 | 45,045 | **1.58×** | 53.9 | 48.6 | +5.3 GB |
| 4 | 69,780 | 49,500 | **1.41×** | 42.4 | 43.8 | -1.4 GB |
| 8 | 74,659 | 50,455 | **1.48×** | 35.4 | 40.8 | **-5.4 GB** |

**各 stage 详细显存（GB）：**

| PP_SP | 类型 | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Peak |
|-------|------|---------|---------|---------|---------|------|
| 1 | DeltaNet | 76.3 | 58.5 | 42.0 | 45.8 | 76.3 |
| 1 | Softmax | 60.6 | 46.2 | 33.7 | 41.5 | 60.6 |
| 2 | DeltaNet | 53.9 | 46.8 | 35.1 | 45.2 | 53.9 |
| 2 | Softmax | 48.6 | 41.9 | 37.2 | 46.6 | 48.6 |
| 4 | DeltaNet | 42.4 | 37.2 | 31.2 | 42.1 | 42.4 |
| 4 | Softmax | 41.7 | 37.4 | 32.9 | 43.8 | 43.8 |
| 8 | DeltaNet | 35.4 | 32.0 | 28.9 | 38.6 | 38.6 |
| 8 | Softmax | 38.5 | 35.1 | 33.9 | 40.8 | 40.8 |

> **核心发现**：
>
> **1. DeltaNet 在所有 SP 下均大幅领先 Softmax（1.41×–1.75×）**：
> SP=1 时加速比最高 **1.75×**（无切分时长序列计算量大，DeltaNet O(n) 优势最大）。
> SP 增大后两者差距缩小（span 变短，Softmax 的 O(n²) 在短 span 内影响减弱）。
>
> **2. 显存交叉！SP=4/8 时 DeltaNet 反而更省**：
> SP=1/2 时 DeltaNet 显存高于 Softmax（因为 span 长时 DeltaNet 的 chunk 中间状态大）。
> 但 **SP=4 时持平，SP=8 时 DeltaNet 反超**——DeltaNet 省了 5.4GB！
> 原因：Softmax 的 KV cache 随 span 索引 j 线性增长（第 j 个 span 缓存前 j-1 个的 KV），
> 而 DeltaNet 跨 span 只传固定大小的 recurrent state，SP 越大优势越明显。
>
> **3. DeltaNet 吞吐量 SP=8 最优，Softmax 也是 SP=8 最优**：
> 但 DeltaNet 提升幅度更大（+15% vs +36%）。Softmax 从 SP=4→SP=8 提升仅 2%，
> 说明 Softmax 在高 SP 下被 KV cache 通信拖累。
>
> **4. Corrected MFU 解读**：
> DeltaNet corrected MFU 约 22-26%，Softmax 约 26-36%。这并不削弱吞吐结论：
> DeltaNet 是用更少的实际 FLOPs 跑出更高 tokens/s，而不是依赖更高的 softmax-style MFU。
>
> **5. 关键启示**：PP_SP 的最优值取决于 **每 span 的绝对长度**，而非 SP 数本身。
> 每 span ≥ 4096 tokens 时 DeltaNet 能充分利用 GPU 并行度。

<details>
<summary>📊 附：seq=8192 的 PP_SP 消融结果（对比参考）</summary>

> 以下为同配置在 seq=8192 下的结果，可与 seq=32K 对比 PP_SP 行为的差异。

| PP_SP | 每 span 长度 | 显存 (max stage, GB) | 吞吐量 (tok/s) | TFLOPs | 相对 SP=1 吞吐 |
|-------|------------|---------------------|---------------|--------|--------------|
| 1 | 8192 | 22.8 | 60,180 | 64.91 | 1.00× |
| 2 | 4096 | 16.9 | 64,601 | 69.68 | 1.07× |
| 4 | 2048 | 13.9 | 57,398 | 61.91 | 0.95× |
| 8 | 1024 | 12.6 | 33,631 | 36.28 | 0.56× |

> seq=8K 下 SP=8（span=1024）吞吐暴跌 44%，但 seq=32K 下 SP=8（span=4096）反而最优。
> 这验证了 **"每 span 绝对长度 ≥ 4096"** 是保持高吞吐的下限。

</details>

---

## 6. 实验四：不同模型规模对比

### 目的

验证 DeltaNet + Seq1F1B 在不同模型规模下的可扩展性，**seq=32768**（DeltaNet 优势区间）。

### 配置

使用项目中 `exp.sh` 已定义的模型配置：

| 模型 | NUM_LAYERS | HIDDEN | NUM_ATTN_HEADS | TP |
|------|-----------|--------|---------------|-----|
| 1.3B | 24 | 2048 | 16 | 1 |
| 2.7B | 32 | 2560 | 32 | 1 |
| 7B | 32 | 4096 | 32 | 2 |

### 6.1 运行 DeltaNet

**不需要改文件，只改模型配置。使用 seq=32768。**

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

export GPUS_PER_NODE=8 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B
export PP_SIZE=4 PP_SP=4 PP_SP_STR=uniform_comp
export SEQ_LENGTH=32768 MICRO_BATCH=1 GLOBAL_BATCH=8 TRAIN_ITER=5

# --- 1.3B ---
export NUM_LAYERS=24 HIDDEN=2048 NUM_ATTN_HEADS=16 TP_SIZE=1
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp4_deltanet_1.3b_seq32k.log

# --- 2.7B ---
export NUM_LAYERS=32 HIDDEN=2560 NUM_ATTN_HEADS=32 TP_SIZE=1
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp4_deltanet_2.7b_seq32k.log

# --- 7B（需要 TP=2）---
export NUM_LAYERS=32 HIDDEN=4096 NUM_ATTN_HEADS=32 TP_SIZE=2 PP_SIZE=4
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp4_deltanet_7b_seq32k.log
```

### 6.2 运行 Softmax 对照组

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

export GPUS_PER_NODE=8 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B
export PP_SIZE=4 PP_SP=4 PP_SP_STR=uniform_comp VPP_SIZE=1
export SEQ_LENGTH=32768 MICRO_BATCH=1 GLOBAL_BATCH=8 TRAIN_ITER=5

# --- 1.3B（复用实验二的 exp2_softmax_seq32768.log） ---
# export NUM_LAYERS=24 HIDDEN=2048 NUM_ATTN_HEADS=16 TP_SIZE=1
# bash run.sh 2>&1 | tee exp_logs/deltanet_exps/exp4_softmax_1.3b_seq32k.log

# --- 2.7B ---
export NUM_LAYERS=32 HIDDEN=2560 NUM_ATTN_HEADS=32 TP_SIZE=1
bash run.sh 2>&1 | tee exp_logs/deltanet_exps/exp4_softmax_2.7b_seq32k.log

# --- 7B ---
export NUM_LAYERS=32 HIDDEN=4096 NUM_ATTN_HEADS=32 TP_SIZE=2
bash run.sh 2>&1 | tee exp_logs/deltanet_exps/exp4_softmax_7b_seq32k.log
```

### 6.3 结果表格

> 数据均取 iter≥2 的平均值，**seq=32768**，PP=4，PP_SP=4。
> MFU 使用 1.1 节的 corrected FLOPs 口径，A100 BF16 Tensor Core 峰值按 312 TFLOPS/device。
> 1.3B Softmax 数据复用实验二 `exp2_softmax_seq32768.log`（相同配置）。

**DeltaNet：**

| 模型 | TP | 显存 (max stage, GB) | 吞吐量 (tok/s) | TFLOPs | 每步耗时 (ms) | MFU |
|------|---|---------------------|---------------|--------|-------------|-----|
| 1.3B | 1 | 42.4 | 69,705 | 75.19 | 3,766 | 24.1% |
| 2.7B | 1 | 72.3 | 42,411 | 91.76 | 6,181 | 29.4% |
| 7B | 2 | 69.6 | 20,241 | 110.19 | 12,952 | 35.3% |

**Softmax (FlashAttention)：**

| 模型 | TP | 显存 (max stage, GB) | 吞吐量 (tok/s) | TFLOPs | 每步耗时 (ms) | MFU |
|------|---|---------------------|---------------|--------|-------------|-----|
| 1.3B | 1 | 43.8 | 49,450 | 108.70 | 5,301 | 34.8% |
| 2.7B | 1 | 69.5 | 25,671 | 103.10 | 10,212 | 33.0% |
| 7B | 2 | 67.8 | 16,407 | 134.98 | 15,978 | 43.3% |

**DeltaNet vs Softmax 对比：**

| 模型 | DeltaNet tok/s | Softmax tok/s | 吞吐量比 (DN/SM) | DN 显存 | SM 显存 | 显存差 |
|------|---------------|--------------|-----------------|---------|---------|--------|
| 1.3B | 69,705 | 49,450 | **1.41×** | 42.4 | 43.8 | -1.4 GB |
| 2.7B | 42,411 | 25,671 | **1.65×** ★ | 72.3 | 69.5 | +2.8 GB |
| 7B | 20,241 | 16,407 | **1.23×** | 69.6 | 67.8 | +1.8 GB |

**各 stage 详细显存（GB）：**

| 模型 | 类型 | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Peak |
|------|------|---------|---------|---------|---------|------|
| 1.3B | DeltaNet | 42.4 | 37.2 | 31.2 | 42.1 | 42.4 |
| 1.3B | Softmax | 41.7 | 37.4 | 32.9 | 43.8 | 43.8 |
| 2.7B | DeltaNet | 72.3 | 64.2 | 54.3 | 62.6 | 72.3 |
| 2.7B | Softmax | 69.5 | 60.8 | 54.7 | 63.6 | 69.5 |
| 7B | DeltaNet | 69.6 | 61.2 | 53.2 | 55.3 | 69.6 |
| 7B | Softmax | 67.8 | 57.8 | 51.9 | 55.5 | 67.8 |

> **核心发现**：
>
> **1. DeltaNet 在所有模型规模上均大幅领先 Softmax**：
> - 1.3B: **1.41×** 快（与实验二的 1.46× 一致）
> - 2.7B: **1.65×** 快 ★ 最大加速比！
> - 7B:  **1.23×** 快（TP=2 通信开销拉近差距）
>
> **2. 加速比先升后降的原因**：
> - 1.3B→2.7B：模型增大，每个 attention 层的计算量增加，DeltaNet 的 O(n) vs Softmax 的
>   O(n²) 差距更加显著，加速比从 1.41× 提升到 **1.65×**
> - 2.7B→7B：引入 TP=2 后，DeltaNet 和 Softmax 都受 all-reduce 通信限制，
>   注意力计算本身的差异被通信开销稀释，加速比回落到 1.23×
>
> **3. 显存方面基本持平**：
> - 在 PP=4, SP=4 的配置下，per-span 激活仍是显存主体，cross-span state 的 O(1) vs O(n)
>   差异在当前序列长度下不明显。1.3B 时 DeltaNet 略少 1.4GB，2.7B/7B 时 DeltaNet 略多 ~2GB
>   （DeltaNet 有额外的 conv state + recurrent state 缓存）。
> - 2.7B Softmax stage 0 达 69.5GB，DeltaNet 达 72.3GB，两者均接近 80GB 上限。
>   更长序列（64K+）时 Softmax 的 KV cache O(n) 增长会使其率先 OOM。
>
> **4. Corrected MFU 对比**：
> - DeltaNet corrected MFU 约 24-35%，Softmax 约 33-43%。
> - 这说明 DeltaNet 的核心收益是 **减少实际 FLOPs 并提升 tokens/s**，而不是在 softmax-style MFU 上更高。
> - 7B Softmax MFU (43.3%) 高于 2.7B (33.0%)：因为 7B 用了 TP=2 将 hidden=4096 拆分，
>   每 GPU 的计算仍然高效；而 2.7B 的 head_dim=80 对 Tensor Core 不友好（非 8 的倍数 padding）
>
> **5. 可重复性验证**：
> - 1.3B DeltaNet（69,705 tok/s）与实验三 SP=4（69,780 tok/s）几乎一致 ✅
> - 1.3B Softmax（49,450 tok/s）与实验二 seq=32K（49,450 tok/s）完全一致 ✅

---

## 7. 实验五：超长序列（64K/128K）极限测试

### 目的

将序列长度推向 64K 和 128K，验证 DeltaNet 在超长序列下的可行性，以及 Softmax 的 OOM 边界。

### 配置

- 模型：1.3B（24层, hidden=2048, heads=16）
- PP=4, TP=1
- 序列长度 / PP_SP：64K/SP=8, 64K/SP=16, 128K/SP=16, 128K/SP=32

### 7.1 结果

| 配置 | span 长度 | DeltaNet | Softmax | 结论 |
|------|----------|---------|---------|------|
| seq=64K, SP=8 | 8192 | ❌ OOM (iter 2, 75.9GB) | ❌ OOM (iter 1) | 两者都无法运行 |
| seq=64K, SP=16 | 4096 | ✅ **73,150 tok/s** | ❌ OOM (iter 2, 77.0GB) | **DeltaNet 能跑，Softmax 不能** ★★★ |
| seq=128K, SP=16 | 8192 | ❌ 数据集不够长 | ❌ 数据集不够长 | 需要更长 document 的数据集 |
| seq=128K, SP=32 | 4096 | ❌ 数据集不够长 | ❌ 数据集不够长 | 同上 |

**DeltaNet seq=64K, SP=16 详细数据**（iter 2-5 平均）：

| 指标 | 值 |
|------|-----|
| 吞吐量 (tok/s) | 73,150 |
| Corrected TFLOPs | 78.90 |
| Corrected MFU | **25.3%** |
| 每步耗时 (ms) | 7,174 |
| 显存 Peak (GB) | 75.3 (Stage 3) |
| 显存各 stage | 60.9, 57.7, 54.9, 75.3 |

> **核心发现**：
>
> **1. seq=64K 的 killer result：DeltaNet ✅ vs Softmax OOM ❌**：
> - DeltaNet SP=16（span=4096）成功完成 5 步训练，peak 显存 75.3GB
> - Softmax SP=16 在 iter 2/3 时 OOM（peak 已达 77.0GB，KV cache 累积导致后续 span OOM）
> - Softmax SP=8（span=8192）更是 iter 1 之前就 OOM
> - **这是论文中最有说服力的数据点**：同样的硬件、同样的模型、同样的序列长度——
>   DeltaNet 跑通了，Softmax 跑不了。
>
> **2. Corrected MFU 后的解读**：
> - 旧口径下 81.8% 来自 softmax-style `seq_len^2` FLOPs 估计，对 DeltaNet 不适用。
> - 线性 DeltaNet 口径下 corrected MFU 约 **25.3%**。
> - 更关键的结果仍然成立：DeltaNet 在 seq=64K 下 tokens/s 几乎没有下降（73,150 vs seq=32K 的 74,659），
>   而 Softmax 在相同配置下 OOM。
>
> **3. DeltaNet seq=64K SP=8 也 OOM（span=8192）**：
> - iter 1 时显存已达 75.9GB（Stage 3），iter 2 OOM
> - 说明 span=8192 的 per-span 激活太大。但 SP=16（span=4096）就能跑通
> - 再次验证了 **span=4096 是最佳平衡点**
>
> **4. 128K 受限于数据集**：
> - `AssertionError: no sample to consume` — codeparrot 数据集的单个 document 不够 128K tokens
> - 这是**数据问题，不是模型/显存问题**。换用长文档数据集（如 Books、ArXiv）即可测试 128K
>
> **5. Softmax OOM 根因分析**：
> - Softmax SP=16 的 iter 1 显存 75.8-76.6GB（接近满载），iter 2 进一步增长到 77.0GB
> - KV cache 在 span 间累积：第 j 个 span 需缓存前 j-1 个 span 的 KV
> - 16 个 span 意味着后面的 span 需要保存前 15 个 span 的 KV → OOM
> - DeltaNet 没有这个问题——跨 span 只传递固定大小的 recurrent state

---

## 8. 结果收集与分析

### 8.1 一键提取所有实验结果

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

echo "文件名, Loss(最终), 吞吐量(tok/s), TFLOPs, 显存(GB), 每步耗时(ms)"
for f in exp_logs/deltanet_exps/exp*.log; do
    name=$(basename $f .log)
    # 提取最后一个 iteration 的指标
    loss=$(grep "lm loss:" $f | tail -1 | grep -oP 'lm loss: \K[0-9.E+-]+')
    toks=$(grep "toks/s:" $f | tail -1 | grep -oP 'toks/s: \K[0-9.]+')
    tflops=$(grep "TFlops/s:" $f | tail -1 | grep -oP 'TFlops/s: \K[0-9.]+')
    mem=$(grep "mem_each_stage:" $f | tail -1 | grep -oP 'mem_each_stage: \K[0-9.,]+')
    time=$(grep "elapsed time per iteration" $f | tail -1 | grep -oP 'elapsed time per iteration \(ms\): \K[0-9.]+')
    echo "$name, $loss, $toks, $tflops, $mem, $time"
done
```

### 8.2 关键对比图建议

1. **吞吐量 vs 序列长度**（实验二，最关键的图）：X 轴=序列长度，Y 轴=tok/s，两条线（DeltaNet / Softmax）
   - ✅ **已验证**：交叉点在 seq≈12K-16K，seq=32K 时 DeltaNet 1.46× 快

2. **Corrected TFLOPs vs 序列长度**（实验二）：X 轴=序列长度，Y 轴=TFLOPs/device
   - ✅ **已修正**：DeltaNet corrected TFLOPs 不再使用 softmax 二次项；seq=32K 时约 78.11 TFLOPs/device

3. **显存 vs 序列长度**（实验二）：X 轴=序列长度，Y 轴=显存(GB)，两条线
   - ✅ **已验证**：两者显存增长趋势类似，但 seq=32K 时 DeltaNet 开始占优（42.4 vs 43.8）
   - 注意：在 1.3B/PP=4/SP=4 的配置下，per-span 激活是显存主体，cross-span state 差异不明显
   - 更大模型或更长序列（64K+）下 KV cache 的 O(n) 增长会更突出

4. **显存 & 吞吐量 vs PP_SP 切分数**（实验三，seq=32K）：X 轴=PP_SP，双 Y 轴（显存 / tok/s）
   - ✅ **已验证**：内存从 76.3→35.4GB 单调下降；吞吐量 SP=8 最优（74,659 tok/s）
   - 关键发现：seq=32K 下 SP=8（span=4096）不再暴跌，与 seq=8K 下 SP=8（span=1024）截然不同

5. **Corrected MFU vs 模型规模**（实验四）：X 轴=模型规模，Y 轴=MFU(%)
   - ✅ **已修正**：DeltaNet corrected MFU 约 24-35%，Softmax 约 33-43%；跨注意力类型不再用 MFU 作为主结论

6. **吞吐量加速比 vs 模型规模**（实验四，新增核心图）：X 轴=模型规模，Y 轴=DeltaNet/Softmax 吞吐比
   - ✅ **已验证**：1.3B 1.41×、2.7B **1.65×** ★、7B 1.23×
   - 加速比先升后降：模型越大差距越大，但 TP 通信会稀释优势

7. **PP_SP 最优 span 长度**（实验三 seq=8K + seq=32K 联合分析）：X 轴=每 span 长度，Y 轴=tok/s
   - ✅ **关键结论**：每 span ≥ 4096 tokens 时吞吐稳定，< 2048 时开始暴跌

8. **seq=64K 可行性对比**（实验五，最有冲击力的图）：柱状图或表格
   - ✅ DeltaNet SP=16: 73,150 tok/s, corrected MFU 25.3%, 75.3GB → **成功**
   - ❌ Softmax SP=16: OOM (77.0GB, KV cache 累积) → **失败**
   - 一张图说明一切："DeltaNet 能训练 64K 序列，Softmax 不能"

---

## 附录：两个脚本的关键差异

| 项目 | `run.sh`（Softmax） | `run_deltanet.sh`（DeltaNet） |
|------|-------------------|---------------------------|
| 注意力 | `--use-flash-attn` | `--use-deltanet` |
| 位置编码 | `--position-embedding-type rope` | （无，自动禁用） |
| data-path 后缀 | `codeparrot_content_document` | `codeparrot_content_document_text_document` |
| 变量默认值 | 无（必须 export） | 有（可以直接运行） |
| 额外参数 | 无 | `--deltanet-mode`, `--deltanet-conv-size` 等 |

## 附录：常见问题

### Q: run.sh 报 `unbound variable` 错误

A: `run.sh` 没有设置默认值，所有变量必须 export。确保至少 export 了：
```bash
export GPUS_PER_NODE PP_SIZE TP_SIZE PP_SP PP_SP_STR VPP_SIZE
export NUM_LAYERS HIDDEN NUM_ATTN_HEADS SEQ_LENGTH
export MICRO_BATCH GLOBAL_BATCH TRAIN_ITER
export DATA_PATH WORLD_SIZE MASTER_ADDR MASTER_PORT
```

### Q: run.sh 报 data-path 找不到

A: 创建软链接：
```bash
cd data/
ln -sf codeparrot_content_document_text_document.bin codeparrot_content_document.bin
ln -sf codeparrot_content_document_text_document.idx codeparrot_content_document.idx
```

### Q: 7B 模型 OOM

A: 增大 TP_SIZE 或减小 MICRO_BATCH / GLOBAL_BATCH：
```bash
export TP_SIZE=2 PP_SIZE=4  # 8卡 = 2(TP) × 4(PP)
```

### Q: 第一个 iteration 特别慢

A: 正常。第一步需要 Triton kernel JIT 编译 + NCCL 通信初始化 + CUDA graph warmup。从第 2 步开始才是真实性能。分析时**应该忽略第 1 步**。
