# DeltaNet + Seq1F1B 实验指南

## 目录

1. [实验概述](#1-实验概述)
2. [环境准备](#2-环境准备)
3. [实验一：DeltaNet vs Softmax 基线对比](#3-实验一deltanet-vs-softmax-基线对比)
4. [实验二：不同序列长度下的显存对比](#4-实验二不同序列长度下的显存对比)
5. [实验三：Seq1F1B 切分数对 DeltaNet 的影响](#5-实验三seq1f1b-切分数对-deltanet-的影响)
6. [实验四：不同模型规模对比](#6-实验四不同模型规模对比)
7. [结果收集与分析](#7-结果收集与分析)

---

## 1. 实验概述

### 目标

验证 DeltaNet（线性注意力）与 Seq1F1B（序列级流水线并行）结合后，相比原版 Softmax Attention + Seq1F1B 的优势：

| 对比维度 | 预期结果 |
|---------|---------|
| **显存** | DeltaNet 的 recurrent state 是 O(1)，Softmax 的 KV cache 是 O(n)，长序列下 DeltaNet 显存更低 |
| **吞吐量** | DeltaNet 无 softmax 计算，理论上更快，但 chunk kernel 有额外开销 |
| **Loss 收敛** | 两者架构不同，loss 绝对值不可直接比较，但各自应该正常下降 |

### 核心变量

- **注意力类型**：DeltaNet（`--use-deltanet`） vs Softmax（`--use-flash-attn`）
- **序列长度**：4096 / 8192 / 16384 / 32768
- **Seq1F1B 切分数**：PP_SP = 1（不切分）/ 4 / 8
- **模型规模**：1.3B / 2.7B / 7B

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

| 指标 | DeltaNet | Softmax |
|------|----------|---------|
| Loss (iter 10) | | |
| 吞吐量 (tok/s) | | |
| TFLOPs | | |
| 显存 stage 0 (GB) | | |
| 显存 stage 1 (GB) | | |
| 显存 stage 2 (GB) | | |
| 显存 stage 3 (GB) | | |
| 每步耗时 (ms) | | |

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

> **预期**：Softmax 在 seq=32768 时可能 OOM，这正好说明 DeltaNet 的优势。

### 4.3 结果表格

| 序列长度 | DeltaNet 显存 (max stage, GB) | Softmax 显存 (max stage, GB) | DeltaNet tok/s | Softmax tok/s |
|---------|---------------------------|---------------------------|---------------|--------------|
| 4096  | | | | |
| 8192  | | | | |
| 16384 | | | | |
| 32768 | | | | |

提取命令：

```bash
for f in exp_logs/deltanet_exps/exp2_*.log; do
    echo "=== $(basename $f) ==="
    grep "mem_each_stage" $f | tail -1
    grep "toks/s" $f | tail -1
done
```

---

## 5. 实验三：Seq1F1B 切分数对 DeltaNet 的影响

### 目的

对比不同 PP_SP（序列切分数）对 DeltaNet 的显存和吞吐量的影响。

### 配置

- 模型：1.3B
- PP=4, TP=1, seq=8192
- PP_SP = 1（不切分）/ 2 / 4 / 8
- 训练 5 步

### 5.1 运行

**不需要改文件，只改 PP_SP。**

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

export GPUS_PER_NODE=8 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B
export PP_SIZE=4 TP_SIZE=1 PP_SP_STR=uniform_comp
export NUM_LAYERS=24 HIDDEN=2048 NUM_ATTN_HEADS=16
export SEQ_LENGTH=8192 MICRO_BATCH=1 GLOBAL_BATCH=8 TRAIN_ITER=5

# --- PP_SP=1（不做序列切分） ---
export PP_SP=1
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp3_deltanet_sp1.log

# --- PP_SP=2 ---
export PP_SP=2
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp3_deltanet_sp2.log

# --- PP_SP=4 ---
export PP_SP=4
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp3_deltanet_sp4.log

# --- PP_SP=8 ---
export PP_SP=8
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp3_deltanet_sp8.log
```

> **注意**：PP_SP=1 时 `run_deltanet.sh` 中 `pipe_sp_splits=1`，DeltaNet 不会缓存 recurrent state（`output_final_state=False`），行为等价于普通 DeltaNet。

### 5.2 结果表格

| PP_SP | 显存 (max stage, GB) | 吞吐量 (tok/s) | 每步耗时 (ms) |
|-------|---------------------|---------------|-------------|
| 1 | | | |
| 2 | | | |
| 4 | | | |
| 8 | | | |

---

## 6. 实验四：不同模型规模对比

### 目的

验证 DeltaNet + Seq1F1B 在不同模型规模下的可扩展性。

### 配置

使用项目中 `exp.sh` 已定义的模型配置：

| 模型 | NUM_LAYERS | HIDDEN | NUM_ATTN_HEADS |
|------|-----------|--------|---------------|
| 1.3B | 24 | 2048 | 16 |
| 2.7B | 32 | 2560 | 32 |
| 7B | 32 | 4096 | 32 |

### 6.1 运行

**不需要改文件，只改模型配置。**

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

export GPUS_PER_NODE=8 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B
export PP_SIZE=4 TP_SIZE=1 PP_SP=4 PP_SP_STR=uniform_comp
export SEQ_LENGTH=4096 MICRO_BATCH=1 GLOBAL_BATCH=8 TRAIN_ITER=5

# --- 1.3B ---
export NUM_LAYERS=24 HIDDEN=2048 NUM_ATTN_HEADS=16
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp4_deltanet_1.3b.log

# --- 2.7B ---
export NUM_LAYERS=32 HIDDEN=2560 NUM_ATTN_HEADS=32
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp4_deltanet_2.7b.log

# --- 7B（可能需要 TP=2 才能放下）---
export NUM_LAYERS=32 HIDDEN=4096 NUM_ATTN_HEADS=32 TP_SIZE=2 PP_SIZE=4
bash run_deltanet.sh 2>&1 | tee exp_logs/deltanet_exps/exp4_deltanet_7b.log
```

> **注意**：7B 模型在 8 卡 A100-80G 上 TP=1, PP=4 可能显存不够。如果 OOM，改为 TP=2, PP=4（使用全部 8 卡）。
> 如果还是 OOM，可以减小 GLOBAL_BATCH 或增加 PP_SIZE。

---

## 7. 结果收集与分析

### 7.1 一键提取所有实验结果

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

### 7.2 关键对比图建议

1. **显存 vs 序列长度**（实验二）：X 轴=序列长度，Y 轴=显存(GB)，两条线（DeltaNet / Softmax）
   - 预期：Softmax 线性上升，DeltaNet 基本平坦

2. **吞吐量 vs 序列长度**（实验二）：X 轴=序列长度，Y 轴=tok/s
   - 预期：Softmax 下降更快（O(n²) attention），DeltaNet 下降更慢（O(n) recurrent）

3. **显存 vs PP_SP 切分数**（实验三）：X 轴=PP_SP，Y 轴=显存(GB)
   - 预期：切分越多，每个 span 的激活显存越小

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
