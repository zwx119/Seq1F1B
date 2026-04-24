# DeltaNet + Seq1F1B 正式实验设计

## 1. 实验目标

这轮实验的核心问题只有一个：

> 在 DeltaNet 长序列预训练中，Seq1F1B 是否相比普通 pipeline parallelism 带来稳定的吞吐和显存收益，同时保持训练数学等价？

因此主实验只比较两类配置：

| 名称 | 含义 | 关键参数 |
|------|------|----------|
| DeltaNet + PP | 普通 pipeline baseline | `--pipe-sp-splits 1` |
| DeltaNet + Seq1F1B | 序列切分流水 | `--pipe-sp-splits N, N > 1` |

Softmax / FlashAttention 只作为背景材料，不再作为这份正式报告的主线。

## 2. 当前已验证结论

### 2.1 SP8 代码路径已通过 FineWeb-Edu 长训 sanity

已完成的 8 卡 FineWeb-Edu 运行：

```text
OUT_DIR = tests/fineweb_outputs_long_8g
model   = 24 layers, hidden 1024, heads 16, FFN 4096
params  = ~431M total, ~328M non-embedding
seq_len = 16384
PP      = 8
SP      = 1 vs 8
iters   = 3000
global batch = 16
```

关键结果：

| 指标 | SP1 | SP8 | 结论 |
|------|-----|-----|------|
| final train loss | 3.574157 | 3.576120 | 对齐 |
| final validation loss | 3.533869 | 3.536217 | 对齐 |
| final validation PPL | ~34.26 | ~34.34 | 对齐 |
| grad norm | 曲线贴合 | 曲线贴合 | 反向动态一致 |
| skipped / NaN iter | 0 / 0 | 0 / 0 | 稳定 |

这说明当前实现中的训练、eval、loss scaling、pipeline 通信顺序，在这个配置下已经可信。

### 2.2 这组 sanity 不用于证明性能优势

当前 sanity 配置对 Seq1F1B 不友好：

```text
hidden = 1024
seq_len = 16384
PP = 8
SP8 span_len = 2048
每个 PP stage 只有 3 层
```

实测吞吐：

| 配置 | 稳态吞吐 | 结论 |
|------|----------|------|
| SP1 | ~133K tok/s | 更快 |
| SP8 | ~80K tok/s | 被过度切分拖慢 |

这组结果应写成：**SP8 correctness 通过，但当前小模型、短 span、PP=8 不是 Seq1F1B 的性能优势区间。**

### 2.3 目标配置 smoke 已得到正结果

在更接近旧实验优势区间的配置上，`GLOBAL_BATCH=8` 的 SP1 会在 80GB A100 上贴边 OOM。因此先把全局 batch 降到 4，保持模型规模和序列长度不变：

```text
model   = 24 layers, hidden 2048, heads 16, FFN 8192
seq_len = 32768
PP      = 4
DP      = 2
GBS     = 4
iters   = 200
```

200-step smoke 结果：

| 配置 | final toks/s | step time | peak mem | final train loss |
|------|--------------|-----------|----------|------------------|
| SP1 | ~47.3K | ~2.77s | 45.6GB | 6.102583 |
| SP8 | ~64.9K | ~2.02s | 33.7GB | 6.024371 |

结论：

```text
SP8 throughput ≈ 1.36x SP1
SP8 peak memory 下降约 11.9GB，约 26%
SP1 与 SP8 loss 都正常下降
```

这已经是正式实验线上的第一个性能正结果。下一步应补齐 `SP=2/4/16` 的 span sweep，然后用最优 SP 与 SP1 做长训。

## 3. 正式假设

正式实验验证四个假设：

| 假设 | 内容 | 验证方式 |
|------|------|----------|
| H1 | Seq1F1B 不改变 DeltaNet 训练数学 | train loss / val loss / grad norm 对齐 |
| H2 | Seq1F1B 显著降低长序列显存 | 比较 `mem_each_stage` peak |
| H3 | 在合适配置下 Seq1F1B 可提升吞吐 | 比较 steady-state `toks/s` |
| H4 | 过度切分会伤吞吐 | span sweep 找最佳 `span_len` |

经验上最值得押注的区域：

```text
PP = 4
DP = 2
hidden = 2048
seq_len = 32768
global batch = 4
Seq1F1B SP = 8
span_len = 4096
```

理由：

| 条件 | 解释 |
|------|------|
| `span_len >= 4096` | 避免每段太短导致调度和通信 overhead 主导 |
| `PP=4` | 每个 stage 约 6 层，比 `PP=8` 的 3 层更厚 |
| `seq_len=32768` | 足够长，普通 PP bubble 和显存压力更明显 |
| `hidden=2048` | 计算更厚，更接近旧实验中 Seq1F1B 有收益的区间 |
| `global_batch=4` | 当前环境下 SP1 可稳定跑通；`global_batch=8` 对 SP1 已接近/超过 80GB 边界 |

## 4. 主指标和判据

### 4.1 Correctness 指标

| 指标 | 通过标准 |
|------|----------|
| `lm loss` | SP1 与 SPN 曲线贴合，无系统性偏移 |
| `lm loss validation` | 每个 eval 点接近，最终差距最好 `< 0.03` |
| `grad-norm` | 趋势一致，无异常尖峰或发散 |
| skipped iter | 必须为 0 |
| NaN iter | 必须为 0 |
| eval | 能正常跑完，不 hang，不翻倍 |

### 4.2 Performance 指标

| 指标 | 解释 |
|------|------|
| `toks/s` | 主吞吐指标，比 `TFlops/s` 更可靠 |
| `elapsed time per iteration` | 每步耗时 |
| `mem_each_stage` | 每个 PP stage 的显存 |
| peak memory | 判断是否打开更长序列空间 |

性能判定：

| 结果 | 解释 |
|------|------|
| SPN 更快且显存更低 | 强正结果 |
| SPN 接近 SP1 且显存显著更低 | memory scaling 正结果 |
| SPN 更慢但显存更低 | 需要调 span / PP / 模型厚度 |
| SPN 更慢且显存不降 | 配置或实现需要重新检查 |

### 4.3 MFU 只放附录

旧日志中的 `TFlops/s` 使用 Megatron softmax attention 风格公式，会把 DeltaNet 的线性 attention 按二次项高估。正式结论优先使用：

```text
tokens/s
step time
peak memory
loss / validation loss alignment
```

corrected MFU 可以放附录，但不作为主结论。

## 5. 实验阶段

### 阶段 S0：已完成 sanity

目的：证明当前代码路径能在真实 FineWeb-Edu 长训中跑通，并且 SP8 与 SP1 loss 对齐。

状态：已完成。

结果路径：

```text
tests/fineweb_outputs_long_8g/log_fineweb_sp1.txt
tests/fineweb_outputs_long_8g/log_fineweb_sp8.txt
tests/fineweb_outputs_long_8g/tb/fineweb_sp1
tests/fineweb_outputs_long_8g/tb/fineweb_sp8
```

报告中这组实验只承担 correctness sanity，不承担性能结论。

### 阶段 E1：目标配置 smoke test

目的：先在最可能有收益的配置上跑短测，确认不会 OOM、不会 hang、loss 正常。

推荐命令：

```bash
OUT_DIR=tests/exp_seqpp_pp4_h2048_seq32k_gbs4_smoke \
GPUS_PER_NODE=8 PP_SIZE=4 SEQ1F1B_SP=8 \
NUM_LAYERS=24 HIDDEN=2048 NUM_HEADS=16 \
SEQ_LEN=32768 MICRO_BATCH=1 GLOBAL_BATCH=4 \
TRAIN_ITERS=200 WARMUP_ITERS=20 \
EVAL_ITERS=0 SAVE_INTERVAL=100000 \
ONLY=both bash tests/run_fineweb_long.sh
```

配置解释：

| 项 | 值 |
|----|----|
| GPUs | 8 |
| TP | 1 |
| PP | 4 |
| DP | 2 |
| layers | 24 |
| hidden | 2048 |
| heads | 16 |
| FFN | 8192 |
| seq_len | 32768 |
| global batch | 4 |
| tokens / step | 131K |
| Seq1F1B SP | 8 |
| span_len | 4096 |

通过标准：

| 项 | 标准 |
|----|------|
| SP1 | 能跑完 200 iter |
| SP8 | 能跑完 200 iter |
| loss | 两条曲线无明显偏移 |
| memory | SP8 peak memory 明显低于 SP1 |
| throughput | SP8 不应明显慢于 SP1；若更快则进入长训 |

### 阶段 E2：目标配置正式长训

目的：在目标配置上给出正式 loss / validation / throughput / memory 结论。

推荐命令：

```bash
OUT_DIR=tests/exp_seqpp_pp4_h2048_seq32k_gbs4_long \
GPUS_PER_NODE=8 PP_SIZE=4 SEQ1F1B_SP=8 \
NUM_LAYERS=24 HIDDEN=2048 NUM_HEADS=16 \
SEQ_LEN=32768 MICRO_BATCH=1 GLOBAL_BATCH=4 \
TRAIN_ITERS=3000 WARMUP_ITERS=200 \
EVAL_INTERVAL=200 EVAL_ITERS=20 SAVE_INTERVAL=1000 \
ONLY=both bash tests/run_fineweb_long.sh
```

这不是完整 Chinchilla-optimal 预训练，而是一个受预算约束的 pretraining benchmark：

```text
tokens / step = 4 * 32768 = 131K
3000 steps    = 393M tokens
```

它足够比较 SP1 与 SP8 的训练动态和系统性能，但不宣称模型已经充分训练。

成功标准：

| 指标 | 期望 |
|------|------|
| train loss | SP1 / SP8 曲线贴合 |
| validation loss | 每 200 iter 的 eval 点贴合 |
| grad norm | 曲线趋势一致 |
| SP8 throughput | 相比 SP1 更快，或至少接近 |
| SP8 memory | 相比 SP1 明显降低 |

如果 E2 得到强正结果，这就是正式报告主结果。

### 阶段 E3：span length 消融

目的：如果 E2 中 SP8 没有明显吞吐优势，或者想找最优切分粒度，就做 span sweep。

固定配置：

```text
GPUs = 8
PP = 4
DP = 2
hidden = 2048
seq_len = 32768
global batch = 4
```

比较：

| SP | span_len |
|----|----------|
| 1 | 32768 |
| 2 | 16384 |
| 4 | 8192 |
| 8 | 4096 |
| 16 | 2048 |

建议每个先跑 200 iter：

```bash
OUT_DIR=tests/exp_span_sweep_gbs4_sp1 \
GPUS_PER_NODE=8 PP_SIZE=4 SEQ1F1B_SP=8 \
NUM_LAYERS=24 HIDDEN=2048 NUM_HEADS=16 \
SEQ_LEN=32768 MICRO_BATCH=1 GLOBAL_BATCH=4 \
TRAIN_ITERS=200 WARMUP_ITERS=20 EVAL_ITERS=0 SAVE_INTERVAL=100000 \
ONLY=sp1 bash tests/run_fineweb_long.sh
```

```bash
OUT_DIR=tests/exp_span_sweep_gbs4_sp2 \
GPUS_PER_NODE=8 PP_SIZE=4 SEQ1F1B_SP=2 \
NUM_LAYERS=24 HIDDEN=2048 NUM_HEADS=16 \
SEQ_LEN=32768 MICRO_BATCH=1 GLOBAL_BATCH=4 \
TRAIN_ITERS=200 WARMUP_ITERS=20 EVAL_ITERS=0 SAVE_INTERVAL=100000 \
ONLY=seq bash tests/run_fineweb_long.sh
```

```bash
OUT_DIR=tests/exp_span_sweep_gbs4_sp4 \
GPUS_PER_NODE=8 PP_SIZE=4 SEQ1F1B_SP=4 \
NUM_LAYERS=24 HIDDEN=2048 NUM_HEADS=16 \
SEQ_LEN=32768 MICRO_BATCH=1 GLOBAL_BATCH=4 \
TRAIN_ITERS=200 WARMUP_ITERS=20 EVAL_ITERS=0 SAVE_INTERVAL=100000 \
ONLY=seq bash tests/run_fineweb_long.sh
```

```bash
OUT_DIR=tests/exp_span_sweep_gbs4_sp8 \
GPUS_PER_NODE=8 PP_SIZE=4 SEQ1F1B_SP=8 \
NUM_LAYERS=24 HIDDEN=2048 NUM_HEADS=16 \
SEQ_LEN=32768 MICRO_BATCH=1 GLOBAL_BATCH=4 \
TRAIN_ITERS=200 WARMUP_ITERS=20 EVAL_ITERS=0 SAVE_INTERVAL=100000 \
ONLY=seq bash tests/run_fineweb_long.sh
```

```bash
OUT_DIR=tests/exp_span_sweep_gbs4_sp16 \
GPUS_PER_NODE=8 PP_SIZE=4 SEQ1F1B_SP=16 \
NUM_LAYERS=24 HIDDEN=2048 NUM_HEADS=16 \
SEQ_LEN=32768 MICRO_BATCH=1 GLOBAL_BATCH=4 \
TRAIN_ITERS=200 WARMUP_ITERS=20 EVAL_ITERS=0 SAVE_INTERVAL=100000 \
ONLY=seq bash tests/run_fineweb_long.sh
```

预期：

| span_len | 预期 |
|----------|------|
| 32768 | 普通 PP，显存最高 |
| 16384 | 显存下降，吞吐可能提升 |
| 8192 | 通常较稳 |
| 4096 | 当前最推荐 |
| 2048 | 可能过切，吞吐下降 |

span sweep 的输出决定最终长训使用 `SP=4/8/16` 中的哪一个。

### 阶段 E4：64K 可行性实验

目的：验证 Seq1F1B + DeltaNet 能否把同等模型推进到更长序列，并保持可训练。

推荐先跑短测：

```bash
OUT_DIR=tests/exp_seqpp_pp4_h2048_seq64k_gbs4_sp16_smoke \
GPUS_PER_NODE=8 PP_SIZE=4 SEQ1F1B_SP=16 \
NUM_LAYERS=24 HIDDEN=2048 NUM_HEADS=16 \
SEQ_LEN=65536 MICRO_BATCH=1 GLOBAL_BATCH=4 \
TRAIN_ITERS=50 WARMUP_ITERS=10 \
EVAL_ITERS=0 SAVE_INTERVAL=100000 \
ONLY=seq bash tests/run_fineweb_long.sh
```

配置：

| 项 | 值 |
|----|----|
| seq_len | 65536 |
| SP | 16 |
| span_len | 4096 |
| PP | 4 |
| DP | 2 |

主要看：

| 指标 | 目标 |
|------|------|
| OOM | 不能 OOM |
| loss | 正常下降 |
| peak memory | 不超过 80GB |
| toks/s | 不出现灾难性下降 |

这组实验是长序列 capability 结果，不要求和 SP1 对比，因为普通 PP 可能显存压力过高。

## 6. 数据读取和分析命令

看训练 loss：

```bash
grep 'lm loss' tests/exp_seqpp_pp4_h2048_seq32k_gbs4_long/log_fineweb_sp*.txt
```

看验证 loss：

```bash
grep 'validation loss' tests/exp_seqpp_pp4_h2048_seq32k_gbs4_long/log_fineweb_sp*.txt
```

看末尾吞吐：

```bash
grep 'toks/s' tests/exp_seqpp_pp4_h2048_seq32k_gbs4_long/log_fineweb_sp1.txt | tail -20
grep 'toks/s' tests/exp_seqpp_pp4_h2048_seq32k_gbs4_long/log_fineweb_sp8.txt | tail -20
```

看 TensorBoard：

```bash
.venv/bin/python -m tensorboard.main \
  --logdir tests/exp_seqpp_pp4_h2048_seq32k_gbs4_long/tb \
  --port 6006
```

TensorBoard 中优先看：

| 曲线 | 用途 |
|------|------|
| `lm loss` | 训练 loss |
| `lm loss vs samples` | 按样本数对齐训练 loss |
| `lm loss validation` | 验证 loss |
| `lm loss validation vs samples` | 按样本数对齐验证 loss |
| `grad-norm` | 梯度稳定性 |
| `iteration-time` | 每步耗时 |

## 7. 最终报告图表

正式报告至少需要这些图：

| 图 | 横轴 | 纵轴 | 目的 |
|----|------|------|------|
| training loss | step 或 samples | lm loss | 证明训练动态一致 |
| validation loss | step 或 samples | val lm loss | 证明质量一致 |
| grad norm | step | grad norm | 证明 backward 稳定 |
| throughput bar | config | tok/s | 展示吞吐 |
| memory bar | config | peak GB | 展示显存收益 |
| span sweep | span_len | tok/s / peak memory | 找最佳切分粒度 |

最终主表建议：

| 实验 | model | seq_len | PP | DP | SP | span_len | val loss | tok/s | step time | peak mem |
|------|-------|---------|----|----|----|----------|----------|-------|-----------|----------|
| E2 | DeltaNet | 32K | 4 | 2 | 1 | 32K | ... | ... | ... | ... |
| E2 | DeltaNet | 32K | 4 | 2 | 8 | 4K | ... | ... | ... | ... |
| E3 | DeltaNet | 32K | 4 | 2 | best | ... | ... | ... | ... | ... |
| E4 | DeltaNet | 64K | 4 | 2 | 16 | 4K | ... | ... | ... | ... |

## 8. 旧实验如何使用

旧实验可以作为 pilot motivation，但不作为正式主结果。

最有参考价值的旧日志：

```text
exp_logs/deltanet_exps/exp3_deltanet_seq32k_sp1.log
exp_logs/deltanet_exps/exp3_deltanet_seq32k_sp8.log
```

旧结果：

| 配置 | seq_len | PP | SP | tok/s | peak memory |
|------|---------|----|----|-------|-------------|
| DeltaNet + PP | 32K | 4 | 1 | ~64.8K | 76.3GB |
| DeltaNet + Seq1F1B | 32K | 4 | 8 | ~74.7K | 38.6GB |

这组结果说明 `PP=4, seq=32K, SP=8, span=4K` 值得正式复现。

但正式报告中要明确：

```text
旧实验用于选择配置；
正式结论来自 FineWeb-Edu 重新跑出的 E1/E2/E3/E4。
```

## 9. 当前建议执行顺序

1. 先跑 E1 smoke：`H=2048, seq=32K, PP=4, SP=1 vs 8, 200 iter`。
2. 如果 E1 loss 正常且 SP8 显存更低，跑 E2 长训。
3. 如果 E2 中 SP8 不够快，跑 E3 span sweep。
4. 用 E3 找到的最佳 SP 再补一条 3000-step 长训。
5. 最后跑 E4 64K smoke，作为长序列能力结果。

这条路径能把实验叙事收紧成一句话：

> Seq1F1B does not change DeltaNet training dynamics; in the right long-sequence regime it trades sequence-span granularity for lower memory and potentially higher throughput.
