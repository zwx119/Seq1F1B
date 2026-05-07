# DeltaNet Seq1F1B Fused-Off 主实验记录

更新时间：2026-05-07

## 结论口径

后续主实验统一使用 `FLA_USE_FUSED_SOLVE_WU=0`，也就是关闭当前实验性的 `solve_wu` fused Triton 路径，回到原始 FLA WY 路径：

```text
chunk_scaled_dot_kkt -> solve_tril -> recompute_w_u
```

原因是 H100 microbenchmark 已经显示当前 fused 实现虽然数值正确，但在真实形状上更慢：

```text
T8192  H32 K80  V80 : original 0.3247 ms, fused 0.4238 ms, speedup 0.766x
T8192  H32 K128 V128: original 0.3153 ms, fused 0.4303 ms, speedup 0.733x
T32768 H32 K80  V80 : original 1.1502 ms, fused 1.6318 ms, speedup 0.705x
```

因此论文主结果不应该混入这个尚未优化成功的 fused kernel。当前 fused kernel 可以作为“尝试过 CC+TC 融合但还需要更底层 warp-specialized 实现”的负结果或附录讨论，不作为主表性能来源。

## 要跑的主实验

第一组是 Hybrid DeltaNet，用于回答 global attention 层存在时 Seq1F1B 的收益，以及 flops 均分是否优于平均切分。

```text
模型：L32 H2560 heads32
序列：16K / 24K / 32K
并行：PP8 TP1 GBS16
SP：4 / 8
hybrid：global2 / period4
切分：average / hybrid_comp
kernel：FLA_USE_FUSED_SOLVE_WU=0
```

`global2` 指两层全局 causal attention，层号为 `2,15`。`period4` 指每 4 层一层 attention，默认 offset=0，因此 L32 下是 `4,8,12,16,20,24,28,32` 共 8 层 attention。

第二组是原始 DeltaNet main result，用 fused-off 重跑 2.0B 和 2.7B，用于作为最终主结论表。

```text
2.0B 近似配置：L24 H2560 heads32
2.7B 近似配置：L32 H2560 heads32
序列：16K / 24K / 32K
并行：PP8 TP1 GBS16
SP：1 / 2 / 4 / 8
切分：average
kernel：FLA_USE_FUSED_SOLVE_WU=0
```

如果 SP1 在 24K/32K OOM，这是可作为主结论的一部分：Seq1F1B 通过切序列降低 activation/state 峰值，使长序列训练从 OOM 变成可跑。

## Flops 均分公式审计

`hybrid_comp` 使用的 cost model 是：

```text
chunk_cost =
  embedding_linear(chunk)
+ num_layers * [attention_linear(chunk) + ffn_linear(chunk)]
+ softmax_layers * causal_attention_quadratic(chunk, prefix)
+ output_projection_linear(chunk)
```

其中 causal attention 的 prefix chunk cost 写成：

```text
chunk_len * prefix_len * (4 * head_dim + 3) * num_heads
- chunk_len^2 * (4 * head_dim + 3) * num_heads / 2
```

这里的 `prefix_len` 是“从序列开头到当前 chunk 结束”的累计长度，所以越靠后的 chunk 同长度下 causal attention FLOPs 越大。为了均分 attention FLOPs，前面的 chunk 可以更长，后面的 chunk 要更短。代码里的 solver 从右往左解，最终得到的 `hybrid_comp` 通常是前大后小，例如 `[9519, 8459, 7690, 7100]`。审计脚本会直接用 `get_prefix_tflops(length, prefix)` 重新计算每段 cost，检查 `max_imbalance`。

当前没有发现明显代数错误。更可能的问题是这个公式只平衡 FLOPs，不包含以下真实开销：

```text
PP stage 上 attention 层的位置分布
DeltaNet / MLP 线性项占大头时，softmax 二次项权重不足
非均匀 shape 造成 Triton kernel/通信/调度效率下降
更长尾 chunk 增加局部峰值显存和 pipeline 等待
```

这解释了为什么之前 A100 上 average 仍然比 hybrid_comp 快，即使 hybrid_comp 在公式上更“均分 FLOPs”。

## 切分长度参考

L32 H2560 heads32, vocab 50304, `global2=2` 个 softmax 层：

```text
seq16384 SP4 average     [4096, 4096, 4096, 4096]
seq16384 SP4 hybrid_comp [4190, 4124, 4064, 4006]
seq16384 SP8 hybrid_comp [2105, 2086, 2070, 2054, 2039, 2024, 2010, 1996]

seq24576 SP4 average     [6144, 6144, 6144, 6144]
seq24576 SP4 hybrid_comp [6352, 6206, 6072, 5946]
seq24576 SP8 hybrid_comp [3197, 3156, 3120, 3085, 3052, 3020, 2988, 2958]

seq32768 SP4 average     [8192, 8192, 8192, 8192]
seq32768 SP4 hybrid_comp [8557, 8299, 8064, 7848]
seq32768 SP8 hybrid_comp [4314, 4244, 4180, 4118, 4060, 4004, 3950, 3898]
```

L32 H2560 heads32, vocab 50304, `period4=8` 个 softmax 层：

```text
seq16384 SP4 average     [4096, 4096, 4096, 4096]
seq16384 SP4 hybrid_comp [4451, 4187, 3967, 3779]
seq16384 SP8 hybrid_comp [2264, 2188, 2123, 2064, 2009, 1958, 1911, 1867]

seq24576 SP4 average     [6144, 6144, 6144, 6144]
seq24576 SP4 hybrid_comp [6914, 6320, 5858, 5484]
seq24576 SP8 hybrid_comp [3546, 3371, 3224, 3095, 2980, 2877, 2784, 2699]

seq32768 SP4 average     [8192, 8192, 8192, 8192]
seq32768 SP4 hybrid_comp [9519, 8459, 7690, 7100]
seq32768 SP8 hybrid_comp [4922, 4600, 4339, 4118, 3928, 3762, 3615, 3484]
```

注意：上面的 non-uniform split 是公式结果，不等于一定更快。它应该作为 ablation：如果 average 仍然更快，结论就是 DeltaNet 主体的 per-token 线性计算和实际 kernel/PP 调度主导性能，简单 flops 均分不是最佳切法。

## 运行命令

先检查切分公式：

```bash
python3 tests/inspect_pipe_sp_splits.py --pattern global2
python3 tests/inspect_pipe_sp_splits.py --pattern period4
```

跑 hybrid fused-off 主实验：

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

export FLA_DIR=$PWD/flash-linear-attention
export PYTHONPATH="$FLA_DIR:$PWD:${PYTHONPATH:-}"
export FLA_USE_FUSED_SOLVE_WU=0

OUT_ROOT=tests/fused_off_hybrid_main \
SEQ_LIST="16384 24576 32768" \
SP_LIST="4 8" \
HYBRID_LIST="global2 period4" \
STRATEGY_LIST="average hybrid_comp" \
bash tests/run_fused_off_hybrid_experiments.sh
```

跑原始 DeltaNet fused-off main result：

```bash
cd /mlx_devbox/users/zhaowenxuan.119/playground/Seq1F1B

export FLA_DIR=$PWD/flash-linear-attention
export PYTHONPATH="$FLA_DIR:$PWD:${PYTHONPATH:-}"
export FLA_USE_FUSED_SOLVE_WU=0

OUT_ROOT=tests/fused_off_deltanet_main \
MODEL_SPECS="m2p0b:24:2560:32 m2p7b:32:2560:32" \
SEQ_LIST="16384 24576 32768" \
SP_LIST="1 2 4 8" \
bash tests/run_fused_off_deltanet_main_sweep.sh
```

汇总结果：

```bash
python3 tests/summarize_fineweb_logs.py --root tests/fused_off_hybrid_main
python3 tests/summarize_fineweb_logs.py --root tests/fused_off_deltanet_main
```

也可以导出 CSV：

```bash
python3 tests/summarize_fineweb_logs.py --root tests/fused_off_hybrid_main --csv tests/fused_off_hybrid_main.csv
python3 tests/summarize_fineweb_logs.py --root tests/fused_off_deltanet_main --csv tests/fused_off_deltanet_main.csv
```

## 结果表待填

Hybrid fused-off 结果：

```text
global2 / period4
seq16K / 24K / 32K
SP4 / SP8
average / hybrid_comp
```

Pure DeltaNet fused-off 结果：

```text
m2p0b / m2p7b
seq16K / 24K / 32K
SP1 / SP2 / SP4 / SP8
average
```

预期主结论写法：

```text
1. 关闭实验性 fused solve_wu 后，主结果回到原始 FLA kernel，避免将一个 H100 上已证实更慢的融合路径混入性能评估。
2. Seq1F1B 的核心收益应该体现在长序列可训练性和 SP scaling：SP1 可能 OOM，而 SP2/4/8 可以跑通。
3. 对 DeltaNet-heavy 模型，average split 很可能仍然是强 baseline；hybrid_comp 是合理 ablation，但不一定赢，因为 softmax 层少且 DeltaNet/MLP 线性项、kernel shape、PP bubble 共同主导。
4. 如果 period4 比 global2 更能放大 hybrid_comp 的差异，说明 attention 层占比上升后二次项切分开始更重要；如果仍不赢，说明真实系统开销超过 FLOPs 模型。
```
