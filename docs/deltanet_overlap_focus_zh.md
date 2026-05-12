# DeltaNet Overlap Focus

日期：2026-05-12

这份记录用于固定当前结论，避免继续混淆三类不同问题：force-seq-chunks 验证、FLA natural chunk 改写、真实 Seq1F1B span 间 overlap。

## 当前结论

DeltaNet 原始前向基本按 Seq1F1B span 粒度执行：

```text
qkvg(span)
shortconv/activation(span)
beta(span)
PRE/WY(span)
H(span)        # H kernel 内部按 FLA natural chunk 顺序递推
O(span)
output projection(span)
```

因此，真实优化应优先围绕 span 粒度，而不是把 FLA natural chunk 提到 Python 外层循环。之前的 natural-chunk H+beta pipeline 能验证想法，但会改变原始执行粒度并引入额外调度开销，不再作为主方向。

## 两个主方向

### 1. 修正确版 H/O pipeline

目标是在同一个 span 内让 `H(C_i)` 与 `O(C_{i-1})` 或等价的 H/O 流水重叠，不需要读取下一个 span 的 hidden。这个方向更贴近 FLA 内部结构，也避免改 Seq1F1B scheduler。

优先级最高，但必须先解决正确性问题：之前 H/O pipeline 端到端出现过 `grad norm: nan`，不能只看速度。

推进顺序：

1. 先做单算子 forward 对齐：`fused H/O` vs `chunk_gated_delta_rule_fwd_h + chunk_fwd_o`。
2. 再做 backward / 训练 smoke：确认 loss、grad norm 不变成 nan。
3. 最后接入 Megatron flag，跑真实 16K/32K 配置对比端到端。

### 2. qkvg span lookahead

目标是在 `H(span i)` 里 fused 计算 `qkvg(span i+1)`，写入 cache，下一 span 直接复用。这个方向更可能带来端到端收益，因为 qkvg projection 明显大于 beta projection。

工程影响比 beta 大：真实 Seq1F1B 当前每次 layer forward 只拿到当前 span hidden，`H(span i)` 里没有 `span i+1` 的 hidden。要做真 span lookahead，需要 scheduler/model 层把下一 span 的 layer input 暴露给当前 span，或者增加明确的跨 span lookahead cache。

推进顺序：

1. 先确认 scheduler 能否安全传递 next-span hidden。
2. 再把现有 `H + projection` fused kernel 接到 span cache。
3. 最后做端到端和 nsys 验证，确认 projection CTA 真正填在 H kernel 内。

## 暂停方向

暂时不再投入以下方向：

```text
FLA natural chunk H+beta Python loop
FLA natural chunk H+beta CUDA Graph replay
force-seq-chunks 伪真实路径
beta-only span lookahead 作为主要性能方向
```

其中 beta-only span lookahead 可以保留作机制验证，但预期收益小，不作为主性能优化目标。

## 当前代码边界

当前主线只保留 `--deltanet-fused-h-o-pipeline` 和 H/O pipeline 测试。旧的 H+beta、H+qkvg chunk-level bench 以及 Megatron 外层 force-seq-chunks 伪 lookahead 接线已经清理掉。

FLA 里仍保留低层 `H + projection` fused kernel 能力，后续如果做真正的 qkvg span lookahead，可以在 scheduler/model 层拿到 next-span hidden 后重新接入；但不再从 Megatron 外层强行切 chunk 调用。
