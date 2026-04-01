# Seq1F1B × DeltaNet 开发文档

## 一、改了什么？总览

本次修改将 **DeltaNet 线性注意力** 集成到 Seq1F1B 流水线并行框架中，作为
softmax attention (FlashAttention) 的 **drop-in 替换**。

总共 **新增 4 个文件，修改 2 个文件**，共 987 行新增代码：

| 文件 | 类型 | 行数 | 做了什么 |
|------|------|------|----------|
| `megatron/model/deltanet_attention.py` | 新增 | 368 | DeltaNet 注意力核心模块 |
| `megatron/model/deltanet_layer.py` | 新增 | 204 | 使用 DeltaNet 的 Transformer 层 |
| `run_deltanet.sh` | 新增 | 127 | DeltaNet 启动脚本 |
| `docs/deltanet.md` | 新增 | 208 | 英文文档 |
| `megatron/arguments.py` | 修改 | +77 | 新增命令行参数 |
| `megatron/model/transformer.py` | 修改 | +9 | 修改 `build_layer()` |

---

## 二、为什么 DeltaNet 天然适配 Seq1F1B？

**原来的 softmax attention + Seq1F1B** 的问题：

Seq1F1B 把每个 microbatch 的序列切成 `pipe_sp_splits` 份。对于 softmax attention，
第 2 份序列需要"看到"第 1 份的 KV cache，所以必须：
1. 缓存前一份的 K、V 张量（大小 = O(s × d)，随序列长度线性增长）
2. 在 `FlashAttnVarlenFunc` 里拼接 KV cache，用 `cu_seqlens` 标记边界
3. 自定义 backward 来处理梯度

**DeltaNet 的核心公式：**
```
S_t = (I - β_t · k_t · k_t^T) · S_{t-1} + β_t · k_t · v_t^T
o_t = q_t^T · S_t
```

DeltaNet 用一个 **常量大小的递归状态 S ∈ R^{d_k × d_v}** 代替 KV cache。
无论序列多长，S 的大小不变。所以 Seq1F1B 的跨 split 状态传递变成：

- `micro_sp_idx == 0`：清空 S（新 microbatch 开始）
- `micro_sp_idx > 0`：把上一个 split 的 S 作为 `initial_state` 传入

**O(1) 内存**，不需要拼接，不需要自定义 backward。

---

## 三、新增文件详解

### 3.1 `megatron/model/deltanet_attention.py`（368 行）

这是最核心的文件。定义了 `DeltaNetAttention` 类，替换原来的 `ParallelAttention`。

#### 文件顶部：依赖导入（第 1-57 行）

```python
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
from fla.modules import RMSNorm as FLARMSNorm, ShortConvolution
from einops import rearrange
```

- `chunk_delta_rule`：训练用的分块 delta rule 算子（Triton 内核），把序列切成 chunk 处理
- `fused_recurrent_delta_rule`：推理/短序列用的逐步递归算子
- `ShortConvolution`：fla 提供的因果 1-D 深度卷积，替代 RoPE 位置编码
- `rearrange`：einops 的张量变形工具

三个 `try/except` 块做了优雅降级：如果 fla 没装，不会崩溃，只是标记 `HAS_FLA=False`。

#### `class DeltaNetAttention(MegatronModule)`

##### `__init__`（第 92-217 行）

**头数切分（张量并行）：**
```python
world_size = mpu.get_tensor_model_parallel_world_size()
self.num_attention_heads_per_partition = core.utils.divide(
    config.num_attention_heads, world_size
)
```
跟原来的 `ParallelAttention` 一样，把注意力头均分到各 TP rank。
比如 32 个头、4 路 TP → 每个 rank 8 个头。

**Q/K/V 投影（第 126-149 行）：**
```python
self.q_proj = tensor_parallel.ColumnParallelLinear(
    self.hidden_size, self.hidden_size,
    gather_output=False,  # 保持切分状态
)
```
- 使用 `ColumnParallelLinear`：输出维度按 TP 切分
- `gather_output=False`：不做 all-gather，输出直接就是当前 rank 那部分头的结果
- Q、K、V 各一个，都不加 bias

**Beta 投影（第 152-157 行）：**
```python
self.b_proj = nn.Linear(
    self.hidden_size,
    self.num_attention_heads_per_partition,
    bias=False,
)
```
Beta 是 delta rule 的步长 β_t ∈ (0,1)，每个头一个标量。
因为输出维度很小（只有 `num_heads_per_partition` 个），用普通 `nn.Linear` 就行，
不需要 TP 切分。

**短卷积（第 160-177 行）：**
```python
self.q_conv1d = ShortConvolution(
    hidden_size=self.key_dim_per_partition,
    kernel_size=self.conv_size,  # 默认 4
    activation='silu',
)
```
- Q、K、V 各一个因果 1-D 卷积（depthwise Conv1d，kernel=4）
- 作用：替代 RoPE，给模型提供局部位置信息
- 因果性：只看当前和前 `kernel_size-1` 个位置，不泄露未来信息
- 卷积作用在 TP partition 的维度上（不需要跨 rank 通信）

**输出门控 + 归一化（第 179-191 行）：**
```python
if self.use_gate:
    self.g_proj = tensor_parallel.ColumnParallelLinear(...)
    self.o_norm = FusedRMSNormGated(self.head_dim)
else:
    self.o_norm = FLARMSNorm(self.head_dim)
```
- 有门控时：额外做一个线性投影得到 gate g，然后 `o_norm(o, g)` 融合归一化+门控
- 无门控时：只做 RMSNorm

**输出投影（第 194-207 行）：**
```python
self.o_proj = tensor_parallel.RowParallelLinear(
    self.hidden_size, self.hidden_size,
    input_is_parallel=True,  # 输入已经按 TP 切分
    skip_bias_add=True,      # bias 不在这里加，返回给上层
)
```
- `RowParallelLinear`：输入是各 rank 的部分结果，内部做 all-reduce 聚合
- `skip_bias_add=True`：不在投影里加 bias，返回 `(output, bias)` 元组，
  让上层的 bias-dropout-add 融合算子一起处理（减少一次 kernel launch）

**Seq1F1B 状态（第 209-217 行）：**
```python
self.recurrent_state = None  # DeltaNet 递归状态 S
self.conv_state_q = None     # Q 的短卷积缓存
self.conv_state_k = None     # K 的短卷积缓存
self.conv_state_v = None     # V 的短卷积缓存
```
这四个变量是 Seq1F1B 的核心。它们不是 `nn.Parameter`（不参与梯度更新），
而是运行时缓存，用于在序列 split 之间传递状态。

##### `_clear_states()`（第 219-224 行）

```python
def _clear_states(self):
    self.recurrent_state = None
    self.conv_state_q = None
    self.conv_state_k = None
    self.conv_state_v = None
```
新 microbatch 的第一个 split（`micro_sp_idx == 0`）时调用，清空所有缓存。

##### `forward()`（第 226-368 行）

完整的前向传播，分 7 个阶段：

**阶段 1：Seq1F1B 状态管理（第 247-254 行）**
```python
if micro_sp_idx is not None:
    if torch.is_tensor(micro_sp_idx):
        micro_sp_idx = micro_sp_idx.item()
    if micro_sp_idx == 0:
        self._clear_states()
```
- `micro_sp_idx` 可能是 tensor（从 activation checkpointing 传来），需要 `.item()` 转标量
- `== 0` 时清空状态（这个 microbatch 的第一个 split）

**阶段 2：转置（第 256-258 行）**
```python
hidden_states = hidden_states.transpose(0, 1).contiguous()  # [s,b,h] → [b,s,h]
```
Megatron 内部用 `[s, b, h]`（sequence first），fla 用 `[b, s, h]`（batch first）。

**阶段 3：Q/K/V 投影 + 短卷积（第 260-294 行）**
```python
q, _ = self.q_proj(hidden_states)  # ColumnParallelLinear → [b, s, dim_per_partition]
# ...
q, self.conv_state_q = self.q_conv1d(
    x=q,
    cache=self.conv_state_q if use_conv_cache else None,
    output_final_state=(args.pipe_sp_splits > 1),
)
```
- 投影后过短卷积
- `cache` 参数：如果是第 2+ 个 split (`micro_sp_idx > 0`)，传入上一个 split 的卷积状态
- `output_final_state`：如果开了 Seq1F1B（`pipe_sp_splits > 1`），需要输出最终卷积状态供下一个 split 使用
- 返回值 `self.conv_state_q` 会自动缓存到实例变量

**阶段 4：Reshape 多头（第 296-300 行）**
```python
q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
```
将平坦的 `[b, s, heads_per_part * head_dim]` 拆成 `[b, s, heads_per_part, head_dim]`。

**阶段 5：Beta 计算（第 308-316 行）**
```python
beta = self.b_proj(hidden_states).sigmoid()  # [b, s, num_heads_per_partition]
```
- 线性投影 + sigmoid 压到 (0,1) 区间
- beta 控制 delta rule 的更新幅度：β=0 完全忘记，β=1 完全更新

**阶段 6：DeltaNet 核心计算（第 318-345 行）**
```python
o, recurrent_state = chunk_delta_rule(
    q=q, k=k, v=v, beta=beta,
    initial_state=self.recurrent_state,     # 上一个 split 的状态
    output_final_state=output_final_state,  # 是否输出最终状态
)
if output_final_state and recurrent_state is not None:
    self.recurrent_state = recurrent_state.detach()  # 缓存，切断梯度
```
- 短序列 (≤64) 自动切换到 `fused_recurrent_delta_rule`（更快）
- `initial_state`：如果是 `micro_sp_idx > 0`，传入上一个 split 缓存的 `recurrent_state`
- `.detach()`：**关键！** 切断跨 split 的梯度流，每个 split 独立反向传播。
  这跟原版 Seq1F1B 处理 KV cache 的方式一致

**阶段 7：归一化 + 输出投影 + 转置回（第 347-368 行）**
```python
o = self.o_norm(o)                    # RMSNorm per head
o = rearrange(o, 'b s h d -> b s (h d)')  # 合并多头
output, bias = self.o_proj(o)         # RowParallelLinear (all-reduce)
output = output.transpose(0, 1)       # [b,s,h] → [s,b,h] 回 Megatron 格式
return output, bias
```

---

### 3.2 `megatron/model/deltanet_layer.py`（204 行）

定义了 `DeltaNetTransformerLayer`，直接替换原来的 `ParallelTransformerLayer`。

#### `class DeltaNetTransformerLayer(MegatronModule)`

##### `__init__()`（第 62-107 行）

结构跟 `ParallelTransformerLayer` 完全一致，唯一区别是 `self.self_attention`：

```python
# 原来：
self.self_attention = ParallelAttention(config, layer_number, ...)
# 现在：
self.self_attention = DeltaNetAttention(config, layer_number)
```

其他组件完全复用：
- `self.input_layernorm`：输入 LayerNorm（Pre-LN 架构）
- `self.post_attention_layernorm`：attention 后的 LayerNorm
- `self.mlp`：ParallelMLP（SwiGLU 或 GELU，不变）
- `self.drop_path`：DropPath（可选）
- `self.bias_dropout_add_exec_handler`：bias+dropout+add 融合执行器

##### `forward()`（第 109-204 行）

函数签名与 `ParallelTransformerLayer.forward()` **完全一致**：

```python
def forward(self, hidden_states, attention_mask,
            encoder_output=None, enc_dec_attn_mask=None,
            retriever_input=None, retriever_output=None,
            retriever_attn_mask=None, inference_params=None,
            rotary_pos_emb=None, micro_sp_idx=None):
```

这很重要，因为 `ParallelTransformer` 容器会用这些参数调用每一层。
有些参数 DeltaNet 不用（`attention_mask`、`rotary_pos_emb`、`retriever_*`），
但必须接收以保持接口兼容。

前向流程：
```
hidden_states [s,b,h]
  │
  ├─ input_layernorm
  ├─ DeltaNetAttention(micro_sp_idx=micro_sp_idx)  ← 传递 split 索引
  ├─ bias_dropout_add + 第一次残差
  ├─ post_attention_layernorm
  ├─ ParallelMLP（不变）
  ├─ bias_dropout_add + 第二次残差
  │
  └─ output [s,b,h]
```

---

## 四、修改的文件详解

### 4.1 `megatron/arguments.py`（+77 行）

做了 3 处修改：

#### 修改 1：注册参数解析函数（第 41 行）

```python
parser = _add_deltanet_args(parser)
```
在 `parse_args()` 里，紧跟 `_add_retro_args` 之后，注册 DeltaNet 参数组。

#### 修改 2：参数验证逻辑（第 380-400 行）

在 `validate_args()` 里添加 DeltaNet 相关验证：

```python
if args.use_deltanet:
    args.position_embedding_type = 'none'     # 不用任何位置编码
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = False
    if args.use_flash_attn:
        raise RuntimeError('互斥')            # 不能同时用 FlashAttn
    try:
        from fla.ops.delta_rule import chunk_delta_rule
    except ImportError:
        raise RuntimeError('缺少 fla 库')     # 启动时就检查依赖
```

设计意图：
- **自动禁用 RoPE**：DeltaNet 用短卷积替代，如果保留 RoPE 会在 `language_model.py`
  里创建不需要的 `RotaryEmbedding` 模块
- **互斥检查**：`--use-deltanet` 和 `--use-flash-attn` 不能共存
- **提前检查 fla**：不等到创建模型时再报错，在参数解析阶段就失败

同时修改了旧的验证规则，让 `position_embedding_type='none'` 不触发旧的报错：
```python
# 旧：
if not args.add_position_embedding and args.position_embedding_type != 'rope':
# 新：
if not args.add_position_embedding and args.position_embedding_type not in ('rope', 'none'):
```

#### 修改 3：新增 `_add_deltanet_args()` 函数（第 558-603 行）

定义了 11 个命令行参数：

| 参数 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `--use-deltanet` | flag | False | 总开关，启用 DeltaNet |
| `--deltanet-mode` | str | `chunk` | 计算模式：`chunk`（训练）或 `fused_recurrent`（推理） |
| `--deltanet-use-short-conv` | flag | True | 启用短卷积 |
| `--no-deltanet-short-conv` | flag | → False | 禁用短卷积 |
| `--deltanet-conv-size` | int | 4 | 卷积核大小 |
| `--deltanet-use-beta` | flag | True | 启用可学习 β |
| `--no-deltanet-beta` | flag | → False | 禁用可学习 β（固定 β=1） |
| `--deltanet-use-output-gate` | flag | True | 启用输出门控 |
| `--no-deltanet-output-gate` | flag | → False | 禁用输出门控 |
| `--deltanet-qk-activation` | str | `silu` | Q/K 激活函数 |
| `--deltanet-qk-norm` | str | `l2` | Q/K 归一化方式 |
| `--deltanet-head-dim` | int | None | 每头维度（默认从 kv_channels 取） |

另外还给 `--position-embedding-type` 增加了 `'none'` 选项。

### 4.2 `megatron/model/transformer.py`（+9 行）

只修改了一个地方：`ParallelTransformer.__init__()` 里的 `build_layer()` 函数。

```python
def build_layer(layer_number):
    if args.transformer_impl == 'local':
        current_layer_type = _get_layer_type(...)
        # ======== 新增 ========
        if getattr(args, 'use_deltanet', False):
            from megatron.model.deltanet_layer import DeltaNetTransformerLayer
            return DeltaNetTransformerLayer(
                config, layer_number,
                layer_type=current_layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])
        # ======== 新增结束 ========
        return ParallelTransformerLayer(...)
```

设计意图：
- **条件分支**：只在 `--use-deltanet` 时走新路径，不影响现有代码
- **惰性导入**：`from ... import DeltaNetTransformerLayer` 放在函数内部，
  不用 DeltaNet 时完全不加载 fla 等依赖
- **`getattr(args, 'use_deltanet', False)`**：用 `getattr` 带默认值，
  即使旧的 checkpoint 没有这个参数也不会崩溃
- 参数传递完全一致（`config`, `layer_number`, `layer_type`, `self_attn_mask_type`, `drop_path_rate`）

---

## 五、`run_deltanet.sh` 启动脚本（127 行）

基于 `run.sh` 修改，主要区别：

1. 把 `--use-flash-attn` 替换为 `--use-deltanet`
2. 去掉 `--position-embedding-type rope`（DeltaNet 自动设为 `none`）
3. 新增 DeltaNet 特有参数：`--deltanet-mode`、`--deltanet-conv-size` 等
4. 所有参数都支持环境变量覆盖

使用方式：
```bash
export GPUS_PER_NODE=8 WORLD_SIZE=1
export MASTER_ADDR=localhost MASTER_PORT=12345
export DATA_PATH=/path/to/data
export PP_SIZE=4 TP_SIZE=2 PP_SP=4
bash run_deltanet.sh
```

---

## 六、不需要修改的文件（以及为什么）

| 文件 | 为什么不用改 |
|------|-------------|
| `pretrain_gpt.py` | `micro_sp_idx` 已经通过 `get_batch_sp` 闭包传递，无需修改 |
| `megatron/model/gpt_model.py` | 只做 `language_model()` + `post_language_model_processing()`，不涉及注意力 |
| `megatron/model/language_model.py` | 位置编码通过 `args.position_embedding_type` 控制，设为 `'none'` 后自动跳过 |
| `megatron/schedules.py` | 调度逻辑完全不变，`micro_sp_idx` 的生成和传递机制不受注意力类型影响 |
| `megatron/sp_utils.py` | `sp_queue` 逻辑不变 |

---

## 七、数据流全链路

一个 Seq1F1B 训练 step 中，DeltaNet 的完整数据流：

```
pretrain_gpt.py::forward_step()
  │
  ├─ get_batch_sp() 闭包
  │   └─ 从 sp_queue 取数据，包含 micro_sp_idx
  │
  ├─ model(tokens, position_ids, attention_mask, micro_sp_idx=micro_sp_idx)
  │   │
  │   └─ GPTModel.forward()
  │       └─ TransformerLanguageModel.forward()
  │           ├─ Embedding（只有 word embedding，没有 position embedding）
  │           ├─ rotary_pos_emb = None（因为 position_embedding_type='none'）
  │           └─ ParallelTransformer.forward(micro_sp_idx=micro_sp_idx)
  │               │
  │               └─ for each layer:  # DeltaNetTransformerLayer
  │                   ├─ input_layernorm
  │                   ├─ DeltaNetAttention(hidden_states, micro_sp_idx=micro_sp_idx)
  │                   │   ├─ 如果 micro_sp_idx==0：清空 recurrent_state + conv_state
  │                   │   ├─ Q/K/V 投影 (ColumnParallelLinear)
  │                   │   ├─ 短卷积（传入/输出 conv_state）
  │                   │   ├─ chunk_delta_rule(initial_state=缓存的 S)
  │                   │   ├─ 缓存新的 recurrent_state（.detach()）
  │                   │   └─ 输出投影 (RowParallelLinear)
  │                   ├─ bias_dropout_add + 残差
  │                   ├─ post_attention_layernorm
  │                   ├─ ParallelMLP
  │                   └─ bias_dropout_add + 残差
  │
  └─ loss_func(output, micro_sp_idx)
```

---

## 八、依赖

| 库 | 用途 | 安装 |
|----|------|------|
| `flash-linear-attention` (fla) | `chunk_delta_rule`、`ShortConvolution`、`RMSNorm` | `pip install flash-linear-attention` |
| `einops` | `rearrange` 张量变形 | `pip install einops` |
| `triton` | fla 的 Triton 算子后端 | `pip install triton` |
