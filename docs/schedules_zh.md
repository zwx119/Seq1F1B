# Seq1F1B 调度核心代码详解

> 本文档逐行分析 `schedules.py` 和 `sp_utils.py` 中 Seq1F1B 的调度逻辑。
> 所有代码引用均来自实际源文件。

---

## 一、标准 1F1B（无 VPP）: `forward_backward_pipelining_without_interleaving`

### 1. Seq1F1B 对 warmup 数量的修改

**原始 Megatron 1F1B 的 warmup 公式：**
```python
# 原始 Megatron-LM（未被 Seq1F1B 修改前）:
num_warmup_microbatches = (
    pipeline_model_parallel_world_size
    - pipeline_model_parallel_rank
    - 1
)
```
意思是：rank 0 需要 warmup `PP_size - 1` 个，rank 1 需要 `PP_size - 2` 个，
最后一个 rank 需要 0 个。这样刚好填满 pipeline。

**Seq1F1B 修改后（`schedules.py` 第 1176-1183 行）：**
```python
    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        + global_args.pipe_sp_splits
        - 2
    )
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches * global_args.pipe_sp_splits
    )
    num_microbatches_remaining = (
        num_microbatches * global_args.pipe_sp_splits
        - num_warmup_microbatches
    )
```

**为什么要 `+ pipe_sp_splits - 2`？**

以 PP=4, rank=0, pipe_sp_splits=4 为例：
- 原始：warmup = 4 - 0 - 1 = 3
- Seq1F1B：warmup = 4 - 0 + 4 - 2 = 6

每个 microbatch 被切成 4 个片段（f₀, f₁, f₂, f₃），它们是**串行**发出的。
当 f₀ 到达最后一个 rank 开始做 backward 时，f₁, f₂, f₃ 可能还没发完。
额外的 `pipe_sp_splits - 2` 次 warmup 确保在第一个 backward 回来之前，
pipeline 里已经有足够多的 forward 在飞。

（`-2` 而不是 `-1` 是因为原始公式已经有 `-1`，两者合在一起）

### 2. 总的调度单元数

```python
    num_microbatches_remaining = (
        num_microbatches * global_args.pipe_sp_splits
        - num_warmup_microbatches
    )
```

**总调度单元 = `num_microbatches × pipe_sp_splits`**。

比如 4 个 microbatch，pipe_sp_splits=4，总共 **16 个调度单元**。
每个调度单元对应一个序列片段的 forward 或 backward。

### 3. 四个通信 wrapper

在进入三个阶段之前，先定义通信包装函数（第 1235-1238 行）：

```python
    recv_forward_wrapper = lambda: recv_forward(recv_forward_tensor_shapes, config)
    recv_backward_wrapper = lambda: recv_backward(recv_backward_tensor_shapes, config)
    send_forward_wrapper = lambda output: send_forward(output, send_forward_tensor_shapes, config)
    send_backward_wrapper = lambda output: send_backward(output, send_backward_tensor_shapes, config)
```

这四个 lambda 把 `tensor_shapes` 和 `config` 绑定进去。
当 `pipe_sp_strategy == 'uniform_comp'` 时，`tensor_shapes` 是一个 `sp_shape_queue`
对象，每次调用会**自动推进到下一个片段的 shape**（后面第三节详解）。

### 4. 输入/输出队列用 sp_queue

```python
    input_tensors = sp_queue(global_args.pipe_sp_splits)
    output_tensors = sp_queue(global_args.pipe_sp_splits)
```

这是 Seq1F1B 引入的核心数据结构，不是普通 list！详见第四节。

### 5. 三个阶段

#### 阶段一：Warmup（第 1252-1280 行）：只做 forward + send

```python
    for i in range(num_warmup_microbatches):
        # ...checkpoint 判断省略...
        mega_args.schedule_info['micro_seq_id'] = i
        input_tensor, output_tensor = recv_forward_send(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches * global_args.pipe_sp_splits,
            None,               # input_tensor 占位，里面会被 recv 覆盖
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
        )

        if not forward_only:
            input_tensors.append(input_tensor[0])
            output_tensors.append(output_tensor[0])
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
```

`recv_forward_send` 是一个组合函数（第 1054-1059 行）：
```python
    def recv_forward_send(*args):
        args = list(args)
        input_tensor = recv_forward_wrapper()    # 1. 从上一个 rank 接收
        args[4] = input_tensor                   # 2. 替换 input_tensor 参数
        output_tensor = forward_step(*args)      # 3. 执行 forward
        send_forward_wrapper(output_tensor)      # 4. 发送到下一个 rank
        return input_tensor, output_tensor
```

每次 warmup 做 3 件事：**recv → forward → send**。
forward 的 input 和 output 都存进 `sp_queue`，供后续 backward 使用。

#### 阶段二：1F1B Steady State（第 1285-1321 行）：交替 1F + 1B

```python
    for i in range(num_microbatches_remaining):
        mega_args.schedule_info['micro_seq_id'] = num_warmup_microbatches + i
        last_iteration = i == (num_microbatches_remaining - 1)
        first_iteration = i == 0

        # ...checkpoint 判断省略...

        if forward_only:
            # forward-only 分支，略
            pass
        else:
            input_tensor = _1f1b_with_comm(
                last_iteration, first_iteration,
                forward_step_func,
                data_iterator,
                model,
                num_microbatches * global_args.pipe_sp_splits,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
            )
```

**`_1f1b_with_comm` 是核心**（第 1073-1115 行）！它做的事情：

```python
    def _1f1b_with_comm(last_iteration, first_iteration, *args):
        args = list(args)
        # ① 第一次迭代：需要从上游 recv
        if first_iteration:
            input_tensor = recv_forward_wrapper()
            args[4] = input_tensor
        else:
            input_tensor = args[4]   # 后续迭代：input 由上次循环尾部 recv 得到

        # ② 执行 forward
        output_tensor = forward_step(*args)

        # ③ 偶偶数/奇数 rank 交错通信（避免死锁！）
        if parallel_state.get_pipeline_model_parallel_rank() % 2 == 0:
            send_forward_wrapper(output_tensor)       # 偶数 rank：先发 fwd
            output_tensor_grad = recv_backward_wrapper()  # 再收 bwd
        else:
            output_tensor_grad = recv_backward_wrapper()  # 奇数 rank：先收 bwd
            send_forward_wrapper(output_tensor)           # 再发 fwd

        # ④ 把 forward 的 input/output 存入 sp_queue
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        if global_args.pipe_sp_splits != 1:
            deallocate_output_tensor_lis(output_tensor, config.deallocate_pipeline_outputs)
        else:
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        # ⑤ 从 sp_queue 弹出最早的 input/output 做 backward
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # ⑥ 通信：发 backward 梯度 + 收下一个 forward
        if last_iteration:
            input_tensor = None
            send_backward_wrapper(input_tensor_grad)
        else:
            if parallel_state.get_pipeline_model_parallel_rank() % 2 == 0:
                send_backward_wrapper(input_tensor_grad)  # 偶数：先发 bwd
                input_tensor = recv_forward_wrapper()     # 再收 fwd
            else:
                input_tensor = recv_forward_wrapper()     # 奇数：先收 fwd
                send_backward_wrapper(input_tensor_grad)  # 再发 bwd

        return input_tensor  # 返回给下一轮的 args[4]
```

这个函数每次调用做了 **1 forward + 1 backward**，中间穿插 4 次通信（send_fwd, recv_bwd, send_bwd, recv_fwd），偶偶数/奇数 rank 交错顺序避免死锁。

#### 阶段三：Cooldown（第 1323-1340 行）：清空剩余的 backward

```python
    if not forward_only:
        for i in range(num_warmup_microbatches):
            # 最后一个 backward 时开启 grad sync
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            recv_backward_send()
```

`recv_backward_send`（第 1118-1127 行）：
```python
    def recv_backward_send():
        input_tensor = input_tensors.pop(0)       # 从 sp_queue 弹出
        output_tensor = output_tensors.pop(0)      # 从 sp_queue 弹出
        output_tensor_grad = recv_backward_wrapper()
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )
        send_backward_wrapper(input_tensor_grad)
```

Cooldown 阶段只做 backward：**recv_grad → backward → send_grad**。
warmup 阶段存进 `sp_queue` 的 `num_warmup_microbatches` 个 tensor 在这里被消费完。

---

## 二、带 VPP 的 Interleaved 1F1B: `forward_backward_pipelining_with_interleaving`

### 关键改动

#### 总调度单元数（第 452-453 行）

```python
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks * args.pipe_sp_splits
```

**三重乘积！** 比如 4 microbatch × 2 chunks × 4 sp_splits = **32 个调度单元**。

#### Warmup 数量（第 461-471 行）

```python
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (
                (num_model_chunks - 1) * pipeline_parallel_size
            )
            num_warmup_microbatches += (
                (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            )
            num_warmup_microbatches += args.pipe_sp_splits - 1
            num_warmup_microbatches = min(
                num_warmup_microbatches, total_num_microbatches
            )
```

原始 Megatron VPP 的 warmup 公式上额外加了 `pipe_sp_splits - 1`，
和非 VPP 版本的原理相同：补偿序列切分带来的额外填充延迟。

#### 输入/输出管理（第 422-424 行）

```python
    input_tensors = [
        sp_queue(args.pipe_sp_splits, print=False, chunk=i, add_msg="input")
        for i in range(len(model))
    ]
    output_tensors = [
        sp_queue(args.pipe_sp_splits, print=False, chunk=i, add_msg="output")
        for i in range(len(model))
    ]
```

**每个 model chunk 有自己的 sp_queue！** VPP 把模型分成 `num_model_chunks` 个 chunk
（比如 2 个），每个 chunk 的 forward 激活独立存取。

#### Tensor shape（第 447-449 行）

```python
    tensor_shape = [
        seq_length // args.pipe_sp_splits,
        micro_batch_size,
        config.hidden_size,
    ]
```

通信的 tensor 大小是**原来的 1/pipe_sp_splits**。
这是 Seq1F1B 降低峰值内存的关键原因之一：同时在飞的激活更小。

---

## 三、`sp_shape_queue` — 非等长切分的通信 shape 管理

当使用 `uniform_comp` 策略时，每个片段长度不同（前短后长），
所以通信的 tensor shape 也不同。

### `get_tensor_shapes` 如何返回 shape queue（第 950-955 行）

```python
def get_tensor_shapes(*, rank, model_type, seq_length, micro_batch_size,
                      decoder_seq_length, config, backward=False):
    global_args = get_args()
    if global_args.pipe_sp_splits != 1 and global_args.pipe_sp_strategy == 'average':
        seq_length = seq_length // global_args.pipe_sp_splits
        # 等分：直接除，后面走正常路径
    elif global_args.pipe_sp_splits != 1 and global_args.pipe_sp_strategy == "uniform_comp":
        return sp_shape_queue(
            seq_length, micro_batch_size, config.hidden_size, backward=backward
        )
        # 非等分：返回 sp_shape_queue 对象，不是普通 list！
```

### `sp_shape_queue` 类（`sp_utils.py` 第 152-172 行）

```python
class sp_shape_queue:
    def __init__(self, seqlen, bs, sz, backward=False):
        self.splits = get_splits()   # 比如 [1024, 1536, 2048, 3584]（非等长）
        self.idx = 0 if not backward else len(self.splits) - 1
        self.shape = [[[s, bs, sz]] for s in self.splits]
        args = get_args()
        if args.sequence_parallel:
            self.shape = [
                [[s // args.tensor_model_parallel_size, bs, sz]]
                for s in self.splits
            ]
        self.backward = backward

    def __iter__(self):
        iter = self.shape[self.idx].__iter__()
        if not self.backward:
            self.idx = (self.idx + 1) % len(self.splits)  # forward: 0→1→2→3→0→...
        else:
            self.idx = (self.idx - 1) % len(self.splits)  # backward: 3→2→1→0→3→...
        return iter

    def get(self):
        res = self.shape[self.idx][0]
        if not self.backward:
            self.idx = (self.idx + 1) % len(self.splits)
        else:
            self.idx = (self.idx - 1) % len(self.splits)
        return res
```

**工作原理：**

假设 `get_splits()` 返回 `[1024, 1536, 2048, 3584]`（4 个非等长片段）。

- Forward 通信时，`tensor_shapes` 是 `sp_shape_queue(backward=False)`，
  每次 `for shape in tensor_shapes:` 会依次返回 `[1024, B, H]`, `[1536, B, H]`,
  `[2048, B, H]`, `[3584, B, H]`，然后循环回 `[1024, ...]`。
- Backward 通信时，`tensor_shapes` 是 `sp_shape_queue(backward=True)`，
  顺序反过来：`[3584, B, H]`, `[2048, B, H]`, `[1536, B, H]`, `[1024, B, H]`。

这样每次通信时，shape queue 自动给出当前片段的正确大小，不需要额外索引计算。

---

## 四、`sp_queue` — 组间 FIFO、组内 LIFO 的两级队列

这是 Seq1F1B 引入的**最关键数据结构**（`sp_utils.py` 第 55-100 行）。

### 为什么不能用普通 list？

考虑 pipe_sp_splits=3，3 个 microbatch，forward 顺序是：
```
warmup 入队顺序：f₀⁰ f₁⁰ f₂⁰  f₀¹ f₁¹ f₂¹  f₀² f₁² f₂²
                 ─── mb 0 ───  ─── mb 1 ───  ─── mb 2 ───
```

backward 顺序不是简单的 FIFO。同一个 microbatch 内，最后的片段要先 backward
（因为它是最晚 forward 的）。但不同 microbatch 之间是先进先出。所以 pop 顺序是：

```
pop 顺序：f₂⁰ f₁⁰ f₀⁰  f₂¹ f₁¹ f₀¹  f₂² f₁² f₀²
          ─── mb 0 ───  ─── mb 1 ───  ─── mb 2 ───
          组内 LIFO      组内 LIFO      组内 LIFO
          ←────────── 组间 FIFO ──────────→
```

### 数据结构

```python
class sp_queue:
    def __init__(self, pipe_sp_splits=4, print=False, chunk=None, add_msg=""):
        self.queues = [[]]       # 二维列表：外层是组，内层是组内元素
        self._offset = 0         # 当前正在写入的组索引
        self._idx = 0            # 当前组内已写入的元素数
        self.count = 0           # 总元素数
        self.pipe_sp_splits = pipe_sp_splits
        self.tail_obj = None     # 最近 append 的对象（用于 __getitem__(-1)）
```

### `append` — 自动分组

```python
    def append(self, obj):
        self.tail_obj = obj
        self.queues[self._offset].append(obj)
        self._idx += 1
        if self._idx == self.pipe_sp_splits:     # 满一组（pipe_sp_splits 个）
            self.queues.append([])               # 创建新组
            self._idx = 0
            self._offset += 1
        self.count += 1
```

每 `pipe_sp_splits` 次 append，自动"换行"到下一组。
比如 pipe_sp_splits=3：
```
append(f₀⁰) → queues = [[f₀⁰]]
append(f₁⁰) → queues = [[f₀⁰, f₁⁰]]
append(f₂⁰) → queues = [[f₀⁰, f₁⁰, f₂⁰], []]    ← 满了，换行
append(f₀¹) → queues = [[f₀⁰, f₁⁰, f₂⁰], [f₀¹]]
```

### `pop(0)` — 组间 FIFO、组内 LIFO

```python
    def pop(self, idx=0):
        assert idx == 0, "only pop head item"
        self.count -= 1
        if len(self.queues[0]) == 1:         # 当前组只剩 1 个元素
            if self._offset > 0:             # 还有后续组
                self._offset -= 1
                return self.queues.pop(0)[0] # 弹出整个组，返回最后一个
            else:
                return self.queues[0].pop(-1)  # 只剩一个组，弹最后一个
        else:
            return self.queues[0].pop(-1)    # 从第一组的 **尾部** 弹出
```

**关键：`self.queues[0].pop(-1)` 是从尾部弹出（LIFO），但始终操作第一个组（FIFO）！**

逐步演示（pipe_sp_splits=3）：
```
初始：queues = [[f₀⁰, f₁⁰, f₂⁰], [f₀¹, f₁¹, f₂¹]]
pop(0) → f₂⁰  (queues[0] 尾部)   queues = [[f₀⁰, f₁⁰], [f₀¹, f₁¹, f₂¹]]
pop(0) → f₁⁰  (queues[0] 尾部)   queues = [[f₀⁰], [f₀¹, f₁¹, f₂¹]]
pop(0) → f₀⁰  (queues[0] 只剩1个，弹出整组)  queues = [[f₀¹, f₁¹, f₂¹]]
pop(0) → f₂¹  (新的 queues[0] 尾部)
pop(0) → f₁¹
pop(0) → f₀¹
```

效果：**先处理 microbatch 0 的所有片段（倒序），再处理 microbatch 1（倒序）**。

这正好匹配 Seq1F1B 的 backward 需求：
- 同一个 microbatch 内，后面的片段先 backward（依赖少）
- 不同 microbatch 之间，先 forward 的先 backward（FIFO）

---

## 五、图示对比

### 原始 Megatron 1F1B（PP=4, microbatch=4）

```
时间 →
Rank 0: F₀  F₁  F₂  F₃  B₀  B₁  B₂  B₃
Rank 1:     F₀  F₁  F₂  B₀  F₃  B₁  B₂  B₃
Rank 2:         F₀  F₁  B₀  F₂  B₁  F₃  B₂  B₃
Rank 3:             F₀  B₀  F₁  B₁  F₂  B₂  F₃  B₃
```

### Seq1F1B（PP=4, microbatch=2, sp_splits=2）

每个 F 变成 f₀f₁（两个半长的 forward），每个 B 变成 b₁b₀（倒序 backward）：

```
时间 →
Rank 0: f₀⁰ f₁⁰ f₀¹ f₁¹ b₁⁰ b₀⁰ b₁¹ b₀¹
Rank 1:     f₀⁰ f₁⁰ f₀¹ b₁⁰ f₁¹ b₀⁰ b₁¹ b₀¹
Rank 2:         f₀⁰ f₁⁰ b₁⁰ f₀¹ b₀⁰ f₁¹ b₁¹ b₀¹
Rank 3:             f₀⁰ b₁⁰ f₁⁰ b₀⁰ f₀¹ b₁¹ f₁¹ b₀¹
```

（上标 ⁰¹ = microbatch index，下标 ₀₁ = split index）

**核心优势：**
1. **更细粒度的调度** → 更少的 pipeline bubble
2. **每个片段的 tensor 更小** → 峰值内存更低（同时在飞的激活更小）
3. **前面的片段更早开始 backward** → 内存释放更及时

---

## 六、总结：Seq1F1B 改了哪些地方

| 文件 | 改动 | 作用 |
|------|------|------|
| `pretrain_gpt.py` | `get_batch_sp()` 闭包 | 数据按序列维度切片，传递 `micro_sp_idx` |
| `transformer.py` | RoPE 切片 + KV cache 拼接 | 保证注意力计算跨 split 的正确性 |
| `sp_utils.py` | `sp_queue`, `sp_shape_queue`, `get_splits()` | 调度用数据结构 + 切分策略 |
| `split_solver.py` | SymPy 求解器 | `uniform_comp` 等计算量切分 |
| **`schedules.py`** | warmup 公式、总调度数、`_1f1b_with_comm`、`recv_backward_send`、sp_queue 管理 | **⭐ 核心：pipeline 排布** |
| `p2p_communication.py` | shape queue 支持 | 支持非等长片段的变长通信 |
| `arguments.py` | `--pipe-sp-splits`, `--pipe-sp-strategy` | 参数定义 |

**`schedules.py` 是最重要的**，它决定了：
- 什么时候发 forward，什么时候做 backward
- 通信顺序（偶数/奇数 rank 交错避免死锁）
- 输入/输出 tensor 的存取顺序（`sp_queue`：组内 LIFO、组间 FIFO）
- warmup / steady / cooldown 各阶段的长度
