# DeltaNet Seq1F1B 实验表格草稿

本文档把当前 DeltaNet + Seq1F1B 的关键实验结果整理成论文表格风格，方便后续复制到报告或论文草稿里。

说明：

- 吞吐单位为 `k toks/s`。
- 显存为 `mem_each_stage` 的峰值，单位为 `GB`。
- 除特别说明外，结果取对应 log 末尾的 `time/toks/tflops/mem_arr` summary。
- `SP1` 表示普通 PP/1F1B baseline；`SP2/SP4/SP8/SP16` 表示 Seq1F1B 的 sequence split 数。
- `GBS` 是 global batch size；本批实验中 `MICRO_BATCH=1`。

## 1. 2.7B DeltaNet, PP8/TP1, Paper-Style 设置

配置：

| Item | Value |
|---|---|
| Model | DeltaNet 2.7B-style |
| Layers / Hidden / Heads | `32 / 2560 / 32` |
| GPUs | `8 x 80G` |
| PP / TP / DP | `8 / 1 / 1` |
| Micro batch size | `1` |
| GBS | `16 / 32` |
| Number of microbatches | `16 / 32` |
| Train iters | `30` |
| Save | disabled |
| DeltaNet optimizations | fused short-conv chunk path + fused `qkvg` projection |

### 1.1 Main Results

表格格式对齐 Seq1F1B 论文 Table 2：上层按 `Sequence Length` 分组，下层按 `Global Batch / microbatch number` 分列；左侧按指标分块。粗体表示该列中已完成结果的最高吞吐或最低显存。

<table>
  <thead>
    <tr>
      <th rowspan="3">Model Size</th>
      <th rowspan="3">Metric</th>
      <th rowspan="3">Method</th>
      <th colspan="2">Sequence Length 16384</th>
      <th colspan="2">Sequence Length 24576</th>
      <th colspan="2">Sequence Length 32768</th>
      <th>Sequence Length 65536</th>
    </tr>
    <tr>
      <th colspan="2">Global Batch</th>
      <th colspan="2">Global Batch</th>
      <th colspan="2">Global Batch</th>
      <th>Global Batch</th>
    </tr>
    <tr>
      <th>16</th>
      <th>32</th>
      <th>16</th>
      <th>32</th>
      <th>16</th>
      <th>32</th>
      <th>16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">2.7B<br/>DeltaNet</td>
      <td rowspan="4">Throughput<br/>(k toks/s)</td>
      <td>SP1 / PP</td>
      <td>37.62</td>
      <td>43.45</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP2</td>
      <td>41.73</td>
      <td>45.51</td>
      <td>42.56</td>
      <td>46.10</td>
      <td>42.72</td>
      <td>46.30</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td><strong>43.32</strong></td>
      <td><strong>45.53</strong></td>
      <td><strong>44.10</strong></td>
      <td><strong>46.23</strong></td>
      <td><strong>45.12</strong></td>
      <td><strong>47.26</strong></td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td>40.85</td>
      <td>42.42</td>
      <td>42.97</td>
      <td>44.29</td>
      <td>44.88</td>
      <td>46.14</td>
      <td><strong>47.14</strong></td>
    </tr>
    <tr>
      <td rowspan="4">Logged TFLOPS/s</td>
      <td>SP1 / PP</td>
      <td>116.51</td>
      <td>134.55</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP2</td>
      <td>129.21</td>
      <td>140.93</td>
      <td>153.41</td>
      <td>166.17</td>
      <td>175.68</td>
      <td>190.43</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td><strong>134.15</strong></td>
      <td><strong>140.99</strong></td>
      <td><strong>158.96</strong></td>
      <td><strong>166.65</strong></td>
      <td><strong>185.55</strong></td>
      <td><strong>194.37</strong></td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td>126.51</td>
      <td>131.35</td>
      <td>154.91</td>
      <td>159.67</td>
      <td>184.58</td>
      <td>189.77</td>
      <td><strong>289.66</strong></td>
    </tr>
    <tr>
      <td rowspan="4">Peak Memory<br/>(GB)</td>
      <td>SP1 / PP</td>
      <td>68.7</td>
      <td>69.0</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP2</td>
      <td>42.3</td>
      <td>42.6</td>
      <td>59.3</td>
      <td>59.3</td>
      <td>76.2</td>
      <td>76.2</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>28.9</td>
      <td>28.9</td>
      <td>39.2</td>
      <td>39.2</td>
      <td>49.6</td>
      <td>49.6</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td><strong>22.2</strong></td>
      <td><strong>22.1</strong></td>
      <td><strong>29.1</strong></td>
      <td><strong>29.1</strong></td>
      <td><strong>36.2</strong></td>
      <td><strong>36.2</strong></td>
      <td><strong>64.0</strong></td>
    </tr>
  </tbody>
</table>

### 1.2 Observations

| Observation | Evidence |
|---|---|
| SP4 是当前 8 卡 PP8 的主要吞吐甜点 | `16K/24K/32K` 的已完成列中，SP4 基本取得最高吞吐；`16K/GBS32` 下 SP2/SP4 几乎持平 |
| SP8 是最低显存和最长序列可行点 | `64K/GBS16` 只有 SP8 完成，吞吐 `47.14k`，峰值显存 `64.0G` |
| 24K 起 SP1 OOM，Seq1F1B 变成可训练性收益 | `24K/32K/64K` 下 SP1 均未产出 `toks/s`，SP2/SP4/SP8 可覆盖更多长序列点 |
| fused short-conv + qkvg projection 后，Seq1F1B 同时带来吞吐与显存收益 | `16K/GBS16`: SP4 `43.32k, 28.9G` vs SP1 `37.62k, 68.7G` |
| SP2 是中间折中点，但多数列不如 SP4 | SP2 显存明显低于 SP1，但吞吐多数低于或持平 SP4 |

## 2. 2.7B 补充实验结果

### 2.1 2.7B PP8/TP1, GBS8 Common Setting

目的：固定 `GBS=8`，获得 `SP1/SP4/SP8` 尽量都能跑的 common setting。这个表用于 fair comparison；`GBS16/32` 表保留为 paper-style feasibility/memory-bound 对照。

配置：

| Item | Value |
|---|---|
| Model | `m2p7b:32:2560:32` |
| GPUs | `8 x 80G` |
| PP / TP / DP | `8 / 1 / 1` |
| MICRO_BATCH / GBS | `1 / 8` |
| Number of microbatches | `8` |
| SP | `1 / 4 / 8` |
| Train iters | `100` |

<table>
  <thead>
    <tr>
      <th rowspan="2">Model Size</th>
      <th rowspan="2">Metric</th>
      <th rowspan="2">Method</th>
      <th colspan="4">Sequence Length</th>
    </tr>
    <tr>
      <th>16384</th>
      <th>24576</th>
      <th>32768</th>
      <th>65536</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="9">2.7B<br/>DeltaNet</td>
      <td rowspan="3">Throughput<br/>(k toks/s)</td>
      <td>SP1 / PP</td>
      <td>30.58</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>34.59</td>
      <td>34.67</td>
      <td>35.53</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td><strong>34.72</strong></td>
      <td>failed</td>
      <td><strong>37.20</strong></td>
      <td><strong>38.43</strong></td>
    </tr>
    <tr>
      <td rowspan="3">Logged TFLOPS/s</td>
      <td>SP1 / PP</td>
      <td>94.71</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>107.12</td>
      <td>124.97</td>
      <td>146.14</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td><strong>107.50</strong></td>
      <td>failed</td>
      <td><strong>152.98</strong></td>
      <td><strong>236.13</strong></td>
    </tr>
    <tr>
      <td rowspan="3">Peak Memory<br/>(GB)</td>
      <td>SP1 / PP</td>
      <td>68.1</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>32.4</td>
      <td>44.1</td>
      <td>56.0</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td><strong>24.7</strong></td>
      <td>failed</td>
      <td><strong>40.9</strong></td>
      <td><strong>73.5</strong></td>
    </tr>
  </tbody>
</table>

观察：

| Observation | Evidence |
|---|---|
| `GBS8` 下 16K 不是降低 baseline，反而能比较出吞吐收益 | SP8 `34.72k` vs SP1 `30.58k` |
| SP8 在 32K 和 64K 长序列下最有价值 | 32K: SP8 `37.20k`，SP1 OOM；64K: 只有 SP8 完成 |
| SP8 同时显著省显存 | 16K: `24.7G` vs SP1 `68.1G`; 32K: `40.9G` vs SP4 `56.0G` |
| 24K SP8 需要重跑确认 | 当前 log 未产出 `toks/s`，标记为 `failed` |

### 2.2 2.7B PP8/TP1, 32K GBS Sweep

目的：固定 `PP=8, TP=1, DP=1, seq=32K`，扫 `GBS/mbn`，观察 PP bubble 和 SP1 OOM 边界。

配置：

| Item | Value |
|---|---|
| Model | `m2p7b:32:2560:32` |
| SeqLen | `32768` |
| PP / TP / DP | `8 / 1 / 1` |
| MICRO_BATCH | `1` |
| GBS | `4 / 8 / 16 / 32` |
| SP | `1 / 4 / 8` |

#### Result Table

<table>
  <thead>
    <tr>
      <th rowspan="3">Model Size</th>
      <th rowspan="3">Metric</th>
      <th rowspan="3">Method</th>
      <th colspan="4">Sequence Length 32768</th>
    </tr>
    <tr>
      <th colspan="4">Global Batch / microbatch number</th>
    </tr>
    <tr>
      <th>4 / 4</th>
      <th>8 / 8</th>
      <th>16 / 16</th>
      <th>32 / 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="9">2.7B<br/>DeltaNet</td>
      <td rowspan="3">Throughput<br/>(k toks/s)</td>
      <td>SP1 / PP</td>
      <td>22.09</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>30.95</td>
      <td>35.56</td>
      <td>38.40</td>
      <td>39.86</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td><strong>34.30</strong></td>
      <td><strong>37.25</strong></td>
      <td><strong>38.77</strong></td>
      <td>39.58</td>
    </tr>
    <tr>
      <td rowspan="3">Logged TFLOPS/s</td>
      <td>SP1 / PP</td>
      <td>90.84</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>127.29</td>
      <td>146.25</td>
      <td>157.94</td>
      <td>163.95</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td><strong>141.08</strong></td>
      <td><strong>153.21</strong></td>
      <td>159.44</td>
      <td>162.79</td>
    </tr>
    <tr>
      <td rowspan="3">Peak Memory<br/>(GB)</td>
      <td>SP1 / PP</td>
      <td>70.7</td>
      <td>OOM</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>55.3</td>
      <td>56.0</td>
      <td>57.3</td>
      <td>57.3</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td><strong>40.3</strong></td>
      <td><strong>40.9</strong></td>
      <td><strong>41.6</strong></td>
      <td><strong>41.5</strong></td>
    </tr>
  </tbody>
</table>

观察：

| Observation | Evidence |
|---|---|
| 32K 下 SP1 只有 `GBS=4` 能跑 | `GBS4: 22.09k, 70.7G`; `GBS>=8` OOM |
| SP8 在低/中 GBS 下吞吐和显存都最好 | `GBS4: 34.30k, 40.3G`; `GBS8: 37.25k, 40.9G` |
| GBS 越大，SP4/SP8 吞吐差距越小 | `GBS16: 38.40k vs 38.77k`; `GBS32: 39.86k vs 39.58k` |
| SP8 显存基本稳定在 `40-42G` | `GBS4/8/16/32: 40.3/40.9/41.6/41.5G` |

### 2.3 2.7B PP4/DP2, Sequence Sweep

目的：降低 PP 长度和 microbatch 数，让 SP1 baseline 更可能跑起来，作为 sanity check。该设置改变了 pipeline 长度和 DP，不作为主实验对齐表。

配置：

| Item | Value |
|---|---|
| Model | `m2p7b:32:2560:32` |
| GPUs | `8` |
| PP / TP / DP | `4 / 1 / 2` |
| MICRO_BATCH | `1` |
| GBS | `16 / 32` |
| mbn | `8 / 16` |
| SP | `1 / 4 / 8` |

#### Result Table

<table>
  <thead>
    <tr>
      <th rowspan="3">Model Size</th>
      <th rowspan="3">Metric</th>
      <th rowspan="3">Method</th>
      <th colspan="2">Sequence Length 16384</th>
      <th colspan="2">Sequence Length 24576</th>
      <th colspan="2">Sequence Length 32768</th>
      <th colspan="2">Sequence Length 65536</th>
    </tr>
    <tr>
      <th colspan="2">Global Batch / mbn</th>
      <th colspan="2">Global Batch / mbn</th>
      <th colspan="2">Global Batch / mbn</th>
      <th colspan="2">Global Batch / mbn</th>
    </tr>
    <tr>
      <th>16 / 8</th>
      <th>32 / 16</th>
      <th>16 / 8</th>
      <th>32 / 16</th>
      <th>16 / 8</th>
      <th>32 / 16</th>
      <th>16 / 8</th>
      <th>32 / 16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">2.7B<br/>DeltaNet</td>
      <td rowspan="3">Throughput<br/>(k toks/s)</td>
      <td>SP1 / PP</td>
      <td><strong>43.52</strong></td>
      <td><strong>49.64</strong></td>
      <td>OOM</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>41.44</td>
      <td>43.17</td>
      <td>41.79</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td>39.44</td>
      <td>40.40</td>
      <td>40.43@20</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td rowspan="3">Peak Memory<br/>(GB)</td>
      <td>SP1 / PP</td>
      <td>69.8</td>
      <td>70.1</td>
      <td>OOM</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>40.2</td>
      <td>40.7</td>
      <td>55.4</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td>33.3</td>
      <td>33.8</td>
      <td>45.3@20</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
  </tbody>
</table>

观察：

| Observation | Evidence |
|---|---|
| PP4/DP2 下 16K baseline 很强 | `GBS16: SP1 43.52k`; `GBS32: SP1 49.64k` |
| Seq1F1B 仍然显著省显存 | 16K GBS32: SP1 `70.1G`, SP4 `40.7G`, SP8 `33.8G` |
| 24K 下 SP1 仍 OOM，SP4 可完成 | SP4 `41.79k, 55.4G`; SP8 只有 iter 20，需重跑确认 |
| 该组不作为主表 | PP/DP 都变了，适合作 sanity/appendix |

### 2.4 Hybrid DeltaNet + Full Attention

目的：测试 DeltaNet 层中周期性插入 full attention 层后的 Seq1F1B 表现，并比较 `average` 与 computation-balanced sequence split。

当前实现开关示例：

```bash
EXTRA_ARGS="--deltanet-hybrid-attention-period 4 --use-flash-attn"
```

配置模板：

| Item | Value |
|---|---|
| Hybrid pattern | every 4th layer uses full attention |
| Softmax layers for 32L model | `4,8,12,16,20,24,28,32` |
| DeltaNet layers | all other layers |
| Required for SP>1 | `--use-flash-attn` |

#### Result Table

<table>
  <thead>
    <tr>
      <th rowspan="2">Model Size</th>
      <th rowspan="2">Split Policy</th>
      <th rowspan="2">Method</th>
      <th colspan="3">Sequence Length</th>
      <th rowspan="2">Notes</th>
    </tr>
    <tr>
      <th>16384</th>
      <th>32768</th>
      <th>65536</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">2.7B<br/>Hybrid DeltaNet</td>
      <td rowspan="3">Average</td>
      <td>SP1 / PP</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>baseline</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>current default split</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>current default split</td>
    </tr>
    <tr>
      <td rowspan="3">Computation-balanced</td>
      <td>SP1 / PP</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>same as baseline for SP1</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>compare with average</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>compare with average</td>
    </tr>
  </tbody>
</table>

## 3. 之前已测结果

### 3.1 1.3B, SeqLen Sweep, PP4/TP1/GBS4

配置：

| Item | Value |
|---|---|
| Model | `m1p3b:24:2048:16` |
| PP / TP / DP | `4 / 1 / 2` |
| MICRO_BATCH / GBS | `1 / 4` |
| Number of microbatches | `2` |
| Train iters | `100` |

<table>
  <thead>
    <tr>
      <th rowspan="2">Model Size</th>
      <th rowspan="2">Metric</th>
      <th rowspan="2">Method</th>
      <th colspan="5">Sequence Length</th>
    </tr>
    <tr>
      <th>8192</th>
      <th>16384</th>
      <th>32768</th>
      <th>65536</th>
      <th>131072</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8">1.3B<br/>DeltaNet</td>
      <td rowspan="4">Throughput<br/>(k toks/s)</td>
      <td>SP1 / PP</td>
      <td>43.97</td>
      <td>46.04</td>
      <td>47.05</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP2</td>
      <td>44.87</td>
      <td>48.66</td>
      <td>50.45</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td><strong>49.44</strong></td>
      <td>56.33</td>
      <td>60.70</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td>37.70</td>
      <td><strong>57.57</strong></td>
      <td><strong>64.38</strong></td>
      <td><strong>68.78</strong></td>
      <td>OOM</td>
    </tr>
    <tr>
      <td rowspan="4">Peak Memory<br/>(GB)</td>
      <td>SP1 / PP</td>
      <td>15.4</td>
      <td>24.4</td>
      <td>45.6</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP2</td>
      <td>15.5</td>
      <td>25.4</td>
      <td>46.0</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP4</td>
      <td>14.1</td>
      <td>22.9</td>
      <td>41.1</td>
      <td>OOM</td>
      <td>OOM</td>
    </tr>
    <tr>
      <td>Seq1F1B SP8</td>
      <td><strong>12.2</strong></td>
      <td><strong>19.1</strong></td>
      <td><strong>33.7</strong></td>
      <td><strong>61.1</strong></td>
      <td>OOM</td>
    </tr>
  </tbody>
</table>

### 3.2 1.3B, 32K SP Sweep, PP4/TP1/GBS4

配置：

| Item | Value |
|---|---|
| Model | `m1p3b:24:2048:16` |
| SeqLen | `32768` |
| PP / TP / DP | `4 / 1 / 2` |
| MICRO_BATCH / GBS | `1 / 4` |
| Train iters | `200` |

| Method | Throughput | TFLOPS/s | Peak Memory | Notes |
|---|---:|---:|---:|---|
| SP1 / PP | 47.33k | 107.71 | 45.6G | baseline |
| SP2 | 50.87k | 115.76 | 46.0G | mild speedup |
| SP4 | 61.39k | 139.69 | 41.1G | strong speedup |
| SP8 | 64.89k | 147.65 | 33.7G | best throughput |
| SP16 | 60.20k | 137.00 | 32.0G | lower memory, slower than SP8 |

### 3.3 Original FLA vs Fused Solve-WU, 1.3B/32K/GBS4

配置：

| Item | Value |
|---|---|
| Model | `m1p3b:24:2048:16` |
| SeqLen | `32768` |
| PP / TP / DP | `4 / 1 / 2` |
| MICRO_BATCH / GBS | `1 / 4` |
| Train iters | `200` |

| FLA Path | SP1 | SP8 | Observation |
|---|---:|---:|---|
| Original FLA (`FLA_USE_FUSED_SOLVE_WU=0`) | 47.52k, 45.8G | 64.22k, 33.1G | reference |
| Fused solve-WU enabled | 47.33k, 45.6G | 64.89k, 33.7G | end-to-end gain about `0-1%` |

### 3.4 300M Long-Run Correctness, Original FLA

配置：

| Item | Value |
|---|---|
| Model | `24L / H1024 / 16 heads` |
| GPUs | `8` |
| PP / TP | `8 / 1` |
| SeqLen | `16384` |
| MICRO_BATCH / GBS | `1 / 16` |
| Train iters | `20000` |
| FLA path | original FLA, `FLA_USE_FUSED_SOLVE_WU=0` |

| Method | Train loss @ 20K | Val loss @ 20K | Val PPL @ 20K | Observation |
|---|---:|---:|---:|---|
| SP1 / PP | 3.0238 | 2.9687 | 19.47 | baseline |
| Seq1F1B SP8 | 3.0266 | 2.9665 | 19.42 | curve overlaps with SP1 |

结论：`SP1` 与 `SP8` 的 20K-step loss 曲线几乎完全重叠，说明 Seq1F1B 的 sequence split/state passing 没有破坏训练动力学。

## 4. 当前结论草稿

| Claim | Evidence |
|---|---|
| `GBS8` common setting 下 Seq1F1B 同时提升吞吐和显存效率 | 2.7B/PP8/16K/GBS8: SP8 `34.72k, 24.7G` vs SP1 `30.58k, 68.1G` |
| fused 2.7B/PP8 paper-style 设置下，SP4 是主要吞吐甜点 | 16K/24K/32K 的 GBS16/32 已完成列中，SP4 基本最高；32K/GBS32 达 `47.26k` |
| SP8 是最低显存和最长序列可行点 | 64K/GBS16 只有 SP8 完成，`47.14k, 64.0G`; 32K 下 SP8 峰值约 `36.2G` |
| Seq1F1B 在低 mbn/大 PP bubble 场景显著提升吞吐 | 1.3B/32K/GBS4: SP8 `64.89k` vs SP1 `47.33k` |
| Seq1F1B 在 paper-style 强 baseline 场景同时提升吞吐、显存和可训练长度 | 2.7B/PP8/16K/GBS16: SP4 `43.32k, 28.9G` vs SP1 `37.62k, 68.7G`; 24K 起 SP1 OOM |
| DeltaNet + Seq1F1B 长序列吞吐高于论文 GPT Seq1F1B 的 nominal 对照 | 2.7B/32K: DeltaNet SP4/SP8 约 `45-47k`, 论文 GPT Seq1F1B 约 `29-30k` |
| DeltaNet SP1 显存没有理论直觉中那么轻 | 2.7B/PP8/24K: SP1 OOM；SP2/SP4/SP8 通过 sequence split 显著降低峰值 activation |
| Hybrid attention 的核心问题是 computation-balanced sequence split | full attention 层 `O(S^2)`，DeltaNet/MLP 层 `O(S)`，average split 对 hybrid 不一定最优 |
