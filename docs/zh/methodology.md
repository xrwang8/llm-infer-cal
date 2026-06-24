# 方法论：每个数字的出处

本文审计工具用的每一个公式和系数，让用户有据可查地评估是否相信输出。发现错误来源或更好的引用，欢迎 PR。

---

## 容量侧（显存）数字

这些是**从公开数据直接推导**，不含经验系数：

### 权重字节数
**公式**：从所选数据源（HuggingFace 或 ModelScope）的文件元数据中，把 safetensors 文件大小求和。

**标签**：`[已验证]`。

**为什么可信**：直接读取数据源返回的文件元数据。无估算步骤。这个字节数就是下载模型权重时得到的字节数。

### Activation 工作集
**公式**：
```
activation_bytes = max_num_batched_tokens × hidden_size × dtype_bytes × activation_factor
moe_activation_correction = 1 + active_experts / total_experts * 0.5
activation_bytes_moe = activation_bytes × moe_activation_correction
```

默认 `max_num_batched_tokens` 为 `2048`，`dtype_bytes` 为 `2`，
`activation_factor` 为 `2`。Dense 模型修正系数是 `1.0`；MoE 模型按上面的
routed-expert 修正项计算，和
`crates/llm-infer-cal-core/src/architecture/formulas/activation.rs` 保持一致。

**Planner 用法**：activation 按**每个 engine/GPU layout 一次性**计入，并按
TP 的 hidden 维度切分。它不会乘以请求并发。长 prompt 会被 serving engine
分块处理；真正随请求数和上下文长度常驻增长的是 KV cache，不是 activation tensor。

**来源**：vLLM 启动时会 profile peak activation memory，并在给 KV cache 分配前扣除
non-KV memory。vLLM chunked prefill 使用 `max_num_batched_tokens` 作为每步 token
预算。SGLang 也通过 `--chunked-prefill-size` / `--max-prefill-tokens` 暴露同类调参。

### 单请求 KV Cache
**公式（标准 attention）**：
```
per_token_per_layer_bytes = 2 × num_kv_heads × head_dim × dtype_bytes
effective_seq_len = sliding_window ? min(seq_len, sliding_window) : seq_len
raw_bytes = per_token_per_layer_bytes × effective_seq_len × num_layers
```

**公式（MLA）**：
```
per_token_per_layer_bytes = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
raw_bytes = per_token_per_layer_bytes × seq_len × num_layers
```

`kv_lora_rank` 是压缩 latent KV，`qk_rope_head_dim` 是 MLA 类模型 config 中暴露的 decoupled RoPE key 维度。

**公式（CSA+HCA / NSA）**：
```
CSA+HCA raw_bytes = baseline × avg(1 / compress_ratio)
NSA raw_bytes = baseline × min(nsa_topk / effective_seq_len, 1.0)
paged_attention_factor = 0.75   # 启用 paged attention 时，否则为 1.00
total_bytes = raw_bytes × paged_attention_factor
```

sliding-window 截断只应用在非 sparse attention 变体上。CSA+HCA 和 NSA
会在 baseline 之后再应用稀疏缩放。

**标签**：`[估算]`——依赖 runtime 的具体行为（例如 KV 量化），可能和 baseline 假设有出入。

**来源**：
- 标准 attention KV 公式：transformer 推理文献里普遍引用
- MLA 细节：DeepSeek-V2 paper（DeepSeek-AI，2024-05）
- CSA+HCA 细节：DeepSeek-V4 技术报告 + 模型 config.json `compress_ratios` 字段语义

### TP/PP 感知的 KV 分摊
**公式**：
```
tp_kv_shards = 1                          # MLA
tp_kv_shards = min(tp_size, num_kv_heads) # MQA/GQA/MHA
effective_kv_shards = pp_size × tp_kv_shards
per_gpu_KV = total_KV / effective_kv_shards
```

**来源**：vLLM 的 TP KV 分摊实现，以及 vLLM / SGLang 多机 TP/PP 启动约定。MQA（kv_heads=1）时 KV 在 TP 维度复制；GQA（kv_heads=G）最多切 G 份；MLA 在 planner 里不按 TP 切 latent KV，但 PP stage 会分摊层数。

**为什么重要**：只看单节点会把超大模型误报成 8 卡可运行。TP/PP 规划会先暴露 8 卡放不下，再搜索 `TP8 × PP6` 这类有效多机布局。

### Fleet 显存预算
**公式**：
```
reserved_per_gpu = max(3GB, 10% × HBM)
usable_per_gpu = HBM - reserved_per_gpu
concurrent_KV = concurrent_requests × per_gpu_KV
decode_required = resident_weight_per_gpu
                + decode_activation_per_gpu
                + concurrent_KV
prefill_active_requests = max(concurrent_requests / 8, 1)
prefill_tokens = prefill_active_requests × 1500
prefill_peak_activation = activation_working_set × prefill_tokens / 2048
prefill_required = resident_weight_per_gpu
                 + prefill_peak_activation_per_gpu
                 + concurrent_KV
required_per_gpu = max(decode_required, prefill_required)
```

10% reserve 对齐 `--gpu-memory-utilization 0.9` / SGLang
`--mem-fraction-static 0.9` 这类 serving 默认口径；3GB floor 避免小卡低估
CUDA/runtime/framework 预留。

**MoE 常驻权重**：所有 expert 权重都会 resident。对 MoE 模型，planner 从 config
估算 routed expert 占比，并按 `min(tp_size, num_routed_experts)` 切 routed experts；
static/shared 部分按 TP 切分，PP 则切分 layer stack。这样可以避免在 expert 数小于
TP 时把 routed experts 过度切分。

---

## 性能侧（算力 + 带宽）数字

这些**依赖经验利用率系数**。所有系数都可以通过 CLI 覆盖。

### Prefill 延迟
**公式**：
```
FLOPs = 2 × params × input_tokens
effective_TFLOPS = peak_TFLOPS × num_gpus × utilization
latency_ms = (FLOPs / (effective_TFLOPS × 1e12)) × 1000
```

**来源**：
- **"2 × params × tokens"**：Kaplan et al. 2020 "Scaling Laws for Neural Language Models"——这是每参数每 token 的前向计算成本，被 Chinchilla paper、Scaling Laws 系列论文和此后每一篇 transformer 推理分析反复使用。
- **论文链接**：https://arxiv.org/abs/2001.08361

**注意**：这个公式忽略了 self-attention 的 `O(seq_len²)` 项。超长序列（>32K）会略低估。典型 LLM 推理 prefill 场景误差 ~5%。

### Decode 每秒 tokens
**公式**：
```
per_gpu_tokens_per_sec = memory_bandwidth × bw_util / weight_bytes_per_gpu
cluster_tokens_per_sec = per_gpu × num_gpus × cluster_comm_efficiency
```

**来源**：Decode 是内存带宽受限的，因为自回归生成每生成一个 token 都要把整个模型权重读一遍。

- **vLLM paper**：Kwon et al. SOSP 2023 "Efficient Memory Management for Large Language Model Serving with PagedAttention"。2.3 节讨论了显存带宽瓶颈。
- **NVIDIA 技术博客**："Mastering LLM Techniques: Inference Optimization"（2023）——明确指出 decode 的显存带宽瓶颈。
- **论文链接**：https://arxiv.org/abs/2309.06180

**注意（MoE 模型）**：严格说 MoE decode 只读 `num_experts_per_tok` 个激活专家，不是全部。但实际 vLLM 批处理下一批 tokens 会触及大部分专家，所以"全量"假设往往更接近真实。工具**同时**输出：
- 保守（全量权重）——默认主显示
- 仅激活（乐观）——单独一行，标注 `optimistic`

### 利用率系数——全部经验值，全部可覆盖

| 系数 | 默认值 | 文献区间 | 来源 |
|---|---|---|---|
| **Prefill 算力利用率** | 0.40 | 0.30–0.50 | vLLM 在 H100 上 Llama-70B 的 benchmark；NVIDIA MLPerf 报告 |
| **Decode 显存带宽利用率** | 0.50 | 0.40–0.65 | vLLM paper（A100 trace）；NVIDIA 推理指南 |
| **集群通信效率（TP）** | 0.90 | 0.80–0.95 | NCCL AllReduce 在 TP=8、NVLink 上的 benchmark |
| **并发退化系数** | **1.00** | **用户决定** | ⚠️ **没有 primary source。** 之前默认 1.5 是从一份 LLM 生成的报告里抄来的。已经改回 1.0（无退化，诚实基线）。你应该根据 **你的 engine 在你的负载下实测到的**来调。 |

**通过 CLI 覆盖**：
```bash
llm-infer-cal <model> --gpu H800 \
  --prefill-util 0.45 \
  --decode-bw-util 0.55 \
  --concurrency-degradation 1.3
```

**如果你真的在自己的 stack 上测过这些数字**：请 PR 到 `docs/benchmark-results/` 作为贡献的参考点。这正是工具"诚实标签"哲学最欢迎的数据。

---

## 并发上限（K / L）

### K 上限（显存容量）
```
K = floor(单卡余量字节 / 单请求 KV 字节)
```
和 fleet planner 同一逻辑。给定 K 的输入，结果确定。

这里 `单卡余量字节 = usable_per_gpu - resident_weight_per_gpu -
activation_working_set_per_gpu`；activation 只计一次，KV 才随并发请求数增长。

**标签**：`[估算]`——"余量"的计算假设 `--gpu-memory-utilization 0.9` 和 vLLM 实际表现一致。通常误差 2-3%。

### L 上限（SLA 下的算力/带宽）
```
L = floor(集群 tokens_per_sec / 每用户目标 tokens_per_sec / 退化系数)
```

**标签**：`[估算]`——依赖上面 4 个经验系数。

### 瓶颈分类
```
K ≤ L: 显存容量瓶颈
L < K: 显存带宽 / 算力瓶颈
```

v0.1 不区分"显存带宽"和"算力"——统一标为"memory bandwidth / compute"，因为 decode 普遍是带宽瓶颈。

---

## 工具不能告诉你什么

最大程度地诚实：

1. **真实吞吐**——这些数字是某个理论模型的**上界**。真实生产吞吐取决于：
   - Kernel 融合质量（FlashAttention vs 朴素 attention：2-3× 差异）
   - KV cache 预取策略
   - 请求到达分布（突发 vs 稳态）
   - 内存压力下的 OS page fault 行为
   - 多机部署时的网络拓扑

2. **量化对精度的影响**——工具假设量化线性减少内存+带宽（INT4 = FP8 字节数的一半）。它不判断模型量化后输出质量还能不能接受。

3. **调度 / 批处理策略**——工具把集群视为一池同质 worker。真实调度器（vLLM continuous batching、SGLang radix caching）有复杂行为，可能比这里的朴素模型好 2-3 倍，也可能差。

4. **多轮 / 对话效应**——decode 吞吐数字假设独立单轮请求。有 prefix caching 的多轮对话效率显著更高。

---

## 怎样让数字更可信

如果你关心某个具体部署：

1. **测量你的 engine 的真实 MFU + 带宽利用率**：
   ```bash
   # vLLM 通过 Prometheus metrics 暴露
   curl http://your-vllm/metrics | grep -E "gpu_(util|bandwidth)"
   ```
2. 通过 `--prefill-util` 和 `--decode-bw-util` 传进来。
3. 做并发 ramp 测试，记录每个并发水位的实测 p95 tokens/sec。
4. 拟合 `degradation_factor = 理论上限 / 实测上限`，通过 `--concurrency-degradation` 传入。
5. 工具输出会匹配**你的 stack**，误差百分之几。

---

## 贡献方法论改进

如果你找到更好的来源、修正公式、或者发现不同的默认值有已发表的数据支持，请 PR 到本文件。每个系数都该有引用，不能是"凭感觉"。
