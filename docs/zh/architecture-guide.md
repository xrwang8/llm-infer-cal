# 架构指南

本文说明 Rust core 如何把模型 config 转成显存估算和 serving 建议。

## 识别流程

1. 从 `config.json` 读取 `model_type`、`architectures` 和常见形状字段。
2. 在 `crates/llm-infer-cal-core/src/architecture/profile.rs` 构建 `ArchitectureProfile`。
3. 在 `crates/llm-infer-cal-core/src/architecture/traits.rs` 识别 attention trait。
4. 在 `crates/llm-infer-cal-core/src/architecture/formulas/` 计算权重和 KV。
5. 在 `crates/llm-infer-cal-core/src/fleet/planner.rs` 做 GPU 规划。
6. 在 `crates/llm-infer-cal-core/src/command_generator/` 生成 vLLM / SGLang 命令。

未知架构会 graceful degrade：工具仍然报告已验证的 `safetensors` 字节数，但不会编造无法证明的 KV 公式和引擎兼容结论。

## Attention 优先级

识别顺序优先匹配更具体的行为：

1. 有经过校验的 sparse 字段时识别 CSA+HCA 或 NSA。
2. 存在 `q_lora_rank` 或 `kv_lora_rank` 时识别 MLA。
3. `num_key_value_heads == 1` 时识别 MQA。
4. `num_key_value_heads < num_attention_heads` 时识别 GQA。
5. 其他情况为 MHA。

CSA+HCA 必须确认 `compress_ratios` 和层数对齐，避免未来模型把同名字段用于其他语义时误识别。

## KV Cache 公式

标准 attention：

```text
per_token_per_layer = 2 x num_kv_heads x head_dim x dtype_bytes
effective_seq_len = sliding_window ? min(seq_len, sliding_window) : seq_len
raw_KV = per_token_per_layer x effective_seq_len x num_hidden_layers
```

MLA：

```text
per_token_per_layer = (kv_lora_rank + qk_rope_head_dim) x dtype_bytes
raw_KV = per_token_per_layer x seq_len x num_hidden_layers
```

CSA+HCA / NSA 在 baseline 公式之后再应用稀疏压缩系数：

```text
CSA+HCA raw_KV = baseline x avg(1 / compress_ratio)
NSA raw_KV = baseline x min(nsa_topk / effective_seq_len, 1.0)
paged_attention_factor = 0.75   # 启用 paged attention 时，否则为 1.00
total_KV = raw_KV x paged_attention_factor
```

MoE activation 在基础公式上再应用：

```text
moe_activation_correction = 1 + active_experts / total_experts * 0.5
```

## TP 和 PP 分摊

单节点候选是能整除 `num_attention_heads` 的 TP 张数，上限为 8 卡。

如果单节点放不下，planner 会继续尝试多机 pipeline parallel 候选：

```text
candidate_gpus = max_tp x pp_size
```

`pp_size` 必须整除 `num_hidden_layers`，当前搜索上限为 8 个 PP stage。

单卡 KV 使用有效分片数：

```text
tp_kv_shards = 1                         # MLA
tp_kv_shards = min(tp_size, num_kv_heads) # MQA/GQA/MHA
effective_kv_shards = pp_size x tp_kv_shards
per_gpu_KV = total_KV / effective_kv_shards
```

这就是为什么一个模型可能 8 张 H100 放不下，但会得到更大的多机建议，例如 `TP8 x PP6`。

## Fleet 显存预算

Planner 把权重和 activation 当作固定常驻项，把 KV cache 当作随并发增长的项：

```text
reserved_per_gpu = max(3GB, 10% x HBM)
usable_per_gpu = HBM - reserved_per_gpu
concurrent_KV = concurrent_requests x per_gpu_KV
decode_required = resident_weight_per_gpu
                + decode_activation_per_gpu
                + concurrent_KV
prefill_active_requests = max(concurrent_requests / 8, 1)
prefill_tokens = prefill_active_requests x 1500
prefill_peak_activation = activation_working_set x prefill_tokens / 2048
prefill_required = resident_weight_per_gpu
                 + prefill_peak_activation_per_gpu
                 + concurrent_KV
required_per_gpu = max(decode_required, prefill_required)
```

Activation 按 serving engine 的 batched-token profiling 预算计算（默认 `2048`），
不是按完整上下文长度计算。对 MoE 模型，routed expert 权重最多按
`min(tp_size, num_routed_experts)` 切分，static/shared 权重按 TP 切分，PP 则切分
layer stack。

## 新增支持

1. 先在 `crates/llm-infer-cal-core/tests/` 增加或更新 Rust 测试。
2. 修改 `crates/llm-infer-cal-core/src/` 下的识别逻辑或公式。
3. 只有有来源证据时才更新 `data/engine_compat/matrix.yaml`。
4. 运行 `CONTRIBUTING.md` 里的完整本地验证。

公式要小、明确，并且有回归测试覆盖。
