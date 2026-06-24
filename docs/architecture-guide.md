# Architecture Guide

This page explains how the Rust core turns a model config into memory and
serving recommendations.

## Detection Pipeline

1. `model_type`, `architectures`, and common shape fields are read from
   `config.json`.
2. `ArchitectureProfile` is built in
   `crates/llm-infer-cal-core/src/architecture/profile.rs`.
3. Attention traits are detected in
   `crates/llm-infer-cal-core/src/architecture/traits.rs`.
4. Weight and KV formulas run from
   `crates/llm-infer-cal-core/src/architecture/formulas/`.
5. Fleet planning runs in `crates/llm-infer-cal-core/src/fleet/planner.rs`.
6. vLLM / SGLang commands are generated from
   `crates/llm-infer-cal-core/src/command_generator/`.

Unknown architectures degrade gracefully: the tool still reports verified
`safetensors` bytes, but skips formula-derived KV and engine claims that it
cannot justify.

## Attention Priority

Detection prefers the most specific attention behavior first:

1. CSA+HCA or NSA sparse attention when config carries validated sparse fields.
2. MLA when `q_lora_rank` or `kv_lora_rank` is present.
3. MQA when `num_key_value_heads == 1`.
4. GQA when `num_key_value_heads < num_attention_heads`.
5. MHA otherwise.

For CSA+HCA, `compress_ratios` must line up with the layer count before the
detector accepts it. That guard prevents unrelated future fields with the same
name from being misclassified.

## KV Cache Formula

Standard attention:

```text
per_token_per_layer = 2 x num_kv_heads x head_dim x dtype_bytes
total_KV = per_token_per_layer x seq_len x num_hidden_layers
```

MLA:

```text
per_token_per_layer = (kv_lora_rank + qk_rope_head_dim) x dtype_bytes
total_KV = per_token_per_layer x seq_len x num_hidden_layers
```

CSA+HCA / NSA apply their sparse reduction after the baseline formula.

## TP And PP Sharding

Single-node candidates are valid TP sizes that divide `num_attention_heads`,
capped at 8 GPUs.

When a single node does not fit, the planner also tries multi-node pipeline
parallel candidates:

```text
candidate_gpus = max_tp x pp_size
```

`pp_size` must divide `num_hidden_layers`, and the current search cap is 8 PP
stages.

Per-GPU KV uses effective shards:

```text
tp_kv_shards = 1                         # MLA
tp_kv_shards = min(tp_size, num_kv_heads) # MQA/GQA/MHA
effective_kv_shards = pp_size x tp_kv_shards
per_gpu_KV = total_KV / effective_kv_shards
```

This is why a model can fail on 8 H100s but receive a larger multi-node
recommendation such as `TP8 x PP6`.

## Fleet Memory Budget

The planner treats weights and activation as fixed resident terms, and KV cache
as the term that scales with concurrency:

```text
reserved_per_gpu = max(3GB, 10% x HBM)
usable_per_gpu = HBM - reserved_per_gpu
needed_per_gpu = resident_weight_per_gpu
               + activation_working_set_per_gpu
               + concurrent_requests x per_gpu_KV
```

Activation is sized from the serving engine's batched-token profiling budget
(`2048` by default), not from full context length. For MoE models, routed expert
weights shard up to `min(tp_size, num_routed_experts)`, while static/shared
weights shard by TP; PP divides the layer stack.

## Adding Support

1. Add or update Rust tests under `crates/llm-infer-cal-core/tests/`.
2. Update architecture detection or formulas in `crates/llm-infer-cal-core/src/`.
3. Update `data/engine_compat/matrix.yaml` only when you have source evidence.
4. Run the full local verification loop from `CONTRIBUTING.md`.

Keep formulas small, explicit, and covered by regression tests.
