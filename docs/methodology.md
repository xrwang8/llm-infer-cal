# Methodology — Where Every Number Comes From

This document audits every formula and coefficient the tool uses, so users
can evaluate trust in the output. If you spot a wrong source or a better
citation, please open a PR.

---

## Capacity-side (memory) numbers

These are **derived from public data**, no empirical factors:

### Weight bytes
**Formula**: sum of `safetensors` file sizes from the selected model source
(HuggingFace or ModelScope).

**Label**: `[verified]`.

**Why trustworthy**: direct read from the source file metadata. No estimation
step. The file sizes are the bytes you download with the model weights.

### Activation working set
**Formula**:
```
activation_bytes = max_num_batched_tokens × hidden_size × dtype_bytes × activation_factor
moe_activation_correction = 1 + active_experts / total_experts * 0.5
activation_bytes_moe = activation_bytes × moe_activation_correction
```

Default `max_num_batched_tokens` is `2048`, `dtype_bytes` is `2`, and
`activation_factor` is `2`. Dense models use a correction factor of `1.0`.
MoE models apply the routed-expert correction above, matching
`crates/llm-infer-cal-core/src/architecture/formulas/activation.rs`.

**Planner use**: activation is charged **once per engine/GPU layout**, after
TP sharding by hidden dimension. It is not multiplied by request concurrency.
Long prompts are chunked by serving engines; the long-lived per-request state is
KV cache, not activation tensors.

**Source**: vLLM profiles peak activation memory during startup and subtracts
non-KV memory from the executor memory budget before allocating KV cache. vLLM
chunked prefill uses `max_num_batched_tokens` as the per-step token budget.
SGLang exposes the same tuning shape via `--chunked-prefill-size` /
`--max-prefill-tokens`.

### KV cache per request
**Formula (standard attention)**:
```
per_token_per_layer_bytes = 2 × num_kv_heads × head_dim × dtype_bytes
effective_seq_len = sliding_window ? min(seq_len, sliding_window) : seq_len
raw_bytes = per_token_per_layer_bytes × effective_seq_len × num_layers
```

**Formula (MLA)**:
```
per_token_per_layer_bytes = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
raw_bytes = per_token_per_layer_bytes × seq_len × num_layers
```

The `kv_lora_rank` part is the compressed latent KV. `qk_rope_head_dim` covers
the decoupled RoPE key dimension used by MLA-style models that expose it in
config.

**Formula (CSA+HCA / NSA)**:
```
CSA+HCA raw_bytes = baseline × avg(1 / compress_ratio)
NSA raw_bytes = baseline × min(nsa_topk / effective_seq_len, 1.0)
paged_attention_factor = 0.75   # when paged attention is enabled, else 1.00
total_bytes = raw_bytes × paged_attention_factor
```

Sliding-window capping applies to non-sparse attention variants. CSA+HCA and
NSA apply their sparse reductions after the baseline formula.

**Label**: `[estimated]` — depends on exact runtime behavior (e.g. KV
quantization) that may differ from the baseline assumption.

**Sources**:
- Standard attention KV formula: universally cited in transformer inference
  literature.
- MLA details: DeepSeek-V2 paper (DeepSeek-AI, 2024-05)
- CSA+HCA details: DeepSeek-V4 technical report + model config.json
  `compress_ratios` field semantics.

### TP/PP-aware KV sharding
**Formula**:
```
tp_kv_shards = 1                          # MLA
tp_kv_shards = min(tp_size, num_kv_heads) # MQA/GQA/MHA
effective_kv_shards = pp_size × tp_kv_shards
per_gpu_KV = total_KV / effective_kv_shards
```

**Source**: vLLM TP implementation for KV splitting, plus vLLM/SGLang launch
conventions for multi-node TP/PP deployments. For MQA (kv_heads=1), KV
replicates across TP ranks. For GQA with kv_heads=G, it splits up to G ways.
For MLA, the latent KV cache is not split by TP in the planner; PP stages still
divide the layer stack.

**Why this matters**: single-node-only planning can report an impossible
8-GPU fit for very large models. TP/PP planning makes the failure visible and
then searches larger valid layouts such as `TP8 × PP6`.

### Fleet memory budget
**Formula**:
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

The 10% reserve matches `--gpu-memory-utilization 0.9` / SGLang
`--mem-fraction-static 0.9` style serving defaults; the 3GB floor avoids
under-reserving CUDA/runtime/framework memory on smaller GPUs.

**MoE resident weights**: all expert weights remain resident. For MoE models,
the planner estimates the routed-expert fraction from config and shards routed
experts by `min(tp_size, num_routed_experts)`, while static/shared parts shard
by TP. PP divides the layer stack. This prevents over-sharding routed experts
when expert count is smaller than TP.

---

## Performance-side (compute + bandwidth) numbers

These **depend on empirical utilization factors**. All factors are
CLI-overridable.

### Prefill latency
**Formula**:
```
FLOPs = 2 × params × input_tokens
effective_TFLOPS = peak_TFLOPS × num_gpus × utilization
latency_ms = (FLOPs / (effective_TFLOPS × 1e12)) × 1000
```

**Source**:
- **"2 × params × tokens"**: Kaplan et al. 2020 "Scaling Laws for Neural
  Language Models" — this is the forward-pass cost per parameter per token,
  used throughout the Chinchilla paper, Scaling Laws papers, and every
  transformer inference analysis since.
- **Paper link**: https://arxiv.org/abs/2001.08361

**Caveat**: This formula ignores the `O(seq_len²)` term of self-attention.
For very long sequences (>32K) this underestimates slightly. For typical
LLM inference prefill it's within ~5% of the true FLOPs.

### Decode tokens/second
**Formula**:
```
per_gpu_tokens_per_sec = memory_bandwidth × bw_util / weight_bytes_per_gpu
cluster_tokens_per_sec = per_gpu × num_gpus × cluster_comm_efficiency
```

**Source**: Decode is memory-bandwidth-bound because autoregressive
generation reads the entire model weight once per generated token.

- **vLLM paper**: Kwon et al. SOSP 2023 "Efficient Memory Management for
  Large Language Model Serving with PagedAttention". Section 2.3 discusses
  memory bandwidth as the bottleneck.
- **NVIDIA technical blog**: "Mastering LLM Techniques: Inference
  Optimization" (2023) — explicitly states the memory-bandwidth bound for
  decode.
- **Paper link**: https://arxiv.org/abs/2309.06180

**Caveat (MoE models)**: Strictly, MoE decode only reads `num_experts_per_tok`
active experts, not all experts. But in practice, vLLM batching causes most
experts to be touched across a batch, so the "all weights" assumption is
often closer to reality. The tool reports BOTH:
- Conservative (all weights) — default in summary
- Active-only (optimistic) — shown separately, tagged `optimistic`

### Utilization factors — all empirical, all overridable

| Factor | Default | Range from literature | Source |
|---|---|---|---|
| **Prefill compute utilization** | 0.40 | 0.30–0.50 | vLLM benchmarks on H100 w/ Llama-70B; NVIDIA MLPerf reports |
| **Decode memory-bandwidth utilization** | 0.50 | 0.40–0.65 | vLLM paper (A100 traces); NVIDIA inference guide |
| **Cluster communication efficiency (TP)** | 0.90 | 0.80–0.95 | NCCL AllReduce benchmarks at TP=8 on NVLink |
| **Concurrency degradation factor** | **1.00** | **User's call** | ⚠️ **No primary source.** Previously defaulted to 1.5 based on an LLM-generated report. Reset to 1.0 (no degradation). You should dial in what YOUR engine actually achieves under YOUR load. |

**Override them via CLI**:
```bash
llm-infer-cal <model> --gpu H800 \
  --prefill-util 0.45 \
  --decode-bw-util 0.55 \
  --concurrency-degradation 1.3
```

**If you've actually measured these on your stack**: please PR them to
`docs/benchmark-results/` as a contributed reference point. That's exactly
the kind of data the tool's "honest label" philosophy wants.

---

## Concurrency bounds (K / L)

### K bound (memory-capacity)
```
K = floor(per_GPU_headroom_bytes / per_GPU_KV_bytes_per_request)
```
Same logic as the fleet planner. Deterministic given K inputs.

Here `per_GPU_headroom_bytes = usable_per_gpu - resident_weight_per_gpu -
activation_working_set_per_gpu`; activation is charged once, while KV scales
with concurrent requests.

**Label**: `[estimated]` — the "headroom" calculation assumes `--gpu-memory-utilization 0.9`
matches what vLLM actually achieves. Usually within 2-3% of reality.

### L bound (compute/bandwidth at SLA)
```
L = floor(cluster_tokens_per_sec / target_per_user_tokens_per_sec / degradation_factor)
```

**Label**: `[estimated]` — depends on all four empirical factors above.

### Bottleneck classification
```
if K ≤ L: memory-capacity bound
if L < K: memory-bandwidth / compute bound
```

The split between "memory-bandwidth" and "compute" isn't distinguished in
v0.1; we always label the compute-path bottleneck as "memory bandwidth /
compute" since decode is universally bandwidth-bound.

---

## What the tool CAN'T tell you

To be maximally honest:

1. **Real-world throughput** — these numbers are upper bounds of a particular
   theoretical model. Your actual production throughput depends on:
   - Kernel fusion quality (FlashAttention vs. naive attention: 2-3× difference)
   - KV cache prefetch policies
   - Request arrival distribution (bursty vs. steady-state)
   - OS page fault behavior under memory pressure
   - Network topology if you're running multi-node

2. **Quantization accuracy impact** — the tool assumes quantization reduces
   memory+bandwidth linearly (INT4 = half of FP8 bytes). It says nothing
   about whether the model's output quality survives that quantization.

3. **Scheduling / batching strategy** — the tool treats the cluster as a
   pool of identical workers. Real schedulers (vLLM continuous batching,
   SGLang radix caching) have complex behaviors that can be 2-3× better
   or worse than the naive model here.

4. **Multi-turn / conversation effects** — decode throughput numbers assume
   independent single-turn requests. Multi-turn chat with prefix caching
   can be significantly more efficient.

---

## How to make the numbers more trustworthy

If you care about a specific deployment:

1. **Measure your engine's actual MFU + bandwidth utilization**:
   ```bash
   # vLLM exposes these via Prometheus metrics
   curl http://your-vllm/metrics | grep -E "gpu_(util|bandwidth)"
   ```
2. Pass them via `--prefill-util` and `--decode-bw-util`.
3. Run a concurrency ramp and measure actual p95 tokens/sec at each
   concurrency level.
4. Fit `degradation_factor = theoretical_ceiling / measured_ceiling` and
   pass via `--concurrency-degradation`.
5. The tool's output will now match YOUR stack within a few percent.

---

## Contributing methodology improvements

If you find a better source, a corrected formula, or a different default
that's supported by published data, open a PR against this file. Every
coefficient should have a cited source, not inherited vibes.
