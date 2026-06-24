# llm-infer-cal

LLM inference hardware calculator.

`llm-infer-cal` reads a model's real metadata and answers the questions you ask
before a deployment: how much VRAM the weights take, how fast KV cache grows,
how many GPUs you need and in what TP/PP layout, the rough throughput ceiling,
and the exact vLLM / SGLang launch command to start from.

Every user-facing number carries a provenance label — `verified`, `inferred`,
`estimated`, `cited`, or `unknown` — so you can tell a measured byte count from
a back-of-envelope estimate at a glance.

It is a sizing assistant, not a replacement for a benchmark on your real
hardware and serving stack.

[中文说明](README.zh.md)

## What It Does

- Reads model metadata from HuggingFace, ModelScope, or a built-in catalog.
- Detects architecture traits: MLA, GQA/MQA/MHA, NSA, CSA+HCA, MoE, sliding
  window, RoPE scaling, and max context length.
- Sums real `safetensors` file sizes instead of trusting `params × precision`.
- Reconciles observed bytes against known quantization schemes (FP16/BF16, FP8,
  INT8, FP4/FP8 mixed, INT4, GPTQ/AWQ INT4).
- Plans `min` / `dev` / `prod` GPU fleets with single-node TP, cross-node TP,
  and pipeline-parallel candidates.
- Estimates prefill latency, decode tokens/sec, and a concurrency ceiling with
  an explicit bottleneck (memory-capacity vs bandwidth/compute).
- Generates a ready-to-run vLLM or SGLang command.
- Ships a CLI, an HTTP API, and a React web UI.

## Core Algorithm

The tool splits every estimate into two halves: **capacity** (does it fit in
VRAM) and **performance** (how fast it runs). Capacity numbers are derived from
public data with no fudge factors; performance numbers depend on empirical
utilization factors that you can override.

### 1. Architecture detection

`detect()` parses `config.json` and classifies the model:

- **Family**: transformer or state-space (Mamba/Jamba are flagged unsupported
  in v0.1 — they have no KV cache).
- **Attention variant**: MHA / GQA / MQA / MLA / NSA / CSA+HCA, inferred from
  `num_attention_heads`, `num_key_value_heads`, `kv_lora_rank`, `compress_ratios`,
  `nsa_topk`.
- **MoE traits**: routed/shared expert counts, experts-per-token, expert
  intermediate size.
- **Position**: RoPE type/theta/scaling and `max_position_embeddings`.

The variant drives both the KV-cache formula and the valid TP layouts, so this
step gates everything downstream.

### 2. Weight bytes and quantization reconciliation

Weight VRAM is the **sum of real `safetensors` file sizes** — the bytes you
actually download — labelled `verified`. The tool independently estimates total
parameter count from the shape (`embed + head + layers × (attn + ffn + norms)`,
with the MoE FFN counting all experts), then computes observed bits-per-param:

```
bits_per_param = observed_bytes × 8 / total_params
```

It reconciles that against known schemes (FP16=16b, FP8=8b, INT4=4b, GPTQ/AWQ
≈4.4b, …). When `config.json` declares a quantization scheme **and** the
predicted bytes land within 15% of observed, the scheme is trusted as
`verified`; otherwise the tool samples a real shard's tensor dtypes, and finally
falls back to a tolerance match on bits-per-param.

### 3. KV cache per request

Per-token, per-layer KV bytes depend on the attention variant:

```
standard:  per_layer_per_token = 2 × num_kv_heads × head_dim × dtype_bytes
MLA:       per_layer_per_token = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
```

The baseline is `per_layer_per_token × effective_seq_len × num_layers`, then:

- **Sliding window** caps `effective_seq_len` for non-sparse variants.
- **CSA+HCA** scales by the average compression fraction `avg(1 / compress_ratio)`.
- **NSA** scales by sparsity `min(nsa_topk / effective_seq_len, 1)`.
- **Paged attention** (optional) multiplies by `0.75`.

KV bytes are reported at several context lengths (4K / 32K / 128K / model max),
because KV — not activation — is the per-request state that scales concurrency.

### 4. Fleet planning (TP / PP)

This is the heart of the tool. A single-node-only planner can claim an
impossible "fits in 8 GPUs" for a huge model; this planner makes the failure
visible and then searches larger valid layouts.

**Valid TP** must divide `num_attention_heads`. Single-node TP is capped at 8;
larger head counts unlock cross-node TP up to 64; pipeline parallel (up to 8
stages) divides the layer stack. Candidate GPU counts are all combinations of
those.

**Per-GPU resident weight** shards by the layout. Dense models divide by `TP`
after `PP` splits the layers. MoE models split routed-expert weight by
`min(TP, num_routed_experts)` and static/shared weight by `TP`, so routed
experts are never over-sharded.

**Per-GPU KV** divides by `effective_kv_shards = PP × kv_shards(TP)`, where MLA
replicates (`kv_shards = 1`) and GQA/MHA split up to `min(TP, num_kv_heads)`.

**Fit check** takes the peak of decode and prefill memory:

```
reserved_per_gpu  = max(3 GB, 10% × HBM)        # ~ --gpu-memory-utilization 0.9
usable_per_gpu    = HBM − reserved_per_gpu
concurrent_KV     = concurrent_requests × per_gpu_KV
decode_required   = weight_per_gpu + decode_activation_per_gpu + concurrent_KV
prefill_required  = weight_per_gpu + prefill_peak_activation_per_gpu + concurrent_KV
required_per_gpu  = max(decode_required, prefill_required) ≤ usable_per_gpu
```

`min`/`dev`/`prod` tiers plan for 1 / 8 / 16 concurrent requests; `--target-concurrency`
plans a single tier at your number. Each option also reports the max concurrent
requests it can hold via binary search on the fit check.

### 5. Performance and concurrency bounds

**Prefill** is compute-bound:

```
FLOPs       = 2 × active_params × input_tokens          # Kaplan et al. 2020
latency_ms  = FLOPs / (peak_TFLOPS × num_gpus × prefill_util) × 1000
```

**Decode** is memory-bandwidth-bound (every token reads the resident weights):

```
per_gpu_tok/s     = mem_bandwidth × bw_util / weight_bytes_per_gpu
cluster_tok/s     = per_gpu_tok/s × num_gpus × comm_eff × nvlink_penalty
```

For MoE the tool reports both a conservative "all weights" number and an
optimistic "active-only" number tagged `optimistic`.

**Concurrency ceiling** is the min of two bounds:

```
K (capacity)  = per_gpu_headroom / per_gpu_KV_per_request
L (SLA)       = cluster_tok/s / target_tok/s_per_user / degradation_factor
max_concurrent = min(K, L)
```

`K ≤ L` → memory-capacity bound; `L < K` → bandwidth/compute bound.

### Empirical factors (all overridable)

| Factor | Default | CLI flag |
|---|---|---|
| Prefill compute utilization | 0.40 | `--prefill-util` |
| Decode bandwidth utilization | 0.50 | `--decode-bw-util` |
| Cluster comm efficiency (TP) | 0.90 | — |
| Concurrency degradation | 1.00 | `--concurrency-degradation` |

The full derivation, every coefficient, and its source live in
[`docs/methodology.md`](docs/methodology.md). Run any command with `--explain`
to print the trace for that specific model.

## Install

Requirements: Rust 1.80+, and network access when evaluating remote models.

```bash
# Build from source
cargo build --release
./target/release/llm-infer-cal --help

# Or install into your Cargo bin
cargo install --path crates/llm-infer-cal --locked
llm-infer-cal --help
```

If the installed command is not found, add Cargo's bin dir to `PATH`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## Usage

Basic sizing — model id plus a GPU:

```bash
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800
```

From ModelScope instead of HuggingFace:

```bash
llm-infer-cal ZhipuAI/GLM-5.2 --gpu H100 --source modelscope
```

Generate an SGLang launch command:

```bash
llm-infer-cal Qwen/Qwen2.5-72B-Instruct --gpu H100 --engine sglang
```

Plan for a concrete concurrency target and longer context:

```bash
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800 \
  --context-length 65536 --target-concurrency 64
```

Print the full derivation trace:

```bash
llm-infer-cal mistralai/Mixtral-8x7B-v0.1 --gpu H100 --explain
```

Tune the empirical factors to match your measured stack:

```bash
llm-infer-cal Qwen/Qwen2.5-72B-Instruct --gpu H100 \
  --prefill-util 0.45 --decode-bw-util 0.55 --concurrency-degradation 1.3
```

List supported GPUs, or emit JSON for scripting:

```bash
llm-infer-cal --list-gpus
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800 --json
```

### Common Options

```text
llm-infer-cal [MODEL_ID] [OPTIONS]

Core:
  --gpu TEXT                       GPU id, e.g. H800, A100-80G (see --list-gpus)
  --source [huggingface|modelscope|builtin]
  --engine [vllm|sglang]
  --gpu-count INT                  Force GPU count (otherwise recommended)
  --context-length INT             Context length for KV estimation
  --lang [en|zh]                   Output language (auto-detects from LANG)
  --format [text|json] / --json

Capacity:
  --kv-cache-bits INT              KV precision in bits (default 16)
  --paged-attention                Apply the 0.75 paged-attention KV factor
  --target-concurrency INT         Plan a fleet for this concurrency
  --speculative-draft-model TEXT   Add a draft/EAGLE model's weight to VRAM
  --cpu-offload-gb FLOAT           Per-GPU weight budget moved off the GPU

Performance:
  --input-tokens INT               Prefill token budget (default 2000)
  --output-tokens INT              Decode token budget (default 512)
  --target-tokens-per-sec FLOAT    Per-user decode SLA, drives the L bound
  --prefill-util FLOAT             Prefill compute utilization (default 0.40)
  --decode-bw-util FLOAT           Decode bandwidth utilization (default 0.50)
  --concurrency-degradation FLOAT  High-concurrency throughput penalty

Inspection:
  --explain                        Print the full derivation trace
  --llm-review                     EXPERIMENTAL second opinion from an LLM
  --list-gpus
  --benchmark                      Run the curated regression dataset
  --refresh                        Bypass cache and re-fetch metadata
```

### Web UI and HTTP API

A React frontend (`web/frontend`) talks to the `llm-infer-cal-web` HTTP server.

```bash
# Backend: serves the API on 127.0.0.1:8080 (override with LLM_INFER_CAL_WEB_ADDR)
cargo run -p llm-infer-cal-web

# Frontend: dev server on 127.0.0.1:5173
cd web/frontend && npm install && npm run dev
```

> The web API binds to localhost and has no built-in authentication. Put it
> behind a reverse proxy with auth before exposing it beyond your machine.

API surface:

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/health` | Liveness check |
| `GET`  | `/api/models` | Built-in model catalog |
| `GET`  | `/api/gpus` | Supported GPU specs |
| `POST` | `/api/evaluate` | Run an evaluation (accepts the same knobs as the CLI; pass `gpus` to compare several) |

```bash
curl -s localhost:8080/api/evaluate \
  -H 'content-type: application/json' \
  -d '{"model_id":"deepseek-ai/DeepSeek-V3","gpu":"H800","source":"builtin"}'
```

### Docker and Helm deployment

Build the production image:

```bash
make docker-build IMAGE_REPOSITORY=172.28.0.32:3443/xrwang/llm-infer-cal IMAGE_TAG=0.1.0
```

Lint and render the Helm chart:

```bash
make helm-lint
make helm-template IMAGE_REPOSITORY=172.28.0.32:3443/xrwang/llm-infer-cal IMAGE_TAG=0.1.0
```

Package the chart:

```bash
make helm-package
```

Install or upgrade with Helm:

```bash
make helm-install \
  HELM_RELEASE=llm-infer-cal \
  HELM_NAMESPACE=llm-infer-cal \
  IMAGE_REPOSITORY=172.28.0.32:3443/xrwang/llm-infer-cal \
  IMAGE_TAG=0.1.0 \
  INGRESS_ENABLED=true \
  INGRESS_HOST=llm-infer-cal.example.com \
  INGRESS_PATH=/ \
  INGRESS_PATH_TYPE=Prefix
```

Set `INGRESS_HOST` and `INGRESS_PATH` to the host and path used by your cluster Ingress.
If Ingress is disabled, port-forward the service:

```bash
kubectl -n llm-infer-cal port-forward svc/llm-infer-cal 8080:80
```

Then open `http://127.0.0.1:8080`.

For registry and Ingress overrides:

```bash
helm upgrade --install llm-infer-cal charts/llm-infer-cal \
  --namespace llm-infer-cal \
  --create-namespace \
  --set-string image.repository=172.28.0.32:3443/xrwang/llm-infer-cal \
  --set-string image.tag=0.1.0 \
  --set ingress.enabled=true \
  --set-string ingress.hosts[0].host=llm-infer-cal.example.com \
  --set-string ingress.hosts[0].paths[0].path=/ \
  --set-string ingress.hosts[0].paths[0].pathType=Prefix
```

See [docs/deployment.md](docs/deployment.md) for the full deployment notes.

## Project Layout

```text
crates/llm-infer-cal/        CLI binary
crates/llm-infer-cal-core/   Calculation library (the core algorithm)
crates/llm-infer-cal-web/    Axum HTTP API
web/frontend/                React + Vite UI
data/hardware/               GPU database (47 entries, many vendors)
data/engine_compat/          vLLM / SGLang compatibility matrix
data/benchmark/              Regression benchmark dataset
docs/                        Methodology and usage docs
```

## Verification

```bash
cargo fmt --all --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release
./target/release/llm-infer-cal --benchmark
```

## License

Apache-2.0. See [LICENSE](LICENSE).
