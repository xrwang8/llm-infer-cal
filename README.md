# llm-infer-cal

LLM inference hardware calculator.

`llm-infer-cal` is a Rust command-line tool for estimating model weight memory,
KV-cache pressure, GPU count, rough throughput limits, and vLLM / SGLang launch
commands from real model metadata.

[中文说明](README.zh.md)

## What It Does

- Reads model metadata from HuggingFace or ModelScope.
- Detects architecture traits such as MLA, GQA/MQA/MHA, NSA, CSA+HCA, MoE,
  sliding window, and max context length.
- Reads real `safetensors` file sizes instead of relying only on
  `params x precision` guesses.
- Reconciles observed bytes against known quantization schemes.
- Plans `min` / `dev` / `prod` GPU fleets with TP and multi-node PP candidates.
- Generates vLLM or SGLang launch commands.
- Labels every user-facing number as verified, inferred, estimated, cited,
  unverified, or unknown.

This is a sizing assistant. Real deployment still needs a benchmark on the
target hardware and serving stack.

## Build

Requirements:

- Rust 1.80 or newer
- Network access when evaluating remote models

```bash
cargo build --release
./target/release/llm-infer-cal --help
```

Install from the local workspace:

```bash
cargo install --path crates/llm-infer-cal --locked
llm-infer-cal --help
```

If the installed command is not found:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## Quick Examples

HuggingFace:

```bash
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800 --lang zh
```

ModelScope:

```bash
llm-infer-cal ZhipuAI/GLM-5.2 --gpu H100 --source modelscope --lang zh
```

SGLang command generation:

```bash
llm-infer-cal Qwen/Qwen2.5-72B-Instruct --gpu H100 --engine sglang --lang zh
```

Derivation trace:

```bash
llm-infer-cal mistralai/Mixtral-8x7B-v0.1 --gpu H100 --explain --lang zh
```

Supported GPUs:

```bash
llm-infer-cal --list-gpus --lang zh
```

## Common Options

```text
llm-infer-cal [MODEL_ID] [OPTIONS]

Core:
  --gpu TEXT
  --source [huggingface|modelscope]
  --engine [vllm|sglang]
  --gpu-count INT
  --context-length INT
  --timeout-s FLOAT
  --lang [en|zh]

Performance:
  --input-tokens INT
  --output-tokens INT
  --target-tokens-per-sec FLOAT
  --prefill-util FLOAT
  --decode-bw-util FLOAT
  --concurrency-degradation FLOAT

Inspection:
  --explain
  --llm-review
  --list-gpus
  --benchmark
  --refresh
```

## Data And Layout

```text
crates/llm-infer-cal/        CLI binary
crates/llm-infer-cal-core/   Calculation library
data/hardware/               GPU database
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
