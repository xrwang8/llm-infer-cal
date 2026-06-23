# llm-infer-cal

LLM inference hardware calculator.

`llm-infer-cal` is a command-line tool for estimating whether a model can run on a target GPU setup, how many GPUs are needed, and which serving command is a reasonable starting point.

The Rust CLI is the primary runtime. The Python modules in this repository are kept for compatibility, regression tests, and parity checks while the Rust implementation matures.

[中文说明](README.zh.md)

## What It Does

- Reads model metadata from HuggingFace or ModelScope.
- Detects model architecture, attention layout, MoE traits, sliding window settings, and KV-cache shape.
- Estimates weight memory, KV-cache memory, fleet size, concurrency limits, and rough prefill/decode behavior.
- Matches the model against vLLM and SGLang compatibility rules.
- Generates a serving command that can be edited and used as a deployment starting point.
- Marks output values by source, such as verified data, inferred values, estimates, citations, and unknowns.

This is a sizing assistant, not a replacement for running your own benchmark on real hardware.

## Build The CLI

Requirements:

- Rust 1.80 or newer
- Network access when evaluating remote models

Build a release binary:

```bash
cd /Users/xrwang/go/src/github.com/xrwang8/llm-infer-cal
cargo build --release
```

Run it directly:

```bash
./target/release/llm-infer-cal --help
./target/release/llm-infer-cal --list-gpus --lang zh
```

Install it into your Cargo bin directory:

```bash
cargo install --path crates/llm-infer-cal --locked
llm-infer-cal --help
```

If the installed command is not found, add Cargo's bin directory to your shell path:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## Quick Examples

Evaluate a HuggingFace model:

```bash
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800 --lang zh
```

Use ModelScope as the metadata source:

```bash
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800 --source modelscope --lang zh
```

Estimate a longer context window:

```bash
llm-infer-cal Qwen/Qwen2.5-72B-Instruct \
  --gpu A100-80G \
  --context-length 32768 \
  --lang zh
```

Print the derivation trace:

```bash
llm-infer-cal mistralai/Mixtral-8x7B-v0.1 --gpu H100 --explain --lang zh
```

List supported GPUs:

```bash
llm-infer-cal --list-gpus --lang zh
```

## Output Language

Use `--lang zh` for Chinese output:

```bash
llm-infer-cal Qwen/Qwen2.5-7B --gpu RTX4090 --lang zh
```

Use `--lang en` for English output. If `--lang` is omitted, the tool tries to infer the language from the `LANG` environment variable.

## Data Sources

Default source:

```bash
llm-infer-cal Qwen/Qwen2.5-7B --gpu RTX4090
```

This uses HuggingFace metadata.

ModelScope source:

```bash
llm-infer-cal Qwen/Qwen2.5-7B --gpu RTX4090 --source modelscope
```

Authentication environment variables:

```bash
export HF_TOKEN=...
export MODELSCOPE_API_TOKEN=...
```

The tool also accepts `HUGGING_FACE_HUB_TOKEN` and `MODELSCOPE_TOKEN`.

Use `--refresh` when you want to bypass cached model metadata:

```bash
llm-infer-cal Qwen/Qwen2.5-7B --gpu RTX4090 --refresh --lang zh
```

## Common Options

```text
llm-infer-cal [MODEL_ID] [OPTIONS]

Core:
  --gpu TEXT                     Target GPU, for example H800 or A100-80G
  --source [huggingface|modelscope]
  --engine [vllm|sglang]
  --gpu-count INT                Force a GPU count instead of auto-planning
  --context-length INT           Context length for KV-cache estimation
  --timeout-s FLOAT              Network timeout for model metadata requests
  --lang [en|zh]

Performance inputs:
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

## LLM Review

`--llm-review` is optional. It sends the derivation trace to an OpenAI-compatible API and prints a second opinion. It does not override deterministic calculator output.

Environment variables:

```bash
export LLM_CAL_REVIEWER_API_KEY=...
export LLM_CAL_REVIEWER_BASE_URL=https://api.openai.com/v1
export LLM_CAL_REVIEWER_MODEL=gpt-4o
```

## Benchmark Command

`llm-infer-cal --benchmark` is a regression check for this project. It runs a curated model set and verifies that weight sizing, quantization recognition, and selected planner outputs still match expected values.

It is meant for CI and development. It is not a public leaderboard.

```bash
llm-infer-cal --benchmark
```

Exit code:

- `0`: all checks passed
- `1`: at least one check failed

## Project Layout

```text
crates/llm-infer-cal/        Rust CLI
crates/llm-infer-cal-core/   Rust calculation library
src/llm_cal/                 Python compatibility and reference modules
tests/                       Python parity and regression tests
docs/                        Methodology and generated reference pages
```

## Development Checks

Rust:

```bash
cargo fmt --all --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

Python parity checks:

```bash
PYTHONPATH=src python -m ruff check src tests
PYTHONPATH=src python -m pytest -q
```

Live ModelScope parity check:

```bash
LLM_CAL_LIVE_MODEL_PARITY=1 PYTHONPATH=src python -m pytest tests/test_live_modelscope_value_parity.py -q
```

## License

Apache-2.0. See [LICENSE](LICENSE).
