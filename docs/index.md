# llm-infer-cal

`llm-infer-cal` is an architecture-aware LLM inference hardware calculator.

Give it a HuggingFace or ModelScope model id and a GPU type, and it reports:

- real `safetensors` weight bytes
- detected architecture traits
- KV cache estimates at common context lengths
- `min` / `dev` / `prod` GPU fleet recommendations
- TP and multi-node PP layout choices
- vLLM or SGLang launch commands
- provenance labels for every important number

The calculator is written in Rust and ships as the `llm-infer-cal` CLI.

## Install

Build locally:

```bash
cargo build --release
./target/release/llm-infer-cal --help
```

Install from the workspace:

```bash
cargo install --path crates/llm-infer-cal --locked
```

## Quickstart

```bash
llm-infer-cal ZhipuAI/GLM-5.2 --gpu H100 --source modelscope --lang zh
```

Use [Quickstart](quickstart.md) for common commands, [Architecture Guide](architecture-guide.md)
for the internal formulas, and [Methodology](methodology.md) for provenance.

## Validation

```bash
llm-infer-cal --benchmark
```

The benchmark command is a project regression check. It is intended for CI and
development; it is not a public leaderboard.

## License

Apache-2.0. See [LICENSE](../LICENSE).
