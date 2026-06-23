# Contributing

`llm-infer-cal` is maintained as a Rust workspace. The editable data files live
in `data/`; the CLI and calculation logic live in `crates/`.

## Local Checks

```bash
cargo fmt --all --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release
./target/release/llm-infer-cal --benchmark
```

## Where To Change Things

- GPU specs: `data/hardware/gpu_database.yaml`
- Engine compatibility: `data/engine_compat/matrix.yaml`
- Benchmark dataset: `data/benchmark/dataset.yaml`
- Architecture detector: `crates/llm-infer-cal-core/src/architecture/`
- Formulas: `crates/llm-infer-cal-core/src/architecture/formulas/`
- Fleet planner: `crates/llm-infer-cal-core/src/fleet/`
- Command generation: `crates/llm-infer-cal-core/src/command_generator/`
- CLI: `crates/llm-infer-cal/src/main.rs`

## Review Rules

- Label every user-facing number honestly.
- Keep `verified` for values read directly from a source or proven by hardware
  evidence.
- Keep estimates explainable in `docs/methodology.md`.
- Add regression tests before changing planner or formula behavior.
- Keep `--lang zh` output in Chinese.
