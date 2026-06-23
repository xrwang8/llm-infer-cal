# Contributing to llm-infer-cal

`llm-infer-cal` is a Rust workspace. Data that should remain easy to edit lives
under `data/`; executable logic lives under `crates/`.

## Dev Setup

```bash
cargo build
```

Run the full local verification loop:

```bash
cargo fmt --all --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release
./target/release/llm-infer-cal --benchmark
```

## Useful Changes

Data-only changes:

- New GPU: edit `data/hardware/gpu_database.yaml` and add/adjust a Rust test.
- Engine compatibility: edit `data/engine_compat/matrix.yaml` with source notes.
- Benchmark references: edit `data/benchmark/dataset.yaml` and keep expected
  values tied to a reproducible source.

Code changes:

- Architecture detection: `crates/llm-infer-cal-core/src/architecture/`.
- KV and weight formulas: `crates/llm-infer-cal-core/src/architecture/formulas/`.
- Fleet planning: `crates/llm-infer-cal-core/src/fleet/`.
- vLLM / SGLang command generation:
  `crates/llm-infer-cal-core/src/command_generator/`.
- CLI flags: `crates/llm-infer-cal/src/main.rs`.

## Rules

- Every user-facing number must carry the right provenance label.
- Do not mark an engine matrix entry `verified` unless it has real hardware
  evidence.
- Keep network calls out of unit tests; use static fixtures or mocked clients.
- Prefer focused regression tests for formulas, planner behavior, and output
  contracts.
- Keep Chinese output fully Chinese when `--lang zh` is used.

## Commit Messages

Conventional commit style is preferred:

- `feat(scope): ...`
- `fix(scope): ...`
- `docs: ...`
- `test: ...`
- `chore: ...`

Good scopes include `architecture`, `fleet`, `engine`, `output`, `formulas`,
`model_source`, `i18n`, and `cli`.
