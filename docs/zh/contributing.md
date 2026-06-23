# 参与贡献

`llm-infer-cal` 是 Rust workspace。可编辑数据放在 `data/`，命令行和计算逻辑放在 `crates/`。

## 本地检查

```bash
cargo fmt --all --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release
./target/release/llm-infer-cal --benchmark
```

## 修改位置

- GPU 规格：`data/hardware/gpu_database.yaml`
- 引擎兼容矩阵：`data/engine_compat/matrix.yaml`
- Benchmark 数据集：`data/benchmark/dataset.yaml`
- 架构识别：`crates/llm-infer-cal-core/src/architecture/`
- 公式：`crates/llm-infer-cal-core/src/architecture/formulas/`
- GPU 规划：`crates/llm-infer-cal-core/src/fleet/`
- 启动命令生成：`crates/llm-infer-cal-core/src/command_generator/`
- CLI：`crates/llm-infer-cal/src/main.rs`

## 审查规则

- 每个面向用户的数字都要有诚实标签。
- `verified` 只用于直接读取的数据或有真实硬件证据的条目。
- 估算公式要能在 `docs/methodology.md` 里解释清楚。
- 改 planner 或公式前先补回归测试。
- `--lang zh` 的输出要保持中文。
