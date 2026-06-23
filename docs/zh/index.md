# llm-infer-cal

`llm-infer-cal` 是一个架构感知的大模型推理硬件计算器。

给它一个 HuggingFace 或 ModelScope 模型 ID，以及目标 GPU，它会输出：

- 真实 `safetensors` 权重字节数
- 模型架构特征识别
- 常见上下文长度下的 KV Cache 估算
- `min` / `dev` / `prod` 三档 GPU 推荐
- TP 和多机 PP 布局
- vLLM 或 SGLang 启动命令
- 每个重要数字的来源标签

计算器使用 Rust 编写，命令行名称是 `llm-infer-cal`。

## 安装

本地编译：

```bash
cargo build --release
./target/release/llm-infer-cal --help
```

安装到 Cargo bin 目录：

```bash
cargo install --path crates/llm-infer-cal --locked
```

## 快速开始

```bash
llm-infer-cal ZhipuAI/GLM-5.2 --gpu H100 --source modelscope --lang zh
```

常用命令见[快速开始](quickstart.md)，内部公式见[架构指南](architecture-guide.md)，数字来源见[方法论](methodology.md)。

## 验证

```bash
llm-infer-cal --benchmark
```

benchmark 是项目回归检查，用于开发和 CI，不是公开排行榜。

## License

Apache-2.0，见 [LICENSE](../../LICENSE)。
