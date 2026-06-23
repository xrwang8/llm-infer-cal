# llm-infer-cal

大模型推理硬件计算器。

`llm-infer-cal` 是一个 Rust 命令行工具。它会读取模型真实元数据，估算权重显存、KV Cache 压力、推荐 GPU 数量、粗略吞吐上限，并生成 vLLM / SGLang 启动命令。

[English README](README.md)

## 能做什么

- 从 HuggingFace 或 ModelScope 读取模型元数据。
- 识别 MLA、GQA/MQA/MHA、NSA、CSA+HCA、MoE、滑动窗口、最大上下文等架构特征。
- 读取真实 `safetensors` 文件大小，不只依赖 `参数量 x 精度` 估算。
- 将观测字节数和已知量化方案对账，给出最接近的量化判断。
- 按 TP 和多机 PP 候选规划 `min` / `dev` / `prod` 三档 GPU 数量。
- 生成 vLLM 或 SGLang 启动命令。
- 给每个面向用户的数字标注来源：已验证、推断、估算、引用、未经验证或未知。

它是部署前的选型助手，不替代真实硬件和真实 serving 栈上的压测。

## 编译

要求：

- Rust 1.80 或更新版本
- 评估远程模型时需要网络

```bash
cargo build --release
./target/release/llm-infer-cal --help
```

安装到本机 Cargo bin 目录：

```bash
cargo install --path crates/llm-infer-cal --locked
llm-infer-cal --help
```

如果安装后找不到命令：

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## 快速示例

HuggingFace：

```bash
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800 --lang zh
```

ModelScope：

```bash
llm-infer-cal ZhipuAI/GLM-5.2 --gpu H100 --source modelscope --lang zh
```

生成 SGLang 启动命令：

```bash
llm-infer-cal Qwen/Qwen2.5-72B-Instruct --gpu H100 --engine sglang --lang zh
```

查看完整推导链：

```bash
llm-infer-cal mistralai/Mixtral-8x7B-v0.1 --gpu H100 --explain --lang zh
```

列出支持的 GPU：

```bash
llm-infer-cal --list-gpus --lang zh
```

## 常用参数

```text
llm-infer-cal [MODEL_ID] [OPTIONS]

核心：
  --gpu TEXT
  --source [huggingface|modelscope]
  --engine [vllm|sglang]
  --gpu-count INT
  --context-length INT
  --timeout-s FLOAT
  --lang [en|zh]

性能：
  --input-tokens INT
  --output-tokens INT
  --target-tokens-per-sec FLOAT
  --prefill-util FLOAT
  --decode-bw-util FLOAT
  --concurrency-degradation FLOAT

检查：
  --explain
  --llm-review
  --list-gpus
  --benchmark
  --refresh
```

## 数据和目录

```text
crates/llm-infer-cal/        命令行入口
crates/llm-infer-cal-core/   计算库
data/hardware/               GPU 数据库
data/engine_compat/          vLLM / SGLang 兼容矩阵
data/benchmark/              回归 benchmark 数据集
docs/                        方法说明和使用文档
```

## 验证

```bash
cargo fmt --all --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release
./target/release/llm-infer-cal --benchmark
```

## License

Apache-2.0，见 [LICENSE](LICENSE)。
