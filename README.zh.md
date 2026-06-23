# llm-infer-cal

大模型推理硬件计算器。

`llm-infer-cal` 是一个命令行工具，用来估算一个模型在指定 GPU 上能不能跑、需要多少张卡、KV 缓存压力大不大，以及 vLLM / SGLang 启动命令可以怎么写。

这个项目以 Rust CLI 为主。仓库里的 Python 模块保留为兼容层、参考实现和跨语言回归测试，用来确保 Rust 迁移过程中输出不偏。

[English README](README.md)

## 它解决什么问题

- 从 HuggingFace 或 ModelScope 读取模型元数据。
- 识别模型架构、注意力结构、MoE 特征、滑动窗口和 KV 缓存形状。
- 估算权重显存、KV 缓存显存、推荐 GPU 数量、并发上界，以及粗略的 prefill / decode 表现。
- 匹配 vLLM 和 SGLang 的推理引擎兼容规则。
- 生成一条可修改、可落地的启动命令。
- 给输出值标注来源，例如真实读取、推导、估算、引用和未知。

它是选型和部署前的估算工具，不替代真实硬件上的压测。

## 编译命令行

要求：

- Rust 1.80 或更新版本
- 评估远程模型时需要网络

编译 release 二进制：

```bash
cd /Users/xrwang/go/src/github.com/xrwang8/llm-infer-cal
cargo build --release
```

直接运行：

```bash
./target/release/llm-infer-cal --help
./target/release/llm-infer-cal --list-gpus --lang zh
```

安装成系统命令：

```bash
cargo install --path crates/llm-infer-cal --locked
llm-infer-cal --help
```

如果安装后提示找不到命令，把 Cargo 的 bin 目录加入 PATH：

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## 快速使用

评估 HuggingFace 模型：

```bash
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800 --lang zh
```

走 ModelScope：

```bash
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800 --source modelscope --lang zh
```

估算更长上下文：

```bash
llm-infer-cal Qwen/Qwen2.5-72B-Instruct \
  --gpu A100-80G \
  --context-length 32768 \
  --lang zh
```

查看完整推导链：

```bash
llm-infer-cal mistralai/Mixtral-8x7B-v0.1 --gpu H100 --explain --lang zh
```

列出支持的 GPU：

```bash
llm-infer-cal --list-gpus --lang zh
```

## 中文输出

需要中文输出时传 `--lang zh`：

```bash
llm-infer-cal Qwen/Qwen2.5-7B --gpu RTX4090 --lang zh
```

需要英文输出时传 `--lang en`。不传 `--lang` 时，工具会尝试从 `LANG` 环境变量自动判断。

## 数据源

默认数据源是 HuggingFace：

```bash
llm-infer-cal Qwen/Qwen2.5-7B --gpu RTX4090
```

使用 ModelScope：

```bash
llm-infer-cal Qwen/Qwen2.5-7B --gpu RTX4090 --source modelscope
```

鉴权环境变量：

```bash
export HF_TOKEN=...
export MODELSCOPE_API_TOKEN=...
```

也支持 `HUGGING_FACE_HUB_TOKEN` 和 `MODELSCOPE_TOKEN`。

需要绕过缓存重新拉取时：

```bash
llm-infer-cal Qwen/Qwen2.5-7B --gpu RTX4090 --refresh --lang zh
```

## 常用参数

```text
llm-infer-cal [MODEL_ID] [OPTIONS]

核心参数：
  --gpu TEXT                     目标 GPU，例如 H800 或 A100-80G
  --source [huggingface|modelscope]
  --engine [vllm|sglang]
  --gpu-count INT                强制指定 GPU 数量，不走自动规划
  --context-length INT           KV 缓存估算使用的上下文长度
  --timeout-s FLOAT              模型元数据请求的网络超时时间，单位秒
  --lang [en|zh]

性能输入：
  --input-tokens INT
  --output-tokens INT
  --target-tokens-per-sec FLOAT
  --prefill-util FLOAT
  --decode-bw-util FLOAT
  --concurrency-degradation FLOAT

辅助命令：
  --explain
  --llm-review
  --list-gpus
  --benchmark
  --refresh
```

## LLM 审计

`--llm-review` 是可选能力。它会把推导链发送到 OpenAI 兼容 API，让模型给出第二意见。这个结果只作为参考，不覆盖计算器的确定性输出。

环境变量：

```bash
export LLM_CAL_REVIEWER_API_KEY=...
export LLM_CAL_REVIEWER_BASE_URL=https://api.openai.com/v1
export LLM_CAL_REVIEWER_MODEL=gpt-4o
```

## Benchmark 命令

`llm-infer-cal --benchmark` 是项目自己的回归检查。它会跑一组固定模型，确认权重估算、量化识别和部分规划结果没有因为改代码而偏掉。

它用于开发和 CI，不是公开排行榜。

```bash
llm-infer-cal --benchmark
```

退出码：

- `0`：全部通过
- `1`：至少一项失败

## 项目结构

```text
crates/llm-infer-cal/        Rust 命令行入口
crates/llm-infer-cal-core/   Rust 计算库
src/llm_cal/                 Python 兼容层和参考模块
tests/                       Python 对齐测试和回归测试
docs/                        方法说明和参考页面
```

## 开发检查

Rust：

```bash
cargo fmt --all --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

Python 对齐测试：

```bash
PYTHONPATH=src python -m ruff check src tests
PYTHONPATH=src python -m pytest -q
```

ModelScope live 对齐测试：

```bash
LLM_CAL_LIVE_MODEL_PARITY=1 PYTHONPATH=src python -m pytest tests/test_live_modelscope_value_parity.py -q
```

## License

Apache-2.0，见 [LICENSE](LICENSE)。
