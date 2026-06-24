# llm-infer-cal

大模型推理硬件计算器。

`llm-infer-cal` 读取模型真实元数据，回答部署前最关心的几个问题：权重占多少
显存、KV Cache 增长多快、需要几张卡以及用什么 TP/PP 布局、吞吐大致上限，以及
可以直接拿来用的 vLLM / SGLang 启动命令。

每个面向用户的数字都带来源标签——`verified`（已验证）、`inferred`（推断）、
`estimated`（估算）、`cited`（引用）、`unknown`（未知）——一眼就能区分实测字节
数和粗略估算。

它是选型助手，不替代在真实硬件和真实 serving 栈上的压测。

[English README](README.md)

## 能做什么

- 从 HuggingFace、ModelScope 或内置目录读取模型元数据。
- 识别架构特征：MLA、GQA/MQA/MHA、NSA、CSA+HCA、MoE、滑动窗口、RoPE 缩放、
  最大上下文。
- 累加真实 `safetensors` 文件大小，不只依赖 `参数量 × 精度`。
- 将观测字节数与已知量化方案对账（FP16/BF16、FP8、INT8、FP4/FP8 混合、INT4、
  GPTQ/AWQ INT4）。
- 按单机 TP、跨机 TP、流水并行候选规划 `min` / `dev` / `prod` 三档 GPU 数量。
- 估算 Prefill 延迟、Decode tokens/秒，以及带明确瓶颈（显存容量 vs 带宽/算力）
  的并发上限。
- 生成可直接运行的 vLLM 或 SGLang 命令。
- 提供 CLI、HTTP API 和 React Web 界面。

## 核心算法

工具把每个估算拆成两半：**容量**（能不能塞进显存）和**性能**（跑多快）。容量
数字全部由公开数据推导，没有经验系数；性能数字依赖可被你覆盖的经验利用率因子。

### 1. 架构识别

`detect()` 解析 `config.json` 并对模型分类：

- **家族**：transformer 或 state-space（Mamba/Jamba 在 v0.1 标记为不支持——它们
  没有 KV Cache 概念）。
- **注意力变体**：MHA / GQA / MQA / MLA / NSA / CSA+HCA，依据
  `num_attention_heads`、`num_key_value_heads`、`kv_lora_rank`、`compress_ratios`、
  `nsa_topk` 推断。
- **MoE 特征**：路由/共享专家数、每 token 激活专家数、专家中间维度。
- **位置编码**：RoPE 类型/theta/缩放，以及 `max_position_embeddings`。

变体同时决定 KV Cache 公式和有效 TP 布局，所以这一步卡住下游所有计算。

### 2. 权重字节与量化对账

权重显存是**真实 `safetensors` 文件大小之和**——也就是你实际下载的字节——标为
`verified`。工具独立地从形状估算总参数量（`embed + head + 层数 × (attn + ffn +
norm)`，MoE 的 FFN 计入全部专家），再算出观测的每参数比特：

```
bits_per_param = observed_bytes × 8 / total_params
```

然后与已知方案对账（FP16=16b、FP8=8b、INT4=4b、GPTQ/AWQ ≈4.4b……）。当
`config.json` 声明了量化方案，**且**预测字节落在观测值的 15% 以内时，方案按
`verified` 采信；否则工具采样真实分片的 tensor dtype，最后退化为对每参数比特做
容差匹配。

### 3. 每请求 KV Cache

每 token、每层 KV 字节取决于注意力变体：

```
标准:  per_layer_per_token = 2 × num_kv_heads × head_dim × dtype_bytes
MLA:   per_layer_per_token = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
```

基线是 `per_layer_per_token × effective_seq_len × num_layers`，再叠加：

- **滑动窗口**：对非稀疏变体截断 `effective_seq_len`。
- **CSA+HCA**：按平均压缩比例 `avg(1 / compress_ratio)` 缩放。
- **NSA**：按稀疏度 `min(nsa_topk / effective_seq_len, 1)` 缩放。
- **Paged attention**（可选）：乘 `0.75`。

KV 字节会在多个上下文长度（4K / 32K / 128K / 模型最大）上分别报告，因为随并发
扩张的是 KV，而不是 activation。

### 4. 集群规划（TP / PP）

这是工具的核心。只考虑单机的规划器会对超大模型谎报"8 卡能放下"；这个规划器把
放不下的事实暴露出来，再搜索更大的有效布局。

**有效 TP** 必须整除 `num_attention_heads`。单机 TP 上限 8；头数更多时解锁跨机
TP，上限 64；流水并行（最多 8 段）切分层栈。候选 GPU 数是这些组合的全集。

**每卡常驻权重**按布局切分。稠密模型在 `PP` 切层之后再除以 `TP`。MoE 模型把
路由专家权重按 `min(TP, num_routed_experts)` 切分、静态/共享权重按 `TP` 切分，
避免路由专家被过度切分。

**每卡 KV** 除以 `effective_kv_shards = PP × kv_shards(TP)`，其中 MLA 复制
（`kv_shards = 1`），GQA/MHA 最多切 `min(TP, num_kv_heads)` 份。

**放得下判定**取 Decode 与 Prefill 显存的峰值：

```
reserved_per_gpu  = max(3 GB, 10% × HBM)        # 约等于 --gpu-memory-utilization 0.9
usable_per_gpu    = HBM − reserved_per_gpu
concurrent_KV     = 并发请求数 × per_gpu_KV
decode_required   = weight_per_gpu + decode_activation_per_gpu + concurrent_KV
prefill_required  = weight_per_gpu + prefill_peak_activation_per_gpu + concurrent_KV
required_per_gpu  = max(decode_required, prefill_required) ≤ usable_per_gpu
```

`min`/`dev`/`prod` 三档分别按 1 / 8 / 16 并发规划；`--target-concurrency` 则按
你给的并发数规划单档。每个方案还通过对放得下判定做二分搜索，报告能承载的最大
并发请求数。

### 5. 性能与并发上限

**Prefill** 受算力约束：

```
FLOPs       = 2 × active_params × input_tokens          # Kaplan et al. 2020
latency_ms  = FLOPs / (peak_TFLOPS × num_gpus × prefill_util) × 1000
```

**Decode** 受显存带宽约束（每生成一个 token 都要读一遍常驻权重）：

```
per_gpu_tok/s     = mem_bandwidth × bw_util / weight_bytes_per_gpu
cluster_tok/s     = per_gpu_tok/s × num_gpus × comm_eff × nvlink_penalty
```

对 MoE，工具同时报告保守的"全部权重"数值和标记为 `optimistic` 的"仅激活专家"
数值。

**并发上限**取两个上界的较小值：

```
K（容量）  = per_gpu_headroom / 每请求 per_gpu_KV
L（SLA）   = cluster_tok/s / 每用户目标 tok/s / 降级因子
max_concurrent = min(K, L)
```

`K ≤ L` → 显存容量瓶颈；`L < K` → 带宽/算力瓶颈。

### 经验因子（全部可覆盖）

| 因子 | 默认值 | CLI 参数 |
|---|---|---|
| Prefill 算力利用率 | 0.40 | `--prefill-util` |
| Decode 带宽利用率 | 0.50 | `--decode-bw-util` |
| 集群通信效率（TP） | 0.90 | — |
| 并发降级因子 | 1.00 | `--concurrency-degradation` |

完整推导链、每个系数及其出处见 [`docs/methodology.md`](docs/methodology.md)。
任意命令加 `--explain` 即可打印该模型的具体推导过程。

## 安装

要求：Rust 1.80 或更新版本；评估远程模型时需要网络。

```bash
# 从源码编译
cargo build --release
./target/release/llm-infer-cal --help

# 或安装到本机 Cargo bin 目录
cargo install --path crates/llm-infer-cal --locked
llm-infer-cal --help
```

如果安装后找不到命令，把 Cargo bin 目录加入 `PATH`：

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## 使用

基本选型——模型 id 加一张 GPU：

```bash
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800
```

从 ModelScope 而非 HuggingFace 读取：

```bash
llm-infer-cal ZhipuAI/GLM-5.2 --gpu H100 --source modelscope
```

生成 SGLang 启动命令：

```bash
llm-infer-cal Qwen/Qwen2.5-72B-Instruct --gpu H100 --engine sglang
```

按具体并发目标和更长上下文规划：

```bash
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800 \
  --context-length 65536 --target-concurrency 64
```

打印完整推导链：

```bash
llm-infer-cal mistralai/Mixtral-8x7B-v0.1 --gpu H100 --explain
```

按你实测的栈调整经验因子：

```bash
llm-infer-cal Qwen/Qwen2.5-72B-Instruct --gpu H100 \
  --prefill-util 0.45 --decode-bw-util 0.55 --concurrency-degradation 1.3
```

列出支持的 GPU，或输出 JSON 便于脚本处理：

```bash
llm-infer-cal --list-gpus
llm-infer-cal deepseek-ai/DeepSeek-V3 --gpu H800 --json
```

### 常用参数

```text
llm-infer-cal [MODEL_ID] [OPTIONS]

核心：
  --gpu TEXT                       GPU id，如 H800、A100-80G（见 --list-gpus）
  --source [huggingface|modelscope|builtin]
  --engine [vllm|sglang]
  --gpu-count INT                  强制 GPU 张数（否则自动推荐）
  --context-length INT             KV 估算用的上下文长度
  --lang [en|zh]                   输出语言（默认从 LANG 自动检测）
  --format [text|json] / --json

容量：
  --kv-cache-bits INT              KV 精度（比特，默认 16）
  --paged-attention                应用 0.75 的 paged-attention KV 系数
  --target-concurrency INT         按此并发规划集群
  --speculative-draft-model TEXT   把草稿/EAGLE 模型权重计入显存
  --cpu-offload-gb FLOAT           每卡从 GPU 卸载到 CPU 的权重预算

性能：
  --input-tokens INT               Prefill token 预算（默认 2000）
  --output-tokens INT              Decode token 预算（默认 512）
  --target-tokens-per-sec FLOAT    每用户 Decode SLA，驱动 L 上界
  --prefill-util FLOAT             Prefill 算力利用率（默认 0.40）
  --decode-bw-util FLOAT           Decode 带宽利用率（默认 0.50）
  --concurrency-degradation FLOAT  高并发吞吐降级惩罚

检查：
  --explain                        打印完整推导链
  --llm-review                     实验性：交给 LLM 出第二意见
  --list-gpus
  --benchmark                      运行精选回归数据集
  --refresh                        绕过缓存重新拉取元数据
```

### Web 界面与 HTTP API

React 前端（`web/frontend`）通过 `llm-infer-cal-web` HTTP 服务交互。

```bash
# 后端：在 127.0.0.1:8080 提供 API（可用 LLM_INFER_CAL_WEB_ADDR 覆盖）
cargo run -p llm-infer-cal-web

# 前端：开发服务器在 127.0.0.1:5173
cd web/frontend && npm install && npm run dev
```

> Web API 绑定 localhost 且没有内置鉴权。对外暴露前请放到带鉴权的反向代理后面。

API 一览：

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET`  | `/api/health` | 存活检查 |
| `GET`  | `/api/models` | 内置模型目录 |
| `GET`  | `/api/gpus` | 支持的 GPU 规格 |
| `POST` | `/api/evaluate` | 执行评估（参数与 CLI 一致；传 `gpus` 可多卡对比） |

```bash
curl -s localhost:8080/api/evaluate \
  -H 'content-type: application/json' \
  -d '{"model_id":"deepseek-ai/DeepSeek-V3","gpu":"H800","source":"builtin"}'
```

## 目录结构

```text
crates/llm-infer-cal/        命令行入口
crates/llm-infer-cal-core/   计算库（核心算法）
crates/llm-infer-cal-web/    Axum HTTP API
web/frontend/                React + Vite 界面
data/hardware/               GPU 数据库（47 项，覆盖多家厂商）
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

