# CLI 参数对齐总结

## 完成的改动

### 1. ✅ 参数名统一（与前端对齐）

| 旧 CLI 参数 | 新 CLI 参数 | 前端参数 |
|-------------|-------------|----------|
| `--prefill-util` | `--prefill-utilization` | `prefill_utilization` |
| `--decode-bw-util` | `--decode-bw-utilization` | `decode_bw_utilization` |
| `--target-concurrency` | `--target-concurrent-requests` | `target_concurrent_requests` |
| `--speculative-draft-model` | `--speculative-draft-model-id` | `speculative_draft_model_id` |

### 2. ✅ 新增 Speculative Decoding 参数

- `--speculative-enabled` — 开关，显式启用 speculative decoding
- `--speculative-mode <MODE>` — 模式选择，**默认且仅支持 `mtp`**
- `--speculative-num-draft-tokens <N>` — draft token 数量，默认 8

**注意**：评估时固定使用 **MTP 模式**，这是当前唯一支持的模式。

### 3. ✅ 新增 MoE Expert Offloading 参数

- `--expert-offloading` — 开关，启用 MoE 专家卸载
- `--experts-on-gpu <N>` — GPU 上保留的专家数量

仅对 MoE 模型（如 Qwen3-30B-A3B）有效。

### 4. ⏭️ 跳过：多 GPU 对比

前端支持 `gpus: ['H100', 'A100-80G']` 多选对比，CLI 保持单个 `--gpu` 输入。
多 GPU 对比是前端专属功能，CLI 用户可以通过脚本多次调用实现。

## 示例用法

### 基础用法（使用新参数名）
```bash
llm-infer-cal Qwen/Qwen3-30B-A3B --source builtin --gpu H100 \
  --prefill-utilization 0.5 \
  --decode-bw-utilization 0.6 \
  --target-concurrent-requests 10
```

### Speculative Decoding (MTP 模式)
```bash
llm-infer-cal Qwen/Qwen3-30B-A3B --source builtin --gpu H100 \
  --speculative-enabled \
  --speculative-num-draft-tokens 4
```

### MoE Expert Offloading
```bash
llm-infer-cal Qwen/Qwen3-30B-A3B --source builtin --gpu A100-80G \
  --expert-offloading \
  --experts-on-gpu 16
```

### 组合使用
```bash
llm-infer-cal Qwen/Qwen3-30B-A3B --source builtin --gpu H100 \
  --prefill-utilization 0.5 \
  --decode-bw-utilization 0.6 \
  --target-concurrent-requests 10 \
  --speculative-enabled \
  --speculative-num-draft-tokens 4 \
  --expert-offloading \
  --experts-on-gpu 16
```

## 兼容性说明

- **旧参数已移除**：`--prefill-util`、`--decode-bw-util`、`--target-concurrency`、`--speculative-draft-model` 不再支持
- **Speculative mode**：当前仅实现 `mtp`，`standard` 模式保留但未启用
- **MTP 默认**：所有 speculative 评估默认使用 MTP 模式，这是当前推荐且唯一实现的方式

## 与前端 API 的完全对齐

CLI 现在与 Web 前端的参数完全一致（除了多 GPU 对比），可以无缝对接同一套后端评估逻辑。
