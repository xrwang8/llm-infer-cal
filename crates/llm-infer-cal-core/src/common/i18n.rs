use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

static CURRENT_LOCALE: OnceLock<Mutex<String>> = OnceLock::new();

pub fn set_locale(locale: &str) {
    let normalized = if locale == "zh" { "zh" } else { "en" };
    let mut current = locale_cell().lock().expect("locale lock poisoned");
    *current = normalized.to_string();
}

pub fn get_locale() -> String {
    locale_cell().lock().expect("locale lock poisoned").clone()
}

pub fn detect_locale_from_env() -> &'static str {
    for var in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if std::env::var(var)
            .unwrap_or_default()
            .to_lowercase()
            .starts_with("zh")
        {
            return "zh";
        }
    }
    "en"
}

pub fn detect_locale_from_env_values<'a>(
    values: impl IntoIterator<Item = (&'a str, Option<&'a str>)>,
) -> &'static str {
    let map: HashMap<&str, Option<&str>> = values.into_iter().collect();
    for var in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if map
            .get(var)
            .and_then(|value| *value)
            .unwrap_or("")
            .to_lowercase()
            .starts_with("zh")
        {
            return "zh";
        }
    }
    "en"
}

pub fn t(key: &str) -> String {
    t_with(key, &HashMap::new())
}

pub fn t_with(key: &str, kwargs: &HashMap<&str, String>) -> String {
    let Some((en, zh)) = message(key) else {
        return key.to_string();
    };
    let template = if get_locale() == "zh" { zh } else { en };
    if kwargs.is_empty() {
        return template.to_string();
    }

    let mut out = template.to_string();
    for (key, value) in kwargs {
        out = out.replace(&format!("{{{key}}}"), value);
    }
    out
}

fn locale_cell() -> &'static Mutex<String> {
    CURRENT_LOCALE.get_or_init(|| Mutex::new("en".to_string()))
}

fn message(key: &str) -> Option<(&'static str, &'static str)> {
    match key {
        "cli.help" => Some((
            "LLM inference hardware calculator.",
            "大模型推理硬件计算器。",
        )),
        "cli.arg.model_id" => Some((
            "HuggingFace or ModelScope model id",
            "HuggingFace 或 ModelScope 的 model id",
        )),
        "cli.opt.gpu" => Some((
            "GPU type, e.g. H800, A100-80G",
            "GPU 型号，例如 H800、A100-80G",
        )),
        "cli.opt.engine" => Some(("Inference engine: vllm | sglang", "推理引擎：vllm | sglang")),
        "cli.opt.gpu_count" => Some((
            "Force GPU count (otherwise tool recommends min/dev/prod)",
            "强制指定 GPU 张数（默认由工具推荐 min/dev/prod 三档）",
        )),
        "cli.opt.context_length" => Some((
            "Context length for KV cache estimation",
            "用于 KV cache 估算的上下文长度",
        )),
        "cli.opt.refresh" => Some(("Bypass cache and re-fetch", "绕过缓存重新拉取")),
        "cli.opt.lang" => Some(("Output language: en | zh", "输出语言：en | zh")),
        "panel.via" => Some(("via", "来源")),
        "section.architecture" => Some(("Architecture", "架构")),
        "section.weights" => Some(("Weights", "权重")),
        "section.kv_cache" => Some((
            "KV cache per request (BF16/FP16)",
            "单请求 KV Cache（BF16/FP16）",
        )),
        "section.reconciliation" => Some((
            "Quantization reconciliation (observed vs predicted per scheme)",
            "量化方案对账（观测值 vs 各方案预测值）",
        )),
        "section.engine_compat" => Some(("Engine compatibility", "推理引擎兼容性")),
        "section.hardware" => Some(("Target hardware", "目标硬件")),
        "section.labels" => Some(("labels:", "标签：")),
        "section.fleet" => Some(("Recommended fleet", "推荐 GPU 张数")),
        "section.command" => Some(("Generated command", "生成的启动命令")),
        "section.performance" => Some(("Performance analysis", "性能分析")),
        "section.explain" => Some((
            "Full derivation traces (--explain)",
            "完整推导链（--explain）",
        )),
        "section.llm_review" => Some((
            "LLM second opinion (--llm-review, EXPERIMENTAL)",
            "LLM 审阅（--llm-review，实验性）",
        )),
        "weights.safetensors_bytes" => Some(("safetensors bytes", "safetensors 总字节")),
        "weights.params_estimated" => Some(("estimated total params", "参数量（估算）")),
        "weights.bits_per_param" => Some(("bits/param", "每参数位数")),
        "weights.quant_guess" => Some(("quantization guess", "量化方案推断")),
        "arch.model_type" => Some(("model_type", "模型类型")),
        "arch.family" => Some(("family", "架构族")),
        "arch.confidence" => Some(("confidence", "识别置信度")),
        "arch.layers" => Some(("layers", "层数")),
        "arch.hidden_size" => Some(("hidden_size", "隐藏维度")),
        "arch.vocab_size" => Some(("vocab_size", "词表大小")),
        "arch.attention" => Some(("attention", "注意力机制")),
        "arch.compress_ratios" => Some(("compress_ratios", "压缩比数组")),
        "arch.moe" => Some(("moe", "MoE")),
        "arch.sliding_window" => Some(("sliding_window", "滑动窗口")),
        "arch.max_position" => Some(("max_position_embeddings", "最大上下文长度")),
        "arch.none" => Some(("(none)", "（无）")),
        "arch.attn_summary" => Some((
            "{variant} (heads={heads}, kv_heads={kv_heads}, head_dim={head_dim})",
            "{variant}（heads={heads}，kv_heads={kv_heads}，head_dim={head_dim}）",
        )),
        "arch.compress_ratios_summary" => {
            Some(("len={n}, dense_layers={dense}", "长度={n}，dense 层数={dense}"))
        }
        "arch.moe_summary" => Some((
            "{routed} routed + {shared} shared, top-{topk}",
            "{routed} 个 routed + {shared} 个 shared，top-{topk}",
        )),
        "arch.unsupported_state_space" => Some((
            "State-space models are not supported in v0.1 (planned for v0.3+).",
            "状态空间模型（Mamba 类）在 v0.1 暂不支持，计划在 v0.3+ 加入。",
        )),
        "recon.scheme" => Some(("scheme", "量化方案")),
        "recon.predicted" => Some(("predicted bytes", "预测字节")),
        "recon.delta" => Some(("delta", "差值")),
        "recon.error_pct" => Some(("error %", "误差 %")),
        "recon.over" => Some(("over", "偏高")),
        "recon.under" => Some(("under", "偏低")),
        "recon.best" => Some(("best match:", "最佳匹配：")),
        "kv.context" => Some(("context", "上下文")),
        "kv.kv_cache" => Some(("KV cache", "KV Cache")),
        "kv.label" => Some(("label", "标签")),
        "kv.tokens" => Some(("tokens", "tokens")),
        "engine.version_spec" => Some(("version", "版本要求")),
        "engine.support" => Some(("support", "支持程度")),
        "engine.verification" => Some(("verification", "验证等级")),
        "engine.required_flags" => Some(("required flags", "必需参数")),
        "engine.optional_flags" => Some(("optional flags", "可选参数")),
        "engine.caveats" => Some(("caveats", "注意事项")),
        "engine.sources" => Some(("sources", "来源")),
        "engine.no_match" => Some((
            "No compatibility entry for this model + engine in v0.1 matrix.",
            "v0.1 兼容矩阵中暂无此模型 + 引擎的条目。",
        )),
        "hw.memory" => Some(("memory", "显存")),
        "hw.nvlink_bandwidth" => Some(("NVLink bandwidth", "NVLink 带宽")),
        "hw.fp16_tflops" => Some(("FP16 TFLOPS", "FP16 算力")),
        "hw.fp8_support" => Some(("FP8 support", "FP8 支持")),
        "hw.fp4_support" => Some(("FP4 support", "FP4 支持")),
        "hw.notes" => Some(("notes", "备注")),
        "hw.spec_source" => Some(("spec source", "规格来源")),
        "hw.bool_yes" => Some(("yes", "是")),
        "hw.bool_no" => Some(("no", "否")),
        "gpus.list.title" => Some(("Supported GPUs", "支持的 GPU")),
        "gpus.col.id" => Some(("id", "型号")),
        "gpus.col.memory" => Some(("memory", "显存")),
        "gpus.col.nvlink" => Some(("NVLink / fabric", "互联带宽")),
        "gpus.col.fp16" => Some(("FP16 TFLOPS", "FP16")),
        "gpus.col.fp8" => Some(("FP8", "FP8")),
        "gpus.col.fp4" => Some(("FP4", "FP4")),
        "gpus.col.aliases" => Some(("aliases", "别名")),
        "gpus.total" => Some((
            "Total: {count} GPUs (pass any id or alias to --gpu)",
            "共 {count} 款（--gpu 后面填 ID 或别名均可）",
        )),
        "hw.unknown" => Some((
            "Unknown GPU '{gpu}'. Known: {known}",
            "未知 GPU '{gpu}'。已知型号：{known}",
        )),
        "cli.err.auth_required" => Some(("Authentication required:", "需要认证：")),
        "cli.err.model_not_found" => Some(("Model not found:", "模型未找到：")),
        "cli.err.source_unavailable" => Some(("Source unavailable:", "数据源不可用：")),
        "cli.err.missing_model" => Some((
            "Missing argument MODEL_ID. Use --help for usage.",
            "缺少参数 MODEL_ID。使用 --help 查看用法。",
        )),
        "cli.err.missing_gpu" => Some((
            "Missing option --gpu. Use --list-gpus to see choices.",
            "缺少选项 --gpu。使用 --list-gpus 查看可选 GPU。",
        )),
        "cli.err.unknown_source" => Some((
            "Unknown --source '{source}'. Use 'builtin', 'huggingface', or 'modelscope'.",
            "未知 --source '{source}'。请使用 'builtin'、'huggingface' 或 'modelscope'。",
        )),
        "source.pr" => Some(("PR", "PR")),
        "source.release_notes" => Some(("release notes", "release note")),
        "source.announcement" => Some(("announcement", "官方公告")),
        "source.tested" => Some(("tested", "实测")),
        "source.captured_on" => Some(("captured on", "采集于")),
        "fleet.col.tier" => Some(("tier", "档位")),
        "fleet.col.gpus" => Some(("GPUs", "GPU 数")),
        "fleet.col.weight_per_gpu" => Some(("weight / GPU", "单卡权重")),
        "fleet.col.headroom_per_gpu" => Some(("headroom / GPU", "单卡余量")),
        "fleet.col.fit" => Some(("fit", "评估")),
        "fleet.col.concurrent_at_ctx" => Some(("concurrent @ {ctx}", "并发 @ {ctx}")),
        "fleet.tier.min" => Some(("min", "最小")),
        "fleet.tier.dev" => Some(("dev", "开发")),
        "fleet.tier.prod" => Some(("prod", "生产")),
        "fleet.best_marker" => Some(("= recommended", "= 推荐档位")),
        "fleet.no_recommended_tier" => Some((
            "No recommended tier: the model does not fit on the available TP/PP candidates.",
            "没有推荐档位：模型无法放入当前 TP/PP 候选配置。",
        )),
        "fleet.constraint" => Some(("constraint:", "约束：")),
        "fleet.forced" => Some((
            "Forced GPU count (--gpu-count was set)",
            "已强制指定 GPU 张数（--gpu-count）",
        )),
        "fleet.gpu_spec_unknown" => Some((
            "Fleet planning skipped - GPU spec unknown.",
            "GPU 规格未知，跳过 fleet 规划。",
        )),
        "command.tier_note" => Some((
            "tier: {tier} ({gpus} GPUs)",
            "档位：{tier}（{gpus} 张）",
        )),
        "perf.assumptions_note" => Some((
            "Assumes input={input_tokens} tokens, output={output_tokens} tokens, target {target_tps} tok/s per user. Utilization: prefill={prefill_util} / decode_bw={decode_util} / concurrency_degradation={degradation}x. All numbers are [estimated] - see docs/methodology.md for formula sources and override via --prefill-util / --decode-bw-util / --concurrency-degradation.",
            "假设输入 {input_tokens} tokens、输出 {output_tokens} tokens、每用户目标 {target_tps} tok/s。利用率：prefill={prefill_util} / decode_bw={decode_util} / 并发退化={degradation}x。所有数字都是 [估算] - 公式来源见 docs/methodology.md，可通过 --prefill-util / --decode-bw-util / --concurrency-degradation 覆盖。",
        )),
        "perf.prefill_latency" => Some((
            "Prefill latency (single request)",
            "Prefill 延迟（单请求）",
        )),
        "perf.decode_throughput_cluster" => {
            Some(("Decode throughput (cluster)", "Decode 吞吐（集群）"))
        }
        "perf.decode_throughput_per_gpu" => {
            Some(("Decode throughput (per GPU)", "Decode 吞吐（单卡）"))
        }
        "perf.decode_moe_active_optimistic" => Some((
            "Decode throughput (MoE active-only, optimistic)",
            "Decode 吞吐（MoE 仅激活专家，乐观估算）",
        )),
        "perf.k_bound" => Some(("K bound (memory-capacity)", "K 上限（显存容量）")),
        "perf.l_bound" => Some((
            "L bound (compute / bandwidth @ SLA)",
            "L 上限（算力/带宽 @ SLA）",
        )),
        "perf.max_concurrent" => Some(("Max concurrent", "最大并发")),
        "perf.bottleneck" => Some(("Bottleneck", "瓶颈类型")),
        "perf.bottleneck.memory_capacity" => Some(("Memory capacity", "显存容量")),
        "perf.bottleneck.memory_bandwidth" => {
            Some(("Memory bandwidth / compute", "显存带宽 / 算力"))
        }
        "perf.bottleneck.compute" => Some(("Compute", "算力")),
        "perf.bottleneck.insufficient_data" => Some(("Insufficient data", "数据不足")),
        "perf.optimization.header" => Some(("Optimization suggestions", "优化建议")),
        "perf.opt.quantize_int4" => Some((
            "Quantize to INT4: weight bytes halve -> decode tok/s roughly 2x -> concurrency scales accordingly.",
            "量化到 INT4：权重字节减半 → decode tok/s 约翻倍 → 并发能力随之提升。",
        )),
        "perf.opt.relax_sla" => Some((
            "Relax SLA: if per-user target drops to 15 tok/s, L bound roughly doubles.",
            "放宽 SLA：若每用户目标降至 15 tok/s，L 上限约翻倍。",
        )),
        "perf.opt.kv_fp8" => Some((
            "KV cache FP8 quantization: halves per-request KV, doubles the K bound at long context.",
            "KV cache 量化到 FP8：单请求 KV 减半，长上下文下 K 上限约翻倍。",
        )),
        "perf.opt.moe_offload" => Some((
            "MoE expert offload to CPU: frees HBM for more KV cache at the cost of PCIe latency per new expert.",
            "MoE 专家卸载到 CPU：释放 HBM 给 KV cache，代价是新专家激活时的 PCIe 延迟。",
        )),
        "explain.formula" => Some(("Formula", "公式")),
        "explain.inputs" => Some(("Inputs", "输入")),
        "explain.steps" => Some(("Computation", "计算步骤")),
        "explain.result" => Some(("Result", "结果")),
        "explain.source" => Some(("Source", "来源")),
        "explain.see_also" => Some(("See also", "延伸阅读")),
        "explain.intro" => Some((
            "Each entry below shows the formula used, the inputs that went in, every computation step, and the primary source.",
            "下面每一项都给出所用公式、输入、每一步计算、主要来源。",
        )),
        "llm_review.unavailable" => Some((
            "LLM review unavailable: {error}",
            "LLM 审阅不可用：{error}",
        )),
        "llm_review.setup_hint" => Some((
            "To enable: export LLM_CAL_REVIEWER_API_KEY=<key>  [optional: LLM_CAL_REVIEWER_BASE_URL, LLM_CAL_REVIEWER_MODEL]",
            "启用方法：export LLM_CAL_REVIEWER_API_KEY=<key>  [可选：LLM_CAL_REVIEWER_BASE_URL、LLM_CAL_REVIEWER_MODEL]",
        )),
        "llm_review.disclaimer" => Some((
            "This is a second opinion from an external LLM ({model} via {base_url}). It is tagged [llm-opinion] and NEVER overrides the 6 primary labels.",
            "以下是来自外部 LLM（{model}，经 {base_url}）的第二意见。标签为 [LLM 观点]，永远不覆盖前 6 级主标签。",
        )),
        "label.verified" => Some(("verified", "已验证")),
        "label.inferred" => Some(("inferred", "推断")),
        "label.estimated" => Some(("estimated", "估算")),
        "label.cited" => Some(("cited", "引用")),
        "label.unverified" => Some(("unverified", "未经验证")),
        "label.unknown" => Some(("unknown", "未知")),
        "label.llm-opinion" => Some(("llm-opinion", "LLM 观点")),
        _ => None,
    }
}
