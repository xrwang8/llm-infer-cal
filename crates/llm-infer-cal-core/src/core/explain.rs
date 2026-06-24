use crate::architecture::profile::{ArchitectureProfile, AttentionVariant};
use crate::core::evaluator::EvaluationReport;
use crate::fleet::planner::effective_kv_shards;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExplainInput {
    pub name: String,
    pub value: String,
    pub label: String,
    pub note: String,
}

impl ExplainInput {
    pub fn new(name: &str, value: impl Into<String>, label: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.into(),
            label: label.to_string(),
            note: String::new(),
        }
    }

    pub fn with_note(name: &str, value: impl Into<String>, label: &str, note: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.into(),
            label: label.to_string(),
            note: note.to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExplainEntry {
    pub heading: String,
    pub formula: String,
    pub inputs: Vec<ExplainInput>,
    pub steps: Vec<String>,
    pub result: String,
    pub source: String,
    pub methodology_anchor: String,
}

impl ExplainEntry {
    fn new(heading: &str, formula: &str) -> Self {
        Self {
            heading: heading.to_string(),
            formula: formula.to_string(),
            inputs: Vec::new(),
            steps: Vec::new(),
            result: String::new(),
            source: String::new(),
            methodology_anchor: String::new(),
        }
    }
}

pub fn build(report: &EvaluationReport) -> Vec<ExplainEntry> {
    let mut entries = Vec::new();

    weight_bytes(report, &mut entries);
    quantization(report, &mut entries);
    kv_cache_contexts(report, &mut entries);
    fleet_tiers(report, &mut entries);
    prefill(report, &mut entries);
    decode(report, &mut entries);
    concurrency(report, &mut entries);

    entries
}

fn weight_bytes(report: &EvaluationReport, entries: &mut Vec<ExplainEntry>) {
    let w = &report.weight.total_bytes;
    let mut entry = ExplainEntry::new(
        "Weight bytes (safetensors file sum)",
        "sum(file.size for file in model_source.file_metadata if file.endswith('.safetensors'))",
    );
    entry.inputs.push(ExplainInput::new(
        "model source API",
        format!(
            "source={}, sha={}",
            report.source,
            report.commit_sha.as_deref().unwrap_or("HEAD")
        ),
        "[verified]",
    ));
    entry.steps = vec![
        format!("Raw value from API = {} bytes", fmt_u64(w.value)),
        format!("= {:.2} GB", w.value as f64 / 1e9),
    ];
    entry.result = format!("{} bytes [verified]", fmt_u64(w.value));
    entry.source = w
        .source
        .clone()
        .unwrap_or_else(|| "HF siblings API".to_string());
    entry.methodology_anchor = "#weight-bytes".to_string();
    entries.push(entry);
}

fn quantization(report: &EvaluationReport, entries: &mut Vec<ExplainEntry>) {
    let r = &report.reconciliation;
    if r.candidates.is_empty() {
        return;
    }

    let best = &r.candidates[0];
    let cands_table = r
        .candidates
        .iter()
        .take(6)
        .map(|candidate| {
            format!(
                "      {:<16} predicted={:.2} GB  error={:.1}%",
                candidate.scheme,
                candidate.predicted_bytes as f64 / 1e9,
                candidate.relative_error * 100.0
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut entry = ExplainEntry::new(
        "Quantization scheme (reconciliation)",
        "best_match = argmin_scheme |observed_bytes - scheme.bpp x total_params|",
    );
    entry.inputs = vec![
        ExplainInput::new("observed_bytes", fmt_u64(r.observed_bytes), "[verified]"),
        ExplainInput::with_note(
            "total_params",
            fmt_u64(r.total_params),
            "[estimated]",
            "from architecture formula - see '#params-estimate' entry below",
        ),
    ];
    entry.steps = vec![
        "For each known quantization scheme, predict total bytes = bpp x params:".to_string(),
        cands_table,
        format!(
            "Winner: {} at {:.1}% error",
            best.scheme,
            best.relative_error * 100.0
        ),
    ];
    entry.result = format!("{} [{}]", r.best.value, r.best.label);
    entry.source = "Nearest-anchor match against known bytes-per-param values".to_string();
    entry.methodology_anchor = "#quantization-scheme".to_string();
    entries.push(entry);
}

fn kv_cache_contexts(report: &EvaluationReport, entries: &mut Vec<ExplainEntry>) {
    let profile = &report.profile;
    let Some(attn) = &profile.attention else {
        return;
    };

    for (ctx, av) in &report.kv_cache_by_context {
        if av.value == 0 {
            continue;
        }

        let is_mla = attn.variant == AttentionVariant::Mla;
        let is_csa_hca = attn.variant == AttentionVariant::CsaHca;
        let (per_tok_per_layer, mut formula, mut inputs) = if is_mla
            && attn.kv_lora_rank.unwrap_or(0) > 0
        {
            let rank = attn.kv_lora_rank.unwrap_or(0);
            let rope_dim = attn.qk_rope_head_dim.unwrap_or(0);
            (
                (rank + rope_dim) * 2,
                "per_tok_per_layer = (kv_lora_rank + qk_rope_head_dim) x dtype_bytes   (MLA: latent KV plus decoupled RoPE key)"
                    .to_string(),
                vec![
                    ExplainInput::new("kv_lora_rank", rank.to_string(), "[verified]"),
                    ExplainInput::new("qk_rope_head_dim", rope_dim.to_string(), "[verified]"),
                    ExplainInput::with_note("dtype_bytes", "2", "[verified]", "BF16/FP16"),
                    ExplainInput::new("seq_len", fmt_u64(*ctx), "[verified]"),
                    ExplainInput::new(
                        "num_layers",
                        profile.num_hidden_layers.to_string(),
                        "[verified]",
                    ),
                ],
            )
        } else {
            let per_tok_per_layer = 2 * attn.num_kv_heads * attn.head_dim * 2;
            (
                    per_tok_per_layer,
                    "per_tok_per_layer = 2 x num_kv_heads x head_dim x dtype_bytes   (standard attention)"
                        .to_string(),
                    vec![
                        ExplainInput::new(
                            "num_kv_heads",
                            attn.num_kv_heads.to_string(),
                            "[verified]",
                        ),
                        ExplainInput::new("head_dim", attn.head_dim.to_string(), "[verified]"),
                        ExplainInput::with_note("dtype_bytes", "2", "[verified]", "BF16/FP16"),
                        ExplainInput::new("seq_len", fmt_u64(*ctx), "[verified]"),
                        ExplainInput::new(
                            "num_layers",
                            profile.num_hidden_layers.to_string(),
                            "[verified]",
                        ),
                    ],
                )
        };

        let baseline = per_tok_per_layer * ctx * profile.num_hidden_layers;
        let mut steps = vec![
            format!("per_tok_per_layer = {} bytes", fmt_u64(per_tok_per_layer)),
            format!(
                "baseline = per_tok_per_layer x seq_len x num_layers = {} bytes",
                fmt_u64(baseline)
            ),
        ];

        if is_csa_hca
            && attn
                .compress_ratios
                .as_ref()
                .is_some_and(|ratios| !ratios.is_empty())
        {
            let ratios = attn.compress_ratios.as_ref().unwrap();
            let avg = average_keep_fraction(ratios);
            inputs.push(ExplainInput::new(
                "compress_ratios",
                format!("len={} (avg keep-fraction={avg:.4})", ratios.len()),
                "[verified]",
            ));
            formula.push_str(
                "\napply_csa_hca: baseline x avg(1/r_i for r_i in compress_ratios, 0 = keep-all=1)",
            );
            steps.push(format!("avg_keep_fraction = {avg:.4}"));
            steps.push(format!(
                "result = baseline x avg_keep_fraction = {} bytes",
                fmt_u64(av.value)
            ));
        } else {
            steps.push(format!("result = baseline = {} bytes", fmt_u64(av.value)));
        }

        let mut entry =
            ExplainEntry::new(&format!("KV cache @ {} context", fmt_ctx(*ctx)), &formula);
        entry.inputs = inputs;
        entry.steps = steps;
        entry.result = format!(
            "{} bytes = {:.2} GB [{}]",
            fmt_u64(av.value),
            av.value as f64 / 1e9,
            av.label
        );
        entry.source = "DeepSeek-V2 paper (MLA); DeepSeek-V4 tech report (CSA+HCA); standard attention formula per Attention Is All You Need (Vaswani 2017)".to_string();
        entry.methodology_anchor = "#kv-cache-per-request".to_string();
        entries.push(entry);
    }
}

fn fleet_tiers(report: &EvaluationReport, entries: &mut Vec<ExplainEntry>) {
    let (Some(fleet), Some(gpu_spec)) = (&report.fleet, &report.gpu_spec) else {
        return;
    };

    for opt in &fleet.options {
        let headroom = opt
            .usable_bytes_per_gpu
            .saturating_sub(opt.weight_bytes_per_gpu)
            .saturating_sub(opt.activation_bytes_per_request_per_gpu);
        let fit_criterion = match opt.tier {
            "min" => 1,
            "dev" => 8,
            "prod" => 16,
            _ => 1,
        };
        let layout = if opt.pipeline_parallel_size > 1 {
            format!(
                "TP{} x PP{} ({} nodes)",
                opt.tensor_parallel_size, opt.pipeline_parallel_size, opt.node_count
            )
        } else {
            format!("TP{}", opt.tensor_parallel_size)
        };
        let effective_shards = effective_kv_shards(&report.profile, opt.gpu_count);
        let kv_per_gpu = ceil_div(opt.kv_bytes_per_request, effective_shards.max(1));
        let reference_ctx = fmt_context(opt.kv_reference_context_tokens);
        let mut steps = vec![
            format!(
                "per-GPU HBM usable (HBM - reserve) = {} bytes; reserve = {} bytes",
                fmt_u64(opt.usable_bytes_per_gpu),
                fmt_u64(opt.reserved_bytes_per_gpu)
            ),
            format!("parallel layout = {layout}"),
            format!(
                "resident weight per GPU = {} bytes",
                fmt_u64(opt.weight_bytes_per_gpu)
            ),
            format!(
                "activation working set per GPU = {} bytes",
                fmt_u64(opt.activation_bytes_per_request_per_gpu)
            ),
            format!(
                "total model weight bytes = {} (observed safetensors), allocated over {} GPUs",
                fmt_u64(report.weight.total_bytes.value),
                opt.gpu_count
            ),
            format!(
                "per-GPU KV @ {reference_ctx} = total_KV / effective_kv_shards = {} / {} = {} bytes",
                fmt_u64(opt.kv_bytes_per_request),
                effective_shards,
                fmt_u64(kv_per_gpu)
            ),
            format!(
                "KV headroom per GPU = usable - resident_weight - activation = {} bytes ({:.2} GB)",
                fmt_u64(headroom),
                headroom as f64 / 1e9
            ),
            format!(
                "tier criterion: headroom >= {fit_criterion} x per_gpu_kv_per_request_at_{reference_ctx}"
            ),
            format!(
                "selected smallest candidate satisfying the criterion: {} GPUs ({layout})",
                opt.gpu_count
            ),
        ];
        if !opt.fits {
            steps.push(format!(
                "NOTE: does not fit the criterion - the chosen {} is the best available.",
                opt.gpu_count
            ));
        }

        let mut entry = ExplainEntry::new(
            &format!("Fleet tier: {} ({} GPUs)", opt.tier, opt.gpu_count),
            "smallest TP/PP candidate where resident_weight_per_gpu + activation_working_set_per_gpu + concurrent x per_gpu_kv_per_request <= usable_per_gpu",
        );
        entry.inputs = vec![
            ExplainInput::new(
                "total_weight_bytes",
                fmt_u64(report.weight.total_bytes.value),
                "[verified]",
            ),
            ExplainInput::with_note(
                "valid_TP_sizes",
                format!("{:?}", fleet.valid_tp_sizes),
                "[estimated]",
                "divisors of num_attention_heads capped at 8 (single node)",
            ),
            ExplainInput::new("selected_layout", layout, "[estimated]"),
            ExplainInput::new(
                "GPU memory_gb",
                format!("{} GB", gpu_spec.memory_gb),
                "[verified]",
            ),
        ];
        entry.steps = steps;
        entry.result = format!("{} GPUs, fit={}", opt.gpu_count, opt.fits);
        entry.source =
            "vLLM GPU memory profiling reserves weights, peak activation, and KV cache inside gpu_memory_utilization; SGLang memory pool similarly budgets static weights plus KV cache"
                .to_string();
        entry.methodology_anchor = "#tppp-aware-kv-sharding".to_string();
        entries.push(entry);
    }
}

fn prefill(report: &EvaluationReport, entries: &mut Vec<ExplainEntry>) {
    let (Some(prefill), Some(gpu_spec), Some(fleet), Some(input_tokens)) = (
        &report.prefill,
        &report.gpu_spec,
        &report.fleet,
        report.perf_input_tokens,
    ) else {
        return;
    };
    let chosen = chosen_gpu_count(fleet);
    let mut entry = ExplainEntry::new(
        "Prefill latency (single request)",
        "FLOPs = 2 x params x input_tokens\neffective_TFLOPS = peak_fp16_TFLOPS x num_gpus x utilization\nlatency_ms = (FLOPs / (effective_TFLOPS x 1e12)) x 1000",
    );
    entry.inputs = vec![
        ExplainInput::with_note(
            "params",
            fmt_u64(report.total_params_estimate.value),
            "[estimated]",
            "from Rust architecture weight formula",
        ),
        ExplainInput::new("input_tokens", fmt_u64(input_tokens), "[user-set]"),
        ExplainInput::with_note(
            "peak_fp16_TFLOPS",
            gpu_spec.fp16_tflops.to_string(),
            "[verified]",
            &format!("from GPU database, {} spec", gpu_spec.id),
        ),
        ExplainInput::new("num_gpus", chosen.to_string(), "[estimated]"),
        ExplainInput::with_note(
            "utilization",
            format!("{:.2}", prefill.utilization),
            "[user-set]",
            "empirical MFU, default 0.40 - override with --prefill-util",
        ),
    ];
    entry.steps = vec![
        format!(
            "FLOPs = 2 x {} x {} = {:.3e}",
            fmt_u64(report.total_params_estimate.value),
            fmt_u64(input_tokens),
            prefill.total_flops.value as f64
        ),
        format!(
            "effective_TFLOPS = {} x {} x {:.2} = {:.1}",
            gpu_spec.fp16_tflops, chosen, prefill.utilization, prefill.peak_effective_tflops.value
        ),
        format!(
            "latency = {:.3e} / ({:.1} x 1e12) x 1000 = {:.1} ms",
            prefill.total_flops.value as f64,
            prefill.peak_effective_tflops.value,
            prefill.latency_ms.value
        ),
    ];
    entry.result = format!(
        "{:.1} ms [{}]",
        prefill.latency_ms.value, prefill.latency_ms.label
    );
    entry.source =
        "Kaplan et al. 2020 'Scaling Laws for Neural Language Models' (arxiv.org/abs/2001.08361)"
            .to_string();
    entry.methodology_anchor = "#prefill-latency".to_string();
    entries.push(entry);
}

fn decode(report: &EvaluationReport, entries: &mut Vec<ExplainEntry>) {
    let (Some(decode), Some(gpu_spec), Some(fleet)) =
        (&report.decode, &report.gpu_spec, &report.fleet)
    else {
        return;
    };
    let bw = gpu_spec.memory_bandwidth_gbps.unwrap_or(0);
    let chosen = chosen_gpu_count(fleet);
    let weight_per_gpu = decode.active_weight_bytes_per_gpu.value;
    let effective_bw_gbs = bw as f64 * decode.bw_utilization;
    let steps = vec![
        format!(
            "weight_per_gpu = {} / {} = {} bytes ({:.2} GB)",
            fmt_u64(report.weight.total_bytes.value),
            chosen,
            fmt_u64(weight_per_gpu),
            weight_per_gpu as f64 / 1e9
        ),
        format!(
            "effective_bw = {} x {:.2} = {:.0} GB/s",
            bw, decode.bw_utilization, effective_bw_gbs
        ),
        format!(
            "per_gpu_tok_per_sec = effective_bw / weight_per_gpu = {:.1} tok/s",
            effective_bw_gbs * 1e9 / weight_per_gpu as f64
        ),
        format!(
            "cluster_tok_per_sec = per_gpu x {} x {:.2} = {:.1} tok/s",
            chosen, decode.cluster_comm_efficiency, decode.cluster_tokens_per_sec.value
        ),
    ];

    let mut entry = ExplainEntry::new(
        "Decode throughput (cluster)",
        "per_gpu_tok_per_sec = memory_bandwidth x bw_util / weight_bytes_per_gpu\ncluster_tok_per_sec = per_gpu x num_gpus x cluster_comm_efficiency",
    );
    entry.inputs = vec![
        ExplainInput::with_note(
            "GPU memory_bandwidth_gbps",
            bw.to_string(),
            "[verified]",
            &format!("from GPU database, {}", gpu_spec.id),
        ),
        ExplainInput::with_note(
            "bw_util",
            format!("{:.2}", decode.bw_utilization),
            "[user-set]",
            "empirical, default 0.50 - override with --decode-bw-util",
        ),
        ExplainInput::new(
            "weight_bytes_per_gpu",
            fmt_u64(weight_per_gpu),
            "[estimated]",
        ),
        ExplainInput::new("num_gpus", chosen.to_string(), "[estimated]"),
        ExplainInput::with_note(
            "cluster_comm_efficiency",
            format!("{:.2}", decode.cluster_comm_efficiency),
            "[user-set]",
            "NCCL AllReduce efficiency on NVLink, default 0.90",
        ),
    ];
    entry.steps = steps;
    entry.result = format!(
        "{:.1} tok/s [estimated]",
        decode.cluster_tokens_per_sec.value
    );
    entry.source = "vLLM paper (Kwon et al. SOSP 2023, arxiv.org/abs/2309.06180)".to_string();
    entry.methodology_anchor = "#decode-tokens-per-second".to_string();
    entries.push(entry);
}

fn concurrency(report: &EvaluationReport, entries: &mut Vec<ExplainEntry>) {
    let Some(concurrency) = &report.concurrency else {
        return;
    };

    let mut k_entry = ExplainEntry::new(
        "K bound (memory capacity)",
        "K = floor(per_GPU_headroom_bytes / per_GPU_kv_bytes_per_request)",
    );
    k_entry.inputs = vec![
        ExplainInput::new(
            "per_GPU_headroom_bytes",
            fmt_u64(concurrency.k_source_headroom_bytes),
            "[estimated]",
        ),
        ExplainInput::with_note(
            "per_GPU_kv_bytes_per_request",
            fmt_u64(concurrency.k_source_kv_per_req_bytes),
            "[estimated]",
            "post TP/PP sharding; MLA uses one TP KV shard and PP splits by pipeline stage",
        ),
    ];
    k_entry.steps = vec![format!(
        "K = floor({} / {}) = {}",
        fmt_u64(concurrency.k_source_headroom_bytes),
        fmt_u64(concurrency.k_source_kv_per_req_bytes),
        concurrency.k_bound.value
    )];
    k_entry.result = format!(
        "K = {} [{}]",
        concurrency.k_bound.value, concurrency.k_bound.label
    );
    k_entry.source =
        "TP KV sharding rule plus vLLM/SGLang multi-node TP/PP launch conventions".to_string();
    k_entry.methodology_anchor = "#k-bound-memory-capacity".to_string();
    entries.push(k_entry);

    let cluster_tps = report
        .decode
        .as_ref()
        .map(|decode| decode.cluster_tokens_per_sec.value)
        .unwrap_or(0.0);
    let mut l_entry = ExplainEntry::new(
        "L bound (compute/bandwidth at SLA)",
        "L = floor(cluster_tok_per_sec / target_per_user_tok_per_sec / degradation_factor)",
    );
    l_entry.inputs = vec![
        ExplainInput::new(
            "cluster_tok_per_sec",
            format!("{cluster_tps:.1}"),
            "[estimated]",
        ),
        ExplainInput::with_note(
            "target_per_user_tok_per_sec",
            format!("{:.1}", concurrency.target_tokens_per_sec),
            "[user-set]",
            "SLA, override with --target-tokens-per-sec",
        ),
        ExplainInput::with_note(
            "degradation_factor",
            format!("{:.2}", concurrency.degradation_factor),
            "[user-set]",
            "default 1.0 = no degradation; override with --concurrency-degradation",
        ),
    ];
    l_entry.steps = vec![format!(
        "L = floor({cluster_tps:.1} / {:.1} / {:.2}) = {}",
        concurrency.target_tokens_per_sec,
        concurrency.degradation_factor,
        concurrency.l_bound.value
    )];
    l_entry.result = format!(
        "L = {} [{}]",
        concurrency.l_bound.value, concurrency.l_bound.label
    );
    l_entry.source = "Standard SLA-based capacity planning".to_string();
    l_entry.methodology_anchor = "#l-bound-compute-bandwidth-at-sla".to_string();
    entries.push(l_entry);

    let mut verdict = ExplainEntry::new(
        "Max concurrent + bottleneck verdict",
        "max_concurrent = min(K, L); bottleneck = 'memory_capacity' if K <= L else 'memory_bandwidth / compute'",
    );
    verdict.inputs = vec![
        ExplainInput::new(
            "K",
            concurrency.k_bound.value.to_string(),
            format!("[{}]", concurrency.k_bound.label).as_str(),
        ),
        ExplainInput::new(
            "L",
            concurrency.l_bound.value.to_string(),
            format!("[{}]", concurrency.l_bound.label).as_str(),
        ),
    ];
    verdict.steps = vec![
        format!(
            "max_concurrent = min(K={}, L={}) = {}",
            concurrency.k_bound.value, concurrency.l_bound.value, concurrency.max_concurrent.value
        ),
        format!("bottleneck = {}", concurrency.bottleneck),
    ];
    verdict.result = format!(
        "{} concurrent, bottleneck = {}",
        concurrency.max_concurrent.value, concurrency.bottleneck
    );
    verdict.source = concurrency.bottleneck_reason_en.clone();
    verdict.methodology_anchor = "#concurrency-bounds-k-l".to_string();
    entries.push(verdict);
}

fn chosen_gpu_count(fleet: &crate::fleet::planner::FleetRecommendation) -> u64 {
    fleet
        .best_option()
        .map(|option| option.gpu_count)
        .unwrap_or(1)
}

fn average_keep_fraction(ratios: &[u64]) -> f64 {
    let total = ratios
        .iter()
        .map(|ratio| {
            if *ratio == 0 {
                1.0
            } else {
                1.0 / *ratio as f64
            }
        })
        .sum::<f64>();
    total / ratios.len() as f64
}

fn ceil_div(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    numerator.div_ceil(denominator)
}

fn fmt_ctx(ctx: u64) -> String {
    if ctx >= 1_000_000 {
        format!("{}M", ctx / 1_000_000)
    } else if ctx >= 1024 {
        format!("{}K", ctx / 1024)
    } else {
        ctx.to_string()
    }
}

fn fmt_u64(value: u64) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    let first_group = text.len() % 3;
    for (idx, ch) in text.chars().enumerate() {
        if idx > 0 && (idx == first_group || (idx > first_group && (idx - first_group) % 3 == 0)) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn fmt_context(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        if tokens % 1_000_000 == 0 {
            return format!("{}M", tokens / 1_000_000);
        }
        return format!("{:.1}M", tokens as f64 / 1_000_000.0);
    }
    if tokens >= 1024 {
        return format!("{}K", tokens / 1024);
    }
    tokens.to_string()
}

#[allow(dead_code)]
fn _profile_for_docs(_: &ArchitectureProfile) {}
