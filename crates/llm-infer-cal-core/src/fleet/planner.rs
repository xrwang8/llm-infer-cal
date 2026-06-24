use crate::architecture::formulas::activation::DEFAULT_BATCHED_TOKENS;
use crate::architecture::formulas::weight::estimate_total_params;
use crate::architecture::profile::{ArchitectureProfile, AttentionVariant};
use crate::hardware::loader::GPUSpec;

const OVERHEAD_FRACTION: f64 = 0.10;
const OVERHEAD_FLOOR_BYTES: u64 = 3_000_000_000;
const MAX_TP_SINGLE_NODE: u64 = 8;
const MAX_DISTRIBUTED_TP: u64 = 64;
const MAX_PIPELINE_PARALLEL: u64 = 8;
// Prefill peak heuristic: while decode KV for all concurrent requests remains
// resident, one eighth of those requests are active in a 1500-token chunked
// prefill step. Mirrors the Python implementation.
const PREFILL_ACTIVE_REQUEST_DIVISOR: u64 = 8;
const PREFILL_CHUNK_TOKENS_PER_ACTIVE_REQUEST: u64 = 1500;

#[derive(Clone, Debug, PartialEq)]
pub struct FleetOption {
    pub tier: &'static str,
    pub gpu_count: u64,
    pub tensor_parallel_size: u64,
    pub pipeline_parallel_size: u64,
    pub node_count: u64,
    pub main_weight_bytes_per_gpu: u64,
    pub speculative_weight_bytes_per_gpu: u64,
    pub cpu_offload_bytes_per_gpu: u64,
    pub weight_bytes_per_gpu: u64,
    pub kv_bytes_per_request: u64,
    pub kv_bytes_per_request_per_gpu: u64,
    pub activation_bytes_per_request: u64,
    pub activation_bytes_per_request_per_gpu: u64,
    pub kv_reference_context_tokens: u64,
    pub tier_concurrent_requests: u64,
    pub decode_required_bytes_per_gpu_at_tier: u64,
    pub prefill_activation_bytes_per_gpu_at_tier: u64,
    pub prefill_required_bytes_per_gpu_at_tier: u64,
    pub required_bytes_per_gpu_at_tier: u64,
    pub max_concurrent_at_reference_ctx: u64,
    pub max_concurrent_by_context: Vec<(u64, u64)>,
    pub usable_bytes_per_gpu: u64,
    pub reserved_bytes_per_gpu: u64,
    pub fits: bool,
    pub reason_en: String,
    pub reason_zh: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FleetMemoryOptions {
    pub target_concurrent_requests: Option<u64>,
    pub speculative_weight_bytes: u64,
    pub cpu_offload_bytes_per_gpu: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FleetRecommendation {
    pub options: Vec<FleetOption>,
    pub best_tier: Option<&'static str>,
    pub valid_tp_sizes: Vec<u64>,
    pub constraint_note_en: String,
    pub constraint_note_zh: String,
}

impl FleetRecommendation {
    pub fn best_option(&self) -> Option<&FleetOption> {
        let tier = self.best_tier?;
        self.options
            .iter()
            .find(|option| option.tier == tier && option.fits)
    }
}

struct EvalContext<'a> {
    profile: &'a ArchitectureProfile,
    weight_bytes: u64,
    kv_bytes: u64,
    activation_bytes: u64,
    reference_context_tokens: u64,
    total_memory_per_gpu: u64,
    usable_per_gpu: u64,
    valid_tp: &'a [u64],
    candidates: &'a [u64],
    kv_by_context: &'a [(u64, u64)],
    memory_options: FleetMemoryOptions,
}

pub fn plan(
    profile: &ArchitectureProfile,
    weight_bytes: u64,
    kv_bytes_per_request_at_ref: u64,
    reference_context_tokens: u64,
    gpu: &GPUSpec,
    forced_gpu_count: Option<u64>,
    kv_bytes_by_context: &[(u64, u64)],
) -> FleetRecommendation {
    plan_with_activation(
        profile,
        weight_bytes,
        kv_bytes_per_request_at_ref,
        0,
        reference_context_tokens,
        gpu,
        forced_gpu_count,
        kv_bytes_by_context,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn plan_with_activation(
    profile: &ArchitectureProfile,
    weight_bytes: u64,
    kv_bytes_per_request_at_ref: u64,
    activation_bytes_per_request_at_ref: u64,
    reference_context_tokens: u64,
    gpu: &GPUSpec,
    forced_gpu_count: Option<u64>,
    kv_bytes_by_context: &[(u64, u64)],
    activation_bytes_by_context: &[(u64, u64)],
) -> FleetRecommendation {
    plan_with_memory_options(
        profile,
        weight_bytes,
        kv_bytes_per_request_at_ref,
        activation_bytes_per_request_at_ref,
        reference_context_tokens,
        gpu,
        forced_gpu_count,
        kv_bytes_by_context,
        activation_bytes_by_context,
        FleetMemoryOptions::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn plan_with_memory_options(
    profile: &ArchitectureProfile,
    weight_bytes: u64,
    kv_bytes_per_request_at_ref: u64,
    activation_bytes_per_request_at_ref: u64,
    reference_context_tokens: u64,
    gpu: &GPUSpec,
    forced_gpu_count: Option<u64>,
    kv_bytes_by_context: &[(u64, u64)],
    _activation_bytes_by_context: &[(u64, u64)],
    memory_options: FleetMemoryOptions,
) -> FleetRecommendation {
    let total_memory_per_gpu = (gpu.memory_gb as f64 * 1_000_000_000.0) as u64;
    let reserved_per_gpu =
        ((total_memory_per_gpu as f64 * OVERHEAD_FRACTION) as u64).max(OVERHEAD_FLOOR_BYTES);
    let usable_per_gpu = total_memory_per_gpu.saturating_sub(reserved_per_gpu);
    let valid_tp = valid_tp_sizes(profile);
    let candidates = candidate_gpu_counts(profile, &valid_tp);
    let constraint_en = constraint_note_en(profile, &valid_tp);
    let constraint_zh = constraint_note_zh(profile, &valid_tp);
    let eval = EvalContext {
        profile,
        weight_bytes,
        kv_bytes: kv_bytes_per_request_at_ref,
        activation_bytes: activation_bytes_per_request_at_ref,
        reference_context_tokens,
        total_memory_per_gpu,
        usable_per_gpu,
        valid_tp: &valid_tp,
        candidates: &candidates,
        kv_by_context: kv_bytes_by_context,
        memory_options,
    };

    if let Some(forced_gpu_count) = forced_gpu_count {
        let tier = if memory_options.target_concurrent_requests.is_some() {
            "target"
        } else {
            "min"
        };
        let option = evaluate_count(forced_gpu_count, tier, &eval);
        let best_tier = option.fits.then_some(tier);
        return FleetRecommendation {
            options: vec![option],
            best_tier,
            valid_tp_sizes: valid_tp,
            constraint_note_en: constraint_en,
            constraint_note_zh: constraint_zh,
        };
    }

    let default_tiers = [("min", 1_u64), ("dev", 8), ("prod", 16)];
    let target_tiers = [(
        "target",
        memory_options.target_concurrent_requests.unwrap_or(0),
    )];
    let tiers: &[(&str, u64)] = if memory_options.target_concurrent_requests.is_some() {
        &target_tiers
    } else {
        &default_tiers
    };
    let mut options = Vec::new();
    for (tier, concurrent) in tiers {
        let gpu_count = smallest_fitting_count(&candidates, &eval, *concurrent);
        let chosen = gpu_count.unwrap_or_else(|| *candidates.iter().max().unwrap_or(&1));
        options.push(evaluate_count(chosen, tier, &eval));
    }

    let order: &[&str] = if memory_options.target_concurrent_requests.is_some() {
        &["target"]
    } else {
        &["dev", "min", "prod"]
    };
    let best_tier = order.iter().find_map(|tier| {
        options
            .iter()
            .find(|option| option.tier == *tier && option.fits)
            .map(|option| option.tier)
    });

    FleetRecommendation {
        options,
        best_tier,
        valid_tp_sizes: valid_tp,
        constraint_note_en: constraint_en,
        constraint_note_zh: constraint_zh,
    }
}

pub fn valid_tp_sizes(profile: &ArchitectureProfile) -> Vec<u64> {
    let Some(attention) = &profile.attention else {
        return vec![1];
    };
    if attention.num_heads == 0 {
        return vec![1];
    }
    let cap = attention.num_heads.min(MAX_TP_SINGLE_NODE);
    let divisors = (1..=cap)
        .filter(|candidate| attention.num_heads % candidate == 0)
        .collect::<Vec<_>>();
    if divisors.is_empty() {
        vec![1]
    } else {
        divisors
    }
}

fn candidate_gpu_counts(profile: &ArchitectureProfile, valid_tp: &[u64]) -> Vec<u64> {
    let mut candidates = valid_tp.to_vec();
    candidates.extend(distributed_tp_sizes(profile));
    let max_tp = valid_tp.iter().copied().max().unwrap_or(1);
    if max_tp > 0 {
        let layers = profile.num_hidden_layers;
        for pp in 2..=MAX_PIPELINE_PARALLEL {
            if layers == 0 || layers % pp == 0 {
                candidates.push(max_tp * pp);
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn layout_for_count(profile: &ArchitectureProfile, gpu_count: u64) -> (u64, u64, u64) {
    let valid_tp = valid_tp_sizes(profile);
    let max_tp = valid_tp.iter().copied().max().unwrap_or(1);
    if gpu_count <= max_tp {
        return (gpu_count.max(1), 1, 1);
    }
    if distributed_tp_sizes(profile).contains(&gpu_count) {
        return (gpu_count, 1, ceil_div(gpu_count, MAX_TP_SINGLE_NODE).max(1));
    }
    let pp = ceil_div(gpu_count, max_tp).max(1);
    (max_tp, pp, pp)
}

fn distributed_tp_sizes(profile: &ArchitectureProfile) -> Vec<u64> {
    let Some(attention) = &profile.attention else {
        return Vec::new();
    };
    if attention.num_heads <= MAX_TP_SINGLE_NODE {
        return Vec::new();
    }
    let cap = attention.num_heads.min(MAX_DISTRIBUTED_TP);
    (MAX_TP_SINGLE_NODE + 1..=cap)
        .filter(|candidate| attention.num_heads % candidate == 0)
        .collect()
}

pub fn kv_shards(profile: &ArchitectureProfile, tp_size: u64) -> u64 {
    let Some(attention) = &profile.attention else {
        return 1;
    };
    if attention.variant == AttentionVariant::Mla {
        return 1;
    }
    let kv_heads = attention.num_kv_heads.max(1);
    tp_size.min(kv_heads)
}

pub fn effective_kv_shards(profile: &ArchitectureProfile, gpu_count: u64) -> u64 {
    let (tp, pp, _) = layout_for_count(profile, gpu_count);
    pp * kv_shards(profile, tp)
}

fn smallest_fitting_count(
    candidates: &[u64],
    eval: &EvalContext<'_>,
    concurrent: u64,
) -> Option<u64> {
    candidates
        .iter()
        .copied()
        .find(|count| fits(*count, eval, concurrent))
}

fn fits(gpu_count: u64, eval: &EvalContext<'_>, concurrent: u64) -> bool {
    let weight_per_gpu = resident_weight_per_gpu(
        eval.profile,
        eval.weight_bytes,
        gpu_count,
        eval.memory_options,
    )
    .total;
    let shards = effective_kv_shards(eval.profile, gpu_count);
    let kv_per_gpu = ceil_div(eval.kv_bytes, shards);
    let act_shards = activation_shards(eval.profile, gpu_count);
    let activation_per_gpu = ceil_div(eval.activation_bytes, act_shards);
    let required = peak_memory_per_gpu(
        weight_per_gpu,
        activation_per_gpu,
        eval.activation_bytes,
        act_shards,
        kv_per_gpu,
        concurrent,
    );
    required.required <= eval.usable_per_gpu as u128
}

fn evaluate_count(gpu_count: u64, tier: &'static str, eval: &EvalContext<'_>) -> FleetOption {
    let (tensor_parallel_size, pipeline_parallel_size, node_count) =
        layout_for_count(eval.profile, gpu_count);
    let weight = resident_weight_per_gpu(
        eval.profile,
        eval.weight_bytes,
        gpu_count,
        eval.memory_options,
    );
    let shards = effective_kv_shards(eval.profile, gpu_count);
    let kv_per_gpu = ceil_div(eval.kv_bytes, shards);
    let act_shards = activation_shards(eval.profile, gpu_count);
    let activation_per_gpu = ceil_div(eval.activation_bytes, act_shards);
    let max_concurrent = max_concurrent_for_kv(
        weight.total,
        eval.activation_bytes,
        activation_per_gpu,
        act_shards,
        kv_per_gpu,
        eval.usable_per_gpu,
    );
    let mut max_concurrent_by_context = eval
        .kv_by_context
        .iter()
        .map(|(ctx, kv)| {
            let kv_per_gpu = ceil_div(*kv, shards);
            let concurrent = max_concurrent_for_kv(
                weight.total,
                eval.activation_bytes,
                activation_per_gpu,
                act_shards,
                kv_per_gpu,
                eval.usable_per_gpu,
            );
            (*ctx, concurrent)
        })
        .collect::<Vec<_>>();
    max_concurrent_by_context.sort_by_key(|(ctx, _)| *ctx);

    let tier_concurrent = match tier {
        "min" => 1,
        "target" => eval.memory_options.target_concurrent_requests.unwrap_or(1),
        "dev" => 8,
        "prod" => 16,
        _ => 8,
    };
    let peak = peak_memory_per_gpu(
        weight.total,
        activation_per_gpu,
        eval.activation_bytes,
        act_shards,
        kv_per_gpu,
        tier_concurrent,
    );
    let decode_required_bytes_per_gpu_at_tier = saturating_u128_to_u64(peak.decode_required);
    let prefill_activation_bytes_per_gpu_at_tier = saturating_u128_to_u64(peak.prefill_activation);
    let prefill_required_bytes_per_gpu_at_tier = saturating_u128_to_u64(peak.prefill_required);
    let required_bytes_per_gpu_at_tier = saturating_u128_to_u64(peak.required);
    let fits = fits(gpu_count, eval, tier_concurrent);

    let layout_en = if pipeline_parallel_size > 1 {
        format!("TP{tensor_parallel_size}xPP{pipeline_parallel_size}, {node_count} nodes")
    } else {
        format!("TP{tensor_parallel_size}")
    };
    let layout_zh = if pipeline_parallel_size > 1 {
        format!("TP{tensor_parallel_size}×PP{pipeline_parallel_size}，{node_count} 节点")
    } else {
        format!("TP{tensor_parallel_size}")
    };

    let (reason_en, reason_zh) = if !eval.candidates.contains(&gpu_count) {
        (
            format!(
                "GPU count {gpu_count} does not divide num_heads or match a valid multi-node PP layout - valid single-node TP sizes: {:?}",
                eval.valid_tp
            ),
            format!(
                "GPU 张数 {gpu_count} 无法整除注意力头数，也不匹配有效多节点 PP 布局——有效单节点 TP 张数：{:?}",
                eval.valid_tp
            ),
        )
    } else if !fits {
        (
            format!(
                "Weights + prefill peak activation + {tier_concurrent}x KV would exceed {:.1} GB usable per GPU",
                eval.usable_per_gpu as f64 / 1e9
            ),
            format!(
                "权重 + Prefill 峰值 Activation + {tier_concurrent} 份 KV 超过单卡可用的 {:.1} GB",
                eval.usable_per_gpu as f64 / 1e9
            ),
        )
    } else {
        (
            format!(
                "fits ~{max_concurrent} concurrent @ {} ctx ({layout_en})",
                fmt_context(eval.reference_context_tokens)
            ),
            format!(
                "可容纳约 {max_concurrent} 并发请求 @ {} 上下文（{layout_zh}）",
                fmt_context(eval.reference_context_tokens)
            ),
        )
    };

    FleetOption {
        tier,
        gpu_count,
        tensor_parallel_size,
        pipeline_parallel_size,
        node_count,
        main_weight_bytes_per_gpu: weight.main,
        speculative_weight_bytes_per_gpu: weight.speculative,
        cpu_offload_bytes_per_gpu: weight.cpu_offload,
        weight_bytes_per_gpu: weight.total,
        kv_bytes_per_request: eval.kv_bytes,
        kv_bytes_per_request_per_gpu: kv_per_gpu,
        activation_bytes_per_request: eval.activation_bytes,
        activation_bytes_per_request_per_gpu: activation_per_gpu,
        kv_reference_context_tokens: eval.reference_context_tokens,
        tier_concurrent_requests: tier_concurrent,
        decode_required_bytes_per_gpu_at_tier,
        prefill_activation_bytes_per_gpu_at_tier,
        prefill_required_bytes_per_gpu_at_tier,
        required_bytes_per_gpu_at_tier,
        max_concurrent_at_reference_ctx: max_concurrent,
        max_concurrent_by_context,
        usable_bytes_per_gpu: eval.usable_per_gpu,
        reserved_bytes_per_gpu: eval
            .total_memory_per_gpu
            .saturating_sub(eval.usable_per_gpu),
        fits,
        reason_en,
        reason_zh,
    }
}

struct ResidentWeight {
    main: u64,
    speculative: u64,
    cpu_offload: u64,
    total: u64,
}

struct PeakMemory {
    decode_required: u128,
    prefill_activation: u128,
    prefill_required: u128,
    required: u128,
}

fn resident_weight_per_gpu(
    profile: &ArchitectureProfile,
    weight_bytes: u64,
    gpu_count: u64,
    memory_options: FleetMemoryOptions,
) -> ResidentWeight {
    let base_main = resident_main_weight_per_gpu(profile, weight_bytes, gpu_count);
    let cpu_offload = memory_options.cpu_offload_bytes_per_gpu.min(base_main);
    let main = base_main.saturating_sub(cpu_offload);
    let speculative = ceil_div(memory_options.speculative_weight_bytes, gpu_count.max(1));
    ResidentWeight {
        main,
        speculative,
        cpu_offload,
        total: main + speculative,
    }
}

fn resident_main_weight_per_gpu(
    profile: &ArchitectureProfile,
    weight_bytes: u64,
    gpu_count: u64,
) -> u64 {
    if weight_bytes == 0 {
        return 0;
    }

    let (tp, pp, _) = layout_for_count(profile, gpu_count);
    let pp = pp.max(1);
    let stage_weight_bytes = ceil_div(weight_bytes, pp);

    let Some(moe) = &profile.moe else {
        return ceil_div(stage_weight_bytes, tp.max(1));
    };

    let routed_fraction = moe_routed_weight_fraction(profile);
    let routed_bytes = (stage_weight_bytes as f64 * routed_fraction) as u64;
    let static_bytes = stage_weight_bytes.saturating_sub(routed_bytes);

    let routed_shards = moe.num_routed_experts.max(1).min(tp.max(1));
    let static_shards = tp.max(1);
    ceil_div(routed_bytes, routed_shards) + ceil_div(static_bytes, static_shards)
}

fn moe_routed_weight_fraction(profile: &ArchitectureProfile) -> f64 {
    let Some(moe) = &profile.moe else {
        return 0.0;
    };
    if profile.hidden_size == 0
        || profile.num_hidden_layers == 0
        || moe.moe_intermediate_size == 0
        || moe.num_routed_experts == 0
    {
        return 0.80;
    }

    let single_expert = 3_u128 * profile.hidden_size as u128 * moe.moe_intermediate_size as u128;
    let routed = single_expert * moe.num_routed_experts as u128 * profile.num_hidden_layers as u128;
    let total = estimate_total_params(profile).value as u128;
    if total == 0 || routed == 0 || routed >= total {
        return 0.80;
    }
    routed as f64 / total as f64
}

fn activation_shards(profile: &ArchitectureProfile, gpu_count: u64) -> u64 {
    let (tp, _, _) = layout_for_count(profile, gpu_count);
    tp.max(1)
}

fn peak_memory_per_gpu(
    weight_per_gpu: u64,
    activation_per_gpu: u64,
    activation_bytes: u64,
    act_shards: u64,
    kv_per_gpu: u64,
    concurrent: u64,
) -> PeakMemory {
    let concurrent_kv = concurrent as u128 * kv_per_gpu as u128;
    let decode_required = weight_per_gpu as u128 + activation_per_gpu as u128 + concurrent_kv;
    let prefill_activation = prefill_activation_per_gpu(activation_bytes, act_shards, concurrent);
    let prefill_required = weight_per_gpu as u128 + prefill_activation + concurrent_kv;
    let required = decode_required.max(prefill_required);
    PeakMemory {
        decode_required,
        prefill_activation,
        prefill_required,
        required,
    }
}

fn prefill_activation_per_gpu(activation_bytes: u64, act_shards: u64, concurrent: u64) -> u128 {
    let active_prefill_requests = (concurrent / PREFILL_ACTIVE_REQUEST_DIVISOR).max(1);
    let prefill_tokens =
        active_prefill_requests as u128 * PREFILL_CHUNK_TOKENS_PER_ACTIVE_REQUEST as u128;
    let total_prefill_activation =
        activation_bytes as u128 * prefill_tokens / DEFAULT_BATCHED_TOKENS as u128;
    ceil_div_u128(total_prefill_activation, act_shards.max(1) as u128)
}

fn max_concurrent_for_kv(
    weight_per_gpu: u64,
    activation_bytes: u64,
    activation_per_gpu: u64,
    act_shards: u64,
    kv_per_gpu: u64,
    usable_per_gpu: u64,
) -> u64 {
    if kv_per_gpu == 0 {
        return 0;
    }

    let decode_headroom = usable_per_gpu
        .saturating_sub(weight_per_gpu)
        .saturating_sub(activation_per_gpu);
    let decode_bound = decode_headroom / kv_per_gpu;
    let mut low = 0;
    let mut high = decode_bound;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        let required = peak_memory_per_gpu(
            weight_per_gpu,
            activation_per_gpu,
            activation_bytes,
            act_shards,
            kv_per_gpu,
            mid,
        );
        if required.required <= usable_per_gpu as u128 {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
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

fn constraint_note_en(profile: &ArchitectureProfile, valid_tp: &[u64]) -> String {
    let heads = profile
        .attention
        .as_ref()
        .map(|attention| attention.num_heads)
        .unwrap_or(0);
    let candidates = candidate_gpu_counts(profile, valid_tp);
    let distributed_tp = distributed_tp_sizes(profile);
    let pipeline = candidates
        .iter()
        .copied()
        .filter(|count| !valid_tp.contains(count) && !distributed_tp.contains(count))
        .collect::<Vec<_>>();
    let distributed_note = if distributed_tp.is_empty() {
        String::new()
    } else {
        format!(" Cross-node TP candidates: {distributed_tp:?}.")
    };
    let pipeline_note = if pipeline.is_empty() {
        String::new()
    } else {
        format!(
            " Pipeline-parallel candidates (TP{} x PP): {pipeline:?}.",
            valid_tp.iter().max().unwrap_or(&1)
        )
    };
    format!(
        "TP must divide num_heads={heads}. Candidates within one node (<=8 GPUs): {valid_tp:?}."
    ) + &distributed_note
        + &pipeline_note
}

fn constraint_note_zh(profile: &ArchitectureProfile, valid_tp: &[u64]) -> String {
    let heads = profile
        .attention
        .as_ref()
        .map(|attention| attention.num_heads)
        .unwrap_or(0);
    let candidates = candidate_gpu_counts(profile, valid_tp);
    let distributed_tp = distributed_tp_sizes(profile);
    let pipeline = candidates
        .iter()
        .copied()
        .filter(|count| !valid_tp.contains(count) && !distributed_tp.contains(count))
        .collect::<Vec<_>>();
    let distributed_note = if distributed_tp.is_empty() {
        String::new()
    } else {
        format!(" 跨节点 TP 候选：{distributed_tp:?}。")
    };
    let pipeline_note = if pipeline.is_empty() {
        String::new()
    } else {
        format!(
            " 单节点放不下时尝试流水并行候选（TP{} × PP）：{pipeline:?}。",
            valid_tp.iter().max().unwrap_or(&1)
        )
    };
    format!("TP 张数必须整除 num_heads={heads}。单节点（≤8 卡）候选：{valid_tp:?}。")
        + &distributed_note
        + &pipeline_note
}

fn ceil_div(value: u64, divisor: u64) -> u64 {
    if divisor == 0 {
        return value;
    }
    value.div_ceil(divisor)
}

fn ceil_div_u128(value: u128, divisor: u128) -> u128 {
    if divisor == 0 {
        return value;
    }
    value.div_ceil(divisor)
}

fn saturating_u128_to_u64(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}
