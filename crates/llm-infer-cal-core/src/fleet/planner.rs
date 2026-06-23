use crate::architecture::profile::ArchitectureProfile;
use crate::hardware::loader::GPUSpec;

const OVERHEAD_FRACTION: f64 = 0.10;
const REFERENCE_CTX_TOKENS: u64 = 131_072;
const MAX_TP_SINGLE_NODE: u64 = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct FleetOption {
    pub tier: &'static str,
    pub gpu_count: u64,
    pub weight_bytes_per_gpu: u64,
    pub kv_bytes_per_request: u64,
    pub max_concurrent_at_reference_ctx: u64,
    pub max_concurrent_by_context: Vec<(u64, u64)>,
    pub usable_bytes_per_gpu: u64,
    pub fits: bool,
    pub reason_en: String,
    pub reason_zh: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FleetRecommendation {
    pub options: Vec<FleetOption>,
    pub best_tier: &'static str,
    pub valid_tp_sizes: Vec<u64>,
    pub constraint_note_en: String,
    pub constraint_note_zh: String,
}

struct EvalContext<'a> {
    profile: &'a ArchitectureProfile,
    weight_bytes: u64,
    kv_bytes: u64,
    usable_per_gpu: u64,
    valid_tp: &'a [u64],
    kv_by_context: &'a [(u64, u64)],
}

pub fn plan(
    profile: &ArchitectureProfile,
    weight_bytes: u64,
    kv_bytes_per_request_at_ref: u64,
    gpu: &GPUSpec,
    forced_gpu_count: Option<u64>,
    kv_bytes_by_context: &[(u64, u64)],
) -> FleetRecommendation {
    let usable_per_gpu =
        (gpu.memory_gb as f64 * 1_000_000_000.0 * (1.0 - OVERHEAD_FRACTION)) as u64;
    let valid_tp = valid_tp_sizes(profile);
    let constraint_en = constraint_note_en(profile, &valid_tp);
    let constraint_zh = constraint_note_zh(profile, &valid_tp);
    let eval = EvalContext {
        profile,
        weight_bytes,
        kv_bytes: kv_bytes_per_request_at_ref,
        usable_per_gpu,
        valid_tp: &valid_tp,
        kv_by_context: kv_bytes_by_context,
    };

    if let Some(forced_gpu_count) = forced_gpu_count {
        let option = evaluate_count(forced_gpu_count, "dev", &eval);
        return FleetRecommendation {
            options: vec![option],
            best_tier: "dev",
            valid_tp_sizes: valid_tp,
            constraint_note_en: constraint_en,
            constraint_note_zh: constraint_zh,
        };
    }

    let tiers = [("min", 1_u64), ("dev", 8), ("prod", 16)];
    let mut options = Vec::new();
    for (tier, concurrent) in tiers {
        let gpu_count = smallest_fitting_count(
            &valid_tp,
            profile,
            weight_bytes,
            kv_bytes_per_request_at_ref,
            usable_per_gpu,
            concurrent,
        );
        let chosen = gpu_count.unwrap_or_else(|| *valid_tp.iter().max().unwrap_or(&1));
        options.push(evaluate_count(chosen, tier, &eval));
    }

    let best_tier = if options.get(1).is_some_and(|option| option.fits) {
        "dev"
    } else if options.first().is_some_and(|option| option.fits) {
        "min"
    } else {
        "prod"
    };

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

pub fn kv_shards(profile: &ArchitectureProfile, tp_size: u64) -> u64 {
    let Some(attention) = &profile.attention else {
        return 1;
    };
    let kv_heads = attention.num_kv_heads.max(1);
    tp_size.min(kv_heads)
}

fn smallest_fitting_count(
    valid_tp: &[u64],
    profile: &ArchitectureProfile,
    weight_bytes: u64,
    kv_bytes: u64,
    usable_per_gpu: u64,
    concurrent: u64,
) -> Option<u64> {
    valid_tp.iter().copied().find(|count| {
        fits(
            *count,
            profile,
            weight_bytes,
            kv_bytes,
            usable_per_gpu,
            concurrent,
        )
    })
}

fn fits(
    gpu_count: u64,
    profile: &ArchitectureProfile,
    weight_bytes: u64,
    kv_bytes: u64,
    usable_per_gpu: u64,
    concurrent: u64,
) -> bool {
    let weight_per_gpu = ceil_div(weight_bytes, gpu_count);
    let shards = kv_shards(profile, gpu_count);
    let kv_per_gpu = ceil_div(kv_bytes, shards);
    let needed = weight_per_gpu + concurrent * kv_per_gpu;
    needed <= usable_per_gpu
}

fn evaluate_count(gpu_count: u64, tier: &'static str, eval: &EvalContext<'_>) -> FleetOption {
    let weight_per_gpu = ceil_div(eval.weight_bytes, gpu_count);
    let shards = kv_shards(eval.profile, gpu_count);
    let kv_per_gpu = ceil_div(eval.kv_bytes, shards);
    let headroom = eval.usable_per_gpu.saturating_sub(weight_per_gpu);
    let max_concurrent = headroom.checked_div(kv_per_gpu).unwrap_or(0);
    let mut max_concurrent_by_context = eval
        .kv_by_context
        .iter()
        .map(|(ctx, kv)| {
            let kv_per_gpu = ceil_div(*kv, shards);
            let concurrent = if *kv > 0 {
                headroom.checked_div(kv_per_gpu).unwrap_or(0)
            } else {
                0
            };
            (*ctx, concurrent)
        })
        .collect::<Vec<_>>();
    max_concurrent_by_context.sort_by_key(|(ctx, _)| *ctx);

    let tier_concurrent = match tier {
        "min" => 1,
        "dev" => 8,
        "prod" => 16,
        _ => 8,
    };
    let fits = fits(
        gpu_count,
        eval.profile,
        eval.weight_bytes,
        eval.kv_bytes,
        eval.usable_per_gpu,
        tier_concurrent,
    );

    let (reason_en, reason_zh) = if !eval.valid_tp.contains(&gpu_count) {
        (
            format!(
                "GPU count {gpu_count} does not divide num_heads - valid TP sizes: {:?}",
                eval.valid_tp
            ),
            format!(
                "GPU 张数 {gpu_count} 无法整除注意力头数——有效 TP 张数：{:?}",
                eval.valid_tp
            ),
        )
    } else if !fits {
        (
            format!(
                "Weights + {tier_concurrent}x KV would exceed {:.1} GB usable per GPU",
                eval.usable_per_gpu as f64 / 1e9
            ),
            format!(
                "权重 + {tier_concurrent} 份 KV 超过单卡可用的 {:.1} GB",
                eval.usable_per_gpu as f64 / 1e9
            ),
        )
    } else {
        (
            format!(
                "fits ~{max_concurrent} concurrent @ {}K ctx",
                REFERENCE_CTX_TOKENS / 1024
            ),
            format!(
                "可容纳约 {max_concurrent} 并发请求 @ {}K 上下文",
                REFERENCE_CTX_TOKENS / 1024
            ),
        )
    };

    FleetOption {
        tier,
        gpu_count,
        weight_bytes_per_gpu: weight_per_gpu,
        kv_bytes_per_request: eval.kv_bytes,
        max_concurrent_at_reference_ctx: max_concurrent,
        max_concurrent_by_context,
        usable_bytes_per_gpu: eval.usable_per_gpu,
        fits,
        reason_en,
        reason_zh,
    }
}

fn constraint_note_en(profile: &ArchitectureProfile, valid_tp: &[u64]) -> String {
    let heads = profile
        .attention
        .as_ref()
        .map(|attention| attention.num_heads)
        .unwrap_or(0);
    format!(
        "TP must divide num_heads={heads}. Candidates within one node (<=8 GPUs): {valid_tp:?}."
    )
}

fn constraint_note_zh(profile: &ArchitectureProfile, valid_tp: &[u64]) -> String {
    let heads = profile
        .attention
        .as_ref()
        .map(|attention| attention.num_heads)
        .unwrap_or(0);
    format!("TP 张数必须整除 num_heads={heads}。单节点（≤8 卡）候选：{valid_tp:?}。")
}

fn ceil_div(value: u64, divisor: u64) -> u64 {
    if divisor == 0 {
        return value;
    }
    value.div_ceil(divisor)
}
