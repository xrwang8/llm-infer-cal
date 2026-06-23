use crate::architecture::profile::ArchitectureProfile;
use crate::hardware::loader::GPUSpec;
use crate::output::labels::{AnnotatedValue, Label};

pub const DEFAULT_PREFILL_UTILIZATION: f64 = 0.40;
pub const DEFAULT_DECODE_BW_UTILIZATION: f64 = 0.50;
pub const DEFAULT_CLUSTER_COMM_EFFICIENCY: f64 = 0.90;
pub const DEFAULT_CONCURRENCY_DEGRADATION: f64 = 1.0;

#[derive(Clone, Debug, PartialEq)]
pub struct PrefillEstimate {
    pub total_flops: AnnotatedValue<u64>,
    pub peak_effective_tflops: AnnotatedValue<f64>,
    pub latency_ms: AnnotatedValue<f64>,
    pub utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodeEstimate {
    pub active_weight_bytes_per_gpu: AnnotatedValue<u64>,
    pub per_gpu_tokens_per_sec: AnnotatedValue<f64>,
    pub cluster_tokens_per_sec: AnnotatedValue<f64>,
    pub bw_utilization: f64,
    pub cluster_comm_efficiency: f64,
    pub moe_active_weight_bytes_per_gpu: Option<AnnotatedValue<u64>>,
    pub moe_active_tokens_per_sec: Option<AnnotatedValue<f64>>,
}

pub fn estimate_prefill(
    _profile: &ArchitectureProfile,
    total_params: u64,
    gpu: &GPUSpec,
    num_gpus: u64,
    input_tokens: u64,
    utilization: f64,
) -> PrefillEstimate {
    let flops = 2 * total_params * input_tokens;
    let aggregate_tflops = gpu.fp16_tflops * num_gpus as f64 * utilization;

    if aggregate_tflops <= 0.0 || total_params == 0 || input_tokens == 0 {
        return PrefillEstimate {
            total_flops: AnnotatedValue::new(0, Label::Unknown, Some("insufficient inputs")),
            peak_effective_tflops: AnnotatedValue::new(0.0, Label::Unknown, None),
            latency_ms: AnnotatedValue::new(0.0, Label::Unknown, None),
            utilization,
        };
    }

    let latency_ms = flops as f64 / (aggregate_tflops * 1e12) * 1000.0;
    let total_source = format!(
        "2 × {} params × {} tokens",
        fmt_u64(total_params),
        fmt_u64(input_tokens)
    );
    let peak_source = format!(
        "{:.1} × {} GPUs × {:.0}% util",
        gpu.fp16_tflops,
        num_gpus,
        utilization * 100.0
    );
    let latency_source = format!(
        "{} FLOPs / ({aggregate_tflops:.1} effective TFLOPS × 1e12)",
        fmt_scientific(flops)
    );

    PrefillEstimate {
        total_flops: AnnotatedValue::new(flops, Label::Estimated, Some(&total_source)),
        peak_effective_tflops: AnnotatedValue::new(
            aggregate_tflops,
            Label::Estimated,
            Some(&peak_source),
        ),
        latency_ms: AnnotatedValue::new(latency_ms, Label::Estimated, Some(&latency_source)),
        utilization,
    }
}

pub fn nvlink_efficiency(gpu: &GPUSpec, num_gpus: u64) -> f64 {
    if num_gpus <= 1 {
        return 1.0;
    }
    let nvlink = gpu.nvlink_bandwidth_gbps;
    if nvlink >= 900 {
        return 1.0;
    }
    if nvlink == 0 {
        return 0.80;
    }
    0.85 + 0.15 * (nvlink as f64 / 900.0)
}

pub fn estimate_decode(
    profile: &ArchitectureProfile,
    total_weight_bytes: u64,
    gpu: &GPUSpec,
    num_gpus: u64,
    bw_utilization: f64,
    cluster_comm_efficiency: f64,
    moe_active_params_ratio: Option<f64>,
) -> DecodeEstimate {
    let Some(memory_bandwidth_gbps) = gpu.memory_bandwidth_gbps else {
        return unknown_decode(bw_utilization, cluster_comm_efficiency);
    };
    if memory_bandwidth_gbps == 0 {
        return unknown_decode(bw_utilization, cluster_comm_efficiency);
    }

    let num_gpus = num_gpus.max(1);
    let bw_bytes_per_s = memory_bandwidth_gbps as f64 * 1e9;
    let effective_bw = bw_bytes_per_s * bw_utilization;
    let weight_per_gpu = (total_weight_bytes / num_gpus).max(1);
    let per_gpu_tps = effective_bw / weight_per_gpu as f64;
    let nvlink_eff = nvlink_efficiency(gpu, num_gpus);
    let effective_comm_eff = cluster_comm_efficiency * nvlink_eff;
    let cluster_tps = per_gpu_tps * num_gpus as f64 * effective_comm_eff;

    let mut moe_active_weight = None;
    let mut moe_active_tps = None;
    if profile.is_moe() {
        if let Some(ratio) = moe_active_params_ratio.filter(|ratio| *ratio > 0.0) {
            let active_bytes = (weight_per_gpu as f64 * ratio) as u64;
            let weight_source = format!(
                "{} × {ratio:.3} (active/total ratio)",
                fmt_u64(weight_per_gpu)
            );
            moe_active_weight = Some(AnnotatedValue::new(
                active_bytes,
                Label::Estimated,
                Some(&weight_source),
            ));
            if active_bytes > 0 {
                let active_cluster_tps =
                    (effective_bw / active_bytes as f64) * num_gpus as f64 * effective_comm_eff;
                let tps_source = format!(
                    "optimistic MoE active-only: effective_bw / {} × {num_gpus} × {effective_comm_eff:.3}",
                    fmt_u64(active_bytes)
                );
                moe_active_tps = Some(AnnotatedValue::new(
                    active_cluster_tps,
                    Label::Estimated,
                    Some(&tps_source),
                ));
            }
        }
    }

    let weight_source = format!(
        "{} bytes / {num_gpus} TP ranks",
        fmt_u64(total_weight_bytes)
    );
    let per_gpu_source = format!(
        "{memory_bandwidth_gbps} GB/s × {:.0}% util / {} weight bytes",
        bw_utilization * 100.0,
        fmt_u64(weight_per_gpu)
    );
    let cluster_source = format!(
        "per-GPU × {num_gpus} GPUs × {:.0}% comm × {nvlink_eff:.3} NVLink penalty (NVLink={} GB/s)",
        cluster_comm_efficiency * 100.0,
        gpu.nvlink_bandwidth_gbps
    );

    DecodeEstimate {
        active_weight_bytes_per_gpu: AnnotatedValue::new(
            weight_per_gpu,
            Label::Estimated,
            Some(&weight_source),
        ),
        per_gpu_tokens_per_sec: AnnotatedValue::new(
            per_gpu_tps,
            Label::Estimated,
            Some(&per_gpu_source),
        ),
        cluster_tokens_per_sec: AnnotatedValue::new(
            cluster_tps,
            Label::Estimated,
            Some(&cluster_source),
        ),
        bw_utilization,
        cluster_comm_efficiency,
        moe_active_weight_bytes_per_gpu: moe_active_weight,
        moe_active_tokens_per_sec: moe_active_tps,
    }
}

fn unknown_decode(bw_utilization: f64, cluster_comm_efficiency: f64) -> DecodeEstimate {
    DecodeEstimate {
        active_weight_bytes_per_gpu: AnnotatedValue::new(
            0,
            Label::Unknown,
            Some("GPU memory_bandwidth_gbps not in database"),
        ),
        per_gpu_tokens_per_sec: AnnotatedValue::new(
            0.0,
            Label::Unknown,
            Some("GPU memory_bandwidth_gbps not in database"),
        ),
        cluster_tokens_per_sec: AnnotatedValue::new(
            0.0,
            Label::Unknown,
            Some("GPU memory_bandwidth_gbps not in database"),
        ),
        bw_utilization,
        cluster_comm_efficiency,
        moe_active_weight_bytes_per_gpu: None,
        moe_active_tokens_per_sec: None,
    }
}

fn fmt_u64(value: u64) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    for (idx, ch) in text.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn fmt_scientific(value: u64) -> String {
    let text = format!("{value:.2e}");
    if let Some((mantissa, exponent)) = text.split_once('e') {
        if !exponent.starts_with(['+', '-']) {
            return format!("{mantissa}e+{exponent}");
        }
    }
    text
}
