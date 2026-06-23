use crate::output::labels::{AnnotatedValue, Label};
use crate::performance::compute::{DecodeEstimate, DEFAULT_CONCURRENCY_DEGRADATION};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bottleneck {
    MemoryCapacity,
    MemoryBandwidth,
    Compute,
    InsufficientData,
}

impl Bottleneck {
    pub const fn as_str(self) -> &'static str {
        match self {
            Bottleneck::MemoryCapacity => "memory_capacity",
            Bottleneck::MemoryBandwidth => "memory_bandwidth",
            Bottleneck::Compute => "compute",
            Bottleneck::InsufficientData => "insufficient_data",
        }
    }
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConcurrencyAnalysis {
    pub k_bound: AnnotatedValue<u64>,
    pub k_source_headroom_bytes: u64,
    pub k_source_kv_per_req_bytes: u64,
    pub l_bound: AnnotatedValue<u64>,
    pub target_tokens_per_sec: f64,
    pub degradation_factor: f64,
    pub max_concurrent: AnnotatedValue<u64>,
    pub bottleneck: Bottleneck,
    pub bottleneck_reason_en: String,
    pub bottleneck_reason_zh: String,
}

pub fn analyze(
    cluster_headroom_bytes: u64,
    kv_bytes_per_request: u64,
    decode: &DecodeEstimate,
    target_tokens_per_sec: f64,
    degradation: f64,
) -> ConcurrencyAnalysis {
    let degradation = if degradation == 0.0 {
        DEFAULT_CONCURRENCY_DEGRADATION
    } else {
        degradation
    };

    let (k, k_label, k_source) = if kv_bytes_per_request == 0 {
        (
            0,
            Label::Unknown,
            "KV cache per request is zero or unknown".to_string(),
        )
    } else {
        let k = cluster_headroom_bytes
            .checked_div(kv_bytes_per_request)
            .unwrap_or(0);
        (
            k,
            Label::Estimated,
            format!(
                "{} bytes headroom / {} bytes per request",
                fmt_u64(cluster_headroom_bytes),
                fmt_u64(kv_bytes_per_request)
            ),
        )
    };

    let cluster_tps = decode.cluster_tokens_per_sec.value;
    let (l_bound, l_label, l_source) = if cluster_tps <= 0.0
        || target_tokens_per_sec <= 0.0
        || degradation <= 0.0
    {
        (
            0,
            Label::Unknown,
            "cluster throughput or target is zero / unknown".to_string(),
        )
    } else {
        let l_bound = (cluster_tps / target_tokens_per_sec / degradation).floor() as u64;
        (
                l_bound,
                Label::Estimated,
                format!(
                    "{cluster_tps:.1} tok/s cluster / {target_tokens_per_sec:.1} target / {degradation:.2} degradation"
                ),
            )
    };

    let (max_n, bottleneck, reason_en, reason_zh) = if k == 0 && l_bound == 0 {
        (
            0,
            Bottleneck::InsufficientData,
            "Both K and L unknown - cannot conclude.".to_string(),
            "K 和 L 均未知，无法得出结论。".to_string(),
        )
    } else if k <= l_bound {
        (
            k,
            Bottleneck::MemoryCapacity,
            format!(
                "K ({k}) <= L ({l_bound}) -> memory-capacity bound. KV cache exhausts GPU headroom before throughput SLA does."
            ),
            format!(
                "K ({k}) ≤ L ({l_bound}) → 显存容量瓶颈。先达到 KV cache 容量上限，才到吞吐目标。"
            ),
        )
    } else {
        (
            l_bound,
            Bottleneck::MemoryBandwidth,
            format!(
                "L ({l_bound}) < K ({k}) -> memory-bandwidth / compute bound. Cluster can't sustain target tok/s per user at this concurrency."
            ),
            format!(
                "L ({l_bound}) < K ({k}) → 带宽/算力瓶颈。集群在此并发下无法维持目标 tok/s。"
            ),
        )
    };

    let max_source = format!("min(K={k}, L={l_bound})");
    ConcurrencyAnalysis {
        k_bound: AnnotatedValue::new(k, k_label, Some(&k_source)),
        k_source_headroom_bytes: cluster_headroom_bytes,
        k_source_kv_per_req_bytes: kv_bytes_per_request,
        l_bound: AnnotatedValue::new(l_bound, l_label, Some(&l_source)),
        target_tokens_per_sec,
        degradation_factor: degradation,
        max_concurrent: AnnotatedValue::new(
            max_n,
            if max_n > 0 {
                Label::Estimated
            } else {
                Label::Unknown
            },
            Some(&max_source),
        ),
        bottleneck,
        bottleneck_reason_en: reason_en,
        bottleneck_reason_zh: reason_zh,
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
