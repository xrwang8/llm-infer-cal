use crate::architecture::profile::{
    ArchitectureProfile, AttentionTraits, AttentionVariant, Confidence, Family,
};
use crate::output::labels::{AnnotatedValue, Label};

pub fn compute_kv_cache_bytes(
    profile: &ArchitectureProfile,
    seq_len: u64,
    dtype_bytes: u64,
) -> AnnotatedValue<u64> {
    compute_kv_cache_bits(profile, seq_len, dtype_bytes.saturating_mul(8))
}

pub fn compute_kv_cache_bits(
    profile: &ArchitectureProfile,
    seq_len: u64,
    dtype_bits: u64,
) -> AnnotatedValue<u64> {
    if seq_len == 0 {
        return AnnotatedValue::new(0, Label::Estimated, Some("seq_len <= 0"));
    }

    if profile.family == Family::StateSpace {
        return AnnotatedValue::new(
            0,
            Label::Unknown,
            Some("state-space model has no KV cache concept"),
        );
    }

    if profile.family == Family::Unknown || profile.confidence == Confidence::Low {
        return AnnotatedValue::new(
            0,
            Label::Unknown,
            Some("unknown architecture - cannot estimate KV cache"),
        );
    }

    let Some(attention) = &profile.attention else {
        return AnnotatedValue::new(
            0,
            Label::Unknown,
            Some("missing attention traits or layer count"),
        );
    };
    if profile.num_hidden_layers == 0 {
        return AnnotatedValue::new(
            0,
            Label::Unknown,
            Some("missing attention traits or layer count"),
        );
    }

    let mut effective_seq = seq_len;
    let mut sliding_note = String::new();
    if let Some(sliding_window) = profile.sliding_window {
        if sliding_window > 0 && !attention.variant.is_sparse() {
            effective_seq = seq_len.min(sliding_window);
            if effective_seq < seq_len {
                sliding_note =
                    format!(" (sliding_window={sliding_window} caps {seq_len} -> {effective_seq})");
            }
        }
    }

    let per_layer_per_token = per_layer_per_token_bits(attention, dtype_bits);
    let baseline_bits = per_layer_per_token * effective_seq * profile.num_hidden_layers;
    let mut scale = 1.0;
    let mut variant_note = attention.variant.as_str().to_string();

    if attention.variant == AttentionVariant::CsaHca {
        if let Some(compress_ratios) = &attention.compress_ratios {
            if !compress_ratios.is_empty() {
                let ratio = average_csa_hca_ratio(compress_ratios);
                scale = ratio;
                variant_note = format!("{variant_note} (avg compress ratio {ratio:.3})");
            }
        }
    }

    if attention.variant == AttentionVariant::Nsa {
        if let Some(nsa_topk) = attention.nsa_topk {
            if nsa_topk > 0 {
                let sparsity = (nsa_topk as f64 / effective_seq as f64).min(1.0);
                scale = sparsity;
                variant_note =
                    format!("{variant_note} (nsa_topk={nsa_topk}, sparsity={sparsity:.3})");
            }
        }
    }

    let result_bytes = if dtype_bits % 8 == 0 {
        ((baseline_bits / 8) as f64 * scale) as u64
    } else {
        ceil_div((baseline_bits as f64 * scale) as u64, 8)
    };

    AnnotatedValue::new(
        result_bytes,
        Label::Estimated,
        Some(&format!(
            "{variant_note}: 2*kv_shape*{dtype_bits}b*{effective_seq}*{}{sliding_note}",
            profile.num_hidden_layers
        )),
    )
}

fn per_layer_per_token_bits(attention: &AttentionTraits, dtype_bits: u64) -> u64 {
    if attention.variant == AttentionVariant::Mla {
        if let Some(kv_lora_rank) = attention.kv_lora_rank {
            if kv_lora_rank > 0 {
                return (kv_lora_rank + attention.qk_rope_head_dim.unwrap_or(0)) * dtype_bits;
            }
        }
    }

    2 * attention.num_kv_heads * attention.head_dim * dtype_bits
}

fn average_csa_hca_ratio(compress_ratios: &[u64]) -> f64 {
    if compress_ratios.is_empty() {
        return 1.0;
    }

    let total_fraction = compress_ratios
        .iter()
        .map(|&ratio| if ratio == 0 { 1.0 } else { 1.0 / ratio as f64 })
        .sum::<f64>();
    total_fraction / compress_ratios.len() as f64
}

fn ceil_div(value: u64, divisor: u64) -> u64 {
    if divisor == 0 {
        return value;
    }
    value.div_ceil(divisor)
}
