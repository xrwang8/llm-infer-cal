use crate::architecture::profile::{ArchitectureProfile, AttentionVariant};
use crate::output::labels::{AnnotatedValue, Label};
use crate::weight_analyzer::QuantizationScheme;

pub fn estimate_total_params(profile: &ArchitectureProfile) -> AnnotatedValue<u64> {
    if profile.num_hidden_layers == 0 || profile.hidden_size == 0 {
        return AnnotatedValue::new(
            0,
            Label::Unknown,
            Some("insufficient shape info in profile"),
        );
    }

    let hidden = profile.hidden_size;
    let n_layers = profile.num_hidden_layers;
    let vocab = profile.vocab_size;

    let embed_params = vocab * hidden;
    let output_head_params = if profile.tie_word_embeddings {
        0
    } else {
        vocab * hidden
    };

    let attn_params = attention_params(profile);
    let ffn_params = ffn_params(profile);
    let norm_params = 2 * hidden;

    let total =
        embed_params + output_head_params + (attn_params + ffn_params + norm_params) * n_layers;

    AnnotatedValue::new(
        total,
        Label::Estimated,
        Some(&format!(
            "{vocab} vocab * {hidden} hidden * 2 (embed+head) + {n_layers} layers * ({} attn + {} ffn + norms)",
            fmt_u64(attn_params),
            fmt_u64(ffn_params),
        )),
    )
}

pub fn predicted_bytes_under_quant(total_params: u64, scheme: &str) -> AnnotatedValue<u64> {
    let Some(scheme) = QuantizationScheme::from_name(scheme) else {
        return AnnotatedValue::new(0, Label::Unknown, Some("no bytes-per-param mapping"));
    };
    let Some((numerator, denominator)) = scheme.bpp_ratio() else {
        return AnnotatedValue::new(0, Label::Unknown, Some("no bytes-per-param mapping"));
    };

    AnnotatedValue::new(
        total_params * numerator / denominator,
        Label::Estimated,
        Some("params multiplied by quantization bytes-per-param"),
    )
}

fn attention_params(profile: &ArchitectureProfile) -> u64 {
    let Some(attention) = &profile.attention else {
        return 0;
    };
    let hidden = profile.hidden_size;

    if attention.variant == AttentionVariant::Mla {
        if let Some(q_lora) = attention.q_lora_rank {
            if q_lora > 0 {
                let kv_lora = attention.kv_lora_rank.unwrap_or(q_lora);
                let head_total = attention.num_heads * attention.head_dim;
                return hidden * q_lora
                    + q_lora * head_total
                    + hidden * kv_lora * 2
                    + kv_lora * head_total
                    + head_total * q_lora
                    + q_lora * hidden;
            }
        }
    }

    let q_out = attention.num_heads * attention.head_dim;
    let kv_out = attention.num_kv_heads * attention.head_dim;
    hidden * q_out + hidden * kv_out * 2 + q_out * hidden
}

fn ffn_params(profile: &ArchitectureProfile) -> u64 {
    let hidden = profile.hidden_size;

    if let Some(moe) = &profile.moe {
        let single_expert = 3 * hidden * moe.moe_intermediate_size;
        let total_experts = moe.num_routed_experts + moe.num_shared_experts;
        let router = hidden * moe.num_routed_experts;
        return single_expert * total_experts + router;
    }

    let intermediate = profile
        .intermediate_size
        .filter(|intermediate| *intermediate > 0)
        .unwrap_or(4 * hidden);
    3 * hidden * intermediate
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
