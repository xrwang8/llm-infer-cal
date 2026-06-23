use serde_json::Value;

use crate::architecture::profile::{AttentionTraits, AttentionVariant, MoeTraits, PositionTraits};

pub fn detect_moe(config: &Value) -> Option<MoeTraits> {
    let routed = get_truthy_u64(config, "n_routed_experts")
        .or_else(|| get_truthy_u64(config, "num_local_experts"))
        .or_else(|| get_truthy_u64(config, "num_experts"))?;

    Some(MoeTraits {
        num_routed_experts: routed,
        num_shared_experts: get_u64(config, "n_shared_experts").unwrap_or(0),
        num_experts_per_tok: get_truthy_u64(config, "num_experts_per_tok")
            .or_else(|| get_truthy_u64(config, "num_experts_per_token"))
            .unwrap_or(1),
        moe_intermediate_size: get_truthy_u64(config, "moe_intermediate_size")
            .or_else(|| get_truthy_u64(config, "intermediate_size"))
            .unwrap_or(0),
    })
}

pub fn detect_attention(config: &Value) -> AttentionTraits {
    let num_heads = get_u64(config, "num_attention_heads").unwrap_or(1);
    let num_kv_heads = get_u64(config, "num_key_value_heads").unwrap_or(num_heads);
    let head_dim = get_truthy_u64(config, "head_dim").unwrap_or_else(|| {
        let hidden_size = get_u64(config, "hidden_size").unwrap_or(0);
        let computed = hidden_size.checked_div(num_heads).unwrap_or(0);
        if computed == 0 {
            1
        } else {
            computed
        }
    });
    let num_layers = get_u64(config, "num_hidden_layers").unwrap_or(0);

    let q_lora = get_truthy_u64(config, "q_lora_rank");
    let kv_lora = get_truthy_u64(config, "kv_lora_rank");
    let compress_ratios = get(config, "compress_ratios")
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(value_to_u64).collect::<Vec<_>>());
    let has_nsa = has_key(config, "nsa_config") || has_key(config, "sparse_attention_cfg");

    let nextn = get_u64(config, "num_nextn_predict_layers").unwrap_or(0);
    if let Some(ratios) = compress_ratios {
        let len = ratios.len() as u64;
        if num_layers > 0 && (len == num_layers || len == num_layers + nextn) {
            return AttentionTraits {
                variant: AttentionVariant::CsaHca,
                num_heads,
                num_kv_heads,
                head_dim,
                q_lora_rank: q_lora,
                kv_lora_rank: kv_lora,
                compress_ratios: Some(ratios),
                nsa_topk: None,
            };
        }
    }

    if has_nsa {
        let nsa_cfg = get(config, "nsa_config")
            .or_else(|| get(config, "sparse_attention_cfg"))
            .and_then(Value::as_object);
        let nsa_topk = nsa_cfg.and_then(|cfg| {
            cfg.get("topk")
                .or_else(|| cfg.get("index_topk"))
                .and_then(value_to_u64)
                .filter(|topk| *topk > 0)
        });
        return AttentionTraits {
            variant: AttentionVariant::Nsa,
            num_heads,
            num_kv_heads,
            head_dim,
            q_lora_rank: None,
            kv_lora_rank: None,
            compress_ratios: None,
            nsa_topk,
        };
    }

    if q_lora.is_some() || kv_lora.is_some() {
        return AttentionTraits {
            variant: AttentionVariant::Mla,
            num_heads,
            num_kv_heads,
            head_dim,
            q_lora_rank: q_lora,
            kv_lora_rank: kv_lora,
            compress_ratios: None,
            nsa_topk: None,
        };
    }

    if num_kv_heads < num_heads {
        return AttentionTraits {
            variant: if num_kv_heads == 1 {
                AttentionVariant::Mqa
            } else {
                AttentionVariant::Gqa
            },
            num_heads,
            num_kv_heads,
            head_dim,
            q_lora_rank: None,
            kv_lora_rank: None,
            compress_ratios: None,
            nsa_topk: None,
        };
    }

    AttentionTraits {
        variant: AttentionVariant::Mha,
        num_heads,
        num_kv_heads,
        head_dim,
        q_lora_rank: None,
        kv_lora_rank: None,
        compress_ratios: None,
        nsa_topk: None,
    }
}

pub fn detect_position(config: &Value) -> PositionTraits {
    let rope_scaling = get(config, "rope_scaling").and_then(Value::as_object);
    let mut rope_type = rope_scaling
        .and_then(|scaling| {
            scaling
                .get("type")
                .or_else(|| scaling.get("rope_type"))
                .and_then(Value::as_str)
        })
        .unwrap_or("rope")
        .to_lowercase();
    if !matches!(rope_type.as_str(), "rope" | "yarn" | "alibi" | "none") {
        rope_type = "rope".to_string();
    }

    PositionTraits {
        rope_type: Some(rope_type),
        rope_theta: get(config, "rope_theta")
            .filter(|value| truthy(value))
            .and_then(value_to_f64),
        rope_scaling_factor: rope_scaling
            .and_then(|scaling| scaling.get("factor"))
            .filter(|value| truthy(value))
            .and_then(value_to_f64),
        max_position_embeddings: get_truthy_u64(config, "max_position_embeddings"),
    }
}

pub fn detect_sliding_window(config: &Value) -> Option<u64> {
    get_truthy_u64(config, "sliding_window")
}

pub(crate) fn get<'a>(config: &'a Value, key: &str) -> Option<&'a Value> {
    config.as_object()?.get(key)
}

pub(crate) fn has_key(config: &Value, key: &str) -> bool {
    config
        .as_object()
        .is_some_and(|object| object.contains_key(key))
}

pub(crate) fn get_u64(config: &Value, key: &str) -> Option<u64> {
    get(config, key).and_then(value_to_u64)
}

pub(crate) fn get_truthy_u64(config: &Value, key: &str) -> Option<u64> {
    get(config, key)
        .filter(|value| truthy(value))
        .and_then(value_to_u64)
}

pub(crate) fn value_to_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .or_else(|| number.as_i64().and_then(|n| u64::try_from(n).ok()))
            .or_else(|| number.as_f64().filter(|n| *n >= 0.0).map(|n| n as u64)),
        Value::String(text) => text.parse().ok(),
        Value::Bool(flag) => Some(u64::from(*flag)),
        _ => None,
    }
}

pub(crate) fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse().ok(),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        _ => None,
    }
}

pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => {
            number.as_i64().is_some_and(|n| n != 0)
                || number.as_u64().is_some_and(|n| n != 0)
                || number.as_f64().is_some_and(|n| n != 0.0)
        }
        Value::String(text) => !text.is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(object) => !object.is_empty(),
    }
}
