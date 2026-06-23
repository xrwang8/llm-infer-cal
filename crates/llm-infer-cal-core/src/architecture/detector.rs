use std::collections::HashMap;

use serde_json::Value;

use crate::architecture::profile::{ArchitectureProfile, Confidence, Family};
use crate::architecture::traits::{
    detect_attention, detect_moe, detect_position, detect_sliding_window, get, get_truthy_u64,
    get_u64, has_key, truthy,
};

const KNOWN_MODEL_TYPES: &[&str] = &[
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v3_2",
    "deepseek_v4",
    "gemma",
    "gemma2",
    "gemma3",
    "phi",
    "phi3",
];

const STATE_SPACE_TYPES: &[&str] = &["mamba", "mamba2", "falcon_mamba", "jamba"];

pub fn detect(config: &Value) -> ArchitectureProfile {
    let model_type = model_type(config);
    let architectures = architectures(config);

    if STATE_SPACE_TYPES.contains(&model_type.as_str()) || has_key(config, "ssm_cfg") {
        let mut auxiliary = HashMap::new();
        auxiliary.insert("v0_1_unsupported".to_string(), Value::Bool(true));
        return ArchitectureProfile {
            model_type,
            architectures,
            family: Family::StateSpace,
            num_hidden_layers: get_u64(config, "num_hidden_layers").unwrap_or(0),
            hidden_size: get_u64(config, "hidden_size").unwrap_or(0),
            vocab_size: get_u64(config, "vocab_size").unwrap_or(0),
            confidence: Confidence::High,
            auxiliary,
            ..ArchitectureProfile::default()
        };
    }

    if model_type.is_empty() && architectures.is_empty() {
        return fallback_unknown(config);
    }

    let Some(num_hidden_layers) = get_truthy_u64(config, "num_hidden_layers") else {
        return fallback_unknown(config);
    };
    let Some(hidden_size) = get_truthy_u64(config, "hidden_size") else {
        return fallback_unknown(config);
    };

    let attention = detect_attention(config);
    let moe = detect_moe(config);
    let position = detect_position(config);
    let sliding_window = detect_sliding_window(config);
    let confidence = if KNOWN_MODEL_TYPES.contains(&model_type.as_str()) {
        Confidence::High
    } else {
        Confidence::Medium
    };

    let mut auxiliary = HashMap::new();
    let intermediate_size = get(config, "intermediate_size")
        .filter(|value| value.is_i64() || value.is_u64())
        .and_then(crate::architecture::traits::value_to_u64);
    if let Some(intermediate_size) = intermediate_size {
        auxiliary.insert(
            "intermediate_size".to_string(),
            Value::from(intermediate_size),
        );
    }

    let tie_word_embeddings = get(config, "tie_word_embeddings").map(bool_from_value);
    if let Some(tied) = tie_word_embeddings {
        auxiliary.insert("tie_word_embeddings".to_string(), Value::Bool(tied));
    }

    ArchitectureProfile {
        model_type,
        architectures,
        family: Family::Transformer,
        num_hidden_layers,
        hidden_size,
        vocab_size: get_u64(config, "vocab_size").unwrap_or(0),
        confidence,
        attention: Some(attention),
        moe,
        position: Some(position),
        sliding_window,
        intermediate_size,
        tie_word_embeddings: tie_word_embeddings.unwrap_or(false),
        auxiliary,
    }
}

pub fn fallback_unknown(config: &Value) -> ArchitectureProfile {
    let mut auxiliary = HashMap::new();
    auxiliary.insert(
        "warning".to_string(),
        Value::String(
            "No recognizable model_type or missing essential config fields. Weight estimate from safetensors file size only; KV cache cannot be estimated; engine compatibility unknown."
                .to_string(),
        ),
    );

    ArchitectureProfile {
        model_type: model_type(config),
        architectures: architectures(config),
        family: Family::Unknown,
        num_hidden_layers: get_u64(config, "num_hidden_layers").unwrap_or(0),
        hidden_size: get_u64(config, "hidden_size").unwrap_or(0),
        vocab_size: get_u64(config, "vocab_size").unwrap_or(0),
        confidence: Confidence::Low,
        auxiliary,
        ..ArchitectureProfile::default()
    }
}

fn model_type(config: &Value) -> String {
    get(config, "model_type")
        .map(stringify_value)
        .unwrap_or_default()
        .to_lowercase()
}

fn architectures(config: &Value) -> Vec<String> {
    get(config, "architectures")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|value| stringify_value(value).to_lowercase())
                .collect()
        })
        .unwrap_or_default()
}

fn stringify_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => "None".to_string(),
        _ => value.to_string(),
    }
}

fn bool_from_value(value: &Value) -> bool {
    truthy(value)
}
