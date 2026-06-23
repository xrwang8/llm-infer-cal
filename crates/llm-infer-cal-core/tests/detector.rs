use llm_infer_cal_core::architecture::detector::{detect, fallback_unknown};
use llm_infer_cal_core::architecture::profile::{AttentionVariant, Confidence, Family};
use llm_infer_cal_core::architecture::traits::{
    detect_attention, detect_moe, detect_sliding_window,
};
use serde_json::{json, Value};

fn load_config(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests")
        .join("fixtures")
        .join("configs")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(path).expect("fixture must exist");
    serde_json::from_str(&text).expect("fixture must be valid JSON")
}

#[test]
fn deepseek_v4_flash_traits_stack_like_python() {
    let profile = detect(&load_config("deepseek_v4_flash"));

    assert_eq!(profile.family, Family::Transformer);
    assert_eq!(profile.confidence, Confidence::High);
    let attention = profile.attention.unwrap();
    assert_eq!(attention.variant, AttentionVariant::CsaHca);
    assert_eq!(attention.compress_ratios.unwrap().len(), 44);
    let moe = profile.moe.unwrap();
    assert_eq!(moe.num_routed_experts, 256);
    assert_eq!(moe.num_shared_experts, 1);
    assert_eq!(moe.num_experts_per_tok, 6);
    assert_eq!(profile.sliding_window, Some(128));
}

#[test]
fn llama_and_mistral_detection_match_python() {
    let llama = detect(&load_config("llama3_70b"));
    let mistral = detect(&load_config("mistral_sliding"));

    let attention = llama.attention.unwrap();
    assert_eq!(llama.family, Family::Transformer);
    assert_eq!(attention.variant, AttentionVariant::Gqa);
    assert_eq!(attention.num_kv_heads, 8);
    assert_eq!(attention.num_heads, 64);
    assert!(llama.moe.is_none());
    assert_eq!(llama.sliding_window, None);
    assert_eq!(mistral.sliding_window, Some(4096));
}

#[test]
fn state_space_and_unknown_fallback_match_python() {
    let mamba = detect(&load_config("mamba"));
    let unknown = detect(&load_config("unknown_model"));
    let empty = detect(&json!({}));

    assert_eq!(mamba.family, Family::StateSpace);
    assert_eq!(mamba.auxiliary.get("v0_1_unsupported"), Some(&json!(true)));
    assert_eq!(unknown.family, Family::Unknown);
    assert_eq!(unknown.confidence, Confidence::Low);
    assert!(unknown.auxiliary.contains_key("warning"));
    assert_eq!(empty.family, Family::Unknown);
}

#[test]
fn future_complete_config_is_medium_confidence_transformer() {
    let profile = detect(&json!({
        "model_type": "hypothetical_v1",
        "architectures": ["HypotheticalForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000
    }));

    assert_eq!(profile.family, Family::Transformer);
    assert_eq!(profile.confidence, Confidence::Medium);
}

#[test]
fn csa_hca_length_rules_match_python() {
    let mismatch = detect(&json!({
        "model_type": "hypothetical",
        "architectures": ["Hypothetical"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "compress_ratios": [4, 128, 4]
    }));
    let mtp_extra = detect(&json!({
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "vocab_size": 129280,
        "compress_ratios": vec![0; 44],
        "q_lora_rank": 1024
    }));

    assert_eq!(mismatch.attention.unwrap().variant, AttentionVariant::Gqa);
    assert_eq!(
        mtp_extra.attention.unwrap().variant,
        AttentionVariant::CsaHca
    );
}

#[test]
fn attention_ordering_and_direct_trait_detectors_match_python() {
    let mla = detect(&json!({
        "model_type": "deepseek_v2",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 1,
        "q_lora_rank": 1024,
        "vocab_size": 10
    }));
    let base = json!({
        "model_type": "foo",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 10
    });
    let mut mqa = base.clone();
    mqa["num_key_value_heads"] = json!(1);
    let mut gqa = base.clone();
    gqa["num_key_value_heads"] = json!(8);
    let mut mha = base;
    mha["num_key_value_heads"] = json!(32);

    assert_eq!(mla.attention.unwrap().variant, AttentionVariant::Mla);
    assert_eq!(detect_attention(&mqa).variant, AttentionVariant::Mqa);
    assert_eq!(detect_attention(&gqa).variant, AttentionVariant::Gqa);
    assert_eq!(detect_attention(&mha).variant, AttentionVariant::Mha);
    assert!(detect_moe(&load_config("llama3_70b")).is_none());
    assert_eq!(
        detect_moe(&load_config("deepseek_v4_flash"))
            .unwrap()
            .num_routed_experts,
        256
    );
    assert_eq!(detect_sliding_window(&json!({"sliding_window": 0})), None);
}

#[test]
fn fallback_unknown_preserves_basic_shape() {
    let profile = fallback_unknown(&json!({}));

    assert_eq!(profile.family, Family::Unknown);
    assert_eq!(profile.confidence, Confidence::Low);
    assert_eq!(profile.num_hidden_layers, 0);
}
