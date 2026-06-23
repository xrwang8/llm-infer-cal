use std::collections::HashMap;

use llm_infer_cal_core::architecture::formulas::kv_cache::{
    compute_kv_cache_bits, compute_kv_cache_bytes,
};
use llm_infer_cal_core::architecture::formulas::weight::{
    estimate_active_params, estimate_total_params, predicted_bytes_under_quant,
};
use llm_infer_cal_core::architecture::profile::{
    ArchitectureProfile, AttentionTraits, AttentionVariant, Confidence, Family, MoeTraits,
};
use llm_infer_cal_core::output::labels::Label;

fn base_profile(attention: Option<AttentionTraits>) -> ArchitectureProfile {
    ArchitectureProfile {
        model_type: "llama".to_string(),
        architectures: vec!["llamaforcausallm".to_string()],
        family: Family::Transformer,
        num_hidden_layers: 80,
        hidden_size: 8192,
        vocab_size: 128_256,
        confidence: Confidence::High,
        attention,
        moe: None,
        position: None,
        sliding_window: None,
        intermediate_size: None,
        tie_word_embeddings: false,
        auxiliary: HashMap::new(),
    }
}

fn gqa_attention() -> AttentionTraits {
    AttentionTraits {
        variant: AttentionVariant::Gqa,
        num_heads: 64,
        num_kv_heads: 8,
        head_dim: 128,
        q_lora_rank: None,
        kv_lora_rank: None,
        qk_rope_head_dim: None,
        compress_ratios: None,
        nsa_topk: None,
    }
}

#[test]
fn standard_gqa_kv_formula_matches_rust_contract() {
    let profile = base_profile(Some(gqa_attention()));

    let kv = compute_kv_cache_bytes(&profile, 2048, 2);

    let expected = 2_u64 * 8 * 128 * 2 * 2048 * 80;
    assert_eq!(kv.value, expected);
    assert_eq!(kv.label, Label::Estimated);
}

#[test]
fn kv_cache_bits_precision_scales_cache_size() {
    let profile = base_profile(Some(gqa_attention()));

    let fp16 = compute_kv_cache_bits(&profile, 2048, 16);
    let fp8 = compute_kv_cache_bits(&profile, 2048, 8);
    let int4 = compute_kv_cache_bits(&profile, 2048, 4);

    assert_eq!(fp16.value, compute_kv_cache_bytes(&profile, 2048, 2).value);
    assert_eq!(fp8.value, fp16.value / 2);
    assert_eq!(int4.value, fp16.value / 4);
    assert!(fp8.source.as_deref().unwrap_or("").contains("8b"));
}

#[test]
fn sliding_window_caps_standard_attention_seq_len() {
    let mut profile = base_profile(Some(gqa_attention()));
    profile.sliding_window = Some(4096);

    let capped = compute_kv_cache_bytes(&profile, 32_768, 2);
    let same_as_window = compute_kv_cache_bytes(&profile, 4096, 2);

    assert_eq!(capped.value, same_as_window.value);
}

#[test]
fn zero_seq_len_is_zero_and_estimated() {
    let profile = base_profile(Some(gqa_attention()));

    let kv = compute_kv_cache_bytes(&profile, 0, 2);

    assert_eq!(kv.value, 0);
    assert_eq!(kv.label, Label::Estimated);
}

#[test]
fn csa_hca_kv_uses_average_compress_ratio() {
    let mut ratios = vec![0, 0];
    for _ in 0..20 {
        ratios.push(4);
        ratios.push(128);
    }
    ratios.push(4);
    ratios.push(0);

    let mut profile = base_profile(Some(AttentionTraits {
        variant: AttentionVariant::CsaHca,
        num_heads: 128,
        num_kv_heads: 1,
        head_dim: 512,
        q_lora_rank: None,
        kv_lora_rank: None,
        qk_rope_head_dim: None,
        compress_ratios: Some(ratios.clone()),
        nsa_topk: None,
    }));
    profile.model_type = "deepseek_v4".to_string();
    profile.num_hidden_layers = 43;
    profile.sliding_window = Some(4096);

    let kv = compute_kv_cache_bytes(&profile, 128_000, 2);

    let num_kv_heads = 1_u64;
    let baseline = 2_u64 * num_kv_heads * 512 * 2 * 128_000 * 43;
    let avg_ratio = ratios
        .iter()
        .map(|&r| if r == 0 { 1.0 } else { 1.0 / r as f64 })
        .sum::<f64>()
        / ratios.len() as f64;
    let expected = (baseline as f64 * avg_ratio) as u64;
    assert_eq!(ratios.len(), 44);
    assert_eq!(kv.value, expected);
    assert_eq!(kv.label, Label::Estimated);
}

#[test]
fn csa_hca_scales_linearly_with_context() {
    let mut ratios = vec![0, 0];
    for _ in 0..20 {
        ratios.push(4);
        ratios.push(128);
    }
    ratios.push(4);
    ratios.push(0);

    let mut profile = base_profile(Some(AttentionTraits {
        variant: AttentionVariant::CsaHca,
        num_heads: 128,
        num_kv_heads: 1,
        head_dim: 512,
        q_lora_rank: None,
        kv_lora_rank: None,
        qk_rope_head_dim: None,
        compress_ratios: Some(ratios),
        nsa_topk: None,
    }));
    profile.num_hidden_layers = 43;
    profile.sliding_window = Some(4096);

    let kv_32k = compute_kv_cache_bytes(&profile, 32_000, 2);
    let kv_128k = compute_kv_cache_bytes(&profile, 128_000, 2);

    assert_eq!(kv_128k.value / kv_32k.value, 4);
}

#[test]
fn mla_uses_kv_lora_rank() {
    let mut profile = base_profile(Some(AttentionTraits {
        variant: AttentionVariant::Mla,
        num_heads: 128,
        num_kv_heads: 128,
        head_dim: 128,
        q_lora_rank: Some(1536),
        kv_lora_rank: Some(512),
        qk_rope_head_dim: None,
        compress_ratios: None,
        nsa_topk: None,
    }));
    profile.model_type = "deepseek_v2".to_string();
    profile.num_hidden_layers = 60;
    profile.hidden_size = 5120;
    profile.vocab_size = 102_400;

    let kv = compute_kv_cache_bytes(&profile, 8192, 2);

    assert_eq!(kv.value, 512_u64 * 2 * 8192 * 60);
}

#[test]
fn mla_kv_includes_decoupled_rope_key_dim() {
    let mut profile = base_profile(Some(AttentionTraits {
        variant: AttentionVariant::Mla,
        num_heads: 64,
        num_kv_heads: 64,
        head_dim: 192,
        q_lora_rank: Some(2048),
        kv_lora_rank: Some(512),
        qk_rope_head_dim: Some(64),
        compress_ratios: None,
        nsa_topk: None,
    }));
    profile.model_type = "glm_moe_dsa".to_string();
    profile.num_hidden_layers = 78;

    let kv = compute_kv_cache_bytes(&profile, 8192, 2);

    assert_eq!(kv.value, (512_u64 + 64) * 2 * 8192 * 78);
}

#[test]
fn unknown_and_state_space_kv_return_unknown() {
    let mut unknown = base_profile(None);
    unknown.model_type = "mystery".to_string();
    unknown.family = Family::Unknown;
    unknown.confidence = Confidence::Low;
    unknown.num_hidden_layers = 0;
    unknown.hidden_size = 0;
    unknown.vocab_size = 0;

    let unknown_kv = compute_kv_cache_bytes(&unknown, 128_000, 2);
    assert_eq!(unknown_kv.value, 0);
    assert_eq!(unknown_kv.label, Label::Unknown);

    let mut state_space = base_profile(None);
    state_space.model_type = "mamba".to_string();
    state_space.family = Family::StateSpace;

    let state_space_kv = compute_kv_cache_bytes(&state_space, 8192, 2);
    assert_eq!(state_space_kv.value, 0);
    assert_eq!(state_space_kv.label, Label::Unknown);
}

#[test]
fn dense_weight_estimate_uses_rust_formula() {
    let mut profile = base_profile(Some(AttentionTraits {
        variant: AttentionVariant::Gqa,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        q_lora_rank: None,
        kv_lora_rank: None,
        qk_rope_head_dim: None,
        compress_ratios: None,
        nsa_topk: None,
    }));
    profile.num_hidden_layers = 2;
    profile.hidden_size = 16;
    profile.vocab_size = 100;
    profile.intermediate_size = Some(64);

    let params = estimate_total_params(&profile);
    let active = estimate_active_params(&profile);

    assert_eq!(params.value, 12_480);
    assert_eq!(params.label, Label::Estimated);
    assert_eq!(active.value, params.value);
    assert_eq!(active.label, Label::Estimated);
}

#[test]
fn moe_weight_estimate_counts_all_experts() {
    let mut profile = base_profile(None);
    profile.num_hidden_layers = 2;
    profile.hidden_size = 16;
    profile.vocab_size = 100;
    profile.moe = Some(MoeTraits {
        num_routed_experts: 4,
        num_shared_experts: 1,
        num_experts_per_tok: 2,
        moe_intermediate_size: 32,
    });

    let params = estimate_total_params(&profile);
    let active = estimate_active_params(&profile);

    assert_eq!(params.value, 18_752);
    assert_eq!(params.label, Label::Estimated);
    assert_eq!(active.value, 12_608);
    assert!(active.value < params.value);
    assert!(active.source.as_deref().unwrap_or("").contains("active"));
}

#[test]
fn predicted_bytes_under_quant_matches_rust_contract_bpp_table() {
    let fp16 = predicted_bytes_under_quant(70_000_000_000, "FP16");
    let mixed = predicted_bytes_under_quant(284_000_000_000, "FP4_FP8_MIXED");
    let unknown = predicted_bytes_under_quant(1_000_000, "UNKNOWN");

    assert_eq!(fp16.value, 140_000_000_000);
    assert_eq!(mixed.value, 156_200_000_000);
    assert_eq!(unknown.value, 0);
    assert_eq!(unknown.label, Label::Unknown);
}
