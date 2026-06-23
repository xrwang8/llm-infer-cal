use std::collections::HashMap;

use llm_infer_cal_core::model_source::base::SiblingFile;
use llm_infer_cal_core::output::labels::Label;
use llm_infer_cal_core::weight_analyzer::fingerprint::{
    from_config, from_safetensors_dtypes, QuantFingerprint, SourceType,
};
use llm_infer_cal_core::weight_analyzer::reconciler::reconcile;
use llm_infer_cal_core::weight_analyzer::{analyze, QuantizationScheme};
use serde_json::json;

fn sibling(filename: &str, size: u64) -> SiblingFile {
    SiblingFile {
        filename: filename.to_string(),
        size: Some(size),
    }
}

fn dtypes(entries: &[(&str, &str)]) -> HashMap<String, String> {
    entries
        .iter()
        .map(|(name, dtype)| ((*name).to_string(), (*dtype).to_string()))
        .collect()
}

#[test]
fn analyze_sums_only_safetensors_with_verified_label() {
    let siblings = vec![
        sibling("model-00001-of-00002.safetensors", 100),
        sibling("model-00002-of-00002.safetensors", 200),
        sibling("pytorch_model.bin", 500),
        sibling("tokenizer.json", 5),
    ];

    let report = analyze(&siblings, Some(1000), None);

    assert_eq!(report.total_bytes.value, 300);
    assert_eq!(report.total_bytes.label, Label::Verified);
    assert_eq!(report.bits_per_param.unwrap().value, 2.4);
}

#[test]
fn analyze_skips_quant_guess_when_params_or_weights_missing() {
    let report = analyze(&[sibling("model.safetensors", 0)], Some(0), None);

    assert_eq!(report.total_bytes.value, 0);
    assert!(report.bits_per_param.is_none());
    assert_eq!(report.quantization_guess.value, QuantizationScheme::Unknown);
    assert_eq!(report.quantization_guess.label, Label::Unknown);
}

#[test]
fn analyze_uses_fingerprint_as_verified_quant_guess() {
    let fingerprint = QuantFingerprint {
        scheme: QuantizationScheme::AwqInt4,
        source_type: SourceType::SafetensorsHeader,
        evidence: "safetensors header has AWQ marker".to_string(),
    };

    let report = analyze(
        &[sibling("model.safetensors", 550)],
        Some(1000),
        Some(&fingerprint),
    );

    assert_eq!(report.quantization_guess.value, QuantizationScheme::AwqInt4);
    assert_eq!(report.quantization_guess.label, Label::Verified);
    assert_eq!(
        report.quantization_guess.source.as_deref(),
        Some("safetensors header has AWQ marker")
    );
}

#[test]
fn config_fingerprint_maps_common_quantization_configs() {
    let gptq =
        from_config(&json!({"quantization_config": {"quant_method": "gptq", "bits": 4}})).unwrap();
    let compressed_fp8 = from_config(&json!({
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float"}}}
        }
    }))
    .unwrap();
    let bnb = from_config(&json!({
        "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": true}
    }))
    .unwrap();
    let root_bf16 = from_config(&json!({"torch_dtype": "bfloat16"})).unwrap();
    let nested_bf16 = from_config(&json!({
        "model_type": "qwen3_5_moe",
        "text_config": {"dtype": "bfloat16"}
    }))
    .unwrap();

    assert_eq!(gptq.scheme, QuantizationScheme::GptqInt4);
    assert_eq!(gptq.source_type, SourceType::ConfigJson);
    assert!(gptq.evidence.contains("gptq"));
    assert_eq!(compressed_fp8.scheme, QuantizationScheme::Fp8);
    assert_eq!(bnb.scheme, QuantizationScheme::Int4);
    assert_eq!(root_bf16.scheme, QuantizationScheme::Bf16);
    assert_eq!(nested_bf16.scheme, QuantizationScheme::Bf16);
    assert!(nested_bf16.evidence.contains("text_config.dtype=bfloat16"));
}

#[test]
fn safetensors_fingerprint_detects_packed_int4_markers() {
    let gptq = from_safetensors_dtypes(&dtypes(&[
        ("model.layers.0.self_attn.q_proj.qweight", "I32"),
        ("model.layers.0.self_attn.q_proj.qzeros", "I32"),
        ("model.layers.0.self_attn.q_proj.scales", "F16"),
        ("model.layers.0.self_attn.q_proj.g_idx", "I32"),
    ]))
    .unwrap();
    let awq = from_safetensors_dtypes(&dtypes(&[
        ("model.layers.0.self_attn.q_proj.qweight", "I32"),
        ("model.layers.0.self_attn.q_proj.qzeros", "I32"),
        ("model.layers.0.self_attn.q_proj.scales", "F16"),
    ]))
    .unwrap();

    assert_eq!(gptq.scheme, QuantizationScheme::GptqInt4);
    assert!(gptq.evidence.contains("g_idx"));
    assert_eq!(awq.scheme, QuantizationScheme::AwqInt4);
}

#[test]
fn safetensors_fingerprint_detects_deepseek_mx_mixed_pack() {
    let fp = from_safetensors_dtypes(&dtypes(&[
        ("model.layers.5.mlp.experts.0.w1.weight", "I8"),
        ("model.layers.5.mlp.experts.0.w2.weight", "I8"),
        ("model.layers.5.mlp.experts.0.w1.weight_scale", "F8_E8M0"),
        ("model.layers.5.self_attn.q_proj.weight", "F8_E4M3"),
        ("model.layers.5.input_layernorm.weight", "BF16"),
    ]))
    .unwrap();

    assert_eq!(fp.scheme, QuantizationScheme::Fp4Fp8Mixed);
    assert!(fp.evidence.contains("MX"));
}

#[test]
fn safetensors_fingerprint_detects_pure_float_weight_dtypes() {
    let fp8 = from_safetensors_dtypes(&dtypes(&[
        ("model.layers.0.self_attn.q_proj.weight", "F8_E4M3"),
        ("model.layers.0.mlp.gate_proj.weight", "F8_E4M3"),
        ("model.layers.0.input_layernorm.weight", "BF16"),
    ]))
    .unwrap();
    let fp16 = from_safetensors_dtypes(&dtypes(&[
        ("model.layers.0.self_attn.q_proj.weight", "F16"),
        ("model.layers.0.mlp.gate_proj.weight", "F16"),
    ]))
    .unwrap();
    let bf16 = from_safetensors_dtypes(&dtypes(&[
        ("model.layers.0.self_attn.q_proj.weight", "BF16"),
        ("model.layers.0.mlp.gate_proj.weight", "BF16"),
    ]))
    .unwrap();

    assert_eq!(fp8.scheme, QuantizationScheme::Fp8);
    assert_eq!(fp16.scheme, QuantizationScheme::Fp16);
    assert_eq!(bf16.scheme, QuantizationScheme::Bf16);
    assert!(from_safetensors_dtypes(&HashMap::new()).is_none());
}

#[test]
fn reconcile_identifies_deepseek_fp4_fp8_pack_and_surfaces_ties() {
    let report = reconcile(160_300_000_000, 284_000_000_000, None);

    assert_eq!(report.best.value, QuantizationScheme::Fp4Fp8Mixed);
    assert_eq!(report.best.label, Label::Inferred);
    assert_eq!(report.candidates[0].scheme, QuantizationScheme::Fp4Fp8Mixed);
    assert!(report
        .best
        .source
        .as_deref()
        .unwrap_or("")
        .contains("tied with"));
}

#[test]
fn reconcile_edge_cases_return_unknown() {
    let zero_observed = reconcile(0, 1_000_000, None);
    let zero_params = reconcile(1_000_000, 0, None);
    let too_large = reconcile(10_000_000, 1_000_000, None);

    assert_eq!(zero_observed.best.value, QuantizationScheme::Unknown);
    assert_eq!(zero_observed.best.label, Label::Unknown);
    assert_eq!(zero_params.best.value, QuantizationScheme::Unknown);
    assert_eq!(too_large.best.value, QuantizationScheme::Unknown);
}

#[test]
fn reconcile_uses_fingerprint_to_break_ties_and_report_conflicts() {
    let awq = QuantFingerprint {
        scheme: QuantizationScheme::AwqInt4,
        source_type: SourceType::SafetensorsHeader,
        evidence: "safetensors header has .qweight + .qzeros, no .g_idx (AWQ marker)".to_string(),
    };
    let fp8 = QuantFingerprint {
        scheme: QuantizationScheme::Fp8,
        source_type: SourceType::ConfigJson,
        evidence: "config.json quant_method=fp8".to_string(),
    };
    let unknown = QuantFingerprint {
        scheme: QuantizationScheme::Unknown,
        source_type: SourceType::ConfigJson,
        evidence: "declared UNKNOWN".to_string(),
    };

    let awq_report = reconcile(160_300_000_000, 284_000_000_000, Some(&awq));
    let fp8_report = reconcile(160_300_000_000, 284_000_000_000, Some(&fp8));
    let unknown_report = reconcile(160_300_000_000, 284_000_000_000, Some(&unknown));

    assert_eq!(awq_report.best.value, QuantizationScheme::AwqInt4);
    assert_eq!(awq_report.best.label, Label::Verified);
    assert!(awq_report
        .best
        .source
        .as_deref()
        .unwrap_or("")
        .contains("AWQ marker"));
    assert_eq!(fp8_report.best.value, QuantizationScheme::Fp8);
    assert_eq!(fp8_report.best.label, Label::Verified);
    assert!(fp8_report
        .best
        .source
        .as_deref()
        .unwrap_or("")
        .contains("NOTE"));
    assert_eq!(unknown_report.best.value, QuantizationScheme::Fp4Fp8Mixed);
    assert!(unknown_report
        .best
        .source
        .as_deref()
        .unwrap_or("")
        .contains("fell back"));
}
