use std::collections::BTreeMap;

use llm_infer_cal_core::architecture::profile::{
    ArchitectureProfile, AttentionTraits, AttentionVariant, Confidence, Family, MoeTraits,
};
use llm_infer_cal_core::benchmark::runner::{
    evaluate_field, exit_code_from, load_dataset, CheckResult, Expectation, ExpectedValue, Status,
};
use llm_infer_cal_core::core::evaluator::EvaluationReport;
use llm_infer_cal_core::fleet::planner::{FleetOption, FleetRecommendation};
use llm_infer_cal_core::output::labels::{AnnotatedValue, Label};
use llm_infer_cal_core::weight_analyzer::reconciler::ReconciliationReport;
use llm_infer_cal_core::weight_analyzer::{QuantizationScheme, WeightReport};

fn fake_report(
    attention_variant: AttentionVariant,
    quantization: QuantizationScheme,
    weight_bytes: u64,
    is_moe: bool,
    fleet: Option<FleetRecommendation>,
) -> EvaluationReport {
    let moe = is_moe.then_some(MoeTraits {
        num_routed_experts: 256,
        num_shared_experts: 1,
        num_experts_per_tok: 6,
        moe_intermediate_size: 2048,
    });
    EvaluationReport {
        model_id: "fake".to_string(),
        source: "huggingface".to_string(),
        commit_sha: None,
        gpu: "H800".to_string(),
        gpu_spec: None,
        gpu_error: None,
        engine: "vllm".to_string(),
        profile: ArchitectureProfile {
            model_type: "fake".to_string(),
            architectures: Vec::new(),
            family: Family::Transformer,
            num_hidden_layers: 43,
            hidden_size: 4096,
            vocab_size: 129_280,
            confidence: Confidence::High,
            attention: Some(AttentionTraits {
                variant: attention_variant,
                num_heads: 64,
                num_kv_heads: 1,
                head_dim: 512,
                q_lora_rank: None,
                kv_lora_rank: None,
                qk_rope_head_dim: None,
                compress_ratios: None,
                nsa_topk: None,
            }),
            moe,
            ..ArchitectureProfile::default()
        },
        weight: WeightReport {
            total_bytes: AnnotatedValue::new(weight_bytes, Label::Verified, None),
            bits_per_param: Some(AnnotatedValue::new(4.5, Label::Inferred, None)),
            quantization_guess: AnnotatedValue::new(quantization, Label::Inferred, None),
        },
        total_params_estimate: AnnotatedValue::new(284_000_000_000, Label::Estimated, None),
        active_params_estimate: AnnotatedValue::new(37_000_000_000, Label::Estimated, None),
        reconciliation: ReconciliationReport {
            observed_bytes: weight_bytes,
            total_params: 284_000_000_000,
            candidates: Vec::new(),
            best: AnnotatedValue::new(quantization, Label::Inferred, None),
        },
        kv_cache_by_context: BTreeMap::new(),
        activation_by_context: BTreeMap::new(),
        kv_cache_bits: 16,
        paged_attention: false,
        engine_match: None,
        fleet,
        generated_command: None,
        prefill: None,
        decode: None,
        concurrency: None,
        perf_input_tokens: None,
        perf_output_tokens: None,
        perf_target_tokens_per_sec: None,
    }
}

fn prod_fleet(gpu_count: u64) -> FleetRecommendation {
    FleetRecommendation {
        options: vec![FleetOption {
            tier: "prod",
            gpu_count,
            tensor_parallel_size: gpu_count,
            pipeline_parallel_size: 1,
            node_count: 1,
            weight_bytes_per_gpu: 1,
            kv_bytes_per_request: 1,
            kv_bytes_per_request_per_gpu: 1,
            activation_bytes_per_request: 0,
            activation_bytes_per_request_per_gpu: 0,
            kv_reference_context_tokens: 131_072,
            tier_concurrent_requests: 16,
            required_bytes_per_gpu_at_tier: 17,
            max_concurrent_at_reference_ctx: 1,
            max_concurrent_by_context: Vec::new(),
            usable_bytes_per_gpu: 1,
            reserved_bytes_per_gpu: 0,
            fits: true,
            reason_en: "test".to_string(),
            reason_zh: "test".to_string(),
        }],
        best_tier: Some("prod"),
        valid_tp_sizes: vec![1, 2, 4, 8],
        constraint_note_en: String::new(),
        constraint_note_zh: String::new(),
    }
}

#[test]
fn evaluate_field_matches_rust_contract_pass_fail_skip_rules() {
    let report = fake_report(
        AttentionVariant::CsaHca,
        QuantizationScheme::Fp4Fp8Mixed,
        160_000_000_000,
        true,
        Some(prod_fleet(8)),
    );

    let (predicted, status) = evaluate_field(
        &report,
        &Expectation::expected(
            "attention_variant",
            ExpectedValue::String("CSA_HCA".to_string()),
            "test",
        ),
    );
    assert_eq!(predicted, "CSA_HCA");
    assert_eq!(status, Status::Pass);

    let (_, status) = evaluate_field(
        &report,
        &Expectation::expected(
            "quantization",
            ExpectedValue::String("FP4_FP8_MIXED".to_string()),
            "test",
        ),
    );
    assert_eq!(status, Status::Pass);

    let (_, status) = evaluate_field(
        &report,
        &Expectation::range(
            "weight_bytes",
            Some(150_000_000_000),
            Some(170_000_000_000),
            "test",
        ),
    );
    assert_eq!(status, Status::Pass);

    let (_, status) = evaluate_field(
        &report,
        &Expectation::range("weight_bytes", Some(1), Some(2), "test"),
    );
    assert_eq!(status, Status::Fail);

    let (_, status) = evaluate_field(
        &report,
        &Expectation::expected("is_moe", ExpectedValue::Bool(true), "test"),
    );
    assert_eq!(status, Status::Pass);

    let (predicted, status) = evaluate_field(
        &report,
        &Expectation::expected(
            "nonexistent_field",
            ExpectedValue::String("x".to_string()),
            "test",
        ),
    );
    assert_eq!(predicted, "(unknown field)");
    assert_eq!(status, Status::Skip);
}

#[test]
fn evaluate_field_handles_fleet_checks_like_rust_contract() {
    let no_fleet = fake_report(
        AttentionVariant::Gqa,
        QuantizationScheme::Fp16,
        10,
        false,
        None,
    );
    let prod_4 = fake_report(
        AttentionVariant::Gqa,
        QuantizationScheme::Fp16,
        10,
        false,
        Some(prod_fleet(4)),
    );

    let (predicted, status) = evaluate_field(
        &no_fleet,
        &Expectation::expected("fleet_prod_gpus", ExpectedValue::Int(4), "test"),
    );
    assert_eq!(predicted, "(no fleet)");
    assert_eq!(status, Status::Skip);

    let (predicted, status) = evaluate_field(
        &prod_4,
        &Expectation::expected("fleet_prod_gpus", ExpectedValue::Int(4), "test"),
    );
    assert_eq!(predicted, "4");
    assert_eq!(status, Status::Pass);

    let (predicted, status) = evaluate_field(
        &prod_4,
        &Expectation::expected("fleet_prod_gpus_at_most", ExpectedValue::Int(8), "test"),
    );
    assert_eq!(predicted, "4 (max 8)");
    assert_eq!(status, Status::Pass);
}

#[test]
fn bundled_dataset_loads_and_cites_every_expectation() {
    let dataset = load_dataset().unwrap();

    assert_eq!(dataset.schema_version, 1);
    assert!(dataset.entries.len() >= 4);
    assert!(dataset
        .entries
        .iter()
        .any(|entry| entry.model_id.contains("DeepSeek-V4-Flash")
            && entry
                .expectations
                .iter()
                .any(|expectation| expectation.field == "quantization")
            && entry
                .expectations
                .iter()
                .any(|expectation| expectation.field == "attention_variant")));
    assert!(dataset.entries.iter().all(|entry| entry
        .expectations
        .iter()
        .all(|expectation| !expectation.source.is_empty())));
}

#[test]
fn exit_code_matches_rust_contract_rules() {
    assert_eq!(
        exit_code_from(&[
            CheckResult::new("a", "f1", Status::Pass, "", "", "src"),
            CheckResult::new("a", "f2", Status::Pass, "", "", "src"),
        ]),
        0
    );
    assert_eq!(
        exit_code_from(&[CheckResult::new("a", "f1", Status::Skip, "", "", "src")]),
        0
    );
    assert_eq!(
        exit_code_from(&[
            CheckResult::new("a", "f1", Status::Pass, "", "", "src"),
            CheckResult::new("a", "f2", Status::Fail, "", "", "src"),
        ]),
        1
    );
}
