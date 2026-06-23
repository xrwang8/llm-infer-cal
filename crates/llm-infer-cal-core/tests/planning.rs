use llm_infer_cal_core::architecture::profile::{
    ArchitectureProfile, AttentionTraits, AttentionVariant, Confidence, Family, MoeTraits,
};
use llm_infer_cal_core::fleet::planner::plan;
use llm_infer_cal_core::hardware::loader::{lookup, GPUSpec};
use llm_infer_cal_core::output::labels::{AnnotatedValue, Label};
use llm_infer_cal_core::performance::compute::{
    estimate_decode, nvlink_efficiency, DecodeEstimate,
};
use llm_infer_cal_core::performance::concurrency::{analyze, Bottleneck};

fn gpu(nvlink: u64) -> GPUSpec {
    GPUSpec {
        id: format!("test-nvlink-{nvlink}"),
        aliases: Vec::new(),
        memory_gb: 80,
        nvlink_bandwidth_gbps: nvlink,
        memory_bandwidth_gbps: Some(3350),
        fp16_tflops: 989.0,
        fp8_support: true,
        fp4_support: false,
        notes_en: None,
        notes_zh: None,
        spec_source: Some("test".to_string()),
    }
}

fn deepseek_v4_profile() -> ArchitectureProfile {
    ArchitectureProfile {
        model_type: "deepseek_v4".to_string(),
        architectures: vec!["deepseekv4forcausallm".to_string()],
        family: Family::Transformer,
        num_hidden_layers: 43,
        hidden_size: 4096,
        vocab_size: 129_280,
        confidence: Confidence::High,
        attention: Some(AttentionTraits {
            variant: AttentionVariant::CsaHca,
            num_heads: 64,
            num_kv_heads: 1,
            head_dim: 512,
            q_lora_rank: None,
            kv_lora_rank: None,
            compress_ratios: Some([vec![0], vec![4; 42], vec![0]].concat()),
            nsa_topk: None,
        }),
        moe: Some(MoeTraits {
            num_routed_experts: 256,
            num_shared_experts: 1,
            num_experts_per_tok: 6,
            moe_intermediate_size: 2048,
        }),
        sliding_window: Some(128),
        ..ArchitectureProfile::default()
    }
}

fn llama_profile() -> ArchitectureProfile {
    ArchitectureProfile {
        model_type: "llama".to_string(),
        architectures: vec!["llamaforcausallm".to_string()],
        family: Family::Transformer,
        num_hidden_layers: 80,
        hidden_size: 8192,
        vocab_size: 128_256,
        confidence: Confidence::High,
        attention: Some(AttentionTraits {
            variant: AttentionVariant::Gqa,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            q_lora_rank: None,
            kv_lora_rank: None,
            compress_ratios: None,
            nsa_topk: None,
        }),
        ..ArchitectureProfile::default()
    }
}

#[test]
fn hardware_lookup_matches_python_aliases_and_helpful_errors() {
    let h800 = lookup("H800").unwrap();
    let h800_alias = lookup("h800-sxm5").unwrap();
    let b200 = lookup("B200").unwrap();
    let cn = lookup("曦云C500").unwrap();
    let legacy = lookup("H800x8").unwrap_err();

    assert_eq!(h800.id, "H800");
    assert_eq!(h800.memory_gb, 80);
    assert_eq!(h800.nvlink_bandwidth_gbps, 400);
    assert_eq!(h800_alias.id, "H800");
    assert!(b200.fp4_support);
    assert_eq!(cn.id, "MXC500");
    assert!(legacy.to_string().contains("--gpu-count 8"));
}

#[test]
fn nvlink_efficiency_and_decode_penalty_match_python() {
    assert_eq!(nvlink_efficiency(&gpu(0), 1), 1.0);
    assert_eq!(nvlink_efficiency(&gpu(900), 8), 1.0);
    assert_eq!(nvlink_efficiency(&gpu(0), 8), 0.80);
    let h800_eff = nvlink_efficiency(&gpu(400), 8);
    assert!(h800_eff > 0.91 && h800_eff < 0.92);

    let profile = llama_profile();
    let weight_bytes = 80 * 1024_u64.pow(3);
    let h100 = estimate_decode(&profile, weight_bytes, &gpu(900), 8, 0.50, 0.90, None);
    let h800 = estimate_decode(&profile, weight_bytes, &gpu(400), 8, 0.50, 0.90, None);

    assert!(h100.cluster_tokens_per_sec.value > h800.cluster_tokens_per_sec.value);
    let ratio = h800.cluster_tokens_per_sec.value / h100.cluster_tokens_per_sec.value;
    assert!(ratio > 0.91 && ratio < 0.92);
    assert_eq!(
        h100.per_gpu_tokens_per_sec.value,
        h800.per_gpu_tokens_per_sec.value
    );
    assert!(h800
        .cluster_tokens_per_sec
        .source
        .as_deref()
        .unwrap_or("")
        .contains("NVLink"));
}

#[test]
fn fleet_planner_respects_tp_divisibility_and_context_concurrency() {
    let gpu = lookup("H800").unwrap();
    let rec = plan(
        &deepseek_v4_profile(),
        160_000_000_000,
        2_200_000_000,
        &gpu,
        None,
        &[(131_072, 2_200_000_000), (1_048_576, 17_640_000_000)],
    );

    assert_eq!(rec.valid_tp_sizes, vec![1, 2, 4, 8]);
    assert_eq!(
        rec.options
            .iter()
            .map(|option| option.tier)
            .collect::<Vec<_>>(),
        vec!["min", "dev", "prod"]
    );
    assert!(rec.options.iter().all(|option| 64 % option.gpu_count == 0));
    let prod = rec
        .options
        .iter()
        .find(|option| option.tier == "prod")
        .unwrap();
    assert_eq!(prod.gpu_count, 8);
    assert!(prod.fits);
    let ctx_128k = prod
        .max_concurrent_by_context
        .iter()
        .find(|(ctx, _)| *ctx == 131_072)
        .unwrap()
        .1;
    let ctx_1m = prod
        .max_concurrent_by_context
        .iter()
        .find(|(ctx, _)| *ctx == 1_048_576)
        .unwrap()
        .1;
    assert!(ctx_1m >= 1);
    assert!(ctx_128k > ctx_1m);
    assert!(rec.constraint_note_zh.contains("单节点（≤8 卡）候选"));
}

#[test]
fn fleet_planner_flags_forced_invalid_count_and_shards_gqa_kv() {
    let h800 = lookup("H800").unwrap();
    let forced = plan(
        &deepseek_v4_profile(),
        160_000_000_000,
        2_200_000_000,
        &h800,
        Some(3),
        &[(131_072, 2_200_000_000)],
    );
    assert_eq!(forced.options.len(), 1);
    assert!(forced.options[0]
        .reason_en
        .to_lowercase()
        .contains("divide"));

    let h100 = lookup("H100").unwrap();
    let rec = plan(
        &llama_profile(),
        140_000_000_000,
        10_000_000_000,
        &h100,
        None,
        &[(131_072, 10_000_000_000)],
    );
    let prod = rec
        .options
        .iter()
        .find(|option| option.tier == "prod")
        .unwrap();
    let ctx_128k = prod.max_concurrent_by_context[0].1;
    assert!(ctx_128k > 30);
}

#[test]
fn concurrency_analysis_picks_tighter_bound() {
    let decode = DecodeEstimate {
        active_weight_bytes_per_gpu: AnnotatedValue::new(1, Label::Estimated, None),
        per_gpu_tokens_per_sec: AnnotatedValue::new(50.0, Label::Estimated, None),
        cluster_tokens_per_sec: AnnotatedValue::new(100.0, Label::Estimated, None),
        bw_utilization: 0.5,
        cluster_comm_efficiency: 0.9,
        moe_active_weight_bytes_per_gpu: None,
        moe_active_tokens_per_sec: None,
    };

    let memory_bound = analyze(10_000, 1_000, &decode, 1.0, 1.0);
    let bandwidth_bound = analyze(1_000_000, 1_000, &decode, 25.0, 1.0);

    assert_eq!(memory_bound.max_concurrent.value, 10);
    assert_eq!(memory_bound.bottleneck, Bottleneck::MemoryCapacity);
    assert!(memory_bound.bottleneck_reason_zh.contains("≤"));
    assert!(memory_bound.bottleneck_reason_zh.contains("→"));
    assert_eq!(bandwidth_bound.max_concurrent.value, 4);
    assert_eq!(bandwidth_bound.bottleneck, Bottleneck::MemoryBandwidth);
    assert!(bandwidth_bound
        .bottleneck_reason_zh
        .contains("→ 带宽/算力瓶颈"));
}
