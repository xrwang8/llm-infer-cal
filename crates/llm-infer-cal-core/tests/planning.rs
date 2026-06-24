use llm_infer_cal_core::architecture::profile::{
    ArchitectureProfile, AttentionTraits, AttentionVariant, Confidence, Family, MoeTraits,
};
use llm_infer_cal_core::fleet::planner::{
    kv_shards, plan, plan_with_activation, plan_with_memory_options, FleetMemoryOptions,
};
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

fn small_gpu() -> GPUSpec {
    GPUSpec {
        id: "small-16gb".to_string(),
        aliases: Vec::new(),
        memory_gb: 16,
        nvlink_bandwidth_gbps: 0,
        memory_bandwidth_gbps: Some(300),
        fp16_tflops: 65.0,
        fp8_support: false,
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
            qk_rope_head_dim: None,
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

fn deepseek_v4_pro_profile() -> ArchitectureProfile {
    ArchitectureProfile {
        model_type: "deepseek_v4".to_string(),
        architectures: vec!["deepseekv4forcausallm".to_string()],
        family: Family::Transformer,
        num_hidden_layers: 61,
        hidden_size: 7168,
        vocab_size: 129_280,
        confidence: Confidence::High,
        attention: Some(AttentionTraits {
            variant: AttentionVariant::CsaHca,
            num_heads: 128,
            num_kv_heads: 1,
            head_dim: 512,
            q_lora_rank: Some(1536),
            kv_lora_rank: None,
            qk_rope_head_dim: Some(64),
            compress_ratios: Some([vec![128, 128], [4, 128].repeat(29), vec![4, 128]].concat()),
            nsa_topk: None,
        }),
        moe: Some(MoeTraits {
            num_routed_experts: 384,
            num_shared_experts: 1,
            num_experts_per_tok: 6,
            moe_intermediate_size: 3072,
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
            qk_rope_head_dim: None,
            compress_ratios: None,
            nsa_topk: None,
        }),
        ..ArchitectureProfile::default()
    }
}

fn glm52_profile() -> ArchitectureProfile {
    ArchitectureProfile {
        model_type: "glm_moe_dsa".to_string(),
        architectures: vec!["glmmoedsaforcausallm".to_string()],
        family: Family::Transformer,
        num_hidden_layers: 78,
        hidden_size: 6144,
        vocab_size: 154_880,
        confidence: Confidence::High,
        attention: Some(AttentionTraits {
            variant: AttentionVariant::Mla,
            num_heads: 64,
            num_kv_heads: 64,
            head_dim: 192,
            q_lora_rank: Some(2048),
            kv_lora_rank: Some(512),
            qk_rope_head_dim: Some(64),
            compress_ratios: None,
            nsa_topk: None,
        }),
        moe: Some(MoeTraits {
            num_routed_experts: 256,
            num_shared_experts: 1,
            num_experts_per_tok: 8,
            moe_intermediate_size: 2048,
        }),
        ..ArchitectureProfile::default()
    }
}

fn tiny_moe_few_experts_profile() -> ArchitectureProfile {
    ArchitectureProfile {
        model_type: "tiny_moe".to_string(),
        architectures: vec!["tinymoeforcausallm".to_string()],
        family: Family::Transformer,
        num_hidden_layers: 8,
        hidden_size: 4096,
        vocab_size: 4096,
        confidence: Confidence::High,
        attention: Some(AttentionTraits {
            variant: AttentionVariant::Gqa,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 512,
            q_lora_rank: None,
            kv_lora_rank: None,
            qk_rope_head_dim: None,
            compress_ratios: None,
            nsa_topk: None,
        }),
        moe: Some(MoeTraits {
            num_routed_experts: 4,
            num_shared_experts: 1,
            num_experts_per_tok: 2,
            moe_intermediate_size: 2048,
        }),
        ..ArchitectureProfile::default()
    }
}

#[test]
fn hardware_lookup_matches_rust_contract_aliases_and_helpful_errors() {
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
fn nvlink_efficiency_and_decode_penalty_match_rust_contract() {
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
        131_072,
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
        131_072,
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
        131_072,
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
fn fleet_planner_counts_activation_memory_in_required_bytes() {
    let h100 = lookup("H100").unwrap();

    let without_activation = plan(
        &llama_profile(),
        10_000_000_000,
        10_000_000_000,
        131_072,
        &h100,
        Some(1),
        &[(131_072, 10_000_000_000)],
    );
    let with_activation = plan_with_activation(
        &llama_profile(),
        10_000_000_000,
        10_000_000_000,
        5_000_000_000,
        131_072,
        &h100,
        Some(1),
        &[(131_072, 10_000_000_000)],
        &[(131_072, 5_000_000_000)],
    );

    let plain = &without_activation.options[0];
    let activated = &with_activation.options[0];

    assert_eq!(plain.max_concurrent_at_reference_ctx, 6);
    assert_eq!(activated.max_concurrent_at_reference_ctx, 5);
    assert_eq!(activated.tier_concurrent_requests, 1);
    assert_eq!(activated.kv_bytes_per_request_per_gpu, 10_000_000_000);
    assert_eq!(activated.activation_bytes_per_request, 5_000_000_000);
    assert_eq!(
        activated.required_bytes_per_gpu_at_tier,
        10_000_000_000 + 5_000_000_000 + 10_000_000_000
    );
    assert!(activated.required_bytes_per_gpu_at_tier > plain.required_bytes_per_gpu_at_tier);
}

#[test]
fn forced_gpu_count_uses_minimal_launch_concurrency_by_default() {
    let h100 = lookup("H100").unwrap();
    let rec = plan(
        &llama_profile(),
        60_000_000_000,
        2_000_000_000,
        131_072,
        &h100,
        Some(1),
        &[(131_072, 2_000_000_000)],
    );

    let option = &rec.options[0];
    assert_eq!(option.tier, "min");
    assert_eq!(option.gpu_count, 1);
    assert_eq!(option.tier_concurrent_requests, 1);
    assert!(option.fits);
    assert_eq!(rec.best_tier, Some("min"));
}

#[test]
fn fleet_planner_reserves_at_least_three_gb_for_small_gpus() {
    let rec = plan(
        &llama_profile(),
        1_000_000_000,
        1_000_000_000,
        131_072,
        &small_gpu(),
        Some(1),
        &[(131_072, 1_000_000_000)],
    );

    let option = &rec.options[0];
    assert_eq!(option.reserved_bytes_per_gpu, 3_000_000_000);
    assert_eq!(option.usable_bytes_per_gpu, 13_000_000_000);
}

#[test]
fn fleet_planner_does_not_over_shard_moe_experts_beyond_expert_count() {
    let h100 = lookup("H100").unwrap();
    let rec = plan_with_activation(
        &tiny_moe_few_experts_profile(),
        80_000_000_000,
        1_000_000_000,
        0,
        131_072,
        &h100,
        Some(8),
        &[(131_072, 1_000_000_000)],
        &[(131_072, 0)],
    );

    let option = &rec.options[0];
    assert_eq!(option.tensor_parallel_size, 8);
    assert_eq!(option.pipeline_parallel_size, 1);
    assert!(
        option.main_weight_bytes_per_gpu > 10_000_000_000,
        "routed experts can only shard across 4 experts, so TP8 must hold more than naive weight/8"
    );
    assert!(option.main_weight_bytes_per_gpu < 20_000_000_000);
}

#[test]
fn fleet_planner_applies_moe_expert_offload_to_resident_weight() {
    let h100 = lookup("H100").unwrap();
    let profile = tiny_moe_few_experts_profile();
    let baseline = plan_with_memory_options(
        &profile,
        80_000_000_000,
        1_000_000_000,
        0,
        131_072,
        &h100,
        Some(1),
        &[(131_072, 1_000_000_000)],
        &[(131_072, 0)],
        FleetMemoryOptions::default(),
    );
    let offloaded = plan_with_memory_options(
        &profile,
        80_000_000_000,
        1_000_000_000,
        0,
        131_072,
        &h100,
        Some(1),
        &[(131_072, 1_000_000_000)],
        &[(131_072, 0)],
        FleetMemoryOptions {
            expert_offloading: true,
            experts_on_gpu: Some(2),
            ..FleetMemoryOptions::default()
        },
    );

    let baseline_option = &baseline.options[0];
    let offloaded_option = &offloaded.options[0];

    assert!(offloaded_option.main_weight_bytes_per_gpu < baseline_option.main_weight_bytes_per_gpu);
    assert!(offloaded_option.expert_offload_bytes_per_gpu > 0);
    assert_eq!(
        offloaded_option.main_weight_bytes_per_gpu + offloaded_option.expert_offload_bytes_per_gpu,
        baseline_option.main_weight_bytes_per_gpu
    );
    assert_eq!(
        offloaded_option.cpu_offload_bytes_per_gpu,
        offloaded_option.expert_offload_bytes_per_gpu
    );
}

#[test]
fn fleet_planner_applies_target_concurrency_speculative_weights_and_cpu_offload() {
    let h100 = lookup("H100").unwrap();
    let rec = plan_with_memory_options(
        &llama_profile(),
        50_000_000_000,
        5_000_000_000,
        1_000_000_000,
        131_072,
        &h100,
        Some(1),
        &[(131_072, 5_000_000_000)],
        &[(131_072, 1_000_000_000)],
        FleetMemoryOptions {
            target_concurrent_requests: Some(3),
            speculative_weight_bytes: 2_000_000_000,
            cpu_offload_bytes_per_gpu: 4_000_000_000,
            ..FleetMemoryOptions::default()
        },
    );

    let option = &rec.options[0];
    assert_eq!(option.tier, "target");
    assert_eq!(option.tier_concurrent_requests, 3);
    assert_eq!(option.main_weight_bytes_per_gpu, 46_000_000_000);
    assert_eq!(option.cpu_offload_bytes_per_gpu, 4_000_000_000);
    assert_eq!(option.speculative_weight_bytes_per_gpu, 2_000_000_000);
    assert_eq!(option.weight_bytes_per_gpu, 48_000_000_000);
    assert_eq!(
        option.required_bytes_per_gpu_at_tier,
        48_000_000_000 + 1_000_000_000 + 3 * 5_000_000_000
    );
    assert_eq!(option.max_concurrent_at_reference_ctx, 4);
    assert!(option.fits);
    assert_eq!(rec.best_tier, Some("target"));
}

#[test]
fn fleet_planner_has_no_recommendation_when_no_candidate_fits() {
    let h100 = lookup("H100").unwrap();
    let rec = plan(
        &llama_profile(),
        6_000_000_000_000,
        10_000_000_000,
        131_072,
        &h100,
        None,
        &[(131_072, 10_000_000_000), (1_048_576, 80_000_000_000)],
    );

    assert_eq!(rec.valid_tp_sizes, vec![1, 2, 4, 8]);
    assert!(rec.options.iter().all(|option| !option.fits));
    assert_eq!(rec.best_tier, None);
    assert!(rec.best_option().is_none());
}

#[test]
fn fleet_planner_recommends_multinode_for_glm52_bf16_on_h100() {
    let h100 = lookup("H100").unwrap();
    let rec = plan(
        &glm52_profile(),
        1_506_670_000_000,
        11_780_000_000,
        131_072,
        &h100,
        None,
        &[(131_072, 11_780_000_000), (1_048_576, 94_240_000_000)],
    );

    assert_eq!(kv_shards(&glm52_profile(), 8), 1);
    assert_eq!(
        rec.options
            .iter()
            .map(|option| option.gpu_count)
            .collect::<Vec<_>>(),
        vec![24, 48, 48]
    );
    assert_eq!(rec.best_tier, Some("dev"));
    let best = rec.best_option().unwrap();
    assert_eq!(best.tensor_parallel_size, 8);
    assert_eq!(best.pipeline_parallel_size, 6);
    assert_eq!(best.node_count, 6);
    assert!(best.fits);
}

#[test]
fn fleet_planner_recommends_multinode_tp_when_layers_do_not_divide_pp() {
    let h100 = lookup("H100").unwrap();
    let rec = plan_with_activation(
        &deepseek_v4_pro_profile(),
        864_721_029_744,
        541_139_736,
        59_179_008,
        40_960,
        &h100,
        None,
        &[(40_960, 541_139_736)],
        &[(40_960, 59_179_008)],
    );

    assert_eq!(rec.best_tier, Some("dev"));
    let best = rec.best_option().unwrap();
    assert_eq!(best.gpu_count, 16);
    assert_eq!(best.tensor_parallel_size, 16);
    assert_eq!(best.pipeline_parallel_size, 1);
    assert_eq!(best.node_count, 2);
    assert!(best.fits);
    assert!(best.required_bytes_per_gpu_at_tier <= best.usable_bytes_per_gpu);
}

#[test]
fn fleet_planner_prefill_peak_can_prevent_fit() {
    let h100 = lookup("H100").unwrap();
    let profile = llama_profile();
    let weight_bytes = 10_000_000_000;
    let kv_bytes = 100_000_000;
    let activation_bytes = 35_000_000_000;

    let target = plan_with_memory_options(
        &profile,
        weight_bytes,
        kv_bytes,
        activation_bytes,
        131_072,
        &h100,
        Some(1),
        &[(131_072, kv_bytes)],
        &[(131_072, activation_bytes)],
        FleetMemoryOptions {
            target_concurrent_requests: Some(128),
            ..FleetMemoryOptions::default()
        },
    );

    let option = &target.options[0];
    assert!(!option.fits);
    assert!(option.reason_en.to_lowercase().contains("exceed"));
}

#[test]
fn fleet_planner_prefill_peak_counts_concurrent_kv_cache() {
    let h100 = lookup("H100").unwrap();
    let rec = plan_with_memory_options(
        &llama_profile(),
        10_000_000_000,
        1_000_000_000,
        40_000_000_000,
        131_072,
        &h100,
        Some(1),
        &[(131_072, 1_000_000_000)],
        &[(131_072, 40_000_000_000)],
        FleetMemoryOptions {
            target_concurrent_requests: Some(16),
            ..FleetMemoryOptions::default()
        },
    );

    let option = &rec.options[0];
    assert_eq!(option.required_bytes_per_gpu_at_tier, 84_593_750_000);
    assert!(!option.fits);
}

#[test]
fn fleet_planner_prefill_peak_handles_extreme_concurrency_without_overflow() {
    let h100 = lookup("H100").unwrap();
    let rec = plan_with_memory_options(
        &llama_profile(),
        1,
        1,
        1,
        131_072,
        &h100,
        Some(1),
        &[(131_072, 1)],
        &[(131_072, 1)],
        FleetMemoryOptions {
            target_concurrent_requests: Some(u64::MAX),
            ..FleetMemoryOptions::default()
        },
    );

    let option = &rec.options[0];
    assert_eq!(option.required_bytes_per_gpu_at_tier, u64::MAX);
    assert!(!option.fits);
}

#[test]
fn fleet_planner_max_concurrent_is_capped_by_prefill_peak() {
    let h100 = lookup("H100").unwrap();
    let rec = plan_with_activation(
        &llama_profile(),
        10_000_000_000,
        1_000_000_000,
        40_000_000_000,
        131_072,
        &h100,
        Some(1),
        &[(131_072, 1_000_000_000)],
        &[(131_072, 40_000_000_000)],
    );

    let option = &rec.options[0];
    assert_eq!(option.max_concurrent_at_reference_ctx, 15);
    assert_eq!(option.max_concurrent_by_context, vec![(131_072, 15)]);
}

#[test]
fn fleet_planner_prefill_peak_does_not_break_simple_fit() {
    let h100 = lookup("H100").unwrap();
    let rec = plan_with_activation(
        &llama_profile(),
        10_000_000_000,
        1_000_000_000,
        5_000_000_000,
        131_072,
        &h100,
        Some(1),
        &[(131_072, 1_000_000_000)],
        &[(131_072, 5_000_000_000)],
    );

    assert!(rec.options[0].fits);
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
