use llm_infer_cal_core::architecture::profile::{ArchitectureProfile, PositionTraits};
use llm_infer_cal_core::command_generator::sglang::generate_sglang_command;
use llm_infer_cal_core::command_generator::vllm::generate_vllm_command;
use llm_infer_cal_core::command_generator::Parallelism;
use llm_infer_cal_core::engine_compat::{EngineCompatEntry, EngineFlag};

fn profile(model_type: &str, max_position_embeddings: Option<u64>) -> ArchitectureProfile {
    ArchitectureProfile {
        model_type: model_type.to_string(),
        position: max_position_embeddings.map(|max| PositionTraits {
            rope_type: Some("rope".to_string()),
            rope_theta: None,
            rope_scaling_factor: None,
            max_position_embeddings: Some(max),
        }),
        ..ArchitectureProfile::default()
    }
}

#[test]
fn vllm_basic_shape_matches_rust_contract() {
    let cmd = generate_vllm_command(
        "meta-llama/Llama-3.3-70B",
        &profile("llama", Some(131_072)),
        Parallelism::single(2),
        None,
        None,
    );

    assert!(cmd.contains("vllm serve meta-llama/Llama-3.3-70B"));
    assert!(cmd.contains("--tensor-parallel-size 2"));
    assert!(cmd.contains("--max-model-len 131072"));
    assert!(cmd.contains("--gpu-memory-utilization 0.9"));
}

#[test]
fn vllm_trust_remote_code_heuristic_matches_rust_contract() {
    let llama = generate_vllm_command(
        "meta-llama/Llama-3.3-70B",
        &profile("llama", Some(131_072)),
        Parallelism::single(2),
        None,
        None,
    );
    let deepseek = generate_vllm_command(
        "deepseek-ai/DeepSeek-V4-Flash",
        &profile("deepseek_v4", Some(1_048_576)),
        Parallelism::single(8),
        None,
        None,
    );

    assert!(!llama.contains("--trust-remote-code"));
    assert!(deepseek.contains("--trust-remote-code"));
    assert!(deepseek.contains("--max-model-len 1048576"));
}

#[test]
fn vllm_max_model_len_override_matches_rust_contract() {
    let cmd = generate_vllm_command(
        "foo/bar",
        &profile("llama", Some(131_072)),
        Parallelism::single(2),
        None,
        Some(32_768),
    );

    assert!(cmd.contains("--max-model-len 32768"));
    assert!(!cmd.contains("--max-model-len 131072"));
}

#[test]
fn vllm_entry_flags_are_appended_verbatim() {
    let entry = EngineCompatEntry {
        required_flags: vec![],
        optional_flags: vec![EngineFlag {
            flag: "--attention-backend".to_string(),
            value: Some("auto".to_string()),
            ..EngineFlag::default()
        }],
        ..EngineCompatEntry::default()
    };

    let cmd = generate_vllm_command(
        "deepseek-ai/DeepSeek-V4-Flash",
        &profile("deepseek_v4", Some(1_048_576)),
        Parallelism::single(8),
        Some(&entry),
        None,
    );

    assert!(cmd.contains("--attention-backend auto"));
}

#[test]
fn sglang_basic_shape_matches_rust_contract() {
    let cmd = generate_sglang_command(
        "deepseek-ai/DeepSeek-V3.2",
        &profile("deepseek_v3_2", Some(131_072)),
        Parallelism::single(8),
        None,
        None,
    );

    assert!(cmd.contains("python -m sglang.launch_server"));
    assert!(cmd.contains("--model-path deepseek-ai/DeepSeek-V3.2"));
    assert!(cmd.contains("--tp 8"));
    assert!(cmd.contains("--context-length 131072"));
}

#[test]
fn sglang_entry_required_flags_are_appended_verbatim() {
    let entry = EngineCompatEntry {
        required_flags: vec![EngineFlag {
            flag: "--attention-backend".to_string(),
            value: Some("nsa".to_string()),
            ..EngineFlag::default()
        }],
        optional_flags: vec![],
        ..EngineCompatEntry::default()
    };

    let cmd = generate_sglang_command(
        "deepseek-ai/DeepSeek-V3.2",
        &profile("deepseek_v3_2", Some(131_072)),
        Parallelism::single(8),
        Some(&entry),
        None,
    );

    assert!(cmd.contains("--attention-backend nsa"));
}

#[test]
fn vllm_multinode_command_uses_tp_per_node_and_pp_nodes() {
    let cmd = generate_vllm_command(
        "ZhipuAI/GLM-5.2",
        &profile("glm_moe_dsa", Some(1_048_576)),
        Parallelism {
            total_gpus: 48,
            tensor_parallel_size: 8,
            pipeline_parallel_size: 6,
        },
        None,
        Some(1_048_576),
    );

    assert!(cmd.contains("--tensor-parallel-size 8"));
    assert!(cmd.contains("--pipeline-parallel-size 6"));
    assert!(cmd.contains("--distributed-executor-backend ray"));
}

#[test]
fn sglang_multinode_command_uses_total_tp_and_node_bootstrap_flags() {
    let cmd = generate_sglang_command(
        "ZhipuAI/GLM-5.2",
        &profile("glm_moe_dsa", Some(1_048_576)),
        Parallelism {
            total_gpus: 16,
            tensor_parallel_size: 8,
            pipeline_parallel_size: 2,
        },
        None,
        Some(1_048_576),
    );

    assert!(cmd.contains("--tp 16"));
    assert!(cmd.contains("--nnodes 2"));
    assert!(cmd.contains("--node-rank ${NODE_RANK:-0}"));
    assert!(cmd.contains("--dist-init-addr ${NODE0_IP:-<node0-ip>}:20000"));
}
