use llm_infer_cal_core::core::evaluator::{EvaluationOptions, Evaluator};
use llm_infer_cal_core::model_source::base::{
    ModelArtifact, ModelNotFoundError, ModelSource, ModelSourceError, SiblingFile,
};
use serde_json::json;

#[derive(Clone)]
struct StaticSource {
    name: &'static str,
    artifact: ModelArtifact,
}

impl ModelSource for StaticSource {
    fn name(&self) -> &str {
        self.name
    }

    fn fetch(&self, model_id: &str) -> Result<ModelArtifact, ModelSourceError> {
        assert_eq!(model_id, self.artifact.model_id);
        Ok(self.artifact.clone())
    }
}

#[derive(Clone)]
struct MultiSource {
    name: &'static str,
    artifacts: Vec<ModelArtifact>,
}

impl ModelSource for MultiSource {
    fn name(&self) -> &str {
        self.name
    }

    fn fetch(&self, model_id: &str) -> Result<ModelArtifact, ModelSourceError> {
        self.artifacts
            .iter()
            .find(|artifact| artifact.model_id == model_id)
            .cloned()
            .ok_or_else(|| ModelSourceError::NotFound(ModelNotFoundError(model_id.to_string())))
    }
}

fn llama_artifact() -> ModelArtifact {
    ModelArtifact {
        source: "huggingface".to_string(),
        model_id: "test/llama-mini".to_string(),
        commit_sha: Some("abc1234def".to_string()),
        config: json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "vocab_size": 100,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "max_position_embeddings": 8192
        }),
        siblings: vec![
            SiblingFile {
                filename: "model-00001-of-00002.safetensors".to_string(),
                size: Some(5_472),
            },
            SiblingFile {
                filename: "model-00002-of-00002.safetensors".to_string(),
                size: Some(5_472),
            },
            SiblingFile {
                filename: "tokenizer.json".to_string(),
                size: Some(100),
            },
        ],
    }
}

fn giant_llama_artifact() -> ModelArtifact {
    ModelArtifact {
        source: "modelscope".to_string(),
        model_id: "test/giant-llama".to_string(),
        commit_sha: Some("giantsha".to_string()),
        config: json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 80,
            "hidden_size": 8192,
            "vocab_size": 128256,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 28672,
            "max_position_embeddings": 1048576
        }),
        siblings: vec![SiblingFile {
            filename: "model-00001-of-00001.safetensors".to_string(),
            size: Some(6_000_000_000_000),
        }],
    }
}

fn moe_artifact() -> ModelArtifact {
    ModelArtifact {
        source: "huggingface".to_string(),
        model_id: "test/moe-mini".to_string(),
        commit_sha: Some("moesha".to_string()),
        config: json!({
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "vocab_size": 100,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "num_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "max_position_embeddings": 8192
        }),
        siblings: vec![SiblingFile {
            filename: "model.safetensors".to_string(),
            size: Some(20_288),
        }],
    }
}

fn evaluator() -> Evaluator {
    Evaluator::without_cache(Box::new(StaticSource {
        name: "huggingface",
        artifact: llama_artifact(),
    }))
}

fn moe_evaluator() -> Evaluator {
    Evaluator::without_cache(Box::new(StaticSource {
        name: "huggingface",
        artifact: moe_artifact(),
    }))
}

fn draft_artifact() -> ModelArtifact {
    ModelArtifact {
        source: "huggingface".to_string(),
        model_id: "test/draft-mini".to_string(),
        commit_sha: Some("draftsha".to_string()),
        config: json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "vocab_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "max_position_embeddings": 4096
        }),
        siblings: vec![SiblingFile {
            filename: "draft.safetensors".to_string(),
            size: Some(2_048),
        }],
    }
}

fn speculative_evaluator() -> Evaluator {
    Evaluator::without_cache(Box::new(MultiSource {
        name: "huggingface",
        artifacts: vec![llama_artifact(), draft_artifact()],
    }))
}

fn giant_evaluator() -> Evaluator {
    Evaluator::without_cache(Box::new(StaticSource {
        name: "modelscope",
        artifact: giant_llama_artifact(),
    }))
}

#[test]
fn evaluate_composes_rust_pipeline_for_known_gpu() {
    let report = evaluator()
        .evaluate(
            "test/llama-mini",
            "H800",
            "sglang",
            EvaluationOptions {
                gpu_count: Some(2),
                context_length: Some(4096),
                input_tokens: Some(123),
                output_tokens: Some(45),
                target_tokens_per_sec: Some(17.5),
                prefill_utilization: 0.33,
                decode_bw_utilization: 0.44,
                concurrency_degradation: 1.67,
                ..EvaluationOptions::default()
            },
        )
        .unwrap();

    assert_eq!(report.model_id, "test/llama-mini");
    assert_eq!(report.source, "huggingface");
    assert_eq!(report.commit_sha.as_deref(), Some("abc1234def"));
    assert_eq!(report.engine, "sglang");
    assert_eq!(report.profile.model_type, "llama");
    assert_eq!(report.weight.total_bytes.value, 10_944);
    assert_eq!(report.total_params_estimate.value, 10_944);
    assert_eq!(
        report
            .kv_cache_by_context
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![4096]
    );
    assert_eq!(report.gpu_spec.as_ref().unwrap().id, "H800");
    assert_eq!(report.fleet.as_ref().unwrap().options.len(), 1);
    assert_eq!(report.fleet.as_ref().unwrap().options[0].gpu_count, 2);
    assert!(report
        .generated_command
        .as_deref()
        .unwrap()
        .contains("--context-length 4096"));
    assert!(report
        .generated_command
        .as_deref()
        .unwrap()
        .contains("--tp 2"));
    assert_eq!(report.perf_input_tokens, Some(123));
    assert_eq!(report.perf_output_tokens, Some(45));
    assert_eq!(report.perf_target_tokens_per_sec, Some(17.5));
    assert_eq!(report.prefill.as_ref().unwrap().utilization, 0.33);
    assert_eq!(report.decode.as_ref().unwrap().bw_utilization, 0.44);
    assert_eq!(
        report.concurrency.as_ref().unwrap().target_tokens_per_sec,
        17.5
    );
    assert_eq!(
        report.concurrency.as_ref().unwrap().degradation_factor,
        1.67
    );
}

#[test]
fn evaluate_uses_model_max_context_instead_of_fixed_reference_context() {
    let report = evaluator()
        .evaluate(
            "test/llama-mini",
            "H800",
            "vllm",
            EvaluationOptions::default(),
        )
        .unwrap();

    assert_eq!(
        report
            .kv_cache_by_context
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![4096, 8192]
    );

    let fleet = report.fleet.as_ref().unwrap();
    for option in &fleet.options {
        assert!(option
            .max_concurrent_by_context
            .iter()
            .all(|(ctx, _)| *ctx <= 8192));
        assert_eq!(option.kv_reference_context_tokens, 8192);
        assert!(!option.reason_en.contains("128K"));
        assert!(!option.reason_zh.contains("128K"));
    }
}

#[test]
fn evaluate_applies_kv_bits_and_paged_attention_to_kv_cache() {
    let plain = evaluator()
        .evaluate(
            "test/llama-mini",
            "H800",
            "vllm",
            EvaluationOptions {
                context_length: Some(4096),
                kv_cache_bits: 16,
                paged_attention: false,
                ..EvaluationOptions::default()
            },
        )
        .unwrap();
    let paged_fp8 = evaluator()
        .evaluate(
            "test/llama-mini",
            "H800",
            "vllm",
            EvaluationOptions {
                context_length: Some(4096),
                kv_cache_bits: 8,
                paged_attention: true,
                ..EvaluationOptions::default()
            },
        )
        .unwrap();

    let plain_kv = plain.kv_cache_by_context.get(&4096).unwrap().value;
    let paged_kv = paged_fp8.kv_cache_by_context.get(&4096).unwrap().value;

    assert_eq!(paged_kv, plain_kv * 3 / 8);
    assert_eq!(paged_fp8.kv_cache_bits, 8);
    assert!(paged_fp8.paged_attention);
    assert!(paged_fp8
        .kv_cache_by_context
        .get(&4096)
        .unwrap()
        .source
        .as_deref()
        .unwrap_or("")
        .contains("paged"));
}

#[test]
fn evaluate_adds_draft_model_safetensors_to_gpu_resident_weights() {
    let report = speculative_evaluator()
        .evaluate(
            "test/llama-mini",
            "H800",
            "vllm",
            EvaluationOptions {
                gpu_count: Some(1),
                context_length: Some(4096),
                target_concurrent_requests: Some(3),
                speculative_draft_model_id: Some("test/draft-mini".to_string()),
                ..EvaluationOptions::default()
            },
        )
        .unwrap();

    let option = &report.fleet.as_ref().unwrap().options[0];
    assert_eq!(report.speculative_weight_bytes.value, 2_048);
    assert_eq!(report.speculative_weight_bytes.label.as_str(), "verified");
    assert_eq!(option.tier, "target");
    assert_eq!(option.tier_concurrent_requests, 3);
    assert_eq!(option.speculative_weight_bytes_per_gpu, 2_048);
    assert_eq!(
        option.weight_bytes_per_gpu,
        option.main_weight_bytes_per_gpu + 2_048
    );
    assert!(option.required_bytes_per_gpu_at_tier > option.main_weight_bytes_per_gpu);
    let command = report.generated_command.as_deref().unwrap();
    assert!(command.contains("--speculative-config"));
    assert!(command.contains("\"model\":\"test/draft-mini\""));
    assert!(command.contains("\"num_speculative_tokens\":4"));
}

#[test]
fn evaluate_adds_sglang_draft_model_path_to_generated_command() {
    let report = speculative_evaluator()
        .evaluate(
            "test/llama-mini",
            "H800",
            "sglang",
            EvaluationOptions {
                gpu_count: Some(1),
                context_length: Some(4096),
                target_concurrent_requests: Some(3),
                speculative_draft_model_id: Some("test/draft-mini".to_string()),
                ..EvaluationOptions::default()
            },
        )
        .unwrap();

    let command = report.generated_command.as_deref().unwrap();
    assert!(command.contains("--speculative-algorithm STANDALONE"));
    assert!(command.contains("--speculative-draft-model-path test/draft-mini"));
    assert!(command.contains("--speculative-num-draft-tokens 7"));
}

#[test]
fn evaluate_reuses_one_activation_working_set_across_contexts() {
    let report = evaluator()
        .evaluate(
            "test/llama-mini",
            "H800",
            "vllm",
            EvaluationOptions::default(),
        )
        .unwrap();

    let activation_values = report
        .activation_by_context
        .values()
        .map(|value| value.value)
        .collect::<Vec<_>>();

    assert!(activation_values.len() > 1);
    assert!(activation_values.windows(2).all(|pair| pair[0] == pair[1]));
    assert!(report
        .activation_by_context
        .values()
        .next()
        .unwrap()
        .source
        .as_deref()
        .unwrap_or("")
        .contains("batched_tokens"));
}

#[test]
fn evaluate_uses_moe_active_params_for_prefill_and_decode() {
    let report = moe_evaluator()
        .evaluate(
            "test/moe-mini",
            "H800",
            "vllm",
            EvaluationOptions {
                gpu_count: Some(1),
                context_length: Some(4096),
                input_tokens: Some(100),
                prefill_utilization: 1.0,
                decode_bw_utilization: 1.0,
                ..EvaluationOptions::default()
            },
        )
        .unwrap();

    assert_eq!(report.total_params_estimate.value, 20_288);
    assert_eq!(report.active_params_estimate.value, 14_144);
    assert_eq!(
        report.prefill.as_ref().unwrap().total_flops.value,
        2 * 14_144 * 100
    );
    assert_eq!(
        report
            .decode
            .as_ref()
            .unwrap()
            .active_weight_bytes_per_gpu
            .value,
        14_144
    );
    assert_eq!(
        report
            .decode
            .as_ref()
            .unwrap()
            .moe_active_weight_bytes_per_gpu
            .as_ref()
            .unwrap()
            .value,
        14_144
    );
}

#[test]
fn evaluate_embeds_unknown_gpu_error_without_aborting() {
    let report = evaluator()
        .evaluate(
            "test/llama-mini",
            "ImaginaryGPU",
            "vllm",
            EvaluationOptions::default(),
        )
        .unwrap();

    assert!(report.gpu_spec.is_none());
    assert!(report
        .gpu_error
        .as_deref()
        .unwrap_or("")
        .contains("Unknown GPU"));
    assert!(report.fleet.is_none());
    assert!(report.generated_command.is_none());
    assert!(report.prefill.is_none());
    assert!(report.decode.is_none());
    assert!(report.concurrency.is_none());
}

#[test]
fn evaluate_does_not_emit_command_or_perf_when_no_candidate_fits() {
    let report = giant_evaluator()
        .evaluate(
            "test/giant-llama",
            "H100",
            "vllm",
            EvaluationOptions::default(),
        )
        .unwrap();

    let fleet = report.fleet.as_ref().unwrap();
    assert!(fleet.options.iter().all(|option| !option.fits));
    assert_eq!(fleet.best_tier, None);
    assert!(fleet.best_option().is_none());
    assert!(report.generated_command.is_none());
    assert!(report.prefill.is_none());
    assert!(report.decode.is_none());
    assert!(report.concurrency.is_none());
}
