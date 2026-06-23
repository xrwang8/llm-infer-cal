use llm_infer_cal_core::core::evaluator::{EvaluationOptions, Evaluator};
use llm_infer_cal_core::model_source::base::{
    ModelArtifact, ModelSource, ModelSourceError, SiblingFile,
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

fn evaluator() -> Evaluator {
    Evaluator::without_cache(Box::new(StaticSource {
        name: "huggingface",
        artifact: llama_artifact(),
    }))
}

#[test]
fn evaluate_composes_python_pipeline_for_known_gpu() {
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
