use llm_infer_cal_core::core::evaluator::{EvaluationOptions, Evaluator};
use llm_infer_cal_core::core::explain::{build, ExplainEntry};
use llm_infer_cal_core::model_source::base::{
    ModelArtifact, ModelSource, ModelSourceError, SiblingFile,
};
use serde_json::json;

#[derive(Clone)]
struct StaticSource {
    artifact: ModelArtifact,
}

impl ModelSource for StaticSource {
    fn name(&self) -> &str {
        "huggingface"
    }

    fn fetch(&self, model_id: &str) -> Result<ModelArtifact, ModelSourceError> {
        assert_eq!(model_id, self.artifact.model_id);
        Ok(self.artifact.clone())
    }
}

fn report() -> llm_infer_cal_core::core::evaluator::EvaluationReport {
    report_with_config(
        json!({
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
        EvaluationOptions {
            context_length: Some(4096),
            input_tokens: Some(512),
            target_tokens_per_sec: Some(20.0),
            ..EvaluationOptions::default()
        },
    )
}

fn report_with_config(
    config: serde_json::Value,
    options: EvaluationOptions,
) -> llm_infer_cal_core::core::evaluator::EvaluationReport {
    let artifact = ModelArtifact {
        source: "huggingface".to_string(),
        model_id: "test/llama-mini".to_string(),
        commit_sha: Some("abc1234def".to_string()),
        config,
        siblings: vec![
            SiblingFile {
                filename: "model-00001-of-00002.safetensors".to_string(),
                size: Some(5_472),
            },
            SiblingFile {
                filename: "model-00002-of-00002.safetensors".to_string(),
                size: Some(5_472),
            },
        ],
    };

    Evaluator::without_cache(Box::new(StaticSource { artifact }))
        .evaluate("test/llama-mini", "H800", "vllm", options)
        .unwrap()
}

fn kv_entry(entries: &[ExplainEntry]) -> &ExplainEntry {
    entries
        .iter()
        .find(|entry| entry.heading.starts_with("KV cache @"))
        .expect("KV explain entry")
}

#[test]
fn build_emits_explain_entries_in_report_order() {
    let entries = build(&report());
    let headings = entries
        .iter()
        .map(|entry| entry.heading.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        headings,
        vec![
            "Weight bytes (safetensors file sum)",
            "Quantization scheme (reconciliation)",
            "KV cache @ 4K context",
            "Fleet tier: min (1 GPUs)",
            "Fleet tier: dev (1 GPUs)",
            "Fleet tier: prod (1 GPUs)",
            "Prefill latency (single request)",
            "Decode throughput (cluster)",
            "K bound (memory capacity)",
            "L bound (compute/bandwidth at SLA)",
            "Max concurrent + bottleneck verdict",
        ]
    );

    assert_eq!(
        entries[0].formula,
        "sum(file.size for file in model_source.file_metadata if file.endswith('.safetensors'))"
    );
    assert_eq!(
        entries[0].inputs[0].value,
        "source=huggingface, sha=abc1234def"
    );
    assert_eq!(entries[0].steps[0], "Raw value from API = 10,944 bytes");
    assert_eq!(entries[0].result, "10,944 bytes [verified]");

    assert_eq!(entries[1].inputs[0].name, "observed_bytes");
    assert!(entries[1].steps[1].contains("FP8"));
    assert_eq!(entries[2].inputs[0].name, "num_kv_heads");
    assert_eq!(entries[2].result, "262,144 bytes = 0.00 GB [estimated]");
    assert_eq!(
        entries[3].formula,
        "required_per_gpu = max(decode_required, prefill_required)"
    );
    assert!(entries[3]
        .steps
        .iter()
        .any(|step| step.contains("prefill_required")));
    assert!(entries[3]
        .steps
        .iter()
        .any(|step| step.contains("concurrent_KV")));
    assert!(entries[6].steps[0].contains("FLOPs = 2 x 10,944 x 512"));
    assert!(entries[7].steps[3].contains("cluster_tok_per_sec"));
    assert_eq!(entries[8].methodology_anchor, "#k-bound-memory-capacity");
    assert_eq!(entries[9].inputs[1].label, "[user-set]");
    assert!(entries[10].result.contains("bottleneck = memory_capacity"));
}

#[test]
fn build_skips_fleet_and_perf_sections_when_gpu_is_unknown() {
    let mut report = report();
    report.gpu_spec = None;
    report.fleet = None;
    report.prefill = None;
    report.decode = None;
    report.concurrency = None;

    let headings = build(&report)
        .into_iter()
        .map(|entry| entry.heading)
        .collect::<Vec<_>>();

    assert_eq!(
        headings,
        vec![
            "Weight bytes (safetensors file sum)",
            "Quantization scheme (reconciliation)",
            "KV cache @ 4K context",
        ]
    );
}

#[test]
fn kv_explain_accounts_for_sliding_window_effective_seq_len() {
    let report = report_with_config(
        json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "vocab_size": 100,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "max_position_embeddings": 8192,
            "sliding_window": 1024
        }),
        EvaluationOptions {
            context_length: Some(4096),
            ..EvaluationOptions::default()
        },
    );

    let entry = kv_entry(&build(&report)).clone();

    assert!(entry
        .steps
        .iter()
        .any(|step| step == "effective_seq_len = min(seq_len, sliding_window) = 1,024"));
    assert!(entry.steps.iter().any(|step| step
        == "baseline = per_tok_per_layer x effective_seq_len x num_layers = 65,536 bytes"));
    assert!(entry
        .steps
        .iter()
        .any(|step| step == "result = raw_kv = 65,536 bytes"));
    assert_eq!(entry.result, "65,536 bytes = 0.00 GB [estimated]");
}

#[test]
fn kv_explain_accounts_for_nsa_sparsity() {
    let report = report_with_config(
        json!({
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "vocab_size": 100,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "max_position_embeddings": 8192,
            "nsa_config": { "topk": 128 }
        }),
        EvaluationOptions {
            context_length: Some(4096),
            ..EvaluationOptions::default()
        },
    );

    let entry = kv_entry(&build(&report)).clone();

    assert!(entry
        .steps
        .iter()
        .any(|step| step == "nsa_sparsity = min(nsa_topk / effective_seq_len, 1.0) = 0.0312"));
    assert!(entry
        .steps
        .iter()
        .any(|step| step == "result = baseline x nsa_sparsity = 8,192 bytes"));
    assert_eq!(entry.result, "8,192 bytes = 0.00 GB [estimated]");
}

#[test]
fn kv_explain_accounts_for_paged_attention_factor() {
    let report = report_with_config(
        json!({
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
        EvaluationOptions {
            context_length: Some(4096),
            paged_attention: true,
            ..EvaluationOptions::default()
        },
    );

    let entry = kv_entry(&build(&report)).clone();

    assert!(entry
        .steps
        .iter()
        .any(|step| step == "paged_attention_factor = 0.75"));
    assert!(entry
        .steps
        .iter()
        .any(|step| step == "result = raw_kv x paged_attention_factor = 196,608 bytes"));
    assert_eq!(entry.result, "196,608 bytes = 0.00 GB [estimated]");
}

#[test]
fn methodology_docs_include_runtime_memory_adjustments() {
    let docs = [
        include_str!("../../../docs/methodology.md"),
        include_str!("../../../docs/zh/methodology.md"),
        include_str!("../../../docs/architecture-guide.md"),
        include_str!("../../../docs/zh/architecture-guide.md"),
    ];

    for doc in docs {
        assert!(doc.contains("paged_attention_factor = 0.75"));
        assert!(
            doc.contains("moe_activation_correction = 1 + active_experts / total_experts * 0.5")
        );
        assert!(doc.contains("required_per_gpu = max(decode_required, prefill_required)"));
        assert!(doc.contains("prefill_active_requests = max(concurrent_requests / 8, 1)"));
    }
}
