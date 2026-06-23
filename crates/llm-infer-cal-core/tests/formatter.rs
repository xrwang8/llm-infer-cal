use std::sync::{Mutex, MutexGuard, OnceLock};

use llm_infer_cal_core::common::i18n::set_locale;
use llm_infer_cal_core::core::evaluator::{EvaluationOptions, Evaluator};
use llm_infer_cal_core::core::explain::{build as build_explain, ExplainEntry, ExplainInput};
use llm_infer_cal_core::hardware::loader::load_database;
use llm_infer_cal_core::llm_review::reviewer::LlmReviewResult;
use llm_infer_cal_core::model_source::base::{
    ModelArtifact, ModelSource, ModelSourceError, SiblingFile,
};
use llm_infer_cal_core::output::formatter::{
    fmt_bytes, fmt_params, format_tag, render_explain_text, render_gpu_list_text,
    render_llm_review_text, render_report_text,
};
use llm_infer_cal_core::output::labels::{AnnotatedValue, Label};
use serde_json::json;

fn locale_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

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
    let artifact = ModelArtifact {
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
        ],
    };
    Evaluator::without_cache(Box::new(StaticSource { artifact }))
        .evaluate(
            "test/llama-mini",
            "H800",
            "vllm",
            EvaluationOptions {
                context_length: Some(4096),
                input_tokens: Some(512),
                target_tokens_per_sec: Some(20.0),
                ..EvaluationOptions::default()
            },
        )
        .unwrap()
}

fn unknown_warning_report() -> llm_infer_cal_core::core::evaluator::EvaluationReport {
    let artifact = ModelArtifact {
        source: "huggingface".to_string(),
        model_id: "test/unknown".to_string(),
        commit_sha: Some("abc1234def".to_string()),
        config: json!({
            "model_type": "gpt2",
            "vocab_size": 50257
        }),
        siblings: vec![],
    };
    Evaluator::without_cache(Box::new(StaticSource { artifact }))
        .evaluate("test/unknown", "H800", "vllm", EvaluationOptions::default())
        .unwrap()
}

fn no_fit_report() -> llm_infer_cal_core::core::evaluator::EvaluationReport {
    let artifact = ModelArtifact {
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
    };
    Evaluator::without_cache(Box::new(StaticSource { artifact }))
        .evaluate(
            "test/giant-llama",
            "H100",
            "vllm",
            EvaluationOptions::default(),
        )
        .unwrap()
}

#[test]
fn helper_formatting_matches_rust_contract_formatter() {
    let _guard = locale_lock();
    set_locale("en");
    assert_eq!(fmt_bytes(999), "999 B");
    assert_eq!(fmt_bytes(12_345), "12.35 KB");
    assert_eq!(fmt_bytes(12_345_678), "12.35 MB");
    assert_eq!(fmt_bytes(12_345_678_900), "12.35 GB");
    assert_eq!(fmt_params(999), "999");
    assert_eq!(fmt_params(12_345_678), "12.35M");
    assert_eq!(fmt_params(12_345_678_900), "12.35B");
    assert_eq!(
        format_tag(&AnnotatedValue::new(1_u64, Label::Verified, None)),
        "[verified]"
    );

    set_locale("zh");
    assert_eq!(
        format_tag(&AnnotatedValue::new(1_u64, Label::Estimated, None)),
        "[估算]"
    );
}

#[test]
fn render_report_text_contains_core_sections_and_localized_values() {
    let _guard = locale_lock();
    set_locale("zh");
    let rendered = render_report_text(&report());

    assert!(rendered.contains("test/llama-mini"));
    assert!(rendered.contains("来源 huggingface @ abc1234"));
    assert!(rendered.contains("架构"));
    assert!(rendered.contains("模型类型: llama [已验证]"));
    assert!(rendered.contains("权重"));
    assert!(rendered.contains("safetensors 总字节: 10.94 KB [已验证]"));
    assert!(rendered.contains("单请求 KV Cache"));
    assert!(rendered.contains("4,096 tokens: 262.14 KB [估算]"));
    assert!(rendered.contains("目标硬件 - H800"));
    assert!(rendered.contains("推荐 GPU 张数 - H800"));
    assert!(rendered.contains("假设输入 512 tokens"));
    assert!(!rendered.contains("Assumptions:"));
    assert!(rendered.contains("生成的启动命令"));
    assert!(rendered.contains("标签："));
}

#[test]
fn render_report_text_does_not_quote_warning_strings() {
    let _guard = locale_lock();
    set_locale("zh");
    let rendered = render_report_text(&unknown_warning_report());

    assert!(rendered.contains("WARNING: No recognizable model_type"));
    assert!(!rendered.contains("WARNING: \"No recognizable model_type"));
}

#[test]
fn render_report_text_does_not_recommend_or_generate_command_when_no_candidate_fits() {
    let _guard = locale_lock();
    set_locale("zh");
    let rendered = render_report_text(&no_fit_report());

    assert!(rendered.contains("推荐 GPU 张数 - H100"));
    assert!(rendered.contains("没有推荐档位：模型无法放入当前 TP/PP 候选配置。"));
    assert!(!rendered.contains("生产 *"));
    assert!(!rendered.contains("= 推荐档位"));
    assert!(!rendered.contains("性能分析"));
    assert!(!rendered.contains("生成的启动命令"));
}

#[test]
fn render_explain_and_llm_review_text_match_visible_contract() {
    let _guard = locale_lock();
    set_locale("en");
    let entries = vec![ExplainEntry {
        heading: "Weight bytes".to_string(),
        formula: "sum(safetensors.size)".to_string(),
        inputs: vec![ExplainInput::new("api", "HF siblings", "[verified]")],
        steps: vec!["result = 160 GB".to_string()],
        result: "160 GB [verified]".to_string(),
        source: "HF API".to_string(),
        methodology_anchor: "#weight-bytes".to_string(),
    }];
    let explain = render_explain_text(&entries);
    assert!(explain.contains("Full derivation traces (--explain)"));
    assert!(explain.contains("Formula:"));
    assert!(explain.contains("api = HF siblings [verified]"));
    assert!(explain.contains("docs/methodology.md#weight-bytes"));

    let unavailable = render_llm_review_text(&LlmReviewResult {
        ok: false,
        content: None,
        error: Some("missing".to_string()),
        model: "gpt-4o".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
    });
    assert!(unavailable.contains("LLM review unavailable: missing"));
    assert!(unavailable.contains("LLM_CAL_REVIEWER_API_KEY"));

    let ok = render_llm_review_text(&LlmReviewResult {
        ok: true,
        content: Some("No issues".to_string()),
        error: None,
        model: "gpt-4o".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
    });
    assert!(ok.contains("[llm-opinion]"));
    assert!(ok.contains("No issues"));

    let generated = render_explain_text(&build_explain(&report()));
    assert!(generated.contains("Decode throughput (cluster)"));
}

#[test]
fn render_gpu_list_preserves_database_order_and_boolean_words() {
    let _guard = locale_lock();
    set_locale("en");
    let db = load_database().unwrap();
    let rendered = render_gpu_list_text(&db);

    assert!(rendered.starts_with("Supported GPUs"));
    assert!(rendered.contains("B200 | 192 GB | 1800 GB/s | 2250 | yes | yes"));
    assert!(rendered.contains("H800-SXM5, H800-80G"));
    assert!(rendered.contains(&format!("Total: {} GPUs", db.gpus.len())));
}
