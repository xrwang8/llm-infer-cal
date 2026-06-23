use std::collections::HashMap;

use llm_infer_cal_core::model_source::auth::{
    get_hf_token_from_values, get_modelscope_token_from_values, hf_auth_error_message,
    modelscope_auth_error_message,
};
use llm_infer_cal_core::model_source::base::{
    AuthRequiredError, ModelArtifact, ModelNotFoundError, SiblingFile, SourceUnavailableError,
};
use serde_json::json;

#[test]
fn model_artifact_and_sibling_file_preserve_source_data() {
    let artifact = ModelArtifact {
        source: "modelscope".to_string(),
        model_id: "owner/repo".to_string(),
        commit_sha: Some("deadbeef".to_string()),
        config: json!({"model_type": "qwen3_moe"}),
        siblings: vec![
            SiblingFile {
                filename: "config.json".to_string(),
                size: Some(642),
            },
            SiblingFile {
                filename: "model.safetensors".to_string(),
                size: None,
            },
        ],
    };

    assert_eq!(artifact.source, "modelscope");
    assert_eq!(artifact.commit_sha.as_deref(), Some("deadbeef"));
    assert_eq!(artifact.config["model_type"], "qwen3_moe");
    assert_eq!(artifact.siblings[1].size, None);
}

#[test]
fn auth_token_priority_matches_rust_contract() {
    let hf = HashMap::from([
        ("HF_TOKEN", Some("hf-primary")),
        ("HUGGING_FACE_HUB_TOKEN", Some("hf-legacy")),
    ]);
    let hf_fallback = HashMap::from([
        ("HF_TOKEN", None),
        ("HUGGING_FACE_HUB_TOKEN", Some("hf-legacy")),
    ]);
    let ms = HashMap::from([
        ("MODELSCOPE_API_TOKEN", Some("ms-primary")),
        ("MODELSCOPE_TOKEN", Some("ms-legacy")),
    ]);
    let ms_fallback = HashMap::from([
        ("MODELSCOPE_API_TOKEN", None),
        ("MODELSCOPE_TOKEN", Some("ms-legacy")),
    ]);

    assert_eq!(
        get_hf_token_from_values(&hf),
        Some("hf-primary".to_string())
    );
    assert_eq!(
        get_hf_token_from_values(&hf_fallback),
        Some("hf-legacy".to_string())
    );
    assert_eq!(
        get_modelscope_token_from_values(&ms),
        Some("ms-primary".to_string())
    );
    assert_eq!(
        get_modelscope_token_from_values(&ms_fallback),
        Some("ms-legacy".to_string())
    );
}

#[test]
fn auth_error_messages_match_user_facing_text() {
    let hf = hf_auth_error_message("meta/llama");
    let ms = modelscope_auth_error_message("Qwen/Qwen3");

    assert!(hf.contains("Model 'meta/llama' requires authentication"));
    assert!(hf.contains("HF_TOKEN"));
    assert!(ms.contains("模型 'Qwen/Qwen3' 需要登录"));
    assert!(ms.contains("MODELSCOPE_API_TOKEN"));
}

#[test]
fn model_source_errors_display_message() {
    assert_eq!(
        ModelNotFoundError("missing".to_string()).to_string(),
        "missing"
    );
    assert_eq!(
        AuthRequiredError("private".to_string()).to_string(),
        "private"
    );
    assert_eq!(
        SourceUnavailableError("timeout".to_string()).to_string(),
        "timeout"
    );
}
