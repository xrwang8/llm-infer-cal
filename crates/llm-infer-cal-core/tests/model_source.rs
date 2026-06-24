use std::collections::HashMap;

use llm_infer_cal_core::model_source::auth::{
    get_hf_token_from_values, get_modelscope_token_from_values, hf_auth_error_message,
    modelscope_auth_error_message,
};
use llm_infer_cal_core::model_source::base::{
    AuthRequiredError, ModelArtifact, ModelNotFoundError, SiblingFile, SourceUnavailableError,
};
use llm_infer_cal_core::model_source::builtin::BuiltinSource;
use serde_json::{json, Value};

const BUILTIN_MANIFEST_JSON: &str = include_str!("../data/builtin_model_manifest.json");
const BUILTIN_CATALOG_JSON: &str = include_str!("../data/builtin_models.json");

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

#[test]
fn builtin_source_returns_qwen36_artifact_without_network() {
    let artifact = BuiltinSource
        .fetch("Qwen/Qwen3.6-35B-A3B")
        .expect("Qwen3.6-35B-A3B should be embedded");

    assert_eq!(artifact.source, "builtin");
    assert_eq!(artifact.model_id, "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(
        artifact.commit_sha.as_deref(),
        Some("995ad96eacd98c81ed38be0c5b274b04031597b0")
    );
    assert_eq!(artifact.config["model_type"], "qwen3_5_moe");
    assert_eq!(artifact.config["text_config"]["num_hidden_layers"], 40);
    assert_eq!(artifact.config["text_config"]["num_experts"], 256);
    assert_eq!(
        artifact
            .siblings
            .iter()
            .filter(|sibling| sibling.filename.ends_with(".safetensors"))
            .count(),
        1
    );
    assert_eq!(
        artifact
            .siblings
            .iter()
            .filter_map(|sibling| sibling.size)
            .sum::<u64>(),
        71_903_776_776
    );
}

#[test]
fn builtin_source_returns_glm52_artifact_without_network() {
    let artifact = BuiltinSource
        .fetch("ZhipuAI/GLM-5.2")
        .expect("GLM-5.2 should be embedded");

    assert_eq!(artifact.source, "builtin");
    assert_eq!(artifact.model_id, "ZhipuAI/GLM-5.2");
    assert_eq!(artifact.commit_sha.as_deref(), Some("master"));
    assert_eq!(artifact.config["model_type"], "glm_moe_dsa");
    assert_eq!(artifact.config["num_hidden_layers"], 78);
    assert_eq!(artifact.config["hidden_size"], 6144);
    assert_eq!(artifact.config["num_attention_heads"], 64);
    assert_eq!(artifact.config["n_routed_experts"], 256);
    assert_eq!(artifact.config["n_shared_experts"], 1);
    assert_eq!(
        artifact
            .siblings
            .iter()
            .filter_map(|sibling| sibling.size)
            .sum::<u64>(),
        1_506_667_387_408
    );

    let alias = BuiltinSource
        .fetch("zai-org/GLM-5.2")
        .expect("vLLM recipe id should resolve as an alias");
    assert_eq!(alias.model_id, "ZhipuAI/GLM-5.2");
}

#[test]
fn builtin_source_resolves_every_manifest_model_and_alias_without_network() {
    let manifest: Value = serde_json::from_str(BUILTIN_MANIFEST_JSON).unwrap();
    let models = manifest["models"].as_array().unwrap();
    assert!(
        models.len() >= 100,
        "expected the local manifest to cover the vLLM/SGLang model set"
    );

    let source = BuiltinSource;
    for model in models {
        let id = model["id"].as_str().unwrap();
        let artifact = source.fetch(id).unwrap_or_else(|error| {
            panic!("manifest model {id} should be embedded: {error}");
        });
        assert_eq!(artifact.source, "builtin");
        assert_eq!(artifact.model_id, id);

        for alias in model["aliases"].as_array().unwrap() {
            let alias = alias.as_str().unwrap();
            let alias_artifact = source.fetch(alias).unwrap_or_else(|error| {
                panic!("manifest alias {alias} should resolve to {id}: {error}");
            });
            assert_eq!(alias_artifact.model_id, id);
        }
    }
}

#[test]
fn builtin_manifest_tracks_concrete_recipe_models_not_family_pages() {
    let manifest: Value = serde_json::from_str(BUILTIN_MANIFEST_JSON).unwrap();
    let ids: Vec<_> = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["id"].as_str().unwrap())
        .collect();

    for family_page in [
        "Qwen/Qwen2.5-VL",
        "Qwen/Qwen3",
        "Qwen/Qwen3-Coder",
        "Qwen/Qwen3-Next",
        "Qwen/Qwen3-VL",
        "Qwen/Qwen3.5",
        "Qwen/Qwen3.6",
        "deepseek-ai/DeepSeek-V4",
        "zai-org/Glyph-FP8",
    ] {
        assert!(
            !ids.contains(&family_page),
            "{family_page} is an SGLang recipe family page, not a concrete model repository"
        );
    }

    for concrete_variant in [
        "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "nvidia/DeepSeek-V4-Pro-NVFP4",
        "nvidia/Kimi-K2.6-NVFP4",
        "nvidia/GLM-5.1-NVFP4",
    ] {
        assert!(
            ids.contains(&concrete_variant),
            "{concrete_variant} should be included because vLLM lists it under the target providers"
        );
    }
}

#[test]
fn builtin_catalog_has_no_unavailable_placeholders() {
    let catalog: Value = serde_json::from_str(BUILTIN_CATALOG_JSON).unwrap();
    let unavailable: Vec<_> = catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|model| model["snapshot_status"] == "unavailable")
        .map(|model| model["id"].as_str().unwrap())
        .collect();

    assert!(
        unavailable.is_empty(),
        "builtin catalog should not treat failed network fetches as completed local models: {unavailable:?}"
    );

    let glm_ga = catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == "zai-org/GLM-GA")
        .expect("vLLM recipe-only GLM-GA should still be represented locally");
    assert_eq!(glm_ga["snapshot_status"], "recipe_only");
}
