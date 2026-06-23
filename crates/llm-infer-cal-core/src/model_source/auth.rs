use std::collections::HashMap;

pub fn get_hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::env::var("HUGGING_FACE_HUB_TOKEN")
                .ok()
                .filter(|value| !value.is_empty())
        })
}

pub fn get_modelscope_token() -> Option<String> {
    std::env::var("MODELSCOPE_API_TOKEN")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::env::var("MODELSCOPE_TOKEN")
                .ok()
                .filter(|value| !value.is_empty())
        })
}

pub fn get_hf_token_from_values(values: &HashMap<&str, Option<&str>>) -> Option<String> {
    values
        .get("HF_TOKEN")
        .and_then(|value| *value)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            values
                .get("HUGGING_FACE_HUB_TOKEN")
                .and_then(|value| *value)
                .filter(|value| !value.is_empty())
        })
        .map(str::to_string)
}

pub fn get_modelscope_token_from_values(values: &HashMap<&str, Option<&str>>) -> Option<String> {
    values
        .get("MODELSCOPE_API_TOKEN")
        .and_then(|value| *value)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            values
                .get("MODELSCOPE_TOKEN")
                .and_then(|value| *value)
                .filter(|value| !value.is_empty())
        })
        .map(str::to_string)
}

pub fn hf_auth_error_message(model_id: &str) -> String {
    format!(
        "Model '{model_id}' requires authentication (gated or private).\nSet HF_TOKEN env var or run: huggingface-cli login"
    )
}

pub fn modelscope_auth_error_message(model_id: &str) -> String {
    format!(
        "模型 '{model_id}' 需要登录（gated 或 私有）。\n设置 MODELSCOPE_API_TOKEN 环境变量，或执行：modelscope login"
    )
}
