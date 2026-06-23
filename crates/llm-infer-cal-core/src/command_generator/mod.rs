pub mod sglang;
pub mod vllm;

use crate::engine_compat::EngineFlag;

fn render_flag(flag: &EngineFlag) -> String {
    match &flag.value {
        Some(value) => format!("{} {}", flag.flag, value),
        None => flag.flag.clone(),
    }
}

fn needs_trust_remote_code(model_type: &str) -> bool {
    model_type.starts_with("deepseek")
        || model_type.starts_with("qwen2_moe")
        || model_type.starts_with("qwen3_moe")
        || model_type.starts_with("mixtral")
}
