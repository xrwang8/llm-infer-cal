pub mod sglang;
pub mod vllm;

use crate::engine_compat::{EngineCompatEntry, EngineFlag};

const DEFAULT_GPUS_PER_NODE: u64 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Parallelism {
    pub total_gpus: u64,
    pub tensor_parallel_size: u64,
    pub pipeline_parallel_size: u64,
}

impl Parallelism {
    pub const fn single(tensor_parallel_size: u64) -> Self {
        Self {
            total_gpus: tensor_parallel_size,
            tensor_parallel_size,
            pipeline_parallel_size: 1,
        }
    }

    pub fn node_count(self) -> u64 {
        self.total_gpus
            .div_ceil(DEFAULT_GPUS_PER_NODE)
            .max(self.pipeline_parallel_size)
            .max(1)
    }
}

fn render_flag(flag: &EngineFlag) -> String {
    match &flag.value {
        Some(value) => format!("{} {}", flag.flag, value),
        None => flag.flag.clone(),
    }
}

fn entry_has_flag(entry: Option<&EngineCompatEntry>, flag: &str) -> bool {
    entry.is_some_and(|entry| {
        entry
            .required_flags
            .iter()
            .chain(entry.optional_flags.iter())
            .any(|engine_flag| engine_flag.flag == flag)
    })
}

fn format_gib(value: f64) -> String {
    let mut text = format!("{value:.3}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn needs_trust_remote_code(model_type: &str) -> bool {
    model_type.starts_with("deepseek")
        || model_type.starts_with("glm")
        || model_type.starts_with("qwen2_moe")
        || model_type.starts_with("qwen3_moe")
        || model_type.starts_with("qwen3_5_moe")
        || model_type.starts_with("mixtral")
}
