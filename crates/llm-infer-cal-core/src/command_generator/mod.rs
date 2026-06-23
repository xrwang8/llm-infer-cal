pub mod sglang;
pub mod vllm;

use crate::engine_compat::{EngineCompatEntry, EngineFlag};

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

    pub const fn node_count(self) -> u64 {
        self.pipeline_parallel_size
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

fn needs_trust_remote_code(model_type: &str) -> bool {
    model_type.starts_with("deepseek")
        || model_type.starts_with("glm")
        || model_type.starts_with("qwen2_moe")
        || model_type.starts_with("qwen3_moe")
        || model_type.starts_with("qwen3_5_moe")
        || model_type.starts_with("mixtral")
}
