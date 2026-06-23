use crate::architecture::profile::ArchitectureProfile;
use crate::command_generator::{needs_trust_remote_code, render_flag};
use crate::engine_compat::EngineCompatEntry;

pub fn generate_vllm_command(
    model_id: &str,
    profile: &ArchitectureProfile,
    tensor_parallel_size: u64,
    entry: Option<&EngineCompatEntry>,
    max_model_len: Option<u64>,
) -> String {
    let mut lines = vec![
        format!("vllm serve {model_id}"),
        format!("  --tensor-parallel-size {tensor_parallel_size}"),
    ];

    let effective_max = max_model_len.or_else(|| {
        profile
            .position
            .as_ref()
            .and_then(|position| position.max_position_embeddings)
    });
    if let Some(max) = effective_max {
        lines.push(format!("  --max-model-len {max}"));
    }

    if needs_trust_remote_code(&profile.model_type) {
        lines.push("  --trust-remote-code".to_string());
    }

    lines.push("  --gpu-memory-utilization 0.9".to_string());

    if let Some(entry) = entry {
        for flag in &entry.required_flags {
            lines.push(format!("  {}", render_flag(flag)));
        }
        for flag in &entry.optional_flags {
            lines.push(format!("  {}", render_flag(flag)));
        }
    }

    lines.join(" \\\n")
}
