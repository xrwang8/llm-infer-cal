use crate::architecture::profile::ArchitectureProfile;
use crate::command_generator::{needs_trust_remote_code, render_flag, Parallelism};
use crate::engine_compat::EngineCompatEntry;

pub fn generate_sglang_command(
    model_id: &str,
    profile: &ArchitectureProfile,
    parallelism: Parallelism,
    entry: Option<&EngineCompatEntry>,
    max_model_len: Option<u64>,
) -> String {
    let mut lines = vec![
        "python -m sglang.launch_server".to_string(),
        format!("  --model-path {model_id}"),
        format!("  --tp {}", parallelism.total_gpus),
    ];
    if parallelism.node_count() > 1 {
        lines.push("  --dist-init-addr ${NODE0_IP:-<node0-ip>}:20000".to_string());
        lines.push(format!("  --nnodes {}", parallelism.node_count()));
        lines.push("  --node-rank ${NODE_RANK:-0}".to_string());
    }

    let effective_max = max_model_len.or_else(|| {
        profile
            .position
            .as_ref()
            .and_then(|position| position.max_position_embeddings)
    });
    if let Some(max) = effective_max {
        lines.push(format!("  --context-length {max}"));
    }

    if needs_trust_remote_code(&profile.model_type) {
        lines.push("  --trust-remote-code".to_string());
    }

    lines.push("  --mem-fraction-static 0.9".to_string());

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
