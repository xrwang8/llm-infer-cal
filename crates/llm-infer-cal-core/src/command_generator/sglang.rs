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
    let launch = match entry {
        Some(entry) if !entry.env.is_empty() => {
            let prefix = entry
                .env
                .iter()
                .map(|env| format!("{}={}", env.name, env.value))
                .collect::<Vec<_>>()
                .join(" ");
            format!("{prefix} python -m sglang.launch_server")
        }
        _ => "python -m sglang.launch_server".to_string(),
    };
    let mut lines = vec![
        launch,
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

    if !entry_has_flag(entry, "--mem-fraction-static") {
        lines.push("  --mem-fraction-static 0.9".to_string());
    }

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

fn entry_has_flag(entry: Option<&EngineCompatEntry>, flag: &str) -> bool {
    entry.is_some_and(|entry| {
        entry
            .required_flags
            .iter()
            .chain(entry.optional_flags.iter())
            .any(|engine_flag| engine_flag.flag == flag)
    })
}
