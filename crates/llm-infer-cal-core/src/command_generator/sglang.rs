use crate::architecture::profile::ArchitectureProfile;
use crate::command_generator::{
    entry_has_flag, format_gib, needs_trust_remote_code, nonempty, render_flag, Parallelism,
};
use crate::engine_compat::EngineCompatEntry;

#[allow(clippy::too_many_arguments)]
pub fn generate_sglang_command(
    model_id: &str,
    profile: &ArchitectureProfile,
    parallelism: Parallelism,
    entry: Option<&EngineCompatEntry>,
    max_model_len: Option<u64>,
    max_concurrent_requests: Option<u64>,
    cpu_offload_gb: Option<f64>,
    speculative_draft_model_id: Option<&str>,
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

    if let Some(max_concurrent) = max_concurrent_requests.filter(|value| *value > 0) {
        if !entry_has_flag(entry, "--max-running-requests") {
            lines.push(format!("  --max-running-requests {max_concurrent}"));
        }
    }

    if needs_trust_remote_code(&profile.model_type) {
        lines.push("  --trust-remote-code".to_string());
    }

    if !entry_has_flag(entry, "--mem-fraction-static") {
        lines.push("  --mem-fraction-static 0.9".to_string());
    }
    if let Some(cpu_offload_gb) = cpu_offload_gb.filter(|value| *value > 0.0) {
        if !entry_has_flag(entry, "--cpu-offload-gb") {
            lines.push(format!("  --cpu-offload-gb {}", format_gib(cpu_offload_gb)));
        }
    }
    if let Some(draft_model_id) = nonempty(speculative_draft_model_id) {
        if !entry_has_flag(entry, "--speculative-algorithm") {
            lines.push("  --speculative-algorithm STANDALONE".to_string());
        }
        if !entry_has_flag(entry, "--speculative-draft-model-path") {
            lines.push(format!("  --speculative-draft-model-path {draft_model_id}"));
        }
        if !entry_has_flag(entry, "--speculative-num-steps") {
            lines.push("  --speculative-num-steps 4".to_string());
        }
        if !entry_has_flag(entry, "--speculative-eagle-topk") {
            lines.push("  --speculative-eagle-topk 2".to_string());
        }
        if !entry_has_flag(entry, "--speculative-num-draft-tokens") {
            lines.push("  --speculative-num-draft-tokens 7".to_string());
        }
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
