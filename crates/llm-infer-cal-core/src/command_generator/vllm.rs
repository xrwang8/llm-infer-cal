use crate::architecture::profile::ArchitectureProfile;
use crate::command_generator::{
    entry_has_flag, format_gib, needs_trust_remote_code, nonempty, render_flag, shell_single_quote,
    Parallelism,
};
use crate::engine_compat::EngineCompatEntry;
use serde_json::json;

#[allow(clippy::too_many_arguments)]
pub fn generate_vllm_command(
    model_id: &str,
    profile: &ArchitectureProfile,
    parallelism: Parallelism,
    entry: Option<&EngineCompatEntry>,
    max_model_len: Option<u64>,
    max_concurrent_requests: Option<u64>,
    cpu_offload_gb: Option<f64>,
    speculative_draft_model_id: Option<&str>,
) -> String {
    generate_vllm_command_with_speculative_tokens(
        model_id,
        profile,
        parallelism,
        entry,
        max_model_len,
        max_concurrent_requests,
        cpu_offload_gb,
        speculative_draft_model_id,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn generate_vllm_command_with_speculative_tokens(
    model_id: &str,
    profile: &ArchitectureProfile,
    parallelism: Parallelism,
    entry: Option<&EngineCompatEntry>,
    max_model_len: Option<u64>,
    max_concurrent_requests: Option<u64>,
    cpu_offload_gb: Option<f64>,
    speculative_draft_model_id: Option<&str>,
    speculative_num_draft_tokens: Option<u64>,
) -> String {
    let mut lines = vec![
        format!("vllm serve {model_id}"),
        format!(
            "  --tensor-parallel-size {}",
            parallelism.tensor_parallel_size
        ),
    ];
    if parallelism.pipeline_parallel_size > 1 {
        lines.push(format!(
            "  --pipeline-parallel-size {}",
            parallelism.pipeline_parallel_size
        ));
    }
    if parallelism.node_count() > 1 {
        lines.push("  --distributed-executor-backend ray".to_string());
    }

    let effective_max = max_model_len.or_else(|| {
        profile
            .position
            .as_ref()
            .and_then(|position| position.max_position_embeddings)
    });
    if let Some(max) = effective_max {
        lines.push(format!("  --max-model-len {max}"));
    }

    if let Some(max_concurrent) = max_concurrent_requests.filter(|value| *value > 0) {
        if !entry_has_flag(entry, "--max-num-seqs") {
            lines.push(format!("  --max-num-seqs {max_concurrent}"));
        }
    }

    if needs_trust_remote_code(&profile.model_type) {
        lines.push("  --trust-remote-code".to_string());
    }

    lines.push("  --gpu-memory-utilization 0.9".to_string());
    if let Some(cpu_offload_gb) = cpu_offload_gb.filter(|value| *value > 0.0) {
        if !entry_has_flag(entry, "--cpu-offload-gb") {
            lines.push(format!("  --cpu-offload-gb {}", format_gib(cpu_offload_gb)));
        }
    }
    if let Some(draft_model_id) = nonempty(speculative_draft_model_id) {
        if !entry_has_flag(entry, "--speculative-config")
            && !entry_has_flag(entry, "--speculative-model")
        {
            let num_speculative_tokens = speculative_num_draft_tokens.unwrap_or(4).max(1);
            let config = json!({
                "model": draft_model_id,
                "num_speculative_tokens": num_speculative_tokens
            });
            lines.push(format!(
                "  --speculative-config {}",
                shell_single_quote(&config.to_string())
            ));
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
