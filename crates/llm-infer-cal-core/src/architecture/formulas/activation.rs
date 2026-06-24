use crate::architecture::profile::{ArchitectureProfile, Confidence, Family};
use crate::output::labels::{AnnotatedValue, Label};

pub const DEFAULT_BATCHED_TOKENS: u64 = 2048;
const ACTIVATION_FACTOR: u64 = 2;
const DEFAULT_DTYPE_BYTES: u64 = 2;

pub fn compute_activation_bytes(
    profile: &ArchitectureProfile,
    batched_tokens: u64,
) -> AnnotatedValue<u64> {
    if batched_tokens == 0 {
        return AnnotatedValue::new(0, Label::Estimated, Some("batched_tokens <= 0"));
    }

    if profile.family == Family::StateSpace {
        return AnnotatedValue::new(
            0,
            Label::Unknown,
            Some("state-space model activation memory is not estimated"),
        );
    }

    if profile.family == Family::Unknown || profile.confidence == Confidence::Low {
        return AnnotatedValue::new(
            0,
            Label::Unknown,
            Some("unknown architecture - cannot estimate activation memory"),
        );
    }

    if profile.hidden_size == 0 {
        return AnnotatedValue::new(
            0,
            Label::Unknown,
            Some("missing hidden_size for activation estimate"),
        );
    }

    let base = batched_tokens * profile.hidden_size * DEFAULT_DTYPE_BYTES * ACTIVATION_FACTOR;
    let (value, note) = if let Some(moe) = &profile.moe {
        let total_experts = moe.num_routed_experts;
        let active_experts = moe.num_experts_per_tok;
        let factor = if total_experts > 0 {
            1.0 + (active_experts as f64 / total_experts as f64) * 0.5
        } else {
            1.0
        };
        (
            (base as f64 * factor) as u64,
            format!(
                "{batched_tokens} batched_tokens * {} hidden * {DEFAULT_DTYPE_BYTES}B * {ACTIVATION_FACTOR} activation_factor * MoE({active_experts}/{total_experts} routing -> {factor:.3})",
                profile.hidden_size,
            ),
        )
    } else {
        (
            base,
            format!(
                "{batched_tokens} batched_tokens * {} hidden * {DEFAULT_DTYPE_BYTES}B * {ACTIVATION_FACTOR} activation_factor",
                profile.hidden_size
            ),
        )
    };

    AnnotatedValue::new(value, Label::Estimated, Some(&note))
}
