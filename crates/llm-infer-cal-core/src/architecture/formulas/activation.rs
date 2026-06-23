use crate::architecture::profile::{ArchitectureProfile, Confidence, Family};
use crate::output::labels::{AnnotatedValue, Label};

pub fn compute_activation_bytes(
    profile: &ArchitectureProfile,
    seq_len: u64,
) -> AnnotatedValue<u64> {
    if seq_len == 0 {
        return AnnotatedValue::new(0, Label::Estimated, Some("seq_len <= 0"));
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

    let base = seq_len * profile.hidden_size * 2;
    let (value, note) = if let Some(moe) = &profile.moe {
        let total_experts = moe.num_routed_experts + moe.num_shared_experts;
        let active_experts = moe.num_experts_per_tok + moe.num_shared_experts;
        let factor = if total_experts > 0 {
            1.0 + (active_experts as f64 / total_experts as f64) * 0.5
        } else {
            1.0
        };
        (
            (base as f64 * factor) as u64,
            format!(
                "{seq_len} seq * {} hidden * 2B * {factor:.3} MoE routing factor",
                profile.hidden_size
            ),
        )
    } else {
        (
            base,
            format!("{seq_len} seq * {} hidden * 2B", profile.hidden_size),
        )
    };

    AnnotatedValue::new(value, Label::Estimated, Some(&note))
}
