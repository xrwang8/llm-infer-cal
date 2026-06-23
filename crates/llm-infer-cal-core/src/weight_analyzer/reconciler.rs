use std::cmp::Ordering;

use crate::output::labels::{AnnotatedValue, Label};
use crate::weight_analyzer::fingerprint::QuantFingerprint;
use crate::weight_analyzer::QuantizationScheme;

const TIE_THRESHOLD: f64 = 0.01;
const UNKNOWN_THRESHOLD: f64 = 0.15;

#[derive(Clone, Debug, PartialEq)]
pub struct ReconciliationCandidate {
    pub scheme: QuantizationScheme,
    pub predicted_bytes: u64,
    pub delta_bytes: i128,
    pub relative_error: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReconciliationReport {
    pub observed_bytes: u64,
    pub total_params: u64,
    pub candidates: Vec<ReconciliationCandidate>,
    pub best: AnnotatedValue<QuantizationScheme>,
}

impl ReconciliationReport {
    pub fn summary_line(&self) -> String {
        let Some(candidate) = self.candidates.first() else {
            return format!(
                "{} bytes - no quantization candidates tested",
                self.observed_bytes
            );
        };

        format!(
            "Observed {} bytes. Best match: {} (predicts {} bytes, {:.1}% error)",
            self.observed_bytes,
            candidate.scheme,
            candidate.predicted_bytes,
            candidate.relative_error * 100.0
        )
    }
}

pub fn reconcile(
    observed_bytes: u64,
    total_params: u64,
    fingerprint: Option<&QuantFingerprint>,
) -> ReconciliationReport {
    if observed_bytes == 0 || total_params == 0 {
        return ReconciliationReport {
            observed_bytes,
            total_params,
            candidates: Vec::new(),
            best: AnnotatedValue::new(
                QuantizationScheme::Unknown,
                Label::Unknown,
                Some("observed_bytes or total_params is zero"),
            ),
        };
    }

    let mut candidates = candidates(observed_bytes, total_params);
    candidates.sort_by(|left, right| {
        left.relative_error
            .partial_cmp(&right.relative_error)
            .unwrap_or(Ordering::Equal)
    });

    let argmin_scheme = candidates[0].scheme;
    let argmin_err = candidates[0].relative_error;

    if let Some(fingerprint) = fingerprint {
        return reconcile_with_fingerprint(
            observed_bytes,
            total_params,
            candidates,
            fingerprint,
            argmin_scheme,
            argmin_err,
        );
    }

    if argmin_err > UNKNOWN_THRESHOLD {
        let source = format!(
            "closest candidate ({argmin_scheme}) is off by {:.1}% - no confident match",
            argmin_err * 100.0
        );
        return ReconciliationReport {
            observed_bytes,
            total_params,
            candidates,
            best: AnnotatedValue::new(QuantizationScheme::Unknown, Label::Unknown, Some(&source)),
        };
    }

    let tied_schemes: Vec<QuantizationScheme> = candidates
        .iter()
        .filter(|candidate| {
            (candidate.relative_error - argmin_err).abs() < TIE_THRESHOLD
                && candidate.relative_error <= UNKNOWN_THRESHOLD
        })
        .map(|candidate| candidate.scheme)
        .collect();

    let source = if tied_schemes.len() > 1 {
        let tied = tied_schemes
            .iter()
            .filter(|scheme| **scheme != argmin_scheme)
            .map(|scheme| scheme.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "best match among {} candidates, {:.1}% error - tied with {tied} at the same bits/param; distinguishing requires config.json quantization_config or safetensors per-tensor dtype (neither available for this model)",
            candidates.len(),
            argmin_err * 100.0
        )
    } else {
        format!(
            "best match among {} candidates, {:.1}% error",
            candidates.len(),
            argmin_err * 100.0
        )
    };

    ReconciliationReport {
        observed_bytes,
        total_params,
        candidates,
        best: AnnotatedValue::new(argmin_scheme, Label::Inferred, Some(&source)),
    }
}

fn candidates(observed_bytes: u64, total_params: u64) -> Vec<ReconciliationCandidate> {
    QuantizationScheme::all()
        .into_iter()
        .filter(|scheme| *scheme != QuantizationScheme::Unknown && scheme.bpp() != 0.0)
        .map(|scheme| {
            let predicted_bytes = (scheme.bpp() * total_params as f64) as u64;
            let delta_bytes = observed_bytes as i128 - predicted_bytes as i128;
            let relative_error = if predicted_bytes == 0 {
                f64::INFINITY
            } else {
                delta_bytes.unsigned_abs() as f64 / predicted_bytes as f64
            };
            ReconciliationCandidate {
                scheme,
                predicted_bytes,
                delta_bytes,
                relative_error,
            }
        })
        .collect()
}

fn reconcile_with_fingerprint(
    observed_bytes: u64,
    total_params: u64,
    candidates: Vec<ReconciliationCandidate>,
    fingerprint: &QuantFingerprint,
    argmin_scheme: QuantizationScheme,
    argmin_err: f64,
) -> ReconciliationReport {
    let declared = fingerprint.scheme;
    let matched = candidates
        .iter()
        .find(|candidate| candidate.scheme == declared);

    let Some(matched) = matched else {
        let source = format!(
            "fingerprint declared {declared} ({}) but we have no bpp anchor for it; fell back to bytes match {argmin_scheme} at {:.1}% error",
            fingerprint.evidence,
            argmin_err * 100.0
        );
        return ReconciliationReport {
            observed_bytes,
            total_params,
            candidates,
            best: AnnotatedValue::new(argmin_scheme, Label::Inferred, Some(&source)),
        };
    };

    if matched.relative_error <= UNKNOWN_THRESHOLD {
        let note = if declared != argmin_scheme && argmin_err < matched.relative_error {
            format!(
                " (bytes alone would argmin to {argmin_scheme} at {:.1}%; we trust the declaration)",
                argmin_err * 100.0
            )
        } else {
            String::new()
        };
        let source = format!(
            "{} (predicts {} bytes, {:.1}% error){note}",
            fingerprint.evidence,
            fmt_u64(matched.predicted_bytes),
            matched.relative_error * 100.0
        );
        return ReconciliationReport {
            observed_bytes,
            total_params,
            candidates,
            best: AnnotatedValue::new(declared, Label::Verified, Some(&source)),
        };
    }

    let source = format!(
        "{} (NOTE: bytes predict {}, off by {:.1}% — likely our param estimate is off, not the declaration)",
        fingerprint.evidence,
        fmt_u64(matched.predicted_bytes),
        matched.relative_error * 100.0
    );
    ReconciliationReport {
        observed_bytes,
        total_params,
        candidates,
        best: AnnotatedValue::new(declared, Label::Verified, Some(&source)),
    }
}

fn fmt_u64(value: u64) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    for (idx, ch) in text.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}
