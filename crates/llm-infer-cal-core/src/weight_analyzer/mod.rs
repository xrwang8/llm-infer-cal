pub mod fingerprint;
pub mod reconciler;
pub mod safetensors_reader;

use std::fmt;

use crate::model_source::base::SiblingFile;
use crate::output::labels::{AnnotatedValue, Label};
use crate::weight_analyzer::fingerprint::QuantFingerprint;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum QuantizationScheme {
    Fp16,
    Bf16,
    Fp8,
    Int8,
    Fp4Fp8Mixed,
    Int4,
    GptqInt4,
    AwqInt4,
    Unknown,
}

impl QuantizationScheme {
    pub const fn all() -> [Self; 9] {
        [
            Self::Fp16,
            Self::Bf16,
            Self::Fp8,
            Self::Int8,
            Self::Fp4Fp8Mixed,
            Self::Int4,
            Self::GptqInt4,
            Self::AwqInt4,
            Self::Unknown,
        ]
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp16 => "FP16",
            Self::Bf16 => "BF16",
            Self::Fp8 => "FP8",
            Self::Int8 => "INT8",
            Self::Fp4Fp8Mixed => "FP4_FP8_MIXED",
            Self::Int4 => "INT4",
            Self::GptqInt4 => "GPTQ_INT4",
            Self::AwqInt4 => "AWQ_INT4",
            Self::Unknown => "UNKNOWN",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "FP16" => Some(Self::Fp16),
            "BF16" => Some(Self::Bf16),
            "FP8" => Some(Self::Fp8),
            "INT8" => Some(Self::Int8),
            "FP4_FP8_MIXED" => Some(Self::Fp4Fp8Mixed),
            "INT4" => Some(Self::Int4),
            "GPTQ_INT4" => Some(Self::GptqInt4),
            "AWQ_INT4" => Some(Self::AwqInt4),
            "UNKNOWN" => Some(Self::Unknown),
            _ => None,
        }
    }

    pub const fn bpp(self) -> f64 {
        match self {
            Self::Fp16 | Self::Bf16 => 2.0,
            Self::Fp8 | Self::Int8 => 1.0,
            Self::Fp4Fp8Mixed | Self::GptqInt4 | Self::AwqInt4 => 0.55,
            Self::Int4 => 0.5,
            Self::Unknown => 0.0,
        }
    }

    pub const fn bpp_ratio(self) -> Option<(u64, u64)> {
        match self {
            Self::Fp16 | Self::Bf16 => Some((2, 1)),
            Self::Fp8 | Self::Int8 => Some((1, 1)),
            Self::Fp4Fp8Mixed | Self::GptqInt4 | Self::AwqInt4 => Some((55, 100)),
            Self::Int4 => Some((1, 2)),
            Self::Unknown => None,
        }
    }
}

impl fmt::Display for QuantizationScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightReport {
    pub total_bytes: AnnotatedValue<u64>,
    pub bits_per_param: Option<AnnotatedValue<f64>>,
    pub quantization_guess: AnnotatedValue<QuantizationScheme>,
}

pub fn analyze(
    siblings: &[SiblingFile],
    total_params: Option<u64>,
    fingerprint: Option<&QuantFingerprint>,
) -> WeightReport {
    let observed_bytes = safetensors_total_bytes(siblings);
    let total_bytes = AnnotatedValue::new(
        observed_bytes,
        Label::Verified,
        Some("sum of safetensors siblings from model_info API"),
    );

    let Some(total_params) = total_params.filter(|params| *params > 0) else {
        return unknown_weight_report(total_bytes);
    };
    if observed_bytes == 0 {
        return unknown_weight_report(total_bytes);
    }

    let bpp = observed_bytes as f64 / total_params as f64;
    let bits_per_param = Some(AnnotatedValue::new(
        bpp * 8.0,
        Label::Inferred,
        Some(&format!("{observed_bytes} bytes / {total_params} params")),
    ));

    let quantization_guess = if let Some(fingerprint) = fingerprint {
        AnnotatedValue::new(
            fingerprint.scheme,
            Label::Verified,
            Some(&fingerprint.evidence),
        )
    } else {
        guess_quantization(bpp)
    };

    WeightReport {
        total_bytes,
        bits_per_param,
        quantization_guess,
    }
}

fn safetensors_total_bytes(siblings: &[SiblingFile]) -> u64 {
    siblings
        .iter()
        .filter(|sibling| sibling.filename.ends_with(".safetensors"))
        .map(|sibling| sibling.size.unwrap_or(0))
        .sum()
}

fn unknown_weight_report(total_bytes: AnnotatedValue<u64>) -> WeightReport {
    WeightReport {
        total_bytes,
        bits_per_param: None,
        quantization_guess: AnnotatedValue::new(
            QuantizationScheme::Unknown,
            Label::Unknown,
            Some("total_params unknown or no safetensors files"),
        ),
    }
}

fn guess_quantization(bpp: f64) -> AnnotatedValue<QuantizationScheme> {
    let candidates = [
        (
            QuantizationScheme::Fp16,
            QuantizationScheme::Fp16.bpp(),
            0.05,
        ),
        (QuantizationScheme::Fp8, QuantizationScheme::Fp8.bpp(), 0.05),
        (
            QuantizationScheme::Fp4Fp8Mixed,
            QuantizationScheme::Fp4Fp8Mixed.bpp(),
            0.10,
        ),
        (
            QuantizationScheme::Int4,
            QuantizationScheme::Int4.bpp(),
            0.05,
        ),
        (
            QuantizationScheme::GptqInt4,
            QuantizationScheme::GptqInt4.bpp(),
            0.10,
        ),
    ];

    let mut best: Option<(QuantizationScheme, f64)> = None;
    for (scheme, anchor_bpp, tolerance) in candidates {
        let delta = (bpp - anchor_bpp).abs();
        if delta <= tolerance && best.map_or(true, |(_, best_delta)| delta < best_delta) {
            best = Some((scheme, delta));
        }
    }

    let Some((scheme, _)) = best else {
        let source = format!("bits/param {:.2} does not match known schemes", bpp * 8.0);
        return AnnotatedValue::new(QuantizationScheme::Unknown, Label::Unknown, Some(&source));
    };

    let source = format!("bits/param {:.2} within tolerance of {scheme}", bpp * 8.0);
    AnnotatedValue::new(scheme, Label::Inferred, Some(&source))
}
