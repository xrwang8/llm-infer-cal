use std::collections::HashMap;

use serde_json::Value;

use crate::weight_analyzer::QuantizationScheme;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceType {
    ConfigJson,
    SafetensorsHeader,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantFingerprint {
    pub scheme: QuantizationScheme,
    pub source_type: SourceType,
    pub evidence: String,
}

pub fn from_config(config: &Value) -> Option<QuantFingerprint> {
    let qc = config.get("quantization_config")?.as_object()?;
    let quant_method = qc.get("quant_method").and_then(Value::as_str);
    let bits = qc.get("bits").and_then(Value::as_i64);
    let weight_dtype = qc.get("weight_dtype").and_then(Value::as_str);

    if quant_method == Some("gptq") {
        if bits == Some(4) {
            return Some(fingerprint(
                QuantizationScheme::GptqInt4,
                SourceType::ConfigJson,
                "config.json quantization_config.quant_method=gptq, bits=4",
            ));
        }
        if bits == Some(8) {
            return Some(fingerprint(
                QuantizationScheme::Int8,
                SourceType::ConfigJson,
                "config.json quantization_config.quant_method=gptq, bits=8",
            ));
        }
    }

    if quant_method == Some("awq") && bits == Some(4) {
        return Some(fingerprint(
            QuantizationScheme::AwqInt4,
            SourceType::ConfigJson,
            "config.json quantization_config.quant_method=awq, bits=4",
        ));
    }

    if quant_method == Some("fp8") {
        return Some(fingerprint(
            QuantizationScheme::Fp8,
            SourceType::ConfigJson,
            "config.json quantization_config.quant_method=fp8",
        ));
    }

    if quant_method == Some("compressed-tensors") {
        if let Some(groups) = qc.get("config_groups").and_then(Value::as_object) {
            for group in groups.values() {
                let Some(group) = group.as_object() else {
                    continue;
                };
                let weights = group.get("weights").and_then(Value::as_object);
                let num_bits = weights
                    .and_then(|weights| weights.get("num_bits"))
                    .and_then(Value::as_i64);
                let weight_type = weights
                    .and_then(|weights| weights.get("type"))
                    .and_then(Value::as_str);

                if num_bits == Some(8) && matches!(weight_type, Some("float" | "fp8")) {
                    return Some(fingerprint(
                        QuantizationScheme::Fp8,
                        SourceType::ConfigJson,
                        "config.json compressed-tensors group weights=fp8/8bit",
                    ));
                }
                if num_bits == Some(8) && weight_type == Some("int") {
                    return Some(fingerprint(
                        QuantizationScheme::Int8,
                        SourceType::ConfigJson,
                        "config.json compressed-tensors group weights=int/8bit",
                    ));
                }
                if num_bits == Some(4) && weight_type == Some("int") {
                    return Some(fingerprint(
                        QuantizationScheme::Int4,
                        SourceType::ConfigJson,
                        "config.json compressed-tensors group weights=int/4bit",
                    ));
                }
                break;
            }
        }
    }

    if quant_method == Some("bitsandbytes") {
        if qc
            .get("load_in_4bit")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Some(fingerprint(
                QuantizationScheme::Int4,
                SourceType::ConfigJson,
                "config.json quant_method=bitsandbytes, load_in_4bit=true",
            ));
        }
        if qc
            .get("load_in_8bit")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Some(fingerprint(
                QuantizationScheme::Int8,
                SourceType::ConfigJson,
                "config.json quant_method=bitsandbytes, load_in_8bit=true",
            ));
        }
    }

    if matches!(weight_dtype, Some("float8_e4m3fn" | "float8_e5m2")) {
        let evidence = format!(
            "config.json quantization_config.weight_dtype={}",
            weight_dtype.unwrap()
        );
        return Some(QuantFingerprint {
            scheme: QuantizationScheme::Fp8,
            source_type: SourceType::ConfigJson,
            evidence,
        });
    }

    None
}

pub fn from_safetensors_dtypes(
    tensor_dtypes: &HashMap<String, String>,
) -> Option<QuantFingerprint> {
    if tensor_dtypes.is_empty() {
        return None;
    }

    let has_qweight = tensor_dtypes
        .keys()
        .any(|name| name.ends_with(".qweight") || name.ends_with("_qweight"));
    let has_g_idx = tensor_dtypes
        .keys()
        .any(|name| name.ends_with(".g_idx") || name.ends_with("_g_idx"));
    let has_qzeros = tensor_dtypes
        .keys()
        .any(|name| name.ends_with(".qzeros") || name.ends_with("_qzeros"));

    if has_qweight && has_g_idx {
        return Some(fingerprint(
            QuantizationScheme::GptqInt4,
            SourceType::SafetensorsHeader,
            "safetensors header has .qweight + .g_idx tensors (GPTQ marker)",
        ));
    }
    if has_qweight && has_qzeros && !has_g_idx {
        return Some(fingerprint(
            QuantizationScheme::AwqInt4,
            SourceType::SafetensorsHeader,
            "safetensors header has .qweight + .qzeros, no .g_idx (AWQ marker)",
        ));
    }

    let mut weight_dtypes: Vec<&str> = tensor_dtypes
        .iter()
        .filter(|(name, _)| is_weight_tensor(name))
        .map(|(_, dtype)| dtype.as_str())
        .collect();
    if weight_dtypes.is_empty() {
        weight_dtypes = tensor_dtypes.values().map(String::as_str).collect();
    }

    let has_fp4 = weight_dtypes.iter().any(|dtype| is_fp4(dtype));
    let has_fp8 = weight_dtypes.iter().any(|dtype| is_fp8(dtype));
    let has_fp16 = weight_dtypes.contains(&"F16");
    let has_bf16 = weight_dtypes.contains(&"BF16");
    let has_int8 = weight_dtypes.iter().any(|dtype| is_int8(dtype));
    let has_mx_scale = tensor_dtypes.values().any(|dtype| dtype == "F8_E8M0");

    if has_mx_scale && has_int8 {
        let int8_count = weight_dtypes.iter().filter(|dtype| is_int8(dtype)).count();
        if has_fp8 {
            let fp8_count = weight_dtypes.iter().filter(|dtype| is_fp8(dtype)).count();
            let evidence = format!(
                "safetensors header: F8_E8M0 scale tensors + {int8_count} packed-I8 (FP4) weights + {fp8_count} FP8 weights — MX block-scaled mixed pack"
            );
            return Some(QuantFingerprint {
                scheme: QuantizationScheme::Fp4Fp8Mixed,
                source_type: SourceType::SafetensorsHeader,
                evidence,
            });
        }
        let evidence = format!(
            "safetensors header: F8_E8M0 scale tensors + {int8_count} packed-I8 (FP4) weights — MXFP4 block-scaled"
        );
        return Some(QuantFingerprint {
            scheme: QuantizationScheme::Fp4Fp8Mixed,
            source_type: SourceType::SafetensorsHeader,
            evidence,
        });
    }

    if has_fp4 && has_fp8 {
        let fp4_count = weight_dtypes.iter().filter(|dtype| is_fp4(dtype)).count();
        let fp8_count = weight_dtypes.iter().filter(|dtype| is_fp8(dtype)).count();
        let evidence = format!(
            "safetensors header has both FP4 and FP8 weight tensors ({fp4_count} FP4, {fp8_count} FP8)"
        );
        return Some(QuantFingerprint {
            scheme: QuantizationScheme::Fp4Fp8Mixed,
            source_type: SourceType::SafetensorsHeader,
            evidence,
        });
    }

    if has_fp8 && !(has_fp4 || has_int8) {
        let fp8_count = weight_dtypes.iter().filter(|dtype| is_fp8(dtype)).count();
        let evidence = format!(
            "safetensors header: {fp8_count}/{} weight tensors are FP8",
            weight_dtypes.len()
        );
        return Some(QuantFingerprint {
            scheme: QuantizationScheme::Fp8,
            source_type: SourceType::SafetensorsHeader,
            evidence,
        });
    }

    if has_fp16 && !(has_fp8 || has_fp4 || has_int8 || has_bf16) {
        let evidence = format!(
            "safetensors header: all {} weight tensors are F16",
            weight_dtypes.len()
        );
        return Some(QuantFingerprint {
            scheme: QuantizationScheme::Fp16,
            source_type: SourceType::SafetensorsHeader,
            evidence,
        });
    }

    if has_bf16 && !(has_fp8 || has_fp4 || has_int8 || has_fp16) {
        let evidence = format!(
            "safetensors header: all {} weight tensors are BF16",
            weight_dtypes.len()
        );
        return Some(QuantFingerprint {
            scheme: QuantizationScheme::Bf16,
            source_type: SourceType::SafetensorsHeader,
            evidence,
        });
    }

    if has_int8 && !(has_fp8 || has_fp4 || has_fp16 || has_bf16) {
        let evidence = format!(
            "safetensors header: {} weight tensors are INT8",
            weight_dtypes.len()
        );
        return Some(QuantFingerprint {
            scheme: QuantizationScheme::Int8,
            source_type: SourceType::SafetensorsHeader,
            evidence,
        });
    }

    None
}

fn fingerprint(
    scheme: QuantizationScheme,
    source_type: SourceType,
    evidence: &str,
) -> QuantFingerprint {
    QuantFingerprint {
        scheme,
        source_type,
        evidence: evidence.to_string(),
    }
}

fn is_weight_tensor(name: &str) -> bool {
    let name = name.to_lowercase();
    if [".norm", ".bias", "embed", "lm_head"]
        .iter()
        .any(|excluded| name.contains(excluded))
    {
        return false;
    }
    name.contains("weight") || name.ends_with(".w") || name.ends_with(".proj")
}

fn is_fp8(dtype: &str) -> bool {
    matches!(dtype, "F8_E4M3" | "F8_E5M2")
}

fn is_fp4(dtype: &str) -> bool {
    matches!(dtype, "F4_E2M1" | "F4")
}

fn is_int8(dtype: &str) -> bool {
    matches!(dtype, "I8" | "U8")
}
