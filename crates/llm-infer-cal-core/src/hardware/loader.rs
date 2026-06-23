use std::error::Error;
use std::fmt;

use serde::Deserialize;

const GPU_DATABASE_YAML: &str = include_str!("../../../../data/hardware/gpu_database.yaml");

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GPUSpec {
    pub id: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub memory_gb: u64,
    pub nvlink_bandwidth_gbps: u64,
    #[serde(default)]
    pub memory_bandwidth_gbps: Option<u64>,
    pub fp16_tflops: f64,
    pub fp8_support: bool,
    pub fp4_support: bool,
    #[serde(default)]
    pub notes_en: Option<String>,
    #[serde(default)]
    pub notes_zh: Option<String>,
    #[serde(default)]
    pub spec_source: Option<String>,
}

impl GPUSpec {
    pub fn localized_notes(&self, locale: &str) -> Option<&str> {
        if locale == "zh" {
            self.notes_zh.as_deref().or(self.notes_en.as_deref())
        } else {
            self.notes_en.as_deref().or(self.notes_zh.as_deref())
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GPUDatabase {
    pub schema_version: u64,
    pub gpus: Vec<GPUSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnknownGPUError {
    message: String,
}

impl fmt::Display for UnknownGPUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for UnknownGPUError {}

pub fn load_database() -> Result<GPUDatabase, serde_yaml::Error> {
    serde_yaml::from_str(GPU_DATABASE_YAML)
}

pub fn lookup(gpu: &str) -> Result<GPUSpec, UnknownGPUError> {
    let database = load_database().map_err(|error| UnknownGPUError {
        message: format!("Failed to load GPU database: {error}"),
    })?;
    let target = gpu.trim().to_uppercase();
    for spec in &database.gpus {
        if spec.id.to_uppercase() == target {
            return Ok(spec.clone());
        }
        if spec
            .aliases
            .iter()
            .any(|alias| alias.to_uppercase() == target)
        {
            return Ok(spec.clone());
        }
    }

    if let Some((gpu_id, count)) = target.rsplit_once('X') {
        if !gpu_id.is_empty() && count.chars().all(|ch| ch.is_ascii_digit()) {
            return Err(UnknownGPUError {
                message: format!(
                    "'{gpu}' looks like old 'H800x8' format. Use `--gpu {gpu_id} --gpu-count {count}` instead."
                ),
            });
        }
    }

    let known = database
        .gpus
        .iter()
        .map(|spec| spec.id.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    Err(UnknownGPUError {
        message: format!("Unknown GPU '{gpu}'. Known: {known}"),
    })
}
