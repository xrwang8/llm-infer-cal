use serde::Deserialize;

use crate::engine_compat::EngineCompatEntry;

const MATRIX_YAML: &str = include_str!("../../../../src/llm_cal/engine_compat/matrix.yaml");

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct EngineCompatMatrix {
    pub schema_version: u64,
    pub entries: Vec<EngineCompatEntry>,
}

pub fn load_matrix() -> Result<EngineCompatMatrix, serde_yaml::Error> {
    serde_yaml::from_str(MATRIX_YAML)
}

pub fn find_match(
    engine: &str,
    model_type: &str,
    version: Option<&str>,
    matrix: Option<&EngineCompatMatrix>,
) -> Option<EngineCompatEntry> {
    let owned_matrix;
    let matrix = if let Some(matrix) = matrix {
        matrix
    } else {
        owned_matrix = load_matrix().ok()?;
        &owned_matrix
    };

    let engine_norm = engine.trim().to_lowercase();
    let model_type_norm = model_type.trim().to_lowercase();
    let candidates = matrix
        .entries
        .iter()
        .filter(|entry| entry.engine == engine_norm && entry.matches_model_type == model_type_norm)
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return None;
    }

    let Some(version) = version else {
        return candidates
            .into_iter()
            .max_by_key(|entry| lower_bound_key(&entry.version_spec))
            .cloned();
    };

    let Some(version) = Version::parse(version) else {
        return candidates.first().map(|entry| (*entry).clone());
    };

    candidates
        .into_iter()
        .find(|entry| version_matches(&version, &entry.version_spec))
        .cloned()
}

fn version_matches(version: &Version, spec: &str) -> bool {
    spec.split(',').all(|part| {
        let part = part.trim();
        if part.is_empty() {
            return true;
        }

        for op in [">=", "<=", "==", ">", "<"] {
            if let Some(raw) = part.strip_prefix(op) {
                let Some(required) = Version::parse(raw.trim()) else {
                    return false;
                };
                return match op {
                    ">=" => version >= &required,
                    "<=" => version <= &required,
                    "==" => version == &required,
                    ">" => version > &required,
                    "<" => version < &required,
                    _ => false,
                };
            }
        }
        false
    })
}

fn lower_bound_key(spec: &str) -> Version {
    for part in spec.split(',') {
        let part = part.trim();
        for op in [">=", "==", ">"] {
            if let Some(raw) = part.strip_prefix(op) {
                if let Some(version) = Version::parse(raw.trim()) {
                    return version;
                }
            }
        }
    }
    Version::default()
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct Version {
    major: u64,
    minor: u64,
    patch: u64,
}

impl Version {
    fn parse(raw: &str) -> Option<Self> {
        let raw = raw.trim().trim_start_matches('v');
        let mut parts = raw.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next().unwrap_or("0").parse().ok()?;
        let patch = parts.next().unwrap_or("0").parse().ok()?;
        Some(Self {
            major,
            minor,
            patch,
        })
    }
}
