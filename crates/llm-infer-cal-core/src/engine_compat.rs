pub mod loader;

use serde::Deserialize;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub struct EngineFlag {
    pub flag: String,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub note_en: Option<String>,
    #[serde(default)]
    pub note_zh: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub struct EngineSource {
    #[serde(rename = "type")]
    pub source_type: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub captured_date: Option<String>,
    #[serde(default)]
    pub note_en: Option<String>,
    #[serde(default)]
    pub note_zh: Option<String>,
    #[serde(default)]
    pub tester: Option<String>,
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub hardware: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub struct EngineCompatEntry {
    #[serde(default)]
    pub engine: String,
    #[serde(default)]
    pub version_spec: String,
    #[serde(default)]
    pub matches_model_type: String,
    #[serde(default)]
    pub support: String,
    #[serde(default)]
    pub verification_level: String,
    #[serde(default)]
    pub required_flags: Vec<EngineFlag>,
    #[serde(default)]
    pub optional_flags: Vec<EngineFlag>,
    #[serde(default)]
    pub sources: Vec<EngineSource>,
    #[serde(default)]
    pub caveats_en: Vec<String>,
    #[serde(default)]
    pub caveats_zh: Vec<String>,
}
