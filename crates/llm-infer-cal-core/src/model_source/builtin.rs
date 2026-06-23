use serde::Deserialize;
use serde_json::Value;

use crate::model_source::base::{
    ModelArtifact, ModelNotFoundError, ModelSource, ModelSourceError, SiblingFile,
    SourceUnavailableError,
};

const CATALOG_JSON: &str = include_str!("../../data/builtin_models.json");

#[derive(Clone, Debug, Default)]
pub struct BuiltinSource;

#[derive(Debug, Deserialize)]
struct BuiltinCatalog {
    models: Vec<BuiltinModel>,
}

#[derive(Debug, Deserialize)]
struct BuiltinModel {
    id: String,
    #[serde(default)]
    aliases: Vec<String>,
    commit_sha: Option<String>,
    config: Value,
    siblings: Vec<SiblingFile>,
}

impl BuiltinSource {
    pub fn fetch(&self, model_id: &str) -> Result<ModelArtifact, ModelSourceError> {
        let catalog = load_catalog()?;
        let Some(model) = catalog
            .models
            .into_iter()
            .find(|model| model.id == model_id || model.aliases.iter().any(|id| id == model_id))
        else {
            return Err(ModelNotFoundError(format!(
                "Model '{model_id}' is not in the built-in catalog."
            ))
            .into());
        };

        Ok(ModelArtifact {
            source: "builtin".to_string(),
            model_id: model.id,
            commit_sha: model.commit_sha,
            config: model.config,
            siblings: model.siblings,
        })
    }
}

impl ModelSource for BuiltinSource {
    fn name(&self) -> &str {
        "builtin"
    }

    fn fetch(&self, model_id: &str) -> Result<ModelArtifact, ModelSourceError> {
        BuiltinSource::fetch(self, model_id)
    }
}

fn load_catalog() -> Result<BuiltinCatalog, ModelSourceError> {
    serde_json::from_str(CATALOG_JSON).map_err(|error| {
        SourceUnavailableError(format!("Built-in model catalog is invalid: {error}")).into()
    })
}
