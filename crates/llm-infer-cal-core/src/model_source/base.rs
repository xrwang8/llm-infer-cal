use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SiblingFile {
    pub filename: String,
    pub size: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ModelArtifact {
    pub source: String,
    pub model_id: String,
    pub commit_sha: Option<String>,
    pub config: Value,
    pub siblings: Vec<SiblingFile>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelNotFoundError(pub String);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthRequiredError(pub String);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceUnavailableError(pub String);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelSourceError {
    NotFound(ModelNotFoundError),
    AuthRequired(AuthRequiredError),
    SourceUnavailable(SourceUnavailableError),
}

pub trait ModelSource {
    fn name(&self) -> &str;

    fn fetch(&self, model_id: &str) -> Result<ModelArtifact, ModelSourceError>;
}

impl fmt::Display for ModelNotFoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Display for AuthRequiredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Display for SourceUnavailableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ModelNotFoundError {}
impl Error for AuthRequiredError {}
impl Error for SourceUnavailableError {}

impl fmt::Display for ModelSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelSourceError::NotFound(err) => err.fmt(f),
            ModelSourceError::AuthRequired(err) => err.fmt(f),
            ModelSourceError::SourceUnavailable(err) => err.fmt(f),
        }
    }
}

impl Error for ModelSourceError {}

impl From<ModelNotFoundError> for ModelSourceError {
    fn from(value: ModelNotFoundError) -> Self {
        Self::NotFound(value)
    }
}

impl From<AuthRequiredError> for ModelSourceError {
    fn from(value: AuthRequiredError) -> Self {
        Self::AuthRequired(value)
    }
}

impl From<SourceUnavailableError> for ModelSourceError {
    fn from(value: SourceUnavailableError) -> Self {
        Self::SourceUnavailable(value)
    }
}
