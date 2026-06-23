use std::collections::HashMap;

use serde_json::Value;

use crate::model_source::auth::{get_hf_token, hf_auth_error_message};
use crate::model_source::base::{
    AuthRequiredError, ModelArtifact, ModelNotFoundError, ModelSource, ModelSourceError,
    SiblingFile, SourceUnavailableError,
};
use crate::model_source::modelscope::{HttpClient, HttpResponse, ReqwestHttpClient};

pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

#[derive(Clone, Debug)]
pub struct HuggingFaceSource {
    endpoint: String,
    timeout_s: f64,
}

impl Default for HuggingFaceSource {
    fn default() -> Self {
        Self::new(None, 30.0)
    }
}

impl HuggingFaceSource {
    pub fn new(endpoint: Option<&str>, timeout_s: f64) -> Self {
        Self {
            endpoint: endpoint
                .unwrap_or(DEFAULT_ENDPOINT)
                .trim_end_matches('/')
                .to_string(),
            timeout_s,
        }
    }

    pub fn fetch(&self, model_id: &str) -> Result<ModelArtifact, ModelSourceError> {
        let token = get_hf_token();
        let mut client = ReqwestHttpClient;
        self.fetch_with_client(model_id, &mut client, token.as_deref())
    }

    pub fn fetch_with_client(
        &self,
        model_id: &str,
        client: &mut dyn HttpClient,
        token: Option<&str>,
    ) -> Result<ModelArtifact, ModelSourceError> {
        let headers = auth_headers(token);
        let info = self.fetch_model_info(model_id, client, &headers)?;
        let commit_sha = info
            .get("sha")
            .and_then(Value::as_str)
            .filter(|sha| !sha.is_empty())
            .map(str::to_string);
        let siblings = siblings_from_info(&info);
        let config = self.fetch_config(
            model_id,
            commit_sha.as_deref().unwrap_or("main"),
            client,
            &headers,
        )?;

        Ok(ModelArtifact {
            source: "huggingface".to_string(),
            model_id: model_id.to_string(),
            commit_sha,
            config,
            siblings,
        })
    }

    fn fetch_model_info(
        &self,
        model_id: &str,
        client: &mut dyn HttpClient,
        headers: &HashMap<String, String>,
    ) -> Result<Value, ModelSourceError> {
        let url = format!("{}/api/models/{model_id}", self.endpoint);
        let params = vec![("blobs".to_string(), "true".to_string())];
        let response = client
            .get(&url, headers, &params, self.timeout_s)
            .map_err(|error| {
                SourceUnavailableError(format!(
                    "HuggingFace request timed out after {}s: {}",
                    self.timeout_s, error.0
                ))
            })?;
        self.raise_info_status(&response, model_id)?;
        serde_json::from_slice(&response.body).map_err(|error| {
            SourceUnavailableError(format!("HuggingFace model_info returned non-JSON: {error}"))
                .into()
        })
    }

    fn fetch_config(
        &self,
        model_id: &str,
        revision: &str,
        client: &mut dyn HttpClient,
        headers: &HashMap<String, String>,
    ) -> Result<Value, ModelSourceError> {
        let url = format!(
            "{}/{model_id}/resolve/{revision}/config.json",
            self.endpoint
        );
        let response = client
            .get(&url, headers, &[], self.timeout_s)
            .map_err(|error| {
                SourceUnavailableError(format!("config.json fetch failed: {}", error.0))
            })?;

        self.raise_config_status(&response, model_id)?;
        let parsed: Value = serde_json::from_slice(&response.body).map_err(|error| {
            SourceUnavailableError(format!("config.json is not valid JSON: {error}"))
        })?;
        if !parsed.is_object() {
            return Err(SourceUnavailableError(
                "config.json did not parse to a JSON object.".to_string(),
            )
            .into());
        }
        Ok(parsed)
    }

    fn raise_info_status(
        &self,
        response: &HttpResponse,
        model_id: &str,
    ) -> Result<(), ModelSourceError> {
        match response.status {
            200 => Ok(()),
            404 => Err(
                ModelNotFoundError(format!("Model '{model_id}' not found on HuggingFace.")).into(),
            ),
            401 | 403 => Err(AuthRequiredError(hf_auth_error_message(model_id)).into()),
            429 => {
                let retry = retry_after(response);
                Err(SourceUnavailableError(format!(
                    "HuggingFace rate limit (429). Retry-After: {retry}s. Setting HF_TOKEN increases your quota."
                ))
                .into())
            }
            status => Err(SourceUnavailableError(format!("HuggingFace error ({status})")).into()),
        }
    }

    fn raise_config_status(
        &self,
        response: &HttpResponse,
        model_id: &str,
    ) -> Result<(), ModelSourceError> {
        match response.status {
            0..=399 => Ok(()),
            404 => Err(ModelNotFoundError(format!(
                "Model '{model_id}' exists but has no config.json. May be a GGUF-only or dataset repo (not supported in v0.1)."
            ))
            .into()),
            401 | 403 => Err(AuthRequiredError(hf_auth_error_message(model_id)).into()),
            429 => {
                let retry = retry_after(response);
                Err(SourceUnavailableError(format!(
                    "HuggingFace rate limit (429). Retry-After: {retry}s."
                ))
                .into())
            }
            status => Err(SourceUnavailableError(format!(
                "config.json fetch returned HTTP {status}"
            ))
            .into()),
        }
    }
}

impl ModelSource for HuggingFaceSource {
    fn name(&self) -> &str {
        "huggingface"
    }

    fn fetch(&self, model_id: &str) -> Result<ModelArtifact, ModelSourceError> {
        HuggingFaceSource::fetch(self, model_id)
    }
}

fn siblings_from_info(info: &Value) -> Vec<SiblingFile> {
    info.get("siblings")
        .and_then(Value::as_array)
        .map(|siblings| {
            siblings
                .iter()
                .filter_map(|sibling| {
                    let object = sibling.as_object()?;
                    let filename = object
                        .get("rfilename")
                        .or_else(|| object.get("filename"))
                        .and_then(Value::as_str)?
                        .to_string();
                    let size = object.get("size").and_then(value_to_u64);
                    Some(SiblingFile { filename, size })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn auth_headers(token: Option<&str>) -> HashMap<String, String> {
    token
        .filter(|token| !token.is_empty())
        .map(|token| HashMap::from([("Authorization".to_string(), format!("Bearer {token}"))]))
        .unwrap_or_default()
}

fn retry_after(response: &HttpResponse) -> &str {
    response
        .headers
        .get("Retry-After")
        .or_else(|| response.headers.get("retry-after"))
        .map(String::as_str)
        .unwrap_or("unknown")
}

fn value_to_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}
