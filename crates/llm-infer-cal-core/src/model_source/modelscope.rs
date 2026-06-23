use std::collections::HashMap;
use std::time::Duration;

use serde_json::Value;

use crate::model_source::auth::{get_modelscope_token, modelscope_auth_error_message};
use crate::model_source::base::{
    AuthRequiredError, ModelArtifact, ModelNotFoundError, ModelSource, ModelSourceError,
    SiblingFile, SourceUnavailableError,
};

pub const DEFAULT_ENDPOINT: &str = "https://www.modelscope.cn";
pub const DEFAULT_REVISION: &str = "master";

const INFO_PATH: &str = "/api/v1/models/{model_id}";
const FILES_PATH: &str = "/api/v1/models/{model_id}/repo/files";
const RAW_PATH: &str = "/api/v1/models/{model_id}/repo";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpError(pub String);

pub trait HttpClient {
    fn get(
        &mut self,
        url: &str,
        headers: &HashMap<String, String>,
        params: &[(String, String)],
        timeout_s: f64,
    ) -> Result<HttpResponse, HttpError>;
}

#[derive(Default)]
pub struct ReqwestHttpClient;

impl HttpClient for ReqwestHttpClient {
    fn get(
        &mut self,
        url: &str,
        headers: &HashMap<String, String>,
        params: &[(String, String)],
        timeout_s: f64,
    ) -> Result<HttpResponse, HttpError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs_f64(timeout_s))
            .build()
            .map_err(|error| HttpError(error.to_string()))?;
        let mut request = client.get(url).query(params);
        for (key, value) in headers {
            request = request.header(key, value);
        }
        let response = request
            .send()
            .map_err(|error| HttpError(error.to_string()))?;
        let status = response.status().as_u16();
        let mut response_headers = HashMap::new();
        for (key, value) in response.headers() {
            response_headers.insert(
                key.as_str().to_string(),
                value.to_str().unwrap_or_default().to_string(),
            );
        }
        let body = response
            .bytes()
            .map_err(|error| HttpError(error.to_string()))?
            .to_vec();
        Ok(HttpResponse {
            status,
            headers: response_headers,
            body,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ModelScopeSource {
    endpoint: String,
    timeout_s: f64,
    revision: String,
}

impl Default for ModelScopeSource {
    fn default() -> Self {
        Self::new(None, 30.0, DEFAULT_REVISION)
    }
}

impl ModelScopeSource {
    pub fn new(endpoint: Option<&str>, timeout_s: f64, revision: &str) -> Self {
        Self {
            endpoint: endpoint
                .unwrap_or(DEFAULT_ENDPOINT)
                .trim_end_matches('/')
                .to_string(),
            timeout_s,
            revision: revision.to_string(),
        }
    }

    pub fn fetch(&self, model_id: &str) -> Result<ModelArtifact, ModelSourceError> {
        let token = get_modelscope_token();
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
        let commit_sha = self.fetch_commit_sha(model_id, client, &headers);
        let revision = commit_sha.as_deref().unwrap_or(&self.revision);

        let siblings = self.list_files(model_id, revision, client, &headers)?;
        let config = self.fetch_config(model_id, revision, client, &headers)?;

        Ok(ModelArtifact {
            source: "modelscope".to_string(),
            model_id: model_id.to_string(),
            commit_sha,
            config,
            siblings,
        })
    }

    fn fetch_commit_sha(
        &self,
        model_id: &str,
        client: &mut dyn HttpClient,
        headers: &HashMap<String, String>,
    ) -> Option<String> {
        let url = self.url(INFO_PATH, model_id);
        let response = client.get(&url, headers, &[], self.timeout_s).ok()?;
        if response.status != 200 {
            return None;
        }
        let payload: Value = serde_json::from_slice(&response.body).ok()?;
        let data = payload.get("Data")?.as_object()?;
        for key in ["LatestSha", "latest_sha", "Revision", "Sha"] {
            if let Some(value) = data
                .get(key)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
            {
                return Some(value.to_string());
            }
        }
        None
    }

    fn list_files(
        &self,
        model_id: &str,
        revision: &str,
        client: &mut dyn HttpClient,
        headers: &HashMap<String, String>,
    ) -> Result<Vec<SiblingFile>, ModelSourceError> {
        let url = self.url(FILES_PATH, model_id);
        let params = vec![
            ("Recursive".to_string(), "true".to_string()),
            ("Revision".to_string(), revision.to_string()),
        ];
        let response = client
            .get(&url, headers, &params, self.timeout_s)
            .map_err(|error| {
                SourceUnavailableError(format!("ModelScope file list failed: {}", error.0))
            })?;

        self.raise_for_status(&response, model_id, "file list")?;
        let payload: Value = serde_json::from_slice(&response.body).map_err(|error| {
            SourceUnavailableError(format!("ModelScope file list returned non-JSON: {error}"))
        })?;
        let files = extract_files(&payload).ok_or_else(|| {
            SourceUnavailableError(
                "ModelScope file list payload had unexpected shape — neither Data.Files nor Data is a list."
                    .to_string(),
            )
        })?;

        Ok(files
            .into_iter()
            .filter_map(|file| {
                let object = file.as_object()?;
                if object.get("Type").and_then(Value::as_str).unwrap_or("blob") == "tree" {
                    return None;
                }
                let filename = object.get("Path").and_then(Value::as_str)?.to_string();
                let size = object.get("Size").and_then(value_to_u64);
                Some(SiblingFile { filename, size })
            })
            .collect())
    }

    fn fetch_config(
        &self,
        model_id: &str,
        revision: &str,
        client: &mut dyn HttpClient,
        headers: &HashMap<String, String>,
    ) -> Result<Value, ModelSourceError> {
        let url = self.url(RAW_PATH, model_id);
        let params = vec![
            ("FilePath".to_string(), "config.json".to_string()),
            ("Revision".to_string(), revision.to_string()),
        ];
        let response = client
            .get(&url, headers, &params, self.timeout_s)
            .map_err(|error| {
                SourceUnavailableError(format!("config.json fetch failed: {}", error.0))
            })?;

        self.raise_for_status(&response, model_id, "config.json")?;
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

    fn raise_for_status(
        &self,
        response: &HttpResponse,
        model_id: &str,
        what: &str,
    ) -> Result<(), ModelSourceError> {
        match response.status {
            200 => Ok(()),
            404 => Err(ModelNotFoundError(format!(
                "Model '{model_id}' not found on ModelScope ({what})."
            ))
            .into()),
            401 | 403 => Err(AuthRequiredError(modelscope_auth_error_message(model_id)).into()),
            429 => {
                let retry = response
                    .headers
                    .get("Retry-After")
                    .or_else(|| response.headers.get("retry-after"))
                    .map(String::as_str)
                    .unwrap_or("unknown");
                Err(SourceUnavailableError(format!(
                    "ModelScope rate limit (429). Retry-After: {retry}s. Setting MODELSCOPE_API_TOKEN increases your quota."
                ))
                .into())
            }
            status => Err(SourceUnavailableError(format!(
                "ModelScope {what} returned HTTP {status}"
            ))
            .into()),
        }
    }

    fn url(&self, path: &str, model_id: &str) -> String {
        format!("{}{}", self.endpoint, path.replace("{model_id}", model_id))
    }
}

impl ModelSource for ModelScopeSource {
    fn name(&self) -> &str {
        "modelscope"
    }

    fn fetch(&self, model_id: &str) -> Result<ModelArtifact, ModelSourceError> {
        ModelScopeSource::fetch(self, model_id)
    }
}

pub fn extract_files(payload: &Value) -> Option<Vec<Value>> {
    let data = payload.as_object()?.get("Data")?;
    if let Some(data) = data.as_object() {
        if let Some(files) = data.get("Files").and_then(Value::as_array) {
            return Some(files.clone());
        }
    }
    data.as_array().cloned()
}

fn auth_headers(token: Option<&str>) -> HashMap<String, String> {
    token
        .filter(|token| !token.is_empty())
        .map(|token| HashMap::from([("Authorization".to_string(), format!("Bearer {token}"))]))
        .unwrap_or_default()
}

fn value_to_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}
