use std::collections::{HashMap, VecDeque};

use llm_infer_cal_core::model_source::base::ModelSourceError;
use llm_infer_cal_core::model_source::huggingface::{HuggingFaceSource, DEFAULT_ENDPOINT};
use llm_infer_cal_core::model_source::modelscope::{HttpClient, HttpError, HttpResponse};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq)]
struct RecordedCall {
    url: String,
    headers: HashMap<String, String>,
    params: Vec<(String, String)>,
}

#[derive(Debug)]
struct FakeClient {
    responses: VecDeque<Result<HttpResponse, HttpError>>,
    calls: Vec<RecordedCall>,
}

impl FakeClient {
    fn new(responses: impl IntoIterator<Item = Result<HttpResponse, HttpError>>) -> Self {
        Self {
            responses: responses.into_iter().collect(),
            calls: Vec::new(),
        }
    }
}

impl HttpClient for FakeClient {
    fn get(
        &mut self,
        url: &str,
        headers: &HashMap<String, String>,
        params: &[(String, String)],
        _timeout_s: f64,
    ) -> Result<HttpResponse, HttpError> {
        self.calls.push(RecordedCall {
            url: url.to_string(),
            headers: headers.clone(),
            params: params.to_vec(),
        });
        self.responses
            .pop_front()
            .expect("test must provide enough fake responses")
    }
}

fn json_resp(payload: Value, status: u16) -> Result<HttpResponse, HttpError> {
    Ok(HttpResponse {
        status,
        headers: HashMap::new(),
        body: serde_json::to_vec(&payload).unwrap(),
    })
}

fn raw_resp(content: &[u8], status: u16) -> Result<HttpResponse, HttpError> {
    Ok(HttpResponse {
        status,
        headers: HashMap::new(),
        body: content.to_vec(),
    })
}

#[test]
fn fetch_happy_path_uses_one_model_info_request_and_pins_config_to_sha() {
    let mut client = FakeClient::new([
        json_resp(
            json!({
                "sha": "abc123",
                "siblings": [
                    {"rfilename": "config.json", "size": 642},
                    {"rfilename": "model-00001-of-00002.safetensors", "size": 5_000_000_000_u64},
                    {"rfilename": "tokenizer.json", "size": null}
                ]
            }),
            200,
        ),
        raw_resp(br#"{"model_type": "llama", "hidden_size": 4096}"#, 200),
    ]);

    let artifact = HuggingFaceSource::default()
        .fetch_with_client("owner/repo", &mut client, Some("hf-secret"))
        .unwrap();

    assert_eq!(artifact.source, "huggingface");
    assert_eq!(artifact.model_id, "owner/repo");
    assert_eq!(artifact.commit_sha.as_deref(), Some("abc123"));
    assert_eq!(artifact.config["model_type"], "llama");
    assert_eq!(artifact.siblings.len(), 3);
    assert_eq!(
        artifact.siblings[1].filename,
        "model-00001-of-00002.safetensors"
    );
    assert_eq!(artifact.siblings[1].size, Some(5_000_000_000));

    assert_eq!(client.calls.len(), 2);
    assert_eq!(
        client.calls[0].url,
        "https://huggingface.co/api/models/owner/repo"
    );
    assert!(client.calls[0]
        .params
        .contains(&("blobs".to_string(), "true".to_string())));
    assert_eq!(
        client.calls[1].url,
        "https://huggingface.co/owner/repo/resolve/abc123/config.json"
    );
    assert!(client.calls.iter().all(|call| {
        call.headers.get("Authorization").map(String::as_str) == Some("Bearer hf-secret")
    }));
}

#[test]
fn fetch_uses_custom_endpoint_and_main_when_sha_missing() {
    let mut client = FakeClient::new([
        json_resp(json!({"siblings": []}), 200),
        raw_resp(br#"{"model_type": "mistral"}"#, 200),
    ]);
    let source = HuggingFaceSource::new(Some("https://hf-mirror.example.com"), 30.0);

    let artifact = source.fetch_with_client("o/r", &mut client, None).unwrap();

    assert_eq!(artifact.commit_sha, None);
    assert!(client
        .calls
        .iter()
        .all(|call| call.url.starts_with("https://hf-mirror.example.com/")));
    assert!(!client
        .calls
        .iter()
        .any(|call| call.url.starts_with(DEFAULT_ENDPOINT)));
    assert_eq!(
        client.calls[1].url,
        "https://hf-mirror.example.com/o/r/resolve/main/config.json"
    );
}

#[test]
fn fetch_maps_model_info_errors_like_rust_contract() {
    let mut not_found = FakeClient::new([json_resp(json!({"error": "missing"}), 404)]);
    let err = HuggingFaceSource::default()
        .fetch_with_client("missing/repo", &mut not_found, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::NotFound(_)));
    assert!(err.to_string().contains("not found on HuggingFace"));

    let mut gated = FakeClient::new([json_resp(json!({"error": "gated"}), 403)]);
    let err = HuggingFaceSource::default()
        .fetch_with_client("private/repo", &mut gated, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::AuthRequired(_)));
    assert!(err.to_string().contains("HF_TOKEN"));

    let mut headers = HashMap::new();
    headers.insert("Retry-After".to_string(), "42".to_string());
    let mut rate_limited = FakeClient::new([Ok(HttpResponse {
        status: 429,
        headers,
        body: br#"{"error": "rate"}"#.to_vec(),
    })]);
    let err = HuggingFaceSource::default()
        .fetch_with_client("o/r", &mut rate_limited, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::SourceUnavailable(_)));
    assert!(err.to_string().contains("rate limit"));
    assert!(err.to_string().contains("42"));

    let mut timeout = FakeClient::new([Err(HttpError("timed out".to_string()))]);
    let err = HuggingFaceSource::default()
        .fetch_with_client("o/r", &mut timeout, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::SourceUnavailable(_)));
    assert!(err.to_string().contains("timed out"));
}

#[test]
fn fetch_maps_config_errors_like_rust_contract() {
    let info = json!({"sha": "abc", "siblings": []});

    let mut missing_config =
        FakeClient::new([json_resp(info.clone(), 200), raw_resp(b"not found", 404)]);
    let err = HuggingFaceSource::default()
        .fetch_with_client("o/r", &mut missing_config, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::NotFound(_)));
    assert!(err.to_string().contains("no config.json"));

    let mut unauth_config =
        FakeClient::new([json_resp(info.clone(), 200), raw_resp(b"forbidden", 401)]);
    let err = HuggingFaceSource::default()
        .fetch_with_client("private/repo", &mut unauth_config, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::AuthRequired(_)));

    let mut bad_config = FakeClient::new([
        json_resp(info.clone(), 200),
        raw_resp(b"not valid json {{{", 200),
    ]);
    let err = HuggingFaceSource::default()
        .fetch_with_client("o/r", &mut bad_config, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::SourceUnavailable(_)));
    assert!(err.to_string().contains("config.json is not valid JSON"));

    let mut config_array = FakeClient::new([json_resp(info, 200), raw_resp(b"[1,2,3]", 200)]);
    let err = HuggingFaceSource::default()
        .fetch_with_client("o/r", &mut config_array, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::SourceUnavailable(_)));
    assert!(err.to_string().contains("JSON object"));
}
