use std::collections::{HashMap, VecDeque};

use llm_infer_cal_core::model_source::base::ModelSourceError;
use llm_infer_cal_core::model_source::modelscope::{
    extract_files, HttpClient, HttpError, HttpResponse, ModelScopeSource, DEFAULT_ENDPOINT,
};
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

fn wrapped(data: Value) -> Value {
    json!({"Code": 200, "Message": "ok", "RequestId": "test", "Success": true, "Data": data})
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
fn extract_files_accepts_known_modelscope_shapes() {
    assert_eq!(
        extract_files(&json!({"Data": {"Files": [{"Path": "a.bin", "Size": 1}]}})),
        Some(vec![json!({"Path": "a.bin", "Size": 1})])
    );
    assert_eq!(
        extract_files(&json!({"Data": [{"Path": "a.bin"}]})),
        Some(vec![json!({"Path": "a.bin"})])
    );
    assert!(extract_files(&json!({"Code": 200})).is_none());
    assert!(extract_files(&json!({"Data": "permission denied"})).is_none());
    assert!(extract_files(&Value::Null).is_none());
}

#[test]
fn fetch_happy_path_filters_tree_entries_and_pins_commit_sha() {
    let mut client = FakeClient::new([
        json_resp(wrapped(json!({"LatestSha": "deadbeef"})), 200),
        json_resp(
            wrapped(json!({
                "Files": [
                    {"Path": "config.json", "Type": "blob", "Size": 642},
                    {"Path": "model-00001-of-00002.safetensors", "Type": "blob", "Size": 5_000_000_000_u64},
                    {"Path": "tokenizer", "Type": "tree", "Size": null}
                ]
            })),
            200,
        ),
        raw_resp(br#"{"model_type": "qwen3_moe", "hidden_size": 4096}"#, 200),
    ]);

    let artifact = ModelScopeSource::default()
        .fetch_with_client("owner/repo", &mut client, None)
        .unwrap();

    assert_eq!(artifact.source, "modelscope");
    assert_eq!(artifact.model_id, "owner/repo");
    assert_eq!(artifact.commit_sha.as_deref(), Some("deadbeef"));
    assert_eq!(artifact.config["model_type"], "qwen3_moe");
    assert_eq!(artifact.siblings.len(), 2);
    assert_eq!(
        artifact
            .siblings
            .iter()
            .map(|sibling| sibling.filename.as_str())
            .collect::<Vec<_>>(),
        vec!["config.json", "model-00001-of-00002.safetensors"]
    );

    let file_call = &client.calls[1];
    assert!(file_call
        .params
        .contains(&("Revision".to_string(), "deadbeef".to_string())));
    let config_call = &client.calls[2];
    assert!(config_call
        .params
        .contains(&("Revision".to_string(), "deadbeef".to_string())));
}

#[test]
fn fetch_supports_data_list_shape_and_master_fallback_when_info_fails() {
    let mut client = FakeClient::new([
        Err(HttpError("info slow".to_string())),
        json_resp(
            wrapped(json!([{"Path": "config.json", "Type": "blob", "Size": 100}])),
            200,
        ),
        raw_resp(br#"{"model_type": "llama"}"#, 200),
    ]);

    let artifact = ModelScopeSource::default()
        .fetch_with_client("owner/repo", &mut client, None)
        .unwrap();

    assert_eq!(artifact.commit_sha, None);
    assert_eq!(artifact.siblings.len(), 1);
    assert!(client.calls[1]
        .params
        .contains(&("Revision".to_string(), "master".to_string())));
    assert!(client.calls[2]
        .params
        .contains(&("Revision".to_string(), "master".to_string())));
}

#[test]
fn fetch_uses_custom_endpoint_and_authorization_header() {
    let mut client = FakeClient::new([
        json_resp(wrapped(json!({})), 200),
        json_resp(wrapped(json!({"Files": []})), 200),
        raw_resp(b"{}", 200),
    ]);
    let source = ModelScopeSource::new(Some("https://my-mirror.example.com"), 30.0, "master");

    source
        .fetch_with_client("o/r", &mut client, Some("ms-secret-xyz"))
        .unwrap();

    assert!(client
        .calls
        .iter()
        .all(|call| call.url.starts_with("https://my-mirror.example.com/")));
    assert!(!client
        .calls
        .iter()
        .any(|call| call.url.starts_with(DEFAULT_ENDPOINT)));
    assert!(client.calls.iter().all(|call| {
        call.headers.get("Authorization").map(String::as_str) == Some("Bearer ms-secret-xyz")
    }));
}

#[test]
fn fetch_maps_file_list_status_errors_like_python() {
    let mut not_found = FakeClient::new([
        json_resp(wrapped(json!({})), 200),
        json_resp(json!({"Code": 404, "Message": "model not found"}), 404),
    ]);
    let err = ModelScopeSource::default()
        .fetch_with_client("nonexistent/model", &mut not_found, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::NotFound(_)));
    assert!(err.to_string().contains("not found"));

    let mut unauth = FakeClient::new([
        json_resp(wrapped(json!({})), 200),
        json_resp(json!({"Code": 401, "Message": "auth required"}), 401),
    ]);
    let err = ModelScopeSource::default()
        .fetch_with_client("private/repo", &mut unauth, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::AuthRequired(_)));
    assert!(err.to_string().contains("MODELSCOPE_API_TOKEN"));

    let mut headers = HashMap::new();
    headers.insert("Retry-After".to_string(), "30".to_string());
    let mut rate_limited = FakeClient::new([
        json_resp(wrapped(json!({})), 200),
        Ok(HttpResponse {
            status: 429,
            headers,
            body: br#"{"Code":429}"#.to_vec(),
        }),
    ]);
    let err = ModelScopeSource::default()
        .fetch_with_client("o/r", &mut rate_limited, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::SourceUnavailable(_)));
    assert!(err.to_string().contains("rate limit"));
    assert!(err.to_string().contains("30"));
}

#[test]
fn fetch_maps_payload_and_config_errors_like_python() {
    let mut bad_files = FakeClient::new([
        json_resp(wrapped(json!({})), 200),
        json_resp(
            json!({"Code": 200, "Message": "ok", "Data": "permission denied"}),
            200,
        ),
    ]);
    let err = ModelScopeSource::default()
        .fetch_with_client("o/r", &mut bad_files, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::SourceUnavailable(_)));
    assert!(err.to_string().contains("unexpected shape"));

    let mut bad_config = FakeClient::new([
        json_resp(wrapped(json!({})), 200),
        json_resp(wrapped(json!({"Files": []})), 200),
        raw_resp(b"not valid json {{{", 200),
    ]);
    let err = ModelScopeSource::default()
        .fetch_with_client("o/r", &mut bad_config, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::SourceUnavailable(_)));
    assert!(err.to_string().contains("config.json"));

    let mut config_array = FakeClient::new([
        json_resp(wrapped(json!({})), 200),
        json_resp(wrapped(json!({"Files": []})), 200),
        raw_resp(b"[1, 2, 3]", 200),
    ]);
    let err = ModelScopeSource::default()
        .fetch_with_client("o/r", &mut config_array, None)
        .unwrap_err();
    assert!(matches!(err, ModelSourceError::SourceUnavailable(_)));
    assert!(err.to_string().contains("JSON object"));
}
