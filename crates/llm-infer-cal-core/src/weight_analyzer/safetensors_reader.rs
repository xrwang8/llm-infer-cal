use std::collections::HashMap;

use serde_json::Value;

use crate::model_source::auth::{get_hf_token, get_modelscope_token};
use crate::model_source::base::SiblingFile;
use crate::model_source::huggingface::DEFAULT_ENDPOINT as HF_DEFAULT_ENDPOINT;
use crate::model_source::modelscope::{
    HttpClient, ReqwestHttpClient, DEFAULT_ENDPOINT as MODELSCOPE_DEFAULT_ENDPOINT,
};

const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const RANGE_FETCH_BYTES: u64 = 16 * 1024 * 1024;

pub fn pick_sample_shard(siblings: &[SiblingFile]) -> Option<SiblingFile> {
    let st_files = siblings
        .iter()
        .filter(|sibling| sibling.filename.ends_with(".safetensors"))
        .collect::<Vec<_>>();
    if st_files.is_empty() {
        return None;
    }

    if let Some(single) = st_files
        .iter()
        .find(|sibling| sibling.filename == "model.safetensors")
    {
        return Some((*single).clone());
    }

    let mut sorted_shards = st_files;
    sorted_shards.sort_by(|left, right| left.filename.cmp(&right.filename));
    sorted_shards
        .get(sorted_shards.len() / 2)
        .map(|sibling| (*sibling).clone())
}

pub fn parse_header(content: &[u8]) -> Option<HashMap<String, String>> {
    if content.len() < 8 {
        return None;
    }

    let header_len = u64::from_le_bytes(content[0..8].try_into().ok()?);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return None;
    }

    let header_end = 8usize.checked_add(usize::try_from(header_len).ok()?)?;
    if content.len() < header_end {
        return None;
    }

    let header: Value = serde_json::from_slice(&content[8..header_end]).ok()?;
    let header = header.as_object()?;
    let mut dtypes = HashMap::new();

    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        let Some(info) = info.as_object() else {
            continue;
        };
        if let Some(dtype) = info.get("dtype").and_then(Value::as_str) {
            dtypes.insert(name.clone(), dtype.to_string());
        }
    }

    if dtypes.is_empty() {
        None
    } else {
        Some(dtypes)
    }
}

pub fn fetch_tensor_dtypes(
    source: &str,
    model_id: &str,
    revision: &str,
    shard_filename: &str,
    endpoint: Option<&str>,
) -> Option<HashMap<String, String>> {
    let mut client = ReqwestHttpClient;
    fetch_tensor_dtypes_with_client(
        source,
        model_id,
        revision,
        shard_filename,
        endpoint,
        &mut client,
        15.0,
    )
}

pub fn fetch_tensor_dtypes_with_client(
    source: &str,
    model_id: &str,
    revision: &str,
    shard_filename: &str,
    endpoint: Option<&str>,
    client: &mut dyn HttpClient,
    timeout_s: f64,
) -> Option<HashMap<String, String>> {
    let request = build_request(source, model_id, revision, shard_filename, endpoint)?;
    let response = client
        .get(&request.url, &request.headers, &request.params, timeout_s)
        .ok()?;

    if response.status != 200 && response.status != 206 {
        return None;
    }
    parse_header(&response.body)
}

#[derive(Debug, Eq, PartialEq)]
struct HeaderRequest {
    url: String,
    headers: HashMap<String, String>,
    params: Vec<(String, String)>,
}

fn build_request(
    source: &str,
    model_id: &str,
    revision: &str,
    shard_filename: &str,
    endpoint: Option<&str>,
) -> Option<HeaderRequest> {
    let mut headers = auth_headers(source);
    headers.insert("User-Agent".to_string(), "llm-infer-cal/0.1".to_string());
    headers.insert("Accept".to_string(), "*/*".to_string());
    headers.insert(
        "Range".to_string(),
        format!("bytes=0-{}", RANGE_FETCH_BYTES - 1),
    );

    match source {
        "huggingface" => {
            let base = endpoint
                .unwrap_or(HF_DEFAULT_ENDPOINT)
                .trim_end_matches('/');
            Some(HeaderRequest {
                url: format!("{base}/{model_id}/resolve/{revision}/{shard_filename}"),
                headers,
                params: Vec::new(),
            })
        }
        "modelscope" => {
            let base = endpoint
                .unwrap_or(MODELSCOPE_DEFAULT_ENDPOINT)
                .trim_end_matches('/');
            Some(HeaderRequest {
                url: format!("{base}/api/v1/models/{model_id}/repo"),
                headers,
                params: vec![
                    ("FilePath".to_string(), shard_filename.to_string()),
                    ("Revision".to_string(), revision.to_string()),
                ],
            })
        }
        _ => None,
    }
}

fn auth_headers(source: &str) -> HashMap<String, String> {
    let token = match source {
        "huggingface" => get_hf_token(),
        "modelscope" => get_modelscope_token(),
        _ => None,
    };
    token
        .filter(|token| !token.is_empty())
        .map(|token| HashMap::from([("Authorization".to_string(), format!("Bearer {token}"))]))
        .unwrap_or_default()
}
