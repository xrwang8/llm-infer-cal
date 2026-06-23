use std::collections::HashMap;

use llm_infer_cal_core::model_source::base::SiblingFile;
use llm_infer_cal_core::model_source::modelscope::{HttpClient, HttpError, HttpResponse};
use llm_infer_cal_core::weight_analyzer::safetensors_reader::{
    fetch_tensor_dtypes_with_client, parse_header, pick_sample_shard,
};
use serde_json::json;

#[derive(Debug)]
struct FakeClient {
    response: Result<HttpResponse, HttpError>,
    calls: Vec<RecordedCall>,
}

#[derive(Debug, Eq, PartialEq)]
struct RecordedCall {
    url: String,
    headers: HashMap<String, String>,
    params: Vec<(String, String)>,
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
        self.response.clone()
    }
}

fn sibling(filename: &str, size: u64) -> SiblingFile {
    SiblingFile {
        filename: filename.to_string(),
        size: Some(size),
    }
}

fn build_safetensors_bytes(header: serde_json::Value) -> Vec<u8> {
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&[0_u8; 100]);
    out
}

#[test]
fn pick_sample_shard_matches_python_preference_order() {
    let single = vec![
        sibling("model-00001-of-00010.safetensors", 1000),
        sibling("model.safetensors", 100),
        sibling("config.json", 10),
    ];
    assert_eq!(
        pick_sample_shard(&single).unwrap().filename,
        "model.safetensors"
    );

    let shuffled = vec![
        sibling("model-00003-of-00010.safetensors", 1000),
        sibling("model-00001-of-00010.safetensors", 1000),
        sibling("model-00002-of-00010.safetensors", 1000),
    ];
    assert_eq!(
        pick_sample_shard(&shuffled).unwrap().filename,
        "model-00002-of-00010.safetensors"
    );

    let even = (1..47)
        .map(|idx| sibling(&format!("model-{idx:05}-of-00046.safetensors"), 1000))
        .collect::<Vec<_>>();
    assert_eq!(
        pick_sample_shard(&even).unwrap().filename,
        "model-00024-of-00046.safetensors"
    );

    assert!(pick_sample_shard(&[sibling("pytorch_model.bin", 1000)]).is_none());
}

#[test]
fn parse_header_returns_tensor_dtype_map() {
    let content = build_safetensors_bytes(json!({
        "__metadata__": {"format": "pt"},
        "model.layers.0.weight": {
            "dtype": "F8_E4M3",
            "shape": [4096, 4096],
            "data_offsets": [0, 16777216]
        },
        "model.norm.weight": {
            "dtype": "BF16",
            "shape": [4096],
            "data_offsets": [16777216, 16785408]
        },
        "missing_dtype": {"shape": [1], "data_offsets": [0, 1]},
        "not_an_object": "F16"
    }));

    let dtypes = parse_header(&content).unwrap();
    let expected = HashMap::from([
        ("model.layers.0.weight".to_string(), "F8_E4M3".to_string()),
        ("model.norm.weight".to_string(), "BF16".to_string()),
    ]);
    assert_eq!(dtypes, expected);
}

#[test]
fn fetch_tensor_dtypes_modelscope_range_gets_and_parses_header() {
    let content = build_safetensors_bytes(json!({
        "model.layers.0.mlp.experts.0.down_proj.weight": {
            "dtype": "I8",
            "shape": [1],
            "data_offsets": [0, 1]
        },
        "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv": {
            "dtype": "F8_E8M0",
            "shape": [1],
            "data_offsets": [1, 2]
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F8_E4M3",
            "shape": [1],
            "data_offsets": [2, 3]
        }
    }));
    let mut client = FakeClient {
        response: Ok(HttpResponse {
            status: 206,
            headers: HashMap::new(),
            body: content,
        }),
        calls: Vec::new(),
    };

    let dtypes = fetch_tensor_dtypes_with_client(
        "modelscope",
        "deepseek-ai/DeepSeek-V4-Flash",
        "master",
        "model-00024-of-00046.safetensors",
        None,
        &mut client,
        15.0,
    )
    .unwrap();

    assert_eq!(
        dtypes["model.layers.0.mlp.experts.0.down_proj.weight"],
        "I8"
    );
    assert_eq!(
        dtypes["model.layers.0.mlp.experts.0.down_proj.weight_scale_inv"],
        "F8_E8M0"
    );
    assert_eq!(client.calls.len(), 1);
    assert_eq!(
        client.calls[0].url,
        "https://www.modelscope.cn/api/v1/models/deepseek-ai/DeepSeek-V4-Flash/repo"
    );
    assert!(client.calls[0].params.contains(&(
        "FilePath".to_string(),
        "model-00024-of-00046.safetensors".to_string()
    )));
    assert!(client.calls[0]
        .params
        .contains(&("Revision".to_string(), "master".to_string())));
    assert_eq!(
        client.calls[0].headers.get("Range").map(String::as_str),
        Some("bytes=0-16777215")
    );
    assert_eq!(
        client.calls[0]
            .headers
            .get("User-Agent")
            .map(String::as_str),
        Some("llm-infer-cal/0.1")
    );
    assert_eq!(
        client.calls[0].headers.get("Accept").map(String::as_str),
        Some("*/*")
    );
}

#[test]
fn parse_header_malformed_inputs_return_none() {
    assert!(parse_header(b"abc").is_none());

    let mut oversized_claim = Vec::new();
    oversized_claim.extend_from_slice(&10_000_000_u64.to_le_bytes());
    oversized_claim.extend_from_slice(b"{}{}{}");
    assert!(parse_header(&oversized_claim).is_none());

    let mut absurd = Vec::new();
    absurd.extend_from_slice(&(100_u64 * 1024 * 1024).to_le_bytes());
    absurd.extend_from_slice(b"{}");
    assert!(parse_header(&absurd).is_none());

    let mut zero = Vec::new();
    zero.extend_from_slice(&0_u64.to_le_bytes());
    zero.extend_from_slice(b"{}");
    assert!(parse_header(&zero).is_none());

    let bad_json = b"{ not valid json";
    let mut malformed_json = Vec::new();
    malformed_json.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
    malformed_json.extend_from_slice(bad_json);
    malformed_json.extend_from_slice(&[0_u8; 10]);
    assert!(parse_header(&malformed_json).is_none());

    assert!(parse_header(&build_safetensors_bytes(json!({}))).is_none());
    assert!(parse_header(&build_safetensors_bytes(json!([]))).is_none());
}
