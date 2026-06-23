use axum::body::{to_bytes, Body};
use axum::http::{header, Method, Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

async fn json_response(response: axum::response::Response) -> Value {
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn models_endpoint_lists_builtin_models() {
    let response = llm_infer_cal_web::app()
        .oneshot(
            Request::builder()
                .uri("/api/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_response(response).await;
    let models = body["models"].as_array().unwrap();
    assert!(models.len() >= 100);
    let qwen = models
        .iter()
        .find(|model| model["id"] == "Qwen/Qwen3-30B-A3B")
        .unwrap();
    assert_eq!(qwen["provider"], "Qwen");
}

#[tokio::test]
async fn gpus_endpoint_lists_h100() {
    let response = llm_infer_cal_web::app()
        .oneshot(
            Request::builder()
                .uri("/api/gpus")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_response(response).await;
    let gpus = body["gpus"].as_array().unwrap();
    let h100 = gpus.iter().find(|gpu| gpu["id"] == "H100").unwrap();
    assert_eq!(h100["vendor"], "NVIDIA");
    assert_eq!(h100["memory_gb"], 80);
    assert_eq!(h100["fp8_support"], true);

    let mi300x = gpus.iter().find(|gpu| gpu["id"] == "MI300X").unwrap();
    assert_eq!(mi300x["vendor"], "AMD");

    let ascend = gpus.iter().find(|gpu| gpu["id"] == "910B4").unwrap();
    assert_eq!(ascend["vendor"], "Huawei Ascend");
    assert_eq!(ascend["vendor_zh"], "华为昇腾");

    let biren = gpus.iter().find(|gpu| gpu["id"] == "BR100").unwrap();
    assert_eq!(biren["vendor"], "Biren 壁仞");
    assert_eq!(biren["vendor_zh"], "壁仞 Biren");
}

#[tokio::test]
async fn evaluate_endpoint_returns_report_json() {
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "engine": "vllm"
    });

    let response = llm_infer_cal_web::app()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/evaluate")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_response(response).await;
    assert_eq!(body["schema_version"], "llm-infer-cal.report/v1");
    assert_eq!(body["model"]["id"], "Qwen/Qwen3-30B-A3B");
    let lines = body["generated_command"]["lines"].as_array().unwrap();
    assert!(lines.iter().any(|line| line == "--max-model-len 40960"));
    assert!(lines.iter().any(|line| line == "--max-num-seqs 20"));
}

#[tokio::test]
async fn evaluate_endpoint_can_include_explain_and_llm_review_text() {
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "engine": "vllm",
        "explain": true,
        "llm_review": true
    });

    let response = llm_infer_cal_web::app()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/evaluate")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_response(response).await;
    assert!(body["explain_text"]
        .as_str()
        .unwrap()
        .contains("完整推导链"));
    assert!(body["llm_review_text"]
        .as_str()
        .unwrap()
        .contains("LLM 审阅"));
}
