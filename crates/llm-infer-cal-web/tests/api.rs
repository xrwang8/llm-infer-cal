use axum::body::{to_bytes, Body};
use axum::http::{header, Method, Request, StatusCode};
use serde_json::{json, Value};
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};
use tower::ServiceExt;

async fn json_response(response: axum::response::Response) -> Value {
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn text_response(response: axum::response::Response) -> String {
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[tokio::test]
async fn app_with_static_serves_frontend_and_keeps_api_routes() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let static_dir = std::env::temp_dir().join(format!(
        "llm-infer-cal-web-static-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir_all(&static_dir).unwrap();
    fs::write(
        static_dir.join("index.html"),
        "<html><body>llm-infer-cal static shell</body></html>",
    )
    .unwrap();

    let app = llm_infer_cal_web::app_with_static(&static_dir);
    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    assert_eq!(text_response(health).await, "ok");

    let index = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(index.status(), StatusCode::OK);
    assert!(text_response(index)
        .await
        .contains("llm-infer-cal static shell"));

    fs::remove_dir_all(static_dir).unwrap();
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
async fn evaluate_endpoint_accepts_inference_optimization_options() {
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "engine": "vllm",
        "context_length": 4096,
        "kv_cache_bits": 8,
        "paged_attention": true
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
    assert_eq!(body["inference_options"]["kv_cache_bits"], 8);
    assert_eq!(body["inference_options"]["paged_attention"], true);
    assert!(body["kv_cache_by_context"][0]["bytes"]["source"]
        .as_str()
        .unwrap()
        .contains("paged attention"));
    assert!(
        body["activation_by_context"][0]["bytes"]["value"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        body["fleet"]["options"][0]["activation_bytes_per_request"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[tokio::test]
async fn evaluate_endpoint_accepts_memory_pressure_options() {
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "engine": "vllm",
        "gpu_count": 1,
        "context_length": 4096,
        "target_concurrent_requests": 3,
        "speculative_extra_weight_bytes": 2048,
        "cpu_offload_gb": 1.0
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
    assert_eq!(body["inference_options"]["target_concurrent_requests"], 3);
    assert_eq!(
        body["inference_options"]["speculative_extra_weight_bytes"]["value"],
        2048
    );
    assert_eq!(
        body["inference_options"]["cpu_offload_bytes_per_gpu"],
        1_073_741_824
    );
    assert_eq!(body["fleet"]["best_tier"], "target");
    assert_eq!(body["fleet"]["options"][0]["tier_concurrent_requests"], 3);
    assert_eq!(
        body["fleet"]["options"][0]["speculative_weight_bytes_per_gpu"],
        2048
    );
    assert_eq!(
        body["fleet"]["options"][0]["cpu_offload_bytes_per_gpu"],
        1_073_741_824
    );
}

#[tokio::test]
async fn evaluate_endpoint_includes_draft_model_in_generated_command() {
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "engine": "vllm",
        "target_concurrent_requests": 2,
        "speculative_enabled": true,
        "speculative_mode": "standard",
        "speculative_num_draft_tokens": 9,
        "speculative_draft_model_id": "Qwen/Qwen3-0.6B"
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
    let command = body["generated_command"]["command"].as_str().unwrap();
    assert!(command.contains("--max-num-seqs 2"));
    assert!(command.contains("--speculative-config"));
    assert!(command.contains("\"model\":\"Qwen/Qwen3-0.6B\""));
    assert!(command.contains("\"num_speculative_tokens\":9"));
}

#[tokio::test]
async fn evaluate_endpoint_echoes_mtp_speculative_mode_without_draft_model() {
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "engine": "vllm",
        "speculative_enabled": true,
        "speculative_mode": "mtp",
        "speculative_num_draft_tokens": 6,
        "speculative_extra_weight_gb": 0.3
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
    assert_eq!(body["inference_options"]["speculative_enabled"], true);
    assert_eq!(body["inference_options"]["speculative_mode"], "mtp");
    assert_eq!(body["inference_options"]["speculative_num_draft_tokens"], 6);
    assert!(body["inference_options"]["speculative_draft_model_id"].is_null());
    assert!(
        body["generated_command"]["command"]
            .as_str()
            .unwrap()
            .contains("--speculative-config")
            == false
    );
}

#[tokio::test]
async fn evaluate_endpoint_defaults_enabled_speculative_mode_to_mtp() {
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "engine": "vllm",
        "speculative_enabled": true
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
    assert_eq!(body["inference_options"]["speculative_enabled"], true);
    assert_eq!(body["inference_options"]["speculative_mode"], "mtp");
    assert!(body["inference_options"]["speculative_draft_model_id"].is_null());
}

#[tokio::test]
async fn evaluate_endpoint_applies_expert_offload_to_moe_memory() {
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "engine": "vllm",
        "gpu_count": 1,
        "expert_offloading": true,
        "experts_on_gpu": 8
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
    let option = &body["fleet"]["options"][0];
    assert_eq!(body["inference_options"]["expert_offloading"], true);
    assert_eq!(body["inference_options"]["experts_on_gpu"], 8);
    assert!(
        body["inference_options"]["expert_offload_bytes_per_gpu"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        option["expert_offload_bytes_per_gpu"],
        body["inference_options"]["expert_offload_bytes_per_gpu"]
    );
    assert!(
        option["main_weight_bytes_per_gpu"].as_u64().unwrap()
            < option["main_weight_bytes_before_offload_per_gpu"]
                .as_u64()
                .unwrap()
    );
}

#[tokio::test]
async fn evaluate_endpoint_rejects_invalid_user_optimization_parameters() {
    let cases = [
        (
            json!({
                "model_id": "Qwen/Qwen3-30B-A3B",
                "source": "builtin",
                "gpu": "H100",
                "engine": "vllm",
                "speculative_enabled": true,
                "speculative_mode": "magic"
            }),
            "speculative_mode must be standard or mtp",
        ),
        (
            json!({
                "model_id": "Qwen/Qwen3-30B-A3B",
                "source": "builtin",
                "gpu": "H100",
                "engine": "vllm",
                "speculative_enabled": true,
                "speculative_num_draft_tokens": 0
            }),
            "speculative_num_draft_tokens must be greater than 0",
        ),
        (
            json!({
                "model_id": "Qwen/Qwen3-30B-A3B",
                "source": "builtin",
                "gpu": "H100",
                "engine": "vllm",
                "expert_offloading": true
            }),
            "experts_on_gpu is required",
        ),
    ];

    for (payload, message) in cases {
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

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_response(response).await;
        assert!(body["error"]["message"].as_str().unwrap().contains(message));
    }
}

#[tokio::test]
async fn evaluate_endpoint_returns_comparison_for_multiple_gpus() {
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "gpus": ["H100", "A100-80G"],
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
    let reports = body["comparison"]["reports"].as_array().unwrap();
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[0]["hardware"]["id"], "H100");
    assert_eq!(reports[1]["hardware"]["id"], "A100-80G");
    assert_eq!(body["hardware"]["id"], "H100");
}

#[tokio::test]
async fn evaluate_endpoint_returns_comparison_for_five_gpus() {
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "gpus": ["H100", "A100-80G", "H800", "H200", "L40S"],
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
    let reports = body["comparison"]["reports"].as_array().unwrap();
    assert_eq!(reports.len(), 5);
    assert_eq!(reports[4]["hardware"]["id"], "L40S");
}

#[tokio::test]
async fn evaluate_endpoint_rejects_more_than_sixty_four_comparison_gpus() {
    let gpus = (0..65).map(|idx| format!("GPU{idx}")).collect::<Vec<_>>();
    let payload = json!({
        "model_id": "Qwen/Qwen3-30B-A3B",
        "source": "builtin",
        "gpu": "H100",
        "gpus": gpus,
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

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_response(response).await;
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("at most 64"));
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
