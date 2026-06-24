use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use llm_infer_cal_core::{
    common::i18n::set_locale,
    core::{
        cache::ArtifactCache,
        evaluator::{EvaluationOptions, Evaluator},
        explain::build as build_explain,
    },
    hardware::loader::load_database,
    llm_review::reviewer::{run_review_with_env, EnvConfig},
    model_source::{
        base::{ModelSource, ModelSourceError},
        builtin::BuiltinSource,
        huggingface::HuggingFaceSource,
        modelscope::{ModelScopeSource, DEFAULT_REVISION},
    },
    output::formatter::{render_explain_text, render_llm_review_text, render_report_json},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};

const BUILTIN_MODEL_MANIFEST_JSON: &str =
    include_str!("../../llm-infer-cal-core/data/builtin_model_manifest.json");

pub fn app() -> Router {
    Router::new()
        .route("/api/health", get(|| async { "ok" }))
        .route("/api/models", get(list_models))
        .route("/api/gpus", get(list_gpus))
        .route("/api/evaluate", post(evaluate))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
}

async fn list_models() -> Result<Json<Value>, ApiError> {
    let manifest: BuiltinManifest =
        serde_json::from_str(BUILTIN_MODEL_MANIFEST_JSON).map_err(ApiError::internal)?;

    Ok(Json(json!({
        "models": manifest.models,
    })))
}

async fn list_gpus() -> Result<Json<Value>, ApiError> {
    let database = load_database().map_err(ApiError::internal)?;
    let gpus = database
        .gpus
        .into_iter()
        .map(|gpu| {
            let (vendor, vendor_zh) = gpu_vendor(&gpu.id);
            json!({
                "id": gpu.id,
                "vendor": vendor,
                "vendor_zh": vendor_zh,
                "aliases": gpu.aliases,
                "memory_gb": gpu.memory_gb,
                "nvlink_bandwidth_gbps": gpu.nvlink_bandwidth_gbps,
                "memory_bandwidth_gbps": gpu.memory_bandwidth_gbps,
                "fp16_tflops": gpu.fp16_tflops,
                "fp8_support": gpu.fp8_support,
                "fp4_support": gpu.fp4_support,
                "notes_en": gpu.notes_en,
                "notes_zh": gpu.notes_zh,
                "spec_source": gpu.spec_source,
            })
        })
        .collect::<Vec<_>>();

    Ok(Json(json!({ "gpus": gpus })))
}

fn gpu_vendor(gpu_id: &str) -> (&'static str, &'static str) {
    let id = gpu_id.to_ascii_uppercase();
    if id == "B200"
        || id == "GB200"
        || id == "H100"
        || id == "H800"
        || id == "H200"
        || id == "H20"
        || id == "GH200"
        || id.starts_with("L4")
        || id.starts_with("RTX")
        || id.starts_with("A10")
        || id.starts_with("A100")
        || id.starts_with("A40")
        || id.starts_with("V100")
        || id.starts_with("T4")
    {
        return ("NVIDIA", "NVIDIA");
    }
    if id.starts_with("MI") {
        return ("AMD", "AMD");
    }
    if id.starts_with("GAUDI") {
        return ("Intel Habana", "英特尔 Habana");
    }
    if id.starts_with("910") || id.starts_with("ATLAS") {
        return ("Huawei Ascend", "华为昇腾");
    }
    if id.starts_with("MXC") {
        return ("MetaX 沐曦", "沐曦 MetaX");
    }
    if id.starts_with("KUNLUN") {
        return ("Kunlunxin 昆仑芯", "昆仑芯 Kunlunxin");
    }
    if id.starts_with("BR") {
        return ("Biren 壁仞", "壁仞 Biren");
    }
    if id.starts_with("BI-") {
        return ("Iluvatar 天数智芯", "天数智芯 Iluvatar");
    }
    if id.starts_with("MR-") || id.starts_with("MTT") {
        return ("Moore Threads 摩尔线程", "摩尔线程 Moore Threads");
    }
    if id.starts_with("MLU") {
        return ("Cambricon 寒武纪", "寒武纪 Cambricon");
    }
    if id.starts_with("HYGON") {
        return ("Hygon 海光", "海光 Hygon");
    }
    ("Other", "其他")
}

async fn evaluate(Json(req): Json<EvaluateRequest>) -> Result<Json<Value>, ApiError> {
    if req.model_id.trim().is_empty() {
        return Err(ApiError::bad_request("model_id is required"));
    }
    let gpu_ids = requested_gpus(&req)?;

    let lang = req.lang.as_deref().unwrap_or("zh");
    if !matches!(lang, "en" | "zh") {
        return Err(ApiError::bad_request("lang must be en or zh"));
    }
    set_locale(lang);

    let timeout_s = req.timeout_s.unwrap_or(30.0);
    if timeout_s <= 0.0 {
        return Err(ApiError::bad_request("timeout_s must be greater than 0"));
    }
    if matches!(req.kv_cache_bits, Some(0)) {
        return Err(ApiError::bad_request(
            "kv_cache_bits must be greater than 0",
        ));
    }
    if matches!(req.target_concurrent_requests, Some(0)) {
        return Err(ApiError::bad_request(
            "target_concurrent_requests must be greater than 0",
        ));
    }
    let speculative_extra_weight_bytes = memory_bytes_from_request(
        req.speculative_extra_weight_bytes,
        req.speculative_extra_weight_gb,
        "speculative_extra_weight_gb",
    )?;
    let cpu_offload_bytes_per_gpu = memory_bytes_from_request(
        req.cpu_offload_bytes_per_gpu,
        req.cpu_offload_gb,
        "cpu_offload_gb",
    )?;

    let source_name = req.source.as_deref().unwrap_or("builtin");
    let source = source_from_name(source_name, timeout_s).map_err(ApiError::bad_request)?;
    let evaluator = Evaluator::new(source, ArtifactCache::with_default_ttl(None).ok());
    let defaults = EvaluationOptions::default();
    let options = EvaluationOptions {
        gpu_count: req.gpu_count,
        context_length: req.context_length,
        refresh: req.refresh.unwrap_or(false),
        input_tokens: req.input_tokens,
        output_tokens: req.output_tokens,
        target_tokens_per_sec: req.target_tokens_per_sec,
        prefill_utilization: req
            .prefill_utilization
            .unwrap_or(defaults.prefill_utilization),
        decode_bw_utilization: req
            .decode_bw_utilization
            .unwrap_or(defaults.decode_bw_utilization),
        concurrency_degradation: req
            .concurrency_degradation
            .unwrap_or(defaults.concurrency_degradation),
        kv_cache_bits: req.kv_cache_bits.unwrap_or(defaults.kv_cache_bits),
        paged_attention: req.paged_attention.unwrap_or(defaults.paged_attention),
        target_concurrent_requests: req.target_concurrent_requests,
        speculative_draft_model_id: req
            .speculative_draft_model_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string),
        speculative_extra_weight_bytes,
        cpu_offload_bytes_per_gpu,
    };

    let reports = gpu_ids
        .iter()
        .map(|gpu| {
            evaluator
                .evaluate(req.model_id.trim(), gpu, req.engine.trim(), options.clone())
                .map_err(ApiError::from_source_error)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let report_values = reports
        .iter()
        .map(report_to_value)
        .collect::<Result<Vec<_>, _>>()?;
    let mut value = report_values
        .first()
        .cloned()
        .ok_or_else(|| ApiError::bad_request("gpu is required"))?;

    if report_values.len() >= 2 {
        value["comparison"] = json!({
            "reports": report_values,
        });
        return Ok(Json(value));
    }

    let report = reports
        .first()
        .ok_or_else(|| ApiError::bad_request("gpu is required"))?;
    if req.explain.unwrap_or(false) || req.llm_review.unwrap_or(false) {
        let explain_entries = build_explain(report);
        if req.explain.unwrap_or(false) {
            value["explain_text"] = Value::String(render_explain_text(&explain_entries));
        }
        if req.llm_review.unwrap_or(false) {
            let review =
                run_review_with_env(&explain_entries, lang, reviewer_env_from_request(&req));
            value["llm_review_text"] = Value::String(render_llm_review_text(&review));
        }
    }

    Ok(Json(value))
}

fn memory_bytes_from_request(
    explicit_bytes: Option<u64>,
    gib: Option<f64>,
    field_name: &str,
) -> Result<u64, ApiError> {
    if let Some(value) = gib {
        if !value.is_finite() || value < 0.0 {
            return Err(ApiError::bad_request(format!(
                "{field_name} must be greater than or equal to 0"
            )));
        }
    }
    Ok(explicit_bytes.unwrap_or_else(|| {
        gib.map(|value| (value * 1024.0 * 1024.0 * 1024.0).round() as u64)
            .unwrap_or(0)
    }))
}

fn requested_gpus(req: &EvaluateRequest) -> Result<Vec<String>, ApiError> {
    let raw = req.gpus.clone().unwrap_or_else(|| vec![req.gpu.clone()]);
    let mut gpus = Vec::new();
    for gpu in raw {
        let trimmed = gpu.trim();
        if !trimmed.is_empty() && !gpus.iter().any(|existing| existing == trimmed) {
            gpus.push(trimmed.to_string());
        }
    }
    if gpus.is_empty() {
        return Err(ApiError::bad_request("gpu is required"));
    }
    if gpus.len() > 64 {
        return Err(ApiError::bad_request(
            "gpus accepts at most 64 items for comparison",
        ));
    }
    Ok(gpus)
}

fn report_to_value(
    report: &llm_infer_cal_core::core::evaluator::EvaluationReport,
) -> Result<Value, ApiError> {
    render_report_json(report)
        .and_then(|json| serde_json::from_str::<Value>(&json))
        .map_err(ApiError::internal)
}

fn source_from_name(name: &str, timeout_s: f64) -> Result<Box<dyn ModelSource>, String> {
    match name.to_lowercase().as_str() {
        "builtin" => Ok(Box::new(BuiltinSource)),
        "hf" | "huggingface" => Ok(Box::new(HuggingFaceSource::new(
            env_nonempty("HF_ENDPOINT").as_deref(),
            timeout_s,
        ))),
        "ms" | "modelscope" => Ok(Box::new(ModelScopeSource::new(
            env_nonempty("MODELSCOPE_ENDPOINT").as_deref(),
            timeout_s,
            DEFAULT_REVISION,
        ))),
        _ => Err(format!("unknown source '{name}'")),
    }
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

#[derive(Debug, Deserialize)]
struct EvaluateRequest {
    model_id: String,
    #[serde(default = "default_gpu")]
    gpu: String,
    gpus: Option<Vec<String>>,
    #[serde(default = "default_engine")]
    engine: String,
    source: Option<String>,
    gpu_count: Option<u64>,
    context_length: Option<u64>,
    refresh: Option<bool>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    target_tokens_per_sec: Option<f64>,
    prefill_utilization: Option<f64>,
    decode_bw_utilization: Option<f64>,
    concurrency_degradation: Option<f64>,
    kv_cache_bits: Option<u64>,
    paged_attention: Option<bool>,
    target_concurrent_requests: Option<u64>,
    speculative_draft_model_id: Option<String>,
    speculative_extra_weight_bytes: Option<u64>,
    speculative_extra_weight_gb: Option<f64>,
    cpu_offload_bytes_per_gpu: Option<u64>,
    cpu_offload_gb: Option<f64>,
    explain: Option<bool>,
    llm_review: Option<bool>,
    llm_review_api_key: Option<String>,
    llm_review_base_url: Option<String>,
    llm_review_model: Option<String>,
    timeout_s: Option<f64>,
    lang: Option<String>,
}

fn default_gpu() -> String {
    "H100".to_string()
}

fn default_engine() -> String {
    "vllm".to_string()
}

fn reviewer_env_from_request(req: &EvaluateRequest) -> EnvConfig {
    reviewer_env_from_request_with_base(req, EnvConfig::from_process_env())
}

fn reviewer_env_from_request_with_base(req: &EvaluateRequest, base: EnvConfig) -> EnvConfig {
    EnvConfig {
        api_key: request_nonempty(&req.llm_review_api_key).or(base.api_key),
        base_url: request_nonempty(&req.llm_review_base_url).or(base.base_url),
        model: request_nonempty(&req.llm_review_model).or(base.model),
    }
}

fn request_nonempty(value: &Option<String>) -> Option<String> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

#[derive(Debug, Deserialize, Serialize)]
struct BuiltinManifest {
    models: Vec<BuiltinModelSummary>,
}

#[derive(Debug, Deserialize, Serialize)]
struct BuiltinModelSummary {
    id: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    preferred_source: Option<String>,
    #[serde(default)]
    mentioned_by: Vec<String>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }

    fn from_source_error(error: ModelSourceError) -> Self {
        let status = match error {
            ModelSourceError::AuthRequired(_) => StatusCode::UNAUTHORIZED,
            ModelSourceError::NotFound(_) => StatusCode::NOT_FOUND,
            ModelSourceError::SourceUnavailable(_) => StatusCode::BAD_GATEWAY,
        };
        Self {
            status,
            message: error.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                }
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reviewer_env_uses_request_scoped_llm_config() {
        let req = request_with_review_config(
            Some(" sk-test "),
            Some(" https://api.deepseek.com/v1/ "),
            Some(" deepseek-chat "),
        );

        let env = reviewer_env_from_request_with_base(&req, EnvConfig::default());

        assert_eq!(env.api_key.as_deref(), Some("sk-test"));
        assert_eq!(
            env.base_url.as_deref(),
            Some("https://api.deepseek.com/v1/")
        );
        assert_eq!(env.model.as_deref(), Some("deepseek-chat"));
    }

    #[test]
    fn reviewer_env_ignores_blank_request_scoped_llm_config() {
        let req = request_with_review_config(Some(" "), None, Some("\t"));

        let env = reviewer_env_from_request_with_base(&req, EnvConfig::default());

        assert_eq!(env.api_key, None);
        assert_eq!(env.base_url, None);
        assert_eq!(env.model, None);
    }

    #[test]
    fn reviewer_env_falls_back_to_base_config() {
        let req = request_with_review_config(None, None, None);

        let env = reviewer_env_from_request_with_base(
            &req,
            EnvConfig {
                api_key: Some("sk-env".to_string()),
                base_url: Some("https://api.openai.com/v1".to_string()),
                model: Some("gpt-4o".to_string()),
            },
        );

        assert_eq!(env.api_key.as_deref(), Some("sk-env"));
        assert_eq!(env.base_url.as_deref(), Some("https://api.openai.com/v1"));
        assert_eq!(env.model.as_deref(), Some("gpt-4o"));
    }

    fn request_with_review_config(
        api_key: Option<&str>,
        base_url: Option<&str>,
        model: Option<&str>,
    ) -> EvaluateRequest {
        EvaluateRequest {
            model_id: "Qwen/Qwen3-30B-A3B".to_string(),
            gpu: "H100".to_string(),
            gpus: None,
            engine: "vllm".to_string(),
            source: Some("builtin".to_string()),
            gpu_count: None,
            context_length: None,
            refresh: None,
            input_tokens: None,
            output_tokens: None,
            target_tokens_per_sec: None,
            prefill_utilization: None,
            decode_bw_utilization: None,
            concurrency_degradation: None,
            kv_cache_bits: None,
            paged_attention: None,
            target_concurrent_requests: None,
            speculative_draft_model_id: None,
            speculative_extra_weight_bytes: None,
            speculative_extra_weight_gb: None,
            cpu_offload_bytes_per_gpu: None,
            cpu_offload_gb: None,
            explain: None,
            llm_review: Some(true),
            llm_review_api_key: api_key.map(ToOwned::to_owned),
            llm_review_base_url: base_url.map(ToOwned::to_owned),
            llm_review_model: model.map(ToOwned::to_owned),
            timeout_s: None,
            lang: None,
        }
    }
}
