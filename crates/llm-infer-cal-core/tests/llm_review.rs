use llm_infer_cal_core::core::explain::{ExplainEntry, ExplainInput};
use llm_infer_cal_core::llm_review::reviewer::{
    build_prompt, format_entry, run_review_with_client, run_review_with_env, system_prompt,
    EnvConfig, ReviewHttpClient, ReviewHttpResponse,
};

fn sample_entries() -> Vec<ExplainEntry> {
    vec![
        ExplainEntry {
            heading: "Weight bytes".to_string(),
            formula: "sum(safetensors.size)".to_string(),
            inputs: vec![ExplainInput::new("api", "HF siblings", "[verified]")],
            steps: vec!["result = 160 GB".to_string()],
            result: "160 GB [verified]".to_string(),
            source: "HF API".to_string(),
            methodology_anchor: String::new(),
        },
        ExplainEntry {
            heading: "Prefill latency".to_string(),
            formula: "2 x params x input_tokens / TFLOPS".to_string(),
            inputs: vec![
                ExplainInput::new("params", "284B", "[estimated]"),
                ExplainInput::new("input_tokens", "2000", "[user-set]"),
            ],
            steps: vec!["FLOPs = 1.1e15".to_string(), "latency = 735 ms".to_string()],
            result: "735 ms [estimated]".to_string(),
            source: "Kaplan 2020".to_string(),
            methodology_anchor: String::new(),
        },
    ]
}

#[test]
fn prompts_contain_entry_data_and_locale_specific_instructions() {
    let entries = sample_entries();
    let english = build_prompt(&entries, "en");
    let chinese = build_prompt(&entries, "zh");

    for entry in &entries {
        assert!(english.contains(&entry.heading));
        assert!(english.contains(&entry.result));
        assert!(chinese.contains(&entry.heading));
    }
    assert!(english.contains("Critical issues"));
    assert!(chinese.contains("关键错误"));
    assert!(system_prompt("en").to_lowercase().contains("auditor"));
    assert!(system_prompt("zh").contains("审计"));
}

#[test]
fn format_entry_includes_all_parts_like_python() {
    let entry = &sample_entries()[1];
    let formatted = format_entry(entry);

    assert!(formatted.contains("## Prefill latency"));
    assert!(formatted.contains("Formula:"));
    assert!(formatted.contains("Inputs:"));
    assert!(formatted.contains("Steps:"));
    assert!(formatted.contains("Result:"));
    assert!(formatted.contains("Source:"));
    assert!(formatted.contains("  - params = 284B [estimated]"));
}

#[test]
fn missing_api_key_returns_graceful_error_with_defaults() {
    let result = run_review_with_env(
        &sample_entries(),
        "en",
        EnvConfig {
            api_key: None,
            base_url: None,
            model: None,
        },
    );

    assert!(!result.ok);
    assert!(result.content.is_none());
    assert!(result
        .error
        .as_deref()
        .unwrap_or("")
        .contains("LLM_CAL_REVIEWER_API_KEY"));
    assert_eq!(result.model, "gpt-4o");
    assert_eq!(result.base_url, "https://api.openai.com/v1");
}

#[test]
fn env_config_trims_base_url_and_uses_custom_model() {
    let result = run_review_with_env(
        &sample_entries(),
        "en",
        EnvConfig {
            api_key: None,
            base_url: Some("https://api.deepseek.com/v1/".to_string()),
            model: Some("deepseek-chat".to_string()),
        },
    );

    assert_eq!(result.base_url, "https://api.deepseek.com/v1");
    assert_eq!(result.model, "deepseek-chat");
    assert!(!result.ok);
    assert!(result.error.is_some());
}

#[derive(Default)]
struct FakeClient {
    response: Option<Result<ReviewHttpResponse, String>>,
    seen_url: Option<String>,
    seen_api_key: Option<String>,
    seen_model: Option<String>,
    seen_system: Option<String>,
    seen_user: Option<String>,
}

impl ReviewHttpClient for FakeClient {
    fn post_chat_completion(
        &mut self,
        url: &str,
        api_key: &str,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
        _timeout_s: f64,
    ) -> Result<ReviewHttpResponse, String> {
        self.seen_url = Some(url.to_string());
        self.seen_api_key = Some(api_key.to_string());
        self.seen_model = Some(model.to_string());
        self.seen_system = Some(system_prompt.to_string());
        self.seen_user = Some(user_prompt.to_string());
        self.response
            .take()
            .unwrap_or_else(|| Err("missing fake response".to_string()))
    }
}

fn env_with_key() -> EnvConfig {
    EnvConfig {
        api_key: Some("sk-test".to_string()),
        base_url: Some("https://api.example.com/v1/".to_string()),
        model: Some("audit-model".to_string()),
    }
}

#[test]
fn run_review_with_client_returns_content_on_success() {
    let mut client = FakeClient {
        response: Some(Ok(ReviewHttpResponse {
            status: 200,
            body: r#"{"choices":[{"message":{"content":"looks good"}}]}"#.to_string(),
        })),
        ..FakeClient::default()
    };

    let result = run_review_with_client(&sample_entries(), "zh", env_with_key(), 12.0, &mut client);

    assert!(result.ok);
    assert_eq!(result.content.as_deref(), Some("looks good"));
    assert_eq!(result.error, None);
    assert_eq!(result.model, "audit-model");
    assert_eq!(result.base_url, "https://api.example.com/v1");
    assert_eq!(
        client.seen_url.as_deref(),
        Some("https://api.example.com/v1/chat/completions")
    );
    assert_eq!(client.seen_api_key.as_deref(), Some("sk-test"));
    assert_eq!(client.seen_model.as_deref(), Some("audit-model"));
    assert!(client.seen_system.as_deref().unwrap_or("").contains("审计"));
    assert!(client
        .seen_user
        .as_deref()
        .unwrap_or("")
        .contains("DERIVATION_TRACE"));
}

#[test]
fn run_review_with_client_maps_http_and_malformed_errors() {
    let mut http_client = FakeClient {
        response: Some(Ok(ReviewHttpResponse {
            status: 500,
            body: "server unavailable".to_string(),
        })),
        ..FakeClient::default()
    };
    let http = run_review_with_client(
        &sample_entries(),
        "en",
        env_with_key(),
        60.0,
        &mut http_client,
    );
    assert!(!http.ok);
    assert_eq!(http.error.as_deref(), Some("HTTP 500: server unavailable"));

    let mut malformed_client = FakeClient {
        response: Some(Ok(ReviewHttpResponse {
            status: 200,
            body: r#"{"choices":[]}"#.to_string(),
        })),
        ..FakeClient::default()
    };
    let malformed = run_review_with_client(
        &sample_entries(),
        "en",
        env_with_key(),
        60.0,
        &mut malformed_client,
    );
    assert!(!malformed.ok);
    assert!(malformed
        .error
        .as_deref()
        .unwrap_or("")
        .contains("Malformed response"));

    let mut network_client = FakeClient {
        response: Some(Err("ConnectError: refused".to_string())),
        ..FakeClient::default()
    };
    let network = run_review_with_client(
        &sample_entries(),
        "en",
        env_with_key(),
        60.0,
        &mut network_client,
    );
    assert!(!network.ok);
    assert_eq!(network.error.as_deref(), Some("ConnectError: refused"));
}
