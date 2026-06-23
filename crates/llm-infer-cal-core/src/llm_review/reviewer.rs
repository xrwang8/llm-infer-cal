use crate::core::explain::ExplainEntry;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewHttpResponse {
    pub status: u16,
    pub body: String,
}

pub trait ReviewHttpClient {
    fn post_chat_completion(
        &mut self,
        url: &str,
        api_key: &str,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
        timeout_s: f64,
    ) -> Result<ReviewHttpResponse, String>;
}

#[derive(Default)]
pub struct ReqwestReviewHttpClient;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlmReviewResult {
    pub ok: bool,
    pub content: Option<String>,
    pub error: Option<String>,
    pub model: String,
    pub base_url: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EnvConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
}

pub fn run_review(entries: &[ExplainEntry], locale: &str) -> LlmReviewResult {
    run_review_with_env(entries, locale, EnvConfig::from_process_env())
}

pub fn run_review_with_env(
    entries: &[ExplainEntry],
    locale: &str,
    env: EnvConfig,
) -> LlmReviewResult {
    let mut client = ReqwestReviewHttpClient;
    run_review_with_client(entries, locale, env, 60.0, &mut client)
}

pub fn run_review_with_client(
    entries: &[ExplainEntry],
    locale: &str,
    env: EnvConfig,
    timeout_s: f64,
    client: &mut dyn ReviewHttpClient,
) -> LlmReviewResult {
    let base_url = normalized_base_url(env.base_url);
    let model = env.model.unwrap_or_else(|| "gpt-4o".to_string());

    let api_key = env.api_key.unwrap_or_default();
    if api_key.is_empty() {
        return LlmReviewResult {
            ok: false,
            content: None,
            error: Some(
                "LLM_CAL_REVIEWER_API_KEY env var not set. Set it to the API key of an OpenAI-compatible endpoint (OpenAI, DeepSeek, Moonshot, Zhipu, etc.)."
                    .to_string(),
            ),
            model,
            base_url,
        };
    }

    let prompt = build_prompt(entries, locale);
    let system = system_prompt(locale);
    let url = format!("{base_url}/chat/completions");
    let response =
        match client.post_chat_completion(&url, &api_key, &model, &system, &prompt, timeout_s) {
            Ok(response) => response,
            Err(error) => {
                return LlmReviewResult {
                    ok: false,
                    content: None,
                    error: Some(error),
                    model,
                    base_url,
                };
            }
        };

    if response.status != 200 {
        return LlmReviewResult {
            ok: false,
            content: None,
            error: Some(format!(
                "HTTP {}: {}",
                response.status,
                truncate_chars(&response.body, 500)
            )),
            model,
            base_url,
        };
    }

    let content = match parse_content(&response.body) {
        Ok(content) => content,
        Err(error) => {
            return LlmReviewResult {
                ok: false,
                content: None,
                error: Some(format!("Malformed response: {error}")),
                model,
                base_url,
            };
        }
    };

    LlmReviewResult {
        ok: true,
        content: Some(content),
        error: None,
        model,
        base_url,
    }
}

pub fn system_prompt(locale: &str) -> String {
    if locale == "zh" {
        return ("你是一个大模型推理硬件计算工具的独立审计者。工具产出确定性的推导链，\
你的工作是发现数学错误、不合理假设或遗漏。你不负责重新计算，\
只负责评论和确认。输出简体中文。")
            .to_string();
    }
    ("You are an independent auditor for a deterministic LLM inference hardware \
calculator. The tool produces a derivation trace; your job is to find math \
errors, unreasonable assumptions, or missing considerations. You do NOT \
recalculate; you only critique and confirm.")
        .to_string()
}

pub fn build_prompt(entries: &[ExplainEntry], locale: &str) -> String {
    let trace = entries
        .iter()
        .map(format_entry)
        .collect::<Vec<_>>()
        .join("\n\n");
    if locale == "zh" {
        prompt_zh(&trace)
    } else {
        prompt_en(&trace)
    }
}

pub fn format_entry(entry: &ExplainEntry) -> String {
    let mut parts = vec![
        format!("## {}", entry.heading),
        format!("Formula:\n{}", entry.formula),
    ];

    if !entry.inputs.is_empty() {
        parts.push("Inputs:".to_string());
        for input in &entry.inputs {
            let note = if input.note.is_empty() {
                String::new()
            } else {
                format!(" ({})", input.note)
            };
            parts.push(format!(
                "  - {} = {} {}{}",
                input.name, input.value, input.label, note
            ));
        }
    }

    if !entry.steps.is_empty() {
        parts.push("Steps:".to_string());
        for step in &entry.steps {
            parts.push(format!("  {step}"));
        }
    }

    parts.push(format!("Result: {}", entry.result));
    if !entry.source.is_empty() {
        parts.push(format!("Source: {}", entry.source));
    }

    parts.join("\n")
}

impl EnvConfig {
    pub fn from_process_env() -> Self {
        Self {
            api_key: std::env::var("LLM_CAL_REVIEWER_API_KEY").ok(),
            base_url: std::env::var("LLM_CAL_REVIEWER_BASE_URL").ok(),
            model: std::env::var("LLM_CAL_REVIEWER_MODEL").ok(),
        }
    }
}

impl ReviewHttpClient for ReqwestReviewHttpClient {
    fn post_chat_completion(
        &mut self,
        url: &str,
        api_key: &str,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
        timeout_s: f64,
    ) -> Result<ReviewHttpResponse, String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs_f64(timeout_s))
            .build()
            .map_err(|error| format!("ClientError: {error}"))?;
        let payload = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 6000,
        });
        let response = client
            .post(url)
            .bearer_auth(api_key)
            .header("Content-Type", "application/json")
            .body(payload.to_string())
            .send()
            .map_err(|error| error.to_string())?;
        let status = response.status().as_u16();
        let body = response.text().map_err(|error| error.to_string())?;
        Ok(ReviewHttpResponse { status, body })
    }
}

fn normalized_base_url(base_url: Option<String>) -> String {
    base_url
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
        .trim_end_matches('/')
        .to_string()
}

fn parse_content(body: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(body).map_err(|error| error.to_string())?;
    value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "missing choices[0].message.content".to_string())
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn prompt_en(trace: &str) -> String {
    format!(
        "The deterministic tool produced this derivation trace for one model evaluation. Audit it.\n\n\
<DERIVATION_TRACE>\n{trace}\n</DERIVATION_TRACE>\n\n\
Respond in this structure. If a section has nothing to flag, write \"none\".\n\n\
## Critical issues\n\
(math errors or wrong formulas - would give wrong final answer)\n\n\
## Moderate concerns\n\
(unreasonable assumptions, factors off by 2x+, missing TP/sharding effects, etc.)\n\n\
## Minor notes\n\
(clarifications, stylistic, optional improvements)\n\n\
## Consensus check\n\
(which ExplainEntry headings look correct? name them explicitly)\n\n\
Rules:\n\
  - Cite specific ExplainEntry heading names. Be concrete.\n\
  - Do NOT produce new numbers. Only critique.\n\
  - If you don't know, say so. Do not hallucinate.\n\
  - All your output must be tagged as a second opinion, NOT authoritative."
    )
}

fn prompt_zh(trace: &str) -> String {
    format!(
        "下面是工具产出的一份完整推导链。请审计。\n\n\
<DERIVATION_TRACE>\n{trace}\n</DERIVATION_TRACE>\n\n\
按下面结构回复。没内容的段落写\"无\"。\n\n\
## 关键错误\n\
（数学错误或公式错误 -- 会导致最终答案错）\n\n\
## 中度疑虑\n\
（不合理假设、因子偏差 2x+、遗漏的 TP 分摊等）\n\n\
## 次要备注\n\
（澄清、风格、可选改进）\n\n\
## 一致性核查\n\
（哪些 ExplainEntry 标题看起来是对的？明确列出）\n\n\
规则：\n\
  - 必须引用具体的 ExplainEntry 标题名。具体点。\n\
  - 不要产出新数字，只做评论。\n\
  - 不确定的地方直说。不要编造。\n\
  - 你的所有输出都只是 second opinion，不是权威答案。"
    )
}
