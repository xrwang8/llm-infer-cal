use std::ffi::OsString;

use clap::{Parser, ValueEnum};
use llm_infer_cal_core::benchmark::runner::{
    exit_code_from, load_dataset, run_all, CheckResult, Status,
};
use llm_infer_cal_core::common::i18n::{detect_locale_from_env, get_locale, set_locale, t, t_with};
use llm_infer_cal_core::core::cache::ArtifactCache;
use llm_infer_cal_core::core::evaluator::{EvaluationOptions, Evaluator, SpeculativeMode};
use llm_infer_cal_core::core::explain::build as build_explain;
use llm_infer_cal_core::hardware::loader::load_database;
use llm_infer_cal_core::llm_review::reviewer::run_review;
use llm_infer_cal_core::model_source::base::{ModelSource, ModelSourceError};
use llm_infer_cal_core::model_source::builtin::BuiltinSource;
use llm_infer_cal_core::model_source::huggingface::HuggingFaceSource;
use llm_infer_cal_core::model_source::modelscope::{ModelScopeSource, DEFAULT_REVISION};
use llm_infer_cal_core::output::formatter::{
    render_explain_text, render_gpu_list_text, render_llm_review_text, render_report_text,
};

const HELP_TEXT: &str = r#"LLM inference hardware calculator.

Usage: llm-infer-cal [OPTIONS] [MODEL_ID]

Arguments:
  [MODEL_ID]  HuggingFace or ModelScope model id

Options:
      --gpu <GPU>
          GPU type, e.g. H800, A100-80G
      --engine <ENGINE>
          Inference engine: vllm | sglang [default: vllm]
      --gpu-count <GPU_COUNT>
          Force GPU count (otherwise tool recommends)
      --context-length <CONTEXT_LENGTH>
          Context length for KV cache estimation
      --refresh
          Bypass cache and re-fetch
      --timeout-s <TIMEOUT_S>
          Network timeout in seconds for model metadata requests [default: 30]
      --lang <LANG>
          Output language: en | zh (default auto-detects from LANG env)
      --list-gpus
          List all supported GPUs and exit (no model_id needed)
      --benchmark
          Run the curated benchmark dataset. Requires network
      --input-tokens <INPUT_TOKENS>
          Input token budget for prefill-latency estimation [default: 2000]
      --output-tokens <OUTPUT_TOKENS>
          Output token budget for total-latency math [default: 512]
      --target-tokens-per-sec <TARGET_TOKENS_PER_SEC>
          SLA: per-user decode tokens/second (drives L bound) [default: 30]
      --prefill-util <PREFILL_UTILIZATION>
          Compute utilization factor for prefill [default: 0.4]
          Alias: --prefill-utilization
      --decode-bw-util <DECODE_BW_UTILIZATION>
          Memory-bandwidth utilization factor for decode [default: 0.5]
          Alias: --decode-bw-utilization
      --concurrency-degradation <CONCURRENCY_DEGRADATION>
          High-concurrency throughput degradation factor [default: 1]
      --kv-cache-bits <KV_CACHE_BITS>
          KV cache precision in bits per element [default: 16]
      --paged-attention
          Apply paged-attention KV memory factor (0.75)
      --target-concurrency <TARGET_CONCURRENT_REQUESTS>
          Target concurrent requests for VRAM pressure planning
          Alias: --target-concurrent-requests
      --speculative-enabled
          Enable speculative decoding (MTP mode)
      --speculative-mode <SPECULATIVE_MODE>
          Speculative decoding mode: mtp (default and only supported)
      --speculative-num-draft-tokens <SPECULATIVE_NUM_DRAFT_TOKENS>
          Number of draft tokens for speculative decoding [default: 8]
      --speculative-draft-model <SPECULATIVE_DRAFT_MODEL_ID>
          Draft/EAGLE model id; its safetensors size is added to resident VRAM (standard mode only)
          Alias: --speculative-draft-model-id
      --speculative-extra-weight-gb <SPECULATIVE_EXTRA_WEIGHT_GB>
          Additional speculative decoding resident weight in GiB [default: 0]
      --expert-offloading
          Enable MoE expert offloading (keep subset of experts on GPU)
      --experts-on-gpu <EXPERTS_ON_GPU>
          Number of MoE experts to keep on GPU (requires --expert-offloading)
      --cpu-offload-gb <CPU_OFFLOAD_GB>
          Per-GPU CPU/offload budget in GiB, subtracted from GPU-resident weights [default: 0]
      --explain
          Print the full derivation trace
      --llm-review
          EXPERIMENTAL: send the derivation trace to an LLM for a second opinion
      --source <SOURCE>
          Model source: builtin | huggingface (default) | modelscope [default: huggingface]
      --format <FORMAT>
          Output format: text | json [default: text]
      --json
          Shortcut for --format json
      --install-completion <SHELL>
          Print portable shell completion install instructions
      --show-completion <SHELL>
          Show shell completion script
  -h, --help
          Print help
"#;

const COMPLETION_OPTIONS: &[&str] = &[
    "--gpu",
    "--engine",
    "--gpu-count",
    "--context-length",
    "--refresh",
    "--timeout-s",
    "--lang",
    "--list-gpus",
    "--benchmark",
    "--input-tokens",
    "--output-tokens",
    "--target-tokens-per-sec",
    "--prefill-util",
    "--prefill-utilization",
    "--decode-bw-util",
    "--decode-bw-utilization",
    "--concurrency-degradation",
    "--kv-cache-bits",
    "--paged-attention",
    "--target-concurrency",
    "--target-concurrent-requests",
    "--speculative-enabled",
    "--speculative-mode",
    "--speculative-num-draft-tokens",
    "--speculative-draft-model",
    "--speculative-draft-model-id",
    "--speculative-extra-weight-gb",
    "--expert-offloading",
    "--experts-on-gpu",
    "--cpu-offload-gb",
    "--explain",
    "--llm-review",
    "--source",
    "--format",
    "--json",
    "--install-completion",
    "--show-completion",
    "--help",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliExit {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Parser, Debug)]
#[command(
    name = "llm-infer-cal",
    about = "LLM inference hardware calculator.",
    disable_help_subcommand = true
)]
struct Cli {
    /// HuggingFace or ModelScope model id
    model_id: Option<String>,

    /// GPU type, e.g. H800, A100-80G
    #[arg(long)]
    gpu: Option<String>,

    /// Inference engine: vllm | sglang
    #[arg(long, default_value = "vllm")]
    engine: String,

    /// Force GPU count (otherwise tool recommends)
    #[arg(long = "gpu-count")]
    gpu_count: Option<u64>,

    /// Context length for KV cache estimation
    #[arg(long = "context-length")]
    context_length: Option<u64>,

    /// Bypass cache and re-fetch
    #[arg(long)]
    refresh: bool,

    /// Network timeout in seconds for model metadata requests.
    #[arg(long = "timeout-s", default_value_t = 30.0)]
    timeout_s: f64,

    /// Output language: en | zh (default auto-detects from LANG env)
    #[arg(long)]
    lang: Option<String>,

    /// List all supported GPUs and exit (no model_id needed)
    #[arg(long = "list-gpus")]
    list_gpus: bool,

    /// Run the curated benchmark dataset. Requires network.
    #[arg(long)]
    benchmark: bool,

    /// Input token budget for prefill-latency estimation.
    #[arg(long = "input-tokens", default_value_t = 2000)]
    input_tokens: u64,

    /// Output token budget for total-latency math.
    #[arg(long = "output-tokens", default_value_t = 512)]
    output_tokens: u64,

    /// SLA: per-user decode tokens/second (drives L bound).
    #[arg(long = "target-tokens-per-sec", default_value_t = 30.0)]
    target_tokens_per_sec: f64,

    /// Compute utilization factor for prefill.
    #[arg(
        long = "prefill-utilization",
        alias = "prefill-util",
        default_value_t = 0.40
    )]
    prefill_utilization: f64,

    /// Memory-bandwidth utilization factor for decode.
    #[arg(
        long = "decode-bw-utilization",
        alias = "decode-bw-util",
        default_value_t = 0.50
    )]
    decode_bw_utilization: f64,

    /// High-concurrency throughput degradation factor.
    #[arg(long = "concurrency-degradation", default_value_t = 1.0)]
    concurrency_degradation: f64,

    /// KV cache precision in bits per element.
    #[arg(long = "kv-cache-bits", default_value_t = 16)]
    kv_cache_bits: u64,

    /// Apply paged-attention KV memory factor (0.75).
    #[arg(long = "paged-attention")]
    paged_attention: bool,

    /// Target concurrent requests for VRAM pressure planning.
    #[arg(long = "target-concurrent-requests", alias = "target-concurrency")]
    target_concurrent_requests: Option<u64>,

    /// Enable speculative decoding (MTP mode by default).
    #[arg(long = "speculative-enabled")]
    speculative_enabled: bool,

    /// Speculative decoding mode (only 'mtp' supported; evaluation always uses MTP).
    #[arg(long = "speculative-mode", default_value = "mtp")]
    speculative_mode: String,

    /// Number of draft tokens for speculative decoding (MTP mode).
    #[arg(long = "speculative-num-draft-tokens", default_value_t = 8)]
    speculative_num_draft_tokens: u64,

    /// Draft/EAGLE model id for standard speculative mode; its safetensors size is added to resident VRAM.
    #[arg(long = "speculative-draft-model-id", alias = "speculative-draft-model")]
    speculative_draft_model_id: Option<String>,

    /// Additional speculative decoding resident weight in GiB.
    #[arg(long = "speculative-extra-weight-gb", default_value_t = 0.0)]
    speculative_extra_weight_gb: f64,

    /// Enable MoE expert offloading (keep subset of experts on GPU).
    #[arg(long = "expert-offloading")]
    expert_offloading: bool,

    /// Number of MoE experts to keep on GPU (requires --expert-offloading).
    #[arg(long = "experts-on-gpu")]
    experts_on_gpu: Option<u64>,

    /// Per-GPU CPU/offload budget in GiB, subtracted from GPU-resident weights.
    #[arg(long = "cpu-offload-gb", default_value_t = 0.0)]
    cpu_offload_gb: f64,

    /// Print the full derivation trace.
    #[arg(long)]
    explain: bool,

    /// EXPERIMENTAL: send the derivation trace to an LLM for a second opinion.
    #[arg(long = "llm-review")]
    llm_review: bool,

    /// Model source: builtin | huggingface (default) | modelscope.
    #[arg(long, default_value = "huggingface")]
    source: String,

    /// Output format: text | json.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    /// Shortcut for --format json.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

pub fn run_with_args<I, S>(_args: I) -> CliExit
where
    I: IntoIterator<Item = S>,
    S: Into<OsString> + Clone,
{
    let args = _args.into_iter().map(Into::into).collect::<Vec<_>>();
    set_locale(detect_locale_from_env());

    if args.len() <= 1
        || args
            .iter()
            .skip(1)
            .any(|arg| matches!(arg.to_string_lossy().as_ref(), "-h" | "--help"))
    {
        return CliExit::ok(HELP_TEXT.to_string());
    }

    if let Some(exit) = handle_completion_args(&args) {
        return exit;
    }

    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error) => {
            let output = error.to_string();
            let code = error.exit_code();
            if error.use_stderr() {
                return CliExit::err(code, output);
            }
            return CliExit::ok(output);
        }
    };

    if matches!(cli.lang.as_deref(), Some("en" | "zh")) {
        set_locale(cli.lang.as_deref().unwrap());
    }
    if cli.timeout_s <= 0.0 {
        return CliExit::err(1, "--timeout-s must be greater than 0.\n");
    }
    if cli.kv_cache_bits == 0 {
        return CliExit::err(1, "--kv-cache-bits must be greater than 0.\n");
    }
    if matches!(cli.target_concurrent_requests, Some(0)) {
        return CliExit::err(1, "--target-concurrent-requests must be greater than 0.\n");
    }
    if !cli.speculative_extra_weight_gb.is_finite() || cli.speculative_extra_weight_gb < 0.0 {
        return CliExit::err(
            1,
            "--speculative-extra-weight-gb must be greater than or equal to 0.\n",
        );
    }
    if !cli.cpu_offload_gb.is_finite() || cli.cpu_offload_gb < 0.0 {
        return CliExit::err(1, "--cpu-offload-gb must be greater than or equal to 0.\n");
    }

    if cli.list_gpus {
        return match load_database() {
            Ok(database) => CliExit::ok(with_newline(render_gpu_list_text(&database))),
            Err(error) => CliExit::err(1, format!("Failed to load GPU database: {error}\n")),
        };
    }

    if cli.benchmark {
        let dataset = match load_dataset() {
            Ok(dataset) => dataset,
            Err(error) => {
                return CliExit::err(1, format!("Failed to load benchmark dataset: {error}\n"));
            }
        };
        let evaluator = Evaluator::default();
        let results = run_all(&evaluator, &dataset);
        return CliExit {
            code: exit_code_from(&results),
            stdout: with_newline(render_benchmark_results_text(&results)),
            stderr: String::new(),
        };
    }

    let Some(model_id) = cli.model_id.as_deref() else {
        return CliExit::err(1, format!("{}\n", t("cli.err.missing_model")));
    };
    let Some(gpu) = cli.gpu.as_deref() else {
        return CliExit::err(1, format!("{}\n", t("cli.err.missing_gpu")));
    };

    let source = match source_from_name(&cli.source, cli.timeout_s) {
        Ok(source) => source,
        Err(message) => return CliExit::err(1, message),
    };
    let evaluator = Evaluator::new(source, ArtifactCache::with_default_ttl(None).ok());

    // Speculative decoding draft model (for standard mode only)
    let speculative_draft_model_id = cli
        .speculative_draft_model_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    let speculative_extra_weight_bytes = gib_to_bytes(cli.speculative_extra_weight_gb);

    // Determine speculative mode: always MTP when enabled
    let speculative_mode = if cli.speculative_mode.trim().eq_ignore_ascii_case("mtp") {
        SpeculativeMode::Mtp
    } else if cli.speculative_mode.trim().eq_ignore_ascii_case("standard") {
        SpeculativeMode::Standard
    } else {
        // Default to MTP if unknown
        SpeculativeMode::Mtp
    };

    let options = EvaluationOptions {
        gpu_count: cli.gpu_count,
        context_length: cli.context_length,
        refresh: cli.refresh,
        input_tokens: Some(cli.input_tokens),
        output_tokens: Some(cli.output_tokens),
        target_tokens_per_sec: Some(cli.target_tokens_per_sec),
        prefill_utilization: cli.prefill_utilization,
        decode_bw_utilization: cli.decode_bw_utilization,
        concurrency_degradation: cli.concurrency_degradation,
        kv_cache_bits: cli.kv_cache_bits,
        paged_attention: cli.paged_attention,
        target_concurrent_requests: cli.target_concurrent_requests,
        speculative_enabled: cli.speculative_enabled,
        speculative_mode,
        speculative_num_draft_tokens: if cli.speculative_enabled {
            Some(cli.speculative_num_draft_tokens)
        } else {
            None
        },
        speculative_draft_model_id,
        speculative_extra_weight_bytes,
        cpu_offload_bytes_per_gpu: gib_to_bytes(cli.cpu_offload_gb),
        expert_offloading: cli.expert_offloading,
        experts_on_gpu: cli.experts_on_gpu,
    };

    let report = match evaluator.evaluate(model_id, gpu, &cli.engine, options) {
        Ok(report) => report,
        Err(error) => return source_error_exit(error),
    };

    let output_format = if cli.json {
        OutputFormat::Json
    } else {
        cli.format
    };
    if output_format == OutputFormat::Json {
        return match llm_infer_cal_core::output::formatter::render_report_json(&report) {
            Ok(json) => CliExit::ok(with_newline(json)),
            Err(error) => CliExit::err(1, format!("failed to render JSON: {error}\n")),
        };
    }

    let mut stdout = render_report_text(&report);
    let explain_entries = if cli.explain || cli.llm_review {
        build_explain(&report)
    } else {
        Vec::new()
    };

    if cli.explain {
        stdout.push_str("\n\n");
        stdout.push_str(&render_explain_text(&explain_entries));
    }
    if cli.llm_review {
        let review = run_review(&explain_entries, &get_locale());
        stdout.push_str("\n\n");
        stdout.push_str(&render_llm_review_text(&review));
    }

    CliExit::ok(with_newline(stdout))
}

impl CliExit {
    fn ok(stdout: impl Into<String>) -> Self {
        Self {
            code: 0,
            stdout: stdout.into(),
            stderr: String::new(),
        }
    }

    fn err(code: i32, stderr: impl Into<String>) -> Self {
        Self {
            code,
            stdout: String::new(),
            stderr: stderr.into(),
        }
    }
}

fn source_from_name(name: &str, timeout_s: f64) -> Result<Box<dyn ModelSource>, String> {
    match name.to_lowercase().as_str() {
        "builtin" => Ok(Box::new(BuiltinSource)),
        "hf" | "huggingface" => {
            let endpoint = env_nonempty("HF_ENDPOINT");
            Ok(Box::new(HuggingFaceSource::new(
                endpoint.as_deref(),
                timeout_s,
            )))
        }
        "ms" | "modelscope" => {
            let endpoint = env_nonempty("MODELSCOPE_ENDPOINT");
            Ok(Box::new(ModelScopeSource::new(
                endpoint.as_deref(),
                timeout_s,
                DEFAULT_REVISION,
            )))
        }
        _ => Err(format!(
            "{}\n",
            t_with(
                "cli.err.unknown_source",
                &std::collections::HashMap::from([("source", name.to_string())])
            )
        )),
    }
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

fn source_error_exit(error: ModelSourceError) -> CliExit {
    match error {
        ModelSourceError::AuthRequired(error) => {
            CliExit::err(2, format!("{} {error}\n", t("cli.err.auth_required")))
        }
        ModelSourceError::NotFound(error) => {
            CliExit::err(3, format!("{} {error}\n", t("cli.err.model_not_found")))
        }
        ModelSourceError::SourceUnavailable(error) => {
            CliExit::err(4, format!("{} {error}\n", t("cli.err.source_unavailable")))
        }
    }
}

fn render_benchmark_results_text(results: &[CheckResult]) -> String {
    let mut out = vec![
        "Benchmark results".to_string(),
        "entry | field | predicted | expected | status".to_string(),
    ];
    let mut current_entry = "";

    for result in results {
        let entry = if result.entry_name == current_entry {
            ""
        } else {
            current_entry = &result.entry_name;
            &result.entry_name
        };
        out.push(format!(
            "{} | {} | {} | {} | {}",
            entry,
            result.field,
            result.predicted,
            result.expected,
            result.status.as_str()
        ));
    }

    let total = results.len();
    let passed = results
        .iter()
        .filter(|result| result.status == Status::Pass)
        .count();
    let failed = results
        .iter()
        .filter(|result| result.status == Status::Fail)
        .count();
    let skipped = results
        .iter()
        .filter(|result| result.status == Status::Skip)
        .count();
    out.push(format!(
        "Total: {total}   PASS: {passed}   FAIL: {failed}   SKIP: {skipped}"
    ));

    if failed > 0 {
        out.push(
            "Failures show the tool's prediction diverges from a curated source. Check the source column for the expected-value provenance."
                .to_string(),
        );
    }

    out.join("\n")
}

fn with_newline(mut value: String) -> String {
    if !value.ends_with('\n') {
        value.push('\n');
    }
    value
}

fn gib_to_bytes(value: f64) -> u64 {
    (value * 1024.0 * 1024.0 * 1024.0).round() as u64
}

fn handle_completion_args(args: &[OsString]) -> Option<CliExit> {
    for flag in ["--show-completion", "--install-completion"] {
        let Some(shell) = read_completion_shell(args, flag) else {
            continue;
        };
        let script = match completion_script(&shell) {
            Ok(script) => script,
            Err(message) => return Some(CliExit::err(1, format!("{message}\n"))),
        };
        if flag == "--show-completion" {
            return Some(CliExit::ok(script));
        }
        return Some(CliExit::ok(completion_install_text(&shell, &script)));
    }
    None
}

fn read_completion_shell(args: &[OsString], flag: &str) -> Option<String> {
    let prefix = format!("{flag}=");
    for (idx, arg) in args.iter().enumerate().skip(1) {
        let value = arg.to_string_lossy();
        if value == flag {
            return Some(
                args.get(idx + 1)
                    .and_then(|next| {
                        let next = next.to_string_lossy();
                        (!next.starts_with('-')).then(|| next.to_lowercase())
                    })
                    .unwrap_or_else(default_completion_shell),
            );
        }
        if let Some(shell) = value.strip_prefix(&prefix) {
            return Some(shell.to_lowercase());
        }
    }
    None
}

fn default_completion_shell() -> String {
    std::env::var("SHELL")
        .ok()
        .and_then(|shell| shell.rsplit('/').next().map(str::to_lowercase))
        .filter(|shell| {
            matches!(
                shell.as_str(),
                "bash" | "zsh" | "fish" | "powershell" | "pwsh"
            )
        })
        .unwrap_or_else(|| "zsh".to_string())
}

fn completion_script(shell: &str) -> Result<String, String> {
    match shell {
        "zsh" => {
            let option_lines = COMPLETION_OPTIONS
                .iter()
                .map(|option| format!("    '{option}'"))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(format!(
                "#compdef llm-infer-cal\n\n\
                 _llm_infer_cal_completion() {{\n\
                 \u{20} local -a options\n\
                 \u{20} options=(\n\
                 {option_lines}\n\
                 \u{20} )\n\
                 \u{20} _describe 'llm-infer-cal options' options\n\
                 }}\n\n\
                 compdef _llm_infer_cal_completion llm-infer-cal\n"
            ))
        }
        "bash" => {
            let words = COMPLETION_OPTIONS.join(" ");
            Ok(format!(
                "_llm_infer_cal_completion() {{\n\
                 \u{20} local cur\n\
                 \u{20} cur=\"${{COMP_WORDS[COMP_CWORD]}}\"\n\
                 \u{20} COMPREPLY=( $(compgen -W \"{words}\" -- \"$cur\") )\n\
                 }}\n\
                 complete -F _llm_infer_cal_completion llm-infer-cal\n"
            ))
        }
        "fish" => Ok(COMPLETION_OPTIONS
            .iter()
            .filter_map(|option| option.strip_prefix("--"))
            .map(|option| format!("complete -c llm-infer-cal -l {option}\n"))
            .collect()),
        "powershell" | "pwsh" => {
            let words = COMPLETION_OPTIONS
                .iter()
                .map(|option| format!("'{option}'"))
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!(
                "Register-ArgumentCompleter -Native -CommandName llm-infer-cal -ScriptBlock {{\n\
                 \u{20} param($wordToComplete)\n\
                 \u{20} @({words}) | Where-Object {{ $_ -like \"$wordToComplete*\" }}\n\
                 }}\n"
            ))
        }
        _ => Err(format!(
            "Unsupported shell '{shell}'. Use bash, zsh, fish, or powershell."
        )),
    }
}

fn completion_install_text(shell: &str, script: &str) -> String {
    format!(
        "Completion install instructions for {shell}:\n\n{script}\n\
         Add the script above to your shell completion setup, then restart the terminal.\n"
    )
}
