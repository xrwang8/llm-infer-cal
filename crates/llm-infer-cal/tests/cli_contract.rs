use std::ffi::OsString;
use std::sync::{Mutex, MutexGuard, OnceLock};

use llm_infer_cal::{run_with_args, CliExit};

const USAGE_OPTIONS: &[&str] = &[
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
    "--decode-bw-util",
    "--concurrency-degradation",
    "--kv-cache-bits",
    "--paged-attention",
    "--target-concurrency",
    "--speculative-draft-model",
    "--speculative-extra-weight-gb",
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

fn cli_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

fn run_cli<const N: usize>(args: [&str; N]) -> CliExit {
    let _guard = cli_lock();
    run_with_args(args.map(OsString::from))
}

#[test]
fn list_gpus_zh_short_circuits_without_model_or_gpu() {
    let exit = run_cli(["llm-infer-cal", "--lang", "zh", "--list-gpus"]);

    assert_eq!(exit.code, 0);
    assert!(exit.stderr.is_empty());
    assert!(exit.stdout.contains("支持的 GPU"));
    assert!(exit.stdout.contains("H800"));
    assert!(exit.stdout.contains("共 "));
}

#[test]
fn missing_model_id_returns_localized_error() {
    let exit = run_cli(["llm-infer-cal", "--lang", "zh", "--gpu", "H800"]);

    assert_eq!(exit.code, 1);
    assert!(exit.stdout.is_empty());
    assert!(exit
        .stderr
        .contains("缺少参数 MODEL_ID。使用 --help 查看用法。"));
    assert!(!exit.stderr.contains("Missing argument"));
}

#[test]
fn missing_gpu_returns_localized_error() {
    let exit = run_cli(["llm-infer-cal", "--lang", "zh", "deepseek-ai/DeepSeek-V3"]);

    assert_eq!(exit.code, 1);
    assert!(exit.stdout.is_empty());
    assert!(exit
        .stderr
        .contains("缺少选项 --gpu。使用 --list-gpus 查看可选 GPU。"));
    assert!(!exit.stderr.contains("Missing option"));
}

#[test]
fn unknown_source_returns_localized_error() {
    let exit = run_cli([
        "llm-infer-cal",
        "--lang",
        "zh",
        "deepseek-ai/DeepSeek-V3",
        "--gpu",
        "H800",
        "--source",
        "mirror",
    ]);

    assert_eq!(exit.code, 1);
    assert!(exit.stdout.is_empty());
    assert!(exit
        .stderr
        .contains("未知 --source 'mirror'。请使用 'builtin'、'huggingface' 或 'modelscope'。"));
    assert!(!exit.stderr.contains("Unknown --source"));
}

#[test]
fn help_exposes_cli_flags() {
    let exit = run_cli(["llm-infer-cal", "--help"]);

    assert_eq!(exit.code, 0);
    assert!(exit.stderr.is_empty());
    assert!(exit.stdout.contains("Usage: llm-infer-cal"));
    assert!(!exit.stdout.contains("Usage: llm-cal"));
    for option in USAGE_OPTIONS {
        assert!(exit.stdout.contains(option), "missing {option} in help");
    }
}

#[test]
fn all_runtime_flags_parse_before_source_validation() {
    let exit = run_cli([
        "llm-infer-cal",
        "--lang",
        "zh",
        "deepseek-ai/DeepSeek-V3",
        "--gpu",
        "H800",
        "--engine",
        "sglang",
        "--gpu-count",
        "2",
        "--context-length",
        "4096",
        "--refresh",
        "--timeout-s",
        "90",
        "--input-tokens",
        "123",
        "--output-tokens",
        "45",
        "--target-tokens-per-sec",
        "17.5",
        "--prefill-util",
        "0.33",
        "--decode-bw-util",
        "0.44",
        "--concurrency-degradation",
        "1.67",
        "--kv-cache-bits",
        "8",
        "--paged-attention",
        "--target-concurrency",
        "3",
        "--speculative-extra-weight-gb",
        "0.5",
        "--cpu-offload-gb",
        "1",
        "--explain",
        "--llm-review",
        "--source",
        "mirror",
    ]);

    assert_eq!(exit.code, 1);
    assert!(exit.stdout.is_empty());
    assert!(exit.stderr.contains("未知 --source 'mirror'"));
}

#[test]
fn builtin_qwen36_model_renders_zh_report_without_network() {
    let exit = run_cli([
        "llm-infer-cal",
        "--lang",
        "zh",
        "Qwen/Qwen3.6-35B-A3B",
        "--gpu",
        "H100",
        "--source",
        "builtin",
    ]);

    assert_eq!(exit.code, 0, "stderr: {}", exit.stderr);
    assert!(exit.stderr.is_empty());
    assert!(exit.stdout.contains("Qwen/Qwen3.6-35B-A3B  来源 builtin"));
    assert!(exit.stdout.contains("模型类型: qwen3_5_moe_text"));
    assert!(exit.stdout.contains("MoE: 256 个 routed"));
    assert!(exit.stdout.contains("最大上下文长度: 262,144"));
    assert!(exit.stdout.contains("safetensors 总字节: 71.90 GB"));
    assert!(exit.stdout.contains("量化方案推断: BF16 [已验证]"));
    assert!(exit.stdout.contains("生成的启动命令"));
    assert!(exit.stdout.contains("--max-model-len 262144"));
    assert!(exit.stdout.contains("--max-num-seqs 25"));
    assert!(exit.stdout.contains("--trust-remote-code"));
    assert!(exit.stdout.contains("--enable-auto-tool-choice"));
    assert!(exit.stdout.contains("--tool-call-parser qwen3_xml"));
    assert!(exit.stdout.contains("--reasoning-parser qwen3"));
    assert!(exit.stdout.contains("--mm-encoder-tp-mode data"));
    assert!(!exit.stdout.contains("Source unavailable"));
    assert!(!exit.stderr.contains("数据源不可用"));
}

#[test]
fn builtin_qwen36_sglang_command_includes_recipe_flags() {
    let exit = run_cli([
        "llm-infer-cal",
        "--lang",
        "zh",
        "Qwen/Qwen3.6-35B-A3B",
        "--gpu",
        "H100",
        "--engine",
        "sglang",
        "--source",
        "builtin",
    ]);

    assert_eq!(exit.code, 0, "stderr: {}", exit.stderr);
    assert!(exit.stderr.is_empty());
    assert!(exit
        .stdout
        .contains("SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server"));
    assert!(exit.stdout.contains("--reasoning-parser qwen3"));
    assert!(exit.stdout.contains("--tool-call-parser qwen3_coder"));
    assert!(exit.stdout.contains("--context-length 262144"));
    assert!(exit.stdout.contains("--max-running-requests 25"));
    assert!(exit.stdout.contains("--speculative-algorithm EAGLE"));
    assert!(exit.stdout.contains("--speculative-num-steps 3"));
    assert!(exit.stdout.contains("--speculative-eagle-topk 1"));
    assert!(exit.stdout.contains("--speculative-num-draft-tokens 4"));
    assert!(exit
        .stdout
        .contains("--mamba-scheduler-strategy extra_buffer"));
    assert!(exit.stdout.contains("--mem-fraction-static 0.8"));
    assert!(!exit.stdout.contains("--mem-fraction-static 0.9"));
}

#[test]
fn builtin_qwen36_can_render_machine_readable_json() {
    let exit = run_cli([
        "llm-infer-cal",
        "--lang",
        "zh",
        "Qwen/Qwen3.6-35B-A3B",
        "--gpu",
        "H100",
        "--source",
        "builtin",
        "--format",
        "json",
    ]);

    assert_eq!(exit.code, 0, "stderr: {}", exit.stderr);
    assert!(exit.stderr.is_empty());
    assert!(exit.stdout.trim_start().starts_with('{'));

    let json: serde_json::Value =
        serde_json::from_str(&exit.stdout).expect("stdout should be valid JSON");
    assert_eq!(json["schema_version"], "llm-infer-cal.report/v1");
    assert_eq!(json["language"], "zh");
    assert_eq!(json["model"]["id"], "Qwen/Qwen3.6-35B-A3B");
    assert_eq!(json["model"]["source"], "builtin");
    assert_eq!(
        json["architecture"]["model_type"]["value"],
        "qwen3_5_moe_text"
    );
    assert_eq!(json["architecture"]["moe"]["routed_experts"], 256);
    assert_eq!(
        json["weights"]["safetensors_total_bytes"]["value"],
        71_903_776_776_u64
    );
    assert_eq!(json["weights"]["quantization_guess"]["value"], "BF16");
    assert_eq!(json["hardware"]["id"], "H100");
    assert_eq!(json["fleet"]["best_tier"], "dev");
    assert_eq!(
        json["fleet"]["options"][1]["kv_reference_context_tokens"],
        262_144
    );
    assert_eq!(json["performance"]["max_concurrent"]["value"], 25);
    assert!(json["generated_command"]["command"]
        .as_str()
        .unwrap()
        .contains("--reasoning-parser qwen3"));
    assert!(json["generated_command"]["command"]
        .as_str()
        .unwrap()
        .contains("--max-num-seqs 25"));
    assert_eq!(
        json["generated_command"]["lines"][0],
        "vllm serve Qwen/Qwen3.6-35B-A3B"
    );
    assert!(json["generated_command"]["lines"]
        .as_array()
        .unwrap()
        .iter()
        .any(|line| line == "--tool-call-parser qwen3_xml"));
}

#[test]
fn builtin_qwen36_json_includes_inference_optimization_options() {
    let exit = run_cli([
        "llm-infer-cal",
        "--lang",
        "zh",
        "Qwen/Qwen3.6-35B-A3B",
        "--gpu",
        "H100",
        "--source",
        "builtin",
        "--context-length",
        "4096",
        "--kv-cache-bits",
        "8",
        "--paged-attention",
        "--target-concurrency",
        "3",
        "--speculative-extra-weight-gb",
        "0.5",
        "--cpu-offload-gb",
        "1",
        "--json",
    ]);

    assert_eq!(exit.code, 0, "stderr: {}", exit.stderr);
    assert!(exit.stderr.is_empty());
    let json: serde_json::Value =
        serde_json::from_str(&exit.stdout).expect("stdout should be valid JSON");
    assert_eq!(json["inference_options"]["kv_cache_bits"], 8);
    assert_eq!(json["inference_options"]["paged_attention"], true);
    assert_eq!(json["inference_options"]["target_concurrent_requests"], 3);
    assert_eq!(
        json["inference_options"]["cpu_offload_bytes_per_gpu"],
        1_073_741_824
    );
    assert_eq!(
        json["inference_options"]["speculative_extra_weight_bytes"]["value"],
        536_870_912
    );
    assert_eq!(json["fleet"]["best_tier"], "target");
    assert_eq!(json["fleet"]["options"][0]["tier_concurrent_requests"], 3);
    assert!(json["kv_cache_by_context"][0]["bytes"]["source"]
        .as_str()
        .unwrap()
        .contains("paged attention"));
    assert!(
        json["activation_by_context"][0]["bytes"]["value"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[test]
fn builtin_glm52_model_renders_zh_report_and_json_without_network() {
    let text = run_cli([
        "llm-infer-cal",
        "--lang",
        "zh",
        "ZhipuAI/GLM-5.2",
        "--gpu",
        "H100",
        "--source",
        "builtin",
    ]);

    assert_eq!(text.code, 0, "stderr: {}", text.stderr);
    assert!(text.stderr.is_empty());
    assert!(text.stdout.contains("ZhipuAI/GLM-5.2  来源 builtin"));
    assert!(text.stdout.contains("模型类型: glm_moe_dsa"));
    assert!(text.stdout.contains("safetensors 总字节: 1506.67 GB"));
    assert!(text.stdout.contains("最小 *: 48 GPUs"));
    assert!(text.stdout.contains("并发 @ 1.0M ~2"));
    assert!(!text.stdout.contains("数据源不可用"));

    let json_exit = run_cli([
        "llm-infer-cal",
        "--lang",
        "zh",
        "ZhipuAI/GLM-5.2",
        "--gpu",
        "H100",
        "--source",
        "builtin",
        "--json",
    ]);

    assert_eq!(json_exit.code, 0, "stderr: {}", json_exit.stderr);
    assert!(json_exit.stderr.is_empty());
    let json: serde_json::Value =
        serde_json::from_str(&json_exit.stdout).expect("stdout should be valid JSON");
    assert_eq!(json["model"]["id"], "ZhipuAI/GLM-5.2");
    assert_eq!(json["model"]["source"], "builtin");
    assert_eq!(json["architecture"]["model_type"]["value"], "glm_moe_dsa");
    assert_eq!(
        json["weights"]["safetensors_total_bytes"]["value"],
        1_506_667_387_408_u64
    );
    assert_eq!(json["fleet"]["best_tier"], "min");
    assert_eq!(
        json["fleet"]["options"][0]["kv_reference_context_tokens"],
        1_048_576
    );
    assert_eq!(json["generated_command"]["gpu_count"], 48);
    assert_eq!(json["generated_command"]["tensor_parallel_size"], 8);
    assert_eq!(json["generated_command"]["pipeline_parallel_size"], 6);
}

#[test]
fn json_alias_matches_format_json() {
    let exit = run_cli([
        "llm-infer-cal",
        "--lang",
        "zh",
        "Qwen/Qwen3.6-35B-A3B",
        "--gpu",
        "H100",
        "--source",
        "builtin",
        "--json",
    ]);

    assert_eq!(exit.code, 0, "stderr: {}", exit.stderr);
    assert!(exit.stderr.is_empty());
    let json: serde_json::Value =
        serde_json::from_str(&exit.stdout).expect("stdout should be valid JSON");
    assert_eq!(json["schema_version"], "llm-infer-cal.report/v1");
}

#[test]
fn huggingface_endpoint_env_is_used_by_rust_cli() {
    let _guard = cli_lock();
    std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:9");

    let exit = run_with_args([
        "llm-infer-cal",
        "--lang",
        "zh",
        "owner/repo",
        "--gpu",
        "H800",
        "--timeout-s",
        "0.001",
    ]);

    std::env::remove_var("HF_ENDPOINT");

    assert_eq!(exit.code, 4);
    assert!(exit.stdout.is_empty());
    assert!(exit.stderr.contains("127.0.0.1:9"));
    assert!(!exit.stderr.contains("huggingface.co"));
}

#[test]
fn completion_commands_are_stable_without_shell_detection() {
    let show = run_cli(["llm-infer-cal", "--show-completion", "zsh"]);
    let install = run_cli(["llm-infer-cal", "--install-completion", "zsh"]);

    assert_eq!(show.code, 0);
    assert!(show.stderr.is_empty());
    assert!(show.stdout.contains("#compdef llm-infer-cal"));
    assert!(show.stdout.contains("--gpu-count"));
    assert!(!show.stdout.contains("llm-cal"));

    assert_eq!(install.code, 0);
    assert!(install.stderr.is_empty());
    assert!(install
        .stdout
        .contains("Completion install instructions for zsh"));
    assert!(install.stdout.contains("#compdef llm-infer-cal"));
    assert!(!install.stdout.contains("llm-cal"));
}
