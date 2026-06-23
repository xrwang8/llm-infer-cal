use std::ffi::OsString;
use std::sync::{Mutex, OnceLock};

use llm_infer_cal::{run_with_args, CliExit};

const USAGE_OPTIONS: &[&str] = &[
    "--gpu",
    "--engine",
    "--gpu-count",
    "--context-length",
    "--refresh",
    "--lang",
    "--list-gpus",
    "--benchmark",
    "--input-tokens",
    "--output-tokens",
    "--target-tokens-per-sec",
    "--prefill-util",
    "--decode-bw-util",
    "--concurrency-degradation",
    "--explain",
    "--llm-review",
    "--source",
    "--install-completion",
    "--show-completion",
    "--help",
];

fn run_cli<const N: usize>(args: [&str; N]) -> CliExit {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
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
        .contains("未知 --source 'mirror'。请使用 'huggingface' 或 'modelscope'。"));
    assert!(!exit.stderr.contains("Unknown --source"));
}

#[test]
fn help_exposes_python_cli_flags() {
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
