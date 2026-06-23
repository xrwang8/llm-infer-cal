from __future__ import annotations

import os
import subprocess
import sys
import tomllib

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))

USAGE_OPTIONS = (
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
)


def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": "src"}
    return subprocess.run(
        [sys.executable, "-m", "llm_cal.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_rust(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cargo", "run", "-q", "-p", "llm-infer-cal", "--", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_zh_list_gpus_output_matches_rust_byte_for_byte():
    py = _run_python("--lang", "zh", "--list-gpus")
    rs = _run_rust("--lang", "zh", "--list-gpus")

    assert py.returncode == rs.returncode == 0
    assert py.stderr == rs.stderr == ""
    assert py.stdout == rs.stdout
    assert "Supported GPUs" not in py.stdout
    assert "yes" not in py.stdout
    assert "no" not in py.stdout


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("--help",),
        ("--list-gpus",),
        ("--gpu", "H800"),
        ("deepseek-ai/DeepSeek-V3",),
        ("deepseek-ai/DeepSeek-V3", "--gpu", "H800", "--source", "mirror"),
        (
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
        ),
        ("--lang", "zh", "--list-gpus"),
        ("--lang", "zh", "deepseek-ai/DeepSeek-V3"),
        (
            "--lang",
            "zh",
            "deepseek-ai/DeepSeek-V3",
            "--gpu",
            "H800",
            "--source",
            "mirror",
        ),
    ],
)
def test_offline_cli_paths_match_rust_byte_for_byte(args: tuple[str, ...]):
    py = _run_python(*args)
    rs = _run_rust(*args)

    assert py.returncode == rs.returncode
    assert py.stdout == rs.stdout
    assert py.stderr == rs.stderr


def test_help_lists_every_usage_option_and_matches_rust():
    py = _run_python("--help")
    rs = _run_rust("--help")

    assert py.returncode == rs.returncode == 0
    assert py.stderr == rs.stderr == ""
    assert py.stdout == rs.stdout
    assert "Usage: llm-infer-cal" in py.stdout
    assert "Usage: llm-cal" not in py.stdout
    for option in USAGE_OPTIONS:
        assert option in py.stdout


@pytest.mark.parametrize("args", [("--show-completion", "zsh"), ("--install-completion", "zsh")])
def test_completion_commands_match_rust_byte_for_byte(args: tuple[str, ...]):
    py = _run_python(*args)
    rs = _run_rust(*args)

    assert py.returncode == rs.returncode == 0
    assert py.stderr == rs.stderr == ""
    assert py.stdout == rs.stdout
    assert "llm-infer-cal" in py.stdout
    assert "llm-cal" not in py.stdout


def test_zh_missing_model_error_matches_rust_and_is_chinese():
    py = _run_python("--lang", "zh", "--gpu", "H800")
    rs = _run_rust("--lang", "zh", "--gpu", "H800")

    assert py.returncode == rs.returncode == 1
    assert py.stdout == rs.stdout == ""
    assert py.stderr == rs.stderr
    assert "缺少参数 MODEL_ID" in py.stderr
    assert "Missing argument" not in py.stderr


def test_python_command_metadata_uses_new_name():
    from typer.testing import CliRunner

    from llm_cal.cli import app

    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        pyproject = tomllib.load(f)

    scripts = pyproject["project"]["scripts"]
    assert "llm-infer-cal" in scripts
    assert "llm-cal" not in scripts

    result = CliRunner().invoke(app, ["--help"], prog_name="llm-infer-cal")
    assert result.exit_code == 0
    assert "llm-infer-cal" in result.stdout
    assert "llm-cal" not in result.stdout
