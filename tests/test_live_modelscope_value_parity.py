"""Live Python/Rust value parity for the reference ModelScope model.

This test intentionally compares structured EvaluationReport values, not the
human-rendered CLI text. It is opt-in because it hits ModelScope live.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from llm_cal.core.evaluator import EvaluationReport, Evaluator
from llm_cal.model_source.modelscope import ModelScopeSource

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
GPU = "H800"
ENGINE = "vllm"


pytestmark = pytest.mark.skipif(
    os.environ.get("LLM_CAL_LIVE_MODEL_PARITY") != "1",
    reason="set LLM_CAL_LIVE_MODEL_PARITY=1 to run live ModelScope parity",
)


def _label(label: Any) -> str:
    return getattr(label, "value", str(label))


def _av(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "value": getattr(value, "value"),
        "label": _label(getattr(value, "label")),
        "source": getattr(value, "source"),
    }


def _snapshot(report: EvaluationReport) -> dict[str, Any]:
    profile = report.profile
    attention = profile.attention
    moe = profile.moe
    position = profile.position
    engine_match = report.engine_match
    fleet = report.fleet
    gpu_spec = report.gpu_spec
    prefill = report.prefill
    decode = report.decode
    concurrency = report.concurrency

    return {
        "model_id": report.model_id,
        "source": report.source,
        "commit_sha": report.commit_sha,
        "gpu": report.gpu,
        "engine": report.engine,
        "profile": {
            "model_type": profile.model_type,
            "architectures": list(profile.architectures),
            "family": _label(profile.family),
            "num_hidden_layers": profile.num_hidden_layers,
            "hidden_size": profile.hidden_size,
            "vocab_size": profile.vocab_size,
            "confidence": _label(profile.confidence),
            "attention": None
            if attention is None
            else {
                "variant": attention.variant,
                "num_heads": attention.num_heads,
                "num_kv_heads": attention.num_kv_heads,
                "head_dim": attention.head_dim,
                "q_lora_rank": attention.q_lora_rank,
                "kv_lora_rank": attention.kv_lora_rank,
                "compress_ratios": list(attention.compress_ratios or ()),
                "nsa_topk": attention.nsa_topk,
            },
            "moe": None
            if moe is None
            else {
                "num_routed_experts": moe.num_routed_experts,
                "num_shared_experts": moe.num_shared_experts,
                "num_experts_per_tok": moe.num_experts_per_tok,
                "moe_intermediate_size": moe.moe_intermediate_size,
            },
            "position": None
            if position is None
            else {
                "rope_type": position.rope_type,
                "rope_theta": position.rope_theta,
                "rope_scaling_factor": position.rope_scaling_factor,
                "max_position_embeddings": position.max_position_embeddings,
            },
            "sliding_window": profile.sliding_window,
        },
        "weight": {
            "total_bytes": _av(report.weight.total_bytes),
            "bits_per_param": _av(report.weight.bits_per_param),
            "quantization_guess": _av(report.weight.quantization_guess),
        },
        "total_params_estimate": _av(report.total_params_estimate),
        "reconciliation": {
            "observed_bytes": report.reconciliation.observed_bytes,
            "total_params": report.reconciliation.total_params,
            "best": _av(report.reconciliation.best),
            "candidates": [
                {
                    "scheme": c.scheme,
                    "predicted_bytes": c.predicted_bytes,
                    "delta_bytes": c.delta_bytes,
                    "relative_error": c.relative_error,
                }
                for c in report.reconciliation.candidates
            ],
        },
        "kv_cache_by_context": [
            [ctx, _av(value)] for ctx, value in sorted(report.kv_cache_by_context.items())
        ],
        "engine_match": None
        if engine_match is None
        else {
            "engine": engine_match.engine,
            "version_spec": engine_match.version_spec,
            "support": engine_match.support,
            "verification_level": engine_match.verification_level,
            "required_flags": [
                {"flag": f.flag, "value": f.value} for f in engine_match.required_flags
            ],
            "optional_flags": [
                {"flag": f.flag, "value": f.value} for f in engine_match.optional_flags
            ],
            "caveats_zh": list(engine_match.caveats_zh),
        },
        "gpu_spec": None
        if gpu_spec is None
        else {
            "id": gpu_spec.id,
            "memory_gb": gpu_spec.memory_gb,
            "nvlink_bandwidth_gbps": gpu_spec.nvlink_bandwidth_gbps,
            "memory_bandwidth_gbps": gpu_spec.memory_bandwidth_gbps,
            "fp16_tflops": gpu_spec.fp16_tflops,
            "fp8_support": gpu_spec.fp8_support,
            "fp4_support": gpu_spec.fp4_support,
        },
        "fleet": None
        if fleet is None
        else {
            "best_tier": fleet.best_tier,
            "valid_tp_sizes": list(fleet.valid_tp_sizes),
            "constraint_note_zh": fleet.constraint_note_zh,
            "options": [
                {
                    "tier": option.tier,
                    "gpu_count": option.gpu_count,
                    "weight_bytes_per_gpu": option.weight_bytes_per_gpu,
                    "kv_bytes_per_request": option.kv_bytes_per_request,
                    "max_concurrent_at_reference_ctx": option.max_concurrent_at_reference_ctx,
                    "max_concurrent_by_context": [
                        list(item) for item in option.max_concurrent_by_context
                    ],
                    "usable_bytes_per_gpu": option.usable_bytes_per_gpu,
                    "fits": option.fits,
                    "reason_zh": option.reason_zh,
                }
                for option in fleet.options
            ],
        },
        "generated_command": report.generated_command,
        "prefill": None
        if prefill is None
        else {
            "total_flops": _av(prefill.total_flops),
            "peak_effective_tflops": _av(prefill.peak_effective_tflops),
            "latency_ms": _av(prefill.latency_ms),
            "utilization": prefill.utilization,
        },
        "decode": None
        if decode is None
        else {
            "active_weight_bytes_per_gpu": _av(decode.active_weight_bytes_per_gpu),
            "per_gpu_tokens_per_sec": _av(decode.per_gpu_tokens_per_sec),
            "cluster_tokens_per_sec": _av(decode.cluster_tokens_per_sec),
            "bw_utilization": decode.bw_utilization,
            "cluster_comm_efficiency": decode.cluster_comm_efficiency,
            "moe_active_weight_bytes_per_gpu": _av(decode.moe_active_weight_bytes_per_gpu),
            "moe_active_tokens_per_sec": _av(decode.moe_active_tokens_per_sec),
        },
        "concurrency": None
        if concurrency is None
        else {
            "k_bound": _av(concurrency.k_bound),
            "l_bound": _av(concurrency.l_bound),
            "max_concurrent": _av(concurrency.max_concurrent),
            "bottleneck": concurrency.bottleneck,
            "bottleneck_reason_zh": concurrency.bottleneck_reason_zh,
            "target_tokens_per_sec": concurrency.target_tokens_per_sec,
            "degradation_factor": concurrency.degradation_factor,
        },
        "perf_input_tokens": report.perf_input_tokens,
        "perf_output_tokens": report.perf_output_tokens,
        "perf_target_tokens_per_sec": report.perf_target_tokens_per_sec,
    }


def _normalize(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _python_snapshot() -> dict[str, Any]:
    report = Evaluator(source=ModelScopeSource()).evaluate(
        MODEL_ID,
        gpu=GPU,
        engine=ENGINE,
        refresh=True,
    )
    return _normalize(_snapshot(report))


def _rust_snapshot() -> dict[str, Any]:
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "llm-infer-cal-core",
        "--example",
        "dump_report_values",
        "--",
        MODEL_ID,
        "--gpu",
        GPU,
        "--engine",
        ENGINE,
        "--source",
        "modelscope",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return _normalize(json.loads(proc.stdout))


def test_live_modelscope_deepseek_v4_python_and_rust_values_match() -> None:
    assert _rust_snapshot() == _python_snapshot()
