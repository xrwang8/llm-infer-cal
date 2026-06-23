"""Plain-text, fully i18n'd output for EvaluationReport.

Every visible string flows through `common.i18n.t()`. To add another locale,
add entries to `_MESSAGES` in i18n.py; no changes here needed.
"""

from __future__ import annotations

from typing import Any

from llm_cal.common.i18n import get_locale, t
from llm_cal.core.evaluator import EvaluationReport
from llm_cal.engine_compat.loader import EngineCompatEntry, EngineFlag, EngineSource
from llm_cal.fleet.planner import FleetRecommendation
from llm_cal.hardware.loader import GPUDatabase
from llm_cal.output.labels import AnnotatedValue, Label


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.2f} KB"
    return f"{n} B"


def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return str(n)


def render_gpu_list_text(db: GPUDatabase) -> str:
    yes = t("hw.bool_yes")
    no = t("hw.bool_no")
    out = [
        t("gpus.list.title"),
        " | ".join(
            [
                t("gpus.col.id"),
                t("gpus.col.memory"),
                t("gpus.col.nvlink"),
                t("gpus.col.fp16"),
                t("gpus.col.fp8"),
                t("gpus.col.fp4"),
                t("gpus.col.aliases"),
            ]
        ),
    ]
    for spec in db.gpus:
        aliases = ", ".join(spec.aliases) if spec.aliases else "-"
        nvlink = (
            f"{spec.nvlink_bandwidth_gbps} GB/s" if spec.nvlink_bandwidth_gbps else "-"
        )
        out.append(
            " | ".join(
                [
                    spec.id,
                    f"{spec.memory_gb} GB",
                    nvlink,
                    f"{spec.fp16_tflops:.0f}",
                    yes if spec.fp8_support else no,
                    yes if spec.fp4_support else no,
                    aliases,
                ]
            )
        )
    out.append(t("gpus.total", count=len(db.gpus)))
    return "\n".join(out)


def render_report_text(report: EvaluationReport) -> str:
    sha_frag = f" @ {report.commit_sha[:7]}" if report.commit_sha else ""
    out = [f"{report.model_id}  {t('panel.via')} {report.source}{sha_frag}"]
    _render_architecture_text(report, out)
    _render_weight_text(report, out)
    _render_kv_cache_text(report, out)
    _render_engine_compat_text(report, out)
    _render_hardware_text(report, out)
    _render_fleet_text(report, out)
    _render_performance_text(report, out)
    _render_command_text(report, out)
    _render_label_legend_text(out)
    return "\n".join(out)


def render_explain_text(entries: list[Any]) -> str:
    out = [t("section.explain"), t("explain.intro"), ""]
    for entry in entries:
        out.append(entry.heading)
        out.append(f"{t('explain.formula')}:")
        out.extend(f"  {line}" for line in entry.formula.splitlines())
        if entry.inputs:
            out.append(f"{t('explain.inputs')}:")
            for inp in entry.inputs:
                note = f" ({inp.note})" if inp.note else ""
                out.append(f"  {inp.name} = {inp.value} {inp.label}{note}")
        if entry.steps:
            out.append(f"{t('explain.steps')}:")
            for step in entry.steps:
                out.extend(f"  {line}" for line in step.splitlines())
        out.append(f"{t('explain.result')}: {entry.result}")
        if entry.source:
            out.append(f"{t('explain.source')}: {entry.source}")
        if entry.methodology_anchor:
            out.append(f"{t('explain.see_also')}: docs/methodology.md{entry.methodology_anchor}")
        out.append("")
    return "\n".join(out)


def render_llm_review_text(result: Any) -> str:
    out = [t("section.llm_review")]
    if not result.ok:
        out.append(t("llm_review.unavailable", error=result.error or "unknown"))
        out.append(t("llm_review.setup_hint"))
        return "\n".join(out)

    out.append(t("llm_review.disclaimer", model=result.model, base_url=result.base_url))
    out.append(f"[{t('label.llm-opinion')}]")
    out.append(result.content or "")
    return "\n".join(out)


def _render_architecture_text(report: EvaluationReport, out: list[str]) -> None:
    p = report.profile
    out.append("")
    out.append(t("section.architecture"))
    out.append(_row(t("arch.model_type"), p.model_type or t("arch.none"), _verified_tag_text()))
    out.append(_row(t("arch.family"), p.family.value, _verified_tag_text()))
    out.append(_row(t("arch.confidence"), p.confidence.value, f"[{p.confidence.value}]"))
    out.append(_row(t("arch.layers"), str(p.num_hidden_layers), _verified_tag_text()))
    out.append(_row(t("arch.hidden_size"), str(p.hidden_size), _verified_tag_text()))
    out.append(_row(t("arch.vocab_size"), _fmt_u64(p.vocab_size), _verified_tag_text()))

    if p.attention is not None:
        out.append(
            _row(
                t("arch.attention"),
                t(
                    "arch.attn_summary",
                    variant=p.attention.variant,
                    heads=p.attention.num_heads,
                    kv_heads=p.attention.num_kv_heads,
                    head_dim=p.attention.head_dim,
                ),
                _verified_tag_text(),
            )
        )
        if p.attention.compress_ratios:
            ratios = p.attention.compress_ratios
            out.append(
                _row(
                    t("arch.compress_ratios"),
                    t(
                        "arch.compress_ratios_summary",
                        n=len(ratios),
                        dense=sum(1 for ratio in ratios if ratio == 0),
                    ),
                    _verified_tag_text(),
                )
            )
    if p.moe is not None:
        out.append(
            _row(
                t("arch.moe"),
                t(
                    "arch.moe_summary",
                    routed=p.moe.num_routed_experts,
                    shared=p.moe.num_shared_experts,
                    topk=p.moe.num_experts_per_tok,
                ),
                _verified_tag_text(),
            )
        )
    if p.sliding_window:
        out.append(_row(t("arch.sliding_window"), str(p.sliding_window), _verified_tag_text()))
    if p.position and p.position.max_position_embeddings:
        out.append(
            _row(
                t("arch.max_position"),
                _fmt_u64(p.position.max_position_embeddings),
                _verified_tag_text(),
            )
        )
    if p.auxiliary.get("warning"):
        out.append(f"WARNING: {p.auxiliary['warning']}")
    if p.auxiliary.get("v0_1_unsupported"):
        out.append(f"WARNING: {t('arch.unsupported_state_space')}")


def _render_weight_text(report: EvaluationReport, out: list[str]) -> None:
    out.append("")
    out.append(t("section.weights"))
    w = report.weight
    out.append(_row(t("weights.safetensors_bytes"), _fmt_bytes(w.total_bytes.value), _format_tag_text(w.total_bytes)))
    out.append(
        _row(
            t("weights.params_estimated"),
            _fmt_params(report.total_params_estimate.value),
            _format_tag_text(report.total_params_estimate),
        )
    )
    if w.bits_per_param is not None:
        out.append(
            _row(
                t("weights.bits_per_param"),
                f"{w.bits_per_param.value:.2f}",
                _format_tag_text(w.bits_per_param),
            )
        )
    out.append(
        _row(
            t("weights.quant_guess"),
            str(w.quantization_guess.value),
            _format_tag_text(w.quantization_guess),
        )
    )

    r = report.reconciliation
    if r.candidates:
        out.append(t("section.reconciliation"))
        for candidate in r.candidates[:6]:
            direction = t("recon.over") if candidate.delta_bytes > 0 else t("recon.under")
            out.append(
                f"{candidate.scheme}: {_fmt_bytes(candidate.predicted_bytes)} | "
                f"{_fmt_bytes(abs(candidate.delta_bytes))} {direction} | "
                f"{candidate.relative_error * 100:.1f}%"
            )
        out.append(f"{t('recon.best')} {r.best.value} {_format_tag_text(r.best)}")


def _render_kv_cache_text(report: EvaluationReport, out: list[str]) -> None:
    if not report.kv_cache_by_context:
        return
    out.append("")
    out.append(t("section.kv_cache"))
    tokens = t("kv.tokens")
    for ctx, av in report.kv_cache_by_context.items():
        out.append(f"{_fmt_u64(ctx)} {tokens}: {_fmt_bytes(av.value)} {_format_tag_text(av)}")


def _render_engine_compat_text(report: EvaluationReport, out: list[str]) -> None:
    out.append("")
    m = report.engine_match
    if m is None:
        out.append(f"{t('section.engine_compat')}: {t('engine.no_match')}")
        return
    out.append(f"{t('section.engine_compat')} - {m.engine}")
    out.append(_row(t("engine.version_spec"), m.version_spec, ""))
    out.append(_row(t("engine.support"), m.support, _verif_label_text(m)))
    out.append(_row(t("engine.verification"), m.verification_level, _verif_label_text(m)))
    if m.required_flags:
        out.append(_row(t("engine.required_flags"), ", ".join(_fmt_flag(f) for f in m.required_flags), ""))
    if m.optional_flags:
        out.append(_row(t("engine.optional_flags"), ", ".join(_fmt_flag(f) for f in m.optional_flags), ""))
    caveats = m.caveats_zh if get_locale() == "zh" else m.caveats_en
    if caveats:
        out.append(_row(t("engine.caveats"), "; ".join(caveats), ""))
    if m.sources:
        out.append(_row(t("engine.sources"), "; ".join(_fmt_source(s) for s in m.sources), ""))


def _render_hardware_text(report: EvaluationReport, out: list[str]) -> None:
    out.append("")
    if report.gpu_spec is None:
        msg = report.gpu_error or f"Unknown GPU '{report.gpu}'"
        out.append(f"{t('section.hardware')}: {msg}")
        return

    spec = report.gpu_spec
    out.append(f"{t('section.hardware')} - {spec.id}")
    out.append(_row(t("hw.memory"), f"{spec.memory_gb} GB HBM", ""))
    out.append(_row(t("hw.nvlink_bandwidth"), f"{spec.nvlink_bandwidth_gbps} GB/s", ""))
    out.append(_row(t("hw.fp16_tflops"), f"{spec.fp16_tflops:.0f} TFLOPS", ""))
    out.append(_row(t("hw.fp8_support"), t("hw.bool_yes") if spec.fp8_support else t("hw.bool_no"), ""))
    out.append(_row(t("hw.fp4_support"), t("hw.bool_yes") if spec.fp4_support else t("hw.bool_no"), ""))
    notes = spec.localized_notes(get_locale())
    if notes:
        out.append(_row(t("hw.notes"), notes, ""))
    if spec.spec_source:
        out.append(_row(t("hw.spec_source"), spec.spec_source, ""))


def _render_fleet_text(report: EvaluationReport, out: list[str]) -> None:
    f = report.fleet
    if f is None:
        if report.gpu_spec is not None:
            out.append(t("fleet.gpu_spec_unknown"))
        return
    out.append("")
    out.append(f"{t('section.fleet')} - {report.gpu_spec.id if report.gpu_spec else report.gpu}")
    ctx_cols = _select_concurrency_columns(f)
    for opt in f.options:
        headroom = max(0, opt.usable_bytes_per_gpu - opt.weight_bytes_per_gpu)
        marker = " *" if opt.tier == f.best_tier else ""
        line = (
            f"{t(f'fleet.tier.{opt.tier}')}{marker}: {opt.gpu_count} GPUs, "
            f"weight/GPU {_fmt_bytes(opt.weight_bytes_per_gpu)}, "
            f"headroom {_fmt_bytes(headroom) if headroom > 0 else '-'}"
        )
        for ctx in ctx_cols:
            concurrent = next((count for candidate, count in opt.max_concurrent_by_context if candidate == ctx), 0)
            line += (
                f", {t('fleet.col.concurrent_at_ctx', ctx=_fmt_ctx(ctx))} "
                f"{f'~{concurrent}' if concurrent > 0 else 'x'}"
            )
        if not opt.fits:
            line += " (does not fit)"
        out.append(line)
    note = f.constraint_note_zh if get_locale() == "zh" else f.constraint_note_en
    out.append(f"{t('fleet.constraint')} {note}")
    out.append(f"* {t('fleet.best_marker')}")


def _render_performance_text(report: EvaluationReport, out: list[str]) -> None:
    if (
        report.prefill is None
        or report.decode is None
        or report.concurrency is None
        or report.perf_input_tokens is None
        or report.perf_target_tokens_per_sec is None
    ):
        return
    p = report.prefill
    d = report.decode
    c = report.concurrency
    out.append("")
    out.append(t("section.performance"))
    out.append(
        t(
            "perf.assumptions_note",
            input_tokens=report.perf_input_tokens,
            output_tokens=report.perf_output_tokens or 512,
            target_tps=f"{report.perf_target_tokens_per_sec:.1f}",
            prefill_util=f"{p.utilization * 100:.0f}%",
            decode_util=f"{d.bw_utilization * 100:.0f}%",
            degradation=f"{c.degradation_factor:.2f}",
        )
    )
    out.append(_row(t("perf.prefill_latency"), f"{p.latency_ms.value:.1f} ms", _format_tag_text(p.latency_ms)))
    out.append(_row(t("perf.decode_throughput_per_gpu"), f"{d.per_gpu_tokens_per_sec.value:.1f} tok/s", _format_tag_text(d.per_gpu_tokens_per_sec)))
    out.append(_row(t("perf.decode_throughput_cluster"), f"{d.cluster_tokens_per_sec.value:.1f} tok/s", _format_tag_text(d.cluster_tokens_per_sec)))
    if d.moe_active_tokens_per_sec is not None:
        out.append(
            _row(
                t("perf.decode_moe_active_optimistic"),
                f"{d.moe_active_tokens_per_sec.value:.1f} tok/s",
                _format_tag_text(d.moe_active_tokens_per_sec),
            )
        )
    out.append(_row(t("perf.k_bound"), str(c.k_bound.value), _format_tag_text(c.k_bound)))
    out.append(_row(t("perf.l_bound"), str(c.l_bound.value), _format_tag_text(c.l_bound)))
    out.append(_row(t("perf.max_concurrent"), str(c.max_concurrent.value), _format_tag_text(c.max_concurrent)))
    reason = c.bottleneck_reason_zh if get_locale() == "zh" else c.bottleneck_reason_en
    out.append(_row(t("perf.bottleneck"), f"{t(f'perf.bottleneck.{c.bottleneck}')}: {reason}", ""))
    out.append(f"{t('perf.optimization.header')}:")
    for key in (
        "perf.opt.quantize_int4",
        "perf.opt.relax_sla",
        "perf.opt.kv_fp8",
        "perf.opt.moe_offload",
    ):
        out.append(f"  - {t(key)}")


def _render_command_text(report: EvaluationReport, out: list[str]) -> None:
    if not report.generated_command or report.fleet is None:
        return
    best = next(
        (option for option in report.fleet.options if option.tier == report.fleet.best_tier),
        report.fleet.options[0],
    )
    out.append("")
    out.append(
        f"{t('section.command')} - "
        f"{t('command.tier_note', tier=t(f'fleet.tier.{best.tier}'), gpus=best.gpu_count)}"
    )
    out.append(report.generated_command)


def _render_label_legend_text(out: list[str]) -> None:
    out.append("")
    labels = " ".join(f"[{t(f'label.{label.value}')}]" for label in Label)
    out.append(f"{t('section.labels')} {labels}")


def _format_tag_text(av: AnnotatedValue[Any]) -> str:
    return f"[{t(f'label.{av.label.value}')}]"


def _verified_tag_text() -> str:
    return f"[{t('label.verified')}]"


def _row(field: str, value: Any, label: Any) -> str:
    label_s = str(label)
    if not label_s:
        return f"{field}: {value}"
    return f"{field}: {value} {label_s}"


def _verif_label_text(entry: EngineCompatEntry) -> str:
    label = {
        "verified": Label.VERIFIED,
        "cited": Label.CITED,
        "unverified": Label.UNVERIFIED,
    }.get(entry.verification_level, Label.UNKNOWN)
    return f"[{t(f'label.{label.value}')}]"


def _select_concurrency_columns(f: FleetRecommendation) -> list[int]:
    all_ctxs = sorted({ctx for opt in f.options for ctx, _ in opt.max_concurrent_by_context})
    if not all_ctxs:
        return []
    picks: list[int] = []
    if 131_072 in all_ctxs:
        picks.append(131_072)
    max_ctx = max(all_ctxs)
    if max_ctx > 131_072 and max_ctx not in picks:
        picks.append(max_ctx)
    if not picks:
        picks.append(32_768 if 32_768 in all_ctxs else max_ctx)
    return picks


def _fmt_u64(value: int) -> str:
    return f"{value:,}"


def _fmt_ctx(ctx_tokens: int) -> str:
    if ctx_tokens >= 1_000_000:
        if ctx_tokens % 1_000_000 == 0:
            return f"{ctx_tokens // 1_000_000}M"
        return f"{ctx_tokens / 1_000_000:.1f}M"
    if ctx_tokens >= 1024:
        return f"{ctx_tokens // 1024}K"
    return str(ctx_tokens)


def _fmt_flag(f: EngineFlag) -> str:
    if f.value is None:
        return f.flag
    return f"{f.flag} {f.value}"


def _fmt_source(s: EngineSource) -> str:
    label = t(f"source.{s.type}")
    if s.type == "tested":
        return f"[{label}] {s.tester} @ {s.hardware} ({s.date})"
    if s.url:
        captured = f" ({t('source.captured_on')} {s.captured_date})" if s.captured_date else ""
        return f"[{label}] {s.url}{captured}"
    return f"[{label}]"
