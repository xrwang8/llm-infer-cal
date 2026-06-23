use std::collections::HashMap;

use crate::common::i18n::{get_locale, t, t_with};
use crate::core::evaluator::EvaluationReport;
use crate::core::explain::ExplainEntry;
use crate::engine_compat::{EngineCompatEntry, EngineFlag, EngineSource};
use crate::fleet::planner::FleetRecommendation;
use crate::hardware::loader::GPUDatabase;
use crate::llm_review::reviewer::LlmReviewResult;
use crate::output::labels::{AnnotatedValue, Label};

pub fn format_tag<T>(value: &AnnotatedValue<T>) -> String {
    format!("[{}]", t(&format!("label.{}", value.label.as_str())))
}

pub fn fmt_bytes(n: u64) -> String {
    if n >= 1_000_000_000 {
        return format!("{:.2} GB", n as f64 / 1_000_000_000.0);
    }
    if n >= 1_000_000 {
        return format!("{:.2} MB", n as f64 / 1_000_000.0);
    }
    if n >= 1_000 {
        return format!("{:.2} KB", n as f64 / 1_000.0);
    }
    format!("{n} B")
}

pub fn fmt_params(n: u64) -> String {
    if n >= 1_000_000_000 {
        return format!("{:.2}B", n as f64 / 1_000_000_000.0);
    }
    if n >= 1_000_000 {
        return format!("{:.2}M", n as f64 / 1_000_000.0);
    }
    n.to_string()
}

pub fn render_report_text(report: &EvaluationReport) -> String {
    let mut out = Vec::new();
    let sha_frag = report
        .commit_sha
        .as_deref()
        .map(|sha| format!(" @ {}", &sha[..sha.len().min(7)]))
        .unwrap_or_default();
    out.push(format!(
        "{}  {} {}{}",
        report.model_id,
        t("panel.via"),
        report.source,
        sha_frag
    ));
    render_architecture(report, &mut out);
    render_weight(report, &mut out);
    render_kv_cache(report, &mut out);
    render_engine_compat(report, &mut out);
    render_hardware(report, &mut out);
    render_fleet(report, &mut out);
    render_performance(report, &mut out);
    render_command(report, &mut out);
    render_label_legend(&mut out);
    out.join("\n")
}

pub fn render_explain_text(entries: &[ExplainEntry]) -> String {
    let mut out = vec![t("section.explain"), t("explain.intro"), String::new()];
    for entry in entries {
        out.push(entry.heading.clone());
        out.push(format!("{}:", t("explain.formula")));
        out.extend(entry.formula.lines().map(|line| format!("  {line}")));
        if !entry.inputs.is_empty() {
            out.push(format!("{}:", t("explain.inputs")));
            for input in &entry.inputs {
                let note = if input.note.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", input.note)
                };
                out.push(format!(
                    "  {} = {} {}{}",
                    input.name, input.value, input.label, note
                ));
            }
        }
        if !entry.steps.is_empty() {
            out.push(format!("{}:", t("explain.steps")));
            for step in &entry.steps {
                out.extend(step.lines().map(|line| format!("  {line}")));
            }
        }
        out.push(format!("{}: {}", t("explain.result"), entry.result));
        if !entry.source.is_empty() {
            out.push(format!("{}: {}", t("explain.source"), entry.source));
        }
        if !entry.methodology_anchor.is_empty() {
            out.push(format!(
                "{}: docs/methodology.md{}",
                t("explain.see_also"),
                entry.methodology_anchor
            ));
        }
        out.push(String::new());
    }
    out.join("\n")
}

pub fn render_llm_review_text(result: &LlmReviewResult) -> String {
    let mut out = vec![t("section.llm_review")];
    if !result.ok {
        out.push(t_with(
            "llm_review.unavailable",
            &HashMap::from([(
                "error",
                result
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
            )]),
        ));
        out.push(t("llm_review.setup_hint"));
        return out.join("\n");
    }

    out.push(t_with(
        "llm_review.disclaimer",
        &HashMap::from([
            ("model", result.model.clone()),
            ("base_url", result.base_url.clone()),
        ]),
    ));
    out.push(format!("[{}]", t("label.llm-opinion")));
    out.push(result.content.clone().unwrap_or_default());
    out.join("\n")
}

pub fn render_gpu_list_text(db: &GPUDatabase) -> String {
    let yes = t("hw.bool_yes");
    let no = t("hw.bool_no");
    let mut out = vec![
        t("gpus.list.title"),
        format!(
            "{} | {} | {} | {} | {} | {} | {}",
            t("gpus.col.id"),
            t("gpus.col.memory"),
            t("gpus.col.nvlink"),
            t("gpus.col.fp16"),
            t("gpus.col.fp8"),
            t("gpus.col.fp4"),
            t("gpus.col.aliases")
        ),
    ];
    for spec in &db.gpus {
        let aliases = if spec.aliases.is_empty() {
            "-".to_string()
        } else {
            spec.aliases.join(", ")
        };
        let nvlink = if spec.nvlink_bandwidth_gbps == 0 {
            "-".to_string()
        } else {
            format!("{} GB/s", spec.nvlink_bandwidth_gbps)
        };
        out.push(format!(
            "{} | {} GB | {} | {:.0} | {} | {} | {}",
            spec.id,
            spec.memory_gb,
            nvlink,
            spec.fp16_tflops,
            if spec.fp8_support { &yes } else { &no },
            if spec.fp4_support { &yes } else { &no },
            aliases
        ));
    }
    out.push(t_with(
        "gpus.total",
        &HashMap::from([("count", db.gpus.len().to_string())]),
    ));
    out.join("\n")
}

fn render_architecture(report: &EvaluationReport, out: &mut Vec<String>) {
    let p = &report.profile;
    out.push(String::new());
    out.push(t("section.architecture"));
    out.push(row(
        &t("arch.model_type"),
        if p.model_type.is_empty() {
            t("arch.none")
        } else {
            p.model_type.clone()
        },
        verified_tag(),
    ));
    out.push(row(&t("arch.family"), p.family.as_str(), verified_tag()));
    out.push(row(
        &t("arch.confidence"),
        p.confidence.as_str(),
        format!("[{}]", p.confidence.as_str()),
    ));
    out.push(row(
        &t("arch.layers"),
        p.num_hidden_layers.to_string(),
        verified_tag(),
    ));
    out.push(row(
        &t("arch.hidden_size"),
        p.hidden_size.to_string(),
        verified_tag(),
    ));
    out.push(row(
        &t("arch.vocab_size"),
        fmt_u64(p.vocab_size),
        verified_tag(),
    ));
    if let Some(attention) = &p.attention {
        out.push(row(
            &t("arch.attention"),
            t_with(
                "arch.attn_summary",
                &HashMap::from([
                    ("variant", attention.variant.as_str().to_string()),
                    ("heads", attention.num_heads.to_string()),
                    ("kv_heads", attention.num_kv_heads.to_string()),
                    ("head_dim", attention.head_dim.to_string()),
                ]),
            ),
            verified_tag(),
        ));
        if let Some(ratios) = &attention.compress_ratios {
            out.push(row(
                &t("arch.compress_ratios"),
                t_with(
                    "arch.compress_ratios_summary",
                    &HashMap::from([
                        ("n", ratios.len().to_string()),
                        (
                            "dense",
                            ratios
                                .iter()
                                .filter(|ratio| **ratio == 0)
                                .count()
                                .to_string(),
                        ),
                    ]),
                ),
                verified_tag(),
            ));
        }
    }
    if let Some(moe) = &p.moe {
        out.push(row(
            &t("arch.moe"),
            t_with(
                "arch.moe_summary",
                &HashMap::from([
                    ("routed", moe.num_routed_experts.to_string()),
                    ("shared", moe.num_shared_experts.to_string()),
                    ("topk", moe.num_experts_per_tok.to_string()),
                ]),
            ),
            verified_tag(),
        ));
    }
    if let Some(sliding_window) = p.sliding_window {
        out.push(row(
            &t("arch.sliding_window"),
            sliding_window.to_string(),
            verified_tag(),
        ));
    }
    if let Some(position) = &p.position {
        if let Some(max_pos) = position.max_position_embeddings {
            out.push(row(
                &t("arch.max_position"),
                fmt_u64(max_pos),
                verified_tag(),
            ));
        }
    }
    if let Some(warning) = p.auxiliary.get("warning") {
        out.push(format!(
            "WARNING: {}",
            warning
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| warning.to_string())
        ));
    }
    if p.auxiliary.contains_key("v0_1_unsupported") {
        out.push(format!("WARNING: {}", t("arch.unsupported_state_space")));
    }
}

fn render_weight(report: &EvaluationReport, out: &mut Vec<String>) {
    out.push(String::new());
    out.push(t("section.weights"));
    let w = &report.weight;
    out.push(row(
        &t("weights.safetensors_bytes"),
        fmt_bytes(w.total_bytes.value),
        format_tag(&w.total_bytes),
    ));
    out.push(row(
        &t("weights.params_estimated"),
        fmt_params(report.total_params_estimate.value),
        format_tag(&report.total_params_estimate),
    ));
    if let Some(bits) = &w.bits_per_param {
        out.push(row(
            &t("weights.bits_per_param"),
            format!("{:.2}", bits.value),
            format_tag(bits),
        ));
    }
    out.push(row(
        &t("weights.quant_guess"),
        w.quantization_guess.value.to_string(),
        format_tag(&w.quantization_guess),
    ));

    let r = &report.reconciliation;
    if !r.candidates.is_empty() {
        out.push(t("section.reconciliation"));
        for candidate in r.candidates.iter().take(6) {
            let direction = if candidate.delta_bytes > 0 {
                t("recon.over")
            } else {
                t("recon.under")
            };
            out.push(format!(
                "{}: {} | {} {} | {:.1}%",
                candidate.scheme,
                fmt_bytes(candidate.predicted_bytes),
                fmt_bytes(candidate.delta_bytes.unsigned_abs() as u64),
                direction,
                candidate.relative_error * 100.0
            ));
        }
        out.push(format!(
            "{} {} {}",
            t("recon.best"),
            r.best.value,
            format_tag(&r.best)
        ));
    }
}

fn render_kv_cache(report: &EvaluationReport, out: &mut Vec<String>) {
    if report.kv_cache_by_context.is_empty() {
        return;
    }
    out.push(String::new());
    out.push(t("section.kv_cache"));
    let tokens = t("kv.tokens");
    for (ctx, av) in &report.kv_cache_by_context {
        out.push(format!(
            "{} {}: {} {}",
            fmt_u64(*ctx),
            tokens,
            fmt_bytes(av.value),
            format_tag(av)
        ));
    }
}

fn render_engine_compat(report: &EvaluationReport, out: &mut Vec<String>) {
    out.push(String::new());
    let Some(entry) = &report.engine_match else {
        out.push(format!(
            "{}: {}",
            t("section.engine_compat"),
            t("engine.no_match")
        ));
        return;
    };
    out.push(format!("{} - {}", t("section.engine_compat"), entry.engine));
    out.push(row(&t("engine.version_spec"), &entry.version_spec, ""));
    out.push(row(
        &t("engine.support"),
        &entry.support,
        verif_label(entry),
    ));
    out.push(row(
        &t("engine.verification"),
        &entry.verification_level,
        verif_label(entry),
    ));
    if !entry.required_flags.is_empty() {
        out.push(row(
            &t("engine.required_flags"),
            entry
                .required_flags
                .iter()
                .map(fmt_flag)
                .collect::<Vec<_>>()
                .join(", "),
            "",
        ));
    }
    if !entry.optional_flags.is_empty() {
        out.push(row(
            &t("engine.optional_flags"),
            entry
                .optional_flags
                .iter()
                .map(fmt_flag)
                .collect::<Vec<_>>()
                .join(", "),
            "",
        ));
    }
    let caveats = if get_locale() == "zh" {
        &entry.caveats_zh
    } else {
        &entry.caveats_en
    };
    if !caveats.is_empty() {
        out.push(row(&t("engine.caveats"), caveats.join("; "), ""));
    }
    if !entry.sources.is_empty() {
        out.push(row(
            &t("engine.sources"),
            entry
                .sources
                .iter()
                .map(fmt_source)
                .collect::<Vec<_>>()
                .join("; "),
            "",
        ));
    }
}

fn render_hardware(report: &EvaluationReport, out: &mut Vec<String>) {
    out.push(String::new());
    let Some(spec) = &report.gpu_spec else {
        let msg = report
            .gpu_error
            .clone()
            .unwrap_or_else(|| format!("Unknown GPU '{}'", report.gpu));
        out.push(format!("{}: {}", t("section.hardware"), msg));
        return;
    };
    out.push(format!("{} - {}", t("section.hardware"), spec.id));
    out.push(row(
        &t("hw.memory"),
        format!("{} GB HBM", spec.memory_gb),
        "",
    ));
    out.push(row(
        &t("hw.nvlink_bandwidth"),
        format!("{} GB/s", spec.nvlink_bandwidth_gbps),
        "",
    ));
    out.push(row(
        &t("hw.fp16_tflops"),
        format!("{:.0} TFLOPS", spec.fp16_tflops),
        "",
    ));
    out.push(row(
        &t("hw.fp8_support"),
        if spec.fp8_support {
            t("hw.bool_yes")
        } else {
            t("hw.bool_no")
        },
        "",
    ));
    out.push(row(
        &t("hw.fp4_support"),
        if spec.fp4_support {
            t("hw.bool_yes")
        } else {
            t("hw.bool_no")
        },
        "",
    ));
    if let Some(notes) = spec.localized_notes(&get_locale()) {
        out.push(row(&t("hw.notes"), notes, ""));
    }
    if let Some(source) = &spec.spec_source {
        out.push(row(&t("hw.spec_source"), source, ""));
    }
}

fn render_fleet(report: &EvaluationReport, out: &mut Vec<String>) {
    let Some(fleet) = &report.fleet else {
        if report.gpu_spec.is_some() {
            out.push(t("fleet.gpu_spec_unknown"));
        }
        return;
    };
    out.push(String::new());
    out.push(format!(
        "{} - {}",
        t("section.fleet"),
        report
            .gpu_spec
            .as_ref()
            .map(|spec| spec.id.as_str())
            .unwrap_or(report.gpu.as_str())
    ));
    let ctx_cols = select_concurrency_columns(fleet);
    for opt in &fleet.options {
        let headroom = opt
            .usable_bytes_per_gpu
            .saturating_sub(opt.weight_bytes_per_gpu);
        let marker = if fleet.best_tier == Some(opt.tier) {
            " *"
        } else {
            ""
        };
        let mut line = format!(
            "{}{}: {} GPUs, weight/GPU {}, headroom {}",
            t(&format!("fleet.tier.{}", opt.tier)),
            marker,
            opt.gpu_count,
            fmt_bytes(opt.weight_bytes_per_gpu),
            if headroom > 0 {
                fmt_bytes(headroom)
            } else {
                "-".to_string()
            }
        );
        if opt.pipeline_parallel_size > 1 {
            line.push_str(&format!(
                ", layout TP{}xPP{} ({} nodes)",
                opt.tensor_parallel_size, opt.pipeline_parallel_size, opt.node_count
            ));
        }
        for ctx in &ctx_cols {
            let concurrent = opt
                .max_concurrent_by_context
                .iter()
                .find(|(candidate, _)| candidate == ctx)
                .map(|(_, count)| *count)
                .unwrap_or(0);
            line.push_str(&format!(
                ", {} {}",
                t_with(
                    "fleet.col.concurrent_at_ctx",
                    &HashMap::from([("ctx", fmt_ctx(*ctx))])
                ),
                if concurrent > 0 {
                    format!("~{concurrent}")
                } else {
                    "x".to_string()
                }
            ));
        }
        if !opt.fits {
            line.push_str(" (does not fit)");
        }
        out.push(line);
    }
    let note = if get_locale() == "zh" {
        &fleet.constraint_note_zh
    } else {
        &fleet.constraint_note_en
    };
    out.push(format!("{} {}", t("fleet.constraint"), note));
    if fleet.best_tier.is_some() {
        out.push(format!("* {}", t("fleet.best_marker")));
    } else {
        out.push(t("fleet.no_recommended_tier"));
    }
}

fn render_performance(report: &EvaluationReport, out: &mut Vec<String>) {
    let (Some(prefill), Some(decode), Some(concurrency), Some(input_tokens), Some(target_tps)) = (
        &report.prefill,
        &report.decode,
        &report.concurrency,
        report.perf_input_tokens,
        report.perf_target_tokens_per_sec,
    ) else {
        return;
    };
    out.push(String::new());
    out.push(t("section.performance"));
    out.push(t_with(
        "perf.assumptions_note",
        &HashMap::from([
            ("input_tokens", input_tokens.to_string()),
            (
                "output_tokens",
                report.perf_output_tokens.unwrap_or(512).to_string(),
            ),
            ("target_tps", format!("{target_tps:.1}")),
            (
                "prefill_util",
                format!("{:.0}%", prefill.utilization * 100.0),
            ),
            (
                "decode_util",
                format!("{:.0}%", decode.bw_utilization * 100.0),
            ),
            (
                "degradation",
                format!("{:.2}", concurrency.degradation_factor),
            ),
        ]),
    ));
    out.push(row(
        &t("perf.prefill_latency"),
        format!("{:.1} ms", prefill.latency_ms.value),
        format_tag(&prefill.latency_ms),
    ));
    out.push(row(
        &t("perf.decode_throughput_per_gpu"),
        format!("{:.1} tok/s", decode.per_gpu_tokens_per_sec.value),
        format_tag(&decode.per_gpu_tokens_per_sec),
    ));
    out.push(row(
        &t("perf.decode_throughput_cluster"),
        format!("{:.1} tok/s", decode.cluster_tokens_per_sec.value),
        format_tag(&decode.cluster_tokens_per_sec),
    ));
    if let Some(active) = &decode.moe_active_tokens_per_sec {
        out.push(row(
            &t("perf.decode_moe_active_optimistic"),
            format!("{:.1} tok/s", active.value),
            format_tag(active),
        ));
    }
    out.push(row(
        &t("perf.k_bound"),
        concurrency.k_bound.value.to_string(),
        format_tag(&concurrency.k_bound),
    ));
    out.push(row(
        &t("perf.l_bound"),
        concurrency.l_bound.value.to_string(),
        format_tag(&concurrency.l_bound),
    ));
    out.push(row(
        &t("perf.max_concurrent"),
        concurrency.max_concurrent.value.to_string(),
        format_tag(&concurrency.max_concurrent),
    ));
    let bottleneck_label = t(&format!(
        "perf.bottleneck.{}",
        concurrency.bottleneck.as_str()
    ));
    let reason = if get_locale() == "zh" {
        &concurrency.bottleneck_reason_zh
    } else {
        &concurrency.bottleneck_reason_en
    };
    out.push(row(
        &t("perf.bottleneck"),
        format!("{bottleneck_label}: {reason}"),
        "",
    ));
    out.push(format!("{}:", t("perf.optimization.header")));
    for key in [
        "perf.opt.quantize_int4",
        "perf.opt.relax_sla",
        "perf.opt.kv_fp8",
        "perf.opt.moe_offload",
    ] {
        out.push(format!("  - {}", t(key)));
    }
}

fn render_command(report: &EvaluationReport, out: &mut Vec<String>) {
    let (Some(command), Some(fleet)) = (&report.generated_command, &report.fleet) else {
        return;
    };
    let Some(best) = fleet.best_option() else {
        return;
    };
    out.push(String::new());
    out.push(format!(
        "{} - {}",
        t("section.command"),
        t_with(
            "command.tier_note",
            &HashMap::from([
                ("tier", t(&format!("fleet.tier.{}", best.tier))),
                ("gpus", best.gpu_count.to_string()),
            ])
        )
    ));
    out.push(command.clone());
}

fn render_label_legend(out: &mut Vec<String>) {
    out.push(String::new());
    let labels = Label::all()
        .iter()
        .map(|label| format!("[{}]", t(&format!("label.{}", label.as_str()))))
        .collect::<Vec<_>>()
        .join(" ");
    out.push(format!("{} {}", t("section.labels"), labels));
}

fn row(field: &str, value: impl ToString, label: impl ToString) -> String {
    let label = label.to_string();
    if label.is_empty() {
        format!("{field}: {}", value.to_string())
    } else {
        format!("{field}: {} {label}", value.to_string())
    }
}

fn verified_tag() -> String {
    format!("[{}]", t("label.verified"))
}

fn verif_label(entry: &EngineCompatEntry) -> String {
    let label = match entry.verification_level.as_str() {
        "verified" => Label::Verified,
        "cited" => Label::Cited,
        "unverified" => Label::Unverified,
        _ => Label::Unknown,
    };
    format!("[{}]", t(&format!("label.{}", label.as_str())))
}

fn fmt_flag(flag: &EngineFlag) -> String {
    match &flag.value {
        Some(value) => format!("{} {value}", flag.flag),
        None => flag.flag.clone(),
    }
}

fn fmt_source(source: &EngineSource) -> String {
    let label = t(&format!("source.{}", source.source_type));
    if source.source_type == "tested" {
        return format!(
            "[{}] {} @ {} ({})",
            label,
            source.tester.as_deref().unwrap_or(""),
            source.hardware.as_deref().unwrap_or(""),
            source.date.as_deref().unwrap_or("")
        );
    }
    if let Some(url) = &source.url {
        let captured = source
            .captured_date
            .as_ref()
            .map(|date| format!(" ({} {date})", t("source.captured_on")))
            .unwrap_or_default();
        return format!("[{label}] {url}{captured}");
    }
    format!("[{label}]")
}

fn select_concurrency_columns(fleet: &FleetRecommendation) -> Vec<u64> {
    let mut all_ctxs = fleet
        .options
        .iter()
        .flat_map(|option| option.max_concurrent_by_context.iter().map(|(ctx, _)| *ctx))
        .collect::<Vec<_>>();
    all_ctxs.sort_unstable();
    all_ctxs.dedup();
    if all_ctxs.is_empty() {
        return Vec::new();
    }

    let mut picks = Vec::new();
    if all_ctxs.contains(&131_072) {
        picks.push(131_072);
    }
    let max_ctx = *all_ctxs.last().unwrap();
    if max_ctx > 131_072 && !picks.contains(&max_ctx) {
        picks.push(max_ctx);
    }
    if picks.is_empty() {
        picks.push(if all_ctxs.contains(&32_768) {
            32_768
        } else {
            max_ctx
        });
    }
    picks
}

fn fmt_ctx(ctx_tokens: u64) -> String {
    if ctx_tokens >= 1_000_000 {
        if ctx_tokens % 1_000_000 == 0 {
            return format!("{}M", ctx_tokens / 1_000_000);
        }
        return format!("{:.1}M", ctx_tokens as f64 / 1_000_000.0);
    }
    if ctx_tokens >= 1024 {
        return format!("{}K", ctx_tokens / 1024);
    }
    ctx_tokens.to_string()
}

fn fmt_u64(value: u64) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    let first_group = text.len() % 3;
    for (idx, ch) in text.chars().enumerate() {
        if idx > 0 && (idx == first_group || (idx > first_group && (idx - first_group) % 3 == 0)) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}
