use std::collections::BTreeMap;

use llm_infer_cal_core::architecture::profile::{ArchitectureProfile, AttentionTraits, MoeTraits};
use llm_infer_cal_core::core::evaluator::{EvaluationOptions, EvaluationReport, Evaluator};
use llm_infer_cal_core::engine_compat::EngineCompatEntry;
use llm_infer_cal_core::fleet::planner::FleetRecommendation;
use llm_infer_cal_core::hardware::loader::GPUSpec;
use llm_infer_cal_core::model_source::base::ModelSource;
use llm_infer_cal_core::model_source::huggingface::HuggingFaceSource;
use llm_infer_cal_core::model_source::modelscope::ModelScopeSource;
use llm_infer_cal_core::output::labels::AnnotatedValue;
use llm_infer_cal_core::performance::compute::{DecodeEstimate, PrefillEstimate};
use llm_infer_cal_core::performance::concurrency::ConcurrencyAnalysis;
use llm_infer_cal_core::weight_analyzer::reconciler::ReconciliationReport;
use llm_infer_cal_core::weight_analyzer::{QuantizationScheme, WeightReport};
use serde_json::{json, Value};

fn main() {
    match run() {
        Ok(value) => println!("{}", serde_json::to_string_pretty(&value).unwrap()),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<Value, String> {
    let args = Args::parse(std::env::args().skip(1).collect())?;
    let source: Box<dyn ModelSource> = match args.source.as_str() {
        "modelscope" => Box::<ModelScopeSource>::default(),
        "huggingface" | "hf" => Box::<HuggingFaceSource>::default(),
        other => return Err(format!("unknown --source {other:?}")),
    };
    let evaluator = Evaluator::without_cache(source);
    let report = evaluator
        .evaluate(
            &args.model_id,
            &args.gpu,
            &args.engine,
            EvaluationOptions::default(),
        )
        .map_err(|error| error.to_string())?;
    Ok(snapshot(&report))
}

#[derive(Debug)]
struct Args {
    model_id: String,
    gpu: String,
    engine: String,
    source: String,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut model_id = None;
        let mut gpu = "H800".to_string();
        let mut engine = "vllm".to_string();
        let mut source = "huggingface".to_string();
        let mut idx = 0;
        while idx < args.len() {
            match args[idx].as_str() {
                "--gpu" => {
                    idx += 1;
                    gpu = args
                        .get(idx)
                        .ok_or_else(|| "--gpu requires a value".to_string())?
                        .clone();
                }
                "--engine" => {
                    idx += 1;
                    engine = args
                        .get(idx)
                        .ok_or_else(|| "--engine requires a value".to_string())?
                        .clone();
                }
                "--source" => {
                    idx += 1;
                    source = args
                        .get(idx)
                        .ok_or_else(|| "--source requires a value".to_string())?
                        .clone();
                }
                value if value.starts_with("--") => {
                    return Err(format!("unknown option {value:?}"));
                }
                value => {
                    if model_id.replace(value.to_string()).is_some() {
                        return Err("only one MODEL_ID is supported".to_string());
                    }
                }
            }
            idx += 1;
        }
        Ok(Self {
            model_id: model_id.ok_or_else(|| "MODEL_ID is required".to_string())?,
            gpu,
            engine,
            source,
        })
    }
}

fn snapshot(report: &EvaluationReport) -> Value {
    json!({
        "model_id": report.model_id,
        "source": report.source,
        "commit_sha": report.commit_sha,
        "gpu": report.gpu,
        "engine": report.engine,
        "profile": profile(&report.profile),
        "weight": weight(&report.weight),
        "total_params_estimate": av_u64(&report.total_params_estimate),
        "reconciliation": reconciliation(&report.reconciliation),
        "kv_cache_by_context": kv_cache(&report.kv_cache_by_context),
        "engine_match": report.engine_match.as_ref().map(engine_match),
        "gpu_spec": report.gpu_spec.as_ref().map(gpu_spec),
        "fleet": report.fleet.as_ref().map(fleet),
        "generated_command": report.generated_command,
        "prefill": report.prefill.as_ref().map(prefill),
        "decode": report.decode.as_ref().map(decode),
        "concurrency": report.concurrency.as_ref().map(concurrency),
        "perf_input_tokens": report.perf_input_tokens,
        "perf_output_tokens": report.perf_output_tokens,
        "perf_target_tokens_per_sec": report.perf_target_tokens_per_sec,
    })
}

fn profile(profile: &ArchitectureProfile) -> Value {
    json!({
        "model_type": profile.model_type,
        "architectures": profile.architectures,
        "family": profile.family.as_str(),
        "num_hidden_layers": profile.num_hidden_layers,
        "hidden_size": profile.hidden_size,
        "vocab_size": profile.vocab_size,
        "confidence": profile.confidence.as_str(),
        "attention": profile.attention.as_ref().map(attention),
        "moe": profile.moe.as_ref().map(moe),
        "position": profile.position.as_ref().map(|position| json!({
            "rope_type": position.rope_type,
            "rope_theta": position.rope_theta,
            "rope_scaling_factor": position.rope_scaling_factor,
            "max_position_embeddings": position.max_position_embeddings,
        })),
        "sliding_window": profile.sliding_window,
    })
}

fn attention(attention: &AttentionTraits) -> Value {
    json!({
        "variant": attention.variant.as_str(),
        "num_heads": attention.num_heads,
        "num_kv_heads": attention.num_kv_heads,
        "head_dim": attention.head_dim,
        "q_lora_rank": attention.q_lora_rank,
        "kv_lora_rank": attention.kv_lora_rank,
        "compress_ratios": attention.compress_ratios.clone().unwrap_or_default(),
        "nsa_topk": attention.nsa_topk,
    })
}

fn moe(moe: &MoeTraits) -> Value {
    json!({
        "num_routed_experts": moe.num_routed_experts,
        "num_shared_experts": moe.num_shared_experts,
        "num_experts_per_tok": moe.num_experts_per_tok,
        "moe_intermediate_size": moe.moe_intermediate_size,
    })
}

fn weight(weight: &WeightReport) -> Value {
    json!({
        "total_bytes": av_u64(&weight.total_bytes),
        "bits_per_param": weight.bits_per_param.as_ref().map(av_f64),
        "quantization_guess": av_quant(&weight.quantization_guess),
    })
}

fn reconciliation(reconciliation: &ReconciliationReport) -> Value {
    json!({
        "observed_bytes": reconciliation.observed_bytes,
        "total_params": reconciliation.total_params,
        "best": av_quant(&reconciliation.best),
        "candidates": reconciliation.candidates.iter().map(|candidate| json!({
            "scheme": candidate.scheme.as_str(),
            "predicted_bytes": candidate.predicted_bytes,
            "delta_bytes": candidate.delta_bytes,
            "relative_error": candidate.relative_error,
        })).collect::<Vec<_>>(),
    })
}

fn kv_cache(kv_cache: &BTreeMap<u64, AnnotatedValue<u64>>) -> Value {
    Value::Array(
        kv_cache
            .iter()
            .map(|(ctx, value)| json!([ctx, av_u64(value)]))
            .collect(),
    )
}

fn engine_match(entry: &EngineCompatEntry) -> Value {
    json!({
        "engine": entry.engine,
        "version_spec": entry.version_spec,
        "support": entry.support,
        "verification_level": entry.verification_level,
        "required_flags": entry.required_flags.iter().map(|flag| json!({
            "flag": flag.flag,
            "value": flag.value,
        })).collect::<Vec<_>>(),
        "optional_flags": entry.optional_flags.iter().map(|flag| json!({
            "flag": flag.flag,
            "value": flag.value,
        })).collect::<Vec<_>>(),
        "caveats_zh": entry.caveats_zh,
    })
}

fn gpu_spec(gpu: &GPUSpec) -> Value {
    json!({
        "id": gpu.id,
        "memory_gb": gpu.memory_gb,
        "nvlink_bandwidth_gbps": gpu.nvlink_bandwidth_gbps,
        "memory_bandwidth_gbps": gpu.memory_bandwidth_gbps,
        "fp16_tflops": gpu.fp16_tflops,
        "fp8_support": gpu.fp8_support,
        "fp4_support": gpu.fp4_support,
    })
}

fn fleet(fleet: &FleetRecommendation) -> Value {
    json!({
        "best_tier": fleet.best_tier,
        "valid_tp_sizes": fleet.valid_tp_sizes,
        "constraint_note_zh": fleet.constraint_note_zh,
        "options": fleet.options.iter().map(|option| json!({
            "tier": option.tier,
            "gpu_count": option.gpu_count,
            "weight_bytes_per_gpu": option.weight_bytes_per_gpu,
            "kv_bytes_per_request": option.kv_bytes_per_request,
            "kv_reference_context_tokens": option.kv_reference_context_tokens,
            "max_concurrent_at_reference_ctx": option.max_concurrent_at_reference_ctx,
            "max_concurrent_by_context": option.max_concurrent_by_context.iter()
                .map(|(ctx, value)| json!([ctx, value]))
                .collect::<Vec<_>>(),
            "usable_bytes_per_gpu": option.usable_bytes_per_gpu,
            "fits": option.fits,
            "reason_zh": option.reason_zh,
        })).collect::<Vec<_>>(),
    })
}

fn prefill(prefill: &PrefillEstimate) -> Value {
    json!({
        "total_flops": av_u64(&prefill.total_flops),
        "peak_effective_tflops": av_f64(&prefill.peak_effective_tflops),
        "latency_ms": av_f64(&prefill.latency_ms),
        "utilization": prefill.utilization,
    })
}

fn decode(decode: &DecodeEstimate) -> Value {
    json!({
        "active_weight_bytes_per_gpu": av_u64(&decode.active_weight_bytes_per_gpu),
        "per_gpu_tokens_per_sec": av_f64(&decode.per_gpu_tokens_per_sec),
        "cluster_tokens_per_sec": av_f64(&decode.cluster_tokens_per_sec),
        "bw_utilization": decode.bw_utilization,
        "cluster_comm_efficiency": decode.cluster_comm_efficiency,
        "moe_active_weight_bytes_per_gpu": decode.moe_active_weight_bytes_per_gpu.as_ref().map(av_u64),
        "moe_active_tokens_per_sec": decode.moe_active_tokens_per_sec.as_ref().map(av_f64),
    })
}

fn concurrency(concurrency: &ConcurrencyAnalysis) -> Value {
    json!({
        "k_bound": av_u64(&concurrency.k_bound),
        "l_bound": av_u64(&concurrency.l_bound),
        "max_concurrent": av_u64(&concurrency.max_concurrent),
        "bottleneck": concurrency.bottleneck.as_str(),
        "bottleneck_reason_zh": concurrency.bottleneck_reason_zh,
        "target_tokens_per_sec": concurrency.target_tokens_per_sec,
        "degradation_factor": concurrency.degradation_factor,
    })
}

fn av_u64(av: &AnnotatedValue<u64>) -> Value {
    json!({
        "value": av.value,
        "label": av.label.as_str(),
        "source": av.source,
    })
}

fn av_f64(av: &AnnotatedValue<f64>) -> Value {
    json!({
        "value": av.value,
        "label": av.label.as_str(),
        "source": av.source,
    })
}

fn av_quant(av: &AnnotatedValue<QuantizationScheme>) -> Value {
    json!({
        "value": av.value.as_str(),
        "label": av.label.as_str(),
        "source": av.source,
    })
}
