use std::collections::BTreeMap;

use crate::architecture::detector::detect;
use crate::architecture::formulas::kv_cache::compute_kv_cache_bytes;
use crate::architecture::formulas::weight::estimate_total_params;
use crate::architecture::profile::ArchitectureProfile;
use crate::command_generator::sglang::generate_sglang_command;
use crate::command_generator::vllm::generate_vllm_command;
use crate::core::cache::{ArtifactCache, CacheKey};
use crate::engine_compat::loader::find_match;
use crate::engine_compat::EngineCompatEntry;
use crate::fleet::planner::{kv_shards, plan, FleetRecommendation};
use crate::hardware::loader::{lookup, GPUSpec};
use crate::model_source::base::{ModelArtifact, ModelSource, ModelSourceError};
use crate::model_source::huggingface::HuggingFaceSource;
use crate::output::labels::AnnotatedValue;
use crate::performance::compute::{
    estimate_decode, estimate_prefill, DecodeEstimate, PrefillEstimate,
    DEFAULT_CLUSTER_COMM_EFFICIENCY, DEFAULT_DECODE_BW_UTILIZATION, DEFAULT_PREFILL_UTILIZATION,
};
use crate::performance::concurrency::{analyze as analyze_concurrency, ConcurrencyAnalysis};
use crate::weight_analyzer::fingerprint::{from_config, from_safetensors_dtypes, QuantFingerprint};
use crate::weight_analyzer::reconciler::{reconcile, ReconciliationReport};
use crate::weight_analyzer::safetensors_reader::{fetch_tensor_dtypes, pick_sample_shard};
use crate::weight_analyzer::{analyze, WeightReport};

const KV_REFERENCE_CTX: u64 = 131_072;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvaluationOptions {
    pub gpu_count: Option<u64>,
    pub context_length: Option<u64>,
    pub refresh: bool,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub target_tokens_per_sec: Option<f64>,
    pub prefill_utilization: f64,
    pub decode_bw_utilization: f64,
    pub concurrency_degradation: f64,
}

impl Default for EvaluationOptions {
    fn default() -> Self {
        Self {
            gpu_count: None,
            context_length: None,
            refresh: false,
            input_tokens: None,
            output_tokens: None,
            target_tokens_per_sec: None,
            prefill_utilization: DEFAULT_PREFILL_UTILIZATION,
            decode_bw_utilization: DEFAULT_DECODE_BW_UTILIZATION,
            concurrency_degradation: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationReport {
    pub model_id: String,
    pub source: String,
    pub commit_sha: Option<String>,
    pub gpu: String,
    pub gpu_spec: Option<GPUSpec>,
    pub gpu_error: Option<String>,
    pub engine: String,
    pub profile: ArchitectureProfile,
    pub weight: WeightReport,
    pub total_params_estimate: AnnotatedValue<u64>,
    pub reconciliation: ReconciliationReport,
    pub kv_cache_by_context: BTreeMap<u64, AnnotatedValue<u64>>,
    pub engine_match: Option<EngineCompatEntry>,
    pub fleet: Option<FleetRecommendation>,
    pub generated_command: Option<String>,
    pub prefill: Option<PrefillEstimate>,
    pub decode: Option<DecodeEstimate>,
    pub concurrency: Option<ConcurrencyAnalysis>,
    pub perf_input_tokens: Option<u64>,
    pub perf_output_tokens: Option<u64>,
    pub perf_target_tokens_per_sec: Option<f64>,
}

pub struct Evaluator {
    source: Box<dyn ModelSource>,
    cache: Option<ArtifactCache>,
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new(
            Box::<HuggingFaceSource>::default(),
            ArtifactCache::with_default_ttl(None).ok(),
        )
    }
}

impl Evaluator {
    pub fn new(source: Box<dyn ModelSource>, cache: Option<ArtifactCache>) -> Self {
        Self { source, cache }
    }

    pub fn without_cache(source: Box<dyn ModelSource>) -> Self {
        Self::new(source, None)
    }

    pub fn evaluate(
        &self,
        model_id: &str,
        gpu: &str,
        engine: &str,
        options: EvaluationOptions,
    ) -> Result<EvaluationReport, ModelSourceError> {
        let artifact = self.fetch(model_id, options.refresh)?;
        let profile = detect(&artifact.config);

        let total_params_estimate = estimate_total_params(&profile);
        let total_params = total_params_estimate.value;

        let observed_bytes_for_fp = artifact
            .siblings
            .iter()
            .filter(|sibling| sibling.filename.ends_with(".safetensors"))
            .map(|sibling| sibling.size.unwrap_or(0))
            .sum::<u64>();
        let fingerprint = self.resolve_quant_fingerprint(
            &artifact,
            observed_bytes_for_fp,
            if total_params > 0 { total_params } else { 0 },
        );
        let weight = analyze(
            &artifact.siblings,
            (total_params > 0).then_some(total_params),
            fingerprint.as_ref(),
        );
        let reconciliation = reconcile(
            weight.total_bytes.value,
            if total_params > 0 { total_params } else { 1 },
            fingerprint.as_ref(),
        );

        let contexts_to_report = select_context_lengths(&profile, options.context_length);
        let kv_cache_by_context = contexts_to_report
            .into_iter()
            .map(|ctx| (ctx, compute_kv_cache_bytes(&profile, ctx, 2)))
            .collect::<BTreeMap<_, _>>();

        let engine_match = find_match(engine, &profile.model_type, None, None);

        let (gpu_spec, gpu_error) = match lookup(gpu) {
            Ok(spec) => (Some(spec), None),
            Err(error) => (None, Some(error.to_string())),
        };

        let mut fleet = None;
        let mut generated_command = None;
        if let Some(spec) = &gpu_spec {
            if weight.total_bytes.value > 0 {
                let kv_ref = compute_kv_cache_bytes(&profile, KV_REFERENCE_CTX, 2);
                let kv_by_context_bytes = kv_cache_by_context
                    .iter()
                    .filter_map(|(ctx, value)| (value.value > 0).then_some((*ctx, value.value)))
                    .collect::<Vec<_>>();
                let recommendation = plan(
                    &profile,
                    weight.total_bytes.value,
                    kv_ref.value.max(1),
                    spec,
                    options.gpu_count,
                    &kv_by_context_bytes,
                );
                let chosen_count = options.gpu_count.unwrap_or_else(|| {
                    recommendation
                        .options
                        .iter()
                        .find(|option| option.tier == recommendation.best_tier)
                        .or_else(|| recommendation.options.first())
                        .map(|option| option.gpu_count)
                        .unwrap_or(1)
                });
                generated_command = Some(generate_command(
                    engine,
                    model_id,
                    &profile,
                    chosen_count,
                    engine_match.as_ref(),
                    options.context_length,
                ));
                fleet = Some(recommendation);
            }
        }

        let mut prefill = None;
        let mut decode = None;
        let mut concurrency = None;
        if let (Some(spec), Some(recommendation)) = (&gpu_spec, &fleet) {
            if total_params > 0 {
                let chosen = options.gpu_count.unwrap_or_else(|| {
                    recommendation
                        .options
                        .iter()
                        .find(|option| option.tier == recommendation.best_tier)
                        .or_else(|| recommendation.options.first())
                        .map(|option| option.gpu_count)
                        .unwrap_or(1)
                });
                let input_tokens = options.input_tokens.unwrap_or(2000);
                let target_tokens_per_sec = options.target_tokens_per_sec.unwrap_or(30.0);

                prefill = Some(estimate_prefill(
                    &profile,
                    total_params,
                    spec,
                    chosen,
                    input_tokens,
                    options.prefill_utilization,
                ));

                let moe_active_ratio = profile.moe.as_ref().and_then(|moe| {
                    let active = moe.num_experts_per_tok + moe.num_shared_experts;
                    let total = moe.num_routed_experts + moe.num_shared_experts;
                    (total > 0).then_some(active as f64 / total as f64)
                });

                let decode_estimate = estimate_decode(
                    &profile,
                    weight.total_bytes.value,
                    spec,
                    chosen,
                    options.decode_bw_utilization,
                    DEFAULT_CLUSTER_COMM_EFFICIENCY,
                    moe_active_ratio,
                );

                let chosen_option = recommendation
                    .options
                    .iter()
                    .find(|option| option.gpu_count == chosen)
                    .or_else(|| recommendation.options.last());
                if let Some(chosen_option) = chosen_option {
                    if let Some((kv_ref_ctx, kv_ref_bytes)) =
                        reference_context_bytes(&kv_cache_by_context)
                    {
                        let _ = kv_ref_ctx;
                        let headroom_per_gpu = chosen_option
                            .usable_bytes_per_gpu
                            .saturating_sub(chosen_option.weight_bytes_per_gpu);
                        let shards = kv_shards(&profile, chosen).max(1);
                        let kv_ref_per_gpu = (kv_ref_bytes / shards).max(1);
                        concurrency = Some(analyze_concurrency(
                            headroom_per_gpu,
                            kv_ref_per_gpu,
                            &decode_estimate,
                            target_tokens_per_sec,
                            options.concurrency_degradation,
                        ));
                    }
                }
                decode = Some(decode_estimate);
            }
        }

        let has_fleet = fleet.is_some();
        Ok(EvaluationReport {
            model_id: model_id.to_string(),
            source: artifact.source,
            commit_sha: artifact.commit_sha,
            gpu: gpu.to_string(),
            gpu_spec,
            gpu_error,
            engine: engine.to_string(),
            profile,
            weight,
            total_params_estimate,
            reconciliation,
            kv_cache_by_context,
            engine_match,
            fleet,
            generated_command,
            prefill,
            decode,
            concurrency,
            perf_input_tokens: has_fleet.then_some(options.input_tokens.unwrap_or(2000)),
            perf_output_tokens: has_fleet.then_some(options.output_tokens.unwrap_or(512)),
            perf_target_tokens_per_sec: has_fleet
                .then_some(options.target_tokens_per_sec.unwrap_or(30.0)),
        })
    }

    fn fetch(&self, model_id: &str, refresh: bool) -> Result<ModelArtifact, ModelSourceError> {
        let artifact = self.source.fetch(model_id)?;
        let key = CacheKey::new(self.source.name(), model_id, artifact.commit_sha.as_deref());
        if let Some(cache) = &self.cache {
            if let Ok(Some(cached)) = cache.get(&key, refresh) {
                return Ok(cached);
            }
            let _ = cache.set(&key, &artifact);
        }
        Ok(artifact)
    }

    fn resolve_quant_fingerprint(
        &self,
        artifact: &ModelArtifact,
        observed_bytes: u64,
        total_params: u64,
    ) -> Option<QuantFingerprint> {
        let fp = from_config(&artifact.config);
        if fp
            .as_ref()
            .is_some_and(|fp| fingerprint_matches_bytes(fp, observed_bytes, total_params))
        {
            return fp;
        }

        let shard = match pick_sample_shard(&artifact.siblings) {
            Some(shard) => shard,
            None => return fp,
        };
        let revision = artifact
            .commit_sha
            .as_deref()
            .unwrap_or(match artifact.source.as_str() {
                "modelscope" => "master",
                _ => "main",
            });
        let dtypes = match fetch_tensor_dtypes(
            &artifact.source,
            &artifact.model_id,
            revision,
            &shard.filename,
            None,
        ) {
            Some(dtypes) => dtypes,
            None => return fp,
        };

        from_safetensors_dtypes(&dtypes).or(fp)
    }
}

fn fingerprint_matches_bytes(
    fp: &QuantFingerprint,
    observed_bytes: u64,
    total_params: u64,
) -> bool {
    let bpp = fp.scheme.bpp();
    if bpp <= 0.0 || total_params == 0 || observed_bytes == 0 {
        return true;
    }
    let predicted = bpp * total_params as f64;
    let rel_err = (observed_bytes as f64 - predicted).abs() / predicted;
    rel_err <= 0.15
}

fn select_context_lengths(profile: &ArchitectureProfile, override_ctx: Option<u64>) -> Vec<u64> {
    if let Some(override_ctx) = override_ctx {
        return vec![override_ctx];
    }

    let mut candidates = vec![4_096, 32_768, KV_REFERENCE_CTX];
    let max_pos = profile
        .position
        .as_ref()
        .and_then(|position| position.max_position_embeddings);
    if let Some(max_pos) = max_pos {
        if max_pos > KV_REFERENCE_CTX {
            candidates.push(max_pos);
        }
        candidates.retain(|ctx| *ctx <= max_pos);
    }
    candidates
}

fn reference_context_bytes(
    kv_cache_by_context: &BTreeMap<u64, AnnotatedValue<u64>>,
) -> Option<(u64, u64)> {
    if let Some(value) = kv_cache_by_context.get(&KV_REFERENCE_CTX) {
        return Some((KV_REFERENCE_CTX, value.value));
    }
    kv_cache_by_context
        .iter()
        .next_back()
        .map(|(ctx, value)| (*ctx, value.value))
}

fn generate_command(
    engine: &str,
    model_id: &str,
    profile: &ArchitectureProfile,
    tp: u64,
    entry: Option<&EngineCompatEntry>,
    max_model_len: Option<u64>,
) -> String {
    if engine.trim().eq_ignore_ascii_case("sglang") {
        generate_sglang_command(model_id, profile, tp, entry, max_model_len)
    } else {
        generate_vllm_command(model_id, profile, tp, entry, max_model_len)
    }
}
