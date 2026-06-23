from __future__ import annotations

from llm_cal.core.evaluator import Evaluator
from llm_cal.model_source.base import ModelArtifact, ModelSource, SiblingFile


class StaticSource(ModelSource):
    name = "huggingface"

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    def fetch(self, model_id: str) -> ModelArtifact:
        assert model_id == self.artifact.model_id
        return self.artifact


def _llama_artifact() -> ModelArtifact:
    return ModelArtifact(
        source="huggingface",
        model_id="test/llama-mini",
        commit_sha="abc1234def",
        config={
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "vocab_size": 100,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "max_position_embeddings": 8192,
        },
        siblings=(
            SiblingFile("model-00001-of-00002.safetensors", 5_472),
            SiblingFile("model-00002-of-00002.safetensors", 5_472),
            SiblingFile("tokenizer.json", 100),
        ),
    )


def test_evaluate_applies_all_user_tunable_runtime_options():
    report = Evaluator(source=StaticSource(_llama_artifact())).evaluate(
        model_id="test/llama-mini",
        gpu="H800",
        engine="sglang",
        gpu_count=2,
        context_length=4096,
        input_tokens=123,
        output_tokens=45,
        target_tokens_per_sec=17.5,
        prefill_utilization=0.33,
        decode_bw_utilization=0.44,
        concurrency_degradation=1.67,
    )

    assert report.engine == "sglang"
    assert list(report.kv_cache_by_context) == [4096]
    assert [option.gpu_count for option in report.fleet.options] == [2]  # type: ignore[union-attr]
    assert report.generated_command is not None
    assert "test/llama-mini" in report.generated_command
    assert "4096" in report.generated_command
    assert report.perf_input_tokens == 123
    assert report.perf_output_tokens == 45
    assert report.perf_target_tokens_per_sec == 17.5
    assert report.prefill is not None
    assert report.prefill.utilization == 0.33
    assert report.decode is not None
    assert report.decode.bw_utilization == 0.44
    assert report.concurrency is not None
    assert report.concurrency.target_tokens_per_sec == 17.5
    assert report.concurrency.degradation_factor == 1.67
