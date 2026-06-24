import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { App, defaultSpeculativeFormPatch, isReportCurrent, memoryExplainText, totalRequiredBytes } from './App';

describe('App shell', () => {
  it('does not render header status metrics', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).not.toContain('statusRail');
    expect(html).not.toContain('127.0.0.1:8080');
  });

  it('renders the GPU model picker as a multi-select control', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).toContain('data-testid="gpu-model-picker"');
  });

  it('does not render explain or refresh cache controls', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).not.toContain('输出推导链（--explain）');
    expect(html).not.toContain('刷新缓存');
    expect(html).not.toContain('目标并发');
  });

  it('renders reference-inspired calculator sections', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).toContain('llm-infer-cal');
    expect(html).toContain('GPU Memory Estimator');
    expect(html).toContain('Configuration');
    expect(html).toContain('Estimates');
    expect(html).toContain('VRAM Breakdown');
    expect(html).toContain('Formula Reference');
    expect(html).toContain('required_per_gpu = max(decode_required, prefill_required)');
    expect(html).toContain('effective_seq_len = sliding_window ? min(seq_len, sliding_window) : seq_len');
    expect(html).toContain('NSA: baseline * min(nsa_topk / effective_seq_len, 1.0)');
    expect(html).toContain('MoE activation correction = 1 + active_experts / total_experts * 0.5');
    expect(html).toContain('Inference Optimizations');
    expect(html).toContain('Compare GPUs');
    expect(html).not.toContain('LLM VRAM Calculator');
    expect(html).not.toContain('v4.0');
    expect(html).not.toContain('v0 calculator');
    expect(html).not.toContain('Shun-Calvin/llm-vram-calculator');
  });

  it('keeps context and workload overrides inside advanced settings only', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).not.toContain('Context &amp; Workload');
    expect(html).toContain('Context Window');
    expect(html).toContain('Concurrent Users');
    expect(html).toContain('自动使用模型上下文');
    expect(html).toContain('自动推荐 min/dev/prod');
  });

  it('renders speculative decoding as a selectable workflow instead of raw text inputs', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).toContain('Speculative Decoding');
    expect(html).toContain('Draft / MTP model predicts, main model verifies');
    expect(html).toContain('Reduces KV cache overhead by ~25%');
    expect(html).not.toContain('Draft/EAGLE 模型 ID');
    expect(html).not.toContain('Speculative 额外权重 GiB');
  });

  it('defaults newly enabled speculative decoding to MTP without a draft model', () => {
    expect(defaultSpeculativeFormPatch(4)).toEqual({
      speculative_enabled: true,
      speculative_mode: 'mtp',
      speculative_num_draft_tokens: '4',
      speculative_draft_model_id: '',
      speculative_extra_weight_gb: '0.3',
    });
  });

  it('renders only the core VRAM summary cards', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).toContain('Total VRAM');
    expect(html).toContain('Model Weights / GPU');
    expect(html).toContain('Recommended GPUs');
    expect(html).toContain('Required / GPU (per GPU)');
    expect(html).not.toContain('Time to First Token');
    expect(html).not.toContain('Tokens / Second');
    expect(html).not.toContain('Engine Support');
  });

  it('renders inference process instead of the simulation panel', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).toContain('推理过程');
    expect(html).not.toContain('TTFT / Tokens Per Second');
    expect(html).not.toContain('TTFT');
    expect(html).not.toContain('tok/s');
    expect(html).not.toContain('Inference Simulation');
    expect(html).not.toContain('TTFT (est.)');
    expect(html).not.toContain('Tok/s (est.)');
    expect(html).not.toContain('Press Play to simulate inference');
  });

  it('calculates Total VRAM from per-GPU required bytes and GPU count', () => {
    expect(totalRequiredBytes({ required_bytes_per_gpu_at_tier: 42_620_000_000, gpu_count: 2 })).toBe(85_240_000_000);
  });

  it('filters throughput-only sections from the displayed explain text', () => {
    const filtered = memoryExplainText(
      `完整推导链（--explain）

Weight bytes (safetensors file sum)
结果: 61.07 GB

KV cache @ 4K context
结果: 0.30 GB

KV cache @ 40K context
结果: 3.02 GB

Fleet tier: min (1 GPUs)
结果: 1 GPUs, fit=true

Fleet tier: dev (2 GPUs)
结果: 2 GPUs, fit=true

Prefill latency (single request)
结果: 17.0 ms

Decode throughput (cluster)
结果: 899.2 tok/s

L bound (compute/bandwidth at SLA)
target_per_user_tok_per_sec = 30

Max concurrent + bottleneck verdict
结果: 27 concurrent
`,
      { tier: 'dev', gpuCount: 2, contextTokens: 40960 },
    );

    expect(filtered).toContain('Weight bytes');
    expect(filtered).toContain('KV cache @ 40K context');
    expect(filtered).toContain('Fleet tier: dev (2 GPUs)');
    expect(filtered).not.toContain('KV cache @ 4K context');
    expect(filtered).not.toContain('Fleet tier: min (1 GPUs)');
    expect(filtered).not.toContain('K bound');
    expect(filtered).not.toContain('Prefill latency');
    expect(filtered).not.toContain('Decode throughput');
    expect(filtered).not.toContain('tok/s');
    expect(filtered).not.toContain('target_per_user_tok_per_sec');
  });

  it('treats report data as stale after the selected model changes', () => {
    const report = {
      model: { id: 'Qwen/Qwen3-30B-A3B', source: 'builtin' },
      engine: 'vllm',
      hardware: { id: 'H100' },
    };
    const form = {
      model_id: 'deepseek-ai/DeepSeek-V3',
      source: 'builtin' as const,
      engine: 'vllm' as const,
      gpu: 'H100',
      gpus: ['H100'],
    };

    expect(isReportCurrent(report, form, 'H100')).toBe(false);
  });
});
