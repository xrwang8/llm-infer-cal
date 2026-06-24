import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { App, isReportCurrent } from './App';

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
  });

  it('renders reference-inspired calculator sections', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).toContain('llm-infer-cal');
    expect(html).toContain('GPU Memory &amp; Performance Estimator');
    expect(html).toContain('Configuration');
    expect(html).toContain('Estimates');
    expect(html).toContain('VRAM Breakdown');
    expect(html).toContain('Formula Reference');
    expect(html).toContain('Inference Optimizations');
    expect(html).toContain('Compare GPUs');
    expect(html).not.toContain('LLM VRAM Calculator');
    expect(html).not.toContain('v4.0');
    expect(html).not.toContain('v0 calculator');
    expect(html).not.toContain('Shun-Calvin/llm-vram-calculator');
  });

  it('renders speculative decoding as a selectable workflow instead of raw text inputs', () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).toContain('Speculative Decoding');
    expect(html).toContain('Draft / MTP model predicts, main model verifies');
    expect(html).toContain('Reduces KV cache overhead by ~25%');
    expect(html).not.toContain('Draft/EAGLE 模型 ID');
    expect(html).not.toContain('Speculative 额外权重 GiB');
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
