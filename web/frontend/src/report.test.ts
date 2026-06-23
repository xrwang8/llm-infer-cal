import { describe, expect, it } from 'vitest';
import {
  advancedSettings,
  bestFleetOption,
  buildEvaluatePayload,
  defaultRemoteModelId,
  formatBytes,
  formatNumber,
  groupGpusByVendor,
  groupModelsByProvider,
  gpuVendorOptionLabel,
  itemsForGroup,
  llmReviewSettings,
  modelVendorOptionLabel,
  performanceSettings,
} from './report';

describe('report helpers', () => {
  it('formats bytes using decimal units', () => {
    expect(formatBytes(61_066_575_648)).toBe('61.07 GB');
    expect(formatBytes(402_653_184)).toBe('402.65 MB');
  });

  it('formats large counters compactly', () => {
    expect(formatNumber(30_532_108_288)).toBe('30.53B');
    expect(formatNumber(151_936)).toBe('151,936');
  });

  it('selects the fleet option matching best_tier and fit status', () => {
    const option = bestFleetOption({
      best_tier: 'dev',
      options: [
        { tier: 'min', fits: true, gpu_count: 1 },
        { tier: 'dev', fits: true, gpu_count: 2 },
        { tier: 'prod', fits: true, gpu_count: 4 },
      ],
    });

    expect(option?.gpu_count).toBe(2);
  });

  it('groups builtin models by repository owner for the selector', () => {
    const groups = groupModelsByProvider([
      { id: 'Qwen/Qwen3-30B-A3B', provider: 'Qwen' },
      { id: 'Qwen/Qwen2.5-32B', provider: 'Qwen' },
      { id: 'deepseek-ai/DeepSeek-V3', provider: 'DeepSeek' },
      { id: 'deepseek-ai/DeepSeek-V3-0324', provider: 'deepseek-ai' },
      { id: 'zai-org/GLM-5.2-FP8', provider: 'zai-org' },
      { id: 'moonshotai/Kimi-K2-Instruct', provider: 'Moonshot AI' },
      { id: 'nvidia/Qwen3.6-35B-A3B-NVFP4', provider: 'Qwen' },
    ]);

    expect(groups.map((group) => group.label)).toEqual(['DeepSeek', 'GLM (Z-AI)', 'Moonshot (Kimi)', 'Qwen']);
    expect(groups.find((group) => group.label === 'DeepSeek')?.items).toHaveLength(2);
    expect(itemsForGroup(groups, 'Qwen').map((model) => model.id)).toEqual(['Qwen/Qwen2.5-32B', 'Qwen/Qwen3-30B-A3B']);
    expect(itemsForGroup(groups, 'NVIDIA')).toEqual([]);
  });

  it('renders model vendor options without item counts', () => {
    expect(modelVendorOptionLabel({ label: 'Qwen', items: [{ id: 'Qwen/Qwen3-30B-A3B' }, { id: 'Qwen/Qwen2.5-32B' }] })).toBe('Qwen');
  });

  it('groups GPUs by vendor for the selector', () => {
    const groups = groupGpusByVendor([
      { id: 'H100', vendor: 'NVIDIA', memory_gb: 80 },
      { id: 'A100-80G', vendor: 'NVIDIA', memory_gb: 80 },
      { id: 'MI300X', vendor: 'AMD', memory_gb: 192 },
      { id: '910B4', vendor: 'Huawei Ascend', memory_gb: 32 },
      { id: 'BR100', vendor: 'Biren 壁仞', memory_gb: 64 },
    ]);

    expect(groups.map((group) => group.label)).toEqual(['NVIDIA', 'AMD', 'Huawei Ascend', 'Biren 壁仞']);
    expect(itemsForGroup(groups, 'NVIDIA').map((gpu) => gpu.id)).toEqual(['A100-80G', 'H100']);
  });

  it('renders GPU vendor options without item counts', () => {
    expect(gpuVendorOptionLabel({ label: 'NVIDIA', items: [{ id: 'H100' }, { id: 'A100-80G' }] })).toBe('NVIDIA');
  });

  it('clears model id when switching to remote sources', () => {
    expect(defaultRemoteModelId()).toBe('');
  });

  it('keeps performance tuning fields in a collapsed settings section', () => {
    const settings = performanceSettings();

    expect(settings.map((setting) => setting.label)).toEqual([
      '输入 tokens',
      '输出 tokens',
      '目标 tok/s',
      'Prefill 利用率',
      'Decode 带宽利用率',
      '并发衰减',
    ]);
    expect(settings.map((setting) => setting.key)).toEqual([
      'input_tokens',
      'output_tokens',
      'target_tokens_per_sec',
      'prefill_utilization',
      'decode_bw_utilization',
      'concurrency_degradation',
    ]);
    expect(settings.every((setting) => setting.collapsedByDefault)).toBe(true);
  });

  it('hides explain and refresh from advanced settings', () => {
    const settings = advancedSettings();

    expect(settings.map((setting) => setting.label)).toEqual(['强制 GPU 数', 'LLM 审计（--llm-review）']);
    expect(settings.map((setting) => setting.key)).toEqual(['gpu_count', 'llm_review']);
    expect(settings.every((setting) => setting.collapsedByDefault)).toBe(true);
  });

  it('shows LLM reviewer configuration only as LLM review details', () => {
    const settings = llmReviewSettings();

    expect(settings.map((setting) => setting.label)).toEqual(['LLM API 密钥', 'LLM 基地址', 'LLM 模型名']);
    expect(settings.map((setting) => setting.key)).toEqual(['llm_review_api_key', 'llm_review_base_url', 'llm_review_model']);
    expect(settings.every((setting) => setting.visibleWhen === 'llm_review')).toBe(true);
  });

  it('passes supported advanced settings to the evaluate payload', () => {
    expect(
      buildEvaluatePayload({
        model_id: 'Qwen/Qwen3-30B-A3B',
        source: 'builtin',
        gpu: 'H100',
        gpus: ['H100'],
        engine: 'vllm',
        gpu_count: '2',
        context_length: '',
        input_tokens: '2000',
        output_tokens: '512',
        target_tokens_per_sec: '30',
        prefill_utilization: '0.4',
        decode_bw_utilization: '0.5',
        concurrency_degradation: '1.67',
        llm_review: false,
        llm_review_api_key: '',
        llm_review_base_url: '',
        llm_review_model: '',
      }),
    ).toMatchObject({
      gpu_count: 2,
      concurrency_degradation: 1.67,
      explain: true,
      llm_review: false,
    });
    expect(
      buildEvaluatePayload({
        model_id: 'Qwen/Qwen3-30B-A3B',
        source: 'builtin',
        gpu: 'H100',
        gpus: ['H100'],
        engine: 'vllm',
        gpu_count: '2',
        context_length: '',
        input_tokens: '2000',
        output_tokens: '512',
        target_tokens_per_sec: '30',
        prefill_utilization: '0.4',
        decode_bw_utilization: '0.5',
        concurrency_degradation: '1.67',
        llm_review: false,
        llm_review_api_key: '',
        llm_review_base_url: '',
        llm_review_model: '',
      }),
    ).not.toHaveProperty('refresh');
  });

  it('passes trimmed LLM reviewer settings to the evaluate payload', () => {
    expect(
      buildEvaluatePayload({
        model_id: 'Qwen/Qwen3-30B-A3B',
        source: 'builtin',
        gpu: 'H100',
        gpus: ['H100'],
        engine: 'vllm',
        gpu_count: '',
        context_length: '',
        input_tokens: '2000',
        output_tokens: '512',
        target_tokens_per_sec: '30',
        prefill_utilization: '0.4',
        decode_bw_utilization: '0.5',
        concurrency_degradation: '1',
        llm_review: true,
        llm_review_api_key: ' sk-test ',
        llm_review_base_url: ' https://api.deepseek.com/v1/ ',
        llm_review_model: ' deepseek-chat ',
      }),
    ).toMatchObject({
      llm_review: true,
      llm_review_api_key: 'sk-test',
      llm_review_base_url: 'https://api.deepseek.com/v1/',
      llm_review_model: 'deepseek-chat',
    });
  });

  it('passes up to four selected GPUs for comparison while keeping the first GPU for compatibility', () => {
    expect(
      buildEvaluatePayload({
        model_id: 'Qwen/Qwen3-30B-A3B',
        source: 'builtin',
        gpu: 'H100',
        gpus: ['H100', 'A100-80G', 'H800', 'H200', 'L40S'],
        engine: 'vllm',
        gpu_count: '',
        context_length: '',
        input_tokens: '2000',
        output_tokens: '512',
        target_tokens_per_sec: '30',
        prefill_utilization: '0.4',
        decode_bw_utilization: '0.5',
        concurrency_degradation: '1',
        llm_review: false,
        llm_review_api_key: '',
        llm_review_base_url: '',
        llm_review_model: '',
      }),
    ).toMatchObject({
      gpu: 'H100',
      gpus: ['H100', 'A100-80G', 'H800', 'H200'],
    });
  });
});
