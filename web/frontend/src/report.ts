export type FleetOption = {
  tier?: string;
  fits?: boolean;
  gpu_count?: number;
  node_count?: number;
  tensor_parallel_size?: number;
  pipeline_parallel_size?: number;
  weight_bytes_per_gpu?: number;
  usable_bytes_per_gpu?: number;
  kv_bytes_per_request?: number;
  kv_reference_context_tokens?: number;
  max_concurrent_at_reference_ctx?: number;
  reason_zh?: string;
  reason_en?: string;
};

export type Fleet = {
  best_tier?: string;
  constraint_note_zh?: string;
  constraint_note_en?: string;
  options?: FleetOption[];
};

export type Annotated<T> = {
  value?: T;
  label?: string;
  source?: string | null;
};

export type ModelSummary = {
  id: string;
  aliases?: string[];
  provider?: string | null;
  preferred_source?: string | null;
  mentioned_by?: string[];
};

export type GpuSummary = {
  id: string;
  vendor?: string | null;
  vendor_zh?: string | null;
  aliases?: string[];
  memory_gb?: number;
  nvlink_bandwidth_gbps?: number;
  memory_bandwidth_gbps?: number | null;
  fp16_tflops?: number;
  fp8_support?: boolean;
  fp4_support?: boolean;
  notes_zh?: string | null;
  notes_en?: string | null;
  spec_source?: string | null;
};

export type Report = {
  schema_version?: string;
  model?: {
    id?: string;
    source?: string;
    commit_sha?: string | null;
  };
  engine?: string;
  architecture?: Record<string, any>;
  weights?: Record<string, any>;
  kv_cache_by_context?: Array<{
    context_tokens?: number;
    bytes?: Annotated<number>;
  }>;
  engine_compatibility?: Record<string, any> | null;
  hardware?: GpuSummary | null;
  fleet?: Fleet | null;
  performance?: Record<string, any> | null;
  generated_command?: {
    command?: string;
    lines?: string[];
    gpu_count?: number;
    tier?: string;
    tensor_parallel_size?: number;
    pipeline_parallel_size?: number;
    node_count?: number;
  } | null;
  explain_text?: string | null;
  llm_review_text?: string | null;
  comparison?: {
    reports?: Report[];
  } | null;
};

export type EvaluateForm = {
  model_id: string;
  source: 'builtin' | 'huggingface' | 'modelscope';
  gpu: string;
  gpus: string[];
  engine: 'vllm' | 'sglang';
  gpu_count: string;
  context_length: string;
  input_tokens: string;
  output_tokens: string;
  target_tokens_per_sec: string;
  prefill_utilization: string;
  decode_bw_utilization: string;
  concurrency_degradation: string;
  llm_review: boolean;
  llm_review_api_key: string;
  llm_review_base_url: string;
  llm_review_model: string;
};

export type Group<T> = {
  label: string;
  items: T[];
};

type PerformanceSettingKey =
  | 'input_tokens'
  | 'output_tokens'
  | 'target_tokens_per_sec'
  | 'prefill_utilization'
  | 'decode_bw_utilization'
  | 'concurrency_degradation';

type AdvancedNumberSettingKey = 'gpu_count';
type AdvancedBooleanSettingKey = 'llm_review';
type LlmReviewSettingKey = 'llm_review_api_key' | 'llm_review_base_url' | 'llm_review_model';

export type PerformanceSetting = {
  key: PerformanceSettingKey;
  label: string;
  control: 'number' | 'range';
  min?: string;
  max?: string;
  step?: string;
  collapsedByDefault: true;
};

export type AdvancedSetting =
  | {
      key: AdvancedNumberSettingKey;
      label: string;
      control: 'number';
      collapsedByDefault: true;
    }
  | {
      key: AdvancedBooleanSettingKey;
      label: string;
      control: 'checkbox';
      collapsedByDefault: true;
    };

export type LlmReviewSetting = {
  key: LlmReviewSettingKey;
  label: string;
  placeholder: string;
  type?: 'password' | 'text';
  visibleWhen: 'llm_review';
};

const GPU_VENDOR_ORDER = [
  'NVIDIA',
  'AMD',
  'Intel Habana',
  'Huawei Ascend',
  'MetaX 沐曦',
  'Kunlunxin 昆仑芯',
  'Biren 壁仞',
  'Iluvatar 天数智芯',
  'Moore Threads 摩尔线程',
  'Cambricon 寒武纪',
  'Hygon 海光',
  'Other',
];

const PERFORMANCE_SETTINGS: PerformanceSetting[] = [
  { key: 'input_tokens', label: '输入 tokens', control: 'number', collapsedByDefault: true },
  { key: 'output_tokens', label: '输出 tokens', control: 'number', collapsedByDefault: true },
  { key: 'target_tokens_per_sec', label: '目标 tok/s', control: 'number', collapsedByDefault: true },
  { key: 'prefill_utilization', label: 'Prefill 利用率', control: 'range', min: '0.1', max: '1', step: '0.05', collapsedByDefault: true },
  { key: 'decode_bw_utilization', label: 'Decode 带宽利用率', control: 'range', min: '0.1', max: '1', step: '0.05', collapsedByDefault: true },
  { key: 'concurrency_degradation', label: '并发衰减', control: 'number', collapsedByDefault: true },
];

const ADVANCED_SETTINGS: AdvancedSetting[] = [
  { key: 'gpu_count', label: '强制 GPU 数', control: 'number', collapsedByDefault: true },
  { key: 'llm_review', label: 'LLM 审计（--llm-review）', control: 'checkbox', collapsedByDefault: true },
];

const LLM_REVIEW_SETTINGS: LlmReviewSetting[] = [
  {
    key: 'llm_review_api_key',
    label: 'LLM API 密钥',
    placeholder: 'sk-...',
    type: 'password',
    visibleWhen: 'llm_review',
  },
  {
    key: 'llm_review_base_url',
    label: 'LLM 基地址',
    placeholder: 'https://api.openai.com/v1',
    visibleWhen: 'llm_review',
  },
  {
    key: 'llm_review_model',
    label: 'LLM 模型名',
    placeholder: 'gpt-4o / deepseek-chat / MiniMax-M2',
    visibleWhen: 'llm_review',
  },
];

export function performanceSettings(): PerformanceSetting[] {
  return PERFORMANCE_SETTINGS;
}

export function advancedSettings(): AdvancedSetting[] {
  return ADVANCED_SETTINGS;
}

export function llmReviewSettings(): LlmReviewSetting[] {
  return LLM_REVIEW_SETTINGS;
}

export function groupModelsByProvider(models: ModelSummary[]): Group<ModelSummary>[] {
  return groupBy(
    models,
    (model) => modelVendorLabel(model),
    (a, b) => compareWithOtherLast(a.label, b.label),
    (a, b) => a.id.localeCompare(b.id),
  ).filter((group) => group.label !== 'NVIDIA');
}

export function groupGpusByVendor(gpus: GpuSummary[]): Group<GpuSummary>[] {
  return groupBy(gpus, (gpu) => gpu.vendor?.trim() || '其他', (a, b) => {
    const ai = GPU_VENDOR_ORDER.indexOf(a.label);
    const bi = GPU_VENDOR_ORDER.indexOf(b.label);
    if (ai !== -1 || bi !== -1) {
      return (ai === -1 ? Number.MAX_SAFE_INTEGER : ai) - (bi === -1 ? Number.MAX_SAFE_INTEGER : bi);
    }
    return compareWithOtherLast(a.label, b.label);
  }, (a, b) => a.id.localeCompare(b.id));
}

export function itemsForGroup<T>(groups: Group<T>[], label: string): T[] {
  return groups.find((group) => group.label === label)?.items ?? [];
}

export function modelVendorOptionLabel(group: Group<ModelSummary>): string {
  return group.label;
}

export function gpuVendorOptionLabel(group: Group<GpuSummary>): string {
  return group.label;
}

export function defaultRemoteModelId(): string {
  return '';
}

function groupBy<T>(items: T[], keyFor: (item: T) => string, sortGroups: (a: Group<T>, b: Group<T>) => number, sortItems: (a: T, b: T) => number): Group<T>[] {
  const byKey = new Map<string, T[]>();
  for (const item of items) {
    const key = keyFor(item);
    byKey.set(key, [...(byKey.get(key) ?? []), item]);
  }
  return [...byKey.entries()]
    .map(([label, groupedItems]) => ({ label, items: [...groupedItems].sort(sortItems) }))
    .sort(sortGroups);
}

function compareWithOtherLast(a: string, b: string): number {
  if (a === '其他' && b !== '其他') return 1;
  if (b === '其他' && a !== '其他') return -1;
  return a.localeCompare(b, 'zh-Hans-CN');
}

function modelVendorLabel(model: ModelSummary): string {
  const owner = model.id.split('/')[0]?.trim();
  return normalizeModelVendor(owner || model.provider);
}

function normalizeModelVendor(vendor: string | null | undefined): string {
  const raw = vendor?.trim();
  if (!raw) return '其他';
  const lower = raw.toLowerCase();
  const aliases: Record<string, string> = {
    'deepseek-ai': 'DeepSeek',
    deepseek: 'DeepSeek',
    moonshotai: 'Moonshot (Kimi)',
    minimaxai: 'MiniMax',
    'zai-org': 'GLM (Z-AI)',
    zhipuai: 'GLM (Z-AI)',
    nvidia: 'NVIDIA',
    qwen: 'Qwen',
  };
  return aliases[lower] ?? raw;
}

export function formatBytes(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return '-';
  }
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)} GB`;
  }
  if (abs >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)} MB`;
  }
  if (abs >= 1_000) {
    return `${(value / 1_000).toFixed(2)} KB`;
  }
  return `${value.toLocaleString()} B`;
}

export function formatNumber(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return '-';
  }
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)}B`;
  }
  if (abs >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  return Math.round(value).toLocaleString();
}

export function formatFloat(value: number | null | undefined, digits = 1): string {
  if (value == null || Number.isNaN(value)) {
    return '-';
  }
  return value.toFixed(digits);
}

export function formatMs(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return '-';
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)} s`;
  }
  return `${value.toFixed(1)} ms`;
}

export function bestFleetOption(fleet: Fleet | null | undefined): FleetOption | undefined {
  if (!fleet?.options?.length) {
    return undefined;
  }
  const bestTier = fleet.best_tier;
  return (
    fleet.options.find((option) => option.fits && option.tier === bestTier) ??
    fleet.options.find((option) => option.fits) ??
    fleet.options[fleet.options.length - 1]
  );
}

export function annotatedValue<T>(value: Annotated<T> | null | undefined): T | undefined {
  return value?.value;
}

export function labelText(label: string | null | undefined): string {
  const labels: Record<string, string> = {
    verified: '已验证',
    inferred: '推断',
    estimated: '估算',
    cited: '有出处',
    unverified: '未验证',
    unknown: '未知',
    'llm-opinion': 'LLM 意见',
  };
  return label ? labels[label] ?? label : '未知';
}

export function buildEvaluatePayload(form: EvaluateForm) {
  const numberOrUndefined = (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) {
      return undefined;
    }
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : undefined;
  };
  const stringOrUndefined = (value: string) => {
    const trimmed = value.trim();
    return trimmed || undefined;
  };

  return {
    model_id: form.model_id.trim(),
    source: form.source,
    gpu: selectedGpus(form)[0] ?? '',
    gpus: selectedGpus(form),
    engine: form.engine,
    gpu_count: numberOrUndefined(form.gpu_count),
    context_length: numberOrUndefined(form.context_length),
    input_tokens: numberOrUndefined(form.input_tokens),
    output_tokens: numberOrUndefined(form.output_tokens),
    target_tokens_per_sec: numberOrUndefined(form.target_tokens_per_sec),
    prefill_utilization: numberOrUndefined(form.prefill_utilization),
    decode_bw_utilization: numberOrUndefined(form.decode_bw_utilization),
    concurrency_degradation: numberOrUndefined(form.concurrency_degradation),
    explain: true,
    llm_review: form.llm_review,
    llm_review_api_key: form.llm_review ? stringOrUndefined(form.llm_review_api_key) : undefined,
    llm_review_base_url: form.llm_review ? stringOrUndefined(form.llm_review_base_url) : undefined,
    llm_review_model: form.llm_review ? stringOrUndefined(form.llm_review_model) : undefined,
    lang: 'zh',
  };
}

function selectedGpus(form: EvaluateForm): string[] {
  const raw = form.gpus.length ? form.gpus : [form.gpu];
  const gpus: string[] = [];
  for (const gpu of raw) {
    const trimmed = gpu.trim();
    if (trimmed && !gpus.includes(trimmed)) {
      gpus.push(trimmed);
    }
  }
  return gpus.slice(0, 4);
}

export function tierText(tier: string | null | undefined): string {
  if (tier === 'min') return '最小';
  if (tier === 'dev') return '开发';
  if (tier === 'prod') return '生产';
  return tier ?? '-';
}

export function pct(numerator: number, denominator: number): number {
  if (!denominator || denominator <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(100, (numerator / denominator) * 100));
}
