import {
  Activity,
  AlertTriangle,
  Calculator,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clipboard,
  Cpu,
  Database,
  GitCompare,
  Github,
  HardDrive,
  Info,
  Layers,
  LayoutGrid,
  Play,
  RefreshCcw,
  Search,
  Terminal,
  X,
  Zap,
} from 'lucide-react';
import { FormEvent, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { evaluate, fetchGpus, fetchModels } from './api';
import {
  advancedSettings,
  annotatedValue,
  bestFleetOption,
  defaultRemoteModelId,
  formatBytes,
  formatFloat,
  formatNumber,
  groupGpusByVendor,
  groupModelsByProvider,
  gpuTier,
  gpuTierLabel,
  itemsForGroup,
  labelText,
  llmReviewSettings,
  modelVendorOptionLabel,
  pct,
  tierText,
  type EvaluateForm,
  type FleetOption,
  type GpuSummary,
  type ModelSummary,
  type Report,
} from './report';

const DEFAULT_SPECULATIVE_MODE = 'mtp' as const;
const DEFAULT_MTP_EXTRA_WEIGHT_GB = '0.3';

const DEFAULT_FORM: EvaluateForm = {
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
  kv_cache_bits: '16',
  paged_attention: true,
  speculative_enabled: false,
  speculative_mode: DEFAULT_SPECULATIVE_MODE,
  speculative_num_draft_tokens: '4',
  target_concurrent_requests: '',
  speculative_draft_model_id: '',
  speculative_extra_weight_gb: '',
  expert_offloading: false,
  experts_on_gpu: '',
  cpu_offload_gb: '',
  llm_review: false,
  llm_review_api_key: '',
  llm_review_base_url: '',
  llm_review_model: '',
};

export function defaultSpeculativeFormPatch(numDraftTokens: number) {
  return {
    speculative_enabled: true,
    speculative_mode: DEFAULT_SPECULATIVE_MODE,
    speculative_num_draft_tokens: String(Math.round(numDraftTokens)),
    speculative_draft_model_id: '',
    speculative_extra_weight_gb: DEFAULT_MTP_EXTRA_WEIGHT_GB,
  };
}

const sourceOptions = [
  { value: 'builtin', label: '内置' },
  { value: 'huggingface', label: 'HuggingFace' },
  { value: 'modelscope', label: 'ModelScope' },
] as const;

const engineOptions = [
  { value: 'vllm', label: 'vLLM' },
  { value: 'sglang', label: 'SGLang' },
] as const;

const kvPrecisionOptions = [
  { value: '16', label: 'FP16/BF16', sublabel: '默认 KV cache 精度' },
  { value: '8', label: 'FP8/INT8', sublabel: '减少约 50% KV cache' },
  { value: '4', label: 'INT4', sublabel: '实验性低精度 KV cache' },
] as const;

type SpeculativeMode = 'standard' | 'mtp';

const gpuCountOptions = ['', '1', '2', '4', '8', '16', '32', '64'];

type SelectOption = {
  value: string;
  label: string;
  sublabel?: string;
  badge?: ReactNode;
  disabled?: boolean;
};

export function isReportCurrent(
  report: Pick<Report, 'model' | 'engine' | 'hardware'> | null | undefined,
  form: Pick<EvaluateForm, 'model_id' | 'source' | 'engine' | 'gpu' | 'gpus'>,
  selectedGpuId?: string,
) {
  if (!report) return false;

  const expectedModelId = form.model_id.trim();
  if (expectedModelId && report.model?.id !== expectedModelId) return false;
  if (report.model?.source && report.model.source !== form.source) return false;
  if (report.engine && report.engine !== form.engine) return false;

  const expectedGpuId = (selectedGpuId || form.gpus[0] || form.gpu || '').trim();
  const reportGpu = report.hardware;
  if (expectedGpuId && reportGpu?.id && reportGpu.id !== expectedGpuId && !reportGpu.aliases?.includes(expectedGpuId)) {
    return false;
  }

  return true;
}

export function App() {
  const [form, setForm] = useState<EvaluateForm>(DEFAULT_FORM);
  const [activeTab, setActiveTab] = useState<'calculator' | 'compare'>('calculator');
  const [speculativeEnabled, setSpeculativeEnabled] = useState(false);
  const [speculativeMode, setSpeculativeMode] = useState<SpeculativeMode>(DEFAULT_SPECULATIVE_MODE);
  const [specDraftModelSize, setSpecDraftModelSize] = useState(0.5);
  const [specNumDraftTokens, setSpecNumDraftTokens] = useState(4);
  const [expertOffloading, setExpertOffloading] = useState(false);
  const [numGpuExperts, setNumGpuExperts] = useState(8);
  const [modelVendor, setModelVendor] = useState('Qwen');
  const [gpuVendor, setGpuVendor] = useState('NVIDIA');
  const [compareVendor, setCompareVendor] = useState('all');
  const [compareTier, setCompareTier] = useState('all');
  const [compareGpuCount, setCompareGpuCount] = useState('');
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [gpus, setGpus] = useState<GpuSummary[]>([]);
  const [report, setReport] = useState<Report | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let mounted = true;

    Promise.all([fetchModels(), fetchGpus()])
      .then(([nextModels, nextGpus]) => {
        if (!mounted) return;
        setModels(nextModels);
        setGpus(nextGpus);
      })
      .catch((nextError) => {
        if (!mounted) return;
        setError(nextError instanceof Error ? nextError.message : String(nextError));
      });

    void runEvaluation(DEFAULT_FORM);

    return () => {
      mounted = false;
    };
  }, []);

  const selectedGpuIds = form.gpus.length ? form.gpus : form.gpu ? [form.gpu] : [];
  const selectedGpuId = selectedGpuIds[0] ?? '';
  const selectedGpu = useMemo(
    () => gpus.find((gpu) => gpu.id === selectedGpuId || gpu.aliases?.includes(selectedGpuId)),
    [selectedGpuId, gpus],
  );
  const modelGroups = useMemo(() => groupModelsByProvider(models), [models]);
  const gpuGroups = useMemo(() => groupGpusByVendor(gpus), [gpus]);
  const modelOptions = useMemo(() => itemsForGroup(modelGroups, modelVendor), [modelGroups, modelVendor]);
  const gpuOptions = useMemo(() => itemsForGroup(gpuGroups, gpuVendor), [gpuGroups, gpuVendor]);
  const visibleReport = isReportCurrent(report, form, selectedGpuId) ? report : null;
  const selectedFleet = bestFleetOption(visibleReport?.fleet);
  const command = visibleReport?.generated_command?.command ?? '';
  const comparisonReports = visibleReport?.comparison?.reports ?? [];
  const kvRows = visibleReport?.kv_cache_by_context ?? [];
  const activationRows = visibleReport?.activation_by_context ?? [];
  const weightBytes = annotatedValue<number>(visibleReport?.weights?.safetensors_total_bytes);
  const params = annotatedValue<number>(visibleReport?.weights?.params_estimated);
  const activeParams = annotatedValue<number>(visibleReport?.weights?.active_params_estimated);
  const engineSupport = visibleReport?.engine_compatibility?.support ?? 'unknown';
  const advancedControls = advancedSettings();
  const reviewerSettings = llmReviewSettings();
  const selectedModelLabel = (visibleReport?.model?.id ?? form.model_id) || '选择模型';
  const moeTraits = visibleReport?.architecture?.moe;
  const isMoe = !!moeTraits || (!!params && !!activeParams && activeParams < params);
  const totalExperts = Number(moeTraits?.routed_experts ?? moeTraits?.num_routed_experts ?? 8);
  const activeExperts = Number(moeTraits?.experts_per_token ?? moeTraits?.num_experts_per_tok ?? 1);
  const requiresDraftModel = form.speculative_enabled && form.speculative_mode === 'standard' && !form.speculative_draft_model_id.trim();
  const compareGpuIds = useMemo(() => {
    const filtered = gpus.filter((gpu) => {
      const vendorOk = compareVendor === 'all' || gpu.vendor === compareVendor;
      const tierOk = compareTier === 'all' || gpuTier(gpu) === compareTier;
      return vendorOk && tierOk;
    });
    return filtered.map((gpu) => gpu.id).slice(0, 64);
  }, [gpus, compareVendor, compareTier]);

  useEffect(() => {
    if (!isMoe) {
      setExpertOffloading(false);
      setForm((current) => ({ ...current, expert_offloading: false, experts_on_gpu: '' }));
      return;
    }
    setNumGpuExperts((current) => {
      const next = Math.max(activeExperts || 1, Math.min(current || activeExperts || 1, totalExperts || current || 8));
      setForm((formState) => (formState.expert_offloading ? { ...formState, experts_on_gpu: String(next) } : formState));
      return next;
    });
  }, [isMoe, activeExperts, totalExperts]);

  async function runEvaluation(nextForm = form) {
    setEvaluating(true);
    setError(null);
    try {
      const nextReport = await evaluate(nextForm);
      setReport(nextReport);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError));
    } finally {
      setEvaluating(false);
    }
  }

  function updateField<K extends keyof EvaluateForm>(key: K, value: EvaluateForm[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function updateSource(value: EvaluateForm['source']) {
    setForm((current) => {
      if (value !== 'builtin') {
        return { ...current, source: value, model_id: defaultRemoteModelId() };
      }

      const currentModelStillVisible = modelOptions.some((model) => model.id === current.model_id);
      return {
        ...current,
        source: value,
        model_id: currentModelStillVisible ? current.model_id : modelOptions[0]?.id ?? '',
      };
    });
  }

  function updateModelVendor(value: string) {
    const nextModels = itemsForGroup(modelGroups, value);
    setModelVendor(value);
    setForm((current) => ({ ...current, source: 'builtin', model_id: nextModels[0]?.id ?? '' }));
  }

  function updateGpuVendor(value: string) {
    const nextGpus = itemsForGroup(gpuGroups, value);
    const firstGpu = nextGpus[0]?.id;
    setGpuVendor(value);
    if (firstGpu) {
      setForm((current) => ({ ...current, gpu: firstGpu, gpus: [firstGpu] }));
    }
  }

  function updatePrimaryGpu(gpuId: string) {
    setForm((current) => ({ ...current, gpu: gpuId, gpus: gpuId ? [gpuId] : [] }));
  }

  function updateSpeculativeEnabled(enabled: boolean) {
    setSpeculativeEnabled(enabled);
    if (!enabled) {
      setForm((current) => ({
        ...current,
        speculative_enabled: false,
        speculative_mode: DEFAULT_SPECULATIVE_MODE,
        speculative_num_draft_tokens: '4',
        speculative_draft_model_id: '',
        speculative_extra_weight_gb: '',
      }));
      return;
    }

    setSpeculativeMode(DEFAULT_SPECULATIVE_MODE);
    setForm((current) => ({
      ...current,
      ...defaultSpeculativeFormPatch(specNumDraftTokens),
    }));
  }

  function updateSpeculativeMode(nextMode: SpeculativeMode) {
    setSpeculativeMode(nextMode);
    setForm((current) => {
      return {
        ...current,
        speculative_mode: nextMode,
        speculative_draft_model_id: nextMode === 'mtp' ? '' : current.speculative_draft_model_id,
        speculative_extra_weight_gb: nextMode === 'mtp' ? DEFAULT_MTP_EXTRA_WEIGHT_GB : draftModelWeightGb(specDraftModelSize),
      };
    });
  }

  function updateSpecDraftModelSize(value: number) {
    setSpecDraftModelSize(value);
    if (speculativeEnabled && speculativeMode === 'standard') {
      setForm((current) => ({ ...current, speculative_extra_weight_gb: draftModelWeightGb(value) }));
    }
  }

  function updateSpecNumDraftTokens(value: number) {
    setSpecNumDraftTokens(value);
    setForm((current) => ({ ...current, speculative_num_draft_tokens: String(Math.round(value)) }));
  }

  function updateExpertOffloading(enabled: boolean) {
    setExpertOffloading(enabled);
    setForm((current) => ({
      ...current,
      expert_offloading: enabled,
      experts_on_gpu: enabled ? String(Math.max(activeExperts || 1, Math.min(numGpuExperts || activeExperts || 1, totalExperts || numGpuExperts || 8))) : '',
    }));
  }

  function updateExpertsOnGpu(value: number) {
    const rounded = Math.round(value);
    setNumGpuExperts(rounded);
    setForm((current) => ({ ...current, experts_on_gpu: String(rounded) }));
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (activeTab === 'compare') {
      void runComparison();
      return;
    }
    void runEvaluation({ ...form, gpus: selectedGpuId ? [selectedGpuId] : form.gpus });
  }

  async function runComparison() {
    const gpusForCompare = compareGpuIds.length ? compareGpuIds : selectedGpuIds;
    const nextForm: EvaluateForm = {
      ...form,
      gpu: gpusForCompare[0] ?? form.gpu,
      gpus: gpusForCompare,
      gpu_count: compareGpuCount,
    };
    await runEvaluation(nextForm);
  }

  function resetForm() {
    setForm(DEFAULT_FORM);
    setSpeculativeEnabled(false);
    setSpeculativeMode(DEFAULT_SPECULATIVE_MODE);
    setSpecDraftModelSize(0.5);
    setSpecNumDraftTokens(4);
    setExpertOffloading(false);
    setNumGpuExperts(8);
    setModelVendor('Qwen');
    setGpuVendor('NVIDIA');
    void runEvaluation(DEFAULT_FORM);
  }

  function copyCommand() {
    if (!command) return;
    void navigator.clipboard.writeText(command).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    });
  }

  const providerOptions: SelectOption[] = gpuGroups.length
    ? gpuGroups.map((group) => ({ value: group.label, label: group.label }))
    : [{ value: gpuVendor, label: gpuVendor }];
  const modelVendorOptions: SelectOption[] = modelGroups.length
    ? modelGroups.map((group) => ({ value: group.label, label: modelVendorOptionLabel(group) }))
    : [{ value: modelVendor, label: modelVendor }];
  const builtinModelOptions: SelectOption[] = modelOptions.map((model) => ({
    value: model.id,
    label: model.id,
    sublabel: model.mentioned_by?.length ? model.mentioned_by.join(' + ') : model.preferred_source ?? undefined,
    badge: model.preferred_source ? <span className="miniBadge">{model.preferred_source}</span> : undefined,
  }));
  const gpuModelOptions: SelectOption[] = (gpuOptions.length ? gpuOptions : selectedGpu ? [selectedGpu] : []).map((gpu) => ({
    value: gpu.id,
    label: gpu.id,
    sublabel: gpu.memory_gb
      ? `${gpu.memory_gb} GB VRAM · ${gpu.memory_bandwidth_gbps ?? '-'} GB/s · ${gpu.fp16_tflops ?? '-'} TF`
      : undefined,
  }));

  return (
    <div className="appShell">
      <header className="appHeader">
        <div className="headerInner">
          <div className="brand">
            <div className="brandMark">
              <Cpu size={17} />
            </div>
            <div className="brandText">
              <span>llm-infer-cal</span>
            </div>
            <span className="crumb">
              <ChevronRight size={14} />
              {selectedModelLabel}
            </span>
          </div>
          <div className="headerChips">
            <span>{selectedGpu?.id ?? form.gpu}</span>
            <span>{form.gpu_count || selectedFleet?.gpu_count || 1}× GPU</span>
            <span>{form.engine}</span>
          </div>
        </div>
      </header>

      <main className="pageBody">
        <div className="heroRow">
          <div>
            <h1>GPU Memory Estimator</h1>
            <p>
              估算 LLM 推理显存和多 GPU 方案，辅助选择模型、引擎和 GPU 部署规格。
            </p>
          </div>
          <nav className="viewTabs" aria-label="主视图">
            <button type="button" className={activeTab === 'calculator' ? 'active' : ''} onClick={() => setActiveTab('calculator')}>
              <Calculator size={15} />
              Calculator
            </button>
            <button type="button" className={activeTab === 'compare' ? 'active' : ''} onClick={() => setActiveTab('compare')}>
              <GitCompare size={15} />
              Compare GPUs
            </button>
          </nav>
        </div>

        {error ? (
          <div className="errorBanner">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        ) : null}

        {activeTab === 'calculator' ? (
          <form className="calculatorGrid" onSubmit={submit}>
            <section className="shellCard configCard">
              <PanelTitle dot="primary" title="Configuration" />
              <div className="configBody">
                <ConfigSection icon={<Cpu size={16} />} title="GPU Configuration">
                  <Field label="Provider">
                    <SearchableSelect options={providerOptions} value={gpuVendor} onValueChange={updateGpuVendor} searchPlaceholder="Search provider..." />
                  </Field>
                  <Field label="GPU Model" hint="选择用于推理的 GPU。显存、带宽、FP16 TFLOPS 会影响估算。">
                    <GpuPicker options={gpuModelOptions} selected={selectedGpuId} onSelectPrimary={updatePrimaryGpu} />
                  </Field>
                  <SpecGrid
                    items={[
                      { label: 'VRAM', value: selectedGpu?.memory_gb ? `${selectedGpu.memory_gb} GB` : '-' },
                      { label: 'Mem BW', value: selectedGpu?.memory_bandwidth_gbps ? `${selectedGpu.memory_bandwidth_gbps} GB/s` : '-' },
                      { label: 'FP16 TF', value: selectedGpu?.fp16_tflops ? `${selectedGpu.fp16_tflops} TF` : '-' },
                    ]}
                  />
                  <Field label="Number of GPUs" hint="空值表示由 planner 自动推荐；选择数值会强制 GPU 数。">
                    <SearchableSelect
                      options={gpuCountOptions.map((count) => ({
                        value: count,
                        label: count ? `${count}× GPU${count === '1' ? '' : 's'}` : '自动推荐',
                        sublabel: count && selectedGpu?.memory_gb ? `${Number(count) * selectedGpu.memory_gb} GB total VRAM` : undefined,
                      }))}
                      value={form.gpu_count}
                      onValueChange={(value) => updateField('gpu_count', value)}
                    />
                  </Field>
                </ConfigSection>

                <Separator />

                <ConfigSection icon={<LayoutGrid size={16} />} title="Model">
                  <div className="twoCol">
                    <Field label="Source">
                      <SearchableSelect
                        options={sourceOptions.map((option) => ({ value: option.value, label: option.label }))}
                        value={form.source}
                        onValueChange={(value) => updateSource(value as EvaluateForm['source'])}
                      />
                    </Field>
                    <Field label="Family">
                      <SearchableSelect options={modelVendorOptions} value={modelVendor} onValueChange={updateModelVendor} disabled={form.source !== 'builtin'} />
                    </Field>
                  </div>

                  {form.source === 'builtin' ? (
                    <Field label={`Model (${builtinModelOptions.length} available)`}>
                      <SearchableSelect
                        options={builtinModelOptions}
                        value={form.model_id}
                        onValueChange={(value) => updateField('model_id', value)}
                        placeholder="请选择模型"
                        searchPlaceholder="Type model name..."
                        maxHeight={360}
                      />
                    </Field>
                  ) : (
                    <Field label="Model ID">
                      <input
                        data-testid="remote-model-id-input"
                        value={form.model_id}
                        placeholder="输入 HuggingFace / ModelScope 模型 ID"
                        onChange={(event) => updateField('model_id', event.target.value)}
                      />
                    </Field>
                  )}

                  <ModelSpecGrid report={visibleReport} params={params} activeParams={activeParams} />
                </ConfigSection>

                <Separator />

                <ConfigSection icon={<Database size={16} />} title="Precision">
                  <div className="twoCol">
                    <Field label="Engine">
                      <SearchableSelect
                        options={engineOptions.map((option) => ({ value: option.value, label: option.label }))}
                        value={form.engine}
                        onValueChange={(value) => updateField('engine', value as EvaluateForm['engine'])}
                      />
                    </Field>
                    <Field label="KV Cache Precision">
                      <SearchableSelect
                        options={kvPrecisionOptions.map((option) => ({ value: option.value, label: option.label, sublabel: option.sublabel }))}
                        value={form.kv_cache_bits}
                        onValueChange={(value) => updateField('kv_cache_bits', value)}
                      />
                    </Field>
                  </div>
                </ConfigSection>

                <Separator />

                <ConfigSection icon={<Zap size={16} />} title="Inference Optimizations">
                  <SwitchField
                    label="Paged Attention"
                    description="Reduces KV cache overhead by ~25%"
                    checked={form.paged_attention}
                    onChange={(value) => updateField('paged_attention', value)}
                  />
                  <SwitchField
                    label="Speculative Decoding"
                    description="Draft / MTP model predicts, main model verifies"
                    checked={speculativeEnabled}
                    onChange={updateSpeculativeEnabled}
                  />
                  {speculativeEnabled ? (
                    <div className="speculativeBox">
                      <div className="modeButtons" role="group" aria-label="Speculative decoding mode">
                        <button
                          type="button"
                          className={speculativeMode === 'standard' ? 'active' : ''}
                          onClick={() => updateSpeculativeMode('standard')}
                        >
                          Standard
                        </button>
                        <button
                          type="button"
                          className={speculativeMode === 'mtp' ? 'active mtp' : ''}
                          onClick={() => updateSpeculativeMode('mtp')}
                        >
                          MTP Mode
                        </button>
                      </div>

                      {speculativeMode === 'standard' ? (
                        <>
                          <SliderWithInput
                            label="Draft Model Size"
                            min={0.1}
                            max={3}
                            step={0.1}
                            value={specDraftModelSize}
                            onValueChange={updateSpecDraftModelSize}
                            format={(value) => `${value.toFixed(1)}B`}
                            unit=""
                            markers={[0.5, 1, 1.5, 2]}
                          />
                          <SliderWithInput
                            label="Draft Tokens per Step"
                            min={2}
                            max={16}
                            step={1}
                            value={specNumDraftTokens}
                            onValueChange={updateSpecNumDraftTokens}
                            format={(value) => `${Math.round(value)}`}
                            unit="tokens"
                            markers={[2, 4, 8, 12, 16]}
                          />
                          <TextField
                            label="Draft Model ID"
                            value={form.speculative_draft_model_id}
                            placeholder="例如 Qwen/Qwen3-0.6B"
                            onChange={(value) => updateField('speculative_draft_model_id', value)}
                          />
                          <p>
                            Draft model adds ~{draftModelWeightGb(specDraftModelSize)} GiB VRAM.
                            Estimated speedup: {(1 + (specNumDraftTokens * 0.6) / 2).toFixed(1)}x at ~60% acceptance rate.
                          </p>
                        </>
                      ) : (
                        <div className="optimizationNotice mtp">
                          <p>
                            <strong>MTP (Multi-Token Prediction):</strong> Built-in speculative decoding using the model&apos;s native MTP heads.
                            No separate draft model needed. Adds only ~0.3 GiB VRAM for the MTP heads. Estimated speedup: ~1.8x.
                          </p>
                        </div>
                      )}
                    </div>
                  ) : null}
                  {isMoe ? (
                    <>
                      <SwitchField
                        label="Expert Offloading"
                        description="Keep select experts on GPU, offload rest to CPU"
                        checked={expertOffloading}
                        onChange={updateExpertOffloading}
                      />
                      {expertOffloading ? (
                        <div className="expertBox">
                          <SliderWithInput
                            label="Experts on GPU"
                            min={Math.max(1, activeExperts)}
                            max={Math.max(totalExperts, activeExperts)}
                            step={1}
                            value={Math.max(activeExperts, Math.min(numGpuExperts, totalExperts))}
                            onValueChange={updateExpertsOnGpu}
                            format={(value) => `${Math.round(value)} / ${Math.max(totalExperts, activeExperts)}`}
                            unit=""
                            markers={[Math.max(1, activeExperts), Math.max(totalExperts, activeExperts)]}
                          />
                          <ExpertOffloadNotes
                            activeExperts={activeExperts}
                            totalExperts={Math.max(totalExperts, activeExperts)}
                            expertsOnGpu={Math.max(activeExperts, Math.min(numGpuExperts, totalExperts))}
                            weightBytes={weightBytes}
                          />
                        </div>
                      ) : null}
                    </>
                  ) : null}
                  {form.paged_attention ? (
                    <div className="optimizationNotice paged">
                      <p>Paged attention enabled: KV cache memory reduced by ~25%. This is the default for vLLM, SGLang, and TensorRT-LLM.</p>
                    </div>
                  ) : null}
                  <details className="advancedPanel" data-testid="advanced-settings">
                    <summary>
                      <span>
                        <ChevronDown className="disclosureIcon" size={16} />
                        高级设置
                      </span>
                    </summary>
                    <div className="advancedGrid">
                      <NumberField
                        label="Context Window"
                        value={form.context_length}
                        placeholder="自动使用模型上下文"
                        hint="留空时使用模型配置或内置参考上下文。"
                        onChange={(value) => updateField('context_length', value)}
                      />
                      <NumberField
                        label="Concurrent Users"
                        value={form.target_concurrent_requests}
                        placeholder="自动推荐 min/dev/prod"
                        hint="留空时保留 min/dev/prod 三档；填写后只计算这个并发数。"
                        onChange={(value) => updateField('target_concurrent_requests', value)}
                      />
                      {advancedControls.map((setting) =>
                        setting.control === 'checkbox' ? (
                          <CheckboxField
                            key={setting.key}
                            label={setting.label}
                            checked={form[setting.key]}
                            onChange={(value) => updateField(setting.key, value)}
                          />
                        ) : (
                          <NumberField
                            key={setting.key}
                            label={setting.label}
                            value={form[setting.key]}
                            onChange={(value) => updateField(setting.key, value)}
                          />
                        ),
                      )}
                      <NumberField
                        label="CPU Offload / GPU GiB"
                        value={form.cpu_offload_gb}
                        onChange={(value) => updateField('cpu_offload_gb', value)}
                      />
                      {form.llm_review
                        ? reviewerSettings.map((setting) => (
                            <TextField
                              key={setting.key}
                              label={setting.label}
                              value={form[setting.key]}
                              type={setting.type ?? 'text'}
                              placeholder={setting.placeholder}
                              onChange={(value) => updateField(setting.key, value)}
                            />
                          ))
                        : null}
                    </div>
                  </details>
                </ConfigSection>

                <button className="primaryButton" type="submit" disabled={evaluating || !form.model_id.trim() || !selectedGpuIds.length || requiresDraftModel}>
                  {evaluating ? <RefreshCcw className="spin" size={17} /> : <Play size={17} />}
                  <span>{evaluating ? '计算中' : '开始评估'}</span>
                </button>
              </div>
            </section>

            <section className="shellCard resultCard">
              <PanelTitle dot="success" title="Estimates">
                <button type="button" className="ghostButton" onClick={resetForm}>
                  Reset to defaults
                </button>
              </PanelTitle>
              <div className="resultBody">
                <FitStatus option={selectedFleet} hardware={visibleReport?.hardware ?? selectedGpu} />
                <VramOverview option={selectedFleet} hardware={visibleReport?.hardware ?? selectedGpu} paged={form.paged_attention} />
                <MetricCards
                  selectedFleet={selectedFleet}
                  weightBytes={weightBytes}
                  form={form}
                  hardware={visibleReport?.hardware ?? selectedGpu}
                />
                <SupportCallouts
                  report={visibleReport}
                  form={form}
                  engineSupport={engineSupport}
                  activeParams={activeParams}
                  params={params}
                />

                <section className="innerPanel">
                  <SectionTitle title="VRAM Breakdown" />
                  <MemoryBreakdown option={selectedFleet} hardware={visibleReport?.hardware ?? selectedGpu} />
                  <div className="kvList">
                    {kvRows.map((row) => (
                      <div className="kvRow" key={row.context_tokens}>
                        <span>{formatNumber(row.context_tokens)} ctx</span>
                        <strong>{formatBytes(row.bytes?.value)}</strong>
                        <small>{labelText(row.bytes?.label)}</small>
                      </div>
                    ))}
                    {activationRows.slice(0, 1).map((row) => (
                      <div className="kvRow" key={`activation-${row.context_tokens}`}>
                        <span>activation working set</span>
                        <strong>{formatBytes(row.bytes?.value)}</strong>
                        <small>{labelText(row.bytes?.label)}</small>
                      </div>
                    ))}
                  </div>
                </section>

                <div className="contentGrid">
                  <section className="innerPanel">
                    <SectionTitle title="Fleet Options" />
                    <FleetTable options={visibleReport?.fleet?.options ?? []} />
                    <p className="noteText">{visibleReport?.fleet?.constraint_note_zh ?? visibleReport?.fleet?.constraint_note_en ?? selectedGpu?.notes_zh}</p>
                  </section>

                  <section className="innerPanel">
                    <SectionTitle title="启动命令" />
                    <div className="commandBox">
                      <pre>{command || '等待评估结果'}</pre>
                      <button type="button" className="iconButton" onClick={copyCommand} disabled={!command} aria-label="复制启动命令">
                        {copied ? <CheckCircle2 size={18} /> : <Clipboard size={18} />}
                      </button>
                    </div>
                    <OptimizationList report={visibleReport} />
                  </section>
                </div>

                <section className="innerPanel">
                  <SectionTitle title="Formula Reference" />
                  <FormulaReference report={visibleReport} form={form} activeParams={activeParams} option={selectedFleet} />
                </section>

                <InferenceProcess report={visibleReport} option={selectedFleet} />
              </div>
            </section>
          </form>
        ) : (
          <section className="shellCard compareShell">
            <PanelTitle dot="success" title="Compare GPUs" />
            <div className="compareBody">
              <CompareControls
                vendors={gpuGroups.map((group) => group.label)}
                vendor={compareVendor}
                tier={compareTier}
                gpuCount={compareGpuCount}
                selectedCount={compareGpuIds.length}
                onVendor={setCompareVendor}
                onTier={setCompareTier}
                onGpuCount={setCompareGpuCount}
                onRun={runComparison}
                disabled={evaluating || !form.model_id.trim() || requiresDraftModel}
              />
              <ComparisonTable reports={comparisonReports} />
            </div>
          </section>
        )}
      </main>

      <footer className="pageFooter">
        <p>Estimates are physics-based approximations. Real-world performance varies by framework, driver version, and system memory bandwidth.</p>
        <a href="https://github.com/xrwang8/llm-infer-cal" target="_blank" rel="noopener noreferrer">
          <Github size={14} />
          xrwang8/llm-infer-cal
        </a>
      </footer>
    </div>
  );
}

function PanelTitle({ dot, title, children }: { dot: 'primary' | 'success'; title: string; children?: ReactNode }) {
  return (
    <div className="panelTitle">
      <div className="panelTitleText">
        <span className={`titleDot ${dot}`} />
        <p>{title}</p>
      </div>
      {children}
    </div>
  );
}

function ConfigSection({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <section className="configSection">
      <SectionHeader icon={icon} title={title} />
      <div className="sectionFields">{children}</div>
    </section>
  );
}

function SectionHeader({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="sectionHeader">
      {icon}
      <h3>{title}</h3>
    </div>
  );
}

function SectionTitle({ title }: { title: string }) {
  return <p className="sectionTitle">{title}</p>;
}

function Separator() {
  return <div className="separator" />;
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="field">
      <span>
        {label}
        {hint ? (
          <span className="infoHint" title={hint}>
            <Info size={13} />
          </span>
        ) : null}
      </span>
      {children}
    </label>
  );
}

function SearchableSelect({
  options,
  value,
  onValueChange,
  placeholder = 'Select...',
  searchPlaceholder = 'Search...',
  disabled = false,
  maxHeight = 320,
}: {
  options: SelectOption[];
  value: string;
  onValueChange: (value: string) => void;
  placeholder?: string;
  searchPlaceholder?: string;
  disabled?: boolean;
  maxHeight?: number;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const selected = options.find((option) => option.value === value);
  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return options;
    return options.filter((option) => `${option.label} ${option.sublabel ?? ''}`.toLowerCase().includes(normalized));
  }, [options, query]);

  useEffect(() => {
    function closeOnOutsideClick(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
        setQuery('');
      }
    }

    document.addEventListener('mousedown', closeOnOutsideClick);
    return () => document.removeEventListener('mousedown', closeOnOutsideClick);
  }, []);

  useEffect(() => {
    if (open) {
      window.setTimeout(() => inputRef.current?.focus(), 20);
    }
  }, [open]);

  function selectValue(nextValue: string) {
    onValueChange(nextValue);
    setOpen(false);
    setQuery('');
  }

  return (
    <div className="searchSelect" ref={containerRef}>
      <button type="button" className="selectTrigger" disabled={disabled} onClick={() => setOpen((current) => !current)} aria-expanded={open}>
        <span>
          {selected ? (
            <>
              <b>{selected.label}</b>
              {selected.badge}
            </>
          ) : (
            <em>{placeholder}</em>
          )}
        </span>
        <ChevronDown size={15} />
      </button>
      {open ? (
        <div className="selectMenu" style={{ maxHeight }}>
          <div className="selectSearch">
            <Search size={14} />
            <input
              ref={inputRef}
              value={query}
              placeholder={searchPlaceholder}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Escape') {
                  setOpen(false);
                  setQuery('');
                }
                if (event.key === 'Enter' && filtered.length === 1) {
                  selectValue(filtered[0].value);
                }
              }}
            />
            {query ? (
              <button type="button" onClick={() => setQuery('')} aria-label="清空搜索">
                <X size={13} />
              </button>
            ) : null}
          </div>
          <div className="selectList">
            {filtered.length ? (
              filtered.map((option) => (
                <button key={option.value || option.label} type="button" disabled={option.disabled} className={option.value === value ? 'selected' : ''} onClick={() => selectValue(option.value)}>
                  <span className="checkSlot">{option.value === value ? <Check size={14} /> : null}</span>
                  <span className="optionText">
                    <span>{option.label}</span>
                    {option.sublabel ? <small>{option.sublabel}</small> : null}
                  </span>
                  {option.badge}
                </button>
              ))
            ) : (
              <p className="emptyText">No results for "{query}"</p>
            )}
          </div>
          {filtered.length ? <div className="selectCount">{filtered.length} result{filtered.length === 1 ? '' : 's'}</div> : null}
        </div>
      ) : null}
    </div>
  );
}

function GpuPicker({
  options,
  selected,
  onSelectPrimary,
}: {
  options: SelectOption[];
  selected: string;
  onSelectPrimary: (gpuId: string) => void;
}) {
  return (
    <div className="gpuPicker" data-testid="gpu-model-picker">
      <SearchableSelect options={options} value={selected} onValueChange={onSelectPrimary} searchPlaceholder="Type to search GPUs..." maxHeight={300} />
    </div>
  );
}

function SpecGrid({ items }: { items: Array<{ label: string; value: string }> }) {
  return (
    <div className="specGrid">
      {items.map((item) => (
        <div key={item.label}>
          <p>{item.label}</p>
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  );
}

function ModelSpecGrid({ report, params, activeParams }: { report: Report | null; params?: number; activeParams?: number }) {
  const architecture = report?.architecture ?? {};
  const attention = architecture.attention ?? {};
  const isMoe = !!params && !!activeParams && activeParams < params;
  return (
    <SpecGrid
      items={[
        { label: isMoe ? 'Total Params' : 'Params', value: formatNumber(params) },
        {
          label: isMoe ? 'Active Params' : 'Layers',
          value: isMoe ? formatNumber(activeParams) : formatNumber(architecture.num_hidden_layers ?? architecture.num_layers ?? architecture.layers),
        },
        { label: 'KV Heads', value: formatNumber(attention.num_kv_heads ?? architecture.num_key_value_heads) },
      ]}
    />
  );
}

function SliderWithInput({
  label,
  min,
  max,
  step,
  value,
  onValueChange,
  format,
  unit,
  markers,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onValueChange: (value: number) => void;
  format: (value: number) => string;
  unit: string;
  markers: number[];
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');

  function commit() {
    const parsed = parseCompact(draft);
    setEditing(false);
    if (parsed == null) {
      setDraft('');
      return;
    }
    const bounded = Math.max(min, Math.min(max, parsed));
    const snapped = Math.round(bounded / step) * step;
    onValueChange(snapped);
    setDraft('');
  }

  return (
    <div className="sliderField">
      <div className="sliderLabel">
        <span>{label}</span>
        {editing ? (
          <input
            autoFocus
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={commit}
            onKeyDown={(event) => {
              if (event.key === 'Enter') commit();
              if (event.key === 'Escape') {
                setEditing(false);
                setDraft('');
              }
            }}
          />
        ) : (
          <button
            type="button"
            onClick={() => {
              setEditing(true);
              setDraft(String(value));
            }}
          >
            {format(value)} <small>{unit}</small>
          </button>
        )}
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onValueChange(Number(event.target.value))} />
      <div className="rangeMarkers">
        {markers.map((marker) => (
          <span key={marker}>{format(marker)}</span>
        ))}
      </div>
    </div>
  );
}

function SwitchField({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="switchField">
      <span>
        <b>{label}</b>
        <small>{description}</small>
      </span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <i />
    </label>
  );
}

function NumberField({
  label,
  value,
  placeholder,
  hint,
  onChange,
}: {
  label: string;
  value: string;
  placeholder?: string;
  hint?: string;
  onChange: (value: string) => void;
}) {
  return (
    <Field label={label} hint={hint}>
      <input inputMode="decimal" value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
    </Field>
  );
}

function TextField({
  label,
  value,
  type = 'text',
  placeholder,
  onChange,
}: {
  label: string;
  value: string;
  type?: 'password' | 'text';
  placeholder?: string;
  onChange: (value: string) => void;
}) {
  return (
    <Field label={label}>
      <input type={type} value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
    </Field>
  );
}

function CheckboxField({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="checkField">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function FitStatus({ option, hardware }: { option?: FleetOption; hardware?: GpuSummary | null }) {
  const memoryBytes = (hardware?.memory_gb ?? 0) * 1_000_000_000;
  const required = option?.required_bytes_per_gpu_at_tier ?? requiredBytes(option);
  const usage = pct(required, memoryBytes);
  const overflow = option ? !option.fits : false;
  const tight = !overflow && usage > 85;
  const tone = overflow ? 'overflow' : tight ? 'tight' : 'fits';
  const label = overflow ? 'Insufficient VRAM' : tight ? 'Tight fit (>85%)' : 'Fits comfortably';
  const Icon = overflow || tight ? AlertTriangle : CheckCircle2;

  return (
    <div className={`fitStatus ${tone}`}>
      <Icon size={20} />
      <div>
        <strong>{label}</strong>
        <p>
          {formatBytes(required)} required / {formatBytes(memoryBytes)} available ({option?.gpu_count ?? 1}× {hardware?.id ?? 'GPU'})
        </p>
      </div>
    </div>
  );
}

function VramOverview({ option, hardware, paged }: { option?: FleetOption; hardware?: GpuSummary | null; paged: boolean }) {
  const memoryBytes = (hardware?.memory_gb ?? 0) * 1_000_000_000;
  const concurrent = option?.tier_concurrent_requests ?? 1;
  const mainWeight = option?.main_weight_bytes_per_gpu ?? Math.max((option?.weight_bytes_per_gpu ?? 0) - (option?.speculative_weight_bytes_per_gpu ?? 0), 0);
  const speculativeWeight = option?.speculative_weight_bytes_per_gpu ?? 0;
  const kv = (option?.kv_bytes_per_request_per_gpu ?? option?.kv_bytes_per_request ?? 0) * concurrent;
  const activation = option?.activation_bytes_per_request_per_gpu ?? 0;
  const required = option?.required_bytes_per_gpu_at_tier ?? mainWeight + speculativeWeight + kv + activation;
  const usage = pct(required, memoryBytes);
  const overflow = option ? !option.fits : false;
  const tight = !overflow && usage > 85;
  const barClass = overflow ? 'overflow' : tight ? 'tight' : 'fits';

  return (
    <div className="vramOverview">
      <div className="utilHeader">
        <span>VRAM Utilization</span>
        <strong>
          {formatFloat(usage, 1)}% of {formatBytes(memoryBytes)}
        </strong>
      </div>
      <div className="utilTrack">
        <div className={barClass} style={{ width: `${Math.min(usage, 100)}%` }} />
      </div>
      <div className="stackedTrack">
        <div className="weight" style={{ width: `${pct(mainWeight, required)}%` }} />
        {speculativeWeight > 0 ? <div className="speculative" style={{ width: `${pct(speculativeWeight, required)}%` }} /> : null}
        <div className="kv" style={{ width: `${pct(kv, required)}%` }} />
        <div className="activation" style={{ width: `${pct(activation, required)}%` }} />
      </div>
      <div className="legendList">
        <Legend tone="weight" label={`Weights ${formatBytes(mainWeight)}`} percent={pct(mainWeight, required)} />
        {speculativeWeight > 0 ? <Legend tone="speculative" label={`Draft ${formatBytes(speculativeWeight)}`} percent={pct(speculativeWeight, required)} /> : null}
        <Legend tone="kv" label={`KV Cache ${formatBytes(kv)}${paged ? ' (paged)' : ''}`} percent={pct(kv, required)} />
        <Legend tone="activation" label={`Activations ${formatBytes(activation)}`} percent={pct(activation, required)} />
      </div>
    </div>
  );
}

function Legend({ tone, label, percent }: { tone: string; label: string; percent: number }) {
  return (
    <span>
      <i className={tone} />
      {label} <small>({formatFloat(percent, 0)}%)</small>
    </span>
  );
}

function MetricCards({
  selectedFleet,
  weightBytes,
  form,
  hardware,
}: {
  selectedFleet?: FleetOption;
  weightBytes?: number;
  form: EvaluateForm;
  hardware?: GpuSummary | null;
}) {
  const totalRequired = totalRequiredBytes(selectedFleet);
  const modelWeightPerGpu = selectedFleet?.main_weight_bytes_per_gpu ?? selectedFleet?.weight_bytes_per_gpu ?? weightBytes;

  return (
    <div className="metricGrid">
      <MetricCard
        icon={<HardDrive size={17} />}
        label="Total VRAM"
        value={formatBytes(totalRequired)}
        sub={`${selectedFleet?.gpu_count ?? 1}× ${hardware?.id ?? form.gpu} total required`}
        accent
      />
      <MetricCard icon={<Layers size={17} />} label="Model Weights / GPU" value={formatBytes(modelWeightPerGpu)} sub="GPU resident model weights" />
      <MetricCard icon={<Activity size={17} />} label="Recommended GPUs" value={selectedFleet?.gpu_count ? `${selectedFleet.gpu_count} 张` : '-'} sub={tierText(selectedFleet?.tier)} />
    </div>
  );
}

function MetricCard({ icon, label, value, sub, accent = false }: { icon: ReactNode; label: string; value: string; sub?: string; accent?: boolean }) {
  return (
    <div className={`metricCard ${accent ? 'accent' : ''}`}>
      <div>
        {icon}
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      {sub ? <p>{sub}</p> : null}
    </div>
  );
}

function SupportCallouts({
  report,
  form,
  engineSupport,
  activeParams,
  params,
}: {
  report: Report | null;
  form: EvaluateForm;
  engineSupport: string;
  activeParams?: number;
  params?: number;
}) {
  const isMoe = !!params && !!activeParams && activeParams < params;
  const caveats = report?.engine_compatibility?.caveats_zh ?? report?.engine_compatibility?.caveats_en ?? [];
  return (
    <div className="calloutStack">
      <div className={`callout ${engineSupport === 'full' ? 'success' : 'warning'}`}>
        <CheckCircle2 size={16} />
        <p>
          <strong>{engineSupport === 'full' ? '完整支持' : `Engine support: ${engineSupport}`}</strong>
          {caveats.length ? ` ${caveats[0]}` : ' 当前模型和引擎配置可生成启动命令。'}
        </p>
      </div>
      {isMoe ? (
        <div className="callout info">
          <Layers size={16} />
          <p>
            <strong>MoE Calculation Notes:</strong> 权重显存按总参数常驻，Prefill/Decode 按 active params 估算。
          </p>
        </div>
      ) : null}
      {(form.paged_attention || form.speculative_draft_model_id || form.speculative_extra_weight_gb) ? (
        <div className="callout info">
          <Zap size={16} />
          <p>
            <strong>Inference Optimizations Active:</strong>
            {form.paged_attention ? ' Paged Attention 已启用。' : ''}
            {form.speculative_draft_model_id || form.speculative_extra_weight_gb ? ' Speculative / Draft 额外权重会进入 GPU resident weights。' : ''}
          </p>
        </div>
      ) : null}
    </div>
  );
}

function MemoryBreakdown({ option, hardware }: { option?: FleetOption; hardware?: GpuSummary | null }) {
  const memoryBytes = (hardware?.memory_gb ?? 0) * 1_000_000_000;
  const concurrent = option?.tier_concurrent_requests ?? 1;
  const weight = option?.weight_bytes_per_gpu ?? 0;
  const mainWeight = option?.main_weight_bytes_per_gpu ?? Math.max(weight - (option?.speculative_weight_bytes_per_gpu ?? 0), 0);
  const speculativeWeight = option?.speculative_weight_bytes_per_gpu ?? 0;
  const cpuOffload = option?.cpu_offload_bytes_per_gpu ?? 0;
  const kv = (option?.kv_bytes_per_request_per_gpu ?? option?.kv_bytes_per_request ?? 0) * concurrent;
  const activation = option?.activation_bytes_per_request_per_gpu ?? 0;
  const reserved = option?.reserved_bytes_per_gpu ?? Math.max(memoryBytes - (option?.usable_bytes_per_gpu ?? 0), 0);
  const required = option?.required_bytes_per_gpu_at_tier ?? weight + kv + activation;

  return (
    <div className="memoryBars">
      <div className="breakdownSummary">
        <span>Required / GPU (per GPU)</span>
        <strong>{formatBytes(required)}</strong>
      </div>
      <Bar label="Main Weights / GPU" value={mainWeight} total={memoryBytes} tone="weight" />
      {speculativeWeight > 0 ? <Bar label="Draft / Speculative Weights" value={speculativeWeight} total={memoryBytes} tone="speculative" /> : null}
      {cpuOffload > 0 ? <Bar label="CPU Offload Saved / GPU" value={cpuOffload} total={memoryBytes} tone="offload" /> : null}
      <Bar label={`KV Cache × ${concurrent}`} value={kv} total={memoryBytes} tone="kv" />
      <Bar label="Activation Working Set" value={activation} total={memoryBytes} tone="activation" />
      <Bar label="Reserved / Framework" value={reserved} total={memoryBytes} tone="reserved" />
    </div>
  );
}

function Bar({ label, value, total, tone }: { label: string; value: number; total: number; tone: string }) {
  return (
    <div className="barRow">
      <div className="barText">
        <span>{label}</span>
        <strong>{formatBytes(value)}</strong>
      </div>
      <div className="barTrack">
        <div className={`barFill ${tone}`} style={{ width: `${pct(value, total)}%` }} />
      </div>
    </div>
  );
}

function FleetTable({ options }: { options: FleetOption[] }) {
  if (!options.length) {
    return <p className="emptyText">暂无方案</p>;
  }

  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>层级</th>
            <th>GPU</th>
            <th>TP</th>
            <th>并发</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody>
          {options.map((option, index) => (
            <tr key={`${option.tier}-${option.gpu_count}-${index}`}>
              <td>{tierText(option.tier)}</td>
              <td>{option.gpu_count ?? '-'}</td>
              <td>{option.tensor_parallel_size ?? '-'}</td>
              <td>{formatNumber(option.max_concurrent_at_reference_ctx)}</td>
              <td>
                <span className={option.fits ? 'fitText' : 'missText'}>{option.fits ? '可行' : '不足'}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CompareControls({
  vendors,
  vendor,
  tier,
  gpuCount,
  selectedCount,
  disabled,
  onVendor,
  onTier,
  onGpuCount,
  onRun,
}: {
  vendors: string[];
  vendor: string;
  tier: string;
  gpuCount: string;
  selectedCount: number;
  disabled: boolean;
  onVendor: (value: string) => void;
  onTier: (value: string) => void;
  onGpuCount: (value: string) => void;
  onRun: () => void;
}) {
  return (
    <div className="compareControls">
      <Field label="Provider">
        <SearchableSelect
          options={[{ value: 'all', label: '全部厂商' }, ...vendors.map((nextVendor) => ({ value: nextVendor, label: nextVendor }))]}
          value={vendor}
          onValueChange={onVendor}
        />
      </Field>
      <Field label="Tier">
        <SearchableSelect
          options={[
            { value: 'all', label: gpuTierLabel('all') },
            { value: 'datacenter', label: gpuTierLabel('datacenter') },
            { value: 'prosumer', label: gpuTierLabel('prosumer') },
            { value: 'consumer', label: gpuTierLabel('consumer') },
          ]}
          value={tier}
          onValueChange={onTier}
        />
      </Field>
      <Field label="GPUs per node">
        <SearchableSelect
          options={gpuCountOptions.map((count) => ({ value: count, label: count ? `${count}× GPU` : '自动推荐' }))}
          value={gpuCount}
          onValueChange={onGpuCount}
        />
      </Field>
      <button className="secondaryButton" type="button" onClick={onRun} disabled={disabled || selectedCount === 0}>
        对比 {selectedCount} 张 GPU
      </button>
    </div>
  );
}

function ComparisonTable({ reports }: { reports: Report[] }) {
  if (!reports.length) {
    return <p className="emptyText">选择过滤条件后点击对比。</p>;
  }

  const sorted = [...reports].sort((a, b) => {
    const aOption = bestFleetOption(a.fleet);
    const bOption = bestFleetOption(b.fleet);
    const fitDelta = Number(!!bOption?.fits) - Number(!!aOption?.fits);
    if (fitDelta !== 0) {
      return fitDelta;
    }
    const gpuDelta = (aOption?.gpu_count ?? Number.MAX_SAFE_INTEGER) - (bOption?.gpu_count ?? Number.MAX_SAFE_INTEGER);
    if (gpuDelta !== 0) {
      return gpuDelta;
    }
    return (b.hardware?.memory_gb ?? 0) - (a.hardware?.memory_gb ?? 0);
  });

  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>GPU</th>
            <th>Min GPUs</th>
            <th>GPU VRAM</th>
            <th>Mem BW</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((item) => {
            const option = bestFleetOption(item.fleet);
            return (
              <tr key={item.hardware?.id ?? item.generated_command?.command}>
                <td>{item.hardware?.id ?? '-'}</td>
                <td>{option?.gpu_count ? `${option.gpu_count} 张` : '-'}</td>
                <td>{item.hardware?.memory_gb ? `${item.hardware.memory_gb}GB` : '-'}</td>
                <td>{item.hardware?.memory_bandwidth_gbps ? `${item.hardware.memory_bandwidth_gbps} GB/s` : '-'}</td>
                <td>
                  <span className={option?.fits ? 'fitText' : 'missText'}>{option?.fits ? '可行' : '不足'}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function FormulaReference({
  report,
  form,
  activeParams,
  option,
}: {
  report: Report | null;
  form: EvaluateForm;
  activeParams?: number;
  option?: FleetOption;
}) {
  const params = annotatedValue<number>(report?.weights?.params_estimated);
  const weight = annotatedValue<number>(report?.weights?.safetensors_total_bytes);
  const kv = report?.kv_cache_by_context?.at(-1);
  const isMoe = !!params && !!activeParams && activeParams < params;
  const architecture = report?.architecture ?? {};
  const hiddenSize = architecture.hidden_size ?? architecture.hidden;
  const kvBits = report?.inference_options?.kv_cache_bits ?? Number(form.kv_cache_bits || 16);
  const paged = report?.inference_options?.paged_attention ?? form.paged_attention;
  const speculativeWeight = annotatedValue<number>(report?.inference_options?.speculative_extra_weight_bytes);
  const cpuOffload = report?.inference_options?.cpu_offload_bytes_per_gpu ?? 0;
  const tierConcurrency = option?.tier_concurrent_requests ?? report?.inference_options?.target_concurrent_requests ?? Number(form.target_concurrent_requests || 0);

  return (
    <div className="formulaList">
      <details open>
        <summary>Model Weights VRAM</summary>
        <pre>
{`weights = safetensors_total_bytes
${weight ? `observed = ${formatBytes(weight)}` : 'observed = waiting for report'}
${speculativeWeight ? `draft/speculative extra = ${formatBytes(speculativeWeight)}` : 'draft/speculative extra = 0'}
${cpuOffload ? `cpu_offload_per_gpu = ${formatBytes(cpuOffload)} saved from GPU resident weights` : 'cpu_offload_per_gpu = 0'}
${isMoe ? 'MoE: all expert weights stay resident in VRAM.' : 'Dense: all weights participate in memory and compute.'}`}
        </pre>
      </details>
      <details open>
        <summary>KV Cache VRAM</summary>
        <pre>
{`standard: per_token_per_layer_bits = 2 * num_kv_heads * head_dim * ${kvBits}
MLA: per_token_per_layer_bits = (kv_lora_rank + qk_rope_head_dim) * ${kvBits}
effective_seq_len = sliding_window ? min(seq_len, sliding_window) : seq_len
baseline = per_token_per_layer_bits * effective_seq_len * num_layers / 8
CSA+HCA: baseline * avg(1 / compress_ratio)
NSA: baseline * min(nsa_topk / effective_seq_len, 1.0)
paged_attention_factor = ${paged ? '0.75' : '1.00'}
${kv?.bytes?.value ? `selected = ${formatBytes(kv.bytes.value)}` : 'selected = waiting for report'}`}
        </pre>
      </details>
      <details>
        <summary>Activations</summary>
        <pre>
{`activation = max_num_batched_tokens * hidden_size * dtype_bytes * activation_factor
max_num_batched_tokens = 2048; dtype_bytes = 2; activation_factor = 2
MoE activation correction = 1 + active_experts / total_experts * 0.5
${hiddenSize ? `hidden_size = ${formatNumber(hiddenSize)}` : 'hidden_size = waiting for report'}
${isMoe ? 'MoE correction is applied by the backend.' : 'Dense model uses correction factor 1.0.'}`}
        </pre>
      </details>
      <details>
        <summary>Required / GPU Fit</summary>
        <pre>
{`decode_required = weights_per_gpu + decode_activation_per_gpu + concurrent * kv_per_gpu
prefill_required = weights_per_gpu + prefill_peak_activation_per_gpu + concurrent * kv_per_gpu
required_per_gpu = max(decode_required, prefill_required)
${tierConcurrency ? `concurrent = ${formatNumber(tierConcurrency)} requests` : 'concurrent = waiting for selected tier'}
${option?.decode_required_bytes_per_gpu_at_tier ? `decode_required = ${formatBytes(option.decode_required_bytes_per_gpu_at_tier)}` : 'decode_required = waiting for report'}
${option?.prefill_activation_bytes_per_gpu_at_tier ? `prefill_peak_activation = ${formatBytes(option.prefill_activation_bytes_per_gpu_at_tier)}` : 'prefill_peak_activation = waiting for report'}
${option?.prefill_required_bytes_per_gpu_at_tier ? `prefill_required = ${formatBytes(option.prefill_required_bytes_per_gpu_at_tier)}` : 'prefill_required = waiting for report'}
${option?.required_bytes_per_gpu_at_tier ? `selected = ${formatBytes(option.required_bytes_per_gpu_at_tier)}` : 'selected = waiting for report'}`}
        </pre>
      </details>
    </div>
  );
}

function InferenceProcess({ report, option }: { report: Report | null; option?: FleetOption }) {
  const explainText = memoryExplainText(report?.explain_text, {
    tier: option?.tier,
    gpuCount: option?.gpu_count,
    contextTokens: option?.kv_reference_context_tokens,
  });

  return (
    <section className="innerPanel processPanel">
      <div className="processHeader">
        <SectionTitle title="推理过程" />
      </div>
      <div className="processBox">
        <pre>{explainText || '等待评估结果'}</pre>
      </div>
    </section>
  );
}

function OptimizationList({ report }: { report: Report | null }) {
  const required = report?.engine_compatibility?.required_flags ?? [];
  const optional = report?.engine_compatibility?.optional_flags ?? [];
  const caveats = report?.engine_compatibility?.caveats_zh ?? report?.engine_compatibility?.caveats_en ?? [];
  const allItems = [...required, ...optional].slice(0, 5);

  if (!allItems.length && !caveats.length) {
    return <p className="noteText">当前引擎没有额外必选参数。</p>;
  }

  return (
    <div className="optimizationList">
      {allItems.map((item: any) => (
        <div className="optimizationItem" key={`${item.flag}-${item.value ?? ''}`}>
          <span>{item.flag}</span>
          <p>{item.note_zh ?? item.note_en ?? item.value ?? '建议保留'}</p>
        </div>
      ))}
      {caveats.map((caveat: string) => (
        <div className="optimizationItem warnItem" key={caveat}>
          <span>注意</span>
          <p>{caveat}</p>
        </div>
      ))}
    </div>
  );
}

function ExpertOffloadNotes({
  activeExperts,
  totalExperts,
  expertsOnGpu,
  weightBytes,
}: {
  activeExperts: number;
  totalExperts: number;
  expertsOnGpu: number;
  weightBytes?: number;
}) {
  const recommendedMin = Math.max(activeExperts, Math.ceil(totalExperts * 0.15));
  const recommendedMax = Math.min(totalExperts, Math.ceil(totalExperts * 0.4));
  const fullWeightsGb = (weightBytes ?? 0) / 1_000_000_000;
  const offloadedFraction = 1 - expertsOnGpu / totalExperts;
  const savedVram = Math.max(0, fullWeightsGb * offloadedFraction * 0.7);

  return (
    <div className="expertNotes">
      {expertsOnGpu < recommendedMin ? (
        <div className="optimizationNotice warn">
          <p>
            <strong>Suggestion:</strong> Try keeping at least {recommendedMin} experts on GPU.
            {activeExperts > 1
              ? ` This model activates ${activeExperts} experts per token, so fewer than this causes frequent PCIe transfers.`
              : ' Keeping a small routing buffer on GPU helps when routing shifts.'}
          </p>
        </div>
      ) : null}
      {expertsOnGpu < totalExperts ? (
        <div className="optimizationNotice expert">
          <p>
            <strong>VRAM saved:</strong> ~{savedVram.toFixed(2)} GB on weights.
            <span> Offloaded experts: {totalExperts - expertsOnGpu}/{totalExperts}</span>
          </p>
        </div>
      ) : null}
      {expertsOnGpu >= recommendedMin && expertsOnGpu <= recommendedMax ? (
        <div className="optimizationNotice paged">
          <p>
            <strong>Recommended range:</strong> {recommendedMin}-{recommendedMax} experts on GPU balances VRAM savings with performance.
          </p>
        </div>
      ) : null}
      <div className="optimizationNotice neutral">
        <p>
          <strong>Why this matters:</strong> In MoE models, only {activeExperts} of {totalExperts} experts compute per token,
          but expert weights still drive resident VRAM. Offloading saves memory but adds PCIe transfer latency when offloaded experts are routed.
        </p>
      </div>
    </div>
  );
}

function requiredBytes(option?: FleetOption): number {
  if (!option) return 0;
  const concurrent = option.tier_concurrent_requests ?? 1;
  const kv = (option.kv_bytes_per_request_per_gpu ?? option.kv_bytes_per_request ?? 0) * concurrent;
  return (option.weight_bytes_per_gpu ?? 0) + kv + (option.activation_bytes_per_request_per_gpu ?? 0);
}

export function totalRequiredBytes(option?: FleetOption): number {
  const perGpu = option?.required_bytes_per_gpu_at_tier ?? requiredBytes(option);
  return perGpu * (option?.gpu_count ?? 1);
}

type ExplainTextFilter = {
  tier?: string;
  gpuCount?: number;
  contextTokens?: number;
};

export function memoryExplainText(explainText?: string | null, filter: ExplainTextFilter = {}): string {
  const trimmed = explainText?.trim();
  if (!trimmed) {
    return '';
  }
  const selectedContextHeading = filter.contextTokens ? `KV cache @ ${explainContextLabel(filter.contextTokens)} context` : null;
  const selectedFleetPrefix = filter.tier ? `Fleet tier: ${filter.tier}` : null;

  return trimmed
    .split(/\n{2,}/)
    .filter((block) => {
      const heading = block.split('\n', 1)[0]?.trim() ?? '';
      if (/^(prefill latency|decode throughput|k bound|l bound|max concurrent\b)/i.test(heading)) {
        return false;
      }
      if (selectedContextHeading && heading.startsWith('KV cache @')) {
        return heading === selectedContextHeading;
      }
      if (selectedFleetPrefix && heading.startsWith('Fleet tier:')) {
        if (!heading.startsWith(selectedFleetPrefix)) {
          return false;
        }
        return filter.gpuCount ? heading.includes(`(${filter.gpuCount} GPUs)`) : true;
      }
      return !/(\btok\/s\b|tokens per second|tokens-per-second|tok_per_sec|tokens_per_sec)/i.test(block);
    })
    .join('\n\n')
    .trim();
}

function explainContextLabel(tokens: number): string {
  if (tokens >= 1_000_000) {
    return tokens % 1_000_000 === 0 ? `${tokens / 1_000_000}M` : `${(tokens / 1_000_000).toFixed(1)}M`;
  }
  if (tokens >= 1024) {
    return `${Math.floor(tokens / 1024)}K`;
  }
  return String(tokens);
}

function numericValue(value: string): number | undefined {
  const parsed = Number(value);
  return Number.isFinite(parsed) && value.trim() ? parsed : undefined;
}

function draftModelWeightGb(sizeBillionParams: number): string {
  return ((sizeBillionParams * 1_000_000_000 * 2) / 1024 ** 3).toFixed(2);
}

function formatCompact(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1000) return `${value / 1000 >= 10 ? (value / 1000).toFixed(0) : (value / 1000).toFixed(1)}K`;
  return String(Math.round(value));
}

function parseCompact(value: string): number | undefined {
  const normalized = value.trim().toLowerCase();
  if (!normalized) return undefined;
  const multiplier = normalized.endsWith('m') ? 1_000_000 : normalized.endsWith('k') ? 1_000 : 1;
  const parsed = Number(normalized.replace(/[km]/g, ''));
  return Number.isFinite(parsed) ? parsed * multiplier : undefined;
}
