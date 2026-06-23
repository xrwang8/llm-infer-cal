import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Clipboard,
  Cpu,
  Gauge,
  Layers,
  Play,
  RefreshCcw,
  Settings2,
  Terminal,
  Zap,
} from 'lucide-react';
import { FormEvent, useEffect, useMemo, useState } from 'react';
import { evaluate, fetchGpus, fetchModels } from './api';
import {
  advancedSettings,
  annotatedValue,
  bestFleetOption,
  defaultRemoteModelId,
  formatBytes,
  formatFloat,
  formatMs,
  formatNumber,
  groupGpusByVendor,
  groupModelsByProvider,
  gpuVendorOptionLabel,
  gpuTier,
  gpuTierLabel,
  itemsForGroup,
  labelText,
  llmReviewSettings,
  modelVendorOptionLabel,
  performanceSettings,
  pct,
  tierText,
  type EvaluateForm,
  type FleetOption,
  type GpuSummary,
  type ModelSummary,
  type Report,
} from './report';

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
  llm_review: false,
  llm_review_api_key: '',
  llm_review_base_url: '',
  llm_review_model: '',
};

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
  { value: '16', label: 'FP16/BF16' },
  { value: '8', label: 'FP8/INT8' },
  { value: '4', label: 'INT4' },
] as const;

export function App() {
  const [form, setForm] = useState<EvaluateForm>(DEFAULT_FORM);
  const [activeTab, setActiveTab] = useState<'calculator' | 'compare'>('calculator');
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
  const selectedFleet = bestFleetOption(report?.fleet);
  const command = report?.generated_command?.command ?? '';
  const comparisonReports = report?.comparison?.reports ?? [];
  const kvRows = report?.kv_cache_by_context ?? [];
  const activationRows = report?.activation_by_context ?? [];
  const weightBytes = annotatedValue<number>(report?.weights?.safetensors_total_bytes);
  const params = annotatedValue<number>(report?.weights?.params_estimated);
  const activeParams = annotatedValue<number>(report?.weights?.active_params_estimated);
  const maxConcurrent = annotatedValue<number>(report?.performance?.max_concurrent);
  const prefillMs = annotatedValue<number>(report?.performance?.prefill?.latency_ms);
  const clusterTps = annotatedValue<number>(report?.performance?.decode?.cluster_tokens_per_sec);
  const engineSupport = report?.engine_compatibility?.support ?? 'unknown';
  const tuningSettings = performanceSettings();
  const advancedControls = advancedSettings();
  const reviewerSettings = llmReviewSettings();
  const compareGpuIds = useMemo(() => {
    const filtered = gpus.filter((gpu) => {
      const vendorOk = compareVendor === 'all' || gpu.vendor === compareVendor;
      const tierOk = compareTier === 'all' || gpuTier(gpu) === compareTier;
      return vendorOk && tierOk;
    });
    return filtered.map((gpu) => gpu.id).slice(0, 64);
  }, [gpus, compareVendor, compareTier]);

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
      if (!currentModelStillVisible) {
        return { ...current, source: value, model_id: '' };
      }
      return { ...current, source: value };
    });
  }

  function updateModelVendor(value: string) {
    setModelVendor(value);
    setForm((current) => ({ ...current, source: 'builtin', model_id: '' }));
  }

  function updateGpuVendor(value: string) {
    setGpuVendor(value);
  }

  function updatePrimaryGpu(gpuId: string) {
    setForm((current) => ({ ...current, gpu: gpuId, gpus: gpuId ? [gpuId] : [] }));
  }

  function updateGpuSelection(gpuId: string, checked: boolean) {
    setForm((current) => {
      const currentGpus = current.gpus.length ? current.gpus : current.gpu ? [current.gpu] : [];
      const nextGpus = checked
        ? [...currentGpus, gpuId].filter((gpu, index, all) => all.indexOf(gpu) === index).slice(0, 64)
        : currentGpus.filter((gpu) => gpu !== gpuId);
      const boundedGpus = nextGpus.length ? nextGpus : currentGpus;
      return { ...current, gpu: boundedGpus[0] ?? '', gpus: boundedGpus };
    });
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

  function copyCommand() {
    if (!command) return;
    void navigator.clipboard.writeText(command).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    });
  }

  return (
    <div className="appShell">
      <header className="topBar">
        <div className="brand">
          <div className="brandMark">
            <Cpu size={22} />
          </div>
          <div>
            <h1>llm-infer-cal</h1>
            <p>LLM 推理硬件计算器</p>
          </div>
        </div>
      </header>

      <nav className="viewTabs" aria-label="主视图">
        <button type="button" className={activeTab === 'calculator' ? 'active' : ''} onClick={() => setActiveTab('calculator')}>
          计算器
        </button>
        <button type="button" className={activeTab === 'compare' ? 'active' : ''} onClick={() => setActiveTab('compare')}>
          GPU 对比
        </button>
      </nav>

      {error ? (
        <div className="errorBanner">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      ) : null}

      <main className="workspace">
        <form className="controlPanel" onSubmit={submit}>
          <SectionHeader icon={<Settings2 size={18} />} title="配置" />

          <div className="fieldStack">
            <Segmented
              label="模型来源"
              options={sourceOptions}
              value={form.source}
              onChange={updateSource}
            />

            {form.source === 'builtin' ? (
              <div className="twoCol">
                <label className="field">
                  <span>模型厂商</span>
                  <select
                    data-testid="model-vendor-select"
                    value={modelVendor}
                    onChange={(event) => updateModelVendor(event.target.value)}
                  >
                    {modelGroups.map((group) => (
                      <option key={group.label} value={group.label}>
                        {modelVendorOptionLabel(group)}
                      </option>
                    ))}
                    {!modelGroups.length ? <option value={modelVendor}>{modelVendor}</option> : null}
                  </select>
                </label>

                <label className="field">
                  <span>内置模型</span>
                  <select
                    data-testid="builtin-model-select"
                    value={form.model_id}
                    onChange={(event) => updateField('model_id', event.target.value)}
                  >
                    <option value="">请选择模型</option>
                    {modelOptions.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.id}
                      </option>
                    ))}
                    {!modelOptions.length && form.model_id ? <option value={form.model_id}>{form.model_id}</option> : null}
                  </select>
                </label>
              </div>
            ) : (
              <label className="field">
                <span>模型 ID</span>
                <input
                  data-testid="remote-model-id-input"
                  value={form.model_id}
                  placeholder="例如 Qwen/Qwen3-30B-A3B"
                  onChange={(event) => updateField('model_id', event.target.value)}
                />
              </label>
            )}

            <div className="twoCol">
              <label className="field">
                <span>GPU 厂商</span>
                <select data-testid="gpu-vendor-select" value={gpuVendor} onChange={(event) => updateGpuVendor(event.target.value)}>
                  {gpuGroups.map((group) => (
                    <option key={group.label} value={group.label}>
                      {gpuVendorOptionLabel(group)}
                    </option>
                  ))}
                  {!gpuGroups.length ? <option value={gpuVendor}>{gpuVendor}</option> : null}
                </select>
              </label>

              <label className="field">
                <span>GPU 型号</span>
                <GpuPicker options={gpuOptions} selected={selectedGpuIds} onSelectPrimary={updatePrimaryGpu} onToggle={updateGpuSelection} />
              </label>
            </div>

            <div className="twoCol">
              <Segmented
                label="引擎"
                options={engineOptions}
                value={form.engine}
                onChange={(value) => updateField('engine', value)}
              />

              <NumberField
                label="上下文长度"
                value={form.context_length}
                onChange={(value) => updateField('context_length', value)}
              />
            </div>

            <details className="advancedPanel" data-testid="performance-settings">
              <summary>
                <span>
                  <ChevronDown className="disclosureIcon" size={16} />
                  性能参数
                </span>
                <strong>{tuningSettings.length}</strong>
              </summary>
              <div className="advancedGrid">
                {tuningSettings.map((setting) =>
                  setting.control === 'range' ? (
                    <RangeField
                      key={setting.key}
                      label={setting.label}
                      value={form[setting.key]}
                      min={setting.min ?? '0'}
                      max={setting.max ?? '1'}
                      step={setting.step ?? '0.05'}
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
              </div>
            </details>

            <details className="advancedPanel" data-testid="advanced-settings">
              <summary>
                <span>
                  <ChevronDown className="disclosureIcon" size={16} />
                  高级设置
                </span>
                <strong>{advancedControls.length}</strong>
              </summary>
              <div className="advancedGrid">
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
                <div className="advancedGroup">
                  <div className="advancedGroupTitle">Inference Optimizations</div>
                  <label className="field">
                    <span>KV Cache 精度</span>
                    <select value={form.kv_cache_bits} onChange={(event) => updateField('kv_cache_bits', event.target.value)}>
                      {kvPrecisionOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <CheckboxField
                    label="Paged Attention（KV 显存 × 0.75）"
                    checked={form.paged_attention}
                    onChange={(value) => updateField('paged_attention', value)}
                  />
                </div>
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
          </div>

          <button className="primaryButton" type="submit" disabled={evaluating || !form.model_id.trim() || !selectedGpuIds.length}>
            {evaluating ? <RefreshCcw className="spin" size={18} /> : <Play size={18} />}
            <span>{evaluating ? '计算中' : '开始评估'}</span>
          </button>
        </form>

        <section className="resultSurface">
          <div className="resultHeader">
            <div>
              <p className="eyebrow">{report?.model?.source ?? form.source}</p>
              <h2>{report?.model?.id ?? form.model_id}</h2>
            </div>
            <div className="resultBadges">
              <LabelBadge label={engineSupport === 'full' ? '完整支持' : engineSupport} tone={engineSupport === 'full' ? 'good' : 'warn'} />
              <LabelBadge label={selectedGpu?.fp8_support ? 'FP8' : 'BF16/FP16'} tone="neutral" />
            </div>
          </div>

          {activeTab === 'compare' ? (
            <section className="panel comparisonPanel">
              <SectionHeader icon={<Activity size={18} />} title="GPU 对比" />
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
                disabled={evaluating || !form.model_id.trim()}
              />
              <ComparisonTable reports={comparisonReports} />
            </section>
          ) : null}

          <div className="metricGrid">
            <Metric icon={<Cpu size={18} />} label="推荐 GPU" value={selectedFleet?.gpu_count ? `${selectedFleet.gpu_count} 张` : '-'} detail={tierText(selectedFleet?.tier)} />
            <Metric icon={<Activity size={18} />} label="并发上限" value={formatNumber(maxConcurrent)} detail="K/L 取小" />
            <Metric icon={<Layers size={18} />} label="权重显存" value={formatBytes(weightBytes)} detail={labelText(report?.weights?.safetensors_total_bytes?.label)} />
            <Metric icon={<Gauge size={18} />} label="参数量" value={formatNumber(params)} detail={labelText(report?.weights?.params_estimated?.label)} />
            <Metric icon={<Zap size={18} />} label="Decode 集群" value={`${formatFloat(clusterTps, 1)} tok/s`} detail="估算吞吐" />
            <Metric icon={<RefreshCcw size={18} />} label="Prefill" value={formatMs(prefillMs)} detail={`${form.input_tokens || 2000} tokens`} />
          </div>

          {comparisonReports.length >= 2 ? (
            <section className="panel comparisonPanel">
              <SectionHeader icon={<Activity size={18} />} title="GPU 对比" />
              <ComparisonTable reports={comparisonReports} />
            </section>
          ) : null}

          <div className="contentGrid">
            <section className="panel">
              <SectionHeader icon={<Layers size={18} />} title="VRAM Breakdown" />
              <MemoryBreakdown option={selectedFleet} hardware={report?.hardware ?? selectedGpu} />
              <div className="kvList">
                {kvRows.map((row) => (
                  <div className="kvRow" key={row.context_tokens}>
                    <span>{formatNumber(row.context_tokens)} ctx</span>
                    <strong>{formatBytes(row.bytes?.value)}</strong>
                    <small>{labelText(row.bytes?.label)}</small>
                  </div>
                ))}
                {activationRows.map((row) => (
                  <div className="kvRow" key={`activation-${row.context_tokens}`}>
                    <span>{formatNumber(row.context_tokens)} ctx activation</span>
                    <strong>{formatBytes(row.bytes?.value)}</strong>
                    <small>{labelText(row.bytes?.label)}</small>
                  </div>
                ))}
              </div>
            </section>

            <section className="panel">
              <SectionHeader icon={<Activity size={18} />} title="方案表" />
              <FleetTable options={report?.fleet?.options ?? []} />
              <p className="noteText">{report?.fleet?.constraint_note_zh ?? report?.fleet?.constraint_note_en ?? selectedGpu?.notes_zh}</p>
            </section>
          </div>

          <div className="contentGrid">
            <section className="panel">
              <SectionHeader icon={<Gauge size={18} />} title="Formula Reference" />
              <FormulaReference report={report} form={form} activeParams={activeParams} />
            </section>

            <section className="panel">
              <SectionHeader icon={<Terminal size={18} />} title="启动命令" />
              <div className="commandBox">
                <pre>{command || '等待评估结果'}</pre>
                <button type="button" className="iconButton" onClick={copyCommand} disabled={!command} aria-label="复制启动命令">
                  {copied ? <CheckCircle2 size={18} /> : <Clipboard size={18} />}
                </button>
              </div>
              <OptimizationList report={report} />
            </section>
          </div>
        </section>
      </main>
    </div>
  );
}

function SectionHeader({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="sectionHeader">
      {icon}
      <h3>{title}</h3>
    </div>
  );
}

function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="field">
      <span>{label}</span>
      <div className="segmented">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            className={value === option.value ? 'active' : ''}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function GpuPicker({
  options,
  selected,
  onSelectPrimary,
  onToggle,
}: {
  options: GpuSummary[];
  selected: string[];
  onSelectPrimary: (gpuId: string) => void;
  onToggle: (gpuId: string, checked: boolean) => void;
}) {
  const visibleOptions: GpuSummary[] = options.length ? options : selected.map((id) => ({ id }));
  const selectedSet = new Set(selected);

  return (
    <div className="gpuPicker" data-testid="gpu-model-picker">
      <select value={selected[0] ?? ''} onChange={(event) => onSelectPrimary(event.target.value)}>
        {visibleOptions.map((gpu) => (
          <option key={gpu.id} value={gpu.id}>
            {gpu.id}
            {gpu.memory_gb ? ` · ${gpu.memory_gb}GB` : ''}
          </option>
        ))}
      </select>
      <div className="selectedGpuList">
        {selected.map((gpu) => (
          <button
            className="selectedGpuChip"
            key={gpu}
            type="button"
            disabled={selectedSet.has(gpu) && selected.length <= 1}
            onClick={() => onToggle(gpu, false)}
            title="从当前选择中移除"
          >
            {gpu}
          </button>
        ))}
      </div>
    </div>
  );
}

function NumberField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input inputMode="decimal" value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
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
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
    </label>
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

function RangeField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: string;
  min: string;
  max: string;
  step: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="field rangeField">
      <span>
        {label}
        <strong>{value}</strong>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function LabelBadge({ label, tone }: { label: string; tone: 'good' | 'warn' | 'neutral' }) {
  return <span className={`labelBadge ${tone}`}>{label}</span>;
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
      <label className="field">
        <span>Provider</span>
        <select value={vendor} onChange={(event) => onVendor(event.target.value)}>
          <option value="all">全部厂商</option>
          {vendors.map((nextVendor) => (
            <option key={nextVendor} value={nextVendor}>
              {nextVendor}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Tier</span>
        <select value={tier} onChange={(event) => onTier(event.target.value)}>
          <option value="all">{gpuTierLabel('all')}</option>
          <option value="datacenter">{gpuTierLabel('datacenter')}</option>
          <option value="prosumer">{gpuTierLabel('prosumer')}</option>
          <option value="consumer">{gpuTierLabel('consumer')}</option>
        </select>
      </label>
      <label className="field">
        <span>GPUs per node</span>
        <select value={gpuCount} onChange={(event) => onGpuCount(event.target.value)}>
          <option value="">自动推荐</option>
          {[1, 2, 4, 8, 16, 32, 64].map((count) => (
            <option key={count} value={String(count)}>
              {count}× GPU
            </option>
          ))}
        </select>
      </label>
      <button className="secondaryButton" type="button" onClick={onRun} disabled={disabled || selectedCount === 0}>
        对比 {selectedCount} 张 GPU
      </button>
    </div>
  );
}

function Metric({ icon, label, value, detail }: { icon: React.ReactNode; label: string; value: string; detail: string }) {
  return (
    <div className="metric">
      <div className="metricIcon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </div>
  );
}

function MemoryBreakdown({ option, hardware }: { option?: FleetOption; hardware?: GpuSummary | null }) {
  const memoryBytes = (hardware?.memory_gb ?? 0) * 1_000_000_000;
  const concurrent = option?.tier_concurrent_requests ?? 1;
  const weight = option?.weight_bytes_per_gpu ?? 0;
  const kv = (option?.kv_bytes_per_request_per_gpu ?? option?.kv_bytes_per_request ?? 0) * concurrent;
  const activation = (option?.activation_bytes_per_request_per_gpu ?? 0) * concurrent;
  const reserved = option?.reserved_bytes_per_gpu ?? Math.max(memoryBytes - (option?.usable_bytes_per_gpu ?? 0), 0);
  const required = option?.required_bytes_per_gpu_at_tier ?? weight + kv + activation;

  return (
    <div className="memoryBars">
      <div className="breakdownSummary">
        <span>Required / GPU</span>
        <strong>{formatBytes(required)}</strong>
      </div>
      <Bar label="Weights / GPU" value={weight} total={memoryBytes} tone="weight" />
      <Bar label={`KV Cache × ${concurrent}`} value={kv} total={memoryBytes} tone="kv" />
      <Bar label={`Activations × ${concurrent}`} value={activation} total={memoryBytes} tone="activation" />
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

function ComparisonTable({ reports }: { reports: Report[] }) {
  if (!reports.length) {
    return <p className="emptyText">选择过滤条件后点击对比。</p>;
  }

  const sorted = [...reports].sort((a, b) => {
    const aTps = annotatedValue<number>(a.performance?.decode?.cluster_tokens_per_sec) ?? 0;
    const bTps = annotatedValue<number>(b.performance?.decode?.cluster_tokens_per_sec) ?? 0;
    return bTps - aTps;
  });

  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>GPU</th>
            <th>Min GPUs</th>
            <th>GPU VRAM</th>
            <th>TTFT</th>
            <th>Tok/s</th>
            <th>Total Tok/s</th>
            <th>Mem BW</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((item) => {
            const option = bestFleetOption(item.fleet);
            const concurrent = annotatedValue<number>(item.performance?.max_concurrent);
            const decode = annotatedValue<number>(item.performance?.decode?.cluster_tokens_per_sec);
            const prefill = annotatedValue<number>(item.performance?.prefill?.latency_ms);
            const perUserTps = decode && concurrent ? decode / concurrent : decode;
            return (
              <tr key={item.hardware?.id ?? item.generated_command?.command}>
                <td>{item.hardware?.id ?? '-'}</td>
                <td>{option?.gpu_count ? `${option.gpu_count} 张` : '-'}</td>
                <td>{item.hardware?.memory_gb ? `${item.hardware.memory_gb}GB` : '-'}</td>
                <td>{formatMs(prefill)}</td>
                <td>{`${formatFloat(perUserTps, 1)} tok/s`}</td>
                <td>{`${formatFloat(decode, 0)} tok/s`}</td>
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

function FormulaReference({ report, form, activeParams }: { report: Report | null; form: EvaluateForm; activeParams?: number }) {
  const params = annotatedValue<number>(report?.weights?.params_estimated);
  const weight = annotatedValue<number>(report?.weights?.safetensors_total_bytes);
  const kv = report?.kv_cache_by_context?.at(-1);
  const activation = report?.activation_by_context?.at(-1);
  const isMoe = !!params && !!activeParams && activeParams < params;
  const kvBits = report?.inference_options?.kv_cache_bits ?? Number(form.kv_cache_bits || 16);
  const paged = report?.inference_options?.paged_attention ?? form.paged_attention;

  return (
    <div className="formulaList">
      <details open>
        <summary>Model Weights VRAM</summary>
        <pre>
{`weights = safetensors_total_bytes
${weight ? `observed = ${formatBytes(weight)}` : 'observed = waiting for report'}
${isMoe ? 'MoE: all expert weights stay resident in VRAM.' : 'Dense: all weights participate in memory and compute.'}`}
        </pre>
      </details>
      <details open>
        <summary>KV Cache VRAM</summary>
        <pre>
{`kv = architecture_kv_shape * seq_len * layers * ${kvBits} bits
${paged ? 'paged_attention factor = 0.75' : 'paged_attention factor = 1.00'}
${kv?.bytes?.value ? `selected = ${formatBytes(kv.bytes.value)}` : 'selected = waiting for report'}`}
        </pre>
      </details>
      <details>
        <summary>Activations</summary>
        <pre>
{`activation ~= seq_len * hidden_size * 2B
${isMoe ? 'MoE routing factor is included.' : 'Dense activation estimate.'}
${activation?.bytes?.value ? `selected = ${formatBytes(activation.bytes.value)}` : 'selected = waiting for report'}`}
        </pre>
      </details>
      <details>
        <summary>TTFT / Tokens Per Second</summary>
        <pre>
{`ttft_flops = 2 * ${isMoe ? 'active_params' : 'params'} * input_tokens
decode_tps = effective_bandwidth / active_weight_bytes_per_gpu
${isMoe ? `MoE active params = ${formatNumber(activeParams)} / total ${formatNumber(params)}` : ''}`}
        </pre>
      </details>
    </div>
  );
}

function EvidenceList({ report }: { report: Report | null }) {
  const items = [
    {
      label: '权重字节',
      value: report?.weights?.safetensors_total_bytes?.source,
    },
    {
      label: '参数估算',
      value: report?.weights?.params_estimated?.source,
    },
    {
      label: 'KV Cache',
      value: report?.kv_cache_by_context?.at(-1)?.bytes?.source,
    },
    {
      label: '并发上限',
      value: report?.performance?.max_concurrent?.source,
    },
  ].filter((item) => item.value);
  const explainText = report?.explain_text?.trim();
  const reviewText = report?.llm_review_text?.trim();

  if (!items.length && !explainText && !reviewText) {
    return <p className="emptyText">等待评估结果</p>;
  }

  return (
    <div className="evidenceList">
      {items.map((item) => (
        <div className="evidenceItem" key={item.label}>
          <span>{item.label}</span>
          <p>{item.value}</p>
        </div>
      ))}
      {explainText ? <pre className="longTextBlock">{explainText}</pre> : null}
      {reviewText ? <pre className="longTextBlock">{reviewText}</pre> : null}
    </div>
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
