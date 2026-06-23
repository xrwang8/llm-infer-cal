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

export function App() {
  const [form, setForm] = useState<EvaluateForm>(DEFAULT_FORM);
  const [modelVendor, setModelVendor] = useState('Qwen');
  const [gpuVendor, setGpuVendor] = useState('NVIDIA');
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
  const weightBytes = annotatedValue<number>(report?.weights?.safetensors_total_bytes);
  const params = annotatedValue<number>(report?.weights?.params_estimated);
  const maxConcurrent = annotatedValue<number>(report?.performance?.max_concurrent);
  const prefillMs = annotatedValue<number>(report?.performance?.prefill?.latency_ms);
  const clusterTps = annotatedValue<number>(report?.performance?.decode?.cluster_tokens_per_sec);
  const engineSupport = report?.engine_compatibility?.support ?? 'unknown';
  const tuningSettings = performanceSettings();
  const advancedControls = advancedSettings();
  const reviewerSettings = llmReviewSettings();

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

  function updateGpuSelection(gpuId: string, checked: boolean) {
    setForm((current) => {
      const currentGpus = current.gpus.length ? current.gpus : current.gpu ? [current.gpu] : [];
      const nextGpus = checked
        ? [...currentGpus, gpuId].filter((gpu, index, all) => all.indexOf(gpu) === index).slice(0, 4)
        : currentGpus.filter((gpu) => gpu !== gpuId);
      const boundedGpus = nextGpus.length ? nextGpus : currentGpus;
      return { ...current, gpu: boundedGpus[0] ?? '', gpus: boundedGpus };
    });
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void runEvaluation();
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
                <GpuPicker options={gpuOptions} selected={selectedGpuIds} onToggle={updateGpuSelection} />
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
              <SectionHeader icon={<Layers size={18} />} title="显存拆解" />
              <MemoryBreakdown option={selectedFleet} hardware={report?.hardware ?? selectedGpu} />
              <div className="kvList">
                {kvRows.map((row) => (
                  <div className="kvRow" key={row.context_tokens}>
                    <span>{formatNumber(row.context_tokens)} ctx</span>
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
              <SectionHeader icon={<Gauge size={18} />} title="公式依据" />
              <EvidenceList report={report} />
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
  onToggle,
}: {
  options: GpuSummary[];
  selected: string[];
  onToggle: (gpuId: string, checked: boolean) => void;
}) {
  const visibleOptions: GpuSummary[] = options.length ? options : selected.map((id) => ({ id }));
  const selectedSet = new Set(selected);
  const atLimit = selected.length >= 4;

  return (
    <div className="gpuPicker" data-testid="gpu-model-picker">
      <div className="gpuOptionGrid">
        {visibleOptions.map((gpu) => {
          const checked = selectedSet.has(gpu.id);
          const disabled = (!checked && atLimit) || (checked && selected.length <= 1);
          return (
            <label className={`gpuOption ${checked ? 'active' : ''}`} key={gpu.id}>
              <input
                type="checkbox"
                checked={checked}
                disabled={disabled}
                onChange={(event) => onToggle(gpu.id, event.target.checked)}
              />
              <span>{gpu.id}</span>
              <small>{gpu.memory_gb ? `${gpu.memory_gb}GB` : ''}</small>
            </label>
          );
        })}
      </div>
      <div className="selectedGpuList">
        {selected.map((gpu) => (
          <span className="selectedGpuChip" key={gpu}>
            {gpu}
          </span>
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
  const weight = option?.weight_bytes_per_gpu ?? 0;
  const usable = option?.usable_bytes_per_gpu ?? 0;
  const headroom = Math.max(usable - weight, 0);
  const reserved = Math.max(memoryBytes - usable, 0);

  return (
    <div className="memoryBars">
      <Bar label="权重 / GPU" value={weight} total={memoryBytes} tone="weight" />
      <Bar label="KV 余量 / GPU" value={headroom} total={memoryBytes} tone="kv" />
      <Bar label="预留" value={reserved} total={memoryBytes} tone="reserved" />
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
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>GPU</th>
            <th>显存</th>
            <th>推荐张数</th>
            <th>并发</th>
            <th>Decode</th>
            <th>Prefill</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody>
          {reports.map((item) => {
            const option = bestFleetOption(item.fleet);
            const concurrent = annotatedValue<number>(item.performance?.max_concurrent);
            const decode = annotatedValue<number>(item.performance?.decode?.cluster_tokens_per_sec);
            const prefill = annotatedValue<number>(item.performance?.prefill?.latency_ms);
            return (
              <tr key={item.hardware?.id ?? item.generated_command?.command}>
                <td>{item.hardware?.id ?? '-'}</td>
                <td>{item.hardware?.memory_gb ? `${item.hardware.memory_gb}GB` : '-'}</td>
                <td>{option?.gpu_count ? `${option.gpu_count} 张` : '-'}</td>
                <td>{formatNumber(concurrent)}</td>
                <td>{`${formatFloat(decode, 1)} tok/s`}</td>
                <td>{formatMs(prefill)}</td>
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
