import { buildEvaluatePayload, type EvaluateForm, type GpuSummary, type ModelSummary, type Report } from './report';

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8080';

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchModels(): Promise<ModelSummary[]> {
  const body = await getJson<{ models: ModelSummary[] }>('/api/models');
  return body.models ?? [];
}

export async function fetchGpus(): Promise<GpuSummary[]> {
  const body = await getJson<{ gpus: GpuSummary[] }>('/api/gpus');
  return body.gpus ?? [];
}

export async function evaluate(form: EvaluateForm): Promise<Report> {
  const response = await fetch(`${API_BASE}/api/evaluate`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
    },
    body: JSON.stringify(buildEvaluatePayload(form)),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body?.error?.message ?? `${response.status} ${response.statusText}`);
  }
  return body as Report;
}
