import type {
  BatchJobStartResponse,
  BatchJobStatus,
  BatchRow,
  CodeReferenceItem,
  PredictionResult,
  StatusResponse
} from "../types";

function resolveApiBase(): string {
  const configured = import.meta.env.VITE_API_BASE;
  if (configured) {
    return configured;
  }

  if (typeof window !== "undefined" && window.location.port === "5173") {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }

  return "/api";
}

export const API_BASE = resolveApiBase();

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchStatus(): Promise<StatusResponse> {
  return readJson<StatusResponse>(await fetch(`${API_BASE}/status`));
}

export async function predict(query: string, useReranker = false): Promise<PredictionResult> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, use_reranker: useReranker })
  });
  return readJson<PredictionResult>(response);
}

export async function batchPredict(file: File, useReranker = false): Promise<BatchRow[]> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("use_reranker", String(useReranker));
  const response = await fetch(`${API_BASE}/batch_predict`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const blob = await response.blob();
  const parsed = await convertWorkbookToRows(blob);
  return parsed;
}

async function convertWorkbookToRows(blob: Blob): Promise<BatchRow[]> {
  const XLSX = await import("xlsx");
  const arrayBuffer = await blob.arrayBuffer();
  const workbook = XLSX.read(arrayBuffer, { type: "array" });
  const worksheet = workbook.Sheets[workbook.SheetNames[0]];
  const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(worksheet);
  return rows.map((row) => ({
    source_name: String(row["Исходное название"] ?? row["РСЃС…РѕРґРЅРѕРµ РЅР°Р·РІР°РЅРёРµ"] ?? ""),
    predicted_code: toNullableString(row["Предсказанный код"] ?? row["РџСЂРµРґСЃРєР°Р·Р°РЅРЅС‹Р№ РєРѕРґ"]),
    final_code: toNullableString(
      row["Финальный код"] ??
      row["Р¤РёРЅР°Р»СЊРЅС‹Р№ РєРѕРґ"] ??
      row["Предсказанный код"] ??
      row["РџСЂРµРґСЃРєР°Р·Р°РЅРЅС‹Р№ РєРѕРґ"]
    ),
    confidence: Number(row["Уверенность"] ?? row["РЈРІРµСЂРµРЅРЅРѕСЃС‚СЊ"] ?? 0),
    mode: String(row["Режим"] ?? row["Р РµР¶РёРј"] ?? "MANUAL") as BatchRow["mode"],
    explanation: "",
    top_candidates: [],
    vat_rate: toNullableString(row["Ставка НДС"] ?? row["РЎС‚Р°РІРєР° РќР”РЎ"]),
    vat_exempt: String(row["Ставка НДС"] ?? row["РЎС‚Р°РІРєР° РќР”РЎ"] ?? "") === "10%",
    closest_exempt_code: toNullableString(row["Ближайший льготный код"] ?? row["Р‘Р»РёР¶Р°Р№С€РёР№ Р»СЊРіРѕС‚РЅС‹Р№ РєРѕРґ"]),
    closest_exempt_name: toNullableString(row["Название ближайшего льготного кода"] ?? row["РќР°Р·РІР°РЅРёРµ Р±Р»РёР¶Р°Р№С€РµРіРѕ Р»СЊРіРѕС‚РЅРѕРіРѕ РєРѕРґР°"]),
    closest_exempt_distance: toNullableNumber(row["Расстояние до льготного"] ?? row["Р Р°СЃСЃС‚РѕСЏРЅРёРµ РґРѕ Р»СЊРіРѕС‚РЅРѕРіРѕ"]),
    is_ood: false,
    packaging_risk: false,
    current_code: toNullableString(row["Текущий код"] ?? row["РўРµРєСѓС‰РёР№ РєРѕРґ"]),
    code_match: toNullableBoolean(row["Совпадение"] ?? row["РЎРѕРІРїР°РґРµРЅРёРµ"]),
    vat_mismatch: toNullableBoolean(row["Расхождение НДС"] ?? row["Р Р°СЃС…РѕР¶РґРµРЅРёРµ РќР”РЎ"])
  }));
}

export async function batchPredictJson(rows: Array<{ query: string; current_code?: string | null }>, useReranker = false): Promise<BatchRow[]> {
  const response = await fetch(`${API_BASE}/batch_predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items: rows, use_reranker: useReranker })
  });
  const payload = await readJson<{ rows: BatchRow[] }>(response);
  return payload.rows.map((row) => ({
    ...row,
    final_code: row.final_code ?? row.predicted_code
  }));
}

export async function fetchCodeReference(): Promise<CodeReferenceItem[]> {
  const response = await fetch(`${API_BASE}/reference/codes`);
  const payload = await readJson<{ items: CodeReferenceItem[] }>(response);
  return payload.items;
}

export async function startBatchUploadJob(file: File, useReranker = false): Promise<BatchJobStartResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("use_reranker", String(useReranker));
  const response = await fetch(`${API_BASE}/batch_jobs/upload`, {
    method: "POST",
    body: formData
  });
  return readJson<BatchJobStartResponse>(response);
}

export async function fetchBatchJobStatus(jobId: string): Promise<BatchJobStatus> {
  return readJson<BatchJobStatus>(await fetch(`${API_BASE}/batch_jobs/${jobId}`));
}

export async function fetchBatchJobResults(jobId: string): Promise<BatchRow[]> {
  const response = await fetch(`${API_BASE}/batch_jobs/${jobId}/results`);
  const payload = await readJson<{ rows: BatchRow[] }>(response);
  return payload.rows.map((row) => ({
    ...row,
    final_code: row.final_code ?? row.predicted_code
  }));
}

function toNullableString(value: unknown): string | null {
  if (value === undefined || value === null || value === "") return null;
  return String(value);
}

function toNullableNumber(value: unknown): number | null {
  if (value === undefined || value === null || value === "") return null;
  return Number(value);
}

function toNullableBoolean(value: unknown): boolean | null {
  if (value === undefined || value === null || value === "") return null;
  if (typeof value === "boolean") return value;
  return String(value).toLowerCase() === "true";
}
