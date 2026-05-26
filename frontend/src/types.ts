export type Mode = "AUTO" | "REVIEW" | "MANUAL";
export type FilterKind = "all" | "codeMismatch" | "vatRisk";

export interface Candidate {
  code: string;
  score: number;
  name: string;
}

export interface PredictionResult {
  predicted_code: string | null;
  predicted_name?: string | null;
  confidence: number;
  mode: Mode;
  explanation: string;
  top_candidates: Candidate[];
  vat_rate: string | null;
  vat_exempt: boolean;
  closest_exempt_code: string | null;
  closest_exempt_name: string | null;
  closest_exempt_distance: number | null;
  is_ood: boolean;
  packaging_risk: boolean;
  current_code?: string | null;
  code_match?: boolean | null;
  vat_mismatch?: boolean | null;
}

export interface BatchRow extends PredictionResult {
  source_name: string;
  final_code: string | null;
}

export interface StatusResponse {
  models_loaded: boolean;
  loading: boolean;
  progress: number;
  error: string | null;
}

export interface CodeReferenceItem {
  code: string;
  name: string;
}

export interface BatchJobStartResponse {
  job_id: string;
  status: string;
  filename: string;
  total_rows: number;
}

export interface BatchJobStatus {
  job_id: string;
  filename: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  processed_rows: number;
  total_rows: number;
  error: string | null;
  logs: string[];
  created_at: number;
  updated_at: number;
}
