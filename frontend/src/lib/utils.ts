import type { BatchRow, FilterKind } from "../types";

export function cn(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

export function modeTone(mode: string): string {
  if (mode === "AUTO") return "text-success border-success/40 bg-success/10";
  if (mode === "REVIEW") return "text-warning border-warning/40 bg-warning/10";
  return "text-danger border-danger/40 bg-danger/10";
}

export function filterRows(rows: BatchRow[], filter: FilterKind): BatchRow[] {
  if (filter === "codeMismatch") {
    return rows.filter((row) => row.code_match === false);
  }
  if (filter === "vatRisk") {
    return rows.filter((row) => row.vat_mismatch === true);
  }
  return rows;
}

export function confidencePercent(confidence: number): number {
  return Math.max(0, Math.min(100, Math.round(confidence * 100)));
}

export async function downloadRowsAsExcel(rows: BatchRow[], fileName: string): Promise<void> {
  const XLSX = await import("xlsx");
  const data = rows.map((row) => ({
    "Исходное название": row.source_name,
    "Предсказанный код": row.predicted_code,
    "Финальный код": row.final_code,
    "Уверенность": row.confidence,
    "Режим": row.mode,
    "Ставка НДС": row.vat_rate,
    "Расхождение НДС": row.vat_mismatch,
    "Текущий код": row.current_code,
    "Совпадение": row.code_match,
    "Ближайший льготный код": row.closest_exempt_code,
    "Название ближайшего льготного кода": row.closest_exempt_name,
    "Расстояние до льготного": row.closest_exempt_distance
  }));
  const worksheet = XLSX.utils.json_to_sheet(data);
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, "results");
  XLSX.writeFile(workbook, fileName);
}
