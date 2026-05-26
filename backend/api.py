from __future__ import annotations

import io
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.models.engine import DecisionEngine
from backend.models.retriever import Retriever
from config.settings import MODELS_DIR, REFERENCE_DIR


REFERENCE_OKPD_PATH = REFERENCE_DIR / "okpd_2.xlsx"
REFERENCE_VAT_PATH = REFERENCE_DIR / "vat_exempt_codes.xlsx"
REFERENCE_OKPD_CACHE_PATH = REFERENCE_DIR / "okpd_2.cached.csv"
REFERENCE_VAT_CACHE_PATH = REFERENCE_DIR / "vat_exempt_codes.cached.csv"
CLASSIFIER_PATH = MODELS_DIR / "berta_classifier_improved"
RETRIEVER_MODEL_PATH = str(MODELS_DIR / "bge-m3-frozen-3epoch")
DEFAULT_TEXT_COLUMNS = ("Номенклатура", "Наименование", "Название", "name", "query")
DEFAULT_CODE_COLUMNS = ("Код ОКПД2", "Код ОКПД 2", "OKPD2", "code", "current_code")


class PredictRequest(BaseModel):
    query: str = Field(..., min_length=1)
    use_reranker: bool = False


class BatchJsonItem(BaseModel):
    query: str = Field(..., min_length=1)
    current_code: str | None = None


class BatchJsonRequest(BaseModel):
    items: list[BatchJsonItem]
    use_reranker: bool = False


class ExportRow(BaseModel):
    source_name: str
    predicted_code: str | None = None
    confidence: float | None = None
    mode: str | None = None
    vat_rate: str | None = None
    vat_risk: bool | None = None
    closest_exempt_code: str | None = None
    closest_exempt_name: str | None = None
    closest_exempt_distance: float | None = None
    current_code: str | None = None
    code_match: bool | None = None
    vat_mismatch: bool | None = None
    final_code: str | None = None


class ExportRequest(BaseModel):
    rows: list[ExportRow]


@dataclass
class BatchJob:
    job_id: str
    filename: str
    use_reranker: bool = False
    status: str = "pending"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    total_rows: int = 0
    processed_rows: int = 0
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        self.updated_at = time.time()

    def set_progress(self, value: float) -> None:
        self.progress = max(0.0, min(1.0, value))
        self.updated_at = time.time()


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    ready_event: threading.Event = field(default_factory=threading.Event)
    batch_lock: threading.Lock = field(default_factory=threading.Lock)
    models_loaded: bool = False
    loading: bool = False
    progress: float = 0.0
    error: str | None = None
    retriever: Retriever | None = None
    engine: DecisionEngine | None = None
    okpd_df: pd.DataFrame | None = None
    code_to_name: dict[str, str] = field(default_factory=dict)
    exempt_codes: list[dict[str, str]] = field(default_factory=list)
    batch_jobs: dict[str, BatchJob] = field(default_factory=dict)

    def set_progress(self, progress: float, error: str | None = None) -> None:
        self.progress = max(0.0, min(1.0, progress))
        self.error = error


state = AppState()

app = FastAPI(title="OKPD-2 Classifier V3 API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).replace("\xa0", " ").strip()


def normalize_code(code: Any) -> str:
    return normalize_text(code).replace(" ", "").rstrip(".")


def parse_code_segments(code: str) -> list[str]:
    return [segment for segment in normalize_code(code).split(".") if segment]


def find_first_existing_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {normalize_text(column).lower(): column for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match:
            return match
    return None


def vat_is_exempt(code: str) -> bool:
    normalized = normalize_code(code)
    if not normalized:
        return False
    for item in state.exempt_codes:
        exempt_code = item["code"]
        if normalized == exempt_code:
            return True
        if normalized.startswith(exempt_code + ".") or exempt_code.startswith(normalized + "."):
            return True
    return False


def code_distance(code_a: str, code_b: str) -> float:
    parts_a = parse_code_segments(code_a)
    parts_b = parse_code_segments(code_b)
    if not parts_a or not parts_b:
        return 1.0
    shared = 0
    for left, right in zip(parts_a, parts_b):
        if left != right:
            break
        shared += 1
    depth = max(len(parts_a), len(parts_b))
    return round(1.0 - (shared / depth), 4)


def closest_exempt_info(code: str) -> dict[str, Any]:
    normalized = normalize_code(code)
    if not normalized or not state.exempt_codes:
        return {
            "closest_exempt_code": None,
            "closest_exempt_name": None,
            "closest_exempt_distance": None,
        }
    best = min(state.exempt_codes, key=lambda item: (code_distance(normalized, item["code"]), item["code"]))
    return {
        "closest_exempt_code": best["code"],
        "closest_exempt_name": best["name"],
        "closest_exempt_distance": code_distance(normalized, best["code"]),
    }


def get_vat_info(code: str) -> dict[str, Any]:
    exempt = vat_is_exempt(code)
    return {
        "vat_rate": "10%" if exempt else "20%",
        "vat_exempt": exempt,
        **closest_exempt_info(code),
    }


def enrich_prediction(result: dict[str, Any], current_code: str | None = None) -> dict[str, Any]:
    predicted_code = normalize_code(result.get("predicted_code"))
    vat_info = get_vat_info(predicted_code) if predicted_code else {
        "vat_rate": None,
        "vat_exempt": False,
        "closest_exempt_code": None,
        "closest_exempt_name": None,
        "closest_exempt_distance": None,
    }
    normalized_current = normalize_code(current_code)
    current_vat_info = get_vat_info(normalized_current) if normalized_current else None
    code_match = None
    vat_mismatch = None
    if normalized_current and predicted_code:
        code_match = normalized_current == predicted_code
        vat_mismatch = (
            current_vat_info is not None
            and current_vat_info["vat_rate"] != vat_info["vat_rate"]
        )
    source_name = None
    if predicted_code:
        source_name = state.code_to_name.get(predicted_code)

    return {
        **result,
        "predicted_code": predicted_code or None,
        "predicted_name": source_name,
        "current_code": normalized_current or None,
        "code_match": code_match,
        "vat_mismatch": vat_mismatch,
        **vat_info,
    }


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    buffer.seek(0)
    return buffer.getvalue()


def load_cached_reference(source_path: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists() and cache_path.stat().st_mtime >= source_path.stat().st_mtime:
        return pd.read_csv(cache_path, dtype=str)

    df = pd.read_excel(source_path, dtype=str)
    try:
        df.to_csv(cache_path, index=False)
    except OSError:
        # Кэш — это оптимизация. Если окружение не даёт писать файл, продолжаем без него.
        pass
    return df


def prepare_reference_data() -> None:
    state.set_progress(0.12)
    okpd_df = load_cached_reference(REFERENCE_OKPD_PATH, REFERENCE_OKPD_CACHE_PATH)
    okpd_df = okpd_df.rename(columns={column: normalize_text(column) for column in okpd_df.columns})
    okpd_df["code"] = okpd_df["code"].map(normalize_code)
    okpd_df["name"] = okpd_df["name"].map(normalize_text)
    okpd_df = okpd_df[(okpd_df["code"] != "") & (okpd_df["name"] != "")]
    okpd_df = okpd_df.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)

    state.set_progress(0.22)
    exempt_df = load_cached_reference(REFERENCE_VAT_PATH, REFERENCE_VAT_CACHE_PATH)
    exempt_df = exempt_df.rename(columns={column: normalize_text(column) for column in exempt_df.columns})
    exempt_df["code"] = exempt_df["code"].map(normalize_code)
    exempt_df["name"] = exempt_df["name"].map(normalize_text)
    exempt_df = exempt_df[(exempt_df["code"] != "") & (exempt_df["name"] != "")]
    exempt_df = exempt_df.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)

    state.okpd_df = okpd_df
    state.code_to_name = dict(zip(okpd_df["code"], okpd_df["name"], strict=False))
    state.exempt_codes = exempt_df[["code", "name"]].to_dict(orient="records")
    state.set_progress(0.35)


def build_engine() -> DecisionEngine:
    retriever = Retriever(model_name=RETRIEVER_MODEL_PATH)
    retriever._lazy_load()
    engine = DecisionEngine(retriever=retriever, classifier_path=CLASSIFIER_PATH)
    state.retriever = retriever
    return engine


def load_models_sync() -> None:
    with state.lock:
        if state.models_loaded:
            return
        if state.loading:
            return
        state.loading = True
        state.ready_event.clear()
        state.set_progress(0.02, error=None)

    try:
        state.set_progress(0.1)
        prepare_reference_data()
        state.set_progress(0.35)
        engine = build_engine()
        state.set_progress(0.95)
        state.engine = engine
        state.models_loaded = True
        state.set_progress(1.0)
    except Exception as exc:
        state.models_loaded = False
        state.set_progress(0.0, error=str(exc))
        raise
    finally:
        state.loading = False
        state.ready_event.set()


def start_background_loading() -> None:
    with state.lock:
        if state.models_loaded or state.loading:
            return
        state.loading = True
        state.ready_event.clear()
        state.set_progress(0.01, error=None)

    def runner() -> None:
        try:
            state.set_progress(0.1)
            prepare_reference_data()
            state.set_progress(0.55)
            engine = build_engine()
            state.engine = engine
            state.models_loaded = True
            state.set_progress(1.0)
        except Exception as exc:
            state.models_loaded = False
            state.set_progress(0.0, error=str(exc))
        finally:
            state.loading = False
            state.ready_event.set()

    thread = threading.Thread(target=runner, daemon=True, name="okpd-model-loader")
    thread.start()


def ensure_initialized(wait: bool = True) -> DecisionEngine:
    if not state.models_loaded and not state.loading:
        start_background_loading()
    if wait and state.loading and not state.models_loaded:
        state.ready_event.wait()
    if state.error and not state.models_loaded:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {state.error}")
    if not state.engine:
        raise HTTPException(status_code=503, detail="Models are still loading.")
    return state.engine


def run_prediction(query: str, use_reranker: bool, current_code: str | None = None) -> dict[str, Any]:
    engine = ensure_initialized(wait=True)
    result = engine.predict(query=query, top_k=10, use_reranker=use_reranker)
    return enrich_prediction(result, current_code=current_code)


def batch_dataframe_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "Исходное название",
        "Предсказанный код",
        "Уверенность",
        "Режим",
        "Ставка НДС",
        "Риск НДС",
        "Ближайший льготный код",
        "Название ближайшего льготного кода",
        "Расстояние до льготного",
        "Текущий код",
        "Совпадение",
        "Расхождение НДС",
        "Финальный код",
    ]
    dataframe_rows = []
    for row in rows:
        dataframe_rows.append({
            "Исходное название": row.get("source_name"),
            "Предсказанный код": row.get("predicted_code"),
            "Уверенность": row.get("confidence"),
            "Режим": row.get("mode"),
            "Ставка НДС": row.get("vat_rate"),
            "Риск НДС": row.get("vat_mismatch"),
            "Ближайший льготный код": row.get("closest_exempt_code"),
            "Название ближайшего льготного кода": row.get("closest_exempt_name"),
            "Расстояние до льготного": row.get("closest_exempt_distance"),
            "Текущий код": row.get("current_code"),
            "Совпадение": row.get("code_match"),
            "Расхождение НДС": row.get("vat_mismatch"),
            "Финальный код": row.get("final_code") or row.get("predicted_code"),
        })
    return pd.DataFrame(dataframe_rows, columns=columns)


def parse_batch_file(upload: UploadFile) -> tuple[list[dict[str, Any]], str]:
    filename = upload.filename or "batch.xlsx"
    content = upload.file.read()
    try:
        df = pd.read_excel(io.BytesIO(content), dtype=str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read Excel file: {exc}") from exc
    df = df.rename(columns={column: normalize_text(column) for column in df.columns})
    text_column = find_first_existing_column(list(df.columns), DEFAULT_TEXT_COLUMNS)
    if not text_column:
        raise HTTPException(status_code=400, detail="Excel file must contain a nomenclature column.")
    code_column = find_first_existing_column(list(df.columns), DEFAULT_CODE_COLUMNS)

    items: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        query = normalize_text(row.get(text_column))
        if not query:
            continue
        items.append({
            "query": query,
            "current_code": normalize_code(row.get(code_column)) if code_column else None,
        })
    return items, filename


def parse_batch_json(payload: BatchJsonRequest) -> list[dict[str, Any]]:
    return [{"query": item.query, "current_code": normalize_code(item.current_code)} for item in payload.items]


def create_batch_job(filename: str, use_reranker: bool) -> BatchJob:
    job = BatchJob(job_id=uuid.uuid4().hex, filename=filename, use_reranker=use_reranker)
    job.log(f"Файл '{filename}' принят в обработку.")
    with state.batch_lock:
        state.batch_jobs[job.job_id] = job
    return job


def get_batch_job_or_404(job_id: str) -> BatchJob:
    with state.batch_lock:
        job = state.batch_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return job


def process_batch_job(job_id: str, batch_items: list[dict[str, Any]]) -> None:
    job = get_batch_job_or_404(job_id)
    try:
        job.status = "running"
        job.total_rows = len(batch_items)
        job.set_progress(0.02)
        job.log("Начинаем обработку строк.")

        if not state.models_loaded:
            job.log("Ожидание готовности моделей.")
        ensure_initialized(wait=True)
        job.set_progress(0.12)
        job.log("Модели готовы, запускаем предсказания.")

        rows: list[dict[str, Any]] = []
        total = len(batch_items)
        for index, item in enumerate(batch_items, start=1):
            prediction = run_prediction(item["query"], job.use_reranker, current_code=item.get("current_code"))
            rows.append({
                "source_name": item["query"],
                "predicted_code": prediction.get("predicted_code"),
                "predicted_name": prediction.get("predicted_name"),
                "confidence": prediction.get("confidence"),
                "mode": prediction.get("mode"),
                "vat_rate": prediction.get("vat_rate"),
                "vat_exempt": prediction.get("vat_exempt"),
                "closest_exempt_code": prediction.get("closest_exempt_code"),
                "closest_exempt_name": prediction.get("closest_exempt_name"),
                "closest_exempt_distance": prediction.get("closest_exempt_distance"),
                "is_ood": prediction.get("is_ood"),
                "packaging_risk": prediction.get("packaging_risk"),
                "current_code": prediction.get("current_code"),
                "code_match": prediction.get("code_match"),
                "vat_mismatch": prediction.get("vat_mismatch"),
                "top_candidates": prediction.get("top_candidates", []),
                "final_code": prediction.get("predicted_code"),
            })
            job.processed_rows = index
            job.set_progress(0.12 + 0.82 * (index / max(total, 1)))
            if index <= 5 or index == total or index % max(1, total // 10) == 0:
                job.log(f"Обработано {index} из {total}: {item['query'][:80]}")

        job.rows = rows
        job.status = "completed"
        job.set_progress(1.0)
        job.log(f"Обработка завершена. Готово строк: {len(rows)}.")
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.set_progress(job.progress if job.progress > 0 else 0.0)
        job.log(f"Ошибка обработки: {exc}")


def start_batch_job(batch_items: list[dict[str, Any]], filename: str, use_reranker: bool) -> BatchJob:
    job = create_batch_job(filename=filename, use_reranker=use_reranker)
    thread = threading.Thread(
        target=process_batch_job,
        args=(job.job_id, batch_items),
        daemon=True,
        name=f"okpd-batch-job-{job.job_id[:8]}",
    )
    thread.start()
    return job


@app.get("/status")
def get_status() -> dict[str, Any]:
    if not state.models_loaded and not state.loading:
        start_background_loading()          # запускаем поток немедленно, а не в background_tasks
    return {
        "models_loaded": state.models_loaded,
        "loading": state.loading,
        "progress": round(state.progress, 3),
        "error": state.error,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    return run_prediction(payload.query, payload.use_reranker)


@app.post("/batch_predict", response_model=None)
async def batch_predict(request: Request, file: UploadFile | None = File(default=None)):
    content_type = request.headers.get("content-type", "")
    use_reranker = False
    rows: list[dict[str, Any]]
    filename = "batch_results.xlsx"

    if "application/json" in content_type:
        raw_payload = await request.json()
        payload = BatchJsonRequest.model_validate(raw_payload)
        use_reranker = payload.use_reranker
        batch_items = parse_batch_json(payload)
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="Excel file is required for multipart uploads.")
        batch_items, source_filename = parse_batch_file(file)
        filename = f"{source_filename.rsplit('.', 1)[0]}_results.xlsx"
        form = await request.form()
        use_reranker = str(form.get("use_reranker", "false")).lower() == "true"

    if not batch_items:
        raise HTTPException(status_code=400, detail="No valid rows were found for batch processing.")

    rows = []
    for item in batch_items:
        prediction = run_prediction(item["query"], use_reranker, current_code=item.get("current_code"))
        rows.append({
            "source_name": item["query"],
            "predicted_code": prediction.get("predicted_code"),
            "predicted_name": prediction.get("predicted_name"),
            "confidence": prediction.get("confidence"),
            "mode": prediction.get("mode"),
            "vat_rate": prediction.get("vat_rate"),
            "vat_exempt": prediction.get("vat_exempt"),
            "closest_exempt_code": prediction.get("closest_exempt_code"),
            "closest_exempt_name": prediction.get("closest_exempt_name"),
            "closest_exempt_distance": prediction.get("closest_exempt_distance"),
            "is_ood": prediction.get("is_ood"),
            "packaging_risk": prediction.get("packaging_risk"),
            "current_code": prediction.get("current_code"),
            "code_match": prediction.get("code_match"),
            "vat_mismatch": prediction.get("vat_mismatch"),
            "top_candidates": prediction.get("top_candidates", []),
            "final_code": prediction.get("predicted_code"),
        })

    if "application/json" in content_type:
        return JSONResponse(content={"rows": rows, "count": len(rows)})

    export_df = batch_dataframe_from_rows(rows)
    excel_bytes = dataframe_to_excel_bytes(export_df)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.post("/batch_jobs/upload")
async def batch_job_upload(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    form = await request.form()
    use_reranker = str(form.get("use_reranker", "false")).lower() == "true"
    batch_items, source_filename = parse_batch_file(file)
    if not batch_items:
        raise HTTPException(status_code=400, detail="No valid rows were found for batch processing.")
    job = start_batch_job(batch_items=batch_items, filename=source_filename, use_reranker=use_reranker)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "filename": job.filename,
        "total_rows": job.total_rows,
    }


@app.get("/batch_jobs/{job_id}")
def batch_job_status(job_id: str) -> dict[str, Any]:
    job = get_batch_job_or_404(job_id)
    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status,
        "progress": round(job.progress, 3),
        "processed_rows": job.processed_rows,
        "total_rows": job.total_rows,
        "error": job.error,
        "logs": job.logs[-30:],
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@app.get("/batch_jobs/{job_id}/results")
def batch_job_results(job_id: str) -> dict[str, Any]:
    job = get_batch_job_or_404(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Batch job is not completed yet.")
    return {"rows": job.rows, "count": len(job.rows), "filename": job.filename}


@app.get("/batch_jobs/{job_id}/export")
def batch_job_export(job_id: str) -> StreamingResponse:
    job = get_batch_job_or_404(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Batch job is not completed yet.")
    export_df = batch_dataframe_from_rows(job.rows)
    excel_bytes = dataframe_to_excel_bytes(export_df)
    export_name = f"{job.filename.rsplit('.', 1)[0]}_results.xlsx"
    headers = {"Content-Disposition": f'attachment; filename="{export_name}"'}
    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.post("/export")
def export_rows(payload: ExportRequest) -> StreamingResponse:
    export_df = batch_dataframe_from_rows([row.model_dump() for row in payload.rows])
    excel_bytes = dataframe_to_excel_bytes(export_df)
    headers = {"Content-Disposition": 'attachment; filename="editor_export.xlsx"'}
    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.get("/code_info/{code}")
def code_info(code: str) -> dict[str, Any]:
    ensure_initialized(wait=True)
    normalized = normalize_code(code)
    if not normalized:
        raise HTTPException(status_code=400, detail="Code is required.")
    name = state.code_to_name.get(normalized)
    if not name:
        raise HTTPException(status_code=404, detail="Code not found in reference.")
    return {
        "code": normalized,
        "name": name,
        **get_vat_info(normalized),
    }


@app.get("/reference/codes")
def reference_codes() -> dict[str, Any]:
    ensure_initialized(wait=True)
    assert state.okpd_df is not None
    items = state.okpd_df[["code", "name"]].to_dict(orient="records")
    return {"items": items, "count": len(items)}


@app.get("/health")
def health() -> dict[str, Literal["ok"]]:
    return {"status": "ok"}
