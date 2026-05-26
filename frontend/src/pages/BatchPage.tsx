import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { useMutation, useQuery } from "@tanstack/react-query";
import { fetchBatchJobResults, fetchBatchJobStatus, startBatchUploadJob } from "../lib/api";
import { downloadRowsAsExcel, filterRows } from "../lib/utils";
import { FilterTabs } from "../components/FilterTabs";
import { VirtualizedResultsTable } from "../components/VirtualizedResultsTable";
import type { BatchJobStatus, BatchRow, FilterKind } from "../types";

interface BatchPageProps {
  rows: BatchRow[];
  onRowsChange: (rows: BatchRow[]) => void;
}

function progressPercent(value: number | undefined): number {
  return Math.round((value ?? 0) * 100);
}

export function BatchPage({ rows, onRowsChange }: BatchPageProps): JSX.Element {
  const [dragging, setDragging] = useState(false);
  const [filter, setFilter] = useState<FilterKind>("all");
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeFileName, setActiveFileName] = useState<string>("");
  const loadedJobIdRef = useRef<string | null>(null);

  const startJobMutation = useMutation({
    mutationFn: async (file: File) => startBatchUploadJob(file),
    onSuccess: (result) => {
      loadedJobIdRef.current = null;
      setActiveJobId(result.job_id);
      setActiveFileName(result.filename);
    }
  });

  const jobQuery = useQuery({
    queryKey: ["batch-job", activeJobId],
    queryFn: () => fetchBatchJobStatus(activeJobId!),
    enabled: Boolean(activeJobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (!status) return 1200;
      return status === "completed" || status === "failed" ? false : 1200;
    }
  });

  useEffect(() => {
    const job = jobQuery.data;
    if (!activeJobId || !job || job.status !== "completed") {
      return;
    }
    if (loadedJobIdRef.current === activeJobId) {
      return;
    }
    loadedJobIdRef.current = activeJobId;
    void fetchBatchJobResults(activeJobId).then(onRowsChange);
  }, [activeJobId, jobQuery.data, onRowsChange]);

  const filteredRows = useMemo(() => filterRows(rows, filter), [filter, rows]);
  const activeJob = jobQuery.data;
  const isProcessing = startJobMutation.isPending || jobQuery.isFetching || (activeJob?.status === "pending" || activeJob?.status === "running");
  const errorMessage =
    (startJobMutation.error as Error | undefined)?.message ??
    activeJob?.error ??
    (jobQuery.error as Error | undefined)?.message;

  return (
    <div className="space-y-6">
      <motion.section
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-[32px] border border-white/10 bg-white/5 p-6 shadow-neon backdrop-blur-xl"
      >
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.3em] text-neon">Batch Pipeline</p>
            <h2 className="mt-3 text-3xl font-semibold text-white">Массовая обработка Excel</h2>
          </div>
          <div className="flex gap-3">
            <button
              type="button"
              disabled={!rows.length}
              onClick={() => void downloadRowsAsExcel(rows, "okpd_batch_results.xlsx")}
              className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200 disabled:opacity-40"
            >
              Экспорт в Excel
            </button>
          </div>
        </div>

        <label
          onDragEnter={() => setDragging(true)}
          onDragLeave={() => setDragging(false)}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const file = event.dataTransfer.files?.[0];
            if (file) startJobMutation.mutate(file);
          }}
          className={`mt-6 flex min-h-48 cursor-pointer flex-col items-center justify-center rounded-[28px] border border-dashed px-6 py-8 text-center transition ${
            dragging ? "border-neon/60 bg-neon/10" : "border-white/15 bg-slate-950/40"
          }`}
        >
          <input
            type="file"
            accept=".xlsx,.xls"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) startJobMutation.mutate(file);
            }}
          />
          <div className="rounded-full border border-neon/30 bg-neon/10 px-4 py-2 font-mono text-xs uppercase tracking-[0.2em] text-neon">
            Drag & Drop
          </div>
          <p className="mt-4 text-lg text-white">Перетащите Excel-файл или нажмите для выбора</p>
          <p className="mt-2 text-sm text-slate-400">Ожидается столбец «Номенклатура» и опционально «Код ОКПД2».</p>
        </label>

        {activeJobId ? (
          <BatchJobPanel
            fileName={activeFileName}
            job={activeJob}
            loading={isProcessing}
            errorMessage={errorMessage}
          />
        ) : null}

        {!activeJobId && errorMessage ? (
          <p className="mt-4 rounded-2xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
            {errorMessage}
          </p>
        ) : null}
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.08 }}
        className="rounded-[32px] border border-white/10 bg-slate-950/65 p-6 backdrop-blur-xl"
      >
        <div className="flex flex-wrap items-center justify-between gap-4">
          <FilterTabs value={filter} onChange={setFilter} />
          <div className="text-sm text-slate-400">
            {isProcessing ? "Идёт обработка файла..." : `Записей: ${filteredRows.length}`}
          </div>
        </div>

        {filteredRows.length ? (
          <div className="mt-6">
            <VirtualizedResultsTable
              rows={filteredRows}
              onSelect={(index) => setSelectedName(filteredRows[index]?.source_name ?? null)}
            />
            {selectedName ? <p className="mt-3 text-sm text-slate-400">Выбрано: {selectedName}</p> : null}
          </div>
        ) : (
          <div className="mt-6 rounded-[28px] border border-dashed border-white/15 bg-white/[0.03] p-8 text-slate-400">
            После загрузки файла здесь появится виртуализированная таблица с фильтрами, НДС и льготными кодами.
          </div>
        )}
      </motion.section>
    </div>
  );
}

interface BatchJobPanelProps {
  fileName: string;
  job?: BatchJobStatus;
  loading: boolean;
  errorMessage?: string;
}

function BatchJobPanel({ fileName, job, loading, errorMessage }: BatchJobPanelProps): JSX.Element {
  const percent = progressPercent(job?.progress);
  const statusLabel =
    job?.status === "completed"
      ? "Завершено"
      : job?.status === "failed"
        ? "Ошибка"
        : loading
          ? "В обработке"
          : "Ожидание";

  return (
    <div className="mt-6 rounded-[28px] border border-white/10 bg-slate-950/65 p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-neon">Текущая задача</p>
          <h3 className="mt-2 text-xl font-semibold text-white">{fileName || job?.filename || "Файл"}</h3>
          <p className="mt-2 text-sm text-slate-400">
            Статус: {statusLabel}
            {job ? ` • ${job.processed_rows}/${job.total_rows || "?"} строк` : ""}
          </p>
        </div>
        <div className="rounded-2xl border border-neon/30 bg-neon/10 px-4 py-3 font-mono text-2xl text-neon">
          {percent}%
        </div>
      </div>

      <div className="mt-4 h-4 overflow-hidden rounded-full bg-white/10">
        <motion.div
          className="h-full rounded-full bg-[linear-gradient(90deg,#00F0FF,#A855F7)] shadow-neon"
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>

      {errorMessage ? (
        <p className="mt-4 rounded-2xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
          {errorMessage}
        </p>
      ) : null}

      <div className="mt-4 rounded-[24px] border border-white/10 bg-black/30 p-4">
        <div className="mb-3 flex items-center justify-between gap-4">
          <p className="font-mono text-xs uppercase tracking-[0.22em] text-slate-400">Лог обработки</p>
          <p className="text-xs text-slate-500">Последние сообщения сервера</p>
        </div>
        <div className="max-h-60 space-y-2 overflow-auto font-mono text-xs text-slate-300">
          {job?.logs?.length ? (
            job.logs.map((line, index) => (
              <div key={`${line}-${index}`} className="rounded-xl bg-white/5 px-3 py-2">
                {line}
              </div>
            ))
          ) : (
            <div className="rounded-xl bg-white/5 px-3 py-2 text-slate-500">
              Ожидаем первые сообщения от сервера...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
