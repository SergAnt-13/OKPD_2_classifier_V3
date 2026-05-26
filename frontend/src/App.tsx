import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Navigate, Route, Routes } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { API_BASE, fetchStatus } from "./lib/api";
import { BackgroundCanvas } from "./components/BackgroundCanvas";
import { InitializationScreen } from "./components/InitializationScreen";
import { StatusPill } from "./components/StatusPill";
import { TabsNav } from "./components/TabsNav";
import { SingleItemPage } from "./pages/SingleItemPage";
import { BatchPage } from "./pages/BatchPage";
import { EditorPage } from "./pages/EditorPage";
import type { BatchRow } from "./types";

export default function App(): JSX.Element {
  const [rows, setRows] = useState<BatchRow[]>([]);
  const [online] = useState(true);
  const statusQuery = useQuery({
    queryKey: ["status"],
    queryFn: fetchStatus,
    refetchInterval: (query) => (query.state.data?.models_loaded ? false : 2000),
    refetchOnWindowFocus: true,
    retry: 1
  });

  const initialized = statusQuery.data?.models_loaded ?? false;
  const loading = statusQuery.data?.loading ?? true;
  const stats = useMemo(() => {
    const total = rows.length;
    const auto = rows.filter((row) => row.mode === "AUTO").length;
    const vat = rows.filter((row) => row.vat_mismatch).length;
    return { total, auto, vat };
  }, [rows]);

  if (!initialized) {
    return (
      <>
        <BackgroundCanvas />
        <InitializationScreen
          status={statusQuery.data}
          errorMessage={statusQuery.error instanceof Error ? statusQuery.error.message : undefined}
          apiBase={API_BASE}
        />
      </>
    );
  }

  return (
    <div className="min-h-screen px-5 pb-12 pt-6 text-white md:px-8">
      <BackgroundCanvas />
      <motion.header
        initial={{ opacity: 0, y: -18 }}
        animate={{ opacity: 1, y: 0 }}
        className="mx-auto mb-8 max-w-[1600px] rounded-[32px] border border-white/10 bg-white/5 p-6 shadow-neon backdrop-blur-xl"
      >
        <div className="flex flex-col gap-6 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-3">
              <p className="font-mono text-xs uppercase tracking-[0.34em] text-neon">OKPD-2 Classifier V3</p>
              <StatusPill loading={loading} ready={initialized} />
            </div>
            <h1 className="mt-4 text-4xl font-semibold tracking-tight text-white">Технологичный интерфейс проверки кодов и НДС</h1>
            <p className="mt-3 max-w-3xl text-sm text-slate-400">
              Единое рабочее место для проверки одного товара, пакетной обработки Excel и ручной редакторской валидации.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            {[
              { label: "Записей", value: stats.total.toString() },
              { label: "AUTO", value: stats.auto.toString() },
              { label: "НДС-рисков", value: stats.vat.toString() }
            ].map((item) => (
              <div key={item.label} className="rounded-2xl border border-white/10 bg-slate-950/50 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.22em] text-slate-500">{item.label}</p>
                <p className="mt-2 font-mono text-2xl text-white">{item.value}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
          <TabsNav />
          <button
            type="button"
            className={`inline-flex items-center gap-3 rounded-full border px-4 py-2 text-sm ${
              online ? "border-neon/40 bg-neon/10 text-neon" : "border-white/10 bg-white/5 text-slate-300"
            }`}
          >
            <span className={`h-2.5 w-2.5 rounded-full ${online ? "bg-neon" : "bg-slate-500"}`} />
            Онлайн / оффлайн
          </button>
        </div>
      </motion.header>

      <main className="mx-auto max-w-[1600px]">
        <Routes>
          <Route path="/" element={<SingleItemPage />} />
          <Route path="/batch" element={<BatchPage rows={rows} onRowsChange={setRows} />} />
          <Route path="/editor" element={<EditorPage rows={rows} onRowsChange={setRows} />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
