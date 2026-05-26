import { motion } from "framer-motion";
import type { StatusResponse } from "../types";

interface InitializationScreenProps {
  status?: StatusResponse;
  errorMessage?: string;
  apiBase?: string;
}

export function InitializationScreen({ status, errorMessage, apiBase }: InitializationScreenProps): JSX.Element {
  const percent = Math.round((status?.progress ?? 0) * 100);
  const message = status?.error ?? errorMessage;

  return (
    <div className="flex min-h-screen items-center justify-center px-6">
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-2xl rounded-[32px] border border-white/10 bg-white/5 p-8 shadow-violet backdrop-blur-xl"
      >
        <div className="mb-6 flex items-center justify-between">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.34em] text-neon">OKPD-2 V3</p>
            <h1 className="mt-3 text-4xl font-semibold text-white">Инициализация моделей...</h1>
          </div>
          <div className="rounded-2xl border border-neon/30 bg-neon/10 px-4 py-3 font-mono text-2xl text-neon">
            {percent}%
          </div>
        </div>

        <div className="h-4 overflow-hidden rounded-full bg-white/10">
          <motion.div
            className="h-full rounded-full bg-[linear-gradient(90deg,#00F0FF,#A855F7)] shadow-neon"
            initial={{ width: 0 }}
            animate={{ width: `${percent}%` }}
            transition={{ duration: 0.6 }}
          />
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {[
            "Поднимаем retrieval-стек",
            "Подключаем DecisionEngine",
            "Готовим НДС-справочники"
          ].map((label, index) => (
            <motion.div
              key={label}
              animate={{ y: [0, -5, 0] }}
              transition={{ repeat: Number.POSITIVE_INFINITY, duration: 2.4, delay: index * 0.2 }}
              className="rounded-2xl border border-white/10 bg-slate-950/60 p-4"
            >
              <div className="mb-3 h-2 w-2 rounded-full bg-neon shadow-neon" />
              <p className="text-sm text-slate-300">{label}</p>
            </motion.div>
          ))}
        </div>

        {message ? (
          <p className="mt-5 rounded-2xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
            {message}
          </p>
        ) : (
          <p className="mt-5 text-sm text-slate-400">
            Экран автоматически исчезнет, как только модель и справочники будут готовы к запросам.
          </p>
        )}

        <div className="mt-4 rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 font-mono text-xs text-slate-400">
          API: {apiBase ?? "unknown"} • loaded={String(status?.models_loaded ?? false)} • loading={String(status?.loading ?? false)}
        </div>
      </motion.div>
    </div>
  );
}
