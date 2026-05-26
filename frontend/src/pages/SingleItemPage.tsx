import { useState } from "react";
import { motion } from "framer-motion";
import { useMutation } from "@tanstack/react-query";
import { predict } from "../lib/api";
import { Badge } from "../components/Badge";
import { ProgressBar } from "../components/ProgressBar";

export function SingleItemPage(): JSX.Element {
  const [query, setQuery] = useState("");
  const [useReranker, setUseReranker] = useState(false);
  const mutation = useMutation({
    mutationFn: () => predict(query, useReranker)
  });

  const result = mutation.data;

  return (
    <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
      <motion.section
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-[32px] border border-white/10 bg-white/5 p-6 shadow-neon backdrop-blur-xl"
      >
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-neon">Одиночная проверка</p>
        <h2 className="mt-3 text-3xl font-semibold text-white">Проверка одного наименования</h2>
        <p className="mt-3 max-w-2xl text-sm text-slate-400">
          Введите товарную позицию и получите код, режим принятия, НДС и ближайшую льготную категорию.
        </p>

        <div className="mt-8 space-y-4">
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Введите название товара..."
            className="min-h-40 w-full rounded-[28px] border border-white/10 bg-slate-950/70 px-5 py-4 text-base text-white outline-none transition placeholder:text-slate-500 focus:border-neon/50 focus:shadow-neon"
          />
          <div className="flex flex-wrap items-center justify-between gap-4">
            <label className="inline-flex items-center gap-3 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={useReranker}
                onChange={(event) => setUseReranker(event.target.checked)}
                className="h-4 w-4 accent-cyan-400"
              />
              Включить reranker
            </label>
            <button
              type="button"
              disabled={!query.trim() || mutation.isPending}
              onClick={() => mutation.mutate()}
              className="rounded-full bg-[linear-gradient(135deg,#00F0FF,#A855F7)] px-6 py-3 text-sm font-semibold text-slate-950 transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {mutation.isPending ? "Проверяем..." : "Проверить"}
            </button>
          </div>
        </div>

        {mutation.error ? (
          <p className="mt-6 rounded-2xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
            {(mutation.error as Error).message}
          </p>
        ) : null}
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.08 }}
        className="rounded-[32px] border border-white/10 bg-slate-950/70 p-6 backdrop-blur-xl"
      >
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-violet">Decision Result</p>
        {result ? (
          <div className="mt-5 space-y-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <button
                  type="button"
                  onClick={() => navigator.clipboard.writeText(result.predicted_code ?? "")}
                  className="font-mono text-4xl text-neon"
                >
                  {result.predicted_code ?? "—"}
                </button>
                <p className="mt-2 text-sm text-slate-400">{result.predicted_name ?? "Наименование будет показано после выбора кода"}</p>
              </div>
              <Badge label={result.mode} mode={result.mode} />
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between text-sm text-slate-300">
                <span>Уверенность</span>
                <span>{Math.round(result.confidence * 100)}%</span>
              </div>
              <ProgressBar value={result.confidence} />
            </div>

            <details className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <summary className="cursor-pointer text-sm font-medium text-white">Пояснение модели</summary>
              <p className="mt-3 text-sm text-slate-300">{result.explanation}</p>
            </details>

            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">НДС</p>
                <p className="mt-2 text-2xl text-white">{result.vat_rate ?? "—"}</p>
                <p className="mt-2 text-sm text-slate-400">
                  {result.vat_exempt ? "Льготный код" : "Базовая ставка"} {result.packaging_risk ? "• упаковочный риск" : ""}
                </p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Близость к льготе</p>
                <p className="mt-2 font-mono text-xl text-neon">{result.closest_exempt_code ?? "—"}</p>
                <p className="mt-2 text-sm text-slate-300">{result.closest_exempt_name ?? "Нет данных"}</p>
                <p className="mt-2 text-xs text-slate-500">Расстояние: {result.closest_exempt_distance ?? "—"}</p>
              </div>
            </div>

            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Top-5 кандидатов</p>
              {result.top_candidates.map((candidate, index) => (
                <motion.div
                  key={`${candidate.code}-${index}`}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="rounded-2xl border border-white/10 bg-white/5 p-4"
                >
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="font-mono text-sm text-neon">{candidate.code}</p>
                      <p className="mt-1 text-sm text-slate-300">{candidate.name}</p>
                    </div>
                    <span className="text-sm text-slate-400">{candidate.score.toFixed(3)}</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        ) : (
          <div className="mt-8 rounded-[28px] border border-dashed border-white/15 bg-white/[0.03] p-8 text-slate-400">
            После первого запроса здесь появятся код, уверенность, top-кандидаты и блок НДС.
          </div>
        )}
      </motion.section>
    </div>
  );
}
