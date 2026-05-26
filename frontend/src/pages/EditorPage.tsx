import { startTransition, useDeferredValue, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { fetchCodeReference } from "../lib/api";
import { downloadRowsAsExcel, filterRows } from "../lib/utils";
import { FilterTabs } from "../components/FilterTabs";
import { Badge } from "../components/Badge";
import { ProgressBar } from "../components/ProgressBar";
import type { BatchRow, FilterKind } from "../types";

interface EditorPageProps {
  rows: BatchRow[];
  onRowsChange: (rows: BatchRow[]) => void;
}

export function EditorPage({ rows, onRowsChange }: EditorPageProps): JSX.Element {
  const [filter, setFilter] = useState<FilterKind>("all");
  const [search, setSearch] = useState("");
  const deferredSearch = useDeferredValue(search);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const { data: reference = [] } = useQuery({
    queryKey: ["reference", "codes"],
    queryFn: fetchCodeReference
  });

  const visibleRows = useMemo(() => {
    const filtered = filterRows(rows, filter);
    if (!deferredSearch.trim()) return filtered;
    const term = deferredSearch.toLowerCase();
    return filtered.filter((row) => row.source_name.toLowerCase().includes(term));
  }, [deferredSearch, filter, rows]);

  useEffect(() => {
    if (selectedIndex >= visibleRows.length) {
      setSelectedIndex(0);
    }
  }, [selectedIndex, visibleRows.length]);

  const row = visibleRows[selectedIndex];

  const updateRowCode = (sourceName: string, nextCode: string) => {
    startTransition(() => {
      onRowsChange(
        rows.map((item) =>
          item.source_name === sourceName
            ? { ...item, final_code: nextCode }
            : item
        )
      );
    });
  };

  const acceptAllAuto = () => {
    startTransition(() => {
      onRowsChange(
        rows.map((item) =>
          item.mode === "AUTO"
            ? { ...item, final_code: item.predicted_code }
            : item
        )
      );
    });
  };

  if (!rows.length) {
    return (
      <div className="rounded-[32px] border border-dashed border-white/15 bg-white/[0.03] p-10 text-slate-400">
        Сначала выполните массовую обработку, затем здесь появится интерактивный редактор с подбором финальных кодов.
      </div>
    );
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[0.95fr_1.25fr]">
      <motion.section
        initial={{ opacity: 0, x: -16 }}
        animate={{ opacity: 1, x: 0 }}
        className="rounded-[32px] border border-white/10 bg-white/5 p-6 shadow-violet backdrop-blur-xl"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.3em] text-violet">Editor</p>
            <h2 className="mt-2 text-2xl font-semibold text-white">Список товаров</h2>
          </div>
          <button
            type="button"
            onClick={acceptAllAuto}
            className="rounded-full bg-[linear-gradient(135deg,#A855F7,#00F0FF)] px-4 py-2 text-sm font-medium text-slate-950"
          >
            Принять все AUTO
          </button>
        </div>

        <input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Поиск по наименованию..."
          className="mt-5 w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm text-white outline-none placeholder:text-slate-500"
        />

        <div className="mt-4">
          <FilterTabs value={filter} onChange={setFilter} />
        </div>

        <div className="mt-5 space-y-3">
          {visibleRows.map((item, index) => (
            <button
              key={`${item.source_name}-${index}`}
              type="button"
              onClick={() => setSelectedIndex(index)}
              className={`w-full rounded-2xl border px-4 py-4 text-left transition ${
                index === selectedIndex
                  ? "border-neon/40 bg-neon/10"
                  : "border-white/10 bg-slate-950/55 hover:border-white/20"
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm text-white">{item.source_name}</p>
                  <p className="mt-2 font-mono text-xs text-slate-400">
                    {item.final_code ?? item.predicted_code ?? "—"} • {item.vat_rate ?? "—"}
                  </p>
                </div>
                <Badge label={item.mode} mode={item.mode} />
              </div>
            </button>
          ))}
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, x: 16 }}
        animate={{ opacity: 1, x: 0 }}
        className="rounded-[32px] border border-white/10 bg-slate-950/65 p-6 backdrop-blur-xl"
      >
        {row ? (
          <div className="space-y-6">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="font-mono text-xs uppercase tracking-[0.3em] text-neon">Детали строки</p>
                <h3 className="mt-3 text-2xl font-semibold text-white">{row.source_name}</h3>
                <p className="mt-2 text-sm text-slate-400">
                  Ближайшая льгота: {row.closest_exempt_code ?? "—"} • расстояние {row.closest_exempt_distance ?? "—"}
                </p>
              </div>
              <Badge label={row.mode} mode={row.mode} />
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between text-sm text-slate-300">
                <span>Уверенность модели</span>
                <span>{Math.round(row.confidence * 100)}%</span>
              </div>
              <ProgressBar value={row.confidence} />
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <label className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <span className="mb-2 block text-xs uppercase tracking-[0.24em] text-slate-400">Финальный код</span>
                <select
                  value={row.final_code ?? row.predicted_code ?? ""}
                  onChange={(event) => updateRowCode(row.source_name, event.target.value)}
                  className="w-full rounded-xl border border-white/10 bg-slate-950/70 px-3 py-3 text-sm text-white outline-none"
                >
                  {[row.predicted_code, ...reference.map((item) => item.code)]
                    .filter((value, index, array) => value && array.indexOf(value) === index)
                    .map((code) => {
                      const ref = reference.find((item) => item.code === code);
                      return (
                        <option key={code} value={code ?? ""}>
                          {code} {ref ? `— ${ref.name}` : ""}
                        </option>
                      );
                    })}
                </select>
              </label>

              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">НДС и флаги</p>
                <p className="mt-3 text-sm text-white">Ставка: {row.vat_rate ?? "—"}</p>
                <p className="mt-2 text-sm text-slate-300">Расхождение НДС: {String(row.vat_mismatch ?? false)}</p>
                <p className="mt-2 text-sm text-slate-300">Совпадение кода: {String(row.code_match ?? false)}</p>
              </div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Кандидаты retrieval</p>
              {row.top_candidates.length ? (
                <div className="mt-4 space-y-3">
                  {row.top_candidates.map((candidate, index) => (
                    <div key={`${candidate.code}-${index}`} className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="font-mono text-sm text-neon">{candidate.code}</p>
                          <p className="mt-1 text-sm text-slate-300">{candidate.name}</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => updateRowCode(row.source_name, candidate.code)}
                          className="rounded-full border border-neon/30 px-3 py-2 text-xs uppercase tracking-[0.18em] text-neon"
                        >
                          Выбрать
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-4 text-sm text-slate-400">
                  Если данные пришли из Excel-обработки, список кандидатов можно дополнить повторной обработкой через JSON API.
                </p>
              )}
            </div>

            <button
              type="button"
              onClick={() => void downloadRowsAsExcel(rows, "okpd_editor_export.xlsx")}
              className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200"
            >
              Сохранить изменения и экспортировать Excel
            </button>
          </div>
        ) : null}
      </motion.section>
    </div>
  );
}
