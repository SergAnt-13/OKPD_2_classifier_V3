import { useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { BatchRow } from "../types";
import { Badge } from "./Badge";

interface VirtualizedResultsTableProps {
  rows: BatchRow[];
  selectedIndex?: number;
  onSelect?: (index: number) => void;
}

export function VirtualizedResultsTable({ rows, selectedIndex, onSelect }: VirtualizedResultsTableProps): JSX.Element {
  const parentRef = useRef<HTMLDivElement | null>(null);
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 64,
    overscan: 12
  });

  const items = useMemo(() => virtualizer.getVirtualItems(), [virtualizer]);

  return (
    <div className="overflow-hidden rounded-[28px] border border-white/10 bg-slate-950/50">
      <div className="grid grid-cols-[2.2fr_1fr_1fr_0.9fr_0.9fr_1fr] gap-4 border-b border-white/10 px-5 py-4 text-xs uppercase tracking-[0.24em] text-slate-400">
        <span>Наименование</span>
        <span>Код</span>
        <span>Текущий</span>
        <span>НДС</span>
        <span>Режим</span>
        <span>Льгота</span>
      </div>
      <div ref={parentRef} className="h-[420px] overflow-auto">
        <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
          {items.map((item) => {
            const row = rows[item.index];
            const active = selectedIndex === item.index;
            return (
              <button
                key={item.key}
                type="button"
                onClick={() => onSelect?.(item.index)}
                className={`absolute left-0 top-0 grid w-full grid-cols-[2.2fr_1fr_1fr_0.9fr_0.9fr_1fr] gap-4 border-b border-white/5 px-5 py-4 text-left transition ${
                  active ? "bg-neon/8" : "hover:bg-white/5"
                }`}
                style={{ transform: `translateY(${item.start}px)` }}
              >
                <span className="truncate text-sm text-white">{row.source_name}</span>
                <span className="font-mono text-sm text-neon">{row.final_code ?? row.predicted_code ?? "—"}</span>
                <span className="font-mono text-sm text-slate-300">{row.current_code ?? "—"}</span>
                <span className={`text-sm ${row.vat_mismatch ? "text-danger" : "text-slate-300"}`}>{row.vat_rate ?? "—"}</span>
                <Badge label={row.mode} mode={row.mode} />
                <span className="text-sm text-slate-300">{row.closest_exempt_distance ?? "—"}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
