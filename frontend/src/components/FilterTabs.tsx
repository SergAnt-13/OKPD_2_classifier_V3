import type { FilterKind } from "../types";

interface FilterTabsProps {
  value: FilterKind;
  onChange: (filter: FilterKind) => void;
}

const filters: Array<{ id: FilterKind; label: string }> = [
  { id: "all", label: "Все записи" },
  { id: "codeMismatch", label: "Только расхождения кодов" },
  { id: "vatRisk", label: "Только риски НДС" }
];

export function FilterTabs({ value, onChange }: FilterTabsProps): JSX.Element {
  return (
    <div className="flex flex-wrap gap-2">
      {filters.map((filter) => (
        <button
          key={filter.id}
          type="button"
          onClick={() => onChange(filter.id)}
          className={`rounded-full border px-3 py-2 text-sm transition ${
            value === filter.id
              ? "border-violet/50 bg-violet/15 text-white shadow-violet"
              : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20"
          }`}
        >
          {filter.label}
        </button>
      ))}
    </div>
  );
}
