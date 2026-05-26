import { motion } from "framer-motion";

interface StatusPillProps {
  loading: boolean;
  ready: boolean;
}

export function StatusPill({ loading, ready }: StatusPillProps): JSX.Element {
  const tone = ready
    ? "bg-success/15 text-success border-success/40"
    : "bg-neon/10 text-neon border-neon/40";

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      initial={{ opacity: 0, y: -12 }}
      className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-xs uppercase tracking-[0.24em] ${tone}`}
    >
      <span className={`h-2.5 w-2.5 rounded-full ${ready ? "bg-success" : "bg-neon animate-pulse"}`} />
      {ready ? "Модели готовы" : loading ? "Загрузка..." : "Ожидание"}
    </motion.div>
  );
}
