import { motion } from "framer-motion";
import { cn, modeTone } from "../lib/utils";

interface BadgeProps {
  label: string;
  mode?: string;
}

export function Badge({ label, mode }: BadgeProps): JSX.Element {
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em]",
        mode ? modeTone(mode) : "border-white/10 bg-white/5 text-slate-200"
      )}
    >
      {label}
    </motion.span>
  );
}
