import { motion } from "framer-motion";
import { confidencePercent } from "../lib/utils";

interface ProgressBarProps {
  value: number;
}

export function ProgressBar({ value }: ProgressBarProps): JSX.Element {
  const percent = confidencePercent(value);
  return (
    <div className="h-3 w-full overflow-hidden rounded-full bg-white/10">
      <motion.div
        className="h-full rounded-full bg-[linear-gradient(90deg,#00F0FF,#A855F7,#4ADE80)] bg-[length:200%_100%] shadow-neon animate-shimmer"
        initial={{ width: 0 }}
        animate={{ width: `${percent}%` }}
        transition={{ duration: 0.7, ease: "easeOut" }}
      />
    </div>
  );
}
