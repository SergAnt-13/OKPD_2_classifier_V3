import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";

const tabs = [
  { to: "/", label: "Одно наименование" },
  { to: "/batch", label: "Массовая обработка" },
  { to: "/editor", label: "Редактор" }
];

export function TabsNav(): JSX.Element {
  return (
    <div className="flex flex-wrap gap-3">
      {tabs.map((tab, index) => (
        <motion.div
          key={tab.to}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
        >
          <NavLink
            to={tab.to}
            end={tab.to === "/"}
            className={({ isActive }) =>
              `inline-flex rounded-full border px-4 py-2 text-sm transition ${
                isActive
                  ? "border-neon/50 bg-neon/10 text-white shadow-neon"
                  : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20 hover:text-white"
              }`
            }
          >
            {tab.label}
          </NavLink>
        </motion.div>
      ))}
    </div>
  );
}
