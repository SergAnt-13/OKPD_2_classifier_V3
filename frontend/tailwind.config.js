/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: "#0B0F19",
        panel: "#111827",
        neon: "#00F0FF",
        violet: "#A855F7",
        success: "#4ADE80",
        warning: "#FBBF24",
        danger: "#FB7185"
      },
      boxShadow: {
        neon: "0 0 0 1px rgba(0,240,255,0.2), 0 12px 50px rgba(0,240,255,0.16)",
        violet: "0 0 0 1px rgba(168,85,247,0.2), 0 14px 50px rgba(168,85,247,0.18)"
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"]
      },
      keyframes: {
        floaty: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" }
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" }
        }
      },
      animation: {
        floaty: "floaty 8s ease-in-out infinite",
        shimmer: "shimmer 2.4s linear infinite"
      }
    }
  },
  plugins: []
};
