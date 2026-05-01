/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["DM Sans", "system-ui", "sans-serif"],
      },
      colors: {
        ink: { 950: "#0a0f1a", 900: "#111827", 700: "#374151", 500: "#6b7280" },
        sky: { glow: "#22d3ee", deep: "#0369a1" },
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { boxShadow: "0 0 12px 2px rgba(34, 211, 238, 0.45)" },
          "50%": { boxShadow: "0 0 22px 6px rgba(34, 211, 238, 0.75)" },
        },
        spinSlow: { to: { transform: "rotate(360deg)" } },
      },
      animation: {
        pulseGlow: "pulseGlow 2.4s ease-in-out infinite",
        spinSlow: "spinSlow 14s linear infinite",
      },
    },
  },
  plugins: [],
};
