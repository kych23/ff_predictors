/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0D1117", surface: "#161B22", line: "#30363D",
        ink: "#C9D1D9", muted: "#8B949E",
        primary: "#3B82F6", accent: "#F59E0B",
        good: "#3FB950", warn: "#D29922",
      },
      fontFamily: {
        mono: ["Fira Code", "ui-monospace", "monospace"],
        sans: ["Fira Sans", "ui-sans-serif", "system-ui"],
      },
    },
  },
  plugins: [],
};
