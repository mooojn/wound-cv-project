export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Manrope", "system-ui", "sans-serif"],
        display: ["Space Grotesk", "Manrope", "sans-serif"],
      },
      colors: {
        paper: "#f5f4f0",
        card: "#fffdf8",
        ink: "#1d1d1a",
        clay: "#e9e3d7",
        accent: "#0f766e",
        warm: "#b45309",
      },
      boxShadow: {
        soft: "0 15px 45px rgba(28, 27, 23, 0.12)",
      },
    },
  },
  plugins: [],
};
