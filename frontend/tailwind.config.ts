import type { Config } from "tailwindcss";

// Design tokens (palette, type scale, spacing) are defined in Phase 11 together with the
// customer UI; Phase 1 only wires Tailwind so the toolchain is proven end to end.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};

export default config;
