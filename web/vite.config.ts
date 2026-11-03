/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development `/api` goes to `newsletter serve` on its default address. The API only
// accepts writes from an allowed Origin, so add this page's origin to
// NEWSLETTER_API_ORIGINS (for example http://localhost:5173).
const api = process.env.NEWSLETTER_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": api } },
  preview: { proxy: { "/api": api } },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["src/setup-tests.ts"],
    coverage: {
      include: ["src/**"],
      exclude: ["src/main.tsx", "src/setup-tests.ts", "src/fixtures.ts", "src/**/*.test.*"],
      thresholds: { lines: 80 },
    },
  },
});
