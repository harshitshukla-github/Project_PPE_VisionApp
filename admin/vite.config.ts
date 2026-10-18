import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

/**
 * In production FastAPI serves the built app at /admin (AGENTS.md §15), so
 * assets are built under that base. In development the dev server forwards
 * /api to FastAPI: the browser only ever talks to localhost:5173, so the
 * session cookie (SameSite=Strict, Path=/api) stays first-party and no CORS
 * setup is needed (docs/auth-api.md §6, §8).
 */
export default defineConfig({
  base: "/admin/",
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/testing/setup.ts"],
  },
});
