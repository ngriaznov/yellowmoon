import react from "@vitejs/plugin-react";
import { configDefaults, defineConfig } from "vitest/config";

const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/ and https://v2.tauri.app/start/frontend/vite/
export default defineConfig({
  plugins: [react()],

  // Tauri options, applied in `tauri dev` and `tauri build`.
  // 1. Keep Rust compiler errors visible in the terminal.
  clearScreen: false,
  // 2. Tauri expects a fixed port and fails if it is taken.
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: "ws", host, port: 1421 } : undefined,
    watch: {
      // 3. Never watch the Rust side: it is large and changes constantly.
      ignored: [
        "**/src-tauri/**",
        "**/crates/**",
        "**/tools/**",
        "**/target/**",
        "**/testdata/**",
        "**/training/**",
        "**/research/**",
        "**/data/**",
      ],
    },
  },
  // Expose Tauri's build variables (TAURI_ENV_PLATFORM, TAURI_ENV_DEBUG, ...) to import.meta.env.
  envPrefix: ["VITE_", "TAURI_ENV_*"],

  build: {
    // macOS 12 ships Safari 15, the oldest WebKit the app supports.
    target: "safari15",
    minify: process.env.TAURI_ENV_DEBUG ? false : "esbuild",
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
  },

  // Vitest. Playwright owns e2e/ with its own runner, so keep Vitest out of it.
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    exclude: [...configDefaults.exclude, "e2e/**"],
    css: false,
  },
});
