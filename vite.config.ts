import { resolve } from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { configDefaults, defineConfig } from "vitest/config";

const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/ and https://v2.tauri.app/start/frontend/vite/
export default defineConfig({
  plugins: [react(), tailwindcss()],

  // glasscn's components import each other and the helpers through the shadcn alias.
  resolve: { alias: { "@": resolve(import.meta.dirname, "src") } },

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
        // Playwright's HTML report (npm run e2e writes playwright-report/index.html when it ends): Vite ignores test-results/ by default, and
        // an .html file it does not know sends the open pages a "page reload" in the dev server's log.
        "**/playwright-report/**",
      ],
    },
  },
  // Expose Tauri's build variables (TAURI_ENV_PLATFORM, TAURI_ENV_DEBUG, ...) to import.meta.env.
  envPrefix: ["VITE_", "TAURI_ENV_*"],

  build: {
    // Two pages: the app, and the component dev page (docs/15 M3-01: "a Storybook-free
    // components.html dev page shows every component state").
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        components: resolve(import.meta.dirname, "components.html"),
      },
      output: {
        // Two big libraries are their own files, so that no chunk reaches Vite's 500 kB warning: React (react-dom alone is about 180 kB
        // minified) and the node view's graph library (xyflow and what it brings: d3's zoom and drag, zustand). They load with the page,
        // not on demand: the app is local, and a lazy import would make the node view wait for a file it can already have (and its tests
        // wait for a tick).
        manualChunks(id) {
          if (/node_modules\/(react|react-dom|scheduler)\//.test(id)) return "react";
          if (/node_modules\/(@xyflow|d3-[a-z-]+|zustand|classcat|use-sync-external-store)\//.test(id)) return "graph";
          return undefined;
        },
      },
    },
    // The script and style syntax is lowered to Safari 17, which macOS 14 ships: the oldest macOS the app runs on
    // (`minimumSystemVersion` in src-tauri/tauri.conf.json; owner's decision of 2026-10-17, docs/16). The dark glass interface's CSS
    // (Tailwind 4: @layer, :has(), @property, color-mix()) needs WebKit 16.4 or later, so every supported Mac has it.
    target: "safari17",
    minify: process.env.TAURI_ENV_DEBUG ? false : "esbuild",
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
  },

  // Vitest. Playwright owns e2e/ with its own runner, so keep Vitest out of it.
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    exclude: [...configDefaults.exclude, "e2e/**"],
    css: false,
    // The screens render whole panels in jsdom; give a busy machine room before a test is called slow.
    testTimeout: 20_000,
    hookTimeout: 20_000,
  },
});
