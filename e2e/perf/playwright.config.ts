import { defineConfig } from "@playwright/test";

// The browser-side latency measurement of docs/11 (M3-03): 500 photos developing in the background, keys pressed meanwhile.
// It is not part of `npm run e2e` (that suite runs against the dev server, whose React is two to three times slower than the
// shipped one, and its testMatch does not see `*.perf.ts`). This config serves the production build:
//
//   npx playwright test -c e2e/perf/playwright.config.ts          (or e2e/perf/run.sh)
//
// Environment: YM_PERF_OUT (report folder, default test-results/perf), YM_PERF_PORT (default 4173), YM_PERF_SPEED (the mock's
// speed, default 10), YM_PERF_KEYS (key presses, default 80), YM_PERF_P95_MS (the bound on the 95th percentile, default 150).
const port = Number(process.env.YM_PERF_PORT ?? 4173);
const outDir = "node_modules/.cache/ym-perf-dist";

export default defineConfig({
  testDir: ".",
  testMatch: "**/*.perf.ts",
  timeout: 180_000,
  workers: 1,
  reporter: [["list"]],
  outputDir: "../../test-results/perf-artifacts",
  use: {
    baseURL: `http://localhost:${port}`,
    // As the e2e suite: a retina window.
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  },
  webServer: {
    // A fresh production build every time (minified, no HMR, no dev-only checks), served by vite's own static server.
    command: `npx vite build --outDir ${outDir} --emptyOutDir && npx vite preview --outDir ${outDir} --port ${port} --strictPort`,
    url: `http://localhost:${port}`,
    reuseExistingServer: false,
    timeout: 300_000,
    cwd: "../..",
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
