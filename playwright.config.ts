import { defineConfig } from "@playwright/test";

// The e2e suite runs against the Vite dev server with the mocked API (no Tauri).
// Chromium is the build that ships in /opt/pw-browsers (1194 = Playwright 1.56.x).
export default defineConfig({
  testDir: "e2e",
  // Every spec runs on the mock with timers it controls, so nothing here waits for real time. The generous limits are for a busy
  // machine (CI runners share their CPUs), not for slow tests.
  timeout: 60_000,
  globalSetup: "./e2e/global-setup.ts",
  reporter: [["html", { open: "never" }], ["list"]],
  // Baseline screenshots are copies of design/screenshots/*.png (2880 × 1800), kept in one folder
  // under their own names, whatever the platform: `await expect(page).toHaveScreenshot("start-idle.png")`.
  snapshotPathTemplate: "{testDir}/__screenshots__/{arg}{ext}",
  expect: {
    timeout: 10_000,
    // The mockups are drawn with CSS shapes and the app with SVG, so edges anti-alias differently; that costs at most 0.25 % of the
    // pixels (Playwright's own count at `threshold`, which forgives anti-aliasing). `maxDiffPixelRatio` is the share allowed to
    // differ, so a real change (a moved line of text, a wrong colour) fails. The few states that differ on purpose have their own
    // budget in `e2e/helpers/compare.ts`.
    toHaveScreenshot: { threshold: 0.2, maxDiffPixelRatio: 0.0035, scale: "device" },
  },
  use: {
    baseURL: "http://localhost:1420",
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:1420",
    reuseExistingServer: !process.env.CI,
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
