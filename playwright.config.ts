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
  // The references are design/screenshots-c2/*.png (2880 × 1800), the 24 states of the dark glass interface (C2), kept in one folder
  // under their own names, whatever the platform: `await expect(page).toHaveScreenshot("start-idle.png")`. The older pictures in
  // design/screenshots belong to the pre-glass design and are kept for history only.
  snapshotPathTemplate: "{testDir}/../design/screenshots-c2/{arg}{ext}",
  expect: {
    timeout: 10_000,
    // The references are this app's own pictures, taken with the flags below, so a repeat run on the same machine differs by next to
    // nothing (see the numbers in `e2e/helpers/compare.ts`). The allowance is for another machine: text and the blur of the glass
    // anti-alias a little differently on another GPU or another Chromium build. `maxDiffPixelRatio` is the share of pixels that may
    // differ (Playwright's count at `threshold`, which forgives anti-aliasing), so a real change (a moved line of text, a wrong
    // colour) fails. The few states that need more have their own budget in `e2e/helpers/compare.ts`.
    toHaveScreenshot: { threshold: 0.2, maxDiffPixelRatio: 0.0035, scale: "device" },
  },
  use: {
    baseURL: "http://localhost:1420",
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
    // Draw text the way the references were drawn: no hinting and grey anti-aliasing. Chromium's headless shell hints fully by
    // default, which moves the glyphs by a fraction of a pixel everywhere: on the GitHub runner (2026-10-15) 1 to 2.5 % more of
    // the pixels differed on every screen with text, and 12 of the 24 screenshot states went over their budget. The browser
    // in the agent sandbox is wrapped to set these two flags itself, and the references were taken there. Naming them here makes the
    // run the same everywhere. (macOS and headed runs ignore them.)
    launchOptions: { args: ["--font-render-hinting=none", "--disable-lcd-text"] },
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:1420",
    reuseExistingServer: !process.env.CI,
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
