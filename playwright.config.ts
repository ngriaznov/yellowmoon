import { defineConfig } from "@playwright/test";

// The e2e suite runs against the Vite dev server with the mocked API (no Tauri).
// Chromium is the build that ships in /opt/pw-browsers (1194 = Playwright 1.56.x).
export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  reporter: [["html", { open: "never" }], ["list"]],
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
