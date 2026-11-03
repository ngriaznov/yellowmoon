import { defineConfig } from "@playwright/test";
import base from "./playwright.config";

// The same specs' settings, run on the BUILT app (`vite build`, then `vite preview`) instead of the dev server: `npm run e2e:built`.
// Only the specs in e2e/built/ run here (the pictures of the main suite are taken on the dev server, and the built app's are the same).
export default defineConfig({
  ...base,
  testDir: "e2e/built",
  testIgnore: [],
  globalSetup: undefined,
  use: { ...base.use, baseURL: "http://localhost:4173" },
  webServer: {
    command: "npm run build && npx vite preview --port 4173 --strictPort",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
