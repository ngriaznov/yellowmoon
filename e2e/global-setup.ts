import { chromium } from "@playwright/test";
import type { FullConfig } from "@playwright/test";

/**
 * Loads both pages once before the first test, so the dev server has transformed every module and the first test of each
 * worker does not pay for it. On a busy machine that cold start was longer than a test's patience.
 */
export default async function globalSetup(config: FullConfig): Promise<void> {
  const baseURL = config.projects[0]?.use.baseURL ?? "http://localhost:1420";
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    for (const path of ["/?mockSpeed=Infinity", "/components.html"]) {
      await page.goto(`${baseURL}${path}`, { waitUntil: "networkidle", timeout: 120_000 });
      await page.evaluate(() => document.fonts.ready);
    }
  } finally {
    await browser.close();
  }
}
