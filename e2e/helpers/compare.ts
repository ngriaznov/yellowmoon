import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect } from "@playwright/test";
import type { Page } from "@playwright/test";
import { settle } from "./app";
import { dumpBoxes } from "./domDump";
import { pixelDiff } from "./pixelDiff";

/** Where the baselines live: byte-identical copies of `design/screenshots/*.png` (a test of each spec checks that). */
export const BASELINES = resolve(import.meta.dirname, "../__screenshots__");
export const DESIGN = resolve(import.meta.dirname, "../../design/screenshots");
/**
 * Where the reports go. Playwright empties `test-results/` at the start of every run, so a run of one spec would erase the
 * numbers of the others; this folder, inside the ignored `node_modules`, is only ever overwritten state by state.
 */
const OUT = resolve(import.meta.dirname, "../../node_modules/.cache/yellowmoon-e2e");
const REPORT_DIR = resolve(OUT, "screenshot-diffs");
/** The page as captured, so a difference can be looked at without re-running the test. */
const ACTUAL_DIR = resolve(OUT, "screenshot-actual");
const DOM_DIR = resolve(OUT, "screenshot-dom");

/**
 * Share of pixels (Playwright's count, 0–1) a state may differ from its mockup screenshot by, where it differs on purpose; every
 * other state gets the default of `playwright.config.ts` (0.35 %). Measured on 2026-10-12.
 */
const BUDGETS: Readonly<Record<string, number>> = {
  // The Light tab's histogram and the pixel readout are drawn from the photo; the mockup's are made-up numbers.
  "photos-light": 0.01,
  // The DNG note carries Adobe's notice and the white-balance caveat (docs/09 §Legal): three lines more than the mockup.
  "export-dng": 0.02,
  // The mockup shows the zip as arrived and Next ready; the app waits for the file, so the footer's line and button differ.
  "import-2-get-files": 0.006,
};

/** The baseline PNG of a state. */
export const baseline = (name: string): Buffer => readFileSync(resolve(BASELINES, `${name}.png`));

/** True when the copy in `e2e/__screenshots__` is the very file of `design/screenshots`. */
export function isCopyOfDesign(name: string): boolean {
  return baseline(name).equals(readFileSync(resolve(DESIGN, `${name}.png`)));
}

/**
 * Compares the page with a baseline: `toHaveScreenshot` is the assertion (tolerance in `playwright.config.ts`), and the share
 * of pixels that differ by more than 24 on a channel, the measure of the M3-01 report, is printed and kept in
 * `node_modules/.cache/yellowmoon-e2e/screenshot-diffs/<state>.json`. Returns that share, 0–1.
 */
export async function compareScreen(page: Page, name: string, options: { maxDiffPixelRatio?: number } = {}): Promise<number> {
  await settle(page);
  await page.mouse.move(1000, 400);
  await page.waitForTimeout(100);
  const png = await page.screenshot({ animations: "disabled" });
  mkdirSync(ACTUAL_DIR, { recursive: true });
  writeFileSync(resolve(ACTUAL_DIR, `${name}.png`), png);
  if (process.env.YM_DOM_DUMP) {
    // For comparing boxes and text styles with the mockup in the same state (see helpers/domDump.ts).
    mkdirSync(DOM_DIR, { recursive: true });
    writeFileSync(resolve(DOM_DIR, `${name}.json`), JSON.stringify(await page.evaluate(dumpBoxes)));
  }
  const diff = await pixelDiff(page.context().browser()!, png, baseline(name), { threshold: 24 });
  mkdirSync(REPORT_DIR, { recursive: true });
  writeFileSync(resolve(REPORT_DIR, `${name}.json`), JSON.stringify({ name, mismatch: diff.mismatch, pixels: diff.pixels }));
  console.log(`[screenshot] ${name}: ${(diff.mismatch * 100).toFixed(3)} % of pixels differ by more than 24`);
  const budget = options.maxDiffPixelRatio ?? BUDGETS[name];
  await expect(page).toHaveScreenshot(`${name}.png`, { animations: "disabled", ...(budget === undefined ? {} : { maxDiffPixelRatio: budget }) });
  return diff.mismatch;
}
