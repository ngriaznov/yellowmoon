import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect } from "@playwright/test";
import type { Page } from "@playwright/test";
import { settle } from "./app";
import { dumpBoxes } from "./domDump";
import { pixelDiff } from "./pixelDiff";

/** Where the references live: `design/screenshots-c2/*.png`, the pictures of the dark glass interface (`playwright.config.ts` points there). */
export const REFERENCES = resolve(import.meta.dirname, "../../design/screenshots-c2");
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
 * Share of pixels (Playwright's count, 0–1) a state may differ from its reference by, where it needs more than the default of
 * `playwright.config.ts` (0.35 %). None does: the references are this app's own pictures, and two repeat runs on the machine that took them
 * (2026-10-16, twice over all 24 states) differ from them by 0.000 % of the pixels, at Playwright's count and at the count of more than
 * 24 on a channel. The 0.35 % is room for another machine (a CI runner's GPU and Chromium build); if a state needs more there, it gets a
 * line here with the number measured and the date, as the states of the pre-glass comparison did.
 */
const BUDGETS: Readonly<Record<string, number>> = {
  // The consent dialog over Preferences (2026-10-26): the Preferences text under the overlay is drawn one CSS pixel to the left in about
  // half of the runs (the boxes measure the same in both; 1.595 % of the pixels differ at Playwright's count, 0.000 % in the other runs),
  // so the state may differ by 2 %. The dialog itself and everything outside the overlay's rim are identical in both.
  "photos-preferences-consent": 0.02,
};

/** The reference PNG of a state. */
export const reference = (name: string): Buffer => readFileSync(resolve(REFERENCES, `${name}.png`));

/**
 * Compares the page with a reference: `toHaveScreenshot` is the assertion (tolerance in `playwright.config.ts`), and the share
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
    // For looking at the boxes and text styles of a state without re-running the test (see helpers/domDump.ts).
    mkdirSync(DOM_DIR, { recursive: true });
    writeFileSync(resolve(DOM_DIR, `${name}.json`), JSON.stringify(await page.evaluate(dumpBoxes)));
  }
  // A state with no reference yet (the first run, or a new state) has nothing to count against: `toHaveScreenshot` writes the
  // reference when the run is `--update-snapshots`, and fails the test otherwise.
  let mismatch = 0;
  if (existsSync(resolve(REFERENCES, `${name}.png`))) {
    const diff = await pixelDiff(page.context().browser()!, png, reference(name), { threshold: 24 });
    mismatch = diff.mismatch;
    mkdirSync(REPORT_DIR, { recursive: true });
    writeFileSync(resolve(REPORT_DIR, `${name}.json`), JSON.stringify({ name, mismatch: diff.mismatch, pixels: diff.pixels }));
    console.log(`[screenshot] ${name}: ${(diff.mismatch * 100).toFixed(3)} % of pixels differ by more than 24`);
  } else {
    console.log(`[screenshot] ${name}: no reference yet`);
  }
  const budget = options.maxDiffPixelRatio ?? BUDGETS[name];
  await expect(page).toHaveScreenshot(`${name}.png`, { animations: "disabled", ...(budget === undefined ? {} : { maxDiffPixelRatio: budget }) });
  return mismatch;
}
