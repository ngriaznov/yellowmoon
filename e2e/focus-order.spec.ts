import { expect, test } from "@playwright/test";
import type { Browser, BrowserContext, Page } from "@playwright/test";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { openApp } from "./helpers/app";
import { classifyRing, describeStop, roleAndName, walkTabs } from "./helpers/focusWalk";
import type { Walk } from "./helpers/focusWalk";
import { openState } from "./helpers/states";

// The golden Tab lists (`docs/28` A-04, M4-04): for each of 26 page states (and the Photos screen with 500 photos), press Tab from the start
// of the page until the focus wraps or leaves the document, and compare each stop (its role, its accessible name, its computed `outline-style`
// and how the ring shows) with the list kept in `e2e/golden/focus-order/<state>.json`.
//
// The test fails when a stop is added, removed, reordered, renamed, or has no ring. A change that is meant (a new control) is taken into the
// goldens on purpose, by running it with YM_UPDATE_FOCUS_GOLDEN=1: the files are then rewritten and the diff goes into the same commit. The
// test never writes them on its own, and it never writes a list with a stop that has no ring.

const GOLDEN = resolve(import.meta.dirname, "golden/focus-order");
const UPDATE = process.env.YM_UPDATE_FOCUS_GOLDEN === "1";

/**
 * The 26 states of the audit's table (`docs/28` 2.4): Start empty and ready; the Photos screen on each of its seven tabs; Crop and Pick white;
 * Colour warp; Preferences, About, the .cube import and the question before closing a session; the error bar; the Export panel and a running
 * export (their strips are the two scrollers of A-02); the node view (default, Light, masks, Film, Parallel); the wizard's first and last step.
 */
const STATE_NAMES = [
  "01-start-empty",
  "03-start-ready",
  "10-tab-look",
  "11-tab-light",
  "12-tab-colour",
  "13-tab-detail",
  "14-tab-curve",
  "15-tab-raw",
  "37-tab-lens",
  "16-mode-crop",
  "17-mode-pick-white",
  "31-colour-warp",
  "20-dialog-preferences",
  "22-dialog-about",
  "21-dialog-import-cube",
  "23-dialog-close-session",
  "24-error-bar",
  "25-export-jpeg",
  "29-export-running",
  "42-nodes-default",
  "54-nodes-corrector-light",
  "58-nodes-masks-two-rows",
  "67-film-tab",
  "78-mixers-parallel-inputs",
  "05-wizard-1-choose",
  "09-wizard-5-finish",
] as const;

/** The Photos screen with 500 photos: the filmstrip is windowed there. */
const MANY = "many-photos-500";

/** The 500-photo Photos screen, as `e2e/axe.spec.ts` opens it. */
async function manyPhotos(browser: Browser): Promise<{ page: Page; context: BrowserContext }> {
  const context = await browser.newContext({ baseURL: "http://localhost:1420", viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4", mockPhotos: 500 });
  await page.getByRole("button", { name: "Open files…" }).click();
  await page.getByRole("button", { name: /Open photos/ }).click({ timeout: 30_000 });
  await expect(page.getByRole("tab", { name: "Look" })).toBeVisible();
  await expect(page.locator(".ym-filmstrip__thumb").first()).toBeVisible();
  // Which photo Open photos shows depends on how many of the 500 are developed by then (the mock's current photo, or the first), and the
  // strip's one Tab stop is the current thumbnail. Choosing the mockup's photo makes the list the same on every run.
  const mockup = page.locator('.ym-filmstrip__thumb[aria-label^="Show DSC04204"]');
  await mockup.click();
  await expect(mockup).toHaveAttribute("aria-current", "true");
  await expect(mockup).toHaveAttribute("tabindex", "0");
  // Not `settle`: it waits for every image to decode, and 500 photos are still developing; the walk needs no picture.
  await page.evaluate(() => new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))));
  return { page, context };
}

const goldenPath = (name: string): string => resolve(GOLDEN, `${name}.json`);

/** The file's text: one stop to a line, so a change reads as one changed line in a diff. */
function goldenText(name: string, walk: Walk): string {
  const lines = walk.stops.map((stop) => `    ${JSON.stringify(stop)}`);
  return `{\n  "state": ${JSON.stringify(name)},\n  "end": ${JSON.stringify(walk.end)},\n  "stops": [\n${lines.join(",\n")}\n  ]\n}\n`;
}

/** Walks the page and checks it against its golden list. */
async function check(name: string, page: Page): Promise<void> {
  const walk = await walkTabs(page);
  // A stop without a ring is a defect whatever the golden says; the golden must not be able to bless it.
  expect(
    walk.stops.map((stop, i) => ({ stop, i })).filter(({ stop }) => stop.ring === "none").map(({ stop, i }) => describeStop(i, stop)),
    `${name}: Tab stops with no visible focus ring`,
  ).toEqual([]);
  expect(walk.stops.length, `${name}: no Tab stop at all`).toBeGreaterThan(0);

  const file = goldenPath(name);
  if (UPDATE) {
    mkdirSync(GOLDEN, { recursive: true });
    writeFileSync(file, goldenText(name, walk));
    return;
  }
  expect(existsSync(file), `${name}: no golden list at e2e/golden/focus-order/${name}.json. Run this spec with YM_UPDATE_FOCUS_GOLDEN=1 to write it, and read it before committing.`).toBe(true);
  const golden = JSON.parse(readFileSync(file, "utf8")) as Walk;
  const lines = (w: Walk): string[] => w.stops.map((stop, i) => describeStop(i, stop));
  expect(lines(walk), `${name}: the Tab order changed (a stop was added, removed, renamed or reordered). If it is meant, rewrite the goldens with YM_UPDATE_FOCUS_GOLDEN=1.`).toEqual(lines(golden));
  expect(walk.end, `${name}: how the walk ends changed`).toBe(golden.end);
}

test.describe("the golden Tab lists", () => {
  // Each state opens its own page, so they run side by side, as `e2e/axe.spec.ts` does.
  test.describe.configure({ mode: "parallel" });

  for (const name of STATE_NAMES) {
    test(name, async ({ browser }) => {
      test.setTimeout(90_000);
      const { page, context } = await openState(browser, name);
      try {
        await check(name, page);
      } finally {
        await context.close();
      }
    });
  }

  test(MANY, async ({ browser }) => {
    test.setTimeout(120_000);
    const { page, context } = await manyPhotos(browser);
    try {
      await check(MANY, page);
    } finally {
      await context.close();
    }
  });

  test("every golden list belongs to a state of this spec", () => {
    const known = new Set<string>([...STATE_NAMES, MANY]);
    const files = existsSync(GOLDEN) ? readdirSync(GOLDEN).filter((f) => f.endsWith(".json")) : [];
    expect(files.filter((f) => !known.has(f.replace(/\.json$/, ""))), "golden lists of states that are not walked (delete them)").toEqual([]);
  });
});

test.describe("what the walk reads", () => {
  test("a ring is an outline of 2 px or more, or a hard box-shadow of 2 px or more; anything else is none", () => {
    expect(classifyRing("solid", "3px", "none")).toBe("outline");
    expect(classifyRing("solid", "3px", "oklab(0.126 0 0 / 0.6) 0px 2px 6px 0.7px")).toBe("outline");
    expect(classifyRing("none", "3px", "none")).toBe("none");
    expect(classifyRing("solid", "1px", "none")).toBe("none");
    expect(classifyRing("none", "0px", "rgb(255, 194, 14) 0px 0px 0px 3px")).toBe("shadow");
    expect(classifyRing("none", "0px", "oklab(0.8 0.1 0.1 / 0.9) 0px 0px 0px 2px, rgba(0, 0, 0, 0.45) 0px 3px 9px 0px")).toBe("shadow");
    // glasscn's gap between a control and its outline is a hard 2 px shadow in the dark glass tint: not a ring.
    expect(classifyRing("none", "0px", "oklab(0.126 0 0 / 0.92) 0px 0px 0px 2px")).toBe("none");
    expect(classifyRing("none", "0px", "rgba(20, 20, 24, 0.92) 0px 0px 0px 2px")).toBe("none");
    // A drop shadow, an inset shadow, a faint line and a 1 px line are not a ring.
    expect(classifyRing("none", "0px", "rgba(0, 0, 0, 0.45) 0px 3px 9px 0px")).toBe("none");
    expect(classifyRing("none", "0px", "rgb(255, 194, 14) 0px 0px 0px 3px inset")).toBe("none");
    expect(classifyRing("none", "0px", "rgb(255, 194, 14) 0px 0px 0px 1px")).toBe("none");
    expect(classifyRing("none", "0px", "rgba(255, 194, 14, 0.1) 0px 0px 0px 3px")).toBe("none");
  });

  test("the role and name come from the first line of the snapshot", () => {
    expect(roleAndName('- button "Open files ⌘O"')).toEqual({ role: "button", name: "Open files ⌘O" });
    expect(roleAndName('- region "Photos to develop":\n  - img "DSC04201"')).toEqual({ role: "region", name: "Photos to develop" });
    expect(roleAndName(`- 'tab "Colour: warp" [selected]'`)).toEqual({ role: "tab", name: "Colour: warp" });
    expect(roleAndName('- textbox "Search looks": hello')).toEqual({ role: "textbox", name: "Search looks" });
    expect(roleAndName("- generic")).toEqual({ role: "generic", name: "" });
    expect(roleAndName("")).toEqual({ role: "generic", name: "" });
  });
});
