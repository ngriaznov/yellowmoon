import { expect } from "@playwright/test";
import type { Browser, BrowserContext, Page } from "@playwright/test";
import { configureMock, dropFiles, failNext, lookTilesLoaded, openApp, openPhotos, settle, showUpdateReady } from "./app";

// Every page state of the app that can be reached on the mock API: the 30 states of the dark glass interface (C2), the ones its
// screenshots were taken of and its page audit ran over (`e2e/glass.spec.ts`), the two of Colour › Colour warp (2026-10-17) and the three of
// Apple Auto (`docs/19`: the Automatic card with Apple's answer, Preferences with its row, and the instructions for Apple's model edited,
// with Reset to default). A state is a function that leaves a page in it; the numbers in the names are only an order.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;
const LL4 = "ARRI_LookLibrary_LogC4.zip";
const LL3 = "ARRI_LookLibrary_LogC3.zip";

/** The mockup's curves (`src/dev/demoData.ts` has the same numbers). */
const CURVES = {
  luma: [[0, 0], [0.25, 0.191], [0.5, 141 / 255], [0.75, 0.845], [1, 1]],
  red: [[0, 0], [0.25, 0.2], [0.5, 0.57], [0.75, 0.86], [1, 1]],
  green: [[0, 0], [0.5, 0.5], [1, 1]],
  blue: [[0, 0], [0.25, 0.1], [0.5, 0.45], [0.75, 0.82], [1, 1]],
} satisfies Record<string, Array<[number, number]>>;

/** What the mockup's 4th photo has been given: Light's Highlights and Shadows, Colour's wheels and Saturation, a curve. */
const MOCKUP_SETTINGS = {
  DSC04204: {
    highlights: -18,
    shadows: 12,
    saturation: { kind: "manual", value: 6 },
    color_balance: { shadows: [-0.08, -0.12], midtones: [0, 0], highlights: [0.12, 0.12] },
    curve: CURVES,
  },
} as const;

/**
 * Colour › Colour warp on the mockup's photo: two moving pins (the blue of the sky toward teal, the warm white of the sun toward orange)
 * and a lock in the magentas. `e2e/screenshots.spec.ts` draws the same pins for `photos-color-warp.png`.
 */
export const WARP_PINS = [
  { source: [-0.088, -0.064], target: [-0.13, 0.03], locked: false, range: 0.04, shadows: 1, highlights: 1, pivot_ev: 0, exposure_ev: 0 },
  { source: [0.028, 0.032], target: [0.11, 0.07], locked: false, range: 0.04, shadows: 1, highlights: 1, pivot_ev: 0, exposure_ev: 0 },
  { source: [0.12, -0.11], target: [0.12, -0.11], locked: true, range: 0.04, shadows: 1, highlights: 1, pivot_ev: 0, exposure_ev: 0 },
] as const;

type Options = Record<string, string | number>;

/** Start, with the update cell showing "Update ready" as the screenshots do. */
async function start(page: Page, options: Options = { mockLooks: 0 }): Promise<void> {
  await openApp(page, options);
  await showUpdateReady(page);
}

/** The Photos screen on the mockup's state (DSC04204 current, three picked), with the mockup's settings on the current photo. */
async function photos(page: Page, options: Options = MOCKUP, settings: boolean | Record<string, unknown> = true): Promise<void> {
  await openApp(page, options);
  if (settings) {
    const extra = settings === true ? {} : settings;
    await configureMock(page, { initialSettings: { DSC04204: { ...MOCKUP_SETTINGS.DSC04204, ...extra } } as never });
  }
  await openPhotos(page);
  await showUpdateReady(page);
  await page.waitForTimeout(300);
}

const tab = async (page: Page, name: string): Promise<void> => {
  await page.getByRole("tab", { name, exact: true }).click();
  await expect(page.getByRole("tab", { name, exact: true })).toHaveAttribute("aria-selected", "true");
  await page.waitForTimeout(300);
};

const wizardNext = (page: Page) => page.locator(".ym-wizard__footer").getByRole("button", { name: /^(Next|Import|Done)/ });

async function wizard(page: Page, extra: Options = {}): Promise<void> {
  await photos(page, { mockLooks: 0, ...MOCKUP, ...extra }, false);
  await page.getByRole("button", { name: "Import ARRI looks →" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Choose what to import" })).toBeVisible();
}

async function toRecognise(page: Page, ...names: string[]): Promise<void> {
  await wizardNext(page).click();
  await expect(page.getByRole("heading", { level: 1, name: "Download from ARRI" })).toBeVisible();
  await dropFiles(page, names);
  await expect(wizardNext(page)).not.toHaveAttribute("aria-disabled", "true");
  await wizardNext(page).click();
  await expect(page.getByRole("table", { name: "What arrived" })).toBeVisible();
  await expect(page.getByText("Checking", { exact: true })).toHaveCount(0);
}

async function exportPanel(page: Page, options: Options = MOCKUP): Promise<void> {
  await photos(page, options);
  await page.keyboard.press("Meta+e");
  await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toBeVisible();
  await page.waitForTimeout(400);
}

const format = async (page: Page, name: string): Promise<void> => {
  await page.getByRole("radiogroup", { name: "Format" }).getByRole("radio", { name }).click();
  await page.waitForTimeout(300);
};

/** The Colour tab with the warp's pins, scrolled to the Colour warp card, its first pin selected and the photo's colours drawn. */
async function colourWarp(page: Page): Promise<void> {
  await photos(page, MOCKUP, { color_warp: { pins: WARP_PINS } });
  await tab(page, "Colour");
  await expect(page.locator('.ym-warp-plane:not([data-scatter-points="0"])')).toBeVisible();
  await page.getByRole("slider", { name: "Pin 1" }).focus();
  await page.locator(".ym-tools__scroll").evaluate((el) => {
    el.scrollTop = el.scrollHeight;
  });
  await page.waitForTimeout(300);
}

/** A state: its name and what leaves a page in it. */
export type PageState = readonly [name: string, open: (page: Page) => Promise<void>];

export const STATES: readonly PageState[] = [
  ["01-start-empty", (p) => start(p)],
  [
    "02-start-developing",
    async (p) => {
      await start(p, { mockHold: 5.4, mockLooks: 0 });
      await p.getByRole("button", { name: "Open files…" }).click();
      await expect(p.getByRole("list", { name: "Photos being developed" }).getByText("Ready", { exact: true })).toHaveCount(5);
    },
  ],
  [
    "03-start-ready",
    async (p) => {
      await start(p);
      await p.getByRole("button", { name: "Open files…" }).click();
      await expect(p.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    },
  ],
  [
    "04-start-no-arw",
    async (p) => {
      await start(p);
      await dropFiles(p, ["holiday.jpg", "notes.txt"]);
      await expect(p.getByRole("alert")).toBeVisible();
    },
  ],
  ["05-wizard-1-choose", (p) => wizard(p)],
  [
    "06-wizard-2-get-files",
    async (p) => {
      await wizard(p);
      await wizardNext(p).click();
      await p.getByRole("switch", { name: "Watch my Downloads folder" }).click();
      await expect(p.getByText("Watching Downloads for Look Library · LogC4…")).toBeVisible();
    },
  ],
  [
    "07-wizard-3-recognise",
    async (p) => {
      await wizard(p);
      await toRecognise(p, LL4, LL3, "Warm_Film.cube");
    },
  ],
  [
    "08-wizard-4-importing",
    async (p) => {
      await wizard(p, { mockImportHold: 40 });
      await toRecognise(p, LL4);
      await wizardNext(p).click();
      await expect(p.getByText("of 87 files")).toBeVisible();
    },
  ],
  [
    "09-wizard-5-finish",
    async (p) => {
      await wizard(p);
      await toRecognise(p, LL4);
      await wizardNext(p).click();
      await expect(p.getByRole("heading", { level: 1, name: "Imported" })).toBeVisible({ timeout: 15_000 });
      await wizardNext(p).click();
      await expect(p.getByRole("heading", { level: 1, name: "87 looks ready" })).toBeVisible();
      await expect(p.locator(".ym-wizard__tile-picture img")).toHaveCount(7);
    },
  ],
  [
    "10-tab-look",
    async (p) => {
      await photos(p);
      await lookTilesLoaded(p);
    },
  ],
  ["11-tab-light", async (p) => (await photos(p), tab(p, "Light"))],
  ["12-tab-colour", async (p) => (await photos(p), tab(p, "Colour"))],
  ["13-tab-detail", async (p) => (await photos(p), tab(p, "Detail"))],
  ["14-tab-curve", async (p) => (await photos(p), tab(p, "Curve"))],
  ["15-tab-raw", async (p) => (await photos(p), tab(p, "RAW"))],
  [
    "16-mode-crop",
    async (p) => {
      await photos(p);
      await p.getByRole("button", { name: "Crop", exact: true }).click();
      await p.waitForTimeout(500);
    },
  ],
  [
    "17-mode-pick-white",
    async (p) => {
      await photos(p);
      await p.getByRole("button", { name: "Pick white", exact: true }).click();
      await p.mouse.move(420, 300);
      await p.waitForTimeout(300);
    },
  ],
  [
    "18-before-on",
    async (p) => {
      await photos(p);
      await p.getByRole("button", { name: "Before", exact: true }).click();
      await expect(p.getByText("Before · camera default")).toBeVisible();
    },
  ],
  [
    "19-pixel-readout",
    async (p) => {
      await photos(p);
      await p.keyboard.down("Alt");
      await p.mouse.move(380, 330);
      await p.mouse.move(420, 320);
      await p.locator(".ym-photo-cell__readout").waitFor({ timeout: 5000 }).catch(() => undefined);
      await p.waitForTimeout(300);
    },
  ],
  [
    "20-dialog-preferences",
    async (p) => {
      await photos(p);
      await p.keyboard.press("Meta+,");
      await expect(p.getByRole("dialog", { name: "Preferences" })).toBeVisible();
      await p.waitForTimeout(400);
    },
  ],
  [
    "21-dialog-import-cube",
    async (p) => {
      await photos(p);
      await p.evaluate(() => {
        const { mockApi } = window.__yellowmoon!;
        mockApi.queueChoice("files", ["~/LUTs/Warm_Film.cube", "~/LUTs/Cool_Day.cube"]);
        mockApi.emitEvent("menu_action", { action: "import_cube" });
      });
      await expect(p.getByRole("dialog", { name: "Import LUTs" })).toBeVisible();
      await p.waitForTimeout(400);
    },
  ],
  [
    "22-dialog-about",
    async (p) => {
      await photos(p);
      await p.evaluate(() => window.__yellowmoon!.mockApi.emitEvent("menu_action", { action: "about" }));
      await expect(p.getByRole("dialog", { name: "About Yellowmoon" })).toBeVisible();
      await p.waitForTimeout(400);
    },
  ],
  [
    "23-dialog-close-session",
    async (p) => {
      await exportPanel(p, { ...MOCKUP, mockExportHold: 1.45 });
      await p.getByRole("button", { name: /^Export 3 photos/ }).click();
      await p.waitForTimeout(600);
      await p.keyboard.press("Meta+w");
      await expect(p.getByRole("dialog")).toBeVisible();
      await p.waitForTimeout(300);
    },
  ],
  [
    "24-error-bar",
    async (p) => {
      await start(p);
      await failNext(p, "openDialog", { scope: "open", message: "That folder can't be read. Choose another." });
      await p.getByRole("button", { name: "Open folder…" }).click();
      await expect(p.getByRole("alert")).toBeVisible();
    },
  ],
  ["25-export-jpeg", (p) => exportPanel(p)],
  ["26-export-tiff", async (p) => (await exportPanel(p), format(p, "TIFF"))],
  ["27-export-png", async (p) => (await exportPanel(p), format(p, "PNG"))],
  ["28-export-dng", async (p) => (await exportPanel(p), format(p, "DNG"))],
  [
    "29-export-running",
    async (p) => {
      await exportPanel(p, { ...MOCKUP, mockExportHold: 1.45 });
      await p.getByRole("button", { name: /^Export 3 photos/ }).click();
      await p.waitForTimeout(900);
    },
  ],
  // The same run with the panel closed: the Export cell of the header carries the progress strip.
  [
    "30-export-running-strip",
    async (p) => {
      await exportPanel(p, { ...MOCKUP, mockExportHold: 1.45 });
      await p.getByRole("button", { name: /^Export 3 photos/ }).click();
      await p.waitForTimeout(600);
      await p.getByRole("button", { name: "← Back to photos" }).click();
      await expect(p.locator("[data-export-strip]")).toBeVisible();
    },
  ],
  // Colour › Colour warp (the owner's request of 2026-10-17): the card with two moving pins and a lock, and Pick colour on the photo.
  ["31-colour-warp", (p) => colourWarp(p)],
  [
    "32-mode-pick-colour",
    async (p) => {
      await colourWarp(p);
      await p.getByRole("button", { name: "Pick colour", exact: true }).click();
      await expect(p.getByText("Click a colour in the photo.")).toBeVisible();
      await p.mouse.move(420, 300);
      await p.waitForTimeout(300);
    },
  ],
  // Apple Auto (docs/19 sections 2 and 10), on a Mac where it is available: the Automatic card with a row for each value Apple's answer
  // set, its line and the plan; Preferences with the row "Auto with Apple Intelligence" and the instructions under it; and the instructions
  // edited, with Reset to default enabled next to the field.
  [
    "33-tab-look-apple-auto",
    async (p) => {
      await photos(p, { ...MOCKUP, mockAppleAuto: "answered" });
      await expect(p.getByText("Set with Apple Intelligence.")).toBeVisible();
    },
  ],
  [
    "34-dialog-preferences-apple-auto",
    async (p) => {
      await photos(p, { ...MOCKUP, mockAppleAuto: "answered" });
      await p.keyboard.press("Meta+,");
      await expect(p.getByRole("switch", { name: "Auto with Apple Intelligence" })).toBeVisible();
      await p.waitForTimeout(400);
    },
  ],
  [
    "35-dialog-preferences-apple-instructions",
    async (p) => {
      await photos(p, { ...MOCKUP, mockAppleAuto: "answered" });
      await p.keyboard.press("Meta+,");
      const field = p.getByRole("textbox", { name: "Instructions for Apple Intelligence" });
      await field.fill("Grade this photo as a warm, quiet documentary still. Keep skin natural and the blacks soft.");
      await field.blur();
      const reset = p.getByRole("button", { name: "Reset to default" });
      await expect(reset).toBeEnabled();
      await reset.scrollIntoViewIfNeeded();
      await p.waitForTimeout(400);
    },
  ],
];

/** Opens one state in a new page of the browser, at the size of the screenshots. */
export async function openState(browser: Browser, name: string): Promise<{ page: Page; context: BrowserContext }> {
  const def = STATES.find(([n]) => n === name);
  if (!def) throw new Error(`there is no page state called ${name}`);
  const context = await browser.newContext({
    baseURL: "http://localhost:1420",
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();
  await def[1](page);
  await settle(page);
  return { page, context };
}
