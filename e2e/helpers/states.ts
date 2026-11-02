import { expect } from "@playwright/test";
import type { Browser, BrowserContext, Page } from "@playwright/test";
import { configureMock, dropFiles, failNext, lookTilesLoaded, openApp, openPhotos, settle, showUpdateReady } from "./app";

// Every page state of the app that can be reached on the mock API: the 30 states of the dark glass interface (C2), the ones its
// screenshots were taken of and its page audit ran over (`e2e/glass.spec.ts`), the two of Colour › Colour warp (2026-10-17), the three of
// Apple Auto (`docs/19`: the Automatic card with Apple's answer, Preferences with its row, and the instructions for Apple's model edited,
// with Reset to default), the two of the Auto providers (`docs/22` section 12.9: Preferences with every provider listed and the dropdown
// open, and the consent dialog), the one of Auto crop (`docs/20`: the Crop panel with Auto on and its note) and the three of the Lens tab
// (`docs/21`: no lens, a lens chosen at its start amounts, and the lens data missing), and the one of Pick white with the CPU note under its hint
// (`docs/23` section 8; state 17 is Pick white alone), the node view with its empty canvas (N-07: the toolbar, the divider, the photo switcher;
// N-08 to N-12 add 43 to 48 (N-12 adds 58 to 61, the Mask tab; N-14 adds 62, a Brush's card with a stroke drawn; N-15 adds 63, a Background's card): N-08 has 43, the graph with its Add node menu open, and two of its own, 51 and 52, the graph with a bypassed,
// a dead and a masked card and a key link, and a selected link) and the simple view of a photo that has four nodes, with the notice (state 49). A state is a function that leaves a
// page in it; the numbers in the names are only an order.

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
const MOCKUP_GRADE = {
  highlights: -18,
  shadows: 12,
  saturation: { kind: "manual", value: 6 },
  color_balance: { shadows: [-0.08, -0.12], midtones: [0, 0], highlights: [0.12, 0.12] },
  curve: CURVES,
} as const;

/**
 * Colour › Colour warp on the mockup's photo: two moving pins (the blue of the sky toward teal, the warm white of the sun toward orange)
 * and a lock in the magentas. `e2e/screenshots.spec.ts` draws the same pins for `photos-color-warp.png`.
 */
export const WARP_PINS = [
  { source: [-0.088, -0.064], target: [-0.13, 0.03], locked: false, range: 0.04, shadows: 1, highlights: 1, pivot_ev: 0, exposure_ev: 0, saturation: 0 },
  { source: [0.028, 0.032], target: [0.11, 0.07], locked: false, range: 0.04, shadows: 1, highlights: 1, pivot_ev: 0, exposure_ev: 0, saturation: 0 },
  { source: [0.12, -0.11], target: [0.12, -0.11], locked: true, range: 0.04, shadows: 1, highlights: 1, pivot_ev: 0, exposure_ev: 0, saturation: 0 },
] as const;

type Options = Record<string, string | number>;

/** Start, with the update cell showing "Update ready" as the screenshots do. */
async function start(page: Page, options: Options = { mockLooks: 0 }): Promise<void> {
  await openApp(page, options);
  await showUpdateReady(page);
}

/**
 * What the mockup's 4th photo has when its graph has four nodes (`design/nodes/04-simple-photo-has-nodes.jpg`): Main, then Mountain, Sky and
 * Warmth in a line (a node patch: the new ids give their kind, the links are the line). It is the mock's `initialSettings` of that photo.
 */
export const FOUR_NODES = {
  nodes: {
    "2": { kind: "corrector", name: "Mountain", pos: [1, 0] },
    "3": { kind: "corrector", name: "Sky", pos: [2, 0] },
    "4": { kind: "corrector", name: "Warmth", pos: [3, 0] },
  },
  links: [
    { from: { node: "input", port: "picture" }, to: { node: 1, port: "picture" } },
    { from: { node: 1, port: "picture" }, to: { node: 2, port: "picture" } },
    { from: { node: 2, port: "picture" }, to: { node: 3, port: "picture" } },
    { from: { node: 3, port: "picture" }, to: { node: 4, port: "picture" } },
    { from: { node: 4, port: "picture" }, to: { node: "output", port: "picture" } },
  ],
} as const;

/**
 * The graph of screens 06 and 07 of `design/nodes` (the node view's cards): Main, then Mountain (an Object mask minus a Radial one, a trim of
 * +0.35 EV), Sky (a Linear mask, a cooler trim) and Warmth (a warmer trim), in a line. It is a node patch like `FOUR_NODES` with the masks and
 * trims that give the cards their glyphs; the mock does not render masks, so only the cards and the panel tell them apart.
 */
export const GRADED_NODES = {
  nodes: {
    "2": {
      kind: "corrector",
      name: "Mountain",
      pos: [1, 0],
      trim: { exposure_ev: 0.35, temperature: 0, tint: 0 },
      // Screen 06's Light tab: Exposure trim +0.35 and Highlights −12 on Mountain.
      grade: { highlights: { kind: "manual", value: -12 } },
      masks: {
        invert: false,
        items: [
          { kind: "object", mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, points: [[0.41, 0.33, true], [0.52, 0.31, true]], box: null },
          { kind: "radial", mode: "subtract", invert: false, enabled: true, opacity: 1, center: [0.55, 0.32], radii: [0.06, 0.03], angle_deg: 0, feather: 0.45 },
        ],
      },
    },
    "3": {
      kind: "corrector",
      name: "Sky",
      pos: [2, 0],
      trim: { exposure_ev: 0, temperature: -18, tint: 4 },
      masks: { invert: false, items: [{ kind: "linear", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.2], angle_deg: 4, feather: 0.25 }] },
    },
    "4": { kind: "corrector", name: "Warmth", pos: [3, 0], trim: { exposure_ev: 0, temperature: 30, tint: 0 } },
  },
  links: FOUR_NODES.links,
} as const;

/** The Photos screen on the mockup's state (DSC04204 current, three picked), with the mockup's settings on the current photo. */
async function photos(page: Page, options: Options = MOCKUP, settings: boolean | Record<string, unknown> = true): Promise<void> {
  await openApp(page, options);
  if (settings) {
    // The settings are a graph of nodes: the grade is node 1's, Main (`docs/23` section 3).
    const extra = settings === true ? {} : settings;
    await configureMock(page, { initialSettings: { DSC04204: { nodes: { "1": { grade: { ...MOCKUP_GRADE, ...extra } } } } } as never });
  }
  await openPhotos(page);
  await showUpdateReady(page);
  await page.waitForTimeout(300);
}

/** The node view of the graph of `GRADED_NODES`, opened with the Nodes button, with `select` (a node's number or "input", "output") on. */
async function nodeViewOfGradedPhoto(page: Page, select: number | "input" | "output" = 2): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, {
    initialSettings: { DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE }, ...GRADED_NODES.nodes }, links: GRADED_NODES.links } } as never,
  });
  await openPhotos(page);
  await showUpdateReady(page);
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await expect(page.locator('[data-node-card="4"]')).toBeVisible();
  await page.locator(`[data-node-card="${select}"]`).click();
  await expect(page.locator(`[data-node-card="${select}"]`)).toHaveAttribute("aria-pressed", "true");
  await page.waitForTimeout(400);
}

/**
 * The graph of `GRADED_NODES` with Mountain's masks replaced by `rows`, on the Mask tab of node `select`, the node view open. `kinds` is the mock's
 * `?mockMasks=` (the kinds `mask_kinds` says this machine can make; default: the four that are built).
 */
async function maskTabOfGradedPhoto(page: Page, select: number, rows?: readonly object[], kinds?: string): Promise<void> {
  await openApp(page, kinds ? { ...MOCKUP, mockMasks: kinds } : MOCKUP);
  const mountain = { ...GRADED_NODES.nodes["2"], ...(rows ? { masks: { invert: false, items: rows } } : {}) };
  await configureMock(page, {
    initialSettings: { DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE }, ...GRADED_NODES.nodes, "2": mountain }, links: GRADED_NODES.links } } as never,
  });
  await openPhotos(page);
  await showUpdateReady(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await expect(page.locator('[data-node-card="4"]')).toBeVisible();
  await page.locator(`[data-node-card="${select}"]`).click();
  await expect(page.locator(`[data-node-card="${select}"]`)).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("tab", { name: "Mask", exact: true }).click();
  await expect(page.getByRole("region", { name: "Masks" })).toBeVisible();
  await page.waitForTimeout(400);
}

/** Screen 07b's rows: an Object (two points, Add) and a Radial that is Subtract and inverted. */
export const ROWS_07B = [
  { kind: "object", mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, points: [[0.41, 0.33, true], [0.52, 0.31, true]], box: null },
  { kind: "radial", mode: "subtract", invert: true, enabled: true, opacity: 1, center: [0.55, 0.32], radii: [0.06, 0.03], angle_deg: 0, feather: 0.45 },
] as const;

/** A Brush row with no stroke yet (N-14): the state draws its stroke by hand. */
export const BRUSH_ROW = { kind: "brush", mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0, strokes: [] } as const;

/** An Object row with two clicks on the mock's mountain (N-18), with the eye on. */
export const OBJECT_ROW = { kind: "object", mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, points: [[0.42, 0.7, true], [0.6, 0.72, true]], box: null } as const;

/** A Background row at its defaults (N-15), with the eye on. */
export const BACKGROUND_ROW = { kind: "background", mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, level: 0.5, softness: 0.5 } as const;

/** A Colour range on Warmth (node 4), whose Mask tab it is. */
export const WARMTH_RANGE = { kind: "color_range", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.06, 0.04], width: 0.05, softness: 0.5 } as const;

/** The Photos screen as `photos` leaves it, on a photo whose graph has four nodes: Main (with the mockup's grade) and three more. */
async function photosWithNodes(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, {
    initialSettings: { DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE }, ...FOUR_NODES.nodes }, links: FOUR_NODES.links } } as never,
  });
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
    // Pick white is a row of the Colour tab's White balance card (docs/23 section 8): the toggle pressed, the hint as a pill on the photo
    // and the crosshair under the pointer. The header has nothing of it.
    async (p) => {
      await photos(p);
      await tab(p, "Colour");
      await p.getByRole("button", { name: "Pick white", exact: true }).click();
      await expect(p.getByText("Click something neutral: grey or white. Esc cancels.")).toBeVisible();
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
  // set, its line and the plan; Preferences with the row "Auto grading" and the instructions under it; and the instructions edited, with
  // Reset to default enabled next to the field.
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
      await expect(p.getByRole("combobox", { name: "Auto grading" })).toBeVisible();
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
  // Auto crop (docs/20 section 1, the owner's request of 2026-10-22): the Crop panel with the Frame card's AUTO on, the photo levelled by
  // the mock's 1.4°, its frame drawn over it, and the note under the card.
  [
    "36-crop-auto",
    async (p) => {
      await photos(p);
      await p.getByRole("button", { name: "Crop", exact: true }).click();
      await p.getByRole("button", { name: "Auto crop" }).click();
      await expect(p.getByText("Levelled by 1.4°.")).toBeVisible();
      await p.waitForTimeout(500);
    },
  ],
  // The Lens tab (docs/21 section 1, the owner's request of 2026-10-25): the list of seven lenses with Default chosen and the amounts at
  // 0; Petzval chosen, with its start amounts on the sliders; and the lens data missing (one row off, the sliders off, the note).
  ["37-tab-lens", async (p) => (await photos(p), tab(p, "Lens"))],
  [
    "38-tab-lens-chosen",
    async (p) => {
      await photos(p);
      await tab(p, "Lens");
      await p.getByRole("radio", { name: /^Petzval / }).click();
      await expect(p.getByRole("textbox", { name: "Edges value" })).toHaveValue("70");
      await p.waitForTimeout(300);
    },
  ],
  [
    "39-tab-lens-missing",
    async (p) => {
      await photos(p, { ...MOCKUP, mockLenses: 0 });
      await tab(p, "Lens");
      await expect(p.getByText("Lens data are missing, so lenses can't be used.")).toBeVisible();
    },
  ],
  // The Auto providers (docs/22 section 12.9): Preferences with every provider listed (Apple Intelligence available, Claude Code
  // available, Antigravity signed out) and the dropdown open, with Claude Code the saved choice; and the consent dialog over Preferences
  // when Claude Code is chosen for the first time.
  [
    "40-dialog-preferences-providers",
    async (p) => {
      await photos(p, { ...MOCKUP, mockProviders: "apple,claude,agy:signed_out", mockAutoProvider: "claude", mockConsent: "claude", mockAppleAuto: "answered" });
      await p.keyboard.press("Meta+,");
      const select = p.getByRole("combobox", { name: "Auto grading" });
      await expect(select).toHaveText("Claude Code");
      await select.click();
      await expect(p.getByRole("option", { name: "Antigravity (agy), experimental, not signed in" })).toBeVisible();
      await p.waitForTimeout(400);
    },
  ],
  [
    "41-dialog-consent",
    async (p) => {
      await photos(p, { ...MOCKUP, mockProviders: "apple,claude,agy:signed_out", mockAppleAuto: "answered" });
      await p.keyboard.press("Meta+,");
      await p.getByRole("combobox", { name: "Auto grading" }).click();
      await p.getByRole("option", { name: "Claude Code" }).click();
      await expect(p.getByRole("dialog", { name: "Send photos to Claude Code?" })).toBeVisible();
      await p.waitForTimeout(400);
    },
  ],
  // The node view (docs/23 section 7.1, N-07 and N-08; screen 05 of design/nodes), opened with the Nodes button: the photo on top, the divider, the
  // graph under it with its toolbar, Main between the Input and the Output with its ports and the line under its link, the zoom control, the
  // photo switcher in place of the filmstrip, the panel as before.
  [
    "42-nodes-default",
    async (p) => {
      await photos(p);
      // The node view opens on the node and the tab of the simple view's tab: Light is Main's (N-09, `docs/23` §7.5).
      await tab(p, "Light");
      await p.getByRole("button", { name: "Nodes", exact: true }).click();
      await expect(p.getByRole("separator", { name: "Resize the graph" })).toBeVisible();
      await expect(p.getByRole("group", { name: "Photos" })).toContainText("4 of 12");
      await expect(p.locator('[data-node-card="1"]')).toBeVisible();
      await p.waitForTimeout(300);
    },
  ],
  // The Add node menu over the graph (screen 06): Mountain selected in the graded graph, the menu opened upward from the button.
  [
    "43-nodes-add-node",
    async (p) => {
      await nodeViewOfGradedPhoto(p, 2);
      await p.getByRole("button", { name: "Add node" }).click();
      await expect(p.getByRole("menu", { name: "Add node" })).toBeVisible();
      await p.waitForTimeout(400);
    },
  ],
  // The graph with every state a card has (N-08): a masked card with both mask glyphs, a bypassed one, one that is not connected to the output
  // (dimmed), a key link in the mask accent and the Input selected.
  [
    "51-nodes-graph-states",
    async (p) => {
      await nodeViewOfGradedPhoto(p, 3);
      await p.keyboard.press("b");
      await expect(p.locator('[data-node-card="3"]')).toHaveAccessibleName(/bypassed/);
      // Sky's key feeds Warmth's key input; Sky's picture comes from Main, which leaves Mountain with nothing after it (dead).
      const from = await p.locator('.ym-port__hit[data-port-node="3"][data-port-name="key"][data-port-side="out"]').boundingBox();
      const to = await p.locator('.ym-port__hit[data-port-node="4"][data-port-name="key"][data-port-side="in"]').boundingBox();
      await p.mouse.move(from!.x + from!.width / 2, from!.y + from!.height / 2);
      await p.mouse.down();
      await p.mouse.move(to!.x + to!.width / 2, to!.y + to!.height / 2, { steps: 10 });
      await p.mouse.up();
      await expect(p.locator('.react-flow__edge[data-id="3.key>4.key"]')).toHaveCount(1);
      const main = await p.locator('.ym-port__hit[data-port-node="1"][data-port-name="picture"][data-port-side="out"]').boundingBox();
      const sky = await p.locator('.ym-port__hit[data-port-node="3"][data-port-name="picture"][data-port-side="in"]').boundingBox();
      await p.mouse.move(main!.x + main!.width / 2, main!.y + main!.height / 2);
      await p.mouse.down();
      await p.mouse.move(sky!.x + sky!.width / 2, sky!.y + sky!.height / 2, { steps: 10 });
      await p.mouse.up();
      await expect(p.locator('[data-node-card="2"]')).toHaveAccessibleName(/not connected/);
      await p.waitForTimeout(300);
    },
  ],
  // A selected link (N-08): thick and yellow, with the Delete link menu from a right click on it.
  [
    "52-nodes-link-selected",
    async (p) => {
      await nodeViewOfGradedPhoto(p, 2);
      const a = await p.locator('.ym-port__hit[data-port-node="2"][data-port-name="picture"][data-port-side="out"]').boundingBox();
      const b = await p.locator('.ym-port__hit[data-port-node="3"][data-port-name="picture"][data-port-side="in"]').boundingBox();
      const at = { x: (a!.x + a!.width / 2 + b!.x + b!.width / 2) / 2, y: a!.y + a!.height / 2 };
      await p.mouse.click(at.x, at.y, { button: "right" });
      await expect(p.getByRole("menuitem", { name: "Delete link" })).toBeVisible();
      await p.waitForTimeout(300);
    },
  ],
  // The node's panel (docs/23 section 7.5, N-09): the Input's Colour tab with the White balance card (screen 10 of design/nodes), a corrector's
  // Light tab with the trim card (screen 06) and its Colour tab with Temperature and tint and the Colour warp (screen 09), and the Output's
  // Look tab (screen 11). The graph is the graded one of the mockups, with the node on the graph selected.
  [
    "53-nodes-input-colour",
    async (p) => {
      await nodeViewOfGradedPhoto(p, "input");
      await tab(p, "Colour");
      await expect(p.getByRole("button", { name: "Pick white" })).toBeVisible();
    },
  ],
  [
    "54-nodes-corrector-light",
    async (p) => {
      await nodeViewOfGradedPhoto(p, 2);
      await expect(p.getByRole("slider", { name: "Exposure trim" })).toBeVisible();
      await expect(p.locator(".ym-histogram path").first()).toHaveAttribute("d", /L/);
    },
  ],
  [
    "55-nodes-corrector-colour",
    async (p) => {
      await nodeViewOfGradedPhoto(p, 3);
      await tab(p, "Colour");
      await expect(p.getByRole("slider", { name: "Temperature" })).toBeVisible();
      await expect(p.locator('.ym-warp-plane:not([data-scatter-points="0"])')).toBeVisible();
    },
  ],
  [
    "56-nodes-output-look",
    async (p) => {
      await nodeViewOfGradedPhoto(p, "output");
      await expect(p.getByRole("slider", { name: "Intensity" })).toBeVisible();
    },
  ],
  // Show mask (docs/23 section 5.8 and 7.6, N-11): Mountain's Mask tab with the Masks card and the View card under it, the view on (the toolbar's
  // Show mask pressed, the Overlay segment chosen). Mountain has an Object and a Radial row; the mock draws the Radial over the photo.
  [
    "57-nodes-mask-view",
    async (p) => {
      await nodeViewOfGradedPhoto(p, 2);
      await tab(p, "Mask");
      await p.getByRole("radio", { name: "Overlay" }).click();
      await expect(p.getByRole("group", { name: "Graph tools" }).getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-pressed", "true");
      await expect(p.getByRole("region", { name: "View" })).toBeVisible();
      await p.waitForTimeout(300);
    },
  ],
  // The Mask tab (docs/23 section 7.6, N-12). The Masks card with two rows, as screen 07b has them: an Object (Add, selected, so its ring shows) and a Radial that
  // is Subtract and inverted (its Invert pressed in the mask accent).
  [
    "58-nodes-masks-two-rows",
    async (p) => {
      await maskTabOfGradedPhoto(p, 2, ROWS_07B);
      await p.getByRole("button", { name: "Object mask, 2 points" }).click();
      await expect(p.getByRole("button", { name: "Object mask, 2 points" })).toHaveAttribute("aria-pressed", "true");
      await expect(p.getByRole("button", { name: "Invert this mask" }).nth(1)).toHaveAttribute("aria-pressed", "true");
      await p.waitForTimeout(300);
    },
  ],
  // The Add mask menu (screen 08): to the left of the panel, over the photo and the graph, with all three groups (the mock says this machine can make every kind).
  [
    "59-nodes-add-mask-menu",
    async (p) => {
      await maskTabOfGradedPhoto(p, 2, ROWS_07B, "all");
      await p.getByRole("button", { name: "Object mask, 2 points" }).click();
      await p.getByRole("button", { name: "Add mask" }).click();
      await expect(p.getByRole("menu", { name: "Add mask" })).toBeVisible();
      await expect(p.getByRole("menuitem")).toHaveCount(9);
      await p.waitForTimeout(400);
    },
  ],
  // A Linear's card with its handles on the photo (screen 09): Sky's Linear selected, the card with Angle, Feather and Opacity under the Masks card, the centre line,
  // the dashed start and end lines and the rotate handle over the photo, and the chip "MASK · LINEAR".
  [
    "60-nodes-linear-card-handles",
    async (p) => {
      await maskTabOfGradedPhoto(p, 3);
      await p.getByRole("button", { name: /^Linear mask/ }).click();
      await expect(p.getByRole("region", { name: "Linear" })).toBeVisible();
      await expect(p.getByRole("button", { name: /^Rotate the Linear mask/ })).toBeVisible();
      await expect(p.getByText("MASK · LINEAR")).toBeVisible();
      await p.waitForTimeout(300);
    },
  ],
  // Pick colour of a Colour range: its card with the toggle pressed, the crosshair over the photo and the hint in the header.
  [
    "61-nodes-colour-range-picking",
    async (p) => {
      await openApp(p, MOCKUP);
      await configureMock(p, {
        initialSettings: {
          DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE }, ...GRADED_NODES.nodes, "4": { ...GRADED_NODES.nodes["4"], masks: { invert: false, items: [WARMTH_RANGE] } } }, links: GRADED_NODES.links },
        } as never,
      });
      await openPhotos(p);
      await showUpdateReady(p);
      await p.getByRole("tab", { name: "Light", exact: true }).click();
      await p.getByRole("button", { name: "Nodes", exact: true }).click();
      await expect(p.locator('[data-node-card="4"]')).toBeVisible();
      await p.locator('[data-node-card="4"]').click();
      await p.getByRole("tab", { name: "Mask", exact: true }).click();
      await p.getByRole("button", { name: /^Colour range mask/ }).click();
      await p.getByRole("button", { name: "Pick colour" }).click();
      await expect(p.getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "true");
      await expect(p.getByRole("application", { name: /^Pick colour for the Colour range mask/ })).toBeVisible();
      await p.waitForTimeout(400);
    },
  ],
  // A Brush row (docs/23 section 7.6, N-14): its BRUSH card (Paint · Erase, Size, Hardness, then Refine edges, Feather, Grow, Opacity) with a stroke drawn by hand
  // on the photo, the cursor circle at the brush's size, the mask view on (the mock draws the stroke), and the chip "MASK · BRUSH".
  [
    "62-nodes-brush-card",
    async (p) => {
      await maskTabOfGradedPhoto(p, 2, [BRUSH_ROW]);
      await p.getByRole("button", { name: /^Brush mask/ }).click();
      await expect(p.getByRole("region", { name: "Brush" })).toBeVisible();
      await p.getByRole("radio", { name: "Overlay" }).click();
      const cell = (await p.locator("[data-viewport]").boundingBox())!;
      await p.mouse.move(cell.x + cell.width * 0.2, cell.y + cell.height * 0.7);
      await p.mouse.down();
      await p.mouse.move(cell.x + cell.width * 0.5, cell.y + cell.height * 0.55, { steps: 8 });
      await p.mouse.move(cell.x + cell.width * 0.8, cell.y + cell.height * 0.7, { steps: 8 });
      await p.mouse.up();
      await expect(p.getByRole("button", { name: "Brush mask, 1 stroke" })).toBeVisible();
      await expect(p.locator("[data-brush-layer] polyline")).toHaveCount(0);
      await expect(p.locator("[data-brush-cursor]")).toBeVisible();
      await p.waitForTimeout(300);
    },
  ],
  // An Object row (docs/23 section 7.6, N-18; mockup 07): its OBJECT card (Add point · Remove point · Box with its line, then Refine edges at 50, Feather, Grow, Opacity),
  // two clicks on the mountain drawn as circles with a plus, the pill "Object found · 0.4 s" from a third click, the mask view on in Overlay, and the chip "MASK · OBJECT".
  [
    "64-nodes-object-card",
    async (p) => {
      await maskTabOfGradedPhoto(p, 2, [OBJECT_ROW]);
      await p.getByRole("button", { name: /^Object mask/ }).click();
      await expect(p.getByRole("region", { name: "Object" })).toBeVisible();
      await p.getByRole("radio", { name: "Overlay" }).click();
      await expect(p.locator("[data-object-layer]")).toHaveAttribute("data-tool", "add");
      await expect(p.locator("[data-object-point]")).toHaveCount(2);
      const cell = (await p.locator("[data-viewport]").boundingBox())!;
      await p.mouse.click(cell.x + cell.width * 0.5, cell.y + cell.height * 0.68);
      await expect(p.getByText("Object found · 0.4 s")).toBeVisible();
      await expect(p.locator("[data-object-point]")).toHaveCount(3);
      await p.waitForTimeout(300);
    },
  ],
  // A Background row (docs/23 section 7.6, N-15): its BACKGROUND card (Level, Softness, then Refine edges, Feather, Grow, Opacity) with the row at "Level 50" (the photo's
  // first frame made its blur map, "Finding…" for 400 ms on the mock, before the state is taken), the mask view on in Overlay (the mock draws the sky as out of focus), and
  // the chip "MASK · BACKGROUND".
  [
    "63-nodes-background-card",
    async (p) => {
      await maskTabOfGradedPhoto(p, 2, [BACKGROUND_ROW]);
      await p.getByRole("button", { name: /^Background mask/ }).click();
      await expect(p.getByRole("region", { name: "Background" })).toBeVisible();
      await p.getByRole("radio", { name: "Overlay" }).click();
      await expect(p.getByRole("button", { name: "Background mask, Level 50" })).toBeVisible();
      await expect(p.locator(".ym-photo-cell__image")).toHaveAttribute("src", /data-background-row/);
      await p.waitForTimeout(300);
    },
  ],
  // The simple view of a photo that has four nodes (docs/23 section 6, N-07; screen 04 of design/nodes): the notice under the tab row, with
  // Open nodes at the end of its text, and the nodes glyph on the photo's thumbnail. The Light tab, as the mockup has it.
  [
    "49-simple-has-nodes",
    async (p) => {
      await photosWithNodes(p);
      await tab(p, "Light");
      await expect(p.getByText("This photo has 4 nodes.", { exact: false })).toBeVisible();
      await expect(p.locator(".ym-filmstrip__nodes")).toHaveCount(1);
    },
  ],
  // Pick white in the Colour tab (docs/23 section 8, N-06), started with W from the Look tab, with "Rendering on the CPU" under its hint:
  // the two glass pills of the photo cell, one above the other. (State 17 is the same screen without the CPU note.)
  [
    "50-colour-picking-white",
    async (p) => {
      await photos(p);
      await p.keyboard.press("w");
      await expect(p.getByText("Click something neutral: grey or white. Esc cancels.")).toBeVisible();
      await p.evaluate(() => window.__yellowmoon!.mockApi.emitEvent("viewport_busy", { busy: true, cpu: true, finding: false }));
      await expect(p.getByText("Rendering on the CPU")).toBeVisible();
      await p.waitForTimeout(300);
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
