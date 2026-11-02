import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { MockOptions } from "../src/api.mock";
import type { GradePatch } from "../src/api/types";
import { configureMock, lookTilesLoaded, openApp, openPhotos, showUpdateReady } from "./helpers/app";
import { compareScreen } from "./helpers/compare";
import { FOUR_NODES, GRADED_NODES, ROWS_07B, WARP_PINS } from "./helpers/states";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

// Screenshot comparisons of the screens (docs/11 §Tests): the app at 1440 × 900 at 2× compared with `design/screenshots-c2/*.png`,
// the references of the dark glass interface (C2). They are the app's own pictures on the mock API, taken with the browser flags of
// `playwright.config.ts` (`npx playwright test --update-snapshots` retakes them; a person looks at them before they are committed).
// `toHaveScreenshot` is the assertion (threshold 0.2, a share of pixels allowed to differ, see playwright.config.ts and the budgets in
// `helpers/compare.ts`). The share of pixels that differ by more than 24 on a channel is printed and kept in
// `node_modules/.cache/yellowmoon-e2e/screenshot-diffs/<state>.json` for each state.
//
// The mock photos are flat-shape photos graded with the mock's own grading function, so the picture is the same on every run.
// Nothing moves while the picture is taken: animations are off and the aurora is still.

const MOCKUP: Record<string, string | number> = { mockCurrent: 3, mockPicked: "2,3,4" };

/** The mockup's curves (`src/dev/demoData.ts` has the same numbers; that file needs Vite, so they are repeated here). */
const CURVE_DEMO_POINTS = {
  luma: [[0, 0], [0.25, 0.191], [0.5, 141 / 255], [0.75, 0.845], [1, 1]],
  red: [[0, 0], [0.25, 0.2], [0.5, 0.57], [0.75, 0.86], [1, 1]],
  green: [[0, 0], [0.5, 0.5], [1, 1]],
  blue: [[0, 0], [0.25, 0.1], [0.5, 0.45], [0.75, 0.82], [1, 1]],
} satisfies Record<string, Array<[number, number]>>;

/** What the mockup's 4th photo has been given: Light's Highlights and Shadows, Colour's wheels and Saturation, a curve. */
const MOCKUP_GRADE: GradePatch = {
  highlights: manual(-18),
  shadows: manual(12),
  saturation: manual(6),
  color_balance: {
    shadows: { kind: "manual", point: [-0.08, -0.12] },
    midtones: [0, 0],
    highlights: { kind: "manual", point: [0.12, 0.12] },
  },
  curve: { luma: CURVE_DEMO_POINTS.luma, red: CURVE_DEMO_POINTS.red, green: CURVE_DEMO_POINTS.green, blue: CURVE_DEMO_POINTS.blue },
};

/** The settings of the mockup's 4th photo: node 1, Main, has the grade above (settings are a graph of nodes, `docs/23` section 3). */
const gradeOfMainOn = (grade: GradePatch): MockOptions["initialSettings"] => ({ DSC04204: { nodes: { "1": { grade } } } });
const MOCKUP_SETTINGS = gradeOfMainOn(MOCKUP_GRADE);

/** Waits until the dialogs and the list have finished opening: a busy machine stretches their 300 ms, and a picture taken in the middle differs. */
async function dialogsSettled(page: Page): Promise<void> {
  await page.waitForTimeout(200);
  await page.waitForFunction(() => document.getAnimations().every((a) => a.playState !== "running"));
}

/** Waits until each of the `count` cards shows its own picture (the core's, after its node), not the photo's thumbnail it shows meanwhile. */
async function cardPicturesMade(page: Page, count: number): Promise<void> {
  await page.waitForFunction(async (n) => {
    const photo = (await window.__yellowmoon!.mockApi.thumbnails()).find((t) => t.id === 4)?.url;
    const srcs = [...document.querySelectorAll<HTMLImageElement>(".ym-gnode__thumb img")].map((img) => img.src);
    return srcs.length === n && srcs.every((src) => src !== photo);
  }, count);
}

test.describe("Start", () => {
  test("idle (start-idle.png)", async ({ page }) => {
    await openApp(page, { mockLooks: 0 });
    await showUpdateReady(page);
    await compareScreen(page, "start-idle");
  });

  test("developing, 5 of 12 ready (start-busy.png)", async ({ page }) => {
    await openApp(page, { mockHold: 5.4, mockLooks: 0 });
    await showUpdateReady(page);
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("list", { name: "Photos being developed" }).getByText("Ready", { exact: true })).toHaveCount(5);
    await compareScreen(page, "start-busy");
  });

  test("all 12 ready (start-done.png)", async ({ page }) => {
    await openApp(page, { mockLooks: 0 });
    await showUpdateReady(page);
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    await compareScreen(page, "start-done");
  });
});

test.describe("Photos", () => {
  async function photos(page: Page, settings: MockOptions["initialSettings"] = MOCKUP_SETTINGS): Promise<void> {
    await openApp(page, MOCKUP);
    await configureMock(page, { initialSettings: settings });
    await openPhotos(page);
    await showUpdateReady(page);
    await expect(page.getByText("Neural · 42 ms · Display P3")).toBeVisible();
    // The look tiles ask for their thumbnails as they scroll into view; wait for every one that shows.
    await lookTilesLoaded(page);
    await expect(page.locator(".ym-filmstrip__thumb img")).toHaveCount(12);
  }

  const tab = (page: Page, name: string) => page.getByRole("tab", { name }).click();

  test("Look (photos-look.png)", async ({ page }) => {
    await photos(page);
    await compareScreen(page, "photos-look");
  });

  test("Lens: Default chosen, the amounts at 0 (photos-lens.png)", async ({ page }) => {
    // The owner's request of 2026-10-25 (docs/11 *Lens card*): the second tab, the seven lenses as rows, the Amounts card below them.
    await photos(page);
    await tab(page, "Lens");
    await expect(page.getByRole("radiogroup", { name: "Lens" }).getByRole("radio")).toHaveCount(7);
    await compareScreen(page, "photos-lens");
  });

  test("Light (photos-light.png)", async ({ page }) => {
    await photos(page);
    await tab(page, "Light");
    await expect(page.locator(".ym-histogram path").first()).toHaveAttribute("d", /L/);
    await compareScreen(page, "photos-light");
  });

  test("A photo that has four nodes: the notice under the tab row, the glyph on its thumbnail (photos-has-nodes.png)", async ({ page }) => {
    // docs/23 section 6, screen 04 of design/nodes: node 1 keeps the mockup's grade, and Mountain, Sky and Warmth follow it. The Light tab, as the
    // mockup has it; the simple view shows the photo with all four nodes applied.
    await photos(page, {
      DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE }, ...FOUR_NODES.nodes }, links: FOUR_NODES.links },
    } as never);
    await tab(page, "Light");
    await expect(page.locator(".ym-histogram path").first()).toHaveAttribute("d", /L/);
    await expect(page.getByText("This photo has 4 nodes.", { exact: false })).toBeVisible();
    await expect(page.locator(".ym-filmstrip__nodes")).toHaveCount(1);
    await compareScreen(page, "photos-has-nodes");
  });

  test("The node view: Main between the Input and the Output, Main selected (photos-nodes.png)", async ({ page }) => {
    // docs/23 section 7.2 and 7.5, screen 05 of design/nodes: the default graph, with Main's head and its Light tab (the node view opens on the node
    // and the tab of the simple view's tab). The card shows the picture after Main (N-05), not the photo's thumbnail.
    await photos(page);
    await tab(page, "Light");
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(page.locator('[data-node-card="1"]')).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("slider", { name: "Exposure trim" })).toBeVisible();
    await expect(page.locator(".ym-histogram path").first()).toHaveAttribute("d", /L/);
    await expect(page.locator(".ym-gnode__thumb img")).toHaveCount(1);
    await cardPicturesMade(page, 1);
    await expect(page.getByText("Add a node to change only part of the photo.")).toBeVisible();
    await compareScreen(page, "photos-nodes");
  });

  test("The node view: the graded graph with the Add node menu open (photos-nodes-add-node.png)", async ({ page }) => {
    // Screen 06: Main, Mountain (selected, with an Object and a Radial mask), Sky (a Linear mask) and Warmth, and the menu opened upward from
    // the button. Each card shows the picture after its node (N-05).
    await photos(page, { DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE }, ...GRADED_NODES.nodes }, links: GRADED_NODES.links } } as never);
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await page.locator('[data-node-card="2"]').click();
    await expect(page.locator('[data-node-card="2"]')).toHaveAttribute("aria-pressed", "true");
    // The panel is Mountain's: its head and its Light tab, with the Exposure trim of +0.35 and Highlights of −12.
    await expect(page.getByRole("slider", { name: "Exposure trim" })).toBeVisible();
    await expect(page.locator(".ym-histogram path").first()).toHaveAttribute("d", /L/);
    await cardPicturesMade(page, 4);
    await page.getByRole("button", { name: "Add node" }).click();
    await expect(page.getByRole("menu", { name: "Add node" })).toBeVisible();
    await dialogsSettled(page);
    await compareScreen(page, "photos-nodes-add-node");
  });

  /** Main with the mockup's grade and a Radial at the middle of the photo, the graded graph after it; the Mask tab of Main, the view on in `style`. */
  async function maskView(page: Page, style: "Overlay" | "Mask only"): Promise<void> {
    const radial = { kind: "radial", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.52], radii: [0.26, 0.17], angle_deg: 0, feather: 0.45 };
    await photos(page, {
      DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE, masks: { invert: false, items: [radial] } }, ...GRADED_NODES.nodes }, links: GRADED_NODES.links },
    } as never);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await page.locator('[data-node-card="1"]').click();
    await expect(page.locator('[data-node-card="1"]')).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await page.getByRole("radio", { name: style, exact: true }).click();
    await expect(page.getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator(".ym-photo-cell__image")).toHaveAttribute("src", /data-mask-view/);
    await cardPicturesMade(page, 4);
  }

  test("The node view: the Mask tab with the View card, the mask tinted over the photo, Show mask on (photos-nodes-mask-overlay.png)", async ({ page }) => {
    // docs/23 section 5.8 and 7.6, screens 07 and 09 of design/nodes: Main's Radial in the mask accent over the photo, the toolbar's Show mask yellow,
    // the View card at the bottom of the Mask tab with Overlay chosen. The core draws the tint (`ym-core` `mask_view.rs`); the mock draws it on its flat scene.
    await maskView(page, "Overlay");
    await compareScreen(page, "photos-nodes-mask-overlay");
  });

  test("The node view: the same, Mask only: the mask as grey (photos-nodes-mask-only.png)", async ({ page }) => {
    // Mask only: white where m is 1, black where it is 0, the paper around the picture as it is.
    await maskView(page, "Mask only");
    await compareScreen(page, "photos-nodes-mask-only");
  });

  test("The node view: Main's Brush row with two strokes, one painted and one erased, its BRUSH card and the cursor circle on the photo (photos-nodes-mask-brush.png)", async ({ page }) => {
    // docs/23 section 7.6 (there is no mockup for Brush): the Masks card with the Brush row selected ("Brush", "2 strokes"), the BRUSH card under it (Paint · Erase, Size,
    // Hardness, then Refine edges at 0, Feather, Grow, Opacity), the View card with Overlay, the stroke painted over the sky and the hill in the mask accent with the
    // erase stroke cutting across it (the mock draws the strokes on its flat scene; the core draws the real mask), the cursor circle at the brush's size where the pointer
    // rests, and the chip "MASK · BRUSH".
    const brush = {
      kind: "brush",
      mode: "add",
      invert: false,
      enabled: true,
      opacity: 1,
      feather: 0,
      grow: 0,
      refine: 0,
      strokes: [
        { size: 0.09, hardness: 0.35, erase: false, points: [[0.16, 0.66], [0.3, 0.5], [0.5, 0.4], [0.7, 0.45], [0.86, 0.6]] },
        { size: 0.05, hardness: 0.8, erase: true, points: [[0.5, 0.26], [0.52, 0.5], [0.5, 0.78]] },
      ],
    };
    await photos(page, {
      DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE, masks: { invert: false, items: [brush] } }, ...GRADED_NODES.nodes }, links: GRADED_NODES.links },
    } as never);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await page.locator('[data-node-card="1"]').click();
    await expect(page.locator('[data-node-card="1"]')).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await page.getByRole("button", { name: "Brush mask, 2 strokes" }).click();
    await expect(page.getByRole("region", { name: "Brush" })).toBeVisible();
    await page.getByRole("radio", { name: "Overlay", exact: true }).click();
    await expect(page.getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator(".ym-photo-cell__image")).toHaveAttribute("src", /data-brush-row/);
    await expect(page.locator("[data-brush-layer]")).toBeVisible();
    await cardPicturesMade(page, 4);
    // The pointer rests near the end of the painted stroke, inside the photo, so that the circle shows.
    const cell = (await page.locator("[data-viewport]").boundingBox())!;
    await page.mouse.move(cell.x + cell.width * 0.5, cell.y + cell.height * 0.4);
    await expect(page.locator("[data-brush-cursor]")).toBeVisible();
    await compareScreen(page, "photos-nodes-mask-brush", { pointer: { x: cell.x + cell.width * 0.5, y: cell.y + cell.height * 0.4 } });
  });

  /**
   * The graded graph (Main with the mockup's grade, then Mountain, Sky and Warmth) on the Mask tab of node `select`, with `rows` in Mountain's masks when given
   * (screen 07b's two rows) and `kinds` as the mock's `?mockMasks=` (every kind, for the menu of screen 08).
   */
  async function maskTab(page: Page, select: number, rows?: readonly object[], kinds?: string): Promise<void> {
    await openApp(page, kinds ? { ...MOCKUP, mockMasks: kinds } : MOCKUP);
    const mountain = { ...GRADED_NODES.nodes["2"], ...(rows ? { masks: { invert: false, items: rows } } : {}) };
    await configureMock(page, {
      initialSettings: { DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE }, ...GRADED_NODES.nodes, "2": mountain }, links: GRADED_NODES.links } } as never,
    });
    await openPhotos(page);
    await showUpdateReady(page);
    await expect(page.getByText("Neural · 42 ms · Display P3")).toBeVisible();
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await page.locator(`[data-node-card="${select}"]`).click();
    await expect(page.locator(`[data-node-card="${select}"]`)).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(page.getByRole("region", { name: "Masks" })).toBeVisible();
    await cardPicturesMade(page, 4);
  }

  test("The node view: Mountain's Mask tab with two rows, one Subtract and one inverted, the Object row selected (photos-nodes-mask-rows.png)", async ({ page }) => {
    // docs/23 section 7.6, screens 07 and 07b of design/nodes: the Masks card with Add mask, Invert and Reset, the Object row (two points, Add) with the
    // 2 px ring of the selected row, and the Radial row (Feather 45) that is Subtract and inverted, its Invert pressed in the mask accent; the line under them,
    // and the View card. The chip says "MASK · OBJECT". An Object row has no card of its own yet (N-18).
    await maskTab(page, 2, ROWS_07B);
    await page.getByRole("button", { name: "Object mask, 2 points" }).click();
    await expect(page.getByRole("button", { name: "Object mask, 2 points" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByText("MASK · OBJECT")).toBeVisible();
    await compareScreen(page, "photos-nodes-mask-rows");
  });

  test("The node view: the Add mask menu open to the left of the panel, with every kind (photos-nodes-add-mask-menu.png)", async ({ page }) => {
    // Screen 08: the menu is 520 px tall at most and opens to the left of the panel over the photo and the graph, in three groups (Find, Shapes, Keys) with a line
    // under each name. The mock says this machine can make every kind (`?mockMasks=all`), as the mockup's does.
    await maskTab(page, 2, ROWS_07B, "all");
    await page.getByRole("button", { name: "Object mask, 2 points" }).click();
    await page.getByRole("button", { name: "Add mask" }).click();
    await expect(page.getByRole("menu", { name: "Add mask" })).toBeVisible();
    await expect(page.getByRole("menuitem")).toHaveCount(9);
    await dialogsSettled(page);
    await compareScreen(page, "photos-nodes-add-mask-menu");
  });

  test("The node view: Sky's Linear mask, its card on the Mask tab and its handles on the photo (photos-nodes-mask-linear.png)", async ({ page }) => {
    // Screen 09 of design/nodes on the Mask tab: the Linear row selected, its card (Angle, Feather, Opacity) under the Masks card, the solid centre line, the
    // dashed start and end lines and the rotate handle over the photo, and the chip "MASK · LINEAR".
    await maskTab(page, 3);
    await page.getByRole("button", { name: /^Linear mask/ }).click();
    await expect(page.getByRole("region", { name: "Linear" })).toBeVisible();
    await expect(page.getByRole("button", { name: /^Rotate the Linear mask/ })).toBeVisible();
    await expect(page.getByText("MASK · LINEAR")).toBeVisible();
    await compareScreen(page, "photos-nodes-mask-linear");
  });

  test("The node view: the Input selected, its Colour tab with the White balance card (photos-nodes-input-colour.png)", async ({ page }) => {
    // docs/23 section 7.5 and 8, screen 10 of design/nodes: the graded graph, the Input's head (a lock, "Turns the camera's data into the scene.")
    // and the tab that holds the White balance card and Pick white.
    await photos(page, { DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE }, ...GRADED_NODES.nodes }, links: GRADED_NODES.links } } as never);
    await tab(page, "Colour");
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    // The simple view stood on Colour, which is Main's tab: the Input is chosen on the graph, and its Colour tab is the one to show.
    await page.locator('[data-node-card="input"]').click();
    await expect(page.locator('[data-node-card="input"]')).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("tab", { name: "Colour", exact: true }).click();
    await expect(page.getByRole("button", { name: "Pick white" })).toBeVisible();
    await cardPicturesMade(page, 4);
    await compareScreen(page, "photos-nodes-input-colour");
  });

  test("The node view: the Output selected, its Look tab with the Look card (photos-nodes-output-look.png)", async ({ page }) => {
    // docs/23 section 7.5, screen 11 of design/nodes: the Output's head and today's Look card. The simple view stood on Look, which is the Output's tab.
    await photos(page, { DSC04204: { nodes: { "1": { grade: MOCKUP_GRADE }, ...GRADED_NODES.nodes }, links: GRADED_NODES.links } } as never);
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(page.locator('[data-node-card="output"]')).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("slider", { name: "Intensity" })).toBeVisible();
    await lookTilesLoaded(page);
    await cardPicturesMade(page, 4);
    await compareScreen(page, "photos-nodes-output-look");
  });

  test("Colour (photos-color.png)", async ({ page }) => {
    await photos(page);
    await tab(page, "Colour");
    await compareScreen(page, "photos-color");
  });

  test("Colour while picking white: the card's toggle pressed, the hint pill on the photo, the crosshair (photos-pick-white.png)", async ({ page }) => {
    // docs/23 section 8, screen 03 of design/nodes: Pick white is a row of the White balance card; while it is on, the toggle is yellow, the
    // hint is a glass pill at the top centre of the photo and the crosshair (white, with a dark outline) is under the pointer, which rests
    // where the mockup's crosshair is.
    await photos(page);
    await tab(page, "Colour");
    await page.getByRole("button", { name: "Pick white" }).click();
    await expect(page.getByRole("status").filter({ hasText: "Click something neutral: grey or white. Esc cancels." })).toBeVisible();
    await compareScreen(page, "photos-pick-white", { pointer: { x: 784, y: 227 } });
  });

  test("Colour warp: two moving pins and a lock, the card in view, its first pin selected (photos-color-warp.png)", async ({ page }) => {
    // The owner's request of 2026-10-17 (docs/11 §Colour warp): the sky's blue moved toward teal, the sun's warm white toward orange, a
    // lock in the magentas. The photo shows the move (the mock warps its flat colours, src/fixtures/warp.ts).
    await photos(page, gradeOfMainOn({ ...MOCKUP_GRADE, color_warp: { pins: WARP_PINS as never } }));
    await tab(page, "Colour");
    // The photo's colours have been drawn behind the plane.
    await expect(page.locator('.ym-warp-plane:not([data-scatter-points="0"])')).toBeVisible();
    await page.getByRole("slider", { name: "Pin 1" }).focus();
    await page.locator(".ym-tools__scroll").evaluate((el) => {
      el.scrollTop = el.scrollHeight;
    });
    await expect(page.getByRole("textbox", { name: "Range value" })).toBeInViewport();
    await compareScreen(page, "photos-color-warp");
  });

  test("Detail (photos-detail.png)", async ({ page }) => {
    await photos(page);
    await tab(page, "Detail");
    await compareScreen(page, "photos-detail");
  });

  test("Curve (photos-curve.png)", async ({ page }) => {
    await photos(page);
    await tab(page, "Curve");
    await expect(page.getByRole("group", { name: "Luma curve" }).getByRole("slider")).toHaveCount(5);
    await compareScreen(page, "photos-curve");
  });

  test("RAW (photos-raw.png)", async ({ page }) => {
    await photos(page);
    await tab(page, "RAW");
    await compareScreen(page, "photos-raw");
  });

  test("Before (photos-before.png)", async ({ page }) => {
    await photos(page);
    await page.getByRole("button", { name: "Before" }).click();
    await expect(page.getByText("Before · camera default")).toBeVisible();
    await compareScreen(page, "photos-before");
  });

  test("Look: Teal and orange picked, panel scrolled to the library (photos-look-teal.png)", async ({ page }) => {
    await photos(page);
    const thumbs = () => page.locator(".ym-filmstrip__thumb img").evaluateAll((imgs) => imgs.map((img) => (img as HTMLImageElement).src));
    const before = await thumbs();
    await page.getByRole("button", { name: "Teal and orange look, key 6" }).click();
    await expect(page.locator(".ym-photo-cell__chip")).toHaveText("Teal and orange · 100%");
    // Scrolled to 357 px, 3 px short of the very bottom of the panel's travel (the state the pre-glass screenshot was taken in).
    await page.locator(".ym-tools__scroll").evaluate((el) => {
      el.scrollTop = 357;
    });
    await expect.poll(async () => page.locator(".ym-look-tile img").count()).toBe(9);
    await lookTilesLoaded(page);
    // Apply to starts on This photo (the owner's decision of 2026-10-20): only the photo on screen has a new thumbnail.
    await expect.poll(async () => {
      const now = await thumbs();
      return now.length === before.length && now.filter((src, i) => src !== before[i]).length === 1;
    }).toBe(true);
    await compareScreen(page, "photos-look-teal");
  });

  test("Look with Apple Auto's answer: the Automatic card's rows for the values it set, its line and plan (photos-look-apple-auto.png)", async ({ page }) => {
    // docs/19 section 10, not in the mockup: the state of a Mac where Apple Auto is available and has answered (the mock's answer is a
    // little brighter and warmer, darker highlights, deeper blacks, more contrast, less saturation, teal shadows and warm highlights).
    // The photo keeps only the mockup's curve, so every row shows Apple's value as Auto. Without Apple Auto the Look tab is
    // photos-look.png, unchanged.
    await openApp(page, { ...MOCKUP, mockAppleAuto: "answered" });
    await configureMock(page, { initialSettings: gradeOfMainOn({ curve: MOCKUP_GRADE.curve }) });
    await openPhotos(page);
    await showUpdateReady(page);
    await expect(page.getByText("Neural · 42 ms · Display P3")).toBeVisible();
    await expect(page.getByText("Set with Apple Intelligence.")).toBeVisible();
    // The card is six rows and two lines taller, so the look tiles start below the fold: wait only for the ones that show.
    await page.waitForFunction(() => {
      const box = document.querySelector(".ym-tools__scroll")?.getBoundingClientRect();
      if (!box) return false;
      return [...document.querySelectorAll<HTMLElement>(".ym-look-tile")]
        .filter((tile) => tile.getBoundingClientRect().top < box.bottom && tile.getBoundingClientRect().bottom > box.top)
        .every((tile) => {
          const img = tile.querySelector("img");
          return !!img && img.complete && img.naturalWidth > 0;
        });
    });
    await expect(page.locator(".ym-filmstrip__thumb img")).toHaveCount(12);
    await compareScreen(page, "photos-look-apple-auto");
  });

  test("Preferences with every provider listed and the list open (photos-preferences-providers.png)", async ({ page }) => {
    // docs/22 section 12.9, not in the mockup: a computer with Apple Intelligence and Claude Code available and Antigravity installed but
    // signed out, Claude Code the saved choice, the Auto grading list open: the disabled entry with its Experimental badge and its tag,
    // the note under the row and the instructions for Claude Code.
    await openApp(page, { ...MOCKUP, mockProviders: "apple,claude,agy:signed_out", mockAutoProvider: "claude", mockConsent: "claude", mockAppleAuto: "answered" });
    await openPhotos(page);
    await showUpdateReady(page);
    await page.keyboard.press("Meta+,");
    // Wait for the dialog to finish opening before the click: a click on a dialog that is still scaling up can scroll its body.
    await expect(page.getByRole("combobox", { name: "Auto grading" })).toBeVisible();
    await dialogsSettled(page);
    await page.getByRole("combobox", { name: "Auto grading" }).click();
    await expect(page.getByRole("option", { name: "Antigravity (agy), experimental, not signed in" })).toBeVisible();
    await dialogsSettled(page);
    await compareScreen(page, "photos-preferences-providers");
  });

  test("the consent dialog over Preferences (photos-preferences-consent.png)", async ({ page }) => {
    // docs/22 section 12.3: Claude Code chosen for the first time, the question over Preferences with Cancel in focus.
    await openApp(page, { ...MOCKUP, mockProviders: "apple,claude,agy:signed_out", mockAppleAuto: "answered" });
    await openPhotos(page);
    await showUpdateReady(page);
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("combobox", { name: "Auto grading" })).toBeVisible();
    await dialogsSettled(page);
    await page.getByRole("combobox", { name: "Auto grading" }).click();
    await page.getByRole("option", { name: "Claude Code" }).click();
    await expect(page.getByRole("dialog", { name: "Send photos to Claude Code?" })).toBeVisible();
    await dialogsSettled(page);
    await compareScreen(page, "photos-preferences-consent");
  });

  test("Look: search \"film\" (photos-look-search.png)", async ({ page }) => {
    await photos(page);
    await page.getByRole("searchbox", { name: "Search looks" }).fill("film");
    await expect(page.getByText("2 of 7")).toBeVisible();
    await expect.poll(async () => page.locator(".ym-look-tile img").count()).toBe(2);
    await lookTilesLoaded(page);
    await compareScreen(page, "photos-look-search");
  });
});
