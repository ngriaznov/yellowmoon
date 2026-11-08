import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, recordCalls } from "./helpers/app";
import { GRADED_NODES } from "./helpers/states";

// The Depth mask's interface (docs/23 §4.8 and §7.6, docs/11 *The Depth card*; task D-05), in a real browser on the mock API: the Add mask entry, a new
// row ("Finding…", Pick depth on), Pick depth, Focus plane and its reasons, Show depth on and off, a map that failed and Try again, and the weights missing.
// The core makes the real map (`crates/ym-core/tests/mask_depth.rs`); the mock gives fixed depth positions to the flat scenes' areas. The pictures of the
// built card and of Show depth go to $YM_DEPTH_SHOTS when it is set (the lead compares them with `design/depth/`).

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;
const SHOTS = process.env.YM_DEPTH_SHOTS;

type Row = Record<string, unknown> & { kind: string };
const depthRow = (extra: Record<string, unknown> = {}) => ({ kind: "depth", mode: "add", invert: false, enabled: true, opacity: 1, far: 1, feather: 0, grow: 0, refine: 0.5, ...extra });

const card = (page: Page, id: number | string) => page.locator(`[data-node-card="${id}"]`);
const rowList = (page: Page) => page.getByRole("list", { name: "Masks of this node" });
const rowButton = (page: Page, name: RegExp | string) => rowList(page).getByRole("button", { name });
const depthCard = (page: Page) => page.getByRole("region", { name: "Depth", exact: true });
const maskView = (page: Page) => page.evaluate(() => window.__yellowmoon!.mockApi.viewport.maskView as { node: number; style: string; row?: number | null; depth?: boolean } | null);
const picture = async (page: Page): Promise<string> => decodeURIComponent(((await page.locator(".ym-photo-cell__image").getAttribute("src")) ?? "").replace(/^data:image\/svg\+xml;charset=utf-8,/, ""));
const cellBox = async (page: Page) => (await page.locator("[data-viewport]").boundingBox())!;

/** The rows of node `node` of the photo on show, as the mock has them. */
const rowsOf = (page: Page, node: number): Promise<Row[]> =>
  page.evaluate(async (id) => {
    const { mockApi } = window.__yellowmoon!;
    const state = await mockApi.sessionState();
    const rs = await mockApi.getSettings(state.photos[state.current_index!]!.id);
    const item = rs.settings.nodes.items.find((n) => n.id === id);
    return item && "masks" in item ? (item.masks.items as unknown as Row[]) : [];
  }, node);

/**
 * The mockup's graph on the photo with the index `current` (3 is DSC04204, which has a focus point; 0 is DSC04201, which has none; 11 is DSC04212, whose point is
 * the exact centre), Haze (node 4) selected on its Mask tab, with these rows.
 */
async function haze(page: Page, rows: Row[] = [], options: Record<string, string | number> = {}): Promise<void> {
  const current = Number(options.mockCurrent ?? MOCKUP.mockCurrent);
  await openApp(page, { ...MOCKUP, ...options });
  await configureMock(page, {
    initialSettings: {
      [`DSC042${String(current + 1).padStart(2, "0")}`]: {
        nodes: { ...GRADED_NODES.nodes, "4": { kind: "corrector", name: "Haze", pos: [3, 0], masks: { invert: false, items: rows } } },
        links: GRADED_NODES.links,
      },
    } as never,
  });
  await openPhotos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await expect(card(page, 4)).toBeVisible();
  await card(page, 4).click();
  await expect(card(page, 4)).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("tab", { name: "Mask", exact: true }).click();
  await expect(page.getByRole("region", { name: "Masks" })).toBeVisible();
}

async function addDepth(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Add mask" }).click();
  await page.getByRole("menuitem", { name: /^Depth/ }).click();
}

async function shot(page: Page, name: string): Promise<void> {
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png` });
}

test.describe("adding a Depth row", () => {
  test("is in the Add mask menu after Background, with its line; the new row says Finding… while the map is made, with Pick depth on and the pill", async ({ page }) => {
    await haze(page, [], { mockSpeed: 1 });
    await page.getByRole("button", { name: "Add mask" }).click();
    const menu = page.getByRole("menu", { name: "Add mask" });
    await expect(menu.getByRole("menuitem")).toHaveCount(10);
    await expect(menu.getByRole("menuitem").nth(4)).toContainText("Depth");
    await expect(menu.getByText("Near or far, from depth")).toBeVisible();
    // The menu grows to hold its tenth item.
    const box = (await menu.boundingBox())!;
    expect(box.height).toBeGreaterThan(520);
    expect(box.height).toBeLessThanOrEqual(600);
    await shot(page, "01-add-mask-menu");
    await menu.getByRole("menuitem", { name: /^Depth/ }).click();
    await expect(rowButton(page, "Depth mask, Finding…")).toBeVisible();
    await expect(rowList(page).getByText("Finding…")).toHaveAttribute("data-finding", "");
    await expect(depthCard(page)).toBeVisible();
    await expect(depthCard(page).getByRole("button", { name: "Pick depth" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByText("Getting ready…")).toBeVisible();
    await shot(page, "02-finding");
    // The map is made: the row has its words, and the pill says what to do.
    await expect(rowButton(page, "Depth mask, Whole depth")).toBeVisible();
    await expect(page.getByText("Click the part of the photo to select. Esc stops.")).toBeVisible();
    expect(await rowsOf(page, 4)).toMatchObject([{ kind: "depth", far: 1, feather: 0, grow: 0, refine: 0.5 }]);
  });

  test("the card has Pick depth, Focus plane, Show depth, the line that says where the mask ends and the raster sliders, and no slider for the far limit", async ({ page }) => {
    await haze(page, [depthRow({ far: 0.35 })]);
    await rowButton(page, /^Depth mask/).click();
    const c = depthCard(page);
    await expect(c.getByRole("button", { name: "Pick depth" })).toHaveAttribute("aria-pressed", "false");
    await expect(c.getByRole("button", { name: "Focus plane" })).toBeVisible();
    await expect(c.getByRole("slider")).toHaveCount(4);
    for (const gone of ["Near", "Far", "Softness"]) await expect(c.getByRole("textbox", { name: `${gone} value` })).toHaveCount(0);
    await expect(c.getByRole("textbox", { name: "Refine edges value" })).toHaveValue("50%");
    await expect(c.getByText("Fades out at 35 % of the depth")).toBeVisible();
    await expect(rowButton(page, "Depth mask, To 35 %")).toBeVisible();
    await shot(page, "03-depth-card");
    // Reset puts the limit back to the whole depth, and the line and the row say so.
    await c.getByRole("button", { name: "Reset" }).click();
    await expect.poll(async () => (await rowsOf(page, 4))[0]).toMatchObject({ far: 1 });
    await expect(c.getByText("Fades out at the farthest point")).toBeVisible();
    await expect(rowButton(page, "Depth mask, Whole depth")).toBeVisible();
  });
});

test.describe("Pick depth", () => {
  test("on for a new row: the first click on the photo sets the far limit a little behind the depth there, as one patch; the tool stays on; Esc stops it", async ({ page }) => {
    await haze(page);
    await addDepth(page);
    await expect(rowButton(page, "Depth mask, Whole depth")).toBeVisible();
    await expect(page.getByRole("application", { name: /^Pick depth/ })).toBeVisible();
    await expect(page.getByText("Click the part of the photo to select. Esc stops.")).toBeVisible();
    await shot(page, "04-pick-depth");
    const calls = await recordCalls(page, "updateSettings");
    const cell = await cellBox(page);
    // The near ground, the bottom of the photo.
    await page.mouse.click(cell.x + cell.width * 0.5, cell.y + cell.height * 0.9);
    await expect.poll(async () => ((await rowsOf(page, 4))[0]?.far as number | undefined) ?? 1).toBeCloseTo(0.25, 6);
    expect(await calls()).toHaveLength(1);
    await expect(rowButton(page, "Depth mask, To 25 %")).toBeVisible();
    await shot(page, "04b-pick-depth-click");
    // The tool stays on: the sky, at the top, is the far end, so the limit is the whole depth.
    await page.mouse.click(cell.x + cell.width * 0.2, cell.y + cell.height * 0.05);
    await expect.poll(async () => (await rowsOf(page, 4))[0]).toMatchObject({ far: 1 });
    await expect(rowButton(page, "Depth mask, Whole depth")).toBeVisible();
    await expect(page.getByRole("application", { name: /^Pick depth/ })).toBeVisible();
    // Esc ends it; the toggle is off and a click on the photo picks nothing.
    await page.keyboard.press("Escape");
    await expect(page.getByRole("application", { name: /^Pick depth/ })).toHaveCount(0);
    await expect(depthCard(page).getByRole("button", { name: "Pick depth" })).toHaveAttribute("aria-pressed", "false");
    // The toggle brings it back, and another tab takes it away.
    await depthCard(page).getByRole("button", { name: "Pick depth" }).click();
    await expect(page.getByRole("application", { name: /^Pick depth/ })).toBeVisible();
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await expect(page.getByRole("application", { name: /^Pick depth/ })).toHaveCount(0);
  });
});

test.describe("Focus plane", () => {
  test("sets the far limit a little behind the plane the camera focused on, as one patch", async ({ page }) => {
    await haze(page, [depthRow({ far: 0.9 })]);
    await rowButton(page, /^Depth mask/).click();
    const calls = await recordCalls(page, "updateSettings");
    await depthCard(page).getByRole("button", { name: "Focus plane" }).click();
    await expect.poll(async () => ((await rowsOf(page, 4))[0]?.far as number | undefined) ?? 1).toBeCloseTo(0.25, 6);
    expect(await calls()).toHaveLength(1);
    await expect(depthCard(page).getByText("The camera may not have recorded where it focused.")).toHaveCount(0);
  });

  test("is off, with the reason in the card, for a photo that has no focus point", async ({ page }) => {
    await haze(page, [depthRow()], { mockCurrent: 0 });
    await rowButton(page, /^Depth mask/).click();
    const button = depthCard(page).getByRole("button", { name: "Focus plane" });
    await expect(button).toHaveAttribute("aria-disabled", "true");
    // The card also has the line that says where the mask ends; the reason is the other note.
    await expect(depthCard(page).locator("p.ym-row-note:not(.ym-depth-fade)")).toHaveText("This photo has no focus point.");
    await shot(page, "05-no-focus-point");
    const calls = await recordCalls(page, "updateSettings");
    await button.click({ force: true });
    expect(await calls()).toHaveLength(0);
  });

  test("says to check the result when the camera's point is the exact centre of the frame", async ({ page }) => {
    await haze(page, [depthRow()], { mockCurrent: 11 });
    await rowButton(page, /^Depth mask/).click();
    await depthCard(page).getByRole("button", { name: "Focus plane" }).click();
    await expect(depthCard(page).locator("p.ym-row-note:not(.ym-depth-fade)")).toHaveText("The camera may not have recorded where it focused. Check the result.");
    await shot(page, "05b-focus-centre");
    await expect.poll(async () => ((await rowsOf(page, 4))[0]?.far as number | undefined) ?? 1).toBeCloseTo(0.25, 6);
  });
});

test.describe("Show depth", () => {
  test("draws the depth map as grey, white for near, with the chip DEPTH; it is a view (no patch), exclusive with Show mask and the eyes, and Esc ends it", async ({ page }) => {
    await haze(page, [depthRow()]);
    await rowButton(page, /^Depth mask/).click();
    const show = depthCard(page).getByRole("button", { name: "Show depth" });
    const calls = await recordCalls(page, "updateSettings");
    await expect(show).toHaveAttribute("aria-pressed", "false");
    await show.click();
    await expect(show).toHaveAttribute("aria-pressed", "true");
    expect(await maskView(page)).toEqual({ node: 4, style: "mask_only", row: 0, depth: true });
    await expect.poll(() => picture(page)).toContain('data-mask-view="depth"');
    expect(await picture(page)).not.toContain("#ff6fd2");
    await expect(page.getByText("DEPTH", { exact: true }).first()).toBeVisible();
    await shot(page, "06-show-depth");
    expect(await calls()).toHaveLength(0);
    // Show mask turns it off, the row's eye turns it off, and it turns them off.
    await page.getByRole("group", { name: "Graph tools" }).getByRole("button", { name: "Show mask" }).click();
    await expect(show).toHaveAttribute("aria-pressed", "false");
    expect(await maskView(page)).toEqual({ node: 4, style: "overlay" });
    await show.click();
    await expect(page.getByRole("group", { name: "Graph tools" }).getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-pressed", "false");
    await rowList(page).getByRole("button", { name: "Show this mask" }).click();
    await expect(show).toHaveAttribute("aria-pressed", "false");
    expect(await maskView(page)).toMatchObject({ node: 4, row: 0 });
    expect((await maskView(page))?.depth).toBeFalsy();
    await expect.poll(() => picture(page)).not.toContain('data-mask-view="depth"');
    // Show depth again, then Esc.
    await show.click();
    await expect(show).toHaveAttribute("aria-pressed", "true");
    await page.keyboard.press("Escape");
    await expect(show).toHaveAttribute("aria-pressed", "false");
    expect(await maskView(page)).toBeNull();
  });

  test("goes when the card does: another row or another tab ends it", async ({ page }) => {
    await haze(page, [depthRow()]);
    await rowButton(page, /^Depth mask/).click();
    await depthCard(page).getByRole("button", { name: "Show depth" }).click();
    expect((await maskView(page))?.depth).toBe(true);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await expect.poll(() => maskView(page)).toBeNull();
  });
});

test.describe("a map that cannot be made", () => {
  test("the row says Can't be made, the card says why with Try again, and Try again makes the map", async ({ page }) => {
    await haze(page, [depthRow()], { mockDepth: "fail" });
    await expect(rowButton(page, "Depth mask, Can't be made")).toBeVisible();
    await rowButton(page, /^Depth mask/).click();
    const c = depthCard(page);
    await expect(c.locator("p.ym-row-note")).toHaveText("The Depth mask can't be made: the model stopped before it finished. Try again, or remove the mask.");
    await expect(c.getByRole("slider")).toHaveCount(0);
    await shot(page, "07b-cant-be-made");
    const retry = await recordCalls(page, "retryDepth");
    await c.getByRole("button", { name: "Try again" }).click();
    await expect.poll(async () => (await retry()).length).toBe(1);
    await expect(rowButton(page, "Depth mask, Whole depth")).toBeVisible();
    await expect(depthCard(page).getByRole("slider")).toHaveCount(4);
    // The row was not changed by any of it.
    expect(await rowsOf(page, 4)).toMatchObject([{ kind: "depth", far: 1 }]);
  });

  test("a model that fails again brings the sentence back", async ({ page }) => {
    await haze(page, [depthRow()], { mockDepth: "fail-always" });
    await rowButton(page, /^Depth mask/).click();
    await depthCard(page).getByRole("button", { name: "Try again" }).click();
    await expect(depthCard(page).getByRole("button", { name: "Try again" })).toBeVisible();
    await expect(rowButton(page, "Depth mask, Can't be made")).toBeVisible();
  });
});

test.describe("where the depth weights are missing", () => {
  test("the menu has no Depth; a row that a file brought says so, selects nothing and has no card", async ({ page }) => {
    await haze(page, [depthRow()], { mockMasks: "nodepth" });
    await page.getByRole("button", { name: "Add mask" }).click();
    const menu = page.getByRole("menu", { name: "Add mask" });
    await expect(menu.getByRole("menuitem")).toHaveCount(9);
    await expect(menu.getByRole("menuitem", { name: /^Depth/ })).toHaveCount(0);
    await page.keyboard.press("Escape");
    const name = "Depth mask, Depth masks need the depth model. Reinstall the app.";
    await expect(rowButton(page, name)).toBeVisible();
    await shot(page, "07-no-model");
    await rowButton(page, name).click();
    await expect(depthCard(page)).toHaveCount(0);
    await expect(page.getByText("Finding…")).toHaveCount(0);
    expect(await rowsOf(page, 4)).toHaveLength(1);
  });
});
