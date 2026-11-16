import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, recordCalls, setValue } from "./helpers/app";
import { GRADED_NODES } from "./helpers/states";

// Show mask (docs/23 §5.8, §7.6 View card, §7.7 M; task N-11), in a real browser on the mock API: the toolbar's toggle and M, the View
// card's two segments, the Input and the Output (no mask), leaving the node view and Crop, and what the mock's own picture shows. The core
// draws the real mask and overlay (`crates/ym-core/tests/mask_view.rs`); here the page asks for it with `viewport_set_mode(mode, mask_view)`
// and the mock's viewport records what it was asked.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

/** A Radial at the middle of the photo, as the screenshots have it on Main. */
const RADIAL = { kind: "radial", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], radii: [0.3, 0.2], angle_deg: 0, feather: 0.4 };

type MaskView = { node: number; style: "overlay" | "mask_only"; row?: number | null } | null;
const maskView = (page: Page): Promise<MaskView> => page.evaluate(() => window.__yellowmoon!.mockApi.viewport.maskView as MaskView);
const showMask = (page: Page) => page.getByRole("group", { name: "Graph tools" }).getByRole("button", { name: "Show mask" });
const card = (page: Page, id: number | string) => page.locator(`[data-node-card="${id}"]`);

/** The mockup's photo with a Radial on Main and the graded graph after it, the node view open on Main's Mask tab. */
async function nodeView(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, {
    initialSettings: { DSC04204: { nodes: { "1": { masks: { items: [RADIAL] } }, ...GRADED_NODES.nodes }, links: GRADED_NODES.links } } as never,
  });
  await openPhotos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await expect(card(page, 4)).toBeVisible();
  await card(page, 1).click();
  await expect(card(page, 1)).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("tab", { name: "Mask", exact: true }).click();
}

/** The picture the cell shows (the mock's own, a data URL of an SVG), decoded. */
async function picture(page: Page): Promise<string> {
  const src = await page.locator(".ym-photo-cell__image").getAttribute("src");
  return decodeURIComponent((src ?? "").replace(/^data:image\/svg\+xml;charset=utf-8,/, ""));
}

test.describe("the toolbar's Show mask and M", () => {
  test("is a toggle with the tooltip 'Show mask (M)': a click asks the core for the Overlay of the selected corrector, another takes it away", async ({ page }) => {
    await nodeView(page);
    const calls = await recordCalls(page, "viewportSetMode");
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "false");
    await expect(showMask(page)).toHaveAccessibleDescription("Show mask (M)");
    expect(await maskView(page)).toBeNull();
    await showMask(page).click();
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "true");
    expect(await maskView(page)).toEqual({ node: 1, style: "overlay" });
    // The mode goes with it (Fit).
    expect((await calls()).at(-1)?.args).toEqual([{ kind: "fit" }, { node: 1, style: "overlay" }]);
    await showMask(page).click();
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "false");
    expect(await maskView(page)).toBeNull();
    expect((await calls()).at(-1)?.args).toEqual([{ kind: "fit" }, null]);
  });

  test("is M too, with the last style, and follows the corrector that is selected", async ({ page }) => {
    await nodeView(page);
    await page.getByRole("radio", { name: "Mask only" }).click();
    expect(await maskView(page)).toEqual({ node: 1, style: "mask_only" });
    await page.keyboard.press("m");
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "false");
    expect(await maskView(page)).toBeNull();
    await page.keyboard.press("m");
    expect(await maskView(page)).toEqual({ node: 1, style: "mask_only" });
    // Another corrector is another mask; the style stays.
    await card(page, 3).click();
    await expect(card(page, 3)).toHaveAttribute("aria-pressed", "true");
    await expect.poll(() => maskView(page)).toEqual({ node: 3, style: "mask_only" });
  });

  test("is off on the Input and the Output, with its sentence, and comes back with a corrector", async ({ page }) => {
    await nodeView(page);
    await showMask(page).click();
    await expect.poll(() => maskView(page)).toEqual({ node: 1, style: "overlay" });
    for (const marker of ["input", "output"]) {
      await card(page, marker).click();
      await expect(card(page, marker)).toHaveAttribute("aria-pressed", "true");
      await expect.poll(() => maskView(page)).toBeNull();
      await expect(showMask(page)).toHaveAttribute("aria-disabled", "true");
      await expect(showMask(page)).toHaveAttribute("aria-pressed", "false");
      await expect(showMask(page)).toHaveAccessibleDescription("The input and the output have no mask.");
      // M does nothing there either.
      await page.locator("body").press("m");
      expect(await maskView(page)).toBeNull();
    }
    await card(page, 1).click();
    await expect(showMask(page)).not.toHaveAttribute("aria-disabled", "true");
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "true");
    await expect.poll(() => maskView(page)).toEqual({ node: 1, style: "overlay" });
  });

  test("ends when the node view is left and when Crop opens, and comes back as it was", async ({ page }) => {
    await nodeView(page);
    await page.getByRole("radio", { name: "Mask only" }).click();
    expect(await maskView(page)).toEqual({ node: 1, style: "mask_only" });
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(page.locator('.ym-screen[data-screen="photos"]')).not.toHaveAttribute("data-nodes", "true");
    await expect.poll(() => maskView(page)).toBeNull();
    // M is the node view's key: in the simple view it does nothing.
    await page.keyboard.press("m");
    expect(await maskView(page)).toBeNull();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(card(page, 4)).toBeVisible();
    await card(page, 1).click();
    await expect.poll(() => maskView(page)).toEqual({ node: 1, style: "mask_only" });
    await page.getByRole("button", { name: "Crop", exact: true }).click();
    await expect.poll(() => maskView(page)).toBeNull();
    await page.getByRole("button", { name: "Crop", exact: true }).click();
    await expect.poll(() => maskView(page)).toEqual({ node: 1, style: "mask_only" });
  });
});

test.describe("the View card", () => {
  test("is under the Masks card: 'View', the key chip M, and Show mask with Overlay and Mask only (no Off)", async ({ page }) => {
    await nodeView(page);
    const view = page.getByRole("region", { name: "View" });
    await expect(view).toBeVisible();
    await expect(view.getByText("M", { exact: true })).toBeVisible();
    await expect(view.getByText("Show mask", { exact: true })).toBeVisible();
    const segments = view.getByRole("radiogroup", { name: "Show mask" }).getByRole("radio");
    await expect(segments).toHaveText(["Overlay", "Mask only"]);
    await expect(segments.first()).toHaveAttribute("aria-checked", "true");
    const masks = page.getByRole("region", { name: "Masks" });
    const [a, b] = [(await masks.boundingBox())!, (await view.boundingBox())!];
    expect(b.y).toBeGreaterThan(a.y + a.height - 1);
  });

  test("a segment sets the style and turns the view on, the chosen one too; the toolbar toggle keeps the style", async ({ page }) => {
    await nodeView(page);
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "false");
    await page.getByRole("radio", { name: "Overlay" }).click();
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "true");
    expect(await maskView(page)).toEqual({ node: 1, style: "overlay" });
    await page.getByRole("radio", { name: "Mask only" }).click();
    expect(await maskView(page)).toEqual({ node: 1, style: "mask_only" });
    await expect(page.getByRole("radio", { name: "Mask only" })).toHaveAttribute("aria-checked", "true");
    await showMask(page).click();
    expect(await maskView(page)).toBeNull();
    await expect(page.getByRole("radio", { name: "Mask only" })).toHaveAttribute("aria-checked", "true");
    await showMask(page).click();
    expect(await maskView(page)).toEqual({ node: 1, style: "mask_only" });
  });

  test("is one Tab stop (the chosen segment), and the arrows choose a segment", async ({ page }) => {
    await nodeView(page);
    const group = page.getByRole("radiogroup", { name: "Show mask" });
    // The last stop of the Masks card (Main's one row: its eye) is the one before the View card.
    await page.getByRole("button", { name: "Show this mask" }).focus();
    await page.keyboard.press("Tab");
    await expect(group.getByRole("radio", { name: "Overlay" })).toBeFocused();
    // One stop: the next Tab leaves the group.
    await page.keyboard.press("Tab");
    await expect(group.locator(":focus")).toHaveCount(0);
    await group.getByRole("radio", { name: "Overlay" }).focus();
    await page.keyboard.press("ArrowRight");
    await expect(group.getByRole("radio", { name: "Mask only" })).toBeFocused();
    expect(await maskView(page)).toEqual({ node: 1, style: "mask_only" });
  });
});

test.describe("what the mock's picture shows", () => {
  test("carries the Overlay or the mask, and neither in the Before view", async ({ page }) => {
    await nodeView(page);
    expect(await picture(page)).not.toContain("data-mask-view");
    await page.getByRole("radio", { name: "Overlay" }).click();
    await expect.poll(() => picture(page)).toContain('data-mask-view="overlay"');
    expect(await picture(page)).toContain("#ff6fd2");
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(() => picture(page)).toContain('data-mask-view="mask_only"');
    await page.getByRole("button", { name: "Before", exact: true }).click();
    await expect.poll(() => picture(page)).not.toContain("data-mask-view");
    await page.getByRole("button", { name: "Before", exact: true }).click();
    await expect.poll(() => picture(page)).toContain('data-mask-view="mask_only"');
  });
});


// ---- The Mask tab (docs/23 §7.6, task N-12) ----------------------------------------------------------------------------------------------------

/** What the page has set on a node's masks, read from the mock (the very one the app runs on). */
type Row = Record<string, unknown> & { kind: string; mode: string; invert: boolean; enabled: boolean; opacity: number };
const masksOf = (page: Page, node: number): Promise<{ invert: boolean; items: Row[] } | null> =>
  page.evaluate(async (id) => {
    const { mockApi } = window.__yellowmoon!;
    const state = await mockApi.sessionState();
    const rs = await mockApi.getSettings(state.photos[state.current_index!]!.id);
    const item = rs.settings.nodes.items.find((n) => n.id === id);
    return item && "masks" in item ? (item.masks as { invert: boolean; items: Row[] }) : null;
  }, node);
const rowsOf = async (page: Page, node: number): Promise<Row[]> => (await masksOf(page, node))?.items ?? [];

const rowList = (page: Page) => page.getByRole("list", { name: "Masks of this node" });
const rowButton = (page: Page, name: RegExp | string) => rowList(page).getByRole("button", { name });
const handle = (page: Page, name: RegExp | string) => page.getByRole("button", { name });
const layer = (page: Page) => page.locator("[data-mask-handles]");
const maskCard = (page: Page, name: string) => page.getByRole("region", { name, exact: true });

/** The mockup's photo with a graph of four nodes; Warmth (4), which has no mask, selected on its Mask tab. */
async function warmth(page: Page, options: Record<string, string | number> = {}, extra: Record<string, unknown> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...options });
  await configureMock(page, { initialSettings: { DSC04204: { nodes: { ...GRADED_NODES.nodes, ...extra }, links: GRADED_NODES.links } } as never });
  await openPhotos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await expect(card(page, 4)).toBeVisible();
  await card(page, 4).click();
  await expect(card(page, 4)).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("tab", { name: "Mask", exact: true }).click();
  await expect(page.getByRole("region", { name: "Masks" })).toBeVisible();
}

async function addMask(page: Page, name: string): Promise<void> {
  await page.getByRole("button", { name: "Add mask" }).click();
  await page.getByRole("menuitem", { name: new RegExp(`^${name}`) }).click();
}

/** The photo cell's box in the page: where a handle's pointer position is a share of the photo. */
async function cellBox(page: Page): Promise<{ x: number; y: number; width: number; height: number }> {
  return (await page.locator("[data-viewport]").boundingBox())!;
}

/** A drag of a handle by `dx`, `dy` CSS pixels, as a hand does it: down on the handle, a few moves, up. */
async function dragHandle(page: Page, name: RegExp | string, dx: number, dy: number): Promise<void> {
  const box = (await handle(page, name).boundingBox())!;
  const [x, y] = [box.x + box.width / 2, box.y + box.height / 2];
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (const k of [0.25, 0.5, 0.75, 1]) await page.mouse.move(x + dx * k, y + dy * k);
  await page.mouse.up();
}

/** The first `translate(x y)` of the mock's mask picture (a Radial's centre in the photo's own pixels), or null. */
async function radialAt(page: Page): Promise<[number, number] | null> {
  const found = /translate\(([-\d.]+) ([-\d.]+)\)/.exec(await picture(page));
  return found ? [Number(found[1]), Number(found[2])] : null;
}

/** The first Linear gradient of the mock's picture: the distance in the photo's own pixels between its two ends (the feather's width). */
async function gradientLength(page: Page): Promise<number> {
  const g = /x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"/.exec(await picture(page));
  return g ? Math.hypot(Number(g[3]) - Number(g[1]), Number(g[4]) - Number(g[2])) : 0;
}

/** Lets go of the focus so that a key press is the window's (a text box leaves ⌘Z to itself). */
const blur = (page: Page) => page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());

test.describe("adding a mask", () => {
  test("a Linear and a Radial: each is selected as it is added, with its card and its handles, and the chip says what is shown", async ({ page }) => {
    await warmth(page);
    await expect(page.getByRole("region", { name: "Masks" }).getByText("Masks add up. Subtract cuts a mask out.", { exact: false })).toBeVisible();
    await addMask(page, "Linear");
    await expect(rowList(page).getByRole("listitem")).toHaveCount(1);
    await expect(rowButton(page, "Linear mask, Feather 20")).toHaveAttribute("aria-pressed", "true");
    await expect(maskCard(page, "Linear")).toBeVisible();
    await expect(layer(page)).toBeVisible();
    for (const name of [/^Move the Linear mask/, /^Linear mask feather, start line/, /^Linear mask feather, end line/, /^Rotate the Linear mask/]) await expect(handle(page, name)).toBeVisible();
    await expect(page.getByText("MASK · LINEAR")).toBeVisible();
    await addMask(page, "Radial");
    await expect(rowList(page).getByRole("listitem")).toHaveCount(2);
    await expect(rowButton(page, "Radial mask, Feather 30")).toHaveAttribute("aria-pressed", "true");
    await expect(maskCard(page, "Radial")).toBeVisible();
    await expect(maskCard(page, "Linear")).toHaveCount(0);
    for (const name of [/^Move the Radial mask/, /^Radial mask width/, /^Radial mask height/, /^Rotate the Radial mask/]) await expect(handle(page, name)).toBeVisible();
    await expect(handle(page, /^Move the Linear mask/)).toHaveCount(0);
    await expect(page.getByText("MASK · RADIAL")).toBeVisible();
    // The core has them, with their defaults, in the order they were added.
    expect((await rowsOf(page, 4)).map((r) => r.kind)).toEqual(["linear", "radial"]);
    expect((await rowsOf(page, 4))[0]).toMatchObject({ mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], angle_deg: 0, feather: 0.2 });
    // The card's glyph in the node's card: a mask.
    await expect(card(page, 4).locator('[data-glyph="mask"], svg').first()).toBeVisible();
  });

  test("the Add mask menu lists the kinds the core says it can make, and no others", async ({ page }) => {
    await warmth(page);
    await page.getByRole("button", { name: "Add mask" }).click();
    const menu = page.getByRole("menu", { name: "Add mask" });
    await expect(menu.getByRole("menuitem")).toHaveText([/^Object/, /^Subject/, /^People/, /^Background/, /^Depth/, /^Linear/, /^Radial/, /^Brush/, /^Colour range/, /^Brightness range/]);
    await expect(menu.getByText("A gradient across the photo")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(menu).toHaveCount(0);
  });

  test("with every kind made, it has the groups of the spec; Subject and People are hidden, not greyed, where a Mac cannot make them", async ({ page }) => {
    await warmth(page, { mockMasks: "all" });
    await page.getByRole("button", { name: "Add mask" }).click();
    const menu = page.getByRole("menu", { name: "Add mask" });
    await expect(menu.getByRole("menuitem")).toHaveCount(10);
    for (const title of ["Find", "Shapes", "Keys"]) await expect(menu.getByText(title, { exact: true })).toBeVisible();
    await page.keyboard.press("Escape");
    await page.reload();
    await warmth(page, { mockMasks: "nomac" });
    await page.getByRole("button", { name: "Add mask" }).click();
    const noMac = page.getByRole("menu", { name: "Add mask" });
    await expect(noMac.getByRole("menuitem")).toHaveCount(8);
    await expect(noMac.getByRole("menuitem", { name: /^(Subject|People)/ })).toHaveCount(0);
    await expect(noMac.getByRole("menuitem", { name: /^Object/ })).toBeVisible();
  });

  test("opens to the left of the panel, over the photo, and grows to hold its ten items (at most 600 px tall)", async ({ page }) => {
    await warmth(page, { mockMasks: "all" });
    await page.getByRole("button", { name: "Add mask" }).click();
    const menu = (await page.getByRole("menu", { name: "Add mask" }).boundingBox())!;
    // A menu hides the page behind it from the accessibility tree, so the panel is found by its class.
    const panel = (await page.locator("aside.ym-tools").boundingBox())!;
    expect(menu.x + menu.width).toBeLessThanOrEqual(panel.x + 1);
    expect(menu.height).toBeLessThanOrEqual(600);
    expect(menu.height).toBeGreaterThan(520);
  });
});

test.describe("the handles on the photo", () => {
  test("a drag of the Radial's centre handle moves the mask in Mask only, and is one undo step", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Radial");
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(() => radialAt(page)).not.toBeNull();
    const before = (await radialAt(page))!;
    const cell = await cellBox(page);
    await dragHandle(page, /^Move the Radial mask/, 100, 60);
    await expect.poll(async () => ((await rowsOf(page, 4))[0]?.center as number[])[0]).toBeCloseTo(0.5 + 100 / cell.width, 2);
    const row = (await rowsOf(page, 4))[0]!;
    expect((row.center as number[])[1]).toBeCloseTo(0.5 + 60 / cell.height, 2);
    // The mask view follows: the mock's Radial is drawn at the new place.
    await expect.poll(async () => (await radialAt(page))![0]).toBeGreaterThan(before[0]);
    expect((await radialAt(page))![1]).toBeGreaterThan(before[1]);
    // The handle is where the mask is: the core's mapping of the new centre.
    const moved = (await handle(page, /^Move the Radial mask/).boundingBox())!;
    expect(moved.x + moved.width / 2).toBeCloseTo(cell.x + (row.center as number[])[0]! * cell.width, 0);
    // One step takes it all back.
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.center).toEqual([0.5, 0.5]);
    await expect.poll(async () => (await radialAt(page))![0]).toBeCloseTo(before[0], 0);
  });

  test("a drag of a Radial's radius handle sets the radius, and its rotate handle the angle", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Radial");
    const cell = await cellBox(page);
    // Radius 0.25 of the long edge (the cell's width); the handle is 60 px further out.
    await dragHandle(page, /^Radial mask width/, 60, 0);
    await expect.poll(async () => ((await rowsOf(page, 4))[0]?.radii as number[])[0]).toBeCloseTo(0.25 + 60 / cell.width, 2);
    expect(((await rowsOf(page, 4))[0]!.radii as number[])[1]).toBeCloseTo(0.18, 3);
    // The rotate handle is above the outline: a drag to the right of the centre turns the oval a quarter turn clockwise.
    const rotate = (await handle(page, /^Rotate the Radial mask/).boundingBox())!;
    const centre = (await handle(page, /^Move the Radial mask/).boundingBox())!;
    const [cx, cy] = [centre.x + centre.width / 2, centre.y + centre.height / 2];
    await page.mouse.move(rotate.x + rotate.width / 2, rotate.y + rotate.height / 2);
    await page.mouse.down();
    await page.mouse.move(cx + 40, cy - 30);
    await page.mouse.move(cx + 150, cy);
    await page.mouse.up();
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.angle_deg).toBeCloseTo(90, 0);
  });

  test("a drag of a Linear's feather grip changes the width of the gradient in Mask only, and its rotate handle turns the line", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Linear");
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(() => gradientLength(page)).toBeGreaterThan(0);
    const before = await gradientLength(page);
    const cell = await cellBox(page);
    // The end line is 0.1 of the long edge below the centre line; 60 px further down makes the feather 0.2 + 2 · 60 / 666.
    await dragHandle(page, /^Linear mask feather, end line/, 0, 60);
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.feather).toBeCloseTo(0.2 + (2 * 60) / cell.width, 2);
    await expect.poll(() => gradientLength(page)).toBeGreaterThan(before * 1.5);
    // Straight down from the centre is 90°.
    const rotate = (await handle(page, /^Rotate the Linear mask/).boundingBox())!;
    const centre = (await handle(page, /^Move the Linear mask/).boundingBox())!;
    const [cx, cy] = [centre.x + centre.width / 2, centre.y + centre.height / 2];
    await page.mouse.move(rotate.x + rotate.width / 2, rotate.y + rotate.height / 2);
    await page.mouse.down();
    await page.mouse.move(cx + 80, cy + 60);
    await page.mouse.move(cx, cy + 120);
    await page.mouse.up();
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.angle_deg).toBeCloseTo(90, 0);
  });

  test("they are the selected row's, on whatever tab the panel shows, and hide with the row's switch, Crop and a pick", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Radial");
    await page.getByRole("tab", { name: "Colour", exact: true }).click();
    await expect(layer(page)).toBeVisible();
    await expect(page.getByText("MASK · RADIAL")).toBeVisible();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await page.getByRole("switch", { name: "Turn this mask off" }).click();
    await expect(layer(page)).toHaveCount(0);
    await page.getByRole("switch", { name: "Turn this mask on" }).click();
    await expect(layer(page)).toBeVisible();
    await page.getByRole("button", { name: "Crop", exact: true }).click();
    await expect(layer(page)).toHaveCount(0);
    await page.getByRole("button", { name: "Crop", exact: true }).click();
    await expect(layer(page)).toBeVisible();
    // Another node has none of its own selected: no handles.
    await card(page, 3).click();
    await expect(layer(page)).toHaveCount(0);
    await expect(page.getByText(/^MASK · /)).toHaveCount(0);
  });

  test("a handle is a Tab stop, and the arrows nudge it 1 px, Shift 10 px, each as one undo step", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Linear");
    const cell = await cellBox(page);
    const position = ((await page.getByRole("group", { name: "Photos" }).textContent()) ?? "").trim();
    await handle(page, /^Move the Linear mask/).focus();
    await expect(handle(page, /^Move the Linear mask/)).toBeFocused();
    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => ((await rowsOf(page, 4))[0]?.center as number[])[0]).toBeCloseTo(0.5 + 1 / cell.width, 3);
    await page.keyboard.press("Shift+ArrowDown");
    await expect.poll(async () => ((await rowsOf(page, 4))[0]?.center as number[])[1]).toBeCloseTo(0.5 + 10 / cell.height, 3);
    // The arrows moved the handle, not the photo or the selection.
    await expect(page.getByRole("group", { name: "Photos" })).toContainText(position);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => ((await rowsOf(page, 4))[0]?.center as number[])[1]).toBeCloseTo(0.5, 3);
    expect(((await rowsOf(page, 4))[0]!.center as number[])[0]).toBeCloseTo(0.5 + 1 / cell.width, 3);
  });
});

test.describe("the rows", () => {
  test("Subtract, a row's Invert, the node's Invert and the switch each change the mask view, and each is a toggle of the row", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Radial");
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(() => picture(page)).toContain("radialGradient");
    const row = rowList(page).getByRole("listitem").first();
    // Subtract: black over the photo, the lone Subtract starts from the whole photo.
    await row.getByRole("radio", { name: "Subtract" }).click();
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.mode).toBe("subtract");
    await expect.poll(() => picture(page)).toMatch(/<mask [^>]*><g [^>]*><rect [^>]*fill="#fff"/);
    await row.getByRole("radio", { name: "Add" }).click();
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.mode).toBe("add");
    // The row's own Invert.
    const invertRow = row.getByRole("button", { name: "Invert this mask" });
    await invertRow.click();
    await expect(invertRow).toHaveAttribute("aria-pressed", "true");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.invert).toBe(true);
    await invertRow.click();
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.invert).toBe(false);
    // The node's Invert: the whole picture inverted.
    expect(await picture(page)).not.toContain("mvinvert)");
    await page.getByRole("region", { name: "Masks" }).getByRole("button", { name: "Invert", exact: true }).click();
    await expect.poll(async () => (await masksOf(page, 4))?.invert).toBe(true);
    await expect.poll(() => picture(page)).toContain('filter="url(#mvinvert)"');
    await page.getByRole("region", { name: "Masks" }).getByRole("button", { name: "Invert", exact: true }).click();
    await expect.poll(async () => (await masksOf(page, 4))?.invert).toBe(false);
    // The switch: a row that is off counts as absent, so the node's mask is the whole photo.
    await row.getByRole("switch", { name: "Turn this mask off" }).click();
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.enabled).toBe(false);
    await expect.poll(() => picture(page)).not.toContain("radialGradient");
    await row.getByRole("switch", { name: "Turn this mask on" }).click();
    await expect.poll(() => picture(page)).toContain("radialGradient");
  });

  test("a row's eye shows that row's mask alone, one at a time; Show mask, M, Esc and the Before view end it; the eye writes nothing", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Radial");
    await addMask(page, "Linear");
    await page.getByRole("radio", { name: "Mask only" }).click();
    await showMask(page).click();
    expect(await maskView(page)).toBeNull();
    const writes = await recordCalls(page, "updateSettings");
    const eye = (i: number) => rowList(page).getByRole("listitem").nth(i).getByRole("button", { name: /^(Show|Hide) this mask$/ });
    // Row 0, the Radial: its own gradient, in the style of the View card, and Show mask stays off.
    await eye(0).click();
    await expect(eye(0)).toHaveAttribute("aria-pressed", "true");
    await expect(eye(0)).toHaveAccessibleName("Hide this mask");
    expect(await maskView(page)).toEqual({ node: 4, style: "mask_only", row: 0 });
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "false");
    await expect.poll(() => picture(page)).toContain("radialGradient");
    expect(await picture(page)).not.toContain("linearGradient");
    // Row 1: the first goes off, the picture is the Linear.
    await eye(1).click();
    expect(await maskView(page)).toEqual({ node: 4, style: "mask_only", row: 1 });
    await expect(eye(0)).toHaveAttribute("aria-pressed", "false");
    await expect.poll(() => picture(page)).toContain("linearGradient");
    expect(await picture(page)).not.toContain("radialGradient");
    // The style comes from the View card, and choosing one leaves the eye on.
    await page.getByRole("radio", { name: "Overlay" }).click();
    expect(await maskView(page)).toEqual({ node: 4, style: "overlay", row: 1 });
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "false");
    // M turns Show mask on and the eye off.
    await blur(page);
    await page.keyboard.press("m");
    await expect(eye(1)).toHaveAttribute("aria-pressed", "false");
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "true");
    expect(await maskView(page)).toEqual({ node: 4, style: "overlay" });
    // The eye turns Show mask off; Esc ends the eye.
    await eye(0).click();
    await expect(showMask(page)).toHaveAttribute("aria-pressed", "false");
    await page.keyboard.press("Escape");
    await expect(eye(0)).toHaveAttribute("aria-pressed", "false");
    expect(await maskView(page)).toBeNull();
    // The Before view ends it.
    await eye(0).click();
    await page.getByRole("button", { name: "Before", exact: true }).click();
    await expect(eye(0)).toHaveAttribute("aria-pressed", "false");
    await page.getByRole("button", { name: "Before", exact: true }).click();
    expect(await maskView(page)).toBeNull();
    // None of it wrote the settings.
    expect(await writes()).toEqual([]);
  });

  test("the eye ends with the node, and shows a row that is off; the switch turns a row off, and Undo turns it back", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Radial");
    const row = rowList(page).getByRole("listitem").first();
    await row.getByRole("button", { name: "Show this mask" }).click();
    expect(await maskView(page)).toEqual({ node: 4, style: "overlay", row: 0 });
    // The switch: one patch, the row is dimmed, and its eye still shows it.
    await row.getByRole("switch", { name: "Turn this mask off" }).click();
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.enabled).toBe(false);
    await expect(row).toHaveAttribute("data-off", "true");
    expect(await maskView(page)).toEqual({ node: 4, style: "overlay", row: 0 });
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.enabled).toBe(true);
    await expect(row.getByRole("switch", { name: "Turn this mask off" })).toBeChecked();
    // Another node ends the eye.
    await card(page, 3).click();
    await expect.poll(() => maskView(page)).toBeNull();
    await card(page, 4).click();
    await expect(row.getByRole("button", { name: "Show this mask" })).toHaveAttribute("aria-pressed", "false");
  });

  test("Delete takes out the selected row, and the context menu moves a row and deletes it", async ({ page }) => {
    await warmth(page);
    const position = ((await page.getByRole("group", { name: "Photos" }).textContent()) ?? "").trim();
    await addMask(page, "Linear");
    await addMask(page, "Radial");
    await expect(rowList(page).getByRole("listitem")).toHaveCount(2);
    // A right click: Move up puts the Radial first.
    await rowButton(page, /^Radial mask/).click({ button: "right" });
    const menu = page.getByRole("menu", { name: "Radial mask actions" });
    await expect(menu.getByRole("menuitem")).toHaveText(["Move up", "Move down", "Delete mask"]);
    await menu.getByRole("menuitem", { name: "Move up" }).click();
    await expect.poll(async () => (await rowsOf(page, 4)).map((r) => r.kind)).toEqual(["radial", "linear"]);
    await expect(rowList(page).getByRole("listitem").first()).toContainText("Radial");
    // The selection followed the Radial: its card is still the one shown.
    await expect(maskCard(page, "Radial")).toBeVisible();
    // Delete, with the focus on the selected row's button.
    await rowButton(page, /^Radial mask/).click();
    await page.keyboard.press("Delete");
    await expect.poll(async () => (await rowsOf(page, 4)).map((r) => r.kind)).toEqual(["linear"]);
    // The node is still there, and so is the photo.
    await expect(card(page, 4)).toBeVisible();
    await expect(page.getByRole("group", { name: "Photos" })).toContainText(position);
    // The menu's Delete mask takes out the last one.
    await rowButton(page, /^Linear mask/).click({ button: "right" });
    await page.getByRole("menuitem", { name: "Delete mask" }).click();
    await expect.poll(async () => (await rowsOf(page, 4)).length).toBe(0);
    await expect(rowList(page)).toHaveCount(0);
    await expect(layer(page)).toHaveCount(0);
  });

  test("Reset removes every row and un-inverts; undo and redo bring them back, one step each", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Linear");
    await addMask(page, "Radial");
    await page.getByRole("region", { name: "Masks" }).getByRole("button", { name: "Invert", exact: true }).click();
    await expect.poll(async () => (await masksOf(page, 4))?.invert).toBe(true);
    await page.getByRole("region", { name: "Masks" }).getByRole("button", { name: "Reset" }).click();
    await expect.poll(async () => (await masksOf(page, 4))?.items.length).toBe(0);
    expect((await masksOf(page, 4))?.invert).toBe(false);
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await masksOf(page, 4))?.items.length).toBe(2);
    expect((await masksOf(page, 4))?.invert).toBe(true);
    await page.keyboard.press("Meta+Shift+z");
    await expect.poll(async () => (await masksOf(page, 4))?.items.length).toBe(0);
    // Back through the invert and both adds.
    for (const [rows, invert] of [[2, true], [2, false], [1, false], [0, false]] as const) {
      await page.keyboard.press("Meta+z");
      await expect.poll(async () => (await masksOf(page, 4))?.items.length).toBe(rows);
      expect((await masksOf(page, 4))?.invert).toBe(invert);
    }
  });
});

test.describe("the ranges", () => {
  test("a Colour range starts with Pick colour on: a click on the photo asks for the chroma at the node and sets the centre; Esc ends the pick", async ({ page }) => {
    await warmth(page);
    const asked = await recordCalls(page, "sampleChroma");
    await addMask(page, "Colour range");
    const toggle = page.getByRole("region", { name: "Colour range" }).getByRole("button", { name: "Pick colour" });
    await expect(toggle).toHaveAttribute("aria-pressed", "true");
    const pickLayer = page.getByRole("application", { name: /^Pick colour for the Colour range mask/ });
    await expect(pickLayer).toBeVisible();
    // The handles of a range: none, and the pick is on.
    await expect(layer(page)).toHaveCount(0);
    await expect(page.getByText("Click a colour in the photo.")).toBeVisible();
    const cell = await cellBox(page);
    await page.mouse.click(cell.x + cell.width * 0.5, cell.y + cell.height * 0.1);
    await expect.poll(async () => (await asked()).length).toBe(1);
    // At the node, and at its input: the picture it reads, before its own trim (Warmth has one), which is what the range selects on.
    const [x, y, at, tap] = (await asked())[0]!.args as [number, number, number, string];
    expect([at, tap]).toEqual([4, "node_input"]);
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.center).not.toEqual([0, 0]);
    const center = (await rowsOf(page, 4))[0]!.center as number[];
    expect(Math.hypot(center[0]!, center[1]!)).toBeLessThanOrEqual(0.5 + 1e-9);
    // The centre is the chroma of the clicked pixel at the node's input, and not the one at its warp's input, which its trim has moved.
    const read = await page.evaluate(
      async ([px, py]) => {
        const { mockApi } = window.__yellowmoon!;
        return { input: await mockApi.sampleChroma(px!, py!, 4, "node_input"), warp: await mockApi.sampleChroma(px!, py!, 4) };
      },
      [x, y] as const,
    );
    expect(center).toEqual([read.input!.a, read.input!.b]);
    expect(read.warp).not.toEqual(read.input);
    // It stays on for another click, and Esc ends it.
    await expect(pickLayer).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(pickLayer).toHaveCount(0);
    await expect(toggle).toHaveAttribute("aria-pressed", "false");
    // The toggle starts it again, and it ends with the tab.
    await toggle.click();
    await expect(pickLayer).toBeVisible();
    await page.getByRole("tab", { name: "Colour", exact: true }).click();
    await expect(pickLayer).toHaveCount(0);
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(toggle).toHaveAttribute("aria-pressed", "false");
  });

  test("the Colour range's sliders: Width on the square scale, Softness and Opacity", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Colour range");
    await page.keyboard.press("Escape");
    await setValue(page, "Width", "40");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.width).toBeCloseTo(0.005 + 0.295 * 0.16, 4);
    await setValue(page, "Softness", "80");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.softness).toBeCloseTo(0.8, 3);
    await setValue(page, "Opacity", "60");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.opacity).toBeCloseTo(0.6, 3);
    await expect(rowButton(page, /^Colour range mask, Width 40/)).toBeVisible();
  });

  test("a Brightness range: Low and High as the code values times 100, and Low stays at or under High", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Brightness range");
    await expect(rowButton(page, "Brightness range mask, 35 to 100")).toBeVisible();
    await setValue(page, "High", "70");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.high).toBeCloseTo(0.7, 3);
    await setValue(page, "Low", "20");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.low).toBeCloseTo(0.2, 3);
    await expect(rowButton(page, "Brightness range mask, 20 to 70")).toBeVisible();
    await setValue(page, "Low", "90");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.low).toBeCloseTo(0.7, 3);
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.low).toBeCloseTo(0.2, 3);
  });

  test("a slider's drag is one undo step", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Linear");
    const slider = page.getByRole("slider", { name: "Feather" });
    const box = (await slider.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + 60, box.y + box.height / 2, { steps: 5 });
    await page.mouse.move(box.x + 100, box.y + box.height / 2, { steps: 5 });
    await page.mouse.up();
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.feather).not.toBe(0.2);
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await rowsOf(page, 4))[0]?.feather).toBe(0.2);
    await expect.poll(async () => (await rowsOf(page, 4)).length).toBe(1);
  });
});

test.describe("Copy settings and Paste", () => {
  test("the rows come to the other photo as the spec says: an Object row arrives empty and reads 'Click the object on this photo'", async ({ page }) => {
    const object = { kind: "object", mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, points: [[0.41, 0.33, true], [0.52, 0.31, true]], box: null };
    const linear = { kind: "linear", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.3], angle_deg: 12, feather: 0.4 };
    await warmth(page, {}, { "4": { kind: "corrector", name: "Warmth", pos: [3, 0], masks: { invert: true, items: [object, linear] } } });
    await expect(rowButton(page, "Object mask, 2 points")).toBeVisible();
    const here = ((await page.getByRole("group", { name: "Photos" }).textContent()) ?? "").trim();
    await page.getByRole("button", { name: "Copy settings" }).click();
    await page.getByRole("button", { name: "Previous photo" }).click();
    await expect(page.getByRole("group", { name: "Photos" })).not.toContainText(here);
    await page.getByRole("button", { name: "Paste", exact: true }).click();
    await expect(card(page, 4)).toBeVisible();
    await card(page, 4).click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(rowButton(page, "Object mask, Click the object on this photo")).toBeVisible();
    await expect(rowButton(page, "Linear mask, Feather 40")).toBeVisible();
    const rows = await rowsOf(page, 4);
    expect(rows[0]).toMatchObject({ kind: "object", points: [], box: null });
    expect(rows[1]).toEqual(linear);
    expect((await masksOf(page, 4))?.invert).toBe(true);
  });
});

// ---- The Brush tool (docs/23 §7.6, task N-14) -------------------------------------------------------------------------------------------------

const brushLayer = (page: Page) => page.locator("[data-brush-layer]");
const brushCursor = (page: Page) => page.locator("[data-brush-cursor]");
/** The strokes of a Brush row in the mock's Mask only picture: their ink (white paints, black erases), points and width. */
async function strokesDrawn(page: Page): Promise<Array<{ ink: string; points: string; width: number }>> {
  const svg = await picture(page);
  return [...svg.matchAll(/<polyline points="([^"]+)" fill="none" stroke="(#[0-9a-f]{3})" stroke-width="([\d.]+)"/g)].map((m) => ({ ink: m[2]!, points: m[1]!, width: Number(m[3]) }));
}
const brushRows = async (page: Page) => (await rowsOf(page, 4)).filter((r) => r.kind === "brush") as Array<Row & { strokes: Array<{ size: number; hardness: number; erase: boolean; points: number[][] }> }>;

/** A stroke by hand over the photo cell: `from` and `to` are shares of the cell; down, a few moves, up. */
async function paint(page: Page, from: [number, number], to: [number, number], steps = 6): Promise<void> {
  const cell = await cellBox(page);
  await page.mouse.move(cell.x + cell.width * from[0], cell.y + cell.height * from[1]);
  await page.mouse.down();
  await page.mouse.move(cell.x + cell.width * to[0], cell.y + cell.height * to[1], { steps });
  await page.mouse.up();
}

test.describe("the Brush tool", () => {
  test("adding a Brush row selects it, shows the BRUSH card and the cursor circle, and a stroke changes the Mask only view", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Brush");
    await expect(rowButton(page, "Brush mask, Paint on the photo")).toHaveAttribute("aria-pressed", "true");
    await expect(maskCard(page, "Brush")).toBeVisible();
    await expect(brushLayer(page)).toBeVisible();
    await expect(page.getByText("MASK · BRUSH")).toBeVisible();
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(async () => await picture(page)).toContain("data-mask-view");
    expect(await strokesDrawn(page)).toEqual([]);
    // The cursor is a circle at the brush's size (0.03 of the long edge, the cell's width here) that follows the pointer.
    const cell = await cellBox(page);
    await page.mouse.move(cell.x + cell.width * 0.3, cell.y + cell.height * 0.4);
    await expect(brushCursor(page)).toBeVisible();
    await expect.poll(async () => Number(await brushCursor(page).getAttribute("data-radius"))).toBeCloseTo((0.03 * cell.width) / 2, 0);
    const ring = (await brushCursor(page).boundingBox())!;
    expect(ring.x + ring.width / 2).toBeCloseTo(cell.x + cell.width * 0.3, 0);
    expect(ring.y + ring.height / 2).toBeCloseTo(cell.y + cell.height * 0.4, 0);
    // A stroke: drawn on the page as it goes, sent when the pointer is released.
    await page.mouse.down();
    await page.mouse.move(cell.x + cell.width * 0.6, cell.y + cell.height * 0.5, { steps: 6 });
    await expect(page.locator("[data-brush-layer] polyline[data-live]")).toHaveCount(1);
    expect(await brushRows(page)).toHaveLength(1);
    expect((await brushRows(page))[0]!.strokes).toHaveLength(0);
    await page.mouse.up();
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(1);
    const stroke = (await brushRows(page))[0]!.strokes[0]!;
    expect(stroke).toMatchObject({ size: 0.03, hardness: 0.5, erase: false });
    expect(stroke.points[0]![0]).toBeCloseTo(0.3, 2);
    expect(stroke.points[0]![1]).toBeCloseTo(0.4, 2);
    expect(stroke.points.at(-1)![0]).toBeCloseTo(0.6, 2);
    expect(stroke.points.at(-1)![1]).toBeCloseTo(0.5, 2);
    // A straight drag is stored as its two ends.
    expect(stroke.points).toHaveLength(2);
    // The mask view shows it: a white line as wide as the brush (30 px of the scene's 1000-wide long edge ... the scene's own units).
    await expect.poll(async () => (await strokesDrawn(page)).length).toBe(1);
    expect((await strokesDrawn(page))[0]!.ink).toBe("#fff");
    await expect(rowButton(page, "Brush mask, 1 stroke")).toBeVisible();
    // The page's own drawing of the stroke goes once the core has the picture.
    await expect(page.locator("[data-brush-layer] polyline")).toHaveCount(0);
  });

  test("Erase takes paint away: the stroke is an erase stroke and the view draws it in black over the paint", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Brush");
    await page.getByRole("radio", { name: "Mask only" }).click();
    await paint(page, [0.2, 0.5], [0.8, 0.5]);
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(1);
    await maskCard(page, "Brush").getByRole("radio", { name: "Erase" }).click();
    await expect(maskCard(page, "Brush").getByRole("radio", { name: "Erase" })).toBeChecked();
    await expect(brushCursor(page).or(brushLayer(page))).toBeVisible();
    await paint(page, [0.45, 0.4], [0.55, 0.6]);
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(2);
    expect((await brushRows(page))[0]!.strokes.map((s) => s.erase)).toEqual([false, true]);
    await expect.poll(async () => (await strokesDrawn(page)).map((s) => s.ink)).toEqual(["#fff", "#000"]);
    await expect(rowButton(page, "Brush mask, 2 strokes")).toBeVisible();
    // Back to Paint.
    await maskCard(page, "Brush").getByRole("radio", { name: "Paint" }).click();
    await paint(page, [0.2, 0.3], [0.8, 0.3]);
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.map((s) => s.erase)).toEqual([false, true, false]);
  });

  test("a stroke is one undo step, redo puts it back, and the view follows", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Brush");
    await page.getByRole("radio", { name: "Mask only" }).click();
    await paint(page, [0.2, 0.3], [0.8, 0.3]);
    await paint(page, [0.2, 0.6], [0.8, 0.6]);
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(2);
    await expect.poll(async () => (await strokesDrawn(page)).length).toBe(2);
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(1);
    await expect.poll(async () => (await strokesDrawn(page)).length).toBe(1);
    await expect(rowButton(page, "Brush mask, 1 stroke")).toBeVisible();
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(0);
    await expect(rowButton(page, "Brush mask, Paint on the photo")).toBeVisible();
    // The row is still there: adding it was a step of its own.
    expect(await brushRows(page)).toHaveLength(1);
    await page.keyboard.press("Meta+Shift+z");
    await page.keyboard.press("Meta+Shift+z");
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(2);
    await expect.poll(async () => (await strokesDrawn(page)).length).toBe(2);
  });

  test("Size and Hardness are for the next strokes: they change the cursor and the next stroke, not the ones drawn", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Brush");
    await paint(page, [0.2, 0.3], [0.8, 0.3]);
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(1);
    await setValue(page, "Size", "100");
    await setValue(page, "Hardness", "100");
    const first = (await brushRows(page))[0]!.strokes[0]!;
    expect(first).toMatchObject({ size: 0.03, hardness: 0.5 });
    const cell = await cellBox(page);
    await page.mouse.move(cell.x + cell.width * 0.5, cell.y + cell.height * 0.5);
    // Size 100 is 0.25 of the long edge: a radius of an eighth of the cell's width.
    await expect.poll(async () => Number(await brushCursor(page).getAttribute("data-radius"))).toBeCloseTo((0.25 * cell.width) / 2, 0);
    await expect(brushCursor(page).locator(".ym-brush-cursor__core")).toHaveCount(0);
    await paint(page, [0.3, 0.7], [0.7, 0.7]);
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(2);
    const rows = await brushRows(page);
    expect(rows[0]!.strokes[0]).toMatchObject({ size: 0.03, hardness: 0.5 });
    expect(rows[0]!.strokes[1]).toMatchObject({ size: 0.25, hardness: 1 });
    // No patch for the sliders themselves: the row holds its two strokes and its fields at their defaults.
    expect(rows[0]).toMatchObject({ feather: 0, grow: 0, refine: 0, opacity: 1 });
    await expect(page.getByRole("textbox", { name: "Size value" })).toHaveValue("100");
    // A small brush is within reach: Size 1 is 0.001025 of the long edge, under a pixel across on this cell (the circle is drawn at least 2 px wide).
    await setValue(page, "Size", "1");
    await page.mouse.move(cell.x + cell.width * 0.4, cell.y + cell.height * 0.5);
    await expect.poll(async () => Number(await brushCursor(page).getAttribute("data-radius"))).toBeLessThanOrEqual(1);
  });

  test("the raster sliders write the row: Refine edges, Feather, Grow and Opacity, with the view following Feather", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Brush");
    await page.getByRole("radio", { name: "Mask only" }).click();
    await paint(page, [0.2, 0.5], [0.8, 0.5]);
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(1);
    await setValue(page, "Refine edges", "40");
    await setValue(page, "Feather", "50");
    await setValue(page, "Grow", "-30");
    await setValue(page, "Opacity", "80");
    await expect.poll(async () => (await brushRows(page))[0]).toMatchObject({ refine: 0.4, feather: 0.5, grow: -0.3, opacity: 0.8 });
    expect((await brushRows(page))[0]!.strokes).toHaveLength(1);
    await expect.poll(async () => await picture(page)).toContain('fill-opacity="0.8"');
    await expect.poll(async () => await picture(page)).toContain("feGaussianBlur");
    // Reset puts the row back as a new Brush row: no strokes, the fields at their defaults.
    await maskCard(page, "Brush").getByRole("button", { name: "Reset", exact: true }).click();
    await expect.poll(async () => (await brushRows(page))[0]).toMatchObject({ refine: 0, feather: 0, grow: 0, opacity: 1, strokes: [] });
    await expect(rowButton(page, "Brush mask, Paint on the photo")).toBeVisible();
    expect(await strokesDrawn(page)).toEqual([]);
  });

  test("Esc ends the tool, and the row, Paint or Erase starts it again; another row or node ends it", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Brush");
    await expect(brushLayer(page)).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(brushLayer(page)).toHaveCount(0);
    await expect(maskCard(page, "Brush").getByText("Choose Paint or Erase to draw on the photo.")).toBeVisible();
    await maskCard(page, "Brush").getByRole("radio", { name: "Paint" }).click();
    await expect(brushLayer(page)).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(brushLayer(page)).toHaveCount(0);
    await rowButton(page, /^Brush mask/).click();
    await expect(brushLayer(page)).toBeVisible();
    // Crop hides it; closing Crop brings it back.
    await page.getByRole("button", { name: "Crop", exact: true }).click();
    await expect(brushLayer(page)).toHaveCount(0);
    await page.getByRole("button", { name: "Crop", exact: true }).click();
    await expect(brushLayer(page)).toBeVisible();
    // Before hides it.
    await page.getByRole("button", { name: "Before", exact: true }).click();
    await expect(brushLayer(page)).toHaveCount(0);
    await page.getByRole("button", { name: "Before", exact: true }).click();
    await expect(brushLayer(page)).toBeVisible();
    // Another node ends it for good.
    await card(page, 3).click();
    await expect(brushLayer(page)).toHaveCount(0);
    await card(page, 4).click();
    await expect(brushLayer(page)).toHaveCount(0);
  });

  test("a stroke does not pan the photo, and the wheel and the zoom keys still work", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Brush");
    const calls = await recordCalls(page, "viewportInput");
    await blur(page);
    await page.keyboard.press("z");
    await expect(page.getByRole("button", { name: /^100%/ })).toBeVisible();
    const before = (await calls()).length;
    await paint(page, [0.3, 0.4], [0.6, 0.5]);
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(1);
    expect((await calls()).length).toBe(before);
    // The wheel zooms with ⌘ held, as it does without the tool.
    const cell = await cellBox(page);
    await page.mouse.move(cell.x + cell.width / 2, cell.y + cell.height / 2);
    await page.keyboard.down("Meta");
    await page.mouse.wheel(0, -200);
    await page.keyboard.up("Meta");
    await expect.poll(async () => (await calls()).length).toBeGreaterThan(before);
    await expect(brushLayer(page)).toBeVisible();
  });

  test("the drawing surface is not a Tab stop and has no name; the card's controls are", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Brush");
    await expect(brushLayer(page)).toHaveAttribute("aria-hidden", "true");
    expect(await brushLayer(page).evaluate((el) => el.querySelectorAll("button, [tabindex], input, a[href]").length)).toBe(0);
    expect(await brushLayer(page).evaluate((el) => el.hasAttribute("tabindex"))).toBe(false);
  });

  test("Copy settings and Paste carry the Brush row, strokes and all, to another photo", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Brush");
    await paint(page, [0.2, 0.3], [0.8, 0.4]);
    await maskCard(page, "Brush").getByRole("radio", { name: "Erase" }).click();
    await paint(page, [0.4, 0.2], [0.5, 0.6]);
    await expect.poll(async () => (await brushRows(page))[0]?.strokes.length).toBe(2);
    const copied = (await brushRows(page))[0]!;
    const here = ((await page.getByRole("group", { name: "Photos" }).textContent()) ?? "").trim();
    await page.getByRole("button", { name: "Copy settings" }).click();
    await page.getByRole("button", { name: "Previous photo" }).click();
    await expect(page.getByRole("group", { name: "Photos" })).not.toContainText(here);
    await page.getByRole("button", { name: "Paste", exact: true }).click();
    await expect(card(page, 4)).toBeVisible();
    await card(page, 4).click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(rowButton(page, "Brush mask, 2 strokes")).toBeVisible();
    expect((await brushRows(page))[0]).toEqual(copied);
  });
});

// ---- The Background mask (docs/23 §4.6 and §7.6, task N-15) ----------------------------------------------------------------------------------

/** The Background rows of node 4. */
const backgroundRows = async (page: Page) => (await rowsOf(page, 4)).filter((r) => r.kind === "background") as Array<Row & { level: number; softness: number; feather: number; grow: number; refine: number }>;
/** The mask the mock draws for a Background row in its Mask only picture: the row's Level and Softness are on the rect, the grey of each area in the map. */
const backgroundDrawn = async (page: Page): Promise<string> => {
  const svg = await picture(page);
  return [...svg.matchAll(/<mask id="mvg\d+"[^>]*>(.*?)<\/mask>/gs)].map((m) => m[1]).join("|");
};

test.describe("the Background mask", () => {
  test("adding a Background row selects it and shows the BACKGROUND card; while the core finds the photo's map the row says 'Finding…' with a spinner", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Background");
    await expect(maskCard(page, "Background")).toBeVisible();
    await expect(page.getByText("MASK · BACKGROUND")).toBeVisible();
    await expect(rowButton(page, "Background mask, Level 50")).toHaveAttribute("aria-pressed", "true");
    // The core says it is making the photo's first blur map (`viewport_busy.finding`; the mock does for 400 ms at the speed of a real
    // run, and at once under `mockSpeed=Infinity`, so the test holds the event itself): the subtitle is a spinner and the words.
    await page.evaluate(() => window.__yellowmoon!.mockApi.emitEvent("viewport_busy", { busy: true, cpu: false, finding: true }));
    await expect(page.locator(".ym-mask-row__sub[data-finding]")).toBeVisible();
    await expect(page.locator(".ym-mask-row__sub[data-finding] svg")).toBeVisible();
    await expect(rowButton(page, "Background mask, Finding…")).toBeVisible();
    await page.evaluate(() => window.__yellowmoon!.mockApi.emitEvent("viewport_busy", { busy: false, cpu: false, finding: false }));
    await expect(page.locator(".ym-mask-row__sub[data-finding]")).toHaveCount(0);
    await expect(rowButton(page, "Background mask, Level 50")).toBeVisible();
    // The card: Level and Softness, then the four raster sliders, at their defaults.
    const sliders = maskCard(page, "Background").getByRole("slider");
    await expect(sliders).toHaveCount(6);
    for (const name of ["Level", "Softness", "Refine edges", "Feather", "Grow", "Opacity"]) await expect(maskCard(page, "Background").getByRole("slider", { name })).toBeVisible();
    expect(await backgroundRows(page)).toMatchObject([{ level: 0.5, softness: 0.5, feather: 0, grow: 0, refine: 0.5, opacity: 1, mode: "add" }]);
    await setValue(page, "Level", "60");
    await expect.poll(async () => (await backgroundRows(page))[0]?.level).toBeCloseTo(0.6, 3);
    await expect(rowButton(page, "Background mask, Level 60")).toBeVisible();
  });

  test("the mock finds a photo's map on its first frame, at the speed of a real run: 'Finding…' for 400 ms, once", async ({ page }) => {
    await warmth(page, { mockSpeed: 1 });
    const seen: Array<{ busy: boolean; finding: boolean }> = [];
    await page.evaluate(() => {
      (window as unknown as { __busy: unknown[] }).__busy = [];
      window.__yellowmoon!.mockApi.on("viewport_busy", (e) => (window as unknown as { __busy: unknown[] }).__busy.push({ busy: e.busy, finding: e.finding }));
    });
    await addMask(page, "Background");
    await expect.poll(() => page.evaluate(() => (window as unknown as { __busy: unknown[] }).__busy.length), { timeout: 5000 }).toBeGreaterThanOrEqual(2);
    seen.push(...(await page.evaluate(() => (window as unknown as { __busy: Array<{ busy: boolean; finding: boolean }> }).__busy)));
    expect(seen.at(0)).toEqual({ busy: true, finding: true });
    expect(seen.at(-1)).toEqual({ busy: false, finding: false });
    await expect(rowButton(page, "Background mask, Level 50")).toBeVisible();
  });

  test("Level and Softness change the Mask only picture, undo takes the change back, and the raster sliders write their fields", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Background");
    await expect(rowButton(page, "Background mask, Level 50")).toBeVisible();
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(async () => await picture(page)).toContain("data-background-row");
    const first = await backgroundDrawn(page);
    expect(first).not.toBe("");
    await setValue(page, "Level", "20");
    await expect.poll(async () => (await backgroundRows(page))[0]?.level).toBeCloseTo(0.2, 3);
    await expect(rowButton(page, "Background mask, Level 20")).toBeVisible();
    await expect.poll(async () => await backgroundDrawn(page)).not.toBe(first);
    const lowLevel = await backgroundDrawn(page);
    await setValue(page, "Softness", "90");
    await expect.poll(async () => (await backgroundRows(page))[0]?.softness).toBeCloseTo(0.9, 3);
    await expect.poll(async () => await backgroundDrawn(page)).not.toBe(lowLevel);
    // Undo takes the Softness back, then the Level, and the picture follows each.
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await backgroundRows(page))[0]?.softness).toBeCloseTo(0.5, 3);
    await expect.poll(async () => await backgroundDrawn(page)).toBe(lowLevel);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await backgroundRows(page))[0]?.level).toBeCloseTo(0.5, 3);
    await expect.poll(async () => await backgroundDrawn(page)).toBe(first);
    await page.keyboard.press("Meta+Shift+z");
    await expect.poll(async () => (await backgroundRows(page))[0]?.level).toBeCloseTo(0.2, 3);
    // The four raster sliders.
    await setValue(page, "Refine edges", "80");
    await expect.poll(async () => (await backgroundRows(page))[0]?.refine).toBeCloseTo(0.8, 3);
    await setValue(page, "Feather", "30");
    await expect.poll(async () => (await backgroundRows(page))[0]?.feather).toBeCloseTo(0.3, 3);
    await setValue(page, "Grow", "-25");
    await expect.poll(async () => (await backgroundRows(page))[0]?.grow).toBeCloseTo(-0.25, 3);
    await setValue(page, "Opacity", "70");
    await expect.poll(async () => (await backgroundRows(page))[0]?.opacity).toBeCloseTo(0.7, 3);
    // Reset puts the row back as a new Background in the same mode.
    await maskCard(page, "Background").getByRole("button", { name: "Reset", exact: true }).click();
    await expect.poll(async () => (await backgroundRows(page))[0]).toMatchObject({ level: 0.5, softness: 0.5, feather: 0, grow: 0, refine: 0.5, opacity: 1 });
    await expect(rowButton(page, "Background mask, Level 50")).toBeVisible();
  });

  test("the mock's map: the sky is background at the default Level, the far hill is not, and a row with the eye off is no mask", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Background");
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(async () => await picture(page)).toContain("data-background-row");
    // Three areas, in the order sky, hill, near ground: at Level 50 and Softness 50 the sky is lit, the hill dark.
    const greys = [...(await backgroundDrawn(page)).matchAll(/fill="rgb\((\d+),\d+,\d+\)"/g)].map((m) => Number(m[1]));
    expect(greys).toHaveLength(3);
    expect(greys[0]).toBe(255);
    expect(greys[1]).toBe(0);
    expect(greys[2]).toBeGreaterThan(100);
    expect(greys[2]).toBeLessThan(255);
    // The row off: it is absent, so the node (with no row left) shows the whole photo.
    await rowList(page).getByRole("switch", { name: "Turn this mask off" }).click();
    await expect.poll(async () => await picture(page)).not.toContain("data-background-row");
  });

  test("Copy settings and Paste carry the Background row, and the other photo finds its own map", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Background");
    await setValue(page, "Level", "65");
    await setValue(page, "Feather", "20");
    await expect.poll(async () => (await backgroundRows(page))[0]?.feather).toBeCloseTo(0.2, 3);
    await expect(rowButton(page, "Background mask, Level 65")).toBeVisible();
    const copied = (await backgroundRows(page))[0]!;
    const here = ((await page.getByRole("group", { name: "Photos" }).textContent()) ?? "").trim();
    await page.getByRole("button", { name: "Copy settings" }).click();
    await page.getByRole("button", { name: "Previous photo" }).click();
    await expect(page.getByRole("group", { name: "Photos" })).not.toContainText(here);
    await page.getByRole("button", { name: "Paste", exact: true }).click();
    await expect(card(page, 4)).toBeVisible();
    await card(page, 4).click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    // The row is the same, and the new photo's map is made for it.
    await expect(rowButton(page, /^Background mask, (Level 65|Finding…)/)).toBeVisible();
    await expect(rowButton(page, "Background mask, Level 65")).toBeVisible();
    expect((await backgroundRows(page))[0]).toEqual(copied);
  });

  test("the Add mask menu lists Background in the Find group, after Object, Subject and People, with its line", async ({ page }) => {
    await warmth(page);
    await page.getByRole("button", { name: "Add mask" }).click();
    const menu = page.getByRole("menu", { name: "Add mask" });
    await expect(menu.getByRole("menuitem").nth(3)).toHaveText(/^Background/);
    await expect(menu.getByText("What is out of focus, from blur")).toBeVisible();
    await expect(menu.getByText("Find", { exact: true })).toBeVisible();
  });
});

// ---- The Object tool (docs/23 §4.4 and §7.6, task N-18) --------------------------------------------------------------------------------------

const objectLayer = (page: Page) => page.locator("[data-object-layer]");
const objectRows = async (page: Page) =>
  (await rowsOf(page, 4)).filter((r) => r.kind === "object") as Array<Row & { points: Array<[number, number, boolean]>; box: [number, number, number, number] | null }>;
/** A click on the photo at a share of the cell, with ⌥ when asked. */
async function clickPhoto(page: Page, at: [number, number], alt = false): Promise<void> {
  const cell = await cellBox(page);
  if (alt) await page.keyboard.down("Alt");
  await page.mouse.click(cell.x + cell.width * at[0], cell.y + cell.height * at[1]);
  if (alt) await page.keyboard.up("Alt");
}

test.describe("the Object tool", () => {
  test("adding an Object row shows the OBJECT card; a click on the mountain finds it: one patch, the point and the pill, and the Mask only view draws it", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Object");
    await expect(rowButton(page, /^Object mask, (Getting ready…|Click the object on this photo)/)).toHaveAttribute("aria-pressed", "true");
    await expect(maskCard(page, "Object")).toBeVisible();
    await expect(page.getByText("MASK · OBJECT")).toBeVisible();
    await expect(objectLayer(page)).toBeVisible();
    await expect(maskCard(page, "Object").getByText("Click the photo to add to the object. ⌥-click to take a part away.")).toBeVisible();
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(async () => await picture(page)).toContain("data-mask-view");
    // Nothing is selected yet.
    expect(await picture(page)).not.toContain("data-object-row");
    const calls = await recordCalls(page, "updateSettings");
    await clickPhoto(page, [0.5, 0.7]);
    await expect(page.getByText("Object found · 0.4 s")).toBeVisible();
    await expect.poll(async () => (await objectRows(page))[0]?.points).toEqual([[0.5, 0.7, true]]);
    expect(await calls()).toHaveLength(1);
    await expect(page.locator("[data-object-point]")).toHaveCount(1);
    await expect(rowButton(page, "Object mask, 1 point")).toBeVisible();
    await expect.poll(async () => await picture(page)).toContain("data-object-row");
    // The pill goes after three seconds.
    await expect(page.getByText("Object found · 0.4 s")).toHaveCount(0, { timeout: 5000 });
  });

  test("a click that finds nothing leaves the row and the settings as they were and says so in the error bar", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Object");
    await expect(objectLayer(page)).toBeVisible();
    const calls = await recordCalls(page, "updateSettings");
    await clickPhoto(page, [0.2, 0.1]);
    await expect(page.getByRole("alert")).toContainText("Nothing found there. Click another part of the object, or draw a box.");
    expect(await calls()).toHaveLength(0);
    expect((await objectRows(page))[0]!.points).toEqual([]);
    await expect(page.locator("[data-object-point]")).toHaveCount(0);
  });

  test("⌥-click takes a part away (a point with a minus), a drag with Box draws a box, and each is one undo step", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Object");
    await expect(objectLayer(page)).toBeVisible();
    await clickPhoto(page, [0.45, 0.7]);
    await expect.poll(async () => (await objectRows(page))[0]?.points.length).toBe(1);
    await clickPhoto(page, [0.6, 0.75], true);
    await expect.poll(async () => (await objectRows(page))[0]?.points.map((p) => p[2])).toEqual([true, false]);
    await expect(page.locator('[data-object-point="remove"]')).toHaveCount(1);
    await maskCard(page, "Object").getByRole("radio", { name: "Box" }).click();
    await expect(maskCard(page, "Object").getByText("Drag a box around the object.")).toBeVisible();
    const cell = await cellBox(page);
    await page.mouse.move(cell.x + cell.width * 0.25, cell.y + cell.height * 0.45);
    await page.mouse.down();
    await page.mouse.move(cell.x + cell.width * 0.5, cell.y + cell.height * 0.7, { steps: 6 });
    await expect(page.locator("[data-object-band]")).toHaveCount(1);
    await page.mouse.move(cell.x + cell.width * 0.75, cell.y + cell.height * 0.9, { steps: 6 });
    await page.mouse.up();
    await expect.poll(async () => (await objectRows(page))[0]?.box).not.toBeNull();
    const box = (await objectRows(page))[0]!.box!;
    expect(box[0]).toBeCloseTo(0.25, 1);
    expect(box[3]).toBeCloseTo(0.9, 1);
    await expect(rowButton(page, "Object mask, Box and 2 points")).toBeVisible();
    await expect(page.locator("[data-object-box]")).toHaveCount(1);
    // One undo step each, newest first.
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await objectRows(page))[0]?.box).toBeNull();
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await objectRows(page))[0]?.points.length).toBe(1);
  });

  test("Esc ends the tool and leaves the points drawn, and the card brings it back", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Object");
    await clickPhoto(page, [0.5, 0.7]);
    await expect(page.locator("[data-object-point]")).toHaveCount(1);
    await blur(page);
    await page.keyboard.press("Escape");
    await expect(objectLayer(page)).not.toHaveAttribute("data-tool", /.*/);
    await expect(page.locator("[data-object-point]")).toHaveCount(1);
    await maskCard(page, "Object").getByRole("radio", { name: "Add point" }).click();
    await expect(objectLayer(page)).toHaveAttribute("data-tool", "add");
  });

  test("Place points (docs/28 A-11): Return on the Object row, the arrows, Return and ⇧Return place two points that are the two mouse clicks' points; Backspace and Esc", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Object");
    await expect(objectLayer(page)).toBeVisible();
    // The row says how, to a screen reader.
    await expect(rowButton(page, /^Object mask/)).toHaveAccessibleDescription("Press Return to place points with the keyboard.");
    await rowButton(page, /^Object mask/).focus();
    await page.keyboard.press("Enter");
    const place = page.locator(".ym-pick-layer");
    await expect(place).toHaveAttribute("data-mode", "points");
    await expect(place).toBeFocused();
    await expect(place).toHaveAccessibleName(/^Place points\. /);
    const calls = await recordCalls(page, "updateSettings");
    // The middle of the hill, (0.5, 0.7): ⇧ twice. Return adds a point there.
    await page.keyboard.press("Shift+ArrowDown");
    await page.keyboard.press("Shift+ArrowDown");
    await page.keyboard.press("Enter");
    await expect.poll(async () => (await objectRows(page))[0]?.points).toEqual([[0.5, 0.7, true]]);
    await expect(page.getByText("Point added. 1 point.")).toBeAttached();
    // A little to the right and down, and a part taken away with ⇧Return.
    for (const key of ["ArrowRight", "ArrowRight", "ArrowRight", "ArrowRight", "ArrowRight", "ArrowDown", "ArrowDown"]) await page.keyboard.press(key);
    await page.keyboard.press("Shift+Enter");
    await expect.poll(async () => (await objectRows(page))[0]?.points).toEqual([[0.5, 0.7, true], [0.55, 0.72, false]]);
    await expect(page.locator("[data-object-point]")).toHaveCount(2);
    const byKeys = (await objectRows(page))[0]!.points;
    expect(await calls()).toHaveLength(2);
    // Backspace takes the last point out, and the first.
    await page.keyboard.press("Backspace");
    await expect.poll(async () => (await objectRows(page))[0]?.points).toEqual([[0.5, 0.7, true]]);
    await page.keyboard.press("Backspace");
    await expect.poll(async () => (await objectRows(page))[0]?.points).toEqual([]);
    // Esc ends it and the focus goes back to the row; the Object tool is still on.
    await page.keyboard.press("Escape");
    await expect(place).toHaveCount(0);
    await expect(rowButton(page, /^Object mask/)).toBeFocused();
    await expect(objectLayer(page)).toHaveAttribute("data-tool", "add");
    // The mouse, at the same places, gives the same points.
    await clickPhoto(page, [0.5, 0.7]);
    await expect.poll(async () => (await objectRows(page))[0]?.points.length).toBe(1);
    await clickPhoto(page, [0.55, 0.72], true);
    await expect.poll(async () => (await objectRows(page))[0]?.points.length).toBe(2);
    expect((await objectRows(page))[0]!.points).toEqual(byKeys);
  });

  test("Place points on a zoomed photo: the arrows are the crosshair's while it is on (the photo does not pan), and the pan keys are the photo's again when it ends", async ({ page }) => {
    await warmth(page);
    await addMask(page, "Object");
    const offset = (): Promise<{ x: number; y: number }> =>
      page.evaluate(() => (window as unknown as { __yellowmoon: { mockApi: { viewport: { offset: { x: number; y: number } } } } }).__yellowmoon.mockApi.viewport.offset);
    await page.getByRole("button", { name: /^(Fit \d+%|\d+%)$/ }).click();
    await page.getByRole("menu").getByText("200%", { exact: true }).click();
    const zoomed = page.getByRole("img", { name: "Photo, zoomed. The arrow keys move it." });
    await expect(zoomed).toHaveAttribute("tabindex", "0");
    await rowButton(page, /^Object mask/).focus();
    await page.keyboard.press("Enter");
    const place = page.locator(".ym-pick-layer");
    await expect(place).toBeFocused();
    // The photo is not a stop while the layer has the cell, and the arrows move the crosshair, not the photo.
    await expect(page.locator(".ym-photo-cell__view")).not.toHaveAttribute("tabindex", /.*/);
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("Shift+ArrowDown");
    expect(await offset()).toEqual({ x: 0, y: 0 });
    await expect(place.locator("svg.ym-pick-layer__mark")).toHaveAttribute("style", /left: 51%; top: 60%/);
    await page.keyboard.press("Escape");
    await expect(place).toHaveCount(0);
    await expect(zoomed).toHaveAttribute("tabindex", "0");
    await zoomed.focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(offset).not.toEqual({ x: 0, y: 0 });
  });

  test("?mockMaskFind=none finds nothing anywhere, and ?mockMasks=noai leaves Object out of the Add mask menu", async ({ page }) => {
    await warmth(page, { mockMaskFind: "none" });
    await addMask(page, "Object");
    await clickPhoto(page, [0.5, 0.7]);
    await expect(page.getByRole("alert")).toContainText("Nothing found there.");
    await warmth(page, { mockMasks: "noai" });
    await page.getByRole("button", { name: "Add mask" }).click();
    await expect(page.getByRole("menuitem", { name: /^Object/ })).toHaveCount(0);
    await expect(page.getByRole("menuitem", { name: /^Background/ })).toBeVisible();
  });
});

// ---- Subject and People (docs/23 §3.10, §4.5 and §7.6, task N-19) -----------------------------------------------------------------------------

const visionRows = async (page: Page) => (await rowsOf(page, 4)).filter((r) => r.kind === "subject" || r.kind === "people");
/** A Subject or People row as a file or a paste would bring it. */
const findRow = (kind: "subject" | "people", extra: Record<string, unknown> = {}) => ({ kind, mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, ...extra });

test.describe("Subject and People", () => {
  test("adding a Subject row says 'Finding…' while the core works, then 'Foreground'; the SUBJECT card has the four raster sliders; the Mask only view draws the far hill; undo takes it all back", async ({ page }) => {
    // At the speed of a real run, so that "Finding…" is on the screen for the 500 ms the mock's Vision takes.
    await warmth(page, { mockSpeed: 1 });
    await addMask(page, "Subject");
    await expect(rowButton(page, "Subject mask, Finding…")).toBeVisible();
    await expect(rowList(page).getByText("Finding…")).toHaveAttribute("data-finding", "");
    await expect(rowButton(page, "Subject mask, Foreground")).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByText("Finding…")).toHaveCount(0);
    // The card: nothing of its own, the sliders every raster card has.
    await expect(maskCard(page, "Subject")).toBeVisible();
    await expect(maskCard(page, "Subject").getByRole("slider")).toHaveCount(4);
    for (const name of ["Refine edges", "Feather", "Grow", "Opacity"]) await expect(maskCard(page, "Subject").getByRole("slider", { name })).toBeVisible();
    await expect(page.getByText("MASK · SUBJECT")).toBeVisible();
    expect(await visionRows(page)).toEqual([findRow("subject")]);
    // The view: a viewer never waits, so the mask is drawn once the core has made it; the mock draws the far hill.
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(async () => await picture(page)).toContain('data-vision-row="subject"');
    // A slider writes its field and the view follows the Feather.
    await setValue(page, "Feather", "30");
    await expect.poll(async () => (await visionRows(page))[0]?.feather).toBe(0.3);
    // Undo: the Feather, then the row; the picture follows.
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await visionRows(page))[0]?.feather).toBe(0);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await visionRows(page)).length).toBe(0);
    await expect.poll(async () => await picture(page)).not.toContain("data-vision-row");
    await page.keyboard.press("Meta+Shift+z");
    await expect(rowButton(page, /^Subject mask/)).toBeVisible();
  });

  test("a People row finds a figure in a photo that has one and says 'Nothing found' in one that has none", async ({ page }) => {
    // DSC04204 (the mockup's photo) has a person on its near ground; DSC04203, the photo before it, has none.
    await warmth(page, {}, { "4": { kind: "corrector", name: "Warmth", pos: [3, 0], masks: { invert: false, items: [findRow("subject"), findRow("people")] } } });
    await expect(rowButton(page, "People mask, People")).toBeVisible();
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(async () => await picture(page)).toContain('data-vision-row="people"');
    expect(await picture(page)).toContain('data-vision-row="subject"');
    await page.getByRole("button", { name: "Copy settings" }).click();
    await page.getByRole("button", { name: "Previous photo" }).click();
    await page.getByRole("button", { name: "Paste", exact: true }).click();
    await expect(card(page, 4)).toBeVisible();
    await card(page, 4).click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    // The rows came as rows, and the masks are worked out again from this photo's picture: Finding…, then its own words.
    await expect(rowButton(page, /^Subject mask, (Finding…|Foreground)/)).toBeVisible();
    await expect(rowButton(page, "People mask, Nothing found")).toBeVisible();
    await expect(rowButton(page, "Subject mask, Foreground")).toBeVisible();
    expect((await visionRows(page)).map((r) => r.kind)).toEqual(["subject", "people"]);
    // Nothing found selects nothing: People is not in the Mask only picture of this photo, Subject is.
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(async () => await picture(page)).toContain('data-vision-row="subject"');
    expect(await picture(page)).not.toContain('data-vision-row="people"');
  });

  test("Copy settings and Paste carry the rows to the other photo, which finds its own masks, and the paste is one undo step", async ({ page }) => {
    await warmth(page, {}, { "4": { kind: "corrector", name: "Warmth", pos: [3, 0], masks: { invert: false, items: [findRow("subject", { feather: 0.2 })] } } });
    await expect(rowButton(page, "Subject mask, Foreground")).toBeVisible();
    await page.getByRole("button", { name: "Copy settings" }).click();
    await page.getByRole("button", { name: "Next photo" }).click();
    await page.getByRole("button", { name: "Paste", exact: true }).click();
    await card(page, 4).click();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(rowButton(page, /^Subject mask/)).toBeVisible();
    expect(await visionRows(page)).toEqual([findRow("subject", { feather: 0.2 })]);
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect(card(page, 4)).toHaveCount(0);
  });

  test("on a Mac that cannot make them (?mockMasks=nomac) the menu hides both; a row that a file brought says the sentence, selects nothing and has no card", async ({ page }) => {
    await warmth(page, { mockMasks: "nomac" }, { "4": { kind: "corrector", name: "Warmth", pos: [3, 0], masks: { invert: false, items: [findRow("subject"), findRow("people")] } } });
    await page.getByRole("button", { name: "Add mask" }).click();
    const menu = page.getByRole("menu", { name: "Add mask" });
    await expect(menu.getByRole("menuitem", { name: /^Object/ })).toBeVisible();
    await expect(menu.getByRole("menuitem", { name: /^(Subject|People)/ })).toHaveCount(0);
    await page.keyboard.press("Escape");
    await expect(menu).toHaveCount(0);
    await expect(rowButton(page, "Subject mask, Subject masks need a Mac that supports them")).toBeVisible();
    await expect(rowButton(page, "People mask, People masks need a Mac that supports them")).toBeVisible();
    await expect(rowList(page).getByText("Subject masks need a Mac that supports them")).toBeVisible();
    // Nothing is being found, and no card shows for the row.
    await expect(page.getByText("Finding…")).toHaveCount(0);
    await rowButton(page, /^Subject mask/).click();
    await expect(maskCard(page, "Subject")).toHaveCount(0);
    await page.getByRole("radio", { name: "Mask only" }).click();
    await expect.poll(async () => await picture(page)).toContain("data-mask-view");
    expect(await picture(page)).not.toContain("data-vision-row");
    // The rows are still in the settings: they select nothing, they are not taken out.
    expect((await visionRows(page)).map((r) => r.kind)).toEqual(["subject", "people"]);
  });
});
