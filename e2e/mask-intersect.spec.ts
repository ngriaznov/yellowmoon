import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, recordCalls } from "./helpers/app";
import { compareScreen } from "./helpers/compare";
import { GRADED_NODES } from "./helpers/states";

// Mask Intersect (docs/23 §3.3, §4.2, §7.6; task MI-06), in a real browser on the mock API: the combine control on the Masks card, Add and
// Intersect, the line under the rows, Show mask (Mask only) drawing only the overlap, Reset, and the screenshot of the card.
// The core draws the real mask (`crates/ym-core/tests/mask_view.rs`); the mock's own picture draws the product of the rows, which is what
// the pixel checks below read.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

const radial = (cx: number, opacity = 1) => ({ kind: "radial", mode: "add", invert: false, enabled: true, opacity, center: [cx, 0.5], radii: [0.25, 0.25], angle_deg: 0, feather: 0.1 });
/** Two ovals of a quarter of the long edge: the left one covers x 0.13 to 0.63, the right one 0.37 to 0.87, so they overlap from 0.37 to 0.63. */
const TWO = [radial(0.38), radial(0.62)];
const LINEAR = { kind: "linear", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], angle_deg: 90, feather: 0.2 };
const COLOUR_RANGE = { kind: "color_range", mode: "add", invert: false, enabled: true, opacity: 1, center: [0, 0], width: 0.2, softness: 0.5 };

type Box = { x: number; y: number; width: number; height: number };
const masksCard = (page: Page) => page.getByRole("region", { name: "Masks" });
const control = (page: Page) => masksCard(page).getByRole("radiogroup", { name: "How the masks combine" });
const card = (page: Page, id: number | string) => page.locator(`[data-node-card="${id}"]`);

/** The mockup's photo with `rows` on Main, the node view open on Main's Mask tab. */
async function nodeView(page: Page, rows: readonly object[], more: object = {}): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, {
    initialSettings: { DSC04204: { nodes: { "1": { masks: { invert: false, items: rows, ...more } }, ...GRADED_NODES.nodes }, links: GRADED_NODES.links } } as never,
  });
  await openPhotos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await expect(card(page, 4)).toBeVisible();
  await card(page, 1).click();
  await expect(card(page, 1)).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("tab", { name: "Mask", exact: true }).click();
  await expect(masksCard(page)).toBeVisible();
}

/** The grey of the mock's picture at fractions of its width and height (Mask only draws the mask as grey over black), 0 to 255. */
async function greys(page: Page, at: ReadonlyArray<readonly [number, number]>): Promise<number[]> {
  const src = await page.locator(".ym-photo-cell__image").getAttribute("src");
  return await page.evaluate(
    async ({ src, at }) => {
      const img = new Image();
      img.src = src ?? "";
      await img.decode();
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth || 600;
      canvas.height = img.naturalHeight || 400;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      return at.map(([x, y]) => ctx.getImageData(Math.round(x! * (canvas.width - 1)), Math.round(y! * (canvas.height - 1)), 1, 1).data[0] ?? -1);
    },
    { src, at: at.map((p) => [...p]) },
  );
}

const PLACES = { overlap: [0.5, 0.5], leftOnly: [0.2, 0.5], rightOnly: [0.8, 0.5], outside: [0.02, 0.05] } as const;
const sample = (page: Page) => greys(page, [PLACES.overlap, PLACES.leftOnly, PLACES.rightOnly, PLACES.outside]);

async function maskOnly(page: Page): Promise<void> {
  await page.getByRole("radio", { name: "Mask only", exact: true }).click();
  await expect(page.getByRole("group", { name: "Graph tools" }).getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-pressed", "true");
}

test.describe("the combine control", () => {
  test("is Add | Intersect between Add mask and Invert, with the Add line, and Intersect swaps the line", async ({ page }) => {
    await nodeView(page, TWO);
    await expect(control(page).getByRole("radio")).toHaveText(["Add", "Intersect"]);
    await expect(control(page).getByRole("radio", { name: "Add" })).toHaveAttribute("aria-checked", "true");
    await expect(masksCard(page).getByText("Masks add up. Subtract cuts a mask out.")).toBeVisible();
    const boxes = await Promise.all([masksCard(page).getByRole("button", { name: "Add mask" }), masksCard(page).getByRole("button", { name: "Invert all masks", exact: true }), masksCard(page).getByRole("button", { name: "Show the mask of all rows", exact: true }), masksCard(page).getByRole("button", { name: "Reset", exact: true }), control(page), masksCard(page).getByRole("list", { name: "Masks of this node" })].map((l) => l.boundingBox()));
    const [add, invert, eye, reset, ctrl, rows] = boxes as [Box, Box, Box, Box, Box, Box];
    // The header keeps Add mask, Invert, the eye and Reset on one line, inside the panel at 1440 x 900; the control is the first line of the card, above the rows.
    expect(add.x + add.width).toBeLessThanOrEqual(invert.x + 1);
    expect(invert.x + invert.width).toBeLessThanOrEqual(eye.x + 1);
    expect(eye.x + eye.width).toBeLessThanOrEqual(reset.x + 1);
    // Invert and the eye are the rows' 32 px icons, and have no text.
    expect([invert.width, invert.height, eye.width, eye.height]).toEqual([32, 32, 32, 32]);
    expect(Math.abs(add.y - reset.y)).toBeLessThan(10);
    expect(ctrl.y).toBeGreaterThan(add.y + add.height - 1);
    expect(ctrl.y + ctrl.height).toBeLessThanOrEqual(rows.y + 1);
    const panel = (await masksCard(page).boundingBox())!;
    expect(reset.x + reset.width).toBeLessThanOrEqual(panel.x + panel.width + 1);
    expect(ctrl.x + ctrl.width).toBeLessThanOrEqual(panel.x + panel.width + 1);
    await control(page).getByRole("radio", { name: "Intersect" }).click();
    await expect(masksCard(page).getByText("Only where every mask overlaps counts. Subtract cuts a mask out.")).toBeVisible();
    await expect(masksCard(page).getByText("Masks add up.")).toHaveCount(0);
  });

  test("shows its tooltip", async ({ page }) => {
    await nodeView(page, TWO);
    await control(page).hover();
    await expect(page.getByRole("tooltip")).toHaveText("How the masks combine: Add joins them, Intersect keeps only where they all overlap.");
  });

  test("a click is one masks.combine patch, and Reset puts Add back", async ({ page }) => {
    await nodeView(page, TWO);
    const calls = await recordCalls(page, "updateSettings");
    await control(page).getByRole("radio", { name: "Intersect" }).click();
    await expect(control(page).getByRole("radio", { name: "Intersect" })).toHaveAttribute("aria-checked", "true");
    expect((await calls()).map((c) => c.args[1])).toEqual([{ nodes: { "1": { masks: { combine: "intersect" } } } }]);
    await masksCard(page).getByRole("button", { name: "Reset", exact: true }).click();
    await expect(control(page).getByRole("radio", { name: "Add" })).toHaveAttribute("aria-checked", "true");
    await expect(page.getByRole("listitem")).toHaveCount(0, { timeout: 2000 }).catch(() => undefined);
    expect((await calls()).at(-1)?.args[1]).toEqual({ nodes: { "1": { masks: { items: [], invert: false, combine: "add" } } } });
  });
});

test.describe("Show mask follows the mode", () => {
  test("two overlapping shapes: Add shows both, Intersect shows only the overlap, and back to Add shows both again", async ({ page }) => {
    await nodeView(page, TWO);
    await maskOnly(page);
    const add = await expect.poll(async () => (await sample(page))[1]!).toBeGreaterThan(150).then(() => sample(page));
    // Add: the overlap and each shape alone are light, outside both is dark.
    expect(add[0]).toBeGreaterThan(200);
    expect(add[1]).toBeGreaterThan(150);
    expect(add[2]).toBeGreaterThan(150);
    expect(add[3]).toBeLessThan(40);
    await control(page).getByRole("radio", { name: "Intersect" }).click();
    await expect.poll(async () => (await sample(page))[1]!).toBeLessThan(40);
    const intersect = await sample(page);
    // Intersect: only the overlap is light.
    expect(intersect[0]).toBeGreaterThan(200);
    expect(intersect[1]).toBeLessThan(40);
    expect(intersect[2]).toBeLessThan(40);
    expect(intersect[3]).toBeLessThan(40);
    await control(page).getByRole("radio", { name: "Add" }).click();
    await expect.poll(async () => (await sample(page))[1]!).toBeGreaterThan(150);
  });

  test("the header eye shows the whole mask: in Intersect and Mask only only the overlap is light, a row's eye takes over, a second click ends it", async ({ page }) => {
    await nodeView(page, TWO, { combine: "intersect" });
    const eye = masksCard(page).getByRole("button", { name: "Show the mask of all rows", exact: true });
    const toolbar = page.getByRole("group", { name: "Graph tools" }).getByRole("button", { name: "Show mask" });
    await expect(eye).toHaveAttribute("aria-pressed", "false");
    await page.getByRole("radio", { name: "Mask only", exact: true }).click();
    await expect(eye).toHaveAttribute("aria-pressed", "true");
    await expect.poll(async () => (await sample(page))[1]!).toBeLessThan(40);
    const lit = await sample(page);
    expect(lit[0]).toBeGreaterThan(200);
    expect(lit[2]).toBeLessThan(40);
    expect(lit[3]).toBeLessThan(40);
    // A row's eye takes over: it shows that row alone, and the header eye and the toolbar let go.
    await masksCard(page).getByRole("button", { name: "Show this mask" }).first().click();
    await expect(eye).toHaveAttribute("aria-pressed", "false");
    await expect(toolbar).toHaveAttribute("aria-pressed", "false");
    await expect.poll(async () => (await sample(page))[1]!).toBeGreaterThan(150);
    // The header eye takes it back: the overlap alone, and the toolbar is pressed with it.
    await eye.click();
    await expect(eye).toHaveAttribute("aria-pressed", "true");
    await expect(toolbar).toHaveAttribute("aria-pressed", "true");
    await expect.poll(async () => (await sample(page))[1]!).toBeLessThan(40);
    // A second click ends the view.
    await eye.click();
    await expect(eye).toHaveAttribute("aria-pressed", "false");
    await expect(toolbar).toHaveAttribute("aria-pressed", "false");
  });

  test("a Subtract row cuts out of the overlap, and Opacity 0 on a row leaves the others as they are", async ({ page }) => {
    const cut = { ...radial(0.5), mode: "subtract", radii: [0.05, 0.05] };
    await nodeView(page, [...TWO, cut], { combine: "intersect" });
    await maskOnly(page);
    // The middle of the overlap is cut out by the small oval; the overlap beside it stays.
    await expect.poll(async () => (await greys(page, [[0.5, 0.5]]))[0]!).toBeLessThan(40);
    expect((await greys(page, [[0.42, 0.5]]))[0]).toBeGreaterThan(150);
    // The same rows with the first at Opacity 0: the second oval alone decides, so the right side is light now.
    await nodeView(page, [radial(0.38, 0), radial(0.62)], { combine: "intersect" });
    await maskOnly(page);
    await expect.poll(async () => (await greys(page, [[0.8, 0.5]]))[0]!).toBeGreaterThan(150);
  });

  test("an Object row with no point empties the mask; a Colour range multiplies in", async ({ page }) => {
    const object = { kind: "object", mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, points: [], box: null };
    await nodeView(page, [...TWO, object], { combine: "intersect" });
    await maskOnly(page);
    await expect.poll(async () => (await sample(page))[0]!).toBeLessThan(40);
    expect((await sample(page)).every((g) => g < 40)).toBe(true);
    await nodeView(page, [...TWO, COLOUR_RANGE], { combine: "intersect" });
    await maskOnly(page);
    // The range keeps only its part of the overlap; whatever the mock's scene gives there, the outside of the ovals stays dark.
    await expect.poll(async () => (await sample(page))[3]!).toBeLessThan(40);
  });
});

test.describe("the reference picture", () => {
  test("the Masks card in Intersect with three rows (photos-masks-intersect.png)", async ({ page }) => {
    await nodeView(page, [radial(0.38), radial(0.62), LINEAR], { combine: "intersect" });
    await page.getByRole("radio", { name: "Mask only", exact: true }).click();
    await compareScreen(page, "photos-masks-intersect");
  });
});
