import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, recordCalls } from "./helpers/app";
import { GRADED_NODES } from "./helpers/states";

// Show mask (docs/23 §5.8, §7.6 View card, §7.7 M; task N-11), in a real browser on the mock API: the toolbar's toggle and M, the View
// card's two segments, the Input and the Output (no mask), leaving the node view and Crop, and what the mock's own picture shows. The core
// draws the real mask and overlay (`crates/ym-core/tests/mask_view.rs`); here the page asks for it with `viewport_set_mode(mode, mask_view)`
// and the mock's viewport records what it was asked.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

/** A Radial at the middle of the photo, as the screenshots have it on Main. */
const RADIAL = { kind: "radial", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], radii: [0.3, 0.2], angle_deg: 0, feather: 0.4 };

type MaskView = { node: number; style: "overlay" | "mask_only" } | null;
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
    await page.getByRole("button", { name: "Add mask" }).focus();
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
