import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { ResolvedSettings } from "../src/api/types";
import { nodeOf } from "../src/state/settingsGraph";
import type { CorrectorItem } from "../src/state/settingsGraph";
import { configureMock, openApp, openPhotos, recordCalls, setValue, watchErrors } from "./helpers/app";
import { GRADED_NODES } from "./helpers/states";

// The panel of the selected node in the node view (docs/23 §7.5, docs/11 §Node view, task N-09), in a real browser on the mock API: the
// compact head, the tabs of each kind of node, the trim cards, the thumbnails that follow a trim, W, undo across nodes, and Copy settings
// and Paste. The graph is the one of the mockups: Main, Mountain (2 masks), Sky (1 mask) and Warmth.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

type Mock = { getSettings(id: number): Promise<ResolvedSettings> };
const mock = <T>(page: Page, run: (api: Mock, ...args: never[]) => T | Promise<T>): Promise<T> =>
  page.evaluate(`(${run.toString()})(window.__yellowmoon.mockApi)`) as Promise<T>;
const settingsOf = (page: Page): Promise<ResolvedSettings> => mock(page, (api) => api.getSettings(4));
const nodeSettings = async (page: Page, id: number): Promise<CorrectorItem> => nodeOf((await settingsOf(page)).settings, id) as CorrectorItem;

/** The graded graph on the mockup's photo, on the simple view's tab `tab`, with the node view opened by the Nodes button. */
async function nodeView(page: Page, tab = "Light"): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, { initialSettings: { DSC04204: { nodes: GRADED_NODES.nodes, links: GRADED_NODES.links } } as never });
  await openPhotos(page);
  await page.getByRole("tab", { name: tab, exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await expect(page.locator('[data-node-card="4"]')).toBeVisible();
}

const panel = (page: Page) => page.getByRole("complementary", { name: "Tools" });
const head = (page: Page) => panel(page).getByRole("group", { name: "Node", exact: true });
const tabs = (page: Page) => panel(page).getByRole("tab");
const card = (page: Page, id: number | string) => page.locator(`[data-node-card="${id}"]`);
const cardPicture = (page: Page, id: number) => card(page, id).locator("img").getAttribute("src");

async function select(page: Page, id: number | string): Promise<void> {
  await card(page, id).click();
  await expect(card(page, id)).toHaveAttribute("aria-pressed", "true");
}

async function tabNames(page: Page): Promise<string[]> {
  return (await tabs(page).allTextContents()).map((t) => t.trim());
}

test.describe("the head", () => {
  test("is one line of 52 px over a tab row of 46 px: the number chip, the name, the quiet line and three round icons; no title block", async ({ page }) => {
    await nodeView(page);
    await select(page, 2);
    const h = head(page);
    await expect(h.getByRole("heading", { level: 2, name: "Mountain" })).toBeVisible();
    await expect(h.locator(".ym-nodehead__chip")).toHaveText("02");
    await expect(h.locator(".ym-nodehead__line")).toHaveText("Corrector · 2 masks");
    const box = (await h.boundingBox())!;
    expect(Math.round(box.height)).toBe(52);
    const row = (await panel(page).locator('[data-slot="tabs"]').first().boundingBox())!;
    expect(Math.round(row.height)).toBe(46);
    expect(Math.abs(row.y - (box.y + box.height))).toBeLessThan(1.5);
    // The name is 20 px bold.
    const name = await h.getByRole("heading", { level: 2 }).evaluate((el) => ({ size: getComputedStyle(el).fontSize, weight: getComputedStyle(el).fontWeight }));
    expect(name).toEqual({ size: "20px", weight: "800" });
    // Three round icon buttons (28 px) at the right with their tooltips.
    const icons = h.getByRole("group", { name: "Node actions" }).getByRole("button");
    await expect(icons).toHaveCount(3);
    for (const [i, label] of ["Bypass this node", "Reset this node", "More"].entries()) {
      await expect(icons.nth(i)).toHaveAccessibleName(label);
      const b = (await icons.nth(i).boundingBox())!;
      expect(Math.round(b.width)).toBe(28);
      expect(Math.round(b.height)).toBe(28);
    }
    await icons.nth(1).hover();
    await expect(page.getByRole("tooltip")).toHaveText("Reset this node");
    await expect(panel(page).locator(".ym-title-block")).toHaveCount(0);
  });

  test("says what each node is, and each kind has its own tabs: the Input's five, a corrector's four and the Output's two", async ({ page }) => {
    await nodeView(page);
    expect(await tabNames(page)).toEqual(["Light", "Colour", "Curve", "Mask"]);
    await select(page, "input");
    await expect(head(page).getByRole("heading", { level: 2, name: "Input" })).toBeVisible();
    await expect(head(page).locator(".ym-nodehead__line")).toHaveText("Turns the camera's data into the scene.");
    expect(await tabNames(page)).toEqual(["RAW", "Lens", "Light", "Colour", "Detail"]);
    await expect(head(page).getByRole("group", { name: "Node actions" }).getByRole("button")).toHaveCount(2);
    await select(page, "output");
    await expect(head(page).getByRole("heading", { level: 2, name: "Output" })).toBeVisible();
    await expect(head(page).locator(".ym-nodehead__line")).toHaveText("Cine rendering, look and export");
    expect(await tabNames(page)).toEqual(["Look", "Detail"]);
    await expect(head(page).getByRole("button", { name: "Reset output" })).toBeVisible();
    await select(page, 3);
    await expect(head(page).locator(".ym-nodehead__line")).toHaveText("Corrector · Linear");
    expect(await tabNames(page)).toEqual(["Light", "Colour", "Curve", "Mask"]);
    await select(page, 4);
    await expect(head(page).locator(".ym-nodehead__line")).toHaveText("Corrector · whole photo");
    await select(page, 1);
    await expect(head(page).getByRole("heading", { level: 2, name: "Main" })).toBeVisible();
    await expect(head(page).locator(".ym-nodehead__chip")).toHaveText("01");
  });

  test("the node view opens on the node and the tab of the simple view's tab", async ({ page }) => {
    for (const [simple, node, tab] of [["Look", "Output", "Look"], ["Detail", "Input", "Detail"], ["Curve", "Main", "Curve"]] as const) {
      await nodeView(page, simple);
      await expect(head(page).getByRole("heading", { level: 2 })).toHaveText(node);
      await expect(panel(page).getByRole("tab", { selected: true })).toHaveText(tab);
    }
  });

  test("Bypass, Reset and More work on the selected node, and a double click renames it in the head", async ({ page }) => {
    await nodeView(page);
    await select(page, 3);
    await head(page).getByRole("button", { name: "Bypass this node" }).click();
    await expect(head(page).getByRole("button", { name: "Bypass this node" })).toHaveAttribute("aria-pressed", "true");
    await expect(head(page).locator(".ym-nodehead__line")).toHaveText("Corrector · bypassed");
    await expect(card(page, 3)).toHaveAccessibleName(/bypassed/);
    await page.keyboard.press("b");
    await expect(head(page).getByRole("button", { name: "Bypass this node" })).toHaveAttribute("aria-pressed", "false");
    // Reset keeps the masks, the name and the links.
    await select(page, 2);
    await setValue(page, "Exposure trim", "1.2");
    await expect((await nodeSettings(page, 2)).trim.exposure_ev).toBe(1.2);
    await head(page).getByRole("button", { name: "Reset this node" }).click();
    await expect.poll(async () => (await nodeSettings(page, 2)).trim.exposure_ev).toBe(0);
    const reset = await nodeSettings(page, 2);
    expect(reset.masks.items).toHaveLength(2);
    expect(reset.name).toBe("Mountain");
    await expect(head(page).locator(".ym-nodehead__line")).toHaveText("Corrector · 2 masks");
    // More opens down the panel, not up over the photo.
    // Read before it opens: while a menu is open the rest of the page is hidden from the accessibility tree.
    const more = (await head(page).getByRole("button", { name: "More" }).boundingBox())!;
    await head(page).getByRole("button", { name: "More" }).click();
    const menu = page.getByRole("menu", { name: "More" });
    await expect(menu).toBeVisible();
    const m = (await menu.boundingBox())!;
    expect(m.y).toBeGreaterThan(more.y + more.height);
    expect(m.x + m.width).toBeLessThanOrEqual(more.x + more.width + 1);
    await menu.getByRole("menuitem", { name: "Rename" }).click();
    const field = head(page).getByRole("textbox", { name: "Node name" });
    await expect(field).toBeFocused();
    await field.fill("Hills");
    await field.press("Enter");
    await expect(head(page).getByRole("heading", { level: 2, name: "Hills" })).toBeVisible();
    await expect(card(page, 2)).toHaveAccessibleName(/^Node 2, Hills/);
    await head(page).getByRole("heading", { level: 2, name: "Hills" }).dblclick();
    await expect(head(page).getByRole("textbox", { name: "Node name" })).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(head(page).getByRole("heading", { level: 2, name: "Hills" })).toBeVisible();
    // The toolbar has no More any more.
    await expect(page.getByRole("group", { name: "Graph tools" }).getByRole("button", { name: "More" })).toHaveCount(0);
  });
});

test.describe("the tabs", () => {
  test("⌘1 to ⌘5 switch the selected node's tabs, and the Input remembers its own", async ({ page }) => {
    await nodeView(page);
    await page.keyboard.press("Meta+4");
    await expect(panel(page).getByRole("tab", { selected: true })).toHaveText("Mask");
    await page.keyboard.press("Meta+5");
    await expect(panel(page).getByRole("tab", { selected: true })).toHaveText("Mask");
    await select(page, "input");
    await page.keyboard.press("Meta+5");
    await expect(panel(page).getByRole("tab", { selected: true })).toHaveText("Detail");
    await select(page, "output");
    await page.keyboard.press("Meta+2");
    await expect(panel(page).getByRole("tab", { selected: true })).toHaveText("Detail");
    await select(page, "input");
    await expect(panel(page).getByRole("tab", { selected: true })).toHaveText("Detail");
    await select(page, 2);
    await expect(panel(page).getByRole("tab", { selected: true })).toHaveText("Mask");
  });

  test("the Input's cards, the Output's cards and a corrector's cards are the simple view's, written to their node", async ({ page }) => {
    await nodeView(page);
    const titles = () => panel(page).getByRole("heading", { level: 3 }).allTextContents();
    await select(page, "input");
    expect(await titles()).toEqual(["Camera", "Decode", "Range"]);
    await panel(page).getByRole("tab", { name: "Colour" }).click();
    expect(await titles()).toEqual(["White balance"]);
    await expect(panel(page).getByRole("button", { name: "Pick white" })).toBeVisible();
    await expect(panel(page).getByRole("textbox", { name: "Kelvin value" })).toBeVisible();
    await panel(page).getByRole("tab", { name: "Detail" }).click();
    expect(await titles()).toEqual(["Sharpening", "Noise reduction"]);
    await expect(panel(page).getByRole("slider", { name: "Output" })).toHaveCount(0);
    await select(page, "output");
    await expect(panel(page).getByRole("slider", { name: "Intensity" })).toBeVisible();
    expect(await titles()).toEqual(["Look"]);
    await panel(page).getByRole("tab", { name: "Detail" }).click();
    expect(await titles()).toEqual(["Sharpening"]);
    await expect(panel(page).getByRole("slider", { name: "Output" })).toBeVisible();
    await select(page, 3);
    await panel(page).getByRole("tab", { name: "Colour" }).click();
    expect(await titles()).toEqual(["Temperature and tint", "Colour balance", "Colour warp"]);
    await expect(panel(page).getByRole("button", { name: "Pick white" })).toHaveCount(0);
    await expect(panel(page).getByRole("button", { name: "Pick colour" })).toBeVisible();
    await panel(page).getByRole("tab", { name: "Mask" }).click();
    expect(await titles()).toEqual(["Masks", "View"]);
    await expect(panel(page).getByRole("list", { name: "Masks of this node" }).getByRole("listitem")).toHaveCount(1);
    // The Mask tab is built (N-12): Add mask is a menu button now.
    await expect(panel(page).getByRole("button", { name: "Add mask" })).not.toHaveAttribute("aria-disabled", "true");
  });
});

test.describe("the trim cards", () => {
  test("moving Mountain's Exposure trim changes its card's picture and the cards after it, not Main's, and one undo takes it back", async ({ page }) => {
    await nodeView(page);
    await select(page, 2);
    // The pictures of the cards are the core's, after each node: wait until each card has its own.
    await expect.poll(async () => new Set(await Promise.all([1, 2, 3, 4].map((id) => cardPicture(page, id)))).size).toBeGreaterThan(1);
    await page.waitForTimeout(300);
    const before = await Promise.all([1, 2, 3, 4].map((id) => cardPicture(page, id)));
    const updates = await recordCalls(page, "updateSettings");
    await setValue(page, "Exposure trim", "1.5");
    await expect.poll(async () => (await nodeSettings(page, 2)).trim.exposure_ev).toBe(1.5);
    // One patch for the typed value, naming the node's trim and nothing else.
    expect((await updates()).map((c) => c.args[1])).toEqual([{ nodes: { "2": { trim: { exposure_ev: 1.5 } } } }]);
    await expect.poll(async () => cardPicture(page, 2)).not.toBe(before[1]);
    await expect.poll(async () => cardPicture(page, 4)).not.toBe(before[3]);
    expect(await cardPicture(page, 1)).toBe(before[0]);
    // A slider drag is frames and one final patch.
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await nodeSettings(page, 2)).trim.exposure_ev).toBe(0.35);
    await expect(panel(page).getByRole("textbox", { name: "Exposure trim value" })).toHaveValue("+0.35");
  });

  test("Temperature and tint write the trim's own fields, −100 to 100", async ({ page }) => {
    await nodeView(page);
    await select(page, 4);
    await panel(page).getByRole("tab", { name: "Colour" }).click();
    const temperature = panel(page).getByRole("slider", { name: "Temperature" });
    await expect(temperature).toHaveAttribute("aria-valuemin", "-100");
    await expect(temperature).toHaveAttribute("aria-valuemax", "100");
    await temperature.focus();
    await page.keyboard.press("ArrowLeft");
    await expect.poll(async () => (await nodeSettings(page, 4)).trim.temperature).toBe(29);
    await setValue(page, "Tint", "-12");
    await expect.poll(async () => (await nodeSettings(page, 4)).trim.tint).toBe(-12);
    expect((await nodeSettings(page, 4)).trim.temperature).toBe(29);
  });

  test("Main's controls keep their Auto, every other corrector's are plain numbers", async ({ page }) => {
    await nodeView(page);
    await select(page, 1);
    await expect(panel(page).getByRole("button", { name: "Auto" })).toHaveCount(0);
    await setValue(page, "Contrast", "20");
    await expect.poll(async () => (await nodeSettings(page, 1)).grade.contrast).toEqual({ kind: "manual", value: 20 });
    await setValue(page, "Contrast", "0");
    await expect.poll(async () => (await nodeSettings(page, 1)).grade.contrast).toEqual({ kind: "auto" });
    await select(page, 4);
    await setValue(page, "Contrast", "20");
    await setValue(page, "Contrast", "0");
    await expect.poll(async () => (await nodeSettings(page, 4)).grade.contrast).toEqual({ kind: "manual", value: 0 });
    // The Input has the AUTO tag on its Exposure.
    await select(page, "input");
    await panel(page).getByRole("tab", { name: "Light" }).click();
    await expect(panel(page).getByRole("button", { name: "Auto", exact: true })).toHaveAttribute("aria-pressed", "true");
  });
});

test.describe("the histogram, the scatter and Pick colour at a node", () => {
  test("name the selected node: `at` is the node's id, or the Input; the simple view and Main name none", async ({ page }) => {
    await nodeView(page);
    const histogram = await recordCalls(page, "histogram");
    const scatter = await recordCalls(page, "chromaScatter");
    await select(page, 3);
    await expect.poll(async () => (await histogram()).map((c) => c.args.slice(1))).toContainEqual([3]);
    await panel(page).getByRole("tab", { name: "Colour" }).click();
    await expect.poll(async () => (await scatter()).map((c) => c.args.slice(1))).toContainEqual([3]);
    await select(page, "input");
    await panel(page).getByRole("tab", { name: "Light" }).click();
    await expect.poll(async () => (await histogram()).map((c) => c.args.slice(1))).toContainEqual(["input"]);
    await select(page, 1);
    await histogram(true);
    await panel(page).getByRole("tab", { name: "Curve" }).click();
    await expect.poll(async () => (await histogram()).length).toBeGreaterThan(0);
    expect((await histogram()).every((c) => c.args.length === 1)).toBe(true);
  });
});

test.describe("W in the node view", () => {
  test("selects the Input, opens its Colour tab and starts Pick white; a node change ends it", async ({ page }) => {
    await nodeView(page);
    await select(page, 3);
    await page.keyboard.press("w");
    await expect(page.getByText("Click something neutral: grey or white. Esc cancels.")).toBeVisible();
    await expect(card(page, "input")).toHaveAttribute("aria-pressed", "true");
    await expect(head(page).getByRole("heading", { level: 2, name: "Input" })).toBeVisible();
    await expect(panel(page).getByRole("tab", { selected: true })).toHaveText("Colour");
    await expect(panel(page).getByRole("button", { name: "Pick white" })).toHaveAttribute("aria-pressed", "true");
    await select(page, 2);
    await expect(page.getByText("Click something neutral: grey or white. Esc cancels.")).toHaveCount(0);
    await page.keyboard.press("w");
    await expect(page.getByText("Click something neutral: grey or white. Esc cancels.")).toBeVisible();
    await page.keyboard.press("w");
    await expect(page.getByText("Click something neutral: grey or white. Esc cancels.")).toHaveCount(0);
    await page.keyboard.press("w");
    await page.keyboard.press("Escape");
    await expect(page.getByText("Click something neutral: grey or white. Esc cancels.")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Nodes", exact: true })).toHaveAttribute("aria-pressed", "true");
  });

  test("a click on the photo picks the white and ends the mode, with the EDITED tag on the Input's White balance card", async ({ page }) => {
    await nodeView(page);
    await select(page, 4);
    await page.keyboard.press("w");
    const layer = page.getByRole("application", { name: /^Pick white\./ });
    await expect(layer).toBeVisible();
    const box = (await layer.boundingBox())!;
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 3);
    await expect(page.getByText("Click something neutral: grey or white. Esc cancels.")).toHaveCount(0);
    await expect(panel(page).getByText("Edited", { exact: true })).toBeVisible();
    const input = (await settingsOf(page)).settings.nodes.items.find((n) => n.kind === "input") as { white_pick: unknown };
    expect(input.white_pick).not.toBeNull();
  });
});

test.describe("undo, Copy settings and Paste", () => {
  test("undo takes back one change of one node at a time, across nodes, and redo does it again", async ({ page }) => {
    await nodeView(page);
    await select(page, 4);
    await setValue(page, "Exposure trim", "-1");
    await select(page, 3);
    await panel(page).getByRole("tab", { name: "Colour" }).click();
    await setValue(page, "Temperature", "-30");
    await expect.poll(async () => (await nodeSettings(page, 3)).trim.temperature).toBe(-30);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await nodeSettings(page, 3)).trim.temperature).toBe(-18);
    expect((await nodeSettings(page, 4)).trim.exposure_ev).toBe(-1);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await nodeSettings(page, 4)).trim.exposure_ev).toBe(0);
    await page.keyboard.press("Meta+Shift+z");
    await expect.poll(async () => (await nodeSettings(page, 4)).trim.exposure_ev).toBe(-1);
  });

  test("Copy settings and Paste carry the whole graph to another photo, in one undo step, from the node view's own foot row", async ({ page }) => {
    const problems = watchErrors(page);
    await nodeView(page);
    await select(page, 4);
    await setValue(page, "Exposure trim", "0.8");
    const foot = panel(page).getByRole("group", { name: "Copy and paste settings" });
    await expect(foot.getByRole("button")).toHaveText(["Copy settings", "Paste"]);
    await foot.getByRole("button", { name: "Copy settings" }).click();
    await page.getByRole("button", { name: "Next photo" }).click();
    await expect(page.getByRole("group", { name: "Photos" })).toContainText("5 of 12");
    await expect(page.locator('[data-node-card="4"]')).toHaveCount(0);
    await foot.getByRole("button", { name: "Paste" }).click();
    await expect(page.locator('[data-node-card="4"]')).toBeVisible();
    await expect(card(page, 2)).toHaveAccessibleName(/^Node 2, Mountain, masked by Object and Radial/);
    const pasted = await mock(page, async (api) => {
      const rs = await api.getSettings(5);
      const four = rs.settings.nodes.items.find((n) => n.id === 4) as { trim: { exposure_ev: number } };
      return four.trim.exposure_ev;
    });
    expect(pasted).toBe(0.8);
    await page.keyboard.press("Meta+z");
    await expect(page.locator('[data-node-card="4"]')).toHaveCount(0);
    // React Flow warns that its attribution is hidden (`docs/14`: the notice travels in THIRD-PARTY.md); nothing else may be said.
    expect(problems.filter((p) => !/React Flow: It seems like you are hiding the attribution/.test(p))).toEqual([]);
  });
});
