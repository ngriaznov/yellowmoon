import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, recordCalls, settle } from "./helpers/app";
import { FOUR_NODES } from "./helpers/states";

// The node view's shell on the mock API (docs/23 sections 6 and 7.1, docs/11 §Node view; task N-07): the Nodes button and N, layout A with its
// divider and Expand, the viewport that follows, the photo switcher, and the notice of the simple view with Open nodes. The canvas is empty
// until N-08; the cards and the panel of a node come with N-08 and N-09.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await openPhotos(page);
}

/** The photo of the mockup's 4th place with four nodes, as `initialSettings` gives them (a node patch). */
async function photosWithNodes(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, { initialSettings: { DSC04204: { nodes: FOUR_NODES.nodes, links: FOUR_NODES.links } } as never });
  await openPhotos(page);
}

const nodesButton = (page: Page) => page.getByRole("button", { name: "Nodes", exact: true });
const divider = (page: Page) => page.getByRole("separator", { name: "Resize the graph" });
const cell = (page: Page) => page.locator("[data-viewport]");
const graph = (page: Page) => page.getByRole("region", { name: "Node editor" });
const expand = (page: Page) => page.getByRole("button", { name: "Expand", exact: true });

/** A box, as the layout makes it. */
interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}
const box = async (locator: Locator): Promise<Box> => (await locator.boundingBox())!;

/** The graph's, the divider's and the cell's boxes, and the panel's. */
async function geometry(page: Page): Promise<{ cell: Box; divider: Box; graph: Box; bar: Box; canvas: Box; panel: Box; header: Box }> {
  return {
    cell: await box(cell(page)),
    divider: await box(divider(page)),
    graph: await box(graph(page)),
    bar: await box(page.getByRole("group", { name: "Graph tools" })),
    canvas: await box(page.locator(".ym-graph__canvas")),
    panel: await box(page.getByRole("complementary", { name: "Tools" })),
    header: await box(page.getByRole("banner")),
  };
}

/** The rectangle the page last told the core the photo cell is at (`viewport_set_rect`), from the calls of the mock. */
async function lastRect(calls: Awaited<ReturnType<typeof recordCalls>>): Promise<Box & { scale: number }> {
  const all = await calls();
  expect(all.length, "viewport_set_rect was called").toBeGreaterThan(0);
  return all[all.length - 1]!.args[0] as Box & { scale: number };
}

/** Waits until the page has told the core where the cell is now, and says that rectangle. */
async function reportedRect(page: Page, calls: Awaited<ReturnType<typeof recordCalls>>): Promise<Box & { scale: number }> {
  await expect
    .poll(async () => {
      // The page measures the cell on its next frame (a `ResizeObserver`), which may come after a box read here has already forced the layout.
      const all = await calls();
      if (all.length === 0) return Number.POSITIVE_INFINITY;
      const rect = all[all.length - 1]!.args[0] as Box;
      const now = await box(cell(page));
      return Math.max(Math.abs(rect.x - now.x), Math.abs(rect.y - now.y), Math.abs(rect.width - now.width), Math.abs(rect.height - now.height));
    })
    .toBeLessThan(0.51);
  return lastRect(calls);
}

test.describe("opening and closing the node view", () => {
  test("the Nodes button, after Crop, goes to the node view and back; so does N", async ({ page }) => {
    await photos(page);
    const labels = await page.getByRole("navigation", { name: "Actions" }).getByRole("button").allTextContents();
    expect(labels.slice(0, 5).map((t) => t.trim())).toEqual(["Before", "Crop", "Nodes", "Add photos", expect.stringMatching(/^Fit/)]);
    await expect(nodesButton(page)).toHaveAttribute("aria-pressed", "false");
    await expect(nodesButton(page)).toHaveAccessibleDescription("Nodes (N)");
    await expect(page.getByRole("navigation", { name: "Photos" })).toBeVisible();
    await expect(divider(page)).toHaveCount(0);

    await nodesButton(page).click();
    await expect(nodesButton(page)).toHaveAttribute("aria-pressed", "true");
    await expect(divider(page)).toBeVisible();
    await expect(graph(page)).toBeVisible();
    // The filmstrip is gone, and its place is the switcher in the graph's toolbar.
    await expect(page.getByRole("navigation", { name: "Photos" })).toHaveCount(0);
    await nodesButton(page).click();
    await expect(nodesButton(page)).toHaveAttribute("aria-pressed", "false");
    await expect(divider(page)).toHaveCount(0);
    await expect(page.getByRole("navigation", { name: "Photos" })).toBeVisible();

    await page.keyboard.press("n");
    await expect(divider(page)).toBeVisible();
    await expect(nodesButton(page)).toHaveAttribute("aria-pressed", "true");
    // Esc does not leave it.
    await page.keyboard.press("Escape");
    await expect(divider(page)).toBeVisible();
    await page.keyboard.press("n");
    await expect(divider(page)).toHaveCount(0);
  });

  test("N is a letter in a text box and in a menu", async ({ page }) => {
    await photos(page);
    const search = page.getByRole("searchbox", { name: "Search looks" });
    await search.click();
    await page.keyboard.type("n");
    await expect(search).toHaveValue("n");
    await expect(divider(page)).toHaveCount(0);
    await search.fill("");
    await page.getByRole("button", { name: /^Fit/ }).click();
    await expect(page.getByRole("menu")).toBeVisible();
    await page.keyboard.press("n");
    await expect(divider(page)).toHaveCount(0);
  });

  test("layout A at 1440 × 900: a photo of 666 × 444 over a 10 px divider and an editor of 371 px, the panel at the right, full height", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    await expect(divider(page)).toBeVisible();
    await settle(page);
    const g = await geometry(page);
    expect(g.header.height).toBe(75);
    // The photo: 3:2, centred in the left column, the height left over.
    expect(g.cell.width).toBeCloseTo(666, 0);
    expect(g.cell.height).toBeCloseTo(444, 0);
    expect(g.cell.width / g.cell.height).toBeCloseTo(1.5, 2);
    expect(g.cell.y).toBeCloseTo(75, 0);
    expect(g.cell.x).toBeCloseTo((1040 - 666) / 2, 0);
    // The divider, the editor (a 44 px toolbar and a 327 px canvas), and nothing between them.
    expect(g.divider.height).toBeCloseTo(10, 1);
    expect(g.divider.y).toBeCloseTo(75 + 444, 0);
    expect(g.graph.height).toBeCloseTo(371, 0);
    expect(g.graph.y).toBeCloseTo(75 + 444 + 10, 0);
    expect(g.graph.y + g.graph.height).toBeCloseTo(900, 0);
    expect(g.graph.width).toBeCloseTo(1040, 0);
    expect(g.bar.height).toBeCloseTo(44, 0);
    expect(g.canvas.height).toBeCloseTo(327, 0);
    // The panel keeps its 400 px and the whole height under the header.
    expect(g.panel.width).toBeCloseTo(400, 0);
    expect(g.panel.y).toBeCloseTo(75, 0);
    expect(g.panel.height).toBeCloseTo(900 - 75, 0);
    // The grab handle is in the middle of the divider, and the divider takes the keyboard.
    const grip = await box(page.locator(".ym-divider__grip"));
    expect(grip.x + grip.width / 2).toBeCloseTo(g.divider.x + g.divider.width / 2, 0);
  });

  test("at 1728 × 1117 the same rules give a photo of about 844 × 563 and an editor of 1328 × 469; at 1180 × 760 the smallest window works too", async ({ page }) => {
    await page.setViewportSize({ width: 1728, height: 1117 });
    await photos(page);
    await nodesButton(page).click();
    await expect(divider(page)).toBeVisible();
    await settle(page);
    let g = await geometry(page);
    expect(g.cell.width).toBeCloseTo(844.5, 0);
    expect(g.cell.height).toBeCloseTo(563, 0);
    expect(g.graph.width).toBeCloseTo(1328, 0);
    expect(g.graph.height).toBeCloseTo(469, 0);
    // The window shrinks: the proportion stays, the cell stays 3:2, and the viewport is told.
    await page.setViewportSize({ width: 1180, height: 760 });
    await expect.poll(async () => (await box(cell(page))).height).toBeCloseTo(367, 0);
    g = await geometry(page);
    expect(g.graph.height).toBeCloseTo(308, 0);
    expect(g.cell.width / g.cell.height).toBeCloseTo(1.5, 2);
    expect(g.cell.width).toBeCloseTo(550.5, 0);
    expect(g.panel.width).toBeCloseTo(400, 0);
  });
});

test.describe("the divider", () => {
  test("is a separator named 'Resize the graph' with the 25 % to 70 % range, a Tab stop with the ring", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    const d = divider(page);
    await expect(d).toHaveAttribute("aria-orientation", "horizontal");
    await expect(d).toHaveAttribute("aria-valuenow", "45");
    await expect(d).toHaveAttribute("aria-valuemin", "25");
    await expect(d).toHaveAttribute("aria-valuemax", /^(69|70)$/);
    await expect(d).toHaveCSS("cursor", "row-resize");
    // A Tab stop: the keyboard reaches it (the ring shows for the keyboard, not for a click).
    await page.getByRole("button", { name: "Nodes", exact: true }).focus();
    for (let i = 0; i < 60 && (await d.evaluate((el) => document.activeElement === el)) === false; i++) await page.keyboard.press("Tab");
    await expect(d).toBeFocused();
    await expect(d).toHaveCSS("outline-width", "3px");
  });

  test("a drag moves it: the photo follows 1:1, stays 3:2 and the core is told where the cell is; Fit is calculated for the smaller cell", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    await expect(divider(page)).toBeVisible();
    await settle(page);
    const calls = await recordCalls(page, "viewportSetRect");
    const fit = (await page.getByRole("button", { name: /^Fit/ }).textContent())!.trim();
    const before = await geometry(page);
    const at = { x: before.divider.x + before.divider.width / 2, y: before.divider.y + before.divider.height / 2 };

    // Up 60 px: the graph is 60 px taller, the photo 60 px lower.
    await page.mouse.move(at.x, at.y);
    await page.mouse.down();
    await page.mouse.move(at.x, at.y - 30, { steps: 3 });
    await page.mouse.move(at.x, at.y - 60, { steps: 3 });
    await page.mouse.up();
    await settle(page);
    const up = await geometry(page);
    expect(up.graph.height).toBeCloseTo(before.graph.height + 60, 0);
    expect(up.cell.height).toBeCloseTo(before.cell.height - 60, 0);
    expect(up.cell.width / up.cell.height).toBeCloseTo(1.5, 2);
    expect(up.cell.width).toBeCloseTo((before.cell.height - 60) * 1.5, 0);
    expect(up.divider.y).toBeCloseTo(before.divider.y - 60, 0);
    // The same fixed total: header + photo + divider + graph = the window.
    expect(75 + up.cell.height + up.divider.height + up.graph.height).toBeCloseTo(900, 0);
    // The rectangle the core was told is the cell's own, and Fit follows the smaller cell.
    const told = await reportedRect(page, calls);
    expect(told.width).toBeCloseTo(up.cell.width, 0);
    expect(told.height).toBeCloseTo(up.cell.height, 0);
    expect(told.scale).toBe(2);
    await expect(page.getByRole("button", { name: /^Fit/ })).not.toHaveText(fit);
    // The graph's toolbar kept its 44 px.
    expect(up.bar.height).toBeCloseTo(44, 0);
    await expect(divider(page)).toHaveAttribute("aria-valuenow", String(Math.round(((before.graph.height + 60) / 825) * 100)));

    // Down, past where it started.
    const now = { x: up.divider.x + up.divider.width / 2, y: up.divider.y + up.divider.height / 2 };
    await page.mouse.move(now.x, now.y);
    await page.mouse.down();
    await page.mouse.move(now.x, now.y + 150, { steps: 5 });
    await page.mouse.up();
    await settle(page);
    const down = await geometry(page);
    expect(down.graph.height).toBeCloseTo(before.graph.height - 90, 0);
    expect(down.cell.width / down.cell.height).toBeCloseTo(1.5, 2);
    const toldDown = await reportedRect(page, calls);
    expect(toldDown.height).toBeCloseTo(down.cell.height, 0);
  });

  test("stops at 70 % of the height, with the photo at least 200 px high, and at 25 %", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    await expect(divider(page)).toBeVisible();
    const d = await box(divider(page));
    const mid = { x: d.x + d.width / 2, y: d.y + d.height / 2 };
    await page.mouse.move(mid.x, mid.y);
    await page.mouse.down();
    await page.mouse.move(mid.x, 80, { steps: 6 });
    await page.mouse.up();
    await settle(page);
    let g = await geometry(page);
    expect(g.graph.height).toBeCloseTo(Math.round(0.7 * 825), 0);
    expect(g.cell.height).toBeGreaterThanOrEqual(200);
    expect(g.cell.width / g.cell.height).toBeCloseTo(1.5, 2);
    const up = await box(divider(page));
    await page.mouse.move(up.x + up.width / 2, up.y + up.height / 2);
    await page.mouse.down();
    await page.mouse.move(up.x + up.width / 2, 895, { steps: 6 });
    await page.mouse.up();
    await settle(page);
    g = await geometry(page);
    expect(g.graph.height).toBeCloseTo(Math.round(0.25 * 825), 0);
    expect(g.cell.width / g.cell.height).toBeCloseTo(1.5, 2);
  });

  test("↑ and ↓ move it 10 px, ⇧ 50, with the cell's rectangle reported each time; a double click returns it to 45 %", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    await expect(divider(page)).toBeVisible();
    await settle(page);
    const calls = await recordCalls(page, "viewportSetRect");
    await divider(page).focus();
    const start = await geometry(page);
    expect(start.graph.height).toBeCloseTo(371, 0);

    await page.keyboard.press("ArrowUp");
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(381, 0);
    let g = await geometry(page);
    expect(g.cell.height).toBeCloseTo(434, 0);
    expect((await reportedRect(page, calls)).height).toBeCloseTo(434, 0);
    await page.keyboard.press("Shift+ArrowUp");
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(431, 0);
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("ArrowDown");
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(411, 0);
    await page.keyboard.press("Shift+ArrowDown");
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(361, 0);
    g = await geometry(page);
    expect(g.cell.width / g.cell.height).toBeCloseTo(1.5, 2);
    expect((await reportedRect(page, calls)).height).toBeCloseTo(g.cell.height, 0);
    // The divider keeps the focus through all of it, and ← and → are still the photos'.
    await expect(divider(page)).toBeFocused();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("img", { name: /^Photo DSC04205/ })).toBeVisible();
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByRole("img", { name: /^Photo DSC04204/ })).toBeVisible();

    await divider(page).dblclick();
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(371, 0);
    expect((await geometry(page)).cell.height).toBeCloseTo(444, 0);
    expect((await reportedRect(page, calls)).height).toBeCloseTo(444, 0);
  });
});

test.describe("Expand", () => {
  test("makes the editor 66 % of the height (545 px, the photo 405 × 270) and back, and the cell is reported each way", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    await expect(divider(page)).toBeVisible();
    await settle(page);
    const calls = await recordCalls(page, "viewportSetRect");
    await expect(expand(page)).toHaveAttribute("aria-pressed", "false");
    await expand(page).click();
    await expect(expand(page)).toHaveAttribute("aria-pressed", "true");
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(545, 0);
    let g = await geometry(page);
    expect(g.cell.width).toBeCloseTo(405, 0);
    expect(g.cell.height).toBeCloseTo(270, 0);
    expect(g.canvas.height).toBeCloseTo(501, 0);
    const told = await reportedRect(page, calls);
    expect(told.width).toBeCloseTo(405, 0);
    expect(told.height).toBeCloseTo(270, 0);
    await expand(page).click();
    await expect(expand(page)).toHaveAttribute("aria-pressed", "false");
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(371, 0);
    g = await geometry(page);
    expect(g.cell.width).toBeCloseTo(666, 0);
    expect((await reportedRect(page, calls)).width).toBeCloseTo(666, 0);
  });

  test("brings the divider back to where it was, and the divider moves Expand off", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    await divider(page).focus();
    await page.keyboard.press("Shift+ArrowUp");
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(421, 0);
    await expand(page).click();
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(545, 0);
    await expand(page).click();
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(421, 0);
    await expand(page).click();
    await divider(page).focus();
    await page.keyboard.press("ArrowDown");
    await expect(expand(page)).toHaveAttribute("aria-pressed", "false");
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(535, 0);
  });

  test("the divider and Expand are the window's: leaving the node view and coming back keeps them, leaving Photos does not", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    await expand(page).click();
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(545, 0);
    await page.keyboard.press("n");
    await page.keyboard.press("n");
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(545, 0);
    // Nothing of it is stored: a new page is 45 % again.
    await page.reload();
    await openPhotos(page);
    await nodesButton(page).click();
    await expect.poll(async () => (await box(graph(page))).height).toBeCloseTo(371, 0);
    expect(await page.evaluate(() => JSON.stringify([localStorage, sessionStorage]))).not.toMatch(/expand|divider|nodes/i);
  });
});

test.describe("the toolbar and the photo switcher", () => {
  test("Add node, Bypass and Delete at the left; the switcher, Show mask and Expand at the right; the canvas is empty", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    const bar = page.getByRole("group", { name: "Graph tools" });
    await expect(bar.getByRole("button")).toHaveText(["Add node", "Bypass", "Delete", "", "", "Show mask", "Expand"]);
    const places = await bar.getByRole("button").evaluateAll((els) => els.map((el) => Math.round(el.getBoundingClientRect().left)));
    expect([...places].sort((a, b) => a - b)).toEqual(places);
    await expect(bar.getByRole("button", { name: "Previous photo" })).toBeVisible();
    await expect(bar.getByRole("button", { name: "Next photo" })).toBeVisible();
    for (const name of ["Bypass", "Delete", "Show mask"]) await expect(bar.getByRole("button", { name })).toHaveAttribute("aria-disabled", "true");
    await expect(bar.getByRole("button", { name: "Show mask" })).toHaveAccessibleDescription("Show mask (M)");
    await expect(page.locator(".ym-graph__canvas")).toBeEmpty();
    await expect(page.locator(".ym-graph__canvas")).toHaveAttribute("data-selected-node", "1");
  });

  test("the Add node menu opens upward over the photo, with Corrector and the four that come later", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    const button = page.getByRole("button", { name: "Add node" });
    // Read before it opens: while a menu is open the rest of the page is hidden from the accessibility tree, the button too.
    const b = await box(button);
    await button.click();
    const menu = page.getByRole("menu", { name: "Add node" });
    await expect(menu).toBeVisible();
    const m = await box(menu);
    expect(m.y + m.height).toBeLessThanOrEqual(b.y + 1);
    await expect(menu.getByRole("menuitem")).toHaveCount(5);
    await expect(menu.getByRole("menuitem").first()).toContainText("Corrector");
    await expect(menu.getByRole("menuitem").first()).toContainText("Grades all or part of the photo.");
    for (const name of ["Parallel", "Layer", "Splitter and combiner", "Outside"]) {
      const item = menu.getByRole("menuitem", { name: new RegExp(`^${name}`) });
      await expect(item).toHaveAttribute("aria-disabled", "true");
      await expect(item).toContainText("Later");
    }
    await page.keyboard.press("Escape");
    await expect(menu).toHaveCount(0);
    await expect(button).toBeFocused();
  });

  test("the switcher reads '4 of 12' and takes the place of the filmstrip: Next and Previous, and ← and → do the same", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    const switcher = page.getByRole("group", { name: "Photos" });
    await expect(switcher).toContainText("4 of 12");
    await expect(page.getByRole("img", { name: /^Photo DSC04204/ })).toBeVisible();
    await switcher.getByRole("button", { name: "Next photo" }).click();
    await expect(switcher).toContainText("5 of 12");
    await expect(page.getByRole("img", { name: /^Photo DSC04205/ })).toBeVisible();
    await page.keyboard.press("ArrowRight");
    await expect(switcher).toContainText("6 of 12");
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("ArrowLeft");
    await expect(switcher).toContainText("4 of 12");
    await switcher.getByRole("button", { name: "Previous photo" }).click();
    await expect(page.getByRole("img", { name: /^Photo DSC04203/ })).toBeVisible();
    // Still the node view, with the same layout.
    await expect(divider(page)).toBeVisible();
    expect((await box(cell(page))).width).toBeCloseTo(666, 0);
  });

  test("P still picks, and Delete takes no photo out of the session in the node view (in the simple view it does)", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    await expect(page.getByRole("button", { name: /^Export 3/ })).toBeVisible();
    await page.keyboard.press("p");
    await expect(page.getByRole("button", { name: /^Export 2/ })).toBeVisible();
    await page.keyboard.press("p");
    await expect(page.getByRole("button", { name: /^Export 3/ })).toBeVisible();
    await page.keyboard.press("Delete");
    await page.keyboard.press("Backspace");
    await expect(page.getByRole("group", { name: "Photos" })).toContainText("4 of 12");
    await expect(page.getByRole("status").filter({ hasText: /Removed/ })).toHaveCount(0);
    await page.keyboard.press("n");
    await expect(page.getByRole("navigation", { name: "Photos" })).toContainText("4 of 12");
    await page.keyboard.press("Delete");
    await expect(page.getByRole("navigation", { name: "Photos" })).toContainText("of 11");
  });

  test("while Crop or Export shows, the graph is dimmed and read only", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    await expect(graph(page)).toHaveCSS("opacity", "1");
    await page.getByRole("button", { name: "Crop", exact: true }).click();
    await expect(page.getByRole("complementary", { name: "Crop" })).toBeVisible();
    await expect(graph(page)).toHaveCSS("opacity", "0.45");
    await expect(graph(page)).toHaveAttribute("inert", "");
    await expect(nodesButton(page)).toHaveAttribute("aria-disabled", "true");
    await page.keyboard.press("Escape");
    await expect(graph(page)).toHaveCSS("opacity", "1");
    await page.keyboard.press("Meta+e");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    await expect(graph(page)).toHaveCSS("opacity", "0.45");
    await expect(graph(page)).toHaveAttribute("inert", "");
    await page.getByRole("button", { name: "← Back to photos" }).click();
    await expect(graph(page)).toHaveCSS("opacity", "1");
  });

  test("full screen is the photo alone, and the node view is back after it", async ({ page }) => {
    await photos(page);
    await nodesButton(page).click();
    await expect(divider(page)).toBeVisible();
    await page.keyboard.press("f");
    await expect(divider(page)).toHaveCount(0);
    expect((await box(cell(page))).width).toBe(1440);
    await page.keyboard.press("f");
    await expect(divider(page)).toBeVisible();
    expect((await box(cell(page))).width).toBeCloseTo(666, 0);
  });
});

test.describe("the native viewport", () => {
  test("the page paints nothing over the cell in the node view, and the paper lies beside it, not over it", async ({ page }) => {
    // The photo is a layer behind the webview (docs/01 §Native viewport): the cell and everything under it is clear, and the paper beside the cell is
    // the page's own. Checked on the pixels of a capture without the page's background, as `viewport.spec.ts` does for the simple view.
    await photos(page, { mockViewport: "native" });
    await nodesButton(page).click();
    await expect(divider(page)).toBeVisible();
    await settle(page);
    const c = await box(cell(page));
    // (Not the top edge: the header's own shadow falls a few pixels over it, in the simple view too.)
    const inside = [
      [0.5, 0.5],
      [0.1, 0.3],
      [0.9, 0.7],
      [0.5, 0.3],
      [0.5, 0.95],
    ].map(([fx, fy]) => ({ x: Math.round(c.x + c.width * fx!), y: Math.round(c.y + c.height * fy!) }));
    const beside = [
      { x: Math.round(c.x / 2), y: Math.round(c.y + c.height / 2) },
      { x: Math.round(c.x + c.width + (1040 - c.x - c.width) / 2), y: Math.round(c.y + c.height / 2) },
    ];
    const boxes = await page.evaluate((at) => {
      const name = (e: Element) => e.tagName.toLowerCase() + [...e.classList].map((cls) => `.${cls}`).join("");
      return at.flatMap(({ x, y }) =>
        document.elementsFromPoint(x, y).map((e) => {
          const css = getComputedStyle(e);
          return { at: `${x},${y}`, element: name(e), color: css.backgroundColor, image: css.backgroundImage, backdrop: css.backdropFilter || "none" };
        }),
      );
    }, inside);
    for (const b of boxes) {
      expect(b, `${b.element} at ${b.at} paints a background over the cell`).toMatchObject({ color: "rgba(0, 0, 0, 0)", image: "none" });
      expect(b.backdrop, `${b.element} at ${b.at} blurs what is behind it`).toBe("none");
    }
    const png = await page.screenshot({ omitBackground: true, animations: "disabled" });
    const alpha = await page.evaluate(
      async ({ data, at, scale }) => {
        const bitmap = await createImageBitmap(await (await fetch(`data:image/png;base64,${data}`)).blob());
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(bitmap, 0, 0);
        return at.map(({ x, y }) => ctx.getImageData(Math.round(x * scale), Math.round(y * scale), 1, 1).data[3]!);
      },
      { data: png.toString("base64"), at: [...inside, ...beside, { x: 400, y: 700 }], scale: 2 },
    );
    expect(alpha.slice(0, inside.length), "alpha inside the cell").toEqual(inside.map(() => 0));
    expect(alpha.slice(inside.length, inside.length + 2), "alpha of the paper on both sides of the cell").toEqual([255, 255]);
    expect(alpha[inside.length + 2], "alpha in the graph").toBeGreaterThan(0);
    // The cell is still told to the core, and the page's cut-out of the aurora follows it as the divider moves.
    await divider(page).focus();
    await page.keyboard.press("Shift+ArrowUp");
    await expect.poll(async () => page.evaluate(() => document.documentElement.style.getPropertyValue("--cell-h"))).toBe(`${Math.round((await box(cell(page))).height * 100) / 100}px`.replace(/\.0+px$/, "px"));
  });
});

test.describe("a photo that has nodes, in the simple view", () => {
  test("says so under the tab row of every tab, with Open nodes, and the thumbnail carries the glyph", async ({ page }) => {
    await photosWithNodes(page);
    const text = "This photo has 4 nodes. These controls change the input, the output and node 1, Main.";
    for (const tab of ["Look", "Light", "Colour", "RAW"]) {
      await page.getByRole("tab", { name: tab }).click();
      const notice = page.getByRole("status").filter({ hasText: "This photo has 4 nodes." });
      await expect(notice, tab).toBeVisible();
      await expect(notice).toContainText(text);
      await expect(notice.getByRole("button", { name: "Open nodes" })).toBeVisible();
    }
    // Under the tab row, over the cards: between the tabs and the scrolling part of the panel.
    const tabs = await box(page.getByRole("tablist", { name: "Tool tabs" }));
    const notice = await box(page.getByRole("status").filter({ hasText: "This photo has 4 nodes." }));
    const scroll = await box(page.locator(".ym-tools__scroll"));
    expect(notice.y).toBeGreaterThanOrEqual(tabs.y + tabs.height);
    expect(notice.y + notice.height).toBeLessThanOrEqual(scroll.y + 1);
    // Two lines, and the link on the second: the pill does not grow to three.
    expect(notice.height).toBeLessThan(80);
    const link = await box(page.getByRole("button", { name: "Open nodes" }));
    expect(link.y).toBeGreaterThan(notice.y + notice.height / 2 - 4);
    // The glyph on the thumbnail of this photo and no other.
    const thumb = page.getByRole("button", { name: "Show DSC04204, picked, 4 nodes" });
    await expect(thumb.locator(".ym-filmstrip__nodes")).toBeVisible();
    await expect(page.locator(".ym-filmstrip__nodes")).toHaveCount(1);
    const t = await box(thumb);
    const glyph = await box(thumb.locator(".ym-filmstrip__nodes"));
    expect(glyph.x - t.x).toBeLessThan(12);
    expect(t.y + t.height - (glyph.y + glyph.height)).toBeLessThan(12);
    await thumb.locator(".ym-filmstrip__nodes").hover();
    await expect(page.getByRole("tooltip")).toHaveText("4 nodes");
    // Another photo has no notice.
    await page.getByRole("button", { name: "Show DSC04205, picked" }).click();
    await expect(page.getByText(/This photo has/)).toHaveCount(0);
  });

  test("Open nodes goes to the node view with Main selected, where the notice is not shown", async ({ page }) => {
    await photosWithNodes(page);
    await page.getByRole("tab", { name: "Light" }).click();
    await page.getByRole("button", { name: "Open nodes" }).click();
    await expect(divider(page)).toBeVisible();
    await expect(nodesButton(page)).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator(".ym-graph__canvas")).toHaveAttribute("data-selected-node", "1");
    await expect(page.getByText(/This photo has/)).toHaveCount(0);
    await nodesButton(page).click();
    await expect(page.getByText(/This photo has 4 nodes/)).toBeVisible();
  });

  test("a photo with only a trim on Main has its own sentence, and a plain photo has none", async ({ page }) => {
    await openApp(page, MOCKUP);
    await configureMock(page, { initialSettings: { DSC04204: { nodes: { "1": { trim: { exposure_ev: 0.5 } } } } } as never });
    await openPhotos(page);
    const notice = page.getByRole("status").filter({ hasText: "Node 1, Main, has a trim, masks or a bypass that these controls don't show." });
    await expect(notice).toContainText("Open nodes");
    await expect(page.getByRole("button", { name: "Show DSC04204, picked, node 1 has masks or a trim" })).toBeVisible();
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByRole("status").filter({ hasText: "that these controls don't show" })).toHaveCount(0);
  });
});
