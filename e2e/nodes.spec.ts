import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, recordCalls, settle } from "./helpers/app";
import { FOUR_NODES, GRADED_NODES } from "./helpers/states";

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
/** The header's zoom button ("Fit 15%"); the canvas has its own "Fit the graph". */
const fitButton = (page: Page) => page.getByRole("navigation", { name: "Actions" }).getByRole("button", { name: /^Fit/ });

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
    const fit = (await fitButton(page).textContent())!.trim();
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
    await expect(fitButton(page)).not.toHaveText(fit);
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
  test("Add node, Bypass and Delete at the left; the switcher, Show mask and Expand at the right; Main selected (More is in the node's head)", async ({ page }) => {
    await photos(page);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await nodesButton(page).click();
    const bar = page.getByRole("group", { name: "Graph tools" });
    await expect(bar.getByRole("button")).toHaveText(["Add node", "Bypass", "Delete", "", "", "Show mask", "Expand"]);
    await expect(bar.getByRole("button", { name: "More" })).toHaveCount(0);
    await expect(page.getByRole("complementary", { name: "Tools" }).getByRole("button", { name: "More" })).toBeVisible();
    const places = await bar.getByRole("button").evaluateAll((els) => els.map((el) => Math.round(el.getBoundingClientRect().left)));
    expect([...places].sort((a, b) => a - b)).toEqual(places);
    await expect(bar.getByRole("button", { name: "Previous photo" })).toBeVisible();
    await expect(bar.getByRole("button", { name: "Next photo" })).toBeVisible();
    // Main is selected: it can be bypassed but not deleted ("Bypass it instead"), and it has a mask to show (N-11).
    await expect(bar.getByRole("button", { name: "Bypass" })).not.toHaveAttribute("aria-disabled", "true");
    await expect(bar.getByRole("button", { name: "Delete" })).toHaveAttribute("aria-disabled", "true");
    await expect(bar.getByRole("button", { name: "Delete" })).toHaveAccessibleDescription("Main can't be deleted. Bypass it instead.");
    await expect(bar.getByRole("button", { name: "Show mask" })).not.toHaveAttribute("aria-disabled", "true");
    await expect(bar.getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-pressed", "false");
    await expect(bar.getByRole("button", { name: "Show mask" })).toHaveAccessibleDescription("Show mask (M)");
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
    await expect(menu.getByRole("menuitem")).toHaveCount(4);
    await expect(menu.getByRole("menuitem").first()).toContainText("Corrector");
    await expect(menu.getByRole("menuitem").first()).toContainText("Grades all or part of the photo.");
    for (const name of ["Parallel", "Layer", "Splitter and combiner"]) {
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

// ---------------------------------------------------------------------------------------------------------------------------------
// The graph (docs/23 section 7.2, task N-08): the markers, the cards, the ports, the links drawn by hand, the editing and the keys.
// ---------------------------------------------------------------------------------------------------------------------------------

/** The graph's canvas. */
const canvas = (page: Page) => page.locator(".ym-graph__canvas");
/** A card or a marker by its node id (`1`, `"input"`). */
const card = (page: Page, id: number | string) => page.locator(`[data-node-card="${id}"]`);
/** The press target of a port of a node. */
const portOf = (page: Page, node: number | string, port: "picture" | "key", side: "in" | "out") =>
  page.locator(`.ym-port__hit[data-port-node="${node}"][data-port-name="${port}"][data-port-side="${side}"]`);
/** The links as xyflow ids ("1.picture>2.picture"), sorted. */
const linkIds = (page: Page): Promise<string[]> =>
  page.locator(".react-flow__edge").evaluateAll((els) => els.map((el) => el.getAttribute("data-id") ?? "").sort());
const linkOf = (page: Page, id: string) => page.locator(`.react-flow__edge[data-id="${id}"]`);
const errorBar = (page: Page) => page.getByRole("alert");
const undo = (page: Page) => page.keyboard.press("Meta+z");
const redo = (page: Page) => page.keyboard.press("Meta+Shift+z");

interface Pt {
  x: number;
  y: number;
}
const centreOf = async (locator: Locator): Promise<Pt> => {
  const b = await box(locator);
  return { x: b.x + b.width / 2, y: b.y + b.height / 2 };
};

/** Presses at a point, drags to another in steps and lets go. */
async function dragBetween(page: Page, from: Pt, to: Pt, steps = 14): Promise<void> {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps });
  await page.mouse.up();
}

/** The node view of the photo with four nodes: Main, Mountain (2), Sky (3), Warmth (4) in a line. */
async function fourNodes(page: Page): Promise<void> {
  await photosWithNodes(page);
  // The node view opens on the node and the tab of the simple view's tab (N-09): Light is Main's, as these tests start.
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await nodesButton(page).click();
  await expect(card(page, 4)).toBeVisible();
  await settle(page);
}

/** The node view of a photo with Main alone. */
async function mainAlone(page: Page): Promise<void> {
  await photos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await nodesButton(page).click();
  await expect(card(page, 1)).toBeVisible();
  await settle(page);
}

/** The picture links of the four-node line. */
const LINE = ["1.picture>2.picture", "2.picture>3.picture", "3.picture>4.picture", "4.picture>output.picture", "input.picture>1.picture"];

test.describe("the graph: markers, cards, ports and links", () => {
  test("draws the Input, the four cards and the Output with their numbers, names and five picture links", async ({ page }) => {
    await fourNodes(page);
    await expect(card(page, "input")).toHaveAccessibleName("Input");
    await expect(card(page, "output")).toHaveAccessibleName("Output");
    await expect(card(page, 1)).toHaveAccessibleName("Node 1, Main");
    await expect(card(page, 2)).toHaveAccessibleName("Node 2, Mountain, picture from node 1");
    await expect(card(page, 3)).toHaveAccessibleName("Node 3, Sky, picture from node 2");
    await expect(card(page, 4)).toHaveAccessibleName("Node 4, Warmth, picture from node 3");
    // The numbers on the chips, and the names above the cards.
    await expect(canvas(page).locator(".ym-gnode__num")).toHaveText(["01", "02", "03", "04"]);
    await expect(canvas(page).locator(".ym-gnode__name")).toHaveText(["Main", "Mountain", "Sky", "Warmth"]);
    await expect(canvas(page).locator(".ym-marker__word")).toHaveText(["Input", "Output"]);
    expect(await linkIds(page)).toEqual(LINE);
    // A card is 150 × 112 px at 100 % zoom, with a thumbnail band of 81 px and a footer of 30.
    await expect(page.locator(".ym-zoom__value")).toHaveText("100%");
    const big = await box(card(page, 2));
    expect(big.width).toBeCloseTo(150, 0);
    expect(big.height).toBeCloseTo(112, 0);
    expect((await box(canvas(page).locator(".ym-gnode__thumb").first())).height).toBeCloseTo(81, 0);
    expect((await box(canvas(page).locator(".ym-gnode__foot").first())).height).toBeCloseTo(30, 0);
    // Four cards a unit apart: 214 px from one to the next, 64 px between them.
    const [a, b] = [await box(card(page, 1)), await box(card(page, 2))];
    expect(b.x - a.x).toBeCloseTo(214, 0);
    // Every card shows a picture: the photo's thumbnail until its own arrives (`node thumbnails`, below).
    await expect(canvas(page).locator(".ym-gnode__thumb img")).toHaveCount(4);
  });

  test("puts four ports on every card (a picture in and out, a key in and out) and one on each marker", async ({ page }) => {
    await fourNodes(page);
    for (const id of [1, 2, 3, 4]) {
      for (const port of ["picture", "key"] as const) {
        await expect(portOf(page, id, port, "in")).toHaveCount(1);
        await expect(portOf(page, id, port, "out")).toHaveCount(1);
      }
    }
    await expect(portOf(page, "input", "picture", "out")).toHaveCount(1);
    await expect(portOf(page, "output", "picture", "in")).toHaveCount(1);
    await expect(portOf(page, "input", "picture", "in")).toHaveCount(0);
    // A picture port is light grey and a key port the mask accent, a triangle in and a square out.
    const colours = await page.evaluate(() => {
      const of = (selector: string): string => getComputedStyle(document.querySelector(selector)!).color;
      return {
        picture: of('.ym-gnode .ym-port[data-port="picture"][data-side="in"]'),
        key: of('.ym-gnode .ym-port[data-port="key"][data-side="in"]'),
        mask: getComputedStyle(document.documentElement).getPropertyValue("--mask").trim(),
      };
    });
    expect(colours.mask).toBe("#ff6fd2");
    expect(colours.key).toBe("rgb(255, 111, 210)");
    expect(colours.picture).not.toBe(colours.key);
    await expect(canvas(page).locator('.ym-port[data-port="picture"][data-side="in"] svg')).not.toHaveCount(0);
    await expect(canvas(page).locator('.ym-port[data-port="picture"][data-side="out"] .ym-port__square')).not.toHaveCount(0);
  });

  test("with Main alone says 'Add a node to change only part of the photo.'", async ({ page }) => {
    await mainAlone(page);
    await expect(canvas(page).getByText("Add a node to change only part of the photo.")).toBeVisible();
    expect(await linkIds(page)).toEqual(["1.picture>output.picture", "input.picture>1.picture"]);
    await expect(card(page, 1)).toHaveAccessibleName("Node 1, Main");
    // With another node the line goes.
    await page.getByRole("button", { name: "Add node" }).click();
    await page.getByRole("menuitem", { name: /Corrector/ }).click();
    await expect(card(page, 2)).toBeVisible();
    await expect(canvas(page).getByText("Add a node to change only part of the photo.")).toHaveCount(0);
  });

  test("the selected card has a 2 px yellow ring, the others none, and a click, or an arrow key, moves it", async ({ page }) => {
    await fourNodes(page);
    const ring = (id: number) => canvas(page).locator(`.ym-gnode:has([data-node-card="${id}"]) .ym-gnode__ring`);
    await expect(card(page, 1)).toHaveAttribute("aria-pressed", "true");
    await expect(ring(1)).toBeVisible();
    await expect(ring(2)).toBeHidden();
    const style = await ring(1).evaluate((el) => {
      const c = getComputedStyle(el);
      return { width: c.borderTopWidth, colour: c.borderTopColor, tab: getComputedStyle(document.documentElement).getPropertyValue("--tab-look").trim() };
    });
    expect(style.width).toBe("2px");
    expect(style.colour).toBe("rgb(255, 194, 14)");
    expect(style.tab).toBe("#ffc20e");
    await card(page, 3).click();
    await expect(card(page, 3)).toHaveAttribute("aria-pressed", "true");
    await expect(card(page, 1)).toHaveAttribute("aria-pressed", "false");
    await expect(canvas(page)).toHaveAttribute("data-selected-node", "3");
    // Arrow keys move the selection to the neighbour in that direction, and the focus with it.
    await page.keyboard.press("ArrowRight");
    await expect(card(page, 4)).toHaveAttribute("aria-pressed", "true");
    await expect(card(page, 4)).toBeFocused();
    await page.keyboard.press("ArrowRight");
    await expect(card(page, "output")).toHaveAttribute("aria-pressed", "true");
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("ArrowLeft");
    await expect(card(page, 3)).toBeFocused();
    // The arrows moved no photo: still the fourth of twelve.
    await expect(page.getByRole("group", { name: "Photos" })).toContainText("4 of 12");
    // Nothing to the left of the Input.
    await card(page, "input").click();
    await page.keyboard.press("ArrowLeft");
    await expect(card(page, "input")).toBeFocused();
  });

  test("selecting a node moves nothing but the selection: no patch is sent", async ({ page }) => {
    await fourNodes(page);
    const calls = await recordCalls(page, "updateSettings");
    await card(page, 2).click();
    await card(page, 3).click();
    await page.keyboard.press("ArrowRight");
    expect(await calls()).toHaveLength(0);
  });

  test("is one Tab stop: the selected card, and arrows move between the cards", async ({ page }) => {
    await fourNodes(page);
    // Of everything in the canvas that a Tab can reach, one is the graph (the selected card) and the rest are the zoom control's buttons.
    const stops = await canvas(page).evaluate((root) => {
      const tabbable = [...root.querySelectorAll<HTMLElement>('a[href], button, input, select, textarea, [tabindex]')].filter(
        (el) => el.tabIndex >= 0 && !el.hasAttribute("disabled") && el.getClientRects().length > 0,
      );
      return tabbable.map((el) => el.getAttribute("aria-label") ?? el.className);
    });
    expect(stops.filter((s) => /^Node |^Input$|^Output$/.test(s))).toEqual(["Node 1, Main"]);
    expect(stops.filter((s) => /^Zoom|^Fit/.test(s))).toEqual(["Zoom out", "Zoom in", "Fit the graph"]);
    expect(stops).toHaveLength(4);
    // Tab from the toolbar's last button goes through the graph's one stop, then the zoom control, and out.
    await page.getByRole("button", { name: "Expand" }).focus();
    await page.keyboard.press("Tab");
    await expect(card(page, 1)).toBeFocused();
    await page.keyboard.press("Tab");
    await expect(page.getByRole("button", { name: "Zoom out" })).toBeFocused();
    // The selection is the stop: after choosing Sky, Tab lands on Sky.
    await card(page, 3).click();
    await page.getByRole("button", { name: "Expand" }).focus();
    await page.keyboard.press("Tab");
    await expect(card(page, 3)).toBeFocused();
  });

  test("a card that is not connected to the output is dimmed to 45 % with 'Not connected to the output.'", async ({ page }) => {
    await fourNodes(page);
    // Take Warmth's link to the Output away by replacing it: Main's output goes to the Output instead.
    await dragBetween(page, await centreOf(portOf(page, 1, "picture", "out")), await centreOf(portOf(page, "output", "picture", "in")));
    await expect.poll(() => linkIds(page)).toContain("1.picture>output.picture");
    for (const id of [2, 3, 4]) {
      await expect(card(page, id)).toHaveAccessibleName(new RegExp(`not connected`));
      await expect(canvas(page).locator(`.ym-gnode:has([data-node-card="${id}"])`)).toHaveAttribute("data-dead", "true");
      const opacity = await card(page, id).evaluate((el) => getComputedStyle(el).opacity);
      expect(Number(opacity)).toBeCloseTo(0.45, 2);
    }
    // The drag ended with the pointer over the graph: a tooltip that the cards' new places opened under it closes when the pointer leaves
    // (Radix keeps one open for its grace area while the pointer is on its way to it), so wait for none before the card is hovered.
    await page.mouse.move(5, 5);
    await expect(page.getByRole("tooltip")).toHaveCount(0);
    await card(page, 3).hover();
    await expect(page.getByRole("tooltip")).toHaveText("Not connected to the output.");
    // Linking it again brings it back.
    await dragBetween(page, await centreOf(portOf(page, 4, "picture", "out")), await centreOf(portOf(page, "output", "picture", "in")));
    for (const id of [2, 3, 4]) await expect(card(page, id)).not.toHaveAccessibleName(/not connected/);
  });

  test("a long name is cut at about ten letters on the card, with the whole name as its tooltip", async ({ page }) => {
    await fourNodes(page);
    await canvas(page).locator(".ym-gnode__name").nth(1).dblclick();
    const field = canvas(page).getByRole("textbox", { name: "Node name" });
    await field.fill("A very long name for a node");
    await field.press("Enter");
    const name = canvas(page).locator(".ym-gnode__name").nth(1);
    await expect(name).toHaveText("A very lo…");
    await name.hover();
    await expect(page.getByRole("tooltip")).toHaveText("A very long name for a node");
    // The card's own name carries the whole name.
    await expect(card(page, 2)).toHaveAccessibleName(/^Node 2, A very long name for a node/);
  });

  test("a bypassed card shows its picture at half strength with a slash, and says so", async ({ page }) => {
    await fourNodes(page);
    await card(page, 3).click();
    await page.keyboard.press("b");
    await expect(card(page, 3)).toHaveAccessibleName(/^Node 3, Sky, bypassed/);
    const thumb = canvas(page).locator(".ym-gnode:has([data-node-card='3'])");
    await expect(thumb.locator(".ym-gnode__slash")).toBeVisible();
    expect(Number(await thumb.locator(".ym-gnode__thumb img").evaluate((el) => getComputedStyle(el).opacity))).toBeCloseTo(0.5, 2);
    await expect(thumb.locator(".ym-gnode__tools")).toBeVisible();
    await expect(canvas(page).locator(".ym-gnode:has([data-node-card='2']) .ym-gnode__slash")).toHaveCount(0);
  });
});

test.describe("the graph: adding, deleting, bypassing, renaming and moving", () => {
  test("Add node › Corrector adds a node after the selected one, selects it, and undo and redo take it away and bring it back", async ({ page }) => {
    await fourNodes(page);
    await card(page, 2).click();
    await page.getByRole("button", { name: "Add node" }).click();
    await page.getByRole("menuitem", { name: /Corrector/ }).click();
    // The new node is node 5 by id and number 3 in the flow: Main → Mountain → new → Sky → Warmth.
    await expect(card(page, 5)).toHaveAttribute("aria-pressed", "true");
    await expect(card(page, 5)).toHaveAccessibleName("Node 3, Corrector, picture from node 2");
    expect(await linkIds(page)).toEqual(
      ["1.picture>2.picture", "2.picture>5.picture", "3.picture>4.picture", "4.picture>output.picture", "5.picture>3.picture", "input.picture>1.picture"],
    );
    // Numbers follow the flow: Sky is 4 and Warmth 5 now.
    await expect(card(page, 3)).toHaveAccessibleName(/^Node 4, Sky/);
    await expect(card(page, 4)).toHaveAccessibleName(/^Node 5, Warmth/);
    await expect(canvas(page).locator(".ym-gnode:has([data-node-card='5']) .ym-gnode__num")).toHaveText("03");
    await expect(canvas(page).locator(".ym-gnode:has([data-node-card='3']) .ym-gnode__num")).toHaveText("04");
    await expect(canvas(page).locator(".ym-gnode__num")).toHaveCount(5);
    // One unit right of Mountain, and the nodes after it one unit right.
    const at = async (id: number) => (await box(card(page, id))).x;
    expect((await at(5)) - (await at(2))).toBeCloseTo(214, 0);
    expect((await at(3)) - (await at(5))).toBeCloseTo(214 * 1, 0);
    // One undo step.
    await undo(page);
    await expect(card(page, 5)).toHaveCount(0);
    expect(await linkIds(page)).toEqual(LINE);
    await expect(card(page, 3)).toHaveAccessibleName(/^Node 3, Sky/);
    // The page is on Main again, where the removed node's selection falls to.
    await expect(canvas(page)).toHaveAttribute("data-selected-node", "1");
    await redo(page);
    await expect(card(page, 5)).toBeVisible();
    expect((await linkIds(page)).length).toBe(6);
  });

  test("with the Input selected adds after Main, and with the Output selected into the Output's link", async ({ page }) => {
    await fourNodes(page);
    await card(page, "input").click();
    await page.getByRole("button", { name: "Add node" }).click();
    await page.getByRole("menuitem", { name: /Corrector/ }).click();
    await expect(card(page, 5)).toHaveAccessibleName("Node 2, Corrector, picture from node 1");
    await undo(page);
    await expect(card(page, 5)).toHaveCount(0);
    await card(page, "output").click();
    await page.getByRole("button", { name: "Add node" }).click();
    await page.getByRole("menuitem", { name: /Corrector/ }).click();
    // Node 5 was made and undone: its id is used up, so this one is 6.
    await expect(card(page, 6)).toHaveAccessibleName("Node 5, Corrector, picture from node 4");
    expect(await linkIds(page)).toContain("6.picture>output.picture");
  });

  test("the new node's id is not one a deleted node had, and is selected with Delete available", async ({ page }) => {
    await fourNodes(page);
    await card(page, 4).click();
    await page.keyboard.press("Delete");
    await expect(card(page, 4)).toHaveCount(0);
    await page.getByRole("button", { name: "Add node" }).click();
    await page.getByRole("menuitem", { name: /Corrector/ }).click();
    // Node 4 is gone; the next id is 5 (one more than the largest the page has used), not 4 again.
    await expect(card(page, 5)).toBeVisible();
    await expect(card(page, 4)).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Delete" })).not.toHaveAttribute("aria-disabled", "true");
  });

  test("Add node's Corrector is off at 64 nodes, with 'A photo can have up to 64 nodes.'", async ({ page }) => {
    await openApp(page, MOCKUP);
    const nodes: Record<string, unknown> = {};
    const links: unknown[] = [{ from: { node: "input", port: "picture" }, to: { node: 1, port: "picture" } }];
    for (let id = 2; id <= 64; id++) {
      nodes[String(id)] = { kind: "corrector", name: `N${id}`, pos: [(id - 1) % 16, Math.floor((id - 1) / 16)] };
    }
    for (let id = 1; id < 64; id++) links.push({ from: { node: id, port: "picture" }, to: { node: id + 1, port: "picture" } });
    links.push({ from: { node: 64, port: "picture" }, to: { node: "output", port: "picture" } });
    await configureMock(page, { initialSettings: { DSC04204: { nodes, links } } as never });
    await openPhotos(page);
    await nodesButton(page).click();
    await expect(card(page, 64)).toBeAttached();
    await page.getByRole("button", { name: "Add node" }).click();
    const item = page.getByRole("menuitem", { name: /Corrector/ });
    await expect(item).toHaveAttribute("aria-disabled", "true");
    await expect(item).toHaveAccessibleDescription("A photo can have up to 64 nodes.");
    await item.hover();
    await expect(page.getByRole("tooltip")).toHaveText("A photo can have up to 64 nodes.");
    // A click adds nothing.
    await item.click({ force: true });
    await expect(card(page, 65)).toHaveCount(0);
  });

  test("Delete (the toolbar, the key, More › Delete node) reconnects the neighbours and selects the previous node; one undo puts it back", async ({ page }) => {
    await fourNodes(page);
    // The toolbar.
    await card(page, 3).click();
    await page.getByRole("button", { name: "Delete", exact: true }).click();
    await expect(card(page, 3)).toHaveCount(0);
    expect(await linkIds(page)).toEqual(["1.picture>2.picture", "2.picture>4.picture", "4.picture>output.picture", "input.picture>1.picture"]);
    await expect(card(page, 2)).toHaveAttribute("aria-pressed", "true");
    await expect(card(page, 4)).toHaveAccessibleName("Node 3, Warmth, picture from node 2");
    await undo(page);
    await expect(card(page, 3)).toBeVisible();
    expect(await linkIds(page)).toEqual(LINE);
    // The key, from the card.
    await card(page, 2).click();
    await page.keyboard.press("Delete");
    await expect(card(page, 2)).toHaveCount(0);
    await expect(card(page, 1)).toHaveAttribute("aria-pressed", "true");
    await undo(page);
    // Backspace, and More › Delete node.
    await card(page, 4).click();
    await page.keyboard.press("Backspace");
    await expect(card(page, 4)).toHaveCount(0);
    await undo(page);
    await expect(card(page, 4)).toBeVisible();
    await card(page, 4).click();
    await page.getByRole("button", { name: "More" }).click();
    await page.getByRole("menuitem", { name: "Delete node" }).click();
    await expect(card(page, 4)).toHaveCount(0);
    expect(await linkIds(page)).toEqual(["1.picture>2.picture", "2.picture>3.picture", "3.picture>output.picture", "input.picture>1.picture"]);
  });

  test("Delete is never Main ('Main can't be deleted. Bypass it instead.'), the Input or the Output, and never a photo", async ({ page }) => {
    await fourNodes(page);
    const del = page.getByRole("button", { name: "Delete", exact: true });
    await expect(del).toHaveAttribute("aria-disabled", "true");
    await expect(del).toHaveAccessibleDescription("Main can't be deleted. Bypass it instead.");
    await page.keyboard.press("Delete");
    await expect(card(page, 1)).toBeVisible();
    // The key on Main says so in the error bar and removes nothing.
    await card(page, 1).focus();
    await page.keyboard.press("Delete");
    await expect(errorBar(page)).toHaveText(/Main can't be deleted. Bypass it instead./);
    expect(await linkIds(page)).toEqual(LINE);
    await card(page, "input").click();
    await expect(del).toHaveAccessibleDescription("The input and the output can't be deleted.");
    await page.keyboard.press("Delete");
    await expect(errorBar(page)).toHaveText(/The input and the output can't be deleted./);
    await card(page, "output").click();
    await expect(del).toHaveAttribute("aria-disabled", "true");
    // The session still has its twelve photos.
    await expect(page.getByRole("group", { name: "Photos" })).toContainText("4 of 12");
    // The Output has no More (its head has Reset only), so there is no Delete node to offer.
    await expect(page.getByRole("button", { name: "More" })).toHaveCount(0);
  });

  test("Bypass (the toolbar, B) toggles the selected node, Main too, and is refused for the Input and the Output", async ({ page }) => {
    await fourNodes(page);
    const bypass = page.getByRole("button", { name: "Bypass", exact: true });
    await card(page, 2).click();
    await bypass.click();
    await expect(bypass).toHaveAttribute("aria-pressed", "true");
    await expect(card(page, 2)).toHaveAccessibleName(/bypassed/);
    await bypass.click();
    await expect(bypass).toHaveAttribute("aria-pressed", "false");
    await expect(card(page, 2)).not.toHaveAccessibleName(/bypassed/);
    // B, with the card focused: one undo step each way.
    await page.keyboard.press("b");
    await expect(card(page, 2)).toHaveAccessibleName(/bypassed/);
    await undo(page);
    await expect(card(page, 2)).not.toHaveAccessibleName(/bypassed/);
    await redo(page);
    await expect(card(page, 2)).toHaveAccessibleName(/bypassed/);
    // Main can be bypassed.
    await card(page, 1).click();
    await page.keyboard.press("b");
    await expect(card(page, 1)).toHaveAccessibleName("Node 1, Main, bypassed");
    // The Input and the Output cannot.
    await card(page, "input").click();
    await expect(bypass).toHaveAttribute("aria-disabled", "true");
    await expect(bypass).toHaveAccessibleDescription("The input and the output can't be bypassed.");
    await page.keyboard.press("b");
    await expect(errorBar(page)).toHaveText(/The input and the output can't be bypassed./);
  });

  test("B is a letter in a menu and in a text box, not a bypass", async ({ page }) => {
    await fourNodes(page);
    await card(page, 2).click();
    await canvas(page).locator(".ym-gnode__name").nth(1).dblclick();
    const field = canvas(page).getByRole("textbox", { name: "Node name" });
    await field.press("Control+a");
    await page.keyboard.type("Bob");
    await expect(field).toHaveValue("Bob");
    await expect(card(page, 2)).not.toHaveAccessibleName(/bypassed/);
    await field.press("Escape");
    await expect(canvas(page).locator(".ym-gnode__name").nth(1)).toHaveText("Mountain");
  });

  test("Rename: double-click the name, Return accepts, Esc cancels, 1 to 32 characters, one undo step", async ({ page }) => {
    await fourNodes(page);
    const name = (n: number) => canvas(page).locator(".ym-gnode__name").nth(n);
    await expect(name(1)).toHaveText("Mountain");
    await name(1).hover();
    await expect(page.getByRole("tooltip")).toHaveText("Double-click to rename");
    await name(1).dblclick();
    const field = canvas(page).getByRole("textbox", { name: "Node name" });
    await expect(field).toBeFocused();
    await expect(field).toHaveValue("Mountain");
    await expect(field).toHaveAttribute("maxlength", "32");
    // Esc cancels.
    await field.fill("Hills");
    await field.press("Escape");
    await expect(field).toHaveCount(0);
    await expect(name(1)).toHaveText("Mountain");
    // Return accepts, trimmed.
    await name(1).dblclick();
    await field.fill("  Hills  ");
    await field.press("Enter");
    await expect(name(1)).toHaveText("Hills");
    await expect(card(page, 2)).toHaveAccessibleName(/^Node 2, Hills/);
    // An empty name changes nothing.
    await name(1).dblclick();
    await field.fill("   ");
    await field.press("Enter");
    await expect(name(1)).toHaveText("Hills");
    // Leaving the box accepts it.
    await name(2).dblclick();
    await field.fill("Cloud");
    await page.getByRole("button", { name: "Expand" }).focus();
    await expect(name(2)).toHaveText("Cloud");
    // Undo: one step each.
    await undo(page);
    await expect(name(2)).toHaveText("Sky");
    await undo(page);
    await expect(name(1)).toHaveText("Mountain");
    await redo(page);
    await expect(name(1)).toHaveText("Hills");
  });

  test("Rename, from More › Rename; the Input and the Output have no Rename and no double-click", async ({ page }) => {
    await fourNodes(page);
    await card(page, 3).click();
    await page.getByRole("button", { name: "More" }).click();
    await page.getByRole("menuitem", { name: "Rename" }).click();
    // More is in the node's head, and so is the text box that Rename opens (a double click on a card's name opens one on the card).
    const field = page.getByRole("complementary", { name: "Tools" }).getByRole("textbox", { name: "Node name" });
    await expect(field).toBeFocused();
    await expect(canvas(page).getByRole("textbox")).toHaveCount(0);
    await field.fill("Blue");
    await field.press("Enter");
    await expect(card(page, 3)).toHaveAccessibleName(/^Node 3, Blue/);
    await card(page, "input").click();
    await page.getByRole("button", { name: "More" }).click();
    await expect(page.getByRole("menuitem", { name: "Rename" })).toHaveCount(0);
    await page.keyboard.press("Escape");
    await canvas(page).locator(".ym-marker__word").first().dblclick();
    await expect(canvas(page).getByRole("textbox")).toHaveCount(0);
  });

  test("a card moves by dragging: its place is pos only, the links follow, and one undo puts it back", async ({ page }) => {
    await fourNodes(page);
    const calls = await recordCalls(page, "updateSettings");
    const before = await box(card(page, 3));
    await dragBetween(page, { x: before.x + 75, y: before.y + 40 }, { x: before.x + 75 + 40, y: before.y + 40 + 80 });
    await expect.poll(async () => (await box(card(page, 3))).y).toBeGreaterThan(before.y + 70);
    const after = await box(card(page, 3));
    expect(after.x - before.x).toBeCloseTo(40, -1);
    // One patch: the place, nothing rewired.
    const sent = await calls();
    expect(sent).toHaveLength(1);
    const patch = sent[0]!.args[1] as { nodes: Record<string, { pos: number[] }>; links?: unknown };
    expect(Object.keys(patch.nodes)).toEqual(["3"]);
    expect(Object.keys(patch.nodes["3"]!)).toEqual(["pos"]);
    expect(patch.links).toBeUndefined();
    expect(await linkIds(page)).toEqual(LINE);
    // The card is selected by the drag.
    await expect(card(page, 3)).toHaveAttribute("aria-pressed", "true");
    // Undo puts it back.
    await undo(page);
    await expect.poll(async () => Math.round((await box(card(page, 3))).y)).toBe(Math.round(before.y));
    await expect.poll(async () => Math.round((await box(card(page, 3))).x)).toBe(Math.round(before.x));
    await redo(page);
    await expect.poll(async () => (await box(card(page, 3))).y).toBeGreaterThan(before.y + 70);
  });

  test("the Input and the Output cannot be dragged (a drag on one pans the canvas), and a click without a move sends nothing", async ({ page }) => {
    await fourNodes(page);
    const calls = await recordCalls(page, "updateSettings");
    const input = await box(card(page, "input"));
    const main = await box(card(page, 1));
    await dragBetween(page, { x: input.x + 16, y: input.y + 16 }, { x: input.x + 16, y: input.y + 60 });
    // The whole graph moved with the canvas: the Input is where it was beside Main.
    const [input2, main2] = [await box(card(page, "input")), await box(card(page, 1))];
    expect(input2.y - main2.y).toBeCloseTo(input.y - main.y, 0);
    expect(await calls()).toHaveLength(0);
    const c = await box(card(page, 2));
    await page.mouse.click(c.x + 75, c.y + 40);
    expect(await calls()).toHaveLength(0);
  });
});

/** The middle of the straight link between two cards' ports, where a click lands on it. */
async function linkMiddle(page: Page, from: number | string, to: number | string, port: "picture" | "key" = "picture"): Promise<Pt> {
  const [a, b] = [await centreOf(portOf(page, from, port, "out")), await centreOf(portOf(page, to, port, "in"))];
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/** The inputs that are lit while a link is dragged, as "node:port". */
const litInputs = (page: Page): Promise<string[]> =>
  canvas(page)
    .locator('.ym-port[data-state="lit"]')
    .evaluateAll((els) => els.map((el) => `${el.getAttribute("data-nodeid")}:${el.getAttribute("data-handleid")}`).sort());
const dimInputs = (page: Page): Promise<string[]> =>
  canvas(page)
    .locator('.ym-port[data-state="dim"]')
    .evaluateAll((els) => els.map((el) => `${el.getAttribute("data-nodeid")}:${el.getAttribute("data-handleid")}`).sort());

test.describe("the graph: links drawn by hand", () => {
  test("drag from an output to an input: the inputs that take it light up and the others dim, and a release on one makes the link", async ({ page }) => {
    await fourNodes(page);
    const from = await centreOf(portOf(page, 2, "key", "out"));
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    const over = await centreOf(portOf(page, 4, "key", "in"));
    await page.mouse.move((from.x + over.x) / 2, from.y + 60, { steps: 6 });
    // A key goes to a key input downstream of it: Sky's and Warmth's. Nothing else lights (Main's is upstream, Mountain's own is a circle).
    expect(await litInputs(page)).toEqual(["3:key", "4:key"]);
    expect((await dimInputs(page)).length).toBeGreaterThan(5);
    expect(await dimInputs(page)).toContain("3:picture");
    // The link follows the pointer from the output that stays.
    await expect(canvas(page).locator(".ym-draft")).toHaveAttribute("data-kind", "key");
    await page.mouse.move(over.x, over.y, { steps: 6 });
    await expect(canvas(page).locator('.ym-port[data-hover="true"]')).toHaveCount(1);
    await page.mouse.up();
    await expect.poll(() => linkIds(page)).toContain("2.key>4.key");
    // Nothing is lit any more, the link is a dashed one in the mask accent, and the picture links are as they were.
    expect(await litInputs(page)).toEqual([]);
    await expect(canvas(page).locator(".ym-draft")).toHaveCount(0);
    const style = await linkOf(page, "2.key>4.key").locator(".react-flow__edge-path").evaluate((el) => {
      const c = getComputedStyle(el);
      return { dash: c.strokeDasharray, colour: c.stroke };
    });
    expect(style.colour).toBe("rgb(255, 111, 210)");
    expect(style.dash).not.toBe("none");
    expect(await linkIds(page)).toEqual([...LINE, "2.key>4.key"].sort());
    // The picture links are drawn solid and grey.
    const picture = await linkOf(page, "1.picture>2.picture").locator(".react-flow__edge-path").evaluate((el) => getComputedStyle(el).strokeDasharray);
    expect(picture).toBe("none");
    // One undo step.
    await undo(page);
    await expect.poll(() => linkIds(page)).toEqual(LINE);
    await redo(page);
    await expect.poll(() => linkIds(page)).toContain("2.key>4.key");
    // Node 4 is told what feeds it.
    await expect(card(page, 4)).toHaveAccessibleName("Node 4, Warmth, picture from node 3, key from node 2");
  });

  test("a picture goes to picture inputs only, never to Main's, and an output may feed any number of inputs", async ({ page }) => {
    await fourNodes(page);
    const from = await centreOf(portOf(page, 1, "picture", "out"));
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    await page.mouse.move(from.x + 40, from.y + 70, { steps: 5 });
    // Main's own link is Main → Mountain (a no-op if dropped back), Sky's and Warmth's replace theirs, and the Output's too.
    expect(await litInputs(page)).toEqual(["2:picture", "3:picture", "4:picture", "output:picture"]);
    expect(await dimInputs(page)).toContain("1:picture");
    expect(await dimInputs(page)).toContain("2:key");
    await page.keyboard.press("Escape");
    await page.mouse.up();
    await expect(canvas(page).locator(".ym-draft")).toHaveCount(0);
    expect(await linkIds(page)).toEqual(LINE);
    // A second branch from Main's output: to a node that has no picture link yet is made by dropping on its input after it is freed.
    await dragBetween(page, await centreOf(portOf(page, 3, "picture", "in")), { x: from.x + 40, y: from.y + 110 });
    await expect.poll(() => linkIds(page)).not.toContain("2.picture>3.picture");
    await dragBetween(page, from, await centreOf(portOf(page, 3, "picture", "in")));
    await expect.poll(() => linkIds(page)).toContain("1.picture>3.picture");
    // Main now feeds Mountain and Sky.
    expect(await linkIds(page)).toEqual(["1.picture>2.picture", "1.picture>3.picture", "3.picture>4.picture", "4.picture>output.picture", "input.picture>1.picture"]);
    await expect(card(page, 2)).toHaveAccessibleName(/^Node 2, Mountain, not connected/);
  });

  test("dropping on an occupied input replaces its link, in one step", async ({ page }) => {
    await fourNodes(page);
    const calls = await recordCalls(page, "updateSettings");
    await dragBetween(page, await centreOf(portOf(page, 1, "picture", "out")), await centreOf(portOf(page, 3, "picture", "in")));
    await expect.poll(() => linkIds(page)).toContain("1.picture>3.picture");
    expect(await linkIds(page)).toEqual(["1.picture>2.picture", "1.picture>3.picture", "3.picture>4.picture", "4.picture>output.picture", "input.picture>1.picture"]);
    // One patch, links only.
    const sent = await calls();
    expect(sent).toHaveLength(1);
    expect(Object.keys(sent[0]!.args[1] as object)).toEqual(["links"]);
    // Mountain is not on a path to the output now: dimmed.
    await expect(card(page, 2)).toHaveAccessibleName(/not connected/);
    await undo(page);
    await expect.poll(() => linkIds(page)).toEqual(LINE);
    await expect(card(page, 2)).not.toHaveAccessibleName(/not connected/);
    await redo(page);
    await expect.poll(() => linkIds(page)).toContain("1.picture>3.picture");
  });

  test("a drop on the Output's input replaces the Output's link", async ({ page }) => {
    await fourNodes(page);
    await dragBetween(page, await centreOf(portOf(page, 2, "picture", "out")), await centreOf(portOf(page, "output", "picture", "in")));
    await expect.poll(() => linkIds(page)).toContain("2.picture>output.picture");
    expect(await linkIds(page)).not.toContain("4.picture>output.picture");
    await expect(card(page, 3)).toHaveAccessibleName(/not connected/);
    await expect(card(page, 4)).toHaveAccessibleName(/not connected/);
    await undo(page);
    await expect.poll(() => linkIds(page)).toEqual(LINE);
  });

  test("pick up a link's end from its input: drop it on empty canvas to remove it, one undo step", async ({ page }) => {
    await fourNodes(page);
    const input = await centreOf(portOf(page, 3, "picture", "in"));
    const empty = { x: input.x - 20, y: input.y + 100 };
    await page.mouse.move(input.x, input.y);
    await page.mouse.down();
    await page.mouse.move(empty.x, empty.y, { steps: 8 });
    // The link has come away from the port: it is not drawn between Mountain and Sky while it is carried, and the line follows the pointer.
    await expect(linkOf(page, "2.picture>3.picture")).toHaveCount(0);
    await expect(canvas(page).locator(".ym-draft")).toHaveCount(1);
    // The input that had it lights up again (putting it back is allowed), with the others of its type.
    expect(await litInputs(page)).toContain("3:picture");
    await page.mouse.up();
    await expect.poll(() => linkIds(page)).toEqual(["1.picture>2.picture", "3.picture>4.picture", "4.picture>output.picture", "input.picture>1.picture"]);
    // Sky read the Input now, and is still on the way to the Output.
    await expect(card(page, 3)).not.toHaveAccessibleName(/picture from/);
    await undo(page);
    await expect.poll(() => linkIds(page)).toEqual(LINE);
  });

  test("pick up a link's end and drop it on another input: the link moves there, replacing what was there", async ({ page }) => {
    await fourNodes(page);
    // Mountain → Sky is picked up at Sky and dropped on Warmth's picture input (which Sky fed).
    await dragBetween(page, await centreOf(portOf(page, 3, "picture", "in")), await centreOf(portOf(page, 4, "picture", "in")));
    await expect.poll(() => linkIds(page)).toContain("2.picture>4.picture");
    expect(await linkIds(page)).toEqual(["1.picture>2.picture", "2.picture>4.picture", "4.picture>output.picture", "input.picture>1.picture"]);
    await expect(card(page, 3)).toHaveAccessibleName(/not connected/);
    await undo(page);
    await expect.poll(() => linkIds(page)).toEqual(LINE);
  });

  test("a picked-up link dropped back on its own input, or let go with Esc, changes nothing", async ({ page }) => {
    await fourNodes(page);
    const calls = await recordCalls(page, "updateSettings");
    const input = await centreOf(portOf(page, 3, "picture", "in"));
    await dragBetween(page, input, { x: input.x - 3, y: input.y + 2 }, 4);
    await dragBetween(page, input, input, 2);
    // A drag that comes back to its input.
    await page.mouse.move(input.x, input.y);
    await page.mouse.down();
    await page.mouse.move(input.x - 60, input.y + 80, { steps: 6 });
    await page.mouse.move(input.x, input.y, { steps: 6 });
    await page.mouse.up();
    expect(await linkIds(page)).toEqual(LINE);
    // Esc while carried.
    await page.mouse.move(input.x, input.y);
    await page.mouse.down();
    await page.mouse.move(input.x - 60, input.y + 80, { steps: 6 });
    await page.keyboard.press("Escape");
    await page.mouse.up();
    expect(await linkIds(page)).toEqual(LINE);
    expect(await calls()).toHaveLength(0);
    await expect(errorBar(page)).toHaveCount(0);
  });

  test("a click on a link selects it (thicker, yellow); Delete removes it and not the node; Esc or a click elsewhere lets go", async ({ page }) => {
    await fourNodes(page);
    const mid = await linkMiddle(page, 2, 3);
    const path = linkOf(page, "2.picture>3.picture").locator(".react-flow__edge-path");
    const widthOf = async (): Promise<number> => parseFloat(await path.evaluate((el) => getComputedStyle(el).strokeWidth));
    const thin = await widthOf();
    await page.mouse.click(mid.x, mid.y);
    await expect(linkOf(page, "2.picture>3.picture")).toHaveClass(/selected/);
    expect(await widthOf()).toBeGreaterThan(thin);
    expect(await path.evaluate((el) => getComputedStyle(el).stroke)).toBe("rgb(255, 194, 14)");
    // The card the panel is on keeps its quiet ring.
    await expect(canvas(page).locator(".ym-gnode[data-quiet]")).toHaveCount(1);
    // Esc lets go.
    await page.keyboard.press("Escape");
    await expect(linkOf(page, "2.picture>3.picture")).not.toHaveClass(/selected/);
    await page.mouse.click(mid.x, mid.y);
    await expect(linkOf(page, "2.picture>3.picture")).toHaveClass(/selected/);
    // A click on the empty canvas lets go too.
    const bare = await box(canvas(page));
    await page.mouse.click(bare.x + bare.width / 2, bare.y + bare.height - 30);
    await expect(linkOf(page, "2.picture>3.picture")).not.toHaveClass(/selected/);
    // Delete takes the link away, not the node that is selected (Main).
    await page.mouse.click(mid.x, mid.y);
    await page.keyboard.press("Delete");
    await expect.poll(() => linkIds(page)).toEqual(["1.picture>2.picture", "3.picture>4.picture", "4.picture>output.picture", "input.picture>1.picture"]);
    await expect(card(page, 1)).toBeVisible();
    await undo(page);
    await expect.poll(() => linkIds(page)).toEqual(LINE);
    // Backspace and the toolbar's Delete do it too.
    await page.mouse.click(mid.x, mid.y);
    await page.getByRole("button", { name: "Delete", exact: true }).click();
    await expect.poll(() => linkIds(page)).not.toContain("2.picture>3.picture");
    await expect(card(page, 3)).toBeVisible();
  });

  test("a right click on a link offers Delete link", async ({ page }) => {
    await fourNodes(page);
    const mid = await linkMiddle(page, 3, 4);
    await page.mouse.click(mid.x, mid.y, { button: "right" });
    const item = page.getByRole("menuitem", { name: "Delete link" });
    await expect(item).toBeVisible();
    await expect(linkOf(page, "3.picture>4.picture")).toHaveClass(/selected/);
    await item.click();
    await expect.poll(() => linkIds(page)).not.toContain("3.picture>4.picture");
    await expect(card(page, 4)).toBeVisible();
  });

  test("the refusals say why in the error bar and change nothing: Main's input, the Output's link, a circle", async ({ page }) => {
    await fourNodes(page);
    const calls = await recordCalls(page, "updateSettings");
    // Anything but the Input on Main's picture input.
    await dragBetween(page, await centreOf(portOf(page, 2, "picture", "out")), await centreOf(portOf(page, 1, "picture", "in")));
    await expect(errorBar(page)).toHaveText(/Main always takes its picture from the input\./);
    expect(await linkIds(page)).toEqual(LINE);
    await page.getByRole("button", { name: "Dismiss" }).click();
    // Picking Input → Main up: it does not come away.
    const main = await centreOf(portOf(page, 1, "picture", "in"));
    await dragBetween(page, main, { x: main.x - 30, y: main.y + 90 });
    await expect(errorBar(page)).toHaveText(/Main always takes its picture from the input\./);
    await expect(linkOf(page, "input.picture>1.picture")).toHaveCount(1);
    await page.getByRole("button", { name: "Dismiss" }).click();
    // The Output's link cannot be taken away.
    const output = await centreOf(portOf(page, "output", "picture", "in"));
    await dragBetween(page, output, { x: output.x - 40, y: output.y + 110 });
    await expect(errorBar(page)).toHaveText(/The output needs a picture\. Link another node to it instead\./);
    expect(await linkIds(page)).toEqual(LINE);
    await page.getByRole("button", { name: "Dismiss" }).click();
    // A circle: Warmth's picture back to Mountain's input, and a key through the pictures.
    await dragBetween(page, await centreOf(portOf(page, 4, "picture", "out")), await centreOf(portOf(page, 2, "picture", "in")));
    await expect(errorBar(page)).toHaveText(/That would link the nodes in a circle\./);
    expect(await linkIds(page)).toEqual(LINE);
    await page.getByRole("button", { name: "Dismiss" }).click();
    await dragBetween(page, await centreOf(portOf(page, 4, "key", "out")), await centreOf(portOf(page, 2, "key", "in")));
    await expect(errorBar(page)).toHaveText(/That would link the nodes in a circle\./);
    expect(await linkIds(page)).toEqual(LINE);
    await page.getByRole("button", { name: "Dismiss" }).click();
    // A key on a picture input.
    await dragBetween(page, await centreOf(portOf(page, 2, "key", "out")), await centreOf(portOf(page, 3, "picture", "in")));
    await expect(errorBar(page)).toHaveText(/A link joins a picture to a picture, or a key to a key\./);
    expect(await linkIds(page)).toEqual(LINE);
    // None of it reached the core.
    expect(await calls()).toHaveLength(0);
  });

  test("a drop on nothing, or a press on an input with no link, does nothing at all", async ({ page }) => {
    await fourNodes(page);
    const calls = await recordCalls(page, "updateSettings");
    const out = await centreOf(portOf(page, 2, "picture", "out"));
    await dragBetween(page, out, { x: out.x + 20, y: out.y + 120 });
    await dragBetween(page, await centreOf(portOf(page, 3, "key", "in")), { x: out.x, y: out.y + 120 });
    expect(await linkIds(page)).toEqual(LINE);
    expect(await calls()).toHaveLength(0);
    await expect(errorBar(page)).toHaveCount(0);
  });

  test("a link's end can be a key's: pick up a key link and remove it", async ({ page }) => {
    await fourNodes(page);
    await dragBetween(page, await centreOf(portOf(page, 2, "key", "out")), await centreOf(portOf(page, 3, "key", "in")));
    await expect.poll(() => linkIds(page)).toContain("2.key>3.key");
    const input = await centreOf(portOf(page, 3, "key", "in"));
    await dragBetween(page, input, { x: input.x - 30, y: input.y + 70 });
    await expect.poll(() => linkIds(page)).toEqual(LINE);
  });
});

test.describe("the graph: reordering", () => {
  test("a card dropped with its centre over a picture link goes into it, which lights up while it is over; one undo step", async ({ page }) => {
    await fourNodes(page);
    const calls = await recordCalls(page, "updateSettings");
    // Mountain is dragged to the middle of the link Sky → Warmth.
    const c = await box(card(page, 2));
    const target = await linkMiddle(page, 3, 4);
    const start = { x: c.x + 75, y: c.y + 40 };
    const centre = { x: c.x + 75, y: c.y + 56 };
    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    await page.mouse.move(start.x + (target.x - centre.x), start.y + (target.y - centre.y), { steps: 14 });
    await expect(canvas(page).locator(".ym-link[data-drop]")).toHaveCount(1);
    await expect(linkOf(page, "3.picture>4.picture").locator(".ym-link")).toHaveAttribute("data-drop", "true");
    await page.mouse.up();
    // Main → Sky → Mountain → Warmth, with Mountain selected and one patch (the node's place, the neighbours', and the links).
    await expect.poll(() => linkIds(page)).toEqual(["1.picture>3.picture", "2.picture>4.picture", "3.picture>2.picture", "4.picture>output.picture", "input.picture>1.picture"]);
    await expect(card(page, 2)).toHaveAttribute("aria-pressed", "true");
    await expect(card(page, 2)).toHaveAccessibleName("Node 3, Mountain, picture from node 2");
    await expect(card(page, 3)).toHaveAccessibleName(/^Node 2, Sky/);
    await expect(canvas(page).locator(".ym-link[data-drop]")).toHaveCount(0);
    const sent = await calls();
    expect(sent).toHaveLength(1);
    expect(Object.keys(sent[0]!.args[1] as object).sort()).toEqual(["links", "nodes"]);
    await undo(page);
    await expect.poll(() => linkIds(page)).toEqual(LINE);
    await expect(card(page, 3)).toHaveAccessibleName(/^Node 3, Sky/);
    await redo(page);
    await expect.poll(() => linkIds(page)).toContain("3.picture>2.picture");
  });

  test("a card dropped on a link that is not under its centre, or on its own links, is only moved", async ({ page }) => {
    await fourNodes(page);
    const calls = await recordCalls(page, "updateSettings");
    const c = await box(card(page, 3));
    // A short drag down: the centre is nowhere near a link.
    await dragBetween(page, { x: c.x + 75, y: c.y + 40 }, { x: c.x + 75, y: c.y + 40 + 90 });
    expect(await linkIds(page)).toEqual(LINE);
    const patch = (await calls())[0]!.args[1] as { links?: unknown };
    expect(patch.links).toBeUndefined();
  });

  test("nothing goes before Main: a drop on the link Input → Main, and Main itself, are refused", async ({ page }) => {
    await fourNodes(page);
    const c = await box(card(page, 3));
    const target = await linkMiddle(page, "input", 1);
    const centre = { x: c.x + 75, y: c.y + 56 };
    await dragBetween(page, { x: c.x + 75, y: c.y + 40 }, { x: c.x + 75 + (target.x - centre.x), y: c.y + 40 + (target.y - centre.y) });
    // That link does not light up, and the drop is a move, not a reorder.
    expect(await linkIds(page)).toEqual(LINE);
    // Main dragged onto a link is only moved.
    const m = await box(card(page, 1));
    const sky = await linkMiddle(page, 3, 4);
    await dragBetween(page, { x: m.x + 75, y: m.y + 40 }, { x: m.x + 75 + (sky.x - (m.x + 75)), y: m.y + 40 + (sky.y - (m.y + 56)) });
    expect(await linkIds(page)).toEqual(LINE);
  });

  test("⌥→ and ⌥← move the focused node later and earlier; Main and the first after Main say why not", async ({ page }) => {
    await fourNodes(page);
    await card(page, 2).click();
    await page.keyboard.press("Alt+ArrowRight");
    await expect.poll(() => linkIds(page)).toEqual(["1.picture>3.picture", "2.picture>4.picture", "3.picture>2.picture", "4.picture>output.picture", "input.picture>1.picture"]);
    await expect(card(page, 2)).toHaveAccessibleName(/^Node 3, Mountain/);
    await page.keyboard.press("Alt+ArrowLeft");
    await expect.poll(() => linkIds(page)).toEqual(LINE);
    // Mountain is the first after Main: nothing goes before Main.
    await page.keyboard.press("Alt+ArrowLeft");
    await expect(errorBar(page)).toHaveText(/Nothing can go before Main\./);
    expect(await linkIds(page)).toEqual(LINE);
    await page.getByRole("button", { name: "Dismiss" }).click();
    // The last node cannot go later.
    await card(page, 4).click();
    await page.keyboard.press("Alt+ArrowRight");
    await expect(errorBar(page)).toHaveText(/This node is already the last one before the output\./);
    await page.getByRole("button", { name: "Dismiss" }).click();
    // Main cannot be reordered.
    await card(page, 1).click();
    await page.keyboard.press("Alt+ArrowRight");
    await expect(errorBar(page)).toHaveText(/Main can't be moved\. It always comes first\./);
    expect(await linkIds(page)).toEqual(LINE);
    // The arrows alone still move the selection, ⌥ moved no photo.
    await expect(page.getByRole("group", { name: "Photos" })).toContainText("4 of 12");
  });

  test("More › Move earlier and Move later do the same, and Main has neither", async ({ page }) => {
    await fourNodes(page);
    await card(page, 4).click();
    await page.getByRole("button", { name: "More" }).click();
    await expect(page.getByRole("menuitem", { name: "Move later" })).toHaveAttribute("aria-disabled", "true");
    await page.getByRole("menuitem", { name: "Move earlier" }).click();
    await expect.poll(() => linkIds(page)).toEqual(["1.picture>2.picture", "2.picture>4.picture", "3.picture>output.picture", "4.picture>3.picture", "input.picture>1.picture"]);
    await expect(card(page, 4)).toHaveAccessibleName(/^Node 3, Warmth/);
    await page.getByRole("button", { name: "More" }).click();
    await page.getByRole("menuitem", { name: "Move later" }).click();
    await expect.poll(() => linkIds(page)).toEqual(LINE);
    await card(page, 1).click();
    await page.getByRole("button", { name: "More" }).click();
    await expect(page.getByRole("menuitem", { name: "Move earlier" })).toHaveCount(0);
    await expect(page.getByRole("menuitem", { name: "Move later" })).toHaveCount(0);
    await expect(page.getByRole("menuitem", { name: "Delete node" })).toHaveAttribute("aria-disabled", "true");
    await expect(page.getByRole("menuitem", { name: "Rename" })).toBeVisible();
  });

  test("a move that would make a circle is refused: 'That would link the nodes in a circle.'", async ({ page }) => {
    await fourNodes(page);
    // Mountain's key feeds Warmth's key input, so Warmth cannot go before Mountain.
    await dragBetween(page, await centreOf(portOf(page, 2, "key", "out")), await centreOf(portOf(page, 4, "key", "in")));
    await expect.poll(() => linkIds(page)).toContain("2.key>4.key");
    const before = await linkIds(page);
    // ⌥← from Warmth puts it before Sky (fine); twice puts it before Mountain, which makes the circle.
    await card(page, 4).click();
    await page.keyboard.press("Alt+ArrowLeft");
    await expect.poll(() => linkIds(page)).toContain("4.picture>3.picture");
    await page.keyboard.press("Alt+ArrowLeft");
    await expect(errorBar(page)).toHaveText(/That would link the nodes in a circle\./);
    expect(await linkIds(page)).toContain("4.picture>3.picture");
    expect(await linkIds(page)).not.toEqual(before);
    // The same by dropping a card on a link: Warmth onto Main → Mountain.
    await page.getByRole("button", { name: "Dismiss" }).click();
    await undo(page);
    await expect.poll(() => linkIds(page)).toEqual(before);
    const c = await box(card(page, 4));
    const target = await linkMiddle(page, 1, 2);
    const centre = { x: c.x + 75, y: c.y + 56 };
    await dragBetween(page, { x: c.x + 75, y: c.y + 40 }, { x: c.x + 75 + (target.x - centre.x), y: c.y + 40 + (target.y - centre.y) });
    await expect(errorBar(page)).toHaveText(/That would link the nodes in a circle\./);
    expect(await linkIds(page)).toEqual(before);
  });
});


/**
 * Clicks a menu item with the pointer travelling to it, as a hand does. A submenu closes when the pointer jumps from its trigger to an item of
 * it in one step (Playwright's click moves in one), which no hand does: the node head's More opens its submenus to the left, away from the trigger.
 */
async function clickSubItem(page: Page, item: Locator): Promise<void> {
  const b = (await item.boundingBox())!;
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 12 });
  await item.click();
}

test.describe("the graph: the More menu's keyboard path", () => {
  test("Link picture to ▸ lists the inputs the node can be linked to, by number and name, and the Output; choosing one makes the link", async ({ page }) => {
    await fourNodes(page);
    await card(page, 2).click();
    await page.getByRole("button", { name: "More" }).click();
    await page.getByRole("menuitem", { name: "Link picture to" }).click();
    const items = page.getByRole("menuitem").filter({ hasText: /^(Node|Output)/ });
    await expect(items).toHaveText(["Node 4, Warmth", "Output"]);
    await clickSubItem(page, page.getByRole("menuitem", { name: "Output", exact: true }));
    await expect.poll(() => linkIds(page)).toContain("2.picture>output.picture");
    await expect(card(page, 4)).toHaveAccessibleName(/not connected/);
    await undo(page);
    await expect.poll(() => linkIds(page)).toEqual(LINE);
  });

  test("Link key to ▸ makes a key link; Remove link ▸ lists the node's links and removes one", async ({ page }) => {
    await fourNodes(page);
    await card(page, 2).click();
    await page.getByRole("button", { name: "More" }).click();
    await page.getByRole("menuitem", { name: "Link key to" }).click();
    await expect(page.getByRole("menuitem").filter({ hasText: /^Node/ })).toHaveText(["Node 3, Sky", "Node 4, Warmth"]);
    await clickSubItem(page, page.getByRole("menuitem", { name: "Node 4, Warmth" }));
    await expect.poll(() => linkIds(page)).toContain("2.key>4.key");
    await expect(card(page, 4)).toHaveAccessibleName("Node 4, Warmth, picture from node 3, key from node 2");
    // Remove link ▸: the picture links of Mountain and its key link.
    await page.getByRole("button", { name: "More" }).click();
    await page.getByRole("menuitem", { name: "Remove link" }).click();
    await expect(page.getByRole("menuitem").filter({ hasText: /^(Picture|Key)/ })).toHaveText(["Picture from Node 1, Main", "Picture to Node 3, Sky", "Key to Node 4, Warmth"]);
    await clickSubItem(page, page.getByRole("menuitem", { name: "Key to Node 4, Warmth" }));
    await expect.poll(() => linkIds(page)).toEqual(LINE);
    await expect(card(page, 4)).toHaveAccessibleName("Node 4, Warmth, picture from node 3");
  });

  test("the same rules apply: Remove link on Input → Main says why not, and the Input's menu has Link picture to only", async ({ page }) => {
    await fourNodes(page);
    await card(page, 1).click();
    await page.getByRole("button", { name: "More" }).click();
    await page.getByRole("menuitem", { name: "Remove link" }).click();
    await clickSubItem(page, page.getByRole("menuitem", { name: "Picture from Input" }));
    await expect(errorBar(page)).toHaveText(/Main always takes its picture from the input\./);
    expect(await linkIds(page)).toEqual(LINE);
    await page.getByRole("button", { name: "Dismiss" }).click();
    // The Output's head has Reset only (docs/23 §7.5): its link is replaced from the other node's Link picture to ▸, and taking it away is
    // refused by the drag and the Delete key.
    await card(page, "output").click();
    await expect(page.getByRole("button", { name: "More" })).toHaveCount(0);
    await card(page, "input").click();
    await page.getByRole("button", { name: "More" }).click();
    const names = await page.getByRole("menuitem").allTextContents();
    expect(names.map((n) => n.trim())).toEqual(["Link picture to", "Remove link"]);
    await page.getByRole("menuitem", { name: "Link picture to" }).click();
    await expect(page.getByRole("menuitem").filter({ hasText: /^(Node|Output)/ })).toHaveText(["Node 2, Mountain", "Node 3, Sky", "Node 4, Warmth", "Output"]);
  });
});

test.describe("the graph: zoom and pan", () => {
  test("the zoom control has '−', '100%', '+' and 'Fit' at the bottom right, from 25 % to 200 %", async ({ page }) => {
    await fourNodes(page);
    const zoom = page.locator(".ym-zoom");
    await expect(zoom.getByRole("button")).toHaveText(["", "", "Fit"]);
    await expect(zoom.getByRole("button", { name: "Zoom out" })).toBeVisible();
    await expect(zoom.getByRole("button", { name: "Zoom in" })).toBeVisible();
    await expect(zoom.getByRole("button", { name: "Fit the graph" })).toHaveText("Fit");
    await expect(page.locator(".ym-zoom__value")).toHaveText("100%");
    // At the canvas's bottom right.
    const [z, c] = [await box(zoom), await box(canvas(page))];
    expect(c.x + c.width - (z.x + z.width)).toBeCloseTo(12, 0);
    expect(c.y + c.height - (z.y + z.height)).toBeCloseTo(12, 0);
    // The sizes of the cards follow the zoom.
    await zoom.getByRole("button", { name: "Zoom in" }).click();
    await expect(page.locator(".ym-zoom__value")).toHaveText("125%");
    await expect.poll(async () => (await box(card(page, 1))).width).toBeCloseTo(187.5, 0);
    await zoom.getByRole("button", { name: "Zoom out" }).click();
    await zoom.getByRole("button", { name: "Zoom out" }).click();
    await expect(page.locator(".ym-zoom__value")).toHaveText("80%");
    for (let i = 0; i < 8; i++) if (await zoom.getByRole("button", { name: "Zoom out" }).isEnabled()) await zoom.getByRole("button", { name: "Zoom out" }).click();
    await expect(page.locator(".ym-zoom__value")).toHaveText("25%");
    await expect(zoom.getByRole("button", { name: "Zoom out" })).toHaveAttribute("aria-disabled", "true");
    for (let i = 0; i < 12; i++) if ((await zoom.getByRole("button", { name: "Zoom in" }).getAttribute("aria-disabled")) !== "true") await zoom.getByRole("button", { name: "Zoom in" }).click();
    await expect(page.locator(".ym-zoom__value")).toHaveText("200%");
    await expect(zoom.getByRole("button", { name: "Zoom in" })).toHaveAttribute("aria-disabled", "true");
    // Fit shows the whole graph again, at 100 % when it fits.
    await zoom.getByRole("button", { name: "Fit the graph" }).click();
    await expect(page.locator(".ym-zoom__value")).not.toHaveText("200%");
    // The view settles in 160 ms.
    const area = await box(canvas(page));
    await expect.poll(async () => (await box(card(page, "input"))).x).toBeGreaterThanOrEqual(area.x);
    await expect.poll(async () => (await box(card(page, "output"))).x + (await box(card(page, "output"))).width).toBeLessThanOrEqual(area.x + area.width);
  });

  test("a scroll pans, ⌘-scroll and a pinch zoom, a drag on the empty canvas pans, and the page never scrolls", async ({ page }) => {
    await fourNodes(page);
    const area = await box(canvas(page));
    const at = { x: area.x + 200, y: area.y + 60 };
    const first = await box(card(page, 1));
    await page.mouse.move(at.x, at.y);
    await page.mouse.wheel(0, 120);
    await expect.poll(async () => (await box(card(page, 1))).y).toBeLessThan(first.y - 20);
    await page.mouse.wheel(200, 0);
    await expect.poll(async () => (await box(card(page, 1))).x).toBeLessThan(first.x - 40);
    // ⌘ and a scroll zooms.
    await page.keyboard.down("Meta");
    await page.mouse.wheel(0, -240);
    await page.keyboard.up("Meta");
    await expect.poll(async () => parseInt((await page.locator(".ym-zoom__value").textContent()) ?? "100")).toBeGreaterThan(100);
    // A pinch is a scroll with Control held.
    await page.keyboard.down("Control");
    await page.mouse.wheel(0, 400);
    await page.keyboard.up("Control");
    await expect.poll(async () => parseInt((await page.locator(".ym-zoom__value").textContent()) ?? "100")).toBeLessThan(100);
    // Nothing of the page scrolled.
    expect(await page.evaluate(() => [document.scrollingElement?.scrollTop, document.scrollingElement?.scrollLeft, window.scrollY])).toEqual([0, 0, 0]);
    // A drag on the empty canvas pans it.
    await page.getByRole("button", { name: "Fit the graph" }).click();
    await expect(card(page, 1)).toBeVisible();
    // Fit takes 160 ms to settle; the pan is measured from where the view stopped.
    await page.waitForTimeout(400);
    const moved = await box(card(page, 1));
    await dragBetween(page, { x: area.x + 400, y: area.y + 290 }, { x: area.x + 460, y: area.y + 310 });
    // The pan starts a few pixels into the drag, so it follows the pointer within those.
    await expect.poll(async () => (await box(card(page, 1))).x - moved.x).toBeGreaterThan(45);
    expect((await box(card(page, 1))).x - moved.x).toBeLessThanOrEqual(61);
    expect(await linkIds(page)).toEqual(LINE);
  });
});

// ---------------------------------------------------------------------------------------------------------------------------------
// The cards' pictures (docs/23 section 5.7, task N-05): the picture after each node and the Output, asked for from the core while the node
// view is on, when a change has landed and never during a drag. The mock grades its flat scene through the nodes up to each card.
// ---------------------------------------------------------------------------------------------------------------------------------

/** The node view of the photo with Main, Mountain (+0.35 EV), Sky (cooler) and Warmth (warmer) in a line, with the cards' pictures made. */
async function gradedNodes(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, { initialSettings: { DSC04204: { nodes: GRADED_NODES.nodes, links: GRADED_NODES.links } } as never });
  await openPhotos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await nodesButton(page).click();
  await expect(card(page, 4)).toBeVisible();
  await picturesMade(page);
  await settle(page);
}

/** The `src` of each card's picture. */
const pictureSrcs = (page: Page): Promise<string[]> =>
  Promise.all([1, 2, 3, 4].map((id) => card(page, id).locator("img").getAttribute("src").then((src) => src ?? "")));

/** Waits until the four cards have four different pictures (before that every card shows the photo's thumbnail). */
async function picturesMade(page: Page): Promise<void> {
  await expect.poll(async () => new Set(await pictureSrcs(page)).size).toBe(4);
}

/** The average colour of each card's picture, as 0 to 255 red, green and blue. */
async function pictureColours(page: Page): Promise<number[][]> {
  return page.evaluate(async () => {
    const out: number[][] = [];
    for (const id of [1, 2, 3, 4]) {
      const img = document.querySelector<HTMLImageElement>(`[data-node-card="${id}"] img`)!;
      await img.decode();
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const context = canvas.getContext("2d")!;
      context.drawImage(img, 0, 0);
      const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
      const sum = [0, 0, 0];
      for (let i = 0; i < data.length; i += 4) for (let c = 0; c < 3; c++) sum[c]! += data[i + c]!;
      out.push(sum.map((v) => v / (data.length / 4)));
    }
    return out;
  });
}

const luma = (rgb: number[]): number => 0.2126 * rgb[0]! + 0.7152 * rgb[1]! + 0.0722 * rgb[2]!;

test.describe("the cards' pictures", () => {
  test("each card shows the picture after its node, 160 px wide, and the first is not the photo's thumbnail", async ({ page }) => {
    await gradedNodes(page);
    const srcs = await pictureSrcs(page);
    expect(new Set(srcs).size).toBe(4);
    for (const id of [1, 2, 3, 4]) {
      const size = await card(page, id).locator("img").evaluate((img: HTMLImageElement) => [img.naturalWidth, img.naturalHeight]);
      expect(size, `card ${id}`).toEqual([160, 107]);
    }
    // The photo's own thumbnail is not on any card.
    const photoThumb = await page.evaluate(async () => (await window.__yellowmoon!.mockApi.thumbnails()).find((t) => t.id === 4)?.url);
    expect(photoThumb).toBeTruthy();
    expect(srcs).not.toContain(photoThumb);
    // Mountain (+0.35 EV) is brighter than Main; Warmth is warmer than Sky, which is cooler than Mountain.
    const colours = await pictureColours(page);
    expect(luma(colours[1]!)).toBeGreaterThan(luma(colours[0]!));
    const warmth = (rgb: number[]): number => rgb[0]! - rgb[2]!;
    expect(warmth(colours[2]!)).toBeLessThan(warmth(colours[1]!));
    expect(warmth(colours[3]!)).toBeGreaterThan(warmth(colours[2]!));
  });

  test("change after a node's grade changes: bypassing Sky gives new pictures to Sky and Warmth, and the cards before it keep theirs", async ({ page }) => {
    await gradedNodes(page);
    const before = await pictureSrcs(page);
    const colours = await pictureColours(page);
    await card(page, 3).click();
    await page.keyboard.press("b");
    await expect(card(page, 3)).toHaveAccessibleName(/^Node 3, Sky,.* bypassed/);
    await expect.poll(async () => (await pictureSrcs(page))[2]).not.toBe(before[2]);
    const after = await pictureSrcs(page);
    expect(after[0]).toBe(before[0]);
    expect(after[1]).toBe(before[1]);
    expect(after[3]).not.toBe(before[3]);
    // A bypassed Sky shows Mountain's colours, and Warmth warms them instead of Sky's cooler ones.
    const now = await pictureColours(page);
    for (let c = 0; c < 3; c++) expect(now[2]![c]!).toBeCloseTo(colours[1]![c]!, 0);
    expect(now[3]![0]! - now[3]![2]!).toBeGreaterThan(colours[3]![0]! - colours[3]![2]!);
    // And back: the cards are as they were.
    await page.keyboard.press("b");
    await expect(card(page, 3)).not.toHaveAccessibleName(/bypassed/);
    await expect.poll(async () => (await pictureColours(page))[2]![2]!).toBeCloseTo(colours[2]![2]!, 0);
  });

  test("a change to Main's light gives every card a new picture; a slider drag asks for nothing and its release asks once", async ({ page }) => {
    await gradedNodes(page);
    const calls = await recordCalls(page, "nodeThumbnails");
    const before = await pictureSrcs(page);
    await expect(page.getByRole("tab", { name: "Light", selected: true })).toBeVisible();
    const slider = page.getByRole("slider", { name: "Exposure trim" }).locator("xpath=ancestor::*[@data-slot='slider'][1]");
    const track = (await slider.boundingBox())!;
    const y = track.y + track.height / 2;
    await page.mouse.move(track.x + track.width * 0.5, y);
    await page.mouse.down();
    await page.mouse.move(track.x + track.width * 0.8, y, { steps: 8 });
    await page.waitForTimeout(150);
    // The thumb is down: frames of the drag have gone to the core, and the cards have not been asked for.
    expect(await calls()).toHaveLength(0);
    expect(await pictureSrcs(page)).toEqual(before);
    await page.mouse.up();
    await expect.poll(async () => (await calls()).length).toBe(1);
    await expect.poll(async () => (await pictureSrcs(page)).filter((src, i) => src !== before[i]).length).toBe(4);
  });
});
