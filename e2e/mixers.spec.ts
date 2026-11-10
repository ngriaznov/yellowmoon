import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, settle } from "./helpers/app";

// The nodes of docs/25 (X-06) on the mock API: the Add node menu with its three items on, the glyph cards with their numbered ports and the "+",
// links dropped on the "+" and the renumbering, the Layer's panel (Mode, Opacity, the rows of its inputs and their drag), the Splitter and
// combiner trio, Delete and the refusals. The rules behind them are `src/state/graphOps.mixers.test.ts`.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;
const canvas = (page: Page) => page.locator(".ym-graph__canvas");
const card = (page: Page, id: number | string) => page.locator(`[data-node-card="${id}"]`);
const tools = (page: Page) => page.getByRole("complementary", { name: "Tools" });
const portHit = (page: Page, node: number | string, port: string, side: "in" | "out") =>
  page.locator(`.ym-port__hit[data-port-node="${node}"][data-port-name="${port}"][data-port-side="${side}"]`);
const linkIds = (page: Page): Promise<string[]> => page.locator(".react-flow__edge").evaluateAll((els) => els.map((el) => el.getAttribute("data-id") ?? "").sort());
const undo = (page: Page) => page.keyboard.press("Meta+z");

interface Pt {
  x: number;
  y: number;
}
const centreOf = async (locator: Locator): Promise<Pt> => {
  const b = (await locator.boundingBox())!;
  return { x: b.x + b.width / 2, y: b.y + b.height / 2 };
};
/** Clicks a menu item with the pointer travelling to it, as a hand does (a submenu closes when the pointer jumps to an item in one step). */
async function clickSubItem(page: Page, item: Locator): Promise<void> {
  const b = (await item.boundingBox())!;
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 12 });
  await item.click();
}
async function dragBetween(page: Page, from: Pt, to: Pt, steps = 14): Promise<void> {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps });
  await page.mouse.up();
}

type L = [string | number, string | number, string?, string?];
const links = (list: L[]) => list.map(([from, to, toPort = "picture", fromPort = "picture"]) => ({ from: { node: from, port: fromPort }, to: { node: to, port: toPort } }));

/** The node view of the mockup's photo with this graph (Main is 1). */
async function nodeView(page: Page, nodes: Record<string, unknown>, list: L[]): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, { initialSettings: { DSC04204: { nodes, links: links(list) } } as never });
  await openPhotos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await expect(card(page, 1)).toBeVisible();
  await settle(page);
}

/** Main feeds Warmth (2), Glow (3) and Sky (4): inputs 1 to 3 of a Layer (5). */
const LAYER_NODES = {
  "2": { kind: "corrector", name: "Warmth", pos: [1, -1] },
  "3": { kind: "corrector", name: "Glow", pos: [1, 0] },
  "4": { kind: "corrector", name: "Sky", pos: [1, 1] },
  "5": { kind: "layer", pos: [2, 0] },
};
const LAYER_LINKS: L[] = [["input", 1], [1, 2], [1, 3], [1, 4], [2, 5, "picture.1"], [3, 5, "picture.2"], [4, 5, "picture.3"], [5, "output"]];

/** Main feeds Skin (2) and Sky (3), the two inputs of a Parallel (4). */
const PARALLEL_NODES = {
  "2": { kind: "corrector", name: "Skin", pos: [1, -1] },
  "3": { kind: "corrector", name: "Sky", pos: [1, 1] },
  "4": { kind: "parallel", pos: [2, 0] },
};
const PARALLEL_LINKS: L[] = [["input", 1], [1, 2], [1, 3], [2, 4, "picture.1"], [3, 4, "picture.2"], [4, "output"]];

const addNode = async (page: Page, item: RegExp): Promise<void> => {
  await page.getByRole("button", { name: "Add node" }).click();
  await page.getByRole("menuitem", { name: item }).click();
};

test.describe("the Add node menu", () => {
  test("has the three items on, each with its line, and no 'Later'", async ({ page }) => {
    await nodeView(page, {}, [["input", 1], [1, "output"]]);
    await page.getByRole("button", { name: "Add node" }).click();
    const menu = page.getByRole("menu", { name: "Add node" });
    for (const [name, line] of [
      ["Parallel", "Grade branches side by side."],
      ["Layer", "Blend one branch over another."],
      ["Splitter and combiner", "Split a picture, then join it again."],
    ] as const) {
      const item = menu.getByRole("menuitem", { name: new RegExp(`^${name}`) });
      await expect(item).not.toHaveAttribute("aria-disabled", "true");
      await expect(item).toContainText(line);
    }
    await expect(menu).not.toContainText("Later");
  });

  test("turns Parallel and Layer off from 63 nodes and Splitter and combiner from 60, with the sentence as the tooltip", async ({ page }) => {
    const nodes: Record<string, unknown> = {};
    const list: L[] = [["input", 1]];
    for (let id = 2; id <= 60; id++) {
      nodes[String(id)] = { kind: "corrector", pos: [id, 0] };
      list.push([id - 1, id]);
    }
    list.push([60, "output"]);
    await nodeView(page, nodes, list);
    await page.getByRole("button", { name: "Add node" }).click();
    const trio = page.getByRole("menuitem", { name: /^Splitter and combiner/ });
    await expect(trio).toHaveAttribute("aria-disabled", "true");
    await expect(trio).toHaveAccessibleDescription("This adds five nodes, and a photo can have up to 64.");
    await expect(page.getByRole("menuitem", { name: /^Parallel/ })).not.toHaveAttribute("aria-disabled", "true");
    await expect(page.getByRole("menuitem", { name: /^Corrector/ })).not.toHaveAttribute("aria-disabled", "true");
  });
});

test.describe("adding the nodes", () => {
  test("Parallel makes a Parallel and a corrector beside Main, selects the corrector, and one undo takes both away", async ({ page }) => {
    await nodeView(page, {}, [["input", 1], [1, "output"]]);
    await addNode(page, /^Parallel/);
    await expect(card(page, 2)).toHaveAccessibleName("Parallel, mixer with 2 inputs, from node 1 and node 2");
    // The corrector reads what Main reads (the Input), so the Parallel's inputs start from the input.
    await expect(card(page, 3)).toHaveAccessibleName("Node 2, Corrector");
    // The corrector is selected with its tab open.
    await expect(card(page, 3)).toHaveAttribute("aria-pressed", "true");
    await expect(tools(page).getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true");
    expect(await linkIds(page)).toEqual(["1.picture>2.picture.1", "2.picture>output.picture", "3.picture>2.picture.2", "input.picture>1.picture", "input.picture>3.picture"]);
    await undo(page);
    await expect(card(page, 2)).toHaveCount(0);
    await expect(card(page, 3)).toHaveCount(0);
    expect(await linkIds(page)).toEqual(["1.picture>output.picture", "input.picture>1.picture"]);
  });

  test("Layer makes a Layer with the corrector on top, and its panel opens on Blend", async ({ page }) => {
    await nodeView(page, {}, [["input", 1], [1, "output"]]);
    await addNode(page, /^Layer/);
    await card(page, 2).click();
    await expect(tools(page).getByRole("tab")).toHaveText(["Blend"]);
    await expect(tools(page).locator(".ym-nodehead__line")).toHaveText("Mixer · 2 inputs");
    const rows = tools(page).locator("[data-input-row]");
    await expect(rows.locator(".ym-input-row__num")).toHaveText(["2", "1"]);
    await expect(rows.locator(".ym-input-row__sub")).toHaveText(["On top", "Below"]);
  });

  test("Splitter and combiner adds the five nodes, selects the Splitter and draws the channel ports tinted with their letters", async ({ page }) => {
    await nodeView(page, {}, [["input", 1], [1, "output"]]);
    await addNode(page, /^Splitter and combiner/);
    await expect(card(page, 2)).toHaveAccessibleName("Splitter, picture from node 1");
    await expect(card(page, 2)).toHaveAttribute("aria-pressed", "true");
    await expect(canvas(page).locator(".ym-gnode__name")).toHaveText(["Main", "Splitter", "Red", "Green", "Blue", "Combiner"]);
    await expect(card(page, 6)).toHaveAccessibleName("Combiner, red from node 2, green from node 3, blue from node 4");
    await expect(card(page, 2).locator(".ym-gnode__letter")).toHaveText(["R", "G", "B"]);
    await expect(card(page, 6).locator(".ym-gnode__letter")).toHaveText(["R", "G", "B"]);
    // 56 px wide, 88 px high with three ports.
    const splitter = (await card(page, 2).boundingBox())!;
    const zoom = (await card(page, 1).boundingBox())!.width / 150;
    expect(Math.round(splitter.width / zoom)).toBe(56);
    expect(Math.round(splitter.height / zoom)).toBe(88);
    expect(await linkIds(page)).toContain("2.red>3.picture");
    expect(await linkIds(page)).toContain("5.picture>6.blue");
    await expect(tools(page).locator(".ym-nodehead__line")).toHaveText("Channels · 3 outputs");
    await undo(page);
    await expect(card(page, 2)).toHaveCount(0);
  });
});

test.describe("the cards", () => {
  test("are 56 px wide with the height of their ports, and stack the inputs 24 px apart with the '+' under the last", async ({ page }) => {
    await nodeView(page, LAYER_NODES, LAYER_LINKS);
    const box = (await card(page, 5).boundingBox())!;
    // The canvas is fitted to the window: the scale is a corrector's width over its 150 px.
    const zoom = (await card(page, 3).boundingBox())!.width / 150;
    // Three inputs and the "+": four ports, 112 px.
    expect(Math.round(box.width / zoom)).toBe(56);
    expect(Math.round(box.height / zoom)).toBe(112);
    const ys = await Promise.all(["picture.1", "picture.2", "picture.3", "picture.4"].map(async (p) => (await centreOf(portHit(page, 5, p, "in"))).y));
    expect((ys[1]! - ys[0]!) / zoom).toBeCloseTo(24, 0);
    expect((ys[2]! - ys[1]!) / zoom).toBeCloseTo(24, 0);
    expect((ys[3]! - ys[2]!) / zoom).toBeCloseTo(24, 0);
    await expect(canvas(page).locator('[data-plus="true"]')).toHaveCount(1);
    expect(await linkIds(page)).toContain("4.picture>5.picture.3");
    // No thumbnail, number or footer on a glyph card.
    await expect(card(page, 5).locator(".ym-gnode__thumb, .ym-gnode__num, .ym-gnode__foot")).toHaveCount(0);
    // The card is centred on the picture line of the corrector next to it.
    const glow = (await card(page, 3).boundingBox())!;
    expect(box.y + box.height / 2).toBeCloseTo(glow.y + 56 * zoom, 0);
  });

  test("a mixer with eight inputs has no '+'", async ({ page }) => {
    const nodes: Record<string, unknown> = { "2": { kind: "parallel", pos: [1, 0] } };
    const list: L[] = [["input", 1], [2, "output"]];
    for (let k = 1; k <= 8; k++) list.push([1, 2, `picture.${k}`]);
    await nodeView(page, nodes, list);
    await expect(canvas(page).locator('[data-plus="true"]')).toHaveCount(0);
    const box = (await card(page, 2).boundingBox())!;
    const main = (await card(page, 1).boundingBox())!;
    expect(Math.round(box.height / (main.width / 150))).toBe(208);
  });

  test("a bypassed mixer has the slash", async ({ page }) => {
    await nodeView(page, { ...PARALLEL_NODES, "4": { kind: "parallel", pos: [2, 0], bypass: true } }, PARALLEL_LINKS);
    await expect(card(page, 4).locator(".ym-gnode__slash")).toHaveCount(1);
    await expect(page.locator(".ym-gnode[data-dead]")).toHaveCount(0);
  });
});

test.describe("links by hand", () => {
  test("a link dragged from an output lights the picture inputs, the '+' and a Combiner's inputs, and dims the inputs that would close a circle", async ({ page }) => {
    await nodeView(page, { ...PARALLEL_NODES, "5": { kind: "combiner", pos: [2, 1.5] } }, [...PARALLEL_LINKS]);
    const from = await centreOf(portHit(page, 3, "picture", "out"));
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    await page.mouse.move(from.x + 60, from.y + 40, { steps: 6 });
    const state = (node: number | string, port: string) => page.locator(`.ym-port[data-port="${port}"][data-side="in"]`).and(page.locator(`[data-state]`)).filter({ has: page.locator(`[data-port-node="${node}"]`) });
    await expect(page.locator('.ym-port[data-state="lit"]')).not.toHaveCount(0);
    // The Combiner's three inputs and the '+' light; Main's and Sky's own inputs (upstream of Sky) dim; a key port dims.
    await expect(state(5, "red")).toHaveAttribute("data-state", "lit");
    await expect(state(5, "blue")).toHaveAttribute("data-state", "lit");
    await expect(state(4, "picture.3")).toHaveAttribute("data-state", "lit");
    await expect(page.locator('.ym-port[data-port="picture"][data-side="in"]').filter({ has: page.locator('[data-port-node="3"]') })).toHaveAttribute("data-state", "dim");
    await expect(page.locator('.ym-port[data-port="key"][data-side="in"][data-state="lit"]')).toHaveCount(0);
    await page.keyboard.press("Escape");
    await page.mouse.up();
    await expect(page.locator(".ym-port[data-state]")).toHaveCount(0);
  });

  test("a drop on the '+' adds the next numbered input; removing a link renumbers; each is one undo step", async ({ page }) => {
    await nodeView(page, PARALLEL_NODES, PARALLEL_LINKS);
    // Main's output to the Parallel's "+" (picture.3).
    await dragBetween(page, await centreOf(portHit(page, 1, "picture", "out")), await centreOf(portHit(page, 4, "picture.3", "in")));
    await expect.poll(() => linkIds(page)).toContain("1.picture>4.picture.3");
    await expect(card(page, 4)).toHaveAccessibleName("Parallel, mixer with 3 inputs, from node 2, node 3 and node 1");
    await expect(canvas(page).locator('[data-plus="true"]')).toHaveCount(1);
    await expect(canvas(page).locator(`.ym-port[data-port="picture.4"][data-plus="true"]`)).toHaveCount(1);
    await undo(page);
    await expect.poll(() => linkIds(page)).not.toContain("1.picture>4.picture.3");
    // Pick input 1 up and drop it on the empty canvas: the link goes and input 2 becomes input 1.
    const empty = await centreOf(page.locator(".react-flow__pane"));
    await dragBetween(page, await centreOf(portHit(page, 4, "picture.1", "in")), { x: empty.x + 100, y: empty.y + 200 });
    await expect.poll(() => linkIds(page)).toEqual(["1.picture>2.picture", "1.picture>3.picture", "3.picture>4.picture.1", "4.picture>output.picture", "input.picture>1.picture"]);
    await expect(card(page, 4)).toHaveAccessibleName("Parallel, mixer with 1 input, from node 3");
    await undo(page);
    await expect.poll(() => linkIds(page)).toContain("2.picture>4.picture.1");
  });

  test("More > Link picture to lists the inputs of a mixer by name, with the next one as 'a new input'", async ({ page }) => {
    await nodeView(page, PARALLEL_NODES, PARALLEL_LINKS);
    await card(page, 1).click();
    await page.getByRole("button", { name: "More" }).last().click();
    await page.getByRole("menuitem", { name: "Link picture to" }).click();
    await expect(page.getByRole("menuitem", { name: "Parallel, a new input" })).toBeVisible();
    await clickSubItem(page, page.getByRole("menuitem", { name: "Parallel, a new input" }));
    await expect.poll(() => linkIds(page)).toContain("1.picture>4.picture.3");
    // Remove link lists the same names.
    await card(page, 4).click();
    await page.getByRole("button", { name: "More" }).last().click();
    await page.getByRole("menuitem", { name: "Remove link" }).click();
    await expect(page.getByRole("menuitem", { name: "Picture from Node 2, Skin" })).toBeVisible();
    await expect(page.getByRole("menuitem", { name: "Picture to Output" })).toBeVisible();
  });
});

test.describe("the Layer's panel", () => {
  test("sets the Mode from seven names and the Opacity, and Reset puts them back", async ({ page }) => {
    await nodeView(page, LAYER_NODES, LAYER_LINKS);
    await card(page, 5).click();
    await expect(tools(page).locator(".ym-nodehead__line")).toHaveText("Mixer · 3 inputs");
    await expect(tools(page).getByRole("tab")).toHaveText(["Blend"]);
    await tools(page).getByRole("combobox", { name: "Blend mode" }).click();
    await expect(page.getByRole("option")).toHaveText(["Normal", "Add", "Subtract", "Multiply", "Lighten", "Darken", "Difference"]);
    await page.getByRole("option", { name: "Multiply" }).click();
    await expect(tools(page).getByRole("combobox", { name: "Blend mode" })).toContainText("Multiply");
    const opacity = tools(page).getByRole("textbox", { name: "Opacity value" });
    await opacity.fill("40");
    await opacity.press("Enter");
    await expect(opacity).toHaveValue("40%");
    await tools(page).getByRole("region", { name: "Blend" }).getByRole("button", { name: "Reset", exact: true }).click();
    await expect(tools(page).getByRole("combobox", { name: "Blend mode" })).toContainText("Normal");
    await expect(opacity).toHaveValue("100%");
    await expect(page.getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-disabled", "true");
  });

  test("dragging a row over another renumbers the ports in one undo step; the card's ports follow", async ({ page }) => {
    await nodeView(page, LAYER_NODES, LAYER_LINKS);
    await card(page, 5).click();
    const rows = tools(page).locator("[data-input-row]");
    // From the top of the stack: Sky (3), Glow (2), Warmth (1).
    await expect(rows.locator(".ym-input-row__name")).toHaveText(["04 Sky", "03 Glow", "02 Warmth"]);
    // Drag Warmth (the bottom) over the top row.
    const grip = rows.nth(2).locator("[data-grip]");
    const top = (await rows.nth(0).boundingBox())!;
    await dragBetween(page, await centreOf(grip), { x: top.x + 60, y: top.y + 6 }, 10);
    await expect(rows.locator(".ym-input-row__name")).toHaveText(["02 Warmth", "04 Sky", "03 Glow"]);
    expect(await linkIds(page)).toEqual(
      ["1.picture>2.picture", "1.picture>3.picture", "1.picture>4.picture", "2.picture>5.picture.3", "3.picture>5.picture.1", "4.picture>5.picture.2", "5.picture>output.picture", "input.picture>1.picture"].sort(),
    );
    await undo(page);
    await expect(rows.locator(".ym-input-row__name")).toHaveText(["04 Sky", "03 Glow", "02 Warmth"]);
  });

  test("a row's context menu has Move up, Move down and Remove input", async ({ page }) => {
    await nodeView(page, LAYER_NODES, LAYER_LINKS);
    await card(page, 5).click();
    const rows = tools(page).locator("[data-input-row]");
    await rows.nth(1).click({ button: "right" });
    await expect(page.getByRole("menuitem")).toHaveText(["Move up", "Move down", "Remove input"]);
    await page.getByRole("menuitem", { name: "Remove input" }).click();
    await expect(rows).toHaveCount(2);
    await expect(card(page, 5)).toHaveAccessibleName("Layer, mixer with 2 inputs, from node 2 and node 4");
  });
});

test.describe("Delete and the refusals", () => {
  test("deleting a Layer leaves its bottom picture to the Output, and selects it", async ({ page }) => {
    await nodeView(page, LAYER_NODES, LAYER_LINKS);
    await card(page, 5).click();
    await page.getByRole("button", { name: "Delete", exact: true }).click();
    await expect(card(page, 5)).toHaveCount(0);
    expect(await linkIds(page)).toContain("2.picture>output.picture");
    await expect(card(page, 2)).toHaveAttribute("aria-pressed", "true");
    await undo(page);
    await expect(card(page, 5)).toBeVisible();
  });

  test("a Parallel is bypassed from the toolbar and B; a Splitter and a Combiner say they can't be", async ({ page }) => {
    await nodeView(page, { ...PARALLEL_NODES, "5": { kind: "splitter", pos: [1, 2] } }, [...PARALLEL_LINKS, [1, 5]]);
    await card(page, 4).click();
    await page.getByRole("button", { name: "Bypass", exact: true }).click();
    await expect(card(page, 4)).toHaveAccessibleName(/^Parallel, mixer with 2 inputs, bypassed/);
    await expect(tools(page).locator(".ym-nodehead__line")).toHaveText("Mixer · bypassed");
    await card(page, 5).click();
    const bypass = page.getByRole("button", { name: "Bypass", exact: true });
    await expect(bypass).toHaveAttribute("aria-disabled", "true");
    await expect(tools(page).getByRole("button", { name: "Bypass this node" })).toHaveAttribute("aria-disabled", "true");
    await expect(tools(page).getByRole("button", { name: "Reset this node" })).toHaveAttribute("aria-disabled", "true");
  });

  test("Move earlier and later are not in the More menu of the four kinds, and dropping one on a link says why it can't", async ({ page }) => {
    await nodeView(page, PARALLEL_NODES, PARALLEL_LINKS);
    await card(page, 4).click();
    await tools(page).getByRole("button", { name: "More" }).click();
    await expect(page.getByRole("menuitem", { name: /^Move (earlier|later)/ })).toHaveCount(0);
    await expect(page.getByRole("menuitem")).toHaveText(["Rename", "Link picture to", "Remove link", "Delete node"]);
    await page.keyboard.press("Escape");
    // Alt+Left is refused with the sentence.
    await card(page, 4).focus();
    await page.keyboard.press("Alt+ArrowLeft");
    await expect(page.getByRole("alert")).toContainText("This node can't be moved along a line. Change its links instead.");
  });
});
