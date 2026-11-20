import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, watchErrors } from "./helpers/app";

// The Colour Uniform node in a browser on the mock API (`docs/32` §2, §4 and §8 "the interface", milestone U, CU-11): the Add node menu,
// Pick colour and Esc, the Colour range row the first pick adds and the one undo for the whole pick, the target by dragging and by typing,
// the Compress sliders, Show mask, the two Resets, the keyboard path, and Copy and Paste. The mock has no pipeline, so what the pictures
// show is the mock's own port of the maths; the tests look at the panel and the settings the page sends.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

const picture = (from: number | "input", to: number | "output") => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

/** Main, then a Colour Uniform node (2), then the Output. */
const UNIFORM_GRAPH = (more: Record<string, unknown> = {}) => ({
  nodes: { "2": { kind: "color_uniform", name: "Uniform", pos: [1, 0], ...more } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

const PICKED = { hue_deg: 40, saturation: 12, lightness_ev: 0.2 };

/** The mock's settings of a photo (the page's own store holds the same). */
async function settingsOf(page: Page, id = 4): Promise<{ nodes: { items: Array<Record<string, unknown>> } }> {
  return page.evaluate(async (photo) => (await window.__yellowmoon!.mockApi.getSettings(photo)).settings as never, id);
}
const uniformOf = async (page: Page, id = 4): Promise<Record<string, any> | undefined> =>
  (await settingsOf(page, id)).nodes.items.find((i) => i.kind === "color_uniform") as Record<string, any> | undefined;

/** The node view of DSC04204 with a Colour Uniform node (2), selected, on its Compress tab. */
async function uniformView(page: Page, more: Record<string, unknown> = {}, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await configureMock(page, { initialSettings: { DSC04204: UNIFORM_GRAPH(more) } as never });
  await openPhotos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await page.getByRole("button", { name: /^Node 2, Uniform/ }).click();
  await expect(page.getByRole("button", { name: /^Node 2, Uniform/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("tab", { name: "Compress", exact: true })).toHaveAttribute("aria-selected", "true");
}

const card = (page: Page, name: string): Locator => page.getByRole("complementary", { name: "Tools" }).getByRole("region", { name });
const pickButton = (page: Page): Locator => card(page, "Target").getByRole("button", { name: "Pick colour" });
const pickLayer = (page: Page): Locator => page.getByRole("application", { name: /Pick colour for the target of Colour Uniform/ });
const field = (page: Page, label: string): Locator => card(page, "Target").getByRole("textbox", { name: `${label} value` });
const amount = (page: Page, label: string): Locator => card(page, "Compress").getByRole("slider", { name: label }).first();

/** Clicks the photo at a share of the pick layer. */
async function clickPhoto(page: Page, fx: number, fy: number): Promise<void> {
  const box = (await pickLayer(page).boundingBox())!;
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
}

async function type(page: Page, label: string, text: string): Promise<void> {
  const box = field(page, label);
  await box.click();
  await box.fill(text);
  await box.press("Enter");
}

test.describe("the Add node menu", () => {
  test("has Colour Uniform with its line; choosing it adds the node, selects it and starts Pick colour", async ({ page }) => {
    const problems = watchErrors(page);
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await page.getByRole("button", { name: "Add node" }).click();
    const menu = page.getByRole("menu", { name: "Add node" });
    await expect(menu).toBeVisible();
    const item = menu.getByRole("menuitem", { name: /^Colour Uniform/ });
    await expect(item).toContainText("Pulls colours toward one colour.");
    await item.click();
    await expect(page.getByRole("button", { name: /^Node 2, Uniform/ })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("tab")).toHaveText(["Compress", "Mask"]);
    await expect(page.getByRole("tab", { name: "Compress", exact: true })).toHaveAttribute("aria-selected", "true");
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "true");
    // It also looks pressed: the tooltip's trigger must not overwrite the toggle's data-state (0.2.4-beta.3: the first click turned it off).
    await expect(pickButton(page)).toHaveAttribute("data-state", "on");
    await expect(pickLayer(page)).toBeVisible();
    await pickButton(page).click();
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "false");
    await expect(pickButton(page)).toHaveAttribute("data-state", "off");
    await expect(pickLayer(page)).toHaveCount(0);
    await pickButton(page).click();
    await expect(pickButton(page)).toHaveAttribute("data-state", "on");
    await expect(pickLayer(page)).toBeVisible();
    await expect(card(page, "Target").getByText("No colour yet")).toBeVisible();
    expect((await uniformOf(page))?.target).toBeNull();
    // React Flow's attribution warning is the node view's own and appears on every page that shows the graph.
    expect(problems.filter((p) => !p.includes("React Flow"))).toEqual([]);
  });
});

test.describe("Pick colour", () => {
  test("a click sets the target, Compress 50 / 50 / 0 and one Colour range row, and one undo takes all three back; Esc ends the pick", async ({ page }) => {
    await uniformView(page);
    await expect(pickLayer(page)).toHaveCount(0);
    await pickButton(page).click();
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "true");
    await expect(pickLayer(page)).toBeVisible();
    await clickPhoto(page, 0.5, 0.3);

    // The card says how to go on, the amounts are on, and the pick stays on for the next click.
    await expect(card(page, "Target").getByText("Pick again on the photo, or set the numbers.")).toBeVisible();
    await expect(card(page, "Target").getByRole("img", { name: /^Target colour: / })).toBeVisible();
    await expect(amount(page, "Hue")).not.toHaveAttribute("data-disabled");
    await expect(pickLayer(page)).toBeVisible();
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "true");
    const node = await uniformOf(page);
    expect(node?.target).not.toBeNull();
    expect(node?.compress).toEqual({ hue: 50, saturation: 50, lightness: 0 });
    expect(node?.masks.items).toHaveLength(1);
    expect(node?.masks.items[0].kind).toBe("color_range");

    // The Colour range row is in the Mask tab, selected.
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(page.getByRole("region", { name: "Masks" }).getByRole("button", { name: /^Colour range mask/ })).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("tab", { name: "Compress", exact: true }).click();

    // One undo for the whole first pick.
    await page.keyboard.press("Escape");
    await expect(pickLayer(page)).toHaveCount(0);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await uniformOf(page))?.target).toBeNull();
    const undone = await uniformOf(page);
    expect(undone?.masks.items).toEqual([]);
    expect(undone?.compress).toEqual({ hue: 0, saturation: 0, lightness: 0 });
  });

  test("Esc ends the pick and the button goes off; so does a second press of the button", async ({ page }) => {
    await uniformView(page);
    await pickButton(page).click();
    await expect(pickLayer(page)).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(pickLayer(page)).toHaveCount(0);
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "false");
    await pickButton(page).click();
    await expect(pickLayer(page)).toBeVisible();
    await pickButton(page).click();
    await expect(pickLayer(page)).toHaveCount(0);
    // Nothing was picked, so nothing changed.
    expect((await uniformOf(page))?.target).toBeNull();
  });

  test("a second click moves the target and adds no second row; the pick leaves with the tab", async ({ page }) => {
    await uniformView(page);
    await pickButton(page).click();
    await clickPhoto(page, 0.5, 0.3);
    await expect.poll(async () => (await uniformOf(page))?.masks.items.length).toBe(1);
    const first = (await uniformOf(page))?.target;
    await clickPhoto(page, 0.5, 0.9);
    await expect.poll(async () => JSON.stringify((await uniformOf(page))?.target)).not.toBe(JSON.stringify(first));
    expect((await uniformOf(page))?.masks.items).toHaveLength(1);
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(pickLayer(page)).toHaveCount(0);
  });
});

test.describe("the target", () => {
  test("shows the values of a picked target, and a typed number sets that field and keeps the others", async ({ page }) => {
    await uniformView(page, { target: PICKED, compress: { hue: 50, saturation: 50, lightness: 0 } });
    await expect(field(page, "Hue")).toHaveValue("40°");
    await expect(field(page, "Saturation")).toHaveValue("12");
    await expect(field(page, "Lightness (EV)")).toHaveValue(/0\.20$/);
    await type(page, "Saturation", "20");
    await expect(field(page, "Saturation")).toHaveValue("20");
    await expect.poll(async () => (await uniformOf(page))?.target).toEqual({ ...PICKED, saturation: 20 });
    await type(page, "Hue", "300");
    await expect.poll(async () => (await uniformOf(page))?.target?.hue_deg).toBe(300);
    expect((await uniformOf(page))?.target.saturation).toBe(20);
  });

  test("with no target the fields show dashes, and a first number sets a target from the resting values", async ({ page }) => {
    await uniformView(page);
    for (const label of ["Hue", "Saturation", "Lightness (EV)"]) await expect(field(page, label)).toHaveValue("–");
    await expect(card(page, "Compress").getByText("Pick or set a target colour first.")).toBeVisible();
    await type(page, "Saturation", "15");
    await expect.poll(async () => (await uniformOf(page))?.target).toEqual({ hue_deg: 0, saturation: 15, lightness_ev: 0 });
    await expect(card(page, "Compress").getByText("At 100 % every colour in the mask takes the target's value.")).toBeVisible();
  });

  test("dragging a slider changes the target and folds into one undo step", async ({ page }) => {
    await uniformView(page, { target: PICKED });
    const slider = card(page, "Target").getByRole("slider", { name: "Hue" });
    const box = (await slider.boundingBox())!;
    const track = (await card(page, "Target").locator('[data-slot="slider-track"]').first().boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(track.x + track.width * 0.6, box.y + box.height / 2, { steps: 8 });
    await page.mouse.move(track.x + track.width * 0.8, box.y + box.height / 2, { steps: 8 });
    await page.mouse.up();
    await expect.poll(async () => (await uniformOf(page))?.target?.hue_deg).toBeGreaterThan(200);
    const dragged = (await uniformOf(page))?.target;
    expect(dragged.saturation).toBe(12);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await uniformOf(page))?.target).toEqual(PICKED);
  });

  test("Reset target clears the target only", async ({ page }) => {
    await uniformView(page, { target: PICKED, compress: { hue: 50, saturation: 50, lightness: 10 } });
    await card(page, "Target").getByRole("button", { name: "Reset target" }).click();
    await expect.poll(async () => (await uniformOf(page))?.target).toBeNull();
    expect((await uniformOf(page))?.compress).toEqual({ hue: 50, saturation: 50, lightness: 10 });
    await expect(card(page, "Target").getByText("No colour yet")).toBeVisible();
    await expect(card(page, "Target").getByRole("button", { name: "Reset target" })).toHaveCount(0);
  });
});

test.describe("Compress", () => {
  test("the sliders are off with no target; with one, a typed amount and the keyboard change it, and Reset Compress sets all three to 0", async ({ page }) => {
    await uniformView(page);
    for (const label of ["Hue", "Saturation", "Lightness"]) await expect(amount(page, label)).toHaveAttribute("data-disabled");
    await expect(card(page, "Compress").getByRole("button", { name: "Reset Compress" })).toHaveCount(0);

    await type(page, "Hue", "40");
    await expect.poll(async () => (await uniformOf(page))?.target).not.toBeNull();
    await expect(amount(page, "Hue")).not.toHaveAttribute("data-disabled");
    await card(page, "Compress").getByRole("textbox", { name: "Hue value" }).fill("70");
    await card(page, "Compress").getByRole("textbox", { name: "Hue value" }).press("Enter");
    await expect.poll(async () => (await uniformOf(page))?.compress.hue).toBe(70);
    await amount(page, "Saturation").focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => (await uniformOf(page))?.compress.saturation).toBe(1);
    await page.keyboard.press("Shift+ArrowRight");
    await expect.poll(async () => (await uniformOf(page))?.compress.saturation).toBeGreaterThan(5);

    await card(page, "Compress").getByRole("button", { name: "Reset Compress" }).click();
    await expect.poll(async () => (await uniformOf(page))?.compress).toEqual({ hue: 0, saturation: 0, lightness: 0 });
    expect((await uniformOf(page))?.target).not.toBeNull();
    await expect(card(page, "Compress").getByRole("button", { name: "Reset Compress" })).toHaveCount(0);
  });

  test("the head's Reset puts the node back to its start", async ({ page }) => {
    await uniformView(page, { target: PICKED, compress: { hue: 50, saturation: 50, lightness: 0 } });
    await page.getByRole("group", { name: "Node" }).getByRole("button", { name: "Reset this node" }).click();
    await expect.poll(async () => (await uniformOf(page))?.target).toBeNull();
    expect((await uniformOf(page))?.compress).toEqual({ hue: 0, saturation: 0, lightness: 0 });
  });
});

test.describe("the Mask tab and Show mask", () => {
  const range = {
    kind: "color_range", mode: "add", invert: false, enabled: true, opacity: 1,
    center: [0.0872, 0.0425], width: 0.05, softness: 0.5,
  };

  test("a node with a Colour range row shows it; Show mask draws the mask over the photo and turns off again", async ({ page }) => {
    await uniformView(page, { target: PICKED, compress: { hue: 50, saturation: 50, lightness: 0 }, masks: { invert: false, items: [range] } });
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(page.getByRole("region", { name: "Masks" }).getByRole("button", { name: /^Colour range mask/ })).toBeVisible();
    const show = page.getByRole("button", { name: "Show mask" });
    await expect(show).toHaveAttribute("aria-pressed", "false");
    await show.click();
    await expect(show).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator(".ym-photo-cell__image")).toHaveAttribute("src", /data-mask-view/);
    await show.click();
    await expect(show).toHaveAttribute("aria-pressed", "false");
    await expect(page.locator(".ym-photo-cell__image")).not.toHaveAttribute("src", /data-mask-view/);
  });

  test("with no mask row the Mask tab has no Colour range row and offers Add mask", async ({ page }) => {
    await uniformView(page, { target: PICKED });
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(page.getByRole("region", { name: "Masks" }).getByRole("button", { name: /^Colour range mask/ })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Add mask" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-pressed", "false");
  });
});

test.describe("the keyboard path", () => {
  test("Tab reaches Pick colour; Return starts the pick; arrows move the crosshair and Return picks; Esc ends", async ({ page }) => {
    await uniformView(page);
    await pickButton(page).focus();
    await expect(pickButton(page)).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(pickLayer(page)).toBeFocused();
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("Enter");
    await expect.poll(async () => (await uniformOf(page))?.target).not.toBeNull();
    expect((await uniformOf(page))?.masks.items).toHaveLength(1);
    // The pick's patch is still coming back to the page when Esc is pressed on a slow run: press until it takes.
    await expect(async () => {
      await page.keyboard.press("Escape");
      await expect(pickLayer(page)).toHaveCount(0, { timeout: 1000 });
    }).toPass();
  });

  test("the fields and the sliders follow in the tab order, each one named, and a slider moves by its arrows", async ({ page }) => {
    await uniformView(page, { target: PICKED, compress: { hue: 50, saturation: 50, lightness: 0 } });
    const names: string[] = [];
    await pickButton(page).focus();
    for (let i = 0; i < 14; i++) {
      await page.keyboard.press("Tab");
      names.push(await page.evaluate(() => document.activeElement?.getAttribute("aria-label") ?? document.activeElement?.textContent ?? ""));
    }
    for (const wanted of ["Hue value", "Saturation value", "Lightness (EV) value"]) expect(names.join("|")).toContain(wanted);
    const slider = card(page, "Target").getByRole("slider", { name: "Saturation" });
    await slider.focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => (await uniformOf(page))?.target?.saturation).toBeCloseTo(12.1, 5);
  });
});

test.describe("Copy and Paste", () => {
  test("the node, its target, amounts and mask row go with the photo's settings to another photo", async ({ page }) => {
    await uniformView(page, { target: PICKED, compress: { hue: 40, saturation: 30, lightness: 10 } });
    const foot = page.getByRole("group", { name: "Copy and paste settings" });
    await foot.getByRole("button", { name: "Copy settings" }).click();
    // The filmstrip is the simple view's: leave the node view for it.
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await page.getByRole("navigation", { name: "Photos" }).getByRole("button", { name: "Show DSC04206" }).click();
    await expect(page.getByText(/^6 of 12 · /)).toBeVisible();
    expect(await uniformOf(page, 6)).toBeUndefined();
    await foot.getByRole("button", { name: "Paste" }).click();
    await expect.poll(async () => (await uniformOf(page, 6))?.target).toEqual(PICKED);
    expect((await uniformOf(page, 6))?.compress).toEqual({ hue: 40, saturation: 30, lightness: 10 });
    // One undo takes the pasted node away again.
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => await uniformOf(page, 6)).toBeUndefined();
  });
});

test.describe("deleting the node", () => {
  const range = {
    kind: "color_range", mode: "add", invert: false, enabled: true, opacity: 1,
    center: [0.0872, 0.0425], width: 0.05, softness: 0.5,
  };
  test("ends the view of its mask with it, and the photo goes on taking edits of Main", async ({ page }) => {
    await uniformView(page, { target: PICKED, compress: { hue: 50, saturation: 50, lightness: 0 }, masks: { invert: false, items: [range] } });
    // What the page tells the viewport, in order.
    await page.evaluate(() => {
      const api = window.__yellowmoon!.mockApi as unknown as Record<string, (...a: unknown[]) => unknown>;
      const log: string[] = [];
      (window as unknown as { __calls: string[] }).__calls = log;
      for (const name of ["viewportSetMode", "updateSettings"]) {
        const call = api[name]!;
        api[name] = (...a: unknown[]) => {
          log.push(name === "viewportSetMode" ? `${name} ${JSON.stringify(a[1] ?? null)}` : `${name} ${JSON.stringify(a[1])}`);
          return call.apply(api, a);
        };
      }
    });
    const calls = (): Promise<string[]> => page.evaluate(() => (window as unknown as { __calls: string[] }).__calls);
    const show = page.getByRole("button", { name: "Show mask" });
    await show.click();
    await expect(page.locator(".ym-photo-cell__image")).toHaveAttribute("src", /data-mask-view/);
    await page.getByRole("complementary", { name: "Tools" }).getByRole("button", { name: /more/i }).first().click();
    await page.getByRole("menuitem", { name: "Delete node" }).click();
    await expect.poll(async () => await uniformOf(page)).toBeUndefined();
    // The view of node 2 ended before the patch that deleted it was sent, and nothing named node 2 again (Show mask stays on, now for Main).
    const log = await calls();
    const patch = log.findIndex((c) => c.startsWith("updateSettings") && c.includes('"remove":[2]'));
    expect(patch).toBeGreaterThan(0);
    expect(log[patch - 1]).toBe("viewportSetMode null");
    expect(log.slice(patch).filter((c) => c.startsWith("viewportSetMode") && c.includes('"node":2'))).toEqual([]);
    // An edit of Main after it reaches the core.
    const slider = page.getByRole("complementary", { name: "Tools" }).getByRole("slider").first();
    await slider.focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => (await calls()).filter((c) => c.startsWith("updateSettings") && c.includes('"1":')).length).toBeGreaterThan(0);
  });
});
