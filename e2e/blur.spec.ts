import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, watchErrors } from "./helpers/app";
import { BLUR_MASKS } from "./helpers/states";

// The Blur node in a browser on the mock API (`docs/33` §2 and §8 "the interface", milestone B, BL-15): the Add node menu, the Type control
// (switching type and back keeps every type's values), the sliders, Lens's Shape, Pick centre and Esc, a drag of the Zoom centre handle, the
// Respect mask edges switch and its lines, Show mask, the two Resets, the keyboard path, and Copy and Paste. The mock draws no blur, so the tests
// look at the panel and at the settings the page sends.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

const picture = (from: number | "input", to: number | "output") => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

/** Main, then a Blur node (2), then the Output. */
const BLUR_GRAPH = (more: Record<string, unknown> = {}) => ({
  nodes: { "2": { kind: "blur", name: "Blur", pos: [1, 0], ...more } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

/** The mock's settings of a photo (the page's own store holds the same). */
async function settingsOf(page: Page, id = 4): Promise<{ nodes: { items: Array<Record<string, unknown>> } }> {
  return page.evaluate(async (photo) => (await window.__yellowmoon!.mockApi.getSettings(photo)).settings as never, id);
}
const blurOf = async (page: Page, id = 4): Promise<Record<string, any> | undefined> =>
  (await settingsOf(page, id)).nodes.items.find((i) => i.kind === "blur") as Record<string, any> | undefined;

/** The node view of DSC04204 with a Blur node (2), selected, on its Blur tab. */
async function blurView(page: Page, more: Record<string, unknown> = {}): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, { initialSettings: { DSC04204: BLUR_GRAPH(more) } as never });
  await openPhotos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await page.getByRole("button", { name: /^Node 2, Blur/ }).click();
  await expect(page.getByRole("button", { name: /^Node 2, Blur/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("tab", { name: "Blur", exact: true })).toHaveAttribute("aria-selected", "true");
}

const card = (page: Page, name: string): Locator => page.getByRole("complementary", { name: "Tools" }).getByRole("region", { name });
const typeControl = (page: Page): Locator => card(page, "Blur").getByRole("radiogroup", { name: "Blur type" });
const typeRadio = (page: Page, name: string): Locator => typeControl(page).getByRole("radio", { name, exact: true });
const slider = (page: Page, label: string): Locator => card(page, "Blur").getByRole("slider", { name: label }).first();
const value = (page: Page, label: string): Locator => card(page, "Blur").getByRole("textbox", { name: `${label} value` });
const pickButton = (page: Page): Locator => card(page, "Blur").getByRole("button", { name: "Pick centre" });
const catcher = (page: Page): Locator => page.locator("[data-zoom-catch]");
const handle = (page: Page): Locator => page.getByRole("button", { name: /^Zoom centre/ });
const edges = (page: Page): Locator => card(page, "Mask edges").getByRole("switch", { name: "Respect mask edges" });
const cardReset = (page: Page): Locator => card(page, "Blur").getByRole("button", { name: "Reset Blur" });
/** Lets go of the focus so that a key press is the window's (a text box leaves ⌘Z to itself). */
const blur = (page: Page) => page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());

async function type(page: Page, label: string, text: string): Promise<void> {
  const box = value(page, label);
  await box.click();
  await box.fill(text);
  await box.press("Enter");
}

test.describe("the Add node menu", () => {
  test("has Blur with its line; choosing it adds the node after Main and selects it on the Blur tab with the start values", async ({ page }) => {
    const problems = watchErrors(page);
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await page.getByRole("button", { name: "Add node" }).click();
    const menu = page.getByRole("menu", { name: "Add node" });
    await expect(menu).toBeVisible();
    const item = menu.getByRole("menuitem", { name: /^Blur/ });
    await expect(item).toContainText("Soft, lens, motion or zoom blur.");
    await item.click();
    await expect(page.getByRole("button", { name: /^Node 2, Blur/ })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("tab")).toHaveText(["Blur", "Mask"]);
    await expect(page.getByRole("tab", { name: "Blur", exact: true })).toHaveAttribute("aria-selected", "true");
    await expect(typeRadio(page, "Soft")).toBeChecked();
    await expect(value(page, "Radius")).toHaveValue("20");
    await expect(edges(page)).toBeChecked();
    const node = await blurOf(page);
    expect(node?.type).toBe("soft");
    expect(node?.soft).toEqual({ radius: 20 });
    expect(node?.respect_mask_edges).toBe(true);
    // React Flow's attribution warning is the node view's own and appears on every page that shows the graph.
    expect(problems.filter((p) => !p.includes("React Flow"))).toEqual([]);
  });
});

test.describe("the Type control", () => {
  test("offers Soft, Lens, Motion and Zoom, each with its own controls, and switching changes the type alone", async ({ page }) => {
    await blurView(page);
    await expect(typeControl(page).getByRole("radio")).toHaveText(["Soft", "Lens", "Motion", "Zoom"]);
    await expect(typeControl(page).getByRole("radio", { disabled: true })).toHaveCount(0);
    await expect(slider(page, "Radius")).toBeVisible();

    await typeRadio(page, "Lens").click();
    await expect.poll(async () => (await blurOf(page))?.type).toBe("lens");
    for (const label of ["Radius", "Rotation", "Highlights"]) await expect(slider(page, label)).toBeVisible();
    await expect(card(page, "Blur").getByRole("radiogroup", { name: "Shape" })).toBeVisible();

    await typeRadio(page, "Motion").click();
    await expect.poll(async () => (await blurOf(page))?.type).toBe("motion");
    for (const label of ["Length", "Angle"]) await expect(slider(page, label)).toBeVisible();
    await expect(slider(page, "Radius")).toHaveCount(0);

    await typeRadio(page, "Zoom").click();
    await expect.poll(async () => (await blurOf(page))?.type).toBe("zoom");
    await expect(slider(page, "Amount")).toBeVisible();
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "false");
    await expect(card(page, "Blur").getByText("Centre of the picture.")).toBeVisible();
  });

  test("switching away and back keeps each type's values, and the patches carry the type alone", async ({ page }) => {
    await blurView(page);
    await type(page, "Radius", "55");
    await expect.poll(async () => (await blurOf(page))?.soft).toEqual({ radius: 55 });
    await typeRadio(page, "Lens").click();
    await type(page, "Highlights", "70");
    await expect.poll(async () => (await blurOf(page))?.lens.highlights).toBe(70);
    await typeRadio(page, "Motion").click();
    await type(page, "Length", "35");
    await expect.poll(async () => (await blurOf(page))?.motion.length).toBe(35);
    await typeRadio(page, "Soft").click();
    await expect(value(page, "Radius")).toHaveValue("55");
    await typeRadio(page, "Lens").click();
    await expect(value(page, "Highlights")).toHaveValue("70");
    await typeRadio(page, "Motion").click();
    await expect(value(page, "Length")).toHaveValue("35");
    const node = await blurOf(page);
    expect(node?.type).toBe("motion");
    expect(node?.soft).toEqual({ radius: 55 });
    expect(node?.lens).toMatchObject({ radius: 20, highlights: 70 });
    expect(node?.motion).toMatchObject({ length: 35, angle_deg: 0 });
  });
});

test.describe("the sliders", () => {
  test("a typed number sets the value, the arrows and Shift+arrows step it, and dragging folds into one undo step", async ({ page }) => {
    await blurView(page);
    await type(page, "Radius", "40");
    await expect.poll(async () => (await blurOf(page))?.soft.radius).toBe(40);
    await slider(page, "Radius").focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => (await blurOf(page))?.soft.radius).toBe(41);
    await page.keyboard.press("Shift+ArrowRight");
    await expect.poll(async () => (await blurOf(page))?.soft.radius).toBeGreaterThan(45);
    await page.keyboard.press("Home");
    await expect.poll(async () => (await blurOf(page))?.soft.radius).toBe(0);

    await type(page, "Radius", "20");
    await expect.poll(async () => (await blurOf(page))?.soft.radius).toBe(20);
    const thumb = (await slider(page, "Radius").boundingBox())!;
    const track = (await card(page, "Blur").locator('[data-slot="slider-track"]').first().boundingBox())!;
    await page.mouse.move(thumb.x + thumb.width / 2, thumb.y + thumb.height / 2);
    await page.mouse.down();
    await page.mouse.move(track.x + track.width * 0.5, thumb.y + thumb.height / 2, { steps: 8 });
    await page.mouse.move(track.x + track.width * 0.8, thumb.y + thumb.height / 2, { steps: 8 });
    await page.mouse.up();
    await expect.poll(async () => (await blurOf(page))?.soft.radius).toBeGreaterThan(60);
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await blurOf(page))?.soft.radius).toBe(20);
  });

  test("Motion's Angle and Length, and Zoom's Amount, write their own group", async ({ page }) => {
    await blurView(page, { type: "motion" });
    await type(page, "Angle", "90");
    await expect.poll(async () => (await blurOf(page))?.motion.angle_deg).toBe(90);
    await type(page, "Length", "60");
    await expect.poll(async () => (await blurOf(page))?.motion.length).toBe(60);
    await typeRadio(page, "Zoom").click();
    await type(page, "Amount", "75");
    await expect.poll(async () => (await blurOf(page))?.zoom.amount).toBe(75);
    expect((await blurOf(page))?.zoom.center).toEqual([0.5, 0.5]);
  });
});

test.describe("Lens's Shape", () => {
  const shape = (page: Page): Locator => card(page, "Blur").getByRole("radiogroup", { name: "Shape" });

  test("Round and 5 to 9 blades; Rotation is off with Round and its value is kept", async ({ page }) => {
    await blurView(page, { type: "lens" });
    await expect(shape(page).getByRole("radio")).toHaveCount(6);
    await expect(shape(page).getByRole("radio", { name: "Round" })).toBeChecked();
    await expect(slider(page, "Rotation")).toHaveAttribute("data-disabled");

    await shape(page).getByRole("radio", { name: "6 blades" }).click();
    await expect.poll(async () => (await blurOf(page))?.lens.blades).toBe(6);
    await expect(slider(page, "Rotation")).not.toHaveAttribute("data-disabled");
    await type(page, "Rotation", "30");
    await expect.poll(async () => (await blurOf(page))?.lens.rotation_deg).toBe(30);

    await shape(page).getByRole("radio", { name: "Round" }).click();
    await expect.poll(async () => (await blurOf(page))?.lens.blades).toBe(0);
    await expect(slider(page, "Rotation")).toHaveAttribute("data-disabled");
    expect((await blurOf(page))?.lens.rotation_deg).toBe(30);

    await shape(page).getByRole("radio", { name: "9 blades" }).click();
    await expect.poll(async () => (await blurOf(page))?.lens.blades).toBe(9);
    await expect(value(page, "Rotation")).toHaveValue("30°");
  });
});

test.describe("Pick centre and the Zoom centre handle", () => {
  const zoom = { type: "zoom", zoom: { amount: 20, center: [0.5, 0.5] } };

  /** The photo cell's box in the page. */
  const cellBox = async (page: Page) => (await page.locator("[data-viewport]").boundingBox())!;

  test("the handle is on the photo of a Zoom node and on no other type", async ({ page }) => {
    await blurView(page, zoom);
    await expect(handle(page)).toBeVisible();
    const cell = await cellBox(page);
    const box = (await handle(page).boundingBox())!;
    expect(box.x + box.width / 2).toBeCloseTo(cell.x + cell.width / 2, 0);
    expect(box.y + box.height / 2).toBeCloseTo(cell.y + cell.height / 2, 0);
    await typeRadio(page, "Soft").click();
    await expect(handle(page)).toHaveCount(0);
  });

  test("Pick centre on, a click on the photo sets the centre as one undo step and ends the pick; Reset centre puts it back", async ({ page }) => {
    await blurView(page, zoom);
    await expect(catcher(page)).toHaveCount(0);
    await pickButton(page).click();
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "true");
    await expect(catcher(page)).toBeVisible();
    await expect(card(page, "Blur").getByText("Click the photo or drag the handle.")).toBeVisible();
    const cell = await cellBox(page);
    await page.mouse.click(cell.x + cell.width * 0.25, cell.y + cell.height * 0.25);
    await expect.poll(async () => (await blurOf(page))?.zoom.center[0]).toBeCloseTo(0.25, 1);
    expect((await blurOf(page))?.zoom.center[1]).toBeCloseTo(0.25, 1);
    await expect(catcher(page)).toHaveCount(0);
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "false");
    await expect(card(page, "Blur").getByText("Moved from the centre.")).toBeVisible();

    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await blurOf(page))?.zoom.center).toEqual([0.5, 0.5]);
    await page.keyboard.press("Meta+Shift+z");
    await expect.poll(async () => (await blurOf(page))?.zoom.center[0]).toBeCloseTo(0.25, 1);
    await card(page, "Blur").getByRole("button", { name: "Reset centre" }).click();
    await expect.poll(async () => (await blurOf(page))?.zoom.center).toEqual([0.5, 0.5]);
    await expect(card(page, "Blur").getByRole("button", { name: "Reset centre" })).toHaveCount(0);
  });

  test("Esc ends the pick and changes nothing; so does a second press of the button", async ({ page }) => {
    await blurView(page, zoom);
    await pickButton(page).click();
    await expect(catcher(page)).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(catcher(page)).toHaveCount(0);
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "false");
    await pickButton(page).click();
    await expect(catcher(page)).toBeVisible();
    await pickButton(page).click();
    await expect(catcher(page)).toHaveCount(0);
    expect((await blurOf(page))?.zoom.center).toEqual([0.5, 0.5]);
  });

  test("the pick leaves with the Mask tab", async ({ page }) => {
    await blurView(page, zoom);
    await pickButton(page).click();
    await expect(catcher(page)).toBeVisible();
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(catcher(page)).toHaveCount(0);
    await expect(handle(page)).toHaveCount(0);
  });

  test("a drag of the handle moves the centre, the handle is where the centre is, and one undo takes it back", async ({ page }) => {
    await blurView(page, zoom);
    const cell = await cellBox(page);
    const box = (await handle(page).boundingBox())!;
    const [x, y] = [box.x + box.width / 2, box.y + box.height / 2];
    await page.mouse.move(x, y);
    await page.mouse.down();
    for (const k of [0.25, 0.5, 0.75, 1]) await page.mouse.move(x + 100 * k, y + 60 * k);
    await page.mouse.up();
    await expect.poll(async () => (await blurOf(page))?.zoom.center[0]).toBeCloseTo(0.5 + 100 / cell.width, 2);
    expect((await blurOf(page))?.zoom.center[1]).toBeCloseTo(0.5 + 60 / cell.height, 2);
    const moved = (await handle(page).boundingBox())!;
    expect(moved.x + moved.width / 2).toBeCloseTo(x + 100, 0);
    expect(moved.y + moved.height / 2).toBeCloseTo(y + 60, 0);
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await blurOf(page))?.zoom.center).toEqual([0.5, 0.5]);
  });

  test("the arrow keys move the focused handle by a pixel, Shift by ten, and switch no photo", async ({ page }) => {
    await blurView(page, zoom);
    const cell = await cellBox(page);
    await handle(page).focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => (await blurOf(page))?.zoom.center[0]).toBeCloseTo(0.5 + 1 / cell.width, 3);
    await page.keyboard.press("Shift+ArrowDown");
    await expect.poll(async () => (await blurOf(page))?.zoom.center[1]).toBeCloseTo(0.5 + 10 / cell.height, 3);
    // The node view stays on the same photo and node: no arrow reached the photo switcher.
    await expect(page.getByRole("button", { name: /^Node 2, Blur/ })).toHaveAttribute("aria-pressed", "true");
    expect(await blurOf(page, 3)).toBeUndefined();
  });
});

test.describe("Respect mask edges", () => {
  test("on by default with its line; with no mask a second line says it changes nothing yet; the switch writes the field", async ({ page }) => {
    await blurView(page);
    const mask = card(page, "Mask edges");
    await expect(edges(page)).toBeChecked();
    await expect(mask.getByText("Colours from outside the mask are not blurred in, so there is no glow around the subject.")).toBeVisible();
    await expect(mask.getByText("This node has no mask, so this changes nothing yet.")).toBeVisible();
    await edges(page).click();
    await expect(edges(page)).not.toBeChecked();
    await expect.poll(async () => (await blurOf(page))?.respect_mask_edges).toBe(false);
    await expect(mask.getByText("The whole picture is blurred, then the mask decides where the blur shows.")).toBeVisible();
    await expect(mask.getByText(/no glow around the subject/)).toHaveCount(0);
    await edges(page).click();
    await expect.poll(async () => (await blurOf(page))?.respect_mask_edges).toBe(true);
  });

  test("a node with a mask row has no 'no mask' line", async ({ page }) => {
    await blurView(page, { masks: BLUR_MASKS });
    await expect(card(page, "Mask edges").getByText("Colours from outside the mask are not blurred in, so there is no glow around the subject.")).toBeVisible();
    await expect(card(page, "Mask edges").getByText(/no mask, so this changes nothing/)).toHaveCount(0);
  });
});

test.describe("Show mask", () => {
  test("a masked Blur node draws its mask over the photo and Show mask turns off again", async ({ page }) => {
    await blurView(page, { masks: BLUR_MASKS });
    const show = page.getByRole("button", { name: "Show mask" });
    await expect(show).toHaveAttribute("aria-pressed", "false");
    await show.click();
    await expect(show).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator(".ym-photo-cell__image")).toHaveAttribute("src", /data-mask-view/);
    await show.click();
    await expect(show).toHaveAttribute("aria-pressed", "false");
    await expect(page.locator(".ym-photo-cell__image")).not.toHaveAttribute("src", /data-mask-view/);
  });
});

test.describe("Reset", () => {
  const changed = {
    type: "zoom",
    soft: { radius: 70 },
    lens: { radius: 50, blades: 7, rotation_deg: 20, highlights: 60 },
    motion: { length: 45, angle_deg: 30 },
    zoom: { amount: 55, center: [0.3, 0.6] },
    respect_mask_edges: false,
    masks: BLUR_MASKS,
  };
  const START = {
    type: "soft",
    soft: { radius: 20 },
    lens: { radius: 20, blades: 0, rotation_deg: 0, highlights: 0 },
    motion: { length: 20, angle_deg: 0 },
    zoom: { amount: 20, center: [0.5, 0.5] },
    respect_mask_edges: true,
  };

  test("the card's Reset sets the type and every value back and keeps the masks; at the start values it is not there", async ({ page }) => {
    await blurView(page, changed);
    await expect(cardReset(page)).toBeVisible();
    await cardReset(page).click();
    await expect.poll(async () => (await blurOf(page))?.type).toBe("soft");
    const node = (await blurOf(page))!;
    expect(node).toMatchObject(START);
    expect(node.masks.items).toHaveLength(1);
    await expect(cardReset(page)).toHaveCount(0);
    await expect(typeRadio(page, "Soft")).toBeChecked();
    await blur(page);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await blurOf(page))?.type).toBe("zoom");
    expect((await blurOf(page))?.zoom).toEqual({ amount: 55, center: [0.3, 0.6] });
  });

  test("the head's Reset puts the node back to its start", async ({ page }) => {
    await blurView(page, changed);
    await page.getByRole("group", { name: "Node" }).getByRole("button", { name: "Reset this node" }).click();
    await expect.poll(async () => (await blurOf(page))?.type).toBe("soft");
    expect(await blurOf(page)).toMatchObject(START);
  });
});

test.describe("the keyboard path", () => {
  test("the Type control moves by its arrows, Tab goes on through the sliders to the switch, and every control has its name", async ({ page }) => {
    await blurView(page);
    await typeRadio(page, "Soft").focus();
    await page.keyboard.press("ArrowRight");
    await expect(typeRadio(page, "Lens")).toBeChecked();
    await expect.poll(async () => (await blurOf(page))?.type).toBe("lens");
    await page.keyboard.press("End");
    await expect(typeRadio(page, "Zoom")).toBeChecked();
    await page.keyboard.press("Home");
    await expect(typeRadio(page, "Soft")).toBeChecked();

    const names: string[] = [];
    for (let i = 0; i < 6; i++) {
      await page.keyboard.press("Tab");
      names.push(await page.evaluate(() => document.activeElement?.getAttribute("aria-label") ?? document.activeElement?.textContent ?? ""));
    }
    expect(names.join("|")).toContain("Radius");
    expect(names.join("|")).toContain("Respect mask edges");
    await edges(page).focus();
    await page.keyboard.press("Space");
    await expect.poll(async () => (await blurOf(page))?.respect_mask_edges).toBe(false);
  });

  test("Zoom: Tab reaches Pick centre; Return starts the pick; Esc ends it", async ({ page }) => {
    await blurView(page, { type: "zoom" });
    await pickButton(page).focus();
    await expect(pickButton(page)).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "true");
    await expect(catcher(page)).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(catcher(page)).toHaveCount(0);
    await expect(pickButton(page)).toHaveAttribute("aria-pressed", "false");
  });
});

test.describe("Copy and Paste", () => {
  test("the node, its type, values, centre, switch and mask row go with the photo's settings to another photo", async ({ page }) => {
    const set = { type: "lens", lens: { radius: 45, blades: 6, rotation_deg: 15, highlights: 50 }, zoom: { amount: 35, center: [0.3, 0.6] }, respect_mask_edges: false, masks: BLUR_MASKS };
    await blurView(page, set);
    const foot = page.getByRole("group", { name: "Copy and paste settings" });
    await foot.getByRole("button", { name: "Copy settings" }).click();
    // The filmstrip is the simple view's: leave the node view for it.
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await page.getByRole("navigation", { name: "Photos" }).getByRole("button", { name: "Show DSC04206" }).click();
    await expect(page.getByText(/^6 of 12 · /)).toBeVisible();
    expect(await blurOf(page, 6)).toBeUndefined();
    await foot.getByRole("button", { name: "Paste" }).click();
    await expect.poll(async () => (await blurOf(page, 6))?.type).toBe("lens");
    const pasted = (await blurOf(page, 6))!;
    expect(pasted.lens).toEqual({ radius: 45, blades: 6, rotation_deg: 15, highlights: 50 });
    expect(pasted.zoom).toEqual({ amount: 35, center: [0.3, 0.6] });
    expect(pasted.respect_mask_edges).toBe(false);
    expect(pasted.masks.items).toHaveLength(1);
    // One undo takes the pasted node away again.
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => await blurOf(page, 6)).toBeUndefined();
  });
});
