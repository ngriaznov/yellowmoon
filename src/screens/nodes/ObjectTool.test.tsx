/**
 * The Object mask on the page (`docs/23` §4.4, §7.6, task N-18): the OBJECT card (the tool Add point · Remove point · Box with its line, then Refine
 * edges at 50 %, Feather, Grow and Opacity), the points and the box drawn on the photo, a click that is a `mask_find` and, when an object is found,
 * one row patch, ⌥-click, a drag with Box, the pill "Object found · 0.4 s", nothing found in the error bar, "Getting ready…" and "Finding…" under the
 * row, and what Esc, Crop, a pick and Before do. The screen runs on the mock, as in a browser: its `mask_find` finds the flat scene's mountain.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import type { MaskItem, NodeLink, SettingsPatch } from "../../api/types";
import { newMask } from "../../state/maskRows";
import { nodeOf, patchOf } from "../../state/settingsGraph";
import type { CorrectorItem } from "../../state/settingsGraph";
import { openPhotos, openTab, renderApp, tools, valueOf } from "../../test/harness";
import type { Harness, UpdateSpy } from "../../test/harness";
import { mockRect } from "../../test/raf";

let h: Harness | undefined;
let windowHeight = window.innerHeight;
beforeEach(() => {
  windowHeight = window.innerHeight;
  Object.defineProperty(window, "innerHeight", { value: 900, configurable: true, writable: true });
});
afterEach(() => {
  h?.unmount();
  h = undefined;
  Object.defineProperty(window, "innerHeight", { value: windowHeight, configurable: true, writable: true });
});

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const OBJECT = (more: Record<string, unknown> = {}): MaskItem => ({ ...newMask("object"), ...more }) as MaskItem;

/** Main, then Mountain (2, an Object row with a click on the hill) and Sky (3, no masks). */
const SETTINGS: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Mountain", pos: [1, 0], masks: { invert: false, items: [OBJECT({ points: [[0.5, 0.7, true]] })] } },
    "3": { kind: "corrector", name: "Sky", pos: [2, 0] },
  },
  links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, "output")],
};

/** DSC04204 in the node view on the Mask tab of the node `onNode`, the photo cell 600 × 400 (a 3:2 photo fills it: a place in the cell is x / 600, y / 400). */
async function setup(onNode: RegExp = /^Node 3, Sky/, options: MockOptions = {}, settings: SettingsPatch = SETTINGS): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: settings }, ...options });
  await openPhotos(h);
  await openTab(h, "Light");
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  fireEvent.click(screen.getByRole("button", { name: onNode }));
  await waitFor(() => expect(screen.getByRole("button", { name: onNode })).toHaveAttribute("aria-pressed", "true"));
  await h.user.click(within(tools()).getByRole("tab", { name: "Mask" }));
  await waitFor(() => expect(within(tools()).getByRole("tab", { name: "Mask" })).toHaveAttribute("aria-selected", "true"));
  const cell = document.querySelector<HTMLElement>("[data-viewport]")!;
  mockRect(cell, { left: 0, top: 0, width: 600, height: 400 });
  await h.api.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
  await act(async () => {
    fireEvent(window, new Event("resize"));
  });
  return vi.spyOn(h.api, "updateSettings");
}

const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const cardTitles = (): string[] => within(tools()).getAllByRole("heading", { level: 3 }).map((e) => e.textContent ?? "");
const card = (name: string): HTMLElement => within(tools()).getByRole("region", { name });
const rowList = (): HTMLElement => within(tools()).getByRole("list", { name: "Masks of this node" });
const rows = (node: number): MaskItem[] => (nodeOf(h!.store.currentSettings()!.settings, node) as CorrectorItem).masks.items;
const objectRow = (node: number, index: number): Extract<MaskItem, { kind: "object" }> => rows(node)[index] as never;
const layer = (): HTMLElement | null => document.querySelector<HTMLElement>("[data-object-layer]");
const dots = (): HTMLElement[] => [...document.querySelectorAll<HTMLElement>("[data-object-point]")];
async function undo(): Promise<void> {
  await h!.user.keyboard("{Meta>}z{/Meta}");
}
async function addObject(): Promise<void> {
  await h!.user.click(within(tools()).getByRole("button", { name: "Add mask" }));
  const menu = await screen.findByRole("menu", { name: "Add mask" });
  await h!.user.click(within(menu).getByRole("menuitem", { name: /^Object/ }));
}
/** A click on the photo: down and up at the place in the cell (CSS pixels), with ⌥ when asked. */
function click(x: number, y: number, alt = false): void {
  fireEvent.pointerDown(layer()!, { clientX: x, clientY: y, button: 0, buttons: 1, pointerId: 1, altKey: alt });
  fireEvent.pointerUp(window, { clientX: x, clientY: y, pointerId: 1 });
}
function drag(from: [number, number], to: [number, number]): void {
  fireEvent.pointerDown(layer()!, { clientX: from[0], clientY: from[1], button: 0, buttons: 1, pointerId: 1 });
  fireEvent.pointerMove(window, { clientX: (from[0] + to[0]) / 2, clientY: (from[1] + to[1]) / 2, pointerId: 1 });
  fireEvent.pointerMove(window, { clientX: to[0], clientY: to[1], pointerId: 1 });
  fireEvent.pointerUp(window, { clientX: to[0], clientY: to[1], pointerId: 1 });
}
/** The cell's place of a photo-frame point on the mock's hill: the middle of the hill is (0.5, 0.7) → (300, 280). */
const HILL: [number, number] = [300, 280];
const SKY: [number, number] = [120, 40];

describe("adding an Object row", () => {
  it("selects it, shows the OBJECT card and starts the tool: the layer is there and is not a control", async () => {
    const spy = await setup();
    await addObject();
    await waitFor(() => expect(patches(spy)).toEqual([patchOf(3, { masks: { items: [newMask("object")] } })]));
    expect(cardTitles()).toEqual(["Masks", "Object", "View"]);
    expect(h!.store.getState().objectTool).toBe(true);
    await waitFor(() => expect(layer()).not.toBeNull());
    expect(screen.getByText("MASK · OBJECT")).toBeInTheDocument();
    expect(within(rowList()).getAllByRole("listitem")[0]).toHaveTextContent("ObjectClick the object on this photo");
    expect(layer()!.matches("button, [tabindex], [role]")).toBe(false);
    expect(layer()!.querySelectorAll("button, [tabindex], input").length).toBe(0);
    expect(layer()).toHaveAttribute("data-tool", "add");
  });

  it("is on offer only where the core can make it: ?mockMasks=noai has no Object line in the Add mask menu", async () => {
    await setup(/^Node 3, Sky/, { maskKinds: ["background", "linear", "radial", "brush", "color_range", "brightness_range"] });
    await h!.user.click(within(tools()).getByRole("button", { name: "Add mask" }));
    const menu = await screen.findByRole("menu", { name: "Add mask" });
    expect(within(menu).queryByRole("menuitem", { name: /^Object/ })).toBeNull();
    expect(within(menu).getByRole("menuitem", { name: /^Background/ })).toBeInTheDocument();
  });
});

describe("the OBJECT card", () => {
  it("has Add point · Remove point · Box with its line, then Refine edges (50 %), Feather, Grow and Opacity", async () => {
    await setup();
    await addObject();
    const object = card("Object");
    expect(within(object).getByRole("radiogroup", { name: "Object tool" })).toBeInTheDocument();
    expect(within(object).getAllByRole("radio").map((r) => r.textContent)).toEqual(["Add point", "Remove point", "Box"]);
    expect(within(object).getByRole("radio", { name: "Add point" })).toBeChecked();
    expect(within(object).getByText("Click the photo to add to the object. ⌥-click to take a part away.")).toBeInTheDocument();
    const order = within(object)
      .getAllByRole("textbox")
      .map((t) => t.getAttribute("aria-label"));
    expect(order).toEqual(["Refine edges value", "Feather value", "Grow value", "Opacity value"]);
    expect([valueOf("Refine edges"), valueOf("Feather"), valueOf("Grow"), valueOf("Opacity")]).toEqual(["50%", "0", "0", "100%"]);
    expect(within(object).getByRole("button", { name: "Reset" })).toBeInTheDocument();
    await h!.user.click(within(object).getByRole("radio", { name: "Remove point" }));
    expect(within(object).getByText("Click the photo to take a part away from the object.")).toBeInTheDocument();
    expect(layer()).toHaveAttribute("data-tool", "remove");
    await h!.user.click(within(object).getByRole("radio", { name: "Box" }));
    expect(within(object).getByText("Drag a box around the object.")).toBeInTheDocument();
    expect(layer()).toHaveAttribute("data-tool", "box");
  });

  it("the tool is a window setting: choosing one sends no patch and makes no undo step", async () => {
    const spy = await setup();
    await addObject();
    await waitFor(() => expect(rows(3)).toHaveLength(1));
    spy.mockClear();
    const before = h!.store.currentSettings();
    await h!.user.click(within(card("Object")).getByRole("radio", { name: "Box" }));
    await h!.user.click(within(card("Object")).getByRole("radio", { name: "Add point" }));
    expect(spy).not.toHaveBeenCalled();
    expect(h!.store.currentSettings()).toBe(before);
  });

  it("Reset puts the row back as a new Object row in the same mode, with no point and no box, and the tool at Add point", async () => {
    const spy = await setup(/^Node 2, Mountain/);
    await h!.user.click(within(rowList()).getByRole("button", { name: /^Object mask, 1 point/ }));
    await h!.user.click(within(card("Object")).getByRole("radio", { name: "Box" }));
    spy.mockClear();
    await h!.user.click(within(card("Object")).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(objectRow(2, 0)).toEqual(newMask("object")));
    expect(patches(spy)).toEqual([patchOf(2, { masks: { edit: { "0": { invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, points: [], box: null } } } })]);
    expect(h!.store.getState().objectMode).toBe("add");
    await undo();
    await waitFor(() => expect(objectRow(2, 0).points).toEqual([[0.5, 0.7, true]]));
  });
});

describe("a click on the photo", () => {
  it("is a mask_find with the tool's sign and, found, one row patch and one undo step; the point is drawn with a plus; the pill says how long", async () => {
    const spy = await setup();
    await addObject();
    await waitFor(() => expect(layer()).not.toBeNull());
    await waitFor(() => expect(rows(3)).toHaveLength(1));
    const find = vi.spyOn(h!.api, "maskFind");
    spy.mockClear();
    click(...HILL);
    await waitFor(() => expect(objectRow(3, 0).points).toEqual([[0.5, 0.7, true]]));
    expect(find).toHaveBeenCalledWith(h!.store.currentSettings()!.photo_id, 3, 0, { kind: "point", x: 0.5, y: 0.7, positive: true });
    expect(patches(spy)).toEqual([patchOf(3, { masks: { edit: { "0": { points: [[0.5, 0.7, true]], box: null } } } })]);
    await waitFor(() => expect(dots()).toHaveLength(1));
    expect(dots()[0]).toHaveAttribute("data-object-point", "add");
    expect(dots()[0]!.querySelector("path")).toHaveAttribute("d", "M-4 0H4M0 -4V4");
    const pill = await screen.findByText("Object found · 0.4 s");
    expect(pill).toHaveAttribute("role", "status");
    expect(within(rowList()).getByRole("button", { name: "Object mask, 1 point" })).toBeInTheDocument();
    // One undo step takes the point away again.
    await undo();
    await waitFor(() => expect(objectRow(3, 0).points).toEqual([]));
    await waitFor(() => expect(dots()).toHaveLength(0));
  });

  it("⌥-click is a negative point, drawn with a minus, whatever the tool is; Remove point does the same without ⌥", async () => {
    await setup(/^Node 2, Mountain/);
    await h!.user.click(within(rowList()).getByRole("button", { name: /^Object mask, 1 point/ }));
    await waitFor(() => expect(layer()).not.toBeNull());
    click(330, 290, true);
    await waitFor(() => expect(objectRow(2, 0).points).toEqual([[0.5, 0.7, true], [0.55, 0.725, false]]));
    await waitFor(() => expect(dots().map((d) => d.dataset.objectPoint)).toEqual(["add", "remove"]));
    expect(dots()[1]!.querySelector("path")).toHaveAttribute("d", "M-4 0H4");
    await h!.user.click(within(card("Object")).getByRole("radio", { name: "Remove point" }));
    click(360, 300);
    await waitFor(() => expect(objectRow(2, 0).points.map((p) => p[2])).toEqual([true, false, false]));
  });

  it("nothing found adds no point, sends no patch and says so in the error bar", async () => {
    const spy = await setup();
    await addObject();
    await waitFor(() => expect(layer()).not.toBeNull());
    await waitFor(() => expect(rows(3)).toHaveLength(1));
    spy.mockClear();
    click(...SKY);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Nothing found there. Click another part of the object, or draw a box."));
    expect(spy).not.toHaveBeenCalled();
    expect(objectRow(3, 0).points).toEqual([]);
    expect(dots()).toHaveLength(0);
    expect(screen.queryByText(/^Object found/)).toBeNull();
  });

  it("a double click finds twice and does not zoom the photo", async () => {
    await setup();
    await addObject();
    await waitFor(() => expect(layer()).not.toBeNull());
    const zoom = vi.spyOn(h!.api, "viewportInput");
    fireEvent.doubleClick(layer()!, { clientX: HILL[0], clientY: HILL[1] });
    expect(zoom).not.toHaveBeenCalled();
  });
});

describe("a drag with the Box tool", () => {
  it("draws a rubber band while it goes and, on release, finds the object in the box: one patch with the box, drawn on the photo", async () => {
    const spy = await setup();
    await addObject();
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(within(card("Object")).getByRole("radio", { name: "Box" }));
    await waitFor(() => expect(rows(3)).toHaveLength(1));
    spy.mockClear();
    fireEvent.pointerDown(layer()!, { clientX: 150, clientY: 200, button: 0, buttons: 1, pointerId: 1 });
    fireEvent.pointerMove(window, { clientX: 400, clientY: 330, pointerId: 1 });
    const band = layer()!.querySelector("[data-object-band]");
    expect(band).not.toBeNull();
    expect(band).toHaveAttribute("x", "150");
    expect(band).toHaveAttribute("width", "250");
    fireEvent.pointerUp(window, { clientX: 400, clientY: 330, pointerId: 1 });
    await waitFor(() => expect(objectRow(3, 0).box).toEqual([0.25, 0.5, 0.6667, 0.825]));
    expect(patches(spy)).toEqual([patchOf(3, { masks: { edit: { "0": { points: [], box: [0.25, 0.5, 0.6667, 0.825] } } } })]);
    expect(layer()!.querySelector("[data-object-band]")).toBeNull();
    await waitFor(() => expect(layer()!.querySelector("[data-object-box]")).not.toBeNull());
    expect(within(rowList()).getByRole("button", { name: "Object mask, Box" })).toBeInTheDocument();
  });

  it("a click or a tiny drag with Box is not a find, and a box that finds nothing says so", async () => {
    const spy = await setup();
    await addObject();
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(within(card("Object")).getByRole("radio", { name: "Box" }));
    await waitFor(() => expect(rows(3)).toHaveLength(1));
    spy.mockClear();
    const find = vi.spyOn(h!.api, "maskFind");
    click(...HILL);
    drag([300, 280], [303, 283]);
    await act(async () => {
      await Promise.resolve();
    });
    expect(find).not.toHaveBeenCalled();
    drag([20, 20], [200, 90]);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Nothing found there."));
    expect(spy).not.toHaveBeenCalled();
    expect(objectRow(3, 0).box).toBeNull();
    // ⌥-click with Box is still a negative point.
    click(...HILL, true);
    await waitFor(() => expect(find).toHaveBeenCalledWith(expect.anything(), 3, 0, { kind: "point", x: 0.5, y: 0.7, positive: false }));
  });
});

describe("what the row says while the core works", () => {
  it("says Getting ready… while the encoder runs and Finding… while a click is answered, and then its own words", async () => {
    // A photo with no Object row yet: its first one starts the encoder (the mock's 0.9 s at speed 1), and the row says so.
    const noObject: SettingsPatch = { ...SETTINGS, nodes: { ...SETTINGS.nodes, "2": { kind: "corrector", name: "Mountain", pos: [1, 0], masks: { invert: false, items: [newMask("radial")] } } } };
    await setup(/^Node 3, Sky/, {}, noObject);
    h!.api.configure({ speed: 1 });
    await addObject();
    await waitFor(() => expect(within(rowList()).getByRole("button", { name: "Object mask, Getting ready…" })).toBeInTheDocument());
    expect(within(rowList()).getAllByRole("listitem")[0]).toHaveTextContent("ObjectGetting ready…");
    // A click is answered once the encoder has finished, and the row says Finding… from the click until then.
    await waitFor(() => expect(layer()).not.toBeNull());
    click(...HILL);
    await waitFor(() => expect(h!.store.getState().objectFinding).not.toBeNull());
    expect(within(rowList()).getAllByRole("listitem")[0]).toHaveTextContent("ObjectFinding…");
    await waitFor(() => expect(within(rowList()).getByRole("button", { name: /^Object mask, 1 point/ })).toBeInTheDocument(), { timeout: 4000 });
    expect(h!.store.getState().objectPreparing).toEqual([]);
  });
});

describe("Esc, Crop, a pick and Before", () => {
  it("Esc ends the tool and leaves the points drawn; the layer takes nothing then", async () => {
    await setup(/^Node 2, Mountain/);
    await h!.user.click(within(rowList()).getByRole("button", { name: /^Object mask, 1 point/ }));
    await waitFor(() => expect(layer()).toHaveAttribute("data-tool", "add"));
    await waitFor(() => expect(dots()).toHaveLength(1));
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(h!.store.getState().objectTool).toBe(false));
    expect(layer()).not.toHaveAttribute("data-tool");
    expect(dots()).toHaveLength(1);
    expect(within(card("Object")).getByText("Choose a tool to find an object on the photo.")).toBeInTheDocument();
    // The way back is on the card.
    await h!.user.click(within(card("Object")).getByRole("radio", { name: "Add point" }));
    expect(layer()).toHaveAttribute("data-tool", "add");
  });

  it("Crop, Before and a pick hide the points and the layer, and they come back with those gone", async () => {
    await setup(/^Node 2, Mountain/);
    await h!.user.click(within(rowList()).getByRole("button", { name: /^Object mask, 1 point/ }));
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(screen.getByRole("button", { name: "Crop" }));
    await waitFor(() => expect(layer()).toBeNull());
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(screen.getByRole("button", { name: "Before" }));
    await waitFor(() => expect(layer()).toBeNull());
    await h!.user.click(screen.getByRole("button", { name: "Before" }));
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(within(tools()).getByRole("tab", { name: "Colour" }));
    await h!.user.click(within(tools()).getByRole("button", { name: "Pick colour" }));
    await waitFor(() => expect(screen.getByRole("application", { name: /^Pick colour/ })).toBeInTheDocument());
    expect(layer()).toBeNull();
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(layer()).not.toBeNull());
    expect(h!.store.getState().objectTool).toBe(true);
  });

  it("a row with the eye off draws nothing and says so on the card", async () => {
    await setup(/^Node 2, Mountain/);
    await h!.user.click(within(rowList()).getByRole("button", { name: /^Object mask, 1 point/ }));
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(within(within(rowList()).getAllByRole("listitem")[0] as HTMLElement).getByRole("switch", { name: "Turn this mask off" }));
    await waitFor(() => expect(layer()).toBeNull());
    expect(within(card("Object")).getByText("This mask is off. Turn it on to find an object.")).toBeInTheDocument();
  });
});

describe("Copy settings and Paste", () => {
  it("an Object row arrives empty on the other photo: no points, no box, and its row says to click the object", async () => {
    await setup(/^Node 2, Mountain/);
    const foot = (): HTMLElement => within(tools()).getByRole("group", { name: "Copy and paste settings" });
    await h!.user.click(within(foot()).getByRole("button", { name: "Copy settings" }));
    await waitFor(() => expect(h!.store.getState().copied).not.toBeNull());
    await h!.user.click(screen.getByRole("button", { name: "Previous photo" }));
    await waitFor(() => expect(h!.store.currentPhoto()?.file_name).not.toBe("DSC04204.ARW"));
    await h!.user.click(within(foot()).getByRole("button", { name: "Paste" }));
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.some((n) => n.id === 2)).toBe(true));
    expect(objectRow(2, 0)).toEqual(newMask("object"));
    fireEvent.click(screen.getByRole("button", { name: /^Node 2, Mountain/ }));
    await h!.user.click(within(tools()).getByRole("tab", { name: "Mask" }));
    expect(within(rowList()).getByRole("button", { name: "Object mask, Click the object on this photo" })).toBeInTheDocument();
  });
});

// ---- Place points: the Object row's points from the keyboard (docs/28 A-11) ----------------------------------------------------------------

const pickLayer = (): HTMLElement | null => document.querySelector<HTMLElement>(".ym-pick-layer");
const crosshair = (): { left: string; top: string } => {
  const mark = pickLayer()!.querySelector<SVGElement>("svg.ym-pick-layer__mark")!;
  // Whole percents, to a rounding error of the sum of steps.
  return { left: `${Math.round(parseFloat(mark.style.left) * 1e6) / 1e6}%`, top: `${Math.round(parseFloat(mark.style.top) * 1e6) / 1e6}%` };
};
/** Sky gets an Object row and the focus goes to it; Return starts "Place points" and the layer, 600 × 400 like the cell, has the focus. */
async function placePoints(): Promise<HTMLElement> {
  await setup();
  await addObject();
  await waitFor(() => expect(rows(3)).toHaveLength(1));
  const row = within(rowList()).getByRole("button", { name: /^Object mask/ });
  row.focus();
  await h!.user.keyboard("{Enter}");
  await waitFor(() => expect(pickLayer()).not.toBeNull());
  mockRect(pickLayer()!, { left: 0, top: 0, width: 600, height: 400 });
  return pickLayer()!;
}

describe("Place points", () => {
  it("Return on the selected Object row starts it: the layer of Pick white, named for what the keys do, with the focus and the crosshair in the middle", async () => {
    const el = await placePoints();
    expect(el).toHaveAttribute("data-mode", "points");
    expect(el).toHaveAttribute("aria-label", "Place points. Move the crosshair with the arrow keys. Return adds a point, Shift Return takes a part away, Backspace removes the last point, Esc ends.");
    expect(el).toHaveFocus();
    expect(crosshair()).toEqual({ left: "50%", top: "50%" });
    expect(h!.store.getState().objectPlacing).toBe(true);
    // The Object tool's layer is under it and takes no press; the row's points are still drawn.
    expect(layer()).not.toHaveAttribute("data-tool");
  });

  it("the row says how, to a screen reader: its description is 'Press Return to place points with the keyboard.'", async () => {
    await setup();
    await addObject();
    const row = await within(rowList()).findByRole("button", { name: /^Object mask/ });
    expect(row).toHaveAccessibleDescription("Press Return to place points with the keyboard.");
  });

  it("Return on a row that is not selected only selects it, and Return elsewhere starts nothing", async () => {
    await setup(/^Node 2, Mountain/);
    const row = within(rowList()).getByRole("button", { name: /^Object mask, 1 point/ });
    expect(row).not.toHaveAccessibleDescription("Press Return to place points with the keyboard.");
    row.focus();
    await h!.user.keyboard("{Enter}");
    expect(h!.store.getState().maskRow).toBe(0);
    expect(h!.store.getState().objectPlacing).toBe(false);
    expect(pickLayer()).toBeNull();
    const slider = within(card("Object")).getAllByRole("textbox")[0]!;
    slider.focus();
    await h!.user.keyboard("{Enter}");
    expect(pickLayer()).toBeNull();
  });

  it("the arrows move the crosshair by one hundredth of the cell, ⇧ by ten, and it stays in the cell", async () => {
    await placePoints();
    const photo = h!.store.currentSettings()!.photo_id;
    await h!.user.keyboard("{ArrowRight}{ArrowRight}{ArrowDown}");
    expect(crosshair()).toEqual({ left: "52%", top: "51%" });
    await h!.user.keyboard("{Shift>}{ArrowLeft}{ArrowUp}{/Shift}");
    expect(crosshair()).toEqual({ left: "42%", top: "41%" });
    await h!.user.keyboard("{Shift>}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowLeft}{/Shift}");
    expect(crosshair().left).toBe("0%");
    // The arrows are the layer's: the photo does not step.
    expect(h!.store.currentSettings()!.photo_id).toBe(photo);
  });

  it("Return adds a positive point and ⇧Return a negative one at the crosshair, Backspace takes the last out, Esc ends: the points are the mouse's", async () => {
    await placePoints();
    // The hill's middle, (0.5, 0.7): two steps of ten down; then a little to the right and down for the part taken away.
    await h!.user.keyboard("{Shift>}{ArrowDown}{ArrowDown}{/Shift}{Enter}");
    await waitFor(() => expect(objectRow(3, 0).points).toEqual([[0.5, 0.7, true]]));
    await screen.findByText(/^Point added\. 1 point\./);
    await h!.user.keyboard("{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowDown}{ArrowDown}{Shift>}{Enter}{/Shift}");
    await waitFor(() => expect(objectRow(3, 0).points).toEqual([[0.5, 0.7, true], [0.55, 0.72, false]]));
    await screen.findByText(/^Negative point added\. 2 points\./);
    const byKeys = objectRow(3, 0).points;
    await waitFor(() => expect(dots().map((d) => d.dataset.objectPoint)).toEqual(["add", "remove"]));

    // Backspace takes the last one out (one patch, one undo step), and again the first.
    await h!.user.keyboard("{Backspace}");
    await waitFor(() => expect(objectRow(3, 0).points).toEqual([[0.5, 0.7, true]]));
    await screen.findByText(/^Point removed\. 1 point\./);
    await h!.user.keyboard("{Backspace}");
    await waitFor(() => expect(objectRow(3, 0).points).toEqual([]));
    // Nothing left to take out: nothing is sent and nothing is said.
    const spyUpdate = vi.spyOn(h!.api, "updateSettings");
    spyUpdate.mockClear();
    await h!.user.keyboard("{Backspace}");
    expect(spyUpdate).not.toHaveBeenCalled();

    // The mouse, at the same places of the cell, gives the same points.
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(pickLayer()).toBeNull());
    expect(h!.store.getState().objectPlacing).toBe(false);
    await screen.findByText(/^No points placed\./);
    await h!.user.click(within(card("Object")).getByRole("radio", { name: "Add point" }));
    click(300, 280);
    await waitFor(() => expect(objectRow(3, 0).points).toHaveLength(1));
    click(330, 288, true);
    await waitFor(() => expect(objectRow(3, 0).points).toHaveLength(2));
    expect(objectRow(3, 0).points).toEqual(byKeys);
  });

  it("Esc ends it, says what was placed, gives the focus back to the row and leaves the Object tool on", async () => {
    await placePoints();
    await h!.user.keyboard("{Shift>}{ArrowDown}{ArrowDown}{/Shift}{Enter}");
    await waitFor(() => expect(objectRow(3, 0).points).toHaveLength(1));
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(pickLayer()).toBeNull());
    await screen.findByText(/^Points placed\. 1 point\./);
    await waitFor(() => expect(within(rowList()).getByRole("button", { name: /^Object mask/ })).toHaveFocus());
    expect(h!.store.getState().objectTool).toBe(true);
  });

  it("a point that finds nothing adds nothing and says nothing of a point added", async () => {
    await placePoints();
    // The corner of the sky, as SKY is for the mouse: nothing is found there.
    await h!.user.keyboard("{Shift>}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{/Shift}{Enter}");
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Nothing found there."));
    expect(objectRow(3, 0).points).toEqual([]);
    expect(screen.queryByText(/^Point added/)).toBeNull();
  });

  it("a key held down adds one point, not a row of them", async () => {
    const el = await placePoints();
    const find = vi.spyOn(h!.api, "maskFind");
    await h!.user.keyboard("{Shift>}{ArrowDown}{ArrowDown}{/Shift}");
    fireEvent.keyDown(el, { key: "Enter", repeat: true });
    fireEvent.keyDown(el, { key: "Enter", repeat: true });
    expect(find).not.toHaveBeenCalled();
  });
});
