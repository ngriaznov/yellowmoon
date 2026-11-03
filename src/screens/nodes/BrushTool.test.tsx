/**
 * The Brush tool (`docs/23` §7.6, task N-14): the BRUSH card (Paint · Erase, Size, Hardness, then Refine edges, Feather, Grow, Opacity: each with its
 * scale, Size and Hardness for the next strokes only), the cursor circle and the stroke drawn on the photo, a stroke sent when the pointer is released
 * as one row patch (`strokes_append`) after one map call, and the start and the end of the tool. The screen runs on the mock, as in a browser.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import type { BrushStroke, MaskItem, NodeLink, SettingsPatch } from "../../api/types";
import { DEFAULT_BRUSH } from "../../state/brushStrokes";
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
const stroke = (x: number, more: Partial<BrushStroke> = {}): BrushStroke => ({ size: 0.03, hardness: 0.5, erase: false, points: [[x, 0.5], [x, 0.6]], ...more });
const BRUSH = (strokes: BrushStroke[] = [], more: Record<string, unknown> = {}): MaskItem => ({ ...newMask("brush"), strokes, ...more }) as MaskItem;
const LINEAR: MaskItem = { kind: "linear", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], angle_deg: 0, feather: 0.2 };

/** Main, then Skin (2, a Linear and a Brush with one stroke), Sky (3, no masks): Skin's Mask tab is where the tests start. */
const SETTINGS: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Skin", pos: [1, 0], masks: { invert: false, items: [LINEAR, BRUSH([stroke(0.3)])] } },
    "3": { kind: "corrector", name: "Sky", pos: [2, 0] },
  },
  links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, "output")],
};

/** DSC04204 in the node view, on the Mask tab of the node `onNode`, the photo cell 600 × 400 and the mock told so. */
async function setup(onNode: RegExp = /^Node 3, Sky/, settings: SettingsPatch = SETTINGS, options: MockOptions = {}): Promise<UpdateSpy> {
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
const rowButton = (name: string | RegExp): HTMLElement => within(rowList()).getByRole("button", { name });
const rows = (node: number): MaskItem[] => (nodeOf(h!.store.currentSettings()!.settings, node) as CorrectorItem).masks.items;
const brushRow = (node: number, index: number): Extract<MaskItem, { kind: "brush" }> => rows(node)[index] as Extract<MaskItem, { kind: "brush" }>;
const edit = (node: number, index: number, row: object): SettingsPatch => patchOf(node, { masks: { edit: { [String(index)]: row } } });
const layer = (): HTMLElement | null => document.querySelector<HTMLElement>("[data-brush-layer]");
const cursor = (): HTMLElement | null => document.querySelector<HTMLElement>("[data-brush-cursor]");
async function undo(): Promise<void> {
  await h!.user.keyboard("{Meta>}z{/Meta}");
}
async function addBrush(): Promise<void> {
  await h!.user.click(within(tools()).getByRole("button", { name: "Add mask" }));
  const menu = await screen.findByRole("menu", { name: "Add mask" });
  await h!.user.click(within(menu).getByRole("menuitem", { name: /^Brush/ }));
}
/** Types a value into a slider's box and presses Return. */
async function type(label: string, text: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: `${label} value` });
  await h!.user.click(box);
  await h!.user.clear(box);
  await h!.user.type(box, `${text}{Enter}`);
}

/** A stroke by hand: down at the first place, moves through the others, up at the last. Returns after the pointer is released. */
function drag(places: Array<[number, number]>): void {
  const [first, ...rest] = places as [[number, number], ...Array<[number, number]>];
  fireEvent.pointerDown(layer()!, { clientX: first[0], clientY: first[1], button: 0, buttons: 1, pointerId: 1 });
  for (const [x, y] of rest) fireEvent.pointerMove(window, { clientX: x, clientY: y, pointerId: 1 });
  const last = rest.at(-1) ?? first;
  fireEvent.pointerUp(window, { clientX: last[0], clientY: last[1], pointerId: 1 });
}

describe("adding a Brush row", () => {
  it("selects it, shows the BRUSH card and starts the tool: the cursor follows the pointer", async () => {
    const spy = await setup();
    await addBrush();
    await waitFor(() => expect(patches(spy)).toEqual([patchOf(3, { masks: { items: [newMask("brush")] } })]));
    expect(cardTitles()).toEqual(["Masks", "Brush", "View"]);
    expect(h!.store.getState().brushTool).toBe(true);
    await waitFor(() => expect(layer()).not.toBeNull());
    expect(screen.getByText("MASK · BRUSH")).toBeInTheDocument();
    expect(within(rowList()).getAllByRole("listitem")[0]).toHaveTextContent("BrushPaint on the photo");
    // The layer is a surface to draw on, not a control.
    expect(layer()!.matches("button, [tabindex], [role]")).toBe(false);
    expect(layer()!.querySelectorAll("button, [tabindex], input").length).toBe(0);
  });

  it("the cursor is a circle at the brush's size, measured by the core at the pointer (the page converts nothing)", async () => {
    await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    expect(cursor()).toBeNull();
    fireEvent.pointerMove(layer()!, { clientX: 300, clientY: 200, pointerId: 1 });
    // A size of 0.03 of the long edge (600 px here) is a diameter of 18 px, a radius of 9.
    await waitFor(() => expect(Number(cursor()!.dataset.radius)).toBeCloseTo(9, 1));
    expect(cursor()!.style.left).toBe("300px");
    expect(cursor()!.style.top).toBe("200px");
    expect(cursor()!.style.width).toBe("18px");
    // A bigger brush is a bigger circle.
    await type("Size", "100");
    await waitFor(() => expect(Number(cursor()!.dataset.radius)).toBeCloseTo(75, 1));
    fireEvent.pointerLeave(layer()!);
    expect(cursor()).toBeNull();
  });

  it("the circle has an inner ring for a soft brush and none for a hard one, and shows Erase as a dashed circle", async () => {
    await setup();
    await addBrush();
    fireEvent.pointerMove(layer()!, { clientX: 300, clientY: 200, pointerId: 1 });
    await waitFor(() => expect(cursor()).not.toBeNull());
    expect(cursor()!.querySelector(".ym-brush-cursor__core")).not.toBeNull();
    expect(cursor()).not.toHaveAttribute("data-erase");
    await type("Hardness", "100");
    await waitFor(() => expect(cursor()!.querySelector(".ym-brush-cursor__core")).toBeNull());
    await h!.user.click(within(card("Brush")).getByRole("radio", { name: "Erase" }));
    await waitFor(() => expect(cursor()).toHaveAttribute("data-erase"));
  });
});

describe("the BRUSH card", () => {
  it("has Paint · Erase, Size, Hardness, then Refine edges (0), Feather, Grow and Opacity", async () => {
    await setup();
    await addBrush();
    const brush = card("Brush");
    expect(within(brush).getByRole("radiogroup", { name: "Paint or erase" })).toBeInTheDocument();
    expect(within(brush).getAllByRole("radio").map((r) => r.textContent)).toEqual(["Paint", "Erase"]);
    expect(within(brush).getByRole("radio", { name: "Paint" })).toBeChecked();
    expect(within(brush).getAllByRole("slider").map((s) => s.getAttribute("aria-valuetext") ?? "")).toHaveLength(6);
    const order = within(brush)
      .getAllByRole("textbox")
      .map((t) => t.getAttribute("aria-label"));
    expect(order).toEqual(["Size value", "Hardness value", "Refine edges value", "Feather value", "Grow value", "Opacity value"]);
    // The defaults: a size of 0.03 is slider 34, hardness 50, Refine edges 0 % (a Brush's own), Feather 0, Grow 0, Opacity 100 %.
    expect([valueOf("Size"), valueOf("Hardness"), valueOf("Refine edges"), valueOf("Feather"), valueOf("Grow"), valueOf("Opacity")]).toEqual(["34", "50", "0%", "0", "0", "100%"]);
    expect(within(brush).getByRole("button", { name: "Reset" })).toBeInTheDocument();
    // The line under the tool says how to draw.
    expect(within(brush).getByText("Drag on the photo to paint. Esc stops.")).toBeInTheDocument();
  });

  it("Refine edges, Feather, Grow and Opacity each write one field of the row with their scale, as a row patch and one undo step", async () => {
    const spy = await setup();
    await addBrush();
    await waitFor(() => expect(rows(3)).toHaveLength(1));
    spy.mockClear();
    await type("Refine edges", "60");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(3, 0, { refine: 0.6 })));
    await type("Feather", "35");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(3, 0, { feather: 0.35 })));
    await type("Grow", "-40");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(3, 0, { grow: -0.4 })));
    await type("Opacity", "70");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(3, 0, { opacity: 0.7 })));
    expect(brushRow(3, 0)).toMatchObject({ refine: 0.6, feather: 0.35, grow: -0.4, opacity: 0.7, strokes: [] });
    expect(patches(spy)).toHaveLength(4);
    // Grow is bipolar and reaches both ends.
    await type("Grow", "100");
    await waitFor(() => expect(brushRow(3, 0).grow).toBe(1));
    await type("Grow", "-100");
    await waitFor(() => expect(brushRow(3, 0).grow).toBe(-1));
    // One undo step each.
    await undo();
    await waitFor(() => expect(brushRow(3, 0).grow).toBe(1));
  });

  it("Size and Hardness are the brush for the next strokes: no patch, no undo step, and the strokes already drawn keep theirs", async () => {
    const spy = await setup(/^Node 2, Skin/);
    await h!.user.click(rowButton(/^Brush mask, 1 stroke/));
    await waitFor(() => expect(cardTitles()).toEqual(["Masks", "Brush", "View"]));
    spy.mockClear();
    const before = h!.store.currentSettings();
    // Size 50 on the slider is 0.001 + 0.249 / 4 = 0.06325; Hardness 80 is 0.8.
    await type("Size", "50");
    await type("Hardness", "80");
    expect(h!.store.getState().brush).toEqual({ size: 0.06325, hardness: 0.8, erase: false });
    expect(spy).not.toHaveBeenCalled();
    expect(h!.store.currentSettings()).toBe(before);
    expect(h!.store.getState().canUndo).toBe(false);
    expect(brushRow(2, 1).strokes).toEqual([stroke(0.3)]);
    // The next stroke is drawn with them.
    await waitFor(() => expect(layer()).not.toBeNull());
    drag([[150, 100], [250, 100]]);
    await waitFor(() => expect(brushRow(2, 1).strokes).toHaveLength(2));
    expect(brushRow(2, 1).strokes[0]).toEqual(stroke(0.3));
    expect(brushRow(2, 1).strokes[1]).toMatchObject({ size: 0.06325, hardness: 0.8, erase: false });
  });

  it("Paint · Erase makes the next strokes erase strokes", async () => {
    await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(within(card("Brush")).getByRole("radio", { name: "Erase" }));
    expect(h!.store.getState().brush.erase).toBe(true);
    expect(within(card("Brush")).getByText("Drag on the photo to take paint away. Esc stops.")).toBeInTheDocument();
    drag([[100, 100], [200, 120]]);
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(1));
    expect(brushRow(3, 0).strokes[0]).toMatchObject({ erase: true });
    await h!.user.click(within(card("Brush")).getByRole("radio", { name: "Paint" }));
    drag([[300, 100], [350, 120]]);
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(2));
    expect(brushRow(3, 0).strokes[1]).toMatchObject({ erase: false });
  });

  it("Reset puts the row back as a new Brush row in the same mode, with no strokes, and the brush at its default", async () => {
    const spy = await setup(/^Node 2, Skin/);
    await h!.user.click(rowButton(/^Brush mask, 1 stroke/));
    await h!.user.click(within(rowList().querySelectorAll("li")[1] as HTMLElement).getByRole("radio", { name: "Subtract" }));
    await type("Opacity", "40");
    await type("Size", "80");
    await h!.user.click(within(card("Brush")).getByRole("radio", { name: "Erase" }));
    spy.mockClear();
    await h!.user.click(within(card("Brush")).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(brushRow(2, 1)).toEqual({ ...newMask("brush", "subtract") }));
    expect(patches(spy)).toEqual([edit(2, 1, { invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0, strokes: [] })]);
    expect(h!.store.getState().brush).toEqual(DEFAULT_BRUSH);
    expect(rowButton(/^Brush mask, Paint on the photo/)).toBeInTheDocument();
    // One undo step brings the stroke back.
    await undo();
    await waitFor(() => expect(brushRow(2, 1).strokes).toEqual([stroke(0.3)]));
  });
});

describe("a stroke on the photo", () => {
  it("is drawn while the pointer is down, in CSS pixels, and sent when it is released: one map call, one row patch, one undo step", async () => {
    const spy = await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    const mapSpy = vi.spyOn(h!.api, "viewportMapPoints");
    await waitFor(() => expect(rows(3)).toHaveLength(1));
    spy.mockClear();
    mapSpy.mockClear();
    fireEvent.pointerDown(layer()!, { clientX: 100, clientY: 100, button: 0, buttons: 1, pointerId: 1 });
    for (const x of [150, 200, 250, 300]) fireEvent.pointerMove(window, { clientX: x, clientY: 100 + (x - 100) / 2, pointerId: 1 });
    // Drawn at once, in the cell's pixels; nothing has gone to the core.
    const live = layer()!.querySelector<SVGPolylineElement>("polyline[data-live]")!;
    expect(live.getAttribute("points")).toBe("100,100 150,125 200,150 250,175 300,200");
    expect(live.getAttribute("stroke-width")).toBe("18");
    expect(layer()).toHaveAttribute("data-stroking", "true");
    expect(spy).not.toHaveBeenCalled();
    expect(mapSpy.mock.calls.filter((c) => c[1] === "cell")).toHaveLength(0);
    fireEvent.pointerUp(window, { clientX: 300, clientY: 200, pointerId: 1 });
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(1));
    // One map call for the whole stroke, with every place in order.
    const toPhoto = mapSpy.mock.calls.filter((c) => c[1] === "cell");
    expect(toPhoto).toHaveLength(1);
    expect(toPhoto[0]![0]).toEqual([[100, 100], [150, 125], [200, 150], [250, 175], [300, 200]]);
    // One row patch with the stroke appended; the straight line is simplified to its two ends, in the photo frame (the cell is 600 × 400).
    await waitFor(() => expect(patches(spy)).toHaveLength(1));
    expect(patches(spy)[0]).toEqual(edit(3, 0, { strokes_append: [{ size: 0.03, hardness: 0.5, erase: false, points: [[0.1667, 0.25], [0.5, 0.5]] }] }));
    expect(layer()).not.toHaveAttribute("data-stroking");
    // The row says so.
    expect(rowButton("Brush mask, 1 stroke")).toBeInTheDocument();
    // One undo step.
    await undo();
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(0));
    expect(rowButton("Brush mask, Paint on the photo")).toBeInTheDocument();
    // The row itself (adding it was a step of its own) is still there; the stroke was the last step.
    expect(rows(3)).toHaveLength(1);
    expect(h!.store.getState().canUndo).toBe(true);
    await h!.user.keyboard("{Meta>}{Shift>}z{/Shift}{/Meta}");
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(1));
  });

  it("keeps a bend that is farther than 1/8 of the brush from the line, and drops points on the line", async () => {
    await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    // A size of 0.03: the tolerance is 0.00375 of the long edge, 2.25 px here. A corner of 60 px is kept; the points along each side are not.
    drag([[100, 100], [140, 100], [180, 100], [220, 100], [220, 140], [220, 180], [220, 220]]);
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(1));
    expect(brushRow(3, 0).strokes[0]!.points).toEqual([[0.1667, 0.25], [0.3667, 0.25], [0.3667, 0.55]]);
  });

  it("a single click is a stroke of one point", async () => {
    await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    drag([[300, 200]]);
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(1));
    expect(brushRow(3, 0).strokes[0]).toMatchObject({ points: [[0.5, 0.5]], erase: false });
    expect(rowButton("Brush mask, 1 stroke")).toBeInTheDocument();
  });

  it("drops the places on the paper, and a stroke wholly on the paper sends nothing", async () => {
    const spy = await setup(/^Node 3, Sky/, SETTINGS, { mockCurrent: 3 } as MockOptions);
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    await waitFor(() => expect(rows(3)).toHaveLength(1));
    // The mapping of the mock answers null for a place outside the photo: stand in for the paper with the cell's own edge.
    vi.spyOn(h!.api, "viewportMapPoints").mockImplementation(async (points) => points.map(([x, y]) => (x < 50 ? null : [x / 600, y / 400])));
    spy.mockClear();
    drag([[10, 100], [30, 100], [100, 100], [200, 100], [210, 100]]);
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(1));
    expect(brushRow(3, 0).strokes[0]!.points).toEqual([[0.1667, 0.25], [0.35, 0.25]]);
    spy.mockClear();
    drag([[10, 100], [30, 100]]);
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(spy).not.toHaveBeenCalled();
    expect(brushRow(3, 0).strokes).toHaveLength(1);
    expect(h!.store.getState().error).toBeNull();
  });

  it("several strokes are several undo steps, in order, and the row counts them", async () => {
    await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    for (const y of [60, 120, 180]) {
      drag([[100, y], [300, y]]);
      await waitFor(() => expect(brushRow(3, 0).strokes.length).toBeGreaterThan([60, 120, 180].indexOf(y)));
    }
    expect(rowButton("Brush mask, 3 strokes")).toBeInTheDocument();
    expect(within(rowList()).getAllByRole("listitem")[0]).toHaveTextContent("Brush3 strokes");
    await undo();
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(2));
    expect(rowButton("Brush mask, 2 strokes")).toBeInTheDocument();
    await undo();
    await undo();
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(0));
  });

  it("a stroke that the system cancels sends nothing, and so does a tool that ends mid-stroke", async () => {
    const spy = await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    await waitFor(() => expect(rows(3)).toHaveLength(1));
    spy.mockClear();
    fireEvent.pointerDown(layer()!, { clientX: 100, clientY: 100, button: 0, buttons: 1, pointerId: 1 });
    fireEvent.pointerMove(window, { clientX: 200, clientY: 100, pointerId: 1 });
    fireEvent.pointerCancel(window, { pointerId: 1 });
    expect(layer()).not.toHaveAttribute("data-stroking");
    fireEvent.pointerDown(layer()!, { clientX: 100, clientY: 100, button: 0, buttons: 1, pointerId: 1 });
    fireEvent.pointerMove(window, { clientX: 200, clientY: 100, pointerId: 1 });
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(layer()).toBeNull());
    fireEvent.pointerUp(window, { clientX: 200, clientY: 100, pointerId: 1 });
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(spy).not.toHaveBeenCalled();
    expect(brushRow(3, 0).strokes).toHaveLength(0);
  });

  it("is not a pan: with the photo zoomed in, a drag paints and does not move it (and moves it once Esc has ended the tool); a double click does not zoom", async () => {
    await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.store.setZoom({ kind: "percent", value: 200 });
    const pan = vi.spyOn(h!.store, "queueViewportInput");
    const zoomed = vi.spyOn(h!.api, "viewportInput");
    drag([[100, 100], [200, 160]]);
    fireEvent.doubleClick(layer()!, { clientX: 300, clientY: 200 });
    await waitFor(() => expect(brushRow(3, 0).strokes).toHaveLength(1));
    expect(pan).not.toHaveBeenCalled();
    expect(zoomed).not.toHaveBeenCalled();
    // The same drag on the bare photo is a pan, so the test above is not vacuous.
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(layer()).toBeNull());
    const cell = document.querySelector<HTMLElement>("[data-viewport]")!;
    fireEvent.pointerDown(cell, { clientX: 100, clientY: 100, button: 0, buttons: 1, pointerId: 1 });
    fireEvent.pointerMove(window, { clientX: 140, clientY: 120, pointerId: 1 });
    fireEvent.pointerUp(window, { clientX: 140, clientY: 120, pointerId: 1 });
    expect(pan).toHaveBeenCalled();
  });

  it("the wheel and the zoom keys work as usual while the tool is on", async () => {
    await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    const queue = vi.spyOn(h!.store, "queueViewportInput");
    fireEvent.wheel(layer()!, { deltaY: -120, ctrlKey: true, clientX: 300, clientY: 200 });
    expect(queue).toHaveBeenCalledWith(expect.objectContaining({ kind: "zoom" }));
    await h!.user.keyboard("z");
    await waitFor(() => expect(h!.store.getState().zoom).toEqual({ kind: "percent", value: 100 }));
    expect(layer()).not.toBeNull();
  });

  it("only the primary button paints", async () => {
    const spy = await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    await waitFor(() => expect(rows(3)).toHaveLength(1));
    spy.mockClear();
    fireEvent.pointerDown(layer()!, { clientX: 100, clientY: 100, button: 2, buttons: 2, pointerId: 1 });
    fireEvent.pointerUp(window, { clientX: 100, clientY: 100, pointerId: 1 });
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(spy).not.toHaveBeenCalled();
  });

  it("is refused at 100,000 points with the core's sentence in the error bar, and changes nothing", async () => {
    const many: BrushStroke[] = Array.from({ length: 100 }, (_, i) => ({ size: 0.03, hardness: 0.5, erase: false, points: Array.from({ length: 1000 }, (_, k): [number, number] => [(k + 1) / 1001, (i + 1) / 101]) }));
    const full: SettingsPatch = { nodes: { "2": { kind: "corrector", name: "Skin", pos: [1, 0], masks: { invert: false, items: [BRUSH(many)] } } }, links: [picture("input", 1), picture(1, 2), picture(2, "output")] };
    const spy = await setup(/^Node 2, Skin/, full);
    await h!.user.click(rowButton(/^Brush mask, 100 strokes/));
    await waitFor(() => expect(layer()).not.toBeNull());
    spy.mockClear();
    drag([[100, 100], [300, 300]]);
    await waitFor(() => expect(screen.getByText("A brush mask can have up to 100,000 points.")).toBeInTheDocument());
    expect(spy).not.toHaveBeenCalled();
    expect(brushRow(2, 0).strokes).toHaveLength(100);
    expect(h!.store.getState().canUndo).toBe(false);
  });
});

describe("the tool starts and ends", () => {
  it("Esc ends it (the cursor and the layer go, the card says how to start it again), and Paint or Erase, the chosen one too, starts it", async () => {
    await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(layer()).toBeNull());
    expect(h!.store.getState().brushTool).toBe(false);
    // The row stays selected, its card stays.
    expect(cardTitles()).toEqual(["Masks", "Brush", "View"]);
    expect(within(card("Brush")).getByText("Choose Paint or Erase to draw on the photo.")).toBeInTheDocument();
    await h!.user.click(within(card("Brush")).getByRole("radio", { name: "Paint" }));
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(layer()).toBeNull());
    await h!.user.click(within(card("Brush")).getByRole("radio", { name: "Erase" }));
    await waitFor(() => expect(layer()).not.toBeNull());
    expect(h!.store.getState().brush.erase).toBe(true);
  });

  it("a click on a Brush row starts it, and choosing another row, or another node, ends it", async () => {
    await setup(/^Node 2, Skin/);
    expect(layer()).toBeNull();
    await h!.user.click(rowButton(/^Brush mask, 1 stroke/));
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(rowButton(/^Linear mask/));
    await waitFor(() => expect(layer()).toBeNull());
    expect(h!.store.getState().brushTool).toBe(false);
    await h!.user.click(rowButton(/^Brush mask, 1 stroke/));
    await waitFor(() => expect(layer()).not.toBeNull());
    fireEvent.click(screen.getByRole("button", { name: /^Node 3, Sky/ }));
    await waitFor(() => expect(layer()).toBeNull());
    expect(h!.store.getState().brushTool).toBe(false);
    // Back on Skin: nothing is selected, nothing is on.
    fireEvent.click(screen.getByRole("button", { name: /^Node 2, Skin/ }));
    await waitFor(() => expect(screen.getByRole("button", { name: /^Node 2, Skin/ })).toHaveAttribute("aria-pressed", "true"));
    expect(layer()).toBeNull();
  });

  it("is on whatever tab the panel shows, and off for a row that is off", async () => {
    await setup(/^Node 2, Skin/);
    await h!.user.click(rowButton(/^Brush mask, 1 stroke/));
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(within(tools()).getByRole("tab", { name: "Light" }));
    expect(layer()).not.toBeNull();
    await h!.user.click(within(tools()).getByRole("tab", { name: "Mask" }));
    await h!.user.click(within(within(rowList()).getAllByRole("listitem")[1] as HTMLElement).getByRole("switch", { name: "Turn this mask off" }));
    await waitFor(() => expect(layer()).toBeNull());
    expect(within(card("Brush")).getByText("This mask is off. Turn it on to draw.")).toBeInTheDocument();
    await h!.user.click(within(within(rowList()).getAllByRole("listitem")[1] as HTMLElement).getByRole("switch", { name: "Turn this mask on" }));
    await waitFor(() => expect(layer()).not.toBeNull());
  });

  it("Crop, Before and a pick hide the cursor, and it comes back with them gone", async () => {
    await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(screen.getByRole("button", { name: "Crop" }));
    await waitFor(() => expect(layer()).toBeNull());
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(screen.getByRole("button", { name: "Before" }));
    await waitFor(() => expect(layer()).toBeNull());
    await h!.user.click(screen.getByRole("button", { name: "Before" }));
    await waitFor(() => expect(layer()).not.toBeNull());
    // A pick (the Colour tab's Pick colour) takes the photo for itself, and Esc ends the pick first, not the tool.
    await h!.user.click(within(tools()).getByRole("tab", { name: "Colour" }));
    await h!.user.click(within(tools()).getByRole("button", { name: "Pick colour" }));
    await waitFor(() => expect(screen.getByRole("application", { name: /^Pick colour/ })).toBeInTheDocument());
    expect(layer()).toBeNull();
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(layer()).not.toBeNull());
    expect(h!.store.getState().brushTool).toBe(true);
  });

  it("the Add mask menu's other kinds do not start it, and a Linear added after a Brush ends it", async () => {
    await setup();
    await addBrush();
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(within(tools()).getByRole("button", { name: "Add mask" }));
    const menu = await screen.findByRole("menu", { name: "Add mask" });
    await h!.user.click(within(menu).getByRole("menuitem", { name: /^Linear/ }));
    await waitFor(() => expect(layer()).toBeNull());
    expect(h!.store.getState().brushTool).toBe(false);
  });
});

describe("Copy settings and Paste", () => {
  it("carry a Brush row as it is, strokes and all, to the other photo", async () => {
    await setup(/^Node 2, Skin/);
    await h!.user.click(rowButton(/^Brush mask, 1 stroke/));
    await waitFor(() => expect(layer()).not.toBeNull());
    drag([[100, 100], [300, 300]]);
    await waitFor(() => expect(brushRow(2, 1).strokes).toHaveLength(2));
    const copied = brushRow(2, 1);
    const foot = (): HTMLElement => within(tools()).getByRole("group", { name: "Copy and paste settings" });
    await h!.user.click(within(foot()).getByRole("button", { name: "Copy settings" }));
    await waitFor(() => expect(h!.store.getState().copied).not.toBeNull());
    await h!.user.click(screen.getByRole("button", { name: "Previous photo" }));
    await waitFor(() => expect(h!.store.currentPhoto()?.file_name).not.toBe("DSC04204.ARW"));
    await h!.user.click(within(foot()).getByRole("button", { name: "Paste" }));
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.some((n) => n.id === 2)).toBe(true));
    expect(brushRow(2, 1)).toEqual(copied);
  });
});
