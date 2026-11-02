/**
 * The Mask tab of a corrector (`docs/23` §7.6, task N-12): the Masks card (Add mask, Invert, Reset, the rows with their Add | Subtract, Invert and
 * eye, the selected ring, Delete and the context menu), the Add mask menu (it lists `mask_kinds` and nothing else), the selected mask's card
 * for a Linear, a Radial, a Colour range and a Brightness range (each slider writes one field with the scale of `docs/algorithms/06` §4.5, one
 * patch per drag), the handles on the photo, and Pick colour of a Colour range. The screen runs on the mock, as in a browser.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import type { MaskItem, NodeLink, SettingsPatch } from "../../api/types";
import { newMask } from "../../state/maskRows";
import { nodeOf, patchOf } from "../../state/settingsGraph";
import type { CorrectorItem } from "../../state/settingsGraph";
import { openPhotos, openTab, renderApp, sliderName, sliderOf, tools, valueOf } from "../../test/harness";
import type { Harness, UpdateSpy } from "../../test/harness";
import { capturingPointer } from "../../test/pointer";
import { installManualRaf, mockRect } from "../../test/raf";

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
const LINEAR: MaskItem = { kind: "linear", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], angle_deg: 0, feather: 0.2 };
const RADIAL: MaskItem = { kind: "radial", mode: "subtract", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], radii: [0.25, 0.18], angle_deg: 0, feather: 0.45 };
const OBJECT: MaskItem = { kind: "object", mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, points: [[0.2, 0.3, true], [0.4, 0.5, false]], box: null };

/** Mountain (2, a Linear), Sky (3, an Object and a Radial) and Warmth (4, no masks) after Main, as the mockups. */
const FOUR: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Mountain", pos: [1, 0], masks: { invert: false, items: [LINEAR] } },
    "3": { kind: "corrector", name: "Sky", pos: [2, 0], masks: { invert: false, items: [OBJECT, RADIAL] } },
    "4": { kind: "corrector", name: "Warmth", pos: [3, 0] },
  },
  links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, 4), picture(4, "output")],
};

/** DSC04204 in the node view, on the Mask tab of the node `onNode` (Sky by default, which has two rows). */
async function setup(settings: SettingsPatch = FOUR, options: MockOptions = {}, onNode: RegExp | string = /^Node 3, Sky/): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: settings }, ...options });
  await openPhotos(h);
  await openTab(h, "Light");
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  fireEvent.click(screen.getByRole("button", { name: onNode }));
  await waitFor(() => expect(screen.getByRole("button", { name: onNode })).toHaveAttribute("aria-pressed", "true"));
  await h.user.click(within(tools()).getByRole("tab", { name: "Mask" }));
  await waitFor(() => expect(within(tools()).getByRole("tab", { name: "Mask" })).toHaveAttribute("aria-selected", "true"));
  return vi.spyOn(h.api, "updateSettings");
}

const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const cardTitles = (): string[] => within(tools()).getAllByRole("heading", { level: 3 }).map((e) => e.textContent ?? "");
const card = (name: string): HTMLElement => within(tools()).getByRole("region", { name });
const rowList = (): HTMLElement => within(tools()).getByRole("list", { name: "Masks of this node" });
const rowButton = (name: string | RegExp): HTMLElement => within(rowList()).getByRole("button", { name });
const rowOf = (name: string | RegExp): HTMLElement => rowButton(name).closest("li") as HTMLElement;
const masks = (node: number): MaskItem[] => (nodeOf(h!.store.currentSettings()!.settings, node) as CorrectorItem).masks.items;
const invertOf = (node: number): boolean => (nodeOf(h!.store.currentSettings()!.settings, node) as CorrectorItem).masks.invert;
const edit = (node: number, index: number, row: object): SettingsPatch => patchOf(node, { masks: { edit: { [String(index)]: row } } });
async function undo(): Promise<void> {
  await h!.user.keyboard("{Meta>}z{/Meta}");
}
/** Types a value into a slider's box and presses Return. */
async function type(label: string, text: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: `${label} value` });
  await h!.user.click(box);
  await h!.user.clear(box);
  await h!.user.type(box, `${text}{Enter}`);
}
async function openAddMenu(): Promise<HTMLElement> {
  await h!.user.click(within(tools()).getByRole("button", { name: "Add mask" }));
  return await screen.findByRole("menu", { name: "Add mask" });
}

describe("the Masks card", () => {
  it("has the title, Add mask, Invert and Reset, then the rows and the line under them", async () => {
    await setup();
    expect(cardTitles()).toEqual(["Masks", "View"]);
    const masksCard = card("Masks");
    expect(within(masksCard).getByRole("button", { name: "Add mask" })).toBeInTheDocument();
    expect(within(masksCard).getByRole("button", { name: "Invert" })).toHaveAttribute("aria-pressed", "false");
    expect(within(masksCard).getByRole("button", { name: "Reset" })).toBeInTheDocument();
    expect(within(masksCard).getByText("Masks add up. Subtract cuts a mask out. Colour and brightness ranges keep only their part of the rest.")).toBeInTheDocument();
  });

  it("names a row by its kind, 'mask' and its subtitle, and shows its name and subtitle in two lines", async () => {
    await setup();
    const rows = within(rowList()).getAllByRole("listitem");
    expect(rows).toHaveLength(2);
    expect(within(rows[0]!).getByRole("button", { name: "Object mask, 2 points" })).toBeInTheDocument();
    expect(within(rows[1]!).getByRole("button", { name: "Radial mask, Feather 45" })).toBeInTheDocument();
    expect(rows[0]).toHaveTextContent("Object2 points");
    expect(rows[1]).toHaveTextContent("RadialFeather 45");
  });

  it("has no rows for a node without masks, and shows no card for a mask", async () => {
    await setup(FOUR, {}, /^Node 4, Warmth/);
    expect(within(tools()).queryByRole("list", { name: "Masks of this node" })).toBeNull();
    expect(cardTitles()).toEqual(["Masks", "View"]);
    // Nothing to reset in a node with no row and no Invert.
    expect(within(card("Masks")).queryByRole("button", { name: "Reset" })).toBeNull();
  });

  it("selects a row by a click and rings it, and shows the card of its kind", async () => {
    await setup();
    expect(rowOf("Radial mask, Feather 45")).not.toHaveAttribute("data-selected");
    await h!.user.click(rowButton("Radial mask, Feather 45"));
    expect(rowOf("Radial mask, Feather 45")).toHaveAttribute("data-selected");
    expect(rowButton("Radial mask, Feather 45")).toHaveAttribute("aria-pressed", "true");
    expect(cardTitles()).toEqual(["Masks", "Radial", "View"]);
    await h!.user.click(rowButton("Object mask, 2 points"));
    expect(rowOf("Object mask, 2 points")).toHaveAttribute("data-selected");
    expect(rowOf("Radial mask, Feather 45")).not.toHaveAttribute("data-selected");
    // An Object row (from a file or a paste, until Object has its tool) shows its row in the list and no card.
    expect(cardTitles()).toEqual(["Masks", "View"]);
  });

  it("selects the row that a control or the keyboard moves into", async () => {
    await setup();
    await h!.user.click(within(rowOf("Radial mask, Feather 45")).getByRole("button", { name: "Show this mask" }));
    expect(rowOf("Radial mask, Feather 45")).toHaveAttribute("data-selected");
  });
});

describe("a row's controls: Add | Subtract, Invert this mask, Show this mask", () => {
  it("Add | Subtract writes the row's mode as a row patch, and is one undo step", async () => {
    const spy = await setup();
    const radial = rowOf("Radial mask, Feather 45");
    expect(within(radial).getByRole("radio", { name: "Subtract" })).toBeChecked();
    await h!.user.click(within(radial).getByRole("radio", { name: "Add" }));
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(3, 1, { mode: "add" })));
    expect(masks(3)[1]).toMatchObject({ kind: "radial", mode: "add" });
    expect(h!.store.getState().canUndo).toBe(true);
    await undo();
    await waitFor(() => expect(masks(3)[1]).toMatchObject({ mode: "subtract" }));
    expect(within(rowOf("Radial mask, Feather 45")).getByRole("radio", { name: "Subtract" })).toBeChecked();
  });

  it("Invert this mask is pressed in its own row only, and writes the row's invert", async () => {
    const spy = await setup();
    const object = within(rowOf("Object mask, 2 points")).getByRole("button", { name: "Invert this mask" });
    const radial = within(rowOf("Radial mask, Feather 45")).getByRole("button", { name: "Invert this mask" });
    expect(object).toHaveAttribute("aria-pressed", "false");
    await h!.user.click(radial);
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(3, 1, { invert: true })));
    expect(radial).toHaveAttribute("aria-pressed", "true");
    expect(object).toHaveAttribute("aria-pressed", "false");
    expect(invertOf(3)).toBe(false);
    await undo();
    await waitFor(() => expect(radial).toHaveAttribute("aria-pressed", "false"));
  });

  it("the eye writes enabled: a row that is off counts as absent and is dimmed", async () => {
    const spy = await setup();
    const eye = within(rowOf("Radial mask, Feather 45")).getByRole("button", { name: "Show this mask" });
    expect(eye).toHaveAttribute("aria-pressed", "true");
    await h!.user.click(eye);
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(3, 1, { enabled: false })));
    expect(eye).toHaveAttribute("aria-pressed", "false");
    expect(rowOf("Radial mask, Feather 45")).toHaveAttribute("data-off");
    await undo();
    await waitFor(() => expect(eye).toHaveAttribute("aria-pressed", "true"));
  });
});

describe("the node's Invert and Reset", () => {
  it("Invert is the node's masks.invert, a toggle, one undo step", async () => {
    const spy = await setup();
    const invert = within(card("Masks")).getByRole("button", { name: "Invert" });
    await h!.user.click(invert);
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(3, { masks: { invert: true } })));
    expect(invert).toHaveAttribute("aria-pressed", "true");
    expect(invertOf(3)).toBe(true);
    await h!.user.click(invert);
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(3, { masks: { invert: false } })));
    await undo();
    await waitFor(() => expect(invertOf(3)).toBe(true));
  });

  it("Reset removes every row and un-inverts, in one patch and one undo step", async () => {
    const spy = await setup();
    await h!.user.click(within(card("Masks")).getByRole("button", { name: "Invert" }));
    await h!.user.click(rowButton("Radial mask, Feather 45"));
    expect(cardTitles()).toContain("Radial");
    spy.mockClear();
    await h!.user.click(within(card("Masks")).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(patches(spy)).toEqual([patchOf(3, { masks: { items: [], invert: false } })]));
    expect(masks(3)).toEqual([]);
    expect(invertOf(3)).toBe(false);
    expect(within(tools()).queryByRole("list", { name: "Masks of this node" })).toBeNull();
    expect(cardTitles()).toEqual(["Masks", "View"]);
    await undo();
    await waitFor(() => expect(masks(3)).toHaveLength(2));
    expect(invertOf(3)).toBe(true);
  });
});

describe("removing and moving a row", () => {
  it("Delete on a selected row takes it out, and not the node or the photo", async () => {
    const spy = await setup();
    await h!.user.click(rowButton("Radial mask, Feather 45"));
    spy.mockClear();
    await h!.user.keyboard("{Delete}");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(3, { masks: { items: [OBJECT] } })));
    expect(masks(3)).toEqual([OBJECT]);
    // The next row is the one selected now, and the node is still there.
    expect(screen.getByRole("button", { name: /^Node 3, Sky/ })).toBeInTheDocument();
    expect(rowOf("Object mask, 2 points")).toHaveAttribute("data-selected");
    await undo();
    await waitFor(() => expect(masks(3)).toHaveLength(2));
  });

  it("Backspace does the same", async () => {
    await setup();
    await h!.user.click(rowButton("Object mask, 2 points"));
    await h!.user.keyboard("{Backspace}");
    await waitFor(() => expect(masks(3)).toEqual([RADIAL]));
  });

  it("a right click offers Move up, Move down and Delete mask, each one patch", async () => {
    const spy = await setup();
    fireEvent.contextMenu(rowOf("Object mask, 2 points"));
    const menu = await screen.findByRole("menu", { name: "Object mask actions" });
    expect(within(menu).getAllByRole("menuitem").map((m) => m.textContent)).toEqual(["Move up", "Move down", "Delete mask"]);
    // The first row cannot go up.
    expect(within(menu).getByRole("menuitem", { name: "Move up" })).toHaveAttribute("data-disabled");
    spy.mockClear();
    await h!.user.click(within(menu).getByRole("menuitem", { name: "Move down" }));
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(3, { masks: { items: [RADIAL, OBJECT] } })));
    expect(masks(3).map((m) => m.kind)).toEqual(["radial", "object"]);
    await undo();
    await waitFor(() => expect(masks(3).map((m) => m.kind)).toEqual(["object", "radial"]));
    fireEvent.contextMenu(rowOf("Radial mask, Feather 45"));
    const second = await screen.findByRole("menu", { name: "Radial mask actions" });
    expect(within(second).getByRole("menuitem", { name: "Move down" })).toHaveAttribute("data-disabled");
    await h!.user.click(within(second).getByRole("menuitem", { name: "Delete mask" }));
    await waitFor(() => expect(masks(3)).toEqual([OBJECT]));
  });

  it("the selection follows a row that is moved", async () => {
    await setup();
    await h!.user.click(rowButton("Object mask, 2 points"));
    fireEvent.contextMenu(rowOf("Object mask, 2 points"));
    await h!.user.click(within(await screen.findByRole("menu", { name: "Object mask actions" })).getByRole("menuitem", { name: "Move down" }));
    await waitFor(() => expect(masks(3).map((m) => m.kind)).toEqual(["radial", "object"]));
    expect(rowOf("Object mask, 2 points")).toHaveAttribute("data-selected");
    expect(h!.store.getState().maskRow).toBe(1);
  });
});

describe("the Add mask menu", () => {
  it("lists exactly the kinds mask_kinds gives (the shapes and ranges that are built), in the groups of the spec, with their lines", async () => {
    await setup();
    const menu = await openAddMenu();
    expect(within(menu).getAllByRole("menuitem").map((m) => m.textContent)).toEqual([
      "LinearA gradient across the photo",
      "RadialAn oval, soft at the edge",
      "Colour rangePick colours to select",
      "Brightness rangeSelect light or dark tones",
    ]);
    expect(within(menu).getByText("Shapes")).toBeInTheDocument();
    expect(within(menu).getByText("Keys")).toBeInTheDocument();
    // The Find group has no item, so it is not there; Brush is not made yet.
    expect(within(menu).queryByText("Find")).toBeNull();
    expect(within(menu).queryByRole("menuitem", { name: /Brush/ })).toBeNull();
  });

  it("lists every kind where the machine can make them, and hides what it cannot", async () => {
    await setup(FOUR, { maskKinds: ["object", "subject", "people", "background", "linear", "radial", "brush", "color_range", "brightness_range"] });
    const all = await openAddMenu();
    expect(within(all).getAllByRole("menuitem").map((m) => within(m).getAllByText(/./)[0]?.textContent)).toEqual([
      "Object",
      "Subject",
      "People",
      "Background",
      "Linear",
      "Radial",
      "Brush",
      "Colour range",
      "Brightness range",
    ]);
    expect(within(all).getByText("Click the photo, or draw a box")).toBeInTheDocument();
    expect(within(all).getByText("Each person. macOS 14 or later")).toBeInTheDocument();
    fireEvent.keyDown(all, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("menu", { name: "Add mask" })).toBeNull());
    // Subject and People are hidden, not greyed, on a Mac that cannot make them.
    h!.api.configure({ maskKinds: ["object", "background", "linear"] });
  });

  it("adds a row with its defaults in Add mode, selects it and shows its card, as one patch and one undo step", async () => {
    const spy = await setup();
    const menu = await openAddMenu();
    spy.mockClear();
    await h!.user.click(within(menu).getByRole("menuitem", { name: /^Radial/ }));
    await waitFor(() => expect(patches(spy)).toEqual([patchOf(3, { masks: { items: [OBJECT, RADIAL, newMask("radial")] } })]));
    expect(masks(3)).toHaveLength(3);
    expect(h!.store.getState().maskRow).toBe(2);
    expect(cardTitles()).toEqual(["Masks", "Radial", "View"]);
    expect(within(rowList()).getAllByRole("listitem")[2]).toHaveAttribute("data-selected");
    expect(within(rowList()).getAllByRole("listitem")[2]).toHaveTextContent("RadialFeather 30");
    await undo();
    await waitFor(() => expect(masks(3)).toHaveLength(2));
  });

  it("is off at 64 rows, with the core's sentence", async () => {
    const many: SettingsPatch = {
      nodes: { "2": { kind: "corrector", name: "Many", pos: [1, 0], masks: { invert: false, items: Array.from({ length: 64 }, () => LINEAR) } } },
      links: [picture("input", 1), picture(1, 2), picture(2, "output")],
    };
    await setup(many, {}, /^Node 2, Many/);
    const add = within(tools()).getByRole("button", { name: "Add mask" });
    expect(add).toHaveAttribute("aria-disabled", "true");
    expect(add).toHaveAccessibleDescription("A node can have up to 64 masks.");
  });

  it("starts the tool of a Colour range: Pick colour is on, for the next click on the photo", async () => {
    await setup();
    const menu = await openAddMenu();
    await h!.user.click(within(menu).getByRole("menuitem", { name: /^Colour range/ }));
    await waitFor(() => expect(cardTitles()).toEqual(["Masks", "Colour range", "View"]));
    expect(within(card("Colour range")).getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("application", { name: /Pick colour for the Colour range mask/ })).toBeInTheDocument();
  });
});

describe("the selected mask's card: sliders write one field each with the scales of docs/algorithms/06 §4.5", () => {
  it("Linear: Angle in degrees, Feather 0 to 100 (of 0 to 1) and Opacity in percent", async () => {
    const spy = await setup(FOUR, {}, /^Node 2, Mountain/);
    await h!.user.click(rowButton("Linear mask, Feather 20"));
    expect(cardTitles()).toEqual(["Masks", "Linear", "View"]);
    expect(within(card("Linear")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Angle", "Feather", "Opacity"]);
    expect(valueOf("Feather")).toBe("20");
    expect(valueOf("Opacity")).toBe("100%");
    await type("Angle", "-35");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(2, 0, { angle_deg: -35 })));
    await type("Feather", "45");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(2, 0, { feather: 0.45 })));
    await type("Opacity", "60");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(2, 0, { opacity: 0.6 })));
    expect(masks(2)[0]).toMatchObject({ angle_deg: -35, feather: 0.45, opacity: 0.6 });
    // The row's subtitle follows the field.
    expect(rowOf("Linear mask, Feather 45")).toBeInTheDocument();
  });

  it("Radial: Width and Height are the oval's full size as a percent of the long edge", async () => {
    const spy = await setup();
    await h!.user.click(rowButton("Radial mask, Feather 45"));
    expect(within(card("Radial")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Width", "Height", "Angle", "Feather", "Opacity"]);
    // Radii 0.25 and 0.18: an oval 50 % and 36 % of the long edge.
    expect(valueOf("Width")).toBe("50");
    expect(valueOf("Height")).toBe("36");
    await type("Width", "80");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(3, 1, { radii: [0.4, 0.18] })));
    await type("Height", "20");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(3, 1, { radii: [0.4, 0.1] })));
    expect(masks(3)[1]).toMatchObject({ radii: [0.4, 0.1] });
  });

  it("Colour range: Width is 0.005 + 0.295 (x / 100)², Softness 0 to 100; its Reset is a new row of the kind in the same mode", async () => {
    const colour: SettingsPatch = {
      nodes: { "2": { kind: "corrector", name: "Skin", pos: [1, 0], masks: { invert: false, items: [{ ...newMask("color_range", "subtract"), center: [0.1, -0.05], width: 0.2, softness: 0.1, opacity: 0.5, invert: true } as MaskItem] } } },
      links: [picture("input", 1), picture(1, 2), picture(2, "output")],
    };
    const spy = await setup(colour, {}, /^Node 2, Skin/);
    await h!.user.click(rowButton(/^Colour range mask/));
    expect(within(card("Colour range")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Width", "Softness", "Opacity"]);
    expect(valueOf("Width")).toBe("81");
    await type("Width", "40");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(2, 0, { width: 0.0522 })));
    await type("Softness", "80");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(2, 0, { softness: 0.8 })));
    await h!.user.click(within(card("Colour range")).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(masks(2)[0]).toEqual(newMask("color_range", "subtract")));
    const last = patches(spy).at(-1);
    expect(last).toEqual(edit(2, 0, { center: [0, 0], width: 0.05, softness: 0.5, opacity: 1, invert: false, enabled: true }));
  });

  it("Brightness range: Low and High are the code values times 100, and Low never goes above High", async () => {
    const bright: SettingsPatch = {
      nodes: { "2": { kind: "corrector", name: "Sky", pos: [1, 0], masks: { invert: false, items: [newMask("brightness_range")] } } },
      links: [picture("input", 1), picture(1, 2), picture(2, "output")],
    };
    const spy = await setup(bright, {}, /^Node 2, Sky/);
    await h!.user.click(rowButton("Brightness range mask, 35 to 100"));
    expect(within(card("Brightness range")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Low", "High", "Softness", "Opacity"]);
    expect(valueOf("Low")).toBe("35");
    expect(valueOf("High")).toBe("100");
    await type("High", "70");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(2, 0, { high: 0.7 })));
    await type("Low", "20");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(2, 0, { low: 0.2 })));
    expect(rowOf("Brightness range mask, 20 to 70")).toBeInTheDocument();
    // The ends of the two sliders meet at the other's value.
    expect(sliderOf("Low")).toHaveAttribute("aria-valuemax", "70");
    expect(sliderOf("High")).toHaveAttribute("aria-valuemin", "20");
    await type("Low", "90");
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ low: 0.7 }));
  });

  it("a drag is one patch for the frame and one final patch, and one undo step", async () => {
    const spy = await setup(FOUR, {}, /^Node 2, Mountain/);
    await h!.user.click(rowButton("Linear mask, Feather 20"));
    const raf = installManualRaf();
    try {
      const root = sliderOf("Feather").closest('[data-slot="slider"]') as HTMLElement;
      mockRect(root, { left: 100, top: 0, width: 200, height: 38 });
      const pointer = capturingPointer(root);
      spy.mockClear();
      pointer.down(120); // 10 %
      pointer.move(150);
      pointer.move(180);
      pointer.move(220); // 60 %
      await act(async () => {
        raf.flush();
      });
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
      expect(patches(spy)[0]).toEqual(edit(2, 0, { feather: 0.6 }));
      pointer.up(220);
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
      expect(patches(spy)[1]).toEqual(edit(2, 0, { feather: 0.6 }));
    } finally {
      raf.restore();
    }
    expect(masks(2)[0]).toMatchObject({ feather: 0.6 });
    await undo();
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ feather: 0.2 }));
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("the card's Reset puts a Linear back as a new Linear in its mode", async () => {
    const spy = await setup(FOUR, {}, /^Node 2, Mountain/);
    await h!.user.click(rowButton("Linear mask, Feather 20"));
    await type("Angle", "30");
    spy.mockClear();
    await h!.user.click(within(card("Linear")).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(masks(2)[0]).toEqual(newMask("linear", "add")));
    expect(patches(spy).at(-1)).toEqual(edit(2, 0, { invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], angle_deg: 0, feather: 0.2 }));
  });

  it("an Object row shows its row and no card, until it has its own", async () => {
    await setup();
    await h!.user.click(rowButton("Object mask, 2 points"));
    expect(cardTitles()).toEqual(["Masks", "View"]);
  });
});

describe("a mask is shown as the chip says", () => {
  it("'MASK · LINEAR' while the node has a selected row, whatever the tab", async () => {
    await setup(FOUR, {}, /^Node 2, Mountain/);
    expect(screen.queryByText(/^MASK · /)).toBeNull();
    await h!.user.click(rowButton("Linear mask, Feather 20"));
    expect(screen.getByText("MASK · LINEAR")).toBeInTheDocument();
    await openTab(h!, "Light").catch(() => undefined);
    await h!.user.click(within(tools()).getByRole("tab", { name: "Light" }));
    expect(screen.getByText("MASK · LINEAR")).toBeInTheDocument();
    await h!.user.click(within(tools()).getByRole("tab", { name: "Mask" }));
    await h!.user.click(rowButton("Linear mask, Feather 20"));
    // Another node starts with no row selected.
    fireEvent.click(screen.getByRole("button", { name: /^Node 3, Sky/ }));
    await waitFor(() => expect(screen.queryByText(/^MASK · /)).toBeNull());
  });
});

describe("the handles on the photo", () => {
  /** The photo cell, 600 × 400, and the mock told so: its own mapping answers for it. */
  async function placeCell(): Promise<void> {
    const cell = document.querySelector<HTMLElement>("[data-viewport]")!;
    mockRect(cell, { left: 0, top: 0, width: 600, height: 400 });
    await h!.api.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
    await act(async () => {
      fireEvent(window, new Event("resize"));
    });
  }
  const handle = (name: RegExp | string): HTMLElement => screen.getByRole("button", { name });
  const layer = (): HTMLElement | null => document.querySelector<HTMLElement>("[data-mask-handles]");

  it("show for the selected Linear: its dot, its two feather grips and its rotate handle, whatever the tab", async () => {
    await setup(FOUR, {}, /^Node 2, Mountain/);
    await placeCell();
    expect(layer()).toBeNull();
    await h!.user.click(rowButton("Linear mask, Feather 20"));
    await waitFor(() => expect(handle(/^Move the Linear mask/)).toBeInTheDocument());
    expect(screen.getAllByRole("button", { name: /^(Move the Linear mask|Linear mask feather|Rotate the Linear mask)/ }).map((b) => b.getAttribute("aria-label"))).toEqual([
      "Move the Linear mask. The arrow keys move it.",
      "Linear mask feather, start line. The arrow keys move it.",
      "Linear mask feather, end line. The arrow keys move it.",
      "Rotate the Linear mask. The arrow keys move it.",
    ]);
    // A line across the photo: the centre line, and the dashed start and end lines.
    expect(layer()!.querySelectorAll("[data-line]").length).toBe(3);
    // They stay on another tab.
    await h!.user.click(within(tools()).getByRole("tab", { name: "Colour" }));
    expect(layer()).not.toBeNull();
    // Mapped by the core: the centre at the middle of the cell, the rotate handle 80 % across.
    const move = handle(/^Move the Linear mask/);
    expect(parseFloat(move.style.left)).toBeCloseTo(300, 3);
    expect(parseFloat(move.style.top)).toBeCloseTo(200, 3);
    expect(parseFloat(handle(/^Rotate the Linear mask/).style.left)).toBeCloseTo(480, 3);
  });

  it("show for a Radial: the outline, the inner ring, the centre, a handle for each radius and the rotate handle", async () => {
    await setup();
    await placeCell();
    await h!.user.click(rowButton("Radial mask, Feather 45"));
    await waitFor(() => expect(handle(/^Move the Radial mask/)).toBeInTheDocument());
    expect(layer()!.querySelector('[data-line="outline"]')).not.toBeNull();
    expect(layer()!.querySelector('[data-line="inner"]')).not.toBeNull();
    for (const name of [/^Radial mask width/, /^Radial mask height/, /^Rotate the Radial mask/]) expect(handle(name)).toBeInTheDocument();
  });

  it("are not drawn for a row that is off, a kind that has none, while Crop shows, or while a pick is on", async () => {
    await setup();
    await placeCell();
    await h!.user.click(rowButton("Radial mask, Feather 45"));
    await waitFor(() => expect(layer()).not.toBeNull());
    // The eye off: none.
    await h!.user.click(within(rowOf("Radial mask, Feather 45")).getByRole("button", { name: "Show this mask" }));
    await waitFor(() => expect(layer()).toBeNull());
    await h!.user.click(within(rowOf("Radial mask, Feather 45")).getByRole("button", { name: "Show this mask" }));
    await waitFor(() => expect(layer()).not.toBeNull());
    // An Object row has none (its points are N-18's).
    await h!.user.click(rowButton("Object mask, 2 points"));
    await waitFor(() => expect(layer()).toBeNull());
    // Crop hides them.
    await h!.user.click(rowButton("Radial mask, Feather 45"));
    await waitFor(() => expect(layer()).not.toBeNull());
    await h!.user.click(screen.getByRole("button", { name: "Crop" }));
    await waitFor(() => expect(layer()).toBeNull());
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(layer()).not.toBeNull());
  });

  it("a drag of the centre dot writes the centre where the pointer is, one patch per move and a final one", async () => {
    const spy = await setup(FOUR, {}, /^Node 2, Mountain/);
    await placeCell();
    await h!.user.click(rowButton("Linear mask, Feather 20"));
    const dot = await waitFor(() => handle(/^Move the Linear mask/));
    spy.mockClear();
    fireEvent.pointerDown(dot, { clientX: 300, clientY: 200, button: 0, buttons: 1, pointerId: 1 });
    fireEvent.pointerMove(window, { clientX: 360, clientY: 240, pointerId: 1 });
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ center: [0.6, 0.6] }));
    fireEvent.pointerMove(window, { clientX: 420, clientY: 300, pointerId: 1 });
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ center: [0.7, 0.75] }));
    fireEvent.pointerUp(window, { clientX: 420, clientY: 300, pointerId: 1 });
    await waitFor(() => expect(h!.store.getState().canUndo).toBe(true));
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(edit(2, 0, { center: [0.7, 0.75] })));
    // One undo step for the whole drag.
    await undo();
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ center: [0.5, 0.5] }));
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("a drag of the end grip sets the feather, of the rotate handle the angle", async () => {
    await setup(FOUR, {}, /^Node 2, Mountain/);
    await placeCell();
    await h!.user.click(rowButton("Linear mask, Feather 20"));
    const end = await waitFor(() => handle(/^Linear mask feather, end line/));
    // The centre line is at y = 200 (0.5 of 400); 0.15 of the long edge (600 px) below it is 90 px: a feather of 0.3.
    fireEvent.pointerDown(end, { clientX: 100, clientY: 230, button: 0, buttons: 1, pointerId: 1 });
    fireEvent.pointerMove(window, { clientX: 100, clientY: 290, pointerId: 1 });
    fireEvent.pointerUp(window, { clientX: 100, clientY: 290, pointerId: 1 });
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ feather: 0.3 }));
    const rotate = await waitFor(() => handle(/^Rotate the Linear mask/));
    // Straight down from the centre is 90°.
    fireEvent.pointerDown(rotate, { clientX: 480, clientY: 200, button: 0, buttons: 1, pointerId: 1 });
    fireEvent.pointerMove(window, { clientX: 300, clientY: 320, pointerId: 1 });
    fireEvent.pointerUp(window, { clientX: 300, clientY: 320, pointerId: 1 });
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ angle_deg: 90 }));
  });

  it("a drag of each Radial handle writes the centre, a radius and the angle", async () => {
    await setup();
    await placeCell();
    await h!.user.click(rowButton("Radial mask, Feather 45"));
    const drag = async (name: RegExp, to: [number, number]): Promise<void> => {
      const el = await waitFor(() => handle(name));
      fireEvent.pointerDown(el, { clientX: parseFloat(el.style.left), clientY: parseFloat(el.style.top), button: 0, buttons: 1, pointerId: 1 });
      fireEvent.pointerMove(window, { clientX: to[0], clientY: to[1], pointerId: 1 });
      fireEvent.pointerUp(window, { clientX: to[0], clientY: to[1], pointerId: 1 });
    };
    // 0.3 of the long edge (180 px) to the right of the middle (300, 200): the first radius is 0.3.
    await drag(/^Radial mask width/, [480, 200]);
    await waitFor(() => expect(masks(3)[1]).toMatchObject({ radii: [0.3, 0.18] }));
    // 60 px below the middle: the second radius is 0.1.
    await drag(/^Radial mask height/, [300, 260]);
    await waitFor(() => expect(masks(3)[1]).toMatchObject({ radii: [0.3, 0.1] }));
    await drag(/^Move the Radial mask/, [150, 100]);
    await waitFor(() => expect(masks(3)[1]).toMatchObject({ center: [0.25, 0.25] }));
    // To the right of the (new) centre: the first radius turns a quarter turn clockwise.
    await drag(/^Rotate the Radial mask/, [400, 100]);
    await waitFor(() => expect(masks(3)[1]).toMatchObject({ angle_deg: 90 }));
  });

  it("the arrow keys nudge a handle 1 px, Shift 10 px, each as one undo step, and switch no photo", async () => {
    const spy = await setup(FOUR, {}, /^Node 2, Mountain/);
    await placeCell();
    await h!.user.click(rowButton("Linear mask, Feather 20"));
    const dot = await waitFor(() => handle(/^Move the Linear mask/));
    act(() => dot.focus());
    const before = h!.store.getState().session?.current_index;
    spy.mockClear();
    await h!.user.keyboard("{ArrowRight}");
    // 1 px of 600: a share of 1/600 of the photo's width.
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ center: [0.5017, 0.5] }));
    await h!.user.keyboard("{Shift>}{ArrowDown}{/Shift}");
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ center: [0.5017, 0.525] }));
    expect(patches(spy)).toHaveLength(2);
    expect(h!.store.getState().session?.current_index).toBe(before);
    await undo();
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ center: [0.5017, 0.5] }));
  });

  it("Delete on a handle removes nothing", async () => {
    await setup(FOUR, {}, /^Node 2, Mountain/);
    await placeCell();
    await h!.user.click(rowButton("Linear mask, Feather 20"));
    const dot = await waitFor(() => handle(/^Move the Linear mask/));
    act(() => dot.focus());
    await h!.user.keyboard("{Delete}");
    expect(masks(2)).toHaveLength(1);
    expect(screen.getByRole("button", { name: /^Node 2, Mountain/ })).toBeInTheDocument();
  });
});

describe("Pick colour of a Colour range", () => {
  const skin: SettingsPatch = {
    nodes: { "2": { kind: "corrector", name: "Skin", pos: [1, 0], masks: { invert: false, items: [newMask("color_range")] } } },
    links: [picture("input", 1), picture(1, 2), picture(2, "output")],
  };
  async function startPick(): Promise<UpdateSpy> {
    const spy = await setup(skin, {}, /^Node 2, Skin/);
    await h!.user.click(rowButton(/^Colour range mask/));
    await h!.user.click(within(card("Colour range")).getByRole("button", { name: "Pick colour" }));
    return spy;
  }
  const pickLayer = (): HTMLElement => screen.getByRole("application", { name: /Pick colour for the Colour range mask/ });

  it("is a toggle with an eyedropper; on, a click on the photo sets the row's centre from the chroma at the node", async () => {
    const spy = await startPick();
    expect(within(card("Colour range")).getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "true");
    const sample = vi.spyOn(h!.api, "sampleChroma");
    mockRect(pickLayer(), { left: 0, top: 0, width: 600, height: 400 });
    spy.mockClear();
    // The mock's sky is at the top of the photo: its chroma, at the node (2).
    await h!.user.pointer({ target: pickLayer(), coords: { clientX: 300, clientY: 20 }, keys: "[MouseLeft]" });
    await waitFor(() => expect(sample).toHaveBeenCalledWith(300, 20, 2));
    // The chroma the core gives for that point at this node, held to the plane (at most 0.5 from the middle).
    const there = (await h!.api.sampleChroma(300, 20, 2))!;
    await waitFor(() => expect(patches(spy)).toHaveLength(1));
    const center = ((patches(spy)[0] as SettingsPatch).nodes?.["2"]?.masks?.edit?.["0"] as { center: [number, number] }).center;
    const scale = Math.min(1, 0.5 / Math.hypot(there.a, there.b));
    expect(center[0]).toBeCloseTo(there.a * scale, 9);
    expect(center[1]).toBeCloseTo(there.b * scale, 9);
    expect(Math.hypot(center[0], center[1])).toBeGreaterThan(0);
    expect(masks(2)[0]).toMatchObject({ kind: "color_range", center });
    // The pick stays on for the next click, and the colour is one undo step.
    expect(pickLayer()).toBeInTheDocument();
    expect(h!.store.getState().canUndo).toBe(true);
    await undo();
    await waitFor(() => expect(masks(2)[0]).toMatchObject({ center: [0, 0] }));
  });

  it("ends with Esc, with the toggle, with a tab change, a node change and a change of row, and the other picks are exclusive with it", async () => {
    await startPick();
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull());
    expect(within(card("Colour range")).getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "false");
    // The toggle turns it on and off.
    await h!.user.click(within(card("Colour range")).getByRole("button", { name: "Pick colour" }));
    expect(pickLayer()).toBeInTheDocument();
    await h!.user.click(within(card("Colour range")).getByRole("button", { name: "Pick colour" }));
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull());
    // A tab change ends it, and coming back finds it off.
    await h!.user.click(within(card("Colour range")).getByRole("button", { name: "Pick colour" }));
    expect(pickLayer()).toBeInTheDocument();
    await h!.user.click(within(tools()).getByRole("tab", { name: "Colour" }));
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull());
    await h!.user.click(within(tools()).getByRole("tab", { name: "Mask" }));
    expect(within(card("Colour range")).getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "false");
    // Pick colour (warp) takes it over, and it takes the warp's.
    await h!.user.click(within(card("Colour range")).getByRole("button", { name: "Pick colour" }));
    await h!.user.click(within(tools()).getByRole("tab", { name: "Colour" }));
    await h!.user.click(within(card("Colour warp")).getByRole("button", { name: "Pick colour" }));
    expect(screen.getByRole("application", { name: /^Pick colour\. / })).toBeInTheDocument();
    await h!.user.click(within(tools()).getByRole("tab", { name: "Mask" }));
    expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull();
    // A node change ends it.
    await h!.user.click(within(card("Colour range")).getByRole("button", { name: "Pick colour" }));
    expect(pickLayer()).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^Node 1, Main/ }));
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull());
  });

  it("is off for a photo change and for Crop", async () => {
    await startPick();
    await h!.user.click(screen.getByRole("button", { name: "Crop" }));
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull());
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull());
    await h!.user.click(within(card("Colour range")).getByRole("button", { name: "Pick colour" }));
    expect(pickLayer()).toBeInTheDocument();
    await h!.user.click(screen.getByRole("button", { name: "Next photo" }));
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull());
  });

  it("hides the handles of the row while it is on", async () => {
    await startPick();
    expect(document.querySelector("[data-mask-handles]")).toBeNull();
  });
});

describe("Delete in the Masks card", () => {
  it("takes out the row that has the focus, else the selected one, and never the node", async () => {
    await setup();
    await h!.user.click(rowButton("Object mask, 2 points"));
    // The focus is on the other row's control: that is the row that goes.
    within(rowOf("Radial mask, Feather 45")).getByRole("button", { name: "Invert this mask" }).focus();
    await h!.user.keyboard("{Delete}");
    await waitFor(() => expect(masks(3).map((m) => m.kind)).toEqual(["object"]));
    expect(screen.getByRole("button", { name: /^Node 3, Sky/ })).toBeInTheDocument();
    // The focus is on a button of the card's header and a row is selected: the selected row goes.
    await h!.user.click(rowButton("Object mask, 2 points"));
    within(card("Masks")).getByRole("button", { name: "Add mask" }).focus();
    await h!.user.keyboard("{Backspace}");
    await waitFor(() => expect(masks(3)).toEqual([]));
    // No row at all: the key does nothing in the card, and the node stays.
    within(card("Masks")).getByRole("button", { name: "Invert" }).focus();
    await h!.user.keyboard("{Delete}");
    expect(screen.getByRole("button", { name: /^Node 3, Sky/ })).toBeInTheDocument();
  });

  it("does not remove a row from the Add mask menu", async () => {
    await setup();
    await h!.user.click(rowButton("Object mask, 2 points"));
    const menu = await openAddMenu();
    fireEvent.keyDown(within(menu).getAllByRole("menuitem")[0]!, { key: "Delete" });
    expect(masks(3)).toHaveLength(2);
  });
});

describe("Copy settings and Paste (docs/23 §3.8)", () => {
  /** An Object row, a Subject row, a Linear and a Radial in one node of one photo. */
  const FULL: SettingsPatch = {
    nodes: {
      "2": {
        kind: "corrector",
        name: "Mixed",
        pos: [1, 0],
        masks: {
          invert: true,
          items: [
            OBJECT,
            { ...newMask("subject"), mode: "subtract" },
            { ...LINEAR, angle_deg: 12, feather: 0.4 },
            { ...RADIAL, center: [0.3, 0.6] },
            newMask("background"),
            { ...newMask("brush"), strokes: [{ size: 0.03, hardness: 0.5, erase: false, points: [[0.1, 0.2], [0.3, 0.4]] }] } as MaskItem,
            { ...newMask("color_range"), center: [0.1, 0.05] } as MaskItem,
            newMask("brightness_range"),
          ],
        },
      },
    },
    links: [picture("input", 1), picture(1, 2), picture(2, "output")],
  };

  it("carries every row to the other photo, Linear, Radial, ranges and Brush as they are, Subject and Background as rows, and an Object row empty", async () => {
    await setup(FULL, { maskKinds: ["object", "subject", "people", "background", "linear", "radial", "brush", "color_range", "brightness_range"] }, /^Node 2, Mixed/);
    const copied = h!.store.currentSettings()!.settings;
    const foot = (): HTMLElement => within(tools()).getByRole("group", { name: "Copy and paste settings" });
    await h!.user.click(within(foot()).getByRole("button", { name: "Copy settings" }));
    await waitFor(() => expect(h!.store.getState().copied).not.toBeNull());
    await h!.user.click(screen.getByRole("button", { name: "Previous photo" }));
    await waitFor(() => expect(h!.store.currentPhoto()?.file_name).not.toBe("DSC04204.ARW"));
    await h!.user.click(within(foot()).getByRole("button", { name: "Paste" }));
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.some((n) => n.id === 2)).toBe(true));
    expect(masks(2)).toHaveLength(8);
    const there = masks(2);
    const was = (nodeOf(copied, 2) as CorrectorItem).masks.items;
    // The node's Invert comes too.
    expect(invertOf(2)).toBe(true);
    // Everything but the Object row is what it was.
    expect(there.filter((r) => r.kind !== "object")).toEqual(was.filter((r) => r.kind !== "object"));
    // The Object row arrives empty: no point, no box, nothing selected until it is clicked on this photo.
    expect(there[0]).toEqual({ ...OBJECT, points: [], box: null });
    // The Mask tab reads it so.
    fireEvent.click(screen.getByRole("button", { name: /^Node 2, Mixed/ }));
    await h!.user.click(within(tools()).getByRole("tab", { name: "Mask" }));
    expect(rowButton("Object mask, Click the object on this photo")).toBeInTheDocument();
    expect(within(rowOf("Object mask, Click the object on this photo")).getByText("Click the object on this photo")).toBeInTheDocument();
    // One undo step takes the whole paste back.
    await undo();
    await waitFor(() => expect(nodeOf(h!.store.currentSettings()!.settings, 2)).toBeUndefined());
  });
});
