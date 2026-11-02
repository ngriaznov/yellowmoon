/**
 * The panel of the selected node in the node view (`docs/23` §7.5, task N-09): the compact head (the number chip, the name, the quiet line,
 * Bypass, Reset and More), the tabs of each kind of node, the trim cards, Main's controls with their Auto against every other
 * corrector's without, the Mask tab until N-12, W in the node view, and the node `at` that the histogram, the scatter and Pick colour
 * name. The screen runs on the mock, as in a browser.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import type { BalanceSetting, MaskItem, NodeLink, SettingsPatch } from "../../api/types";
import { SCENES } from "../../fixtures/scenes";
import { hexChroma } from "../../fixtures/warp";
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
const OBJECT: MaskItem = { kind: "object", mode: "add", invert: false, enabled: false, opacity: 1, feather: 0, grow: 0, refine: 0.5, points: [[0.2, 0.3, true], [0.4, 0.5, false]], box: null };

/** Mountain (2, a Linear mask), Sky (3, Object and Radial) and Warmth (4) after Main, as the mockups. */
const FOUR: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Mountain", pos: [1, 0], trim: { exposure_ev: 0.3 }, masks: { invert: false, items: [LINEAR] } },
    "3": { kind: "corrector", name: "Sky", pos: [2, 0], masks: { invert: false, items: [OBJECT, RADIAL] } },
    "4": { kind: "corrector", name: "Warmth", pos: [3, 0] },
  },
  links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, 4), picture(4, "output")],
};

/** DSC04204 (the mockups' photo, `current: 3`), in the node view on the tab the simple view stands on. */
async function setup(settings: SettingsPatch = FOUR, options: MockOptions = {}, simpleTab = "Light"): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: settings }, ...options });
  await openPhotos(h);
  await openTab(h, simpleTab);
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  return vi.spyOn(h.api, "updateSettings");
}

const head = (): HTMLElement => within(tools()).getByRole("group", { name: "Node" });
const tabsOf = (): string[] => within(tools()).getAllByRole("tab").map((t) => t.textContent ?? "");
const selectedTab = (): string => within(tools()).getByRole("tab", { selected: true }).textContent ?? "";
const nodeCard = (name: RegExp | string): HTMLElement => screen.getByRole("button", { name });
const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const cardTitles = (): string[] => within(tools()).getAllByRole("heading", { level: 3 }).map((e) => e.textContent ?? "");
const regionNamed = (name: string): HTMLElement => within(tools()).getByRole("region", { name });
async function openNodeTab(name: string): Promise<void> {
  await h!.user.click(within(tools()).getByRole("tab", { name }));
  await waitFor(() => expect(within(tools()).getByRole("tab", { name })).toHaveAttribute("aria-selected", "true"));
}
/** A click on a card of the graph (a pointer press on it would reach the canvas's drag library, which jsdom's events have no window for). */
async function select(card: RegExp | string): Promise<void> {
  fireEvent.click(nodeCard(card));
  await waitFor(() => expect(nodeCard(card)).toHaveAttribute("aria-pressed", "true"));
}
/** Types a value into a slider's box and presses Return. */
async function type(label: string, text: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: `${label} value` });
  await h!.user.click(box);
  await h!.user.clear(box);
  await h!.user.type(box, `${text}{Enter}`);
}
const corrector = (id: number): CorrectorItem => nodeOf(h!.store.currentSettings()!.settings, id) as CorrectorItem;

describe("the head", () => {
  it("is one compact head, not a title block: the number chip, the name, the quiet line and three icons", async () => {
    await setup();
    expect(within(head()).getByRole("heading", { level: 2, name: "Main" })).toBeInTheDocument();
    expect(head().querySelector(".ym-nodehead__chip")).toHaveTextContent("01");
    expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Corrector · whole photo");
    const buttons = within(within(head()).getByRole("group", { name: "Node actions" })).getAllByRole("button");
    expect(buttons.map((b) => b.getAttribute("aria-label"))).toEqual(["Bypass this node", "Reset this node", "More"]);
    expect(buttons[0]).toHaveAccessibleDescription("Bypass this node (B)");
    expect(buttons[1]).toHaveAccessibleDescription("Reset this node");
    expect(buttons[2]).toHaveAccessibleDescription("More");
    // The simple view's 40 px title block is not in the node view.
    expect(tools().querySelector(".ym-title-block")).toBeNull();
    expect(within(tools()).queryByRole("heading", { level: 2, name: "Light" })).toBeNull();
  });

  it("says what each node is: a lock and a sentence for the Input and the Output, and what a corrector acts on", async () => {
    await setup();
    await select("Input");
    expect(within(head()).getByRole("heading", { name: "Input" })).toBeInTheDocument();
    expect(head().querySelector(".ym-nodehead__chip svg")).not.toBeNull();
    expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Turns the camera's data into the scene.");
    expect(within(head()).getAllByRole("button").map((b) => b.getAttribute("aria-label"))).toEqual(["Reset input", "More"]);
    await select("Output");
    expect(within(head()).getByRole("heading", { name: "Output" })).toBeInTheDocument();
    expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Cine rendering, look and export");
    expect(within(head()).getAllByRole("button").map((b) => b.getAttribute("aria-label"))).toEqual(["Reset output"]);
    // A corrector: its number as the plan has it, and the kind of its mask, or how many.
    await select(/^Node 2, Mountain/);
    expect(head().querySelector(".ym-nodehead__chip")).toHaveTextContent("02");
    expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Corrector · Linear");
    await select(/^Node 3, Sky/);
    expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Corrector · 2 masks");
    await select(/^Node 4, Warmth/);
    expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Corrector · whole photo");
  });

  it("says 'bypassed' and 'not connected' before what a node acts on", async () => {
    await setup();
    await select(/^Node 3, Sky/);
    await h!.user.click(within(head()).getByRole("button", { name: "Bypass this node" }));
    await waitFor(() => expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Corrector · bypassed"));
    // A branch nothing reads: node 4 is taken out of the line and left where it is.
    await h!.user.keyboard("b");
    await h!.store.update({ links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, "output")] }, true);
    await select(/^Node 4, Warmth/);
    await waitFor(() => expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Corrector · not connected"));
  });

  it("renames on a double click: Return accepts as one patch, Esc cancels, and the card follows", async () => {
    const spy = await setup();
    await select(/^Node 4, Warmth/);
    fireEvent.doubleClick(within(head()).getByRole("heading", { name: "Warmth" }));
    const box = within(head()).getByRole("textbox", { name: "Node name" });
    expect(box).toHaveFocus();
    expect(box).toHaveValue("Warmth");
    await h!.user.clear(box);
    await h!.user.type(box, "Glow{Enter}");
    await waitFor(() => expect(within(head()).getByRole("heading", { name: "Glow" })).toBeInTheDocument());
    expect(patches(spy)).toEqual([patchOf(4, { name: "Glow" })]);
    expect(nodeCard(/^Node 4, Glow/)).toBeInTheDocument();
    // Esc leaves the name as it was and sends nothing.
    fireEvent.doubleClick(within(head()).getByRole("heading", { name: "Glow" }));
    await h!.user.type(within(head()).getByRole("textbox", { name: "Node name" }), "x{Escape}");
    expect(within(head()).getByRole("heading", { name: "Glow" })).toBeInTheDocument();
    expect(spy).toHaveBeenCalledTimes(1);
    // Main, a corrector too, is renamed; the Input and the Output have fixed names.
    await select("Input");
    fireEvent.doubleClick(within(head()).getByRole("heading", { name: "Input" }));
    expect(within(head()).queryByRole("textbox", { name: "Node name" })).toBeNull();
  });

  it("has Bypass, which is the toolbar's Bypass and the key B: a toggle that says what it does", async () => {
    const spy = await setup();
    await select(/^Node 2, Mountain/);
    const bypass = within(head()).getByRole("button", { name: "Bypass this node" });
    expect(bypass).toHaveAttribute("aria-pressed", "false");
    await h!.user.click(bypass);
    await waitFor(() => expect(bypass).toHaveAttribute("aria-pressed", "true"));
    expect(patches(spy)).toEqual([{ nodes: { "2": { bypass: true } } }]);
    expect(within(screen.getByRole("group", { name: "Graph tools" })).getByRole("button", { name: "Bypass" })).toHaveAttribute("aria-pressed", "true");
    await h!.user.keyboard("b");
    await waitFor(() => expect(bypass).toHaveAttribute("aria-pressed", "false"));
    // Not for the Input and the Output: their head has no Bypass.
    await select("Output");
    expect(within(head()).queryByRole("button", { name: "Bypass this node" })).toBeNull();
  });

  it("has Reset: the node's trim and grade at their defaults, its masks, name, bypass and links kept, in one undo step", async () => {
    const spy = await setup({
      nodes: {
        "2": {
          kind: "corrector",
          name: "Mountain",
          pos: [1, 0],
          bypass: true,
          trim: { exposure_ev: 0.8, temperature: 30, tint: -10 },
          grade: { contrast: { kind: "manual", value: 25 }, brightness: 12, color_warp: { pins: [] } },
          masks: { invert: false, items: [LINEAR] },
        },
      },
      links: [picture("input", 1), picture(1, 2), picture(2, "output")],
    });
    await select(/^Node 2, Mountain/);
    const before = h!.store.currentSettings()!.settings;
    await h!.user.click(within(head()).getByRole("button", { name: "Reset this node" }));
    await waitFor(() => expect(corrector(2).trim).toEqual({ exposure_ev: 0, temperature: 0, tint: 0 }));
    const after = corrector(2);
    expect(after.grade.contrast).toEqual({ kind: "manual", value: 0 });
    expect(after.grade.brightness).toBe(0);
    // What the head's Reset leaves alone.
    expect(after.masks.items).toEqual([LINEAR]);
    expect(after.name).toBe("Mountain");
    expect(after.bypass).toBe(true);
    expect(h!.store.currentSettings()!.settings.nodes.links).toEqual(before.nodes.links);
    // One patch, naming only that node's trim and grade.
    expect(patches(spy)).toHaveLength(1);
    expect(Object.keys(patches(spy)[0]!.nodes ?? {})).toEqual(["2"]);
    expect(Object.keys(patches(spy)[0]!.nodes?.["2"] ?? {}).sort()).toEqual(["grade", "trim"]);
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(corrector(2).trim.exposure_ev).toBe(0.8));
    expect(corrector(2).grade.brightness).toBe(12);
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("resets Main to its Auto and does nothing to a node that is already at its defaults", async () => {
    const spy = await setup();
    await select(/^Node 4, Warmth/);
    await h!.user.click(within(head()).getByRole("button", { name: "Reset this node" }));
    expect(spy).not.toHaveBeenCalled();
    await select(/^Node 1, Main/);
    await h!.store.update(patchOf(1, { trim: { temperature: 20 }, grade: { contrast: { kind: "manual", value: 30 } } }), true);
    spy.mockClear();
    await h!.user.click(within(head()).getByRole("button", { name: "Reset this node" }));
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.find((n) => n.id === 1 && n.kind === "corrector")).toMatchObject({ trim: { temperature: 0 } }));
    expect((nodeOf(h!.store.currentSettings()!.settings, 1) as CorrectorItem).grade.contrast).toEqual({ kind: "auto" });
  });

  it("has Reset for the Input (every field at its default, Auto where there is Auto) and the Output (the look, its intensity and the sharpening)", async () => {
    await setup();
    await select("Input");
    await h!.store.update(
      { nodes: { input: { exposure: { kind: "manual", ev: 0.7 }, light_source: "tungsten", rolloff: { highlights: 0.2 } }, output: { look_intensity: 40, output_sharpening: 150 } } },
      true,
    );
    await h!.user.click(within(head()).getByRole("button", { name: "Reset input" }));
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.find((n) => n.kind === "input")).toMatchObject({ exposure: { kind: "auto" }, light_source: "auto", rolloff: { highlights: 0.6, low_light: 0.5 } }));
    // The Output keeps what the Input's Reset does not touch.
    expect(h!.store.currentSettings()!.settings.nodes.items.find((n) => n.kind === "output")).toMatchObject({ look_intensity: 40, output_sharpening: 150 });
    await select("Output");
    await h!.user.click(within(head()).getByRole("button", { name: "Reset output" }));
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.find((n) => n.kind === "output")).toMatchObject({ look_intensity: 100, output_sharpening: 100 }));
  });

  it("has More in the head, not in the toolbar: Rename starts the name's text box in the head", async () => {
    await setup();
    await select(/^Node 2, Mountain/);
    expect(within(screen.getByRole("group", { name: "Graph tools" })).queryByRole("button", { name: "More" })).toBeNull();
    await h!.user.click(within(head()).getByRole("button", { name: "More" }));
    const menu = await screen.findByRole("menu", { name: "More" });
    expect(within(menu).getAllByRole("menuitem").map((i) => i.textContent)).toEqual(["Rename", "Move earlier", "Move later", "Link picture to", "Link key to", "Remove link", "Delete node"]);
    await h!.user.click(within(menu).getByRole("menuitem", { name: "Rename" }));
    const box = await within(head()).findByRole("textbox", { name: "Node name" });
    await waitFor(() => expect(box).toHaveFocus());
    // The card has no second text box.
    expect(screen.getAllByRole("textbox", { name: "Node name" })).toHaveLength(1);
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("textbox", { name: "Node name" })).toBeNull());
  });

  it("has More on the Input (an extra branch from it), and Main's More has no Move items and a Delete that is off", async () => {
    await setup();
    await select("Input");
    await h!.user.click(within(head()).getByRole("button", { name: "More" }));
    let menu = await screen.findByRole("menu", { name: "More" });
    expect(within(menu).getAllByRole("menuitem").map((i) => i.textContent)).toEqual(["Link picture to", "Remove link"]);
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("menu")).toBeNull());
    await select(/^Node 1, Main/);
    await h!.user.click(within(head()).getByRole("button", { name: "More" }));
    menu = await screen.findByRole("menu", { name: "More" });
    expect(within(menu).queryByRole("menuitem", { name: "Move earlier" })).toBeNull();
    expect(within(menu).getByRole("menuitem", { name: "Delete node" })).toHaveAttribute("aria-disabled", "true");
  });
});

describe("the tabs follow the selected node", () => {
  it("are RAW · Lens · Light · Colour · Detail on the Input, Light · Colour · Curve · Mask on a corrector, Look · Detail on the Output", async () => {
    await setup();
    expect(tabsOf()).toEqual(["Light", "Colour", "Curve", "Mask"]);
    await select("Input");
    expect(tabsOf()).toEqual(["RAW", "Lens", "Light", "Colour", "Detail"]);
    await select("Output");
    expect(tabsOf()).toEqual(["Look", "Detail"]);
    await select(/^Node 3, Sky/);
    expect(tabsOf()).toEqual(["Light", "Colour", "Curve", "Mask"]);
  });

  it("open on the node and the tab of the simple view's tab, as the table of docs/23 §7.5 says", async () => {
    const cases: Array<[string, string, string]> = [
      ["Look", "Output", "Look"],
      ["Lens", "Input", "Lens"],
      ["Light", "Main", "Light"],
      ["Colour", "Main", "Colour"],
      ["Detail", "Input", "Detail"],
      ["Curve", "Main", "Curve"],
      ["RAW", "Input", "RAW"],
    ];
    for (const [simple, node, tab] of cases) {
      await setup(FOUR, {}, simple);
      expect(within(head()).getByRole("heading", { level: 2 })).toHaveTextContent(node);
      expect(selectedTab()).toBe(tab);
      h!.unmount();
      h = undefined;
    }
  });

  it("are remembered by kind: nodes of one kind share a tab", async () => {
    await setup();
    await openNodeTab("Colour");
    await select(/^Node 3, Sky/);
    expect(selectedTab()).toBe("Colour");
    await select("Input");
    expect(selectedTab()).toBe("RAW");
    await openNodeTab("Detail");
    await select("Output");
    expect(selectedTab()).toBe("Look");
    await select("Input");
    expect(selectedTab()).toBe("Detail");
    await select(/^Node 1, Main/);
    expect(selectedTab()).toBe("Colour");
  });

  it("are switched by ⌘1 to ⌘5 in the order of the row, the digits past the last tab doing nothing", async () => {
    await setup();
    await h!.user.keyboard("{Meta>}4{/Meta}");
    expect(selectedTab()).toBe("Mask");
    await h!.user.keyboard("{Meta>}5{/Meta}");
    expect(selectedTab()).toBe("Mask");
    await h!.user.keyboard("{Meta>}3{/Meta}");
    expect(selectedTab()).toBe("Curve");
    await select("Input");
    await h!.user.keyboard("{Meta>}5{/Meta}");
    expect(selectedTab()).toBe("Detail");
    await h!.user.keyboard("{Meta>}1{/Meta}");
    expect(selectedTab()).toBe("RAW");
    await select("Output");
    await h!.user.keyboard("{Meta>}2{/Meta}");
    expect(selectedTab()).toBe("Detail");
    await h!.user.keyboard("{Meta>}3{/Meta}");
    expect(selectedTab()).toBe("Detail");
  });

  it("have the cards of the simple view's tabs for the Input, and none of the simple view's Automatic, Tone and Re-run", async () => {
    await setup();
    await select("Input");
    expect(cardTitles()).toEqual(["Camera", "Decode", "Range"]);
    await openNodeTab("Lens");
    expect(cardTitles()).toEqual(["Lens", "Amounts"]);
    await openNodeTab("Light");
    expect(cardTitles()).toEqual(["Exposure"]);
    expect(within(regionNamed("Exposure")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Exposure"]);
    expect(within(regionNamed("Exposure")).getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "true");
    await openNodeTab("Colour");
    expect(cardTitles()).toEqual(["White balance"]);
    expect(within(tools()).getByRole("button", { name: "Pick white" })).toBeInTheDocument();
    await openNodeTab("Detail");
    expect(cardTitles()).toEqual(["Sharpening", "Noise reduction"]);
    expect(within(regionNamed("Sharpening")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Amount", "Radius"]);
    expect(within(tools()).queryByRole("region", { name: "Automatic" })).toBeNull();
    expect(within(tools()).queryByRole("button", { name: "Re-run" })).toBeNull();
  });

  it("have today's Look card on the Output's Look tab, and the Output sharpening alone on its Detail tab", async () => {
    await setup();
    await select("Output");
    expect(cardTitles()).toEqual(["Look"]);
    expect(within(regionNamed("Look")).getByRole("slider", { name: "Intensity" })).toBeInTheDocument();
    expect(within(regionNamed("Look")).getByRole("radiogroup", { name: "Apply look to" })).toBeInTheDocument();
    expect(within(regionNamed("Look")).getByRole("searchbox", { name: "Search looks" })).toBeInTheDocument();
    expect(within(tools()).queryByRole("region", { name: "Tone" })).toBeNull();
    await openNodeTab("Detail");
    expect(cardTitles()).toEqual(["Sharpening"]);
    expect(within(regionNamed("Sharpening")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Output"]);
    expect(valueOf("Output")).toBe("100%");
  });

  it("have the trim cards on a corrector: Exposure trim with Contrast and Brightness, Temperature and tint, no AUTO tag and no Pick white", async () => {
    await setup();
    await select(/^Node 2, Mountain/);
    expect(cardTitles()).toEqual(["Exposure", "High dynamic range"]);
    expect(within(regionNamed("Exposure")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Exposure trim", "Contrast", "Brightness"]);
    expect(within(regionNamed("High dynamic range")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Highlights", "Shadows", "White", "Black"]);
    expect(valueOf("Exposure trim")).toBe("+0.30");
    // Auto belongs to the Input: no AUTO tag in a corrector's cards.
    expect(within(tools()).queryByRole("button", { name: "Auto" })).toBeNull();
    await openNodeTab("Colour");
    expect(cardTitles()).toEqual(["Temperature and tint", "Colour balance", "Colour warp"]);
    expect(within(regionNamed("Temperature and tint")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Temperature", "Tint"]);
    expect(within(tools()).queryByRole("button", { name: "Pick white" })).toBeNull();
    expect(within(tools()).getByRole("button", { name: "Pick colour" })).toBeInTheDocument();
    expect(within(tools()).queryByRole("button", { name: "Auto" })).toBeNull();
    await openNodeTab("Curve");
    expect(cardTitles()).toEqual(["Curve"]);
  });
});

describe("the trim cards write the node's trim", () => {
  it("Exposure trim: −4 to 4 in steps of 0.01, the detent at 0, and a drag is one patch per frame and a final one, as one undo step", async () => {
    const spy = await setup();
    await select(/^Node 4, Warmth/);
    const slider = sliderOf("Exposure trim");
    expect(slider).toHaveAttribute("aria-valuemin", "-4");
    expect(slider).toHaveAttribute("aria-valuemax", "4");
    expect(slider).toHaveAttribute("aria-valuenow", "0");
    const raf = installManualRaf();
    try {
      const root = slider.closest('[data-slot="slider"]') as HTMLElement;
      mockRect(root, { left: 100, top: 0, width: 200, height: 38 });
      const pointer = capturingPointer(root);
      spy.mockClear();
      pointer.down(200); // the middle: 0
      pointer.move(230);
      pointer.move(250);
      pointer.move(260); // 4 · 0.6 = 2.4
      await act(async () => {
        raf.flush();
      });
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
      expect(patches(spy)[0]).toEqual(patchOf(4, { trim: { exposure_ev: 2.4 } }));
      pointer.up(260);
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
      expect(patches(spy)[1]).toEqual(patchOf(4, { trim: { exposure_ev: 2.4 } }));
    } finally {
      raf.restore();
    }
    await waitFor(() => expect(valueOf("Exposure trim")).toBe("+2.40"));
    expect(corrector(4).trim.exposure_ev).toBe(2.4);
    expect(h!.store.getState().canUndo).toBe(true);
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(valueOf("Exposure trim")).toBe("0.00"));
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("Exposure trim: a typed value and the row's Reset send the trim alone, and the node's other trims stay", async () => {
    const spy = await setup({ nodes: { "2": { kind: "corrector", pos: [1, 0], trim: { temperature: 20 } } }, links: [picture("input", 1), picture(1, 2), picture(2, "output")] });
    await select(/^Node 2, Corrector/);
    spy.mockClear();
    await type("Exposure trim", "0.5");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(2, { trim: { exposure_ev: 0.5 } })));
    expect(corrector(2).trim).toEqual({ exposure_ev: 0.5, temperature: 20, tint: 0 });
    await h!.user.click(within(regionNamed("Exposure")).getAllByRole("button", { name: "Reset" })[0]!);
    await waitFor(() => expect(corrector(2).trim.exposure_ev).toBe(0));
    expect(corrector(2).trim.temperature).toBe(20);
  });

  it("Temperature and tint: −100 to 100 in steps of 1, the detent at 0; each writes its own field of the trim; Reset puts both at 0", async () => {
    const spy = await setup();
    await select(/^Node 4, Warmth/);
    await openNodeTab("Colour");
    for (const label of ["Temperature", "Tint"]) {
      const slider = sliderOf(label);
      expect(slider).toHaveAttribute("aria-valuemin", "-100");
      expect(slider).toHaveAttribute("aria-valuemax", "100");
      expect(slider).toHaveAttribute("aria-valuenow", "0");
    }
    spy.mockClear();
    await type("Temperature", "-18");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(4, { trim: { temperature: -18 } })));
    await type("Tint", "4");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(4, { trim: { tint: 4 } })));
    expect(corrector(4).trim).toEqual({ exposure_ev: 0, temperature: -18, tint: 4 });
    expect(valueOf("Temperature")).toBe("−18");
    await h!.user.click(within(regionNamed("Temperature and tint")).getAllByRole("button", { name: "Reset" }).at(-1)!);
    await waitFor(() => expect(corrector(4).trim).toEqual({ exposure_ev: 0, temperature: 0, tint: 0 }));
  });
});

describe("Auto belongs to Main", () => {
  it("Main's Contrast sends Auto for a value equal to Auto; another corrector's sends a manual 0", async () => {
    const spy = await setup();
    await select(/^Node 1, Main/);
    await type("Contrast", "20");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(1, { grade: { contrast: { kind: "manual", value: 20 } } })));
    await type("Contrast", "0");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(1, { grade: { contrast: { kind: "auto" } } })));
    await select(/^Node 4, Warmth/);
    await type("Contrast", "20");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(4, { grade: { contrast: { kind: "manual", value: 20 } } })));
    await type("Contrast", "0");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(4, { grade: { contrast: { kind: "manual", value: 0 } } })));
    expect(corrector(4).grade.contrast).toEqual({ kind: "manual", value: 0 });
  });

  it("Highlights, Shadows and Black follow it, and the card's Reset is Auto on Main and a manual 0 elsewhere", async () => {
    const spy = await setup();
    await select(/^Node 1, Main/);
    await h!.user.click(within(regionNamed("High dynamic range")).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(1, { grade: { highlights: { kind: "auto" }, shadows: { kind: "auto" }, white: 0, black: { kind: "auto" } } })));
    await select(/^Node 4, Warmth/);
    await h!.user.click(within(regionNamed("High dynamic range")).getByRole("button", { name: "Reset" }));
    const zero = { kind: "manual", value: 0 } as const;
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(4, { grade: { highlights: zero, shadows: zero, white: 0, black: zero } })));
  });

  it("the Colour balance wheels and Saturation of another corrector are plain values; Reset writes manual zeros", async () => {
    const spy = await setup();
    await select(/^Node 4, Warmth/);
    await openNodeTab("Colour");
    spy.mockClear();
    await type("Saturation", "15");
    await waitFor(() => expect(patches(spy).at(-1)).toEqual(patchOf(4, { grade: { saturation: { kind: "manual", value: 15 } } })));
    await h!.user.click(within(regionNamed("Colour balance")).getAllByRole("button", { name: "Reset" })[0]!);
    const centre: BalanceSetting = { kind: "manual", point: [0, 0] };
    await waitFor(() =>
      expect(patches(spy).at(-1)).toEqual(patchOf(4, { grade: { color_balance: { shadows: centre, midtones: [0, 0], highlights: centre }, saturation: { kind: "manual", value: 0 } } })),
    );
  });

  it("Main's own card has the same cards, with Auto on its Contrast, and no AUTO tag", async () => {
    await setup();
    await select(/^Node 1, Main/);
    expect(cardTitles()).toEqual(["Exposure", "High dynamic range"]);
    expect(within(regionNamed("Exposure")).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Exposure trim", "Contrast", "Brightness"]);
    expect(within(tools()).queryByRole("button", { name: "Auto" })).toBeNull();
  });
});

describe("the Mask tab until N-12", () => {
  it("lists the node's rows, read only, with the line 'Masks add up…' and Add mask off", async () => {
    await setup();
    await select(/^Node 3, Sky/);
    await openNodeTab("Mask");
    expect(cardTitles()).toEqual(["Masks", "View"]);
    const rows = within(within(tools()).getByRole("list", { name: "Masks of this node" })).getAllByRole("listitem");
    expect(rows.map((r) => r.getAttribute("aria-label"))).toEqual(["Object mask, 2 points", "Radial mask, Feather 45"]);
    expect(rows[0]).toHaveTextContent("Off");
    expect(rows[1]).toHaveTextContent("Subtract");
    expect(within(tools()).getByText("Masks add up. Subtract cuts a mask out. Colour and brightness ranges keep only their part of the rest.")).toBeInTheDocument();
    const add = within(tools()).getByRole("button", { name: "Add mask" });
    expect(add).toHaveAttribute("aria-disabled", "true");
    expect(add).toHaveAccessibleDescription("Masks come in the next step");
    expect(within(tools()).queryByRole("button", { name: "Invert" })).toBeNull();
    // Nothing is a control: a row is not a button.
    expect(within(tools()).queryAllByRole("button", { name: /mask,/ })).toEqual([]);
  });

  it("has no rows for a node without masks, and still the line and the button", async () => {
    await setup();
    await select(/^Node 4, Warmth/);
    await openNodeTab("Mask");
    expect(within(tools()).queryByRole("list", { name: "Masks of this node" })).toBeNull();
    expect(within(tools()).getByText(/^Masks add up\./)).toBeInTheDocument();
    expect(within(tools()).getByRole("button", { name: "Add mask" })).toHaveAttribute("aria-disabled", "true");
  });
});

describe("the View card (N-11)", () => {
  const viewCard = (): HTMLElement => within(tools()).getByRole("region", { name: "View" });
  const styles = (): HTMLElement => within(viewCard()).getByRole("radiogroup", { name: "Show mask" });

  it("is the last card of the Mask tab: 'View', the key chip M, and one row, Show mask, with Overlay and Mask only (no Off)", async () => {
    await setup();
    await select(/^Node 3, Sky/);
    await openNodeTab("Mask");
    expect(cardTitles()).toEqual(["Masks", "View"]);
    expect(within(viewCard()).getByText("M")).toBeInTheDocument();
    expect(within(viewCard()).getByText("Show mask")).toBeInTheDocument();
    const radios = within(styles()).getAllByRole("radio");
    expect(radios.map((r) => r.textContent)).toEqual(["Overlay", "Mask only"]);
    expect(radios[0]).toHaveAttribute("aria-checked", "true");
    expect(radios[1]).toHaveAttribute("aria-checked", "false");
  });

  it("sets the style and turns the view on: a segment, the one already chosen too", async () => {
    await setup();
    await select(/^Node 3, Sky/);
    await openNodeTab("Mask");
    expect(h!.store.getState().maskView).toEqual({ on: false, style: "overlay" });
    // The chosen segment: the view was off, so it is on now, with the same style.
    await h!.user.click(within(styles()).getByRole("radio", { name: "Overlay" }));
    await waitFor(() => expect(h!.store.getState().maskView).toEqual({ on: true, style: "overlay" }));
    expect(h!.api.viewport.maskView).toEqual({ node: 3, style: "overlay" });
    await h!.user.click(within(styles()).getByRole("radio", { name: "Mask only" }));
    await waitFor(() => expect(h!.store.getState().maskView).toEqual({ on: true, style: "mask_only" }));
    expect(h!.api.viewport.maskView).toEqual({ node: 3, style: "mask_only" });
    // The toolbar's toggle turns it off and keeps the style; the card still shows it.
    await h!.user.click(screen.getByRole("button", { name: "Show mask" }));
    await waitFor(() => expect(h!.store.getState().maskView).toEqual({ on: false, style: "mask_only" }));
    expect(within(styles()).getByRole("radio", { name: "Mask only" })).toHaveAttribute("aria-checked", "true");
    expect(h!.api.viewport.maskView).toBeNull();
    // An arrow key moves the selection and turns the view on, as a click does.
    within(styles()).getByRole("radio", { name: "Mask only" }).focus();
    await h!.user.keyboard("{ArrowLeft}");
    await waitFor(() => expect(h!.store.getState().maskView).toEqual({ on: true, style: "overlay" }));
  });
});

describe("W in the node view", () => {
  it("selects the Input, opens its Colour tab and starts Pick white; W again ends it", async () => {
    await setup();
    await select(/^Node 3, Sky/);
    await h!.user.keyboard("w");
    await screen.findByText("Click something neutral: grey or white. Esc cancels.");
    expect(within(head()).getByRole("heading", { name: "Input" })).toBeInTheDocument();
    expect(selectedTab()).toBe("Colour");
    expect(within(tools()).getByRole("button", { name: "Pick white" })).toHaveAttribute("aria-pressed", "true");
    await h!.user.keyboard("w");
    await waitFor(() => expect(screen.queryByText("Click something neutral: grey or white. Esc cancels.")).toBeNull());
    expect(within(tools()).getByRole("button", { name: "Pick white" })).toHaveAttribute("aria-pressed", "false");
    // From the Output too, whatever tab the Input was on.
    await h!.user.click(within(tools()).getByRole("tab", { name: "RAW" }));
    await select("Output");
    await h!.user.keyboard("w");
    await screen.findByText("Click something neutral: grey or white. Esc cancels.");
    expect(selectedTab()).toBe("Colour");
    // A pick sets the white and ends the mode.
    const layer = await screen.findByRole("application", { name: /^Pick white\./ });
    mockRect(layer, { left: 0, top: 0, width: 900, height: 600 });
    fireEvent.click(layer, { clientX: 450, clientY: 100 });
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.find((n) => n.kind === "input")).toMatchObject({ white_pick: expect.any(Array) }));
    await waitFor(() => expect(screen.queryByText("Click something neutral: grey or white. Esc cancels.")).toBeNull());
  });

  it("ends when the node changes, as it ends when the tab is left", async () => {
    await setup();
    await h!.user.keyboard("w");
    await screen.findByText("Click something neutral: grey or white. Esc cancels.");
    await select(/^Node 2, Mountain/);
    await waitFor(() => expect(screen.queryByText("Click something neutral: grey or white. Esc cancels.")).toBeNull());
    await select("Input");
    expect(within(tools()).getByRole("button", { name: "Pick white" })).toHaveAttribute("aria-pressed", "false");
    // Pick colour on a corrector ends with the node too.
    await select(/^Node 2, Mountain/);
    await openNodeTab("Colour");
    await h!.user.click(within(tools()).getByRole("button", { name: "Pick colour" }));
    await screen.findByText("Click a colour in the photo.");
    await select(/^Node 3, Sky/);
    await waitFor(() => expect(screen.queryByText("Click a colour in the photo.")).toBeNull());
    expect(within(tools()).getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "false");
  });

  it("is the simple view's W when the node view is off", async () => {
    await setup();
    await h!.user.keyboard("n");
    await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "false"));
    await h!.user.keyboard("w");
    await screen.findByText("Click something neutral: grey or white. Esc cancels.");
    expect(screen.getByRole("tab", { name: "Colour", selected: true })).toBeInTheDocument();
  });
});

describe("the histogram, the scatter and Pick colour read the signal at the selected node (`at`)", () => {
  it("names no node in the simple view, and the node in the node view: the Input's output for the Input, the corrector's id for a corrector", async () => {
    h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: FOUR } });
    await openPhotos(h);
    const histogram = vi.spyOn(h.api, "histogram");
    const scatter = vi.spyOn(h.api, "chromaScatter");
    const id = h.store.currentPhoto()!.id;
    // The simple view reads Main's, and says nothing of it.
    await openTab(h, "Light");
    await waitFor(() => expect(histogram).toHaveBeenCalledWith(id));
    expect(histogram.mock.calls.every((c) => c.length === 1)).toBe(true);
    await openTab(h, "Colour");
    await waitFor(() => expect(scatter).toHaveBeenCalledWith(id));
    expect(scatter.mock.calls.every((c) => c.length === 1)).toBe(true);
    // The node view: Main's Light tab leaves `at` out; another corrector's names it.
    await h.user.keyboard("n");
    await openNodeTab("Light");
    histogram.mockClear();
    await select(/^Node 3, Sky/);
    await waitFor(() => expect(histogram).toHaveBeenCalledWith(id, 3));
    await select("Input");
    await openNodeTab("Light");
    await waitFor(() => expect(histogram).toHaveBeenCalledWith(id, "input"));
    // The Curve tab and the Colour warp's scatter name the node too.
    await select(/^Node 2, Mountain/);
    histogram.mockClear();
    await openNodeTab("Curve");
    await waitFor(() => expect(histogram).toHaveBeenCalledWith(id, 2));
    await openNodeTab("Colour");
    await waitFor(() => expect(scatter).toHaveBeenCalledWith(id, 2));
    // Main's own readings leave `at` out, as before nodes.
    scatter.mockClear();
    await select(/^Node 1, Main/);
    await waitFor(() => expect(scatter).toHaveBeenCalledWith(id));
    expect(scatter.mock.calls.every((c) => c.length === 1)).toBe(true);
  });

  it("Pick colour on a corrector reads the chroma at that node (`sample_chroma` with `at`) and puts the pin in that node", async () => {
    await setup();
    await select(/^Node 2, Mountain/);
    await openNodeTab("Colour");
    const sample = vi.spyOn(h!.api, "sampleChroma");
    await h!.user.click(within(tools()).getByRole("button", { name: "Pick colour" }));
    const layer = await screen.findByRole("application", { name: /^Pick colour\./ });
    mockRect(layer, { left: 0, top: 0, width: 900, height: 600 });
    await h!.api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    fireEvent.click(layer, { clientX: 450, clientY: 100 });
    await waitFor(() => expect(corrector(2).grade.color_warp.pins).toHaveLength(1));
    expect(sample).toHaveBeenCalledWith(450, 100, 2);
    // The colour is the one at the input of this node's warp: the photo's sky after the node's own trim (node 2 has +0.3 EV), not Main's.
    const there = await h!.api.sampleChroma(450, 100, 2);
    const mains = await h!.api.sampleChroma(450, 100);
    expect(corrector(2).grade.color_warp.pins[0]?.source).toEqual([there!.a, there!.b]);
    expect(there).not.toEqual(mains);
    expect(mains).toEqual({ a: hexChroma(SCENES[3]!.sky)[0], b: hexChroma(SCENES[3]!.sky)[1] });
    // Main's pins are untouched.
    expect((nodeOf(h!.store.currentSettings()!.settings, 1) as CorrectorItem).grade.color_warp.pins).toHaveLength(0);
    // The selected pin is this node's: another node starts with none selected.
    expect(h!.store.getState().warpPin).toBe(0);
    await select(/^Node 3, Sky/);
    await waitFor(() => expect(h!.store.getState().warpPin).toBeNull());
  });
});

describe("undo, Copy settings and Paste work in the node view as in the simple view", () => {
  it("undo takes back one change of one node at a time, across nodes", async () => {
    await setup();
    await select(/^Node 2, Mountain/);
    await type("Exposure trim", "1");
    await waitFor(() => expect(corrector(2).trim.exposure_ev).toBe(1));
    await select(/^Node 4, Warmth/);
    await type("Exposure trim", "-1");
    await waitFor(() => expect(corrector(4).trim.exposure_ev).toBe(-1));
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(corrector(4).trim.exposure_ev).toBe(0));
    expect(corrector(2).trim.exposure_ev).toBe(1);
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(corrector(2).trim.exposure_ev).toBe(0.3));
    await h!.user.keyboard("{Meta>}{Shift>}z{/Shift}{/Meta}");
    await waitFor(() => expect(corrector(2).trim.exposure_ev).toBe(1));
  });

  it("Copy settings and Paste carry the whole graph, and the foot row stays under the panel", async () => {
    await setup();
    // The panel is drawn again for another photo: ask for the row each time.
    const foot = (): HTMLElement => within(tools()).getByRole("group", { name: "Copy and paste settings" });
    expect(within(foot()).getAllByRole("button").map((b) => b.textContent)).toEqual(["Copy settings", "Paste"]);
    await select(/^Node 4, Warmth/);
    await type("Exposure trim", "0.7");
    await waitFor(() => expect(corrector(4).trim.exposure_ev).toBe(0.7));
    await h!.user.click(within(foot()).getByRole("button", { name: "Copy settings" }));
    await waitFor(() => expect(h!.store.getState().copied).not.toBeNull());
    // Another photo: Paste replaces its graph with the whole copy.
    await h!.user.keyboard("{ArrowLeft}");
    await waitFor(() => expect(h!.store.currentPhoto()?.file_name).not.toBe("DSC04204.ARW"));
    await h!.user.click(within(foot()).getByRole("button", { name: "Paste" }));
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.filter((n) => n.kind === "corrector")).toHaveLength(5 - 1));
    expect(corrector(4).trim.exposure_ev).toBe(0.7);
    expect(corrector(2).name).toBe("Mountain");
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.filter((n) => n.kind === "corrector")).toHaveLength(1));
  });
});
