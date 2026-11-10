/**
 * The panels of the four kinds of `docs/25` (§5.5) on the mock, as the app runs in a browser: the heads and their buttons, the one tab of each
 * kind, the Layer's Blend card (Mode, Opacity, Reset) and Inputs rows with their menu, the Parallel's sentence and base, the Splitter's and the
 * Combiner's channel rows.
 */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { NodeLink, SettingsPatch } from "../../api/types";
import { nodeOf } from "../../state/settingsGraph";
import { openPhotos, openTab, renderApp, tools } from "../../test/harness";
import type { Harness, UpdateSpy } from "../../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

const link = (from: number | "input", to: number | "output", toPort: NodeLink["to"]["port"] = "picture", fromPort: NodeLink["from"]["port"] = "picture"): NodeLink => ({
  from: { node: from, port: fromPort },
  to: { node: to, port: toPort },
});

/** Main feeds Warmth (2), Glow (3) and Sky (4): the inputs 1 to 3 of a Layer (5) that feeds the Output. */
const LAYER: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Warmth", pos: [1, -1] },
    "3": { kind: "corrector", name: "Glow", pos: [1, 0] },
    "4": { kind: "corrector", name: "Sky", pos: [1, 1] },
    "5": { kind: "layer", pos: [2, 0] },
  },
  links: [link("input", 1), link(1, 2), link(1, 3), link(1, 4), link(2, 5, "picture.1"), link(3, 5, "picture.2"), link(4, 5, "picture.3"), link(5, "output")],
};

/** Main feeds Skin (2) and Sky (3), the inputs of a Parallel (4); a Splitter (5) and a Combiner (6) sit apart, not connected. */
const PARALLEL: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Skin", pos: [1, -1] },
    "3": { kind: "corrector", name: "Sky", pos: [1, 1] },
    "4": { kind: "parallel", pos: [2, 0] },
    "5": { kind: "splitter", pos: [1, 2] },
    "6": { kind: "combiner", pos: [2, 2] },
  },
  links: [link("input", 1), link(1, 2), link(1, 3), link(2, 4, "picture.1"), link(3, 4, "picture.2"), link(4, "output"), link(1, 5), link(5, 6, "red", "red"), link(5, 3, "picture", "blue")].filter(
    // Sky has one picture link: Main's; the Splitter's blue goes to nowhere here.
    (l) => !(l.to.node === 3 && l.from.node === 5),
  ),
};

async function setup(settings: SettingsPatch): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: settings } });
  await openPhotos(h);
  await openTab(h, "Light");
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  return vi.spyOn(h.api, "updateSettings");
}
const head = (): HTMLElement => within(tools()).getByRole("group", { name: "Node" });
const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
async function select(name: RegExp | string): Promise<void> {
  fireEvent.click(screen.getByRole("button", { name }));
  await waitFor(() => expect(screen.getByRole("button", { name })).toHaveAttribute("aria-pressed", "true"));
}
const links = (): NodeLink[] => h!.store.currentSettings()!.settings.nodes.links;
const has = (from: number, to: number, toPort: string): boolean => links().some((l) => l.from.node === from && l.to.node === to && l.to.port === toPort);

describe("the Layer's panel", () => {
  it("has the head 'Mixer · 3 inputs', Bypass, Reset and More, and the one tab Blend", async () => {
    await setup(LAYER);
    await select(/^Layer, mixer with 3 inputs/);
    expect(within(head()).getByRole("heading", { level: 2, name: "Layer" })).toBeInTheDocument();
    expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Mixer · 3 inputs");
    expect(head().querySelector(".ym-nodehead__chip svg")).not.toBeNull();
    expect(within(within(head()).getByRole("group", { name: "Node actions" })).getAllByRole("button").map((b) => b.getAttribute("aria-label"))).toEqual(["Bypass this node", "Reset this node", "More"]);
    expect(within(tools()).getAllByRole("tab").map((t) => t.textContent)).toEqual(["Blend"]);
    expect(within(tools()).getAllByRole("heading", { level: 3 }).map((e) => e.textContent)).toEqual(["Blend", "Inputs"]);
    expect(tools().querySelector('[data-tab="blend"]')).not.toBeNull();
  });

  it("lists the inputs from the top of the stack, with On top and Below, and the sentence under them", async () => {
    await setup(LAYER);
    await select(/^Layer, mixer/);
    const rows = [...tools().querySelectorAll<HTMLElement>("[data-input-row]")];
    expect(rows.map((r) => r.querySelector(".ym-input-row__num")?.textContent)).toEqual(["3", "2", "1"]);
    expect(rows.map((r) => r.querySelector(".ym-input-row__name")?.textContent)).toEqual(["04 Sky", "03 Glow", "02 Warmth"]);
    expect(rows.map((r) => r.querySelector(".ym-input-row__sub")?.textContent ?? "")).toEqual(["On top", "", "Below"]);
    expect(within(tools()).getByText("Drag a row to change which input sits on top.")).toBeInTheDocument();
  });

  it("sets the Mode from the seven names, with no lines, and Opacity in steps of one percent, each one patch", async () => {
    const spy = await setup(LAYER);
    await select(/^Layer, mixer/);
    await h!.user.click(within(tools()).getByRole("combobox", { name: "Blend mode" }));
    const options = await screen.findAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["Normal", "Add", "Subtract", "Multiply", "Lighten", "Darken", "Difference"]);
    await h!.user.click(screen.getByRole("option", { name: "Multiply" }));
    await waitFor(() => expect(patches(spy).some((p) => (p.nodes?.["5"] as { mode?: string } | undefined)?.mode === "multiply")).toBe(true));
    const box = screen.getByRole("textbox", { name: "Opacity value" });
    await h!.user.click(box);
    await h!.user.clear(box);
    await h!.user.type(box, "40{Enter}");
    await waitFor(() => expect(nodeOf(h!.store.currentSettings()!.settings, 5)).toMatchObject({ mode: "multiply", opacity: 0.4 }));
  });

  it("Reset on the Blend card puts Mode to Normal and Opacity to 100 %", async () => {
    await setup({ ...LAYER, nodes: { ...LAYER.nodes, "5": { kind: "layer", pos: [2, 0], mode: "add", opacity: 0.3 } } });
    await select(/^Layer, mixer/);
    await h!.user.click(within(regionOf("Blend")).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(nodeOf(h!.store.currentSettings()!.settings, 5)).toMatchObject({ mode: "normal", opacity: 1 }));
  });

  it("renumbers the ports from a row's menu: Move up, Move down and Remove input; one patch each", async () => {
    const spy = await setup(LAYER);
    await select(/^Layer, mixer/);
    const row = (n: number): HTMLElement => tools().querySelector<HTMLElement>(`[data-input-row="${n}"]`)!;
    // Warmth (input 1) moves up: Glow is now 1 and Warmth 2.
    fireEvent.contextMenu(row(1));
    await h!.user.click(await screen.findByRole("menuitem", { name: "Move up" }));
    await waitFor(() => expect(has(2, 5, "picture.2")).toBe(true));
    expect(has(3, 5, "picture.1")).toBe(true);
    expect(patches(spy).at(-1)).toEqual({ links: expect.any(Array) });
    // Remove the top input (Sky): the rest stay 1 and 2.
    fireEvent.contextMenu(row(3));
    await h!.user.click(await screen.findByRole("menuitem", { name: "Remove input" }));
    await waitFor(() => expect(has(4, 5, "picture.3")).toBe(false));
    expect(links().filter((l) => l.to.node === 5 && l.to.port.startsWith("picture.")).map((l) => l.to.port).sort()).toEqual(["picture.1", "picture.2"]);
  });

  it("is off for Show mask and Delete is on; Bypass toggles", async () => {
    await setup(LAYER);
    await select(/^Layer, mixer/);
    expect(screen.getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("button", { name: "Show mask" })).toHaveAccessibleDescription("This node has no mask.");
    await h!.user.click(within(head()).getByRole("button", { name: "Bypass this node" }));
    await waitFor(() => expect((nodeOf(h!.store.currentSettings()!.settings, 5) as { bypass: boolean }).bypass).toBe(true));
    expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Mixer · bypassed");
  });
});

/** A region of the panel by its card title. */
function regionOf(name: string): HTMLElement {
  return within(tools()).getByRole("region", { name });
}

describe("the Parallel's panel", () => {
  it("says what it does, where the inputs start from and lists them 1 to N with no handle and no words; Reset is off", async () => {
    await setup(PARALLEL);
    await select(/^Parallel, mixer with 2 inputs/);
    expect(within(tools()).getAllByRole("tab").map((t) => t.textContent)).toEqual(["Inputs"]);
    expect(within(tools()).getByText("Adds the change each input makes to the picture they start from.")).toBeInTheDocument();
    expect(within(tools()).getByText("They start from node 1, Main.")).toBeInTheDocument();
    const rows = [...tools().querySelectorAll<HTMLElement>("[data-input-row]")];
    expect(rows.map((r) => r.querySelector(".ym-input-row__name")?.textContent)).toEqual(["02 Skin", "03 Sky"]);
    expect(tools().querySelector("[data-grip], .ym-input-row__sub")).toBeNull();
    const reset = within(head()).getByRole("button", { name: "Reset this node" });
    expect(reset).toHaveAttribute("aria-disabled", "true");
    expect(reset).toHaveAccessibleDescription("This node has nothing to reset.");
  });

  it("offers Remove input only, and the head says '1 input' after one is removed", async () => {
    await setup(PARALLEL);
    await select(/^Parallel, mixer/);
    fireEvent.contextMenu(tools().querySelector<HTMLElement>('[data-input-row="2"]')!);
    expect(await screen.findByRole("menuitem", { name: "Remove input" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Move up" })).toBeNull();
    await h!.user.click(screen.getByRole("menuitem", { name: "Remove input" }));
    await waitFor(() => expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Mixer · 1 input"));
  });
});

describe("the Splitter's and the Combiner's panels", () => {
  it("say what the node does and where each channel goes or comes from; Bypass and Reset are off", async () => {
    await setup(PARALLEL);
    await select(/^Splitter, not connected, picture from node 1/);
    expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Channels · not connected");
    expect(within(tools()).getAllByRole("tab").map((t) => t.textContent)).toEqual(["Channels"]);
    expect(within(tools()).getByText(/^Sends the red, green and blue of the picture to three outputs\./)).toBeInTheDocument();
    const rows = [...tools().querySelectorAll<HTMLElement>(".ym-channel-row")];
    expect(rows.map((r) => r.textContent)).toEqual(["RRedTo Combiner, red", "GGreenNot linked", "BBlueNot linked"]);
    const bypass = within(head()).getByRole("button", { name: "Bypass this node" });
    expect(bypass).toHaveAttribute("aria-disabled", "true");
    expect(bypass).toHaveAccessibleDescription("A splitter or a combiner can't be bypassed.");
    expect(within(head()).getByRole("button", { name: "Reset this node" })).toHaveAttribute("aria-disabled", "true");
    await select(/^Combiner, not connected, red from Splitter, red/);
    expect(within(tools()).getByText(/^Makes one picture: the red of its red input/)).toBeInTheDocument();
    expect([...tools().querySelectorAll<HTMLElement>(".ym-channel-row")].map((r) => r.textContent)).toEqual([
      "RRedFrom Splitter, red",
      "GGreenFrom the input",
      "BBlueFrom the input",
    ]);
  });
});

describe("the Add node menu", () => {
  const items = async (): Promise<HTMLElement> => {
    await h!.user.click(screen.getByRole("button", { name: "Add node" }));
    return screen.findByRole("menu", { name: "Add node" });
  };

  it("Parallel adds a Parallel and a corrector beside the selected node, selects the corrector and is one undo step", async () => {
    const spy = await setup({ links: [link("input", 1), link(1, "output")] });
    const menu = await items();
    await h!.user.click(within(menu).getByRole("menuitem", { name: /^Parallel/ }));
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.map((i) => i.kind)).toEqual(["input", "corrector", "parallel", "corrector", "output"]));
    expect(patches(spy)).toHaveLength(1);
    // The new corrector is selected, with its own tab open (Light).
    await waitFor(() => expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Corrector · whole photo"));
    expect(within(tools()).getAllByRole("tab").map((t) => t.textContent)).toEqual(["Light", "Colour", "Curve", "Mask"]);
    expect(has(1, 2, "picture.1")).toBe(true);
    expect(has(3, 2, "picture.2")).toBe(true);
    await h!.store.undo();
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items).toHaveLength(3));
  });

  it("Splitter and combiner adds the five nodes in one patch and selects the Splitter", async () => {
    const spy = await setup({ links: [link("input", 1), link(1, "output")] });
    const menu = await items();
    await h!.user.click(within(menu).getByRole("menuitem", { name: /^Splitter and combiner/ }));
    await waitFor(() => expect(h!.store.currentSettings()!.settings.nodes.items.map((i) => i.kind)).toEqual(["input", "corrector", "splitter", "corrector", "corrector", "corrector", "combiner", "output"]));
    expect(patches(spy)).toHaveLength(1);
    await waitFor(() => expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Channels · 3 outputs"));
  });
});
