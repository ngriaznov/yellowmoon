/**
 * The graph's canvas in the browser's stand-in (`docs/23` §7.2 and §7.8): the cards and the markers with their names, numbers, glyphs and
 * states, the one Tab stop and the arrow keys between the nodes, the rename box, and the line a graph with Main alone carries. What
 * needs a real pointer and real geometry (links dragged, cards dropped, zoom and pan) is `e2e/nodes.spec.ts`; the rules behind it are
 * `src/state/graphOps.test.ts`.
 */
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { DevelopSettings, NodeId, NodeLink, NodePatch } from "../../api/types";
import { sortedLinks } from "../../state/graphOps";
import { applyPatch, defaultSettings } from "../../state/settingsGraph";
import { EMPTY_GRAPH_HINT, NodeGraph } from "./NodeGraph";
import type { NodeGraphProps } from "./NodeGraph";
import { maskGlyphsOf, NAME_CUT, shortName } from "./NodeCard";
import { nodeOf } from "../../state/settingsGraph";

const pic = (from: NodeId, to: NodeId): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

const row = (kind: string): never => ({ kind, mode: "add", invert: false, enabled: true, opacity: 1 }) as never;

/** Main → Mountain (Object and Radial masks) → Sky (bypassed) → Warmth, and a spare node that nothing reads. */
function graph(): DevelopSettings {
  const nodes: Record<string, NodePatch> = {
    "2": { kind: "corrector", name: "Mountain", pos: [1, 0], masks: { invert: false, items: [row("object"), row("radial")] } },
    "3": { kind: "corrector", name: "Sky", pos: [2, 0], bypass: true, masks: { invert: false, items: [row("linear")] } },
    "4": { kind: "corrector", name: "Warmth", pos: [3, 0] },
    "5": { kind: "corrector", name: "A long name for a spare", pos: [1, 1] },
  };
  return applyPatch(defaultSettings(), { nodes, links: sortedLinks([pic("input", 1), pic(1, 2), pic(2, 3), pic(3, 4), pic(4, "output"), pic(1, 5)]) });
}

function setup(over: Partial<NodeGraphProps> = {}) {
  const props: NodeGraphProps = {
    settings: graph(),
    photoKey: 1,
    selected: 1,
    selectedLink: null,
    renaming: null,
    thumbnailOf: () => "blob:photo",
    onSelect: vi.fn(),
    onSelectLink: vi.fn(),
    onStartRename: vi.fn(),
    onRename: vi.fn(),
    onMove: vi.fn(),
    onReorder: vi.fn(),
    onConnect: vi.fn(),
    onReconnect: vi.fn(),
    onDisconnect: vi.fn(),
    onRefuse: vi.fn(),
    ...over,
  };
  const view = render(<NodeGraph {...props} />);
  return { props, ...view };
}

const card = (id: NodeId): HTMLElement => document.querySelector<HTMLElement>(`[data-node-card="${id}"]`)!;
const gnode = (id: NodeId): HTMLElement => card(id).closest<HTMLElement>(".ym-gnode")!;

describe("the names", () => {
  it("cut a name at about ten letters, with an ellipsis, and keep a short one whole", () => {
    expect(NAME_CUT).toBe(10);
    expect(shortName("Corrector")).toBe("Corrector");
    expect(shortName("Mountain")).toBe("Mountain");
    expect(shortName("Ten letter")).toBe("Ten letter");
    expect(shortName("Eleven lett")).toBe("Eleven le…");
    expect(shortName("A very long name for a node")).toBe("A very lo…");
    expect(shortName("é".repeat(12))).toBe(`${"é".repeat(9)}…`);
  });
});

describe("the Input, the cards and the Output", () => {
  it("are one button each, named for a screen reader: number, name, masks, bypass, 'not connected' and what feeds them", () => {
    setup();
    const names = screen.getAllByRole("button").map((b) => b.getAttribute("aria-label"));
    expect(names).toContain("Input");
    expect(names).toContain("Output");
    expect(card(1)).toHaveAccessibleName("Node 1, Main");
    expect(card(2)).toHaveAccessibleName("Node 2, Mountain, masked by Object and Radial, picture from node 1");
    // The spare node (id 5) reads Main and nothing reads it: it is not on a path to the Output. It stands in the second row, left of Sky, so
    // the plan numbers it 3 (Main first, then by place), and Sky 4 and Warmth 5.
    expect(card(5)).toHaveAccessibleName("Node 3, A long name for a spare, not connected, picture from node 1");
    expect(card(3)).toHaveAccessibleName("Node 4, Sky, masked by Linear, bypassed, picture from node 2");
    expect(card(4)).toHaveAccessibleName("Node 5, Warmth, picture from node 4");
    expect(card("input")).toHaveAccessibleName("Input");
    expect(card("output")).toHaveAccessibleName("Output");
  });

  it("show the number as two digits in the card's chip, in the plan's order (ids 1, 2, 5, 3, 4 are 01 to 05)", () => {
    setup();
    expect([1, 2, 5, 3, 4].map((id) => gnode(id).querySelector(".ym-gnode__num")?.textContent)).toEqual(["01", "02", "03", "04", "05"]);
  });

  it("show the name above the card, cut, and the Input and the Output by their names", () => {
    setup();
    expect(gnode(5).querySelector(".ym-gnode__name")).toHaveTextContent("A long na…");
    expect(gnode(2).querySelector(".ym-gnode__name")).toHaveTextContent("Mountain");
    expect(document.querySelector('.ym-marker[data-which="input"] .ym-marker__word')).toHaveTextContent("Input");
    expect(document.querySelector('.ym-marker[data-which="output"] .ym-marker__word')).toHaveTextContent("Output");
  });

  it("have a thumbnail band with the picture the page gives, and none without one", () => {
    const { unmount } = setup();
    for (const id of [1, 2, 3, 4, 5]) expect(gnode(id).querySelector(".ym-gnode__thumb img")).toHaveAttribute("src", "blob:photo");
    unmount();
    setup({ thumbnailOf: () => undefined });
    expect(document.querySelectorAll(".ym-gnode__thumb img")).toHaveLength(0);
    expect(document.querySelectorAll(".ym-gnode__thumb")).toHaveLength(5);
  });

  it("show a sparkle for an AI mask and a half disc for any other, and a slash for a bypass", () => {
    setup();
    const glyphs = (id: NodeId): string[] => [...gnode(id).querySelectorAll("[data-glyph]")].map((g) => g.getAttribute("data-glyph") ?? "");
    expect(glyphs(1)).toEqual([]);
    expect(glyphs(2)).toEqual(["ai", "mask"]);
    expect(glyphs(3)).toEqual(["mask", "bypass"]);
    expect(glyphs(4)).toEqual([]);
    expect(gnode(3).querySelector(".ym-gnode__slash")).not.toBeNull();
    expect(gnode(2).querySelector(".ym-gnode__slash")).toBeNull();
  });

  it("name the glyphs in a tooltip: 'AI mask', 'Mask', 'Bypassed'", async () => {
    setup();
    const user = userEvent.setup();
    await user.hover(gnode(2).querySelector('[data-glyph="ai"]')!);
    expect(await screen.findByRole("tooltip")).toHaveTextContent("AI mask");
    await user.unhover(gnode(2).querySelector('[data-glyph="ai"]')!);
    await user.hover(gnode(3).querySelector('[data-glyph="mask"]')!);
    expect((await screen.findAllByText("Mask", { selector: '[role="tooltip"], [data-slot="tooltip-content"] *' })).length).toBeGreaterThan(0);
  });

  it("count an enabled mask only: a row that is switched off is absent, and Subject, People and Background are AI too", () => {
    const g = graph();
    const two = nodeOf(g, 2);
    if (two?.kind !== "corrector") throw new Error("node 2");
    expect(maskGlyphsOf(two)).toEqual({ ai: true, plain: true });
    const off = { ...two, masks: { ...two.masks, items: two.masks.items.map((r) => ({ ...r, enabled: false })) } };
    expect(maskGlyphsOf(off)).toEqual({ ai: false, plain: false });
    for (const kind of ["subject", "people", "background"]) expect(maskGlyphsOf({ ...two, masks: { invert: false, items: [row(kind)] } })).toEqual({ ai: true, plain: false });
    for (const kind of ["linear", "radial", "brush", "color_range", "brightness_range"]) expect(maskGlyphsOf({ ...two, masks: { invert: false, items: [row(kind)] } })).toEqual({ ai: false, plain: true });
  });

  it("mark the selected card (aria-pressed and the ring), a bypassed one and a dead one", () => {
    setup({ selected: 3 });
    expect(card(3)).toHaveAttribute("aria-pressed", "true");
    expect(card(1)).toHaveAttribute("aria-pressed", "false");
    expect(gnode(3)).toHaveAttribute("data-selected", "true");
    expect(gnode(3)).toHaveAttribute("data-bypass", "true");
    expect(gnode(1)).not.toHaveAttribute("data-selected");
    expect(gnode(5)).toHaveAttribute("data-dead", "true");
    expect(gnode(4)).not.toHaveAttribute("data-dead");
    // A selected link quietens the card's ring.
    const link = pic(2, 3);
    setup({ selected: 2, selectedLink: link });
    expect(document.querySelectorAll(".ym-gnode[data-quiet]")).toHaveLength(1);
  });

  it("say 'Not connected to the output.' for a dead card", async () => {
    setup();
    const user = userEvent.setup();
    await user.hover(card(5));
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Not connected to the output.");
  });

  it("have four ports on a card and one on each marker", () => {
    setup();
    for (const id of [1, 2, 3, 4, 5]) {
      expect(gnode(id).querySelectorAll(".ym-port")).toHaveLength(4);
      expect(gnode(id).querySelectorAll('.ym-port[data-port="picture"]')).toHaveLength(2);
      expect(gnode(id).querySelectorAll('.ym-port[data-port="key"]')).toHaveLength(2);
    }
    expect(document.querySelectorAll('.ym-marker[data-which="input"] .ym-port')).toHaveLength(1);
    expect(document.querySelectorAll('.ym-marker[data-which="output"] .ym-port')).toHaveLength(1);
    // The ports that have a link say so.
    expect(gnode(2).querySelector('.ym-port[data-port="picture"][data-side="in"]')).toHaveAttribute("data-linked", "true");
    expect(gnode(2).querySelector('.ym-port[data-port="key"][data-side="in"]')).not.toHaveAttribute("data-linked");
    expect(gnode(4).querySelector('.ym-port[data-port="picture"][data-side="out"]')).toHaveAttribute("data-linked", "true");
  });

  it("have no tab stop of their own in their ports", () => {
    setup();
    for (const port of document.querySelectorAll(".ym-port, .ym-port__hit")) expect(port).not.toHaveAttribute("tabindex");
  });
});

describe("the keyboard path", () => {
  it("is one Tab stop: the selected card", () => {
    const { unmount } = setup({ selected: 2 });
    const stops = [...document.querySelectorAll<HTMLElement>("[tabindex]")].filter((el) => el.tabIndex >= 0);
    expect(stops.map((el) => el.getAttribute("aria-label"))).toEqual([expect.stringMatching(/^Node 2, Mountain/), ...stops.slice(1).map((el) => el.getAttribute("aria-label"))]);
    expect(document.querySelectorAll('[data-node-card][tabindex="0"]')).toHaveLength(1);
    expect(card(2)).toHaveAttribute("tabindex", "0");
    expect(card(1)).toHaveAttribute("tabindex", "-1");
    unmount();
    // The Input can be the one.
    setup({ selected: "input" });
    expect(card("input")).toHaveAttribute("tabindex", "0");
    expect(document.querySelectorAll('[data-node-card][tabindex="0"]')).toHaveLength(1);
  });

  it("moves the selection with the arrow keys to the neighbour in that direction, and the focus with it", () => {
    const { props } = setup({ selected: 2 });
    card(2).focus();
    fireEvent.keyDown(card(2), { key: "ArrowRight" });
    expect(props.onSelect).toHaveBeenLastCalledWith(3);
    expect(card(3)).toHaveFocus();
    fireEvent.keyDown(card(2), { key: "ArrowLeft" });
    expect(props.onSelect).toHaveBeenLastCalledWith(1);
    // Down from Mountain is the spare node in the row under it; up from Main is nothing.
    fireEvent.keyDown(card(2), { key: "ArrowDown" });
    expect(props.onSelect).toHaveBeenLastCalledWith(5);
    const calls = vi.mocked(props.onSelect).mock.calls.length;
    fireEvent.keyDown(card(1), { key: "ArrowUp" });
    expect(vi.mocked(props.onSelect).mock.calls.length).toBe(calls);
    // The Input and the Output are in the row, at its ends.
    fireEvent.keyDown(card(1), { key: "ArrowLeft" });
    expect(props.onSelect).toHaveBeenLastCalledWith("input");
    fireEvent.keyDown(card(4), { key: "ArrowRight" });
    expect(props.onSelect).toHaveBeenLastCalledWith("output");
  });

  it("selects a card with Return or Space, and an arrow with a modifier is left to the screen's shortcuts", () => {
    const { props } = setup({ selected: 1 });
    card(4).focus();
    fireEvent.keyDown(card(4), { key: "Enter" });
    expect(props.onSelect).toHaveBeenLastCalledWith(4);
    fireEvent.keyDown(card(3), { key: " " });
    expect(props.onSelect).toHaveBeenLastCalledWith(3);
    const calls = vi.mocked(props.onSelect).mock.calls.length;
    fireEvent.keyDown(card(3), { key: "ArrowRight", altKey: true });
    fireEvent.keyDown(card(3), { key: "ArrowRight", metaKey: true });
    expect(vi.mocked(props.onSelect).mock.calls.length).toBe(calls);
  });

  it("selects a card by a click, and the Input and the Output too", () => {
    const { props } = setup();
    // `fireEvent`, not user-event: its mouse events carry no `view`, which the pointer handling of the canvas (d3-drag) reads on a press.
    fireEvent.click(card(3));
    expect(props.onSelect).toHaveBeenLastCalledWith(3);
    fireEvent.click(card("output"));
    expect(props.onSelect).toHaveBeenLastCalledWith("output");
  });
});

describe("renaming", () => {
  it("starts with a double click on the name, and the Input and the Output have no name to click", () => {
    const { props } = setup();
    fireEvent.doubleClick(gnode(4).querySelector(".ym-gnode__name")!);
    expect(props.onStartRename).toHaveBeenCalledWith(4);
    expect(document.querySelectorAll(".ym-marker .ym-gnode__name")).toHaveLength(0);
  });

  it("shows a text box with the name, 32 characters at most; Return accepts and Esc cancels", async () => {
    const { props } = setup({ renaming: 2 });
    const user = userEvent.setup();
    const box = screen.getByRole("textbox", { name: "Node name" });
    expect(box).toHaveValue("Mountain");
    expect(box).toHaveAttribute("maxlength", "32");
    expect(box).toHaveFocus();
    await user.clear(box);
    await user.type(box, "Hills{Enter}");
    expect(props.onRename).toHaveBeenCalledTimes(1);
    expect(props.onRename).toHaveBeenCalledWith(2, "Hills");
  });

  it("is cancelled by Esc: the name is null, once, even when the box is left after it", async () => {
    const { props } = setup({ renaming: 2 });
    const user = userEvent.setup();
    const box = screen.getByRole("textbox", { name: "Node name" });
    await user.type(box, "xyz{Escape}");
    expect(props.onRename).toHaveBeenCalledTimes(1);
    expect(props.onRename).toHaveBeenCalledWith(2, null);
    act(() => box.blur());
    expect(props.onRename).toHaveBeenCalledTimes(1);
  });

  it("is accepted by leaving the box", async () => {
    const { props } = setup({ renaming: 3 });
    const box = screen.getByRole("textbox", { name: "Node name" });
    fireEvent.change(box, { target: { value: "Blue" } });
    act(() => box.blur());
    expect(props.onRename).toHaveBeenCalledWith(3, "Blue");
  });

  it("keeps its keys to itself: no shortcut of the screen sees them", () => {
    setup({ renaming: 2 });
    const seen = vi.fn();
    window.addEventListener("keydown", seen);
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Node name" }), { key: "b" });
    window.removeEventListener("keydown", seen);
    expect(seen).not.toHaveBeenCalled();
  });
});

describe("the empty graph", () => {
  it("says 'Add a node to change only part of the photo.' with Main alone, and not with more", () => {
    const { unmount } = setup({ settings: defaultSettings() });
    expect(screen.getByText(EMPTY_GRAPH_HINT)).toBeInTheDocument();
    expect(EMPTY_GRAPH_HINT).toBe("Add a node to change only part of the photo.");
    unmount();
    setup();
    expect(screen.queryByText(EMPTY_GRAPH_HINT)).toBeNull();
  });

  it("draws Main between the Input and the Output: three buttons and no more", () => {
    setup({ settings: defaultSettings() });
    expect(within(document.body).getAllByRole("button").map((b) => b.getAttribute("aria-label"))).toEqual(
      expect.arrayContaining(["Input", "Node 1, Main", "Output"]),
    );
    expect(document.querySelectorAll("[data-node-card]")).toHaveLength(3);
  });
});
