/**
 * The glyph cards of `docs/25` §5.2 in the browser's stand-in: the size and ports of each kind (`glyphGeometry`, `layoutGraph`), the card's
 * name, glyph, states and the letters of the channel ports. What needs a real pointer (links dragged to the "+") is `e2e/mixers.spec.ts`.
 */
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { DevelopSettings, NodeId, NodeLink, NodePatch, Port } from "../../api/types";
import { sortedLinks } from "../../state/graphOps";
import { applyPatch, defaultSettings } from "../../state/settingsGraph";
import { glyphGeometry, layoutGraph, portPoint, PICTURE_PORT_Y } from "./graphLayout";
import { NodeGraph } from "./NodeGraph";
import type { NodeGraphProps } from "./NodeGraph";

const link = (from: NodeId, to: NodeId, toPort: Port = "picture", fromPort: Port = "picture"): NodeLink => ({ from: { node: from, port: fromPort }, to: { node: to, port: toPort } });

function build(nodes: Record<string, NodePatch>, links: NodeLink[]): DevelopSettings {
  return applyPatch(defaultSettings(), { nodes, links: sortedLinks(links) });
}

/** Main → A and B → a Layer with two inputs → Output, and a Splitter and a Combiner off to the side. */
const settings = (): DevelopSettings =>
  build(
    {
      "2": { kind: "corrector", name: "Skin", pos: [1, -1] },
      "3": { kind: "corrector", name: "Sky", pos: [1, 1] },
      "4": { kind: "layer", pos: [2, 0] },
    },
    [link("input", 1), link(1, 2), link(1, 3), link(2, 4, "picture.1"), link(3, 4, "picture.2"), link(4, "output")],
  );

const splitterGraph = (bypass = false): DevelopSettings =>
  build(
    {
      "2": { kind: "splitter", pos: [1, 0] },
      "3": { kind: "combiner", pos: [2, 0] },
      "4": { kind: "parallel", pos: [3, 0], ...(bypass ? { bypass: true } : {}) },
    },
    [link("input", 1), link(1, 2), link(2, 3, "red", "red"), link(2, 3, "blue", "blue"), link(3, 4, "picture.1"), link(4, "output")],
  );

function setup(s: DevelopSettings, selected: NodeId = 1) {
  const props: NodeGraphProps = {
    settings: s,
    photoKey: 1,
    selected,
    selectedLink: null,
    renaming: null,
    thumbnailOf: () => undefined,
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
  };
  return { props, ...render(<NodeGraph {...props} />) };
}
const gnode = (id: NodeId): HTMLElement => document.querySelector<HTMLElement>(`[data-node-card="${id}"]`)!.closest<HTMLElement>(".ym-gnode")!;

describe("the geometry of a glyph card", () => {
  it("is 56 px wide and 64 px high with two ports, 24 more for each further one", () => {
    // A Layer with two inputs: two inputs and the "+" are three ports.
    expect(glyphGeometry(settings(), 4)).toMatchObject({ w: 56, h: 88 });
    const one = build({ "2": { kind: "parallel", pos: [1, 0] } }, [link("input", 1), link(1, "output")]);
    // No linked input: the "+" alone.
    expect(glyphGeometry(one, 2)).toMatchObject({ h: 64, inputs: [{ port: "picture.1", plus: true }] });
    const eight = build({ "2": { kind: "layer", pos: [1, 0] } }, [link("input", 1), ...Array.from({ length: 8 }, (_, i) => link(1, 2, `picture.${i + 1}` as Port)), link(2, "output")]);
    // Eight inputs and no "+": 208 px.
    const g = glyphGeometry(eight, 2);
    expect(g?.h).toBe(208);
    expect(g?.inputs).toHaveLength(8);
    expect(g?.inputs.some((s) => s.plus)).toBe(false);
    // Seven inputs and the "+": eight ports.
    const seven = build({ "2": { kind: "layer", pos: [1, 0] } }, [link("input", 1), ...Array.from({ length: 7 }, (_, i) => link(1, 2, `picture.${i + 1}` as Port)), link(2, "output")]);
    expect(glyphGeometry(seven, 2)?.h).toBe(208);
    expect(glyphGeometry(seven, 2)?.inputs.at(-1)).toMatchObject({ port: "picture.8", plus: true });
  });

  it("stacks the ports 24 px apart and centres them, with the output at mid height", () => {
    const g = glyphGeometry(settings(), 4)!;
    expect(g.inputs.map((s) => s.y)).toEqual([20, 44, 68]);
    expect(g.outputs.map((s) => s.y)).toEqual([44]);
    const t = glyphGeometry(splitterGraph(), 2)!;
    expect(t).toMatchObject({ h: 88, inputs: [{ port: "picture", y: 44 }] });
    expect(t.outputs.map((s) => [s.port, s.y])).toEqual([["red", 20], ["green", 44], ["blue", 68]]);
    const c = glyphGeometry(splitterGraph(), 3)!;
    expect(c.inputs.map((s) => s.port)).toEqual(["red", "green", "blue"]);
    expect(c.outputs.map((s) => s.port)).toEqual(["picture"]);
  });

  it("puts the card's middle on the line of a corrector's picture ports, and the Output marker on that line", () => {
    const layout = layoutGraph(settings());
    const card = layout.boxes.get(4)!;
    expect(card.y + card.h / 2).toBe(PICTURE_PORT_Y);
    expect(card.x).toBe(2 * 214);
    const output = layout.boxes.get("output")!;
    expect(output.y + output.h / 2).toBe(PICTURE_PORT_Y);
    // The ports of a link end where the glyph's ports are.
    expect(portPoint(layout, { node: 4, port: "picture.1" }, "in")).toEqual({ x: card.x - 9, y: card.y + 20 });
    expect(portPoint(layout, { node: 4, port: "picture" }, "out")).toEqual({ x: card.x + 56 + 4, y: card.y + 44 });
  });
});

describe("the card", () => {
  it("is named by its kind's words, with no thumbnail, no number chip and no footer", () => {
    setup(settings());
    const card = gnode(4);
    expect(card).toHaveAttribute("data-glyph", "layer");
    expect(card.querySelector(".ym-gnode__num, .ym-gnode__thumb, .ym-gnode__foot")).toBeNull();
    expect(card.querySelector("[data-node-name]")).toHaveTextContent("Layer");
    expect(card.querySelector('[role="button"]')).toHaveAccessibleName("Layer, mixer with 2 inputs, from node 2 and node 3");
  });

  it("draws one port for each input and the '+' under the last, and no key ports", () => {
    setup(settings());
    const ports = [...gnode(4).querySelectorAll<HTMLElement>(".ym-port")].map((p) => `${p.dataset.side}:${p.dataset.port}${p.dataset.plus ? "+" : ""}`);
    expect(ports).toEqual(["in:picture.1", "in:picture.2", "in:picture.3+", "out:picture"]);
    expect(gnode(4).querySelector('[data-port="key"]')).toBeNull();
  });

  it("tints the channel ports and writes the letters R, G and B inside a Splitter and a Combiner", () => {
    setup(splitterGraph());
    const splitter = gnode(2);
    expect([...splitter.querySelectorAll(".ym-gnode__letter")].map((l) => l.textContent)).toEqual(["R", "G", "B"]);
    expect([...splitter.querySelectorAll<HTMLElement>(".ym-port--ch")].map((p) => p.dataset.port)).toEqual(["red", "green", "blue"]);
    expect(splitter.querySelector<HTMLElement>('[data-port="red"]')?.style.getPropertyValue("--ch")).toBe("#ff7a6e");
    expect([...gnode(3).querySelectorAll(".ym-gnode__letter")].map((l) => l.textContent)).toEqual(["R", "G", "B"]);
    expect(gnode(3).querySelector('[role="button"]')).toHaveAccessibleName("Combiner, red from Splitter, red, green from the input, blue from Splitter, blue");
  });

  it("shows the selected ring, the bypass slash, and the dead state", () => {
    const { rerender, props } = setup(splitterGraph(true), 4);
    expect(gnode(4)).toHaveAttribute("data-selected");
    expect(gnode(4)).toHaveAttribute("data-bypass");
    expect(gnode(4).querySelector(".ym-gnode__slash")).not.toBeNull();
    // A Splitter and a Combiner have no bypass state.
    expect(gnode(2)).not.toHaveAttribute("data-bypass");
    // Dead: the Output is fed from Main, so the mixer is not needed.
    const dead = build({ "2": { kind: "parallel", pos: [1, 0] } }, [link("input", 1), link(1, 2, "picture.1"), link(1, "output")]);
    rerender(<NodeGraph {...props} settings={dead} />);
    expect(gnode(2)).toHaveAttribute("data-dead");
  });
});
