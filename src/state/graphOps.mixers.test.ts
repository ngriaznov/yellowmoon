/**
 * The operations of `docs/25` §5.3 to §5.5 without a screen: adding a Parallel or a Layer with its corrector, adding the Splitter and
 * Combiner trio, Delete, the refusals, links by hand to the numbered inputs and the "+", renumbering, the order of a Layer's inputs, the plan's
 * numbers that skip the four kinds, the base of a Parallel, and the words of the heads and card names. Each answer is checked against the
 * core's rules as the mock plays them (`validateGraph`).
 */
import { describe, expect, it } from "vitest";
import type { DevelopSettings, NodeId, NodeLink, NodePatch, NodeRef, Port, SettingsPatch } from "../api/types";
import { validateGraph } from "../fixtures/graph";
import {
  addsText,
  baseOf,
  cardName,
  cardStates,
  connect,
  disconnect,
  endName,
  GRAPH_OP_TEXT,
  headLine,
  inputStates,
  inputsOf,
  insertMixer,
  insertTrio,
  linkTargets,
  moveCard,
  moveEarlier,
  moveInput,
  moveLater,
  reconnect,
  remove,
  removeInput,
  rename,
  reorderInto,
  resetNode,
  sortedLinks,
  toggleBypass,
} from "./graphOps";
import type { GraphOutcome } from "./graphOps";
import { applyPatch, defaultSettings, nodeOf } from "./settingsGraph";

const out = (node: NodeId, port: Port = "picture"): NodeRef => ({ node, port });
const link = (from: NodeId, to: NodeId, toPort: Port = "picture", fromPort: Port = "picture"): NodeLink => ({ from: out(from, fromPort), to: out(to, toPort) });

type Spec = { kind?: string; name?: string; pos: [number, number]; bypass?: boolean; opacity?: number; mode?: string };

/** A graph of these nodes (Main is 1) and links. */
function graph(nodes: Record<number, Spec>, links: NodeLink[]): DevelopSettings {
  const patch: Record<string, NodePatch> = {};
  for (const [id, n] of Object.entries(nodes)) {
    patch[id] = { ...(Number(id) === 1 ? {} : { kind: n.kind ?? "corrector" }), pos: n.pos, ...(n.name ? { name: n.name } : {}), ...(n.bypass ? { bypass: true } : {}), ...(n.opacity !== undefined ? { opacity: n.opacity } : {}), ...(n.mode ? { mode: n.mode } : {}) } as NodePatch;
  }
  return applyPatch(defaultSettings(), { nodes: patch, links: sortedLinks(links) });
}

/** Main → Sky → Warmth in a line, as the mockups' first graph. */
const line = (): DevelopSettings =>
  graph({ 1: { pos: [0, 0] }, 2: { name: "Sky", pos: [1, 0] }, 3: { name: "Warmth", pos: [2, 0] } }, [link("input", 1), link(1, 2), link(2, 3), link(3, "output")]);

/** Main feeds Skin and Sky, which feed a Parallel (4), which feeds the Output. */
const parallel = (): DevelopSettings =>
  graph(
    { 1: { pos: [0, 0] }, 2: { name: "Skin", pos: [1, -0.5] }, 3: { name: "Sky", pos: [1, 0.5] }, 4: { kind: "parallel", pos: [2, 0] } },
    [link("input", 1), link(1, 2), link(1, 3), link(2, 4, "picture.1"), link(3, 4, "picture.2"), link(4, "output")],
  );

/** Main feeds Warmth (2), Glow (3) and Sky (4), which are the three inputs of a Layer (5) that feeds the Output. */
const layer = (): DevelopSettings =>
  graph(
    { 1: { pos: [0, 0] }, 2: { name: "Warmth", pos: [1, -1] }, 3: { name: "Glow", pos: [1, 0] }, 4: { name: "Sky", pos: [1, 1] }, 5: { kind: "layer", pos: [2, 0] } },
    [link("input", 1), link(1, 2), link(1, 3), link(1, 4), link(2, 5, "picture.1"), link(3, 5, "picture.2"), link(4, 5, "picture.3"), link(5, "output")],
  );

function sent(outcome: GraphOutcome): SettingsPatch {
  if (!outcome.ok) throw new Error(`refused: ${outcome.reason}`);
  return outcome.patch;
}
const reasonOf = (outcome: GraphOutcome): string | null => (outcome.ok ? null : outcome.reason);
const selectedOf = (outcome: GraphOutcome): NodeId | undefined => (outcome.ok ? outcome.select : undefined);

/** The document an outcome makes; it must be valid in the core's rules. */
function made(settings: DevelopSettings, outcome: GraphOutcome): DevelopSettings {
  const next = applyPatch(settings, sent(outcome));
  expect(validateGraph(next)).toBeNull();
  return next;
}
const has = (s: DevelopSettings, l: NodeLink): boolean => s.nodes.links.some((x) => x.from.node === l.from.node && x.from.port === l.from.port && x.to.node === l.to.node && x.to.port === l.to.port);

describe("Add node > Parallel and Layer", () => {
  it("makes the mixer and a corrector beside the selected node: S on input 1, the new corrector on input 2, the mixer feeds what S fed", () => {
    const s = line();
    const outcome = insertMixer(s, 2, "layer");
    const next = made(s, outcome);
    // 4 is the Layer, 5 the corrector: Main → Sky → Layer.1, Main → 5 → Layer.2, Layer → Warmth.
    expect(nodeOf(next, 4)?.kind).toBe("layer");
    expect(nodeOf(next, 5)?.kind).toBe("corrector");
    expect(has(next, link(2, 4, "picture.1"))).toBe(true);
    expect(has(next, link(1, 5))).toBe(true);
    expect(has(next, link(5, 4, "picture.2"))).toBe(true);
    expect(has(next, link(4, 3))).toBe(true);
    expect(has(next, link(2, 3))).toBe(false);
    expect(selectedOf(outcome)).toBe(5);
    expect(nodeOf(next, 4)).toMatchObject({ name: "Layer", mode: "normal", opacity: 1 });
    expect(nodeOf(next, 5)).toMatchObject({ name: "Corrector" });
  });

  it("places the corrector one row under S, the mixer one unit right of S and the nodes after S one unit right", () => {
    const s = line();
    const next = made(s, insertMixer(s, 2, "parallel"));
    expect(nodeOf(next, 5)?.pos).toEqual([1, 1]);
    expect(nodeOf(next, 4)?.pos).toEqual([2, 0]);
    expect(nodeOf(next, 3)?.pos).toEqual([3, 0]);
  });

  it("is one patch that adds both nodes and the links, and the patch undoes in one step", () => {
    const s = line();
    const patch = sent(insertMixer(s, 2, "parallel"));
    // 4 and 5 are new (with a kind); 3 only moves right.
    expect(Object.keys(patch.nodes ?? {}).sort()).toEqual(["3", "4", "5"]);
    expect(Object.values(patch.nodes ?? {}).filter((n) => n.kind !== undefined)).toHaveLength(2);
    expect(patch.links).toBeDefined();
  });

  it("with the Input selected takes Main as S, and with the Output selected the node that feeds it", () => {
    const s = line();
    expect(made(s, insertMixer(s, "input", "parallel")).nodes.links.some((l) => l.to.node === 4 && l.to.port === "picture.1" && l.from.node === 1)).toBe(true);
    const next = made(s, insertMixer(s, "output", "layer"));
    expect(has(next, link(3, 4, "picture.1"))).toBe(true);
    expect(has(next, link(4, "output"))).toBe(true);
  });

  it("moves the nodes in the way of the new corrector down one row", () => {
    const s = graph({ 1: { pos: [0, 0] }, 2: { name: "Sky", pos: [1, 0] }, 3: { name: "Below", pos: [1, 1] }, 4: { name: "Last", pos: [2, 0] } }, [
      link("input", 1),
      link(1, 2),
      link(2, 4),
      link(1, 3),
      link(4, "output"),
    ]);
    const next = made(s, insertMixer(s, 2, "parallel"));
    expect(nodeOf(next, 3)?.pos).toEqual([1, 2]);
  });

  it("can follow a Splitter output and a mixer, taking the link S fed through that output", () => {
    const s = graph(
      { 1: { pos: [0, 0] }, 2: { kind: "splitter", pos: [1, 0] } },
      [link("input", 1), link(1, 2), link(2, "output", "picture", "red")],
    );
    const next = made(s, insertMixer(s, 2, "layer"));
    expect(has(next, link(2, 3, "picture.1", "red"))).toBe(true);
    expect(has(next, link(3, "output"))).toBe(true);
    // The corrector reads what the Splitter reads.
    expect(has(next, link(1, 4))).toBe(true);
  });

  it("is refused from 63 nodes and allowed at 62", () => {
    const many = (count: number): DevelopSettings => {
      const nodes: Record<number, Spec> = { 1: { pos: [0, 0] } };
      for (let i = 2; i <= count; i++) nodes[i] = { pos: [i, 0] };
      const links = [link("input", 1), ...Array.from({ length: count - 1 }, (_, i) => link(i + 1, i + 2)), link(count, "output")];
      return graph(nodes, links);
    };
    expect(reasonOf(insertMixer(many(63), 63, "parallel"))).toBe(GRAPH_OP_TEXT.tooMany);
    expect(insertMixer(many(62), 62, "parallel").ok).toBe(true);
    expect(addsText(2)).toBe("This adds two nodes, and a photo can have up to 64.");
  });
});

describe("Add node > Splitter and combiner", () => {
  it("adds a Splitter, Red, Green and Blue correctors and a Combiner, linked, with the Splitter selected", () => {
    const s = line();
    const outcome = insertTrio(s, 2);
    const next = made(s, outcome);
    // 4 Splitter, 5 Red, 6 Green, 7 Blue, 8 Combiner.
    expect(nodeOf(next, 4)?.kind).toBe("splitter");
    expect(nodeOf(next, 8)?.kind).toBe("combiner");
    expect(["Red", "Green", "Blue"].map((n, i) => nodeOf(next, 5 + i)?.kind === "corrector" && (nodeOf(next, 5 + i) as { name: string }).name === n)).toEqual([true, true, true]);
    expect(has(next, link(2, 4))).toBe(true);
    for (const [i, port] of (["red", "green", "blue"] as const).entries()) {
      expect(has(next, link(4, 5 + i, "picture", port))).toBe(true);
      expect(has(next, link(5 + i, 8, port))).toBe(true);
    }
    expect(has(next, link(8, 3))).toBe(true);
    expect(selectedOf(outcome)).toBe(4);
  });

  it("places the Splitter one unit right of S, the correctors two on the rows y - 1, y and y + 1, the Combiner three, and moves what follows three right", () => {
    const s = line();
    const next = made(s, insertTrio(s, 2));
    expect(nodeOf(next, 4)?.pos).toEqual([2, 0]);
    expect([5, 6, 7].map((id) => nodeOf(next, id)?.pos)).toEqual([[3, -1], [3, 0], [3, 1]]);
    expect(nodeOf(next, 8)?.pos).toEqual([4, 0]);
    expect(nodeOf(next, 3)?.pos).toEqual([5, 0]);
  });

  it("is one patch of five nodes, and is off from 60 nodes", () => {
    const s = line();
    expect(Object.values(sent(insertTrio(s, 2)).nodes ?? {}).filter((n) => n.kind !== undefined)).toHaveLength(5);
    const nodes: Record<number, Spec> = { 1: { pos: [0, 0] } };
    for (let i = 2; i <= 60; i++) nodes[i] = { pos: [i, 0] };
    const big = graph(nodes, [link("input", 1), ...Array.from({ length: 59 }, (_, i) => link(i + 1, i + 2)), link(60, "output")]);
    expect(reasonOf(insertTrio(big, 60))).toBe(GRAPH_OP_TEXT.tooMany);
    expect(addsText(5)).toBe("This adds five nodes, and a photo can have up to 64.");
  });
});

describe("Delete", () => {
  it("links what a deleted Layer fed from its first input's source, and selects that source", () => {
    const s = layer();
    const outcome = remove(s, 5);
    const next = made(s, outcome);
    expect(has(next, link(2, "output"))).toBe(true);
    expect(selectedOf(outcome)).toBe(2);
  });

  it("gives a deleted Parallel's outputs to its first branch", () => {
    const s = parallel();
    const next = made(s, remove(s, 4));
    expect(has(next, link(2, "output"))).toBe(true);
  });

  it("takes a deleted Splitter's three outputs to its source, and a deleted Combiner's output to the source of its red", () => {
    const s = graph(
      { 1: { pos: [0, 0] }, 2: { kind: "splitter", pos: [1, 0] }, 3: { name: "R", pos: [2, 0] }, 4: { kind: "combiner", pos: [3, 0] } },
      [link("input", 1), link(1, 2), link(2, 3, "picture", "red"), link(3, 4, "red"), link(4, "output")],
    );
    const next = made(s, remove(s, 2));
    expect(has(next, link(1, 3))).toBe(true);
    const gone = made(s, remove(s, 4));
    expect(has(gone, link(3, "output"))).toBe(true);
  });

  it("numbers a mixer's inputs from 1 again when an input's corrector goes", () => {
    const s = layer();
    // Deleting Glow (3) links Main → Layer.2 in its place: no gap, three inputs.
    const next = made(s, remove(s, 3));
    expect(inputsOf(next, 5).map((r) => r.port)).toEqual(["picture.1", "picture.2", "picture.3"]);
    expect(has(next, link(1, 5, "picture.2"))).toBe(true);
  });
});

describe("refusals and the plain nodes' operations", () => {
  it("refuses to reorder the four kinds, with the spec's sentence", () => {
    const s = parallel();
    expect(reasonOf(moveEarlier(s, 4))).toBe(GRAPH_OP_TEXT.glyphFixed);
    expect(reasonOf(moveLater(s, 4))).toBe("This node can't be moved along a line. Change its links instead.");
    expect(reasonOf(reorderInto(s, 4, link(1, 2)))).toBe(GRAPH_OP_TEXT.glyphFixed);
  });

  it("toggles bypass on a Parallel and a Layer and refuses it on a Splitter and a Combiner", () => {
    const s = layer();
    expect(sent(toggleBypass(s, 5)).nodes).toEqual({ "5": { bypass: true } });
    const t = graph({ 1: { pos: [0, 0] }, 2: { kind: "splitter", pos: [1, 0] } }, [link("input", 1), link(1, 2), link(2, "output", "picture", "red")]);
    expect(reasonOf(toggleBypass(t, 2))).toBe("A splitter or a combiner can't be bypassed.");
  });

  it("renames all four kinds", () => {
    const s = layer();
    expect(sent(rename(s, 5, "  Stack ")).nodes).toEqual({ "5": { name: "Stack" } });
    const t = graph({ 1: { pos: [0, 0] }, 2: { kind: "combiner", pos: [1, 0] } }, [link("input", 1), link(1, 2, "red"), link(2, "output")]);
    expect(sent(rename(t, 2, "Join")).nodes).toEqual({ "2": { name: "Join" } });
  });

  it("resets a Layer's mode and opacity and has nothing to reset on the others", () => {
    const s = graph({ 1: { pos: [0, 0] }, 2: { kind: "layer", pos: [1, 0], mode: "multiply", opacity: 0.4 } }, [link("input", 1), link(1, 2, "picture.1"), link(2, "output")]);
    expect(sent(resetNode(s, 2)).nodes).toEqual({ "2": { mode: "normal", opacity: 1 } });
    expect(resetNode(parallel(), 4)).toEqual({ ok: true, patch: {} });
  });

  it("moves a glyph card by its pos", () => {
    expect(sent(moveCard(layer(), 5, [3, 0.5])).nodes).toEqual({ "5": { pos: [3, 0.5] } });
  });

  it("puts a corrector dropped on a link into a mixer input into that link, keeping the input's number", () => {
    const s = parallel();
    const next = made(s, reorderInto(s, 3, link(2, 4, "picture.1")));
    expect(has(next, link(2, 3))).toBe(true);
    expect(has(next, link(3, 4, "picture.1"))).toBe(true);
  });

  it("takes a node before a mixer into the link that feeds the mixer's first input (Move earlier)", () => {
    const s = graph(
      { 1: { pos: [0, 0] }, 2: { name: "A", pos: [1, 0] }, 3: { kind: "layer", pos: [2, 0] }, 4: { name: "B", pos: [3, 0] } },
      [link("input", 1), link(1, 2), link(2, 3, "picture.1"), link(3, 4), link(4, "output")],
    );
    const next = made(s, moveEarlier(s, 4));
    expect(has(next, link(2, 4))).toBe(true);
    expect(has(next, link(4, 3, "picture.1"))).toBe(true);
    expect(has(next, link(3, "output"))).toBe(true);
  });
});

describe("links by hand", () => {
  it("lights the picture inputs, the numbered inputs, the '+' and a Combiner's inputs for a picture, and never a key port", () => {
    const s = graph(
      { 1: { pos: [0, 0] }, 2: { name: "A", pos: [1, 0] }, 3: { kind: "layer", pos: [2, 0] }, 4: { kind: "combiner", pos: [2, 1] } },
      [link("input", 1), link(1, 2), link(2, 3, "picture.1"), link(3, "output")],
    );
    const lit = new Map(inputStates(s, out(1)).map((i) => [`${i.to.node}:${i.to.port}`, i.ok]));
    // The Layer has one linked input and the '+' (picture.2).
    expect(lit.get("3:picture.1")).toBe(true);
    expect(lit.get("3:picture.2")).toBe(true);
    expect(lit.has("3:picture.3")).toBe(false);
    expect(lit.get("4:red")).toBe(true);
    expect(lit.get("4:blue")).toBe(true);
    // A key port of a corrector does not take a picture, and the four kinds have none.
    expect(lit.get("2:key")).toBe(false);
    expect([...lit.keys()].filter((k) => k.startsWith("3:") || k.startsWith("4:")).some((k) => k.endsWith(":key"))).toBe(false);
    // A key output lights only key inputs, none on the four kinds.
    const keyLit = inputStates(s, out(2, "key")).filter((i) => i.ok);
    expect(keyLit.some((i) => i.to.node === 3 || i.to.node === 4)).toBe(false);
  });

  it("dims the inputs that would close a circle", () => {
    const s = parallel();
    const circle = inputStates(s, out(4)).find((i) => i.to.node === 2 && i.to.port === "picture");
    expect(circle).toMatchObject({ ok: false, reason: GRAPH_OP_TEXT.circle });
  });

  it("a drop on the '+' adds the next numbered input", () => {
    const s = parallel();
    const withThird = made(s, connect(s, out(1), out(4, "picture.3")));
    expect(inputsOf(withThird, 4).map((r) => r.port)).toEqual(["picture.1", "picture.2", "picture.3"]);
    expect(has(withThird, link(1, 4, "picture.3"))).toBe(true);
  });

  it("refuses a ninth input", () => {
    const s = graph({ 1: { pos: [0, 0] }, 2: { kind: "parallel", pos: [1, 0] } }, [link("input", 1), ...Array.from({ length: 8 }, (_, i) => link(1, 2, `picture.${i + 1}` as Port)), link(2, "output")]);
    expect(reasonOf(connect(s, out(1), out(2, "picture.9" as Port)))).not.toBeNull();
    expect(linkTargets(s, out(1)).some((t) => t.node === 2 && t.port === ("picture.9" as Port))).toBe(false);
  });

  it("numbers the inputs 1 to N again, in their old order, when a link is removed", () => {
    const s = layer();
    const next = made(s, disconnect(s, link(3, 5, "picture.2")));
    expect(has(next, link(2, 5, "picture.1"))).toBe(true);
    expect(has(next, link(4, 5, "picture.2"))).toBe(true);
    expect(inputsOf(next, 5)).toHaveLength(2);
  });

  it("makes the new link first and then removes the lifted one: input 1 dropped on the '+' of its own Layer goes to the top", () => {
    const s = layer();
    const next = made(s, reconnect(s, link(2, 5, "picture.1"), out(5, "picture.4")));
    // Warmth was the bottom; now Glow is 1, Sky is 2 and Warmth is 3 (on top).
    expect(has(next, link(3, 5, "picture.1"))).toBe(true);
    expect(has(next, link(4, 5, "picture.2"))).toBe(true);
    expect(has(next, link(2, 5, "picture.3"))).toBe(true);
  });

  it("a drop on an input that has a link replaces it", () => {
    const s = parallel();
    const next = made(s, connect(s, out(1), out(4, "picture.2")));
    expect(has(next, link(1, 4, "picture.2"))).toBe(true);
    expect(has(next, link(3, 4, "picture.2"))).toBe(false);
  });

  it("links a Splitter's channel to any picture input and a corrector's picture to a Combiner's channel", () => {
    const s = graph(
      { 1: { pos: [0, 0] }, 2: { kind: "splitter", pos: [1, 0] }, 3: { kind: "combiner", pos: [2, 0] } },
      [link("input", 1), link(1, 2), link(3, "output")],
    );
    expect(made(s, connect(s, out(2, "red"), out(3, "green"))).nodes.links.some((l) => l.from.port === "red" && l.to.port === "green")).toBe(true);
    expect(made(s, connect(s, out(1), out(3, "red"))).nodes.links.some((l) => l.to.node === 3 && l.to.port === "red")).toBe(true);
    // A key does not join a picture.
    expect(reasonOf(connect(s, out(1, "key"), out(3, "red")))).toBe(GRAPH_OP_TEXT.portTypes);
  });

  it("names the targets of the keyboard path: 'Layer, input 2', 'Layer, a new input', 'Combiner, red'", () => {
    const s = graph(
      { 1: { pos: [0, 0] }, 2: { name: "A", pos: [1, 0] }, 3: { kind: "layer", pos: [2, 0] }, 4: { kind: "combiner", pos: [2, 1] } },
      [link("input", 1), link(1, 2), link(2, 3, "picture.1"), link(3, "output")],
    );
    expect(endName(s, out(3, "picture.1"), "in")).toBe("Layer, input 1");
    expect(endName(s, out(3, "picture.2"), "in")).toBe("Layer, a new input");
    expect(endName(s, out(4, "red"), "in")).toBe("Combiner, red");
  });
});

describe("the inputs of a Layer", () => {
  it("moves an input to another place and renumbers the ports, nothing else", () => {
    const s = layer();
    // Put input 1 (Warmth) on top.
    const next = made(s, moveInput(s, 5, 1, 3));
    expect(has(next, link(3, 5, "picture.1"))).toBe(true);
    expect(has(next, link(4, 5, "picture.2"))).toBe(true);
    expect(has(next, link(2, 5, "picture.3"))).toBe(true);
    expect(Object.keys(sent(moveInput(s, 5, 1, 3)).nodes ?? {})).toEqual([]);
    expect(moveInput(s, 5, 2, 2)).toEqual({ ok: true, patch: {} });
  });

  it("removes an input and numbers the later ones down", () => {
    const s = layer();
    const next = made(s, removeInput(s, 5, "picture.1"));
    expect(has(next, link(3, 5, "picture.1"))).toBe(true);
    expect(has(next, link(4, 5, "picture.2"))).toBe(true);
  });

  it("lists a mixer with no linked input as one row that reads the Input", () => {
    const s = graph({ 1: { pos: [0, 0] }, 2: { kind: "parallel", pos: [1, 0] } }, [link("input", 1), link(1, "output")]);
    expect(inputsOf(s, 2)).toEqual([{ port: "picture.1", link: null }]);
  });
});

describe("the plan's numbers, liveness and the base", () => {
  it("numbers the correctors and skips the four kinds, and follows every input of a mixer for liveness", () => {
    const s = parallel();
    const states = cardStates(s);
    expect([1, 2, 3].map((id) => states.get(id)?.number)).toEqual([1, 2, 3]);
    expect(states.get(4)).toMatchObject({ number: 0, live: true });
    expect(states.get(2)?.live).toBe(true);
    expect(states.get(3)?.live).toBe(true);
  });

  it("marks a mixer and what only it needs dead when the Output is fed from elsewhere", () => {
    const s = graph(
      { 1: { pos: [0, 0] }, 2: { name: "A", pos: [1, 0] }, 3: { kind: "layer", pos: [2, 0] } },
      [link("input", 1), link(1, 2), link(2, 3, "picture.1"), link(1, "output")],
    );
    expect(cardStates(s).get(3)?.live).toBe(false);
    expect(cardStates(s).get(2)?.live).toBe(false);
    expect(headLine(s, 3)).toBe("Mixer · not connected");
  });

  it("finds the base: the node every branch leaves, else the input", () => {
    expect(baseOf(parallel(), 4)).toEqual(out(1));
    const s = graph(
      { 1: { pos: [0, 0] }, 2: { name: "A", pos: [1, 0] }, 3: { name: "C", pos: [1, 1] }, 4: { kind: "parallel", pos: [2, 0] } },
      [link("input", 1), link(1, 2), link("input", 3), link(2, 4, "picture.1"), link(3, 4, "picture.2"), link(4, "output")],
    );
    expect(baseOf(s, 4)).toEqual(out("input"));
  });

  it("takes the base of branches through one Splitter as the value that feeds the Splitter", () => {
    const s = graph(
      { 1: { pos: [0, 0] }, 2: { kind: "splitter", pos: [1, 0] }, 3: { kind: "parallel", pos: [2, 0] } },
      [link("input", 1), link(1, 2), link(2, 3, "picture.1", "red"), link(2, 3, "picture.2", "green"), link(3, "output")],
    );
    expect(baseOf(s, 3)).toEqual(out(1));
  });
});

describe("words", () => {
  it("writes the head lines", () => {
    const s = layer();
    expect(headLine(s, 5)).toBe("Mixer · 3 inputs");
    expect(headLine(parallel(), 4)).toBe("Mixer · 2 inputs");
    expect(headLine(applyPatch(s, { nodes: { "5": { bypass: true } } }), 5)).toBe("Mixer · bypassed");
    const t = graph({ 1: { pos: [0, 0] }, 2: { kind: "splitter", pos: [1, 0] }, 3: { kind: "combiner", pos: [2, 0] } }, [link("input", 1), link(1, 2), link(2, 3, "red", "red"), link(3, "output")]);
    expect(headLine(t, 2)).toBe("Channels · 3 outputs");
    expect(headLine(t, 3)).toBe("Channels · 3 inputs");
  });

  it("writes the card names", () => {
    const s = layer();
    expect(cardName(s, 5)).toBe("Layer, mixer with 3 inputs, from node 2, node 3 and node 4");
    expect(cardName(applyPatch(s, { nodes: { "5": { bypass: true } } }), 5)).toBe("Layer, mixer with 3 inputs, bypassed, from node 2, node 3 and node 4");
    const t = graph({ 1: { pos: [0, 0] }, 2: { kind: "splitter", pos: [1, 0] }, 3: { name: "R", pos: [2, 0] }, 4: { kind: "combiner", pos: [3, 0] } }, [
      link("input", 1),
      link(1, 2),
      link(2, 3, "picture", "red"),
      link(3, 4, "red"),
      link(4, "output"),
    ]);
    expect(cardName(t, 2)).toBe("Splitter, picture from node 1");
    expect(cardName(t, 4)).toBe("Combiner, red from node 2, green from the input, blue from the input");
    // A corrector fed by a Layer names it by its name.
    const u = graph({ 1: { pos: [0, 0] }, 2: { kind: "layer", pos: [1, 0] }, 3: { name: "After", pos: [2, 0] } }, [link("input", 1), link(1, 2, "picture.1"), link(2, 3), link(3, "output")]);
    expect(cardName(u, 3)).toBe("Node 2, After, picture from Layer");
  });
});
