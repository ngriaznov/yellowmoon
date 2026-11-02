/**
 * The graph's operations (`docs/23` §7.2) without a screen: every operation, its undo, the rules for several links and none, the refusals
 * with their sentences, the limit of 64, the plan's numbers with ties, dead nodes, the key links a delete takes away, and the lighting of
 * the inputs while a link is dragged. Each answer is also checked against the core's rules as the mock plays them (`validateGraph`).
 */
import { describe, expect, it } from "vitest";
import type { DevelopSettings, NodeId, NodeLink, NodePatch, NodeRef, Port, SettingsPatch } from "../api/types";
import { GRAPH_TEXT, MAX_NODES, validateGraph } from "../fixtures/graph";
import {
  addNode,
  cardName,
  cardNumbers,
  cardStates,
  chipText,
  cleanName,
  compareLinks,
  connect,
  disconnect,
  GRAPH_OP_TEXT,
  hasCircle,
  hasPatch,
  inputStates,
  insertAfter,
  joinWords,
  linksOf,
  linkTargets,
  MAX_CORRECTORS,
  moveCard,
  moveEarlier,
  moveLater,
  nextNodeId,
  reconnect,
  remove,
  rename,
  reorderAvailability,
  reorderInto,
  sortedLinks,
  toggleBypass,
} from "./graphOps";
import type { GraphOutcome } from "./graphOps";
import { applyPatch, defaultSettings, diffGraph, isEmptyPatch, nodeOf } from "./settingsGraph";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const pic = (from: NodeId, to: NodeId): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const key = (from: number, to: number): NodeLink => ({ from: { node: from, port: "key" }, to: { node: to, port: "key" } });
const ref = (node: NodeId, port: Port = "picture"): NodeRef => ({ node, port });

/** A graph made of correctors with these names and places (Main is 1, at the origin) and these links. */
function graph(nodes: Record<number, { name?: string; pos?: [number, number]; bypass?: boolean; masks?: string[] }>, links: NodeLink[]): DevelopSettings {
  const patch: Record<string, NodePatch> = {};
  for (const [id, n] of Object.entries(nodes)) {
    const base: NodePatch = { pos: n.pos ?? [Number(id) - 1, 0] };
    if (n.name) base.name = n.name;
    if (n.bypass) base.bypass = true;
    if (n.masks) base.masks = { invert: false, items: n.masks.map((kind) => ({ kind, mode: "add", invert: false, enabled: true, opacity: 1 }) as never) };
    patch[id] = Number(id) === 1 ? base : { kind: "corrector", ...base };
  }
  return applyPatch(defaultSettings(), { nodes: patch, links: sortedLinks(links) });
}

/** Main → 2 → 3 → 4 in a line: Mountain, Sky, Warmth (the mockup's four nodes). */
const line = (): DevelopSettings =>
  graph(
    { 1: {}, 2: { name: "Mountain" }, 3: { name: "Sky" }, 4: { name: "Warmth" } },
    [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, 4), pic(4, "output")],
  );

/** The patch an outcome sends; fails the test when the operation was refused. */
function sent(outcome: GraphOutcome): SettingsPatch {
  if (!outcome.ok) throw new Error(`refused: ${outcome.reason}`);
  return outcome.patch;
}

const reasonOf = (outcome: GraphOutcome): string | null => (outcome.ok ? null : outcome.reason);

/** The settings after an operation, checked against the core's rules and with its undo applied to the result. */
function run(before: DevelopSettings, outcome: GraphOutcome): { after: DevelopSettings; undone: DevelopSettings; select?: NodeId } {
  const patch = sent(outcome);
  const after = applyPatch(before, patch);
  expect(validateGraph(after), "the core would refuse the graph").toBeNull();
  const diff = diffGraph(before, after);
  const undone = diff ? applyPatch(after, diff.before) : after;
  return { after, undone, select: outcome.ok ? outcome.select : undefined };
}

/** The links as "from>to" strings, in the order given, for a short comparison. */
const arrows = (s: DevelopSettings): string[] =>
  s.nodes.links.map((l) => `${l.from.node}${l.from.port === "key" ? ".key" : ""}>${l.to.node}${l.to.port === "key" ? ".key" : ""}`);

const posOf = (s: DevelopSettings, id: NodeId): [number, number] => (nodeOf(s, id) as { pos: [number, number] }).pos;

// ---------------------------------------------------------------------------

describe("the sentences", () => {
  it("are the spec's, word for word", () => {
    expect(GRAPH_OP_TEXT.mainInput).toBe("Main always takes its picture from the input.");
    expect(GRAPH_OP_TEXT.outputNeedsPicture).toBe("The output needs a picture. Link another node to it instead.");
    expect(GRAPH_OP_TEXT.circle).toBe("That would link the nodes in a circle.");
    expect(GRAPH_OP_TEXT.mainDelete).toBe("Main can't be deleted. Bypass it instead.");
    expect(GRAPH_OP_TEXT.tooMany).toBe("A photo can have up to 64 nodes.");
    expect(GRAPH_OP_TEXT.noBypass).toBe("The input and the output can't be bypassed.");
  });

  it("repeat the core's where the core has one", () => {
    expect(GRAPH_OP_TEXT.tooMany).toBe(GRAPH_TEXT.tooManyNodes);
    expect(GRAPH_OP_TEXT.cannotRemove).toBe(GRAPH_TEXT.cannotRemove);
    expect(GRAPH_OP_TEXT.portTypes).toBe(GRAPH_TEXT.linkPortTypes);
    expect(MAX_CORRECTORS).toBe(MAX_NODES);
  });

  it("have no exclamation mark and end like a sentence", () => {
    for (const text of Object.values(GRAPH_OP_TEXT)) {
      expect(text).not.toContain("!");
      expect(text).toMatch(/[.]$/);
    }
  });
});

describe("the plan's numbers and live nodes", () => {
  it("number the default graph's Main 1 and keep it live", () => {
    const states = cardStates(defaultSettings());
    expect([...states]).toEqual([[1, { number: 1, live: true }]]);
  });

  it("follow the links: a line is 1, 2, 3, 4 whatever the places", () => {
    const flipped = graph({ 1: { pos: [0, 0] }, 2: { pos: [9, 0] }, 3: { pos: [5, 0] }, 4: { pos: [1, 0] } }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, 4), pic(4, "output")]);
    expect([...cardNumbers(flipped)]).toEqual([
      [1, 1],
      [2, 2],
      [3, 3],
      [4, 4],
    ]);
  });

  it("break ties by Main, then x, then y, then the id", () => {
    // Main fans out to 2, 3, 4 and 5: all ready together once Main has run.
    const fan = (places: Record<number, [number, number]>): DevelopSettings =>
      graph({ 1: {}, 2: { pos: places[2] }, 3: { pos: places[3] }, 4: { pos: places[4] }, 5: { pos: places[5] } }, [
        pic("input", 1),
        pic(1, 2),
        pic(1, 3),
        pic(1, 4),
        pic(1, 5),
        pic(2, "output"),
      ]);
    // By x first, then y.
    const byPlace = cardNumbers(fan({ 2: [3, 0], 3: [1, 1], 4: [1, 0], 5: [2, 0] }));
    expect([4, 3, 5, 2].map((id) => byPlace.get(id))).toEqual([2, 3, 4, 5]);
    // The same place: the id.
    const byId = cardNumbers(fan({ 2: [1, 0], 3: [1, 0], 4: [1, 0], 5: [1, 0] }));
    expect([2, 3, 4, 5].map((id) => byId.get(id))).toEqual([2, 3, 4, 5]);
    // Main is first even when another node is further left.
    const early = cardNumbers(fan({ 2: [-5, 0], 3: [1, 0], 4: [1, 0], 5: [1, 0] }));
    expect(early.get(1)).toBe(1);
    expect(early.get(2)).toBe(2);
  });

  it("number a node after the ones its key needs, even when it is further left", () => {
    // 3's key comes from 5, which reads Main: 5 runs before 3.
    const keyed = graph({ 1: {}, 2: { pos: [1, 0] }, 3: { pos: [2, 0] }, 5: { pos: [9, 1] } }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output"), pic(1, 5), key(5, 3)]);
    const n = cardNumbers(keyed);
    expect(n.get(5)).toBeLessThan(n.get(3) as number);
    expect(n.get(2)).toBe(2);
  });

  it("find a node dead when the Output does not need it, and live when a live node needs its key", () => {
    const dead = graph({ 1: {}, 2: {}, 3: {} }, [pic("input", 1), pic(1, 2), pic(1, "output"), pic(2, 3)]);
    expect(cardStates(dead).get(2)?.live).toBe(false);
    expect(cardStates(dead).get(3)?.live).toBe(false);
    expect(cardStates(dead).get(1)?.live).toBe(true);
    // Dead nodes keep their numbers.
    expect([...cardNumbers(dead)].map(([, n]) => n).sort()).toEqual([1, 2, 3]);
    // Node 2's key output feeds live node 1? Main takes no key from a later node (a circle); use node 3 live and 2 as its key source.
    const keyLive = graph({ 1: {}, 2: {}, 3: {} }, [pic("input", 1), pic(1, 3), pic(3, "output"), pic(1, 2), key(2, 3)]);
    expect(cardStates(keyLive).get(2)?.live).toBe(true);
  });

  it("read a picture input with no link as the Input, so such a node is live when it feeds the Output", () => {
    const loose = graph({ 1: {}, 2: {} }, [pic("input", 1), pic(2, "output")]);
    expect(cardStates(loose).get(2)?.live).toBe(true);
    expect(cardStates(loose).get(1)?.live).toBe(false);
  });

  it("show a number as two digits", () => {
    expect([1, 9, 10, 64].map(chipText)).toEqual(["01", "09", "10", "64"]);
  });
});

describe("the links' order", () => {
  it("is the core's: from-node (the Input, the correctors by id, the Output), from-port, to-node, to-port", () => {
    const shuffled = [key(2, 4), pic(3, "output"), pic(2, 3), pic("input", 1), pic(2, 4)];
    expect(arrows({ nodes: { items: [], links: sortedLinks(shuffled) } })).toEqual(["input>1", "2>3", "2>4", "2.key>4.key", "3>output"]);
    expect(compareLinks(pic(1, 2), pic(1, 2))).toBe(0);
  });
});

describe("Add after the selected node", () => {
  it("goes into the selected node's outgoing link: S → T becomes S → N → T, with N selected", () => {
    const before = line();
    const { after, undone, select } = run(before, insertAfter(before, 2));
    expect(select).toBe(5);
    // The links come back in the core's order, whatever order they went in.
    expect(arrows(after)).toEqual(["input>1", "1>2", "2>5", "3>4", "4>output", "5>3"]);
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "2>5", "5>3", "3>4", "4>output"]));
    expect(nodeOf(after, 5)).toMatchObject({ kind: "corrector", name: "Corrector", bypass: false });
    expect(undone).toEqual(before);
  });

  it("puts N one unit to the right of S, and moves the nodes downstream of N one unit right", () => {
    const before = line();
    const { after } = run(before, insertAfter(before, 2));
    expect(posOf(after, 5)).toEqual([2, 0]);
    expect([1, 2, 3, 4].map((id) => posOf(after, id))).toEqual([
      [0, 0],
      [1, 0],
      [3, 0],
      [4, 0],
    ]);
  });

  it("is one node patch: the new node whole, the moved places, the links", () => {
    const patch = sent(insertAfter(line(), 2));
    expect(Object.keys(patch.nodes ?? {}).sort()).toEqual(["3", "4", "5"]);
    expect(patch.nodes?.["5"]).toMatchObject({ kind: "corrector", name: "Corrector", pos: [2, 0], bypass: false });
    expect(patch.nodes?.["3"]).toEqual({ pos: [3, 0] });
    expect(patch.remove).toBeUndefined();
    expect(patch.links).toHaveLength(6);
  });

  it("makes a node that is manual 0 everywhere, because Auto belongs to Main", () => {
    const before = defaultSettings();
    const { after } = run(before, insertAfter(before, 1));
    const made = nodeOf(after, 2);
    expect(made?.kind === "corrector" && made.grade.contrast).toEqual({ kind: "manual", value: 0 });
    expect(made?.kind === "corrector" && made.grade.color_balance.shadows).toEqual({ kind: "manual", point: [0, 0] });
    expect(made?.kind === "corrector" && made.masks.items).toEqual([]);
  });

  it("with the Input selected goes after Main", () => {
    const before = line();
    const { after, undone, select } = run(before, insertAfter(before, "input"));
    expect(select).toBe(5);
    expect(arrows(after)).toContain("1>5");
    expect(arrows(after)).toContain("5>2");
    expect(arrows(after)).toContain("input>1");
    expect(posOf(after, 5)).toEqual([1, 0]);
    expect(undone).toEqual(before);
  });

  it("with the Output selected goes into the Output's incoming link", () => {
    const before = line();
    const { after, undone } = run(before, insertAfter(before, "output"));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "2>3", "3>4", "4>5", "5>output"]));
    expect(posOf(after, 5)).toEqual([4, 0]);
    expect(undone).toEqual(before);
  });

  it("with the default graph's Main selected makes Main → N → Output", () => {
    const before = defaultSettings();
    const { after } = run(before, insertAfter(before, 1));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "2>output"]));
    expect(posOf(after, 2)).toEqual([1, 0]);
  });

  it("with several outgoing picture links goes into the one that leads to the Output", () => {
    // Main feeds 2 (a dead end) and 3 (which leads to the Output): N goes into Main → 3.
    const before = graph({ 1: {}, 2: { pos: [1, 1] }, 3: { pos: [1, 0] } }, [pic("input", 1), pic(1, 2), pic(1, 3), pic(3, "output")]);
    const { after } = run(before, insertAfter(before, 1));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "1>4", "4>3", "3>output"]));
  });

  it("with several that all lead to the Output, or none does, goes into the first", () => {
    const none = graph({ 1: {}, 2: { pos: [1, 1] }, 3: { pos: [1, 2] } }, [pic("input", 1), pic(1, 2), pic(1, 3), pic(1, "output")]);
    // Main → Output is the one that leads to the Output.
    expect(new Set(arrows(run(none, insertAfter(none, 1)).after))).toEqual(new Set(["input>1", "1>2", "1>3", "1>4", "4>output"]));
    const first = graph({ 1: {}, 2: { pos: [1, 1] }, 3: { pos: [1, 2] } }, [pic("input", 1), pic(1, 2), pic(1, 3), pic(2, "output")]);
    // 1 → 2 leads to the Output, 1 → 3 does not.
    expect(new Set(arrows(run(first, insertAfter(first, 1)).after))).toEqual(new Set(["input>1", "1>4", "4>2", "1>3", "2>output"]));
    const dead = graph({ 1: {}, 2: { pos: [1, 1] }, 3: { pos: [1, 2] }, 4: { pos: [5, 5] } }, [pic("input", 1), pic(1, 2), pic(1, 3), pic(4, "output")]);
    // Neither 2 nor 3 leads to the Output: the first, 1 → 2.
    expect(new Set(arrows(run(dead, insertAfter(dead, 1)).after))).toEqual(new Set(["input>1", "1>5", "5>2", "1>3", "4>output"]));
  });

  it("with no outgoing picture link links N after S alone, and moves nothing", () => {
    const before = graph({ 1: {}, 2: { pos: [1, 1] } }, [pic("input", 1), pic(1, "output"), pic(1, 2)]);
    // Node 2 has no outgoing link.
    const { after, undone } = run(before, insertAfter(before, 2));
    expect(arrows(after)).toContain("2>3");
    expect(posOf(after, 3)).toEqual([2, 1]);
    expect(posOf(after, 1)).toEqual([0, 0]);
    expect(undone).toEqual(before);
    // The new node is dead like the one it hangs from.
    expect(cardStates(after).get(3)?.live).toBe(false);
  });

  it("keeps the key links around: a node's key still feeds the node it fed", () => {
    const before = graph({ 1: {}, 2: {}, 3: {} }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output"), key(2, 3)]);
    const { after } = run(before, insertAfter(before, 2));
    expect(arrows(after)).toContain("2.key>3.key");
  });

  it("gives an id one more than the largest used, so a deleted id is not reused", () => {
    const before = line();
    expect(nextNodeId(before)).toBe(5);
    expect(nextNodeId(before, 9)).toBe(10);
    const { select } = run(before, insertAfter(before, 4, { usedMax: 7 }));
    expect(select).toBe(8);
  });

  it("is refused at 64 nodes, with the sentence, and allowed at 63", () => {
    const ids = Array.from({ length: 64 }, (_, i) => i + 1);
    const chain = (n: number): DevelopSettings => {
      const nodes = Object.fromEntries(ids.slice(0, n).map((id) => [id, {}]));
      const links = [pic("input", 1), ...ids.slice(1, n).map((id) => pic(id - 1, id)), pic(n, "output")];
      return graph(nodes, links);
    };
    expect(reasonOf(insertAfter(chain(64), 64))).toBe("A photo can have up to 64 nodes.");
    expect(reasonOf(insertAfter(chain(64), "output"))).toBe(GRAPH_OP_TEXT.tooMany);
    const sixtyThree = chain(63);
    const { after } = run(sixtyThree, insertAfter(sixtyThree, 63));
    expect(after.nodes.items.filter((i) => i.kind === "corrector")).toHaveLength(64);
  });

  it("is refused for a node the graph does not have", () => {
    expect(reasonOf(insertAfter(line(), 99))).toBe(GRAPH_OP_TEXT.gone);
  });

  it("builds a new corrector whole with addNode", () => {
    expect(addNode(7, [3, 1], "Glow")["7"]).toMatchObject({ kind: "corrector", name: "Glow", pos: [3, 1], bypass: false, trim: { exposure_ev: 0, temperature: 0, tint: 0 } });
  });
});

describe("Delete", () => {
  it("reconnects the neighbours: U → D → T becomes U → T, and the previous node is selected", () => {
    const before = line();
    const { after, undone, select } = run(before, remove(before, 3));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "2>4", "4>output"]));
    expect(nodeOf(after, 3)).toBeUndefined();
    expect(select).toBe(2);
    expect(undone).toEqual(before);
  });

  it("is one patch: the node in remove and the links whole", () => {
    const patch = sent(remove(line(), 3));
    expect(patch.remove).toEqual([3]);
    expect(patch.nodes).toBeUndefined();
    expect(patch.links).toHaveLength(4);
  });

  it("deletes the last node before the Output: the one before it feeds the Output", () => {
    const before = line();
    const { after, select } = run(before, remove(before, 4));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "2>3", "3>output"]));
    expect(select).toBe(3);
  });

  it("reconnects every node a branching node fed: U → each of T1 … Tn", () => {
    const before = graph({ 1: {}, 2: {}, 3: { pos: [2, 0] }, 4: { pos: [2, 1] } }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(2, 4), pic(3, "output")]);
    const { after, undone } = run(before, remove(before, 2));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>3", "1>4", "3>output"]));
    expect(undone).toEqual(before);
  });

  it("removes the node's key links, both ways: the key it made and the key it took", () => {
    const before = graph({ 1: {}, 2: {}, 3: {}, 5: { pos: [1, 1] } }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output"), pic(1, 5), key(5, 3), key(2, 5)]);
    // Delete 5: it gave 3 its key input and took 2's key; both links go and 3 loses its key input.
    const { after, undone } = run(before, remove(before, 5));
    expect(arrows(after).some((a) => a.includes("key"))).toBe(false);
    expect(undone).toEqual(before);
    // Delete 2: 5 loses the key it took from 2.
    const second = run(before, remove(before, 2));
    expect(arrows(second.after)).not.toContain("2.key>5.key");
    expect(arrows(second.after)).toContain("5.key>3.key");
  });

  it("reads a node with no picture link as one that took the Input", () => {
    const before = graph({ 1: {}, 2: {} }, [pic("input", 1), pic(2, "output")]);
    const { after, select } = run(before, remove(before, 2));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "input>output"]));
    expect(select).toBe(1);
  });

  it("is refused for Main with its sentence, and for the Input and the Output", () => {
    expect(reasonOf(remove(line(), 1))).toBe("Main can't be deleted. Bypass it instead.");
    expect(reasonOf(remove(line(), "input"))).toBe("The input and the output can't be deleted.");
    expect(reasonOf(remove(line(), "output"))).toBe(GRAPH_OP_TEXT.markerDelete);
    expect(reasonOf(remove(line(), 42))).toBe(GRAPH_OP_TEXT.gone);
  });

  it("makes a number free again: the ones after it count one less", () => {
    const before = line();
    const after = run(before, remove(before, 2)).after;
    expect([2, 3, 4].map((id) => cardNumbers(after).get(id))).toEqual([undefined, 2, 3]);
  });
});

describe("Bypass", () => {
  it("toggles a corrector's bypass, Main included, and undoes", () => {
    const before = line();
    for (const id of [1, 3]) {
      const { after, undone } = run(before, toggleBypass(before, id));
      expect(nodeOf(after, id)).toMatchObject({ bypass: true });
      expect(undone).toEqual(before);
      const again = run(after, toggleBypass(after, id));
      expect(nodeOf(again.after, id)).toMatchObject({ bypass: false });
    }
  });

  it("is one node's field and nothing else", () => {
    expect(sent(toggleBypass(line(), 3))).toEqual({ nodes: { "3": { bypass: true } } });
  });

  it("is refused for the Input and the Output with its sentence", () => {
    expect(reasonOf(toggleBypass(line(), "input"))).toBe("The input and the output can't be bypassed.");
    expect(reasonOf(toggleBypass(line(), "output"))).toBe("The input and the output can't be bypassed.");
  });
});

describe("Rename", () => {
  it("sets the name, trimmed, and undoes", () => {
    const before = line();
    const { after, undone } = run(before, rename(before, 2, "  Hills  "));
    expect(nodeOf(after, 2)).toMatchObject({ name: "Hills" });
    expect(undone).toEqual(before);
  });

  it("cuts a name longer than 32 characters, and keeps 32", () => {
    const long = "x".repeat(40);
    expect(cleanName(long)).toHaveLength(32);
    expect(sent(rename(line(), 2, long)).nodes?.["2"]).toEqual({ name: "x".repeat(32) });
    expect(cleanName("é".repeat(33))).toBe("é".repeat(32));
  });

  it("changes nothing for an empty name or the same name", () => {
    expect(isEmptyPatch(sent(rename(line(), 2, "   ")))).toBe(true);
    expect(isEmptyPatch(sent(rename(line(), 2, "Mountain")))).toBe(true);
    expect(hasPatch(rename(line(), 2, "Mountain"))).toBe(false);
    expect(hasPatch(rename(line(), 2, "Hills"))).toBe(true);
  });

  it("is refused for the Input and the Output", () => {
    expect(reasonOf(rename(line(), "input", "Camera"))).toBe(GRAPH_OP_TEXT.noRename);
    expect(reasonOf(rename(line(), "output", "Look"))).toBe(GRAPH_OP_TEXT.noRename);
  });
});

describe("Move a card", () => {
  it("changes pos only, rounded to two decimals, and undoes", () => {
    const before = line();
    const { after, undone } = run(before, moveCard(before, 3, [2.3456, 0.9999]));
    expect(posOf(after, 3)).toEqual([2.35, 1]);
    expect(arrows(after)).toEqual(arrows(before));
    expect(undone).toEqual(before);
    expect(sent(moveCard(before, 3, [4, 1]))).toEqual({ nodes: { "3": { pos: [4, 1] } } });
  });

  it("changes nothing when the card is where it was", () => {
    expect(hasPatch(moveCard(line(), 3, [2, 0]))).toBe(false);
    expect(hasPatch(moveCard(line(), 3, [2.001, 0.002]))).toBe(false);
  });

  it("keeps a place inside the core's bounds", () => {
    expect(sent(moveCard(line(), 3, [5000, -5000])).nodes?.["3"]).toEqual({ pos: [1000, -1000] });
  });

  it("can change the numbers: ties follow the place", () => {
    const fan = graph({ 1: {}, 2: { pos: [1, 0] }, 3: { pos: [2, 0] } }, [pic("input", 1), pic(1, 2), pic(1, 3), pic(2, "output")]);
    expect([cardNumbers(fan).get(2), cardNumbers(fan).get(3)]).toEqual([2, 3]);
    const moved = run(fan, moveCard(fan, 3, [0.5, 0])).after;
    expect([cardNumbers(moved).get(2), cardNumbers(moved).get(3)]).toEqual([3, 2]);
  });
});

describe("Reorder", () => {
  it("by dropping puts the node into another link: the neighbours are reconnected first, and its key links stay", () => {
    const before = graph({ 1: {}, 2: {}, 3: {}, 4: {}, 5: { pos: [1, 1] } }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, 4), pic(4, "output"), pic(1, 5), key(2, 5)]);
    // Drop 2 on the link 3 → 4: 1 → 3 → 2 → 4.
    const { after, undone, select } = run(before, reorderInto(before, 2, pic(3, 4)));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>3", "3>2", "2>4", "4>output", "1>5", "2.key>5.key"]));
    expect(select).toBe(2);
    expect(undone).toEqual(before);
    expect(posOf(after, 2)).toEqual([3, 0]);
    expect(posOf(after, 4)).toEqual([4, 0]);
  });

  it("puts a node into the Output's link, and a node before another", () => {
    const before = line();
    const out = run(before, reorderInto(before, 2, pic(4, "output"))).after;
    expect(new Set(arrows(out))).toEqual(new Set(["input>1", "1>3", "3>4", "4>2", "2>output"]));
    const earlier = run(before, reorderInto(before, 4, pic(1, 2))).after;
    expect(new Set(arrows(earlier))).toEqual(new Set(["input>1", "1>4", "4>2", "2>3", "3>output"]));
    expect([4, 2, 3].map((id) => cardNumbers(earlier).get(id))).toEqual([2, 3, 4]);
  });

  it("changes nothing when dropped on one of its own links", () => {
    expect(hasPatch(reorderInto(line(), 3, pic(2, 3)))).toBe(false);
    expect(hasPatch(reorderInto(line(), 3, pic(3, 4)))).toBe(false);
  });

  it("never moves Main, and nothing goes before Main", () => {
    expect(reasonOf(reorderInto(line(), 1, pic(3, 4)))).toBe(GRAPH_OP_TEXT.mainFixed);
    expect(reasonOf(reorderInto(line(), 3, pic("input", 1)))).toBe("Nothing can go before Main.");
    expect(reasonOf(moveEarlier(line(), 1))).toBe(GRAPH_OP_TEXT.mainFixed);
    expect(reasonOf(moveLater(line(), 1))).toBe(GRAPH_OP_TEXT.mainFixed);
    expect(reasonOf(moveEarlier(line(), 2))).toBe("Nothing can go before Main.");
    expect(reasonOf(reorderInto(line(), "input", pic(3, 4)))).toBe(GRAPH_OP_TEXT.markerFixed);
  });

  it("refuses a link that is not there", () => {
    expect(reasonOf(reorderInto(line(), 3, pic(2, 4)))).toBe(GRAPH_OP_TEXT.gone);
    expect(reasonOf(reorderInto(line(), 3, key(2, 4)))).toBe(GRAPH_OP_TEXT.gone);
  });

  it("is refused with the circle sentence when the node's key feeds a node that would then be upstream", () => {
    // 4's key feeds 2's key input. Dropping 4 on the link Main → 2 puts 4 before 2 (fine); dropping 2 after 4 would make 2 feed 4 and 4's key feed 2.
    const before = graph({ 1: {}, 2: {}, 3: {}, 4: {} }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, 4), pic(4, "output"), key(2, 4)]);
    // 2's key feeds 4. Moving 4 before 2 (into Main → 2) would make 4 upstream of 2 while 2's key feeds 4: a circle.
    expect(reasonOf(reorderInto(before, 4, pic(1, 2)))).toBe("That would link the nodes in a circle.");
    expect(reasonOf(moveEarlier(graph({ 1: {}, 2: {}, 3: {} }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output"), key(2, 3)]), 3))).toBe(GRAPH_OP_TEXT.circle);
    // The other way: 2 after 4 would put 4's key source downstream.
    const keyed = graph({ 1: {}, 2: {}, 3: {}, 4: {} }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, 4), pic(4, "output"), key(3, 2)]);
    expect(hasCircle(keyed, keyed.nodes.links)).toBe(true);
  });

  it("with Move earlier and Move later swaps a node with its neighbour", () => {
    const before = line();
    const later = run(before, moveLater(before, 3));
    expect(new Set(arrows(later.after))).toEqual(new Set(["input>1", "1>2", "2>4", "4>3", "3>output"]));
    expect(later.undone).toEqual(before);
    expect([4, 3].map((id) => cardNumbers(later.after).get(id))).toEqual([3, 4]);
    const earlier = run(before, moveEarlier(before, 4));
    expect(new Set(arrows(earlier.after))).toEqual(new Set(["input>1", "1>2", "2>4", "4>3", "3>output"]));
    expect(posOf(earlier.after, 4)).toEqual([2, 0]);
    expect(earlier.undone).toEqual(before);
  });

  it("refuses Move later for the last node and Move earlier for the first after Main, and says which the menu may offer", () => {
    expect(reasonOf(moveLater(line(), 4))).toBe(GRAPH_OP_TEXT.alreadyLast);
    expect(reasonOf(moveEarlier(line(), 2))).toBe(GRAPH_OP_TEXT.beforeMain);
    expect(reorderAvailability(line(), 3)).toEqual({ earlier: null, later: null });
    expect(reorderAvailability(line(), 2)).toEqual({ earlier: GRAPH_OP_TEXT.beforeMain, later: null });
    expect(reorderAvailability(line(), 4)).toEqual({ earlier: null, later: GRAPH_OP_TEXT.alreadyLast });
    expect(reorderAvailability(line(), 1).earlier).toBe(GRAPH_OP_TEXT.mainFixed);
  });

  it("puts a node that took the Input after the one it sits beside, or alone", () => {
    // 2 and 3 both read the Input; 3 feeds the Output. Moving 3 earlier is refused: it already takes its picture from the input.
    const branch = graph({ 1: {}, 2: { pos: [1, 0] }, 3: { pos: [0, 1] } }, [pic("input", 1), pic(1, 2), pic("input", 3), pic(2, "output")]);
    expect(reasonOf(moveEarlier(branch, 3))).toBe(GRAPH_OP_TEXT.alreadyFirst);
    // Moving 2 earlier is refused too: it follows Main.
    expect(reasonOf(moveEarlier(branch, 2))).toBe(GRAPH_OP_TEXT.beforeMain);
  });

  it("moves a node later past a branching one into the link that leads to the Output", () => {
    const before = graph({ 1: {}, 2: {}, 3: {}, 4: { pos: [2, 1] } }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(2, 4), pic(3, "output")]);
    // 2 feeds 3 (leads to the Output) and 4 (dead): later goes after 3.
    const { after } = run(before, moveLater(before, 2));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>3", "1>4", "3>2", "2>output"]));
  });
});

describe("Connect", () => {
  it("links an output port to an input port, picture to picture, and undoes", () => {
    // 5 reads Main and nothing reads it: link 5 → 3's... use the Output instead.
    const before = graph({ 1: {}, 2: {}, 3: { pos: [1, 1] } }, [pic("input", 1), pic(1, 2), pic(2, "output"), pic(1, 3)]);
    const { after, undone } = run(before, connect(before, ref(3), ref("output")));
    // The Output's link is replaced.
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "3>output", "1>3"]));
    expect(undone).toEqual(before);
  });

  it("links a key to a key", () => {
    const before = graph({ 1: {}, 2: {}, 3: {} }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output")]);
    // 2 reads Main, so 2's key may feed 3's key input.
    const { after, undone } = run(before, connect(before, ref(2, "key"), ref(3, "key")));
    expect(arrows(after)).toContain("2.key>3.key");
    expect(undone).toEqual(before);
  });

  it("drops onto an occupied input and replaces its link in the same step", () => {
    const before = graph({ 1: {}, 2: { pos: [1, 0] }, 3: { pos: [2, 0] }, 4: { pos: [1, 1] } }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output"), pic(1, 4)]);
    // 3 takes its picture from 2; link 4 → 3 instead.
    const outcome = connect(before, ref(4), ref(3));
    const { after, undone } = run(before, outcome);
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "4>3", "3>output", "1>4"]));
    expect(undone).toEqual(before);
    expect(sent(outcome).links).toHaveLength(5);
    // One patch, links only.
    expect(Object.keys(sent(outcome))).toEqual(["links"]);
  });

  it("may feed any number of inputs from one output: a branch", () => {
    const before = graph({ 1: {}, 2: {}, 3: { pos: [1, 1] } }, [pic("input", 1), pic(1, 2), pic(2, "output"), pic(1, 3)]);
    const { after } = run(before, connect(before, ref(1), ref(3, "picture")));
    // Already linked: nothing changes.
    expect(after).toEqual(before);
    const branch = graph({ 1: {}, 2: {}, 3: { pos: [1, 1] }, 4: { pos: [1, 2] } }, [pic("input", 1), pic(1, 2), pic(2, "output"), pic(1, 3)]);
    const made = run(branch, connect(branch, ref(1), ref(4))).after;
    expect(new Set(arrows(made))).toEqual(new Set(["input>1", "1>2", "2>output", "1>3", "1>4"]));
  });

  it("connects the Input to another node's picture input: a second branch from the Input", () => {
    const before = graph({ 1: {}, 2: {}, 3: { pos: [1, 1] } }, [pic("input", 1), pic(1, 2), pic(2, "output"), pic(1, 3)]);
    const { after } = run(before, connect(before, ref("input"), ref(3)));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "2>output", "input>3"]));
  });

  it("changes nothing when the link is already there", () => {
    expect(hasPatch(connect(line(), ref(2), ref(3)))).toBe(false);
    expect(hasPatch(connect(line(), ref("input"), ref(1)))).toBe(false);
  });

  it("refuses Main's picture input for anything but the Input: 'Main always takes its picture from the input.'", () => {
    expect(reasonOf(connect(line(), ref(3), ref(1)))).toBe("Main always takes its picture from the input.");
    expect(reasonOf(connect(line(), ref(2), ref(1)))).toBe(GRAPH_OP_TEXT.mainInput);
  });

  it("refuses a circle, through picture links, through key links and through both", () => {
    // Picture: 4 → 2 while 2 feeds 3 feeds 4.
    expect(reasonOf(connect(line(), ref(4), ref(2)))).toBe("That would link the nodes in a circle.");
    // A node into itself.
    expect(reasonOf(connect(line(), ref(3), ref(3)))).toBe(GRAPH_OP_TEXT.circle);
    expect(reasonOf(connect(line(), ref(3, "key"), ref(3, "key")))).toBe(GRAPH_OP_TEXT.circle);
    // Key: 4's key into 2's key input, with 2 → 3 → 4 pictures.
    expect(reasonOf(connect(line(), ref(4, "key"), ref(2, "key")))).toBe(GRAPH_OP_TEXT.circle);
    // Both: 3's key feeds 2's key input? 2 → 3 by picture: a circle through one picture and one key link.
    expect(reasonOf(connect(line(), ref(3, "key"), ref(2, "key")))).toBe(GRAPH_OP_TEXT.circle);
    // And the other way is fine: 2's key may feed 4.
    expect(connect(line(), ref(2, "key"), ref(4, "key")).ok).toBe(true);
  });

  it("refuses a picture on a key and a key on a picture, and the Output's missing key input", () => {
    expect(reasonOf(connect(line(), ref(2), ref(3, "key")))).toBe(GRAPH_OP_TEXT.portTypes);
    expect(reasonOf(connect(line(), ref(2, "key"), ref(3)))).toBe(GRAPH_OP_TEXT.portTypes);
    expect(reasonOf(connect(line(), ref(2, "key"), ref("output")))).toBe(GRAPH_OP_TEXT.portTypes);
  });

  it("refuses a link out of the Output, into the Input, or from a port the node does not have", () => {
    expect(connect(line(), ref("output"), ref(2)).ok).toBe(false);
    expect(connect(line(), ref(2), ref("input")).ok).toBe(false);
    expect(connect(line(), ref("input", "key"), ref(2, "key")).ok).toBe(false);
    expect(reasonOf(connect(line(), ref(77), ref(2)))).toBe(GRAPH_OP_TEXT.gone);
  });
});

describe("Pick up a link's end", () => {
  it("moves it to another input of the same type, replacing the link there", () => {
    const before = graph({ 1: {}, 2: {}, 3: { pos: [1, 1] }, 4: { pos: [2, 1] } }, [pic("input", 1), pic(1, 2), pic(2, "output"), pic(1, 3), pic(3, 4)]);
    // 3 → 4 picked up at 4 and dropped on the Output: the Output's link (from 2) is replaced.
    const { after, undone } = run(before, reconnect(before, pic(3, 4), ref("output")));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "3>output", "1>3"]));
    expect(undone).toEqual(before);
  });

  it("puts the link back when dropped on its own input: nothing changes", () => {
    expect(hasPatch(reconnect(line(), pic(2, 3), ref(3)))).toBe(false);
  });

  it("moves a key link to another key input", () => {
    const before = graph({ 1: {}, 2: {}, 3: {}, 4: {} }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, 4), pic(4, "output"), key(2, 3)]);
    const { after, undone } = run(before, reconnect(before, key(2, 3), ref(4, "key")));
    expect(arrows(after)).toContain("2.key>4.key");
    expect(arrows(after)).not.toContain("2.key>3.key");
    expect(undone).toEqual(before);
  });

  it("refuses the same things as a new link, and a different type", () => {
    expect(reasonOf(reconnect(line(), pic(2, 3), ref(2)))).toBe(GRAPH_OP_TEXT.circle);
    expect(reasonOf(reconnect(line(), pic(2, 3), ref(4, "key")))).toBe(GRAPH_OP_TEXT.portTypes);
  });

  it("cannot move Input → Main, nor take the Output's link anywhere but another node into the Output", () => {
    expect(reasonOf(reconnect(line(), pic("input", 1), ref(2)))).toBe("Main always takes its picture from the input.");
    expect(reasonOf(reconnect(line(), pic(4, "output"), ref(3)))).toBe("The output needs a picture. Link another node to it instead.");
    // It may be replaced: another node's picture into the Output.
    expect(reconnect(line(), pic(4, "output"), ref("output")).ok).toBe(true);
  });
});

describe("Remove a link", () => {
  it("takes a link away, as dropping its end on the canvas does, and undoes", () => {
    const before = line();
    const { after, undone } = run(before, disconnect(before, pic(2, 3)));
    expect(new Set(arrows(after))).toEqual(new Set(["input>1", "1>2", "3>4", "4>output"]));
    expect(undone).toEqual(before);
    // 3 is dead now, and still numbered.
    expect(cardStates(after).get(3)?.live).toBe(true);
    expect(cardStates(after).get(2)?.live).toBe(false);
  });

  it("takes a key link away", () => {
    const before = graph({ 1: {}, 2: {}, 3: {} }, [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output"), key(2, 3)]);
    const { after, undone } = run(before, disconnect(before, key(2, 3)));
    expect(arrows(after)).not.toContain("2.key>3.key");
    expect(undone).toEqual(before);
  });

  it("refuses Input → Main and the Output's link, with their sentences", () => {
    expect(reasonOf(disconnect(line(), pic("input", 1)))).toBe("Main always takes its picture from the input.");
    expect(reasonOf(disconnect(line(), pic(4, "output")))).toBe("The output needs a picture. Link another node to it instead.");
  });

  it("refuses a link that is not there", () => {
    expect(reasonOf(disconnect(line(), pic(2, 4)))).toBe(GRAPH_OP_TEXT.gone);
  });
});

describe("Which inputs light up", () => {
  const lit = (states: ReturnType<typeof inputStates>): string[] => states.filter((s) => s.ok).map((s) => `${s.to.node}:${s.to.port}`);

  it("are the inputs of the same type that would make no circle, from a picture output", () => {
    // From 2's picture output in the line 1 → 2 → 3 → 4: not Main (fixed), not 2 itself, and 3's input is its own link (replacing it changes nothing).
    const states = inputStates(line(), ref(2));
    expect(lit(states)).toEqual(["3:picture", "4:picture", "output:picture"]);
    // Main's picture input is refused with its sentence.
    expect(states.find((s) => s.to.node === 1 && s.to.port === "picture")).toEqual({ to: ref(1), ok: false, reason: GRAPH_OP_TEXT.mainInput });
    // A key input is the other type.
    expect(states.find((s) => s.to.node === 3 && s.to.port === "key")?.reason).toBe(GRAPH_OP_TEXT.portTypes);
    // A circle: 2 cannot feed itself.
    expect(states.find((s) => s.to.node === 2 && s.to.port === "picture")?.reason).toBe(GRAPH_OP_TEXT.circle);
  });

  it("are the key inputs from a key output, and never a node upstream", () => {
    // 3's key may feed 4 (downstream), not 2 (upstream: 2 → 3 is a picture link) and not 3.
    const states = inputStates(line(), ref(3, "key"));
    expect(lit(states)).toEqual(["4:key"]);
  });

  it("light the Input's branch: the Input's picture reaches any picture input but Main's", () => {
    const states = inputStates(line(), ref("input"));
    // Main's input is already the Input's link: putting it there changes nothing, so it is allowed.
    expect(lit(states)).toEqual(["1:picture", "2:picture", "3:picture", "4:picture", "output:picture"]);
  });

  it("light, for a lifted link, the input it came from too: putting it back is allowed", () => {
    const states = inputStates(line(), ref(2), pic(2, 3));
    expect(lit(states)).toContain("3:picture");
    expect(lit(states)).toContain("output:picture");
    // The lifted link's own end: Input → Main is fixed, whatever input it is dropped on.
    expect(lit(inputStates(line(), ref("input"), pic("input", 1)))).toEqual([]);
    // The Output's link may be moved to the Output only.
    expect(lit(inputStates(line(), ref(4), pic(4, "output")))).toEqual(["output:picture"]);
  });

  it("agree with the connect that a drop would make", () => {
    for (const s of inputStates(line(), ref(2))) {
      expect(connect(line(), ref(2), s.to).ok).toBe(s.ok);
    }
  });

  it("list, for the More menu, the inputs a node can be linked to, not the ones it is linked to already", () => {
    expect(linkTargets(line(), ref(2)).map((r) => `${r.node}:${r.port}`)).toEqual(["4:picture", "output:picture"]);
    expect(linkTargets(line(), ref(2, "key")).map((r) => `${r.node}:${r.port}`)).toEqual(["3:key", "4:key"]);
    expect(linkTargets(line(), ref("input")).map((r) => r.node)).toEqual([2, 3, 4, "output"]);
    expect(linkTargets(line(), ref("output"))).toEqual([]);
    expect(linkTargets(line(), ref("input", "key"))).toEqual([]);
  });

  it("list a node's links for Remove link", () => {
    expect(linksOf(line(), 3).map((l) => `${l.from.node}>${l.to.node}`)).toEqual(["2>3", "3>4"]);
    expect(linksOf(line(), "input")).toHaveLength(1);
  });
});

describe("Names for the cards", () => {
  it("name a card with its number, its name, its masks, what it is, and what feeds it", () => {
    const g = graph(
      { 1: {}, 2: { name: "Mountain", masks: ["object", "radial"] }, 3: { name: "Sky", bypass: true }, 5: { name: "Skin", pos: [1, 1] } },
      [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output"), pic(1, 5), key(5, 3)],
    );
    expect(cardName(g, 1)).toBe("Node 1, Main");
    expect(cardName(g, 2)).toBe("Node 2, Mountain, masked by Object and Radial, picture from node 1");
    expect(cardName(g, 3)).toBe("Node 4, Sky, bypassed, picture from node 2, key from node 3");
    // Skin reads Main and only its key reaches Sky, which is live: so it is live too.
    expect(cardName(g, 5)).toBe("Node 3, Skin, picture from node 1");
    expect(cardName(g, "input")).toBe("Input");
    expect(cardName(g, "output")).toBe("Output");
  });

  it("say a dead node is not connected", () => {
    const g = graph({ 1: {}, 2: { name: "Spare" } }, [pic("input", 1), pic(1, "output"), pic(1, 2)]);
    expect(cardName(g, 2)).toBe("Node 2, Spare, not connected, picture from node 1");
  });

  it("leave out masks that are switched off, and join the kinds in words", () => {
    expect(joinWords([])).toBe("");
    expect(joinWords(["Object"])).toBe("Object");
    expect(joinWords(["Object", "Radial"])).toBe("Object and Radial");
    expect(joinWords(["Object", "Radial", "Colour range"])).toBe("Object, Radial and Colour range");
  });
});
