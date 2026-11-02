import { describe, expect, it } from "vitest";
import { MOCK_ESTIMATES, resolveSettings } from "../api.mock";
import type { DevelopSettings, MaskItem, NodeLink, SettingsPatch } from "../api/types";
import { applyPatch, defaultLinks, defaultSettings, inputOf, inputPatch, newCorrector, nodeOf, outputPatch } from "../state/settingsGraph";
import { copiedPatch, COPY_LEAVES_OUT, GRAPH_TEXT, MAX_NODES, pastedOnto, refusalOfPatch, validateGraph } from "./graph";

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const key = (from: number, to: number): NodeLink => ({ from: { node: from, port: "key" }, to: { node: to, port: "key" } });

/** Main, then nodes 2 and 3 in a line. */
function line(): DevelopSettings {
  const base = defaultSettings();
  return {
    nodes: {
      items: [...base.nodes.items.slice(0, 2), newCorrector(2, [1, 0]), newCorrector(3, [2, 0]), ...base.nodes.items.slice(2)],
      links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, "output")],
    },
  };
}

const withLinks = (s: DevelopSettings, links: NodeLink[]): DevelopSettings => ({ nodes: { ...s.nodes, links } });

describe("validateGraph: the core's rules, with the core's words (docs/23 section 3.5)", () => {
  it("accepts the default graph, a line of nodes and a key link", () => {
    expect(validateGraph(defaultSettings())).toBeNull();
    expect(validateGraph(line())).toBeNull();
    // A node that only makes a mask for another: node 3 reads Main's picture (a dead branch with a live key), and its key feeds node 2.
    const keyed = withLinks(line(), [picture("input", 1), picture(1, 2), picture(2, "output"), picture(1, 3), key(3, 2)]);
    expect(validateGraph(keyed)).toBeNull();
  });

  it("wants the Input, the Output and Main", () => {
    const base = defaultSettings();
    const noOutput: DevelopSettings = { nodes: { items: base.nodes.items.slice(0, 2), links: [picture("input", 1)] } };
    expect(validateGraph(noOutput)).toBe(GRAPH_TEXT.missingFixed);
    expect(GRAPH_TEXT.missingFixed).toBe("A photo needs its input, its output and node 1, Main.");
  });

  it("refuses two nodes with one id", () => {
    const s = line();
    const twice: DevelopSettings = { nodes: { ...s.nodes, items: [...s.nodes.items, newCorrector(2, [5, 5])] } };
    expect(validateGraph(twice)).toBe("Two nodes have the id 2.");
  });

  it("refuses a link that leaves the Output or enters the Input", () => {
    const direction = "A link goes from the output of one node to the input of another.";
    expect(validateGraph(withLinks(defaultSettings(), [...defaultLinks(), { from: { node: "output", port: "picture" }, to: { node: 1, port: "key" } }]))).toBe(direction);
    expect(validateGraph(withLinks(defaultSettings(), [...defaultLinks(), picture(1, "input" as never)]))).toBe(direction);
  });

  it("refuses a link that names a node the photo does not have, or a port the node does not have", () => {
    expect(validateGraph(withLinks(line(), [...line().nodes.links, picture(3, 7)]))).toBe("A link names node 7, which this photo does not have.");
    const noKey = { from: { node: "input", port: "key" }, to: { node: 1, port: "key" } } as NodeLink;
    expect(validateGraph(withLinks(line(), [noKey]))).toBe("The input has no port called “key”.");
    const toOutputKey = { from: { node: 1, port: "key" }, to: { node: "output", port: "key" } } as NodeLink;
    expect(validateGraph(withLinks(line(), [toOutputKey]))).toBe("The output has no port called “key”.");
  });

  it("refuses a picture joined to a key", () => {
    const mixed = { from: { node: 1, port: "picture" }, to: { node: 2, port: "key" } } as NodeLink;
    expect(validateGraph(withLinks(line(), [...line().nodes.links, mixed]))).toBe("A link joins a picture to a picture, or a key to a key.");
  });

  it("refuses an input with two links", () => {
    expect(validateGraph(withLinks(line(), [...line().nodes.links, picture(1, 3)]))).toBe("An input can have one link. Node 3 has two on its picture input.");
    expect(validateGraph(withLinks(line(), [picture("input", 1), picture(1, "output"), picture(1, 2), picture(2, 3), picture(3, "output")]))).toBe(
      "An input can have one link. The output has two on its picture input.",
    );
  });

  it("wants one picture link to the Output, and Main's picture from the Input", () => {
    expect(validateGraph(withLinks(line(), line().nodes.links.slice(0, 3)))).toBe("The output needs one picture link.");
    expect(validateGraph(withLinks(line(), [picture("input", 2), picture(2, 1), picture(1, 3), picture(3, "output")]))).toBe(
      "Node 1, Main, must take its picture from the input.",
    );
  });

  it("refuses circles, through picture links, key links or both", () => {
    const circle = "The nodes of this photo are linked in a circle. Remove a link.";
    expect(validateGraph(withLinks(line(), [picture("input", 1), picture(1, "output"), picture(2, 3), picture(3, 2)]))).toBe(circle);
    const throughKeys = withLinks(line(), [picture("input", 1), picture(1, 2), picture(1, 3), key(3, 2), key(2, 3), picture(2, "output")]);
    expect(validateGraph(throughKeys)).toBe(circle);
    const both = withLinks(line(), [picture("input", 1), picture(1, 2), picture(2, 3), key(3, 2), picture(3, "output")]);
    expect(validateGraph(both)).toBe(circle);
  });

  it("allows 64 correctors, Main included, and no more", () => {
    const many = (n: number): DevelopSettings => {
      const base = defaultSettings();
      const extra = Array.from({ length: n - 1 }, (_, i) => newCorrector(i + 2, [i + 1, 0]));
      const chain = [picture("input", 1), ...Array.from({ length: n - 1 }, (_, i) => picture(i + 1, i + 2)), picture(n, "output")];
      return { nodes: { items: [base.nodes.items[0]!, base.nodes.items[1]!, ...extra, base.nodes.items[2]!], links: chain } };
    };
    expect(validateGraph(many(MAX_NODES))).toBeNull();
    expect(validateGraph(many(MAX_NODES + 1))).toBe("A photo can have up to 64 nodes.");
  });

  it("holds the masks to their limits", () => {
    const row = { kind: "linear", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], angle_deg: 0, feather: 0.2 } as const;
    const withMasks = (items: unknown[]): DevelopSettings => applyPatch(defaultSettings(), { nodes: { "1": { masks: { items: items as never } } } });
    expect(validateGraph(withMasks(Array.from({ length: 64 }, () => row)))).toBeNull();
    expect(validateGraph(withMasks(Array.from({ length: 65 }, () => row)))).toBe("A node can have up to 64 masks.");
    const object = (n: number) => ({ kind: "object", mode: "add", invert: false, enabled: true, opacity: 1, points: Array.from({ length: n }, () => [0.5, 0.5, true]), box: null, feather: 0, grow: 0, refine: 0.5 });
    expect(validateGraph(withMasks([object(256)]))).toBeNull();
    expect(validateGraph(withMasks([object(257)]))).toBe("An Object mask can have up to 256 points.");
    const brush = (n: number) => ({ kind: "brush", mode: "add", invert: false, enabled: true, opacity: 1, strokes: [{ size: 0.03, hardness: 0.5, erase: false, points: Array.from({ length: n }, () => [0.1, 0.1]) }], feather: 0, grow: 0, refine: 0 });
    expect(validateGraph(withMasks([brush(100_000)]))).toBeNull();
    expect(validateGraph(withMasks([brush(100_001)]))).toBe("A brush mask can have up to 100,000 points.");
  });
});

describe("refusalOfPatch: what the core refuses before it applies a patch", () => {
  const s = line();
  const refuse = (patch: SettingsPatch): string | null => refusalOfPatch(s, patch);

  it("lets a node's own fields through, and a node that is added with its kind", () => {
    expect(refuse(inputPatch({ exposure: { kind: "manual", ev: 1 } }))).toBeNull();
    expect(refuse(outputPatch({ look: "builtin:signature" }))).toBeNull();
    expect(refuse({ nodes: { "2": { trim: { tint: 4 }, grade: { white: 3 }, masks: { invert: true }, name: "x" } } })).toBeNull();
    expect(refuse({ nodes: { "9": { kind: "corrector", name: "New" } } })).toBeNull();
    expect(refuse({ remove: [3], links: [] })).toBeNull();
  });

  it("refuses to remove the Input, the Output or Main", () => {
    for (const id of ["input", "output", 1] as const) expect(refuse({ remove: [id] })).toBe("The input, the output and node 1, Main, can't be removed.");
  });

  it("refuses a node the photo does not have, unless the patch gives its kind", () => {
    expect(refuse({ nodes: { "7": { name: "Peak" } } })).toBe("The patch names node 7, which this photo does not have.");
  });

  it("refuses a kind other than the node's", () => {
    expect(refuse({ nodes: { "2": { kind: "input" } } })).toBe("Node 2 is a corrector, not an input.");
    expect(refuse({ nodes: { input: { kind: "corrector" } } })).toBe("The input is an input, not a corrector.");
    expect(refuse({ nodes: { output: { kind: "input" } } })).toBe("The output is an output, not an input.");
    expect(refuse({ nodes: { "7": { kind: "output" } } })).toBe(GRAPH_TEXT.missingFixed);
  });

  it("refuses a field the node's kind does not have", () => {
    expect(refuse({ nodes: { "2": { exposure: { kind: "auto" } } } })).toBe("A corrector has no field called “exposure”.");
    expect(refuse({ nodes: { input: { grade: { white: 1 } } } })).toBe("An input has no field called “grade”.");
    expect(refuse({ nodes: { output: { name: "x" } } })).toBe("An output has no field called “name”.");
  });
});

describe("Copy and Paste: the core's rules (docs/23 section 3.8)", () => {
  it("copies the whole graph, with the Input's crop, orientation and white pick left out", () => {
    const edited = applyPatch(
      line(),
      {
        nodes: {
          input: { exposure: { kind: "manual", ev: 0.8 }, crop: { rect: { x: 0, y: 0, width: 0.5, height: 0.5 }, angle_deg: 1, aspect: "free" }, orientation: 90, white_pick: [0.5, 0.5] },
          "2": { name: "Peak", trim: { exposure_ev: 0.4 }, pos: [3, 1] },
        },
      },
    );
    const patch = copiedPatch(resolveSettings(1, edited, MOCK_ESTIMATES));
    expect(Object.keys(patch.nodes ?? {}).sort()).toEqual(["1", "2", "3", "input", "output"]);
    for (const field of COPY_LEAVES_OUT) expect(patch.nodes?.input).not.toHaveProperty(field);
    expect(patch.nodes?.input).toMatchObject({ kind: "input", exposure: { kind: "manual", ev: 0.8 } });
    expect(patch.nodes?.["2"]).toMatchObject({ kind: "corrector", name: "Peak", pos: [3, 1], trim: { exposure_ev: 0.4 } });
    expect(patch.links).toEqual(edited.nodes.links);
    expect(patch.remove).toBeUndefined();
  });

  it("copies Auto as Auto in the Input and Main, and Main's Auto stays Auto", () => {
    const patch = copiedPatch(resolveSettings(1, defaultSettings(), MOCK_ESTIMATES));
    expect(patch.nodes?.input).toMatchObject({ exposure: { kind: "auto" }, white_balance: { kind: "auto" }, sharpen: { kind: "auto" } });
    expect(patch.nodes?.["1"]?.grade?.contrast).toEqual({ kind: "auto" });
  });

  it("copies an Object mask empty, so it selects nothing on the new photo until it is clicked", () => {
    const object: MaskItem = { kind: "object", mode: "add", invert: false, enabled: true, opacity: 1, points: [[0.4, 0.4, true]], box: [0.1, 0.1, 0.5, 0.5], feather: 0, grow: 0, refine: 0.5 };
    const linear: MaskItem = { kind: "linear", mode: "subtract", invert: false, enabled: true, opacity: 1, center: [0.5, 0.2], angle_deg: 4, feather: 0.25 };
    const s = applyPatch(line(), { nodes: { "2": { masks: { items: [object, linear] } } } });
    const rows = copiedPatch(resolveSettings(1, s, MOCK_ESTIMATES)).nodes?.["2"]?.masks?.items;
    expect(rows?.[0]).toMatchObject({ kind: "object", points: [], box: null, refine: 0.5 });
    expect(rows?.[1]).toEqual(linear);
  });

  it("pastes without the Input's crop, orientation and white pick, whatever the patch says", () => {
    const target = line();
    const sneaky: SettingsPatch = { nodes: { input: { exposure: { kind: "manual", ev: 1 }, crop: null, orientation: 270, white_pick: [0.1, 0.1] } } };
    const pasted = pastedOnto(target, sneaky);
    expect(pasted.nodes?.input).toEqual({ exposure: { kind: "manual", ev: 1 } });
    expect(pasted.remove).toBeUndefined();
  });

  it("replaces the target's graph: a copy with links removes the target's correctors the copy does not have", () => {
    const source = copiedPatch(resolveSettings(1, defaultSettings(), MOCK_ESTIMATES));
    const pasted = pastedOnto(line(), source);
    expect(pasted.remove?.slice().sort()).toEqual([2, 3]);
    expect(refusalOfPatch(line(), pasted)).toBeNull();
    const result = applyPatch(line(), pasted);
    expect(validateGraph(result)).toBeNull();
    expect(result.nodes.items.map((n) => n.id)).toEqual(["input", 1, "output"]);
    expect(inputOf(result).exposure).toEqual({ kind: "auto" });
    expect(nodeOf(result, 2)).toBeUndefined();
  });
});
