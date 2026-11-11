import { describe, expect, it } from "vitest";
import { MOCK_ESTIMATES, resolveSettings } from "../api.mock";
import type { DevelopSettings, MaskItem, NodeLink, SettingsPatch } from "../api/types";
import {
  applyPatch,
  defaultLinks,
  defaultSettings,
  faceSettingsFor,
  inputOf,
  inputPatch,
  newCorrector,
  newFaceRefinement,
  newMixer,
  nodeOf,
  outputPatch,
} from "../state/settingsGraph";
import { copiedPatch, COPY_LEAVES_OUT, GRAPH_TEXT, MAX_NODES, pastedOnto, portsOf, refusalOfPatch, validateGraph } from "./graph";

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

// ---------------------------------------------------------------------------
// The mixers, the Splitter and the Combiner (docs/25 section 2)
// ---------------------------------------------------------------------------

const link = (from: number | "input", fromPort: string, to: number | "output", toPort: string): NodeLink => ({ from: { node: from, port: fromPort }, to: { node: to, port: toPort } }) as NodeLink;

/** Main -> a Layer (node 2, input 1) -> the Output, with the Layer's other links given. */
function withLayer(more: NodeLink[] = [], kind: "layer" | "parallel" = "layer"): DevelopSettings {
  const base = defaultSettings();
  return {
    nodes: {
      items: [...base.nodes.items.slice(0, 2), newMixer(kind, 2, [1, 0]), ...base.nodes.items.slice(2)],
      links: [picture("input", 1), link(1, "picture", 2, "picture.1"), picture(2, "output"), ...more],
    },
  };
}

describe("the nodes of docs/25: ports and rules, with the core's words (docs/25 section 2.3)", () => {
  it("has the ports of each kind", () => {
    expect(portsOf("parallel")).toEqual([["picture.1", "picture.2", "picture.3", "picture.4", "picture.5", "picture.6", "picture.7", "picture.8"], ["picture"]]);
    expect(portsOf("layer")[1]).toEqual(["picture"]);
    expect(portsOf("splitter")).toEqual([["picture"], ["red", "green", "blue"]]);
    expect(portsOf("combiner")).toEqual([["red", "green", "blue"], ["picture"]]);
    expect(portsOf("corrector")).toEqual([["picture", "key"], ["picture", "key"]]);
  });

  it("accepts a Layer, a Parallel and the Splitter and Combiner of docs/25, with a channel swap", () => {
    expect(validateGraph(withLayer())).toBeNull();
    expect(validateGraph(withLayer([link(1, "picture", 2, "picture.2")], "parallel"))).toBeNull();
    const base = defaultSettings();
    const trio: DevelopSettings = {
      nodes: {
        items: [...base.nodes.items.slice(0, 2), newMixer("splitter", 2, [1, 0]), newMixer("combiner", 3, [2, 0]), ...base.nodes.items.slice(2)],
        links: [picture("input", 1), picture(1, 2), link(2, "red", 3, "green"), link(2, "green", 3, "blue"), link(2, "blue", 3, "red"), picture(3, "output")],
      },
    };
    expect(validateGraph(trio)).toBeNull();
  });

  it("refuses a ninth input", () => {
    expect(validateGraph(withLayer([link(1, "picture", 2, "picture.9")]))).toBe("A mixer can have up to 8 inputs.");
    expect(validateGraph(withLayer([link(1, "picture", 2, "picture.10")]))).toBe("A mixer can have up to 8 inputs.");
    expect(validateGraph(withLayer([link(1, "picture", "output", "picture.9")]))).toBe("The output has no port called “picture.9”.");
  });

  it("refuses a gap in a mixer's inputs", () => {
    const text = "Node 2 has a link on input 3 and none on input 2. A mixer's inputs are numbered from 1 with no gap.";
    expect(validateGraph(withLayer([link(1, "picture", 2, "picture.3")]))).toBe(text);
    expect(validateGraph(withLayer([link(1, "picture", 2, "picture.3")], "parallel"))).toBe(text);
    const first = withLayer();
    const noFirst = withLinks(first, [picture("input", 1), link(1, "picture", 2, "picture.2"), picture(2, "output")]);
    expect(validateGraph(noFirst)).toBe("Node 2 has a link on input 2 and none on input 1. A mixer's inputs are numbered from 1 with no gap.");
  });

  it("refuses a port the kind does not have, with the existing text", () => {
    for (const port of ["picture", "picture.0", "picture.01", "key", "red"]) {
      expect(validateGraph(withLayer([link(1, "picture", 2, port)])), port).toBe(`Node 2 has no port called “${port}”.`);
    }
    expect(validateGraph(withLayer([link(1, "key", 2, "key")]))).toBe("Node 2 has no port called “key”.");
    expect(validateGraph(withLayer([link(2, "key", 1, "key")]))).toBe("Node 2 has no port called “key”.");
    expect(validateGraph(withLayer([link(2, "red", 1, "picture")]))).toBe("Node 2 has no port called “red”.");
  });

  it("joins pictures to pictures and refuses a key into a mixer", () => {
    const keyInto = { from: { node: 1, port: "key" }, to: { node: 2, port: "picture.2" } } as NodeLink;
    expect(validateGraph(withLayer([keyInto]))).toBe(GRAPH_TEXT.linkPortTypes);
  });

  it("refuses two links on one mixer input, and counts the new kinds toward the 64 nodes", () => {
    expect(validateGraph(withLayer([link("input", "picture", 2, "picture.1")]))).toBe("An input can have one link. Node 2 has two on its picture.1 input.");
    const base = defaultSettings();
    const many = (n: number): DevelopSettings => ({
      nodes: {
        items: [...base.nodes.items.slice(0, 2), ...Array.from({ length: n }, (_, i) => newMixer("splitter", 2 + i, [1, 0])), ...base.nodes.items.slice(2)],
        links: defaultLinks(),
      },
    });
    expect(validateGraph(many(MAX_NODES - 1))).toBeNull();
    expect(validateGraph(many(MAX_NODES))).toBe("A photo can have up to 64 nodes.");
  });

  it("refuses a field of another kind in a patch, with the kind's name", () => {
    const s = withLayer();
    expect(refusalOfPatch(s, { nodes: { "2": { mode: "add", opacity: 0.5, bypass: true, name: "Glow" } } })).toBeNull();
    expect(refusalOfPatch(s, { nodes: { "2": { masks: { invert: true } } } })).toBe("A layer mixer has no field called “masks”.");
    expect(refusalOfPatch(s, { nodes: { "2": { kind: "corrector" } } })).toBe("Node 2 is a layer mixer, not a corrector.");
    expect(refusalOfPatch(s, { nodes: { "5": { kind: "combiner" } } })).toBeNull();
    expect(refusalOfPatch(s, { nodes: { "5": { kind: "splitter", bypass: true } } })).toBe("A splitter has no field called “bypass”.");
  });

  it("adds each kind with its defaults, changes mode and opacity, and copies the nodes whole", () => {
    let s = defaultSettings();
    s = applyPatch(s, {
      nodes: { "2": { kind: "layer", mode: "multiply", opacity: 0.4 }, "3": { kind: "parallel" }, "4": { kind: "splitter" }, "5": { kind: "combiner" } },
    });
    expect(nodeOf(s, 2)).toMatchObject({ kind: "layer", name: "Layer", bypass: false, mode: "multiply", opacity: 0.4 });
    expect(nodeOf(s, 3)).toMatchObject({ kind: "parallel", name: "Parallel", bypass: false });
    expect(nodeOf(s, 4)).toMatchObject({ kind: "splitter", name: "Splitter" });
    expect(nodeOf(s, 5)).toMatchObject({ kind: "combiner", name: "Combiner" });
    s = applyPatch(s, { nodes: { "2": { mode: "darken" } } });
    expect(nodeOf(s, 2)).toMatchObject({ mode: "darken", opacity: 0.4 });
    const copy = copiedPatch(resolveSettings(1, s, MOCK_ESTIMATES));
    expect(copy.nodes?.["2"]).toMatchObject({ kind: "layer", mode: "darken", opacity: 0.4 });
    expect(copy.nodes?.["4"]).toEqual({ kind: "splitter", name: "Splitter", pos: [3, 0] });
  });
});

describe("the Face Refinement node (docs/27 S-11a)", () => {
  const threeNodes = (): DevelopSettings => {
    const base = defaultSettings();
    return {
      nodes: {
        items: [...base.nodes.items.slice(0, 2), newFaceRefinement(2, [1, 0]), ...base.nodes.items.slice(2)],
        links: [picture("input", 1), picture(1, 2), picture(2, "output")],
      },
    };
  };

  it("starts at the approved amounts, with Same for all faces on and nothing protected", () => {
    expect(newFaceRefinement(2, [1, 0])).toEqual({
      kind: "face_refinement",
      id: 2,
      name: "Face Refinement",
      bypass: false,
      pos: [1, 0],
      same_for_all: true,
      all: { blemish: 30, dark_circles: 30, even_skin: 25, texture: 0, skin_contrast: 0, shadow_desaturation: 0, shine_removal: 0, teeth_whitening: 0 },
      faces: [],
      protect: { strokes: [], feather: 0, grow: 0, refine: 0 },
      masks: { invert: false, items: [] },
    });
  });

  it("has the ports of a corrector, and the core's sentences for its fields", () => {
    expect(portsOf("face_refinement")).toEqual(portsOf("corrector"));
    const s = threeNodes();
    expect(validateGraph(s)).toBeNull();
    expect(refusalOfPatch(s, { nodes: { "2": { same_for_all: false, all: { blemish: 5 }, faces: [], protect: { strokes_keep: 0 }, masks: { invert: true } } } })).toBeNull();
    expect(refusalOfPatch(s, { nodes: { "2": { trim: { tint: 1 } } } })).toBe("A face refinement node has no field called “trim”.");
    expect(refusalOfPatch(s, { nodes: { "1": { faces: [] } } })).toBe("A corrector has no field called “faces”.");
    expect(refusalOfPatch(s, { nodes: { "2": { kind: "corrector" } } })).toBe("Node 2 is a face refinement node, not a corrector.");
    expect(refusalOfPatch(defaultSettings(), { nodes: { "2": { kind: "face_refinement" } } })).toBeNull();
  });

  it("refuses more than 32 faces and holds the protection to the brush limit", () => {
    const s = threeNodes();
    const face = (nodeOf(s, 2) as ReturnType<typeof newFaceRefinement>).all;
    const many = (n: number): DevelopSettings => applyPatch(s, { nodes: { "2": { faces: Array.from({ length: n }, () => face) } } });
    expect(validateGraph(many(32))).toBeNull();
    expect(validateGraph(many(33))).toBe(GRAPH_TEXT.tooManyFaces);
    const stroke = { size: 0.02, hardness: 0.5, erase: false, points: Array.from({ length: 100_001 }, (): [number, number] => [0.5, 0.5]) };
    expect(validateGraph(applyPatch(s, { nodes: { "2": { protect: { strokes: [stroke] } } } }))).toBe(GRAPH_TEXT.tooManyBrushPoints);
  });

  it("merges `all` by amount and applies the protection as a Brush row does", () => {
    let s = applyPatch(threeNodes(), { nodes: { "2": { all: { blemish: 70, texture: -20 } } } });
    expect(nodeOf(s, 2)).toMatchObject({ all: { blemish: 70, texture: -20, dark_circles: 30, even_skin: 25 } });
    const stroke = (x: number) => ({ size: 0.02, hardness: 0.5, erase: false, points: [[x, 0.5]] as [number, number][] });
    s = applyPatch(s, { nodes: { "2": { protect: { strokes_append: [stroke(0.1)] } } } });
    s = applyPatch(s, { nodes: { "2": { protect: { strokes_append: [stroke(0.2), stroke(0.3)] } } } });
    expect(nodeOf(s, 2)).toMatchObject({ protect: { strokes: [stroke(0.1), stroke(0.2), stroke(0.3)] } });
    s = applyPatch(s, { nodes: { "2": { protect: { strokes_keep: 1, feather: 0.4 } } } });
    expect(nodeOf(s, 2)).toMatchObject({ protect: { strokes: [stroke(0.1)], feather: 0.4 } });
  });

  it("gives face N the Nth entry, a face past the list face 1's, and `all` while Same for all faces is on", () => {
    const entry = (blemish: number) => ({ ...newFaceRefinement(2, [0, 0]).all, blemish });
    let s = applyPatch(threeNodes(), { nodes: { "2": { same_for_all: false, all: { blemish: 11 }, faces: [entry(40), entry(50)] } } });
    const node = () => nodeOf(s, 2) as ReturnType<typeof newFaceRefinement>;
    expect([1, 2, 3, 9, 0].map((n) => faceSettingsFor(node(), n).blemish)).toEqual([40, 50, 40, 40, 40]);
    s = applyPatch(s, { nodes: { "2": { same_for_all: true } } });
    expect([1, 2, 9].map((n) => faceSettingsFor(node(), n).blemish)).toEqual([11, 11, 11]);
    s = applyPatch(s, { nodes: { "2": { same_for_all: false, faces: [] } } });
    expect(faceSettingsFor(node(), 3).blemish).toBe(11);
  });

  it("copies the node whole with its masks", () => {
    const s = applyPatch(threeNodes(), { nodes: { "2": { same_for_all: false, faces: [{ ...newFaceRefinement(2, [0, 0]).all, blemish: 12 }] } } });
    const copy = copiedPatch(resolveSettings(1, s, MOCK_ESTIMATES));
    expect(copy.nodes?.["2"]).toMatchObject({ kind: "face_refinement", same_for_all: false, masks: { invert: false, items: [] } });
    expect(copy.nodes?.["2"]?.faces?.[0]?.blemish).toBe(12);
  });
});
