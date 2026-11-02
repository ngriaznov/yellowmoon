import { describe, expect, it } from "vitest";
import type { BrushStroke, DevelopSettings, MaskItem, SettingsPatch } from "../api/types";
import { PhotoHistory } from "./history";
import {
  applyPatch,
  correctorsOf,
  defaultLinks,
  defaultSettings,
  diffGraph,
  foldPatches,
  hiddenByPhoto,
  hiddenFromSimpleView,
  inCanonicalOrder,
  INPUT_ID,
  inputOf,
  inputPatch,
  isEmptyPatch,
  mainGradeOf,
  mainGradePatch,
  MAIN_ID,
  mainOf,
  mainPatch,
  newCorrector,
  nodeAsPatch,
  nodeOf,
  OUTPUT_ID,
  outputOf,
  outputPatch,
  patchOf,
  same,
  withoutLook,
} from "./settingsGraph";

const manual = (value: number) => ({ kind: "manual", value }) as const;

/** A graph of the example of `docs/23` section 3.1: Main, then nodes 2 and 3 in a line, with a Linear mask in node 3. */
function graph(): DevelopSettings {
  const base = defaultSettings();
  const mountain = { ...newCorrector(2, [1, 0], "Mountain"), trim: { exposure_ev: 0.35, temperature: 0, tint: 0 } };
  const sky = {
    ...newCorrector(3, [2, 0], "Sky"),
    masks: {
      invert: false,
      items: [
        { kind: "linear", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.2], angle_deg: 4, feather: 0.25 },
      ] as MaskItem[],
    },
  };
  return {
    nodes: {
      items: [inputOf(base), mainOf(base), mountain, sky, outputOf(base)],
      links: [
        { from: { node: INPUT_ID, port: "picture" }, to: { node: MAIN_ID, port: "picture" } },
        { from: { node: MAIN_ID, port: "picture" }, to: { node: 2, port: "picture" } },
        { from: { node: 2, port: "picture" }, to: { node: 3, port: "picture" } },
        { from: { node: 3, port: "picture" }, to: { node: OUTPUT_ID, port: "picture" } },
      ],
    },
  };
}

describe("the selectors", () => {
  it("find the Input, Main and the Output of the default graph", () => {
    const s = defaultSettings("builtin:signature");
    expect(inputOf(s).kind).toBe("input");
    expect(inputOf(s).id).toBe("input");
    expect(mainOf(s).kind).toBe("corrector");
    expect(mainOf(s).id).toBe(1);
    expect(mainOf(s).name).toBe("Main");
    expect(outputOf(s).look).toBe("builtin:signature");
    expect(mainGradeOf(s)).toBe(mainOf(s).grade);
  });

  it("find them wherever the document lists them, and find Main among other correctors", () => {
    const s = graph();
    const shuffled: DevelopSettings = { nodes: { ...s.nodes, items: [...s.nodes.items].reverse() } };
    expect(inputOf(shuffled).id).toBe("input");
    expect(mainOf(shuffled).id).toBe(1);
    expect(outputOf(shuffled).id).toBe("output");
    expect(correctorsOf(shuffled).map((n) => n.id)).toEqual([3, 2, 1]);
    expect(nodeOf(s, 3)?.kind).toBe("corrector");
    expect(nodeOf(s, 9)).toBeUndefined();
  });

  it("say which node is missing when a document has no such node", () => {
    const noMain: DevelopSettings = { nodes: { items: [inputOf(defaultSettings()), outputOf(defaultSettings())], links: [] } };
    expect(() => mainOf(noMain)).toThrow("node 1, Main");
    expect(() => inputOf({ nodes: { items: [], links: [] } })).toThrow("input node");
  });

  it("start every corrector but Main at manual 0, because Auto belongs to Main", () => {
    expect(mainGradeOf(defaultSettings()).contrast).toEqual({ kind: "auto" });
    const node = newCorrector(4, [3, 0]);
    expect(node.grade.contrast).toEqual(manual(0));
    expect(node.grade.color_balance.shadows).toEqual({ kind: "manual", point: [0, 0] });
    expect(node.name).toBe("Corrector");
  });
});

describe("what the simple view cannot show (docs/23 §6)", () => {
  /** The default graph with Main changed. */
  const withMain = (change: Partial<ReturnType<typeof mainOf>>): DevelopSettings => {
    const base = defaultSettings();
    return { nodes: { ...base.nodes, items: base.nodes.items.map((item) => (item.kind === "corrector" && item.id === MAIN_ID ? { ...item, ...change } : item)) } };
  };

  it("is nothing for the default graph, and for a graph whose only change is in the Input, Main's grade or the Output", () => {
    expect(hiddenFromSimpleView(defaultSettings())).toBeNull();
    const grade = { ...mainOf(defaultSettings()).grade, contrast: manual(30), curve: { ...mainOf(defaultSettings()).grade.curve, luma: [[0, 0], [0.5, 0.6], [1, 1]] as Array<[number, number]> } };
    expect(hiddenFromSimpleView(withMain({ grade }))).toBeNull();
    const base = defaultSettings();
    const changed: DevelopSettings = {
      nodes: { ...base.nodes, items: base.nodes.items.map((item) => (item.kind === "output" ? { ...item, look: "builtin:signature", look_intensity: 40 } : item)) },
    };
    expect(hiddenFromSimpleView(changed)).toBeNull();
  });

  it("counts the correctors, Main included, when there is more than Main", () => {
    expect(hiddenFromSimpleView(graph())).toEqual({ kind: "nodes", count: 3 });
    const base = defaultSettings();
    const two: DevelopSettings = { nodes: { ...base.nodes, items: [inputOf(base), mainOf(base), newCorrector(2, [1, 0]), outputOf(base)] } };
    expect(hiddenFromSimpleView(two)).toEqual({ kind: "nodes", count: 2 });
    const four: DevelopSettings = {
      nodes: { ...base.nodes, items: [inputOf(base), mainOf(base), newCorrector(2, [1, 0]), newCorrector(3, [2, 0]), newCorrector(4, [3, 0]), outputOf(base)] },
    };
    expect(hiddenFromSimpleView(four)).toEqual({ kind: "nodes", count: 4 });
  });

  it("is Main when Main alone has a trim, a mask or a bypass", () => {
    const trim = (patch: Partial<ReturnType<typeof mainOf>["trim"]>) => withMain({ trim: { ...mainOf(defaultSettings()).trim, ...patch } });
    expect(hiddenFromSimpleView(trim({ exposure_ev: 0.3 }))).toEqual({ kind: "main" });
    expect(hiddenFromSimpleView(trim({ temperature: -10 }))).toEqual({ kind: "main" });
    expect(hiddenFromSimpleView(trim({ tint: 5 }))).toEqual({ kind: "main" });
    expect(hiddenFromSimpleView(withMain({ bypass: true }))).toEqual({ kind: "main" });
    const row: MaskItem = { kind: "linear", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], angle_deg: 0, feather: 0.2 };
    expect(hiddenFromSimpleView(withMain({ masks: { invert: false, items: [row] } }))).toEqual({ kind: "main" });
    // A trim of 0 is no trim, and an inverted empty mask list is no mask.
    expect(hiddenFromSimpleView(trim({ exposure_ev: 0 }))).toBeNull();
    expect(hiddenFromSimpleView(withMain({ masks: { invert: true, items: [] } }))).toBeNull();
  });

  it("says nodes before it says Main when both are so", () => {
    const base = graph();
    const trimmed: DevelopSettings = {
      nodes: { ...base.nodes, items: base.nodes.items.map((item) => (item.kind === "corrector" && item.id === MAIN_ID ? { ...item, bypass: true } : item)) },
    };
    expect(hiddenFromSimpleView(trimmed)).toEqual({ kind: "nodes", count: 3 });
  });

  it("is told by the photo, and the same Map comes back while the answer is the same (a slider move does not make the screen draw again)", () => {
    const rs = (settings: DevelopSettings) => ({ settings });
    const first = hiddenByPhoto({ 1: rs(defaultSettings()), 2: rs(graph()), 3: rs(withMain({ bypass: true })) });
    expect([...first]).toEqual([
      [2, { kind: "nodes", count: 3 }],
      [3, { kind: "main" }],
    ]);
    // Other documents, the same answer: the same Map.
    const again = hiddenByPhoto({ 1: rs(defaultSettings()), 2: rs(graph()), 3: rs(withMain({ bypass: true })) });
    expect(again).toBe(first);
    expect(hiddenByPhoto({ 1: rs(defaultSettings()) })).toEqual(new Map());
    expect(hiddenByPhoto({ 4: rs(defaultSettings()) })).toBe(hiddenByPhoto({}));
    // A different answer is a different Map.
    expect(hiddenByPhoto({ 2: rs(graph()) })).not.toBe(first);
  });
});

describe("the patch builders", () => {
  it("send one node's partial under its id", () => {
    expect(patchOf("input", { exposure: { kind: "manual", ev: 0.3 } })).toEqual({ nodes: { input: { exposure: { kind: "manual", ev: 0.3 } } } });
    expect(patchOf(1, { grade: { contrast: manual(12) } })).toEqual({ nodes: { "1": { grade: { contrast: manual(12) } } } });
    expect(patchOf("output", { look: "builtin:signature" })).toEqual({ nodes: { output: { look: "builtin:signature" } } });
    expect(patchOf(7, { name: "Peak" })).toEqual({ nodes: { "7": { name: "Peak" } } });
  });

  it("have one per node of the simple view", () => {
    expect(inputPatch({ light_source: "day" })).toEqual(patchOf("input", { light_source: "day" }));
    expect(mainPatch({ bypass: true })).toEqual(patchOf(1, { bypass: true }));
    expect(mainGradePatch({ brightness: 5 })).toEqual(patchOf(1, { grade: { brightness: 5 } }));
    expect(outputPatch({ look_intensity: 40 })).toEqual(patchOf("output", { look_intensity: 40 }));
  });

  it("know an empty patch", () => {
    expect(isEmptyPatch({})).toBe(true);
    expect(isEmptyPatch({ nodes: {} })).toBe(true);
    expect(isEmptyPatch({ remove: [] })).toBe(true);
    expect(isEmptyPatch(inputPatch({ orientation: 90 }))).toBe(false);
    expect(isEmptyPatch({ remove: [3] })).toBe(false);
    expect(isEmptyPatch({ links: [] })).toBe(false);
  });
});

describe("foldPatches", () => {
  it("merges two partials of one node field by field, the later field winning", () => {
    const folded = foldPatches(inputPatch({ exposure: { kind: "manual", ev: 1 }, light_source: "day" }), inputPatch({ exposure: { kind: "manual", ev: 2 } }));
    expect(folded).toEqual({ nodes: { input: { exposure: { kind: "manual", ev: 2 }, light_source: "day" } } });
  });

  it("merges a grade one field deeper, so a contrast and a brightness dragged in turn are both kept", () => {
    const folded = foldPatches(mainGradePatch({ contrast: manual(5) }), mainGradePatch({ brightness: 3, contrast: manual(9) }));
    expect(folded).toEqual({ nodes: { "1": { grade: { contrast: manual(9), brightness: 3 } } } });
  });

  it("keeps every node a patch names", () => {
    const folded = foldPatches(inputPatch({ orientation: 90 }), foldPatches(mainGradePatch({ white: 4 }), outputPatch({ look_intensity: 50 })));
    expect(Object.keys(folded.nodes ?? {}).sort()).toEqual(["1", "input", "output"]);
  });

  it("takes the union of what is removed and the later links", () => {
    const links = defaultLinks();
    const folded = foldPatches({ remove: [3], links: [] }, { remove: [3, 4], links });
    expect(folded.remove).toEqual([3, 4]);
    expect(folded.links).toBe(links);
    expect(foldPatches({ links }, { remove: [5] }).links).toBe(links);
  });

  it("changes the document as the two patches do in turn", () => {
    const a = foldPatches(mainGradePatch({ contrast: manual(5), brightness: 1 }), inputPatch({ lens: { flares: 20 } }));
    const b = mainGradePatch({ brightness: 8, white: -3 });
    const base = defaultSettings();
    expect(applyPatch(base, foldPatches(a, b))).toEqual(applyPatch(applyPatch(base, a), b));
  });

  it("folds mask edits: a row edited twice is one edit, appended strokes are all appended, new rows drop the edits made before them", () => {
    const stroke = (x: number): BrushStroke => ({ size: 0.03, hardness: 0.5, erase: false, points: [[x, x]] });
    const edit = (rows: Record<string, object>): SettingsPatch => patchOf(2, { masks: { edit: rows } });
    const one = foldPatches(edit({ "0": { feather: 0.1 } }), edit({ "0": { grow: 0.2 }, "1": { opacity: 0.5 } }));
    expect(one.nodes?.["2"]?.masks?.edit).toEqual({ "0": { feather: 0.1, grow: 0.2 }, "1": { opacity: 0.5 } });
    const strokes = foldPatches(edit({ "0": { strokes_append: [stroke(0.1)] } }), edit({ "0": { strokes_append: [stroke(0.2)] } }));
    expect(strokes.nodes?.["2"]?.masks?.edit?.["0"]?.strokes_append).toEqual([stroke(0.1), stroke(0.2)]);
    const rows = foldPatches(edit({ "0": { feather: 0.1 } }), patchOf(2, { masks: { items: [] } }));
    expect(rows.nodes?.["2"]?.masks).toEqual({ items: [] });
  });
});

describe("applyPatch", () => {
  it("changes a node's fields and leaves the rest of the document alone", () => {
    const base = defaultSettings();
    const next = applyPatch(base, inputPatch({ light_source: "tungsten", exposure: { kind: "manual", ev: 0.5 } }));
    expect(inputOf(next).light_source).toBe("tungsten");
    expect(inputOf(next).exposure).toEqual({ kind: "manual", ev: 0.5 });
    expect(mainOf(next)).toBe(mainOf(base));
    expect(outputOf(next)).toBe(outputOf(base));
    expect(inputOf(base).light_source).toBe("auto");
  });

  it("merges the parts of rolloff, decode, lens and trim, and of a grade's wheels and curves", () => {
    const base = defaultSettings();
    let next = applyPatch(base, inputPatch({ rolloff: { highlights: 0.9 }, decode: { neural_demosaic: false }, lens: { flares: 30 } }));
    expect(inputOf(next).rolloff).toEqual({ highlights: 0.9, low_light: 0.5 });
    expect(inputOf(next).decode).toEqual({ neural_demosaic: false, highlight_recovery: true, lens_corrections: true, chromatic_aberration: true });
    expect(inputOf(next).lens).toEqual({ kind: "default", flares: 30, edges: 0, anamorphic: 0 });
    next = applyPatch(next, mainPatch({ trim: { tint: 12 }, grade: { color_balance: { midtones: [0.1, 0] }, curve: { red: [[0, 0.1], [1, 1]] } } }));
    expect(mainOf(next).trim).toEqual({ exposure_ev: 0, temperature: 0, tint: 12 });
    expect(mainGradeOf(next).color_balance).toEqual({ shadows: { kind: "auto" }, midtones: [0.1, 0], highlights: { kind: "auto" } });
    expect(mainGradeOf(next).curve.red).toEqual([[0, 0.1], [1, 1]]);
    expect(mainGradeOf(next).curve.luma).toEqual([[0, 0], [1, 1]]);
  });

  it("clears a crop and a pick with null", () => {
    const crop = { rect: { x: 0, y: 0, width: 0.5, height: 0.5 }, angle_deg: 0, aspect: "free" as const };
    const cropped = applyPatch(defaultSettings(), inputPatch({ crop, white_pick: [0.2, 0.3] }));
    expect(inputOf(cropped).crop).toEqual(crop);
    const clear = applyPatch(cropped, inputPatch({ crop: null, white_pick: null }));
    expect([inputOf(clear).crop, inputOf(clear).white_pick]).toEqual([null, null]);
  });

  it("adds a node that gives its kind, as a corrector at manual 0, and none that does not", () => {
    const base = defaultSettings();
    const added = applyPatch(base, { nodes: { "2": { kind: "corrector", name: "Warmth", trim: { temperature: 30 } } } });
    const node = nodeOf(added, 2);
    expect(node?.kind === "corrector" && node.name).toBe("Warmth");
    expect(node?.kind === "corrector" && node.trim).toEqual({ exposure_ev: 0, temperature: 30, tint: 0 });
    expect(node?.kind === "corrector" && node.grade.contrast).toEqual(manual(0));
    expect(node?.pos).toEqual([1, 0]);
    expect(applyPatch(base, { nodes: { "2": { name: "Nothing" } } })).toEqual(base);
  });

  it("removes a node with the links that touch it, and replaces the links whole when it is given them", () => {
    const s = graph();
    const gone = applyPatch(s, { remove: [2] });
    expect(nodeOf(gone, 2)).toBeUndefined();
    expect(gone.nodes.links.map((l) => [l.from.node, l.to.node])).toEqual([
      [INPUT_ID, MAIN_ID],
      [3, OUTPUT_ID],
    ]);
    const relinked = applyPatch(s, { remove: [2], links: [...gone.nodes.links, { from: { node: MAIN_ID, port: "picture" }, to: { node: 3, port: "picture" } }] });
    expect(relinked.nodes.links).toHaveLength(3);
    expect(applyPatch(s, { links: [] }).nodes.links).toEqual([]);
  });

  it("writes the nodes in the core's order: the Input, the correctors by id, the Output", () => {
    const s = applyPatch(graph(), { nodes: { "9": { kind: "corrector" }, "5": { kind: "corrector" } } });
    expect(s.nodes.items.map((n) => n.id)).toEqual(["input", 1, 2, 3, 5, 9, "output"]);
    expect(inCanonicalOrder([outputOf(s), nodeOf(s, 3) as never, inputOf(s)]).map((n) => n.id)).toEqual(["input", 3, "output"]);
  });

  it("edits mask rows by index and keeps their kind", () => {
    const s = graph();
    const next = applyPatch(s, patchOf(3, { masks: { invert: true, edit: { "0": { feather: 0.6, kind: "radial" } } } }));
    const node = nodeOf(next, 3);
    expect(node?.kind === "corrector" && node.masks.invert).toBe(true);
    expect(node?.kind === "corrector" && node.masks.items[0]).toMatchObject({ kind: "linear", feather: 0.6, angle_deg: 4 });
  });

  it("appends a stroke and keeps the first strokes again (an append and its inverse restore the strokes exactly)", () => {
    const strokes: BrushStroke[] = [
      { size: 0.03, hardness: 0.5, erase: false, points: [[0.1, 0.1], [0.2, 0.2]] },
      { size: 0.05, hardness: 0.2, erase: true, points: [[0.5, 0.5]] },
    ];
    const brush: MaskItem = { kind: "brush", mode: "add", invert: false, enabled: true, opacity: 1, strokes: [strokes[0] as BrushStroke], feather: 0, grow: 0, refine: 0 };
    const withBrush = applyPatch(graph(), patchOf(3, { masks: { items: [brush] } }));
    const appended = applyPatch(withBrush, patchOf(3, { masks: { edit: { "0": { strokes_append: [strokes[1] as BrushStroke] } } } }));
    const rowOf = (s: DevelopSettings) => (nodeOf(s, 3) as { masks: { items: Array<{ strokes: BrushStroke[] }> } }).masks.items[0]?.strokes;
    expect(rowOf(appended)).toEqual(strokes);
    const undone = applyPatch(appended, patchOf(3, { masks: { edit: { "0": { strokes_keep: 1 } } } }));
    expect(rowOf(undone)).toEqual([strokes[0]]);
    expect(undone).toEqual(withBrush);
  });
});

describe("diffGraph", () => {
  it("is null when nothing changed", () => {
    expect(diffGraph(graph(), graph())).toBeNull();
  });

  it("holds only the nodes a change touched, and only their fields that differ", () => {
    const before = graph();
    const after = applyPatch(before, patchOf(2, { trim: { exposure_ev: 0.9 }, name: "Peak" }));
    expect(diffGraph(before, after)).toEqual({
      before: { nodes: { "2": { trim: { exposure_ev: 0.35 }, name: "Mountain" } } },
      after: { nodes: { "2": { trim: { exposure_ev: 0.9 }, name: "Peak" } } },
    });
  });

  it("goes one field into a grade, but takes other fields whole", () => {
    const before = defaultSettings();
    const after = applyPatch(before, foldPatches(mainGradePatch({ contrast: manual(12) }), inputPatch({ lens: { kind: "classic", flares: 35, edges: 30, anamorphic: 0 } })));
    expect(diffGraph(before, after)).toEqual({
      before: { nodes: { input: { lens: { kind: "default", flares: 0, edges: 0, anamorphic: 0 } }, "1": { grade: { contrast: { kind: "auto" } } } } },
      after: { nodes: { input: { lens: { kind: "classic", flares: 35, edges: 30, anamorphic: 0 } }, "1": { grade: { contrast: manual(12) } } } },
    });
  });

  it("adds a node whole with its kind and takes it out on the way back, and the links with it", () => {
    const before = defaultSettings();
    const added = applyPatch(before, {
      nodes: { "2": { kind: "corrector", name: "Warmth", pos: [1, 1] } },
      links: [
        { from: { node: INPUT_ID, port: "picture" }, to: { node: MAIN_ID, port: "picture" } },
        { from: { node: MAIN_ID, port: "picture" }, to: { node: 2, port: "picture" } },
        { from: { node: 2, port: "picture" }, to: { node: OUTPUT_ID, port: "picture" } },
      ],
    });
    const entry = diffGraph(before, added);
    expect(entry?.after.nodes?.["2"]).toMatchObject({ kind: "corrector", name: "Warmth", pos: [1, 1] });
    expect(entry?.before.remove).toEqual([2]);
    expect(entry?.before.links).toEqual(before.nodes.links);
    expect(entry?.after.links).toEqual(added.nodes.links);
    expect(applyPatch(added, entry?.before ?? {})).toEqual(before);
    expect(applyPatch(before, entry?.after ?? {})).toEqual(added);
  });

  it("brings a removed node back whole", () => {
    const before = graph();
    const after = applyPatch(before, { remove: [3], links: [{ from: { node: 2, port: "picture" }, to: { node: OUTPUT_ID, port: "picture" } }, ...before.nodes.links.slice(0, 2)] });
    const entry = diffGraph(before, after);
    expect(entry?.after.remove).toEqual([3]);
    expect(entry?.before.nodes?.["3"]).toEqual(nodeAsPatch(nodeOf(before, 3) as never));
    expect(applyPatch(after, entry?.before ?? {})).toEqual(before);
  });

  it("gives the links whole when they changed, and moving a card is a step", () => {
    const before = graph();
    const moved = applyPatch(before, patchOf(2, { pos: [1, 2] }));
    expect(diffGraph(before, moved)).toEqual({ before: { nodes: { "2": { pos: [1, 0] } } }, after: { nodes: { "2": { pos: [1, 2] } } } });
    const relinked = applyPatch(before, { links: before.nodes.links.slice(0, 3) });
    expect(diffGraph(before, relinked)?.after.links).toHaveLength(3);
    expect(diffGraph(before, relinked)?.before.nodes).toBeUndefined();
  });

  it("round-trips: undo restores what was there and redo makes the change again, for every kind of change", () => {
    const start = graph();
    const changes: SettingsPatch[] = [
      inputPatch({ exposure: { kind: "manual", ev: 1.2 }, crop: { rect: { x: 0, y: 0, width: 0.5, height: 0.5 }, angle_deg: 3, aspect: "free" } }),
      mainGradePatch({ contrast: manual(-20), color_warp: { pins: [] }, color_balance: { midtones: [0.1, 0.1] } }),
      outputPatch({ look: "builtin:signature", look_intensity: 33 }),
      patchOf(3, { masks: { invert: true, edit: { "0": { center: [0.1, 0.9], feather: 0.5 } } } }),
      patchOf(2, { bypass: true, grade: { white: 12 } }),
      { remove: [2], links: [start.nodes.links[0]!, { from: { node: MAIN_ID, port: "picture" }, to: { node: 3, port: "picture" } }, start.nodes.links[3]!] },
    ];
    for (const change of changes) {
      const after = applyPatch(start, change);
      const entry = diffGraph(start, after);
      expect(entry, JSON.stringify(change)).not.toBeNull();
      expect(applyPatch(after, entry?.before ?? {}), JSON.stringify(change)).toEqual(start);
      expect(applyPatch(start, entry?.after ?? {}), JSON.stringify(change)).toEqual(after);
    }
  });

  it("tells a drawn stroke from a replaced list: a stroke is an append and its inverse a keep", () => {
    const stroke = (n: number): BrushStroke => ({ size: 0.03, hardness: 0.5, erase: false, points: [[n / 10, n / 10]] });
    const row = (strokes: BrushStroke[]): MaskItem => ({ kind: "brush", mode: "add", invert: false, enabled: true, opacity: 1, strokes, feather: 0, grow: 0, refine: 0 });
    const start = applyPatch(graph(), patchOf(3, { masks: { items: [row([stroke(1)])] } }));
    const drawn = applyPatch(start, patchOf(3, { masks: { items: [row([stroke(1), stroke(2)])] } }));
    expect(diffGraph(start, drawn)).toEqual({
      before: { nodes: { "3": { masks: { edit: { "0": { strokes_keep: 1 } } } } } },
      after: { nodes: { "3": { masks: { edit: { "0": { strokes_append: [stroke(2)] } } } } } },
    });
    const erased = applyPatch(start, patchOf(3, { masks: { items: [row([stroke(9)])] } }));
    expect(diffGraph(start, erased)?.after.nodes?.["3"]?.masks?.edit?.["0"]).toEqual({ strokes: [stroke(9)] });
    const dropped = applyPatch(start, patchOf(3, { masks: { items: [] } }));
    expect(diffGraph(start, dropped)?.after.nodes?.["3"]?.masks?.items).toEqual([]);
  });

  it("keeps 100 steps over a graph with a 100,000-point brush within 128 KiB, where the strokes alone are over a megabyte", () => {
    const points = (n: number, from: number): Array<[number, number]> => Array.from({ length: n }, (_, i) => [((from + i) % 1000) / 1000, 0.5]);
    // 100 strokes of 1,000 points: 100,000 points, the sanity bound (`docs/23` section 3.5).
    const strokes: BrushStroke[] = Array.from({ length: 100 }, (_, i) => ({ size: 0.03, hardness: 0.5, erase: false, points: points(1000, i * 7) }));
    const brush = (list: BrushStroke[]): MaskItem => ({ kind: "brush", mode: "add", invert: false, enabled: true, opacity: 1, strokes: list, feather: 0, grow: 0, refine: 0 });
    // A graph with 99,000 points, then 100 strokes of 10 points each drawn one after the other.
    let doc = applyPatch(graph(), patchOf(3, { masks: { items: [brush(strokes.slice(0, 99))] } }));
    expect(JSON.stringify(doc).length).toBeGreaterThan(1_000_000);
    const history = new PhotoHistory();
    const entries: string[] = [];
    for (let i = 0; i < 100; i++) {
      const row = (nodeOf(doc, 3) as { masks: { items: Array<{ strokes: BrushStroke[] }> } }).masks.items[0] as { strokes: BrushStroke[] };
      const next = applyPatch(doc, patchOf(3, { masks: { items: [brush([...row.strokes, { size: 0.02, hardness: 0.5, erase: false, points: points(10, i) }])] } }));
      const entry = diffGraph(doc, next);
      expect(entry?.after.nodes?.["3"]?.masks?.edit?.["0"]?.strokes_append).toHaveLength(1);
      history.push(entry!);
      entries.push(JSON.stringify(entry));
      doc = next;
    }
    expect(history.undoDepth).toBe(100);
    const total = entries.reduce((n, e) => n + e.length, 0);
    expect(Math.max(...entries.map((e) => e.length))).toBeLessThan(2048);
    expect(total).toBeLessThan(128 * 1024);
  });
});

describe("withoutLook and same", () => {
  it("drops the Output's look and intensity and nothing else", () => {
    const a = applyPatch(defaultSettings(), outputPatch({ look: "builtin:signature", look_intensity: 20 }));
    expect(same(withoutLook(a), withoutLook(defaultSettings()))).toBe(true);
    const b = applyPatch(a, outputPatch({ output_sharpening: 50 }));
    expect(same(withoutLook(a), withoutLook(b))).toBe(false);
    expect(JSON.stringify(withoutLook(a))).not.toContain("builtin:signature");
  });

  it("compares by structure", () => {
    expect(same({ a: [1, { b: 2 }] }, { a: [1, { b: 2 }] })).toBe(true);
    expect(same({ a: 1 }, { a: 1, b: undefined })).toBe(false);
    expect(same([1, 2], [1, 2, 3])).toBe(false);
    expect(same(null, undefined)).toBe(false);
  });
});
