/**
 * The graph on the mock API (`docs/23` §3 and §9): node patches, the core's refusals with the core's words, Auto in a corrector other
 * than Main, nodes added and removed, copy and paste of the whole graph, and the settings a photo opens with.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "./api.mock";
import type { MockApi } from "./api.mock";
import type { AppError, MaskItem, NodeLink, SettingsPatch } from "./api/types";
import { applyPatch, correctorsOf, defaultSettings, inputOf, mainGradeOf, mainOf, nodeOf, outputOf } from "./state/settingsGraph";

const ARW = ["~/Pictures/Lisbon/DSC04201.ARW", "~/Pictures/Lisbon/DSC04202.ARW", "~/Pictures/Lisbon/DSC04203.ARW"];
const manual = (value: number) => ({ kind: "manual", value }) as const;
const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

/** Node 2 after Main: the patch that adds it with the links that put it in the line. */
const addNode2: SettingsPatch = {
  nodes: { "2": { kind: "corrector", name: "Warmth", trim: { temperature: 30 }, pos: [1, 0] } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
};

let api: MockApi;
beforeEach(async () => {
  vi.useFakeTimers();
  api = createMockApi();
  await api.openPaths(ARW);
  api.flush();
});
afterEach(() => {
  api.reset();
  vi.useRealTimers();
});

const rejection = async (promise: Promise<unknown>): Promise<AppError> => {
  try {
    await promise;
  } catch (error) {
    return error as AppError;
  }
  throw new Error("expected a rejection");
};

describe("the settings of a photo", () => {
  it("are the default graph: the Input, Main and the Output, linked Input → Main → Output", async () => {
    const rs = await api.getSettings(1);
    expect(rs.settings.nodes.items.map((n) => [n.kind, n.id])).toEqual([["input", "input"], ["corrector", 1], ["output", "output"]]);
    expect(rs.settings.nodes.links).toEqual([picture("input", 1), picture(1, "output")]);
    expect(rs.settings).toEqual(defaultSettings());
    expect(mainOf(rs.settings).name).toBe("Main");
    expect(rs.values.nodes).toEqual([]);
  });

  it("open with a node patch of their own (initialSettings), nodes and links included", async () => {
    api.reset({ initialSettings: { DSC04202: { ...addNode2, nodes: { ...addNode2.nodes, output: { look: "builtin:signature" } } } } });
    await api.openPaths(ARW);
    api.flush();
    const rs = await api.getSettings(2);
    expect(correctorsOf(rs.settings).map((n) => n.id)).toEqual([1, 2]);
    expect(outputOf(rs.settings).look).toBe("builtin:signature");
    expect(rs.values.nodes.map((n) => [n.id, n.name])).toEqual([[2, "Warmth"]]);
    expect((await api.getSettings(1)).settings).toEqual(defaultSettings());
  });
});

describe("a node patch", () => {
  it("changes the fields of the node it names and no other, and clamps them", async () => {
    const rs = await api.updateSettings(1, { nodes: { "1": { grade: { contrast: manual(500), brightness: 10 }, trim: { tint: 400 } }, input: { orientation: 90 } } });
    expect(mainGradeOf(rs.settings).contrast).toEqual(manual(100));
    expect(mainGradeOf(rs.settings).brightness).toBe(10);
    expect(mainOf(rs.settings).trim).toEqual({ exposure_ev: 0, temperature: 0, tint: 100 });
    expect(inputOf(rs.settings).orientation).toBe(90);
    expect(outputOf(rs.settings)).toEqual(outputOf(defaultSettings()));
    expect(rs.values.main.grade.contrast).toBe(100);
    expect(rs.values.main.trim.tint).toBe(100);
  });

  it("adds a node that gives its kind, and the resolved values list it", async () => {
    const rs = await api.updateSettings(1, addNode2);
    expect(correctorsOf(rs.settings).map((n) => n.id)).toEqual([1, 2]);
    expect(rs.settings.nodes.links).toEqual(addNode2.links);
    expect(rs.values.nodes).toHaveLength(1);
    expect(rs.values.nodes[0]).toMatchObject({ id: 2, name: "Warmth", trim: { exposure_ev: 0, temperature: 30, tint: 0 } });
    // A new node is manual 0 everywhere: it has no Auto.
    expect(rs.values.nodes[0]?.grade).toMatchObject({ contrast: 0, saturation: 0, highlights: 0 });
    expect(rs.auto.main.contrast).toBe(true);
  });

  it("reads Auto in a corrector other than Main as a manual 0, and keeps Main's Auto", async () => {
    await api.updateSettings(1, addNode2);
    const rs = await api.updateSettings(1, {
      nodes: { "2": { grade: { contrast: { kind: "auto" }, color_balance: { shadows: { kind: "auto" } } } }, "1": { grade: { saturation: { kind: "auto" } } } },
    });
    const node = nodeOf(rs.settings, 2);
    expect(node?.kind === "corrector" && node.grade.contrast).toEqual(manual(0));
    expect(node?.kind === "corrector" && node.grade.color_balance.shadows).toEqual({ kind: "manual", point: [0, 0] });
    expect(mainGradeOf(rs.settings).saturation).toEqual({ kind: "auto" });
    expect(rs.auto.main.saturation).toBe(true);
  });

  it("removes a node with the links that touch it", async () => {
    await api.updateSettings(1, addNode2);
    const rs = await api.updateSettings(1, { remove: [2], links: [picture("input", 1), picture(1, "output")] });
    expect(correctorsOf(rs.settings).map((n) => n.id)).toEqual([1]);
    expect(rs.settings.nodes.links).toEqual(defaultSettings().nodes.links);
    expect(rs.values.nodes).toEqual([]);
  });

  it("applies a row patch to a mask, and a Brush stroke as an append", async () => {
    const brush: MaskItem = { kind: "brush", mode: "add", invert: false, enabled: true, opacity: 1, strokes: [], feather: 0, grow: 0, refine: 0 };
    await api.updateSettings(1, { nodes: { "1": { masks: { items: [brush] } } } });
    const stroke = { size: 0.03, hardness: 0.5, erase: false, points: [[0.1, 0.1], [0.2, 0.2]] as Array<[number, number]> };
    const drawn = await api.updateSettings(1, { nodes: { "1": { masks: { edit: { "0": { strokes_append: [stroke] } } } } } });
    expect(mainOf(drawn.settings).masks.items[0]).toMatchObject({ kind: "brush", strokes: [stroke] });
    const undone = await api.updateSettings(1, { nodes: { "1": { masks: { edit: { "0": { strokes_keep: 0 } } } } } });
    expect(mainOf(undone.settings).masks.items[0]).toMatchObject({ kind: "brush", strokes: [] });
  });
});

describe("a patch the core would refuse", () => {
  const cases: Array<[string, SettingsPatch, string]> = [
    ["a node the photo does not have", { nodes: { "7": { name: "Peak" } } }, "The patch names node 7, which this photo does not have."],
    ["a kind other than the node's", { nodes: { "1": { kind: "input" } } }, "Node 1 is a corrector, not an input."],
    ["a field the kind does not have", { nodes: { "1": { exposure: { kind: "auto" } } } }, "A corrector has no field called “exposure”."],
    ["a field of the Input on the Output", { nodes: { output: { lens: { flares: 3 } } } }, "An output has no field called “lens”."],
    ["removing Main", { remove: [1] }, "The input, the output and node 1, Main, can't be removed."],
    ["removing the Output", { remove: ["output"] }, "The input, the output and node 1, Main, can't be removed."],
    ["links in a circle", { nodes: { "2": { kind: "corrector" }, "3": { kind: "corrector" } }, links: [picture("input", 1), picture(1, "output"), picture(2, 3), picture(3, 2)] }, "The nodes of this photo are linked in a circle. Remove a link."],
    ["a second link on an input", { links: [picture("input", 1), picture(1, "output"), picture("input", "output")] }, "An input can have one link. The output has two on its picture input."],
    ["Main taking its picture from elsewhere", { nodes: { "2": { kind: "corrector" } }, links: [picture("input", 2), picture(2, 1), picture(1, "output")] }, "Node 1, Main, must take its picture from the input."],
    ["no link to the Output", { links: [picture("input", 1)] }, "The output needs one picture link."],
  ];

  it.each(cases)("%s is refused in the core's words, with the render scope", async (_what, patch, text) => {
    const error = await rejection(api.updateSettings(1, patch));
    expect(error).toEqual({ scope: "render", message: text });
  });

  it("changes nothing when any part of a patch is refused (a patch is atomic)", async () => {
    const before = await api.getSettings(1);
    const stats: unknown[] = [];
    api.on("render_stats", (e) => stats.push(e));
    await rejection(api.updateSettings(1, { nodes: { "1": { grade: { contrast: manual(40) } }, "9": { name: "Nobody" } } }));
    await rejection(api.updateSettings(1, { nodes: { input: { orientation: 90 } }, remove: [1] }));
    await vi.advanceTimersByTimeAsync(1);
    expect(await api.getSettings(1)).toEqual(before);
    expect(stats).toEqual([]);
  });
});

describe("copy and paste of the whole graph", () => {
  it("replace the target's graph in one step that undo can take back", async () => {
    // Photo 1: node 2 with a trim, Main's contrast, the Signature look. Photo 2: a node 5 of its own and a crop.
    await api.updateSettings(1, applyPatchWith(addNode2, { nodes: { "1": { grade: { contrast: manual(20) } }, output: { look: "builtin:signature" } } }));
    await api.updateSettings(2, {
      nodes: { "5": { kind: "corrector", name: "Own", pos: [1, 1] }, input: { crop: { rect: { x: 0, y: 0, width: 0.5, height: 0.5 }, angle_deg: 0, aspect: "free" } } },
      links: [picture("input", 1), picture(1, 5), picture(5, "output")],
    });
    const copied = await api.copySettings(1);
    expect(Object.keys(copied.nodes ?? {}).sort()).toEqual(["1", "2", "input", "output"]);
    const before = await api.getSettings(2);

    const report = await api.pasteSettings([2], copied);
    expect(report.pasted).toHaveLength(1);
    const after = await api.getSettings(2);
    // The graph is the source's: node 2 is there, node 5 is gone, the links are the source's, the crop is the target's own.
    expect(correctorsOf(after.settings).map((n) => n.id)).toEqual([1, 2]);
    expect(after.settings.nodes.links).toEqual(addNode2.links);
    expect(mainGradeOf(after.settings).contrast).toEqual(manual(20));
    expect(outputOf(after.settings).look).toBe("builtin:signature");
    expect(inputOf(after.settings).crop?.rect.width).toBe(0.5);
    // The step's two patches hold only what changed, and undo takes the paste back whole.
    const step = report.pasted[0]!;
    expect(step.after.nodes?.["5"]).toBeUndefined();
    expect(step.after.remove).toEqual([5]);
    expect(step.before.remove).toEqual([2]);
    expect(step.before.nodes?.["5"]).toMatchObject({ kind: "corrector", name: "Own" });
    await api.updateSettings(2, step.before);
    expect((await api.getSettings(2)).settings).toEqual(before.settings);
    await api.updateSettings(2, step.after);
    expect((await api.getSettings(2)).settings).toEqual(after.settings);
  });

  it("leaves the target untouched when the copy is of the same graph (an empty step)", async () => {
    const copied = await api.copySettings(1);
    const report = await api.pasteSettings([2], copied);
    expect(report.pasted).toEqual([{ id: 2, before: {}, after: {} }]);
  });
});

/** A patch of the first, then the second: what the first `updateSettings` of a test sends. */
function applyPatchWith(first: SettingsPatch, second: SettingsPatch): SettingsPatch {
  return { ...first, nodes: { ...first.nodes, ...Object.fromEntries(Object.entries(second.nodes ?? {}).map(([k, v]) => [k, { ...first.nodes?.[k], ...v }])) } };
}

describe("applyPatch agrees with the mock", () => {
  it("makes the settings the mock makes of the same patch (the page's mirror of the core's apply)", async () => {
    const mirrored = applyPatch(defaultSettings(), addNode2);
    const rs = await api.updateSettings(1, addNode2);
    // The mock clamps and reads Auto as the core does; the pure mirror checks nothing: for a patch in range they are one document.
    expect(rs.settings.nodes.items.map((n) => n.id)).toEqual(mirrored.nodes.items.map((n) => n.id));
    expect(rs.settings.nodes.links).toEqual(mirrored.nodes.links);
    expect(nodeOf(rs.settings, 2)).toEqual(nodeOf(mirrored, 2));
  });
});
