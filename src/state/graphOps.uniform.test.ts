/**
 * The Colour Uniform node's graph operations (`docs/32` sections 2.1, 2.3, 4 and 5.3): Add node, Reset, the head's line and the one patch a pick sends.
 */
import { describe, expect, it } from "vitest";
import type { DevelopSettings, NodeLink, SettingsPatch } from "../api/types";
import { validateGraph } from "../fixtures/graph";
import { headLine, insertAfter, pickUniform, resetNode } from "./graphOps";
import type { GraphOutcome } from "./graphOps";
import { NEW_COLOR_RANGE_WIDTH, widthSlider } from "./maskRows";
import { applyPatch, defaultSettings, diffGraph, nodeOf } from "./settingsGraph";
import type { ColorUniformItem } from "./settingsGraph";

const pic = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const sent = (o: GraphOutcome): SettingsPatch => {
  if (!o.ok) throw new Error(`refused: ${o.reason}`);
  return o.patch;
};
const uniformOf = (s: DevelopSettings, id = 2): ColorUniformItem => nodeOf(s, id) as ColorUniformItem;

/** Main, then a Colour Uniform node (2), then the Output. */
function withUniform(extra: Record<string, unknown> = {}, links: NodeLink[] = [pic("input", 1), pic(1, 2), pic(2, "output")]): DevelopSettings {
  return applyPatch(defaultSettings(), { nodes: { "2": { kind: "color_uniform", name: "Uniform", pos: [1, 0], ...extra } }, links });
}

/** An Orange: hue about 55 degrees at chroma 0.1, a little above middle grey. */
const ORANGE = { a: 0.06, b: 0.08, tone: 0.35 };

describe("Add node > Colour Uniform", () => {
  it("adds a node named Uniform after the selected one, with no target, Compress at 0 and no masks, and it is in the graph the core accepts", () => {
    const before = defaultSettings();
    const outcome = insertAfter(before, 1, { kind: "color_uniform" });
    const after = applyPatch(before, sent(outcome));
    expect(validateGraph(after)).toBeNull();
    const added = uniformOf(after);
    expect(added.kind).toBe("color_uniform");
    expect(added.name).toBe("Uniform");
    expect(added.target).toBeNull();
    expect(added.compress).toEqual({ hue: 0, saturation: 0, lightness: 0 });
    expect(added.masks.items).toEqual([]);
    expect(outcome.ok && outcome.select).toBe(2);
    expect(after.nodes.links.some((l) => l.from.node === 1 && l.to.node === 2)).toBe(true);
    expect(after.nodes.links.some((l) => l.from.node === 2 && l.to.node === "output")).toBe(true);
  });

  it("undoes as one step", () => {
    const before = defaultSettings();
    const after = applyPatch(before, sent(insertAfter(before, 1, { kind: "color_uniform" })));
    const diff = diffGraph(before, after);
    expect(diff).not.toBeNull();
    expect(nodeOf(applyPatch(after, diff!.before), 2)).toBeUndefined();
  });
});

describe("Reset on a Colour Uniform node", () => {
  it("clears the target and the amounts and keeps the masks, the name, the bypass and the links", () => {
    const mask = { kind: "color_range", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.05, 0.05] };
    const before = withUniform({ name: "Skin", bypass: true, target: { hue_deg: 40, saturation: 10, lightness_ev: 0.5 }, compress: { hue: 70, saturation: 60, lightness: 20 }, masks: { invert: false, items: [mask] } });
    const after = applyPatch(before, sent(resetNode(before, 2)));
    const node = uniformOf(after);
    expect(node.target).toBeNull();
    expect(node.compress).toEqual({ hue: 0, saturation: 0, lightness: 0 });
    expect(node.masks.items).toHaveLength(1);
    expect(node.name).toBe("Skin");
    expect(node.bypass).toBe(true);
    expect(after.nodes.links).toEqual(before.nodes.links);
  });

  it("answers an empty patch on a node that is already at its defaults", () => {
    const outcome = resetNode(withUniform(), 2);
    expect(outcome.ok && Object.keys(outcome.patch.nodes ?? {})).toEqual([]);
  });
});

describe("the head's line of a Colour Uniform node", () => {
  it("reads Colour Uniform, Bypassed, or Not connected", () => {
    expect(headLine(withUniform(), 2)).toBe("Colour Uniform");
    expect(headLine(withUniform({ bypass: true }), 2)).toBe("Bypassed");
    expect(headLine(withUniform({}, [pic("input", 1), pic(1, "output")]), 2)).toBe("Not connected");
  });
});

describe("pickUniform (docs/32 sections 2.1 and 2.3)", () => {
  it("is null on a node that is not a Colour Uniform node", () => {
    expect(pickUniform(defaultSettings(), 1, ORANGE)).toBeNull();
  });

  it("sends one patch for the first pick: the target, Compress 50 / 50 / 0 and one Colour range row centred on the chroma", () => {
    const before = withUniform();
    const pick = pickUniform(before, 2, ORANGE)!;
    expect(pick.first).toBe(true);
    expect(pick.tooGrey).toBe(false);
    expect(Object.keys(pick.patch.nodes ?? {})).toEqual(["2"]);
    const node = uniformOf(applyPatch(before, pick.patch));
    expect(node.target).not.toBeNull();
    expect(node.target!.hue_deg).toBeCloseTo((Math.atan2(0.08, 0.06) * 180) / Math.PI, 6);
    expect(node.target!.saturation).toBeCloseTo(10, 6);
    expect(node.compress).toEqual({ hue: 50, saturation: 50, lightness: 0 });
    expect(node.masks.items).toHaveLength(1);
    const row = node.masks.items[0] as unknown as { kind: string; center: [number, number]; width: number; softness: number };
    expect(row.kind).toBe("color_range");
    // Narrower than the Colour range's own 0.05 (the slider reads 23, not 39), with its default softness (docs/32 §2.3).
    expect(row.width).toBe(NEW_COLOR_RANGE_WIDTH);
    expect(NEW_COLOR_RANGE_WIDTH).toBe(0.02);
    expect(widthSlider(row.width)).toBe(23);
    expect(row.softness).toBe(0.5);
    expect(row.center[0]).toBeCloseTo(0.06, 9);
    expect(row.center[1]).toBeCloseTo(0.08, 9);
  });

  it("is one undo step", () => {
    const before = withUniform();
    const after = applyPatch(before, pickUniform(before, 2, ORANGE)!.patch);
    const diff = diffGraph(before, after);
    const undone = applyPatch(after, diff!.before);
    expect(uniformOf(undone).target).toBeNull();
    expect(uniformOf(undone).masks.items).toEqual([]);
    expect(uniformOf(undone).compress).toEqual({ hue: 0, saturation: 0, lightness: 0 });
  });

  it("changes the target only on a later pick", () => {
    const first = applyPatch(withUniform(), pickUniform(withUniform(), 2, ORANGE)!.patch);
    const pick = pickUniform(first, 2, { a: -0.05, b: 0.02, tone: 0.2 })!;
    expect(pick.first).toBe(false);
    expect(Object.keys(pick.patch.nodes!["2"] ?? {})).toEqual(["target"]);
  });

  it("keeps amounts the person already set on a first pick", () => {
    const before = withUniform({ compress: { hue: 0, saturation: 30, lightness: 0 } });
    const pick = pickUniform(before, 2, ORANGE)!;
    expect(pick.patch.nodes!["2"]).not.toHaveProperty("compress");
  });

  it("adds no row to a node that already has a mask or a key link", () => {
    const mask = { kind: "luminance", mode: "add", invert: false, enabled: true, opacity: 1 };
    const masked = withUniform({ masks: { invert: false, items: [mask] } });
    expect(pickUniform(masked, 2, ORANGE)!.patch.nodes!["2"]).not.toHaveProperty("masks");
    const keyed = applyPatch(withUniform(), { links: [pic("input", 1), pic(1, 2), pic(2, "output"), { from: { node: 1, port: "key" }, to: { node: 2, port: "key" } }] });
    expect(pickUniform(keyed, 2, ORANGE)!.patch.nodes!["2"]).not.toHaveProperty("masks");
  });

  it("holds a grey pick at Saturation 2 and says so, while the row stays where the pick was", () => {
    const before = withUniform();
    const pick = pickUniform(before, 2, { a: 0.003, b: 0.004, tone: 0.3 })!;
    expect(pick.tooGrey).toBe(true);
    const node = uniformOf(applyPatch(before, pick.patch));
    expect(node.target!.saturation).toBe(2);
    const row = node.masks.items[0] as unknown as { center: [number, number] };
    expect(row.center[0]).toBeCloseTo(0.003, 9);
  });
});
