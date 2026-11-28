/**
 * The Reference Grade node's graph operations (`docs/34` sections 2.1, 3.5 and 9.1): Add node after the selected node, Delete, Reorder and the links as a
 * corrector's, the 64-node limit, Reset, the head's line, and the kind's tabs.
 */
import { describe, expect, it } from "vitest";
import type { DevelopSettings, NodeLink, SettingsPatch } from "../api/types";
import { FIRST_NODE_TAB, NODE_TABS, simpleTabOf, tabForShortcut, tabInfo } from "../components";
import { validateGraph } from "../fixtures/graph";
import { cardName, headLine, insertAfter, moveEarlier, moveLater, remove, resetNode } from "./graphOps";
import type { GraphOutcome } from "./graphOps";
import { applyPatch, defaultSettings, diffGraph, isCardItem, nodeOf } from "./settingsGraph";
import type { ReferenceGradeItem } from "./settingsGraph";

const pic = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const sent = (o: GraphOutcome): SettingsPatch => {
  if (!o.ok) throw new Error(`refused: ${o.reason}`);
  return o.patch;
};
const matchOf = (s: DevelopSettings, id = 2): ReferenceGradeItem => nodeOf(s, id) as ReferenceGradeItem;

/** Main (1), a Reference Grade node (2), a corrector (3), then the Output. */
function chain(extra: Record<string, unknown> = {}): DevelopSettings {
  return applyPatch(defaultSettings(), {
    nodes: { "2": { kind: "reference_grade", pos: [1, 0], ...extra }, "3": { kind: "corrector", name: "Node 3", pos: [2, 0] } },
    links: [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output")],
  });
}

describe("Add node > Reference Grade", () => {
  it("adds a node named Match after the selected one, every field at the identity, Strength 100, Skin protection on, and the core accepts the graph", () => {
    const before = defaultSettings();
    const outcome = insertAfter(before, 1, { kind: "reference_grade" });
    const after = applyPatch(before, sent(outcome));
    expect(validateGraph(after)).toBeNull();
    const node = matchOf(after);
    expect(node).toMatchObject({ kind: "reference_grade", name: "Match", bypass: false, strength: 100, skin_protection: true, masks: { invert: false, items: [] } });
    expect(node.hue_curves).toEqual({ hue_vs_hue: [], hue_vs_sat: [] });
    expect(node.elements.sky).toEqual({ exposure_ev: 0, hue_deg: 0, chroma: 1 });
    expect(node.elements.subject).toEqual({ exposure_ev: 0 });
    expect(node.trim).toEqual({ exposure_ev: 0, temperature: 0, tint: 0 });
    expect(outcome.ok && outcome.select).toBe(2);
    expect(after.nodes.links).toContainEqual(pic(1, 2));
    expect(after.nodes.links).toContainEqual(pic(2, "output"));
  });

  it("goes into the selected node's outgoing link, after Main when the Input is selected and before the Output when it is", () => {
    const before = chain();
    const after = applyPatch(before, sent(insertAfter(before, 2, { kind: "reference_grade" })));
    expect(validateGraph(after)).toBeNull();
    expect(after.nodes.links).toContainEqual(pic(2, 4));
    expect(after.nodes.links).toContainEqual(pic(4, 3));
    expect(applyPatch(before, sent(insertAfter(before, "input", { kind: "reference_grade" }))).nodes.links).toContainEqual(pic(1, 4));
    expect(applyPatch(before, sent(insertAfter(before, "output", { kind: "reference_grade" }))).nodes.links).toContainEqual(pic(4, "output"));
  });

  it("undoes as one step", () => {
    const before = defaultSettings();
    const after = applyPatch(before, sent(insertAfter(before, 1, { kind: "reference_grade" })));
    const diff = diffGraph(before, after);
    expect(diff).not.toBeNull();
    expect(nodeOf(applyPatch(after, diff!.before), 2)).toBeUndefined();
  });

  it("is refused at 64 nodes with the sentence, and allowed at 63", () => {
    const fill = (n: number): DevelopSettings => {
      const nodes: Record<string, unknown> = {};
      const links: NodeLink[] = [pic("input", 1)];
      for (let id = 2; id <= n; id += 1) {
        nodes[String(id)] = { kind: "corrector", name: `Node ${id}`, pos: [id - 1, 0] };
        links.push(pic(id - 1, id));
      }
      links.push(pic(n, "output"));
      return applyPatch(defaultSettings(), { nodes: nodes as never, links });
    };
    expect(insertAfter(fill(63), 1, { kind: "reference_grade" }).ok).toBe(true);
    const refused = insertAfter(fill(64), 1, { kind: "reference_grade" });
    expect(!refused.ok && refused.reason).toBe("A photo can have up to 64 nodes.");
  });
});

describe("a Reference Grade node is a card like a corrector", () => {
  it("has a card, masks, a number and a name in its card's accessible name", () => {
    expect(isCardItem(matchOf(chain()))).toBe(true);
    expect(cardName(chain({ name: "Dusk" }), 2)).toContain("Dusk");
  });

  it("Delete reconnects its neighbours; Move later and Move earlier swap it with its neighbour", () => {
    const before = chain();
    expect(applyPatch(before, sent(remove(before, 2))).nodes.links).toContainEqual(pic(1, 3));
    const later = applyPatch(before, sent(moveLater(before, 2)));
    expect(later.nodes.links).toContainEqual(pic(3, 2));
    expect(later.nodes.links).toContainEqual(pic(2, "output"));
    const back = applyPatch(later, sent(moveEarlier(later, 2)));
    expect(back.nodes.links).toContainEqual(pic(2, 3));
  });

  it("its head says Reference Grade, Bypassed or Not connected", () => {
    expect(headLine(chain(), 2)).toBe("Reference Grade");
    expect(headLine(chain({ bypass: true }), 2)).toBe("Bypassed");
    const loose = applyPatch(defaultSettings(), { nodes: { "2": { kind: "reference_grade", pos: [1, 1] } }, links: [pic("input", 1), pic(1, "output")] });
    expect(headLine(loose, 2)).toBe("Not connected");
  });
});

describe("Reset on a Reference Grade node", () => {
  it("puts trim, grade, hue curves, elements, Strength and Skin protection back and keeps the masks, the name, the bypass and the links", () => {
    const mask = { kind: "color_range", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.05, 0.05] };
    const before = chain({
      name: "Dusk",
      bypass: true,
      trim: { exposure_ev: 0.5 },
      hue_curves: { hue_vs_hue: [{ hue_deg: 120, value: 12 }], hue_vs_sat: [{ hue_deg: 120, value: 0.8 }] },
      elements: { sky: { exposure_ev: -0.4, hue_deg: 10, chroma: 1.2 }, subject: { exposure_ev: 0.3 } },
      strength: 40,
      skin_protection: false,
      match_brightness: 70,
      masks: { invert: true, items: [mask] },
    });
    const after = applyPatch(before, sent(resetNode(before, 2)));
    const node = matchOf(after);
    expect(node.trim.exposure_ev).toBe(0);
    expect(node.hue_curves).toEqual({ hue_vs_hue: [], hue_vs_sat: [] });
    expect(node.elements.sky).toEqual({ exposure_ev: 0, hue_deg: 0, chroma: 1 });
    expect(node.elements.subject).toEqual({ exposure_ev: 0 });
    expect([node.strength, node.skin_protection, node.match_brightness]).toEqual([100, true, 0]);
    expect(node.masks.items).toHaveLength(1);
    expect(node.masks.invert).toBe(true);
    expect([node.name, node.bypass]).toEqual(["Dusk", true]);
    expect(after.nodes.links).toEqual(before.nodes.links);
  });

  it("answers an empty patch on a node that is already at its defaults", () => {
    const outcome = resetNode(chain(), 2);
    expect(outcome.ok && Object.keys(outcome.patch.nodes ?? {})).toEqual([]);
  });
});

describe("the tabs of a Reference Grade node (docs/34 section 9.1)", () => {
  it("are Reference, Light, Colour, Curve, Hue, Elements and Mask, and the node opens on Reference", () => {
    expect(NODE_TABS.reference_grade).toEqual(["reference", "light", "color", "curve", "hue", "elements", "mask"]);
    expect(NODE_TABS.reference_grade.map((id) => tabInfo(id).label)).toEqual(["Reference", "Light", "Colour", "Curve", "Hue", "Elements", "Mask"]);
    expect(FIRST_NODE_TAB.reference_grade).toBe("reference");
    for (const id of ["reference", "hue", "elements"] as const) expect(simpleTabOf(id)).toBe("light");
  });

  it("⌘1 to ⌘7 select them in the order of the row", () => {
    NODE_TABS.reference_grade.forEach((id, i) => {
      expect(tabForShortcut({ metaKey: true, shiftKey: false, altKey: false, ctrlKey: false, code: `Digit${i + 1}`, key: String(i + 1) }, NODE_TABS.reference_grade)).toBe(id);
    });
  });
});
