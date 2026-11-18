/**
 * The Blur node's graph operations (`docs/33` sections 2.5, 2.6 and 4): Add node, Delete, Reorder and the links as a corrector's, the 64-node limit, Reset and the head's line.
 */
import { describe, expect, it } from "vitest";
import type { DevelopSettings, NodeLink, SettingsPatch } from "../api/types";
import { BLUR_TAB, FIRST_NODE_TAB, NODE_TABS, simpleTabOf, tabInfo } from "../components/tabs";
import { validateGraph } from "../fixtures/graph";
import { BLUR_TYPE_NAME, cardName, headLine, insertAfter, moveEarlier, moveLater, remove, resetNode } from "./graphOps";
import type { GraphOutcome } from "./graphOps";
import { applyPatch, defaultSettings, diffGraph, isCardItem, nodeOf } from "./settingsGraph";
import type { BlurItem } from "./settingsGraph";

const pic = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const sent = (o: GraphOutcome): SettingsPatch => {
  if (!o.ok) throw new Error(`refused: ${o.reason}`);
  return o.patch;
};
const blurOf = (s: DevelopSettings, id = 2): BlurItem => nodeOf(s, id) as BlurItem;

/** Main (1), a Blur node (2), a corrector (3), then the Output. */
function chain(extra: Record<string, unknown> = {}, links: NodeLink[] = [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output")]): DevelopSettings {
  return applyPatch(defaultSettings(), { nodes: { "2": { kind: "blur", pos: [1, 0], ...extra }, "3": { kind: "corrector", name: "Node 3", pos: [2, 0] } }, links });
}

describe("Add node > Blur", () => {
  it("adds a node named Blur after the selected one: Soft at Radius 20, Respect mask edges on, no masks, and the core accepts the graph", () => {
    const before = defaultSettings();
    const outcome = insertAfter(before, 1, { kind: "blur" });
    const after = applyPatch(before, sent(outcome));
    expect(validateGraph(after)).toBeNull();
    expect(blurOf(after)).toMatchObject({ kind: "blur", name: "Blur", bypass: false, type: "soft", soft: { radius: 20 }, respect_mask_edges: true, masks: { invert: false, items: [] } });
    expect(outcome.ok && outcome.select).toBe(2);
    expect(after.nodes.links).toContainEqual(pic(1, 2));
    expect(after.nodes.links).toContainEqual(pic(2, "output"));
  });

  it("goes into the selected node's outgoing link and moves the nodes after it", () => {
    const before = chain();
    const after = applyPatch(before, sent(insertAfter(before, 2, { kind: "blur" })));
    expect(validateGraph(after)).toBeNull();
    expect(after.nodes.links).toContainEqual(pic(2, 4));
    expect(after.nodes.links).toContainEqual(pic(4, 3));
    expect(nodeOf(after, 4)?.kind).toBe("blur");
  });

  it("goes after Main when the Input is selected, and into the Output's incoming link when the Output is", () => {
    const before = chain();
    expect(applyPatch(before, sent(insertAfter(before, "input", { kind: "blur" }))).nodes.links).toContainEqual(pic(1, 4));
    expect(applyPatch(before, sent(insertAfter(before, "output", { kind: "blur" }))).nodes.links).toContainEqual(pic(4, "output"));
  });

  it("undoes as one step", () => {
    const before = defaultSettings();
    const after = applyPatch(before, sent(insertAfter(before, 1, { kind: "blur" })));
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
    expect(insertAfter(fill(63), 1, { kind: "blur" }).ok).toBe(true);
    const refused = insertAfter(fill(64), 1, { kind: "blur" });
    expect(refused.ok).toBe(false);
    expect(!refused.ok && refused.reason).toBe("A photo can have up to 64 nodes.");
  });
});

describe("a Blur node is a card like a corrector", () => {
  it("has a card, masks and a key output", () => {
    expect(isCardItem(blurOf(chain()))).toBe(true);
  });

  it("Delete reconnects its neighbours, as for a corrector", () => {
    const before = chain();
    const after = applyPatch(before, sent(remove(before, 2)));
    expect(nodeOf(after, 2)).toBeUndefined();
    expect(after.nodes.links).toContainEqual(pic(1, 3));
    expect(validateGraph(after)).toBeNull();
  });

  it("Move later and Move earlier swap it with its neighbour", () => {
    const before = chain();
    const later = applyPatch(before, sent(moveLater(before, 2)));
    expect(later.nodes.links).toContainEqual(pic(1, 3));
    expect(later.nodes.links).toContainEqual(pic(3, 2));
    expect(later.nodes.links).toContainEqual(pic(2, "output"));
    const back = applyPatch(later, sent(moveEarlier(later, 2)));
    expect(back.nodes.links).toContainEqual(pic(1, 2));
    expect(back.nodes.links).toContainEqual(pic(2, 3));
  });

  it("is named in a card's accessible name by its name and number", () => {
    expect(cardName(chain({ name: "Soften" }), 2)).toContain("Soften");
  });
});

describe("Reset on a Blur node", () => {
  it("puts the type and every type's values back and keeps the masks, the name, the bypass and the links", () => {
    const mask = { kind: "color_range", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.05, 0.05] };
    const before = chain({
      name: "Dream",
      bypass: true,
      type: "zoom",
      soft: { radius: 80 },
      lens: { radius: 50, blades: 6, rotation_deg: 30, highlights: 60 },
      motion: { length: 70, angle_deg: 45 },
      zoom: { amount: 60, center: [0.2, 0.7] },
      respect_mask_edges: false,
      masks: { invert: true, items: [mask] },
    });
    const after = applyPatch(before, sent(resetNode(before, 2)));
    const node = blurOf(after);
    expect(node).toMatchObject({
      type: "soft",
      soft: { radius: 20 },
      lens: { radius: 20, blades: 0, rotation_deg: 0, highlights: 0 },
      motion: { length: 20, angle_deg: 0 },
      zoom: { amount: 20, center: [0.5, 0.5] },
      respect_mask_edges: true,
    });
    expect(node.masks.items).toHaveLength(1);
    expect(node.masks.invert).toBe(true);
    expect(node.name).toBe("Dream");
    expect(node.bypass).toBe(true);
    expect(after.nodes.links).toEqual(before.nodes.links);
  });

  it("answers an empty patch on a node that is already at its defaults", () => {
    const outcome = resetNode(chain(), 2);
    expect(outcome.ok && Object.keys(outcome.patch.nodes ?? {})).toEqual([]);
  });
});

describe("the head's line of a Blur node", () => {
  it("is the type, Bypassed, or Not connected", () => {
    expect(headLine(chain(), 2)).toBe("Soft");
    for (const type of ["soft", "lens", "motion", "zoom"] as const) expect(headLine(chain({ type }), 2)).toBe(BLUR_TYPE_NAME[type]);
    expect(headLine(chain({ bypass: true }), 2)).toBe("Bypassed");
    expect(headLine(chain({}, [pic("input", 1), pic(1, 3), pic(3, "output")]), 2)).toBe("Not connected");
  });
});

describe("the Blur tab", () => {
  it("is the node's first tab, before Mask, in the Detail tab's colour, and the simple view has no such tab", () => {
    expect(NODE_TABS.blur).toEqual(["blur", "mask"]);
    expect(FIRST_NODE_TAB.blur).toBe("blur");
    expect(tabInfo("blur")).toBe(BLUR_TAB);
    expect(BLUR_TAB.label).toBe("Blur");
    expect(simpleTabOf("blur")).toBe("light");
  });
});
