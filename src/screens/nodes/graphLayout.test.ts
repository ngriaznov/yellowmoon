/**
 * The graph's geometry without a screen (`docs/23` §3.2 and §7.2): where the cards and the markers are, where the ports are, how far a point
 * is from a link, which link a dropped card is over, and which card is next to the focused one in an arrow key's direction.
 */
import { describe, expect, it } from "vitest";
import type { DevelopSettings, NodeLink, NodePatch } from "../../api/types";
import { sortedLinks } from "../../state/graphOps";
import { applyPatch, defaultSettings } from "../../state/settingsGraph";
import {
  bezierOf,
  CARD_H,
  CARD_W,
  DROP_REACH,
  distanceToLink,
  extentOf,
  isMarker,
  layoutGraph,
  linkEnds,
  linkUnder,
  MARKER,
  MARKER_GAP,
  neighbourIn,
  originOf,
  portPoint,
  posOf,
  UNIT_H,
  UNIT_W,
} from "./graphLayout";

const pic = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

/** A line of correctors at the given places, Main first, Input → Main → … → Output. */
function line(places: Array<[number, number]>): DevelopSettings {
  const nodes: Record<string, NodePatch> = {};
  places.forEach((pos, i) => {
    const id = i + 1;
    nodes[String(id)] = id === 1 ? { pos } : { kind: "corrector", pos };
  });
  const ids = places.map((_, i) => i + 1);
  const links = [pic("input", 1), ...ids.slice(1).map((id) => pic(id - 1, id)), pic(ids[ids.length - 1] as number, "output")];
  return applyPatch(defaultSettings(), { nodes, links: sortedLinks(links) });
}

describe("where the cards are", () => {
  it("puts a card at its pos in grid units: 214 px across and 150 down", () => {
    expect(originOf([0, 0])).toEqual({ x: 0, y: 0 });
    expect(originOf([3, 1])).toEqual({ x: 3 * UNIT_W, y: UNIT_H });
    expect(posOf({ x: UNIT_W * 2, y: UNIT_H * -1 })).toEqual([2, -1]);
    expect(UNIT_W).toBe(214);
    expect(UNIT_H).toBe(150);
  });

  it("makes a card 150 × 112", () => {
    const layout = layoutGraph(line([[0, 0], [1, 0]]));
    expect(layout.boxes.get(2)).toEqual({ x: 214, y: 0, w: CARD_W, h: CARD_H });
    expect([CARD_W, CARD_H]).toEqual([150, 112]);
  });

  it("follows a card that is being dragged", () => {
    const layout = layoutGraph(line([[0, 0], [1, 0]]), new Map([[2, { x: 300.5, y: 40 }]]));
    expect(layout.boxes.get(2)).toMatchObject({ x: 300.5, y: 40 });
  });
});

describe("the markers", () => {
  it("sit just outside the leftmost and the rightmost cards, on the rows of Main and of the node that feeds the Output", () => {
    const layout = layoutGraph(line([[0, 0], [1, 0], [2, 1]]));
    const input = layout.boxes.get("input")!;
    const output = layout.boxes.get("output")!;
    // The Input's centre is a marker-gap left of Main's left edge, on the middle of Main's picture line.
    expect(input.x + input.w / 2).toBe(0 - MARKER_GAP);
    expect(input.y + input.h / 2).toBe(56);
    expect(input.w).toBe(MARKER);
    // The Output is right of the rightmost card (the third, at x = 2 units, one row down), on its row.
    expect(output.x + output.w / 2).toBe(2 * UNIT_W + CARD_W + MARKER_GAP);
    expect(output.y + output.h / 2).toBe(UNIT_H + 56);
  });

  it("move with the cards: they never overlap one", () => {
    const layout = layoutGraph(line([[0, 0], [1, 0]]), new Map([[2, { x: 900, y: 0 }]]));
    const output = layout.boxes.get("output")!;
    expect(output.x).toBeGreaterThan(900 + CARD_W);
    expect(isMarker("input") && isMarker("output") && !isMarker(1)).toBe(true);
  });

  it("are placed from the cards, not from their own pos", () => {
    const settings = line([[0, 0], [1, 0]]);
    const changed = applyPatch(settings, { nodes: { input: { pos: [40, 40] }, output: { pos: [-40, 9] } } });
    expect(layoutGraph(changed).boxes.get("input")).toEqual(layoutGraph(settings).boxes.get("input"));
  });
});

describe("the ports and the links", () => {
  const layout = layoutGraph(line([[0, 0], [1, 0]]));

  it("have a card's picture ports at its middle line and its key ports in the footer", () => {
    expect(portPoint(layout, { node: 1, port: "picture" }, "in")).toEqual({ x: -9, y: 56 });
    expect(portPoint(layout, { node: 1, port: "picture" }, "out")).toEqual({ x: 154, y: 56 });
    expect(portPoint(layout, { node: 1, port: "key" }, "in")).toEqual({ x: -9, y: 97 });
    expect(portPoint(layout, { node: 1, port: "key" }, "out")).toEqual({ x: 154, y: 97 });
  });

  it("have the markers' ports on their discs: a square out of the Input and an arrowhead into the Output", () => {
    const input = layout.boxes.get("input")!;
    expect(portPoint(layout, { node: "input", port: "picture" }, "out")).toEqual({ x: input.x + MARKER + 8, y: input.y + 16 });
    const output = layout.boxes.get("output")!;
    expect(portPoint(layout, { node: "output", port: "picture" }, "in")).toEqual({ x: output.x - 10, y: output.y + 16 });
    expect(portPoint(layout, { node: 77, port: "picture" }, "in")).toEqual({ x: 0, y: 0 });
  });

  it("run a link between the output port of one node and the input port of the next", () => {
    const ends = linkEnds(layout, pic(1, 2));
    expect(ends.from).toEqual({ x: 154, y: 56 });
    expect(ends.to).toEqual({ x: 214 - 9, y: 56 });
  });

  it("draw a straight line between two ports on one row, and a curve between rows", () => {
    const [p0, p1, p2, p3] = bezierOf({ x: 0, y: 10 }, { x: 100, y: 10 });
    expect([p0.y, p1.y, p2.y, p3.y]).toEqual([10, 10, 10, 10]);
    expect([p1.x, p2.x]).toEqual([50, 50]);
    // A link that goes back (the target is left of the source) bends out from both ends.
    const back = bezierOf({ x: 100, y: 0 }, { x: 0, y: 50 });
    expect(back[1].x).toBeGreaterThan(100);
    expect(back[2].x).toBeLessThan(0);
  });

  it("measure how far a point is from a link as it is drawn", () => {
    const link = pic(1, 2);
    expect(distanceToLink(layout, link, { x: 180, y: 56 })).toBeLessThan(2);
    expect(distanceToLink(layout, link, { x: 180, y: 56 + 30 })).toBeCloseTo(30, 0);
    expect(distanceToLink(layout, link, { x: 180, y: 56 - 12 })).toBeCloseTo(12, 0);
  });
});

describe("which link a dropped card is over", () => {
  const settings = line([[0, 0], [1, 0], [2, 0], [3, 0]]);
  const layout = layoutGraph(settings);
  const links = settings.nodes.links;
  // The middle of the link 3 → 4.
  const mid = (a: number, b: number) => {
    const { from, to } = linkEnds(layout, pic(a, b));
    return { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 };
  };

  it("is the picture link nearest to its centre, within the reach", () => {
    const at = mid(3, 4);
    expect(linkUnder(layout, links, 2, at)).toEqual(pic(3, 4));
    expect(linkUnder(layout, links, 2, { x: at.x, y: at.y + DROP_REACH - 1 })).toEqual(pic(3, 4));
    expect(linkUnder(layout, links, 2, { x: at.x, y: at.y + DROP_REACH + 12 })).toBeUndefined();
  });

  it("is never one of the card's own links, nor the link into Main", () => {
    expect(linkUnder(layout, links, 3, mid(2, 3))).toBeUndefined();
    expect(linkUnder(layout, links, 3, mid(3, 4))).toBeUndefined();
    const into = linkEnds(layout, pic("input", 1));
    expect(linkUnder(layout, links, 3, { x: (into.from.x + into.to.x) / 2, y: into.from.y })).toBeUndefined();
  });

  it("is a picture link: a key link is not a place to drop a card", () => {
    const keyed = applyPatch(settings, { links: sortedLinks([...links, { from: { node: 2, port: "key" }, to: { node: 4, port: "key" } }]) });
    const l = layoutGraph(keyed);
    const { from, to } = linkEnds(l, { from: { node: 2, port: "key" }, to: { node: 4, port: "key" } });
    const found = linkUnder(l, keyed.nodes.links, 3, { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 });
    expect(found?.from.port ?? "picture").toBe("picture");
  });
});

describe("the arrow keys", () => {
  const settings = line([[0, 0], [1, 0], [2, 0]]);
  const layout = layoutGraph(settings);

  it("go to the next card along the row, the markers included", () => {
    expect(neighbourIn(layout, 1, "right")).toBe(2);
    expect(neighbourIn(layout, 2, "right")).toBe(3);
    expect(neighbourIn(layout, 3, "right")).toBe("output");
    expect(neighbourIn(layout, 2, "left")).toBe(1);
    expect(neighbourIn(layout, 1, "left")).toBe("input");
  });

  it("stay where they are at the ends, and where nothing lies that way", () => {
    expect(neighbourIn(layout, "output", "right")).toBeUndefined();
    expect(neighbourIn(layout, "input", "left")).toBeUndefined();
    expect(neighbourIn(layout, 2, "up")).toBeUndefined();
    expect(neighbourIn(layout, 2, "down")).toBeUndefined();
    expect(neighbourIn(layout, 99, "left")).toBeUndefined();
  });

  it("go up and down to the card in the next row, and prefer the one in line", () => {
    const rows = line([[0, 0], [1, 0], [1, 1], [3, 1]]);
    const l = layoutGraph(rows);
    expect(neighbourIn(l, 2, "down")).toBe(3);
    expect(neighbourIn(l, 3, "up")).toBe(2);
    // From the card in the lower row, right goes along that row.
    expect(neighbourIn(l, 3, "right")).toBe(4);
  });
});

describe("the extent", () => {
  it("holds every node", () => {
    const layout = layoutGraph(line([[0, 0], [1, 0]]));
    const e = extentOf(layout);
    for (const box of layout.boxes.values()) {
      expect(box.x).toBeGreaterThanOrEqual(e.x);
      expect(box.x + box.w).toBeLessThanOrEqual(e.x + e.w);
    }
  });
});
