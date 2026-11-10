/**
 * Where everything is on the graph's canvas, in flow pixels (the canvas at 100 % zoom): the cards, the two markers, the ports and the links
 * between them (`docs/23-nodes.md` §3.2 and §7.2, `design/nodes/html/pages/states.html` for the numbers).
 *
 * A card's `pos` is in grid units, 214 px across and 150 px down. The Input and the Output are fixed markers that sit just outside the
 * leftmost and rightmost cards, so they are placed from the cards, not from their own `pos`. The same numbers draw the ports (the CSS
 * repeats them) and answer the questions that need geometry without a browser: which link a dropped card is over, which card is next to
 * the focused one in an arrow key's direction.
 *
 * Pure functions, tested without a screen (`graphLayout.test.ts`).
 */
import type { DevelopSettings, NodeId, NodeLink, NodePos, NodeRef, Port } from "../../api/types";
import { CHANNEL_PORTS, inputNumber, isMixerKind, mixerPort, MIXER_INPUTS, portRank, portType } from "../../state/ports";
import { INPUT_ID, MAIN_ID, nodeOf, numberedOf, OUTPUT_ID } from "../../state/settingsGraph";

/** One grid unit, in pixels (`docs/23` §3.2). */
export const UNIT_W = 214;
export const UNIT_H = 150;

/** A card: 150 × 112, a thumbnail band of 81 and a footer row of 30 (`docs/23` §7.2). */
export const CARD_W = 150;
export const CARD_H = 112;
export const THUMB_H = 81;
export const FOOT_H = 30;

/** Where the ports sit on a card, from its top: the picture ports at the card's middle, the key ports in the footer row. */
export const PICTURE_PORT_Y = 56;
export const KEY_PORT_Y = 97;
/** A port's box on a card: how far it sticks out of the card's side (its left edge for an input, in pixels). */
export const PORT_OUT = 9;

/** A marker (the Input, the Output): a round 32 px disc. Its ports stick out 8 px (the square) and 10 px (the arrowhead). */
export const MARKER = 32;
export const MARKER_SQUARE_OUT = 8;
export const MARKER_ARROW_OUT = 10;
/** How far a marker's centre is from the card next to it. */
export const MARKER_GAP = 86;

/** A point, in flow pixels. */
export interface XY {
  x: number;
  y: number;
}

/** A box, in flow pixels. */
export interface Box extends XY {
  w: number;
  h: number;
  /** Where the node's picture line is, from its top: a corrector's picture ports (56), a glyph card's middle. */
  line?: number;
  /** The ports of a glyph card, as the distance from its top to each port's centre, by `in:picture.1`, `out:red`. */
  ports?: Readonly<Record<string, number>>;
}

/** A glyph card (a Parallel, a Layer, a Splitter, a Combiner; `docs/25` §5.2): 56 px wide, 64 high with two ports and 24 more for each further one. */
export const GLYPH_W = 56;
export const GLYPH_BASE_H = 64;
export const GLYPH_PORT_STEP = 24;

/** One port of a glyph card: its place from the card's top (the centre of its glyph), and whether it is the "+" that adds an input. */
export interface PortSpot {
  /** The port it stands for; for the "+" the next number, `picture.N+1`. */
  port: Port;
  y: number;
  plus?: boolean;
}

/** The size of a glyph card and the places of its ports. */
export interface GlyphGeometry {
  w: number;
  h: number;
  inputs: PortSpot[];
  outputs: PortSpot[];
}

/** The ports of a glyph card stacked 24 px apart and centred on `h / 2`. */
function stacked(ports: readonly { port: Port; plus?: boolean }[], h: number): PortSpot[] {
  return ports.map((p, i) => ({ ...p, y: h / 2 + (i - (ports.length - 1) / 2) * GLYPH_PORT_STEP }));
}

/**
 * The size and ports of the card of a Parallel, a Layer, a Splitter or a Combiner, or null for any other node. A mixer shows its linked
 * inputs in port order and, while it has fewer than 8, the "+" under them (an input with no link at all shows the "+" alone); a Splitter's
 * outputs and a Combiner's inputs are red, green and blue from the top. The picture output is at the middle.
 */
export function glyphGeometry(settings: DevelopSettings, id: NodeId): GlyphGeometry | null {
  const node = nodeOf(settings, id);
  if (!node) return null;
  let inputs: { port: Port; plus?: boolean }[];
  let outputs: { port: Port }[];
  if (isMixerKind(node.kind)) {
    const linked = settings.nodes.links
      .filter((l) => l.to.node === id && inputNumber(l.to.port) !== null)
      .map((l) => l.to.port)
      .sort((a, b) => portRank(a) - portRank(b));
    inputs = linked.map((port) => ({ port }));
    if (linked.length < MIXER_INPUTS) inputs.push({ port: mixerPort(linked.length + 1), plus: true });
    outputs = [{ port: "picture" }];
  } else if (node.kind === "splitter") {
    inputs = [{ port: "picture" }];
    outputs = CHANNEL_PORTS.map((port) => ({ port }));
  } else if (node.kind === "combiner") {
    inputs = CHANNEL_PORTS.map((port) => ({ port }));
    outputs = [{ port: "picture" }];
  } else {
    return null;
  }
  const count = Math.max(inputs.length, outputs.length);
  const h = GLYPH_BASE_H + GLYPH_PORT_STEP * Math.max(0, count - 2);
  return { w: GLYPH_W, h, inputs: stacked(inputs, h), outputs: stacked(outputs, h) };
}

/** The place of every node. */
export interface GraphLayout {
  boxes: ReadonlyMap<NodeId, Box>;
}

/** The top left of a card at this `pos`. */
export const originOf = (pos: NodePos): XY => ({ x: pos[0] * UNIT_W, y: pos[1] * UNIT_H });

/** The `pos` of a corrector or Film card whose top left is here: the grid units, as a drop turns them back. */
export const posOf = (at: XY): NodePos => [at.x / UNIT_W, at.y / UNIT_H];

/**
 * The top left of a glyph card of height `h` at this `pos`: its `pos` is the card's middle line, the line of a corrector's picture ports
 * (`design/mixers/mockup-notes.md`, question 3), so a Parallel between two correctors sits on their line.
 */
export const glyphOrigin = (pos: NodePos, h: number): XY => ({ x: pos[0] * UNIT_W, y: pos[1] * UNIT_H + PICTURE_PORT_Y - h / 2 });

/** The `pos` of a glyph card of height `h` whose top left is here. */
export const glyphPosOf = (at: XY, h: number): NodePos => [at.x / UNIT_W, (at.y + h / 2 - PICTURE_PORT_Y) / UNIT_H];

/**
 * The boxes of every node: the cards where their `pos` puts them (or where `moved` says they are now, while one is dragged), the Input
 * one marker-gap left of the leftmost card on Main's row, and the Output one marker-gap right of the rightmost card on the row of the
 * node that feeds it.
 */
export function layoutGraph(settings: DevelopSettings, moved: ReadonlyMap<NodeId, XY> = new Map()): GraphLayout {
  const boxes = new Map<NodeId, Box>();
  for (const node of numberedOf(settings)) {
    const glyph = glyphGeometry(settings, node.id);
    if (glyph) {
      const at = moved.get(node.id) ?? glyphOrigin(node.pos, glyph.h);
      const ports: Record<string, number> = {};
      for (const spot of glyph.inputs) if (!spot.plus) ports[`in:${spot.port}`] = spot.y;
      for (const spot of glyph.outputs) ports[`out:${spot.port}`] = spot.y;
      boxes.set(node.id, { x: at.x, y: at.y, w: glyph.w, h: glyph.h, line: glyph.h / 2, ports });
      continue;
    }
    const at = moved.get(node.id) ?? originOf(node.pos);
    boxes.set(node.id, { x: at.x, y: at.y, w: CARD_W, h: CARD_H });
  }
  const cards = [...boxes.values()];
  const left = cards.length > 0 ? Math.min(...cards.map((b) => b.x)) : 0;
  const right = cards.length > 0 ? Math.max(...cards.map((b) => b.x + b.w)) : CARD_W;
  const main = boxes.get(MAIN_ID);
  const feeder = settings.nodes.links.find((l) => l.to.node === OUTPUT_ID && l.to.port === "picture")?.from.node;
  const feeding = (typeof feeder === "number" ? boxes.get(feeder) : undefined) ?? main;
  const around = (centre: number, row: Box | undefined): Box => ({
    x: centre - MARKER / 2,
    y: (row?.y ?? 0) + (row?.line ?? PICTURE_PORT_Y) - MARKER / 2,
    w: MARKER,
    h: MARKER,
  });
  boxes.set(INPUT_ID, around(left - MARKER_GAP, main));
  boxes.set(OUTPUT_ID, around(right + MARKER_GAP, feeding));
  return { boxes };
}

/** True when a node is a marker (the Input or the Output). */
export const isMarker = (id: NodeId): boolean => id === INPUT_ID || id === OUTPUT_ID;

/** Where a link's end is: the output port's right edge, or the input port's left edge (the marker's own ports are a little different). */
export function portPoint(layout: GraphLayout, end: NodeRef, side: "out" | "in"): XY {
  const box = layout.boxes.get(end.node);
  if (!box) return { x: 0, y: 0 };
  if (isMarker(end.node)) {
    return side === "out"
      ? { x: box.x + box.w + MARKER_SQUARE_OUT, y: box.y + box.h / 2 }
      : { x: box.x - MARKER_ARROW_OUT, y: box.y + box.h / 2 };
  }
  const own = box.ports?.[`${side === "out" ? "out" : "in"}:${end.port}`];
  const y = box.y + (own ?? (end.port === "key" ? KEY_PORT_Y : box.line ?? PICTURE_PORT_Y));
  // An output port is an 8 px square whose right edge is 4 px past the card's; an input port is a triangle that starts 9 px left of it.
  return side === "out" ? { x: box.x + box.w + 4, y } : { x: box.x - PORT_OUT, y };
}

/** The two ends of a link. */
export function linkEnds(layout: GraphLayout, link: NodeLink): { from: XY; to: XY } {
  return { from: portPoint(layout, link.from, "out"), to: portPoint(layout, link.to, "in") };
}

/** xyflow's horizontal Bézier between two ports (its curvature of 0.25): the control points of the curve a link is drawn along. */
export function bezierOf(from: XY, to: XY): [XY, XY, XY, XY] {
  const distance = to.x - from.x;
  const offset = distance >= 0 ? 0.5 * distance : 0.25 * 25 * Math.sqrt(-distance);
  return [from, { x: from.x + offset, y: from.y }, { x: to.x - offset, y: to.y }, to];
}

/** A point of the cubic Bézier at `t` (0 to 1). */
function at([p0, p1, p2, p3]: [XY, XY, XY, XY], t: number): XY {
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const c = 3 * u * t * t;
  const d = t * t * t;
  return { x: a * p0.x + b * p1.x + c * p2.x + d * p3.x, y: a * p0.y + b * p1.y + c * p2.y + d * p3.y };
}

/** How far a point is from the link as it is drawn, in pixels (the curve is sampled every few pixels). */
export function distanceToLink(layout: GraphLayout, link: NodeLink, p: XY): number {
  const { from, to } = linkEnds(layout, link);
  const curve = bezierOf(from, to);
  const steps = Math.max(8, Math.ceil((Math.abs(to.x - from.x) + Math.abs(to.y - from.y)) / 6));
  let best = Number.POSITIVE_INFINITY;
  for (let i = 0; i <= steps; i++) {
    const q = at(curve, i / steps);
    best = Math.min(best, Math.hypot(q.x - p.x, q.y - p.y));
  }
  return best;
}

/** A dropped card is "over" a link when its centre is this close to it (flow pixels). */
export const DROP_REACH = 22;

/**
 * The picture link a card dropped here is over, the nearest within `DROP_REACH` of its centre, or undefined. The card's own links never
 * count (a card is not dropped on its own links), and neither does the one that goes into Main (nothing goes before Main).
 */
export function linkUnder(layout: GraphLayout, links: readonly NodeLink[], dragged: NodeId, centre: XY): NodeLink | undefined {
  let best: { link: NodeLink; d: number } | undefined;
  for (const link of links) {
    if (portType(link.from.port) !== "picture" || link.from.node === dragged || link.to.node === dragged) continue;
    if (link.to.node === MAIN_ID) continue;
    const d = distanceToLink(layout, link, centre);
    if (d <= DROP_REACH && (!best || d < best.d)) best = { link, d };
  }
  return best?.link;
}

/** An arrow key's direction. */
export type Direction = "left" | "right" | "up" | "down";

/**
 * The node next to `from` in an arrow key's direction: among the nodes whose centre lies that way, the one nearest along the direction,
 * with a straying sideways counting double, so a row is followed along and a column down. Undefined when nothing lies that way.
 */
export function neighbourIn(layout: GraphLayout, from: NodeId, direction: Direction): NodeId | undefined {
  const here = layout.boxes.get(from);
  if (!here) return undefined;
  const cx = here.x + here.w / 2;
  const cy = here.y + here.h / 2;
  let best: { id: NodeId; cost: number } | undefined;
  for (const [id, box] of layout.boxes) {
    if (id === from) continue;
    const dx = box.x + box.w / 2 - cx;
    const dy = box.y + box.h / 2 - cy;
    const along = direction === "right" ? dx : direction === "left" ? -dx : direction === "down" ? dy : -dy;
    const across = direction === "left" || direction === "right" ? dy : dx;
    if (along <= 2) continue;
    const cost = along + 2 * Math.abs(across);
    if (!best || cost < best.cost) best = { id, cost };
  }
  return best?.id;
}

/** The box that holds every node, with a margin: what Fit shows. */
export function extentOf(layout: GraphLayout): Box {
  const boxes = [...layout.boxes.values()];
  const x0 = Math.min(...boxes.map((b) => b.x));
  const y0 = Math.min(...boxes.map((b) => b.y));
  const x1 = Math.max(...boxes.map((b) => b.x + b.w));
  const y1 = Math.max(...boxes.map((b) => b.y + b.h));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}
