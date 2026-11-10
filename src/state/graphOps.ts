/**
 * The graph's operations (`docs/23-nodes.md` §7.2): add, delete, bypass, rename, move, reorder and the links drawn by hand. Every one is a
 * **pure function of the settings** that answers the patch to send, or the plain sentence that says why the operation is refused. The
 * patch is one node patch (`nodes`, `remove` and `links`, section 3.7), so it is one undo step; nothing here touches the store, the core
 * or a screen, and everything is tested without one (`graphOps.test.ts`).
 *
 * The same functions say which inputs light up while a link is dragged (`inputStates`) and what a card's number is (`cardNumbers`, the
 * plan's order of section 5.2 computed here as the core computes it), so the canvas, the menus and the keys cannot disagree about a rule.
 * The core validates whatever is sent (`crates/ym-core/src/settings/graph.rs`); these functions never send a graph it would refuse.
 */
import type { DevelopSettings, LookId, NodeId, NodeItem, NodeLink, NodePatch, NodePos, NodeRef, Port, SettingsPatch } from "../api/types";
import {
  applyPatch,
  MIXER_NAMES,
  newMixer,
  numberedOf,
  defaultGrade,
  defaultInput,
  INPUT_ID,
  isEmptyPatch,
  MAIN_ID,
  neutralGrade,
  NEUTRAL_LOOK,
  newCorrector,
  newFilm,
  DEFAULT_STOCK,
  nodeAsPatch,
  nodeOf,
  OUTPUT_ID,
  same,
} from "./settingsGraph";
import type { CardItem, MixerItem } from "./settingsGraph";
import { CHANNEL_PORTS, firstInputPort, inputNumber, isGlyphKind, isMixerKind, MIXER_INPUTS, mixerPort, portRank, portType, portWords } from "./ports";

/** Most corrector nodes a photo can have, Main included (`docs/23` section 3.5). */
export const MAX_CORRECTORS = 64;
/** Longest name a node can have, in characters. */
export const MAX_NAME_LENGTH = 32;
/** The name a new node has. */
export const DEFAULT_NAME = "Corrector";
/** The name a new Film node has. */
export const FILM_DEFAULT_NAME = "Film";
/** The ids a node can have: 1 to 2³¹ − 1. */
const MAX_ID = 2 ** 31 - 1;
/** `pos` is clamped to this, as the core clamps it. */
const POS_LIMIT = 1000;

/**
 * What the graph says when it refuses, word for word (`docs/23` section 7.2, section 14 decision 17). The first six are the spec's; the
 * rest are the same kind of plain sentence for the refusals the spec lists but gives no words for (a node that cannot be moved, one that
 * is already first or last, the Input and the Output that stay as they are).
 */
export const GRAPH_OP_TEXT = {
  /** Dropping something on Main's picture input, or picking its link up. */
  mainInput: "Main always takes its picture from the input.",
  /** Taking the Output's only link away. */
  outputNeedsPicture: "The output needs a picture. Link another node to it instead.",
  /** A link, or a move, that would make a circle. */
  circle: "That would link the nodes in a circle.",
  /** Delete on Main. */
  mainDelete: "Main can't be deleted. Bypass it instead.",
  /** Add at 64 nodes. */
  tooMany: "A photo can have up to 64 nodes.",
  /** Bypass on the Input or the Output. */
  noBypass: "The input and the output can't be bypassed.",
  /** Show mask on the Input or the Output (the toolbar's tooltip). */
  noMask: "The input and the output have no mask.",
  /** Delete on the Input or the Output. */
  markerDelete: "The input and the output can't be deleted.",
  /** The core's sentence for a patch that removes the Input, the Output or Main. */
  cannotRemove: "The input, the output and node 1, Main, can't be removed.",
  /** A picture to a picture, a key to a key (the core's own sentence). */
  portTypes: "A link joins a picture to a picture, or a key to a key.",
  /** Reorder, Main. */
  mainFixed: "Main can't be moved. It always comes first.",
  /** Reorder, the Input and the Output. */
  markerFixed: "The input and the output stay where they are.",
  /** Reorder: into the link Input → Main, or earlier than Main. */
  beforeMain: "Nothing can go before Main.",
  /** Earlier, for a node that already takes its picture from the input. */
  alreadyFirst: "This node already takes its picture from the input.",
  /** Later, for a node with nothing after it, or only the output. */
  alreadyLast: "This node is already the last one before the output.",
  /** Rename of the Input or the Output. */
  noRename: "The input and the output can't be renamed.",
  /** A link that names a node or a port the graph does not have (a stale menu). */
  gone: "That link is not there any more.",
  /** Reorder of a Parallel, a Layer, a Splitter or a Combiner (`docs/25` §5.4). */
  glyphFixed: "This node can't be moved along a line. Change its links instead.",
  /** Bypass of a Splitter or a Combiner (`docs/25` §5.4). */
  glyphNoBypass: "A splitter or a combiner can't be bypassed.",
  /** Reset of a node with nothing to reset (the head's tooltip). */
  nothingToReset: "This node has nothing to reset.",
  /** Show mask on a node with no mask (`docs/25` §5.5). */
  glyphNoMask: "This node has no mask.",
} as const;

/** What an Add node item says when it is off with room for some nodes but not for all it adds (`docs/25` §5.1). */
export const addsText = (count: number): string => `This adds ${count === 2 ? "two" : "five"} nodes, and a photo can have up to 64.`;

// ---------------------------------------------------------------------------
// Links
// ---------------------------------------------------------------------------

/** A node's place in the order the core writes links in: the Input, the correctors by id, the Output. */
const rankOf = (id: NodeId): number => (id === INPUT_ID ? -1 : id === OUTPUT_ID ? Number.POSITIVE_INFINITY : id);

/** Orders two links as the core writes them: by the from-node, the from-port, the to-node and the to-port. */
export function compareLinks(a: NodeLink, b: NodeLink): number {
  const by = (x: number, y: number): number => (x === y ? 0 : x < y ? -1 : 1);
  return (
    by(rankOf(a.from.node), rankOf(b.from.node)) ||
    by(portRank(a.from.port), portRank(b.from.port)) ||
    by(rankOf(a.to.node), rankOf(b.to.node)) ||
    by(portRank(a.to.port), portRank(b.to.port))
  );
}

/** The links in the core's order (the same graph is the same bytes, hard rule 8). */
export const sortedLinks = (links: readonly NodeLink[]): NodeLink[] => [...links].sort(compareLinks);

/** True when two ends are the same node and port. */
export const sameRef = (a: NodeRef, b: NodeRef): boolean => a.node === b.node && a.port === b.port;

/** True when two links join the same ends. */
export const sameLink = (a: NodeLink, b: NodeLink): boolean => sameRef(a.from, b.from) && sameRef(a.to, b.to);

/** The link that feeds this input port, if it has one (an input has at most one). */
export const linkInto = (links: readonly NodeLink[], to: NodeRef): NodeLink | undefined => links.find((l) => sameRef(l.to, to));

/** The node whose output feeds this node's picture or key input, if one does. */
const sourceOf = (links: readonly NodeLink[], node: NodeId, port: Port): NodeId | undefined => linkInto(links, { node, port })?.from.node;

/** The links that leave a node's picture outputs (`picture`, or a Splitter's `red`, `green` and `blue`), in the core's order. */
const picturesOut = (links: readonly NodeLink[], node: NodeId): NodeLink[] =>
  sortedLinks(links.filter((l) => l.from.node === node && portType(l.from.port) === "picture"));

const picture = (node: NodeId): NodeRef => ({ node, port: "picture" });

/** The number of linked inputs of a mixer. */
const linkedInputs = (links: readonly NodeLink[], node: NodeId): number => links.filter((l) => l.to.node === node && inputNumber(l.to.port) !== null).length;

/**
 * The ports a kind of node has: `[inputs, outputs]` (`docs/23` section 3.4, `docs/25` section 2.2). A mixer's inputs are the linked ones
 * and one more, `picture.N+1`, which is what the "+" port adds: the next number, up to the eighth.
 */
function portsOf(settings: DevelopSettings, id: NodeId): { inputs: readonly Port[]; outputs: readonly Port[] } | null {
  const node = nodeOf(settings, id);
  if (!node) return null;
  switch (node.kind) {
    case "input":
      return { inputs: [], outputs: ["picture"] };
    case "output":
      return { inputs: ["picture"], outputs: [] };
    case "parallel":
    case "layer": {
      const next = Math.min(linkedInputs(settings.nodes.links, id) + 1, MIXER_INPUTS);
      return { inputs: Array.from({ length: next }, (_, i) => mixerPort(i + 1)), outputs: ["picture"] };
    }
    case "splitter":
      return { inputs: ["picture"], outputs: CHANNEL_PORTS };
    case "combiner":
      return { inputs: CHANNEL_PORTS, outputs: ["picture"] };
    default:
      return { inputs: ["picture", "key"], outputs: ["picture", "key"] };
  }
}

/** The output a node's picture leaves by when nothing says which: `picture`, a Splitter's `red`. */
const pictureOut = (node: NodeItem): Port => (node.kind === "splitter" ? "red" : "picture");

/**
 * The source of a node's first picture input (`docs/25` §5.4): the output that feeds a mixer's `picture.1`, a Combiner's `red` or any
 * other node's `picture`; the Input's picture when that input has no link.
 */
function firstSource(links: readonly NodeLink[], node: NodeItem): NodeRef {
  return linkInto(links, { node: node.id, port: firstInputPort(node.kind) })?.from ?? picture(INPUT_ID);
}

/**
 * The links with every mixer's inputs numbered from 1 with no gap, in their old order (`docs/25` §5.3): after a link was removed, picked
 * up and dropped elsewhere, or a node upstream went. Written in the core's order.
 */
export function renumbered(settings: DevelopSettings, links: readonly NodeLink[]): NodeLink[] {
  let out = [...links];
  for (const item of settings.nodes.items) {
    if (!isMixerKind(item.kind)) continue;
    const mine = out.filter((l) => l.to.node === item.id && inputNumber(l.to.port) !== null).sort((a, b) => portRank(a.to.port) - portRank(b.to.port));
    const renumber = new Map(mine.map((l, i) => [l, { ...l, to: { node: item.id, port: mixerPort(i + 1) } }]));
    out = out.map((l) => renumber.get(l) ?? l);
  }
  return sortedLinks(out);
}

/** True when these links, picture and key together, make a circle (Kahn's algorithm). */
export function hasCircle(settings: DevelopSettings, links: readonly NodeLink[]): boolean {
  const waiting = new Map<NodeId, number>(settings.nodes.items.map((item) => [item.id, 0]));
  for (const link of links) waiting.set(link.to.node, (waiting.get(link.to.node) ?? 0) + 1);
  const ready = [...waiting].filter(([, n]) => n === 0).map(([id]) => id);
  let seen = 0;
  while (ready.length > 0) {
    const id = ready.pop() as NodeId;
    seen += 1;
    for (const link of links) {
      if (link.from.node !== id) continue;
      const left = (waiting.get(link.to.node) ?? 0) - 1;
      waiting.set(link.to.node, left);
      if (left === 0) ready.push(link.to.node);
    }
  }
  return seen < waiting.size;
}

/** True when a chain of picture links leads from this node to the Output. */
function leadsToOutput(links: readonly NodeLink[], node: NodeId): boolean {
  const seen = new Set<NodeId>();
  const todo: NodeId[] = [node];
  while (todo.length > 0) {
    const id = todo.pop() as NodeId;
    if (id === OUTPUT_ID) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const link of picturesOut(links, id)) todo.push(link.to.node);
  }
  return false;
}

/** Every node a node's links lead to, itself left out: what is downstream of it (picture and key links alike). */
function downstreamOf(links: readonly NodeLink[], node: NodeId): Set<NodeId> {
  const seen = new Set<NodeId>();
  const todo: NodeId[] = [node];
  while (todo.length > 0) {
    const id = todo.pop() as NodeId;
    for (const link of links) {
      if (link.from.node !== id || seen.has(link.to.node)) continue;
      seen.add(link.to.node);
      todo.push(link.to.node);
    }
  }
  seen.delete(node);
  return seen;
}

// ---------------------------------------------------------------------------
// The plan's numbers and live nodes (`docs/23` section 5.2, `crates/ym-color/src/plan.rs`)
// ---------------------------------------------------------------------------

/** What the plan says about one numbered node. */
export interface CardState {
  /** The node's place in the flow, counted from 1 (the number on the card). A Parallel, a Layer, a Splitter and a Combiner have no number: 0. */
  number: number;
  /** The Output needs this node, or a live node needs its key. */
  live: boolean;
}

/**
 * Every numbered node's number and whether it is live, as the core's plan has them. The order is a topological one over the picture and key
 * links together, with ties broken by Main first, then `pos` (x, then y), then the id; the number is the place in that order over the
 * correctors and Film nodes, dead ones included, **skipping the four kinds of `docs/25`** (they are in the order, they have no number).
 * A node is live when the Output needs it: it is on the chain of picture links back from the Output to the Input, through every input of
 * a mixer, a Combiner's three and a Splitter's one, or its key output feeds a live node's key input (and so on). A picture input with no
 * link reads the Input.
 */
export function cardStates(settings: DevelopSettings): Map<number, CardState> {
  const links = settings.nodes.links;
  const nodes = numberedOf(settings);
  const byId = new Map(nodes.map((n) => [n.id as number, n]));
  const needs = (node: NodeItem): number[] => {
    const out: number[] = [];
    for (const link of links) {
      if (link.to.node !== node.id) continue;
      const from = link.from.node;
      if (typeof from === "number" && byId.has(from)) out.push(from);
    }
    return out;
  };
  const precedes = (a: NodeItem, b: NodeItem): number =>
    Number(a.id !== MAIN_ID) - Number(b.id !== MAIN_ID) || a.pos[0] - b.pos[0] || a.pos[1] - b.pos[1] || (a.id as number) - (b.id as number);

  const waiting = new Map<number, Set<number>>(nodes.map((n) => [n.id as number, new Set(needs(n))]));
  const order: number[] = [];
  while (waiting.size > 0) {
    const ready = [...waiting].filter(([, deps]) => deps.size === 0).map(([id]) => byId.get(id) as NodeItem);
    // A validated graph has no circle; a broken one still gets a total order rather than a loop.
    const pool = ready.length > 0 ? ready : [...waiting.keys()].map((id) => byId.get(id) as NodeItem);
    const next = pool.reduce((best, n) => (precedes(n, best) < 0 ? n : best));
    waiting.delete(next.id as number);
    for (const deps of waiting.values()) deps.delete(next.id as number);
    order.push(next.id as number);
  }

  const live = new Set<number>();
  const todo: number[] = [];
  const root = sourceOf(links, OUTPUT_ID, "picture");
  if (typeof root === "number") todo.push(root);
  while (todo.length > 0) {
    const id = todo.pop() as number;
    if (live.has(id) || !byId.has(id)) continue;
    live.add(id);
    todo.push(...needs(byId.get(id) as NodeItem));
  }
  let counted = 0;
  return new Map(order.map((id) => [id, { number: isGlyphKind((byId.get(id) as NodeItem).kind) ? 0 : ++counted, live: live.has(id) }]));
}

/** The number on a card, 1 for Main, or undefined for the Input and the Output. */
export function cardNumbers(settings: DevelopSettings): Map<NodeId, number> {
  return new Map([...cardStates(settings)].map(([id, state]) => [id, state.number]));
}

/** "01", "12": a number as the card's chip shows it. */
export const chipText = (n: number): string => String(n).padStart(2, "0");

// ---------------------------------------------------------------------------
// What an operation answers
// ---------------------------------------------------------------------------

/** What to send, and which node to select afterwards (none: the selection stays). An empty patch means nothing needs doing. */
export interface GraphEdit {
  patch: SettingsPatch;
  select?: NodeId;
}

/** An operation's answer: a patch to send, or the sentence that says why it is refused (the error bar shows it, nothing changes). */
export type GraphOutcome = ({ ok: true } & GraphEdit) | { ok: false; reason: string };

const done = (patch: SettingsPatch, select?: NodeId): GraphOutcome => (select === undefined ? { ok: true, patch } : { ok: true, patch, select });
const refuse = (reason: string): GraphOutcome => ({ ok: false, reason });
const nothing: GraphOutcome = { ok: true, patch: {} };

/** True when an outcome has something to send. */
export const hasPatch = (outcome: GraphOutcome): outcome is { ok: true } & GraphEdit => outcome.ok && !isEmptyPatch(outcome.patch);

// ---------------------------------------------------------------------------
// Add, delete, bypass, rename, move
// ---------------------------------------------------------------------------

/** True for a Parallel, a Layer, a Splitter or a Combiner. */
const isGlyph = (node: NodeItem): node is MixerItem => isGlyphKind(node.kind);

/** True for a node that has a place on the canvas the page moves: every node but the Input and the Output. */
const hasPlace = (node: NodeItem): node is CardItem | MixerItem => node.kind !== "input" && node.kind !== "output";

/** The id a new node gets: one more than the largest the page has used in the session (`usedMax`), so a deleted id is not reused. */
export function nextNodeId(settings: DevelopSettings, usedMax = 0): number {
  const largest = Math.max(usedMax, ...settings.nodes.items.map((item) => (typeof item.id === "number" ? item.id : 0)));
  return Math.min(largest + 1, MAX_ID);
}

/** The picture link a node is put into: the one that leads to the Output, else the first (the core's order). */
function linkToPutInto(links: readonly NodeLink[], node: NodeId): NodeLink | undefined {
  const outgoing = picturesOut(links, node);
  return outgoing.find((l) => leadsToOutput(links, l.to.node)) ?? outgoing[0];
}

const clampPos = (pos: NodePos): NodePos => [Math.min(POS_LIMIT, Math.max(-POS_LIMIT, pos[0])), Math.min(POS_LIMIT, Math.max(-POS_LIMIT, pos[1]))];

/** `pos` patches that move every node in `ids` `delta` units to the right. The Input and the Output have no place the page moves. */
function shifted(settings: DevelopSettings, ids: ReadonlySet<NodeId>, skip: readonly NodeId[] | NodeId, delta: number): Record<string, { pos: NodePos }> {
  const out: Record<string, { pos: NodePos }> = {};
  if (delta <= 0) return out;
  const skipped = new Set(Array.isArray(skip) ? skip : [skip]);
  for (const item of settings.nodes.items) {
    if (!hasPlace(item) || skipped.has(item.id) || !ids.has(item.id)) continue;
    out[String(item.id)] = { pos: clampPos([item.pos[0] + delta, item.pos[1]]) };
  }
  return out;
}

/**
 * Where a node goes when it is put after `after`, into the link `into`: one unit to the right of `after` (on its row), and the nodes
 * downstream of it, the one it goes before included, far enough right to leave it its unit (one unit when they were next to it; none when
 * a gap that a delete or an earlier move left is there to be used, so moving a node back and forth never spreads the graph). After the
 * Input, which has no place of its own, it takes the place of the node it goes before, and that node and what follows move one unit right.
 */
function slotFor(settings: DevelopSettings, after: NodeId, into: NodeLink | undefined): { pos: NodePos; delta: number } | null {
  const base = nodeOf(settings, after);
  if (!base) return null;
  const before = into ? nodeOf(settings, into.to.node) : undefined;
  if (base.kind === "input") {
    return before && hasPlace(before) ? { pos: clampPos([before.pos[0], before.pos[1]]), delta: 1 } : { pos: clampPos([0, 0]), delta: 0 };
  }
  const pos = clampPos([base.pos[0] + 1, base.pos[1]]);
  const delta = before && hasPlace(before) ? Math.max(0, pos[0] + 1 - before.pos[0]) : 0;
  return { pos, delta };
}

/**
 * A new node and the links around it: the node `id` goes after `after`, into `into` (the picture link that was there: `after → to` becomes
 * `after → id → to`, from the port that link left by), or linked after `after` alone when there is none. The links of `id`'s other ports
 * are not touched.
 */
function putAfter(settings: DevelopSettings, links: readonly NodeLink[], id: number, after: NodeId, into: NodeLink | undefined): NodeLink[] {
  const kept = links.filter((l) => !into || !sameLink(l, into));
  const item = nodeOf(settings, after);
  const from: NodeRef = into ? into.from : { node: after, port: item ? pictureOut(item) : "picture" };
  const added: NodeLink[] = [{ from, to: picture(id) }];
  if (into) added.push({ from: picture(id), to: into.to });
  return sortedLinks([...kept, ...added]);
}

/** A new corrector node, manual 0 everywhere, as a patch that adds it whole (`docs/23` section 3.2: a new node is manual 0). */
export function addNode(id: number, pos: NodePos, name = DEFAULT_NAME): Record<string, NodePatch> {
  return { [String(id)]: nodeAsPatch(newCorrector(id, clampPos(pos), name)) };
}

/** A new Film node (`docs/24` section 4.5), Portra 400 on its matching paper, as a patch that adds it whole. */
export function addFilmNode(id: number, pos: NodePos, name = FILM_DEFAULT_NAME): Record<string, NodePatch> {
  return { [String(id)]: nodeAsPatch(newFilm(id, clampPos(pos), name)) };
}

/**
 * **Add after the selected node** (`docs/23` section 7.2): the new node goes into the selected node's outgoing picture link. With S
 * selected and a link S → T the result is S → N → T; with several such links N goes into the one that leads to the Output, else the first;
 * with none N is linked after S alone. With the Input selected N goes after Main (Main stays first); with the Output selected, into the
 * Output's incoming link. N's `pos` is the node's plus one unit to the right, the nodes downstream of N move one unit right (as far as
 * it takes to leave N its unit: none where a gap is), and N is selected. Refused at 64 nodes.
 */
export function insertAfter(settings: DevelopSettings, selected: NodeId, options: { usedMax?: number; name?: string; kind?: "corrector" | "film" } = {}): GraphOutcome {
  if (numberedOf(settings).length >= MAX_CORRECTORS) return refuse(GRAPH_OP_TEXT.tooMany);
  const links = settings.nodes.links;
  let after: NodeId = selected === INPUT_ID ? MAIN_ID : selected;
  let into: NodeLink | undefined;
  if (after === OUTPUT_ID) {
    into = linkInto(links, picture(OUTPUT_ID));
    after = into?.from.node ?? MAIN_ID;
  } else {
    into = linkToPutInto(links, after);
  }
  const slot = slotFor(settings, after, into);
  if (!slot) return refuse(GRAPH_OP_TEXT.gone);
  const id = nextNodeId(settings, options.usedMax);
  const next = putAfter(settings, links, id, after, into);
  const added = options.kind === "film" ? addFilmNode(id, slot.pos, options.name) : addNode(id, slot.pos, options.name);
  const nodes = { ...added, ...shifted(settings, downstreamOf(next, id), id, slot.delta) };
  return done({ nodes, links: next }, id);
}

/**
 * **Delete** (`docs/23` section 7.2, `docs/25` §5.4): any node but Main. The neighbours are reconnected: with U → D → T₁ … Tₙ the result is
 * U → Tᵢ for each, where U is the source of D's **first picture input** (`picture` of a corrector, a Film node or a Splitter, `picture.1`
 * of a mixer, `red` of a Combiner; the Input when that input has no link) and Tᵢ are the nodes D fed through any of its picture outputs.
 * D's other links go, a key's with it (a node that had D's key as its key input loses that input), and the mixers it fed are numbered from 1
 * again. The node U comes from is selected (Main when it is the Input).
 */
export function remove(settings: DevelopSettings, id: NodeId): GraphOutcome {
  if (id === MAIN_ID) return refuse(GRAPH_OP_TEXT.mainDelete);
  if (id === INPUT_ID || id === OUTPUT_ID) return refuse(GRAPH_OP_TEXT.markerDelete);
  const node = nodeOf(settings, id);
  if (!node) return refuse(GRAPH_OP_TEXT.gone);
  const links = settings.nodes.links;
  const before = firstSource(links, node);
  const rest = sortedLinks([...links.filter((l) => l.from.node !== id && l.to.node !== id), ...picturesOut(links, id).map((l) => ({ from: before, to: l.to }))]);
  const gone = { ...settings, nodes: { ...settings.nodes, items: settings.nodes.items.filter((item) => item.id !== id) } };
  return done({ remove: [id], links: renumbered(gone, rest) }, before.node === INPUT_ID ? MAIN_ID : before.node);
}

/**
 * **Bypass** (`docs/23` section 7.2, `docs/25` §5.4): toggles a corrector's, a Film node's, a Parallel's or a Layer's bypass, Main's too;
 * refused for the Input and the Output, and for a Splitter and a Combiner, which have none.
 */
export function toggleBypass(settings: DevelopSettings, id: NodeId): GraphOutcome {
  const node = nodeOf(settings, id);
  if (!node) return refuse(GRAPH_OP_TEXT.gone);
  if (node.kind === "splitter" || node.kind === "combiner") return refuse(GRAPH_OP_TEXT.glyphNoBypass);
  if (node.kind === "input" || node.kind === "output") return refuse(GRAPH_OP_TEXT.noBypass);
  return done({ nodes: { [String(id)]: { bypass: !node.bypass } } });
}

/**
 * **Reset node** (`docs/23` §3.7 and §7.5, the head's Reset): puts the node's values at their defaults and nothing else, as one patch. A
 * corrector's trim and grade go back (Main's to Auto where there is Auto, the others' to a manual 0); **its masks, its name, its bypass and
 * its links stay** (the Masks card has its own Reset). The Input goes back to every default it has (Auto where there is Auto), and the
 * Output's look (`look`, the user's default look, Neutral where the page does not know it), its intensity and its output sharpening.
 * A node that is already at its defaults answers an empty patch.
 */
export function resetNode(settings: DevelopSettings, id: NodeId, defaults: { look?: LookId } = {}): GraphOutcome {
  const node = nodeOf(settings, id);
  if (!node) return refuse(GRAPH_OP_TEXT.gone);
  let patch: NodePatch;
  if (node.kind === "input") {
    const { kind: _kind, id: _id, pos: _pos, ...fields } = defaultInput();
    patch = fields;
  } else if (node.kind === "output") {
    patch = { look: defaults.look ?? NEUTRAL_LOOK, look_intensity: 100, output_sharpening: 100 };
  } else if (node.kind === "film") {
    // Every number back to its default; the stock goes back to Portra 400 on its matching paper. The grain's seed, the masks, the name and the bypass stay.
    patch = {
      film: { stock: DEFAULT_STOCK, exposure_ev: 0 },
      print: { paper: null, exposure_ev: 0, lights: { c: 0, m: 0, y: 0 } },
      halation: { amount: 100, size: 100 },
      grain: { amount: 100, size: 100 },
      strength: 100,
    };
  } else if (node.kind === "layer") {
    // A Layer's Mode and Opacity; its inputs, name and bypass stay.
    patch = { mode: "normal", opacity: 1 };
  } else if (isGlyphKind(node.kind)) {
    // A Parallel, a Splitter and a Combiner have nothing to reset (`docs/25` §5.5).
    return nothing;
  } else {
    patch = { trim: { exposure_ev: 0, temperature: 0, tint: 0 }, grade: node.id === MAIN_ID ? defaultGrade() : neutralGrade() };
  }
  const settingsPatch: SettingsPatch = { nodes: { [String(id)]: patch } };
  return same(applyPatch(settings, settingsPatch), settings) ? nothing : done(settingsPatch);
}

/** A name as a node keeps it: trimmed and cut at 32 characters. */
export const cleanName = (name: string): string => Array.from(name.trim()).slice(0, MAX_NAME_LENGTH).join("");

/**
 * **Rename** (`docs/23` section 7.2): 1 to 32 characters after trimming, a longer name is cut. A name that is empty after trimming, or
 * the one the node has, changes nothing (the editor then simply closes). The Input and the Output keep their fixed names.
 */
export function rename(settings: DevelopSettings, id: NodeId, name: string): GraphOutcome {
  const node = nodeOf(settings, id);
  if (!node) return refuse(GRAPH_OP_TEXT.gone);
  if (node.kind === "input" || node.kind === "output") return refuse(GRAPH_OP_TEXT.noRename);
  const clean = cleanName(name);
  return clean === "" || clean === node.name ? nothing : done({ nodes: { [String(id)]: { name: clean } } });
}

/** `pos` as the page keeps it: two decimals (a hundredth of a unit is about two pixels), inside the core's bounds. */
const roundPos = (pos: NodePos): NodePos => clampPos([Math.round(pos[0] * 100) / 100, Math.round(pos[1] * 100) / 100]);

/** **Move a card** (`docs/23` section 7.2): `pos` only, nothing is rewired. A card that is where it was changes nothing. */
export function moveCard(settings: DevelopSettings, id: NodeId, pos: NodePos): GraphOutcome {
  const node = nodeOf(settings, id);
  if (!node) return refuse(GRAPH_OP_TEXT.gone);
  const next = roundPos(pos);
  if (next[0] === node.pos[0] && next[1] === node.pos[1]) return nothing;
  return done({ nodes: { [String(id)]: { pos: next } } });
}

// ---------------------------------------------------------------------------
// Reorder
// ---------------------------------------------------------------------------

/**
 * The links with a node taken out of its place: its picture links go, and its source feeds each of the nodes it fed (a mixer's input
 * keeps its number). Its key links stay.
 */
function takeOut(settings: DevelopSettings, links: readonly NodeLink[], id: number): NodeLink[] {
  const node = nodeOf(settings, id);
  const before: NodeRef = node ? firstSource(links, node) : picture(INPUT_ID);
  const kept = links.filter((l) => !((l.from.node === id && portType(l.from.port) === "picture") || (l.to.node === id && l.to.port === "picture")));
  return [...kept, ...picturesOut(links, id).map((l) => ({ from: before, to: l.to }))];
}

/** Why a node cannot be reordered, or null when it can. */
function immovable(settings: DevelopSettings, id: NodeId): string | null {
  if (id === MAIN_ID) return GRAPH_OP_TEXT.mainFixed;
  if (id === INPUT_ID || id === OUTPUT_ID) return GRAPH_OP_TEXT.markerFixed;
  const node = nodeOf(settings, id);
  if (!node) return GRAPH_OP_TEXT.gone;
  return isGlyphKind(node.kind) ? GRAPH_OP_TEXT.glyphFixed : null;
}

/**
 * Puts a node (already out of its place, in `links`) after `after`, into `into` or alone, and answers the patch: the links, the node's
 * `pos` (the node it follows plus one unit) and the nodes downstream of it, one unit right. A circle through key links is refused.
 */
function placed(settings: DevelopSettings, links: readonly NodeLink[], id: number, after: NodeId, into: NodeLink | undefined): GraphOutcome {
  const next = putAfter(settings, links, id, after, into);
  if (hasCircle(settings, next)) return refuse(GRAPH_OP_TEXT.circle);
  const slot = slotFor(settings, after, into);
  if (!slot) return refuse(GRAPH_OP_TEXT.gone);
  const nodes: Record<string, NodePatch> = { [String(id)]: { pos: slot.pos } };
  Object.assign(nodes, shifted(settings, downstreamOf(next, id), id, slot.delta));
  return done({ nodes, links: next }, id);
}

function sameLinks(a: readonly NodeLink[], b: readonly NodeLink[]): boolean {
  return a.length === b.length && a.every((l, i) => sameLink(l, b[i] as NodeLink));
}

/**
 * **Reorder by dropping** (`docs/23` section 7.2): a card released with its centre over another picture link. The node is taken out of its
 * place and the neighbours reconnected, then put into that link (the same two operations as Delete and Add); its key links stay. Refused:
 * Main (it cannot be reordered), the link Input → Main (nothing goes before Main), and a move that would make a circle (the node's key
 * output feeds a node that would then be upstream of it, or its key input comes from one that would be downstream). A link of the node
 * itself is where it already is, so nothing changes.
 */
export function reorderInto(settings: DevelopSettings, id: NodeId, target: NodeLink): GraphOutcome {
  const why = immovable(settings, id);
  if (why) return refuse(why);
  const links = settings.nodes.links;
  if (!links.some((l) => sameLink(l, target)) || portType(target.from.port) !== "picture") return refuse(GRAPH_OP_TEXT.gone);
  if (target.to.node === MAIN_ID) return refuse(GRAPH_OP_TEXT.beforeMain);
  if (target.from.node === id || target.to.node === id) return nothing;
  return placed(settings, takeOut(settings, links, id as number), id as number, target.from.node, target);
}

/**
 * **Move earlier** (More › Move earlier, ⌥←): the node goes before the one it takes its picture from, into the link that feeds that one.
 * Nothing goes before Main, and a node that takes its picture from the input is already as early as it can be.
 */
export function moveEarlier(settings: DevelopSettings, id: NodeId): GraphOutcome {
  const why = immovable(settings, id);
  if (why) return refuse(why);
  const links = settings.nodes.links;
  const previous = sourceOf(links, id, "picture");
  if (previous === undefined || previous === INPUT_ID) return refuse(GRAPH_OP_TEXT.alreadyFirst);
  if (previous === MAIN_ID) return refuse(GRAPH_OP_TEXT.beforeMain);
  // Into the link that feeds the previous node's first picture input; when that node is a mixer, a Splitter or a Combiner too (`docs/25` §5.4).
  const was = nodeOf(settings, previous);
  const firstInput: NodeRef = { node: previous, port: was ? firstInputPort(was.kind) : "picture" };
  const rest = takeOut(settings, links, id as number);
  const into = linkInto(rest, firstInput) ?? { from: picture(INPUT_ID), to: firstInput };
  return placed(settings, rest, id as number, into.from.node, into);
}

/**
 * **Move later** (More › Move later, ⌥→): the node goes after the one it feeds (the one that leads to the Output, else the first), into that
 * node's outgoing link or alone after it. A node with nothing after it, or only the Output, is already the last.
 */
export function moveLater(settings: DevelopSettings, id: NodeId): GraphOutcome {
  const why = immovable(settings, id);
  if (why) return refuse(why);
  const links = settings.nodes.links;
  const out = linkToPutInto(links, id);
  const next = out?.to.node;
  if (next === undefined || next === OUTPUT_ID) return refuse(GRAPH_OP_TEXT.alreadyLast);
  const rest = takeOut(settings, links, id as number);
  return placed(settings, rest, id as number, next, linkToPutInto(rest, next));
}

/** Whether Move earlier and Move later are on offer for a node (the More menu greys them), with the sentence that says why not. */
export function reorderAvailability(settings: DevelopSettings, id: NodeId): { earlier: string | null; later: string | null } {
  const earlier = moveEarlier(settings, id);
  const later = moveLater(settings, id);
  return { earlier: earlier.ok ? null : earlier.reason, later: later.ok ? null : later.reason };
}

// ---------------------------------------------------------------------------
// Links drawn by hand
// ---------------------------------------------------------------------------

/**
 * Makes, replaces or moves one link: `from → to`, with `lifted` the link whose end was picked up (it goes, and so does whatever was on the
 * input it lands on). The rules of `docs/23` section 7.2, in this order: the ports must be an output and an input of the same type; the
 * link Input → Main cannot be moved or replaced, and Main's picture input takes only the Input; the Output's only picture link can be
 * replaced but not taken away; no circle, picture and key links together.
 */
function relink(settings: DevelopSettings, from: NodeRef, to: NodeRef, lifted?: NodeLink): GraphOutcome {
  const start = portsOf(settings, from.node);
  const end = portsOf(settings, to.node);
  if (!start || !end) return refuse(GRAPH_OP_TEXT.gone);
  if (!start.outputs.includes(from.port) || !end.inputs.includes(to.port) || portType(from.port) !== portType(to.port)) return refuse(GRAPH_OP_TEXT.portTypes);
  const links = settings.nodes.links;
  if (lifted && lifted.to.node === MAIN_ID && lifted.to.port === "picture") return refuse(GRAPH_OP_TEXT.mainInput);
  if (to.node === MAIN_ID && to.port === "picture" && from.node !== INPUT_ID) return refuse(GRAPH_OP_TEXT.mainInput);
  if (lifted && lifted.to.node === OUTPUT_ID && !(to.node === OUTPUT_ID && to.port === "picture")) return refuse(GRAPH_OP_TEXT.outputNeedsPicture);
  const kept = links.filter((l) => !(lifted && sameLink(l, lifted)) && !sameRef(l.to, to));
  // A drop makes the new link first and then takes the lifted one away; the mixers' inputs are then numbered from 1 again (`docs/25` §5.3).
  const next = renumbered(settings, [...kept, { from, to }]);
  if (sameLinks(next, sortedLinks(links))) return nothing;
  if (hasCircle(settings, next)) return refuse(GRAPH_OP_TEXT.circle);
  return done({ links: next });
}

/**
 * **Connect** (`docs/23` section 7.2): link an output port to an input port. An input has one link, so a drop onto an input that has one
 * replaces it, in the same step; an output may feed any number of inputs.
 */
export const connect = (settings: DevelopSettings, from: NodeRef, to: NodeRef): GraphOutcome => relink(settings, from, to);

/** **Pick up a link's end and drop it on another input** (the output end stays): the link moves, replacing what was on that input. */
export const reconnect = (settings: DevelopSettings, link: NodeLink, to: NodeRef): GraphOutcome => relink(settings, link.from, to, link);

/**
 * **Remove a link** (a link's end dropped on the empty canvas, Delete on a selected link, right click › Delete link, More › Remove link).
 * Refused: Input → Main, which is fixed, and the Output's picture link, which can be replaced but not taken away.
 */
export function disconnect(settings: DevelopSettings, link: NodeLink): GraphOutcome {
  const links = settings.nodes.links;
  if (!links.some((l) => sameLink(l, link))) return refuse(GRAPH_OP_TEXT.gone);
  if (link.to.node === MAIN_ID && link.to.port === "picture") return refuse(GRAPH_OP_TEXT.mainInput);
  if (link.to.node === OUTPUT_ID) return refuse(GRAPH_OP_TEXT.outputNeedsPicture);
  return done({ links: renumbered(settings, links.filter((l) => !sameLink(l, link))) });
}

/** One input port, and whether a link dragged from an output may land on it. */
export interface InputState {
  to: NodeRef;
  /** The input lights up: the same type, no circle, and the rules above allow it. */
  ok: boolean;
  /** Why not, in the sentence the error bar shows when the link is dropped there anyway. */
  reason?: string;
}

/**
 * **Which inputs light up** while a link is dragged from `from` (or while `lifted`'s end is carried: that input is free again, so putting
 * it back is allowed and changes nothing). Every input port of every node, in the order the document lists the nodes: the same type and
 * no circle light up, the others dim, each with the reason it is refused.
 */
export function inputStates(settings: DevelopSettings, from: NodeRef, lifted?: NodeLink): InputState[] {
  const out: InputState[] = [];
  for (const item of settings.nodes.items) {
    const ports = portsOf(settings, item.id);
    for (const port of ports?.inputs ?? []) {
      const to: NodeRef = { node: item.id, port };
      const outcome = relink(settings, from, to, lifted);
      out.push(outcome.ok ? { to, ok: true } : { to, ok: false, reason: outcome.reason });
    }
  }
  return out;
}

/**
 * What the More menu lists for "Link picture to" and "Link key to": the inputs that can take a link from this node's port (the same type,
 * no circle, the rules above), leaving out the ones it is already linked to.
 */
export function linkTargets(settings: DevelopSettings, from: NodeRef): NodeRef[] {
  if (!portsOf(settings, from.node)?.outputs.includes(from.port)) return [];
  return inputStates(settings, from)
    .filter((s) => {
      const current = linkInto(settings.nodes.links, s.to);
      return s.ok && !(current && sameRef(current.from, from));
    })
    .map((s) => s.to);
}

/** The links a node has: the ones that feed it and the ones that leave it, in the core's order. This is what "Remove link" lists. */
export function linksOf(settings: DevelopSettings, id: NodeId): NodeLink[] {
  return sortedLinks(settings.nodes.links.filter((l) => l.from.node === id || l.to.node === id));
}

// ---------------------------------------------------------------------------
// Names for the cards and the menus
// ---------------------------------------------------------------------------

/** What a mask row is called in a sentence (`docs/23` section 7.6). */
export const MASK_KIND_NAME: Readonly<Record<string, string>> = {
  object: "Object",
  subject: "Subject",
  people: "People",
  background: "Background",
  depth: "Depth",
  linear: "Linear",
  radial: "Radial",
  brush: "Brush",
  color_range: "Colour range",
  brightness_range: "Brightness range",
};

/** "Object", "Object and Radial", "Object, Radial and Linear": a list as a sentence. */
export function joinWords(words: readonly string[]): string {
  if (words.length <= 1) return words.join("");
  return `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`;
}

/** The kinds of a node's enabled mask rows, in the order of the rows, each named once. */
export function maskKindNames(node: Pick<CardItem, "masks">): string[] {
  const names: string[] = [];
  for (const row of node.masks.items) {
    const name = MASK_KIND_NAME[row.kind] ?? row.kind;
    if (row.enabled && !names.includes(name)) names.push(name);
  }
  return names;
}

/**
 * The quiet line of a node's head (`docs/23` §7.5). The Input and the Output say what they do; a corrector says what it acts on, after
 * "Corrector ·": "whole photo" with no mask rows, the kind of the row when it has one ("Linear"), "2 masks" from two rows, and "bypassed" or
 * "not connected" before either (a bypassed node does nothing, a dead one is not in the picture).
 */
export function headLine(settings: DevelopSettings, id: NodeId, states: ReadonlyMap<number, CardState> = cardStates(settings)): string {
  const node = nodeOf(settings, id);
  if (!node) return "";
  if (node.kind === "input") return "Turns the camera's data into the scene.";
  if (node.kind === "output") return "Cine rendering, look and export";
  if (isGlyph(node)) {
    const dead = states.get(node.id as number)?.live === false;
    if (node.kind === "splitter") return dead ? "Channels · not connected" : "Channels · 3 outputs";
    if (node.kind === "combiner") return dead ? "Channels · not connected" : "Channels · 3 inputs";
    if (node.bypass) return "Mixer · bypassed";
    if (dead) return "Mixer · not connected";
    const n = Math.max(1, linkedInputs(settings.nodes.links, node.id));
    return `Mixer · ${n} ${n === 1 ? "input" : "inputs"}`;
  }
  const label = node.kind === "film" ? "Film" : "Corrector";
  if (node.bypass) return `${label} · bypassed`;
  if (states.get(node.id as number)?.live === false) return `${label} · not connected`;
  const rows = node.masks.items;
  if (rows.length === 0) return `${label} · whole photo`;
  const only = rows[0];
  if (rows.length === 1 && only) return `${label} · ${MASK_KIND_NAME[only.kind] ?? only.kind}`;
  return `${label} · ${rows.length} masks`;
}

/**
 * How a source is named in a card's accessible name (`docs/25` §5.6): a numbered node as "node 5", the Input as "the input", a node with
 * no number by its name ("Parallel"), and a Splitter's output with its channel ("Splitter, red").
 */
function sourceName(settings: DevelopSettings, from: NodeRef, states: ReadonlyMap<number, CardState>): string {
  if (from.node === INPUT_ID) return "the input";
  const item = nodeOf(settings, from.node);
  if (!item) return "";
  if (isGlyph(item)) return item.kind === "splitter" ? `${item.name}, ${from.port}` : item.name;
  return `node ${states.get(item.id as number)?.number ?? 0}`;
}

/**
 * A card's accessible name (`docs/23` section 7.8, `docs/25` §5.6): "Node 2, Mountain, masked by Object and Radial" (the number as a plain
 * number, "Node 1, Main" with no masks), then ", bypassed" and ", not connected" where they apply, and what feeds it: ", picture from node 1,
 * key from node 5" (a source with no number by its name: ", picture from Layer"). A Parallel or a Layer is "Layer, mixer with 2 inputs, from node 5
 * and node 6"; a Splitter "Splitter, picture from node 1"; a Combiner "Combiner, red from node 3, green from node 4, blue from node 5". The
 * Input and the Output are "Input" and "Output".
 */
export function cardName(settings: DevelopSettings, id: NodeId, states: ReadonlyMap<number, CardState> = cardStates(settings)): string {
  const node = nodeOf(settings, id);
  if (!node) return "";
  if (node.kind === "input") return "Input";
  if (node.kind === "output") return "Output";
  const links = settings.nodes.links;
  const dead = states.get(node.id as number)?.live === false;
  if (isGlyph(node)) {
    const parts: string[] = [node.name];
    const flags = [...("bypass" in node && node.bypass ? ["bypassed"] : []), ...(dead ? ["not connected"] : [])];
    if (isMixerKind(node.kind)) {
      const feeders = inputsOf(settings, node.id).filter((row) => row.link !== null);
      const n = Math.max(1, feeders.length);
      parts.push(`mixer with ${n} ${n === 1 ? "input" : "inputs"}`, ...flags);
      if (feeders.length > 0) parts.push(`from ${joinWords(feeders.map((row) => sourceName(settings, (row.link as NodeLink).from, states)))}`);
    } else {
      parts.push(...flags);
      for (const port of node.kind === "splitter" ? (["picture"] as const) : CHANNEL_PORTS) {
        const from = linkInto(links, { node: node.id, port })?.from;
        if (from === undefined) {
          if (node.kind === "combiner") parts.push(`${port} from the input`);
          continue;
        }
        parts.push(`${port} from ${sourceName(settings, from, states)}`);
      }
    }
    return parts.join(", ");
  }
  const number = states.get(node.id as number)?.number ?? 0;
  const masks = maskKindNames(node);
  const parts = [`Node ${number}`, node.name];
  if (masks.length > 0) parts.push(`masked by ${joinWords(masks)}`);
  if (node.bypass) parts.push("bypassed");
  if (dead) parts.push("not connected");
  for (const port of ["picture", "key"] as const) {
    const from = linkInto(links, { node: node.id, port })?.from;
    if (from === undefined || from.node === INPUT_ID) continue;
    const name = sourceName(settings, from, states);
    if (name !== "") parts.push(`${port} from ${name}`);
  }
  return parts.join(", ");
}

/** "Node 3, Sky" or "Output", and a node with no number by its name ("Layer"): how the More menu names a node in a list. */
export function listName(settings: DevelopSettings, id: NodeId, states: ReadonlyMap<number, CardState> = cardStates(settings)): string {
  const node = nodeOf(settings, id);
  if (!node) return "";
  if (node.kind === "input") return "Input";
  if (node.kind === "output") return "Output";
  if (isGlyph(node)) return node.name;
  return `Node ${states.get(node.id as number)?.number ?? 0}, ${node.name}`;
}

/**
 * One end of a link as the More menu names it (`docs/25` §5.3): "Node 3, Sky", and for the four kinds the port too: "Layer, input 2",
 * "Layer, a new input" (the next number), "Combiner, red", "Splitter, red". The Splitter's input and the outputs of a mixer and a Combiner
 * are the node's name alone.
 */
export function endName(settings: DevelopSettings, end: NodeRef, side: "in" | "out", states: ReadonlyMap<number, CardState> = cardStates(settings)): string {
  const node = nodeOf(settings, end.node);
  const base = listName(settings, end.node, states);
  if (!node) return base;
  if (isMixerKind(node.kind) && side === "in") {
    const n = inputNumber(end.port);
    const linked = linkedInputs(settings.nodes.links, node.id);
    return n !== null && n > linked ? `${base}, a new input` : `${base}, ${portWords(end.port)}`;
  }
  if ((node.kind === "splitter" && side === "out") || (node.kind === "combiner" && side === "in")) return `${base}, ${end.port}`;
  return base;
}

/**
 * Where a link comes from or goes to, in the words of the panels (`docs/25` §5.5): "the input", "the output", "node 3, Red", and for the
 * four kinds their name with the port ("Layer, input 2", "Splitter, red", "Combiner, red"; a mixer's output and a Splitter's input are the
 * name alone).
 */
export function describeEnd(settings: DevelopSettings, end: NodeRef, states: ReadonlyMap<number, CardState> = cardStates(settings)): string {
  const node = nodeOf(settings, end.node);
  if (end.node === INPUT_ID) return "the input";
  if (end.node === OUTPUT_ID) return "the output";
  if (!node) return "";
  if (isGlyph(node)) {
    const n = inputNumber(end.port);
    if (isMixerKind(node.kind) && n !== null) return `${node.name}, input ${n}`;
    return node.kind === "splitter" || node.kind === "combiner" ? (CHANNEL_PORTS.includes(end.port) ? `${node.name}, ${end.port}` : node.name) : node.name;
  }
  if (node.kind === "input" || node.kind === "output") return "";
  return `node ${states.get(node.id as number)?.number ?? 0}, ${node.name}`;
}

// ---------------------------------------------------------------------------
// The four kinds of `docs/25`: adding, inputs and the base
// ---------------------------------------------------------------------------

/**
 * The picture link a new node is put into when S is selected (`docs/25` §5.4): with the Input selected S is Main, with the Output selected S
 * is the node that feeds it; S → T is the link that leads to the Output, else the first. Null when S is gone.
 */
function addTarget(settings: DevelopSettings, selected: NodeId): { node: NodeItem; out: NodeRef; into: NodeLink | undefined } | null {
  const links = settings.nodes.links;
  let at: NodeId = selected === INPUT_ID ? MAIN_ID : selected;
  let into: NodeLink | undefined;
  if (at === OUTPUT_ID) {
    into = linkInto(links, picture(OUTPUT_ID));
    at = into?.from.node ?? MAIN_ID;
  } else {
    into = linkToPutInto(links, at);
  }
  const node = nodeOf(settings, at);
  if (!node) return null;
  return { node, out: into?.from ?? { node: at, port: pictureOut(node) }, into };
}

/**
 * **Add node › Parallel or Layer** (`docs/25` §5.4, the owner's answer "Add a corrector too, as Resolve"). With S selected, fed from U (the
 * source of S's first picture input, the Input when it has none) and S → T: a new corrector N (manual 0, the next default name) is made
 * beside S, fed from U too; S goes to the mixer M's input 1 and N to its input 2 (on a Layer N is on top); M feeds what S fed (every link out
 * of S's picture output moves to M). So U → S → M, U → N → M, M → T, and a Parallel's base is U. With the Input selected S is Main; with the
 * Output selected S is the node that feeds it. N is one row under S at S's x (the nodes in its way move down one row), M one unit right of S,
 * and the nodes after S move one unit right. N is selected. One patch. Refused from 63 nodes: it adds two.
 */
export function insertMixer(settings: DevelopSettings, selected: NodeId, kind: "parallel" | "layer", options: { usedMax?: number; name?: string } = {}): GraphOutcome {
  if (numberedOf(settings).length + 2 > MAX_CORRECTORS) return refuse(GRAPH_OP_TEXT.tooMany);
  const target = addTarget(settings, selected);
  if (!target) return refuse(GRAPH_OP_TEXT.gone);
  const { node: s, out } = target;
  const links = settings.nodes.links;
  const m = nextNodeId(settings, options.usedMax);
  const n = Math.min(m + 1, MAX_ID);
  const u = firstSource(links, s);
  // What S fed through that output moves to the mixer.
  const moved = links.filter((l) => sameRef(l.from, out)).map((l) => ({ from: picture(m), to: l.to }));
  const rest = links.filter((l) => !sameRef(l.from, out));
  const next = sortedLinks([...rest, ...moved, { from: out, to: { node: m, port: "picture.1" } }, { from: u, to: picture(n) }, { from: picture(n), to: { node: m, port: "picture.2" } }]);
  // Places: the mixer one unit right of S on S's line, the corrector one row down at S's x.
  const mixerPos = clampPos([s.pos[0] + 1, s.pos[1]]);
  const newPos = clampPos([s.pos[0], s.pos[1] + 1]);
  const downstream = downstreamOf(next, m);
  downstream.delete(n);
  const into = moved[0]?.to;
  const before = into ? nodeOf(settings, into.node) : undefined;
  const delta = before && hasPlace(before) ? Math.max(0, mixerPos[0] + 1 - before.pos[0]) : 0;
  const nodes: Record<string, NodePatch> = {
    [String(m)]: nodeAsPatch(newMixer(kind, m, mixerPos, options.name ?? MIXER_NAMES[kind])),
    [String(n)]: nodeAsPatch(newCorrector(n, newPos, DEFAULT_NAME)),
  };
  Object.assign(nodes, shifted(settings, downstream, [m, n], delta));
  // The nodes in the new corrector's way (the column of S, one row down or lower) move down one row.
  for (const item of settings.nodes.items) {
    if (!hasPlace(item) || item.id === s.id) continue;
    if (Math.abs(item.pos[0] - s.pos[0]) < 0.5 && item.pos[1] >= newPos[1] - 0.25) {
      const was = (nodes[String(item.id)] as { pos?: NodePos } | undefined)?.pos ?? item.pos;
      nodes[String(item.id)] = { pos: clampPos([was[0], was[1] + 1]) };
    }
  }
  return done({ nodes, links: next }, n);
}

/**
 * **Add node › Splitter and combiner** (`docs/25` §5.4). With S selected and S → T the result is S → Splitter; the Splitter's `red`, `green` and
 * `blue` → three new correctors named Red, Green and Blue (manual 0); them → the Combiner's `red`, `green` and `blue`; Combiner → T. The
 * Splitter is one unit right of S, the correctors two units right on the rows y − 1, y and y + 1, the Combiner three units right, and T and
 * what follows move right as far as it takes (three units when they were next to S). The Splitter is selected. One patch adds all five;
 * refused from 60 nodes.
 */
export function insertTrio(settings: DevelopSettings, selected: NodeId, options: { usedMax?: number } = {}): GraphOutcome {
  if (numberedOf(settings).length + 5 > MAX_CORRECTORS) return refuse(GRAPH_OP_TEXT.tooMany);
  const target = addTarget(settings, selected);
  if (!target) return refuse(GRAPH_OP_TEXT.gone);
  const { node: s, out, into } = target;
  const links = settings.nodes.links;
  const first = nextNodeId(settings, options.usedMax);
  const ids = Array.from({ length: 5 }, (_, i) => Math.min(first + i, MAX_ID));
  const [splitter, red, green, blue, combiner] = ids as [number, number, number, number, number];
  const moved = links.filter((l) => sameRef(l.from, out)).map((l) => ({ from: picture(combiner), to: l.to }));
  const rest = links.filter((l) => !sameRef(l.from, out));
  const channels = [
    ["red", red],
    ["green", green],
    ["blue", blue],
  ] as const;
  const next = sortedLinks([
    ...rest,
    ...moved,
    { from: out, to: picture(splitter) },
    ...channels.flatMap(([port, id]) => [
      { from: { node: splitter, port }, to: picture(id) },
      { from: picture(id), to: { node: combiner, port } },
    ]),
  ]);
  const [x, y] = s.pos;
  const before = into ? nodeOf(settings, into.to.node) : undefined;
  const delta = before && hasPlace(before) ? Math.max(0, x + 4 - before.pos[0]) : 0;
  const nodes: Record<string, NodePatch> = {
    [String(splitter)]: nodeAsPatch(newMixer("splitter", splitter, clampPos([x + 1, y]))),
    [String(red)]: nodeAsPatch(newCorrector(red, clampPos([x + 2, y - 1]), "Red")),
    [String(green)]: nodeAsPatch(newCorrector(green, clampPos([x + 2, y]), "Green")),
    [String(blue)]: nodeAsPatch(newCorrector(blue, clampPos([x + 2, y + 1]), "Blue")),
    [String(combiner)]: nodeAsPatch(newMixer("combiner", combiner, clampPos([x + 3, y]))),
  };
  const downstream = downstreamOf(next, combiner);
  for (const id of ids) downstream.delete(id);
  Object.assign(nodes, shifted(settings, downstream, ids, delta));
  return done({ nodes, links: next }, splitter);
}

/** One input of a mixer, as the panel lists it. */
export interface InputRow {
  /** `picture.k`, k from 1. */
  port: Port;
  /** The link into it, or null for the implicit input of a mixer with no link (the Input's picture). */
  link: NodeLink | null;
}

/** The inputs of a mixer in port order (1 to N). A mixer with no linked input has one implicit row, `picture.1`. */
export function inputsOf(settings: DevelopSettings, id: NodeId): InputRow[] {
  const mine = settings.nodes.links.filter((l) => l.to.node === id && inputNumber(l.to.port) !== null).sort((a, b) => portRank(a.to.port) - portRank(b.to.port));
  return mine.length > 0 ? mine.map((link) => ({ port: link.to.port, link })) : [{ port: "picture.1", link: null }];
}

/**
 * Takes input `port` of a mixer away (the Inputs card's Remove input): the link into it goes and the later inputs are numbered down by one.
 * Refused for an input with no link.
 */
export function removeInput(settings: DevelopSettings, id: NodeId, port: Port): GraphOutcome {
  const link = linkInto(settings.nodes.links, { node: id, port });
  return link ? disconnect(settings, link) : refuse(GRAPH_OP_TEXT.gone);
}

/**
 * Moves input `from` of a Layer (the Inputs card's drag, Move up, Move down) to the place of input `to`, the ones between moving over by
 * one: the ports of the links are renumbered, nothing else changes. One patch (`links`). Moving an input where it is changes nothing.
 */
export function moveInput(settings: DevelopSettings, id: NodeId, from: number, to: number): GraphOutcome {
  const rows = inputsOf(settings, id).flatMap((row) => (row.link ? [row.link] : []));
  if (from === to) return nothing;
  if (from < 1 || to < 1 || from > rows.length || to > rows.length) return refuse(GRAPH_OP_TEXT.gone);
  const order = [...rows];
  const [lifted] = order.splice(from - 1, 1);
  order.splice(to - 1, 0, lifted as NodeLink);
  const renumber = new Map(order.map((link, i) => [link, { ...link, to: { node: id, port: mixerPort(i + 1) } }]));
  const next = sortedLinks(settings.nodes.links.map((l) => renumber.get(l) ?? l));
  return done({ links: next });
}

/**
 * Where a Parallel's inputs start from (`docs/25` §3.2): the **base**, the last picture value every path from the Input's picture to the
 * node goes through (its immediate dominator in the graph of picture values). A value is the Input's picture or one picture output of a
 * node; a node's outputs read all its picture inputs, and an input with no link reads the Input's picture. Answered as the output the
 * base is (the Input's `picture` for the Input).
 */
export function baseOf(settings: DevelopSettings, id: NodeId): NodeRef {
  const links = settings.nodes.links;
  const key = (ref: NodeRef): string => `${ref.node}:${ref.port}`;
  const refs = new Map<string, NodeRef>();
  const readsOf = (node: NodeItem): NodeRef[] => {
    const ports: readonly Port[] = isMixerKind(node.kind) ? inputsOf(settings, node.id).map((row) => row.port) : node.kind === "combiner" ? CHANNEL_PORTS : ["picture"];
    return ports.map((port) => linkInto(links, { node: node.id, port })?.from ?? picture(INPUT_ID));
  };
  /** The values every path to this one goes through, itself included. */
  const memo = new Map<string, Set<string>>();
  const dominators = (ref: NodeRef, visiting: ReadonlySet<string>): Set<string> => {
    const k = key(ref);
    refs.set(k, ref);
    const known = memo.get(k);
    if (known) return known;
    const node = ref.node === INPUT_ID ? undefined : nodeOf(settings, ref.node);
    const own = new Set([k]);
    if (node && !visiting.has(k)) {
      for (const x of commonOf(node, new Set([...visiting, k]))) own.add(x);
      memo.set(k, own);
    }
    return own;
  };
  const commonOf = (node: NodeItem, visiting: ReadonlySet<string>): Set<string> => {
    let common = null as Set<string> | null;
    for (const read of readsOf(node)) {
      const d = dominators(read, visiting);
      common = common === null ? new Set(d) : new Set([...common].filter((x) => d.has(x)));
    }
    return common ?? new Set();
  };
  const node = nodeOf(settings, id);
  if (!node) return picture(INPUT_ID);
  // The nearest of the values they all share is the one with the most dominators of its own.
  let best: NodeRef = picture(INPUT_ID);
  let most = -1;
  for (const k of commonOf(node, new Set([key({ node: id, port: "picture" })]))) {
    const ref = refs.get(k);
    const size = ref ? dominators(ref, new Set()).size : 0;
    if (ref && size > most) {
      most = size;
      best = ref;
    }
  }
  return best;
}
