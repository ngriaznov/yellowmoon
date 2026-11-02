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
import type { DevelopSettings, NodeId, NodeLink, NodePatch, NodePos, NodeRef, Port, SettingsPatch } from "../api/types";
import { correctorsOf, INPUT_ID, isEmptyPatch, MAIN_ID, newCorrector, nodeAsPatch, nodeOf, OUTPUT_ID } from "./settingsGraph";
import type { CorrectorItem } from "./settingsGraph";

/** Most corrector nodes a photo can have, Main included (`docs/23` section 3.5). */
export const MAX_CORRECTORS = 64;
/** Longest name a node can have, in characters. */
export const MAX_NAME_LENGTH = 32;
/** The name a new node has. */
export const DEFAULT_NAME = "Corrector";
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
} as const;

// ---------------------------------------------------------------------------
// Links
// ---------------------------------------------------------------------------

/** A node's place in the order the core writes links in: the Input, the correctors by id, the Output. */
const rankOf = (id: NodeId): number => (id === INPUT_ID ? -1 : id === OUTPUT_ID ? Number.POSITIVE_INFINITY : id);
const portRank = (port: Port): number => (port === "picture" ? 0 : 1);

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

/** The links that leave a node's picture output, in the core's order. */
const picturesOut = (links: readonly NodeLink[], node: NodeId): NodeLink[] =>
  sortedLinks(links.filter((l) => l.from.node === node && l.from.port === "picture"));

const picture = (node: NodeId): NodeRef => ({ node, port: "picture" });

/** The ports a kind of node has: `[inputs, outputs]` (`docs/23` section 3.4). */
function portsOf(settings: DevelopSettings, id: NodeId): { inputs: readonly Port[]; outputs: readonly Port[] } | null {
  const node = nodeOf(settings, id);
  if (!node) return null;
  if (node.kind === "input") return { inputs: [], outputs: ["picture"] };
  if (node.kind === "output") return { inputs: ["picture"], outputs: [] };
  return { inputs: ["picture", "key"], outputs: ["picture", "key"] };
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

/** What the plan says about one corrector. */
export interface CardState {
  /** The node's place in the flow, counted from 1 (the number on the card). */
  number: number;
  /** The Output needs this node, or a live node needs its key. */
  live: boolean;
}

/**
 * Every corrector's number and whether it is live, as the core's plan has them. The order is a topological one over the picture and key
 * links together, with ties broken by Main first, then `pos` (x, then y), then the id; the number is the place in that order over all the
 * nodes, dead ones included. A node is live when the Output needs it: it is on the chain of picture links back from the Output to the
 * Input, or its key output feeds a live node's key input (and so on). A picture input with no link reads the Input.
 */
export function cardStates(settings: DevelopSettings): Map<number, CardState> {
  const links = settings.nodes.links;
  const nodes = correctorsOf(settings);
  const byId = new Map(nodes.map((n) => [n.id as number, n]));
  const needs = (node: CorrectorItem): number[] => {
    const out: number[] = [];
    for (const port of ["picture", "key"] as const) {
      const from = sourceOf(links, node.id, port);
      if (typeof from === "number" && byId.has(from)) out.push(from);
    }
    return out;
  };
  const precedes = (a: CorrectorItem, b: CorrectorItem): number =>
    Number(a.id !== MAIN_ID) - Number(b.id !== MAIN_ID) || a.pos[0] - b.pos[0] || a.pos[1] - b.pos[1] || (a.id as number) - (b.id as number);

  const waiting = new Map<number, Set<number>>(nodes.map((n) => [n.id as number, new Set(needs(n))]));
  const order: number[] = [];
  while (waiting.size > 0) {
    const ready = [...waiting].filter(([, deps]) => deps.size === 0).map(([id]) => byId.get(id) as CorrectorItem);
    // A validated graph has no circle; a broken one still gets a total order rather than a loop.
    const pool = ready.length > 0 ? ready : [...waiting.keys()].map((id) => byId.get(id) as CorrectorItem);
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
    todo.push(...needs(byId.get(id) as CorrectorItem));
  }
  return new Map(order.map((id, i) => [id, { number: i + 1, live: live.has(id) }]));
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

/** `pos` patches that move every node in `ids` one unit to the right. Only correctors have a place the page moves. */
function shifted(settings: DevelopSettings, ids: ReadonlySet<NodeId>, skip: NodeId): Record<string, { pos: NodePos }> {
  const out: Record<string, { pos: NodePos }> = {};
  for (const item of settings.nodes.items) {
    if (item.kind !== "corrector" || item.id === skip || !ids.has(item.id)) continue;
    out[String(item.id)] = { pos: clampPos([item.pos[0] + 1, item.pos[1]]) };
  }
  return out;
}

/**
 * A new node and the links around it: the node `id` goes after `after`, into `into` (the picture link that was there: `after → to` becomes
 * `after → id → to`), or linked after `after` alone when there is none. The links of `id`'s other ports are not touched.
 */
function putAfter(links: readonly NodeLink[], id: number, after: NodeId, into: NodeLink | undefined): NodeLink[] {
  const kept = links.filter((l) => !into || !sameLink(l, into));
  const added: NodeLink[] = [{ from: picture(after), to: picture(id) }];
  if (into) added.push({ from: picture(id), to: into.to });
  return sortedLinks([...kept, ...added]);
}

/** A new corrector node, manual 0 everywhere, as a patch that adds it whole (`docs/23` section 3.2: a new node is manual 0). */
export function addNode(id: number, pos: NodePos, name = DEFAULT_NAME): Record<string, NodePatch> {
  return { [String(id)]: nodeAsPatch(newCorrector(id, clampPos(pos), name)) };
}

/**
 * **Add after the selected node** (`docs/23` section 7.2): the new node goes into the selected node's outgoing picture link. With S
 * selected and a link S → T the result is S → N → T; with several such links N goes into the one that leads to the Output, else the first;
 * with none N is linked after S alone. With the Input selected N goes after Main (Main stays first); with the Output selected, into the
 * Output's incoming link. N's `pos` is the node's plus one unit to the right, the nodes downstream of N move one unit right, and N is
 * selected. Refused at 64 nodes.
 */
export function insertAfter(settings: DevelopSettings, selected: NodeId, options: { usedMax?: number; name?: string } = {}): GraphOutcome {
  if (correctorsOf(settings).length >= MAX_CORRECTORS) return refuse(GRAPH_OP_TEXT.tooMany);
  const links = settings.nodes.links;
  let after: NodeId = selected === INPUT_ID ? MAIN_ID : selected;
  let into: NodeLink | undefined;
  if (after === OUTPUT_ID) {
    into = linkInto(links, picture(OUTPUT_ID));
    after = into?.from.node ?? MAIN_ID;
  } else {
    into = linkToPutInto(links, after);
  }
  const base = nodeOf(settings, after);
  if (!base) return refuse(GRAPH_OP_TEXT.gone);
  const id = nextNodeId(settings, options.usedMax);
  const next = putAfter(links, id, after, into);
  const pos: NodePos = clampPos([base.pos[0] + 1, base.pos[1]]);
  const nodes = { ...addNode(id, pos, options.name), ...shifted(settings, downstreamOf(next, id), id) };
  return done({ nodes, links: next }, id);
}

/**
 * **Delete** (`docs/23` section 7.2): any corrector but Main. The neighbours are reconnected: with U → D → T₁ … Tₙ the result is U → Tᵢ
 * for each, and the links of D's key ports go with it (a node that had D's key as its key input loses that input). A node that took its
 * picture from no link read the Input, so U is then the Input. The previous node is selected.
 */
export function remove(settings: DevelopSettings, id: NodeId): GraphOutcome {
  if (id === MAIN_ID) return refuse(GRAPH_OP_TEXT.mainDelete);
  if (id === INPUT_ID || id === OUTPUT_ID) return refuse(GRAPH_OP_TEXT.markerDelete);
  const node = nodeOf(settings, id);
  if (!node) return refuse(GRAPH_OP_TEXT.gone);
  const links = settings.nodes.links;
  const before: NodeId = sourceOf(links, id, "picture") ?? INPUT_ID;
  const next = sortedLinks([
    ...links.filter((l) => l.from.node !== id && l.to.node !== id),
    ...picturesOut(links, id).map((l) => ({ from: picture(before), to: l.to })),
  ]);
  return done({ remove: [id], links: next }, before === INPUT_ID ? MAIN_ID : before);
}

/** **Bypass** (`docs/23` section 7.2): toggles a corrector's bypass, Main's too; refused for the Input and the Output. */
export function toggleBypass(settings: DevelopSettings, id: NodeId): GraphOutcome {
  const node = nodeOf(settings, id);
  if (!node) return refuse(GRAPH_OP_TEXT.gone);
  if (node.kind !== "corrector") return refuse(GRAPH_OP_TEXT.noBypass);
  return done({ nodes: { [String(id)]: { bypass: !node.bypass } } });
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
  if (node.kind !== "corrector") return refuse(GRAPH_OP_TEXT.noRename);
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

/** The links with a node taken out of its place: its picture links go, and its source feeds each of the nodes it fed. Its key links stay. */
function takeOut(links: readonly NodeLink[], id: number): NodeLink[] {
  const before: NodeId = sourceOf(links, id, "picture") ?? INPUT_ID;
  const kept = links.filter((l) => !((l.from.node === id || l.to.node === id) && l.from.port === "picture" && l.to.port === "picture"));
  return [...kept, ...picturesOut(links, id).map((l) => ({ from: picture(before), to: l.to }))];
}

/** Why a node cannot be reordered, or null when it can. */
function immovable(settings: DevelopSettings, id: NodeId): string | null {
  if (id === MAIN_ID) return GRAPH_OP_TEXT.mainFixed;
  if (id === INPUT_ID || id === OUTPUT_ID) return GRAPH_OP_TEXT.markerFixed;
  return nodeOf(settings, id) ? null : GRAPH_OP_TEXT.gone;
}

/**
 * Puts a node (already out of its place, in `links`) after `after`, into `into` or alone, and answers the patch: the links, the node's
 * `pos` (the node it follows plus one unit) and the nodes downstream of it, one unit right. A circle through key links is refused.
 */
function placed(settings: DevelopSettings, links: readonly NodeLink[], id: number, after: NodeId, into: NodeLink | undefined): GraphOutcome {
  const next = putAfter(links, id, after, into);
  if (hasCircle(settings, next)) return refuse(GRAPH_OP_TEXT.circle);
  const base = nodeOf(settings, after);
  if (!base) return refuse(GRAPH_OP_TEXT.gone);
  // After the Input there is no place of its own (it sits where the cards begin): the node takes the place of the one it goes before.
  const before = into ? nodeOf(settings, into.to.node) : undefined;
  const pos: NodePos = clampPos(base.kind === "input" && before ? [before.pos[0], before.pos[1]] : [base.pos[0] + 1, base.pos[1]]);
  const nodes: Record<string, NodePatch> = { [String(id)]: { pos } };
  Object.assign(nodes, shifted(settings, downstreamOf(next, id), id));
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
  if (!links.some((l) => sameLink(l, target)) || target.from.port !== "picture") return refuse(GRAPH_OP_TEXT.gone);
  if (target.to.node === MAIN_ID) return refuse(GRAPH_OP_TEXT.beforeMain);
  if (target.from.node === id || target.to.node === id) return nothing;
  return placed(settings, takeOut(links, id as number), id as number, target.from.node, target);
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
  const before: NodeId = sourceOf(links, previous, "picture") ?? INPUT_ID;
  const rest = takeOut(links, id as number);
  const into = linkInto(rest, picture(previous)) ?? { from: picture(before), to: picture(previous) };
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
  const rest = takeOut(links, id as number);
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
  if (!start.outputs.includes(from.port) || !end.inputs.includes(to.port) || from.port !== to.port) return refuse(GRAPH_OP_TEXT.portTypes);
  const links = settings.nodes.links;
  if (lifted && lifted.to.node === MAIN_ID && lifted.to.port === "picture") return refuse(GRAPH_OP_TEXT.mainInput);
  if (to.node === MAIN_ID && to.port === "picture" && from.node !== INPUT_ID) return refuse(GRAPH_OP_TEXT.mainInput);
  if (lifted && lifted.to.node === OUTPUT_ID && !(to.node === OUTPUT_ID && to.port === "picture")) return refuse(GRAPH_OP_TEXT.outputNeedsPicture);
  const kept = links.filter((l) => !(lifted && sameLink(l, lifted)) && !sameRef(l.to, to));
  const next = sortedLinks([...kept, { from, to }]);
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
  return done({ links: links.filter((l) => !sameLink(l, link)) });
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
export function maskKindNames(node: CorrectorItem): string[] {
  const names: string[] = [];
  for (const row of node.masks.items) {
    const name = MASK_KIND_NAME[row.kind] ?? row.kind;
    if (row.enabled && !names.includes(name)) names.push(name);
  }
  return names;
}

/**
 * A card's accessible name (`docs/23` section 7.8): "Node 2, Mountain, masked by Object and Radial" (the number as a plain number, "Node 1,
 * Main" with no masks), then ", bypassed" and ", not connected" where they apply, and what feeds it: ", picture from node 1, key from node 5".
 * The Input and the Output are "Input" and "Output".
 */
export function cardName(settings: DevelopSettings, id: NodeId, states: ReadonlyMap<number, CardState> = cardStates(settings)): string {
  const node = nodeOf(settings, id);
  if (!node) return "";
  if (node.kind === "input") return "Input";
  if (node.kind === "output") return "Output";
  const number = states.get(node.id as number)?.number ?? 0;
  const masks = maskKindNames(node);
  const parts = [`Node ${number}`, node.name];
  if (masks.length > 0) parts.push(`masked by ${joinWords(masks)}`);
  if (node.bypass) parts.push("bypassed");
  if (states.get(node.id as number)?.live === false) parts.push("not connected");
  for (const port of ["picture", "key"] as const) {
    const from = sourceOf(settings.nodes.links, node.id, port);
    if (from === undefined || from === INPUT_ID) continue;
    const n = typeof from === "number" ? states.get(from)?.number : undefined;
    if (n !== undefined) parts.push(`${port} from node ${n}`);
  }
  return parts.join(", ");
}

/** "Node 3, Sky" or "Output": how the More menu names a node in a list. */
export function listName(settings: DevelopSettings, id: NodeId, states: ReadonlyMap<number, CardState> = cardStates(settings)): string {
  const node = nodeOf(settings, id);
  if (!node) return "";
  if (node.kind === "input") return "Input";
  if (node.kind === "output") return "Output";
  return `Node ${states.get(node.id as number)?.number ?? 0}, ${node.name}`;
}
