/**
 * The graph's rules, for the mock API (`docs/23-nodes.md` §3.5, §3.7, §3.8, §9). The core owns them (`crates/ym-core/src/settings/`:
 * `graph.rs` for the texts and the checks, `patch.rs` for the patches and `SettingsPatch::copied_from`); the mock plays the core in a
 * browser and in the tests, so it refuses what the core refuses, with the core's words, and copies and pastes by the core's rules.
 *
 * Pure functions on plain data: a patch is checked before it is applied (`refusalOfPatch`), the document it makes is checked after
 * (`validateGraph`), and both answer the plain sentence to show, or null.
 */
import type { DevelopSettings, NodeId, NodeItem, NodeKind, NodeLink, NodePatch, Port, ResolvedSettings, SettingsPatch } from "../api/types";
import { clone, correctorsOf, INPUT_ID, MAIN_ID, nodeAsPatch, nodeIdOfKey, nodeOf, OUTPUT_ID } from "../state/settingsGraph";

/** Most corrector nodes a photo can have, Main included (`docs/23` section 3.5). */
export const MAX_NODES = 64;
/** Most mask rows in a node. */
export const MAX_MASKS = 64;
/** Most points in a Brush mask, all strokes together. */
export const MAX_BRUSH_POINTS = 100_000;
/** Most clicks in an Object mask. */
export const MAX_OBJECT_POINTS = 256;

/** The core's sentences for the graph's rules, word for word (`GraphError` in `crates/ym-core/src/settings/graph.rs`). */
export const GRAPH_TEXT = {
  missingFixed: "A photo needs its input, its output and node 1, Main.",
  linkDirection: "A link goes from the output of one node to the input of another.",
  linkPortTypes: "A link joins a picture to a picture, or a key to a key.",
  outputLink: "The output needs one picture link.",
  mainInput: "Node 1, Main, must take its picture from the input.",
  cycle: "The nodes of this photo are linked in a circle. Remove a link.",
  tooManyNodes: "A photo can have up to 64 nodes.",
  tooManyMasks: "A node can have up to 64 masks.",
  tooManyBrushPoints: "A brush mask can have up to 100,000 points.",
  tooManyObjectPoints: "An Object mask can have up to 256 points.",
  cannotRemove: "The input, the output and node 1, Main, can't be removed.",
  unknownNode: (node: NodeId) => `A link names node ${node}, which this photo does not have.`,
  unknownPort: (node: NodeId, port: string) => `${sentence(node)} has no port called “${port}”.`,
  twoLinks: (node: NodeId, port: Port) => `An input can have one link. ${sentence(node)} has two on its ${port} input.`,
  duplicateId: (id: NodeId) => `Two nodes have the id ${id}.`,
  patchUnknownNode: (id: NodeId) => `The patch names node ${id}, which this photo does not have.`,
  patchWrongKind: (id: NodeId, actual: NodeKind, given: NodeKind) => `${sentence(id)} is ${article(actual)}, not ${article(given)}.`,
  unknownField: (kind: NodeKind, field: string) => `${capital(article(kind))} has no field called “${field}”.`,
} as const;

/** "Node 7", "The input", "The output": how a sentence starts with a node. */
function sentence(id: NodeId): string {
  return id === INPUT_ID ? "The input" : id === OUTPUT_ID ? "The output" : `Node ${id}`;
}

/** "an input", "an output", "a corrector". */
function article(kind: NodeKind): string {
  return kind === "corrector" ? "a corrector" : `an ${kind}`;
}

const capital = (text: string): string => text.charAt(0).toUpperCase() + text.slice(1);

/** The fields a patch of a node of each kind may carry (`docs/23` section 3.2); `kind` and `pos` are every node's. */
const FIELDS: Readonly<Record<NodeKind, ReadonlySet<string>>> = {
  input: new Set([
    "kind",
    "pos",
    "exposure",
    "light_source",
    "white_balance",
    "white_pick",
    "rolloff",
    "decode",
    "noise_reduction",
    "sharpen",
    "lens",
    "crop",
    "orientation",
  ]),
  output: new Set(["kind", "pos", "look", "look_intensity", "output_sharpening"]),
  corrector: new Set(["kind", "pos", "name", "bypass", "trim", "grade", "masks"]),
};

/** The kind a node with this id has to be. */
const kindOfId = (id: NodeId): NodeKind => (id === INPUT_ID ? "input" : id === OUTPUT_ID ? "output" : "corrector");

/**
 * Why the core would refuse this patch before applying it, or null (`docs/23` section 3.5, the rows about patches): it removes the
 * Input, the Output or Main; it names a node the photo does not have and gives no kind (or a kind the id cannot have); it gives a kind
 * other than the node's; it carries a field the node's kind does not have.
 */
export function refusalOfPatch(settings: DevelopSettings, patch: SettingsPatch): string | null {
  for (const id of patch.remove ?? []) {
    if (id === INPUT_ID || id === OUTPUT_ID || id === MAIN_ID) return GRAPH_TEXT.cannotRemove;
  }
  for (const [key, node] of Object.entries(patch.nodes ?? {})) {
    const id = nodeIdOfKey(key);
    const have = nodeOf(settings, id);
    if (!have) {
      if (node.kind === undefined) return GRAPH_TEXT.patchUnknownNode(id);
      if (node.kind !== kindOfId(id)) return GRAPH_TEXT.missingFixed;
    } else if (node.kind !== undefined && node.kind !== have.kind) {
      return GRAPH_TEXT.patchWrongKind(id, have.kind, node.kind);
    }
    const kind = have?.kind ?? (node.kind as NodeKind);
    for (const field of Object.keys(node)) {
      if (!FIELDS[kind].has(field)) return GRAPH_TEXT.unknownField(kind, field);
    }
  }
  return null;
}

/** The ports a node has: `[inputs, outputs]` (`docs/23` section 3.4). */
function portsOf(kind: NodeKind): [Port[], Port[]] {
  if (kind === "input") return [[], ["picture"]];
  if (kind === "output") return [["picture"], []];
  return [["picture", "key"], ["picture", "key"]];
}

/** True when the links, picture and key together, make a circle (Kahn's algorithm). */
function hasCycle(ids: readonly NodeId[], links: readonly NodeLink[]): boolean {
  const waiting = new Map<NodeId, number>(ids.map((id) => [id, 0]));
  for (const link of links) waiting.set(link.to.node, (waiting.get(link.to.node) ?? 0) + 1);
  const ready = ids.filter((id) => waiting.get(id) === 0);
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
  return seen < ids.length;
}

function maskProblem(item: NodeItem): string | null {
  if (item.kind !== "corrector") return null;
  if (item.masks.items.length > MAX_MASKS) return GRAPH_TEXT.tooManyMasks;
  let brush = 0;
  for (const row of item.masks.items) {
    if (row.kind === "object" && row.points.length > MAX_OBJECT_POINTS) return GRAPH_TEXT.tooManyObjectPoints;
    if (row.kind === "brush") brush += row.strokes.reduce((n, stroke) => n + stroke.points.length, 0);
  }
  return brush > MAX_BRUSH_POINTS ? GRAPH_TEXT.tooManyBrushPoints : null;
}

/**
 * Why the core would refuse this document, or null (`docs/23` section 3.5): in the order the core checks them, the three fixed nodes,
 * unique ids, the limits, each link (both ends exist, the direction, the ports, the port types), an input with two links, the Output's
 * picture link, Main's picture from the Input, and circles.
 */
export function validateGraph(settings: DevelopSettings): string | null {
  const { items, links } = settings.nodes;
  const ids = new Set<NodeId>();
  for (const item of items) {
    if (ids.has(item.id)) return GRAPH_TEXT.duplicateId(item.id);
    ids.add(item.id);
  }
  const main = nodeOf(settings, MAIN_ID);
  if (!ids.has(INPUT_ID) || !ids.has(OUTPUT_ID) || main?.kind !== "corrector") return GRAPH_TEXT.missingFixed;
  if (correctorsOf(settings).length > MAX_NODES) return GRAPH_TEXT.tooManyNodes;
  for (const item of items) {
    const problem = maskProblem(item);
    if (problem) return problem;
  }
  const taken = new Map<string, number>();
  for (const { from, to } of links) {
    const start = nodeOf(settings, from.node);
    const end = nodeOf(settings, to.node);
    const missing = start ? (end ? null : to.node) : from.node;
    if (missing !== null) return GRAPH_TEXT.unknownNode(missing);
    if (!start || !end) return GRAPH_TEXT.missingFixed;
    if (start.kind === "output" || end.kind === "input") return GRAPH_TEXT.linkDirection;
    if (!portsOf(start.kind)[1].includes(from.port)) return GRAPH_TEXT.unknownPort(from.node, from.port);
    if (!portsOf(end.kind)[0].includes(to.port)) return GRAPH_TEXT.unknownPort(to.node, to.port);
    if (from.port !== to.port) return GRAPH_TEXT.linkPortTypes;
    const slot = `${to.node}:${to.port}`;
    taken.set(slot, (taken.get(slot) ?? 0) + 1);
  }
  for (const link of links) {
    if ((taken.get(`${link.to.node}:${link.to.port}`) ?? 0) > 1) return GRAPH_TEXT.twoLinks(link.to.node, link.to.port);
  }
  if (!taken.has(`${OUTPUT_ID}:picture`)) return GRAPH_TEXT.outputLink;
  const feeds = links.find((link) => link.to.node === MAIN_ID && link.to.port === "picture");
  if (feeds?.from.node !== INPUT_ID) return GRAPH_TEXT.mainInput;
  return hasCycle([...ids], links) ? GRAPH_TEXT.cycle : null;
}

// ---------------------------------------------------------------------------
// Copy and paste (`docs/23` section 3.8)
// ---------------------------------------------------------------------------

/** The fields of the Input that belong to one photo's own pixels and frame: a copy leaves them out and a paste keeps the target's. */
export const COPY_LEAVES_OUT = ["crop", "orientation", "white_pick"] as const;

/**
 * What Copy settings takes from a photo (the core's `SettingsPatch::copied_from`): the whole graph, every node and every link, as a
 * patch that holds every node whole and `links`, except the Input's `crop`, `orientation` and `white_pick`. Auto stays Auto in the Input
 * and Main; a white balance that came from a picked white (a white picked, the light preset and the white balance both Auto) is the
 * manual Kelvin and tint it resolved to. An Object row arrives empty, so it selects nothing until it is clicked on the new photo.
 */
export function copiedPatch(rs: ResolvedSettings): SettingsPatch {
  const nodes: Record<string, NodePatch> = {};
  for (const item of rs.settings.nodes.items) {
    const patch = nodeAsPatch(item) as Record<string, unknown>;
    if (item.kind === "input") for (const field of COPY_LEAVES_OUT) delete patch[field];
    if (item.kind === "corrector") {
      patch.masks = {
        invert: item.masks.invert,
        items: item.masks.items.map((row) => (row.kind === "object" ? { ...clone(row), points: [], box: null } : clone(row))),
      };
    }
    nodes[String(item.id)] = patch as NodePatch;
  }
  const input = rs.settings.nodes.items.find((item) => item.kind === "input");
  const inputPatch = nodes[INPUT_ID];
  if (input?.kind === "input" && inputPatch && input.white_pick !== null && input.light_source === "auto" && input.white_balance.kind === "auto") {
    inputPatch.white_balance = { kind: "manual", kelvin: rs.values.input.kelvin, tint: rs.values.input.tint };
  }
  return { nodes, links: clone(rs.settings.nodes.links) };
}

/**
 * A copied patch as it lands on a target (the core's `SettingsPatch::pasted_onto`): the Input's `crop`, `orientation` and `white_pick`
 * are never pasted, whatever the patch says, and a patch that carries `links` (a whole-graph copy) replaces the target's graph, so the
 * target's correctors it does not have are removed. A patch without `links` replaces nothing.
 */
export function pastedOnto(target: DevelopSettings, patch: SettingsPatch): SettingsPatch {
  const nodes: Record<string, NodePatch> = {};
  for (const [key, node] of Object.entries(patch.nodes ?? {})) {
    if (nodeIdOfKey(key) === INPUT_ID) {
      const { crop: _crop, orientation: _orientation, white_pick: _pick, ...kept } = node;
      nodes[key] = kept;
    } else {
      nodes[key] = node;
    }
  }
  const out: SettingsPatch = { ...patch, nodes };
  if (patch.links !== undefined) {
    const given = new Set(Object.keys(nodes).map(nodeIdOfKey));
    const extra = correctorsOf(target)
      .map((item) => item.id)
      .filter((id) => id !== MAIN_ID && !given.has(id));
    out.remove = [...new Set([...(patch.remove ?? []), ...extra])];
  }
  return out;
}
