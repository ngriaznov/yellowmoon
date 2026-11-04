/**
 * The settings of a photo are a graph of nodes (`docs/23-nodes.md` §3), and the simple view is three nodes of it: the Input, Main
 * (node 1, the corrector) and the Output. This is the one place that knows that: the selectors a tab reads them with (`inputOf`,
 * `mainOf`, `outputOf`), the builders it writes with (`patchOf` and one per node), and the pure functions the store and the mock share
 * to fold patches, apply them to a document and tell what changed between two documents (undo).
 *
 * Nothing here talks to the core or draws anything, so it is tested without a screen (`settingsGraph.test.ts`). The core owns the
 * rules (validation, clamping, Auto in a corrector other than Main: `crates/ym-core/src/settings.rs`); `applyPatch` is the page's
 * mirror of the way a node patch is applied and checks nothing.
 */
import type {
  BrushStroke,
  Curves,
  DevelopSettings,
  GradePatch,
  LookId,
  MaskItem,
  MaskPatch,
  MasksPatch,
  NodeGrade,
  NodeId,
  NodeItem,
  NodeLink,
  NodePatch,
  NodePos,
  ResolvedCorrector,
  ResolvedValues,
  SettingsPatch,
} from "../api/types";

/** The Input's id. */
export const INPUT_ID = "input" as const;
/** The Output's id. */
export const OUTPUT_ID = "output" as const;
/** Main's id: the corrector every photo has, and the one the simple view edits. */
export const MAIN_ID = 1 as const;
/** The look a photo has before anyone chooses one (the user's default look, Neutral out of the box). */
export const NEUTRAL_LOOK: LookId = "builtin:neutral";

/** The Input as an item of the graph. */
export type InputItem = Extract<NodeItem, { kind: "input" }>;
/** The Output as an item of the graph. */
export type OutputItem = Extract<NodeItem, { kind: "output" }>;
/** A corrector (Main or another) as an item of the graph. */
export type CorrectorItem = Extract<NodeItem, { kind: "corrector" }>;
/** A Film node as an item of the graph (`docs/24`). */
export type FilmItem = Extract<NodeItem, { kind: "film" }>;
/** A node that has a card in the node view and a number: a corrector (Main included) or a Film node. */
export type CardItem = CorrectorItem | FilmItem;

// ---------------------------------------------------------------------------
// Selectors
// ---------------------------------------------------------------------------

/** The node with this id, or undefined. */
export function nodeOf(settings: DevelopSettings, id: NodeId): NodeItem | undefined {
  return settings.nodes.items.find((item) => item.id === id);
}

function required<T>(item: T | undefined, what: string): T {
  // The core always writes the three fixed nodes (`docs/23` section 3.1), so a document without one is a bug on the other side.
  if (item === undefined) throw new Error(`The settings have no ${what}.`);
  return item;
}

/** The Input: the raw, light and framing settings (`docs/23` section 2). */
export function inputOf(settings: DevelopSettings): InputItem {
  return required(
    settings.nodes.items.find((item): item is InputItem => item.kind === "input"),
    "input node",
  );
}

/** Main, node 1: the corrector the simple view's grade controls edit. */
export function mainOf(settings: DevelopSettings): CorrectorItem {
  return required(
    settings.nodes.items.find((item): item is CorrectorItem => item.kind === "corrector" && item.id === MAIN_ID),
    "node 1, Main",
  );
}

/** The Output: the look and the output sharpening. */
export function outputOf(settings: DevelopSettings): OutputItem {
  return required(
    settings.nodes.items.find((item): item is OutputItem => item.kind === "output"),
    "output node",
  );
}

/**
 * The document without the Output's look and its intensity: what everything about a photo but its look comes to (the Look tab's tiles
 * are drawn again when this changes, and the mock's "adjusted photos" compare it). For comparing and keying only, not for sending.
 */
export function withoutLook(settings: DevelopSettings): object {
  const items = settings.nodes.items.map((item) => {
    if (item.kind !== "output") return item;
    const { look: _look, look_intensity: _intensity, ...rest } = item;
    return rest;
  });
  return { nodes: { items, links: settings.nodes.links } };
}

/** Every corrector of the graph, Main included, in the order the document lists them (by id). */
export function correctorsOf(settings: DevelopSettings): CorrectorItem[] {
  return settings.nodes.items.filter((item): item is CorrectorItem => item.kind === "corrector");
}

/** Every node that has a card, in the order the document lists them: the correctors and the Film nodes, by id. */
export function cardsOf(settings: DevelopSettings): CardItem[] {
  return settings.nodes.items.filter((item): item is CardItem => item.kind === "corrector" || item.kind === "film");
}

/** The Film nodes of the graph. */
export function filmsOf(settings: DevelopSettings): FilmItem[] {
  return settings.nodes.items.filter((item): item is FilmItem => item.kind === "film");
}

/** Main's grade: the shorthand for `mainOf(settings).grade`. */
export function mainGradeOf(settings: DevelopSettings): NodeGrade {
  return mainOf(settings).grade;
}

/**
 * The grade of a corrector, Main's or another's: the node view's tabs edit the selected node (`docs/23` §7.5). A node the document does
 * not have (an undo took it away between a render and this read) reads as Main's, as the screen falls back to Main for it.
 */
export function gradeOf(settings: DevelopSettings, id: number): NodeGrade {
  const node = nodeOf(settings, id);
  return node?.kind === "corrector" ? node.grade : mainOf(settings).grade;
}

/** A corrector with its numbers resolved, Main's (`values.main`) or another's (`values.nodes`); undefined for a node the core has not answered for. */
export function resolvedCorrector(values: ResolvedValues, id: number): ResolvedCorrector | undefined {
  return id === MAIN_ID ? values.main : values.nodes.find((node) => node.id === id);
}

// ---------------------------------------------------------------------------
// What the simple view cannot show (`docs/23` section 6: the notice and the filmstrip glyph)
// ---------------------------------------------------------------------------

/**
 * What of a photo's graph the simple view does not show: more than Main (`nodes`, with the number of correctors, Main included), or Main
 * alone with a trim, masks or a bypass (`main`).
 */
export type HiddenFromSimpleView = { kind: "nodes"; count: number } | { kind: "main" };

const hiddenOf = new WeakMap<DevelopSettings, HiddenFromSimpleView | null>();

/**
 * What the simple view cannot show of this graph, or null when it shows all of it. The simple view reads the Input, Main's grade and the
 * Output, so the graph has something hidden when it has another corrector, or when Main has a trim, a mask row or a bypass. The Input's
 * and the Output's fields all have a control, and Main's grade is the simple view's own.
 */
export function hiddenFromSimpleView(settings: DevelopSettings): HiddenFromSimpleView | null {
  const known = hiddenOf.get(settings);
  if (known !== undefined) return known;
  const correctors = correctorsOf(settings);
  const main = correctors.find((item) => item.id === MAIN_ID);
  let hidden: HiddenFromSimpleView | null = null;
  if (correctors.length > 1) hidden = { kind: "nodes", count: correctors.length };
  else if (main && (main.bypass || main.masks.items.length > 0 || main.trim.exposure_ev !== 0 || main.trim.temperature !== 0 || main.trim.tint !== 0)) {
    hidden = { kind: "main" };
  }
  hiddenOf.set(settings, hidden);
  return hidden;
}

let lastByPhoto: { key: string; map: ReadonlyMap<number, HiddenFromSimpleView> } = { key: "", map: new Map() };

/**
 * The photos whose graph the simple view does not show all of, for the filmstrip's glyph: those whose settings the window holds (the ones
 * that were shown, or that a paste reached). The answer is the same Map while nothing about it changes, so a screen that reads it from
 * the store is not drawn again by every slider move.
 */
export function hiddenByPhoto(settings: Readonly<Record<number, { settings: DevelopSettings }>>): ReadonlyMap<number, HiddenFromSimpleView> {
  const found: Array<[number, HiddenFromSimpleView]> = [];
  for (const [id, rs] of Object.entries(settings)) {
    const hidden = hiddenFromSimpleView(rs.settings);
    if (hidden) found.push([Number(id), hidden]);
  }
  const key = found.map(([id, hidden]) => `${id}:${hidden.kind === "nodes" ? hidden.count : "main"}`).join(";");
  if (key === lastByPhoto.key) return lastByPhoto.map;
  lastByPhoto = { key, map: new Map(found) };
  return lastByPhoto.map;
}

// ---------------------------------------------------------------------------
// Patch builders
// ---------------------------------------------------------------------------

/** What a patch of the Input may carry. */
export type InputPatch = Pick<
  NodePatch,
  | "pos"
  | "exposure"
  | "light_source"
  | "white_balance"
  | "white_pick"
  | "rolloff"
  | "decode"
  | "noise_reduction"
  | "sharpen"
  | "lens"
  | "crop"
  | "orientation"
>;
/** What a patch of the Output may carry. */
export type OutputPatch = Pick<NodePatch, "pos" | "look" | "look_intensity" | "output_sharpening">;
/** What a patch of a corrector (Main included) may carry. */
export type CorrectorPatch = Pick<NodePatch, "pos" | "name" | "bypass" | "trim" | "grade" | "masks">;
/** What a patch of a Film node may carry. */
export type FilmPatch = Pick<NodePatch, "pos" | "name" | "bypass" | "film" | "print" | "halation" | "grain" | "strength" | "masks">;

/**
 * The patch that changes the fields of one node: `patchOf("input", { exposure })`, `patchOf(1, { grade: { contrast } })`,
 * `patchOf("output", { look })` (`docs/23` section 3.7). A slider drag sends one node's partial, never the graph, so a tab builds its
 * patch with this and hands it to `store.update`.
 */
export function patchOf(node: typeof INPUT_ID, partial: InputPatch): SettingsPatch;
export function patchOf(node: typeof OUTPUT_ID, partial: OutputPatch): SettingsPatch;
export function patchOf(node: number, partial: CorrectorPatch | FilmPatch): SettingsPatch;
export function patchOf(node: NodeId, partial: NodePatch): SettingsPatch {
  return { nodes: { [String(node)]: partial } };
}

/** The patch of the Input's fields. */
export const inputPatch = (partial: InputPatch): SettingsPatch => patchOf(INPUT_ID, partial);
/** The patch of Main's fields. */
export const mainPatch = (partial: CorrectorPatch): SettingsPatch => patchOf(MAIN_ID, partial);
/** The patch of the grade fields of Main: `mainGradePatch({ contrast })`. */
export const mainGradePatch = (grade: GradePatch): SettingsPatch => patchOf(MAIN_ID, { grade });
/** The patch of the Output's fields. */
export const outputPatch = (partial: OutputPatch): SettingsPatch => patchOf(OUTPUT_ID, partial);

/** The id a patch's `nodes` key names. */
export function nodeIdOfKey(key: string): NodeId {
  return key === INPUT_ID || key === OUTPUT_ID ? key : Number(key);
}

/** True when a patch changes nothing: it names no node, removes none and gives no links. */
export function isEmptyPatch(patch: SettingsPatch): boolean {
  return Object.keys(patch.nodes ?? {}).length === 0 && (patch.remove?.length ?? 0) === 0 && patch.links === undefined;
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

/** Structural equality of two values of a settings document (or a patch). */
export function same(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((value, i) => same(value, b[i]));
  if (isObject(a)) {
    if (!isObject(b)) return false;
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every((key) => key in b && same(a[key], b[key]));
  }
  return false;
}

/** A deep copy of a plain value, so that history never shares a value with settings that later change. */
export function clone<T>(value: T): T {
  return value === undefined ? value : (JSON.parse(JSON.stringify(value)) as T);
}

// ---------------------------------------------------------------------------
// Defaults (the mock's `defaultSettings` and a node patch that adds a node)
// ---------------------------------------------------------------------------

/** The identity curves. */
export function identityCurves(): Curves {
  return {
    luma: [[0, 0], [1, 1]],
    red: [[0, 0], [1, 1]],
    green: [[0, 0], [1, 1]],
    blue: [[0, 0], [1, 1]],
  };
}

/** The grade of Main when nobody has touched it: Auto wherever there is Auto (`docs/23` section 3.2). */
export function defaultGrade(): NodeGrade {
  return {
    contrast: { kind: "auto" },
    brightness: 0,
    highlights: { kind: "auto" },
    shadows: { kind: "auto" },
    white: 0,
    black: { kind: "auto" },
    color_balance: { shadows: { kind: "auto" }, midtones: [0, 0], highlights: { kind: "auto" } },
    saturation: { kind: "auto" },
    color_warp: { pins: [] },
    curve: identityCurves(),
  };
}

/** The grade of a corrector other than Main, and of a new node: manual 0 everywhere, because Auto belongs to Main. */
export function neutralGrade(): NodeGrade {
  return {
    contrast: { kind: "manual", value: 0 },
    brightness: 0,
    highlights: { kind: "manual", value: 0 },
    shadows: { kind: "manual", value: 0 },
    white: 0,
    black: { kind: "manual", value: 0 },
    color_balance: { shadows: { kind: "manual", point: [0, 0] }, midtones: [0, 0], highlights: { kind: "manual", point: [0, 0] } },
    saturation: { kind: "manual", value: 0 },
    color_warp: { pins: [] },
    curve: identityCurves(),
  };
}

/** The Input of a photo nobody has touched: Auto wherever there is Auto. */
export function defaultInput(pos: NodePos = [-1, 0]): InputItem {
  return {
    kind: "input",
    id: INPUT_ID,
    pos,
    exposure: { kind: "auto" },
    light_source: "auto",
    white_balance: { kind: "auto" },
    white_pick: null,
    rolloff: { highlights: 0.6, low_light: 0.5 },
    decode: { neural_demosaic: true, highlight_recovery: true, lens_corrections: true, chromatic_aberration: true },
    noise_reduction: { kind: "auto" },
    sharpen: { kind: "auto" },
    lens: { kind: "default", flares: 0, edges: 0, anamorphic: 0 },
    crop: null,
    orientation: 0,
  };
}

/** The Output: the given look at 100 %, output sharpening at 100. */
export function defaultOutput(look: LookId = NEUTRAL_LOOK, pos: NodePos = [1, 0]): OutputItem {
  return { kind: "output", id: OUTPUT_ID, pos, look, look_intensity: 100, output_sharpening: 100 };
}

/** Main as it is in the default graph: Auto where there is Auto, no trim, no masks. */
export function defaultMain(pos: NodePos = [0, 0]): CorrectorItem {
  return {
    kind: "corrector",
    id: MAIN_ID,
    name: "Main",
    bypass: false,
    pos,
    trim: { exposure_ev: 0, temperature: 0, tint: 0 },
    grade: defaultGrade(),
    masks: { invert: false, items: [] },
  };
}

/** A new corrector node (`graphOps.addNode`): manual 0 everywhere, no masks. */
export function newCorrector(id: number, pos: NodePos, name = "Corrector"): CorrectorItem {
  return {
    kind: "corrector",
    id,
    name,
    bypass: false,
    pos,
    trim: { exposure_ev: 0, temperature: 0, tint: 0 },
    grade: neutralGrade(),
    masks: { invert: false, items: [] },
  };
}

/** The stock a new Film node starts with (`docs/24` section 4.2). */
export const DEFAULT_STOCK = "kodak_portra_400";
/** The name a new Film node has. */
export const FILM_NAME = "Film";

/** A new Film node (`docs/24` section 4.5): Portra 400 on its matching paper, every number at its default, no masks. */
export function newFilm(id: number, pos: NodePos, name = FILM_NAME): FilmItem {
  return {
    kind: "film",
    id,
    name,
    bypass: false,
    pos,
    film: { stock: DEFAULT_STOCK, exposure_ev: 0 },
    print: { paper: null, exposure_ev: 0, lights: { c: 0, m: 0, y: 0 } },
    halation: { amount: 100, size: 100 },
    grain: { amount: 100, size: 100, seed: 0 },
    strength: 100,
    masks: { invert: false, items: [] },
  };
}

/** The default links: Input → Main → Output. */
export function defaultLinks(): NodeLink[] {
  return [
    { from: { node: INPUT_ID, port: "picture" }, to: { node: MAIN_ID, port: "picture" } },
    { from: { node: MAIN_ID, port: "picture" }, to: { node: OUTPUT_ID, port: "picture" } },
  ];
}

/** The settings of a photo nobody has touched: the default graph, Input → Main → Output, with the given look on the Output. */
export function defaultSettings(look: LookId = NEUTRAL_LOOK): DevelopSettings {
  return { nodes: { items: [defaultInput(), defaultMain(), defaultOutput(look)], links: defaultLinks() } };
}

/** The order the core writes nodes in: the Input first, the correctors by id, the Output last. */
export function inCanonicalOrder(items: readonly NodeItem[]): NodeItem[] {
  const rank = (item: NodeItem): number => (item.kind === "input" ? 0 : item.kind === "output" ? 2 : 1);
  return [...items].sort((a, b) => rank(a) - rank(b) || (typeof a.id === "number" && typeof b.id === "number" ? a.id - b.id : 0));
}

// ---------------------------------------------------------------------------
// Folding patches
// ---------------------------------------------------------------------------

/** The fields of a corrector patch that are partial objects whose own fields merge one by one. */
const MERGED_FIELDS: ReadonlySet<string> = new Set(["trim", "grade", "film", "print", "halation", "grain"]);

function foldRows(first: Record<string, MaskPatch> | undefined, second: Record<string, MaskPatch> | undefined): Record<string, MaskPatch> | undefined {
  if (!first) return second;
  if (!second) return first;
  const out: Record<string, MaskPatch> = { ...first };
  for (const [index, row] of Object.entries(second)) {
    const earlier = out[index];
    if (!earlier) {
      out[index] = row;
      continue;
    }
    const folded: MaskPatch = { ...earlier, ...row };
    // Two appends are both appended, in order (a later keep or a replacement of the strokes ends the chain).
    if (earlier.strokes_append && row.strokes_append && row.strokes === undefined && row.strokes_keep === undefined) {
      folded.strokes_append = [...earlier.strokes_append, ...row.strokes_append];
    }
    out[index] = folded;
  }
  return out;
}

function foldMasks(first: MasksPatch, second: MasksPatch): MasksPatch {
  // A row edit made before a new list of rows has nothing left to act on; one made after acts on the new list.
  const edit = second.items !== undefined ? second.edit : foldRows(first.edit, second.edit);
  const out: MasksPatch = { ...first, ...second };
  if (edit === undefined) delete out.edit;
  else out.edit = edit;
  return out;
}

function foldNode(first: NodePatch, second: NodePatch): NodePatch {
  const out: Record<string, unknown> = { ...first };
  for (const [field, value] of Object.entries(second)) {
    if (value === undefined) continue;
    const earlier = out[field];
    if (field === "print" && isObject(earlier) && isObject(value)) {
      // The lights are a part of their own: a yellow and a magenta dragged in turn are both kept.
      const lights = isObject(earlier.lights) || isObject(value.lights) ? { ...(earlier.lights as object), ...(value.lights as object) } : undefined;
      out[field] = lights ? { ...earlier, ...value, lights } : { ...earlier, ...value };
    } else if (MERGED_FIELDS.has(field) && isObject(earlier) && isObject(value)) out[field] = { ...earlier, ...value };
    else if (field === "masks" && isObject(earlier) && isObject(value)) out[field] = foldMasks(earlier as MasksPatch, value as MasksPatch);
    else out[field] = value;
  }
  return out as NodePatch;
}

/**
 * Two patches in a row as one: what the store sends when a drag's patch is still waiting and the next one arrives. The result changes
 * the document as the two do in turn. Two partials of the same node merge field by field (a later field wins whole; `grade` and `trim`
 * merge one level further, so a contrast and a brightness dragged in turn are both kept), a patch that names more nodes keeps them all,
 * `remove` is the union, and `links` are the later ones.
 */
export function foldPatches(first: SettingsPatch, second: SettingsPatch): SettingsPatch {
  const out: SettingsPatch = {};
  if (first.nodes || second.nodes) {
    const nodes: Record<string, NodePatch> = { ...first.nodes };
    for (const [key, patch] of Object.entries(second.nodes ?? {})) {
      const earlier = nodes[key];
      nodes[key] = earlier ? foldNode(earlier, patch) : patch;
    }
    out.nodes = nodes;
  }
  if (first.remove || second.remove) out.remove = [...new Set([...(first.remove ?? []), ...(second.remove ?? [])])];
  const links = second.links ?? first.links;
  if (links !== undefined) out.links = links;
  return out;
}

// ---------------------------------------------------------------------------
// Applying a patch to a document
// ---------------------------------------------------------------------------

function applyStrokes(row: MaskItem, edit: MaskPatch): BrushStroke[] {
  const strokes = "strokes" in row ? row.strokes : [];
  let out = edit.strokes ?? strokes;
  if (edit.strokes_keep !== undefined) out = out.slice(0, edit.strokes_keep);
  if (edit.strokes_append) out = [...out, ...edit.strokes_append];
  return out;
}

/** One row with a row patch applied (`masks.edit`). The row's kind never changes. */
export function applyRowPatch(row: MaskItem, edit: MaskPatch): MaskItem {
  const { kind: _kind, strokes_append, strokes_keep, strokes, ...fields } = edit;
  const next: Record<string, unknown> = { ...row };
  for (const [key, value] of Object.entries(fields)) if (value !== undefined) next[key] = value;
  if (row.kind === "brush" && (strokes !== undefined || strokes_append !== undefined || strokes_keep !== undefined)) {
    next.strokes = applyStrokes(row, { strokes, strokes_append, strokes_keep });
  }
  return next as MaskItem;
}

/** A node's masks with a masks patch applied: the node's Invert, the whole list of rows, then the row patches (`masks.edit`). */
export function applyMasks(masks: CorrectorItem["masks"], patch: MasksPatch): CorrectorItem["masks"] {
  let items = patch.items ?? masks.items;
  if (patch.edit) {
    items = items.map((row, index) => {
      const edit = patch.edit?.[String(index)];
      return edit ? applyRowPatch(row, edit) : row;
    });
  }
  return { invert: patch.invert ?? masks.invert, items };
}

function applyGrade(grade: NodeGrade, patch: GradePatch): NodeGrade {
  const out: NodeGrade = { ...grade };
  const { color_balance, curve, ...whole } = patch;
  for (const [key, value] of Object.entries(whole)) if (value !== undefined) (out as unknown as Record<string, unknown>)[key] = value;
  if (color_balance) out.color_balance = { ...grade.color_balance, ...color_balance };
  if (curve) out.curve = { ...grade.curve, ...curve };
  return out;
}

/** A node with a node patch applied: whole fields replace, `rolloff`, `decode`, `lens`, `trim`, `grade` and `masks` merge their parts. */
function applyNode(item: NodeItem, patch: NodePatch): NodeItem {
  const { kind: _kind, rolloff, decode, lens, trim, grade, masks, film, print, halation, grain, ...fields } = patch;
  const next: Record<string, unknown> = { ...item };
  for (const [key, value] of Object.entries(fields)) if (value !== undefined) next[key] = value;
  if (item.kind === "input") {
    if (rolloff) next.rolloff = { ...item.rolloff, ...rolloff };
    if (decode) next.decode = { ...item.decode, ...decode };
    if (lens) next.lens = { ...item.lens, ...lens };
  }
  if (item.kind === "corrector") {
    if (trim) next.trim = { ...item.trim, ...trim };
    if (grade) next.grade = applyGrade(item.grade, grade);
    if (masks) next.masks = applyMasks(item.masks, masks);
  }
  if (item.kind === "film") {
    // The parts of a Film node merge field by field, as the core merges them (`docs/24` section 4.5).
    if (film) next.film = { ...item.film, ...film };
    if (print) next.print = { ...item.print, ...print, lights: { ...item.print.lights, ...print.lights } };
    if (halation) next.halation = { ...item.halation, ...halation };
    if (grain) next.grain = { ...item.grain, ...grain };
    if (masks) next.masks = applyMasks(item.masks, masks);
  }
  return next as unknown as NodeItem;
}

/** A node a patch adds: the defaults of its kind, then the patch's fields. A corrector starts at manual 0 (`docs/23` section 3.2). */
function addedNode(id: NodeId, patch: NodePatch, correctors: number): NodeItem {
  const pos: NodePos = patch.pos ?? [correctors, 0];
  if (patch.kind === "input") return applyNode(defaultInput(pos), patch);
  if (patch.kind === "output") return applyNode(defaultOutput(NEUTRAL_LOOK, pos), patch);
  if (patch.kind === "film") return applyNode(newFilm(typeof id === "number" ? id : MAIN_ID, pos), patch);
  return applyNode(newCorrector(typeof id === "number" ? id : MAIN_ID, pos), patch);
}

/**
 * The document with a patch applied, as the core applies it, minus the checks: nodes the patch names are changed (a node the document
 * does not have is added when the patch gives its `kind`), `remove` takes nodes out with the links that touch them, and `links`, when
 * given, replaces the links whole. Pure: the document is not changed, and what the patch does not touch is shared with it.
 */
export function applyPatch(settings: DevelopSettings, patch: SettingsPatch): DevelopSettings {
  let items = settings.nodes.items;
  for (const [key, nodePatch] of Object.entries(patch.nodes ?? {})) {
    const id = nodeIdOfKey(key);
    const at = items.findIndex((item) => item.id === id);
    if (at >= 0) {
      const current = items[at] as NodeItem;
      items = items.map((item, i) => (i === at ? applyNode(current, nodePatch) : item));
    } else if (nodePatch.kind !== undefined) {
      items = [...items, addedNode(id, nodePatch, items.filter((item) => item.kind === "corrector" || item.kind === "film").length)];
    }
  }
  let links = settings.nodes.links;
  const removed = new Set(patch.remove ?? []);
  if (removed.size > 0) {
    items = items.filter((item) => !removed.has(item.id));
    links = links.filter((link) => !removed.has(link.from.node) && !removed.has(link.to.node));
  }
  if (patch.links !== undefined) links = patch.links;
  return { nodes: { items: inCanonicalOrder(items), links } };
}

// ---------------------------------------------------------------------------
// What changed between two documents (undo and redo)
// ---------------------------------------------------------------------------

/** The two patches of one change: `before` puts the touched fields back, `after` makes the change again. */
export interface GraphDiff {
  before: SettingsPatch;
  after: SettingsPatch;
}

/** A node as a patch that adds it whole: every field with the node's `kind`. */
export function nodeAsPatch(item: NodeItem): NodePatch {
  const { id: _id, ...fields } = item;
  return clone(fields) as NodePatch;
}

/** The row fields that differ, as the two row patches of `masks.edit`; null when the rows are the same. */
function diffRow(before: MaskItem, after: MaskItem): { before: MaskPatch; after: MaskPatch } | null {
  const b: Record<string, unknown> = {};
  const a: Record<string, unknown> = {};
  for (const key of Object.keys(after)) {
    if (key === "kind" || key === "strokes") continue;
    const was = (before as unknown as Record<string, unknown>)[key];
    const now = (after as unknown as Record<string, unknown>)[key];
    if (!same(was, now)) {
      b[key] = clone(was);
      a[key] = clone(now);
    }
  }
  if (before.kind === "brush" && after.kind === "brush" && !same(before.strokes, after.strokes)) {
    const was = before.strokes;
    const now = after.strokes;
    const prefix = (shorter: BrushStroke[], longer: BrushStroke[]): boolean => shorter.length < longer.length && shorter.every((s, i) => same(s, longer[i]));
    if (prefix(was, now)) {
      // A stroke drawn: an append, and its inverse is to keep the strokes there were (`docs/23` section 3.7).
      a.strokes_append = clone(now.slice(was.length));
      b.strokes_keep = was.length;
    } else if (prefix(now, was)) {
      a.strokes_keep = now.length;
      b.strokes_append = clone(was.slice(now.length));
    } else {
      a.strokes = clone(now);
      b.strokes = clone(was);
    }
  }
  return Object.keys(a).length === 0 ? null : { before: b as MaskPatch, after: a as MaskPatch };
}

function diffMasks(before: CorrectorItem["masks"], after: CorrectorItem["masks"]): { before: MasksPatch; after: MasksPatch } | null {
  const b: MasksPatch = {};
  const a: MasksPatch = {};
  if (before.invert !== after.invert) {
    b.invert = before.invert;
    a.invert = after.invert;
  }
  if (!same(before.items, after.items)) {
    const rowsMatch = before.items.length === after.items.length && before.items.every((row, i) => row.kind === (after.items[i] as MaskItem).kind);
    if (rowsMatch) {
      // The same rows, some of them changed: a row patch for each, so a drag on one row never carries the others.
      const bEdit: Record<string, MaskPatch> = {};
      const aEdit: Record<string, MaskPatch> = {};
      after.items.forEach((row, i) => {
        const d = diffRow(before.items[i] as MaskItem, row);
        if (d) {
          bEdit[String(i)] = d.before;
          aEdit[String(i)] = d.after;
        }
      });
      b.edit = bEdit;
      a.edit = aEdit;
    } else {
      b.items = clone(before.items);
      a.items = clone(after.items);
    }
  }
  return Object.keys(a).length === 0 ? null : { before: b, after: a };
}

/**
 * What changed between two documents, as the patches that undo and redo it; null when nothing did. It diffs per node and for the links
 * (`docs/23` section 3.7): an entry holds the nodes a change touched and only the fields of them that differ (a field whole, except that
 * `grade` and `trim` go one level further, so a contrast change is a contrast and nothing else, and masks go by row: a Brush stroke is
 * a `strokes_append` and its inverse a `strokes_keep`). A node only `after` has is added whole with its `kind` (and `remove`s on the
 * way back), one only `before` has is in `remove` (and comes back whole), and `links` are given whole when they differ.
 */
export function diffGraph(before: DevelopSettings, after: DevelopSettings): GraphDiff | null {
  const b: SettingsPatch = {};
  const a: SettingsPatch = {};
  const bNodes: Record<string, NodePatch> = {};
  const aNodes: Record<string, NodePatch> = {};
  const was = new Map(before.nodes.items.map((item) => [item.id, item]));
  const now = new Map(after.nodes.items.map((item) => [item.id, item]));
  for (const item of after.nodes.items) {
    const old = was.get(item.id);
    if (!old) {
      aNodes[String(item.id)] = nodeAsPatch(item);
      b.remove = [...(b.remove ?? []), item.id];
      continue;
    }
    const nb: Record<string, unknown> = {};
    const na: Record<string, unknown> = {};
    for (const key of Object.keys(item)) {
      if (key === "id" || key === "kind") continue;
      const oldValue = (old as unknown as Record<string, unknown>)[key];
      const newValue = (item as unknown as Record<string, unknown>)[key];
      if (same(oldValue, newValue)) continue;
      if ((key === "grade" || key === "trim") && isObject(oldValue) && isObject(newValue)) {
        const gb: Record<string, unknown> = {};
        const ga: Record<string, unknown> = {};
        for (const field of Object.keys(newValue)) {
          if (same(oldValue[field], newValue[field])) continue;
          gb[field] = clone(oldValue[field]);
          ga[field] = clone(newValue[field]);
        }
        nb[key] = gb;
        na[key] = ga;
      } else if (key === "masks" && (item.kind === "corrector" || item.kind === "film") && (old.kind === "corrector" || old.kind === "film")) {
        const d = diffMasks(old.masks, item.masks);
        if (d) {
          nb[key] = d.before;
          na[key] = d.after;
        }
      } else {
        nb[key] = clone(oldValue);
        na[key] = clone(newValue);
      }
    }
    if (Object.keys(na).length > 0) {
      bNodes[String(item.id)] = nb as NodePatch;
      aNodes[String(item.id)] = na as NodePatch;
    }
  }
  for (const item of before.nodes.items) {
    if (now.has(item.id)) continue;
    bNodes[String(item.id)] = nodeAsPatch(item);
    a.remove = [...(a.remove ?? []), item.id];
  }
  if (Object.keys(aNodes).length > 0) a.nodes = aNodes;
  if (Object.keys(bNodes).length > 0) b.nodes = bNodes;
  if (!same(before.nodes.links, after.nodes.links)) {
    b.links = clone(before.nodes.links);
    a.links = clone(after.nodes.links);
  }
  return isEmptyPatch(a) && isEmptyPatch(b) ? null : { before: b, after: a };
}
