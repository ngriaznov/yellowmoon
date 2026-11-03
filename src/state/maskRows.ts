/**
 * What a mask row says about itself (`docs/23` §7.6): its name, its subtitle under the name and the accessible name of the row. Pure, so the
 * Mask tab's rows (read only in N-09, editable from N-12) and their tests do not need a screen. The numbers are what the sliders show
 * (`docs/algorithms/06` §4.5): Feather and Softness 0 to 100 are 0 to 1, a Colour range's Width is `0.005 + 0.295 · (x / 100)²`, Low and
 * High are the code values times 100.
 */
import type { DevelopSettings, MaskItem, MaskKind, MaskMode, MaskPatch, MasksPatch } from "../api/types";
import { widthSlider } from "../screens/tools/maskScales";
import { MASK_KIND_NAME } from "./graphOps";
import { GETTING_READY } from "./objectMask";
import { nodeOf } from "./settingsGraph";

export { widthSlider };

/**
 * What tells one list of a node's mask rows from another for an eye that is an index into it (`docs/23` §7.6): the kinds in order. A row's own
 * fields are not in it, so editing a row keeps its eye; adding, removing and moving a row (an undo or a paste too) changes it. Empty for no node.
 */
export function maskRowKinds(settings: DevelopSettings | null, node: number | null): string {
  if (settings === null || node === null) return "";
  const item = nodeOf(settings, node);
  return item?.kind === "corrector" ? item.masks.items.map((row) => row.kind).join(",") : "";
}

/** What the subtitle of a raster row says while the core makes its mask for the first time (`docs/23` §7.6). */
export const FINDING = "Finding…";

/** What the subtitle of a Subject or People row says when Vision found no instance in the photo (`docs/23` §7.6). */
export const NOTHING_FOUND = "Nothing found";

/** What the subtitle of a Subject or People row says on a computer that cannot make it (`docs/23` §3.10 and §7.6; the core's `unsupported_row_sentence`). */
export const UNSUPPORTED_ROW: Readonly<Record<"subject" | "people", string>> = {
  subject: "Subject masks need a Mac that supports them",
  people: "People masks need a Mac that supports them",
};

/**
 * Whether a row is waiting for its raster to be made: a Background, Subject or People row with the eye on while the core says it is making the photo's
 * first blur map or its Subject and People masks (`viewport_busy.finding`). The row's subtitle is then {@link FINDING}, with a spinner. (The flag is
 * the photo's, not a row's: while the core finds one kind the other rows of these kinds say it too.)
 */
export const isFinding = (row: MaskItem, finding: boolean): boolean =>
  finding && row.enabled && (row.kind === "background" || row.kind === "subject" || row.kind === "people");

/** What the Mask tab knows about Subject and People on this computer and on the photo (`docs/23` §7.6). */
export interface VisionRows {
  /** The kinds this computer cannot make (the ones `mask_kinds` leaves out): their rows say so. Empty while the kinds are not known. */
  unsupported: readonly MaskKind[];
  /** The kinds whose newest mask for the photo found nothing (`vision_state`): their rows say "Nothing found". */
  nothing: readonly MaskKind[];
}

/** Whether a Subject or People row is of a kind this computer cannot make: it stays in the list and selects nothing. */
export const isUnsupported = (row: MaskItem, vision: VisionRows | undefined): boolean =>
  (row.kind === "subject" || row.kind === "people") && (vision?.unsupported.includes(row.kind) ?? false);

/** Whether a Subject or People row found nothing in the photo (the subtitle says so unless a new mask is being found). */
export const foundNothing = (row: MaskItem, vision: VisionRows | undefined): boolean =>
  (row.kind === "subject" || row.kind === "people") && row.enabled && (vision?.nothing.includes(row.kind) ?? false);

/** What the core is doing for the Object rows of the photo on screen: SAM's encoder is getting the photo ready, or a click is being answered (`docs/23` §7.6). */
export interface ObjectBusy {
  preparing: boolean;
  finding: boolean;
}

/**
 * What an Object row's subtitle says instead of its own while the core works for it (`docs/23` §7.6): "Finding…" while a click or a box is being answered,
 * "Getting ready…" while the encoder runs; only for a row with the eye on. Null for every other row, and when the core is idle.
 */
export function objectBusyText(row: MaskItem, busy: ObjectBusy | undefined): string | null {
  if (row.kind !== "object" || !row.enabled || !busy) return null;
  if (busy.finding) return FINDING;
  return busy.preparing ? GETTING_READY : null;
}

/** The row's name in bold: "Object", "Radial", "Colour range". */
export const maskRowName = (row: MaskItem): string => MASK_KIND_NAME[row.kind] ?? row.kind;

const plural = (n: number, one: string, many: string): string => `${n} ${n === 1 ? one : many}`;

/** The longest centre a Colour range may have: its distance from the middle of the chroma plane (`docs/23` §3.3). */
export const COLOR_RANGE_REACH = 0.5;

/** The most mask rows a node may hold (`docs/23` §3.5) and the sentence the core gives when it would hold more. */
export const MAX_MASKS = 64;
export const MASKS_FULL = "A node can have up to 64 masks.";

/**
 * A new row of a kind, with its defaults (`docs/23` §3.3): Add mode, not inverted, on, at full opacity. Every kind the settings know has
 * one, though the Mask tab builds the cards of the shapes and ranges only (the others come with their tasks).
 */
export function newMask(kind: MaskKind, mode: MaskMode = "add"): MaskItem {
  const base = { mode, invert: false, enabled: true, opacity: 1 } as const;
  const edge = { feather: 0, grow: 0, refine: 0.5 } as const;
  switch (kind) {
    case "object":
      return { ...base, ...edge, kind, points: [], box: null };
    case "subject":
    case "people":
      return { ...base, ...edge, kind };
    case "background":
      return { ...base, ...edge, kind, level: 0.5, softness: 0.5 };
    case "linear":
      return { ...base, kind, center: [0.5, 0.5], angle_deg: 0, feather: 0.2 };
    case "radial":
      return { ...base, kind, center: [0.5, 0.5], radii: [0.25, 0.18], angle_deg: 0, feather: 0.3 };
    case "brush":
      return { ...base, ...edge, refine: 0, kind, strokes: [] };
    case "color_range":
      return { ...base, kind, center: [0, 0], width: 0.05, softness: 0.5 };
    case "brightness_range":
      return { ...base, kind, low: 0.35, high: 1, softness: 0.5 };
  }
}

/**
 * The row patch that puts a row back as a new row of its kind, in the same mode (the selected mask's card's Reset, `docs/23` §7.6): every
 * field of the kind at its default but the mode. The row's kind is not in it (a row patch cannot change it).
 */
export function resetRowPatch(row: MaskItem): MaskPatch {
  const { kind: _kind, mode: _mode, ...fields } = newMask(row.kind, row.mode);
  return fields as MaskPatch;
}

/** A row patch for row `index`: `masks.edit` (`docs/23` §3.7), the small patch a drag sends. */
export const editRow = (index: number, edit: MaskPatch): MasksPatch => ({ edit: { [String(index)]: edit } });

/** The rows with `row` added at the end, or null when the node holds 64 already. */
export function withMask(rows: readonly MaskItem[], row: MaskItem): MasksPatch | null {
  return rows.length >= MAX_MASKS ? null : { items: [...rows, row] };
}

/** The rows without the one at `index`. */
export const withoutMask = (rows: readonly MaskItem[], index: number): MasksPatch => ({ items: rows.filter((_, i) => i !== index) });

/** The rows with the one at `index` moved by `by` places (−1 up, 1 down), or null at the end of the list. */
export function movedMask(rows: readonly MaskItem[], index: number, by: -1 | 1): MasksPatch | null {
  const to = index + by;
  const row = rows[index];
  if (!row || to < 0 || to >= rows.length) return null;
  const items = [...rows];
  items.splice(index, 1);
  items.splice(to, 0, row);
  return { items };
}

/** The row to select after the one at `index` is taken out of `count` rows: the next one, else the one before, else none. */
export const selectionAfterRemove = (index: number, count: number): number | null => (count <= 1 ? null : Math.min(index, count - 2));

/**
 * The subtitle of a row (`docs/23` §7.6): "2 points", "1 point", "Box", "Box and 2 points" and, with none, "Click the object on this photo"
 * (Object); "Foreground" (Subject); "People"; "Level 50" (Background); "Feather 20" (Linear, Radial); "3 strokes", "1 stroke" and "Paint on the
 * photo" (Brush); "Width 40" (Colour range); "35 to 100" (Brightness range). A Subject or People row says "Subject masks need a Mac that supports them"
 * where this computer cannot make it and "Nothing found" where Vision found no instance (`vision`). The state of a mask being made ("Finding…",
 * {@link isFinding}) is the row's own.
 */
export function maskSubtitle(row: MaskItem, vision?: VisionRows): string {
  switch (row.kind) {
    case "object": {
      const points = row.points.length;
      if (row.box && points > 0) return `Box and ${plural(points, "point", "points")}`;
      if (row.box) return "Box";
      return points > 0 ? plural(points, "point", "points") : "Click the object on this photo";
    }
    case "subject":
      return isUnsupported(row, vision) ? UNSUPPORTED_ROW.subject : foundNothing(row, vision) ? NOTHING_FOUND : "Foreground";
    case "people":
      return isUnsupported(row, vision) ? UNSUPPORTED_ROW.people : foundNothing(row, vision) ? NOTHING_FOUND : "People";
    case "background":
      return `Level ${Math.round(row.level * 100)}`;
    case "linear":
    case "radial":
      return `Feather ${Math.round(row.feather * 100)}`;
    case "brush":
      return row.strokes.length > 0 ? plural(row.strokes.length, "stroke", "strokes") : "Paint on the photo";
    case "color_range":
      return `Width ${widthSlider(row.width)}`;
    case "brightness_range":
      return `${Math.round(row.low * 100)} to ${Math.round(row.high * 100)}`;
  }
}

/**
 * The row's accessible name: the kind, "mask" and the subtitle ("Object mask, 2 points", "Radial mask, Feather 45"); with `finding`, a Background, Subject
 * or People row reads "Background mask, Finding…"; a Subject row on a computer that cannot make one reads "Subject mask, Subject masks need a Mac that
 * supports them".
 */
export const maskRowLabel = (row: MaskItem, finding = false, object?: ObjectBusy, vision?: VisionRows): string =>
  `${maskRowName(row)} mask, ${
    isUnsupported(row, vision) ? maskSubtitle(row, vision) : isFinding(row, finding) ? FINDING : (objectBusyText(row, object) ?? maskSubtitle(row, vision))
  }`;
