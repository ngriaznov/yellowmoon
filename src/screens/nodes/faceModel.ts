/**
 * What the Face Refinement panel says and sends, without the drawing (`docs/27` S-11c). Pure: the cards (`FaceTab.tsx`) and the overlay on
 * the photo (`FaceLayer.tsx`) call these, and the tests do.
 */
import type { FaceSettings, FaceState } from "../../api/types";
import { defaultFaceSettings, faceSettingsFor } from "../../state/settingsGraph";
import type { FaceRefinementItem } from "../../state/settingsGraph";

/** Most faces a node holds settings for (`docs/27` S-11a). */
export const MAX_FACES = 32;

/** The quiet line of the node's head: how many faces there are, or what the worker is doing (`docs/27` decision 9). */
export function facesLine(state: FaceState | undefined): string {
  if (!state || state.phase === "finding") return "Finding faces";
  if (state.phase === "failed") return "Can't find faces";
  if (state.phase === "models_missing") return "Face models missing";
  const n = state.faces.length;
  return n === 0 ? "No faces found" : n === 1 ? "1 face" : `${n} faces`;
}

/** The number of the face the tab is on: the one asked for, held to 1 … `count` (1 with no faces). */
export const selectedFace = (selected: number, count: number): number => Math.min(Math.max(1, selected), Math.max(1, count));

/** What a card says it acts on: "All faces" while Same for all faces is on, else "Face 2". */
export const targetName = (item: FaceRefinementItem, count: number, number: number): string => (item.same_for_all && count > 1 ? "All faces" : `Face ${number}`);

/** The title of the Face card: "Face" with one face, "All faces", or "Face 2". */
export const faceCardTitle = (item: FaceRefinementItem, count: number, number: number): string =>
  count <= 1 ? "Face" : item.same_for_all ? "All faces" : `Face ${number}`;

/** The settings the cards show: `all` while Same for all faces is on, else the face's own. */
export const shownSettings = (item: FaceRefinementItem, number: number): FaceSettings => faceSettingsFor(item, number);

/** The entry of `faces` for face index `i` (from 0) as the core reads it: its own, face 1's past the end, `all` for an empty list. */
const entryOf = (item: FaceRefinementItem, i: number): FaceSettings => item.faces[i] ?? item.faces[0] ?? item.all;

/**
 * The list `faces` becomes when face `number` takes `partial`: every face up to the count (or the list's own length, or the face) has an entry, the missing ones
 * as the core would read them, and the face's entry has the change. At most 32 entries.
 */
export function facesWith(item: FaceRefinementItem, count: number, number: number, partial: Partial<FaceSettings>): FaceSettings[] {
  const length = Math.min(MAX_FACES, Math.max(count, number, item.faces.length));
  const list = Array.from({ length }, (_, i) => ({ ...entryOf(item, i) }));
  const at = number - 1;
  if (at >= 0 && at < length) list[at] = { ...list[at], ...partial } as FaceSettings;
  return list;
}

/** The list `faces` becomes on "Copy to other faces": every face but `number` takes face `number`'s settings, which are kept as they are. */
export function facesCopied(item: FaceRefinementItem, count: number, number: number): FaceSettings[] {
  const length = Math.min(MAX_FACES, Math.max(count, number, item.faces.length));
  const mine = { ...entryOf(item, number - 1) };
  return Array.from({ length }, (_, i) => (i === number - 1 ? mine : { ...mine }));
}

/**
 * The list `faces` starts as when Same for all faces is turned off: the list it has, or, when it is empty, one copy of `all` for every face (`docs/27`
 * S-11c: each face starts from what they all had).
 */
export function facesOnSplit(item: FaceRefinementItem, count: number): FaceSettings[] {
  if (item.faces.length > 0) return item.faces;
  return Array.from({ length: Math.min(MAX_FACES, Math.max(1, count)) }, () => ({ ...item.all }));
}

/** The amounts of the Face card, and where Reset puts them back. */
export const FACE_FIELDS = ["blemish", "dark_circles", "even_skin", "texture"] as const;
export const GRADING_FIELDS = ["skin_contrast", "shadow_desaturation", "shine_removal"] as const;
export const TEETH_FIELDS = ["teeth_whitening"] as const;

/** The partial that puts `fields` back to their defaults (the Face card's Reset) or to 0 (the Skin grading and Teeth cards' Reset). */
export function resetPartial(fields: readonly (keyof FaceSettings)[], to: "defaults" | "zero"): Partial<FaceSettings> {
  const defaults = defaultFaceSettings();
  return Object.fromEntries(fields.map((field) => [field, to === "zero" ? 0 : defaults[field]]));
}

/** The long names the tooltips and the page use. */
export const FACE_TEXT = {
  finding: "Finding faces…",
  findingLine: "The first time takes a few seconds. The photo stays as it is until then.",
  none: "No faces found in this photo.",
  noneLine:
    "Face Refinement works on faces that are at least 200 px across and turned towards the camera. Try another photo, or delete this node. It changes nothing until it finds a face.",
  sameForAll: "Same for all faces",
  sameForAllTip: "One set of controls for every face. Turn it off to set each face on its own.",
  eye: "Show skin areas",
  eyeTip: "Show where the tools work on this face",
  eyeOnTip: "Hide the skin areas",
  contrastTip: "Adds or removes contrast in the skin. Fine detail stays.",
  shadowTip: "Lowers the colour in the skin's shadows.",
  shineTip: "Reduces shine on the forehead, nose and cheeks.",
  whiteningTip: "Makes teeth less yellow and a little brighter.",
  gradingResetTip: "Sets Contrast, Shadow desaturation and Shine removal back to 0.",
  teethResetTip: "Sets Whitening back to 0.",
  copy: "Copy to other faces",
  copyTip: "Gives every other face these settings: Face, Skin grading and Teeth.",
  protectLine: "Paint over moles, freckles and scars you want to keep.",
  protectActive: "Paint on the photo. Hold ⌥ to erase.",
  faceTip: (n: number): string => `Face ${n}. Click to retouch it on its own.`,
  faceTipAll: (n: number): string => `Face ${n}. Click to show it on the photo.`,
} as const;
