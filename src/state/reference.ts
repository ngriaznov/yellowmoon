/**
 * What the page knows of a Reference Grade node's reference and fit, for the session (`docs/34` §9.2, §9.3 and §3.7): the words of the card, the
 * status-line state machine and the helpers it uses. Nothing here is saved: a reference, its name, a fit's status and the flag "edited since the fit"
 * live in the store while the window is open and are never written to the settings, a preset or an album (`docs/34` §3.7).
 */
import type { DevelopSettings, ElementShare, NodeId, ReferenceInfo } from "../api/types";
import { newReferenceGrade, nodeOf, outputOf } from "./settingsGraph";
import type { ReferenceGradeItem } from "./settingsGraph";

/** Every string of the Reference tab (`docs/34` §9.3 and §9.2), character for character; a test pins each one. */
export const REFERENCE_TEXT = {
  // The status line, one at a time.
  noReference: "Choose a picture to match.",
  pressFit: "Press Fit to match the photo to this picture.",
  makingScene: "Finding what is in the pictures…",
  fitted: "Fitted to the reference.",
  alreadyClose: "The photo is already close to the reference. Nothing was changed.",
  edited: "Edited by hand since the fit. Fit again to start over.",
  lookChanged: "The look changed since the fit. Fit again to match it with the new look.",
  valuesKept: "The values are kept. Choose a reference to fit again.",
  // The notes under the status.
  spatialAfter: "Nodes after this one that blur or add film aren't part of the fit.",
  sceneMissingNote: "The scene model is missing from this install, so element grades aren't applied. Reinstall Yellowmoon.",
  blackBars: "Black bars were removed from the reference.",
  profileUnreadable: "The picture's colour profile couldn't be read, so it is taken as sRGB.",
  noColour: "This picture has no colour. The photo will be matched in black and white.",
  noSkin: "No skin was found, so Skin protection did nothing.",
  // The errors, shown in the card as a notice with the text and no code.
  unreadable: "This file can't be read. It may be damaged. Choose another picture.",
  unsupported: "Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again.",
  tooSmall: "This picture is too small to match. Use one that is at least 256 pixels on its short side.",
  empty: "There's no picture inside the black bars. Choose another picture.",
  samePhoto: "That's the photo you're grading. Choose another.",
  notReady: "This photo isn't ready yet.",
  fitFailed: "The grade couldn't be fitted. Try another reference.",
  sceneMissing: "The scene model is missing from this install. Reinstall Yellowmoon.",
  // The card's own lines and controls.
  chooseFirst: "Choose a reference first.",
  strengthLine: "Scales the fitted values. The tabs show them before Strength.",
  skinLine: "Keeps skin tones from drifting. Applies at the next fit.",
  noReferenceName: "No reference",
  chooserHint: "Click a photo in the strip. Esc cancels.",
  chooserThisPhoto: "This photo",
  chooserDeveloping: "Developing",
} as const;

/** The notes the card can show, in the order it shows them (`docs/34` §9.3): the reference's own, then the fit's, then the two about the graph and the install. */
const NOTE_ORDER: readonly string[] = [
  REFERENCE_TEXT.blackBars,
  REFERENCE_TEXT.profileUnreadable,
  REFERENCE_TEXT.noColour,
  REFERENCE_TEXT.noSkin,
  REFERENCE_TEXT.spatialAfter,
  REFERENCE_TEXT.sceneMissingNote,
];

/** Where a fit is while it runs (`fit_progress`): the scene is being made, or search `search` of `of`, `fraction` of the way. */
export type FitRun = { step: "scene" } | { step: "search"; search: number; of: number; fraction: number };

/** What the last fit of this session left, for the status line: how it ended, the look it was made under and the numbers it wrote. */
export interface FitResult {
  status: "fitted" | "already_close";
  /** The Output's look and intensity at the fit ({@link lookKey}); a different one now makes the fit stale. */
  look: string;
  /** The fitted fields as the fit left them ({@link fittedKey}); a different one now is a hand edit. */
  fitted: string;
  /** The notes the fit made (the reference's own and "No skin was found…"). */
  notes: string[];
  /** The six elements' shares of the photo (`found`), 0 to 1. */
  shares: ElementShare[];
}

/** What the page holds for one Reference Grade node of one photo, for the session. */
export interface ReferenceCard {
  /** The reference the core holds: its label, notes and picture; null until one is chosen. */
  info: ReferenceInfo | null;
  /** A fit is running. */
  fit: FitRun | null;
  /** The last fit of this session; cleared when another reference is chosen. */
  result: FitResult | null;
  /** The sentence of the last error of the card (a file that cannot be used, a fit that failed), or null. */
  error: string | null;
  /** A fit said the scene model is missing from this install. */
  sceneMissing: boolean;
}

/** A card with nothing in it. */
export const EMPTY_CARD: ReferenceCard = { info: null, fit: null, result: null, error: null, sceneMissing: false };

/** The key of a node's card in the store: the photo and the node. */
export const cardKey = (photo: number, node: NodeId): string => `${photo}:${String(node)}`;

/** The look the fit depends on: the Output's look and its intensity (`docs/34` §2.2). Anything else about the chain is an ordinary edit. */
export function lookKey(settings: DevelopSettings): string {
  const output = outputOf(settings);
  return `${output.look}|${output.look_intensity}`;
}

/** The fitted fields of a node, as one string to compare: the four curves, the saturation, both hue curves and all seven elements (`docs/34` §3.2). */
export function fittedKey(item: ReferenceGradeItem): string {
  return JSON.stringify([item.grade.curve, item.grade.saturation, item.hue_curves, item.elements]);
}

const IDENTITY_KEY = fittedKey(newReferenceGrade(0, [0, 0]));

/** Whether the node holds fitted values: its fitted fields are not the identity. */
export const hasFittedValues = (item: ReferenceGradeItem): boolean => fittedKey(item) !== IDENTITY_KEY;

/** Whether any of the node's elements has a grade (it needs the scene model to be drawn). */
export function hasElementGrades(item: ReferenceGradeItem): boolean {
  const { subject, ...rest } = item.elements;
  return subject.exposure_ev !== 0 || Object.values(rest).some((e) => e.exposure_ev !== 0 || e.hue_deg !== 0 || e.chroma !== 1);
}

/** Why the fit has gone stale or been left by hand: what the status line says about the card, besides the text. */
export type ReferenceStatus = "none" | "ready" | "running" | "fitted" | "already_close" | "edited" | "look_changed" | "values_kept";

/** The state of a card and its status line, one at a time (`docs/34` §9.3). */
export function referenceStatus(card: ReferenceCard | undefined, item: ReferenceGradeItem, settings: DevelopSettings): { state: ReferenceStatus; text: string } {
  const c = card ?? EMPTY_CARD;
  if (c.fit) {
    return { state: "running", text: c.fit.step === "scene" ? REFERENCE_TEXT.makingScene : `Fitting… ${c.fit.search} of ${c.fit.of}` };
  }
  if (!c.info) {
    return hasFittedValues(item) ? { state: "values_kept", text: REFERENCE_TEXT.valuesKept } : { state: "none", text: REFERENCE_TEXT.noReference };
  }
  if (!c.result) return { state: "ready", text: REFERENCE_TEXT.pressFit };
  // A look that is not the one the fit was made under is the stale state and has the accent; a hand edit does not (`docs/34` §2.2, §3.3).
  if (c.result.look !== lookKey(settings)) return { state: "look_changed", text: REFERENCE_TEXT.lookChanged };
  if (c.result.fitted !== fittedKey(item)) return { state: "edited", text: REFERENCE_TEXT.edited };
  return c.result.status === "already_close"
    ? { state: "already_close", text: REFERENCE_TEXT.alreadyClose }
    : { state: "fitted", text: REFERENCE_TEXT.fitted };
}

/** What the Fit button is and says (`docs/34` §9.2, `design/reference-grade/mockup-notes.md` item 3). */
export interface FitButton {
  label: "Fit" | "Fit again";
  /** Off, with the tooltip `Choose a reference first.`. */
  disabled: boolean;
  /** The prominent look: Fit with a reference, and Fit again when the look changed. */
  prominent: boolean;
  /** The stale look's ring and glow (`docs/34` §2.2). */
  accent: boolean;
}

/** The Fit button for a card in this state. */
export function fitButton(state: ReferenceStatus): FitButton {
  switch (state) {
    case "none":
    case "values_kept":
      return { label: "Fit", disabled: true, prominent: false, accent: false };
    case "ready":
      return { label: "Fit", disabled: false, prominent: true, accent: false };
    case "look_changed":
      return { label: "Fit again", disabled: false, prominent: true, accent: true };
    default:
      return { label: "Fit again", disabled: false, prominent: false, accent: false };
  }
}

/** The kinds of node whose picture is made before the colour pass, which the fit does not run (`docs/34` §2.2). */
const SPATIAL_KINDS: ReadonlySet<string> = new Set(["film", "blur", "face_refinement"]);

/** Whether a Film, Blur or Face Refinement node comes after this one on the way to the Output: the card says they are not part of the fit. */
export function hasSpatialAfter(settings: DevelopSettings, node: NodeId): boolean {
  const links = settings.nodes.links;
  const seen = new Set<NodeId>();
  const walk = (id: NodeId): boolean => {
    for (const link of links) {
      if (link.from.node !== id || seen.has(link.to.node)) continue;
      seen.add(link.to.node);
      if (SPATIAL_KINDS.has(nodeOf(settings, link.to.node)?.kind ?? "")) return true;
      if (walk(link.to.node)) return true;
    }
    return false;
  };
  return walk(node);
}

/** The notes under the status line, each once, in the order of {@link NOTE_ORDER} (`docs/34` §9.3). */
export function referenceNotes(card: ReferenceCard | undefined, item: ReferenceGradeItem, settings: DevelopSettings, node: NodeId): string[] {
  const c = card ?? EMPTY_CARD;
  const found = new Set<string>([...(c.info?.notes ?? []), ...(c.result?.notes ?? [])]);
  if (hasSpatialAfter(settings, node)) found.add(REFERENCE_TEXT.spatialAfter);
  if (c.sceneMissing && hasElementGrades(item)) found.add(REFERENCE_TEXT.sceneMissingNote);
  const known = NOTE_ORDER.filter((text) => found.has(text));
  // A sentence the core adds that this page does not know yet still shows, after the known ones.
  return [...known, ...[...found].filter((text) => !NOTE_ORDER.includes(text))];
}

/** One element of the Elements tab and what the last fit found of it. */
export const ELEMENT_NAMES = ["sky", "vegetation", "water", "ground", "built", "skin"] as const;
export type ElementName = (typeof ELEMENT_NAMES)[number];

/** An element's share of the photo as the last fit found it, 0 to 1; null before any fit. */
export function shareOf(card: ReferenceCard | undefined, element: ElementName): number | null {
  const found = card?.result?.shares.find((s) => s.element === element);
  return found ? found.share : null;
}

/** An element that covers under 2 % of the photo is "not found" (`docs/34` §9.4a); unknown until a fit has looked. */
export const NOT_FOUND_SHARE = 0.02;

/** "26 % of the photo", the share the card's header shows (a share under half a percent reads "<1 %"). */
export function shareText(share: number): string {
  return share > 0 && share * 100 < 1 ? "<1 % of the photo" : `${Math.round(share * 100)} % of the photo`;
}
