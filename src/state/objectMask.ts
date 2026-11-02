/**
 * What the Object mask says and keeps on the page (`docs/23` §4.4 and §7.6): the tool, the words, the pill and the points as the layer draws them. Pure, so
 * the card, the layer and the store share the sentences and the tests need no screen.
 */
import type { MaskFindPrompt, MaskItem } from "../api/types";

/** The Object card's tool (`docs/23` §7.6): a plain click adds a point, Remove point takes a part away, Box draws a box. */
export type ObjectMode = "add" | "remove" | "box";

/** The tool a new Object row starts with. */
export const DEFAULT_OBJECT_MODE: ObjectMode = "add";

/** What the error bar says when a click or a box finds nothing (`docs/23` §4.4), word for word. */
export const NOTHING_FOUND = "Nothing found there. Click another part of the object, or draw a box.";

/** What the subtitle of an Object row says while the photo's features are being made (`docs/23` §7.6). */
export const GETTING_READY = "Getting ready…";

/** How long the pill "Object found · 0.4 s" stays on the photo, in milliseconds. */
export const FOUND_PILL_MS = 3000;

/** The pill's words: the decoder's time in seconds, one decimal ("Object found · 0.4 s"). */
export const foundText = (tookMs: number): string => `Object found · ${(Math.max(0, tookMs) / 1000).toFixed(1)} s`;

/** The line under the tool (`docs/23` §7.6). */
export function objectLine(mode: ObjectMode, on: boolean, enabled: boolean): string {
  if (!enabled) return "This mask is off. Turn it on to find an object.";
  if (!on) return "Choose a tool to find an object on the photo.";
  switch (mode) {
    case "add":
      return "Click the photo to add to the object. ⌥-click to take a part away.";
    case "remove":
      return "Click the photo to take a part away from the object.";
    case "box":
      return "Drag a box around the object.";
  }
}

/** The sentence the core gives for a row that has all its points; the page asks no more of it. */
export const OBJECT_POINTS_FULL = "An Object mask can have up to 256 points.";
/** The most points an Object row holds (`docs/23` §3.3). */
export const MAX_OBJECT_POINTS = 256;

/** A drag of the Box tool shorter than this on either side, in CSS pixels, is a click, not a box. */
export const BOX_MIN_PX = 6;

/**
 * What a click does with the tool (`docs/23` §4.4): the sign of the point it adds. ⌥-click takes a part away whatever the tool is; with the Box tool a
 * click is not a find (a box is a drag).
 */
export function clickPrompt(mode: ObjectMode, alt: boolean, x: number, y: number): MaskFindPrompt | null {
  if (mode === "box" && !alt) return null;
  return { kind: "point", x, y, positive: mode === "add" && !alt };
}

/** A box between two photo-frame corners, in either order, as a `mask_find` prompt. */
export function boxPrompt(a: [number, number], b: [number, number]): MaskFindPrompt {
  return { kind: "box", x0: Math.min(a[0], b[0]), y0: Math.min(a[1], b[1]), x1: Math.max(a[0], b[0]), y1: Math.max(a[1], b[1]) };
}

/** An Object row's fields (`docs/23` §3.3). */
export type ObjectRow = Extract<MaskItem, { kind: "object" }>;
