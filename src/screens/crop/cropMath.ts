/**
 * The geometry of the crop panel (`docs/11-ui.md` §Crop). Rectangles are in image coordinates
 * normalised to [0, 1]², as `crop.rect` is. A "pixel ratio" is width ÷ height measured in the
 * photo's pixels; a rectangle's own width ÷ height is in normalised units, and the two differ by
 * the photo's aspect (`imageAspect`).
 */
import type { CropAspect, NormRect, Orientation } from "../../api/types";

export type Rect = NormRect;

/** The whole photo. */
export const FULL_RECT: Rect = { x: 0, y: 0, width: 1, height: 1 };

/** Smallest side of a crop, as a share of the photo. */
export const MIN_CROP = 0.05;

const EPS = 1e-6;
const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

/** Width ÷ height of the photo after its orientation: a quarter turn swaps the sides. */
export function imageAspectOf(widthPx: number, heightPx: number, orientation: Orientation | number): number {
  const turned = orientation % 180 !== 0;
  const w = turned ? heightPx : widthPx;
  const h = turned ? widthPx : heightPx;
  return h > 0 ? w / h : 1.5;
}

/**
 * Where a picture of aspect `imageAspect` sits in a cell of aspect `cellAspect` when it is
 * letterboxed (`object-fit: contain`), as fractions of the cell.
 */
export function fitInCell(cellAspect: number, imageAspect: number): Rect {
  if (imageAspect >= cellAspect) {
    const height = cellAspect / imageAspect;
    return { x: 0, y: (1 - height) / 2, width: 1, height };
  }
  const width = imageAspect / cellAspect;
  return { x: (1 - width) / 2, y: 0, width, height: 1 };
}

/** True when two rectangles are the same to a thousandth. */
export const sameRect = (a: Rect, b: Rect): boolean =>
  Math.abs(a.x - b.x) < 1e-3 && Math.abs(a.y - b.y) < 1e-3 && Math.abs(a.width - b.width) < 1e-3 && Math.abs(a.height - b.height) < 1e-3;

/** A rectangle's width ÷ height in the photo's pixels. */
export const rectRatioPx = (rect: Rect, imageAspect: number): number => (rect.width / rect.height) * imageAspect;

/**
 * The width ÷ height in pixels an aspect preset asks for, or null for Free. `Original` is the
 * photo's own; the others follow the shape the crop already has (a landscape crop stays
 * landscape: 4:5 becomes 5:4).
 */
export function presetRatio(aspect: CropAspect, imageAspect: number, landscape: boolean): number | null {
  const base = { free: null, original: imageAspect, square: 1, four_by_five: 4 / 5, sixteen_by_nine: 16 / 9 }[aspect];
  if (base === null || aspect === "original" || aspect === "square") return base;
  return landscape ? Math.max(base, 1 / base) : Math.min(base, 1 / base);
}

/** The largest rectangle of pixel ratio `ratioPx` that fits inside `rect`, centred in it. */
export function fitRatio(rect: Rect, ratioPx: number, imageAspect: number): Rect {
  const ratioN = ratioPx / imageAspect; // width ÷ height in normalised units
  let width = rect.width;
  let height = width / ratioN;
  if (height > rect.height) {
    height = rect.height;
    width = height * ratioN;
  }
  return { x: rect.x + (rect.width - width) / 2, y: rect.y + (rect.height - height) / 2, width, height };
}

/** The rectangle after the photo turns a quarter turn clockwise: where the same part of the picture is now. */
export function rotateRect90(rect: Rect): Rect {
  return { x: 1 - (rect.y + rect.height), y: rect.x, width: rect.height, height: rect.width };
}

/** Which part of the frame a handle holds. */
export type Handle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

/** The eight handles, clockwise from the top left. */
export const HANDLES: readonly Handle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

/** Where a handle sits on a rectangle, in the same normalised units. */
export function handlePoint(rect: Rect, handle: Handle): { x: number; y: number } {
  const x = handle.includes("w") ? rect.x : handle.includes("e") ? rect.x + rect.width : rect.x + rect.width / 2;
  const y = handle.includes("n") ? rect.y : handle.includes("s") ? rect.y + rect.height : rect.y + rect.height / 2;
  return { x, y };
}

/** Keeps a rectangle inside the photo by sliding it, without changing its size. */
export function keepInside(rect: Rect): Rect {
  const width = Math.min(1, rect.width);
  const height = Math.min(1, rect.height);
  return { x: clamp(rect.x, 0, 1 - width), y: clamp(rect.y, 0, 1 - height), width, height };
}

/** Moves a rectangle by a distance in normalised units, stopping at the edges of the photo. */
export function moveRect(rect: Rect, dx: number, dy: number): Rect {
  return keepInside({ ...rect, x: rect.x + dx, y: rect.y + dy });
}

/**
 * Drags one handle of `start` to the point `to`. With `ratioPx` set the shape is locked: a corner
 * keeps the opposite corner still and follows the pointer along the diagonal; an edge moves and
 * the other side grows or shrinks about the middle. Without it every handle just moves its own
 * edges. The result stays inside the photo and is never smaller than `MIN_CROP` a side.
 */
export function resizeRect(
  start: Rect,
  handle: Handle,
  to: { x: number; y: number },
  ratioPx: number | null,
  imageAspect: number,
  min: number = MIN_CROP,
): Rect {
  const left0 = start.x;
  const top0 = start.y;
  const right0 = start.x + start.width;
  const bottom0 = start.y + start.height;
  const pointer = { x: clamp(to.x, 0, 1), y: clamp(to.y, 0, 1) };

  if (ratioPx === null) {
    let left = left0;
    let top = top0;
    let right = right0;
    let bottom = bottom0;
    if (handle.includes("w")) left = clamp(pointer.x, 0, right0 - min);
    if (handle.includes("e")) right = clamp(pointer.x, left0 + min, 1);
    if (handle.includes("n")) top = clamp(pointer.y, 0, bottom0 - min);
    if (handle.includes("s")) bottom = clamp(pointer.y, top0 + min, 1);
    return { x: left, y: top, width: right - left, height: bottom - top };
  }

  const ratioN = ratioPx / imageAspect; // width ÷ height, normalised
  const isCorner = handle.length === 2;
  if (isCorner) {
    const east = handle.includes("e");
    const south = handle.includes("s");
    const anchorX = east ? left0 : right0;
    const anchorY = south ? top0 : bottom0;
    let width = Math.max(Math.abs(pointer.x - anchorX), Math.abs(pointer.y - anchorY) * ratioN);
    // The room between the anchor and the photo's edges, along each axis.
    const roomX = east ? 1 - anchorX : anchorX;
    const roomY = south ? 1 - anchorY : anchorY;
    // Both sides stay at least `min`: the width by itself, and the height through the ratio.
    width = Math.max(width, min, min * ratioN);
    width = Math.min(width, roomX, roomY * ratioN);
    const height = width / ratioN;
    return { x: east ? anchorX : anchorX - width, y: south ? anchorY : anchorY - height, width, height };
  }

  const centerX = start.x + start.width / 2;
  const centerY = start.y + start.height / 2;
  if (handle === "e" || handle === "w") {
    const edge = handle === "e" ? pointer.x : left0;
    const fixed = handle === "e" ? left0 : pointer.x;
    let width = handle === "e" ? edge - fixed : right0 - pointer.x;
    // The height follows the width about the middle, so it must still fit above and below.
    const maxHeight = 2 * Math.min(centerY, 1 - centerY);
    width = clamp(width, min * Math.max(1, ratioN), Math.min(handle === "e" ? 1 - left0 : right0, maxHeight * ratioN));
    const height = width / ratioN;
    return { x: handle === "e" ? left0 : right0 - width, y: centerY - height / 2, width, height };
  }
  let height = handle === "s" ? pointer.y - top0 : bottom0 - pointer.y;
  const maxWidth = 2 * Math.min(centerX, 1 - centerX);
  height = clamp(height, min * Math.max(1, 1 / ratioN), Math.min(handle === "s" ? 1 - top0 : bottom0, maxWidth / ratioN));
  const width = height * ratioN;
  return { x: centerX - width / 2, y: handle === "s" ? top0 : bottom0 - height, width, height };
}

/** True when the rectangle is the whole photo. */
export const isFullRect = (rect: Rect): boolean => sameRect(rect, FULL_RECT) || (rect.width > 1 - EPS && rect.height > 1 - EPS);

/** The straighten angle as the label of the slider: "0°", "+1.5°", "−12.0°". */
export function angleLabel(degrees: number): string {
  if (Math.abs(degrees) < 0.05) return "0°";
  return `${degrees > 0 ? "+" : "−"}${Math.abs(degrees).toFixed(1)}°`;
}
