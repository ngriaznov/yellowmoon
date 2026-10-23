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

// No border in photos (the owner's decision of 2026-10-22, `docs/11-ui.md` §Crop): a straightened photo leaves empty corners on its
// canvas, and the frame must never reach them. The geometry is the core's (`ym_core::pipeline::geometry`): the canvas is the photo after
// its quarter turns, W × H pixels; an output point q of the canvas samples p = c + L(q − c) about the centre c = (W/2, H/2), where
// L(dx, dy) = (cos θ·dx + sin θ·dy, −sin θ·dx + cos θ·dy) and θ is the straighten angle (clockwise positive, y down). A p outside the
// canvas is black. A frame is inside when its four corners sample the photo at least `FIT_MARGIN_PX` from its edge, and it lies on the
// canvas; at θ = 0 it only has to lie on the canvas.

/** The canvas: the photo after its quarter turns, in its full-resolution pixels. */
export interface Canvas {
  width: number;
  height: number;
}

/**
 * How far, in canvas pixels at full resolution, a straightened frame keeps from the edge of the photo: the reduced views round their
 * sides to whole pixels, and the margin keeps their corners filled too (`ym_core::pipeline::geometry::FIT_MARGIN_PX`).
 */
export const FIT_MARGIN_PX = 2;

/** The largest straighten angle that counts as none, in degrees (the core's `ANGLE_EPS_DEG`). */
const ANGLE_EPS_DEG = 1e-9;

/** How far past a bound, in pixels, a corner may be and still count as inside: floating-point noise, not a margin. */
const INSIDE_TOLERANCE_PX = 1e-6;

/** The canvas of a photo of `widthPx × heightPx` turned by `orientation`: a quarter turn swaps the sides. */
export function canvasOf(widthPx: number, heightPx: number, orientation: Orientation | number): Canvas {
  const turned = orientation % 180 !== 0;
  return turned ? { width: heightPx, height: widthPx } : { width: widthPx, height: heightPx };
}

/** Clamps a rectangle to [0, 1]² as the core does: the corner first, then the size. */
function clampToUnit(rect: Rect): Rect {
  const x = clamp(rect.x, 0, 1);
  const y = clamp(rect.y, 0, 1);
  return { x, y, width: clamp(rect.width, 0, 1 - x), height: clamp(rect.height, 0, 1 - y) };
}

/** The straighten map of a canvas: its linear part, its centre and the bounds a sample must keep to. */
interface Straighten {
  cos: number;
  sin: number;
  w: number;
  h: number;
  m: number;
}

function straightenOf(angleDeg: number, canvas: Canvas, marginPx: number): Straighten {
  // As Rust's `f64::to_radians` rounds it.
  const t = angleDeg * (Math.PI / 180);
  const { width: w, height: h } = canvas;
  return { cos: Math.cos(t), sin: Math.sin(t), w, h, m: clamp(marginPx, 0, 0.25 * Math.min(w, h)) };
}

/** Where the canvas point (x, y), in pixels, samples the photo. */
function sampleAt(g: Straighten, x: number, y: number): [number, number] {
  const dx = x - g.w / 2;
  const dy = y - g.h / 2;
  return [g.cos * dx + g.sin * dy + g.w / 2, -g.sin * dx + g.cos * dy + g.h / 2];
}

/** The four corners of a rectangle in canvas pixels. */
function cornersPx(rect: Rect, canvas: Canvas): Array<[number, number]> {
  const left = rect.x * canvas.width;
  const top = rect.y * canvas.height;
  const right = (rect.x + rect.width) * canvas.width;
  const bottom = (rect.y + rect.height) * canvas.height;
  return [
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom],
  ];
}

const usable = (canvas: Canvas): boolean => canvas.width > 0 && canvas.height > 0;

/**
 * True when the frame shows no empty corner of the photo straightened by `angleDeg` (the rule of `docs/11-ui.md` §Crop): it lies in
 * [0, 1]², and unless the angle is 0 its four corners sample the photo at least `marginPx` from its edge.
 */
export function isInside(rect: Rect, angleDeg: number, canvas: Canvas, marginPx: number = FIT_MARGIN_PX): boolean {
  const tol = INSIDE_TOLERANCE_PX / Math.max(1, canvas.width, canvas.height);
  const onCanvas = rect.x >= -tol && rect.y >= -tol && rect.x + rect.width <= 1 + tol && rect.y + rect.height <= 1 + tol;
  if (!onCanvas) return false;
  if (Math.abs(angleDeg) < ANGLE_EPS_DEG || !usable(canvas)) return true;
  const g = straightenOf(angleDeg, canvas, marginPx);
  const lo = g.m - INSIDE_TOLERANCE_PX;
  return cornersPx(rect, canvas).every(([x, y]) => {
    const [px, py] = sampleAt(g, x, y);
    return px >= lo && px <= g.w - lo && py >= lo && py <= g.h - lo;
  });
}

/**
 * `rect` made to lie inside the photo straightened by `angleDeg`, so that the picture has no empty corner: the same function as the
 * core's `fit_inside` (`ym_core::pipeline::geometry`), step for step.
 *
 * The frame is clamped to [0, 1]² (the corner, then the size). At an angle of 0 that is all, so the whole photo stays exactly the whole
 * photo. Otherwise it shrinks about its own centre, keeping its shape in pixels, by as little as makes its four corners sample the photo at
 * least `marginPx` from its edge and keeps it on the canvas: each corner moves linearly with the scale, so every bound gives its limit
 * directly. A frame already inside is returned as it is. A frame whose centre samples outside the photo becomes the largest frame of its
 * shape centred on the canvas, fitted the same way.
 */
export function fitInside(rect: Rect, angleDeg: number, canvas: Canvas, marginPx: number = FIT_MARGIN_PX): Rect {
  const r = clampToUnit(rect);
  if (Math.abs(angleDeg) < ANGLE_EPS_DEG || !usable(canvas) || r.width <= 0 || r.height <= 0) return r;
  const g = straightenOf(angleDeg, canvas, marginPx);
  const { w, h, m } = g;
  /** The largest scale (at most 1) of the frame centred at (cx, cy) with half sides (a, b) in pixels; null when its centre samples outside. */
  const scale = (cx: number, cy: number, a: number, b: number): number | null => {
    const [px, py] = sampleAt(g, cx, cy);
    if (px < m || px > w - m || py < m || py > h - m) return null;
    let s = 1;
    const limit = (start: number, step: number, lo: number, hi: number): void => {
      if (step > 0) s = Math.min(s, (hi - start) / step);
      else if (step < 0) s = Math.min(s, (lo - start) / step);
    };
    for (const [sx, sy] of [
      [-1, -1],
      [1, -1],
      [-1, 1],
      [1, 1],
    ] as const) {
      const dx = g.cos * sx * a + g.sin * sy * b;
      const dy = -g.sin * sx * a + g.cos * sy * b;
      limit(px, dx, m, w - m);
      limit(py, dy, m, h - m);
      // The canvas itself, unturned.
      limit(cx, sx * a, 0, w);
      limit(cy, sy * b, 0, h);
    }
    return Math.max(s, 0);
  };
  const frame = (cx: number, cy: number, a: number, b: number, s: number): Rect => ({
    x: (cx - s * a) / w,
    y: (cy - s * b) / h,
    width: (2 * s * a) / w,
    height: (2 * s * b) / h,
  });
  const a = 0.5 * r.width * w;
  const b = 0.5 * r.height * h;
  const cx = (r.x + 0.5 * r.width) * w;
  const cy = (r.y + 0.5 * r.height) * h;
  const s = scale(cx, cy, a, b);
  if (s !== null) return s >= 1 ? r : clampToUnit(frame(cx, cy, a, b, s));
  // The centre is outside: the frame of the same shape that fills the canvas, centred, fitted.
  const ratio = a / b;
  const [a2, b2] = ratio >= w / h ? [0.5 * w, (0.5 * w) / ratio] : [0.5 * h * ratio, 0.5 * h];
  // The canvas centre samples itself, which the margin's cap of a quarter of the shorter side keeps inside.
  const s2 = scale(0.5 * w, 0.5 * h, a2, b2) ?? 0;
  return clampToUnit(frame(0.5 * w, 0.5 * h, a2, b2, s2));
}

/** The rectangle a share `t` of the way from `from` to `to`. */
const lerpRect = (from: Rect, to: Rect, t: number): Rect => ({
  x: from.x + t * (to.x - from.x),
  y: from.y + t * (to.y - from.y),
  width: from.width + t * (to.width - from.width),
  height: from.height + t * (to.height - from.height),
});

/**
 * The frame as far from `from` towards `to` as it can go and stay inside: the largest t in [0, 1] for which the rectangle a share t of the
 * way is inside. Every corner, and where it samples the photo, moves linearly with t, and the inside of the photo is convex, so every
 * bound gives its limit directly (the closed form of a bisection between the two). `from` is taken to be inside.
 */
function towards(from: Rect, to: Rect, angleDeg: number, canvas: Canvas, marginPx: number): Rect {
  if (isInside(to, angleDeg, canvas, marginPx)) return to;
  let t = 1;
  const limit = (start: number, step: number, lo: number, hi: number): void => {
    if (step > 0) t = Math.min(t, (hi - start) / step);
    else if (step < 0) t = Math.min(t, (lo - start) / step);
  };
  const a = cornersPx(from, canvas);
  const b = cornersPx(to, canvas);
  const level = Math.abs(angleDeg) < ANGLE_EPS_DEG || !usable(canvas);
  const g = straightenOf(level ? 0 : angleDeg, canvas, level ? 0 : marginPx);
  for (let k = 0; k < 4; k++) {
    const [ax, ay] = a[k]!;
    const [bx, by] = b[k]!;
    // On the canvas.
    limit(ax, bx - ax, 0, canvas.width);
    limit(ay, by - ay, 0, canvas.height);
    if (level) continue;
    // Where it samples the photo.
    const [pax, pay] = sampleAt(g, ax, ay);
    const [pbx, pby] = sampleAt(g, bx, by);
    limit(pax, pbx - pax, g.m, g.w - g.m);
    limit(pay, pby - pay, g.m, g.h - g.m);
  }
  return lerpRect(from, to, Math.max(0, t));
}

/**
 * A frame moved or resized by a drag or an arrow key (`to`), kept inside the photo straightened by `angleDeg` (`docs/11-ui.md` §Crop).
 * `from` is the frame shown before the step, which is inside. A `to` that is inside is returned as it is, so at an angle of 0 (where
 * `moveRect` and `resizeRect` already keep to the photo) nothing changes. Otherwise:
 *
 * - With the shape locked and the size changing (a handle of a frame with an aspect), the frame goes as far towards `to` as it can along
 *   the straight way between the two, which keeps the shape and the handle's anchor.
 * - Otherwise (a move, or a handle of a Free frame) it goes as far as it can across first and then up or down, so a frame pushed against a
 *   slanted edge slides along it instead of stopping where it first touches.
 */
export function constrainFrame(from: Rect, to: Rect, angleDeg: number, canvas: Canvas, lockShape: boolean, marginPx: number = FIT_MARGIN_PX): Rect {
  if (isInside(to, angleDeg, canvas, marginPx)) return to;
  const resized = Math.abs(to.width - from.width) > 1e-12 || Math.abs(to.height - from.height) > 1e-12;
  if (lockShape && resized) return towards(from, to, angleDeg, canvas, marginPx);
  const across = towards(from, { x: to.x, y: from.y, width: to.width, height: from.height }, angleDeg, canvas, marginPx);
  return towards(across, { x: across.x, y: to.y, width: across.width, height: to.height }, angleDeg, canvas, marginPx);
}

/** The straighten angle as the label of the slider: "0°", "+1.5°", "−12.0°". */
export function angleLabel(degrees: number): string {
  if (Math.abs(degrees) < 0.05) return "0°";
  return `${degrees > 0 ? "+" : "−"}${Math.abs(degrees).toFixed(1)}°`;
}
