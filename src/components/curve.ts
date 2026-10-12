/**
 * Curve maths for the Curve tab: the monotone cubic spline of `docs/02-colour-pipeline.md`
 * §C11 (Fritsch–Carlson), and the editing operations the plot needs (insert, move, remove).
 *
 * The core bakes the same spline into four 1024-entry 1D textures, so the plot and the render
 * agree. Points are `[x, y]` in [0, 1]², sorted by x, at least two and at most
 * `MAX_CURVE_POINTS`. Outside the first and last point the curve is flat.
 */
import type { CurvePoint } from "../api/types";
import { clamp } from "./format";

export type { CurvePoint } from "../api/types";

/** Most control points a curve may have (`docs/01` §DevelopSettings). */
export const MAX_CURVE_POINTS = 16;
/** Fewest control points: the two ends. */
export const MIN_CURVE_POINTS = 2;
/** Entries in the baked 1D curve (`docs/02` §C11). */
export const CURVE_LUT_SIZE = 1024;
/** Closest two control points may sit on the x axis. */
export const MIN_POINT_GAP = 1 / 255;

/** The identity curve. A fresh array every call, so callers may mutate it. */
export const identityCurve = (): CurvePoint[] => [
  [0, 0],
  [1, 1],
];

/** True when the curve is the identity: two points, `[0,0]` and `[1,1]`. */
export function isIdentityCurve(points: readonly CurvePoint[]): boolean {
  const first = points[0];
  const last = points[points.length - 1];
  return points.length === 2 && !!first && !!last && first[0] === 0 && first[1] === 0 && last[0] === 1 && last[1] === 1;
}

/**
 * Tangents of the monotone cubic spline through `points` (Fritsch & Carlson, 1980): secant
 * averages, zeroed at local extrema, then limited so each segment stays monotone.
 */
export function monotoneTangents(points: readonly CurvePoint[]): number[] {
  const n = points.length;
  const tangents = new Array<number>(n).fill(0);
  if (n < 2) return tangents;
  const dx: number[] = [];
  const secant: number[] = [];
  for (let k = 0; k < n - 1; k++) {
    const a = points[k] as CurvePoint;
    const b = points[k + 1] as CurvePoint;
    const h = b[0] - a[0];
    dx.push(h);
    secant.push(h > 0 ? (b[1] - a[1]) / h : 0);
  }
  tangents[0] = secant[0] as number;
  tangents[n - 1] = secant[n - 2] as number;
  for (let k = 1; k < n - 1; k++) {
    const left = secant[k - 1] as number;
    const right = secant[k] as number;
    tangents[k] = left * right <= 0 ? 0 : (left + right) / 2;
  }
  for (let k = 0; k < n - 1; k++) {
    const d = secant[k] as number;
    if (d === 0) {
      tangents[k] = 0;
      tangents[k + 1] = 0;
      continue;
    }
    const alpha = (tangents[k] as number) / d;
    const beta = (tangents[k + 1] as number) / d;
    const s = alpha * alpha + beta * beta;
    if (s > 9) {
      const tau = 3 / Math.sqrt(s);
      tangents[k] = tau * alpha * d;
      tangents[k + 1] = tau * beta * d;
    }
  }
  return tangents;
}

/** Index of the segment `[points[k], points[k+1]]` that holds `x`, for x inside the curve. */
function segmentOf(points: readonly CurvePoint[], x: number): number {
  let lo = 0;
  let hi = points.length - 2;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if ((points[mid] as CurvePoint)[0] <= x) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** The curve's value at `x`, clamped to [0, 1]. Flat outside the first and last point. */
export function evaluateCurve(points: readonly CurvePoint[], x: number, tangents = monotoneTangents(points)): number {
  const first = points[0];
  const last = points[points.length - 1];
  if (!first || !last) return clamp(x, 0, 1);
  if (points.length === 1 || x <= first[0]) return clamp(first[1], 0, 1);
  if (x >= last[0]) return clamp(last[1], 0, 1);
  const k = segmentOf(points, x);
  const a = points[k] as CurvePoint;
  const b = points[k + 1] as CurvePoint;
  const h = b[0] - a[0];
  const t = (x - a[0]) / h;
  const t2 = t * t;
  const t3 = t2 * t;
  const value =
    (2 * t3 - 3 * t2 + 1) * a[1] +
    (t3 - 2 * t2 + t) * h * (tangents[k] as number) +
    (-2 * t3 + 3 * t2) * b[1] +
    (t3 - t2) * h * (tangents[k + 1] as number);
  return clamp(value, 0, 1);
}

/** Samples the curve into a table of `size` entries, the shape of the core's 1D textures. */
export function sampleCurve(points: readonly CurvePoint[], size = CURVE_LUT_SIZE): Float32Array {
  const table = new Float32Array(size);
  const tangents = monotoneTangents(points);
  for (let i = 0; i < size; i++) table[i] = evaluateCurve(points, i / (size - 1), tangents);
  return table;
}

/** Code value (0–255) for a unit value, as the Curve tab's readout shows it. */
export const toCodeValue = (unit: number): number => Math.round(clamp(unit, 0, 1) * 255);

const fmt = (n: number): string => String(Math.round(n * 100) / 100);

/**
 * SVG path of the curve in a `width` × `height` box with the origin at the bottom left (y is
 * flipped). Each spline segment is written as one cubic Bézier, which is the same curve
 * exactly. The ends extend flat to the plot edges.
 */
export function curvePath(points: readonly CurvePoint[], width = 320, height = 220): string {
  const first = points[0];
  const last = points[points.length - 1];
  if (!first || !last) return "";
  const X = (x: number): number => x * width;
  const Y = (y: number): number => (1 - y) * height;
  const tangents = monotoneTangents(points);
  let d = `M${fmt(0)} ${fmt(Y(first[1]))}`;
  if (first[0] > 0) d += ` L${fmt(X(first[0]))} ${fmt(Y(first[1]))}`;
  for (let k = 0; k < points.length - 1; k++) {
    const a = points[k] as CurvePoint;
    const b = points[k + 1] as CurvePoint;
    const h = b[0] - a[0];
    const c1x = a[0] + h / 3;
    const c1y = a[1] + ((tangents[k] as number) * h) / 3;
    const c2x = b[0] - h / 3;
    const c2y = b[1] - ((tangents[k + 1] as number) * h) / 3;
    d += ` C${fmt(X(c1x))} ${fmt(Y(c1y))}, ${fmt(X(c2x))} ${fmt(Y(c2y))}, ${fmt(X(b[0]))} ${fmt(Y(b[1]))}`;
  }
  if (last[0] < 1) d += ` L${fmt(width)} ${fmt(Y(last[1]))}`;
  return d;
}

/**
 * Adds a point on the curve at `x`. Returns the new list and the index of the new point, or
 * null when the curve is full or `x` is closer than `MIN_POINT_GAP` to a neighbour.
 */
export function insertPoint(
  points: readonly CurvePoint[],
  x: number,
  y?: number,
): { points: CurvePoint[]; index: number } | null {
  if (points.length >= MAX_CURVE_POINTS) return null;
  const px = clamp(x, 0, 1);
  if (points.some((p) => Math.abs(p[0] - px) < MIN_POINT_GAP)) return null;
  const py = clamp(y ?? evaluateCurve(points, px), 0, 1);
  const next: CurvePoint[] = [...points.map((p): CurvePoint => [p[0], p[1]]), [px, py]];
  next.sort((a, b) => a[0] - b[0]);
  return { points: next, index: next.findIndex((p) => p[0] === px && p[1] === py) };
}

/**
 * Moves point `index` to `(x, y)`. The two end points keep their x (they only move up and
 * down); an inner point stays between its neighbours. Returns a new list.
 */
export function movePoint(points: readonly CurvePoint[], index: number, x: number, y: number): CurvePoint[] {
  const current = points[index];
  if (!current) return points.map((p): CurvePoint => [p[0], p[1]]);
  const isEnd = index === 0 || index === points.length - 1;
  const prev = points[index - 1];
  const next = points[index + 1];
  const lo = prev ? prev[0] + MIN_POINT_GAP : 0;
  const hi = next ? next[0] - MIN_POINT_GAP : 1;
  const nx = isEnd ? current[0] : clamp(x, lo, Math.max(lo, hi));
  return points.map((p, i): CurvePoint => (i === index ? [nx, clamp(y, 0, 1)] : [p[0], p[1]]));
}

/** Removes an inner point. The two ends stay: the curve never has fewer than two points. */
export function removePoint(points: readonly CurvePoint[], index: number): CurvePoint[] {
  const keep = points.length <= MIN_CURVE_POINTS || index <= 0 || index >= points.length - 1;
  return points.filter((_, i) => keep || i !== index).map((p): CurvePoint => [p[0], p[1]]);
}

/** Where a new point should go to open up the widest gap on the x axis (for the keyboard). */
export function widestGapX(points: readonly CurvePoint[]): number {
  let best = 0.5;
  let bestGap = -1;
  for (let k = 0; k < points.length - 1; k++) {
    const a = points[k] as CurvePoint;
    const b = points[k + 1] as CurvePoint;
    if (b[0] - a[0] > bestGap) {
      bestGap = b[0] - a[0];
      best = (a[0] + b[0]) / 2;
    }
  }
  return best;
}

/**
 * The inner point nearest the middle of the plot, or `null` when the curve has only its two ends. The Curve tab starts with it
 * selected (drawn filled) as the mockup does, so the Input and Output readout names a point of the curve instead of the
 * middle of the plot.
 */
export function centralInnerPoint(points: readonly CurvePoint[]): number | null {
  let best: number | null = null;
  let bestDistance = Infinity;
  for (let k = 1; k < points.length - 1; k++) {
    const distance = Math.abs((points[k] as CurvePoint)[0] - 0.5);
    if (distance < bestDistance) {
      best = k;
      bestDistance = distance;
    }
  }
  return best;
}
