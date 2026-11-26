/**
 * Hue curve maths for the Hue tab (`docs/34` §4.1, §9.4): the periodic monotone cubic spline of the core (`fritsch_carlson_tangents_periodic` in
 * `ym_color::curve`, the parametric look's `Spline::new_periodic`), and the scales of the two plots: how a hue and a value map onto the plot, the zoomed
 * default view that fits a fit's values and the full range, and the editing operations (add, move, remove) on a list of points `{ hue_deg, value }`.
 *
 * A curve has at most 16 points. An empty list is the identity (a turn of 0, a multiplier of 1); one point is a constant. The curve is periodic with period
 * 360°, C1 across 0° and 360°, and never overshoots the values given. The plot draws the spline itself; the core bakes the same spline into a table of 1024
 * samples, whose error against it is far below one code (`docs/34` §19).
 */
import type { HuePoint } from "../api/types";
import { clamp } from "./format";

export type { HuePoint } from "../api/types";

/** Most points a hue curve may have (`docs/34` §4.1). */
export const MAX_HUE_POINTS = 16;
/** The period of a hue curve, in degrees. */
export const HUE_PERIOD = 360;
/** The closest two points may sit on the hue axis, in degrees (the core refuses two at the same hue). */
export const MIN_HUE_GAP_DEG = 1;

/** The two hue curves of a Hue tab. */
export type HueCurveKind = "hue_vs_hue" | "hue_vs_sat";

/** What a curve is at the identity and the values it holds (`docs/34` §3.4). */
export const HUE_CURVES: Readonly<Record<HueCurveKind, { identity: number; min: number; max: number }>> = {
  hue_vs_hue: { identity: 0, min: -180, max: 180 },
  hue_vs_sat: { identity: 1, min: 0, max: 8 },
};

/** A hue reduced into [0, 360). */
export const wrapHue = (hue: number): number => ((hue % HUE_PERIOD) + HUE_PERIOD) % HUE_PERIOD || 0;

/** The points sorted by hue, each hue read modulo 360, as the core reads them. */
export function sortedPoints(points: readonly HuePoint[]): HuePoint[] {
  return points.map((p) => ({ hue_deg: wrapHue(p.hue_deg), value: p.value })).sort((a, b) => a.hue_deg - b.hue_deg);
}

/** Tangents of the periodic monotone cubic through the points (sorted, at least two): mean of the two secants, zeroed where they disagree in sign, then limited. */
export function periodicTangents(points: readonly HuePoint[]): number[] {
  const n = points.length;
  const x = points.map((p) => p.hue_deg);
  const y = points.map((p) => p.value);
  const width = (k: number): number => (k + 1 < n ? (x[k + 1] as number) - (x[k] as number) : (x[0] as number) + HUE_PERIOD - (x[n - 1] as number));
  const delta = Array.from({ length: n }, (_, k) => ((y[(k + 1) % n] as number) - (y[k] as number)) / width(k));
  const m = Array.from({ length: n }, (_, k) => {
    const before = delta[(k + n - 1) % n] as number;
    const after = delta[k] as number;
    return before * after > 0 ? 0.5 * (before + after) : 0;
  });
  for (let k = 0; k < n; k++) {
    const next = (k + 1) % n;
    const d = delta[k] as number;
    if (d === 0) {
      m[k] = 0;
      m[next] = 0;
      continue;
    }
    const alpha = (m[k] as number) / d;
    const beta = (m[next] as number) / d;
    const norm2 = alpha * alpha + beta * beta;
    if (norm2 > 9) {
      const tau = 3 / Math.sqrt(norm2);
      m[k] = tau * alpha * d;
      m[next] = tau * beta * d;
    }
  }
  return m;
}

/** The curve's value at a hue: the identity for no point, a constant for one, the periodic spline for more. `points` must be sorted ({@link sortedPoints}). */
export function evaluateHueCurve(points: readonly HuePoint[], hue: number, identity: number, tangents?: readonly number[]): number {
  const n = points.length;
  if (n === 0) return identity;
  const first = points[0] as HuePoint;
  if (n === 1) return first.value;
  const m = tangents ?? periodicTangents(points);
  let h = wrapHue(hue);
  // The segment that holds `h`: the last point at or before it; before the first point it is the closing segment, which starts at the last point.
  let k = n - 1;
  for (let i = 0; i < n; i++) {
    if ((points[i] as HuePoint).hue_deg <= h) k = i;
    else break;
  }
  if (h < first.hue_deg) h += HUE_PERIOD;
  const a = points[k] as HuePoint;
  const b = points[(k + 1) % n] as HuePoint;
  const x0 = a.hue_deg;
  const x1 = k + 1 < n ? b.hue_deg : b.hue_deg + HUE_PERIOD;
  const w = x1 - x0;
  const t = (h - x0) / w;
  const t2 = t * t;
  const t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * a.value + (t3 - 2 * t2 + t) * w * (m[k] as number) + (-2 * t3 + 3 * t2) * b.value + (t3 - t2) * w * (m[(k + 1) % n] as number);
}

/** How a hue and a value map onto a plot: x is the hue over 0° to 360°; y is linear (a turn in degrees) or logarithmic (a multiplier), over a view. */
export interface HueScale {
  kind: HueCurveKind;
  /** The value at the bottom and at the top edge of the plot. For the multiplier the bottom is where the axis starts (1/k), not 0. */
  low: number;
  high: number;
}

/** The fraction (0 at the bottom, 1 at the top) a value sits at on the plot's y axis. */
export function valueToUnit(scale: HueScale, value: number): number {
  if (scale.kind === "hue_vs_hue") return (value - scale.low) / (scale.high - scale.low);
  // A multiplier is on a log scale (its identity is 1, ×½ and ×2 are the same distance from it); 0 (grey) is below the axis, at its foot.
  const floor = Math.log2(scale.low);
  const top = Math.log2(scale.high);
  return value <= 0 ? 0 : (Math.log2(value) - floor) / (top - floor);
}

/** The value at a fraction of the plot's y axis. The foot of a multiplier plot that reaches 1/8 or lower is a multiplier of 0 ("down to grey"). */
export function unitToValue(scale: HueScale, unit: number): number {
  if (scale.kind === "hue_vs_hue") return scale.low + unit * (scale.high - scale.low);
  const floor = Math.log2(scale.low);
  const top = Math.log2(scale.high);
  if (unit <= 0.01 && scale.low <= FULL_SAT_LOW) return 0;
  return 2 ** (floor + unit * (top - floor));
}

/** The full range of each plot: ±180° and ⅛ to 8 (`docs/34` §9.4). */
export const FULL_SCALE: Readonly<Record<HueCurveKind, HueScale>> = {
  hue_vs_hue: { kind: "hue_vs_hue", low: -180, high: 180 },
  hue_vs_sat: { kind: "hue_vs_sat", low: 1 / 8, high: 8 },
};
const FULL_SAT_LOW = 1 / 8;

/** The half-ranges of Hue vs hue a zoomed view can have, in degrees. */
export const TURN_STEPS = [15, 30, 45, 90, 180] as const;
/** The factors of Hue vs saturation a zoomed view can have: the axis runs from 1/k to k. */
export const FACTOR_STEPS = [1.25, 1.5, 2, 3, 4, 8] as const;

/**
 * The zoomed default view of a curve (`docs/34` §9.4, the owner's answer at RG-H2): the smallest step of {@link TURN_STEPS} or {@link FACTOR_STEPS} that holds
 * every point with a quarter to spare, and never smaller than the first step. A fit's values are small (a few degrees, ×0.85 to ×1.15), so the view opens on them;
 * the full range is always reachable. A curve with no point opens at the smallest step.
 */
export function zoomedScale(kind: HueCurveKind, points: readonly HuePoint[]): HueScale {
  if (kind === "hue_vs_hue") {
    const widest = Math.max(0, ...points.map((p) => Math.abs(p.value)));
    const step = TURN_STEPS.find((s) => s >= widest * 1.25) ?? 180;
    return { kind, low: -step, high: step };
  }
  // A multiplier of 0 (grey) is the widest there is: the view then is the full one.
  const widest = Math.max(1, ...points.map((p) => (p.value <= 0 ? Infinity : Math.max(p.value, 1 / p.value))));
  const step = FACTOR_STEPS.find((s) => s >= widest * 1.1) ?? 8;
  return { kind, low: 1 / step, high: step };
}

/** Whether the scale is the full range. */
export const isFullScale = (scale: HueScale): boolean => scale.high === FULL_SCALE[scale.kind].high && scale.low === FULL_SCALE[scale.kind].low;

/** The next view that holds `value`, no smaller than `scale`: a point moved past the edge by a key widens the view, so it is never drawn off the plot. */
export function widenedScale(scale: HueScale, points: readonly HuePoint[]): HueScale {
  const wanted = zoomedScale(scale.kind, points);
  return wanted.high > scale.high ? wanted : scale;
}

/** The labels on the y axis, from the top to the bottom. */
export function axisTicks(scale: HueScale): Array<{ value: number; label: string }> {
  if (scale.kind === "hue_vs_hue") {
    const half = scale.high;
    const quarter = half / 2;
    const text = (v: number): string => (v === 0 ? "0" : `${v > 0 ? "+" : "−"}${Math.abs(v)}°`);
    return [half, quarter, 0, -quarter, -half].map((value) => ({ value, label: text(value) }));
  }
  const text = (v: number): string => (v >= 1 ? String(v) : v === 0.5 ? "½" : v === 0.25 ? "¼" : v === 0.125 ? "⅛" : v.toFixed(2).replace(/0+$/, ""));
  const ups = scale.high >= 8 ? [8, 4, 2, 1] : [scale.high, 1];
  const downs = scale.high >= 8 ? [0.5, 0.25] : [1 / scale.high];
  const full = scale.low <= FULL_SAT_LOW;
  return [...ups, ...downs].map((value) => ({ value, label: text(value) })).concat(full ? [{ value: 0, label: "0" }] : []);
}

/** Where on the plot's y axis (0 bottom, 1 top) a tick sits; a multiplier of 0 is the foot. */
export const tickUnit = (scale: HueScale, value: number): number => valueToUnit(scale, value);

/** The SVG path of the curve in a `width` × `height` box (y down), sampled every 2° round the circle; the identity is a straight line. */
export function hueCurvePath(points: readonly HuePoint[], scale: HueScale, width: number, height: number): string {
  const identity = HUE_CURVES[scale.kind].identity;
  const sorted = sortedPoints(points);
  const tangents = sorted.length >= 2 ? periodicTangents(sorted) : undefined;
  const parts: string[] = [];
  for (let hue = 0; hue <= HUE_PERIOD; hue += 2) {
    const value = evaluateHueCurve(sorted, hue, identity, tangents);
    const unit = clamp(valueToUnit(scale, value), -0.02, 1.02);
    parts.push(`${hue === 0 ? "M" : "L"}${Math.round((hue / HUE_PERIOD) * width * 100) / 100} ${Math.round((1 - unit) * height * 100) / 100}`);
  }
  return parts.join(" ");
}

/** Rounds a value as the plot's keys and drags do: a tenth of a degree for a turn, a hundredth for a multiplier. */
export function roundValue(kind: HueCurveKind, value: number): number {
  const step = kind === "hue_vs_hue" ? 0.1 : 0.01;
  const { min, max } = HUE_CURVES[kind];
  return Number(clamp(Math.round(value / step) * step, min, max).toFixed(2));
}

/** Adds a point at `hue` (rounded to a degree) with the curve's own value there. Null when the curve is full or `hue` is within a degree of a point. */
export function addHuePoint(points: readonly HuePoint[], kind: HueCurveKind, hue: number, value?: number): { points: HuePoint[]; index: number } | null {
  if (points.length >= MAX_HUE_POINTS) return null;
  const sorted = sortedPoints(points);
  const at = wrapHue(Math.round(hue));
  if (sorted.some((p) => circularGap(p.hue_deg, at) < MIN_HUE_GAP_DEG)) return null;
  const v = roundValue(kind, value ?? evaluateHueCurve(sorted, at, HUE_CURVES[kind].identity));
  const next = [...sorted, { hue_deg: at, value: v }].sort((a, b) => a.hue_deg - b.hue_deg);
  return { points: next, index: next.findIndex((p) => p.hue_deg === at) };
}

/** The distance between two hues round the circle, in degrees. */
export const circularGap = (a: number, b: number): number => {
  const d = Math.abs(wrapHue(a) - wrapHue(b));
  return Math.min(d, HUE_PERIOD - d);
};

/**
 * Moves point `index` to `(hue, value)`. The hue stays between its two neighbours (a point never passes another, and keeps a degree from each); with one or
 * two points it can go anywhere that keeps the gap. Returns a new list.
 */
export function moveHuePoint(points: readonly HuePoint[], kind: HueCurveKind, index: number, hue: number, value: number): HuePoint[] {
  const sorted = sortedPoints(points);
  const current = sorted[index];
  if (!current) return sorted;
  const n = sorted.length;
  let lo = 0;
  let hi = HUE_PERIOD - MIN_HUE_GAP_DEG;
  if (n > 1) {
    const before = sorted[index - 1];
    const after = sorted[index + 1];
    if (before) lo = before.hue_deg + MIN_HUE_GAP_DEG;
    else lo = Math.max(0, (sorted[n - 1] as HuePoint).hue_deg - HUE_PERIOD + MIN_HUE_GAP_DEG);
    if (after) hi = after.hue_deg - MIN_HUE_GAP_DEG;
    else hi = Math.min(HUE_PERIOD - MIN_HUE_GAP_DEG, (sorted[0] as HuePoint).hue_deg + HUE_PERIOD - MIN_HUE_GAP_DEG);
  }
  const h = clamp(Math.round(hue), lo, Math.max(lo, hi));
  return sorted.map((p, i) => (i === index ? { hue_deg: wrapHue(h), value: roundValue(kind, value) } : p));
}

/** Removes a point. The list may become empty, which is the identity. */
export function removeHuePoint(points: readonly HuePoint[], index: number): HuePoint[] {
  return sortedPoints(points).filter((_, i) => i !== index);
}

/** The hue where a new point would open the widest gap (for the keyboard). */
export function widestGapHue(points: readonly HuePoint[]): number {
  const sorted = sortedPoints(points);
  if (sorted.length === 0) return 0;
  let best = 0;
  let bestGap = -1;
  sorted.forEach((p, k) => {
    const next = sorted[(k + 1) % sorted.length] as HuePoint;
    const gap = k + 1 < sorted.length ? next.hue_deg - p.hue_deg : next.hue_deg + HUE_PERIOD - p.hue_deg;
    if (gap > bestGap) {
      bestGap = gap;
      best = wrapHue(p.hue_deg + gap / 2);
    }
  });
  return Math.round(best);
}

/** The words under a plot for a point: "Hue 120°, Turn +22°" and "Hue 120°, Saturation 80%" (the multiplier as a percentage). */
export function pointReadout(kind: HueCurveKind, p: HuePoint): { hue: string; label: string; value: string } {
  const hue = `${Math.round(p.hue_deg)}°`;
  if (kind === "hue_vs_hue") {
    const rounded = Math.round(p.value);
    return { hue, label: "Turn", value: rounded === 0 ? "0°" : `${rounded > 0 ? "+" : "−"}${Math.abs(rounded)}°` };
  }
  return { hue, label: "Saturation", value: `${Math.round(p.value * 100)}%` };
}
