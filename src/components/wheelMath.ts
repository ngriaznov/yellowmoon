/**
 * Maths for the colour balance wheel: pointer position to a point in the unit disc, and the
 * colour the puck shows. A point is `[a, b]`: `a` to the right, `b` upwards, within the unit
 * disc (`docs/01` §DevelopSettings `color_balance`).
 */
import type { ColorBalancePoint } from "../api/types";
import { clamp } from "./format";

export type { ColorBalancePoint } from "../api/types";

/** The colours around the wheel, clockwise from 3 o'clock (the mockup's conic gradient). */
export const WHEEL_STOPS = ["#FF5A1F", "#FFC20E", "#77E05A", "#0DB7E8", "#0B5CC2", "#C36BD9"] as const;
/** CSS `conic-gradient` of the wheel face. */
export const WHEEL_GRADIENT = `conic-gradient(from 90deg, ${[...WHEEL_STOPS, WHEEL_STOPS[0]].join(", ")})`;

/** Keeps a point inside the unit disc by pulling it back along its radius. */
export function clampToDisc(a: number, b: number): ColorBalancePoint {
  const radius = Math.hypot(a, b);
  if (radius <= 1) return [a, b];
  return [a / radius, b / radius];
}

/**
 * Point under the pointer. `rect` is the wheel's box, `travel` the radius in CSS pixels that
 * maps to 1 (the puck's centre can reach it).
 */
export function pointFromPointer(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  travel: number,
): ColorBalancePoint {
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const a = (clientX - cx) / travel;
  const b = (cy - clientY) / travel;
  return clampToDisc(a, b);
}

/** Strength of a point, 0 at the centre and 1 on the rim. */
export const strengthOf = ([a, b]: ColorBalancePoint): number => Math.min(1, Math.hypot(a, b));

/** Angle of a point in degrees clockwise from 12 o'clock, in [0, 360). Zero at the centre. */
export function angleOf([a, b]: ColorBalancePoint): number {
  if (a === 0 && b === 0) return 0;
  const deg = (Math.atan2(a, b) * 180) / Math.PI;
  return (deg + 360) % 360;
}

const hexToRgb = (hex: string): [number, number, number] => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const toHex = (c: readonly number[]): string =>
  "#" + c.map((x) => clamp(Math.round(x), 0, 255).toString(16).padStart(2, "0")).join("");

/** Linear mix of two `#RRGGBB` colours. */
export function mixHex(a: string, b: string, t: number): string {
  const A = hexToRgb(a);
  const B = hexToRgb(b);
  return toHex(A.map((x, i) => x + ((B[i] as number) - x) * t));
}

/** The wheel colour at an angle (degrees clockwise from 12 o'clock), as drawn by the gradient. */
export function wheelColorAt(angleDeg: number): string {
  // The gradient starts at 90deg (3 o'clock) with the first stop and turns clockwise.
  const fromStart = (((angleDeg - 90) % 360) + 360) % 360;
  const slot = fromStart / 60;
  const i = Math.floor(slot) % WHEEL_STOPS.length;
  const t = slot - Math.floor(slot);
  return mixHex(WHEEL_STOPS[i] as string, WHEEL_STOPS[(i + 1) % WHEEL_STOPS.length] as string, t);
}

/** Fill of the puck: paper at the centre, the hue of its direction as it moves out. */
export function puckColor(point: ColorBalancePoint, paper = "#FFFFFF"): string {
  const strength = strengthOf(point);
  if (strength === 0) return paper;
  return mixHex(paper, wheelColorAt(angleOf(point)), Math.min(1, strength * 1.15));
}

/** True when the point is at the centre (the default). */
export const isCentered = ([a, b]: ColorBalancePoint): boolean => a === 0 && b === 0;
