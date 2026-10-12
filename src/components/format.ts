/**
 * Number helpers shared by the controls: clamping, snapping to a step, and the value formats
 * the mockups use (a real minus sign, explicit plus, units).
 */

/** U+2212, the minus sign the mockups draw ("−18"). */
export const MINUS = "−";

/** Clamps `value` into [min, max]. */
export const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

/** Number of decimals in a step: 0.01 gives 2, 50 gives 0. */
export function stepDecimals(step: number): number {
  if (!Number.isFinite(step) || step <= 0) return 0;
  const text = String(step);
  if (text.includes("e-")) return Number(text.split("e-")[1]);
  const dot = text.indexOf(".");
  return dot < 0 ? 0 : text.length - dot - 1;
}

/** Rounds to the decimals of `step`, removing float noise (0.30000000000000004 gives 0.3). */
export function roundToStep(value: number, step: number): number {
  return Number(value.toFixed(stepDecimals(step)));
}

/** Clamps `value` into [min, max] and snaps it to the step grid that starts at `min`. */
export function clampSnap(value: number, min: number, max: number, step: number): number {
  const clamped = clamp(value, min, max);
  if (!(step > 0)) return clamped;
  const snapped = roundToStep(min + Math.round((clamped - min) / step) * step, step);
  // Snapping can step past `max` when the range is not a whole number of steps.
  return snapped > max ? roundToStep(min + Math.floor((max - min) / step) * step, step) : snapped;
}

/** True when two values are the same within a thousandth of `step`. */
export const sameValue = (a: number, b: number, step = 1): boolean => Math.abs(a - b) <= step / 1000;

/**
 * Parses what a person typed into a value box. Accepts a real minus sign, a leading plus, a
 * decimal comma and trailing units ("5450 K", "−18", "0,5"). Returns null for nothing usable.
 */
export function parseNumber(text: string): number | null {
  const cleaned = text.replace(/−|–/g, "-").replace(",", ".");
  const match = /[-+]?\s*(?:\d+\.?\d*|\.\d+)/.exec(cleaned);
  if (!match) return null;
  const value = Number(match[0].replace(/\s+/g, ""));
  return Number.isFinite(value) ? value : null;
}

/** Plain text a value box shows while it is being edited: "0.30", "-18", "5450". */
export function editText(value: number, step: number): string {
  return value.toFixed(stepDecimals(step));
}

/** Value formats for `Slider` and `ValueBox`, as in the mockups. */
export const formats = {
  /** 0 → "0", 18 → "+18", -18 → "−18". */
  signed: (value: number): string => (value > 0 ? "+" : value < 0 ? MINUS : "") + Math.abs(Math.round(value)),
  /** 0.3 → "+0.30", -1.25 → "−1.25", 0 → "0.00". Exposure in EV. */
  ev: (value: number): string => {
    const text = Math.abs(value).toFixed(2);
    return (text === "0.00" ? "" : value > 0 ? "+" : MINUS) + text;
  },
  /** 5450 → "5450 K". */
  kelvin: (value: number): string => `${Math.round(value)} K`,
  /** 100 → "100%". */
  percent: (value: number): string => `${Math.round(value)}%`,
  /** 0.8 → "0.8 px". */
  px1: (value: number): string => `${value.toFixed(1)} px`,
  /** 3000 → "3000 px". */
  px0: (value: number): string => `${Math.round(value)} px`,
  /** 0.5 → "0.50". */
  fixed2: (value: number): string => value.toFixed(2),
  /** 120.4 → "120". */
  integer: (value: number): string => String(Math.round(value)),
} as const;
