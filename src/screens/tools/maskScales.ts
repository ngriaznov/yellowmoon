/**
 * What each slider of a mask's card shows and the field it writes (`docs/algorithms/06` §4.5, `docs/23` §7.6): the one place for the
 * scales, so a card, a row's subtitle and a test all say the same. Pure.
 *
 * - Feather, Softness, Low and High: 0 to 100 are 0 to 1 (Low and High are LogC4 code values times 100).
 * - Opacity: 0 to 100 % are 0 to 1.
 * - A Colour range's Width: `0.005 + 0.295 · (x / 100)²` (the slider is fine where the width is small, and reaches the plane's 0.30).
 * - A Radial's Width and Height: the oval's full width and height as a share of the photo's long edge, 1 to 200 (the radii are half of
 *   that, 0.002 to 1 of the long edge, and the slider's end is the field's); the spec gives no rule for them, this is the plain one.
 * - A Linear or Radial's Angle: degrees, −180 to 180, as stored.
 * - Refine edges (the raster masks): 0 to 100 % are 0 to 1. Grow: −100 to 100 are −1 to 1.
 * - A Brush's Size: `0.001 + 0.249 · (x / 100)²` of the long edge (the slider is fine where the brush is small, and reaches the stored 0.25).
 */

/** Rounds to a whole number of the slider (a slider shows whole numbers; the field keeps what was typed to a step of 1 on the slider). */
const whole = (n: number): number => Math.round(n);

const clamp = (n: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, n));

/** A 0 to 1 field (Feather, Softness, Low, High) as its slider shows it, 0 to 100. */
export const unitSlider = (field: number): number => whole(clamp(field, 0, 1) * 100);
/** The field a 0 to 100 slider of a 0 to 1 field writes. */
export const unitField = (slider: number): number => clamp(slider, 0, 100) / 100;

/** Opacity as its slider shows it, 0 to 100 %. */
export const opacitySlider = unitSlider;
/** The opacity a slider writes, 0 to 1. */
export const opacityField = unitField;

/** The smallest and the largest width of a Colour range, and the plane's reach (`docs/23` §3.3). */
export const COLOR_WIDTH_RANGE: readonly [number, number] = [0.005, 0.3];

/** A Colour range's width as its slider shows it: the inverse of `0.005 + 0.295 · (x / 100)²`. */
export const widthSlider = (width: number): number => whole(100 * Math.sqrt(Math.max(0, (clamp(width, COLOR_WIDTH_RANGE[0], COLOR_WIDTH_RANGE[1]) - 0.005) / 0.295)));
/** The width a slider of 0 to 100 writes: `0.005 + 0.295 · (x / 100)²`, to a millionth (no float noise in a document). */
export const widthField = (slider: number): number => Math.round((0.005 + 0.295 * (clamp(slider, 0, 100) / 100) ** 2) * 1e6) / 1e6;

/** Refine edges as its slider shows it, 0 to 100 %. */
export const refineSlider = unitSlider;
/** The refine edges a slider writes, 0 to 1. */
export const refineField = unitField;

/** Grow as its slider shows it: whole numbers from −100 to 100 for −1 to 1. */
export const growSlider = (field: number): number => whole(clamp(field, -1, 1) * 100);
/** The grow a slider of −100 to 100 writes, −1 to 1. */
export const growField = (slider: number): number => clamp(slider, -100, 100) / 100;

/** The smallest and the largest size of a brush stroke, fractions of the photo's long edge (`docs/23` §3.3). */
export const BRUSH_SIZE_RANGE: readonly [number, number] = [0.001, 0.25];

/** A brush's Size as its slider shows it: the inverse of `0.001 + 0.249 · (x / 100)²`, 0 to 100. */
export const sizeSlider = (size: number): number => whole(100 * Math.sqrt(Math.max(0, (clamp(size, BRUSH_SIZE_RANGE[0], BRUSH_SIZE_RANGE[1]) - 0.001) / 0.249)));
/** The size a slider of 0 to 100 writes: `0.001 + 0.249 · (x / 100)²`, to a millionth (no float noise in a document). */
export const sizeField = (slider: number): number => Math.round((0.001 + 0.249 * (clamp(slider, 0, 100) / 100) ** 2) * 1e6) / 1e6;

/** The radii's range, fractions of the long edge (`docs/23` §3.3). */
export const RADII_RANGE: readonly [number, number] = [0.002, 1];
/** A Radial's width or height as its slider shows it: the radius doubled, as a percent of the long edge, 1 to 200. */
export const radiusSlider = (radius: number): number => clamp(whole(clamp(radius, RADII_RANGE[0], RADII_RANGE[1]) * 200), 1, 200);
/** The radius a slider writes: half the width or height, held to the field's range. */
export const radiusField = (slider: number): number => clamp(slider / 200, RADII_RANGE[0], RADII_RANGE[1]);

/** The angle the field holds, as the slider shows it: whole degrees, −180 to 180. */
export const angleSlider = (degrees: number): number => clamp(whole(degrees), -180, 180);

/** An angle held to −180…180 (a turn past the end comes round the other side). */
export function wrapDegrees(degrees: number): number {
  const d = ((((degrees + 180) % 360) + 360) % 360) - 180;
  // −180 and 180 are the same line; keep the end the field has.
  return d === -180 && degrees > 0 ? 180 : d;
}
