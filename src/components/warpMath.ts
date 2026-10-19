/**
 * The colour warp's maths in TypeScript (`docs/02` C9b): a port of `crates/ym-color/src/warp.rs` for the plane of Colour › Colour warp
 * and for the mock API, which shows the warp on its flat-shape photos. The core renders with its own CPU reference and GPU kernel; this
 * copy only draws the plane (where a pin's colour lands, the colour of a point) and fakes the picture in a browser build.
 *
 * The chroma of a LogC4 pixel `v` is `c = (e1·v, e2·v)` with `e1 = (1, −½, −½)/√1.5` (red against cyan) and `e2 = (0, 1, −1)/√2`
 * (green against blue); its tone is `L = encode(Y(decode(v)))`. The constants below mirror `warp.rs`: `warpMath.node.test.ts` reads
 * that file and fails when one of them drifts. Nothing here touches the DOM, so Node runs it as it is.
 */
import type { ChromaPoint, ColorWarpPin } from "../api/types";

export type { ChromaPoint, ColorWarpPin } from "../api/types";

/** A LogC4 or linear triple. */
export type Rgb = [number, number, number];

// ---------------------------------------------------------------------------
// The constants of warp.rs
// ---------------------------------------------------------------------------

/** At most this many pins; later ones are ignored (`WARP_MAX_PINS`). */
export const WARP_MAX_PINS = 16;
/** How far out the plane is drawn: `|c|` at the middle of each edge (`WARP_PLANE_EXTENT`). */
export const WARP_PLANE_EXTENT = 0.3;
/** Sources and targets are scaled back to `|c| ≤ WARP_COORD_LIMIT`. */
export const WARP_COORD_LIMIT = 0.5;
/** The smallest range of a pin. */
export const WARP_RANGE_MIN = 0.01;
/** The largest range of a pin. */
export const WARP_RANGE_MAX = 0.25;
/** The range of a new pin. */
export const WARP_RANGE_DEFAULT = 0.04;
/** The safety factor of the flow's steps: `N = ceil(WARP_STEP_SAFETY · Σ|δ| / (ρ√e))`, so no step can fold the plane for one pin. */
export const WARP_STEP_SAFETY = 1.25;
/** The most flow steps a frame takes (one pin never needs more than 61). */
export const WARP_MAX_STEPS = 64;
/** Every Gaussian's exponent is limited to this (`exp(−80)` rather than `exp(−∞)`). */
export const WARP_EXP_LIMIT = 80;
/** Chroma below which the warp fades out: `g = smoothstep(0, WARP_NEUTRAL_GUARD, |c|)`. */
export const WARP_NEUTRAL_GUARD = 0.02;
/** Half the width of a pin's tonal crossover, in stops (`WARP_TONE_HALF_WIDTH` is this many `LOGC4_STOP`). */
export const WARP_TONE_HALF_WIDTH_STOPS = 2;
/** A pin's pivot is limited to this many stops either side of middle grey. */
export const WARP_PIVOT_LIMIT_EV = 6;
/** A pin's exposure is limited to this many stops either way. */
export const WARP_EXPOSURE_LIMIT_EV = 2;

// ---------------------------------------------------------------------------
// LogC4 and the plane's axes
// ---------------------------------------------------------------------------

/** The LogC4 constants (`ym_color::consts::LOGC4`). */
export const LOGC4 = {
  a: 2231.8263090676883,
  b: 0.9071358748778103,
  c: 0.09286412512218964,
  s: 0.1135972086105891,
  t: -0.01805699611991131,
} as const;

/** One stop in LogC4 code values on the log segment, `b / 14 = 0.06480`. */
export const LOGC4_STOP = LOGC4.b / 14;
/** Half the width of a pin's tonal crossover in LogC4 code values. */
export const WARP_TONE_HALF_WIDTH = WARP_TONE_HALF_WIDTH_STOPS * LOGC4_STOP;
/** Scene-linear middle grey. */
export const MIDDLE_GRAY = 0.18;

/** LogC4 encode of a scene-linear value. */
export function logc4Encode(x: number): number {
  return x >= LOGC4.t ? ((Math.log2(LOGC4.a * x + 64) - 6) / 14) * LOGC4.b + LOGC4.c : (x - LOGC4.t) / LOGC4.s;
}

/** LogC4 decode of a code value to scene-linear. */
export function logc4Decode(e: number): number {
  return e >= 0 ? (2 ** ((14 * (e - LOGC4.c)) / LOGC4.b + 6) - 64) / LOGC4.a : e * LOGC4.s + LOGC4.t;
}

/** The LogC4 code of middle grey `stops` stops up or down (`grade::logc4_at_stops`). */
export const logc4AtStops = (stops: number): number => logc4Encode(MIDDLE_GRAY * 2 ** stops);

/** Red against cyan. */
export const E1: Readonly<Rgb> = [1 / Math.sqrt(1.5), -0.5 / Math.sqrt(1.5), -0.5 / Math.sqrt(1.5)];
/** Green against blue. */
export const E2: Readonly<Rgb> = [0, Math.SQRT1_2, -Math.SQRT1_2];

/** The `Y` row of AWG4 to XYZ (`gamut::awg4_luminance_row`, from the AWG4 primaries and D65). */
export const AWG4_LUMINANCE: Readonly<Rgb> = [0.2545241764040269, 0.7814777327120022, -0.03600190911602904];

const dot = (a: Readonly<Rgb>, b: Readonly<Rgb>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** The chroma of a LogC4 pixel: `(e1·v, e2·v)`. */
export const chromaOf = (v: Readonly<Rgb>): ChromaPoint => [dot(E1, v), dot(E2, v)];

/** The tone of a LogC4 pixel: the LogC4 code of its linear AWG4 luminance. */
export const toneOf = (v: Readonly<Rgb>): number => logc4Encode(dot(AWG4_LUMINANCE, [logc4Decode(v[0]), logc4Decode(v[1]), logc4Decode(v[2])]));

/** The LogC4 pixel of chroma `c` around the neutral `n`: `n + a·e1 + b·e2`. */
export const pixelAt = (n: number, c: ChromaPoint): Rgb => [n + c[0] * E1[0] + c[1] * E2[0], n + c[0] * E1[1] + c[1] * E2[1], n + c[0] * E1[2] + c[1] * E2[2]];

/** The Hermite step of WGSL's `smoothstep`: 0 below `e0`, 1 above `e1`. */
export function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

const finiteOr = (x: number, fallback: number): number => (Number.isFinite(x) ? x : fallback);
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

/** `p` scaled back onto the disc of radius `limit` when it lies outside; a coordinate that is not a finite number counts as 0. */
export function clampToDisc(p: ChromaPoint, limit: number): ChromaPoint {
  const q: ChromaPoint = [finiteOr(p[0], 0), finiteOr(p[1], 0)];
  const r = Math.hypot(q[0], q[1]);
  return r > limit ? [(q[0] * limit) / r, (q[1] * limit) / r] : q;
}

// ---------------------------------------------------------------------------
// Pins
// ---------------------------------------------------------------------------

/** The fields of a new pin besides its points (`WarpPin::default`). */
export const PIN_DEFAULTS = { range: WARP_RANGE_DEFAULT, shadows: 1, highlights: 1, pivot_ev: 0, exposure_ev: 0 } as const;

/** A moving pin from `source` to `target` (the source itself by default) with the default range and weights. */
export const movingPin = (source: ChromaPoint, target: ChromaPoint = source): ColorWarpPin => ({
  source: [source[0], source[1]],
  target: [target[0], target[1]],
  locked: false,
  ...PIN_DEFAULTS,
});

/** A locked pin at `at` with the default range. */
export const lockedPin = (at: ChromaPoint): ColorWarpPin => ({ source: [at[0], at[1]], target: [at[0], at[1]], locked: true, ...PIN_DEFAULTS });

/**
 * Every value in its range, as the core keeps it (`WarpPin::clamped`): points scaled back to `WARP_COORD_LIMIT`, the range, weights,
 * pivot and exposure clamped, a value that is not a finite number replaced by its default. A locked pin's target is its source and its
 * weights, pivot and exposure are the defaults.
 */
export function clampPin(pin: ColorWarpPin): ColorWarpPin {
  const source = clampToDisc(pin.source, WARP_COORD_LIMIT);
  const range = clamp(finiteOr(pin.range, WARP_RANGE_DEFAULT), WARP_RANGE_MIN, WARP_RANGE_MAX);
  if (pin.locked) return { ...lockedPin(source), range };
  return {
    source,
    target: clampToDisc(pin.target, WARP_COORD_LIMIT),
    locked: false,
    range,
    shadows: clamp(finiteOr(pin.shadows, 1), 0, 1),
    highlights: clamp(finiteOr(pin.highlights, 1), 0, 1),
    pivot_ev: clamp(finiteOr(pin.pivot_ev, 0), -WARP_PIVOT_LIMIT_EV, WARP_PIVOT_LIMIT_EV),
    exposure_ev: clamp(finiteOr(pin.exposure_ev, 0), -WARP_EXPOSURE_LIMIT_EV, WARP_EXPOSURE_LIMIT_EV),
  };
}

/** The pins as the core keeps them: the first sixteen, each clamped. */
export const clampPins = (pins: readonly ColorWarpPin[]): ColorWarpPin[] => pins.slice(0, WARP_MAX_PINS).map(clampPin);

/** Whether a pin changes anything on its own: it moves its colour or changes its exposure (a locked pin only holds colours). */
export const pinActs = (pin: ColorWarpPin): boolean =>
  !pin.locked && (pin.target[0] !== pin.source[0] || pin.target[1] !== pin.source[1] || pin.exposure_ev !== 0);

/** A moving pin, ready to evaluate. */
export interface PreparedMove {
  source: ChromaPoint;
  /** `target − source`. */
  delta: ChromaPoint;
  /** `1 / (2ρ²)` with the pin's own range. */
  invTwoSigmaSq: number;
  shadows: number;
  highlights: number;
  /** The pivot as a LogC4 code value. */
  pivotCode: number;
  exposureEv: number;
}

/** A locked pin, ready to evaluate. */
export interface PreparedLock {
  at: ChromaPoint;
  /** `1 / (2ρ²)`. */
  invTwoSigmaSq: number;
}

/** Stage C9b, prepared from the pins (`Warp`). */
export interface PreparedWarp {
  moves: PreparedMove[];
  /** Empty when no pin acts. */
  locks: PreparedLock[];
  /** The flow's steps, 1 to `WARP_MAX_STEPS` (0 when no pin acts). */
  steps: number;
}

/**
 * The number of flow steps for moves `delta` at ranges `range` (`flow_steps`): `clamp(ceil(1.25 · Σ|δ| / (ρ√e)), 1, 64)`. The Lipschitz
 * constant of one pin's move field is `|δ| / (ρ√e)`; a step of `1/N` of it is one to one while `N` is larger.
 */
export function flowSteps(moves: ReadonlyArray<{ delta: ChromaPoint; range: number }>): number {
  const sqrtE = Math.sqrt(Math.E);
  let lipschitz = 0;
  for (const { delta, range } of moves) lipschitz += Math.sqrt(delta[0] * delta[0] + delta[1] * delta[1]) / (range * sqrtE);
  const n = Math.ceil(WARP_STEP_SAFETY * lipschitz);
  return Number.isNaN(n) ? WARP_MAX_STEPS : clamp(n, 1, WARP_MAX_STEPS);
}

/**
 * Prepares the pins (`Warp::new`): the first sixteen, clamped. Pins that neither move nor change exposure are left out (they would only
 * dilute their neighbours), and so are the locks when no pin acts. The number of flow steps comes from the moving pins' moves and ranges.
 */
export function prepareWarp(pins: readonly ColorWarpPin[]): PreparedWarp {
  const moves: PreparedMove[] = [];
  const locks: PreparedLock[] = [];
  const reach: Array<{ delta: ChromaPoint; range: number }> = [];
  for (const pin of clampPins(pins)) {
    if (pin.locked) {
      locks.push({ at: pin.source, invTwoSigmaSq: 1 / (2 * pin.range * pin.range) });
    } else if (pinActs(pin)) {
      const delta: ChromaPoint = [pin.target[0] - pin.source[0], pin.target[1] - pin.source[1]];
      reach.push({ delta, range: pin.range });
      moves.push({
        source: pin.source,
        delta,
        invTwoSigmaSq: 1 / (2 * pin.range * pin.range),
        shadows: pin.shadows,
        highlights: pin.highlights,
        pivotCode: logc4AtStops(pin.pivot_ev),
        exposureEv: pin.exposure_ev,
      });
    }
  }
  if (moves.length === 0) return { moves, locks: [], steps: 0 };
  return { moves, locks, steps: flowSteps(reach) };
}

/** Whether the stage changes nothing (no pin acts). */
export const warpIsIdentity = (warp: PreparedWarp): boolean => warp.moves.length === 0;

/** What the warp does at a chroma `c` and a tone `l` (a LogC4 code value): the chroma move `d` and the exposure `e` in stops. */
export interface Displacement {
  d: ChromaPoint;
  e: number;
}

/** A Gaussian `exp(−d2 · invTwoSigmaSq)` with its exponent limited to `WARP_EXP_LIMIT`. */
const gauss = (d2: number, invTwoSigmaSq: number): number => Math.exp(-Math.min(d2 * invTwoSigmaSq, WARP_EXP_LIMIT));

/** A moving pin's tonal weight at the tone `l`: Shadows below the pivot, Highlights above it. */
const tonalWeight = (m: PreparedMove, l: number): number =>
  m.shadows + (m.highlights - m.shadows) * smoothstep(m.pivotCode - WARP_TONE_HALF_WIDTH, m.pivotCode + WARP_TONE_HALF_WIDTH, l);

/**
 * The displacement of the chroma `c` at the tone `l` (`Warp::displacement`). The pixel follows a flow of `warp.steps` steps while each
 * pin's centre slides from its source by the share of its move this pixel takes, `γ·w·δ` (γ: the neutral guard times the locks' hold at
 * `c`, w: the pin's tonal weight at `l`): `c += (γ/N) · Σ(w φ δ) / max(1, Σφ)`. The exposure is the one-step formula at `c`.
 */
export function warpDisplacement(warp: PreparedWarp, c: ChromaPoint, l: number): Displacement {
  const none: Displacement = { d: [0, 0], e: 0 };
  if (warp.moves.length === 0) return none;
  const guard = smoothstep(0, WARP_NEUTRAL_GUARD, Math.hypot(c[0], c[1]));
  if (guard === 0) return none;
  let hold = 1;
  for (const lock of warp.locks) {
    hold *= 1 - gauss((c[0] - lock.at[0]) ** 2 + (c[1] - lock.at[1]) ** 2, lock.invTwoSigmaSq);
  }
  if (hold === 0) return none;
  const gamma = guard * hold;
  const weights = warp.moves.map((m) => tonalWeight(m, l));
  const shares = weights.map((w) => gamma * w);
  let reach = 0;
  let exposure = 0;
  warp.moves.forEach((m, i) => {
    const phi = gauss((c[0] - m.source[0]) ** 2 + (c[1] - m.source[1]) ** 2, m.invTwoSigmaSq);
    reach += phi;
    exposure += (weights[i] as number) * phi * m.exposureEv;
  });
  const e = (gamma / Math.max(1, reach)) * exposure;
  const n = warp.steps;
  const h = gamma / n;
  let d0 = 0;
  let d1 = 0;
  for (let k = 0; k < n; k++) {
    const tau = k / n;
    const p0 = c[0] + d0;
    const p1 = c[1] + d1;
    let sum = 0;
    let v0 = 0;
    let v1 = 0;
    warp.moves.forEach((m, i) => {
      const slide = tau * (shares[i] as number);
      const phi = gauss((p0 - (m.source[0] + slide * m.delta[0])) ** 2 + (p1 - (m.source[1] + slide * m.delta[1])) ** 2, m.invTwoSigmaSq);
      sum += phi;
      const wphi = (weights[i] as number) * phi;
      v0 += wphi * m.delta[0];
      v1 += wphi * m.delta[1];
    });
    const step = h / Math.max(1, sum);
    d0 += step * v0;
    d1 += step * v1;
  }
  return { d: [d0, d1], e };
}

/** Applies the warp to one LogC4 pixel (`Warp::apply`). */
export function applyWarp(warp: PreparedWarp, v: Readonly<Rgb>): Rgb {
  if (warp.moves.length === 0) return [v[0], v[1], v[2]];
  const { d, e } = warpDisplacement(warp, chromaOf(v), toneOf(v));
  const lift = e * LOGC4_STOP;
  return [0, 1, 2].map((k) => (v[k] as number) + d[0] * (E1[k] as number) + d[1] * (E2[k] as number) + lift) as Rgb;
}

// ---------------------------------------------------------------------------
// Hue names (the pins' accessible values)
// ---------------------------------------------------------------------------

/** A colour's name, by its chroma angle. */
export type HueName = "red" | "orange" | "yellow" | "green" | "cyan" | "blue" | "purple" | "magenta";

/** The angle of a chroma, in degrees counter-clockwise from `+e1` (red), in [0, 360). */
export function hueAngle(c: ChromaPoint): number {
  const deg = (Math.atan2(c[1], c[0]) * 180) / Math.PI;
  return (deg + 360) % 360;
}

/** The mean of two angles in degrees, the short way round. */
const midway = (a: number, b: number): number => {
  const diff = ((b - a + 540) % 360) - 180;
  return (a + diff / 2 + 360) % 360;
};

/** Pure LogC4 colours around a mid grey: one or two channels up, the rest down. */
const PURE: ReadonlyArray<readonly [HueName, Rgb]> = [
  ["red", [0.5, 0.2, 0.2]],
  ["yellow", [0.5, 0.5, 0.2]],
  ["green", [0.2, 0.5, 0.2]],
  ["cyan", [0.2, 0.5, 0.5]],
  ["blue", [0.2, 0.2, 0.5]],
  ["magenta", [0.5, 0.2, 0.5]],
];

/**
 * The hue names and where they sit on this plane, measured: the angle of each pure LogC4 colour (red at 0°, yellow 60°, green 120°,
 * cyan 180°, blue 240°, magenta 300°), with orange halfway between red and yellow and purple halfway between blue and magenta.
 */
export const HUE_ANCHORS: ReadonlyArray<readonly [HueName, number]> = (() => {
  const at = new Map(PURE.map(([name, v]) => [name, hueAngle(chromaOf(v))]));
  const angle = (name: HueName): number => at.get(name) ?? 0;
  return [
    ["red", angle("red")],
    ["orange", midway(angle("red"), angle("yellow"))],
    ["yellow", angle("yellow")],
    ["green", angle("green")],
    ["cyan", angle("cyan")],
    ["blue", angle("blue")],
    ["purple", midway(angle("blue"), angle("magenta"))],
    ["magenta", angle("magenta")],
  ];
})();

/** The name of the hue a chroma points at: the nearest anchor; "grey" within the neutral guard, where the warp moves nothing. */
export function hueName(c: ChromaPoint): HueName | "grey" {
  if (Math.hypot(c[0], c[1]) < WARP_NEUTRAL_GUARD) return "grey";
  const angle = hueAngle(c);
  let best: HueName = "red";
  let distance = Infinity;
  for (const [name, at] of HUE_ANCHORS) {
    const d = Math.abs(((angle - at + 540) % 360) - 180);
    if (d < distance) {
      distance = d;
      best = name;
    }
  }
  return best;
}

const samePoint = (a: ChromaPoint, b: ChromaPoint): boolean => Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9;

/**
 * What a pin does, in words, for its `aria-valuetext`: "holds red" (a lock), "moves red toward orange", "moves red toward a stronger
 * red" or "a paler red" (a move within one hue), "red, not moved" (a pin whose target is still its source).
 */
export function pinValueText(pin: ColorWarpPin): string {
  const from = hueName(pin.source);
  if (pin.locked) return `holds ${from}`;
  if (samePoint(pin.source, pin.target)) return `${from}, not moved`;
  const to = hueName(pin.target);
  if (to !== from || from === "grey") return `moves ${from} toward ${to}`;
  const stronger = Math.hypot(pin.target[0], pin.target[1]) >= Math.hypot(pin.source[0], pin.source[1]);
  return `moves ${from} toward a ${stronger ? "stronger" : "paler"} ${to}`;
}

// ---------------------------------------------------------------------------
// The plane's geometry and colours
// ---------------------------------------------------------------------------

/** A point of a square plane `size` pixels wide, centred on neutral, `|c| = extent` at the middle of each edge; `+b` is up. */
export function planeToPx(c: ChromaPoint, size: number, extent = WARP_PLANE_EXTENT): { x: number; y: number } {
  const scale = size / 2 / extent;
  return { x: size / 2 + c[0] * scale, y: size / 2 - c[1] * scale };
}

/** The chroma under a pixel of the plane (the inverse of `planeToPx`). */
export function pxToPlane(x: number, y: number, size: number, extent = WARP_PLANE_EXTENT): ChromaPoint {
  const scale = size / 2 / extent;
  return [(x - size / 2) / scale, (size / 2 - y) / scale];
}

/** Keeps a point on the drawn square, `|a|, |b| ≤ extent`, so a pin dragged past the edge stays in sight. */
export const clampToPlane = (c: ChromaPoint, extent = WARP_PLANE_EXTENT): ChromaPoint => [clamp(c[0], -extent, extent), clamp(c[1], -extent, extent)];

/** The bright grey the plane's colours are shown at, as a LogC4 code value (about 2.8 stops over middle grey). */
export const PLANE_GREY = 0.45;

/** Linear AWG4 to linear Display P3 (D65). */
export const AWG4_TO_P3: ReadonlyArray<Readonly<Rgb>> = [
  [1.520502397, -0.4042986627, -0.1162037343],
  [-0.1360315837, 1.2698479799, -0.1338163962],
  [0.0058785171, -0.0548756605, 1.0489971435],
];
/** Linear Display P3 to linear sRGB (Rec.709 primaries, D65). */
export const P3_TO_SRGB: ReadonlyArray<Readonly<Rgb>> = [
  [1.2249401763, -0.2249401763, 0],
  [-0.0420569547, 1.0420569547, 0],
  [-0.0196375546, -0.0786360456, 1.0982736002],
];

/** A 3 × 3 matrix times a column. */
export const mul3 = (m: ReadonlyArray<Readonly<Rgb>>, v: Readonly<Rgb>): Rgb => [0, 1, 2].map((r) => dot(m[r] as Rgb, v)) as Rgb;

/** The inverse of a 3 × 3 matrix. */
export function invert3(m: ReadonlyArray<Readonly<Rgb>>): Rgb[] {
  const [a, b, c] = m[0] as Rgb;
  const [d, e, f] = m[1] as Rgb;
  const [g, h, i] = m[2] as Rgb;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

/** The sRGB transfer curve (also Display P3's). */
export const srgbEncode = (x: number): number => (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055);
/** Its inverse. */
export const srgbDecode = (x: number): number => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4);

/** The `Y` row of linear Display P3. */
const P3_LUMINANCE: Readonly<Rgb> = [0.2289745641, 0.6917385218, 0.0792869141];

/**
 * A gentle display rendering for the plane, which keeps each point's hue: a colour outside Display P3 (a channel below 0) is pulled
 * toward its own grey (its luminance, at least 8 % of its largest channel) until it fits, then the three channels are scaled together by `m / (0.6 + m)` of the largest, so the bright grey
 * reads light (0.68) and saturated colours roll off instead of clipping into flat bands.
 */
function renderP3(rgb: Readonly<Rgb>): Rgb {
  // AWG4's blue has a negative luminance, so the deepest blues of the plane have none: they are pulled toward a dim grey instead.
  const y = Math.max(dot(P3_LUMINANCE, rgb), 0.08 * Math.max(rgb[0], rgb[1], rgb[2]), 0);
  const low = Math.min(rgb[0], rgb[1], rgb[2]);
  const fit = low < 0 ? (y > 0 ? y / (y - low) : 0) : 1;
  const inGamut = rgb.map((x) => y + (x - y) * fit) as Rgb;
  const top = Math.max(inGamut[0], inGamut[1], inGamut[2]);
  if (!(top > 0)) return [0, 0, 0];
  const scale = 1 / (0.6 + top);
  return inGamut.map((x) => Math.max(0, x * scale)) as Rgb;
}

/** Where a plane colour is shown: a Display P3 canvas, or an sRGB one where the browser has no wide-gamut canvas. */
export type PlaneSpace = "display-p3" | "srgb";

/**
 * The colour of a point of the plane, as encoded values 0–1 in `space`: the LogC4 pixel `PLANE_GREY + a·e1 + b·e2`, decoded, taken to
 * Display P3, rendered gently (`renderP3`), taken on to sRGB for an sRGB canvas (clamped there), and encoded with the sRGB curve.
 */
export function planeColor(c: ChromaPoint, space: PlaneSpace = "display-p3"): Rgb {
  const v = pixelAt(PLANE_GREY, c);
  let rgb = renderP3(mul3(AWG4_TO_P3, [logc4Decode(v[0]), logc4Decode(v[1]), logc4Decode(v[2])]));
  if (space === "srgb") rgb = mul3(P3_TO_SRGB, rgb);
  return rgb.map((x) => srgbEncode(clamp(x, 0, 1))) as Rgb;
}

/** A plane colour as an sRGB `#rrggbb` (a pin's dot). */
export function planeHex(c: ChromaPoint): string {
  return `#${planeColor(c, "srgb")
    .map((x) => Math.round(clamp(x, 0, 1) * 255).toString(16).padStart(2, "0"))
    .join("")}`;
}

/** How much of a plane colour shows at a radius: none at neutral, rising toward the edge, fading again past the extent. */
export function planeAlpha(r: number, extent = WARP_PLANE_EXTENT): number {
  return 0.92 * smoothstep(0, 0.75 * extent, r) * (1 - 0.8 * smoothstep(extent, 1.4 * extent, r));
}

/**
 * The plane's hue field as RGBA bytes, `resolution` pixels square (row by row, from the top): each pixel the colour of its point
 * (`planeColor`) at the opacity of `planeAlpha`, over a dark ground that shows through. It is decorative and costs a few milliseconds,
 * so `WarpPlane` makes it once per size and keeps it.
 */
export function hueFieldPixels(resolution: number, space: PlaneSpace, extent = WARP_PLANE_EXTENT): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(resolution * resolution * 4);
  for (let j = 0; j < resolution; j++) {
    for (let i = 0; i < resolution; i++) {
      const c = pxToPlane(i + 0.5, j + 0.5, resolution, extent);
      const [r, g, b] = planeColor(c, space);
      const k = (j * resolution + i) * 4;
      out[k] = r * 255;
      out[k + 1] = g * 255;
      out[k + 2] = b * 255;
      out[k + 3] = planeAlpha(Math.hypot(c[0], c[1]), extent) * 255;
    }
  }
  return out;
}
