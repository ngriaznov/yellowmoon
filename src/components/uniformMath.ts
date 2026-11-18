/**
 * The Colour Uniform node's maths in TypeScript (`docs/32` §3): a port of `crates/ym-color/src/uniform.rs` for the panel (the stored
 * units of a pick, the swatch's colour and name) and for the mock API, which draws the node on its flat-shape photos. The core renders
 * with its own CPU reference and GPU kernel. The constants mirror the Rust file's; `uniformMath.node.test.ts` reads it and fails when
 * one of them drifts. Nothing here touches the DOM, so Node runs it as it is.
 */
import type { ChromaPoint, TargetValue, UniformAmounts } from "../api/types";
import { chromaOf, E1, E2, hueAngle, hueName, LOGC4_STOP, renderP3, logc4Decode, mul3, AWG4_TO_P3, P3_TO_SRGB, srgbEncode, WARP_NEUTRAL_GUARD } from "./warpMath";
import type { HueName, Rgb } from "./warpMath";

/** Chroma below which a colour has no hue: the colour warp's guard (`UNIFORM_NEUTRAL_GUARD`). */
export const UNIFORM_NEUTRAL_GUARD = WARP_NEUTRAL_GUARD;
/** The hue pull is full up to this angle from the target's hue (`HUE_FULL`, 90°). */
export const HUE_FULL = Math.PI / 2;
/** The hue pull fades to nothing over this angle after `HUE_FULL` (`HUE_FADE`, 60°). */
export const HUE_FADE = Math.PI / 3;
/** Smallest target chroma (Saturation 2) and largest (Saturation 50). */
export const TARGET_CHROMA_MIN = UNIFORM_NEUTRAL_GUARD;
export const TARGET_CHROMA_MAX = 0.5;
/** Smallest and largest target tone (a LogC4 code). */
export const TARGET_TONE_MIN = 0;
export const TARGET_TONE_MAX = 1;
/** Code value of scene-linear 0.18 in LogC4 (`LOGC4_GRAY`): middle grey, Lightness 0. */
export const LOGC4_GRAY = 0.2783958365482653;

/** The panel's ranges (`docs/32` §2.1, §5.1). */
export const SATURATION_MIN = 2;
export const SATURATION_MAX = 50;
export const LIGHTNESS_MIN_EV = -4;
export const LIGHTNESS_MAX_EV = 8;

/** A pick reads the mean of a square of `2 · radius + 1` pixels, 7 × 7 (`docs/32` §3.11): the `radius_px` it asks `sample_chroma` for. */
export const UNIFORM_PICK_RADIUS_PX = 3;

/** What the first pick sets Compress to when all three amounts are 0 (`docs/32` §2.1). */
export const FIRST_PICK_AMOUNTS: Readonly<UniformAmounts> = { hue: 50, saturation: 50, lightness: 0 };

/** The target in the node's plane: chroma `[c1, c2]` and the tone `m`. */
export interface PlaneTarget {
  c: ChromaPoint;
  tone: number;
}

const fin = (x: number): number => (Number.isFinite(x) ? x : 0);
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

/** `Target::new`: chroma held between 2 and 50 (a chroma of 0 is put on `+e1`), tone between 0 and 1. */
export function planeTarget(c1: number, c2: number, tone: number): PlaneTarget {
  let a = fin(c1);
  let b = fin(c2);
  const r = Math.hypot(a, b);
  if (r < TARGET_CHROMA_MIN) {
    if (r === 0) [a, b] = [TARGET_CHROMA_MIN, 0];
    else [a, b] = [(a * TARGET_CHROMA_MIN) / r, (b * TARGET_CHROMA_MIN) / r];
  } else if (r > TARGET_CHROMA_MAX) {
    [a, b] = [(a * TARGET_CHROMA_MAX) / r, (b * TARGET_CHROMA_MAX) / r];
  }
  return { c: [a, b], tone: clamp(fin(tone), TARGET_TONE_MIN, TARGET_TONE_MAX) };
}

/** `Target::from_stored`: the stored units (hue read modulo 360, Saturation as chroma times 100, Lightness in stops from middle grey). */
export function fromStored(t: TargetValue): PlaneTarget {
  const r = t.saturation / 100;
  const h = (((t.hue_deg % 360) + 360) % 360) * (Math.PI / 180);
  return planeTarget(r * Math.cos(h), r * Math.sin(h), LOGC4_GRAY + t.lightness_ev * LOGC4_STOP);
}

/** `Target::to_stored`: hue in [0, 360), Saturation, Lightness in stops; held to the stored ranges of `docs/32` §5.1. */
export function toStored(t: PlaneTarget): TargetValue {
  const deg = (Math.atan2(t.c[1], t.c[0]) * 180) / Math.PI;
  return {
    hue_deg: ((deg % 360) + 360) % 360,
    saturation: clamp(100 * Math.hypot(t.c[0], t.c[1]), SATURATION_MIN, SATURATION_MAX),
    lightness_ev: clamp((t.tone - LOGC4_GRAY) / LOGC4_STOP, LIGHTNESS_MIN_EV, LIGHTNESS_MAX_EV),
  };
}

/** What a pick makes of a `sample_chroma` answer `{a, b, tone}`: the stored target, and whether it was too grey for a hue (held at Saturation 2). */
export function targetFromSample(sample: { a: number; b: number; tone: number }): { target: TargetValue; tooGrey: boolean } {
  const tooGrey = Math.hypot(sample.a, sample.b) < TARGET_CHROMA_MIN;
  return { target: toStored(planeTarget(sample.a, sample.b, sample.tone)), tooGrey };
}

/** The tone of a LogC4 triple: the mean of its three codes (`tone_of`). */
export const toneOf = (v: Readonly<Rgb>): number => (v[0] + v[1] + v[2]) / 3;

const smooth = (x: number): number => {
  const u = Number.isNaN(x) ? 0 : clamp(x, 0, 1);
  return u * u * (3 - 2 * u);
};

/** The neutral guard `g(r)` (`neutral_guard`). */
export const neutralGuard = (r: number): number => smooth(r / UNIFORM_NEUTRAL_GUARD);
/** The antipode fade `a(δ)` (`antipode_fade`). */
export const antipodeFade = (delta: number): number => 1 - smooth((Math.abs(delta) - HUE_FULL) / HUE_FADE);

/** `apply`: the node on one LogC4 / AWG4 triple with its mask at 1. No target or three zero amounts return the input. */
export function applyUniform(v: Readonly<Rgb>, target: PlaneTarget | null, amounts: UniformAmounts): Rgb {
  if (!target) return [v[0], v[1], v[2]];
  const kh = clamp(fin(amounts.hue) / 100, 0, 1);
  const ks = clamp(fin(amounts.saturation) / 100, 0, 1);
  const kl = clamp(fin(amounts.lightness) / 100, 0, 1);
  if (kh === 0 && ks === 0 && kl === 0) return [v[0], v[1], v[2]];
  const [t1, t2] = target.c;
  const rt = Math.hypot(t1, t2);
  const [c1, c2] = chromaOf(v);
  const m = toneOf(v);
  const r = Math.hypot(c1, c2);
  const g = neutralGuard(r);
  const r2 = ks !== 0 ? rt + (r - rt) * (1 - ks * g) : r;
  let n1 = 0;
  let n2 = 0;
  if (r > 0) {
    if (kh !== 0) {
      const delta = Math.atan2(t1 * c2 - t2 * c1, t1 * c1 + t2 * c2);
      const th = Math.atan2(t2, t1) + delta * (1 - kh * g * antipodeFade(delta));
      [n1, n2] = [r2 * Math.cos(th), r2 * Math.sin(th)];
    } else {
      [n1, n2] = [(c1 * r2) / r, (c2 * r2) / r];
    }
  }
  const d1 = n1 - c1;
  const d2 = n2 - c2;
  const dm = kl !== 0 ? target.tone + (m - target.tone) * (1 - kl) - m : 0;
  return [0, 1, 2].map((i) => (v[i] as number) + d1 * (E1[i] as number) + d2 * (E2[i] as number) + (kl !== 0 ? dm : 0)) as Rgb;
}

/** The LogC4 triple of a plane target: `m (1,1,1) + c1 e1 + c2 e2` (`to_rgb`). */
export const targetRgb = (t: PlaneTarget): Rgb => [0, 1, 2].map((i) => t.tone + t.c[0] * (E1[i] as number) + t.c[1] * (E2[i] as number)) as Rgb;

/** The swatch's colour as `#rrggbb`: the target's own colour, rendered as the warp's plane renders its colours. */
export function targetHex(t: TargetValue): string {
  const v = targetRgb(fromStored(t));
  const p3 = renderP3(mul3(AWG4_TO_P3, [logc4Decode(v[0]), logc4Decode(v[1]), logc4Decode(v[2])]));
  const srgb = mul3(P3_TO_SRGB, p3).map((x) => srgbEncode(clamp(x, 0, 1)));
  return `#${srgb.map((x) => Math.round(x * 255).toString(16).padStart(2, "0")).join("")}`;
}

/** The hue track's colour at `deg` degrees, at the warp plane's grey: the rainbow of the Hue slider. */
export function hueTrackHex(deg: number): string {
  return targetHex({ hue_deg: deg, saturation: 18, lightness_ev: 1.5 });
}

/** A CSS gradient of the hue ramp, 0 to 360 degrees, for the Hue slider's track. */
export function hueGradient(steps = 12): string {
  const stops = Array.from({ length: steps + 1 }, (_, i) => `${hueTrackHex((360 * i) / steps)} ${((100 * i) / steps).toFixed(1)}%`);
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}

const capital = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);

/**
 * The colour's name: the warp's eight hue names with a word for how pale and how light it is ("Orange", "Orange, a little pale",
 * "Dark red, strong"). Saturation under 4 is "pale", under 12 "a little pale", over 30 "strong"; Lightness under −1.5 stops is "dark",
 * over +1.5 "light". The same words name the swatch and its accessible name.
 */
export function colourName(t: TargetValue): string {
  const rad = (t.hue_deg * Math.PI) / 180;
  const base: HueName | "grey" = hueName([Math.cos(rad), Math.sin(rad)]);
  const tone = t.lightness_ev < -1.5 ? "dark " : t.lightness_ev > 1.5 ? "light " : "";
  const word = t.saturation < 4 ? ", pale" : t.saturation < 12 ? ", a little pale" : t.saturation > 30 ? ", strong" : "";
  return capital(`${tone}${base}`) + word;
}

const signed = (x: number): string => (x < 0 ? `minus ${Math.abs(x).toFixed(2)}` : x.toFixed(2));

/** The swatch's accessible name (`docs/32` §2.1): "Target colour: orange, hue 26 degrees, saturation 9.7, lightness minus 0.03 stops." */
export function swatchLabel(t: TargetValue): string {
  const sat = Number(t.saturation.toFixed(1));
  const ev = Number(t.lightness_ev.toFixed(2));
  return `Target colour: ${colourName(t).toLowerCase()}, hue ${Math.round(t.hue_deg) % 360} degrees, saturation ${sat}, lightness ${signed(ev)} stops.`;
}

/** The angle name the warp uses, re-exported so the panel needs one import. */
export { hueAngle };
