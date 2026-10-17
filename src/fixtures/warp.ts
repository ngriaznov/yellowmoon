/**
 * The colour warp on the mock's flat-shape photos. The mock has no pipeline: its photos are four flat sRGB colours each
 * (`scenes.ts`). To show what Colour › Colour warp does in a browser build, a colour is taken to an approximate LogC4 pixel (sRGB
 * decoded, to Display P3, to linear AWG4, read as scene-linear, encoded, and its chroma multiplied by `CHROMA_GAIN`), moved by the
 * TypeScript port of the warp (`components/warpMath.ts`), and brought back the same way. The plane's scatter and Pick colour read the
 * same approximate chroma, so a pin made from a click lands on the colour it was made from.
 *
 * Nothing here is colour science: it exists so that a pin visibly moves a colour in screenshots and end-to-end tests.
 */
import type { ChromaPoint, ColorWarpPin } from "../api/types";
import {
  applyWarp,
  AWG4_TO_P3,
  chromaOf,
  invert3,
  logc4Decode,
  logc4Encode,
  mul3,
  P3_TO_SRGB,
  prepareWarp,
  srgbDecode,
  srgbEncode,
  warpIsIdentity,
} from "../components/warpMath";
import type { Rgb } from "../components/warpMath";
import { layerAt } from "./scenes";
import type { Scene } from "./scenes";

const SRGB_TO_P3 = invert3(P3_TO_SRGB);
const P3_TO_AWG4 = invert3(AWG4_TO_P3);
const KEYS = ["sky", "far", "near", "sun"] as const;

/**
 * How much stronger a flat colour's chroma is at the warp's input than on screen. The mockup's colours are muted, as a picture is after
 * a look and the display rendering; read as they are, they would sit within 0.06 of neutral, a dot on the plane. Three times puts the
 * blue sky of DSC04204 at |c| = 0.11 and an orange sky at 0.19, where a camera's colours sit (`docs/02` C9b: a red of linear
 * (0.18, 0.009, 0.009) reads 0.13).
 */
export const CHROMA_GAIN = 3;

/** Scales the chroma of a LogC4 pixel about its channel mean, which keeps its brightness along the neutral axis. */
const scaleChroma = (v: Readonly<Rgb>, gain: number): Rgb => {
  const mean = (v[0] + v[1] + v[2]) / 3;
  return [mean + gain * (v[0] - mean), mean + gain * (v[1] - mean), mean + gain * (v[2] - mean)];
};

const channels = (hex: string): Rgb => {
  const n = Number.parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};

/** An sRGB `#RRGGBB` as an approximate LogC4 pixel: its display-linear values read as scene-linear AWG4, its chroma × `CHROMA_GAIN`. */
export function hexToLogC4(hex: string): Rgb {
  const linear = channels(hex).map(srgbDecode) as Rgb;
  return scaleChroma(mul3(P3_TO_AWG4, mul3(SRGB_TO_P3, linear)).map(logc4Encode) as Rgb, CHROMA_GAIN);
}

/** The inverse of `hexToLogC4`, clamped to the sRGB cube: an uppercase `#RRGGBB`. */
export function logC4ToHex(v: Readonly<Rgb>): string {
  const u = scaleChroma(v, 1 / CHROMA_GAIN);
  const linear = mul3(P3_TO_SRGB, mul3(AWG4_TO_P3, [logc4Decode(u[0]), logc4Decode(u[1]), logc4Decode(u[2])]));
  return (
    "#" +
    linear
      .map((x) => Math.round(srgbEncode(Math.min(1, Math.max(0, x))) * 255).toString(16).padStart(2, "0"))
      .join("")
      .toUpperCase()
  );
}

/** Where an sRGB colour sits on the warp's plane. */
export const hexChroma = (hex: string): ChromaPoint => chromaOf(hexToLogC4(hex));

/** The scene with its four colours through the warp. No pin that acts leaves it as it is, to the byte. */
export function warpScene(scene: Scene, pins: readonly ColorWarpPin[]): Scene {
  const warp = prepareWarp(pins);
  if (warpIsIdentity(warp)) return scene;
  const out: Scene = { ...scene };
  for (const key of KEYS) out[key] = logC4ToHex(applyWarp(warp, hexToLogC4(scene[key])));
  return out;
}

/** A small seeded generator (mulberry32), so the scatter is the same on every run. */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** How far the dots of one flat colour spread, in chroma units: the texture a real photo would have. */
const SPREAD = 0.012;

/**
 * The photo's colours as the core's `chroma_scatter` gives them: a grid of `columns × rows` points over the frame (1536 by default,
 * under the 2048 of the contract), each the chroma of the shape under it plus a little seeded spread, rounded to 1e-4 as the core's
 * are. Flat `[a0, b0, a1, b1, …]`.
 */
export function sceneScatter(scene: Scene, seed: number, columns = 48, rows = 32): number[] {
  const random = seeded(seed);
  const gauss = (): number => {
    const u = Math.max(1e-12, random());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
  };
  const at = Object.fromEntries(KEYS.map((key) => [key, hexChroma(scene[key])])) as Record<(typeof KEYS)[number], ChromaPoint>;
  const points: number[] = [];
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < columns; i++) {
      const [a, b] = at[layerAt(scene, (i + 0.5) / columns, (j + 0.5) / rows)];
      points.push(Math.round((a + SPREAD * gauss()) * 1e4) / 1e4, Math.round((b + SPREAD * gauss()) * 1e4) / 1e4);
    }
  }
  return points;
}
