/**
 * A deterministic histogram for a flat-shape scene, for the mock API. The scenes have four
 * flat colours, so the honest histogram would be four spikes. This one puts a soft bump at
 * each colour, weighted by the share of the frame it covers, so the plot looks like a photo's.
 */
import type { HistogramData } from "../api/types";
import type { Scene } from "./scenes";

/** Bins in the mock's histogram. */
export const HISTOGRAM_BINS = 64;

/** Share of the frame each shape covers (the sun is small). */
const AREA = { sky: 0.5, far: 0.2, near: 0.26, sun: 0.04 } as const;

const channel = (hex: string, index: 0 | 1 | 2): number => Number.parseInt(hex.slice(1 + index * 2, 3 + index * 2), 16) / 255;

const bump = (x: number, centre: number, width: number): number => Math.exp(-((x - centre) ** 2) / (2 * width * width));

/**
 * The histogram of a scene. `shift` moves every bump along the tonal scale (positive is
 * brighter), so exposure changes visibly move the plot. The tallest bin of all channels is 1.
 */
export function sceneHistogram(scene: Scene, shift = 0, bins = HISTOGRAM_BINS): HistogramData {
  const make = (index: 0 | 1 | 2): number[] =>
    Array.from({ length: bins }, (_, i) => {
      const x = i / (bins - 1);
      let total = 0.02;
      for (const key of ["sky", "far", "near", "sun"] as const) {
        total += AREA[key] * bump(x, Math.min(1, Math.max(0, channel(scene[key], index) + shift)), 0.07);
      }
      return total;
    });
  const red = make(0);
  const green = make(1);
  const blue = make(2);
  const peak = Math.max(...red, ...green, ...blue) || 1;
  const norm = (values: number[]): number[] => values.map((v) => Math.round((v / peak) * 1000) / 1000);
  return { red: norm(red), green: norm(green), blue: norm(blue) };
}
