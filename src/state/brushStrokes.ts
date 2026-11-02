/**
 * What a Brush stroke is on the page (`docs/23` §7.6 *On the photo*, §3.7; `docs/algorithms/06` §5.5): the brush for the next strokes, a stroke
 * made from the points the pointer went through, and the rule of 100,000 points. Pure: the component (`BrushLayer.tsx`) collects the pointer's
 * places in CSS pixels, the core maps them to the photo frame in one call (`viewport_map_points`), and this turns the answer into the stroke
 * that `masks.edit` appends. Nothing here converts a place between the screen and the photo: that is the core's.
 */
import type { BrushStroke, MaskItem } from "../api/types";
import { toIso } from "../screens/nodes/maskGeometry";
import type { FrameUnits, Pt } from "../screens/nodes/maskGeometry";
import { BRUSH_SIZE_RANGE } from "../screens/tools/maskScales";

/** The brush the next strokes are drawn with. It belongs to the window: it is not saved, undone or copied, and a stroke that is drawn keeps what it was drawn with. */
export interface BrushSettings {
  /** The brush's diameter, 0.001 to 0.25 of the photo's long edge. */
  size: number;
  /** 0 (soft all the way in) to 1 (a hard edge). */
  hardness: number;
  /** The strokes take paint away instead of adding it. */
  erase: boolean;
}

/** A new Brush row's brush (`docs/23` §3.3): a size of 0.03, a hardness of 0.5, painting. */
export const DEFAULT_BRUSH: BrushSettings = { size: 0.03, hardness: 0.5, erase: false };

/** Most points a Brush mask may hold in all its strokes (`docs/23` §3.5), and the core's sentence when a stroke would make it hold more. */
export const MAX_BRUSH_POINTS = 100_000;
export const BRUSH_FULL = "A brush mask can have up to 100,000 points.";

/** A point is stored to four decimals of the photo frame: a fifth of a pixel of the 2048 px plane the brush is drawn on. */
const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;
const clamp01 = (n: number): number => Math.min(1, Math.max(0, n));

/** The points in all the strokes of a Brush row (0 for another kind). */
export function brushPointCount(row: MaskItem | undefined): number {
  return row?.kind === "brush" ? row.strokes.reduce((n, stroke) => n + stroke.points.length, 0) : 0;
}

/**
 * Where a stroke is simplified: the points nearer than 1/8 of the brush size to the line between their neighbours are dropped, so a document
 * stays small (`docs/23` §7.6). The distance is in the photo frame as a length (the long edge is 1, `frameUnits`).
 */
export const simplifyTolerance = (size: number): number => size / 8;

/** The distance from `p` to the segment `a`–`b` (to `a` when they are one point). */
function distanceToSegment(p: Pt, a: Pt, b: Pt): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.min(1, Math.max(0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

/**
 * The points that stay (Ramer–Douglas–Peucker): the first and the last always, and between two that stay, the point farthest from the line
 * between them, if it is `tolerance` or more away. Every point that goes is therefore nearer than `tolerance` to the line between the
 * points kept on either side of it. `points` are in the isotropic plane (lengths as fractions of the long edge); the answer is their indices.
 */
export function simplifiedIndices(points: readonly Pt[], tolerance: number): number[] {
  const n = points.length;
  if (n <= 2) return points.map((_, i) => i);
  const keep = new Uint8Array(n);
  keep[0] = 1;
  keep[n - 1] = 1;
  // Iterative, so that a stroke of ten thousand points cannot overflow the stack.
  const todo: Array<[number, number]> = [[0, n - 1]];
  for (let range = todo.pop(); range; range = todo.pop()) {
    const [first, last] = range;
    const a = points[first] as Pt;
    const b = points[last] as Pt;
    let far = -1;
    let farthest = tolerance;
    for (let i = first + 1; i < last; i++) {
      const d = distanceToSegment(points[i] as Pt, a, b);
      if (d >= farthest) {
        farthest = d;
        far = i;
      }
    }
    if (far < 0) continue;
    keep[far] = 1;
    todo.push([first, far], [far, last]);
  }
  const out: number[] = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(i);
  return out;
}

/**
 * The stroke to append from what the core made of the pointer's places. `mapped` are the answers of `viewport_map_points("cell")` for the
 * places the pointer went through, in order: a place on the paper (or in a straightening's empty corner) is `null` and is dropped; the others
 * are held to the photo frame, simplified (section above) and rounded. A stroke with no place on the photo is nothing: `null`. A single click
 * is a stroke of one point.
 */
export function buildStroke(mapped: ReadonlyArray<Pt | null>, brush: BrushSettings, units: FrameUnits): BrushStroke | null {
  const onPhoto: Pt[] = [];
  for (const place of mapped) {
    if (!place || !Number.isFinite(place[0]) || !Number.isFinite(place[1])) continue;
    onPhoto.push([clamp01(place[0]), clamp01(place[1])]);
  }
  if (onPhoto.length === 0) return null;
  const size = Math.min(BRUSH_SIZE_RANGE[1], Math.max(BRUSH_SIZE_RANGE[0], brush.size));
  const kept = simplifiedIndices(
    onPhoto.map((p) => toIso(p, units)),
    simplifyTolerance(size),
  );
  return {
    size,
    hardness: clamp01(brush.hardness),
    erase: brush.erase,
    points: kept.map((i): Pt => [round4((onPhoto[i] as Pt)[0]), round4((onPhoto[i] as Pt)[1])]),
  };
}
