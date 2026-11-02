/**
 * The handles of a Linear and a Radial mask on the photo (`docs/23` §7.6 *On the photo*): where they are, what a drag of each one writes.
 * Pure: the component (`MaskHandles.tsx`) asks the core to map the points (`viewport_map_points`: the straightening, the turn, the crop,
 * the lens's bend and the view's zoom and pan are the core's, the page never converts) and draws what comes back.
 *
 * Everything is in the **photo frame** (`docs/23` §3.6): points are fractions `[x, y]` of the photo as the camera oriented it, and sizes are
 * fractions of its long edge. A circle on the photo is a circle in "iso" coordinates, the frame's fractions scaled so that the long
 * edge is 1 (`frameUnits`): the maths of `docs/algorithms/06` §4.1 and §4.2 (the angle is clockwise from the horizontal, y down).
 *
 * A drag is **absolute**: the handle goes where the pointer is (a few pixels' jump to the pointer's place when it was grabbed off its
 * centre), so a drag needs only where the pointer is now, in the photo frame, and the row as it was.
 */
import type { MaskItem, MaskPatch } from "../../api/types";
import { RADII_RANGE, wrapDegrees } from "../tools/maskScales";

/** A point: photo-frame fractions, or CSS pixels from the cell's corner. */
export type Pt = [x: number, y: number];

/** The shapes that have handles. */
export type ShapeRow = Extract<MaskItem, { kind: "linear" | "radial" }>;

/** Whether a row has handles on the photo: a Linear or a Radial that is on. */
export const hasHandles = (row: MaskItem | undefined): row is ShapeRow => row !== undefined && row.enabled && (row.kind === "linear" || row.kind === "radial");

/** The photo frame's sides with the long edge as 1: what a fraction is multiplied by to be a length (`docs/23` §3.6). */
export interface FrameUnits {
  w: number;
  h: number;
}

/** The frame's units from its aspect (width ÷ height of the oriented photo). A shape on a square photo and a wide one is the same shape. */
export function frameUnits(aspect: number): FrameUnits {
  const a = Number.isFinite(aspect) && aspect > 0 ? aspect : 1.5;
  return a >= 1 ? { w: 1, h: 1 / a } : { w: a, h: 1 };
}

/** A point of the frame as a point of the isotropic plane. */
export const toIso = ([x, y]: Pt, u: FrameUnits): Pt => [x * u.w, y * u.h];
/** The inverse of {@link toIso}. */
export const fromIso = ([x, y]: Pt, u: FrameUnits): Pt => [x / u.w, y / u.h];

const rad = (degrees: number): number => (degrees * Math.PI) / 180;
const deg = (radians: number): number => (radians * 180) / Math.PI;
const clamp = (n: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, n));
/** Four decimals: a handle's number is not a measurement, and a row patch is not noise. */
const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

/** The unit vectors of a shape's axes in the isotropic plane: along the line (Linear) or the first radius (Radial), and across it. */
export function axes(angleDeg: number): { t: Pt; n: Pt } {
  const a = rad(angleDeg);
  return { t: [Math.cos(a), Math.sin(a)], n: [-Math.sin(a), Math.cos(a)] };
}

const add = (a: Pt, b: Pt, k = 1): Pt => [a[0] + b[0] * k, a[1] + b[1] * k];
const sub = (a: Pt, b: Pt): Pt => [a[0] - b[0], a[1] - b[1]];
const dot = (a: Pt, b: Pt): number => a[0] * b[0] + a[1] * b[1];

/** The handles' names: what a drag of each one changes. */
export type HandleId = "move" | "line" | "feather-start" | "feather-end" | "rotate" | "radius-x" | "radius-y";

/** What the page draws for a shape: lines of CSS points and the handles. */
export interface HandleDrawing {
  lines: Array<{ id: "centre" | "start" | "end" | "outline" | "inner"; points: Pt[]; dashed: boolean }>;
  handles: Array<{ id: HandleId; label: string; at: Pt }>;
}

/** Which photo-frame points to map, and how to read the answers. */
export interface HandlePlan {
  /** The photo-frame points to hand to `viewport_map_points("photo")`, in order. */
  points: Pt[];
  /** The drawing from the answers (the same order; `null` where the core has no place for a point). */
  draw: (mapped: ReadonlyArray<Pt | null>) => HandleDrawing;
}

/** Samples along a Linear's lines and round a Radial's outline: enough for the lens's bend to show as a curve. */
const LINE_SAMPLES = 16;
const RING_SAMPLES = 48;

/** The part `[u0, u1]` of the line `p + t u` (iso) inside the frame `[0, w] × [0, h]`, or null when it misses the frame. */
export function clipLine(p: Pt, t: Pt, u: FrameUnits): [number, number] | null {
  let lo = -Infinity;
  let hi = Infinity;
  for (const [k, size] of [
    [0, u.w],
    [1, u.h],
  ] as const) {
    const dir = t[k];
    if (Math.abs(dir) < 1e-12) {
      if (p[k] < 0 || p[k] > size) return null;
      continue;
    }
    const a = (0 - p[k]) / dir;
    const b = (size - p[k]) / dir;
    lo = Math.max(lo, Math.min(a, b));
    hi = Math.min(hi, Math.max(a, b));
  }
  return lo < hi ? [lo, hi] : null;
}

const labelOf = (name: string): string => `${name}. The arrow keys move it.`;

/** The plan for a Linear: the centre line, the start and end lines, a dot at the centre, a grip on each feather line and the rotate handle. */
function linearPlan(row: Extract<ShapeRow, { kind: "linear" }>, u: FrameUnits): HandlePlan {
  const c = toIso(row.center, u);
  const { t, n } = axes(row.angle_deg);
  const half = row.feather / 2;
  const at = (s: number, along: number): Pt => fromIso(add(add(c, n, s), t, along), u);
  const span = clipLine(c, t, u);
  // The line's own part inside the photo. A line that misses the photo is not drawn, and has only its centre dot (the way back to it).
  const [u0, u1] = span ?? [0, 0];
  const along = (k: number): number => u0 + (u1 - u0) * k;
  const samples = (s: number): Pt[] =>
    span ? Array.from({ length: LINE_SAMPLES + 1 }, (_, i) => at(s, along(i / LINE_SAMPLES))) : [];
  const centre = samples(0);
  const start = samples(-half);
  const end = samples(half);
  const grips: Pt[] = span ? [at(-half, along(0.2)), at(half, along(0.2))] : [];
  // The rotate handle sits on the line, on the side the angle points to (the centre is a point of the photo, so the line has some of its
  // length on that side), three fifths of the way to the photo's edge: at 80 % across a centred line, as in the mockup.
  const rotate: Pt[] = span ? [at(0, u1 > 0.1 ? 0.6 * u1 : u1)] : [];
  const points: Pt[] = [row.center, ...centre, ...start, ...end, ...grips, ...rotate];
  return {
    points,
    draw: (mapped) => {
      let i = 0;
      const take = (count: number): Pt[] => mapped.slice(i, (i += count)).filter((p): p is Pt => p !== null);
      const middle = mapped[i++] ?? null;
      const lines = [
        { id: "centre" as const, points: take(centre.length), dashed: false },
        { id: "start" as const, points: take(start.length), dashed: true },
        { id: "end" as const, points: take(end.length), dashed: true },
      ].filter((l) => l.points.length > 1);
      const handles: HandleDrawing["handles"] = [];
      if (middle) handles.push({ id: "move", label: labelOf("Move the Linear mask"), at: middle });
      const [gs, ge, rot] = [mapped[i++] ?? null, mapped[i++] ?? null, mapped[i++] ?? null];
      if (gs && span) handles.push({ id: "feather-start", label: labelOf("Linear mask feather, start line"), at: gs });
      if (ge && span) handles.push({ id: "feather-end", label: labelOf("Linear mask feather, end line"), at: ge });
      if (rot && span) handles.push({ id: "rotate", label: labelOf("Rotate the Linear mask"), at: rot });
      return { lines, handles };
    },
  };
}

/** How far outside the outline, as a share of the long edge, a Radial's rotate handle stands. */
const ROTATE_OUT = 0.04;

/** The plan for a Radial: the outline, the inner (feather) ring, the centre handle, a handle for each radius and the rotate handle. */
function radialPlan(row: Extract<ShapeRow, { kind: "radial" }>, u: FrameUnits): HandlePlan {
  const c = toIso(row.center, u);
  const { t, n } = axes(row.angle_deg);
  const [rx, ry] = row.radii;
  const ring = (d: number): Pt[] =>
    Array.from({ length: RING_SAMPLES + 1 }, (_, i) => {
      const a = (i / RING_SAMPLES) * 2 * Math.PI;
      return fromIso(add(add(c, t, Math.cos(a) * rx * d), n, Math.sin(a) * ry * d), u);
    });
  const inner = 1 - row.feather;
  // The inner ring is the feather's other end; at no feather it is the outline, and at full feather it is a point.
  const hasInner = row.feather > 0.02 && inner > 0.04;
  const outline = ring(1);
  const innerRing = hasInner ? ring(inner) : [];
  const handles: Pt[] = [fromIso(add(c, t, rx), u), fromIso(add(c, n, ry), u), fromIso(add(c, n, -(ry + ROTATE_OUT)), u)];
  const points: Pt[] = [row.center, ...outline, ...innerRing, ...handles];
  return {
    points,
    draw: (mapped) => {
      let i = 0;
      const take = (count: number): Pt[] => mapped.slice(i, (i += count)).filter((p): p is Pt => p !== null);
      const middle = mapped[i++] ?? null;
      const lines = [
        { id: "outline" as const, points: take(outline.length), dashed: false },
        { id: "inner" as const, points: take(innerRing.length), dashed: true },
      ].filter((l) => l.points.length > 1);
      const [hx, hy, hr] = [mapped[i++] ?? null, mapped[i++] ?? null, mapped[i++] ?? null];
      const out: HandleDrawing["handles"] = [];
      if (middle) out.push({ id: "move", label: labelOf("Move the Radial mask"), at: middle });
      if (hx) out.push({ id: "radius-x", label: labelOf("Radial mask width"), at: hx });
      if (hy) out.push({ id: "radius-y", label: labelOf("Radial mask height"), at: hy });
      if (hr) out.push({ id: "rotate", label: labelOf("Rotate the Radial mask"), at: hr });
      return { lines, handles: out };
    },
  };
}

/** The plan of a shape's handles for a photo whose frame has these units. */
export function handlePlan(row: ShapeRow, units: FrameUnits): HandlePlan {
  return row.kind === "linear" ? linearPlan(row, units) : radialPlan(row, units);
}

/**
 * The row patch a drag of `handle` writes with the pointer at `now` (photo-frame fractions): the handle goes where the pointer is, in
 * the fields it moves. `null` when this handle changes nothing on this row.
 */
export function dragPatch(handle: HandleId, row: ShapeRow, now: Pt, units: FrameUnits): MaskPatch | null {
  const c = toIso(row.center, units);
  const p = toIso(now, units);
  const rel = sub(p, c);
  const { t, n } = axes(row.angle_deg);
  const centre = (at: Pt): MaskPatch => ({ center: [round4(clamp(at[0], 0, 1)), round4(clamp(at[1], 0, 1))] });
  switch (handle) {
    case "move":
      return centre(now);
    case "line":
      // The line goes where the pointer is, across it: along the line nothing changes.
      return row.kind === "linear" ? centre(fromIso(add(c, n, dot(rel, n)), units)) : null;
    case "feather-start":
    case "feather-end":
      return row.kind === "linear" ? { feather: round4(clamp(2 * Math.abs(dot(rel, n)), 0, 1)) } : null;
    case "rotate": {
      if (Math.hypot(rel[0], rel[1]) < 1e-6) return null;
      // A Linear's handle is on the line (the angle is the pointer's direction from the centre); a Radial's stands on the far side of the
      // second radius, a quarter turn behind the first, so its direction is the angle less a quarter turn the other way.
      const a = deg(Math.atan2(rel[1], rel[0]));
      return { angle_deg: round4(wrapDegrees(row.kind === "linear" ? a : a + 90)) };
    }
    case "radius-x":
      return row.kind === "radial" ? { radii: [round4(clamp(Math.abs(dot(rel, t)), RADII_RANGE[0], RADII_RANGE[1])), row.radii[1]] } : null;
    case "radius-y":
      return row.kind === "radial" ? { radii: [row.radii[0], round4(clamp(Math.abs(dot(rel, n)), RADII_RANGE[0], RADII_RANGE[1]))] } : null;
  }
}
