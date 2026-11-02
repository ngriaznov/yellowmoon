/**
 * What a Brush stroke is on the page (`docs/23` §7.6, §3.7): the stroke made from the places the core mapped, the paper dropped, the simplification
 * with its tolerance of 1/8 of the brush size, and the rule of 100,000 points.
 */
import { describe, expect, it } from "vitest";
import type { MaskItem } from "../api/types";
import { frameUnits } from "../screens/nodes/maskGeometry";
import type { Pt } from "../screens/nodes/maskGeometry";
import { brushPointCount, buildStroke, DEFAULT_BRUSH, MAX_BRUSH_POINTS, simplifiedIndices, simplifyTolerance } from "./brushStrokes";
import { newMask } from "./maskRows";

/** A 3:2 photo: the long edge is the width, so a fraction of the height is 2/3 as long as the same fraction of the width. */
const UNITS = frameUnits(1.5);
const SQUARE = frameUnits(1);

/** A straight line of `n` points along y = 0.5, from x = 0.1 to 0.9. */
const line = (n: number): Pt[] => Array.from({ length: n }, (_, i) => [0.1 + (0.8 * i) / (n - 1), 0.5]);

describe("the brush", () => {
  it("starts at a size of 0.03, a hardness of 0.5, painting (a new Brush row's own)", () => {
    expect(DEFAULT_BRUSH).toEqual({ size: 0.03, hardness: 0.5, erase: false });
  });

  it("counts the points in all the strokes of a Brush row, and none for another kind", () => {
    const brush = { ...newMask("brush"), strokes: [{ size: 0.03, hardness: 0.5, erase: false, points: [[0, 0], [1, 1]] }, { size: 0.03, hardness: 0.5, erase: true, points: [[0.5, 0.5]] }] } as MaskItem;
    expect(brushPointCount(brush)).toBe(3);
    expect(brushPointCount(newMask("brush"))).toBe(0);
    expect(brushPointCount(newMask("radial"))).toBe(0);
    expect(brushPointCount(undefined)).toBe(0);
    expect(MAX_BRUSH_POINTS).toBe(100_000);
  });
});

describe("building a stroke from the places the core mapped", () => {
  it("takes the brush's size, hardness and erase, and the places in order", () => {
    const stroke = buildStroke([[0.1, 0.2], [0.1, 0.9]], { size: 0.05, hardness: 0.8, erase: true }, UNITS);
    expect(stroke).toEqual({ size: 0.05, hardness: 0.8, erase: true, points: [[0.1, 0.2], [0.1, 0.9]] });
  });

  it("drops the places on the paper, and a stroke with none on the photo is nothing", () => {
    const stroke = buildStroke([null, [0.4, 0.4], null, [0.6, 0.5], null], DEFAULT_BRUSH, UNITS);
    expect(stroke?.points).toEqual([[0.4, 0.4], [0.6, 0.5]]);
    expect(buildStroke([null, null], DEFAULT_BRUSH, UNITS)).toBeNull();
    expect(buildStroke([], DEFAULT_BRUSH, UNITS)).toBeNull();
    expect(buildStroke([[Number.NaN, 0.5]], DEFAULT_BRUSH, UNITS)).toBeNull();
  });

  it("makes a single click a stroke of one point", () => {
    expect(buildStroke([[0.3, 0.7]], DEFAULT_BRUSH, UNITS)).toEqual({ size: 0.03, hardness: 0.5, erase: false, points: [[0.3, 0.7]] });
  });

  it("holds the places to the photo frame and the size and hardness to their ranges, and rounds to four decimals", () => {
    const stroke = buildStroke([[-0.2, 1.4], [0.123456789, 0.987654321]], { size: 9, hardness: -1, erase: false }, UNITS);
    expect(stroke?.size).toBe(0.25);
    expect(stroke?.hardness).toBe(0);
    expect(stroke?.points).toEqual([[0, 1], [0.1235, 0.9877]]);
    expect(buildStroke([[0.5, 0.5]], { size: 0, hardness: 1, erase: false }, UNITS)?.size).toBe(0.001);
  });
});

describe("simplifying a stroke (the points nearer than 1/8 of the brush size to the line between their neighbours go)", () => {
  it("has a tolerance of an eighth of the size", () => {
    expect(simplifyTolerance(0.08)).toBeCloseTo(0.01, 12);
    expect(simplifyTolerance(0.001)).toBeCloseTo(0.000125, 12);
  });

  it("keeps the two ends of a straight line and nothing between", () => {
    const stroke = buildStroke(line(200), DEFAULT_BRUSH, UNITS);
    expect(stroke?.points).toEqual([[0.1, 0.5], [0.9, 0.5]]);
  });

  it("keeps a bend that is farther from the line than the tolerance, and drops one that is nearer", () => {
    // Size 0.08: the tolerance is 0.01 of the long edge. On a square photo a y of 0.5 → 0.52 is 0.02 away: kept; 0.505 is 0.005 away: dropped.
    const big = buildStroke([[0.1, 0.5], [0.5, 0.52], [0.9, 0.5]], { size: 0.08, hardness: 0.5, erase: false }, SQUARE);
    expect(big?.points).toHaveLength(3);
    const small = buildStroke([[0.1, 0.5], [0.5, 0.505], [0.9, 0.5]], { size: 0.08, hardness: 0.5, erase: false }, SQUARE);
    expect(small?.points).toEqual([[0.1, 0.5], [0.9, 0.5]]);
  });

  it("measures in lengths of the photo, not in fractions: the same y step is 2/3 as far on a 3:2 photo", () => {
    // 0.012 of the height on a 3:2 photo is 0.008 of the long edge (dropped at a tolerance of 0.01) and 0.012 on a square one (kept).
    const path: Pt[] = [[0.1, 0.5], [0.5, 0.512], [0.9, 0.5]];
    const brush = { size: 0.08, hardness: 0.5, erase: false };
    expect(buildStroke(path, brush, UNITS)?.points).toHaveLength(2);
    expect(buildStroke(path, brush, SQUARE)?.points).toHaveLength(3);
  });

  it("keeps the shape of a curve within the tolerance, with far fewer points", () => {
    const arc: Pt[] = Array.from({ length: 400 }, (_, i) => {
      const t = (Math.PI * i) / 399;
      return [0.5 + 0.4 * Math.cos(t), 0.5 - 0.4 * Math.sin(t)];
    });
    const brush = { size: 0.04, hardness: 0.5, erase: false };
    const kept = simplifiedIndices(arc.map(([x, y]) => [x * SQUARE.w, y * SQUARE.h] as Pt), simplifyTolerance(brush.size));
    expect(kept.length).toBeLessThan(40);
    expect(kept[0]).toBe(0);
    expect(kept.at(-1)).toBe(399);
    // Every point that went is nearer to the polyline of the ones that stayed than the tolerance.
    const poly = kept.map((i) => arc[i] as Pt);
    const near = (p: Pt): number =>
      Math.min(
        ...poly.slice(1).map((b, k) => {
          const a = poly[k] as Pt;
          const [dx, dy] = [b[0] - a[0], b[1] - a[1]];
          const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
          return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
        }),
      );
    for (const p of arc) expect(near(p)).toBeLessThan(simplifyTolerance(brush.size) + 1e-9);
  });

  it("keeps a closed path's far side (the ends are one point)", () => {
    const loop: Pt[] = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7], [0.3, 0.3]];
    expect(buildStroke(loop, DEFAULT_BRUSH, SQUARE)?.points).toHaveLength(5);
  });

  it("leaves one point and two points as they are, and does not overflow the stack on a path of ten thousand", () => {
    expect(simplifiedIndices([[0, 0]], 0.1)).toEqual([0]);
    expect(simplifiedIndices([[0, 0], [1, 1]], 0.1)).toEqual([0, 1]);
    const spiral: Pt[] = Array.from({ length: 10_000 }, (_, i) => {
      const t = (i / 10_000) * 40;
      return [0.5 + 0.004 * t * Math.cos(t), 0.5 + 0.004 * t * Math.sin(t)];
    });
    const kept = simplifiedIndices(spiral, 0.002);
    expect(kept.length).toBeGreaterThan(100);
    expect(kept.length).toBeLessThan(2_000);
  });

  it("keeps every point of a zigzag that is wider than the tolerance", () => {
    const zigzag: Pt[] = Array.from({ length: 40 }, (_, i) => [0.05 + i * 0.02, i % 2 === 0 ? 0.2 : 0.5]);
    expect(simplifiedIndices(zigzag, 0.01)).toHaveLength(40);
  });
});
