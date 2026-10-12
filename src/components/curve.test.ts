import { describe, expect, it } from "vitest";
import {
  centralInnerPoint,
  CURVE_LUT_SIZE,
  curvePath,
  evaluateCurve,
  identityCurve,
  insertPoint,
  isIdentityCurve,
  MAX_CURVE_POINTS,
  MIN_POINT_GAP,
  monotoneTangents,
  movePoint,
  removePoint,
  sampleCurve,
  toCodeValue,
  widestGapX,
} from "./curve";
import type { CurvePoint } from "./curve";

describe("monotone cubic spline (Fritsch–Carlson)", () => {
  it("is the identity for the identity curve", () => {
    const points = identityCurve();
    for (const x of [0, 0.1, 0.25, 0.5, 0.9, 1]) expect(evaluateCurve(points, x)).toBeCloseTo(x, 12);
    expect(isIdentityCurve(points)).toBe(true);
  });

  it("passes through every control point", () => {
    const points: CurvePoint[] = [[0, 0], [0.25, 0.1], [0.5, 0.6], [0.8, 0.7], [1, 1]];
    for (const [x, y] of points) expect(evaluateCurve(points, x)).toBeCloseTo(y, 12);
  });

  it("matches a hand calculation: points (0,0) (0.5,0.8) (1,1) at x = 0.25", () => {
    // Secants 1.6 and 0.4; the middle tangent is their mean, 1.0; the ends take their secants.
    // Hermite at t = 0.5 on the first segment: 0.125·0.5·1.6 + 0.5·0.8 − 0.125·0.5·1.0 = 0.4375.
    const points: CurvePoint[] = [[0, 0], [0.5, 0.8], [1, 1]];
    const tangents = monotoneTangents(points);
    expect(tangents[0]).toBeCloseTo(1.6, 12);
    expect(tangents[1]).toBeCloseTo(1, 12);
    expect(tangents[2]).toBeCloseTo(0.4, 12);
    expect(evaluateCurve(points, 0.25)).toBeCloseTo(0.4375, 12);
  });

  it("zeroes the tangent at a local extremum, so the curve never overshoots", () => {
    const points: CurvePoint[] = [[0, 0.2], [0.5, 0.9], [1, 0.3]];
    expect(monotoneTangents(points)[1]).toBe(0);
    for (let i = 0; i <= 1000; i++) expect(evaluateCurve(points, i / 1000)).toBeLessThanOrEqual(0.9 + 1e-12);
  });

  it("limits steep tangents so monotone data stays monotone and within range", () => {
    const points: CurvePoint[] = [[0, 0], [0.1, 0.9], [1, 1]];
    let previous = -1;
    for (let i = 0; i <= 2000; i++) {
      const y = evaluateCurve(points, i / 2000);
      expect(y).toBeGreaterThanOrEqual(previous - 1e-12);
      expect(y).toBeLessThanOrEqual(1);
      previous = y;
    }
  });

  it("keeps flat segments flat", () => {
    const points: CurvePoint[] = [[0, 0.2], [0.4, 0.5], [0.6, 0.5], [1, 0.9]];
    for (let i = 0; i <= 20; i++) expect(evaluateCurve(points, 0.4 + (0.2 * i) / 20)).toBeCloseTo(0.5, 12);
  });

  it("is flat outside the first and last point", () => {
    const points: CurvePoint[] = [[0.2, 0.3], [0.8, 0.7]];
    expect(evaluateCurve(points, 0)).toBe(0.3);
    expect(evaluateCurve(points, 1)).toBe(0.7);
  });

  it("samples a 1024-entry table that agrees with evaluateCurve", () => {
    const points: CurvePoint[] = [[0, 0], [0.3, 0.2], [0.7, 0.85], [1, 1]];
    const table = sampleCurve(points);
    expect(table).toHaveLength(CURVE_LUT_SIZE);
    expect(table[0]).toBe(0);
    expect(table[CURVE_LUT_SIZE - 1]).toBe(1);
    const i = 512;
    expect(table[i]).toBeCloseTo(evaluateCurve(points, i / (CURVE_LUT_SIZE - 1)), 6);
  });

  it("reads code values as 0–255", () => {
    expect(toCodeValue(0)).toBe(0);
    expect(toCodeValue(0.5)).toBe(128);
    expect(toCodeValue(1)).toBe(255);
    expect(toCodeValue(2)).toBe(255);
  });
});

describe("curvePath", () => {
  it("starts bottom left and ends top right for the identity curve", () => {
    expect(curvePath(identityCurve(), 320, 220)).toBe("M0 220 C106.67 146.67, 213.33 73.33, 320 0");
  });

  it("extends flat to the plot edges when the ends are inside", () => {
    const d = curvePath([[0.25, 0.25], [0.75, 0.75]], 320, 220);
    expect(d.startsWith("M0 165 L80 165 C")).toBe(true);
    expect(d.endsWith("L320 55")).toBe(true);
  });
});

describe("editing", () => {
  it("inserts a point on the curve, keeping the list sorted", () => {
    const result = insertPoint(identityCurve(), 0.5);
    expect(result?.index).toBe(1);
    expect(result?.points).toEqual([[0, 0], [0.5, 0.5], [1, 1]]);
  });

  it("refuses a point too close to a neighbour and a seventeenth point", () => {
    expect(insertPoint(identityCurve(), MIN_POINT_GAP / 2)).toBeNull();
    const full: CurvePoint[] = Array.from({ length: MAX_CURVE_POINTS }, (_, i) => [i / (MAX_CURVE_POINTS - 1), i / (MAX_CURVE_POINTS - 1)]);
    expect(insertPoint(full, 0.033)).toBeNull();
  });

  it("moves an inner point freely between its neighbours", () => {
    const points: CurvePoint[] = [[0, 0], [0.5, 0.5], [1, 1]];
    expect(movePoint(points, 1, 0.3, 0.7)).toEqual([[0, 0], [0.3, 0.7], [1, 1]]);
    const clamped = movePoint(points, 1, 2, -1);
    expect(clamped[1]?.[0]).toBeCloseTo(1 - MIN_POINT_GAP, 12);
    expect(clamped[1]?.[1]).toBe(0);
  });

  it("moves the end points only up and down", () => {
    const points: CurvePoint[] = [[0, 0], [1, 1]];
    expect(movePoint(points, 0, 0.5, 0.2)).toEqual([[0, 0.2], [1, 1]]);
    expect(movePoint(points, 1, 0.5, 0.8)).toEqual([[0, 0], [1, 0.8]]);
  });

  it("removes an inner point but never an end or the last two points", () => {
    const points: CurvePoint[] = [[0, 0], [0.5, 0.5], [1, 1]];
    expect(removePoint(points, 1)).toEqual([[0, 0], [1, 1]]);
    expect(removePoint(points, 0)).toEqual(points);
    expect(removePoint(points, 2)).toEqual(points);
    expect(removePoint([[0, 0], [1, 1]], 1)).toEqual([[0, 0], [1, 1]]);
  });

  it("never mutates its input", () => {
    const points: CurvePoint[] = [[0, 0], [0.5, 0.5], [1, 1]];
    const copy = JSON.stringify(points);
    movePoint(points, 1, 0.2, 0.2);
    insertPoint(points, 0.7);
    removePoint(points, 1);
    expect(JSON.stringify(points)).toBe(copy);
  });

  it("finds the middle of the widest gap", () => {
    expect(widestGapX([[0, 0], [0.2, 0.2], [1, 1]])).toBeCloseTo(0.6, 12);
    expect(isIdentityCurve([[0, 0], [0.5, 0.5], [1, 1]])).toBe(false);
  });
});

describe("centralInnerPoint", () => {
  it("picks the inner point nearest the middle, and none for the bare ends", () => {
    expect(centralInnerPoint([[0, 0], [1, 1]])).toBeNull();
    expect(centralInnerPoint([[0, 0], [0.25, 0.2], [0.5, 0.5], [0.75, 0.8], [1, 1]])).toBe(2);
    expect(centralInnerPoint([[0, 0], [0.2, 0.2], [0.45, 0.4], [0.9, 0.8], [1, 1]])).toBe(2);
    expect(centralInnerPoint([[0, 0], [0.3, 0.2], [1, 1]])).toBe(1);
  });
});
