import { describe, expect, it } from "vitest";
import {
  addHuePoint,
  axisTicks,
  circularGap,
  evaluateHueCurve,
  FACTOR_STEPS,
  FULL_SCALE,
  hueCurvePath,
  isFullScale,
  MAX_HUE_POINTS,
  moveHuePoint,
  periodicTangents,
  pointReadout,
  removeHuePoint,
  roundValue,
  sortedPoints,
  TURN_STEPS,
  unitToValue,
  valueToUnit,
  widenedScale,
  widestGapHue,
  wrapHue,
  zoomedScale,
} from "./hueCurve";
import type { HuePoint } from "./hueCurve";

const SIX = (values: number[]): HuePoint[] => [0, 60, 120, 180, 240, 300].map((hue_deg, i) => ({ hue_deg, value: values[i] as number }));

describe("the periodic monotone spline (docs/34 §4.1)", () => {
  it("is the identity with no point, a constant with one, and passes through its points", () => {
    expect(evaluateHueCurve([], 123, 1)).toBe(1);
    expect(evaluateHueCurve([{ hue_deg: 40, value: 7 }], 300, 0)).toBe(7);
    const points = SIX([-4, 6, 12, 4, -3, 2]);
    for (const p of points) expect(evaluateHueCurve(points, p.hue_deg, 0)).toBeCloseTo(p.value, 12);
  });

  it("is closed at the seam: 0° and 360° are one value and a sweep through it has no step", () => {
    const points = SIX([-4, 6, 12, 4, -3, 2]);
    expect(evaluateHueCurve(points, 360, 0)).toBeCloseTo(evaluateHueCurve(points, 0, 0), 12);
    let worst = 0;
    let last = evaluateHueCurve(points, 290, 0);
    for (let hue = 290.1; hue <= 430; hue += 0.1) {
      const v = evaluateHueCurve(points, hue, 0);
      worst = Math.max(worst, Math.abs(v - last));
      last = v;
    }
    // The steepest slope of the data is 6 per 60° (0.1 per degree): a sweep of a tenth of a degree moves at most about that much.
    expect(worst).toBeLessThan(0.2);
  });

  it("is C1 across the seam: the slope just before 360° is the slope just after", () => {
    const points = SIX([-4, 6, 12, 4, -3, 2]);
    const slope = (hue: number): number => (evaluateHueCurve(points, hue + 0.01, 0) - evaluateHueCurve(points, hue - 0.01, 0)) / 0.02;
    expect(slope(359.99)).toBeCloseTo(slope(0.01), 2);
  });

  it("never overshoots the values it is given", () => {
    const points = SIX([0, 0, 5, 5, 0, 0]);
    for (let hue = 0; hue < 360; hue += 0.5) {
      const v = evaluateHueCurve(points, hue, 0);
      expect(v).toBeGreaterThanOrEqual(-1e-9);
      expect(v).toBeLessThanOrEqual(5 + 1e-9);
    }
  });

  it("has zero tangents at an extremum and for a flat segment", () => {
    const m = periodicTangents(SIX([0, 1, 2, 1, 0, 0]));
    expect(m[2]).toBe(0);
    expect(m[4]).toBe(0);
  });

  it("reads hues modulo 360 and sorts the points", () => {
    expect(wrapHue(-30)).toBe(330);
    expect(wrapHue(720)).toBe(0);
    expect(sortedPoints([{ hue_deg: 370, value: 1 }, { hue_deg: -10, value: 2 }, { hue_deg: 5, value: 3 }]).map((p) => p.hue_deg)).toEqual([5, 10, 350]);
  });
});

describe("the two scales", () => {
  it("map the turn linearly and the multiplier on a log scale with its detent at 1", () => {
    const turn = FULL_SCALE.hue_vs_hue;
    expect(valueToUnit(turn, 0)).toBeCloseTo(0.5, 12);
    expect(valueToUnit(turn, 180)).toBe(1);
    expect(unitToValue(turn, 0.25)).toBeCloseTo(-90, 9);
    const sat = FULL_SCALE.hue_vs_sat;
    expect(valueToUnit(sat, 1)).toBeCloseTo(0.5, 12);
    expect(valueToUnit(sat, 8)).toBeCloseTo(1, 12);
    expect(valueToUnit(sat, 0.125)).toBeCloseTo(0, 12);
    expect(valueToUnit(sat, 0)).toBe(0);
    expect(unitToValue(sat, 0.75)).toBeCloseTo(2 ** 1.5, 9);
    // The foot of the full range is "down to grey".
    expect(unitToValue(sat, 0)).toBe(0);
  });

  it("round-trip a value through the plot", () => {
    for (const kind of ["hue_vs_hue", "hue_vs_sat"] as const) {
      for (const v of kind === "hue_vs_hue" ? [-120, -3.5, 0, 22] : [0.4, 0.85, 1, 1.15, 3]) {
        const scale = FULL_SCALE[kind];
        expect(unitToValue(scale, valueToUnit(scale, v))).toBeCloseTo(v, 9);
      }
    }
  });

  it("label the axis: the turn's detent at 0, the multiplier's at 1 with ½, ¼ and 0 below it", () => {
    expect(axisTicks(FULL_SCALE.hue_vs_hue).map((t) => t.label)).toEqual(["+180°", "+90°", "0", "−90°", "−180°"]);
    expect(axisTicks(FULL_SCALE.hue_vs_sat).map((t) => t.label)).toEqual(["8", "4", "2", "1", "½", "¼", "0"]);
  });
});

describe("the zoomed default view that fits the fitted values (docs/34 §9.4)", () => {
  it("opens a fit's small turns on a small range and keeps the full range reachable", () => {
    expect(zoomedScale("hue_vs_hue", SIX([-4, 6, 12, 4, -3, 2]))).toMatchObject({ low: -15, high: 15 });
    expect(zoomedScale("hue_vs_hue", SIX([-4, 6, 22, 4, -3, 2]))).toMatchObject({ low: -30, high: 30 });
    expect(zoomedScale("hue_vs_hue", [])).toMatchObject({ high: TURN_STEPS[0] });
    expect(zoomedScale("hue_vs_hue", [{ hue_deg: 10, value: 170 }])).toMatchObject({ high: 180 });
    expect(isFullScale(zoomedScale("hue_vs_hue", [{ hue_deg: 10, value: 170 }]))).toBe(true);
    expect(isFullScale(zoomedScale("hue_vs_hue", []))).toBe(false);
  });

  it("opens a fit's multipliers (0.85 to 1.12) on a narrow factor range around 1", () => {
    const scale = zoomedScale("hue_vs_sat", SIX([1.05, 1.08, 0.85, 1.12, 1.06, 0.92]));
    expect(scale.high).toBe(FACTOR_STEPS[1]);
    expect(scale.low).toBeCloseTo(1 / FACTOR_STEPS[1], 12);
    expect(zoomedScale("hue_vs_sat", SIX([1.02, 1.05, 0.97, 1.1, 1.04, 0.99])).high).toBe(FACTOR_STEPS[0]);
    expect(valueToUnit(scale, 1)).toBeCloseTo(0.5, 12);
    expect(zoomedScale("hue_vs_sat", [{ hue_deg: 0, value: 0 }]).high).toBe(8);
    expect(zoomedScale("hue_vs_sat", [{ hue_deg: 0, value: 3.5 }]).high).toBe(4);
  });

  it("widens as a point is moved past the edge and never narrows by itself", () => {
    const small = zoomedScale("hue_vs_hue", SIX([1, 1, 1, 1, 1, 1]));
    const wider = widenedScale(small, SIX([1, 1, 40, 1, 1, 1]));
    expect(wider.high).toBe(90);
    expect(widenedScale(wider, SIX([1, 1, 1, 1, 1, 1])).high).toBe(90);
  });
});

describe("editing", () => {
  it("adds a point on the curve at a hue, up to 16, a degree from the others", () => {
    const points = SIX([0, 10, 0, -10, 0, 10]);
    const added = addHuePoint(points, "hue_vs_hue", 30)!;
    expect(added.points).toHaveLength(7);
    expect(added.points[added.index]).toMatchObject({ hue_deg: 30 });
    expect(addHuePoint(points, "hue_vs_hue", 60.4)).toBeNull();
    expect(addHuePoint(points, "hue_vs_hue", 359.8)).toBeNull();
    const full = Array.from({ length: MAX_HUE_POINTS }, (_, i) => ({ hue_deg: i * 20, value: 0 }));
    expect(addHuePoint(full, "hue_vs_hue", 10)).toBeNull();
    expect(addHuePoint([], "hue_vs_sat", 100)!.points).toEqual([{ hue_deg: 100, value: 1 }]);
  });

  it("moves a point between its neighbours, a degree from each, and never past them", () => {
    const points = SIX([0, 0, 0, 0, 0, 0]);
    expect(moveHuePoint(points, "hue_vs_hue", 2, 500, 5)[2]).toEqual({ hue_deg: 179, value: 5 });
    expect(moveHuePoint(points, "hue_vs_hue", 2, -50, 5)[2]).toEqual({ hue_deg: 61, value: 5 });
    // The first point can go left to the seam only as far as the last allows.
    expect(moveHuePoint(points, "hue_vs_hue", 0, -100, 0)[0]?.hue_deg).toBe(0);
    expect(moveHuePoint(points, "hue_vs_hue", 5, 400, 0)[5]?.hue_deg).toBe(359);
  });

  it("holds a value to the field's limits and rounds it", () => {
    expect(roundValue("hue_vs_hue", 200.04)).toBe(180);
    expect(roundValue("hue_vs_hue", 12.34)).toBe(12.3);
    expect(roundValue("hue_vs_sat", 9)).toBe(8);
    expect(roundValue("hue_vs_sat", -1)).toBe(0);
    expect(roundValue("hue_vs_sat", 0.8549)).toBe(0.85);
  });

  it("removes a point, down to none", () => {
    expect(removeHuePoint(SIX([1, 2, 3, 4, 5, 6]), 2).map((p) => p.value)).toEqual([1, 2, 4, 5, 6]);
    expect(removeHuePoint([{ hue_deg: 4, value: 1 }], 0)).toEqual([]);
  });

  it("finds the widest gap round the circle, the seam included", () => {
    expect(widestGapHue([])).toBe(0);
    expect(widestGapHue([{ hue_deg: 0, value: 0 }])).toBe(180);
    expect(widestGapHue([{ hue_deg: 10, value: 0 }, { hue_deg: 100, value: 0 }, { hue_deg: 200, value: 0 }])).toBe(285);
    expect(circularGap(350, 10)).toBe(20);
  });
});

describe("the plot's words", () => {
  it("reads a point as 'Hue 120°, Turn +22°' and 'Hue 120°, Saturation 80%'", () => {
    expect(pointReadout("hue_vs_hue", { hue_deg: 120, value: 22 })).toEqual({ hue: "120°", label: "Turn", value: "+22°" });
    expect(pointReadout("hue_vs_hue", { hue_deg: 120, value: -3.2 })).toEqual({ hue: "120°", label: "Turn", value: "−3°" });
    expect(pointReadout("hue_vs_sat", { hue_deg: 120, value: 0.8 })).toEqual({ hue: "120°", label: "Saturation", value: "80%" });
  });

  it("draws the curve as one path round the circle, and the identity as a straight line", () => {
    const path = hueCurvePath(SIX([0, 10, 0, -10, 0, 10]), zoomedScale("hue_vs_hue", SIX([0, 10, 0, -10, 0, 10])), 320, 170);
    expect(path.startsWith("M0 ")).toBe(true);
    expect(path.split("L")).toHaveLength(181);
    const flat = hueCurvePath([], FULL_SCALE.hue_vs_hue, 320, 170);
    expect(new Set(flat.split(/[ML]/).filter(Boolean).map((p) => p.split(" ")[1]))).toEqual(new Set(["85"]));
  });
});
