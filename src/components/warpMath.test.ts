import { describe, expect, it } from "vitest";
import type { ColorWarpPin } from "../api/types";
import {
  applyWarp,
  chromaOf,
  clampPin,
  E1,
  E2,
  HUE_ANCHORS,
  hueAngle,
  hueName,
  LOGC4_STOP,
  logc4AtStops,
  logc4Decode,
  logc4Encode,
  lockedPin,
  movingPin,
  pinValueText,
  pixelAt,
  planeHex,
  planeToPx,
  prepareWarp,
  pxToPlane,
  toneOf,
  WARP_COORD_LIMIT,
  WARP_EXPOSURE_LIMIT_EV,
  WARP_MAX_PINS,
  WARP_PIVOT_LIMIT_EV,
  WARP_RANGE_MAX,
  warpDisplacement,
  warpIsIdentity,
} from "./warpMath";

// The values below are worked out by hand from the formulas of docs/02 C9b (the CPU reference is crates/ym-color/src/warp.rs):
//   δ = t − s, φ = exp(−|c − s|²/(2ρ²)), n = max(1, Σφ), λ = Π(1 − exp(−|c − l|²/(2ρ²))), g = smoothstep(0, 0.02, |c|), γ = g·λ,
//   N = clamp(ceil(1.25·Σ|δ|/(ρ√e)), 1, 64). With N = 1 (a move of up to 1.32 ranges) D = γ·Σ(w φ δ)/n; with more, the pixel follows
//   N steps while each centre slides to s + γ·w·δ. E = γ·Σ(w φ x)/n.

const mid = logc4AtStops(0);
const close = (a: readonly number[], b: readonly number[], digits = 12): void => {
  expect(a.length).toBe(b.length);
  a.forEach((x, i) => expect(x).toBeCloseTo(b[i] as number, digits));
};

describe("LogC4 and the plane's axes", () => {
  it("encodes middle grey at 0.27840 and decodes back", () => {
    expect(logc4Encode(0.18)).toBeCloseTo(0.2783958365482653, 14);
    expect(logc4Decode(logc4Encode(0.18))).toBeCloseTo(0.18, 14);
    expect(logc4Decode(-0.1)).toBeCloseTo(-0.1 * 0.1135972086105891 - 0.01805699611991131, 15);
    expect(LOGC4_STOP).toBeCloseTo(0.06479541963412931, 15);
  });

  it("has orthonormal axes in the plane of no brightness change", () => {
    const dot = (a: readonly number[], b: readonly number[]) => a.reduce((s, x, i) => s + x * (b[i] as number), 0);
    expect(dot(E1, E1)).toBeCloseTo(1, 15);
    expect(dot(E2, E2)).toBeCloseTo(1, 15);
    expect(dot(E1, E2)).toBeCloseTo(0, 15);
    expect(E1[0] + E1[1] + E1[2]).toBeCloseTo(0, 15);
    close(chromaOf(pixelAt(0.4, [0.1, -0.05])), [0.1, -0.05], 14);
  });
});

describe("the warp (the CPU reference's maths)", () => {
  it("is the identity with no pins, locked pins only, or pins that have not moved", () => {
    const v = pixelAt(0.4, [0.1, -0.05]);
    for (const pins of [[], [lockedPin([0.1, 0])], [movingPin([0.1, 0])]]) {
      const warp = prepareWarp(pins);
      expect(warpIsIdentity(warp)).toBe(true);
      expect(warp.locks).toEqual([]);
      expect(applyWarp(warp, v)).toEqual(v);
    }
  });

  it("lands a single pin's source on its target and keeps the channel mean", () => {
    const warp = prepareWarp([movingPin([0.12, 0.03], [0.08, 0.1])]);
    for (const n of [0.2, mid, 0.5, 0.7]) {
      const out = applyWarp(warp, pixelAt(n, [0.12, 0.03]));
      close(chromaOf(out), [0.08, 0.1], 12);
      expect((out[0] + out[1] + out[2]) / 3).toBeCloseTo(n, 12);
    }
  });

  it("follows the Gaussian around the source: one range away a short move is exp(−½) of it", () => {
    // s = (0.1, 0), δ = (0, 0.05), ρ = 0.04: 1.25 · 0.05 / (0.04 √e) = 0.95, one step. At c = (0.14, 0), |c − s| = ρ: φ = exp(−0.5).
    const warp = prepareWarp([movingPin([0.1, 0], [0.1, 0.05])]);
    expect(warp.steps).toBe(1);
    const { d, e } = warpDisplacement(warp, [0.14, 0], mid);
    close(d, [0, 0.05 * 0.6065306597126334], 12);
    expect(e).toBe(0);
  });

  it("fades out toward neutral: half the guard gives half the weight", () => {
    // c = (0.01, 0): g = smoothstep(0, 0.02, 0.01) = 0.5; |c − s| = 0.09, φ = exp(−0.0081 / 0.0032) = exp(−2.53125).
    const warp = prepareWarp([movingPin([0.1, 0], [0.1, 0.05])]);
    const { d } = warpDisplacement(warp, [0.01, 0], mid);
    close(d, [0, 0.5 * Math.exp(-2.53125) * 0.05], 14);
    // Neutral itself never moves, whatever the pin.
    expect(warpDisplacement(prepareWarp([{ ...movingPin([0.03, 0], [0.2, 0.1]), range: WARP_RANGE_MAX, exposure_ev: 1 }]), [0, 0], mid)).toEqual({ d: [0, 0], e: 0 });
    for (const n of [0.1, 0.3, 0.6, 0.9]) {
      expect(applyWarp(prepareWarp([movingPin([0.03, 0], [0.2, 0.1])]), [n, n, n])).toEqual([n, n, n]);
    }
  });

  it("keeps the range of a long move and follows it in steps: |δ| = 0.25 at ρ = 0.01 takes 19", () => {
    const s: [number, number] = [0.1, 0];
    const t: [number, number] = [-0.1, 0.15];
    const warp = prepareWarp([{ ...movingPin(s, t), range: 0.01 }]);
    expect(warp.moves[0]?.invTwoSigmaSq).toBeCloseTo(1 / (2 * 0.01 * 0.01), 10);
    // ceil(1.25 · 0.25 / (0.01 √e)) = ceil(18.95) = 19.
    expect(warp.steps).toBe(19);
    // The source lands on the target.
    close(chromaOf(applyWarp(warp, pixelAt(0.4, s))), t, 12);
    // A colour three ranges beside the source moves by at most exp(−4.5) of the move (the grown range of 0.167 moved it by 0.98 of it).
    const across: [number, number] = [0.6, 0.8]; // perpendicular to δ = (−0.2, 0.15) / 0.25
    const { d } = warpDisplacement(warp, [s[0] + 0.03 * across[0], s[1] + 0.03 * across[1]], mid);
    expect(Math.hypot(d[0], d[1])).toBeLessThan(Math.exp(-4.5) * 0.25);
    expect(Math.hypot(d[0], d[1])).toBeGreaterThan(0);
  });

  it("moves the source by its share of a long move: tonal weight 0.4 gives 0.4 of it", () => {
    const warp = prepareWarp([{ ...movingPin([0.1, 0], [0.1, 0.3]), range: 0.01, shadows: 0.4, highlights: 0.4 }]);
    expect(warp.steps).toBe(23);
    close(chromaOf(applyWarp(warp, pixelAt(0.4, [0.1, 0]))), [0.1, 0.12], 12);
  });

  it("takes the step count from every moving pin, at most 64", () => {
    expect(prepareWarp([]).steps).toBe(0);
    expect(prepareWarp([lockedPin([0.1, 0])]).steps).toBe(0);
    expect(prepareWarp([{ ...movingPin([0.1, 0]), exposure_ev: 1 }]).steps).toBe(1);
    expect(prepareWarp([movingPin([0.1, 0], [0.1, 0.1])]).steps).toBe(2);
    expect(prepareWarp([{ ...movingPin([0.5, 0], [-0.5, 0]), range: 0.01 }]).steps).toBe(64);
    expect(prepareWarp([movingPin([0.1, 0], [0.1, 0.1]), movingPin([-0.1, 0], [-0.1, 0.1])]).steps).toBe(4);
  });

  it("holds a locked pin's colour at every tone, and lets a colour one range off the lock move by 1 − exp(−½) of the move", () => {
    const lock = lockedPin([0.1, 0.05]);
    const move = { ...movingPin([0.12, 0.02], [0.02, 0.15]), range: 0.08, exposure_ev: 0.5 };
    const warp = prepareWarp([move, lock]);
    for (const n of [0.15, 0.3, 0.5, 0.8]) {
      const v = pixelAt(n, [0.1, 0.05]);
      expect(applyWarp(warp, v)).toEqual(v);
    }
    // One lock alone, a pin far away from the colour so φ is the same with and without it.
    const alone = prepareWarp([movingPin([0.14, 0.05], [0.14, 0.11])]);
    const held = prepareWarp([movingPin([0.14, 0.05], [0.14, 0.11]), lockedPin([0.1, 0.05])]);
    const free = warpDisplacement(alone, [0.14, 0.05], mid).d;
    const less = warpDisplacement(held, [0.14, 0.05], mid).d;
    close(less, [free[0] * (1 - Math.exp(-0.5)), free[1] * (1 - Math.exp(-0.5))], 12);
  });

  it("shares overlapping pins rather than adding them", () => {
    const one = prepareWarp([movingPin([0.1, 0], [0.1, 0.06])]);
    const two = prepareWarp([movingPin([0.1, 0], [0.1, 0.06]), movingPin([0.1, 0], [0.1, 0.06])]);
    const v = pixelAt(0.4, [0.1, 0]);
    close(chromaOf(applyWarp(two, v)), chromaOf(applyWarp(one, v)), 12);
  });

  it("splits shadows from highlights around the pivot over ±2 stops", () => {
    const pin: ColorWarpPin = { ...movingPin([0.1, 0], [0.1, 0.1]), shadows: 0, highlights: 1 };
    const warp = prepareWarp([pin]);
    // At the pivot the crossover is halfway: w = 0.5; two stops below it 0, two above 1.
    close(warpDisplacement(warp, [0.1, 0], mid).d, [0, 0.05], 12);
    close(warpDisplacement(warp, [0.1, 0], mid - 2 * LOGC4_STOP).d, [0, 0], 12);
    close(warpDisplacement(warp, [0.1, 0], mid + 2 * LOGC4_STOP).d, [0, 0.1], 12);
    // A pivot moves the crossover: at +1 stop with the pivot at +1 the weight is a half again.
    const moved = prepareWarp([{ ...pin, pivot_ev: 1 }]);
    close(warpDisplacement(moved, [0.1, 0], logc4AtStops(1)).d, [0, 0.05], 12);
  });

  it("moves the three channels together by the exposure, one stop being LOGC4_STOP", () => {
    const warp = prepareWarp([{ ...movingPin([0.1, 0.02]), exposure_ev: 1 }]);
    const v = pixelAt(0.4, [0.1, 0.02]);
    const out = applyWarp(warp, v);
    close(chromaOf(out), [0.1, 0.02], 12);
    out.forEach((x, k) => expect(x - (v[k] as number)).toBeCloseTo(LOGC4_STOP, 12));
  });

  it("uses the first sixteen pins only", () => {
    const pins = Array.from({ length: WARP_MAX_PINS }, () => lockedPin([0.2, 0.2]));
    expect(warpIsIdentity(prepareWarp([...pins, movingPin([0.1, 0], [0, 0.1])]))).toBe(true);
    expect(warpIsIdentity(prepareWarp([...pins.slice(1), movingPin([0.1, 0], [0, 0.1])]))).toBe(false);
  });

  it("clamps every value as the core does", () => {
    const pin = clampPin({ source: [3, 4], target: [Number.NaN, 0.1], locked: false, range: 9, shadows: -1, highlights: 2, pivot_ev: 40, exposure_ev: -9 });
    expect(Math.hypot(...pin.source)).toBeCloseTo(WARP_COORD_LIMIT, 12);
    expect(pin.target).toEqual([0, 0.1]);
    expect(pin.range).toBe(WARP_RANGE_MAX);
    expect([pin.shadows, pin.highlights]).toEqual([0, 1]);
    expect(pin.pivot_ev).toBe(WARP_PIVOT_LIMIT_EV);
    expect(pin.exposure_ev).toBe(-WARP_EXPOSURE_LIMIT_EV);
    const locked = clampPin({ ...movingPin([0.1, 0], [0.2, 0]), locked: true, exposure_ev: 1, shadows: 0.2 });
    expect(locked).toEqual(lockedPin([0.1, 0]));
  });

  it("measures the tone as the LogC4 code of the luminance", () => {
    expect(toneOf([mid, mid, mid])).toBeCloseTo(mid, 12);
  });
});

describe("hue names", () => {
  it("are measured on the plane: red at 0°, yellow 60°, green 120°, cyan 180°, blue 240°, magenta 300°", () => {
    const at = Object.fromEntries(HUE_ANCHORS);
    expect(at.red).toBeCloseTo(0, 9);
    expect(at.orange).toBeCloseTo(30, 9);
    expect(at.yellow).toBeCloseTo(60, 9);
    expect(at.green).toBeCloseTo(120, 9);
    expect(at.cyan).toBeCloseTo(180, 9);
    expect(at.blue).toBeCloseTo(240, 9);
    expect(at.purple).toBeCloseTo(270, 9);
    expect(at.magenta).toBeCloseTo(300, 9);
  });

  it("name a chroma by its nearest anchor, and grey within the neutral guard", () => {
    expect(hueName([0.1, 0])).toBe("red");
    expect(hueName([0.1, 0.06])).toBe("orange");
    expect(hueName([-0.05, 0.09])).toBe("green");
    expect(hueName([-0.1, 0])).toBe("cyan");
    expect(hueName([-0.05, -0.09])).toBe("blue");
    expect(hueName([0.05, -0.09])).toBe("magenta");
    expect(hueName([0.01, 0.01])).toBe("grey");
    expect(hueAngle([0, -1])).toBeCloseTo(270, 12);
  });

  it("say what a pin does", () => {
    expect(pinValueText(movingPin([0.1, 0], [0.1, 0.06]))).toBe("moves red toward orange");
    expect(pinValueText(lockedPin([0.1, 0]))).toBe("holds red");
    expect(pinValueText(movingPin([-0.1, 0]))).toBe("cyan, not moved");
    expect(pinValueText(movingPin([0.1, 0], [0.15, 0]))).toBe("moves red toward a stronger red");
    expect(pinValueText(movingPin([0.1, 0], [0.05, 0]))).toBe("moves red toward a paler red");
  });
});

describe("the plane", () => {
  it("puts |c| = 0.30 at the middle of each edge, red to the right and +b up", () => {
    expect(planeToPx([0, 0], 352)).toEqual({ x: 176, y: 176 });
    expect(planeToPx([0.3, 0], 352)).toEqual({ x: 352, y: 176 });
    expect(planeToPx([0, 0.3], 352).y).toBeCloseTo(0, 12);
    close(pxToPlane(352, 0, 352), [0.3, 0.3], 12);
  });

  it("paints a pin's dot in the colour of its point: red to the right is red, cyan to the left cyan", () => {
    const rgb = (hex: string) => [1, 3, 5].map((i) => Number.parseInt(hex.slice(i, i + 2), 16));
    const [r, g, b] = rgb(planeHex([0.15, 0]));
    expect(r).toBeGreaterThan(g as number);
    expect(r).toBeGreaterThan(b as number);
    const [r2, g2, b2] = rgb(planeHex([-0.15, 0]));
    expect(g2).toBeGreaterThan(r2 as number);
    expect(b2).toBeGreaterThan(r2 as number);
  });
});
