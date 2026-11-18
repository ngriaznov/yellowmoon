// @vitest-environment node
// The Colour Uniform node's constants in TypeScript are copies of the core's (`crates/ym-color/src/uniform.rs`, `consts.rs`): this reads the
// Rust files and fails when a number on either side changes without the other. It also checks the port's maths against the worked
// example and the properties of `docs/32` §8 that the page relies on.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  antipodeFade,
  applyUniform,
  colourName,
  fromStored,
  HUE_FADE,
  HUE_FULL,
  LOGC4_GRAY,
  neutralGuard,
  swatchLabel,
  TARGET_CHROMA_MAX,
  TARGET_CHROMA_MIN,
  TARGET_TONE_MAX,
  TARGET_TONE_MIN,
  targetFromSample,
  targetRgb,
  toStored,
  UNIFORM_NEUTRAL_GUARD,
} from "./uniformMath";
import { chromaOf } from "./warpMath";
import type { Rgb } from "./warpMath";

const root = new URL("../../", import.meta.url);
const read = (path: string): string => readFileSync(new URL(path, root), "utf8");

/** The value of `pub const NAME: type = <value>;` in a Rust source, as written. */
function rustConst(source: string, name: string): string {
  const match = new RegExp(`pub const ${name}: \\w+ = ([^;]+);`).exec(source);
  if (!match?.[1]) throw new Error(`${name} is not a constant of the Rust file`);
  const text = match[1].trim();
  return /^[\d._]+$/.test(text) ? text.replaceAll("_", "") : text;
}

describe("the Colour Uniform constants match crates/ym-color/src/uniform.rs", () => {
  const uniform = read("crates/ym-color/src/uniform.rs");
  it("the guard and the targets' limits", () => {
    expect(rustConst(uniform, "UNIFORM_NEUTRAL_GUARD")).toBe("WARP_NEUTRAL_GUARD");
    expect(UNIFORM_NEUTRAL_GUARD).toBe(0.02);
    expect(rustConst(uniform, "TARGET_CHROMA_MIN")).toBe("UNIFORM_NEUTRAL_GUARD");
    expect(TARGET_CHROMA_MIN).toBe(UNIFORM_NEUTRAL_GUARD);
    expect(Number(rustConst(uniform, "TARGET_CHROMA_MAX"))).toBe(TARGET_CHROMA_MAX);
    expect(Number(rustConst(uniform, "TARGET_TONE_MIN"))).toBe(TARGET_TONE_MIN);
    expect(Number(rustConst(uniform, "TARGET_TONE_MAX"))).toBe(TARGET_TONE_MAX);
  });
  it("the antipode fade's two angles", () => {
    expect(rustConst(uniform, "HUE_FULL")).toBe("std::f64::consts::FRAC_PI_2");
    expect(rustConst(uniform, "HUE_FADE")).toBe("std::f64::consts::FRAC_PI_3");
    expect(HUE_FULL).toBe(Math.PI / 2);
    expect(HUE_FADE).toBe(Math.PI / 3);
  });
  it("middle grey's code", () => {
    expect(Number(rustConst(read("crates/ym-color/src/consts.rs"), "LOGC4_GRAY"))).toBe(LOGC4_GRAY);
  });
  it("the guards' formulas are the ones written down", () => {
    expect(uniform).toContain("smooth(r / UNIFORM_NEUTRAL_GUARD)");
    expect(uniform).toContain("1.0 - smooth((delta.abs() - HUE_FULL) / HUE_FADE)");
  });
});

describe("the conversions", () => {
  it("stored units and the plane agree both ways", () => {
    const stored = { hue_deg: 26, saturation: 9.7, lightness_ev: -0.03 };
    const back = toStored(fromStored(stored));
    expect(back.hue_deg).toBeCloseTo(26, 9);
    expect(back.saturation).toBeCloseTo(9.7, 9);
    expect(back.lightness_ev).toBeCloseTo(-0.03, 9);
  });
  it("a hue is read modulo 360", () => {
    expect(toStored(fromStored({ hue_deg: -334, saturation: 10, lightness_ev: 0 })).hue_deg).toBeCloseTo(26, 9);
  });
  it("a pick becomes Hue, Saturation and Lightness", () => {
    const { target, tooGrey } = targetFromSample({ a: 0.0875, b: 0.0423, tone: LOGC4_GRAY });
    expect(tooGrey).toBe(false);
    expect(target.hue_deg).toBeCloseTo((Math.atan2(0.0423, 0.0875) * 180) / Math.PI, 9);
    expect(target.saturation).toBeCloseTo(100 * Math.hypot(0.0875, 0.0423), 9);
    expect(target.lightness_ev).toBeCloseTo(0, 9);
  });
  it("a pick too grey for a hue is held at Saturation 2 on its direction", () => {
    const { target, tooGrey } = targetFromSample({ a: 0.006, b: 0.008, tone: 0.3 });
    expect(tooGrey).toBe(true);
    expect(target.saturation).toBeCloseTo(2, 9);
    expect(target.hue_deg).toBeCloseTo((Math.atan2(0.008, 0.006) * 180) / Math.PI, 9);
    expect(targetFromSample({ a: 0, b: 0, tone: 0.3 }).target.hue_deg).toBe(0);
  });
  it("a strong pick is held at Saturation 50 and the tone at its limits", () => {
    expect(targetFromSample({ a: 0.9, b: 0, tone: 0.3 }).target.saturation).toBeCloseTo(50, 9);
    expect(targetFromSample({ a: 0.1, b: 0, tone: 3 }).target.lightness_ev).toBe(8);
    expect(targetFromSample({ a: 0.1, b: 0, tone: -3 }).target.lightness_ev).toBe(-4);
  });
});

describe("the maths", () => {
  const target = fromStored({ hue_deg: 26, saturation: 9.7, lightness_ev: -0.03 });
  const px = (c: [number, number], m: number): Rgb => [0, 1, 2].map((i) => m + c[0] * [1 / Math.sqrt(1.5), -0.5 / Math.sqrt(1.5), -0.5 / Math.sqrt(1.5)][i]! + c[1] * [0, Math.SQRT1_2, -Math.SQRT1_2][i]!) as Rgb;
  it("no target, or three zero amounts, change nothing", () => {
    const v: Rgb = [0.4, 0.3, 0.2];
    expect(applyUniform(v, null, { hue: 50, saturation: 50, lightness: 0 })).toEqual(v);
    expect(applyUniform(v, target, { hue: 0, saturation: 0, lightness: 0 })).toEqual(v);
  });
  it("the target is a fixed point", () => {
    const v = targetRgb(target);
    const out = applyUniform(v, target, { hue: 70, saturation: 60, lightness: 30 });
    out.forEach((x, i) => expect(x).toBeCloseTo(v[i] as number, 12));
  });
  it("100 % puts a colour near the target's hue on the target's chroma and tone", () => {
    const v = px([0.12, 0.01], 0.35);
    const out = applyUniform(v, target, { hue: 100, saturation: 100, lightness: 100 });
    const c = chromaOf(out);
    expect(Math.hypot(c[0], c[1])).toBeCloseTo(Math.hypot(target.c[0], target.c[1]), 12);
    expect(Math.atan2(c[1], c[0])).toBeCloseTo(Math.atan2(target.c[1], target.c[0]), 12);
    expect((out[0] + out[1] + out[2]) / 3).toBeCloseTo(target.tone, 12);
  });
  it("greys are kept, and nothing is NaN on black", () => {
    const grey = applyUniform([0.3, 0.3, 0.3], target, { hue: 100, saturation: 100, lightness: 0 });
    grey.forEach((x) => expect(x).toBeCloseTo(0.3, 12));
    applyUniform([0, 0, 0], target, { hue: 100, saturation: 100, lightness: 100 }).forEach((x) => expect(Number.isFinite(x)).toBe(true));
  });
  it("the guards", () => {
    expect(neutralGuard(0)).toBe(0);
    expect(neutralGuard(0.02)).toBe(1);
    expect(antipodeFade(HUE_FULL)).toBe(1);
    expect(antipodeFade(Math.PI)).toBe(0);
  });
});

describe("the colour's name", () => {
  it("names the picture's example", () => {
    const t = { hue_deg: 26, saturation: 9.7, lightness_ev: -0.03 };
    expect(colourName(t)).toBe("Orange, a little pale");
    expect(swatchLabel(t)).toBe("Target colour: orange, a little pale, hue 26 degrees, saturation 9.7, lightness minus 0.03 stops.");
  });
  it("says dark, light, pale and strong", () => {
    expect(colourName({ hue_deg: 0, saturation: 20, lightness_ev: -2 })).toBe("Dark red");
    expect(colourName({ hue_deg: 120, saturation: 40, lightness_ev: 2 })).toBe("Light green, strong");
    expect(colourName({ hue_deg: 240, saturation: 3, lightness_ev: 0 })).toBe("Blue, pale");
  });
});
