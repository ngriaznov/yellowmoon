import { describe, expect, it } from "vitest";
import {
  angleSlider,
  COLOR_WIDTH_RANGE,
  opacityField,
  opacitySlider,
  RADII_RANGE,
  radiusField,
  radiusSlider,
  unitField,
  unitSlider,
  widthField,
  widthSlider,
  wrapDegrees,
} from "./maskScales";

describe("the scales of a mask's sliders (docs/algorithms/06 §4.5)", () => {
  it("Feather, Softness, Low and High: 0 to 100 are 0 to 1", () => {
    expect(unitSlider(0)).toBe(0);
    expect(unitSlider(0.2)).toBe(20);
    expect(unitSlider(0.45)).toBe(45);
    expect(unitSlider(1)).toBe(100);
    expect(unitField(0)).toBe(0);
    expect(unitField(35)).toBeCloseTo(0.35, 12);
    expect(unitField(100)).toBe(1);
    // Outside the slider is held to it.
    expect(unitField(140)).toBe(1);
    expect(unitField(-5)).toBe(0);
  });

  it("Opacity shows percent, 0 to 100 %, and writes 0 to 1", () => {
    expect(opacitySlider(1)).toBe(100);
    expect(opacitySlider(0.6)).toBe(60);
    expect(opacityField(60)).toBeCloseTo(0.6, 12);
    expect(opacityField(100)).toBe(1);
  });

  it("a Colour range's Width is 0.005 + 0.295 (x / 100)² and reaches both ends of the stored range", () => {
    expect(widthField(0)).toBe(COLOR_WIDTH_RANGE[0]);
    expect(widthField(100)).toBe(COLOR_WIDTH_RANGE[1]);
    expect(widthField(40)).toBe(0.0522);
    for (const x of [0, 1, 17, 39, 40, 64, 99, 100]) expect(widthSlider(widthField(x))).toBe(x);
    // The default width, 0.05, is slider 39.
    expect(widthSlider(0.05)).toBe(39);
    // Every slider position is inside the stored range (0.005 to 0.3).
    for (let x = 0; x <= 100; x++) {
      expect(widthField(x)).toBeGreaterThanOrEqual(COLOR_WIDTH_RANGE[0]);
      expect(widthField(x)).toBeLessThanOrEqual(COLOR_WIDTH_RANGE[1] + 1e-12);
    }
  });

  it("a Radial's Width and Height are the oval's full size as a percent of the long edge, 1 to 200", () => {
    expect(radiusSlider(0.25)).toBe(50);
    expect(radiusSlider(0.18)).toBe(36);
    expect(radiusSlider(1)).toBe(200);
    // The smallest radius the field has (0.002) is slider 1; none is shown as 0.
    expect(radiusSlider(RADII_RANGE[0])).toBe(1);
    expect(radiusField(50)).toBeCloseTo(0.25, 12);
    expect(radiusField(200)).toBe(1);
    expect(radiusField(1)).toBeCloseTo(0.005, 12);
    for (const x of [1, 2, 36, 50, 100, 200]) expect(radiusSlider(radiusField(x))).toBe(x);
    // A radius is held to the stored range, whatever is asked.
    expect(radiusField(0)).toBe(RADII_RANGE[0]);
    expect(radiusField(900)).toBe(RADII_RANGE[1]);
  });

  it("an angle is whole degrees from −180 to 180, and one past the end comes round", () => {
    expect(angleSlider(12.4)).toBe(12);
    expect(angleSlider(-200)).toBe(-180);
    expect(wrapDegrees(190)).toBe(-170);
    expect(wrapDegrees(-190)).toBe(170);
    expect(wrapDegrees(540)).toBe(180);
    expect(wrapDegrees(30)).toBe(30);
    expect(wrapDegrees(-180)).toBe(-180);
  });
});
