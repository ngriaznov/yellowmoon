import { describe, expect, it } from "vitest";
import {
  angleSlider,
  BRUSH_SIZE_RANGE,
  COLOR_WIDTH_RANGE,
  growField,
  growSlider,
  opacityField,
  opacitySlider,
  RADII_RANGE,
  radiusField,
  radiusSlider,
  refineField,
  refineSlider,
  sizeField,
  sizeSlider,
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

  it("Refine edges is percent, 0 to 100 % for 0 to 1; Grow is −100 to 100 for −1 to 1", () => {
    expect(refineSlider(0.5)).toBe(50);
    expect(refineSlider(0)).toBe(0);
    expect(refineField(50)).toBeCloseTo(0.5, 12);
    expect(refineField(140)).toBe(1);
    expect(growSlider(-1)).toBe(-100);
    expect(growSlider(0)).toBe(0);
    expect(growSlider(0.37)).toBe(37);
    expect(growSlider(-2)).toBe(-100);
    expect(growField(-100)).toBe(-1);
    expect(growField(25)).toBeCloseTo(0.25, 12);
    expect(growField(400)).toBe(1);
    for (const x of [-100, -37, -1, 0, 1, 64, 100]) expect(growSlider(growField(x))).toBe(x);
  });

  it("a Brush's Size is 0.001 + 0.249 (x / 100)² of the long edge and reaches both ends of the stored range", () => {
    expect(sizeField(0)).toBe(BRUSH_SIZE_RANGE[0]);
    expect(sizeField(100)).toBe(BRUSH_SIZE_RANGE[1]);
    expect(sizeField(50)).toBe(0.06325);
    // A small brush is within reach: a third of the slider is still under 0.03, and the first step from the smallest is a fraction of a pixel on a 6000 px photo.
    expect(sizeField(1) - sizeField(0)).toBeLessThan(0.0001);
    expect(sizeField(34)).toBeLessThan(0.03);
    for (const x of [0, 1, 2, 10, 17, 34, 35, 64, 99, 100]) expect(sizeSlider(sizeField(x))).toBe(x);
    // The default size, 0.03, is slider 34.
    expect(sizeSlider(0.03)).toBe(34);
    for (let x = 0; x <= 100; x++) {
      expect(sizeField(x)).toBeGreaterThanOrEqual(BRUSH_SIZE_RANGE[0]);
      expect(sizeField(x)).toBeLessThanOrEqual(BRUSH_SIZE_RANGE[1] + 1e-12);
      if (x > 0) expect(sizeField(x)).toBeGreaterThan(sizeField(x - 1));
    }
    // Held to the stored range, whatever is asked.
    expect(sizeSlider(0)).toBe(0);
    expect(sizeSlider(9)).toBe(100);
    expect(sizeField(-5)).toBe(BRUSH_SIZE_RANGE[0]);
    expect(sizeField(900)).toBe(BRUSH_SIZE_RANGE[1]);
  });
});
