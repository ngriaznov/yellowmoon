import { describe, expect, it } from "vitest";
import { angleOf, clampToDisc, isCentered, mixHex, pointFromPointer, puckColor, strengthOf, WHEEL_GRADIENT, WHEEL_STOPS, wheelColorAt } from "./wheel";

const box = { left: 100, top: 200, width: 92, height: 92 };

describe("pointFromPointer", () => {
  it("is [0, 0] at the centre", () => {
    expect(pointFromPointer(146, 246, box, 37)).toEqual([0, 0]);
  });

  it("measures to the right and upwards", () => {
    const [a, b] = pointFromPointer(146 + 11.1, 246 - 14.8, box, 37);
    expect(a).toBeCloseTo(0.3, 12);
    expect(b).toBeCloseTo(0.4, 12);
  });

  it("pulls a point outside the disc back to its rim", () => {
    const [a, b] = pointFromPointer(400, 246, box, 37);
    expect(a).toBeCloseTo(1, 12);
    expect(b).toBeCloseTo(0, 12);
    expect(clampToDisc(3, 4)).toEqual([0.6, 0.8]);
    expect(clampToDisc(0.3, 0.4)).toEqual([0.3, 0.4]);
  });
});

describe("strength, angle and colour", () => {
  it("measures strength from 0 to 1", () => {
    expect(strengthOf([0, 0])).toBe(0);
    expect(strengthOf([0.6, 0.8])).toBeCloseTo(1, 12);
    expect(isCentered([0, 0])).toBe(true);
    expect(isCentered([0.01, 0])).toBe(false);
  });

  it("measures the angle clockwise from twelve o'clock", () => {
    expect(angleOf([0, 1])).toBe(0);
    expect(angleOf([1, 0])).toBe(90);
    expect(angleOf([0, -1])).toBe(180);
    expect(angleOf([-1, 0])).toBe(270);
  });

  it("reads the wheel gradient: orange at three o'clock, then clockwise through yellow", () => {
    expect(wheelColorAt(90)).toBe(WHEEL_STOPS[0].toLowerCase());
    expect(wheelColorAt(150)).toBe(WHEEL_STOPS[1].toLowerCase());
    expect(wheelColorAt(90 + 360)).toBe(WHEEL_STOPS[0].toLowerCase());
    expect(WHEEL_GRADIENT).toContain("from 90deg");
    expect(mixHex("#000000", "#ffffff", 0.5)).toBe("#808080");
  });

  it("paints the puck paper at the centre and the hue of its direction away from it", () => {
    expect(puckColor([0, 0])).toBe("#FFFFFF");
    expect(puckColor([1, 0])).toBe(WHEEL_STOPS[0].toLowerCase());
    expect(puckColor([0.05, 0]).toLowerCase()).not.toBe("#ffffff");
  });
});
