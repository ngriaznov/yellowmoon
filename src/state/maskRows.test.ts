import { describe, expect, it } from "vitest";
import type { MaskItem } from "../api/types";
import { maskRowLabel, maskRowName, maskSubtitle, widthSlider } from "./maskRows";

const base = { mode: "add", invert: false, enabled: true, opacity: 1 } as const;
const edge = { feather: 0, grow: 0, refine: 0.5 } as const;

describe("a mask row's words (docs/23 §7.6)", () => {
  it("names the kinds and gives the accessible name 'Radial mask, Feather 45'", () => {
    const radial: MaskItem = { ...base, kind: "radial", center: [0.5, 0.5], radii: [0.25, 0.18], angle_deg: 0, feather: 0.45 };
    expect(maskRowName(radial)).toBe("Radial");
    expect(maskSubtitle(radial)).toBe("Feather 45");
    expect(maskRowLabel(radial)).toBe("Radial mask, Feather 45");
  });

  it("says what an Object row has: points, a box, both, or what to do", () => {
    const object = (points: Array<[number, number, boolean]>, box: [number, number, number, number] | null): MaskItem => ({ ...base, ...edge, kind: "object", points, box });
    expect(maskSubtitle(object([], null))).toBe("Click the object on this photo");
    expect(maskSubtitle(object([[0.1, 0.2, true]], null))).toBe("1 point");
    expect(maskSubtitle(object([[0.1, 0.2, true], [0.3, 0.4, false]], null))).toBe("2 points");
    expect(maskSubtitle(object([], [0.1, 0.1, 0.5, 0.5]))).toBe("Box");
    expect(maskSubtitle(object([[0.1, 0.2, true], [0.3, 0.4, true]], [0.1, 0.1, 0.5, 0.5]))).toBe("Box and 2 points");
    expect(maskRowLabel(object([[0.1, 0.2, true], [0.3, 0.4, false]], null))).toBe("Object mask, 2 points");
  });

  it("gives the other kinds' subtitles", () => {
    expect(maskSubtitle({ ...base, ...edge, kind: "subject" })).toBe("Foreground");
    expect(maskSubtitle({ ...base, ...edge, kind: "people" })).toBe("People");
    expect(maskSubtitle({ ...base, ...edge, kind: "background", level: 0.5, softness: 0.5 })).toBe("Level 50");
    expect(maskSubtitle({ ...base, kind: "linear", center: [0.5, 0.5], angle_deg: 0, feather: 0.2 })).toBe("Feather 20");
    expect(maskSubtitle({ ...base, ...edge, kind: "brush", strokes: [] })).toBe("Paint on the photo");
    const stroke = { size: 0.03, hardness: 0.5, erase: false, points: [[0.1, 0.1]] as Array<[number, number]> };
    expect(maskSubtitle({ ...base, ...edge, kind: "brush", strokes: [stroke] })).toBe("1 stroke");
    expect(maskSubtitle({ ...base, ...edge, kind: "brush", strokes: [stroke, stroke, stroke] })).toBe("3 strokes");
    expect(maskSubtitle({ ...base, kind: "brightness_range", low: 0.35, high: 1, softness: 0.5 })).toBe("35 to 100");
    expect(maskRowName({ ...base, kind: "color_range", center: [0, 0], width: 0.05, softness: 0.5 })).toBe("Colour range");
  });

  it("reads a Colour range's width as its slider shows it (the inverse of 0.005 + 0.295 (x / 100)²)", () => {
    expect(widthSlider(0.005)).toBe(0);
    expect(widthSlider(0.3)).toBe(100);
    expect(widthSlider(0.005 + 0.295 * 0.16)).toBe(40);
    expect(maskSubtitle({ ...base, kind: "color_range", center: [0, 0], width: 0.005 + 0.295 * 0.16, softness: 0.5 })).toBe("Width 40");
  });
});
