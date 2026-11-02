import { describe, expect, it } from "vitest";
import {
  areaHeight,
  DEFAULT_FRACTION,
  DIVIDER_H,
  editorBounds,
  EXPANDED_FRACTION,
  fractionOf,
  HEADER_H,
  layoutOf,
  MAX_FRACTION,
  MIN_FRACTION,
  MIN_PHOTO_H,
  MIN_WINDOW_H,
} from "./layout";

/** The photo cell's width: 3:2 on the row's height, or the column's width when that is less. */
const cellWidth = (viewer: number, column: number): number => Math.min(column, viewer * 1.5);

describe("the node view's geometry (docs/23 §7.1)", () => {
  it("at 1440 × 900: a 75 px header, a photo of 666 × 444, a 10 px divider and an editor of 371 px (a 44 px toolbar and a 327 px canvas)", () => {
    const l = layoutOf(900, DEFAULT_FRACTION);
    expect(HEADER_H).toBe(75);
    expect(l.area).toBe(825);
    expect(l.editor).toBe(371);
    expect(l.editor).toBe(Math.round(0.45 * (900 - 75)));
    expect(l.viewer).toBe(444);
    expect(HEADER_H + l.viewer + DIVIDER_H + l.editor).toBe(900);
    expect(cellWidth(l.viewer, 1040)).toBe(666);
    expect(l.editor - 44).toBe(327);
  });

  it("Expand makes the editor 66 % of the height under the header: 545 px, and a photo of 405 × 270", () => {
    const l = layoutOf(900, EXPANDED_FRACTION);
    expect(l.editor).toBe(545);
    expect(l.viewer).toBe(270);
    expect(cellWidth(l.viewer, 1040)).toBe(405);
  });

  it("at 1728 × 1117: a photo of about 844 × 563 and an editor 1328 × 469", () => {
    const l = layoutOf(1117, DEFAULT_FRACTION);
    expect(l.editor).toBe(469);
    expect(l.viewer).toBe(563);
    expect(cellWidth(l.viewer, 1728 - 400)).toBeCloseTo(844.5, 5);
    expect(1728 - 400).toBe(1328);
  });

  it("a window shorter than 760 px keeps the height of the screen, which is 760 px", () => {
    expect(areaHeight(600)).toBe(MIN_WINDOW_H - HEADER_H);
    expect(layoutOf(600, DEFAULT_FRACTION)).toEqual(layoutOf(760, DEFAULT_FRACTION));
  });

  it("the divider may be anywhere between 25 % and 70 %, and the photo keeps at least 200 px", () => {
    for (const height of [760, 800, 900, 1000, 1117, 1440, 2000]) {
      const area = areaHeight(height);
      const { min, max } = editorBounds(area);
      expect(min, `${height}: least`).toBe(Math.round(MIN_FRACTION * area));
      expect(max, `${height}: most`).toBeLessThanOrEqual(Math.round(MAX_FRACTION * area));
      expect(area - DIVIDER_H - max, `${height}: the photo at the most`).toBeGreaterThanOrEqual(MIN_PHOTO_H);
      // Asked for more or less than that, the editor stops at the bounds.
      expect(layoutOf(height, 0).editor).toBe(min);
      expect(layoutOf(height, 1).editor).toBe(max);
      for (const fraction of [0, 0.25, 0.3, 0.45, 0.6, 0.66, 0.7, 0.9, 1]) {
        const l = layoutOf(height, fraction);
        expect(l.viewer, `${height} at ${fraction}`).toBeGreaterThanOrEqual(MIN_PHOTO_H);
        expect(l.editor + DIVIDER_H + l.viewer).toBe(area);
      }
    }
  });

  it("at the smallest window the 70 % bound gives way to the photo's 200 px, and Expand still fits", () => {
    const area = areaHeight(760);
    // 70 % of 685 is 479, and the photo's 200 px leaves 475.
    expect(Math.round(MAX_FRACTION * area)).toBe(479);
    expect(editorBounds(area).max).toBe(area - DIVIDER_H - MIN_PHOTO_H);
    expect(layoutOf(760, EXPANDED_FRACTION).viewer).toBeGreaterThanOrEqual(MIN_PHOTO_H);
  });

  it("keeps the proportion, not the pixels: the share of an editor of this height gives the same height back", () => {
    for (const height of [760, 900, 1117]) {
      const area = areaHeight(height);
      for (let editor = editorBounds(area).min; editor <= editorBounds(area).max; editor += 7) {
        expect(layoutOf(height, fractionOf(editor, area)).editor).toBe(editor);
      }
    }
    // The same share in a taller window is a taller editor.
    const share = fractionOf(layoutOf(900, 0.4).editor, areaHeight(900));
    expect(layoutOf(1117, share).editor).toBeGreaterThan(layoutOf(900, share).editor);
  });
});
