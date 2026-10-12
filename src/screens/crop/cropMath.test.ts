import { describe, expect, it } from "vitest";
import {
  angleLabel,
  fitInCell,
  fitRatio,
  FULL_RECT,
  handlePoint,
  HANDLES,
  imageAspectOf,
  isFullRect,
  keepInside,
  MIN_CROP,
  moveRect,
  presetRatio,
  rectRatioPx,
  resizeRect,
  rotateRect90,
} from "./cropMath";
import type { Rect } from "./cropMath";

const ASPECT = 1.5; // an 8640 × 5760 photo

const inside = (r: Rect): boolean => r.x >= -1e-9 && r.y >= -1e-9 && r.x + r.width <= 1 + 1e-9 && r.y + r.height <= 1 + 1e-9;

describe("image geometry", () => {
  it("swaps the sides for a quarter turn", () => {
    expect(imageAspectOf(8640, 5760, 0)).toBeCloseTo(1.5);
    expect(imageAspectOf(8640, 5760, 90)).toBeCloseTo(2 / 3);
    expect(imageAspectOf(8640, 5760, 180)).toBeCloseTo(1.5);
    expect(imageAspectOf(8640, 5760, 270)).toBeCloseTo(2 / 3);
  });

  it("letterboxes a picture in the 3:2 cell", () => {
    expect(fitInCell(1.5, 1.5)).toEqual({ x: 0, y: 0, width: 1, height: 1 });
    const wide = fitInCell(1.5, 3);
    expect(wide.width).toBe(1);
    expect(wide.height).toBeCloseTo(0.5);
    expect(wide.y).toBeCloseTo(0.25);
    const tall = fitInCell(1.5, 2 / 3);
    expect(tall.height).toBe(1);
    expect(tall.width).toBeCloseTo(4 / 9);
    expect(tall.x).toBeCloseTo((1 - 4 / 9) / 2);
  });

  it("recognises the whole photo", () => {
    expect(isFullRect(FULL_RECT)).toBe(true);
    expect(isFullRect({ x: 0.0004, y: 0, width: 0.9996, height: 1 })).toBe(true);
    expect(isFullRect({ x: 0.1, y: 0, width: 0.9, height: 1 })).toBe(false);
  });

  it("labels the straighten angle with a true minus sign", () => {
    expect(angleLabel(0)).toBe("0°");
    expect(angleLabel(1.5)).toBe("+1.5°");
    expect(angleLabel(-12)).toBe("−12.0°");
    expect(angleLabel(0.01)).toBe("0°");
  });
});

describe("aspect presets", () => {
  it("Original is the photo's own shape, Free is none", () => {
    expect(presetRatio("original", ASPECT, true)).toBe(ASPECT);
    expect(presetRatio("free", ASPECT, true)).toBeNull();
    expect(presetRatio("square", ASPECT, true)).toBe(1);
  });

  it("4:5 and 16:9 follow the shape of the crop: landscape stays landscape", () => {
    expect(presetRatio("four_by_five", ASPECT, true)).toBeCloseTo(1.25);
    expect(presetRatio("four_by_five", ASPECT, false)).toBeCloseTo(0.8);
    expect(presetRatio("sixteen_by_nine", ASPECT, true)).toBeCloseTo(16 / 9);
    expect(presetRatio("sixteen_by_nine", ASPECT, false)).toBeCloseTo(9 / 16);
  });

  it("fits the largest rectangle of a ratio inside another, centred", () => {
    const square = fitRatio(FULL_RECT, 1, ASPECT);
    expect(rectRatioPx(square, ASPECT)).toBeCloseTo(1);
    expect(square.height).toBeCloseTo(1);
    expect(square.width).toBeCloseTo(1 / 1.5);
    expect(square.x).toBeCloseTo((1 - 1 / 1.5) / 2);
    const wide = fitRatio(FULL_RECT, 16 / 9, ASPECT);
    expect(rectRatioPx(wide, ASPECT)).toBeCloseTo(16 / 9);
    expect(wide.width).toBeCloseTo(1);
    expect(inside(wide)).toBe(true);
  });
});

describe("moving and turning", () => {
  it("moves a rectangle and stops at the edges", () => {
    const r = { x: 0.2, y: 0.2, width: 0.4, height: 0.4 };
    expect(moveRect(r, 0.1, -0.1)).toEqual({ x: 0.30000000000000004, y: 0.1, width: 0.4, height: 0.4 });
    expect(moveRect(r, 5, 5)).toEqual({ x: 0.6, y: 0.6, width: 0.4, height: 0.4 });
    expect(moveRect(r, -5, -5)).toEqual({ x: 0, y: 0, width: 0.4, height: 0.4 });
  });

  it("keeps an oversized rectangle inside by sliding it", () => {
    expect(keepInside({ x: 0.9, y: -0.2, width: 0.3, height: 0.3 })).toEqual({ x: 0.7, y: 0, width: 0.3, height: 0.3 });
  });

  it("rotates a rectangle with the picture, a quarter turn at a time", () => {
    const r = { x: 0.1, y: 0.2, width: 0.3, height: 0.4 };
    const once = rotateRect90(r);
    expect(once.x).toBeCloseTo(0.4);
    expect(once.y).toBeCloseTo(0.1);
    expect(once.width).toBeCloseTo(0.4);
    expect(once.height).toBeCloseTo(0.3);
    const full = rotateRect90(rotateRect90(rotateRect90(rotateRect90(r))));
    expect(full.x).toBeCloseTo(r.x);
    expect(full.y).toBeCloseTo(r.y);
    expect(full.width).toBeCloseTo(r.width);
    expect(full.height).toBeCloseTo(r.height);
  });

  it("puts each handle on the frame", () => {
    const r = { x: 0.1, y: 0.2, width: 0.4, height: 0.2 };
    expect(handlePoint(r, "nw")).toEqual({ x: 0.1, y: 0.2 });
    expect(handlePoint(r, "se")).toEqual({ x: 0.5, y: 0.4 });
    expect(handlePoint(r, "n")).toEqual({ x: 0.30000000000000004, y: 0.2 });
    expect(handlePoint(r, "w")).toEqual({ x: 0.1, y: 0.30000000000000004 });
  });
});

describe("resizing a free frame", () => {
  const start = { x: 0.25, y: 0.25, width: 0.5, height: 0.5 };

  it("a corner moves its two edges and leaves the opposite corner alone", () => {
    const r = resizeRect(start, "se", { x: 0.9, y: 0.8 }, null, ASPECT);
    expect(r).toEqual({ x: 0.25, y: 0.25, width: 0.65, height: 0.55 });
    const nw = resizeRect(start, "nw", { x: 0.1, y: 0.05 }, null, ASPECT);
    expect(nw.x).toBeCloseTo(0.1);
    expect(nw.y).toBeCloseTo(0.05);
    expect(nw.x + nw.width).toBeCloseTo(0.75);
    expect(nw.y + nw.height).toBeCloseTo(0.75);
  });

  it("an edge moves only its own side", () => {
    const e = resizeRect(start, "e", { x: 0.95, y: 0.1 }, null, ASPECT);
    expect(e).toEqual({ x: 0.25, y: 0.25, width: 0.7, height: 0.5 });
    const n = resizeRect(start, "n", { x: 0.1, y: 0.05 }, null, ASPECT);
    expect(n.x).toBe(0.25);
    expect(n.y).toBeCloseTo(0.05);
    expect(n.height).toBeCloseTo(0.7);
  });

  it("never leaves the photo and never collapses", () => {
    const out = resizeRect(start, "se", { x: 4, y: 4 }, null, ASPECT);
    expect(inside(out)).toBe(true);
    const tiny = resizeRect(start, "se", { x: 0, y: 0 }, null, ASPECT);
    expect(tiny.width).toBeCloseTo(MIN_CROP);
    expect(tiny.height).toBeCloseTo(MIN_CROP);
  });
});

describe("resizing a locked frame", () => {
  const start = { x: 0.2, y: 0.2, width: 0.6, height: 0.4 }; // 3:2 in pixels for an aspect of 1
  const cases: Array<[string, number, number]> = [
    ["3:2 on a 3:2 photo", 1.5, 1.5],
    ["square on a 3:2 photo", 1, 1.5],
    ["16:9 on a portrait photo", 16 / 9, 2 / 3],
  ];

  it.each(cases)("keeps the ratio from every handle: %s", (_name, ratioPx, aspect) => {
    const base = fitRatio({ x: 0.1, y: 0.1, width: 0.8, height: 0.8 }, ratioPx, aspect);
    for (const handle of HANDLES) {
      for (const to of [{ x: 0.05, y: 0.05 }, { x: 0.95, y: 0.95 }, { x: 0.5, y: 0.5 }, { x: 0.7, y: 0.2 }]) {
        const out = resizeRect(base, handle, to, ratioPx, aspect);
        expect(inside(out), `${handle} to ${to.x},${to.y}`).toBe(true);
        expect(rectRatioPx(out, aspect), `${handle} to ${to.x},${to.y}`).toBeCloseTo(ratioPx, 5);
        expect(out.width).toBeGreaterThanOrEqual(MIN_CROP - 1e-9);
        expect(out.height).toBeGreaterThanOrEqual(MIN_CROP - 1e-9);
      }
    }
  });

  it("a corner keeps the opposite corner still", () => {
    const out = resizeRect(start, "se", { x: 0.9, y: 0.9 }, 1.5, 1.5);
    expect(out.x).toBeCloseTo(start.x);
    expect(out.y).toBeCloseTo(start.y);
  });

  it("an edge grows the other side about the middle", () => {
    const out = resizeRect(start, "e", { x: 0.9, y: 0.5 }, 1.5, 1.5);
    expect(out.x).toBeCloseTo(start.x);
    expect(out.y + out.height / 2).toBeCloseTo(start.y + start.height / 2);
    expect(out.width).toBeCloseTo(0.7);
  });
});
