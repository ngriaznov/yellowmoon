import { describe, expect, it } from "vitest";
import {
  angleLabel,
  canvasOf,
  constrainFrame,
  FIT_MARGIN_PX,
  fitInCell,
  fitInside,
  fitRatio,
  FULL_RECT,
  handlePoint,
  HANDLES,
  imageAspectOf,
  isFullRect,
  isInside,
  keepInside,
  MIN_CROP,
  moveRect,
  presetRatio,
  rectRatioPx,
  resizeRect,
  rotateRect90,
} from "./cropMath";
import type { Canvas, Rect } from "./cropMath";

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

// No border in photos (the owner's decision of 2026-10-22, docs/11-ui.md §Crop): the frame never shows an empty corner of the
// straightened photo. The checks below work the rule out on their own, from the brief's map, rather than through `isInside`.

/** A small fixed-seed generator, so the random cases are the same on every run. */
function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

const SONY = { width: 8640, height: 5760 }; // the mock's photos: a 3:2 canvas

/** Where each corner of `rect` samples the photo straightened by `deg` (q → c + L(q − c)), in canvas pixels. */
function samples(rect: Rect, deg: number, c: Canvas): Array<[number, number]> {
  const t = (deg * Math.PI) / 180;
  const [cos, sin] = [Math.cos(t), Math.sin(t)];
  const xs = [rect.x, rect.x + rect.width].map((v) => v * c.width);
  const ys = [rect.y, rect.y + rect.height].map((v) => v * c.height);
  return xs.flatMap((x) =>
    ys.map((y): [number, number] => {
      const [dx, dy] = [x - c.width / 2, y - c.height / 2];
      return [cos * dx + sin * dy + c.width / 2, -sin * dx + cos * dy + c.height / 2];
    }),
  );
}

/** The rule: on the canvas, and unless the angle is 0, every corner samples the photo at least 2 px from its edge (to 1e-6 px). */
function noEmptyCorner(rect: Rect, deg: number, c: Canvas): boolean {
  const tol = 1e-9;
  if (rect.x < -tol || rect.y < -tol || rect.x + rect.width > 1 + tol || rect.y + rect.height > 1 + tol) return false;
  if (Math.abs(deg) < 1e-9) return true;
  const m = FIT_MARGIN_PX - 1e-6;
  return samples(rect, deg, c).every(([x, y]) => x >= m && x <= c.width - m && y >= m && y <= c.height - m);
}

/** `rect` scaled by `k` about its own centre. */
const grown = (rect: Rect, k: number): Rect => {
  const [cx, cy] = [rect.x + rect.width / 2, rect.y + rect.height / 2];
  return { x: cx - (rect.width * k) / 2, y: cy - (rect.height * k) / 2, width: rect.width * k, height: rect.height * k };
};

const pxRatio = (rect: Rect, c: Canvas): number => (rect.width * c.width) / (rect.height * c.height);

describe("fitInside: no empty corner", () => {
  it("the canvas is the photo after its quarter turns", () => {
    expect(canvasOf(8640, 5760, 0)).toEqual({ width: 8640, height: 5760 });
    expect(canvasOf(8640, 5760, 90)).toEqual({ width: 5760, height: 8640 });
    expect(canvasOf(8640, 5760, 180)).toEqual({ width: 8640, height: 5760 });
    expect(canvasOf(8640, 5760, 270)).toEqual({ width: 5760, height: 8640 });
  });

  it("puts random frames inside at any angle, shape and turn, keeping their shape and shrinking no more than it must", () => {
    const next = lcg(20261022);
    const photos = [
      [8640, 5760],
      [6000, 4000],
      [4024, 6048],
      [3000, 3000],
      [7008, 2304],
    ];
    let shrunk = 0;
    for (let i = 0; i < 2000; i++) {
      const [w, h] = photos[i % photos.length]!;
      const turn = [0, 90, 180, 270][Math.floor(next() * 4)]!;
      const canvas = canvasOf(w!, h!, turn);
      const deg = (next() * 2 - 1) * 45;
      const width = 0.05 + 0.95 * next();
      const height = 0.05 + 0.95 * next();
      const rect = { x: next() * (1 - width), y: next() * (1 - height), width, height };
      const fitted = fitInside(rect, deg, canvas);
      const what = `${w}x${h} turned ${turn}, ${deg.toFixed(3)}°, ${JSON.stringify(rect)} -> ${JSON.stringify(fitted)}`;
      expect(noEmptyCorner(fitted, deg, canvas), what).toBe(true);
      expect(isInside(fitted, deg, canvas), what).toBe(true);
      expect(pxRatio(fitted, canvas) / pxRatio(rect, canvas), what).toBeCloseTo(1, 9);
      if (fitted.x === rect.x && fitted.y === rect.y && fitted.width === rect.width && fitted.height === rect.height) continue;
      shrunk += 1;
      // Maximal: 0.1 % larger about its own centre shows an empty corner.
      expect(noEmptyCorner(grown(fitted, 1.001), deg, canvas), what).toBe(false);
    }
    expect(shrunk).toBeGreaterThan(1000);
  });

  it("leaves a frame already inside alone", () => {
    const small = { x: 0.4, y: 0.4, width: 0.2, height: 0.2 };
    expect(fitInside(small, 5, SONY)).toEqual(small);
    expect(isInside(small, 5, SONY)).toBe(true);
  });

  it("at 0° leaves the frame alone: the whole photo stays exactly the whole photo", () => {
    const rect = { x: 0.1, y: 0.2, width: 0.5, height: 0.6 };
    expect(fitInside(rect, 0, SONY)).toEqual(rect);
    expect(fitInside(FULL_RECT, 0, SONY)).toEqual(FULL_RECT);
    expect(fitInside(FULL_RECT, 1e-10, SONY)).toEqual(FULL_RECT);
    expect(isFullRect(fitInside(FULL_RECT, -0, SONY))).toBe(true);
    expect(isInside(FULL_RECT, 0, SONY)).toBe(true);
    // At 0° there is no margin: only the canvas counts. A frame over its edge is clamped as the core clamps it, corner then size.
    expect(fitInside({ x: -0.1, y: 0.5, width: 0.5, height: 0.7 }, 0, SONY)).toEqual({ x: 0, y: 0.5, width: 0.5, height: 0.5 });
    expect(isInside({ x: 0.9, y: 0, width: 0.2, height: 0.5 }, 0, SONY)).toBe(false);
  });

  it("the whole photo at an angle becomes the largest frame of its shape, h = min(W / (r|cos| + |sin|), H / (r|sin| + |cos|))", () => {
    const { width: W, height: H } = SONY;
    const r = W / H;
    for (const deg of [0.3, -1.4, 2.5, 5, -10, 25, -45]) {
      const fitted = fitInside(FULL_RECT, deg, SONY);
      const t = (deg * Math.PI) / 180;
      const [c, s] = [Math.abs(Math.cos(t)), Math.abs(Math.sin(t))];
      const largest = (w: number, h: number): number => Math.min(w / (r * c + s), h / (r * s + c));
      const hPx = fitted.height * H;
      // Within the margin of the formula (it keeps 2 px from every edge), and exactly the formula on the canvas less the margin.
      expect(Math.abs(hPx - largest(W, H)), `${deg}°`).toBeLessThanOrEqual(2 * FIT_MARGIN_PX);
      expect(hPx, `${deg}°`).toBeCloseTo(largest(W - 2 * FIT_MARGIN_PX, H - 2 * FIT_MARGIN_PX), 6);
      expect(pxRatio(fitted, SONY)).toBeCloseTo(r, 9);
      expect(fitted.x + fitted.width / 2).toBeCloseTo(0.5, 12);
      expect(fitted.y + fitted.height / 2).toBeCloseTo(0.5, 12);
    }
  });

  it("a frame whose centre samples outside the photo becomes the largest frame of its shape about the canvas's centre", () => {
    // At 40° the canvas's top left corner is empty: a small square there has nothing of the photo under its centre.
    const corner = { x: 0, y: 0, width: 0.1, height: 0.15 };
    expect(samples({ x: 0.05, y: 0.075, width: 0, height: 0 }, 40, SONY)[0]![0]).toBeLessThan(0);
    const fitted = fitInside(corner, 40, SONY);
    expect(fitted.x + fitted.width / 2).toBeCloseTo(0.5, 12);
    expect(fitted.y + fitted.height / 2).toBeCloseTo(0.5, 12);
    expect(pxRatio(fitted, SONY)).toBeCloseTo(pxRatio(corner, SONY), 9);
    expect(noEmptyCorner(fitted, 40, SONY)).toBe(true);
    expect(noEmptyCorner(grown(fitted, 1.001), 40, SONY)).toBe(false);
    // The same as fitting the centred frame of that shape that fills the canvas.
    const filling = fitRatio(FULL_RECT, pxRatio(corner, SONY), SONY.width / SONY.height);
    const direct = fitInside(filling, 40, SONY);
    expect(fitted.width).toBeCloseTo(direct.width, 12);
    expect(fitted.height).toBeCloseTo(direct.height, 12);
  });
});

describe("constrainFrame: a drag stays inside", () => {
  const DEG = 5;
  const whole = fitInside(FULL_RECT, DEG, SONY);
  const small = { x: 0.3, y: 0.3, width: 0.3, height: 0.3 };

  it("passes a frame that is inside as it is, and changes nothing at 0°", () => {
    const moved = { ...small, x: 0.35 };
    expect(constrainFrame(small, moved, DEG, SONY, false)).toBe(moved);
    for (const to of [FULL_RECT, { x: 0, y: 0, width: 0.4, height: 1 }, { x: 0.6, y: 0.7, width: 0.4, height: 0.3 }]) {
      expect(constrainFrame(small, to, 0, SONY, false)).toBe(to);
      expect(constrainFrame(small, to, 0, SONY, true)).toBe(to);
    }
  });

  it("a move past the turned photo's edge stops where the frame touches it", () => {
    const asked = moveRect(small, 0.5, 0);
    expect(noEmptyCorner(asked, DEG, SONY)).toBe(false);
    const got = constrainFrame(small, asked, DEG, SONY, false);
    expect(got.width).toBe(small.width);
    expect(got.height).toBe(small.height);
    expect(got.x).toBeGreaterThan(small.x);
    expect(got.x).toBeLessThan(asked.x);
    expect(noEmptyCorner(got, DEG, SONY)).toBe(true);
    // It touches: a thousandth further right shows an empty corner.
    expect(noEmptyCorner({ ...got, x: got.x + 0.001 }, DEG, SONY)).toBe(false);
  });

  it("a move into a slanted edge slides along it rather than stopping", () => {
    // Up and to the right, towards the top right corner: across goes as far as it can, then up as far as it can.
    const asked = moveRect(small, 0.5, -0.5);
    const got = constrainFrame(small, asked, DEG, SONY, false);
    expect(noEmptyCorner(got, DEG, SONY)).toBe(true);
    expect(got.x).toBeGreaterThan(small.x);
    expect(got.y).toBeLessThan(small.y);
  });

  it("the largest frame can't move or grow, and a handle pulled out of it stays on it", () => {
    for (const [dx, dy] of [
      [0.01, 0],
      [-0.01, 0],
      [0, 0.01],
      [0, -0.01],
    ] as const) {
      const got = constrainFrame(whole, moveRect(whole, dx, dy), DEG, SONY, true);
      expect(got.x).toBeCloseTo(whole.x, 9);
      expect(got.y).toBeCloseTo(whole.y, 9);
    }
    const out = resizeRect(whole, "se", { x: 1, y: 1 }, null, SONY.width / SONY.height);
    const got = constrainFrame(whole, out, DEG, SONY, false);
    expect(got.width).toBeCloseTo(whole.width, 9);
    expect(got.height).toBeCloseTo(whole.height, 9);
  });

  it("a locked handle keeps the shape and the opposite corner, and stops at the edge", () => {
    const ratio = 1.5;
    const aspect = SONY.width / SONY.height;
    const start = fitRatio({ x: 0.25, y: 0.25, width: 0.4, height: 0.4 }, ratio, aspect);
    const asked = resizeRect(start, "se", { x: 1, y: 1 }, ratio, aspect);
    expect(noEmptyCorner(asked, DEG, SONY)).toBe(false);
    const got = constrainFrame(start, asked, DEG, SONY, true);
    expect(noEmptyCorner(got, DEG, SONY)).toBe(true);
    expect(rectRatioPx(got, aspect)).toBeCloseTo(ratio, 9);
    expect(got.x).toBeCloseTo(start.x, 12);
    expect(got.y).toBeCloseTo(start.y, 12);
    expect(got.width).toBeGreaterThan(start.width);
    expect(noEmptyCorner({ ...got, width: got.width * 1.001, height: got.height * 1.001 }, DEG, SONY)).toBe(false);
  });

  it("keeps every random drag inside", () => {
    const next = lcg(7);
    const aspect = SONY.width / SONY.height;
    for (let i = 0; i < 500; i++) {
      const deg = (next() * 2 - 1) * 45;
      const from = fitInside({ x: next() * 0.5, y: next() * 0.5, width: 0.1 + next() * 0.5, height: 0.1 + next() * 0.5 }, deg, SONY);
      const handle = HANDLES[Math.floor(next() * HANDLES.length)]!;
      const to = { x: next() * 1.4 - 0.2, y: next() * 1.4 - 0.2 };
      const locked = next() < 0.5;
      const asked = next() < 0.3 ? moveRect(from, to.x - 0.5, to.y - 0.5) : resizeRect(from, handle, to, locked ? rectRatioPx(from, aspect) : null, aspect);
      const got = constrainFrame(from, asked, deg, SONY, locked);
      expect(noEmptyCorner(got, deg, SONY), `${deg}° ${handle} ${JSON.stringify(asked)} -> ${JSON.stringify(got)}`).toBe(true);
      if (locked) expect(rectRatioPx(got, aspect)).toBeCloseTo(rectRatioPx(from, aspect), 6);
    }
  });
});
