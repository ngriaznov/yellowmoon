/** The mask view on the mock's flat scene (`maskView.ts`): the shapes the page tests and the reference screenshots rely on. */
import { describe, expect, it } from "vitest";
import type { MaskItem, MaskSettings } from "../api/types";
import { backgroundValue, BLUR_RAMP_MIN, DEPTH_PICK_MARGIN, depthValue, depthViewSvg, MOCK_DEPTH, pickDepthFar, hasBackgroundRow, MASK_ACCENT, maskViewSvg, masksOfView, MOCK_BLUR, OVERLAY_WEIGHT } from "./maskView";
import type { DevelopSettings } from "../api/types";

const BOX = { x: 100, y: 50, width: 600, height: 400 };
const row = (more: Record<string, unknown>): MaskItem => ({ mode: "add", invert: false, enabled: true, opacity: 1, ...more }) as MaskItem;
const radial = (more: Record<string, unknown> = {}): MaskItem => row({ kind: "radial", center: [0.5, 0.5], radii: [0.3, 0.2], angle_deg: 0, feather: 0.4, ...more });
const masks = (items: MaskItem[], invert = false): MaskSettings => ({ invert, items });

describe("maskViewSvg", () => {
  it("tints the accent at 0.55 over the picture's rectangle through the mask, for Overlay", () => {
    const svg = maskViewSvg("overlay", masks([radial()]), BOX);
    expect(svg).toContain('data-mask-view="overlay"');
    expect(svg).toContain(`fill="${MASK_ACCENT}" fill-opacity="${OVERLAY_WEIGHT}"`);
    expect(svg).toContain('mask="url(#mvmask)"');
    expect(svg).toContain('x="100" y="50" width="600" height="400"');
    // The Radial's centre is the middle of the picture; its radii are of the long edge (600): 0.3 × 600 and 0.2 × 600.
    expect(svg).toContain("translate(400 250)");
    expect(svg).toContain("scale(180 120)");
  });

  it("paints the mask white over black, for Mask only", () => {
    const svg = maskViewSvg("mask_only", masks([radial()]), BOX);
    expect(svg).toContain('data-mask-view="mask_only"');
    expect(svg).toContain('fill="#000"');
    expect(svg).toContain('fill="#fff" mask="url(#mvmask)"');
  });

  it("shows 1 everywhere for a node with no rows, and 0 when the node is inverted; rows that are off count as absent", () => {
    expect(maskViewSvg("mask_only", masks([]), BOX)).toMatch(/<mask [^>]*><rect [^>]*fill="#fff"\/><\/mask>/);
    expect(maskViewSvg("mask_only", masks([], true), BOX)).toMatch(/<mask [^>]*><rect [^>]*fill="#000"\/><\/mask>/);
    expect(maskViewSvg("mask_only", masks([radial({ enabled: false })]), BOX)).toMatch(/<mask [^>]*><rect [^>]*fill="#fff"\/><\/mask>/);
  });

  it("draws a Subtract in black over what came before, a lone Subtract over the whole photo, and the node's Invert as the same picture inverted", () => {
    const both = maskViewSvg("mask_only", masks([radial(), radial({ mode: "subtract", radii: [0.1, 0.1] })]), BOX);
    expect(both).toContain('stop-color="#fff"');
    expect(both).toContain('stop-color="#000"');
    const lone = maskViewSvg("mask_only", masks([radial({ mode: "subtract" })]), BOX);
    expect(lone).toMatch(/<mask [^>]*><rect [^>]*fill="#fff"\/>/);
    expect(maskViewSvg("mask_only", masks([radial()], true), BOX)).toContain('<g filter="url(#mvinvert)">');
  });

  it("draws a Linear as a gradient across the photo, and every other kind as nothing", () => {
    const linear = maskViewSvg("overlay", masks([row({ kind: "linear", center: [0.5, 0.5], angle_deg: 0, feather: 0.2 })]), BOX);
    expect(linear).toContain("linearGradient");
    const brightness = maskViewSvg("overlay", masks([row({ kind: "brightness_range", low: 0.2, high: 0.8, softness: 0.4 })]), BOX);
    expect(brightness).not.toContain("Gradient");
  });

  it("draws an Object as the flat scene's mountain, with a hole at each negative point, cut to the box, and nothing for a row with no point and no box", () => {
    const object = (more: Record<string, unknown>) => maskViewSvg("mask_only", masks([row({ kind: "object", points: [], box: null, feather: 0, grow: 0, refine: 0.5, ...more })]), BOX);
    // A row that has asked for nothing selects nothing.
    expect(object({})).not.toContain("data-object-row");
    const hill = object({ points: [[0.5, 0.7, true]] });
    expect(hill).toContain("data-object-row");
    // The far hill is the half ellipse of the scene: centre 0.82 down, a radius 0.6 of the width across and 0.42 of the height up (box 600 × 400 at 100, 50).
    expect(hill).toContain('<ellipse cx="400" cy="378"');
    expect(hill).toContain('rx="360" ry="168"');
    // The near band is in front of it.
    expect(hill).toContain('<rect x="100" y="346" width="600" height="104" fill="#000"/>');
    expect(hill).not.toContain('r="24"');
    // A negative point takes a round hole out of it (0.06 of the long edge); a positive one adds nothing to the picture.
    const holed = object({ points: [[0.5, 0.7, true], [0.3, 0.6, false]] });
    expect(holed).toContain('<circle cx="280" cy="290" r="36" fill="#000"/>');
    // A box cuts the row to its rectangle.
    expect(object({ box: [0.25, 0.5, 0.75, 0.9] })).toContain("mvo0b");
    // Invert and Opacity are the row's as for every raster row.
    expect(object({ points: [[0.5, 0.7, true]], invert: true })).toContain("mvo0i");
    expect(object({ points: [[0.5, 0.7, true]], opacity: 0.4 })).toContain('fill-opacity="0.4"');
  });

  it("draws a Linear as the core's Linear: at an angle of 0 the line is horizontal and the gradient runs down, at 90° the line is vertical", () => {
    const gradient = (angle: number): number[] => {
      const svg = maskViewSvg("mask_only", masks([row({ kind: "linear", center: [0.5, 0.5], angle_deg: angle, feather: 0.2 })]), BOX);
      const [, x1, y1, x2, y2] = /x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"/.exec(svg) ?? [];
      return [x1, y1, x2, y2].map(Number);
    };
    // 0.2 of the long edge (600) across: 120 px, 60 either side of the centre (400, 250); 1 above the start line, 0 below the end line.
    const [x1, y1, x2, y2] = gradient(0);
    expect([x1, x2]).toEqual([400, 400]);
    expect(y1).toBeCloseTo(190, 6);
    expect(y2).toBeCloseTo(310, 6);
    const [a1, b1, a2, b2] = gradient(90);
    expect([b1, b2]).toEqual([250, 250]);
    // At 90° the normal points to the left of the picture (−sin 90°, cos 90°): the gradient starts to the right of the centre.
    expect(a1).toBeCloseTo(460, 6);
    expect(a2).toBeCloseTo(340, 6);
  });

  it("carries the shapes of the photo frame to a turned or cropped picture by the map it is given", () => {
    const upright = maskViewSvg("mask_only", masks([radial()]), BOX, { width: 600, height: 400, matrix: [1, 0, 0, 1, 100, 50] });
    // The shapes are drawn on the photo's own pixels (600 × 400) and the group places them at the picture.
    expect(upright).toContain('<g transform="matrix(1 0 0 1 100 50)">');
    expect(upright).toContain("translate(300 200)");
    expect(upright).toContain("scale(180 120)");
    // The picture's rectangle still clips the mask.
    expect(upright).toContain('<mask id="mvmask" maskUnits="userSpaceOnUse" x="100" y="50" width="600" height="400">');
    const turned = maskViewSvg("mask_only", masks([radial()]), BOX, { width: 600, height: 400, matrix: [0, 1, -1, 0, 700, 50] });
    expect(turned).toContain('<g transform="matrix(0 1 -1 0 700 50)">');
  });
});

describe("a Brush row on the flat scene (task N-14)", () => {
  const stroke = (more: Record<string, unknown> = {}) => ({ size: 0.05, hardness: 1, erase: false, points: [[0.2, 0.3], [0.8, 0.3]], ...more });
  const brush = (strokes: unknown[], more: Record<string, unknown> = {}): MaskItem =>
    row({ kind: "brush", strokes, feather: 0, grow: 0, refine: 0, ...more });
  const svg = (items: MaskItem[], invert = false): string => maskViewSvg("mask_only", masks(items, invert), BOX);

  it("draws a stroke as a round-capped line as wide as the brush, along its points", () => {
    const out = svg([brush([stroke()])]);
    // The long edge is 600: a size of 0.05 is 30 px; the points are fractions of the picture's rectangle.
    expect(out).toContain('points="220,170 580,170"');
    expect(out).toContain('stroke="#fff" stroke-width="30" stroke-linecap="round"');
    expect(out).toContain("data-brush-row");
    // The row's plane is its own mask, black where nothing is painted.
    expect(out).toMatch(/<mask id="mvb0"[^>]*><rect [^>]*fill="#000"\/><polyline/);
  });

  it("draws a click as a dot: the same place twice", () => {
    expect(svg([brush([stroke({ points: [[0.5, 0.5]] })])])).toContain('points="400,250 400,250"');
  });

  it("draws an erase stroke in black over the paint before it, in the order the strokes were drawn", () => {
    const out = svg([brush([stroke(), stroke({ erase: true, points: [[0.5, 0.3]] })])]);
    const paint = out.indexOf('stroke="#fff" stroke-width="30"');
    const erase = out.indexOf('stroke="#000" stroke-width="30"');
    expect(paint).toBeGreaterThan(-1);
    expect(erase).toBeGreaterThan(paint);
  });

  it("softens a stroke by its hardness (none at 1) and the plane by the row's Feather", () => {
    expect(svg([brush([stroke({ hardness: 1 })])])).not.toContain("feGaussianBlur");
    const soft = svg([brush([stroke({ hardness: 0 })])]);
    // A ramp of the whole radius: 15 px, a third of it as the blur.
    expect(soft).toContain('<feGaussianBlur stdDeviation="4.95"/>');
    expect(soft).toContain('filter="url(#mvb0s0)"');
    // The region is given in the scene's units, so a horizontal line (no height) is still drawn.
    expect(soft).toContain('filterUnits="userSpaceOnUse" x="100" y="50" width="600" height="400"');
    const feathered = svg([brush([stroke()], { feather: 0.5 })]);
    // σ = feather · 0.02 of the long edge = 6 px.
    expect(feathered).toContain('<feGaussianBlur stdDeviation="6"/>');
    expect(feathered).toContain('<g filter="url(#mvb0f)">');
  });

  it("paints the row's ink at its opacity: white for Add, black for Subtract, and the plane the other way round when the row is inverted", () => {
    expect(svg([brush([stroke()], { opacity: 0.4 })])).toMatch(/<rect data-brush-row="" [^>]*fill="#fff" fill-opacity="0.4" mask="url\(#mvb0\)"\/>/);
    expect(svg([brush([stroke()], { mode: "subtract" })])).toMatch(/<rect data-brush-row="" [^>]*fill="#000"/);
    const inverted = svg([brush([stroke()], { invert: true })]);
    expect(inverted).toMatch(/<mask id="mvb0"[^>]*><rect [^>]*fill="#fff"\/><polyline [^>]*stroke="#000"/);
  });

  it("selects nothing with no strokes, and adds to another row's mask without erasing it", () => {
    expect(svg([brush([])])).not.toContain("<polyline");
    const out = svg([radial(), brush([stroke({ erase: true })])]);
    // The erase is inside the Brush row's own plane: the Radial's gradient is painted before it and is not touched.
    expect(out.indexOf("radialGradient")).toBeLessThan(out.indexOf('<mask id="mvb1"'));
    expect(out).toContain('<mask id="mvb1"');
  });
});

describe("a Background row on the flat scene (docs/algorithms/06 §5.4)", () => {
  const background = (more: Record<string, unknown> = {}): MaskItem => row({ kind: "background", level: 0.5, softness: 0.5, feather: 0, grow: 0, refine: 0.5, ...more });
  const out = (rows: MaskItem[]) => maskViewSvg("mask_only", masks(rows), BOX);
  const greys = (svg: string): number[] => [...svg.matchAll(/fill="rgb\((\d+),\d+,\d+\)"/g)].map((m) => Number(m[1]));

  it("cuts the blur map with S((P − (1 − level) + w / 2) / w): the ends, the middle and the minimum ramp", () => {
    expect(BLUR_RAMP_MIN).toBe(0.05);
    expect(backgroundValue(0.5, 0.5, 0.5)).toBeCloseTo(0.5, 12);
    expect(backgroundValue(0.25, 0.5, 0.5)).toBe(0);
    expect(backgroundValue(0.75, 0.5, 0.5)).toBe(1);
    expect(backgroundValue(0, 1, 0.5)).toBeCloseTo(0.5, 12);
    expect(backgroundValue(0.474, 0.5, 0)).toBe(0);
    expect(backgroundValue(0.526, 0.5, 0)).toBe(1);
    expect(backgroundValue(Number.NaN, 0.5, 0.5)).toBe(0);
  });

  it("paints the sky lit, the far hill dark and the near ground in between at the default Level and Softness", () => {
    const [sky, hill, near] = greys(out([background()]));
    expect([sky, hill]).toEqual([255, 0]);
    expect(near).toBeGreaterThan(100);
    expect(near).toBeLessThan(255);
    expect(MOCK_BLUR.sky).toBeGreaterThan(MOCK_BLUR.near);
    expect(MOCK_BLUR.near).toBeGreaterThan(MOCK_BLUR.hill);
  });

  it("follows Level and Softness, and nothing else of the map moves", () => {
    const base = greys(out([background()]));
    const low = greys(out([background({ level: 0.2 })]));
    const high = greys(out([background({ level: 0.9 })]));
    expect(low[2]).toBeLessThan(base[2]!);
    expect(high[2]).toBeGreaterThan(base[2]!);
    expect(greys(out([background({ softness: 0.9 })]))).not.toEqual(base);
    expect(greys(out([background({ feather: 0.5, grow: 0.5, refine: 1 })]))).toEqual(base);
  });

  it("is blurred by Feather, inverted with the row, and painted at the row's opacity in the row's ink", () => {
    expect(out([background({ feather: 0.5 })])).toContain("feGaussianBlur");
    expect(out([background({ invert: true })])).toContain("feColorMatrix");
    expect(out([background({ opacity: 0.4 })])).toMatch(/<rect data-background-row="" data-level="0.5" data-softness="0.5" [^>]*fill="#fff" fill-opacity="0.4" mask="url\(#mvg0\)"\/>/);
    expect(out([background({ mode: "subtract" })])).toMatch(/data-background-row="" [^>]*fill="#000"/);
  });

  it("is found on a photo whose graph has one with the eye on", () => {
    const graph = (items: MaskItem[]) => ({ nodes: { items: [{ id: 2, kind: "corrector", masks: { invert: false, items } }], links: [] } }) as unknown as DevelopSettings;
    expect(hasBackgroundRow(graph([background()]))).toBe(true);
    expect(hasBackgroundRow(graph([background({ enabled: false })]))).toBe(false);
    expect(hasBackgroundRow(graph([radial()]))).toBe(false);
  });
});

describe("masksOfView (a row's own mask, docs/23 §5.8)", () => {
  const items = [radial({ center: [0.2, 0.5] }), radial({ center: [0.8, 0.5], mode: "subtract", invert: true, enabled: false, opacity: 0.5 })];

  it("is the node's own masks without a row, and for a row the node does not have", () => {
    const own = masks(items, true);
    expect(masksOfView(own, null)).toBe(own);
    expect(masksOfView(own, undefined)).toBe(own);
    expect(masksOfView(own, 2)).toBe(own);
  });

  it("is that row alone: added, on, with its own Invert and Opacity, and no Invert of the node's", () => {
    const only = masksOfView(masks(items, true), 1);
    expect(only.invert).toBe(false);
    expect(only.items).toEqual([{ ...items[1], mode: "add", enabled: true }]);
    // Drawn, it is the row's gradient over black: a Subtract row reads as the positive mask it takes away, and the other row is not there.
    const svg = maskViewSvg("mask_only", only, BOX);
    expect(svg).toContain("translate(580 250)");
    expect(svg).not.toContain("translate(220 250)");
    expect(svg).toMatch(/<mask [^>]*><rect [^>]*fill="#000"\/>/);
  });
});

describe("Show depth (docs/23 §4.8)", () => {
  const depthRow = { mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, kind: "depth", far: 0.45 } as MaskItem;
  const only = masksOfView({ invert: false, items: [depthRow] }, 0);

  it("draws the whole map as grey, white for near (the near ground lighter than the far hill, the hill lighter than the sky), and not the row's gradient", () => {
    const svg = maskViewSvg("overlay", only, BOX, undefined, undefined, { on: true, ready: true });
    expect(svg).toContain('data-mask-view="depth"');
    // No tint and no row ink: the picture is grey.
    expect(svg).not.toContain("#ff6fd2");
    const grey = (z: number): number => Math.round(255 * (1 - z));
    expect(svg).toContain(`rgb(${grey(MOCK_DEPTH.near)},${grey(MOCK_DEPTH.near)},${grey(MOCK_DEPTH.near)})`);
    expect(grey(MOCK_DEPTH.near)).toBeGreaterThan(grey(MOCK_DEPTH.far));
    expect(grey(MOCK_DEPTH.far)).toBeGreaterThan(grey(MOCK_DEPTH.sky));
  });

  it("is black until the map is made, and means nothing for a row that is not a Depth row or when it is off", () => {
    expect(depthViewSvg(BOX, undefined, false)).not.toContain("<ellipse");
    const linear = { mode: "add", invert: false, enabled: true, opacity: 1, kind: "linear", center: [0.5, 0.5], angle_deg: 0, feather: 0.2 } as MaskItem;
    expect(maskViewSvg("mask_only", masksOfView({ invert: false, items: [linear] }, 0), BOX, undefined, undefined, { on: true, ready: true })).toContain('data-mask-view="mask_only"');
    expect(maskViewSvg("mask_only", only, BOX, undefined, undefined, { on: false, ready: true })).toContain('data-mask-view="mask_only"');
  });
});

describe("a Depth row's gradient (docs/algorithms/06 §5.8)", () => {
  const row = (extra: Record<string, unknown> = {}): MaskItem => ({ mode: "add", invert: false, enabled: true, opacity: 1, feather: 0, grow: 0, refine: 0.5, kind: "depth", far: 1, ...extra }) as MaskItem;
  const out = (r: MaskItem, ready = true): string => maskViewSvg("mask_only", masksOfView({ invert: false, items: [r] }, undefined), BOX, undefined, undefined, { on: false, ready });
  /** The grey codes the three areas are painted with, in the order sky, hill, near ground. */
  const greys = (svg: string): number[] => [...svg.matchAll(/fill="rgb\((\d+),\d+,\d+\)"/g)].map((m) => Number(m[1]));

  it("is 1 at the nearest, 0 at the far limit and beyond, linear between, and 1 - z at the whole depth", () => {
    expect(depthValue(0, 0.5)).toBe(1);
    expect(depthValue(0.5, 0.5)).toBe(0);
    expect(depthValue(0.9, 0.5)).toBe(0);
    expect(depthValue(0.25, 0.5)).toBeCloseTo(0.5, 12);
    expect(depthValue(0.3, 1)).toBeCloseTo(0.7, 12);
    // The least limit is held, and NaN reads as the whole depth.
    expect(depthValue(0.01, 0)).toBeCloseTo(0.5, 12);
    expect(depthValue(0.3, Number.NaN)).toBeCloseTo(0.7, 12);
  });

  it("paints the near ground white and fades with depth: at the whole depth the greys are 1 - z of the sky, the hill and the ground", () => {
    const [sky = -1, hill = -1, near = -1] = greys(out(row()));
    expect(near).toBe(Math.round(255 * (1 - MOCK_DEPTH.near)));
    expect(hill).toBe(Math.round(255 * (1 - MOCK_DEPTH.far)));
    expect(sky).toBe(Math.round(255 * (1 - MOCK_DEPTH.sky)));
    expect(near).toBeGreaterThan(0);
    expect(near).toBeGreaterThan(hill);
    expect(hill).toBeGreaterThan(sky);
  });

  it("a nearer far limit takes the hill and the sky out and keeps part of the ground; Invert, Opacity and a row not ready", () => {
    const [sky = -1, hill = -1, near = -1] = greys(out(row({ far: 0.25 })));
    expect([sky, hill]).toEqual([0, 0]);
    expect(near).toBe(Math.round(255 * (1 - MOCK_DEPTH.near / 0.25)));
    expect(out(row({ opacity: 0.4 }))).toContain('fill-opacity="0.4"');
    expect(out(row({ invert: true }))).toContain("feColorMatrix");
    expect(out(row({ far: 0.25 }))).toContain('data-far="0.25"');
    // Until the map is made the row selects nothing.
    expect(out(row(), false)).not.toContain("data-depth-row");
    expect(out(row(), true)).toContain("data-depth-row");
  });

  it("Pick depth's limit is the depth plus the margin, at most 1", () => {
    expect(DEPTH_PICK_MARGIN).toBe(0.1);
    expect(pickDepthFar(MOCK_DEPTH.near)).toBeCloseTo(0.25, 12);
    expect(pickDepthFar(MOCK_DEPTH.sky)).toBe(1);
    expect(pickDepthFar(0)).toBe(0.1);
  });
});
