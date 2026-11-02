/** The mask view on the mock's flat scene (`maskView.ts`): the shapes the page tests and the reference screenshots rely on. */
import { describe, expect, it } from "vitest";
import type { MaskItem, MaskSettings } from "../api/types";
import { MASK_ACCENT, maskViewSvg, OVERLAY_WEIGHT } from "./maskView";

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
