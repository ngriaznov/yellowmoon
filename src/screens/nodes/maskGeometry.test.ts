import { describe, expect, it } from "vitest";
import type { MaskItem } from "../../api/types";
import { newMask } from "../../state/maskRows";
import { applyRowPatch } from "../../state/settingsGraph";
import { axes, clipLine, dragPatch, frameUnits, fromIso, handlePlan, hasHandles, toIso } from "./maskGeometry";
import type { HandleId, Pt, ShapeRow } from "./maskGeometry";

const landscape = frameUnits(1.5);
const linear = (over: Partial<Extract<ShapeRow, { kind: "linear" }>> = {}): Extract<ShapeRow, { kind: "linear" }> => ({ ...(newMask("linear") as Extract<ShapeRow, { kind: "linear" }>), ...over });
const radial = (over: Partial<Extract<ShapeRow, { kind: "radial" }>> = {}): Extract<ShapeRow, { kind: "radial" }> => ({ ...(newMask("radial") as Extract<ShapeRow, { kind: "radial" }>), ...over });
/** The core's own answer for a flat, unturned photo: fractions to CSS pixels of a 600 × 400 cell, no bend. */
const flat = (p: Pt): Pt => [p[0] * 600, p[1] * 400];

describe("the photo frame's units (docs/23 §3.6)", () => {
  it("make the long edge 1, so a circle is a circle on a wide photo and a tall one", () => {
    expect(frameUnits(1.5)).toEqual({ w: 1, h: 1 / 1.5 });
    expect(frameUnits(2 / 3)).toEqual({ w: 2 / 3, h: 1 });
    expect(frameUnits(1)).toEqual({ w: 1, h: 1 });
    // Not a number, or not positive: the camera's 3:2.
    expect(frameUnits(Number.NaN)).toEqual(frameUnits(1.5));
    expect(frameUnits(0)).toEqual(frameUnits(1.5));
    const p: Pt = [0.3, 0.8];
    expect(fromIso(toIso(p, landscape), landscape)).toEqual(p);
  });

  it("clip a line to the photo", () => {
    // A horizontal line through the middle of a 1 × 2/3 frame crosses all of it.
    const c = toIso([0.5, 0.5], landscape);
    const span = clipLine(c, [1, 0], landscape);
    expect(span?.[0]).toBeCloseTo(-0.5, 12);
    expect(span?.[1]).toBeCloseTo(0.5, 12);
    // A vertical one crosses the frame's height.
    const up = clipLine(c, [0, 1], landscape);
    expect(up?.[1]).toBeCloseTo(1 / 3, 12);
    // A line past the frame misses it.
    expect(clipLine([0.5, 2], [1, 0], landscape)).toBeNull();
  });
});

describe("which rows have handles", () => {
  it("is a Linear or a Radial that is on", () => {
    expect(hasHandles(newMask("linear"))).toBe(true);
    expect(hasHandles(newMask("radial"))).toBe(true);
    expect(hasHandles({ ...newMask("radial"), enabled: false })).toBe(false);
    expect(hasHandles(newMask("color_range"))).toBe(false);
    expect(hasHandles(newMask("brightness_range"))).toBe(false);
    expect(hasHandles(undefined)).toBe(false);
  });
});

describe("a Linear's handles", () => {
  it("are a centre line, the feather's two lines, a dot at the centre, a grip on each feather line and the rotate handle", () => {
    const row = linear({ feather: 0.3 });
    const plan = handlePlan(row, landscape);
    const drawing = plan.draw(plan.points.map(flat));
    expect(drawing.lines.map((l) => [l.id, l.dashed])).toEqual([
      ["centre", false],
      ["start", true],
      ["end", true],
    ]);
    expect(drawing.handles.map((h) => h.id)).toEqual(["move", "feather-start", "feather-end", "rotate"]);
    // At angle 0 the centre line is horizontal across the photo, and the feather's lines are f/2 of the long edge either side of it.
    const centre = drawing.lines[0]?.points ?? [];
    expect(centre[0]).toEqual([0, 200]);
    expect(centre[centre.length - 1]?.[0]).toBeCloseTo(600, 9);
    expect(centre.every((p) => Math.abs(p[1] - 200) < 1e-9)).toBe(true);
    const start = drawing.lines[1]?.points ?? [];
    expect(start[0]?.[1]).toBeCloseTo(200 - 0.15 * 600, 9);
    expect(drawing.lines[2]?.points[0]?.[1]).toBeCloseTo(200 + 0.15 * 600, 9);
    // The rotate handle is on the line, at 80 % across.
    const rotate = drawing.handles.find((h) => h.id === "rotate");
    expect(rotate?.at[1]).toBeCloseTo(200, 9);
    expect(rotate?.at[0]).toBeCloseTo(480, 9);
  });

  it("have a name each, and the arrows' hint", () => {
    const plan = handlePlan(linear(), landscape);
    for (const h of plan.draw(plan.points.map(flat)).handles) expect(h.label).toMatch(/\. The arrow keys move it\.$/);
  });

  it("draw nothing but the centre dot for a line that misses the photo", () => {
    // The centre is held to the photo, but a Linear turned so that it only grazes a corner is a short line: no handles beyond the dot
    // when it misses entirely.
    const row = linear({ center: [0.5, 0.5], angle_deg: 0 });
    const plan = handlePlan({ ...row, center: [0.5, 2] }, landscape);
    const drawing = plan.draw(plan.points.map(flat));
    expect(drawing.lines).toEqual([]);
    expect(drawing.handles.map((h) => h.id)).toEqual(["move"]);
  });

  it("is turned by its angle: at 90° the centre line is vertical", () => {
    const plan = handlePlan(linear({ angle_deg: 90 }), landscape);
    const centre = plan.draw(plan.points.map(flat)).lines[0]?.points ?? [];
    expect(centre.every((p) => Math.abs(p[0] - 300) < 1e-6)).toBe(true);
    expect(centre[0]?.[1]).toBeCloseTo(0, 6);
    expect(centre[centre.length - 1]?.[1]).toBeCloseTo(400, 6);
  });

  it("is moved across the line by its line, and anywhere by its dot", () => {
    const row = linear({ angle_deg: 30 });
    const { n } = axes(30);
    const toLine = dragPatch("line", row, [0.9, 0.2], landscape);
    // The line passes through the pointer's place.
    const c = toLine?.center as Pt;
    const [px, py] = toIso([0.9, 0.2], landscape);
    const [qx, qy] = toIso(c, landscape);
    expect((px - qx) * n[0] + (py - qy) * n[1]).toBeCloseTo(0, 3);
    // Along the line nothing moves: the centre stays on its own line.
    const own = toIso(row.center, landscape);
    const cn = toIso(c, landscape);
    expect((cn[0] - own[0]) * axes(30).t[0] + (cn[1] - own[1]) * axes(30).t[1]).toBeCloseTo(0, 3);
    expect(dragPatch("move", row, [0.2, 0.7], landscape)).toEqual({ center: [0.2, 0.7] });
    // Held to the photo.
    expect(dragPatch("move", row, [1.4, -0.2], landscape)).toEqual({ center: [1, 0] });
  });

  it("has a feather that is twice the distance of either feather line from the centre line", () => {
    const row = linear({ angle_deg: 0, feather: 0.2 });
    // 0.15 of the long edge below the line.
    const below: Pt = fromIso([0.5, toIso(row.center, landscape)[1] + 0.15], landscape);
    expect(dragPatch("feather-end", row, below, landscape)).toEqual({ feather: 0.3 });
    const above: Pt = fromIso([0.5, toIso(row.center, landscape)[1] - 0.15], landscape);
    expect(dragPatch("feather-start", row, above, landscape)).toEqual({ feather: 0.3 });
    // On the line: none. Far away: 1.
    expect(dragPatch("feather-end", row, row.center, landscape)).toEqual({ feather: 0 });
    expect(dragPatch("feather-end", row, [0.5, 3], landscape)).toEqual({ feather: 1 });
  });

  it("is turned to where the pointer is from the centre, clockwise from the horizontal", () => {
    const row = linear();
    expect(dragPatch("rotate", row, [0.9, 0.5], landscape)).toEqual({ angle_deg: 0 });
    // Straight down is 90° (y is down, the turn is clockwise).
    expect(dragPatch("rotate", row, [0.5, 0.9], landscape)).toEqual({ angle_deg: 90 });
    expect(dragPatch("rotate", row, [0.5, 0.1], landscape)).toEqual({ angle_deg: -90 });
    // 45° in the isotropic plane is not 45° of the fractions on a wide photo.
    const a = dragPatch("rotate", row, fromIso([toIso(row.center, landscape)[0] + 0.2, toIso(row.center, landscape)[1] + 0.2], landscape), landscape);
    expect(a?.angle_deg).toBeCloseTo(45, 3);
    // On the centre: nothing to turn to.
    expect(dragPatch("rotate", row, row.center, landscape)).toBeNull();
  });
});

describe("a Radial's handles", () => {
  it("are the outline, the inner ring, the centre, a handle for each radius and the rotate handle", () => {
    const row = radial({ radii: [0.2, 0.1], feather: 0.4, angle_deg: 0 });
    const plan = handlePlan(row, landscape);
    const drawing = plan.draw(plan.points.map(flat));
    expect(drawing.lines.map((l) => [l.id, l.dashed])).toEqual([
      ["outline", false],
      ["inner", true],
    ]);
    expect(drawing.handles.map((h) => h.id)).toEqual(["move", "radius-x", "radius-y", "rotate"]);
    // The outline is the ellipse: 0.2 of the long edge across the photo, 0.1 down.
    const outline = drawing.lines[0]?.points ?? [];
    expect(outline[0]?.[0]).toBeCloseTo(300 + 0.2 * 600, 6);
    const xs = outline.map((p) => p[0]);
    const ys = outline.map((p) => p[1]);
    expect(Math.max(...xs) - Math.min(...xs)).toBeCloseTo(0.4 * 600, 6);
    expect(Math.max(...ys) - Math.min(...ys)).toBeCloseTo(0.2 * 600, 6);
    // The inner ring is the outline scaled by 1 − feather.
    const inner = drawing.lines[1]?.points ?? [];
    expect(inner[0]?.[0]).toBeCloseTo(300 + 0.2 * 0.6 * 600, 6);
    // The handles: the first radius to the right of the centre, the second below it, the rotate handle above the outline.
    const at = (id: HandleId): Pt => drawing.handles.find((h) => h.id === id)?.at ?? [0, 0];
    expect(at("radius-x")[0]).toBeCloseTo(300 + 0.2 * 600, 6);
    expect(at("radius-y")[1]).toBeCloseTo(200 + 0.1 * 600, 6);
    expect(at("rotate")[1]).toBeLessThan(200 - 0.1 * 600);
  });

  it("has no inner ring at Feather 0 or at a full feather", () => {
    for (const feather of [0, 1]) {
      const plan = handlePlan(radial({ feather }), landscape);
      expect(plan.draw(plan.points.map(flat)).lines.map((l) => l.id)).toEqual(["outline"]);
    }
  });

  it("is turned with its angle: the first radius's handle follows", () => {
    const plan = handlePlan(radial({ radii: [0.2, 0.1], angle_deg: 90, feather: 0.4 }), landscape);
    const x = plan.draw(plan.points.map(flat)).handles.find((h) => h.id === "radius-x");
    // 90° clockwise: the first radius points down the photo.
    expect(x?.at[0]).toBeCloseTo(300, 6);
    expect(x?.at[1]).toBeCloseTo(200 + 0.2 * 600, 6);
  });

  it("takes its radii from the pointer's distance along each axis, held to the field's range", () => {
    const row = radial({ angle_deg: 0, radii: [0.25, 0.18] });
    const c = toIso(row.center, landscape);
    expect(dragPatch("radius-x", row, fromIso([c[0] + 0.3, c[1]], landscape), landscape)).toEqual({ radii: [0.3, 0.18] });
    expect(dragPatch("radius-y", row, fromIso([c[0], c[1] - 0.12], landscape), landscape)).toEqual({ radii: [0.25, 0.12] });
    // On the centre it is the smallest radius, not 0; far away it is the largest, 1.
    expect(dragPatch("radius-x", row, row.center, landscape)).toEqual({ radii: [0.002, 0.18] });
    expect(dragPatch("radius-y", row, [0.5, 40], landscape)).toEqual({ radii: [0.25, 1] });
    // Along the turned axis.
    const turned = radial({ angle_deg: 90, radii: [0.25, 0.18] });
    expect(dragPatch("radius-x", turned, fromIso([c[0], c[1] + 0.3], landscape), landscape)).toEqual({ radii: [0.3, 0.18] });
    // A Linear has none of these.
    expect(dragPatch("radius-x", linear(), [0.9, 0.5], landscape)).toBeNull();
    expect(dragPatch("feather-end", row, [0.9, 0.5], landscape)).toBeNull();
  });

  it("is turned by its rotate handle, which stands a quarter turn behind the first radius", () => {
    const row = radial({ angle_deg: 0 });
    const c = toIso(row.center, landscape);
    // The handle is above the outline at angle 0: dragging it to the top of the centre is the angle it has now.
    expect(dragPatch("rotate", row, fromIso([c[0], c[1] - 0.3], landscape), landscape)).toEqual({ angle_deg: 0 });
    // Dragging it to the right of the centre turns the first radius a quarter turn clockwise.
    expect(dragPatch("rotate", row, fromIso([c[0] + 0.3, c[1]], landscape), landscape)).toEqual({ angle_deg: 90 });
  });
});

describe("dragging a handle to where it already is changes nothing", () => {
  it("for every handle of every shape, whatever the photo's shape", () => {
    for (const aspect of [1.5, 1, 2 / 3]) {
      const units = frameUnits(aspect);
      const rows: ShapeRow[] = [
        linear({ center: [0.4, 0.6], angle_deg: 25, feather: 0.3 }),
        linear({ center: [0.5, 0.5], angle_deg: -70, feather: 0 }),
        radial({ center: [0.6, 0.4], radii: [0.2, 0.1], angle_deg: 33, feather: 0.4 }),
        radial({ center: [0.3, 0.5], radii: [0.08, 0.15], angle_deg: -120, feather: 0.1 }),
      ];
      for (const row of rows) {
        const plan = handlePlan(row, units);
        // The answers of a core that draws the photo as it is: the plan's own points.
        const drawing = plan.draw(plan.points);
        expect(drawing.handles.length).toBeGreaterThanOrEqual(3);
        for (const h of drawing.handles) {
          const patch = dragPatch(h.id, row, h.at, units);
          expect(patch, `${row.kind} ${h.id} at ${aspect}`).not.toBeNull();
          const after = applyRowPatch(row as MaskItem, patch ?? {}) as ShapeRow;
          if (row.kind === "linear") {
            const o = row as Extract<ShapeRow, { kind: "linear" }>;
            const a = after as typeof o;
            expect(a.center[0], `${h.id}`).toBeCloseTo(o.center[0], 3);
            expect(a.center[1], `${h.id}`).toBeCloseTo(o.center[1], 3);
            expect(a.angle_deg, `${h.id}`).toBeCloseTo(o.angle_deg, 2);
            // A grip on a feather line gives the same feather.
            expect(a.feather, `${h.id}`).toBeCloseTo(o.feather, 3);
          } else {
            const o = row as Extract<ShapeRow, { kind: "radial" }>;
            const a = after as typeof o;
            expect(a.center[0], `${h.id}`).toBeCloseTo(o.center[0], 3);
            expect(a.center[1], `${h.id}`).toBeCloseTo(o.center[1], 3);
            expect(a.radii[0], `${h.id}`).toBeCloseTo(o.radii[0], 3);
            expect(a.radii[1], `${h.id}`).toBeCloseTo(o.radii[1], 3);
            expect(a.angle_deg, `${h.id}`).toBeCloseTo(o.angle_deg, 2);
          }
        }
      }
    }
  });
});
