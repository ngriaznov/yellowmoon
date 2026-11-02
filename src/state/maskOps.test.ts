import { describe, expect, it } from "vitest";
import type { MaskItem, MaskKind } from "../api/types";
import { applyRowPatch } from "./settingsGraph";
import { editRow, MAX_MASKS, movedMask, newMask, resetRowPatch, selectionAfterRemove, withMask, withoutMask } from "./maskRows";

const KINDS: MaskKind[] = ["object", "subject", "people", "background", "linear", "radial", "brush", "color_range", "brightness_range"];

describe("a new mask row (docs/23 §3.3)", () => {
  it("has the defaults of its kind, in Add mode, on, not inverted, at full opacity", () => {
    expect(newMask("linear")).toEqual({ kind: "linear", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], angle_deg: 0, feather: 0.2 });
    expect(newMask("radial")).toMatchObject({ center: [0.5, 0.5], radii: [0.25, 0.18], angle_deg: 0, feather: 0.3 });
    expect(newMask("color_range")).toMatchObject({ center: [0, 0], width: 0.05, softness: 0.5 });
    expect(newMask("brightness_range")).toMatchObject({ low: 0.35, high: 1, softness: 0.5 });
    expect(newMask("object")).toMatchObject({ points: [], box: null, feather: 0, grow: 0, refine: 0.5 });
    expect(newMask("brush")).toMatchObject({ strokes: [], feather: 0, grow: 0, refine: 0 });
    expect(newMask("background")).toMatchObject({ level: 0.5, softness: 0.5, refine: 0.5 });
    for (const kind of KINDS) expect(newMask(kind).kind).toBe(kind);
  });

  it("takes the mode it is made in", () => {
    expect(newMask("radial", "subtract").mode).toBe("subtract");
  });

  it("is put back by its card's Reset as a new row of the kind in the same mode, whatever was changed", () => {
    const changed: MaskItem = {
      ...newMask("radial", "subtract"),
      invert: true,
      enabled: false,
      opacity: 0.4,
      center: [0.1, 0.9],
      radii: [0.5, 0.05],
      angle_deg: 33,
      feather: 0.9,
    } as MaskItem;
    const patch = resetRowPatch(changed);
    // The kind and the mode are not in a row patch's reset: a row patch cannot change the kind, and the mode stays.
    expect(patch).not.toHaveProperty("kind");
    expect(patch).not.toHaveProperty("mode");
    expect(applyRowPatch(changed, patch)).toEqual(newMask("radial", "subtract"));
    for (const kind of KINDS) expect(applyRowPatch(newMask(kind, "add"), resetRowPatch(newMask(kind, "add")))).toEqual(newMask(kind, "add"));
  });
});

describe("the row operations (one node patch, one undo step)", () => {
  const rows = [newMask("linear"), newMask("radial"), newMask("color_range")];

  it("adds at the end, up to 64", () => {
    expect(withMask(rows, newMask("brightness_range"))?.items).toHaveLength(4);
    expect(withMask(rows, newMask("brightness_range"))?.items?.[3]?.kind).toBe("brightness_range");
    expect(withMask(Array.from({ length: MAX_MASKS }, () => newMask("linear")), newMask("linear"))).toBeNull();
  });

  it("removes a row and says which one to select next", () => {
    expect(withoutMask(rows, 1).items?.map((r) => r.kind)).toEqual(["linear", "color_range"]);
    expect(selectionAfterRemove(1, 3)).toBe(1);
    expect(selectionAfterRemove(2, 3)).toBe(1);
    expect(selectionAfterRemove(0, 1)).toBeNull();
  });

  it("moves a row up or down, and nowhere past the ends", () => {
    expect(movedMask(rows, 1, -1)?.items?.map((r) => r.kind)).toEqual(["radial", "linear", "color_range"]);
    expect(movedMask(rows, 1, 1)?.items?.map((r) => r.kind)).toEqual(["linear", "color_range", "radial"]);
    expect(movedMask(rows, 0, -1)).toBeNull();
    expect(movedMask(rows, 2, 1)).toBeNull();
  });

  it("edits one row by its index, as a string", () => {
    expect(editRow(2, { opacity: 0.5 })).toEqual({ edit: { "2": { opacity: 0.5 } } });
  });
});
