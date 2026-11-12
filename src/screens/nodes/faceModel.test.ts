import { describe, expect, it } from "vitest";
import { mockFaceBoxes } from "../../api.mock";
import { defaultFaceSettings, newFaceRefinement } from "../../state/settingsGraph";
import { facesCopied, facesLine, facesOnSplit, facesWith, resetPartial, selectedFace } from "./faceModel";

const item = () => newFaceRefinement(2, [1, 0]);
const state = (phase: "finding" | "ready" | "failed" | "models_missing", n = 0) => ({ photo_id: 1, phase, faces: mockFaceBoxes(n), message: null });

describe("the head's line", () => {
  it("names the phase and the count", () => {
    expect(facesLine(undefined)).toBe("Finding faces");
    expect(facesLine(state("finding"))).toBe("Finding faces");
    expect(facesLine(state("ready", 0))).toBe("No faces found");
    expect(facesLine(state("ready", 1))).toBe("1 face");
    expect(facesLine(state("ready", 5))).toBe("5 faces");
    expect(facesLine(state("failed"))).toBe("Can't find faces");
    expect(facesLine(state("models_missing"))).toBe("Face models missing");
  });
});

describe("the lists of settings", () => {
  it("holds the chosen face to the faces there are", () => {
    expect(selectedFace(0, 3)).toBe(1);
    expect(selectedFace(7, 3)).toBe(3);
    expect(selectedFace(2, 0)).toBe(1);
  });

  it("splits from `all` when the list is empty, and keeps the list it has", () => {
    const n = { ...item(), all: { ...defaultFaceSettings(), blemish: 60 } };
    expect(facesOnSplit(n, 3)).toEqual([n.all, n.all, n.all]);
    expect(facesOnSplit({ ...n, faces: [defaultFaceSettings()] }, 3)).toEqual([defaultFaceSettings()]);
  });

  it("changes one face's entry and fills the others as the core reads them", () => {
    const n = { ...item(), same_for_all: false, faces: [{ ...defaultFaceSettings(), blemish: 5 }] };
    const list = facesWith(n, 3, 2, { even_skin: 90 });
    expect(list.map((f) => [f.blemish, f.even_skin])).toEqual([[5, 25], [5, 90], [5, 25]]);
    expect(facesWith(item(), 1, 1, { texture: 4 })).toHaveLength(1);
  });

  it("copies one face's settings to every other face", () => {
    const n = { ...item(), same_for_all: false, faces: [defaultFaceSettings(), { ...defaultFaceSettings(), blemish: 77 }] };
    expect(facesCopied(n, 4, 2).map((f) => f.blemish)).toEqual([77, 77, 77, 77]);
  });

  it("caps the list at 32 faces", () => {
    expect(facesWith(item(), 40, 1, {})).toHaveLength(32);
  });

  it("makes the partial of a Reset", () => {
    expect(resetPartial(["blemish", "texture"], "defaults")).toEqual({ blemish: 30, texture: 0 });
    expect(resetPartial(["skin_contrast", "shine_removal"], "zero")).toEqual({ skin_contrast: 0, shine_removal: 0 });
  });
});

describe("the mock's faces", () => {
  it("lays out one, five and twelve faces left to right inside the frame without overlap", () => {
    for (const n of [1, 5, 12]) {
      const faces = mockFaceBoxes(n);
      expect(faces).toHaveLength(n);
      expect(faces.map((f) => f.number)).toEqual(Array.from({ length: n }, (_, i) => i + 1));
      for (const f of faces) {
        expect(f.x).toBeGreaterThan(0);
        expect(f.x + f.width).toBeLessThan(1);
        expect(f.y).toBeGreaterThan(0);
        expect(f.y + f.height).toBeLessThan(1);
      }
      for (const a of faces) for (const b of faces) if (a !== b) expect(a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y).toBe(true);
    }
  });
});
