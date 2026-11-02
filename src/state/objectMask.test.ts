import { describe, expect, it } from "vitest";
import { boxPrompt, clickPrompt, foundText, GETTING_READY, NOTHING_FOUND, objectLine } from "./objectMask";
import { maskRowLabel, maskSubtitle, objectBusyText } from "./maskRows";
import { newMask } from "./maskRows";
import type { MaskItem } from "../api/types";

const OBJECT = (more: Record<string, unknown> = {}): MaskItem => ({ ...newMask("object"), ...more }) as MaskItem;

describe("the words of the Object mask", () => {
  it("the pill gives the decoder's time in seconds to one decimal", () => {
    expect(foundText(412)).toBe("Object found · 0.4 s");
    expect(foundText(1260)).toBe("Object found · 1.3 s");
    expect(foundText(40)).toBe("Object found · 0.0 s");
  });

  it("nothing found and the encoder's wait are word for word", () => {
    expect(NOTHING_FOUND).toBe("Nothing found there. Click another part of the object, or draw a box.");
    expect(GETTING_READY).toBe("Getting ready…");
  });

  it("each tool has its line under the segmented control", () => {
    expect(objectLine("add", true, true)).toBe("Click the photo to add to the object. ⌥-click to take a part away.");
    expect(objectLine("remove", true, true)).toBe("Click the photo to take a part away from the object.");
    expect(objectLine("box", true, true)).toBe("Drag a box around the object.");
    expect(objectLine("add", false, true)).toBe("Choose a tool to find an object on the photo.");
    expect(objectLine("add", true, false)).toBe("This mask is off. Turn it on to find an object.");
  });
});

describe("what a click and a drag ask for", () => {
  it("the tool sets the sign of a click, and ⌥-click takes a part away whatever the tool is", () => {
    expect(clickPrompt("add", false, 0.4, 0.5)).toEqual({ kind: "point", x: 0.4, y: 0.5, positive: true });
    expect(clickPrompt("remove", false, 0.4, 0.5)).toEqual({ kind: "point", x: 0.4, y: 0.5, positive: false });
    expect(clickPrompt("add", true, 0.4, 0.5)).toEqual({ kind: "point", x: 0.4, y: 0.5, positive: false });
    expect(clickPrompt("box", true, 0.4, 0.5)).toEqual({ kind: "point", x: 0.4, y: 0.5, positive: false });
  });

  it("a click with the Box tool is not a find", () => {
    expect(clickPrompt("box", false, 0.4, 0.5)).toBeNull();
  });

  it("a box has its corners in order whichever way it was dragged", () => {
    expect(boxPrompt([0.6, 0.7], [0.2, 0.1])).toEqual({ kind: "box", x0: 0.2, y0: 0.1, x1: 0.6, y1: 0.7 });
  });
});

describe("the row says what the core is doing", () => {
  it("Finding… while a click is answered, Getting ready… while the encoder runs, and only for an Object row with the eye on", () => {
    expect(objectBusyText(OBJECT(), { preparing: true, finding: false })).toBe("Getting ready…");
    expect(objectBusyText(OBJECT(), { preparing: true, finding: true })).toBe("Finding…");
    expect(objectBusyText(OBJECT(), { preparing: false, finding: true })).toBe("Finding…");
    expect(objectBusyText(OBJECT(), { preparing: false, finding: false })).toBeNull();
    expect(objectBusyText(OBJECT({ enabled: false }), { preparing: true, finding: true })).toBeNull();
    expect(objectBusyText(newMask("radial"), { preparing: true, finding: true })).toBeNull();
    expect(objectBusyText(OBJECT(), undefined)).toBeNull();
  });

  it("the subtitles of the row and its accessible name", () => {
    expect(maskSubtitle(OBJECT())).toBe("Click the object on this photo");
    expect(maskSubtitle(OBJECT({ points: [[0.5, 0.5, true]] }))).toBe("1 point");
    expect(maskSubtitle(OBJECT({ points: [[0.5, 0.5, true], [0.4, 0.4, false]] }))).toBe("2 points");
    expect(maskSubtitle(OBJECT({ box: [0.1, 0.1, 0.5, 0.5] }))).toBe("Box");
    expect(maskSubtitle(OBJECT({ box: [0.1, 0.1, 0.5, 0.5], points: [[0.5, 0.5, true], [0.4, 0.4, false]] }))).toBe("Box and 2 points");
    expect(maskRowLabel(OBJECT({ points: [[0.5, 0.5, true]] }))).toBe("Object mask, 1 point");
    expect(maskRowLabel(OBJECT(), false, { preparing: true, finding: false })).toBe("Object mask, Getting ready…");
    expect(maskRowLabel(OBJECT(), false, { preparing: false, finding: true })).toBe("Object mask, Finding…");
  });
});
