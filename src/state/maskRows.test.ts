import { describe, expect, it } from "vitest";
import type { MaskItem } from "../api/types";
import { CANT_BE_MADE, DEPTH_ROW_NO_MODEL, FINDING, foundNothing, isFinding, isUnsupported, maskRowKinds, maskRowLabel, maskRowName, maskSubtitle, newMask, NOTHING_FOUND, UNSUPPORTED_ROW, widthSlider } from "./maskRows";
import type { VisionRows } from "./maskRows";

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

describe("a Background row while the photo's blur map is made (docs/23 §7.6)", () => {
  const background: MaskItem = { ...base, ...edge, kind: "background", level: 0.5, softness: 0.5 };

  it("is 'Finding…' for a Background row that is on, while the core finds, and only then", () => {
    expect(FINDING).toBe("Finding…");
    expect(isFinding(background, true)).toBe(true);
    expect(isFinding(background, false)).toBe(false);
    // The eye off: the row is absent, nothing is made for it.
    expect(isFinding({ ...background, enabled: false }, true)).toBe(false);
    // Every other kind has its raster at once (Brush) or none to wait for.
    for (const row of [
      { ...base, ...edge, kind: "brush", strokes: [] },
      { ...base, kind: "linear", center: [0.5, 0.5], angle_deg: 0, feather: 0.2 },
      { ...base, ...edge, kind: "object", points: [], box: null },
    ] as MaskItem[]) {
      expect(isFinding(row, true)).toBe(false);
    }
  });

  it("changes the accessible name with it, and not the subtitle's own words", () => {
    expect(maskRowLabel(background)).toBe("Background mask, Level 50");
    expect(maskRowLabel(background, false)).toBe("Background mask, Level 50");
    expect(maskRowLabel(background, true)).toBe("Background mask, Finding…");
    expect(maskRowLabel({ ...background, enabled: false }, true)).toBe("Background mask, Level 50");
    expect(maskSubtitle(background)).toBe("Level 50");
  });
});

describe("a Subject or People row (docs/23 §3.10 and §7.6, N-19)", () => {
  const subject: MaskItem = { ...base, ...edge, kind: "subject" };
  const people: MaskItem = { ...base, ...edge, kind: "people" };
  const nothing = (...kinds: Array<"subject" | "people">): VisionRows => ({ unsupported: [], nothing: kinds });
  const cannot = (...kinds: Array<"subject" | "people">): VisionRows => ({ unsupported: kinds, nothing: [] });

  it("is 'Finding…' while the core finds, like a Background row, for rows that are on", () => {
    expect(isFinding(subject, true)).toBe(true);
    expect(isFinding(people, true)).toBe(true);
    expect(isFinding(subject, false)).toBe(false);
    expect(isFinding({ ...people, enabled: false }, true)).toBe(false);
    expect(maskRowLabel(subject, true)).toBe("Subject mask, Finding…");
    expect(maskRowLabel(people, true)).toBe("People mask, Finding…");
  });

  it("says 'Nothing found' for the kinds the core named, for rows that are on, and 'Foreground' and 'People' otherwise", () => {
    expect(NOTHING_FOUND).toBe("Nothing found");
    expect(maskSubtitle(subject, nothing())).toBe("Foreground");
    expect(maskSubtitle(people, nothing())).toBe("People");
    expect(maskSubtitle(subject, nothing("subject"))).toBe("Nothing found");
    expect(maskSubtitle(people, nothing("subject"))).toBe("People");
    expect(maskSubtitle(people, nothing("people"))).toBe("Nothing found");
    expect(foundNothing(subject, nothing("subject"))).toBe(true);
    expect(foundNothing({ ...subject, enabled: false }, nothing("subject"))).toBe(false);
    expect(maskRowLabel(subject, false, undefined, nothing("subject"))).toBe("Subject mask, Nothing found");
    // "Finding…" comes first while the masks are made again.
    expect(maskRowLabel(subject, true, undefined, nothing("subject"))).toBe("Subject mask, Finding…");
    // No other kind says it.
    expect(foundNothing({ ...base, ...edge, kind: "background", level: 0.5, softness: 0.5 }, nothing("subject", "people"))).toBe(false);
  });

  it("says the sentences of the spec, word for word, on a computer that cannot make them", () => {
    expect(UNSUPPORTED_ROW).toEqual({ subject: "Subject masks need a Mac that supports them", people: "People masks need a Mac that supports them" });
    expect(isUnsupported(subject, cannot("subject"))).toBe(true);
    expect(isUnsupported(people, cannot("subject"))).toBe(false);
    expect(isUnsupported(subject, undefined)).toBe(false);
    expect(isUnsupported(subject, { unsupported: [], nothing: [] })).toBe(false);
    expect(maskSubtitle(subject, cannot("subject", "people"))).toBe("Subject masks need a Mac that supports them");
    expect(maskSubtitle(people, cannot("subject", "people"))).toBe("People masks need a Mac that supports them");
    // The sentence comes before "Finding…" and "Nothing found": nothing is being made, nothing was looked for.
    expect(maskRowLabel(subject, true, undefined, cannot("subject"))).toBe("Subject mask, Subject masks need a Mac that supports them");
    expect(maskSubtitle(subject, { unsupported: ["subject"], nothing: ["subject"] })).toBe("Subject masks need a Mac that supports them");
    // The other kinds do not care.
    const linear: MaskItem = { ...base, kind: "linear", center: [0.5, 0.5], angle_deg: 0, feather: 0.2 };
    expect(isUnsupported(linear, cannot("subject", "people"))).toBe(false);
  });
});

describe("the kinds an eye's index is checked against (docs/23 §7.6)", () => {
  it("lists a node's row kinds in order, and nothing for no node, the Input or the Output", () => {
    expect(maskRowKinds(null, 1)).toBe("");
    const settings = { nodes: { items: [{ id: "input", kind: "input" }, { id: 2, kind: "corrector", masks: { invert: false, items: [{ ...base, kind: "linear" }, { ...base, kind: "brush" }] } }] } };
    expect(maskRowKinds(settings as never, 2)).toBe("linear,brush");
    expect(maskRowKinds(settings as never, null)).toBe("");
    expect(maskRowKinds(settings as never, 9)).toBe("");
  });
});

describe("a Depth row's words (docs/23 §7.6)", () => {
  const depth: MaskItem = { ...base, feather: 0, grow: 0, refine: 0.5, kind: "depth", far: 0.44 } as MaskItem;
  const none: VisionRows = { unsupported: [], nothing: [] };

  it("is where the mask fades out, \"To 44 %\", and \"Whole depth\" at the default", () => {
    expect(maskSubtitle(depth)).toBe("To 44 %");
    expect(maskRowLabel(depth)).toBe("Depth mask, To 44 %");
    expect(maskSubtitle({ ...depth, far: 1 } as MaskItem)).toBe("Whole depth");
    expect(maskSubtitle(newMask("depth"))).toBe("Whole depth");
    // A limit that rounds to 100 % is the whole depth too, and the least limit is 2 %.
    expect(maskSubtitle({ ...depth, far: 0.996 } as MaskItem)).toBe("Whole depth");
    expect(maskSubtitle({ ...depth, far: 0.02 } as MaskItem)).toBe("To 2 %");
  });

  it("says Finding… while the map is made, Can't be made when it failed, and the sentence of the missing model before both", () => {
    expect(maskRowLabel(depth, true)).toBe("Depth mask, Finding…");
    expect(isFinding(depth, true)).toBe(true);
    expect(isFinding({ ...depth, enabled: false }, true)).toBe(false);
    const failed: VisionRows = { ...none, depthFailed: true };
    expect(maskSubtitle(depth, failed)).toBe(CANT_BE_MADE);
    expect(CANT_BE_MADE).toBe("Can't be made");
    // A failed map is not "finding" for the row, whatever else the core is making.
    expect(isFinding(depth, true, failed)).toBe(false);
    expect(maskRowLabel(depth, true, undefined, failed)).toBe("Depth mask, Can't be made");
    const missing: VisionRows = { ...none, depthMissing: true, depthFailed: true };
    expect(isUnsupported(depth, missing)).toBe(true);
    expect(DEPTH_ROW_NO_MODEL).toBe("Depth masks need the depth model. Reinstall the app.");
    expect(maskRowLabel(depth, true, undefined, missing)).toBe(`Depth mask, ${DEPTH_ROW_NO_MODEL}`);
    expect(isUnsupported(depth, none)).toBe(false);
    expect(isUnsupported(depth, undefined)).toBe(false);
  });
});
