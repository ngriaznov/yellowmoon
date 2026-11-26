/**
 * What the page knows of a Reference Grade node's reference (`docs/34` §9.2, §9.3): **every string of §9.3, pinned** character for character, the
 * status-line state machine (including "look changed since the fit", "edited by hand since the fit" and the progress text), the Fit button's looks, the notes
 * and the elements' shares.
 */
import { describe, expect, it } from "vitest";
import type { DevelopSettings, ReferenceInfo } from "../api/types";
import {
  EMPTY_CARD,
  fitButton,
  fittedKey,
  hasElementGrades,
  hasFittedValues,
  hasSpatialAfter,
  lookKey,
  NOT_FOUND_SHARE,
  referenceNotes,
  referenceStatus,
  REFERENCE_TEXT,
  shareOf,
  shareText,
} from "./reference";
import type { ReferenceCard } from "./reference";
import { applyPatch, defaultSettings, nodeOf } from "./settingsGraph";
import type { ReferenceGradeItem } from "./settingsGraph";

const pic = (from: number | "input", to: number | "output") => ({ from: { node: from, port: "picture" as const }, to: { node: to, port: "picture" as const } });

/** Main, a Reference Grade node (2), then the Output. */
function graph(extra: Record<string, unknown> = {}): DevelopSettings {
  return applyPatch(defaultSettings(), { nodes: { "2": { kind: "reference_grade", pos: [1, 0], ...extra } }, links: [pic("input", 1), pic(1, 2), pic(2, "output")] });
}
const matchOf = (s: DevelopSettings): ReferenceGradeItem => nodeOf(s, 2) as ReferenceGradeItem;
const INFO: ReferenceInfo = { photo_id: 1, node_id: 2, label: "harbour-dusk.jpg", source: "file", notes: [], grey: false, thumbnail: "ym-frame://x.png" };
const FITTED = { hue_curves: { hue_vs_hue: [{ hue_deg: 120, value: 12 }] }, elements: { sky: { exposure_ev: -0.3, hue_deg: 8, chroma: 1.1 } } };

/** The card of a node that was fitted under `settings`, as the store writes it. */
function fittedCard(settings: DevelopSettings, status: "fitted" | "already_close" = "fitted", notes: string[] = []): ReferenceCard {
  return { ...EMPTY_CARD, info: INFO, result: { status, look: lookKey(settings), fitted: fittedKey(matchOf(settings)), notes, shares: [] } };
}

describe("the copy of docs/34 §9.3, character for character", () => {
  it("status lines", () => {
    expect(REFERENCE_TEXT.noReference).toBe("Choose a picture to match.");
    expect(REFERENCE_TEXT.pressFit).toBe("Press Fit to match the photo to this picture.");
    expect(REFERENCE_TEXT.makingScene).toBe("Finding what is in the pictures…");
    expect(REFERENCE_TEXT.fitted).toBe("Fitted to the reference.");
    expect(REFERENCE_TEXT.alreadyClose).toBe("The photo is already close to the reference. Nothing was changed.");
    expect(REFERENCE_TEXT.edited).toBe("Edited by hand since the fit. Fit again to start over.");
    expect(REFERENCE_TEXT.lookChanged).toBe("The look changed since the fit. Fit again to match it with the new look.");
    expect(REFERENCE_TEXT.valuesKept).toBe("The values are kept. Choose a reference to fit again.");
  });

  it("notes", () => {
    expect(REFERENCE_TEXT.spatialAfter).toBe("Nodes after this one that blur or add film aren't part of the fit.");
    expect(REFERENCE_TEXT.sceneMissingNote).toBe("The scene model is missing from this install, so element grades aren't applied. Reinstall Yellowmoon.");
    expect(REFERENCE_TEXT.blackBars).toBe("Black bars were removed from the reference.");
    expect(REFERENCE_TEXT.profileUnreadable).toBe("The picture's colour profile couldn't be read, so it is taken as sRGB.");
    expect(REFERENCE_TEXT.noColour).toBe("This picture has no colour. The photo will be matched in black and white.");
    expect(REFERENCE_TEXT.noSkin).toBe("No skin was found, so Skin protection did nothing.");
  });

  it("errors, and the tile tooltip", () => {
    expect(REFERENCE_TEXT.unreadable).toBe("This file can't be read. It may be damaged. Choose another picture.");
    expect(REFERENCE_TEXT.unsupported).toBe("Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again.");
    expect(REFERENCE_TEXT.tooSmall).toBe("This picture is too small to match. Use one that is at least 256 pixels on its short side.");
    expect(REFERENCE_TEXT.empty).toBe("There's no picture inside the black bars. Choose another picture.");
    expect(REFERENCE_TEXT.samePhoto).toBe("That's the photo you're grading. Choose another.");
    expect(REFERENCE_TEXT.notReady).toBe("This photo isn't ready yet.");
    expect(REFERENCE_TEXT.fitFailed).toBe("The grade couldn't be fitted. Try another reference.");
    expect(REFERENCE_TEXT.sceneMissing).toBe("The scene model is missing from this install. Reinstall Yellowmoon.");
  });

  it("the card's own lines and tooltips", () => {
    expect(REFERENCE_TEXT.chooseFirst).toBe("Choose a reference first.");
    expect(REFERENCE_TEXT.strengthLine).toBe("Scales the fitted values. The tabs show them before Strength.");
    expect(REFERENCE_TEXT.skinLine).toBe("Keeps skin tones from drifting. Applies at the next fit.");
    expect(REFERENCE_TEXT.chooserHint).toBe("Click a photo in the strip. Esc cancels.");
  });

  it("has no exclamation mark anywhere", () => {
    for (const text of Object.values(REFERENCE_TEXT)) expect(text).not.toContain("!");
  });
});

describe("the status line, one state at a time", () => {
  it("no reference: choose a picture", () => {
    expect(referenceStatus(undefined, matchOf(graph()), graph())).toEqual({ state: "none", text: "Choose a picture to match." });
  });

  it("a reference, not fitted: press Fit", () => {
    const s = graph();
    expect(referenceStatus({ ...EMPTY_CARD, info: INFO }, matchOf(s), s)).toEqual({ state: "ready", text: "Press Fit to match the photo to this picture." });
  });

  it("making the scene, then Fitting… 1 of 3 to 3 of 3", () => {
    const s = graph();
    const run = (fit: ReferenceCard["fit"]) => referenceStatus({ ...EMPTY_CARD, info: INFO, fit }, matchOf(s), s);
    expect(run({ step: "scene" })).toEqual({ state: "running", text: "Finding what is in the pictures…" });
    for (const k of [1, 2, 3]) expect(run({ step: "search", search: k, of: 3, fraction: k / 3 }).text).toBe(`Fitting… ${k} of 3`);
  });

  it("fitted, then the photo is already close", () => {
    const s = graph(FITTED);
    expect(referenceStatus(fittedCard(s), matchOf(s), s)).toEqual({ state: "fitted", text: "Fitted to the reference." });
    expect(referenceStatus(fittedCard(s, "already_close"), matchOf(s), s)).toEqual({ state: "already_close", text: "The photo is already close to the reference. Nothing was changed." });
  });

  it("edited by hand: a fitted value changed since the fit", () => {
    const s = graph(FITTED);
    const card = fittedCard(s);
    const edited = applyPatch(s, { nodes: { "2": { elements: { vegetation: { hue_deg: 32 } } } } });
    expect(referenceStatus(card, matchOf(edited), edited)).toEqual({ state: "edited", text: "Edited by hand since the fit. Fit again to start over." });
  });

  it("is not edited by Strength, Skin protection, the name or the node's trim: they are not fitted values", () => {
    const s = graph(FITTED);
    const card = fittedCard(s);
    const other = applyPatch(s, { nodes: { "2": { strength: 40, skin_protection: false, name: "Dusk", trim: { exposure_ev: 0.5 } } } });
    expect(referenceStatus(card, matchOf(other), other).state).toBe("fitted");
  });

  it("the look changed since the fit: another look, or another intensity of it, and it wins over an edit", () => {
    const s = graph(FITTED);
    const card = fittedCard(s);
    const looked = applyPatch(s, { nodes: { output: { look_intensity: 50 } } });
    expect(referenceStatus(card, matchOf(looked), looked)).toEqual({ state: "look_changed", text: "The look changed since the fit. Fit again to match it with the new look." });
    const other = applyPatch(s, { nodes: { output: { look: "builtin:signature" } } });
    expect(referenceStatus(card, matchOf(other), other).state).toBe("look_changed");
    const both = applyPatch(looked, { nodes: { "2": { elements: { sky: { hue_deg: -5 } } } } });
    expect(referenceStatus(card, matchOf(both), both).state).toBe("look_changed");
  });

  it("a change that is neither the look nor a fitted value (a node's exposure) marks nothing stale", () => {
    const s = graph(FITTED);
    const card = fittedCard(s);
    const later = applyPatch(s, { nodes: { "1": { trim: { exposure_ev: 0.7 } } } });
    expect(referenceStatus(card, matchOf(later), later).state).toBe("fitted");
  });

  it("values, no reference: after a restart, a paste or a preset the node holds numbers and the page holds nothing", () => {
    const s = graph(FITTED);
    expect(hasFittedValues(matchOf(s))).toBe(true);
    expect(referenceStatus(undefined, matchOf(s), s)).toEqual({ state: "values_kept", text: "The values are kept. Choose a reference to fit again." });
    expect(hasFittedValues(matchOf(graph()))).toBe(false);
  });

  it("choosing another reference clears the last fit: Press Fit again, though the values stay", () => {
    const s = graph(FITTED);
    expect(referenceStatus({ ...fittedCard(s), result: null }, matchOf(s), s).state).toBe("ready");
  });

  it("a fit that runs says its step before anything else", () => {
    const s = graph(FITTED);
    expect(referenceStatus({ ...fittedCard(s), fit: { step: "scene" } }, matchOf(s), s).state).toBe("running");
  });
});

describe("the Fit button", () => {
  it("is off with no reference, prominent with one, plain after a fit, and accented only when the look changed", () => {
    expect(fitButton("none")).toEqual({ label: "Fit", disabled: true, prominent: false, accent: false });
    expect(fitButton("values_kept").disabled).toBe(true);
    expect(fitButton("ready")).toEqual({ label: "Fit", disabled: false, prominent: true, accent: false });
    for (const state of ["fitted", "already_close", "edited"] as const) expect(fitButton(state)).toEqual({ label: "Fit again", disabled: false, prominent: false, accent: false });
    expect(fitButton("look_changed")).toEqual({ label: "Fit again", disabled: false, prominent: true, accent: true });
  });
});

describe("the notes", () => {
  it("show only those that apply, each once, in a fixed order", () => {
    const s = graph();
    const card: ReferenceCard = {
      ...EMPTY_CARD,
      info: { ...INFO, notes: [REFERENCE_TEXT.noColour, REFERENCE_TEXT.blackBars] },
      result: { status: "fitted", look: lookKey(s), fitted: fittedKey(matchOf(s)), notes: [REFERENCE_TEXT.noSkin, REFERENCE_TEXT.blackBars], shares: [] },
    };
    expect(referenceNotes(card, matchOf(s), s, 2)).toEqual([REFERENCE_TEXT.blackBars, REFERENCE_TEXT.noColour, REFERENCE_TEXT.noSkin]);
    expect(referenceNotes(undefined, matchOf(s), s, 2)).toEqual([]);
  });

  it("say that nodes after this one that blur or add film are not part of the fit", () => {
    const after = applyPatch(graph(), { nodes: { "3": { kind: "blur", pos: [2, 0] } }, links: [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output")] });
    expect(hasSpatialAfter(after, 2)).toBe(true);
    expect(referenceNotes(undefined, matchOf(after), after, 2)).toEqual([REFERENCE_TEXT.spatialAfter]);
    // A corrector after it, or a Blur before it, is not a spatial node after it.
    const corrector = applyPatch(graph(), { nodes: { "3": { kind: "corrector", pos: [2, 0] } }, links: [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output")] });
    expect(hasSpatialAfter(corrector, 2)).toBe(false);
    const before = applyPatch(defaultSettings(), { nodes: { "2": { kind: "blur", pos: [1, 0] }, "3": { kind: "reference_grade", pos: [2, 0] } }, links: [pic("input", 1), pic(1, 2), pic(2, 3), pic(3, "output")] });
    expect(hasSpatialAfter(before, 3)).toBe(false);
  });

  it("say the scene model is missing only after a fit found it so and while element grades are set", () => {
    const withElements = graph(FITTED);
    const card = { ...EMPTY_CARD, sceneMissing: true };
    expect(hasElementGrades(matchOf(withElements))).toBe(true);
    expect(referenceNotes(card, matchOf(withElements), withElements, 2)).toEqual([REFERENCE_TEXT.sceneMissingNote]);
    expect(referenceNotes(card, matchOf(graph()), graph(), 2)).toEqual([]);
    expect(referenceNotes(EMPTY_CARD, matchOf(withElements), withElements, 2)).toEqual([]);
  });
});

describe("the elements' shares", () => {
  it("are what the last fit found; none before, and under 2 % is not found", () => {
    const s = graph();
    const card: ReferenceCard = { ...fittedCard(s), result: { ...(fittedCard(s).result as NonNullable<ReferenceCard["result"]>), shares: [{ element: "sky", share: 0.26 }, { element: "water", share: 0.006 }] } };
    expect(shareOf(card, "sky")).toBe(0.26);
    expect(shareOf(card, "ground")).toBeNull();
    expect(shareOf(undefined, "sky")).toBeNull();
    expect(NOT_FOUND_SHARE).toBe(0.02);
    expect(shareText(0.26)).toBe("26 % of the photo");
    expect(shareText(0.006)).toBe("<1 % of the photo");
  });
});
