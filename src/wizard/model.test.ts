import { describe, expect, it } from "vitest";
import type { LookInput, PackageId, SourceRecognition } from "../api/types";
import {
  anyChecking,
  canAdvance,
  downloadTotalMb,
  footNote,
  heading,
  importPlan,
  initialWizard,
  joinList,
  lead,
  matchedPackages,
  offersDisplayChoice,
  PACKAGES,
  preferredDisplayLut,
  recogniseBlocker,
  recogniseNote,
  reduceWizard,
  standsInFor,
  stepLocked,
  trashCandidates,
  unmatchedPackages,
  verdictFor,
  visiblePackages,
  wantList,
} from "./model";
import type { WizardAction, WizardState } from "./model";

function run(actions: WizardAction[], from: WizardState = initialWizard()): WizardState {
  return actions.reduce(reduceWizard, from);
}

function pkg(path: string, id: PackageId, title = "Look Library · LogC4", files = 87): SourceRecognition {
  return {
    path,
    name: path.split("/").pop() ?? path,
    source: "zip",
    size_bytes: 181_000_000,
    recognition: { kind: "package", package_id: id, title, what: `${title} · log to log`, contents: `${files} .cube · 65-point`, file_count: files, note: null },
  };
}
const unusable = (path: string, reason: string): SourceRecognition => ({ path, name: path.split("/").pop() ?? path, source: "zip", size_bytes: 120_000_000, recognition: { kind: "unusable", package_id: null, reason } });
const missing = (path: string): SourceRecognition => ({ path, name: path.split("/").pop() ?? path, source: "zip", size_bytes: null, recognition: { kind: "missing", reason: "That file is not there any more. Download it again, or choose it again." } });
const cube = (path: string, suggestion: LookInput | null = null): SourceRecognition => ({
  path,
  name: path.split("/").pop() ?? path,
  source: "file",
  size_bytes: 1_200_000,
  recognition: { kind: "needs_answer", file_count: 1, lut_size: 33, suggestion, note: "1 file is not from an ARRI package we know." },
});

/** A zip or folder of unknown `.cube` files (the real LogC4 zip when nothing recognises it): `suggestion` is what the names say. */
const unknownSet = (path: string, suggestion: LookInput | null = "logc4", source: "zip" | "folder" = "zip"): SourceRecognition => ({
  path,
  name: path.split("/").pop() ?? path,
  source,
  size_bytes: 181_000_000,
  recognition: { kind: "needs_answer", file_count: 87, lut_size: 65, suggestion, note: "87 files are not from an ARRI package we know. Say what they expect as input: LogC4, LogC3, or LogC3 with Rec.709 baked in." },
});

/** A wizard that has got to `step` with the sources recognised as given. */
function at(step: number, found: SourceRecognition[], chosen?: PackageId[]): WizardState {
  let state = initialWizard();
  if (chosen) state = { ...state, chosen };
  for (const f of found) state = run([{ type: "received", path: f.path }, { type: "recognised", found: f }], state);
  return { ...state, step: step as WizardState["step"] };
}

const LL4 = "~/Downloads/ARRI_LookLibrary_LogC4.zip";

describe("Choose", () => {
  it("starts with the recommended package ticked and the older ones hidden", () => {
    const state = initialWizard();
    expect(state.chosen).toEqual(["look_library_logc4"]);
    expect(visiblePackages(state).map((p) => p.id)).toEqual(["look_library_logc4", "display_luts_logc4"]);
    expect(PACKAGES.filter((p) => p.recommended).map((p) => p.id)).toEqual(["look_library_logc4"]);
  });

  it("toggles packages and keeps them in the order of the list", () => {
    let state = run([{ type: "toggle-package", id: "display_luts_logc4" }]);
    expect(state.chosen).toEqual(["look_library_logc4", "display_luts_logc4"]);
    state = run([{ type: "toggle-package", id: "look_library_logc4" }], state);
    expect(state.chosen).toEqual(["display_luts_logc4"]);
    state = run([{ type: "toggle-package", id: "look_library_logc4" }], state);
    expect(state.chosen).toEqual(["look_library_logc4", "display_luts_logc4"]);
  });

  it("sums the download and says so in the footer", () => {
    let state = initialWizard();
    expect(downloadTotalMb(state)).toBe(181);
    expect(footNote(state)).toBe("Download total: 181 MB");
    state = run([{ type: "toggle-package", id: "display_luts_logc4" }], state);
    expect(downloadTotalMb(state)).toBe(213);
    expect(footNote(state)).toBe("Download total: 213 MB");
  });

  it("locks Next until a package is chosen", () => {
    const none = run([{ type: "toggle-package", id: "look_library_logc4" }]);
    expect(canAdvance(none)).toBe(false);
    expect(footNote(none)).toBe("Choose at least one package");
    expect(run([{ type: "advance" }], none).step).toBe(0);
  });

  it("shows three older packages when asked, and unchooses them when they are hidden again", () => {
    let state = run([{ type: "toggle-older" }]);
    expect(visiblePackages(state)).toHaveLength(5);
    state = run([{ type: "toggle-package", id: "look_library_logc3" }], state);
    expect(state.chosen).toContain("look_library_logc3");
    state = run([{ type: "toggle-older" }], state);
    expect(state.chosen).toEqual(["look_library_logc4"]);
  });

  it("lists the chosen packages in a sentence", () => {
    expect(joinList([])).toBe("");
    expect(joinList(["A"])).toBe("A");
    expect(joinList(["A", "B"])).toBe("A and B");
    expect(joinList(["A", "B", "C"])).toBe("A, B and C");
    expect(wantList(initialWizard())).toBe("Look Library · LogC4");
    expect(wantList({ ...initialWizard(), chosen: [] })).toBe("the packages you chose");
  });
});

describe("step locking", () => {
  it("locks every step ahead of the current one", () => {
    const state = initialWizard();
    expect([0, 1, 2, 3, 4].map((i) => stepLocked(state, i))).toEqual([false, true, true, true, true]);
  });

  it("lets the rail go back to a finished step but never forward", () => {
    const state = at(2, [pkg(LL4, "look_library_logc4")]);
    expect(run([{ type: "select", step: 0 }], state).step).toBe(0);
    expect(run([{ type: "select", step: 3 }], state).step).toBe(2);
    expect(run([{ type: "select", step: 2 }], state).step).toBe(2);
    expect(run([{ type: "back" }], state).step).toBe(1);
    expect(run([{ type: "back" }], initialWizard()).step).toBe(0);
  });

  it("moves on one step at a time, and only when the step allows it", () => {
    let state = run([{ type: "advance" }]);
    expect(state.step).toBe(1);
    // Nothing has arrived: Get the files stays.
    expect(canAdvance(state)).toBe(false);
    expect(run([{ type: "advance" }], state).step).toBe(1);
    state = run([{ type: "received", path: LL4 }], state);
    expect(canAdvance(state)).toBe(true);
    expect(run([{ type: "advance" }], state).step).toBe(2);
  });
});

describe("Get the files", () => {
  it("unlocks Next when a file has arrived for each chosen package", () => {
    let state: WizardState = { ...initialWizard(), step: 1, chosen: ["look_library_logc4", "display_luts_logc4"] };
    expect(footNote(state)).toBe("Next unlocks when all 2 files have arrived.");
    state = run([{ type: "received", path: LL4 }], state);
    expect(canAdvance(state)).toBe(false);
    state = run([{ type: "received", path: "~/Downloads/ARRI_DisplayLUTs_LogC4.zip" }], state);
    expect(canAdvance(state)).toBe(true);
    expect(footNote(state)).toBe("Everything has arrived.");
  });

  it("says Next unlocks when the zip arrives for one package", () => {
    expect(footNote({ ...initialWizard(), step: 1 })).toBe("Next unlocks when the zip arrives.");
  });

  it("takes a file once, however often it is dropped", () => {
    const state = run([{ type: "received", path: LL4 }, { type: "received", path: LL4 }]);
    expect(state.sources).toHaveLength(1);
  });

  it("forgets a file that is removed", () => {
    const state = run([{ type: "received", path: LL4 }, { type: "remove", path: LL4 }]);
    expect(state.sources).toEqual([]);
  });

  it("waits for the recogniser while a file is unchecked", () => {
    const state = run([{ type: "received", path: LL4 }]);
    expect(anyChecking(state)).toBe(true);
    expect(verdictFor(state, state.sources[0]!)).toEqual({ kind: "checking" });
    expect(anyChecking(run([{ type: "recognised", found: pkg(LL4, "look_library_logc4") }], state))).toBe(false);
  });
});

describe("Recognise", () => {
  it("matches a package that was chosen", () => {
    const state = at(2, [pkg(LL4, "look_library_logc4")]);
    const v = verdictFor(state, state.sources[0]!);
    expect(v).toMatchObject({ kind: "match", packageId: "look_library_logc4", contents: "87 .cube · 65-point" });
    expect(matchedPackages(state)).toEqual(["look_library_logc4"]);
    expect(unmatchedPackages(state)).toEqual([]);
    expect(recogniseBlocker(state)).toBeNull();
    expect(canAdvance(state)).toBe(true);
    expect(footNote(state)).toBe("1 package matches what you chose");
    expect(heading(state)).toBe("Found it");
  });

  it("says what the wrong package is and what to download instead", () => {
    const state = at(2, [pkg("~/Downloads/ARRI_LookLibrary_LogC3.zip", "look_library_logc3", "Look Library · LogC3")]);
    const v = verdictFor(state, state.sources[0]!);
    expect(v).toMatchObject({ kind: "other-package", packageId: "look_library_logc3" });
    expect(v.kind === "other-package" && v.explanation).toBe(
      "This is Look Library · LogC3, not Look Library · LogC4, which you chose. Download 'Look Library for LogC4 (3D-LUTs)' from ARRI's page, or go back and choose this one.",
    );
    expect(canAdvance(state)).toBe(false);
    expect(recogniseBlocker(state)).toBe("Look Library · LogC4 is missing. Go back and add it.");
    expect(heading(state)).toBe("Check what arrived");
  });

  it("explains an ALF4-only zip in the words of docs/08", () => {
    const reason = "These are ARRI look files for the camera (.alf4). Yellowmoon needs the 3D-LUT version: download 'Look Library for LogC4 (3D-LUTs)' from ARRI's look page.";
    const state = at(2, [unusable("~/Downloads/ARRI_LookFiles_ALF4.zip", reason)]);
    expect(verdictFor(state, state.sources[0]!)).toEqual({ kind: "unusable", explanation: reason });
    expect(canAdvance(state)).toBe(false);
  });

  it("reports a file that has gone missing", () => {
    const state = at(2, [missing("~/Downloads/ARRI_LookLibrary_LogC4.zip")]);
    expect(verdictFor(state, state.sources[0]!)).toMatchObject({ kind: "missing" });
    expect(canAdvance(state)).toBe(false);
    expect(footNote(state)).toBe("Look Library · LogC4 is missing. Go back and add it.");
  });

  it("imports the chosen package even when another file that did not work is listed too", () => {
    const state = at(2, [pkg(LL4, "look_library_logc4"), unusable("~/Downloads/notes.zip", "This zip has no .cube files, so there is nothing to import.")]);
    expect(canAdvance(state)).toBe(true);
    expect(importPlan(state)).toEqual([{ path: LL4, files: 87, kind: "package", packageId: "look_library_logc4" }]);
  });

  it("imports what is there when only some of the chosen packages arrived, and says what is not here", () => {
    const state = at(2, [pkg(LL4, "look_library_logc4")], ["look_library_logc4", "display_luts_logc4"]);
    expect(unmatchedPackages(state).map((p) => p.id)).toEqual(["display_luts_logc4"]);
    expect(recogniseBlocker(state)).toBeNull();
    expect(canAdvance(state)).toBe(true);
    expect(recogniseNote(state)).toBe("Display LUTs · LogC4 isn't here. You can import the rest now and add it later.");
    expect(footNote(state)).toBe("Display LUTs · LogC4 isn't here. You can import the rest now and add it later.");
    expect(heading(state)).toBe("Check what arrived");
    expect(importPlan(state)).toEqual([{ path: LL4, files: 87, kind: "package", packageId: "look_library_logc4" }]);
    const three = at(2, [pkg(LL4, "look_library_logc4")], ["look_library_logc4", "display_luts_logc4", "look_library_logc3"]);
    expect(canAdvance(three)).toBe(true);
    expect(footNote(three)).toBe("Some of the packages you chose aren't here. You can import the rest now and add them later.");
  });

  it("blocks Import only when there is nothing to import", () => {
    const none = at(2, [], ["look_library_logc4", "display_luts_logc4"]);
    expect(canAdvance(none)).toBe(false);
    expect(footNote(none)).toBe("Some packages are missing. Go back and add them.");
    const unusableOnly = at(2, [unusable("~/Downloads/notes.zip", "No cube files.")]);
    expect(canAdvance(unusableOnly)).toBe(false);
    expect(recogniseNote(unusableOnly)).toBeNull();
    expect(footNote(unusableOnly)).toBe("Look Library · LogC4 is missing. Go back and add it.");
    // Not even the wrong package is imported: nothing to import, so no note, only the lock.
    expect(canAdvance(at(2, [pkg("~/Downloads/ll3.zip", "look_library_logc3", "Look Library · LogC3")]))).toBe(false);
  });

  it("counts an answered set of unknown files for the chosen package with the same input", () => {
    const state = at(2, [unknownSet(LL4)]);
    // The names suggest LogC4 and the answer starts there, so the person who agrees only presses Import.
    expect(state.sources[0]?.answer).toBe("logc4");
    expect(standsInFor(state, state.sources[0]!)?.id).toBe("look_library_logc4");
    expect(verdictFor(state, state.sources[0]!)).toMatchObject({ kind: "question", input: "logc4", standsFor: "Look Library · LogC4" });
    expect(matchedPackages(state)).toEqual(["look_library_logc4"]);
    expect(unmatchedPackages(state)).toEqual([]);
    expect(recogniseBlocker(state)).toBeNull();
    expect(recogniseNote(state)).toBeNull();
    expect(canAdvance(state)).toBe(true);
    expect(footNote(state)).toBe("1 package matches what you chose");
    expect(heading(state)).toBe("Found it");
    expect(importPlan(state)).toEqual([{ path: LL4, input: "logc4", files: 87, kind: "cube" }]);
    // The same when the person picks the answer themselves, and for a folder.
    const picked = run([{ type: "answer", path: LL4, input: "logc4" }], at(2, [unknownSet(LL4, null)]));
    expect(canAdvance(picked)).toBe(true);
    expect(footNote(picked)).toBe("1 package matches what you chose");
    expect(canAdvance(at(2, [unknownSet("~/Downloads/ARRI looks", "logc4", "folder")]))).toBe(true);
    // And for the other two looks packages, by their own input.
    const logc3 = at(2, [unknownSet("~/Downloads/x.zip", "logc3_ei800")], ["look_library_logc3"]);
    expect(footNote(logc3)).toBe("1 package matches what you chose");
    const rec709 = at(2, [unknownSet("~/Downloads/x.zip", "logc3_to_rec709")], ["look_library_logc3_to_rec709"]);
    expect(footNote(rec709)).toBe("1 package matches what you chose");
  });

  it("does not count an answer for a package it is not the answer for", () => {
    // LogC3 answered, Look Library · LogC4 chosen: it can still be imported, but the chosen package is not here.
    const state = at(2, [unknownSet(LL4, "logc3_ei800")]);
    expect(standsInFor(state, state.sources[0]!)).toBeNull();
    expect(unmatchedPackages(state).map((p) => p.id)).toEqual(["look_library_logc4"]);
    expect(canAdvance(state)).toBe(true);
    expect(footNote(state)).toBe("Look Library · LogC4 isn't here. You can import the rest now and add it later.");
    // Display LUTs cannot be answered for: an answer is always about looks.
    const display = at(2, [unknownSet("~/Downloads/d.zip", "logc4")], ["display_luts_logc4"]);
    expect(standsInFor(display, display.sources[0]!)).toBeNull();
    expect(footNote(display)).toBe("Display LUTs · LogC4 isn't here. You can import the rest now and add it later.");
    // One loose .cube is not a set.
    const loose = at(2, [cube("~/Downloads/a.cube", "logc4")]);
    expect(standsInFor(loose, loose.sources[0]!)).toBeNull();
    expect(unmatchedPackages(loose).map((p) => p.id)).toEqual(["look_library_logc4"]);
    expect(canAdvance(loose)).toBe(true);
  });

  it("stays locked while an unknown set has no answer, and says so before anything else", () => {
    const state = at(2, [unknownSet(LL4, null)]);
    expect(state.sources[0]?.answer).toBeNull();
    expect(recogniseBlocker(state)).toBe("Say what the other LUTs expect as input.");
    expect(canAdvance(state)).toBe(false);
    expect(footNote(state)).toBe("Say what the other LUTs expect as input.");
    expect(run([{ type: "answer", path: LL4, input: "logc4" }], state)).toSatisfy((next: WizardState) => canAdvance(next));
    // With the chosen package also unmatched, the question still comes first.
    const both = at(2, [unknownSet(LL4, null), pkg("~/Downloads/dl.zip", "display_luts_logc4", "Display LUTs · LogC4", 22)], ["look_library_logc3", "display_luts_logc4"]);
    expect(recogniseBlocker(both)).toBe("Say what the other LUTs expect as input.");
  });

  it("does not let Import start while a file is still being checked", () => {
    const state = { ...at(2, [pkg(LL4, "look_library_logc4")]), checking: true };
    expect(recogniseBlocker(state)).toBe("Checking the files…");
    expect(canAdvance(state)).toBe(false);
  });

  it("asks the generic question for .cube files of unknown origin, and waits for the answer", () => {
    const state = at(2, [pkg(LL4, "look_library_logc4"), cube("~/Downloads/Warm_Film.cube")]);
    expect(verdictFor(state, state.sources[1]!)).toMatchObject({ kind: "question", input: null });
    expect(recogniseBlocker(state)).toBe("Say what the other LUTs expect as input.");
    expect(canAdvance(state)).toBe(false);
    const answered = run([{ type: "answer", path: "~/Downloads/Warm_Film.cube", input: "logc3_ei800" }], state);
    expect(recogniseBlocker(answered)).toBeNull();
    expect(canAdvance(answered)).toBe(true);
    expect(importPlan(answered)).toEqual([
      { path: LL4, files: 87, kind: "package", packageId: "look_library_logc4" },
      { path: "~/Downloads/Warm_Film.cube", input: "logc3_ei800", files: 1, kind: "cube" },
    ]);
  });

  it("starts from what the names suggest and keeps an answer when the file is looked at again", () => {
    let state = at(2, [cube("~/Downloads/a.cube", "logc4")]);
    expect(state.sources[0]?.answer).toBe("logc4");
    state = run([{ type: "answer", path: "~/Downloads/a.cube", input: "logc3_to_rec709" }, { type: "recognised", found: cube("~/Downloads/a.cube", "logc4") }], state);
    expect(state.sources[0]?.answer).toBe("logc3_to_rec709");
  });

  it("says the chosen package is not here when only a loose .cube arrived, and still lets it be imported", () => {
    const state = at(2, [cube("~/Downloads/a.cube", "logc4")]);
    expect(canAdvance(state)).toBe(true);
    expect(recogniseNote(state)).toBe("Look Library · LogC4 isn't here. You can import the rest now and add it later.");
  });
});

describe("Import and Finish", () => {
  it("counts files and waits for the copy before Next unlocks", () => {
    let state = run([{ type: "import-start", total: 87 }], at(3, [pkg(LL4, "look_library_logc4")]));
    expect(canAdvance(state)).toBe(false);
    expect(footNote(state)).toBe("This takes a few seconds");
    expect(heading(state)).toBe("Importing");
    state = run([{ type: "import-progress", done: 40 }], state);
    expect(state.import).toMatchObject({ done: 40, total: 87, status: "running" });
    state = run([{ type: "import-progress", done: 500 }], state);
    expect(state.import.done).toBe(87);
    state = run([{ type: "import-done", outcome: { looks: 87, displayLuts: 0, zips: [LL4] } }], state);
    expect(canAdvance(state)).toBe(true);
    expect(heading(state)).toBe("Imported");
    expect(footNote(state)).toBe("All files copied and checked");
  });

  it("says when the copy failed", () => {
    const state = run([{ type: "import-start", total: 87 }, { type: "import-failed", error: "The looks folder can't be written to." }], at(3, [pkg(LL4, "look_library_logc4")]));
    expect(state.import.status).toBe("failed");
    expect(heading(state)).toBe("Not imported");
    expect(footNote(state)).toBe("Something went wrong. Go back and try again.");
    expect(canAdvance(state)).toBe(false);
  });

  it("ignores progress that arrives when no import is running", () => {
    const state = initialWizard();
    expect(run([{ type: "import-progress", done: 5 }], state)).toBe(state);
  });

  it("names what was imported on the last step", () => {
    const base = at(4, [pkg(LL4, "look_library_logc4")]);
    expect(heading(base)).toBe("Done");
    const looks = run([{ type: "import-done", outcome: { looks: 87, displayLuts: 22, zips: [LL4] } }], base);
    expect(heading(looks)).toBe("87 looks ready");
    expect(lead(looks)).toBe("Here they are on your current photo. Pick a default and you are done.");
    expect(offersDisplayChoice(looks)).toBe(true);
    expect(footNote(looks)).toBe("You can change the default look any time");
    const displayOnly = run([{ type: "import-done", outcome: { looks: 0, displayLuts: 22, zips: [] } }], base);
    expect(heading(displayOnly)).toBe("22 display LUTs ready");
    const one = run([{ type: "import-done", outcome: { looks: 1, displayLuts: 0, zips: [] } }], base);
    expect(heading(one)).toBe("1 look ready");
    expect(offersDisplayChoice(one)).toBe(false);
  });

  it("starts with Neutral kept, ours as the display and the zip going to the Trash", () => {
    const state = initialWizard();
    expect(state.finish).toEqual({ defaultLook: null, display: "ours", zip: "trash" });
    expect(run([{ type: "finish", patch: { zip: "keep" } }], state).finish.zip).toBe("keep");
  });

  it("offers the Trash only for zips that are imported", () => {
    const state = at(4, [pkg(LL4, "look_library_logc4"), unusable("~/Downloads/notes.zip", "No cube files.")]);
    expect(trashCandidates(state)).toEqual([LL4]);
    const folder = at(4, [{ ...pkg("~/Downloads/ARRI_LookLibrary_LogC4", "look_library_logc4"), source: "folder" }]);
    expect(trashCandidates(folder)).toEqual([]);
  });
});

describe("copy", () => {
  it("has a heading and a lead for each step", () => {
    const headings = [0, 1, 2, 3].map((step) => heading({ ...at(step, [pkg(LL4, "look_library_logc4")]), step: step as WizardState["step"] }));
    expect(headings[0]).toBe("Choose what to import");
    expect(headings[1]).toBe("Download from ARRI");
    expect(lead(initialWizard())).toContain("ARRI publishes these looks for free.");
  });

  it("never shouts", () => {
    const texts = [lead(initialWizard()), footNote(initialWizard()), ...PACKAGES.map((p) => p.description)];
    for (const text of texts) expect(text).not.toContain("!");
  });
});

describe("preferredDisplayLut", () => {
  const lut = (id: string, input: "logc4" | "logc3_ei800", target: string) => ({ id, name: id, input, target });
  it("prefers LogC4, then Rec.709 at gamma 2.4", () => {
    expect(preferredDisplayLut(null)).toBeNull();
    expect(preferredDisplayLut([])).toBeNull();
    const luts = [lut("p3", "logc4", "p3-d65"), lut("c3", "logc3_ei800", "rec709-g24"), lut("g22", "logc4", "rec709-g22"), lut("g24", "logc4", "rec709-g24")];
    expect(preferredDisplayLut(luts)?.id).toBe("g24");
    expect(preferredDisplayLut(luts.slice(0, 3))?.id).toBe("g22");
    expect(preferredDisplayLut([lut("c3", "logc3_ei800", "rec709-g24"), lut("p3", "logc4", "p3-d65")])?.id).toBe("p3");
  });
});
