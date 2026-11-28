/**
 * The Reference Grade commands on the mock API (`docs/34` §9.5): a stand-in reference held per node, the errors and notes a file name asks for, a fixed fit with
 * `fit_progress` events, Cancel that changes nothing, and the rules of the core the page relies on (the node must be a Reference Grade node, the photo cannot be
 * its own reference). The contract with the shell is `api.test.ts` and the Rust `bindings` test.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "./api.mock";
import type { MockApi } from "./api.mock";
import type { AppError, FitProgress, NodeLink, ReferenceGradeNode, SettingsPatch } from "./api/types";
import { nodeOf } from "./state/settingsGraph";
import type { ReferenceGradeItem } from "./state/settingsGraph";

const ARW = ["~/Pictures/Lisbon/DSC04201.ARW", "~/Pictures/Lisbon/DSC04202.ARW", "~/Pictures/Lisbon/DSC04203.ARW"];
const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const ADD: SettingsPatch = { nodes: { "2": { kind: "reference_grade", name: "Match", pos: [1, 0] } }, links: [picture("input", 1), picture(1, 2), picture(2, "output")] };

let api: MockApi;
beforeEach(async () => {
  vi.useFakeTimers();
  api = createMockApi();
  await api.openPaths(ARW);
  api.flush();
  await api.updateSettings(1, ADD);
});
afterEach(() => {
  api.reset();
  vi.useRealTimers();
});

const rejection = async (promise: Promise<unknown>): Promise<AppError> => {
  try {
    await promise;
  } catch (error) {
    return error as AppError;
  }
  throw new Error("expected a rejection");
};
const node = async (): Promise<ReferenceGradeItem> => nodeOf((await api.getSettings(1)).settings, 2) as ReferenceGradeItem;

describe("Match brightness", () => {
  it("is a stored field that starts at 0 and is held to 0 to 100 like the core", async () => {
    expect((await node()).match_brightness).toBe(0);
    await api.updateSettings(1, { nodes: { "2": { match_brightness: 140 } } });
    expect((await node()).match_brightness).toBe(100);
    await api.updateSettings(1, { nodes: { "2": { match_brightness: -5 } } });
    expect((await node()).match_brightness).toBe(0);
    await api.updateSettings(1, { nodes: { "2": { match_brightness: 35 } } });
    expect((await node()).match_brightness).toBe(35);
  });
});

describe("choosing a reference", () => {
  it("holds a file for the node and answers its label, notes and a picture; the settings do not name it", async () => {
    api.queueChoice("reference", "~/Pictures/harbour-dusk.jpg");
    const info = await api.referenceChooseFile(1, 2);
    expect(info).toMatchObject({ photo_id: 1, node_id: 2, label: "harbour-dusk.jpg", source: "file", notes: [], grey: false });
    expect(info?.thumbnail.startsWith("data:image/svg+xml")).toBe(true);
    expect(JSON.stringify((await api.getSettings(1)).settings)).not.toContain("harbour");
  });

  it("answers null when the panel is cancelled, and a name asks for notes", async () => {
    api.queueChoice("reference", null);
    expect(await api.referenceChooseFile(1, 2)).toBeNull();
    api.queueChoice("reference", "~/Pictures/bars-profile-grey.png");
    const info = await api.referenceChooseFile(1, 2);
    expect(info?.notes).toEqual([
      "Black bars were removed from the reference.",
      "The picture's colour profile couldn't be read, so it is taken as sRGB.",
      "This picture has no colour. The photo will be matched in black and white.",
    ]);
    expect(info?.grey).toBe(true);
  });

  it.each([
    ["damaged.png", "This file can't be read. It may be damaged. Choose another picture."],
    ["photo.HEIC", "Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again."],
    ["tiny.png", "This picture is too small to match. Use one that is at least 256 pixels on its short side."],
    ["empty.png", "There's no picture inside the black bars. Choose another picture."],
  ])("refuses %s with the core's sentence", async (file, message) => {
    api.queueChoice("reference", `~/Pictures/${file}`);
    expect(await rejection(api.referenceChooseFile(1, 2))).toEqual({ scope: "render", message });
  });

  it("refuses a node that is not a Reference Grade node", async () => {
    expect((await rejection(api.referenceChooseFile(1, 1))).message).toBe("Node 1 is not a Reference Grade node.");
  });

  it("takes a photo of the strip but not the photo itself, and not one that is not ready", async () => {
    expect(await api.referenceChoosePhoto(1, 2, 2, "Photo 2 of 3")).toMatchObject({ label: "Photo 2 of 3", source: "photo" });
    expect(await rejection(api.referenceChoosePhoto(1, 2, 1, "Photo 1 of 3"))).toEqual({ scope: "render", message: "That's the photo you're grading. Choose another." });
  });
});

describe("the fit", () => {
  it("needs a reference", async () => {
    expect((await rejection(api.fitReference(1, 2))).message).toBe("Choose a reference first.");
  });

  it("reports its steps, then writes the fixed fit as one patch of the node's fitted fields and answers the settings and the shares", async () => {
    await api.referenceChooseFile(1, 2);
    const steps: FitProgress[] = [];
    api.on("fit_progress", (e) => steps.push(e));
    const before = await node();
    const running = api.fitReference(1, 2);
    await vi.advanceTimersByTimeAsync(10_000);
    const report = await running;
    expect(steps.map((s) => [s.step, s.search])).toEqual([["scene", 0], ["search", 1], ["search", 2], ["search", 3]]);
    expect(steps.every((s) => s.photo_id === 1 && s.node_id === 2 && s.of === 3)).toBe(true);
    expect(steps.map((s) => s.fraction)).toEqual([...steps.map((s) => s.fraction)].sort((a, b) => a - b));
    expect(report.status).toBe("fitted");
    expect(report.shares.map((s) => s.element)).toEqual(["sky", "vegetation", "water", "ground", "built", "skin"]);
    expect(report.shares.find((s) => s.element === "water")!.share).toBeLessThan(0.02);
    const after = nodeOf(report.settings.settings, 2) as ReferenceGradeItem;
    expect(after.hue_curves.hue_vs_hue).toHaveLength(6);
    expect(after.hue_curves.hue_vs_sat).toHaveLength(6);
    expect(after.elements.water).toEqual({ exposure_ev: 0, hue_deg: 0, chroma: 1 });
    expect(after.grade.saturation).toEqual({ kind: "manual", value: -12 });
    // What the fit does not write stays: the name, Strength, Skin protection, the trim and the masks.
    expect([after.name, after.strength, after.skin_protection, after.match_brightness, after.trim, after.masks]).toEqual([before.name, before.strength, before.skin_protection, before.match_brightness, before.trim, before.masks]);
    expect((await api.getSettings(1)).settings).toEqual(report.settings.settings);
  });

  it("is the same fit every time, whatever the reference", async () => {
    await api.referenceChooseFile(1, 2);
    const first = api.fitReference(1, 2);
    await vi.advanceTimersByTimeAsync(10_000);
    const one = (nodeOf((await first).settings.settings, 2) as ReferenceGradeNode).elements;
    api.queueChoice("reference", "~/Pictures/other.png");
    await api.referenceChooseFile(1, 2);
    const second = api.fitReference(1, 2);
    await vi.advanceTimersByTimeAsync(10_000);
    expect((nodeOf((await second).settings.settings, 2) as ReferenceGradeNode).elements).toEqual(one);
  });

  it("a cancel answers cancelled and changes nothing", async () => {
    await api.referenceChooseFile(1, 2);
    const before = JSON.stringify((await api.getSettings(1)).settings);
    const running = api.fitReference(1, 2);
    await vi.advanceTimersByTimeAsync(500);
    await api.cancelFit(1, 2);
    const report = await running;
    expect(report.status).toBe("cancelled");
    expect(report.shares).toEqual([]);
    expect(JSON.stringify(report.settings.settings)).toBe(before);
    expect(JSON.stringify((await api.getSettings(1)).settings)).toBe(before);
    // A cancel with no fit is nothing.
    await expect(api.cancelFit(1, 2)).resolves.toBeUndefined();
  });

  it("clear_reference lets go of the reference, so a fit needs a new one", async () => {
    await api.referenceChooseFile(1, 2);
    await api.clearReference(1, 2);
    expect((await rejection(api.fitReference(1, 2))).message).toBe("Choose a reference first.");
  });

  it("an error of the next fit can be injected, as for every command", async () => {
    await api.referenceChooseFile(1, 2);
    api.failNext("fitReference", { scope: "render", message: "The grade couldn't be fitted. Try another reference." });
    expect((await rejection(api.fitReference(1, 2))).message).toBe("The grade couldn't be fitted. Try another reference.");
  });
});

describe("the node in the resolved settings (resolvedGraded)", () => {
  it("has an entry in values.nodes with its trim and grade, as Colour Uniform has", async () => {
    const rs = await api.getSettings(1);
    expect(rs.values.nodes.map((n) => n.id)).toContain(2);
    const entry = rs.values.nodes.find((n) => n.id === 2)!;
    expect(entry.trim).toEqual({ exposure_ev: 0, temperature: 0, tint: 0 });
    expect(entry.grade.saturation).toBe(0);
  });
});
