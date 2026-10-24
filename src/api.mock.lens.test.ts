/**
 * The lens on the mock API (`docs/21` §2 and §10): the setting, its clamping, partial patches, copy and paste, and `lens_list`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clampSettings, copiedPatch, createMockApi, defaultSettings, lensSettings, LENS_KINDS, MOCK_ESTIMATES, MOCK_LENSES, resolveSettings } from "./api.mock";
import type { MockApi } from "./api.mock";
import type { LensSettings } from "./api/types";

const ARW = ["~/Pictures/Lisbon/DSC04201.ARW", "~/Pictures/Lisbon/DSC04202.ARW"];
const NO_LENS: LensSettings = { kind: "default", flares: 0, edges: 0, anamorphic: 0 };

let api: MockApi;
beforeEach(() => {
  vi.useFakeTimers();
  api = createMockApi();
});
afterEach(() => {
  api.reset();
  vi.useRealTimers();
});

describe("the setting", () => {
  it("is the Default lens with all three amounts at 0 by default, and resolves as it is stored", () => {
    expect(defaultSettings().lens).toEqual(NO_LENS);
    const settings = { ...defaultSettings(), lens: { kind: "vintage", flares: 45, edges: 40, anamorphic: 10 } } as const;
    expect(resolveSettings(1, settings, MOCK_ESTIMATES).values.lens).toEqual(settings.lens);
  });

  it("is clamped as the core clamps it: 0 to 100, and an unknown lens is Default", () => {
    expect(clampSettings({ ...defaultSettings(), lens: { kind: "sonnar", flares: 140, edges: -5, anamorphic: 100 } }).lens).toEqual({
      kind: "sonnar",
      flares: 100,
      edges: 0,
      anamorphic: 100,
    });
    expect(lensSettings({ kind: "leica" as never, flares: 3 })).toEqual({ ...NO_LENS, flares: 3 });
    // Settings from before the lens have none.
    const { lens: _gone, ...older } = defaultSettings();
    expect(clampSettings(older as never).lens).toEqual(NO_LENS);
  });

  it("a patch may carry only the parts it changes", async () => {
    await api.openPaths(ARW);
    await api.updateSettings(1, { lens: { kind: "triplet" } as never });
    const rs = await api.updateSettings(1, { lens: { flares: 40 } as never });
    expect(rs.settings.lens).toEqual({ kind: "triplet", flares: 40, edges: 0, anamorphic: 0 });
    // What the card sends on choosing a lens: the kind and the starting amounts, in one patch.
    const [vintage] = MOCK_LENSES.filter((l) => l.kind === "vintage");
    const chosen = await api.updateSettings(1, { lens: { kind: "vintage", flares: vintage!.start_flares, edges: vintage!.start_edges, anamorphic: 25 } });
    expect(chosen.settings.lens).toEqual({ kind: "vintage", flares: vintage!.start_flares, edges: vintage!.start_edges, anamorphic: 25 });
    expect(chosen.values.lens).toEqual(chosen.settings.lens);
  });

  it("is copied with the rest and pasted onto other photos as one step; Re-run leaves it alone", async () => {
    await api.openPaths(ARW);
    api.flush();
    await api.updateSettings(1, { lens: { kind: "petzval", flares: 30, edges: 70, anamorphic: 5 } });
    const patch = await api.copySettings(1);
    expect(patch.lens).toEqual({ kind: "petzval", flares: 30, edges: 70, anamorphic: 5 });
    expect(copiedPatch(resolveSettings(1, { ...defaultSettings(), lens: { ...NO_LENS, flares: 9 } }, MOCK_ESTIMATES)).lens).toEqual({ ...NO_LENS, flares: 9 });
    const report = await api.pasteSettings([2], patch);
    expect(report.pasted).toHaveLength(1);
    expect(report.pasted[0]!.after).toEqual({ lens: { kind: "petzval", flares: 30, edges: 70, anamorphic: 5 } });
    expect(report.pasted[0]!.before).toEqual({ lens: NO_LENS });
    expect((await api.getSettings(2)).settings.lens).toEqual({ kind: "petzval", flares: 30, edges: 70, anamorphic: 5 });
    expect((await api.rerunAuto(1)).settings.lens).toEqual({ kind: "petzval", flares: 30, edges: 70, anamorphic: 5 });
  });
});

describe("lens_list", () => {
  it("answers the seven lenses in menu order, each with its name, line and starting amounts", async () => {
    const list = await api.lensList();
    expect(list.map((l) => l.kind)).toEqual([...LENS_KINDS]);
    expect(list.map((l) => l.kind)).toEqual(["default", "classic", "vintage", "petzval", "triplet", "sonnar", "detuned"]);
    for (const lens of list) {
      expect(lens.name.length).toBeGreaterThan(0);
      expect(lens.line).toMatch(/\.$/);
      expect(lens.start_flares).toBeGreaterThanOrEqual(0);
      expect(lens.start_flares).toBeLessThanOrEqual(100);
      expect(lens.start_edges).toBeGreaterThanOrEqual(0);
      expect(lens.start_edges).toBeLessThanOrEqual(100);
      expect(lens.version).toBeGreaterThanOrEqual(1);
    }
    // Default starts at 0 and 0: choosing it clears the lens.
    expect(list[0]).toMatchObject({ kind: "default", name: "Default", start_flares: 0, start_edges: 0 });
    expect(list[0]!.line).toBe("Clean and modern: even sharpness, faint glow.");
    // A copy: changing the answer does not change the next one.
    list[0]!.name = "Changed";
    expect((await api.lensList())[0]!.name).toBe("Default");
  });
});
