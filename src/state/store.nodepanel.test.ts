/**
 * The store's part in the node view's panel (`docs/23` §7.5, task N-09): the histogram, the scatter and Pick colour name the node they are
 * read at (`at`, `docs/01`), another corrector's grade shows at once, and a pin picked on a corrector goes into that corrector.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi } from "../api.mock";
import type { HistogramData, NodeLink, SettingsPatch } from "../api/types";
import { applyDirect, createStore } from "./store";
import type { AppStore } from "./store";
import { gradeOf, nodeOf, patchOf } from "./settingsGraph";
import type { CorrectorItem } from "./settingsGraph";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
});

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const TWO: SettingsPatch = {
  nodes: { "2": { kind: "corrector", name: "Warmth", pos: [1, 0], trim: { exposure_ev: 1 } } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
};

/** DSC04204 open (current), with node 2 after Main. */
async function opened(): Promise<{ api: MockApi; store: AppStore; id: number }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3, initialSettings: { DSC04204: TWO } });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
  await store.setCurrent(4);
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  return { api, store, id: 4 };
}

describe("the histogram at a node", () => {
  it("asks without a node for Main (the simple view), and with the node for another: `at` reaches the API", async () => {
    const { api, store, id } = await opened();
    const histogram = vi.spyOn(api, "histogram");
    const stop = store.watchHistogram();
    await vi.waitFor(() => expect(histogram).toHaveBeenCalledWith(id));
    stop();
    histogram.mockClear();
    const stopAt = store.watchHistogram({ at: 2 });
    await vi.waitFor(() => expect(histogram).toHaveBeenCalledWith(id, 2));
    stopAt();
    histogram.mockClear();
    const stopInput = store.watchHistogram({ readout: true, at: "input" });
    await vi.waitFor(() => expect(histogram).toHaveBeenCalledWith(id, "input"));
    stopInput();
  });

  it("is the node's: another node's answer, which arrives late, is not shown, and the new one is asked for", async () => {
    const { api, store, id } = await opened();
    const real = api.histogram.bind(api);
    const late = new Map<unknown, (data: HistogramData) => void>();
    const lateHistogram: HistogramData = { red: Array(64).fill(0.9), green: Array(64).fill(0.9), blue: Array(64).fill(0.9) };
    vi.spyOn(api, "histogram").mockImplementation((photoId, at) => {
      if (at === 2) return new Promise<HistogramData>((resolve) => late.set(at, resolve));
      return real(photoId, at);
    });
    const stopTwo = store.watchHistogram({ at: 2 });
    await vi.waitFor(() => expect(late.has(2)).toBe(true));
    // The panel moves to Main before node 2's answer is back.
    stopTwo();
    const stopMain = store.watchHistogram();
    late.get(2)?.(lateHistogram);
    await vi.waitFor(() => expect(store.getState().histogram).not.toBeNull());
    expect(store.getState().histogram).not.toEqual(lateHistogram);
    expect(store.getState().histogram).toEqual(await real(id));
    stopMain();
  });
});

describe("the scatter at a node", () => {
  it("asks without a node for Main and with the node for another", async () => {
    const { api, store, id } = await opened();
    const scatter = vi.spyOn(api, "chromaScatter");
    const stop = store.watchChromaScatter();
    await vi.waitFor(() => expect(scatter).toHaveBeenCalledWith(id));
    stop();
    scatter.mockClear();
    const stopAt = store.watchChromaScatter({ at: 2 });
    await vi.waitFor(() => expect(scatter).toHaveBeenCalledWith(id, 2));
    stopAt();
    expect(store.getState().chromaScatter?.points.length).toBeGreaterThan(0);
  });
});

describe("Pick colour on a node", () => {
  it("reads the chroma at that node's warp and puts the moving pin into that node, selected, as one undo step", async () => {
    const { api, store } = await opened();
    await api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    const sample = vi.spyOn(api, "sampleChroma");
    await store.pickColor(450, 100, 2);
    expect(sample).toHaveBeenCalledWith(450, 100, 2);
    const settings = store.currentSettings()!.settings;
    expect(gradeOf(settings, 2).color_warp.pins).toHaveLength(1);
    expect(gradeOf(settings, 1).color_warp.pins).toHaveLength(0);
    expect(store.getState().warpPin).toBe(0);
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    expect(gradeOf(store.currentSettings()!.settings, 2).color_warp.pins).toHaveLength(0);
    // Main's is as before nodes: no node in the call.
    sample.mockClear();
    await store.pickColor(450, 100);
    expect(sample).toHaveBeenCalledWith(450, 100);
    expect(gradeOf(store.currentSettings()!.settings, 1).color_warp.pins).toHaveLength(1);
    expect(gradeOf(store.currentSettings()!.settings, 2).color_warp.pins).toHaveLength(0);
  });
});

describe("another corrector's grade shows at once (the node view's sliders)", () => {
  it("is in the mirror before the core answers, as a plain number, and Auto in it is a manual 0", async () => {
    const { store } = await opened();
    const rs = store.currentSettings()!;
    const next = applyDirect(rs, patchOf(2, { grade: { brightness: 20, white: -10, contrast: { kind: "manual", value: 15 }, saturation: { kind: "auto" } } }));
    const stored = (nodeOf(next.settings, 2) as CorrectorItem).grade;
    expect([stored.brightness, stored.white, stored.contrast, stored.saturation]).toEqual([20, -10, { kind: "manual", value: 15 }, { kind: "manual", value: 0 }]);
    const resolved = next.values.nodes.find((n) => n.id === 2)!.grade;
    expect([resolved.brightness, resolved.white, resolved.contrast, resolved.saturation]).toEqual([20, -10, 15, 0]);
    // Main's Auto flags are not touched by it, and Main's grade is as it was.
    expect(next.auto).toBe(rs.auto);
    expect(next.values.main).toBe(rs.values.main);
    // A wheel's Auto is a manual point at the centre, the midtones wheel is the point.
    const wheels = applyDirect(rs, patchOf(2, { grade: { color_balance: { shadows: { kind: "auto" }, midtones: [0.1, 0.2] } } }));
    expect((nodeOf(wheels.settings, 2) as CorrectorItem).grade.color_balance).toEqual({ shadows: { kind: "manual", point: [0, 0] }, midtones: [0.1, 0.2], highlights: { kind: "manual", point: [0, 0] } });
    expect(wheels.values.nodes.find((n) => n.id === 2)!.grade.color_balance.midtones).toEqual([0.1, 0.2]);
  });

  it("shows the trim and the curve at once too, and leaves a node the page has no values for to the core's reply", async () => {
    const { store } = await opened();
    const rs = store.currentSettings()!;
    const curve = { luma: [[0, 0], [0.5, 0.6], [1, 1]] as Array<[number, number]> };
    const next = applyDirect(rs, patchOf(2, { trim: { temperature: 40 }, grade: { curve } }));
    expect((nodeOf(next.settings, 2) as CorrectorItem).trim).toEqual({ exposure_ev: 1, temperature: 40, tint: 0 });
    expect(next.values.nodes.find((n) => n.id === 2)!.grade.curve.luma).toEqual(curve.luma);
    // A node the resolved values do not have: its stored grade is not changed under values that cannot follow.
    const without = { ...rs, values: { ...rs.values, nodes: [] } };
    const same = applyDirect(without, patchOf(2, { grade: { brightness: 20 } }));
    expect((nodeOf(same.settings, 2) as CorrectorItem).grade.brightness).toBe(0);
  });
});
