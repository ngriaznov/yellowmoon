/**
 * The store's part in the Mask tab (`docs/23` §7.6): a mask edit shows at once, the selected row is window state, the kinds of mask are asked
 * for once, the points are mapped by the core, and Pick colour of a Colour range sets a row's centre from the chroma at its node.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi } from "../api.mock";
import type { MaskItem, SettingsPatch } from "../api/types";
import { newMask } from "./maskRows";
import { applyDirect, createStore } from "./store";
import type { AppStore } from "./store";
import { nodeOf, patchOf } from "./settingsGraph";
import type { CorrectorItem } from "./settingsGraph";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
  vi.useRealTimers();
});

const picture = (from: number | "input", to: number | "output") => ({ from: { node: from, port: "picture" as const }, to: { node: to, port: "picture" as const } });
const RADIAL = { ...newMask("radial"), center: [0.3, 0.4] } as MaskItem;
const SETTINGS: SettingsPatch = {
  nodes: { "2": { kind: "corrector", name: "Skin", pos: [1, 0], masks: { invert: false, items: [RADIAL, newMask("color_range")] } } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
};

/** DSC04204 current, with a node 2 that has a Radial and a Colour range; the cell reported as 600 × 400. */
async function opened(options: Parameters<typeof createMockApi>[0] = {}): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3, initialSettings: { DSC04204: SETTINGS }, ...options });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
  await store.setCurrent((await api.sessionState()).photos[3]!.id);
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  await api.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
  return { api, store };
}

const rowsOf = (store: AppStore, node = 2): MaskItem[] => (nodeOf(store.currentSettings()!.settings, node) as CorrectorItem).masks.items;

describe("a mask edit shows at once", () => {
  it("applyDirect puts a row patch, the node's Invert and a new list of rows into the stored and the resolved masks, of Main and of any other corrector", async () => {
    const { store } = await opened();
    const rs = store.currentSettings()!;
    const edited = applyDirect(rs, patchOf(2, { masks: { edit: { "0": { feather: 0.9 } }, invert: true } }));
    expect((nodeOf(edited.settings, 2) as CorrectorItem).masks).toMatchObject({ invert: true, items: [{ kind: "radial", feather: 0.9 }, { kind: "color_range" }] });
    expect(edited.values.nodes.find((n) => n.id === 2)?.masks).toMatchObject({ invert: true, items: [{ feather: 0.9 }, { kind: "color_range" }] });
    // Main too (its own `values`).
    const main = applyDirect(rs, patchOf(1, { masks: { items: [newMask("linear")] } }));
    expect((nodeOf(main.settings, 1) as CorrectorItem).masks.items).toEqual([newMask("linear")]);
    expect(main.values.main.masks.items).toEqual([newMask("linear")]);
    // A patch that has no masks part leaves them alone.
    expect((nodeOf(applyDirect(rs, patchOf(2, { name: "Faces" })).settings, 2) as CorrectorItem).masks).toEqual((nodeOf(rs.settings, 2) as CorrectorItem).masks);
  });

  it("the update shows before the core answers, and the core's answer is the same", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "updateSettings");
    const sent = store.update(patchOf(2, { masks: { edit: { "1": { width: 0.2 } } } }), true);
    expect(rowsOf(store)[1]).toMatchObject({ width: 0.2 });
    await sent;
    expect(spy).toHaveBeenCalledTimes(1);
    expect(rowsOf(store)[1]).toMatchObject({ width: 0.2 });
    await store.undo();
    expect(rowsOf(store)[1]).toMatchObject({ width: 0.05 });
  });
});

describe("the selected mask row", () => {
  it("is window state: not a setting, and none on another photo", async () => {
    const { api, store } = await opened();
    const before = store.currentSettings();
    store.selectMaskRow(1);
    expect(store.getState().maskRow).toBe(1);
    expect(store.currentSettings()).toBe(before);
    expect(store.getState().canUndo).toBe(false);
    store.selectMaskRow(1);
    store.selectMaskRow(null);
    expect(store.getState().maskRow).toBeNull();
    store.selectMaskRow(0);
    await store.setCurrent((await api.sessionState()).photos[4]!.id);
    expect(store.getState().maskRow).toBeNull();
  });
});

describe("the kinds of mask", () => {
  it("are asked for once, and what the core says is what the store holds", async () => {
    const { api, store } = await opened({ maskKinds: ["linear", "radial", "brush"] });
    const spy = vi.spyOn(api, "maskKinds");
    expect(store.getState().maskKinds).toBeNull();
    store.ensureMaskKinds();
    store.ensureMaskKinds();
    await vi.waitFor(() => expect(store.getState().maskKinds).toEqual(["linear", "radial", "brush"]));
    store.ensureMaskKinds();
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("a failure is told, and asked for again next time", async () => {
    const { api, store } = await opened();
    api.failNext("maskKinds", { scope: "viewport", message: "No." });
    store.ensureMaskKinds();
    await vi.waitFor(() => expect(store.getState().error?.message).toBe("No."));
    expect(store.getState().maskKinds).toBeNull();
    store.ensureMaskKinds();
    await vi.waitFor(() => expect(store.getState().maskKinds).not.toBeNull());
  });
});

describe("the core maps the points", () => {
  it("answers one place per point, and null when the core could not", async () => {
    const { api, store } = await opened();
    expect(await store.mapPoints([[0.5, 0.5], [0, 0]], "photo")).toEqual([[300, 200], [0, 0]]);
    expect(await store.mapPoints([[300, 200]], "cell")).toEqual([[0.5, 0.5]]);
    api.failNext("viewportMapPoints", { scope: "viewport", message: "No." });
    expect(await store.mapPoints([[0.5, 0.5]], "photo")).toBeNull();
    // No error bar for it: the next frame asks again.
    expect(store.getState().error).toBeNull();
  });
});

describe("Pick colour of a Colour range", () => {
  it("sets the row's centre from the chroma at its node, held to 0.5 from the middle, as one undo step", async () => {
    const { api, store } = await opened();
    const sample = vi.spyOn(api, "sampleChroma");
    await store.pickRange(300, 20, 2, 1);
    expect(sample).toHaveBeenCalledWith(300, 20, 2);
    const there = (await api.sampleChroma(300, 20, 2))!;
    const scale = Math.min(1, 0.5 / Math.hypot(there.a, there.b));
    const center = (rowsOf(store)[1] as Extract<MaskItem, { kind: "color_range" }>).center;
    expect(center[0]).toBeCloseTo(there.a * scale, 9);
    expect(center[1]).toBeCloseTo(there.b * scale, 9);
    expect(center).not.toEqual([0, 0]);
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    expect(rowsOf(store)[1]).toMatchObject({ center: [0, 0] });
  });

  it("changes nothing for a click on the paper, or for a row that is not a Colour range", async () => {
    const { api, store } = await opened();
    // A cell wider than the 3:2 photo has paper at its sides: the mock's picture is 3:2, so a cell of 3:4 has paper above and below.
    await api.viewportSetRect({ x: 0, y: 0, width: 300, height: 400, scale: 1 });
    await store.pickRange(150, 5, 2, 1);
    expect(rowsOf(store)[1]).toMatchObject({ center: [0, 0] });
    expect(store.getState().canUndo).toBe(false);
    // Row 0 is a Radial.
    await store.pickRange(150, 200, 2, 0);
    expect(rowsOf(store)[0]).toEqual(RADIAL);
    // A row past the end.
    await store.pickRange(150, 200, 2, 7);
    expect(store.getState().canUndo).toBe(false);
  });
});
