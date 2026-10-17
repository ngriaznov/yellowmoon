/**
 * The store's part in Colour › Colour warp (`docs/11` §Colour warp): `color_warp` shows at once and a drag of pins is one undo step,
 * the selected pin is window state, the scatter is asked for while the plane is on show, and Pick colour adds a pin from a click.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi } from "../api.mock";
import { lockedPin, movingPin, WARP_MAX_PINS } from "../components/warpMath";
import { hexChroma } from "../fixtures/warp";
import { SCENES } from "../fixtures/scenes";
import { applyDirect, createStore, HISTOGRAM_THROTTLE_MS } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
  vi.useRealTimers();
});

/** The twelve photos open and developed, DSC04204 current (the mockup's photo), its settings loaded. */
async function opened(): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3 });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
  await store.setCurrent(4);
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  return { api, store };
}

const pinsOf = (store: AppStore) => store.currentSettings()?.settings.color_warp.pins ?? [];
const SKY = hexChroma(SCENES[3]!.sky);

describe("color_warp in the store", () => {
  it("shows a patch at once, before the core answers", async () => {
    const { store } = await opened();
    const rs = store.currentSettings()!;
    const pins = [movingPin([0.1, 0], [0.1, 0.05])];
    const next = applyDirect(rs, { color_warp: { pins } });
    expect(next.settings.color_warp.pins).toEqual(pins);
    expect(next.values.color_warp.pins).toEqual(pins);
  });

  it("a drag of a pin is one undo step: frames without a step, the release with one", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "updateSettings");
    const at = (b: number) => ({ color_warp: { pins: [movingPin([0.1, 0], [0.1, b])] } });
    void store.update(at(0), false);
    void store.update(at(0.02), false);
    void store.update(at(0.04), false);
    await store.update(at(0.05), true);
    expect(spy.mock.calls.length).toBeGreaterThanOrEqual(1);
    expect(pinsOf(store)).toEqual([movingPin([0.1, 0], [0.1, 0.05])]);
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    expect(pinsOf(store)).toEqual([]);
    expect(store.getState().canUndo).toBe(false);
    await store.redo();
    expect(pinsOf(store)).toEqual([movingPin([0.1, 0], [0.1, 0.05])]);
  });

  it("keeps the selected pin as window state: not a setting, and none on another photo", async () => {
    const { store } = await opened();
    await store.update({ color_warp: { pins: [movingPin([0.1, 0])] } });
    store.selectWarpPin(0);
    expect(store.getState().warpPin).toBe(0);
    expect(store.currentSettings()?.settings).not.toHaveProperty("warpPin");
    await store.setCurrent(5);
    expect(store.getState().warpPin).toBeNull();
  });
});

describe("the scatter", () => {
  it("is asked for while watched, again after frames at most every 200 ms, and dropped with the photo", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "chromaScatter");
    const stop = store.watchChromaScatter();
    await vi.waitFor(() => expect(store.getState().chromaScatter?.points.length).toBeGreaterThan(0));
    expect(spy).toHaveBeenCalledWith(4);
    vi.useFakeTimers();
    spy.mockClear();
    api.emitEvent("render_stats", { ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null });
    api.emitEvent("render_stats", { ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null });
    await vi.advanceTimersByTimeAsync(HISTOGRAM_THROTTLE_MS + 10);
    expect(spy).toHaveBeenCalledTimes(1);
    stop();
    spy.mockClear();
    api.emitEvent("render_stats", { ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null });
    await vi.advanceTimersByTimeAsync(HISTOGRAM_THROTTLE_MS + 10);
    expect(spy).not.toHaveBeenCalled();
    vi.useRealTimers();
    await store.setCurrent(5);
    expect(store.getState().chromaScatter).toBeNull();
  });
});

describe("Pick colour", () => {
  it("adds a moving pin at the colour under the click, its target on its source, selected, as one undo step", async () => {
    const { api, store } = await opened();
    store.setViewportRect({ x: 0, y: 0, width: 1500, height: 1000, scale: 1 });
    const sample = vi.spyOn(api, "sampleChroma");
    await store.pickColor(20, 20);
    expect(sample).toHaveBeenCalledWith(20, 20);
    expect(pinsOf(store)).toEqual([movingPin(SKY)]);
    expect(store.getState().warpPin).toBe(0);
    // A second pick adds a second pin and selects it.
    await store.pickColor(750, 950);
    expect(pinsOf(store)).toHaveLength(2);
    expect(store.getState().warpPin).toBe(1);
    await store.undo();
    expect(pinsOf(store)).toEqual([movingPin(SKY)]);
  });

  it("adds nothing on the paper around a letterboxed photo, nor at sixteen pins (where the selection goes)", async () => {
    const { store } = await opened();
    await store.update({ orientation: 90 });
    store.setViewportRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    await store.pickColor(20, 300);
    expect(pinsOf(store)).toEqual([]);
    const full = Array.from({ length: WARP_MAX_PINS }, (_, i) => lockedPin([0.2 * Math.cos(i), 0.2 * Math.sin(i)]));
    await store.update({ color_warp: { pins: full } });
    store.selectWarpPin(3);
    await store.pickColor(450, 300);
    expect(pinsOf(store)).toHaveLength(WARP_MAX_PINS);
    expect(store.getState().warpPin).toBeNull();
  });

  it("shows the core's words when it can't sample, and changes nothing", async () => {
    const { api, store } = await opened();
    api.failNext("sampleChroma", { scope: "viewport", message: "Open a photo first." });
    await store.pickColor(20, 20);
    expect(store.getState().error?.message).toBe("Open a photo first.");
    expect(pinsOf(store)).toEqual([]);
    expect(store.getState().canUndo).toBe(false);
  });
});
