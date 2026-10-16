/**
 * The store's side of the performance work (docs/15 M3-16 to M3-19): wheel, pinch and drag input collected per frame, the histogram
 * asked for at most five times a second, and the look tiles asked for two at a time.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import type { RenderStats, ViewportInput } from "../api/types";
import { installManualRaf } from "../test/raf";
import type { ManualRaf } from "../test/raf";
import { isSuperseded } from "./requestLimiter";
import { createStore, HISTOGRAM_THROTTLE_MS, LOOK_THUMBNAIL_CONCURRENCY } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
let raf: ManualRaf | undefined;
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
  raf?.restore();
  raf = undefined;
  vi.useRealTimers();
});

/** A store with the twelve photos open and developed, and the current photo's settings loaded. */
async function opened(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, ...options });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => {
    const photos = store.getState().session?.photos ?? [];
    expect(photos).toHaveLength(12);
    expect(photos.every((p) => p.status.kind === "ready" || p.status.kind === "skipped")).toBe(true);
  });
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  return { api, store };
}

const STATS: RenderStats = { ms: 12, path: "neural", space: "srgb", zoom_percent: null, fit_percent: null };

/** Inputs the core has been sent, in order. */
const inputsOf = (api: MockApi): ViewportInput[] => [...api.viewport.inputs];

describe("wheel, pinch and drag input", () => {
  it("sends a hundred wheel events and one animation frame as one viewport_input with the sum", async () => {
    raf = installManualRaf();
    const { api, store } = await opened({ current: 3 });
    const sent = vi.spyOn(api, "viewportInput");
    for (let i = 0; i < 100; i++) store.queueViewportInput({ kind: "pan", dx: -1, dy: -2 });
    expect(raf.pending).toBe(1);
    expect(sent).not.toHaveBeenCalled();
    raf.flush();
    await store.flushViewportInput();
    expect(sent).toHaveBeenCalledTimes(1);
    expect(inputsOf(api)).toEqual([{ kind: "pan", dx: -100, dy: -200 }]);
  });

  it("moves the zoom mirror once per frame, not once per event, by the factor the core was asked for", async () => {
    raf = installManualRaf();
    const { store } = await opened({ current: 3 });
    store.setViewportRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    // How many times the zoom was replaced (other news, such as a status line, may also arrive meanwhile).
    let moved = 0;
    let seen = store.getState().zoom;
    const stop = store.subscribe(() => {
      if (store.getState().zoom !== seen) moved += 1;
      seen = store.getState().zoom;
    });
    const factor = Math.exp(0.05);
    for (let i = 0; i < 40; i++) store.queueViewportInput({ kind: "zoom", factor, x: 300, y: 200 });
    expect(store.getState().zoom).toEqual({ kind: "fit" });
    expect(moved).toBe(0);
    raf.flush();
    await store.flushViewportInput();
    stop();
    expect(moved).toBe(1);
    const zoom = store.getState().zoom;
    expect(zoom.kind).toBe("percent");
    // 40 steps of exp(0.05) from the Fit zoom (about 24 % for this cell): the product, once.
    const fit = store.fitPercentExact() ?? 0;
    expect(zoom.kind === "percent" && zoom.value).toBeCloseTo(Math.min(800, fit * Math.exp(2)), 0);
  });

  it("keeps pans and zooms in the order they came, and merges only what sits next to each other", async () => {
    raf = installManualRaf();
    const { api, store } = await opened({ current: 3 });
    store.setViewportRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    store.queueViewportInput({ kind: "zoom", factor: 2, x: 100, y: 100 });
    store.queueViewportInput({ kind: "pan", dx: 4, dy: 0 });
    store.queueViewportInput({ kind: "pan", dx: 6, dy: 1 });
    store.queueViewportInput({ kind: "zoom", factor: 1.5, x: 100, y: 100 });
    raf.flush();
    await store.flushViewportInput();
    expect(inputsOf(api)).toEqual([
      { kind: "zoom", factor: 2, x: 100, y: 100 },
      { kind: "pan", dx: 10, dy: 1 },
      { kind: "zoom", factor: 1.5, x: 100, y: 100 },
    ]);
  });

  it("sends at once when a drag ends (flushViewportInput), without waiting for the frame", async () => {
    raf = installManualRaf();
    const { api, store } = await opened({ current: 3 });
    store.queueViewportInput({ kind: "pan", dx: 3, dy: 4 });
    await store.flushViewportInput();
    expect(inputsOf(api)).toEqual([{ kind: "pan", dx: 3, dy: 4 }]);
    // The frame that was asked for finds nothing left.
    raf.flush();
    await store.flushViewportInput();
    expect(inputsOf(api)).toHaveLength(1);
  });

  it("sends a double click after the input that came before it, and mirrors the zoom the wheel left", async () => {
    raf = installManualRaf();
    const { api, store } = await opened({ current: 3 });
    store.setViewportRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    store.queueViewportInput({ kind: "zoom", factor: 8, x: 50, y: 50 });
    // No frame has run: the click is a click on the picture as the wheel left it.
    await store.viewportInput({ kind: "toggle_zoom", x: 10, y: 10 });
    expect(inputsOf(api).map((i) => i.kind)).toEqual(["zoom", "toggle_zoom"]);
    // The mirror had followed the zoom, so the double click went back to Fit.
    expect(store.getState().zoom).toEqual({ kind: "fit" });
  });

  it("sends Pick white after the pan that came before it", async () => {
    raf = installManualRaf();
    const { api, store } = await opened({ current: 3 });
    store.setViewportRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    store.queueViewportInput({ kind: "pan", dx: 9, dy: 9 });
    await store.pickWhite(450, 300);
    expect(inputsOf(api).map((i) => i.kind)).toEqual(["pan", "pick_white"]);
  });

  it("sends a zoom that is still waiting before a zoom mode chosen from the menu", async () => {
    raf = installManualRaf();
    const { api, store } = await opened({ current: 3 });
    store.setViewportRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    const order: string[] = [];
    vi.spyOn(api, "viewportInput").mockImplementation(async (e) => void order.push(`input:${e.kind}`));
    vi.spyOn(api, "viewportSetMode").mockImplementation(async (m) => void order.push(`mode:${m.kind}`));
    store.queueViewportInput({ kind: "zoom", factor: 3, x: 10, y: 10 });
    await store.setZoom({ kind: "fit" });
    expect(order).toEqual(["input:zoom", "mode:fit"]);
    expect(store.getState().zoom).toEqual({ kind: "fit" });
  });

  it("drops what has not been sent when the photo changes, and when the session closes", async () => {
    raf = installManualRaf();
    const { api, store } = await opened({ current: 3 });
    store.queueViewportInput({ kind: "pan", dx: 5, dy: 5 });
    await store.setCurrent(5);
    raf.flush();
    await store.flushViewportInput();
    expect(inputsOf(api)).toEqual([]);

    store.queueViewportInput({ kind: "pan", dx: 5, dy: 5 });
    await store.closeSession();
    raf.flush();
    await store.flushViewportInput();
    expect(inputsOf(api)).toEqual([]);
  });

  it("shows one error when the core refuses the input, and carries on with the next", async () => {
    raf = installManualRaf();
    const { api, store } = await opened({ current: 3 });
    api.failNext("viewportInput", { scope: "viewport", message: "The viewport is busy. Try again." });
    store.queueViewportInput({ kind: "pan", dx: 1, dy: 1 });
    raf.flush();
    await store.flushViewportInput();
    expect(store.getState().error).toEqual({ scope: "viewport", message: "The viewport is busy. Try again." });
    store.dismissError();
    store.queueViewportInput({ kind: "pan", dx: 2, dy: 2 });
    await store.flushViewportInput();
    expect(inputsOf(api).at(-1)).toEqual({ kind: "pan", dx: 2, dy: 2 });
    expect(store.getState().error).toBeNull();
  });
});

describe("the histogram", () => {
  /** Frames drawn for `ms`, one every 10 ms, as a drag makes them. */
  async function frames(api: MockApi, ms: number): Promise<void> {
    for (let t = 0; t < ms; t += 10) {
      api.emitEvent("render_stats", STATS);
      await vi.advanceTimersByTimeAsync(10);
    }
  }

  it("is asked for at most five times a second while frames keep coming, and once after the last frame (a throttle, not a debounce)", async () => {
    const { api, store } = await opened({ current: 3 });
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const histogram = vi.spyOn(api, "histogram");
    const sample = vi.spyOn(api, "samplePixel");
    const unwatch = store.watchHistogram({ readout: true });
    await vi.advanceTimersByTimeAsync(0);
    // Asked for at once when the tab shows.
    expect(histogram).toHaveBeenCalledTimes(1);
    expect(sample).toHaveBeenCalledTimes(1);

    // A hundred frames in one second.
    await frames(api, 1000);
    const during = histogram.mock.calls.length - 1;
    expect(during, "a debounce would wait for the drag to end and ask for nothing").toBeGreaterThanOrEqual(4);
    expect(during).toBeLessThanOrEqual(1000 / HISTOGRAM_THROTTLE_MS);
    expect(sample.mock.calls.length - 1).toBeLessThanOrEqual(1000 / HISTOGRAM_THROTTLE_MS);

    // One more after the last frame.
    const before = histogram.mock.calls.length;
    api.emitEvent("render_stats", STATS);
    await vi.advanceTimersByTimeAsync(HISTOGRAM_THROTTLE_MS + 5);
    expect(histogram.mock.calls.length).toBe(before + 1);

    unwatch();
  });

  it("is not asked for once nobody watches, or while nobody has", async () => {
    const { api, store } = await opened({ current: 3 });
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const histogram = vi.spyOn(api, "histogram");
    await frames(api, 400);
    expect(histogram).not.toHaveBeenCalled();

    const unwatch = store.watchHistogram();
    await vi.advanceTimersByTimeAsync(0);
    expect(histogram).toHaveBeenCalledTimes(1);
    api.emitEvent("render_stats", STATS);
    unwatch();
    unwatch();
    await vi.advanceTimersByTimeAsync(1000);
    await frames(api, 400);
    expect(histogram).toHaveBeenCalledTimes(1);
  });

  it("is kept up for as long as one of two watchers is left, and the middle readout only for the one that wants it", async () => {
    const { api, store } = await opened({ current: 3 });
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const histogram = vi.spyOn(api, "histogram");
    const sample = vi.spyOn(api, "samplePixel");
    const curve = store.watchHistogram();
    const light = store.watchHistogram({ readout: true });
    await vi.advanceTimersByTimeAsync(0);
    expect(sample).toHaveBeenCalledTimes(1);
    light();
    sample.mockClear();
    histogram.mockClear();
    api.emitEvent("render_stats", STATS);
    await vi.advanceTimersByTimeAsync(HISTOGRAM_THROTTLE_MS + 5);
    expect(histogram).toHaveBeenCalledTimes(1);
    expect(sample).not.toHaveBeenCalled();
    curve();
  });

  it("holds the middle readout in the state, and forgets it when the photo changes", async () => {
    const { store } = await opened({ current: 3 });
    store.setViewportRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    const unwatch = store.watchHistogram({ readout: true });
    await vi.waitFor(() => expect(store.getState().middleReadout).not.toBeNull());
    expect(store.getState().middleReadout).toMatchObject({ r: expect.any(Number), g: expect.any(Number), b: expect.any(Number) });
    await store.setCurrent(5);
    expect(store.getState().middleReadout).toBeNull();
    unwatch();
  });
});

describe("look tiles", () => {
  /** A store whose look tiles the test answers by hand. */
  async function held(): Promise<{
    store: AppStore;
    asked: Array<{ look: string; photo: number }>;
    answer: (index: number, url?: string) => void;
    refuse: (index: number, error?: unknown) => void;
    inFlight: () => number;
    maxInFlight: () => number;
  }> {
    const { api, store } = await opened({ current: 3 });
    const asked: Array<{ look: string; photo: number }> = [];
    const gates: Array<{ resolve: (url: string) => void; reject: (e: unknown) => void; done: boolean }> = [];
    let live = 0;
    let peak = 0;
    vi.spyOn(api, "lookThumbnail").mockImplementation(
      (photo, look) =>
        new Promise<string>((resolve, reject) => {
          asked.push({ look, photo });
          live += 1;
          peak = Math.max(peak, live);
          const gate = {
            done: false,
            resolve: (url: string) => {
              live -= 1;
              gate.done = true;
              resolve(url);
            },
            reject: (e: unknown) => {
              live -= 1;
              gate.done = true;
              reject(e);
            },
          };
          gates.push(gate);
        }),
    );
    return {
      store,
      asked,
      answer: (index, url = `blob:tile-${index}`) => gates[index]?.resolve(url),
      refuse: (index, error = { scope: "app", message: "The core says no." }) => gates[index]?.reject(error),
      inFlight: () => live,
      maxInFlight: () => peak,
    };
  }

  const LOOKS = Array.from({ length: 12 }, (_, i) => `look-${i + 1}`);
  const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

  it("asks the core for two at a time, however many tiles are in view, and answers every one", async () => {
    const t = await held();
    const urls = LOOKS.map((look) => t.store.lookThumbnail(3, look, "k1"));
    await settle();
    expect(t.asked).toHaveLength(LOOK_THUMBNAIL_CONCURRENCY);
    // The ones made together are asked for in the order they were made.
    expect(t.asked.map((a) => a.look)).toEqual(["look-1", "look-2"]);
    for (let answered = 0; answered < LOOKS.length; answered++) {
      expect(t.inFlight()).toBeLessThanOrEqual(LOOK_THUMBNAIL_CONCURRENCY);
      t.answer(answered);
      await settle();
    }
    expect(t.maxInFlight()).toBe(LOOK_THUMBNAIL_CONCURRENCY);
    expect(t.asked).toHaveLength(LOOKS.length);
    await expect(Promise.all(urls)).resolves.toHaveLength(LOOKS.length);
  });

  it("asks for what has just come into view first", async () => {
    const t = await held();
    void t.store.lookThumbnail(3, "look-1", "k1");
    void t.store.lookThumbnail(3, "look-2", "k1");
    await settle();
    // Two are out. A row of three scrolls into view, then another row of three.
    for (const look of ["look-3", "look-4", "look-5"]) void t.store.lookThumbnail(3, look, "k1");
    await settle();
    for (const look of ["look-6", "look-7", "look-8"]) void t.store.lookThumbnail(3, look, "k1");
    await settle();
    t.answer(0);
    await settle();
    t.answer(1);
    await settle();
    // The newest row first, each row from the left.
    expect(t.asked.map((a) => a.look).slice(2, 4)).toEqual(["look-6", "look-7"]);
  });

  it("drops the requests that have not started when the photo's settings change, and says they were superseded, not that they failed", async () => {
    const t = await held();
    const old = LOOKS.map((look) => t.store.lookThumbnail(3, look, "k1"));
    const outcomes = old.map((p) => p.then(() => "url", (e: unknown) => (isSuperseded(e) ? "superseded" : "failed")));
    await settle();
    expect(t.asked).toHaveLength(2);

    // The settings changed: tiles ask again with a new key.
    const fresh = t.store.lookThumbnail(3, "look-1", "k2");
    await settle();
    // The ten that had not started are dropped. The two that had are left to finish.
    expect(await Promise.all(outcomes.slice(2))).toEqual(Array(10).fill("superseded"));
    expect(t.asked).toHaveLength(2);
    t.answer(0);
    t.answer(1);
    await settle();
    expect(await Promise.all(outcomes.slice(0, 2))).toEqual(["url", "url"]);
    // The new request goes out now that places are free, and only it.
    expect(t.asked.map((a) => a.look).slice(2)).toEqual(["look-1"]);
    t.answer(2, "blob:fresh");
    await expect(fresh).resolves.toBe("blob:fresh");
  });

  it("asks again for a look whose request was superseded, with a request of its own", async () => {
    const t = await held();
    for (const look of LOOKS.slice(0, 4)) void t.store.lookThumbnail(3, look, "k1").catch(() => undefined);
    await settle();
    void t.store.lookThumbnail(3, "look-4", "k2");
    await settle();
    // look-4 had not started under k1 (it was dropped); under k2 it is a new request, not the dropped one.
    expect(t.asked.map((a) => a.look)).toEqual(["look-1", "look-2"]);
    t.answer(0);
    t.answer(1);
    await settle();
    expect(t.asked.map((a) => a.look)).toEqual(["look-1", "look-2", "look-4"]);
  });

  it("drops the requests of the photo that was left when the person moves to another", async () => {
    const t = await held();
    const left = ["look-1", "look-2", "look-3", "look-4"].map((look) => t.store.lookThumbnail(3, look, "k1").then(() => "url", (e: unknown) => (isSuperseded(e) ? "superseded" : "failed")));
    await settle();
    await t.store.setCurrent(5);
    await settle();
    expect(await Promise.all(left.slice(2))).toEqual(["superseded", "superseded"]);
    t.answer(0);
    t.answer(1);
    await settle();
    // Nothing of the old photo is asked for after that.
    expect(t.asked.every((a) => a.photo === 3)).toBe(true);
    expect(t.asked).toHaveLength(2);
  });

  it("keeps a failed request out of the cache so that asking again makes a new one", async () => {
    const t = await held();
    const first = t.store.lookThumbnail(3, "look-1", "k1");
    await settle();
    t.refuse(0);
    await expect(first).rejects.toMatchObject({ scope: "app" });
    const second = t.store.lookThumbnail(3, "look-1", "k1");
    await settle();
    expect(t.asked).toHaveLength(2);
    t.answer(1, "blob:again");
    await expect(second).resolves.toBe("blob:again");
  });
});
