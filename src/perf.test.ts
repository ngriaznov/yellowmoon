import { describe, expect, it, vi } from "vitest";
import { createPerf, installPerf, LONG_FRAME_MS, perfRequested } from "./perf";

/** A clock the test moves. */
function clock(start = 1000) {
  let t = start;
  return { now: () => t, advance: (ms: number) => void (t += ms) };
}

describe("perfRequested", () => {
  const storage = (value: string | null) => ({ getItem: () => value });

  it("is on with ?perf or ?perf=1, and off with ?perf=0 or ?perf=false, whatever is stored", () => {
    expect(perfRequested("?perf", null)).toBe(true);
    expect(perfRequested("?mockSpeed=Infinity&perf=1", null)).toBe(true);
    expect(perfRequested("?perf=0", storage("1"))).toBe(false);
    expect(perfRequested("?perf=false", storage("1"))).toBe(false);
  });

  it("is on with localStorage.ymPerf set, off when it is unset, empty, 0 or false", () => {
    expect(perfRequested("", storage("1"))).toBe(true);
    expect(perfRequested("", storage("true"))).toBe(true);
    expect(perfRequested("", storage(null))).toBe(false);
    expect(perfRequested("", storage(""))).toBe(false);
    expect(perfRequested("", storage("0"))).toBe(false);
    expect(perfRequested("", storage("false"))).toBe(false);
    expect(perfRequested("", null)).toBe(false);
  });

  it("is off, not an error, when the storage cannot be read", () => {
    expect(
      perfRequested("", {
        getItem: () => {
          throw new Error("blocked");
        },
      }),
    ).toBe(false);
  });
});

describe("a page that is not measuring", () => {
  it("records nothing and hands the api back unwrapped", () => {
    const perf = createPerf(false);
    perf.count("Photos");
    const api = { ping: () => 1 };
    expect(perf.wrapApi(api)).toBe(api);
    expect(perf.watchFrames(() => 0)()).toBeUndefined();
    expect(perf.snapshot().counts).toEqual({});
    const win = {} as Window;
    installPerf(perf, win);
    expect(win.__ymPerf).toBeUndefined();
  });
});

describe("counters", () => {
  it("count each name, and remember the busiest second", () => {
    const c = clock();
    const perf = createPerf(true, c.now);
    for (let i = 0; i < 5; i++) {
      perf.count("Photos");
      c.advance(10);
    }
    c.advance(1000);
    for (let i = 0; i < 2; i++) perf.count("Photos");
    perf.count("LookTab");
    const { counts } = perf.snapshot();
    expect(counts.Photos).toEqual({ total: 7, peak_per_second: 5 });
    expect(counts.LookTab).toEqual({ total: 1, peak_per_second: 1 });
  });

  it("start again on reset", () => {
    const c = clock();
    const perf = createPerf(true, c.now);
    perf.count("Photos");
    c.advance(500);
    perf.reset();
    expect(perf.snapshot().counts).toEqual({});
    expect(perf.snapshot().uptime_ms).toBe(0);
  });
});

describe("the api proxy", () => {
  it("counts the calls of each command and their round trips, failures included, and leaves results alone", async () => {
    const c = clock();
    const perf = createPerf(true, c.now);
    const api = {
      fast: (a: number) => a + 1,
      slow: () =>
        new Promise<string>((resolve) => {
          c.advance(30);
          resolve("done");
        }),
      broken: () => Promise.reject(new Error("no")),
    };
    const wrapped = perf.wrapApi(api);
    expect(wrapped.fast(1)).toBe(2);
    await expect(wrapped.slow()).resolves.toBe("done");
    await expect(wrapped.slow()).resolves.toBe("done");
    await expect(wrapped.broken()).rejects.toThrow("no");
    const { commands } = perf.snapshot();
    expect(commands.fast).toMatchObject({ calls: 1, failed: 0 });
    expect(commands.slow).toMatchObject({ calls: 2, failed: 0, mean_ms: 30, worst_ms: 30 });
    expect(commands.broken).toMatchObject({ calls: 1, failed: 1 });
  });

  it("wraps once, and sees a command that a test replaced on the api afterwards", async () => {
    const perf = createPerf(true, clock().now);
    const api: { ping: () => number } = { ping: () => 1 };
    const wrapped = perf.wrapApi(api);
    expect(perf.wrapApi(api)).toBe(wrapped);
    api.ping = () => 2;
    expect(wrapped.ping()).toBe(2);
    expect(perf.snapshot().commands.ping?.calls).toBe(1);
  });

  it("counts the events the api delivers, and still delivers them", () => {
    const perf = createPerf(true, clock().now);
    const handlers = new Map<string, (payload: unknown) => void>();
    const api = {
      on(event: string, handler: (payload: unknown) => void) {
        handlers.set(event, handler);
        return () => handlers.delete(event);
      },
    };
    const got: unknown[] = [];
    const stop = perf.wrapApi(api).on("render_stats", (payload) => got.push(payload));
    for (let i = 0; i < 3; i++) handlers.get("render_stats")?.({ i });
    expect(got).toHaveLength(3);
    expect(perf.snapshot().counts["event.render_stats"]?.total).toBe(3);
    stop();
    expect(handlers.has("render_stats")).toBe(false);
  });
});

describe("the frame watch", () => {
  it("records the frames that came late, by how late, and not the ones on time", () => {
    const perf = createPerf(true, clock().now);
    let callback: FrameRequestCallback = () => undefined;
    let cancelled = 0;
    const stop = perf.watchFrames(
      (cb) => {
        callback = cb;
        return 7;
      },
      () => void (cancelled += 1),
    );
    let t = 0;
    const frame = (gap: number): void => {
      t += gap;
      callback(t);
    };
    frame(0);
    frame(17);
    frame(LONG_FRAME_MS);
    frame(30);
    frame(60);
    frame(250);
    const { frames } = perf.snapshot();
    expect(frames.watching).toBe(true);
    expect(frames.long_frames).toBe(3);
    expect(frames.worst_gap_ms).toBe(250);
    expect(frames.buckets).toEqual({ "24-50": 1, "50-100": 1, "100+": 1 });
    expect(frames.recent.map((r) => r.gap_ms)).toEqual([30, 60, 250]);
    stop();
    expect(cancelled).toBe(1);
    expect(perf.snapshot().frames.watching).toBe(false);
  });

  it("keeps only the latest late frames", () => {
    const perf = createPerf(true, clock().now);
    let callback: FrameRequestCallback = () => undefined;
    perf.watchFrames((cb) => ((callback = cb), 1));
    let t = 0;
    callback((t += 1));
    for (let i = 0; i < 80; i++) callback((t += 40));
    const { frames } = perf.snapshot();
    expect(frames.long_frames).toBe(80);
    expect(frames.recent).toHaveLength(50);
  });

  it("is published on window.__ymPerf with a snapshot that is plain JSON", () => {
    const perf = createPerf(true, clock().now);
    const win = {} as Window;
    const watch = vi.spyOn(perf, "watchFrames").mockReturnValue(() => undefined);
    installPerf(perf, win);
    expect(watch).toHaveBeenCalledTimes(1);
    perf.count("Photos");
    const snapshot = win.__ymPerf?.snapshot();
    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
    expect(snapshot?.counts.Photos?.total).toBe(1);
  });
});
