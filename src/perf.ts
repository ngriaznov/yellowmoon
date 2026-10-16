/**
 * Optional measuring for the web view (`docs/13` §Perf counters, task M3-20). WebKit has no Long Tasks or Event Timing API, so the page
 * counts what it does itself: the animation frames that came late, how often some screens draw, how often the store tells its
 * listeners, and what the commands and events of the core cost.
 *
 * Off unless the page was opened with `?perf` or `localStorage.ymPerf` is set (`1`). Off, nothing is recorded and nothing is wrapped:
 * `perfCount` is an empty function and `getApi()` hands out the plain api. On, `window.__ymPerf.snapshot()` returns one JSON object
 * (for Playwright, and for the Mac job that reads it back after a scripted drag). Everything stays in the page: nothing is sent
 * anywhere (AGENTS rule 7).
 */

/** A frame that came more than this long after the one before it was a visible stall (two refreshes at 60 Hz, three at 120 Hz). */
export const LONG_FRAME_MS = 24;

/** What `window.__ymPerf.snapshot()` returns. */
export interface PerfSnapshot {
  /** Milliseconds since the page started measuring. */
  uptime_ms: number;
  /** Named counters (`Photos`, `LookTab`, `LazyLookTile`, `store.notify`, `event.render_stats`): the total and the busiest second. */
  counts: Record<string, { total: number; peak_per_second: number }>;
  /** The core's commands as the page called them: how many, how many failed, and the round trip. */
  commands: Record<string, { calls: number; failed: number; mean_ms: number; worst_ms: number }>;
  /** Animation frames that came late (`LONG_FRAME_MS` and more after the one before). */
  frames: {
    watching: boolean;
    long_frames: number;
    worst_gap_ms: number;
    /** Late frames by how late: 24 to 50 ms, 50 to 100 ms, 100 ms and more. */
    buckets: { "24-50": number; "50-100": number; "100+": number };
    /** The latest ones, newest last: when (ms since the start) and how late. */
    recent: Array<{ at_ms: number; gap_ms: number }>;
  };
}

/** The measuring. One instance serves the page (`perf`); tests make their own with a clock they drive. */
export interface Perf {
  readonly enabled: boolean;
  /** Counts one occurrence of `name`. */
  count: (name: string) => void;
  /** Counts the calls of an `Api`, their round trips, and the events it delivers. Returns `api` itself when measuring is off. */
  wrapApi: <T extends object>(api: T) => T;
  /** Starts recording late animation frames; returns the function that stops it. */
  watchFrames: (raf?: (callback: FrameRequestCallback) => number, cancel?: (handle: number) => void) => () => void;
  snapshot: () => PerfSnapshot;
  reset: () => void;
}

interface Counter {
  total: number;
  windowStart: number;
  windowCount: number;
  peak: number;
}

interface CommandStats {
  calls: number;
  failed: number;
  totalMs: number;
  worstMs: number;
}

const RECENT_FRAMES = 50;

const round1 = (n: number): number => Math.round(n * 10) / 10;

/** Makes the measuring. `enabled` false makes every method do nothing. */
export function createPerf(enabled: boolean, now: () => number = () => performance.now()): Perf {
  let started = enabled ? now() : 0;
  const counters = new Map<string, Counter>();
  const commands = new Map<string, CommandStats>();
  const wrapped = new WeakMap<object, object>();
  let watching = false;
  let frames = emptyFrames();

  function emptyFrames() {
    return { long: 0, worst: 0, buckets: { "24-50": 0, "50-100": 0, "100+": 0 }, recent: [] as Array<{ at_ms: number; gap_ms: number }> };
  }

  function count(name: string): void {
    if (!enabled) return;
    const t = now();
    let c = counters.get(name);
    if (!c) {
      c = { total: 0, windowStart: t, windowCount: 0, peak: 0 };
      counters.set(name, c);
    }
    if (t - c.windowStart >= 1000) {
      c.peak = Math.max(c.peak, c.windowCount);
      c.windowStart = t;
      c.windowCount = 0;
    }
    c.total += 1;
    c.windowCount += 1;
  }

  function track(name: string, began: number, result: unknown): void {
    let stats = commands.get(name);
    if (!stats) {
      stats = { calls: 0, failed: 0, totalMs: 0, worstMs: 0 };
      commands.set(name, stats);
    }
    const entry = stats;
    entry.calls += 1;
    if (!(result instanceof Promise)) return;
    const done = (failed: boolean): void => {
      const ms = now() - began;
      entry.totalMs += ms;
      entry.worstMs = Math.max(entry.worstMs, ms);
      if (failed) entry.failed += 1;
    };
    result.then(
      () => done(false),
      () => done(true),
    );
  }

  function wrapApi<T extends object>(api: T): T {
    if (!enabled) return api;
    const known = wrapped.get(api);
    if (known) return known as T;
    const proxy = new Proxy(api, {
      get(target, key) {
        // `this` is the target, not the proxy: the api's methods are plain closures, and a test may replace one on the target later.
        const value: unknown = Reflect.get(target, key, target);
        if (typeof value !== "function" || typeof key !== "string") return value;
        if (key === "on") {
          return (event: string, handler: (payload: unknown) => void) =>
            (value as (event: string, handler: (payload: unknown) => void) => unknown).call(target, event, (payload: unknown) => {
              count(`event.${event}`);
              handler(payload);
            });
        }
        return (...args: unknown[]) => {
          const began = now();
          const result: unknown = (value as (...a: unknown[]) => unknown).apply(target, args);
          track(key, began, result);
          return result;
        };
      },
    });
    wrapped.set(api, proxy);
    return proxy;
  }

  function watchFrames(
    raf: (callback: FrameRequestCallback) => number = (callback) => requestAnimationFrame(callback),
    cancel: (handle: number) => void = (handle) => cancelAnimationFrame(handle),
  ): () => void {
    if (!enabled || watching) return () => undefined;
    watching = true;
    let last: number | null = null;
    let handle = 0;
    const tick = (time: number): void => {
      if (last !== null) {
        const gap = time - last;
        if (gap > LONG_FRAME_MS) {
          frames.long += 1;
          frames.worst = Math.max(frames.worst, gap);
          frames.buckets[gap < 50 ? "24-50" : gap < 100 ? "50-100" : "100+"] += 1;
          frames.recent.push({ at_ms: round1(time - started), gap_ms: round1(gap) });
          if (frames.recent.length > RECENT_FRAMES) frames.recent.shift();
        }
      }
      last = time;
      handle = raf(tick);
    };
    handle = raf(tick);
    return () => {
      watching = false;
      cancel(handle);
    };
  }

  function snapshot(): PerfSnapshot {
    const counts: PerfSnapshot["counts"] = {};
    for (const [name, c] of counters) counts[name] = { total: c.total, peak_per_second: Math.max(c.peak, c.windowCount) };
    const cmds: PerfSnapshot["commands"] = {};
    for (const [name, s] of commands) {
      const settled = Math.max(1, s.calls);
      cmds[name] = { calls: s.calls, failed: s.failed, mean_ms: round1(s.totalMs / settled), worst_ms: round1(s.worstMs) };
    }
    return {
      uptime_ms: round1(now() - started),
      counts,
      commands: cmds,
      frames: {
        watching,
        long_frames: frames.long,
        worst_gap_ms: round1(frames.worst),
        buckets: { ...frames.buckets },
        recent: frames.recent.map((r) => ({ ...r })),
      },
    };
  }

  function reset(): void {
    counters.clear();
    commands.clear();
    frames = emptyFrames();
    started = now();
  }

  return { enabled, count, wrapApi, watchFrames, snapshot, reset };
}

/**
 * Whether this page was asked to measure: `?perf` (or `?perf=1`) in the address, or `ymPerf` set in `localStorage` (not "0", "false" or
 * empty). Either may be unreadable (a private window, blocked site data): that means off.
 */
export function perfRequested(search: string, storage: Pick<Storage, "getItem"> | null): boolean {
  const off = (value: string | null): boolean => value === null || value === "0" || value === "false";
  try {
    const params = new URLSearchParams(search);
    if (params.has("perf")) return !off(params.get("perf"));
  } catch {
    // An address that can't be read asks for nothing.
  }
  try {
    const stored = storage?.getItem("ymPerf") ?? null;
    return !off(stored) && stored !== "";
  } catch {
    return false;
  }
}

function pageStorage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

/** The page's measuring: on with `?perf` or `localStorage.ymPerf`. */
export const perf: Perf = createPerf(perfRequested(typeof location === "undefined" ? "" : location.search, pageStorage()));

const noop = (): void => undefined;

/** Counts one occurrence of `name`; an empty function while measuring is off. */
export const perfCount: (name: string) => void = perf.enabled ? perf.count : noop;

declare global {
  interface Window {
    /** The page's measuring, when it is on (`src/perf.ts`). */
    __ymPerf?: Pick<Perf, "snapshot" | "reset">;
  }
}

/** Starts the frame watch and publishes `window.__ymPerf`. Does nothing while measuring is off. */
export function installPerf(target: Pick<Perf, "enabled" | "watchFrames" | "snapshot" | "reset"> = perf, win: Window = window): void {
  if (!target.enabled) return;
  target.watchFrames();
  win.__ymPerf = { snapshot: target.snapshot, reset: target.reset };
}
