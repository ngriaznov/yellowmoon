import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { UpdateState } from "./api/types";

// The real plugins, replaced: the default `updater` is wired to these.
const pluginCheck = vi.fn();
const pluginRelaunch = vi.fn();
vi.mock("@tauri-apps/plugin-updater", () => ({ check: (...args: unknown[]) => pluginCheck(...args) }));
vi.mock("@tauri-apps/plugin-process", () => ({ relaunch: (...args: unknown[]) => pluginRelaunch(...args) }));

const {
  CHECK_INTERVAL_MS,
  createUpdater,
  FIRST_CHECK_DELAY_MS,
  RESTART_AFTER_EXPORT,
  restartAvailability,
  updateCell,
  updater,
  useUpdater,
} = await import("./updater");
type UpdateHandle = import("./updater").UpdateHandle;
type DownloadCallback = Parameters<UpdateHandle["downloadAndInstall"]>[0];

/** A fake update whose download reports the given events. */
function fakeUpdate(version = "0.0.2", events: Array<Parameters<NonNullable<DownloadCallback>>[0]> = []): UpdateHandle {
  return {
    version,
    downloadAndInstall: vi.fn(async (onEvent?: DownloadCallback) => {
      for (const event of events) onEvent?.(event);
    }),
  };
}

const IDLE: UpdateState = { phase: "idle" };

beforeEach(() => {
  vi.useFakeTimers();
  pluginCheck.mockReset();
  pluginRelaunch.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
  delete (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__;
});

describe("quiet check", () => {
  it("starts idle", () => {
    expect(createUpdater({ check: vi.fn(), relaunch: vi.fn() }).getState()).toEqual(IDLE);
  });

  it("moves to available when an update exists, and tells subscribers", async () => {
    const check = vi.fn().mockResolvedValue(fakeUpdate("0.0.2"));
    const u = createUpdater({ check, relaunch: vi.fn() });
    const listener = vi.fn();
    u.subscribe(listener);
    await u.quietCheck();
    expect(u.getState()).toEqual({ phase: "available", version: "0.0.2" });
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("stays idle when this version is current", async () => {
    const u = createUpdater({ check: vi.fn().mockResolvedValue(null), relaunch: vi.fn() });
    await u.quietCheck();
    expect(u.getState()).toEqual(IDLE);
  });

  it("stays silent when the endpoint can't be reached", async () => {
    const u = createUpdater({ check: vi.fn().mockRejectedValue(new Error("offline")), relaunch: vi.fn() });
    const listener = vi.fn();
    u.subscribe(listener);
    await expect(u.quietCheck()).resolves.toBeUndefined();
    expect(u.getState()).toEqual(IDLE);
    expect(listener).not.toHaveBeenCalled();
  });

  it("leaves a download or a ready update alone", async () => {
    const check = vi.fn().mockResolvedValue(fakeUpdate("0.0.3"));
    const u = createUpdater({ check, relaunch: vi.fn() });
    u.applyExternal({ phase: "ready", version: "0.0.2" });
    await u.quietCheck();
    u.applyExternal({ phase: "downloading", version: "0.0.2", percent: 10 });
    await u.quietCheck();
    expect(check).not.toHaveBeenCalled();
  });

  it("replaces an old failure with a fresh result", async () => {
    const check = vi.fn().mockResolvedValue(fakeUpdate("0.0.2"));
    const u = createUpdater({ check, relaunch: vi.fn() });
    u.applyExternal({ phase: "error", message: "Download failed." });
    await u.quietCheck();
    expect(u.getState()).toEqual({ phase: "available", version: "0.0.2" });
  });
});

describe("schedule (docs/12: 10 s after launch, then every 6 hours)", () => {
  it("constants match the spec", () => {
    expect(FIRST_CHECK_DELAY_MS).toBe(10_000);
    expect(CHECK_INTERVAL_MS).toBe(6 * 60 * 60 * 1000);
  });

  it("checks at 10 s, then every 6 h, until stopped", async () => {
    const check = vi.fn().mockResolvedValue(null);
    const u = createUpdater({ check, relaunch: vi.fn() });
    const stop = u.start();

    await vi.advanceTimersByTimeAsync(9_999);
    expect(check).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(check).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(CHECK_INTERVAL_MS - 1);
    expect(check).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(check).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(CHECK_INTERVAL_MS);
    expect(check).toHaveBeenCalledTimes(3);

    stop();
    await vi.advanceTimersByTimeAsync(CHECK_INTERVAL_MS * 3);
    expect(check).toHaveBeenCalledTimes(3);
  });

  it("can be stopped before the first check", async () => {
    const check = vi.fn().mockResolvedValue(null);
    const stop = createUpdater({ check, relaunch: vi.fn() }).start();
    stop();
    await vi.advanceTimersByTimeAsync(CHECK_INTERVAL_MS * 2);
    expect(check).not.toHaveBeenCalled();
  });

  it("restarting the schedule replaces the old one", async () => {
    const check = vi.fn().mockResolvedValue(null);
    const u = createUpdater({ check, relaunch: vi.fn() });
    u.start();
    u.start();
    await vi.advanceTimersByTimeAsync(FIRST_CHECK_DELAY_MS);
    expect(check).toHaveBeenCalledTimes(1);
  });

  it("asks the preference before every check, so switching it off takes effect at once", async () => {
    const check = vi.fn().mockResolvedValue(null);
    let on = false;
    const u = createUpdater({ check, relaunch: vi.fn(), autoCheck: async () => on });
    u.start();
    await vi.advanceTimersByTimeAsync(FIRST_CHECK_DELAY_MS);
    await vi.advanceTimersByTimeAsync(CHECK_INTERVAL_MS);
    expect(check).not.toHaveBeenCalled(); // off: nothing is asked

    on = true;
    await vi.advanceTimersByTimeAsync(CHECK_INTERVAL_MS);
    expect(check).toHaveBeenCalledTimes(1);

    on = false;
    await vi.advanceTimersByTimeAsync(CHECK_INTERVAL_MS);
    expect(check).toHaveBeenCalledTimes(1);
  });

  it("a scheduled check that finds an update shows it", async () => {
    const u = createUpdater({ check: vi.fn().mockResolvedValue(fakeUpdate("0.0.5")), relaunch: vi.fn() });
    u.start();
    await vi.advanceTimersByTimeAsync(FIRST_CHECK_DELAY_MS);
    expect(u.getState()).toEqual({ phase: "available", version: "0.0.5" });
  });
});

describe("manual check (Check for Updates…)", () => {
  it("reports an available update", async () => {
    const u = createUpdater({ check: vi.fn().mockResolvedValue(fakeUpdate("0.1.0")), relaunch: vi.fn() });
    await expect(u.checkNow()).resolves.toEqual({ kind: "available", version: "0.1.0" });
    expect(u.getState()).toEqual({ phase: "available", version: "0.1.0" });
  });

  it("reports that the app is up to date", async () => {
    const u = createUpdater({ check: vi.fn().mockResolvedValue(null), relaunch: vi.fn() });
    u.applyExternal({ phase: "error", message: "old" });
    await expect(u.checkNow()).resolves.toEqual({ kind: "up_to_date" });
    expect(u.getState()).toEqual(IDLE);
  });

  it("reports a failure, unlike the quiet check", async () => {
    const u = createUpdater({ check: vi.fn().mockRejectedValue(new Error("offline")), relaunch: vi.fn() });
    await expect(u.checkNow()).resolves.toEqual({ kind: "error", message: "offline" });
    expect(u.getState()).toEqual({ phase: "error", message: "offline" });
  });

  it("describes a rejection that isn't an Error", async () => {
    const u = createUpdater({ check: vi.fn().mockRejectedValue("denied"), relaunch: vi.fn() });
    await expect(u.checkNow()).resolves.toEqual({ kind: "error", message: "denied" });
  });

  it("answers from the update already downloading or ready, without asking again", async () => {
    const check = vi.fn();
    const u = createUpdater({ check, relaunch: vi.fn() });
    u.applyExternal({ phase: "ready", version: "0.0.9" });
    await expect(u.checkNow()).resolves.toEqual({ kind: "available", version: "0.0.9" });
    u.applyExternal({ phase: "downloading", version: "0.0.9" });
    await expect(u.checkNow()).resolves.toEqual({ kind: "available", version: "0.0.9" });
    expect(check).not.toHaveBeenCalled();
  });
});

describe("download", () => {
  const events = [
    { event: "Started", data: { contentLength: 1000 } },
    { event: "Progress", data: { chunkLength: 250 } },
    { event: "Progress", data: { chunkLength: 500 } },
    { event: "Progress", data: { chunkLength: 400 } },
    { event: "Finished" },
  ] as const;

  it("walks available, downloading with percentages, then ready", async () => {
    const update = fakeUpdate("0.0.2", [...events]);
    const u = createUpdater({ check: vi.fn().mockResolvedValue(update), relaunch: vi.fn() });
    await u.quietCheck();
    const seen: UpdateState[] = [];
    u.subscribe(() => seen.push(u.getState()));
    await u.download();
    expect(seen).toEqual([
      { phase: "downloading", version: "0.0.2" },
      { phase: "downloading", version: "0.0.2", percent: 0 },
      { phase: "downloading", version: "0.0.2", percent: 25 },
      { phase: "downloading", version: "0.0.2", percent: 75 },
      { phase: "downloading", version: "0.0.2", percent: 100 }, // 1150 bytes of 1000 caps at 100
      { phase: "downloading", version: "0.0.2", percent: 100 },
      { phase: "ready", version: "0.0.2" },
    ]);
    expect(update.downloadAndInstall).toHaveBeenCalledTimes(1);
  });

  it("reports no percentage when the server sends no length", async () => {
    const update = fakeUpdate("0.0.2", [
      { event: "Started", data: {} },
      { event: "Progress", data: { chunkLength: 300 } },
    ]);
    const u = createUpdater({ check: vi.fn().mockResolvedValue(update), relaunch: vi.fn() });
    await u.quietCheck();
    const percents: Array<number | undefined> = [];
    u.subscribe(() => {
      const s = u.getState();
      if (s.phase === "downloading") percents.push(s.percent);
    });
    await u.download();
    expect(percents).toEqual([undefined, undefined]);
    expect(u.getState().phase).toBe("ready");
  });

  it("fails into the error phase with the reason", async () => {
    const update = fakeUpdate();
    vi.mocked(update.downloadAndInstall).mockRejectedValue(new Error("signature does not match"));
    const u = createUpdater({ check: vi.fn().mockResolvedValue(update), relaunch: vi.fn() });
    await u.quietCheck();
    await u.download();
    expect(u.getState()).toEqual({ phase: "error", message: "signature does not match" });
  });

  it("does nothing unless an update is available", async () => {
    const update = fakeUpdate();
    const u = createUpdater({ check: vi.fn().mockResolvedValue(update), relaunch: vi.fn() });
    await u.download();
    expect(u.getState()).toEqual(IDLE);
    await u.quietCheck();
    await u.download();
    await u.download(); // the second call finds it ready, not available
    expect(update.downloadAndInstall).toHaveBeenCalledTimes(1);
  });
});

describe("restart", () => {
  async function ready() {
    const relaunch = vi.fn().mockResolvedValue(undefined);
    const u = createUpdater({ check: vi.fn().mockResolvedValue(fakeUpdate()), relaunch });
    await u.quietCheck();
    await u.download();
    return { u, relaunch };
  }

  it("relaunches when the update is ready and no export is running", async () => {
    const { u, relaunch } = await ready();
    await expect(u.restart(false)).resolves.toBe(true);
    expect(relaunch).toHaveBeenCalledTimes(1);
  });

  it("never interrupts an export", async () => {
    const { u, relaunch } = await ready();
    await expect(u.restart(true)).resolves.toBe(false);
    expect(relaunch).not.toHaveBeenCalled();
    // The export finishes; the same update can now restart.
    await expect(u.restart(false)).resolves.toBe(true);
  });

  it("does nothing before an update is ready", async () => {
    const relaunch = vi.fn();
    const u = createUpdater({ check: vi.fn(), relaunch });
    await expect(u.restart(false)).resolves.toBe(false);
    u.applyExternal({ phase: "available", version: "0.0.2" });
    await expect(u.restart(false)).resolves.toBe(false);
    expect(relaunch).not.toHaveBeenCalled();
  });

  it("reports a relaunch that fails", async () => {
    const relaunch = vi.fn().mockRejectedValue(new Error("could not relaunch"));
    const u = createUpdater({ check: vi.fn(), relaunch });
    u.applyExternal({ phase: "ready", version: "0.0.2" });
    await expect(u.restart(false)).resolves.toBe(false);
    expect(u.getState()).toEqual({ phase: "error", message: "could not relaunch" });
  });
});

describe("restartAvailability (the export-aware rule)", () => {
  const states: UpdateState[] = [
    { phase: "idle" },
    { phase: "available", version: "1" },
    { phase: "downloading", version: "1", percent: 3 },
    { phase: "ready", version: "1" },
    { phase: "error", message: "x" },
  ];

  it.each(states.map((s) => [s.phase, s] as const))("%s: only a ready update can restart, and only when no export runs", (_phase, state) => {
    expect(restartAvailability(state, false).enabled).toBe(state.phase === "ready");
    expect(restartAvailability(state, true).enabled).toBe(false);
  });

  it("explains the wait only for a ready update during an export", () => {
    expect(restartAvailability({ phase: "ready", version: "1" }, true)).toEqual({ enabled: false, reason: RESTART_AFTER_EXPORT });
    expect(RESTART_AFTER_EXPORT).toBe("Restarts after the export finishes");
    expect(restartAvailability({ phase: "ready", version: "1" }, false)).toEqual({ enabled: true, reason: null });
    expect(restartAvailability({ phase: "idle" }, true)).toEqual({ enabled: false, reason: null });
  });
});

describe("updateCell (the header cell of docs/11)", () => {
  it("is hidden while idle", () => {
    expect(updateCell({ phase: "idle" }, false)).toBeNull();
  });

  it("offers the download, shows progress, then offers the restart", () => {
    expect(updateCell({ phase: "available", version: "0.0.2" }, false)).toEqual({
      phase: "available",
      label: "Update available",
      action: "download",
      disabled: false,
      title: null,
    });
    expect(updateCell({ phase: "downloading", version: "0.0.2", percent: 42 }, false)?.label).toBe("Downloading 42%");
    expect(updateCell({ phase: "downloading", version: "0.0.2" }, false)?.label).toBe("Downloading…");
    expect(updateCell({ phase: "downloading", version: "0.0.2", percent: 42 }, false)?.action).toBeNull();
    expect(updateCell({ phase: "ready", version: "0.0.2" }, false)).toEqual({
      phase: "ready",
      label: "Update ready",
      action: "restart",
      disabled: false,
      title: null,
    });
  });

  it("disables Update ready during an export, with the tooltip from docs/11", () => {
    expect(updateCell({ phase: "ready", version: "0.0.2" }, true)).toMatchObject({
      label: "Update ready",
      disabled: true,
      title: "Restarts after the export finishes",
    });
  });

  it("shows the failure reason on hover", () => {
    expect(updateCell({ phase: "error", message: "Signature mismatch." }, false)).toEqual({
      phase: "error",
      label: "Update failed",
      action: null,
      disabled: false,
      title: "Signature mismatch.",
    });
  });
});

describe("useUpdater", () => {
  it("follows the state and runs the cell's action", async () => {
    const update = fakeUpdate("0.0.2", [{ event: "Finished" }]);
    const relaunch = vi.fn().mockResolvedValue(undefined);
    const u = createUpdater({ check: vi.fn().mockResolvedValue(update), relaunch });
    const { result } = renderHook(() => useUpdater(false, u));
    expect(result.current.cell).toBeNull();

    await act(async () => {
      await u.quietCheck();
    });
    expect(result.current.cell?.label).toBe("Update available");

    await act(async () => {
      result.current.activate(); // downloads
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.state).toEqual({ phase: "ready", version: "0.0.2" });
    expect(result.current.cell?.label).toBe("Update ready");

    await act(async () => {
      result.current.activate(); // restarts
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(relaunch).toHaveBeenCalledTimes(1);
  });

  it("does not restart while an export is running", async () => {
    const relaunch = vi.fn().mockResolvedValue(undefined);
    const u = createUpdater({ check: vi.fn(), relaunch });
    u.applyExternal({ phase: "ready", version: "0.0.2" });
    const { result, rerender } = renderHook(({ exporting }) => useUpdater(exporting, u), { initialProps: { exporting: true } });
    expect(result.current.cell).toMatchObject({ disabled: true, title: RESTART_AFTER_EXPORT });
    await act(async () => {
      result.current.activate();
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(relaunch).not.toHaveBeenCalled();

    rerender({ exporting: false });
    expect(result.current.cell?.disabled).toBe(false);
    await act(async () => {
      result.current.activate();
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(relaunch).toHaveBeenCalledTimes(1);
  });

  it("takes a state from the core's update_state event", () => {
    const u = createUpdater({ check: vi.fn(), relaunch: vi.fn() });
    const { result } = renderHook(() => useUpdater(false, u));
    act(() => u.applyExternal({ phase: "downloading", version: "0.0.2", percent: 7 }));
    expect(result.current.cell?.label).toBe("Downloading 7%");
  });
});

describe("the app's updater, wired to the Tauri plugins", () => {
  it("finds nothing outside a Tauri window, without touching the plugin", async () => {
    await updater.checkNow();
    expect(pluginCheck).not.toHaveBeenCalled();
    expect(updater.getState()).toEqual(IDLE);
  });

  it("checks, downloads and relaunches through the plugins inside Tauri", async () => {
    (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {};
    const downloadAndInstall = vi.fn(async (onEvent?: DownloadCallback) => {
      onEvent?.({ event: "Started", data: { contentLength: 100 } });
      onEvent?.({ event: "Progress", data: { chunkLength: 100 } });
      onEvent?.({ event: "Finished" });
    });
    pluginCheck.mockResolvedValue({ version: "0.0.2", downloadAndInstall });
    pluginRelaunch.mockResolvedValue(undefined);

    await expect(updater.checkNow()).resolves.toEqual({ kind: "available", version: "0.0.2" });
    expect(pluginCheck).toHaveBeenCalledTimes(1);
    await updater.download();
    expect(downloadAndInstall).toHaveBeenCalledTimes(1);
    expect(updater.getState()).toEqual({ phase: "ready", version: "0.0.2" });
    await expect(updater.restart(true)).resolves.toBe(false);
    expect(pluginRelaunch).not.toHaveBeenCalled();
    await expect(updater.restart(false)).resolves.toBe(true);
    expect(pluginRelaunch).toHaveBeenCalledTimes(1);
  });
});
