import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ExportJob, ViewportRect } from "./api/types";

const invoke = vi.fn();
const listen = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invoke(...args) }));
vi.mock("@tauri-apps/api/event", () => ({ listen: (...args: unknown[]) => listen(...args) }));

// Imported after the mocks are registered.
const { getApi, isAppError, isTauri, tauriApi, toAppError } = await import("./api");
const { createMockApi, mockApi } = await import("./api.mock");

const job: ExportJob = {
  scope: "picked",
  options: {
    format: "jpeg",
    jpeg_quality: 92,
    bit_depth: 16,
    tiff_compression: "lzw",
    color_space: "srgb",
    size: { kind: "full" },
    destination: null,
    name_template: "{name}_{look}",
    if_exists: "keep_both",
  },
};
const rect: ViewportRect = { x: 0, y: 75, width: 1020, height: 680, scale: 2 };

/** Each method with the command and arguments it must send: snake_case names, camelCase keys. */
const calls: Array<[string, () => Promise<unknown>, string, Record<string, unknown> | undefined]> = [
  ["openPaths", () => tauriApi.openPaths(["/a.ARW"]), "open_paths", { paths: ["/a.ARW"] }],
  ["openDialog", () => tauriApi.openDialog("folder"), "open_dialog", { kind: "folder" }],
  ["sessionState", () => tauriApi.sessionState(), "session_state", undefined],
  ["closeSession", () => tauriApi.closeSession(), "close_session", undefined],
  ["setCurrent", () => tauriApi.setCurrent(3), "set_current", { photoId: 3 }],
  ["setPicked", () => tauriApi.setPicked(3, true), "set_picked", { photoId: 3, picked: true }],
  ["getSettings", () => tauriApi.getSettings(3), "get_settings", { photoId: 3 }],
  ["updateSettings", () => tauriApi.updateSettings(3, { contrast: 4 }), "update_settings", { photoId: 3, patch: { contrast: 4 } }],
  ["applyLook", () => tauriApi.applyLook("all", "builtin:neutral", 80), "apply_look", { scope: "all", lookId: "builtin:neutral", intensity: 80 }],
  ["rerunAuto", () => tauriApi.rerunAuto(3), "rerun_auto", { photoId: 3 }],
  ["histogram", () => tauriApi.histogram(3), "histogram", { photoId: 3 }],
  ["viewportSetRect", () => tauriApi.viewportSetRect(rect), "viewport_set_rect", { rect }],
  ["viewportInput", () => tauriApi.viewportInput({ kind: "pan", dx: 1, dy: 2 }), "viewport_input", { event: { kind: "pan", dx: 1, dy: 2 } }],
  ["viewportSetMode", () => tauriApi.viewportSetMode({ kind: "fit" }), "viewport_set_mode", { mode: { kind: "fit" } }],
  ["setBefore", () => tauriApi.setBefore(true), "set_before", { before: true }],
  ["samplePixel", () => tauriApi.samplePixel(10, 20), "sample_pixel", { x: 10, y: 20 }],
  ["looksList", () => tauriApi.looksList(), "looks_list", undefined],
  ["lookThumbnail", () => tauriApi.lookThumbnail(3, "builtin:signature"), "look_thumbnail", { photoId: 3, lookId: "builtin:signature" }],
  ["looksImport", () => tauriApi.looksImport("/x.zip"), "looks_import", { path: "/x.zip", input: null }],
  ["looksImport (answered)", () => tauriApi.looksImport("/x.cube", "logc3_ei800"), "looks_import", { path: "/x.cube", input: "logc3_ei800" }],
  ["looksRecognise", () => tauriApi.looksRecognise("/x.zip"), "looks_recognise", { path: "/x.zip" }],
  ["looksDisplayLuts", () => tauriApi.looksDisplayLuts(), "looks_display_luts", undefined],
  ["looksWatchDownloads", () => tauriApi.looksWatchDownloads(true), "looks_watch_downloads", { enabled: true }],
  ["chooseFolder", () => tauriApi.chooseFolder("~/Pictures"), "choose_folder", { start: "~/Pictures" }],
  ["chooseLookFiles", () => tauriApi.chooseLookFiles(), "choose_look_files", undefined],
  ["openUrl", () => tauriApi.openUrl("https://example.com/"), "open_url", { url: "https://example.com/" }],
  ["revealInFinder", () => tauriApi.revealInFinder("~/Pictures/Lisbon export"), "reveal_in_finder", { path: "~/Pictures/Lisbon export" }],
  ["trashPath", () => tauriApi.trashPath("~/Downloads/x.zip"), "trash_path", { path: "~/Downloads/x.zip" }],
  ["looksSetDefault", () => tauriApi.looksSetDefault("builtin:signature"), "looks_set_default", { id: "builtin:signature" }],
  ["looksRemove", () => tauriApi.looksRemove("user:x"), "looks_remove", { id: "user:x" }],
  ["exportEstimate", () => tauriApi.exportEstimate(job), "export_estimate", { job }],
  ["exportStart", () => tauriApi.exportStart(job), "export_start", { job }],
  ["exportCancel", () => tauriApi.exportCancel(7), "export_cancel", { jobId: 7 }],
  ["cliPreview", () => tauriApi.cliPreview(job), "cli_preview", { job }],
  ["preferencesGet", () => tauriApi.preferencesGet(), "preferences_get", undefined],
  ["preferencesSet", () => tauriApi.preferencesSet({ remove_location_from_exports: true }), "preferences_set", { patch: { remove_location_from_exports: true } }],
  ["installCliLink", () => tauriApi.installCliLink(), "install_cli_link", undefined],
  ["cliLinkStatus", () => tauriApi.cliLinkStatus(), "cli_link_status", undefined],
  ["appInfo", () => tauriApi.appInfo(), "app_info", undefined],
];

beforeEach(() => {
  invoke.mockReset().mockResolvedValue("result");
  listen.mockReset();
});

describe("tauriApi commands", () => {
  it.each(calls)("%s calls invoke with the command and arguments of docs/01", async (_name, call, command, args) => {
    await expect(call()).resolves.toBe("result");
    expect(invoke).toHaveBeenCalledTimes(1);
    expect(invoke.mock.calls[0]).toEqual(args === undefined ? [command] : [command, args]);
  });

  it("covers every method of the mock, so the two can't drift apart", () => {
    const mockOnly = ["options", "configure", "reset", "flush", "failNext", "cancelNextDialog", "fixtureUrl", "viewport", "system", "queueChoice", "addSource", "removeSource", "simulateDownload", "emitEvent"];
    const mock = createMockApi();
    const contract = Object.keys(mock).filter((k) => !mockOnly.includes(k)).sort();
    expect(Object.keys(tauriApi).sort()).toEqual(contract);
    expect(calls.map(([name]) => name.replace(/ \(.*\)$/, "")).filter((name, i, all) => all.indexOf(name) === i).sort()).toEqual(contract.filter((k) => k !== "on"));
    mock.reset();
  });

  it("passes a rejection from the core through untouched", async () => {
    const error = { scope: "export", message: "The folder isn't writable. Choose another folder." };
    invoke.mockRejectedValue(error);
    await expect(tauriApi.exportStart(job)).rejects.toBe(error);
  });
});

describe("tauriApi.on", () => {
  it("hands the handler the event payload, not the envelope", async () => {
    let deliver: (e: { payload: unknown }) => void = () => {};
    listen.mockImplementation((_name: string, cb: typeof deliver) => {
      deliver = cb;
      return Promise.resolve(() => {});
    });
    const seen: unknown[] = [];
    tauriApi.on("render_stats", (stats) => seen.push(stats));
    expect(listen.mock.calls[0]?.[0]).toBe("render_stats");
    await Promise.resolve();
    deliver({ payload: { ms: 42, path: "neural", space: "display_p3" } });
    expect(seen).toEqual([{ ms: 42, path: "neural", space: "display_p3" }]);
  });

  it("unsubscribes through the function listen resolved with", async () => {
    const unlisten = vi.fn();
    listen.mockResolvedValue(unlisten);
    const stop = tauriApi.on("looks_changed", () => {});
    await Promise.resolve();
    stop();
    expect(unlisten).toHaveBeenCalledTimes(1);
    stop();
    expect(unlisten).toHaveBeenCalledTimes(1);
  });

  it("honours an unsubscribe that arrives before listen has resolved", async () => {
    const unlisten = vi.fn();
    let resolve: (fn: () => void) => void = () => {};
    listen.mockReturnValue(new Promise<() => void>((r) => (resolve = r)));
    const stop = tauriApi.on("error", () => {});
    stop();
    expect(unlisten).not.toHaveBeenCalled();
    resolve(unlisten);
    await Promise.resolve();
    await Promise.resolve();
    expect(unlisten).toHaveBeenCalledTimes(1);
  });

  it("survives a failed registration", async () => {
    listen.mockRejectedValue(new Error("window is closing"));
    const stop = tauriApi.on("error", () => {});
    await Promise.resolve();
    expect(() => stop()).not.toThrow();
  });
});

describe("getApi", () => {
  afterEach(() => {
    delete (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__;
  });

  it("returns the mock in a plain browser", () => {
    expect(isTauri()).toBe(false);
    expect(getApi()).toBe(mockApi);
  });

  it("returns the Tauri API inside a Tauri window", () => {
    (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {};
    expect(isTauri()).toBe(true);
    expect(getApi()).toBe(tauriApi);
  });
});

describe("errors", () => {
  it("isAppError recognises the core's error shape", () => {
    expect(isAppError({ scope: "open", message: "Nope." })).toBe(true);
    expect(isAppError({ message: "Nope." })).toBe(false);
    expect(isAppError(new Error("x"))).toBe(false);
    expect(isAppError(null)).toBe(false);
    expect(isAppError("open")).toBe(false);
  });

  it("toAppError wraps anything else in plain words", () => {
    expect(toAppError({ scope: "looks", message: "Gone." })).toEqual({ scope: "looks", message: "Gone." });
    expect(toAppError(new Error("boom"))).toEqual({ scope: "app", message: "boom" });
    expect(toAppError("text")).toEqual({ scope: "app", message: "text" });
    expect(toAppError(42)).toEqual({ scope: "app", message: "Something went wrong. Try again." });
  });
});
