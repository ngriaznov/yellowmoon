import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ExportJob, ViewportRect } from "./api/types";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

const invoke = vi.fn();
const listen = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invoke(...args) }));
vi.mock("@tauri-apps/api/event", () => ({ listen: (...args: unknown[]) => listen(...args) }));

// Imported after the mocks are registered.
const { getApi, isAppError, isTauri, tauriApi, toAppError } = await import("./api");
const { commands } = await import("./api/bindings");
const { default: generatedBindings } = await import("./api/bindings.ts?raw");
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

/** Each method with the command and arguments it must send: snake_case names, camelCase keys. A unit command answers `null` on the wire and nothing in `Api`. */
type Row = [name: string, call: () => Promise<unknown>, command: string, args: Record<string, unknown> | undefined, unit?: true];
const calls: Row[] = [
  ["openPaths", () => tauriApi.openPaths(["/a.ARW"]), "open_paths", { paths: ["/a.ARW"] }],
  ["openDialog", () => tauriApi.openDialog("folder"), "open_dialog", { kind: "folder" }],
  ["sessionState", () => tauriApi.sessionState(), "session_state", undefined],
  ["closeSession", () => tauriApi.closeSession(), "close_session", undefined, true],
  ["setCurrent", () => tauriApi.setCurrent(3), "set_current", { photoId: 3 }, true],
  ["setPicked", () => tauriApi.setPicked(3, true), "set_picked", { photoId: 3, picked: true }, true],
  ["removePhotos", () => tauriApi.removePhotos([3, 5]), "remove_photos", { photoIds: [3, 5] }],
  ["restorePhotos", () => tauriApi.restorePhotos(), "restore_photos", undefined],
  ["getSettings", () => tauriApi.getSettings(3), "get_settings", { photoId: 3 }],
  ["updateSettings", () => tauriApi.updateSettings(3, { contrast: manual(4) }), "update_settings", { photoId: 3, patch: { contrast: manual(4) } }],
  ["applyLook", () => tauriApi.applyLook("all", "builtin:neutral", 80), "apply_look", { scope: "all", lookId: "builtin:neutral", intensity: 80 }, true],
  ["rerunAuto", () => tauriApi.rerunAuto(3), "rerun_auto", { photoId: 3 }],
  ["copySettings", () => tauriApi.copySettings(3), "copy_settings", { photoId: 3 }],
  ["pasteSettings", () => tauriApi.pasteSettings([4, 5], { contrast: manual(4) }), "paste_settings", { photoIds: [4, 5], patch: { contrast: manual(4) } }],
  ["setMenuState", () => tauriApi.setMenuState({ copy_settings: true, paste_settings: false, paste_settings_to_picked: false }), "set_menu_state", { state: { copy_settings: true, paste_settings: false, paste_settings_to_picked: false } }, true],
  ["histogram", () => tauriApi.histogram(3), "histogram", { photoId: 3 }],
  ["chromaScatter", () => tauriApi.chromaScatter(3), "chroma_scatter", { photoId: 3 }],
  ["viewportInfo", () => tauriApi.viewportInfo(), "viewport_info", undefined],
  ["viewportSetRect", () => tauriApi.viewportSetRect(rect), "viewport_set_rect", { rect }, true],
  ["viewportInput", () => tauriApi.viewportInput({ kind: "pan", dx: 1, dy: 2 }), "viewport_input", { event: { kind: "pan", dx: 1, dy: 2 } }, true],
  ["viewportInput (pick white)", () => tauriApi.viewportInput({ kind: "pick_white", x: 40, y: 20 }), "viewport_input", { event: { kind: "pick_white", x: 40, y: 20 } }, true],
  ["viewportSetMode", () => tauriApi.viewportSetMode({ kind: "fit" }), "viewport_set_mode", { mode: { kind: "fit" } }, true],
  ["setBefore", () => tauriApi.setBefore(true), "set_before", { before: true }, true],
  ["samplePixel", () => tauriApi.samplePixel(10, 20), "sample_pixel", { x: 10, y: 20 }],
  ["sampleChroma", () => tauriApi.sampleChroma(10, 20), "sample_chroma", { x: 10, y: 20 }],
  ["looksList", () => tauriApi.looksList(), "looks_list", undefined],
  ["lookThumbnail", () => tauriApi.lookThumbnail(3, "builtin:signature"), "look_thumbnail", { photoId: 3, lookId: "builtin:signature" }],
  ["looksImport", () => tauriApi.looksImport("/x.zip"), "looks_import", { path: "/x.zip", input: null }],
  ["looksImport (answered)", () => tauriApi.looksImport("/x.cube", "logc3_ei800"), "looks_import", { path: "/x.cube", input: "logc3_ei800" }],
  ["looksRecognise", () => tauriApi.looksRecognise("/x.zip"), "looks_recognise", { path: "/x.zip" }],
  ["looksDisplayLuts", () => tauriApi.looksDisplayLuts(), "looks_display_luts", undefined],
  ["looksWatchDownloads", () => tauriApi.looksWatchDownloads(true), "looks_watch_downloads", { enabled: true }],
  ["chooseFolder", () => tauriApi.chooseFolder("~/Pictures"), "choose_folder", { start: "~/Pictures" }],
  ["chooseLookFiles", () => tauriApi.chooseLookFiles(), "choose_look_files", undefined],
  ["openUrl", () => tauriApi.openUrl("https://example.com/"), "open_url", { url: "https://example.com/" }, true],
  ["revealInFinder", () => tauriApi.revealInFinder("~/Pictures/Lisbon export"), "reveal_in_finder", { path: "~/Pictures/Lisbon export" }, true],
  ["trashPath", () => tauriApi.trashPath("~/Downloads/x.zip"), "trash_path", { path: "~/Downloads/x.zip" }, true],
  ["looksSetDefault", () => tauriApi.looksSetDefault("builtin:signature"), "looks_set_default", { id: "builtin:signature" }, true],
  ["looksRemove", () => tauriApi.looksRemove("user:x"), "looks_remove", { id: "user:x" }, true],
  ["exportEstimate", () => tauriApi.exportEstimate(job), "export_estimate", { job }],
  ["exportStart", () => tauriApi.exportStart(job), "export_start", { job }],
  ["exportCancel", () => tauriApi.exportCancel(7), "export_cancel", { jobId: 7 }, true],
  ["cliPreview", () => tauriApi.cliPreview(job), "cli_preview", { job }],
  ["exportBusy", () => tauriApi.exportBusy(), "export_busy", undefined],
  ["restartApp", () => tauriApi.restartApp(), "restart_app", undefined, true],
  ["thumbnails", () => tauriApi.thumbnails(), "thumbnails", undefined],
  ["preferencesGet", () => tauriApi.preferencesGet(), "preferences_get", undefined],
  ["preferencesSet", () => tauriApi.preferencesSet({ remove_location_from_exports: true }), "preferences_set", { patch: { remove_location_from_exports: true } }],
  ["installCliLink", () => tauriApi.installCliLink(), "install_cli_link", undefined],
  ["appleAutoStatus", () => tauriApi.appleAutoStatus(), "apple_auto_status", undefined],
  ["cliLinkStatus", () => tauriApi.cliLinkStatus(), "cli_link_status", undefined],
  ["appInfo", () => tauriApi.appInfo(), "app_info", undefined],
];

beforeEach(() => {
  invoke.mockReset().mockResolvedValue("result");
  listen.mockReset();
});

describe("tauriApi commands", () => {
  it.each(calls)("%s calls invoke with the command and arguments of docs/01", async (_name, call, command, args, unit) => {
    // A unit command answers `null` on the wire; the interface says it resolves with nothing.
    invoke.mockResolvedValue(unit ? null : "result");
    await expect(call()).resolves.toBe(unit ? undefined : "result");
    expect(invoke).toHaveBeenCalledTimes(1);
    expect(invoke.mock.calls[0]).toEqual(args === undefined ? [command] : [command, args]);
  });

  it("covers every method of the mock, so the two can't drift apart", () => {
    const mockOnly = ["options", "configure", "reset", "flush", "failNext", "cancelNextDialog", "fixtureUrl", "viewport", "menuState", "system", "queueChoice", "addSource", "removeSource", "simulateDownload", "emitEvent", "setAppleAuto"];
    const mock = createMockApi();
    const contract = Object.keys(mock).filter((k) => !mockOnly.includes(k)).sort();
    expect(Object.keys(tauriApi).sort()).toEqual(contract);
    expect(calls.map(([name]) => name.replace(/ \(.*\)$/, "")).filter((name, i, all) => all.indexOf(name) === i).sort()).toEqual(contract.filter((k) => k !== "on"));
    mock.reset();
  });

  it("calls the commands tauri-specta generated: each has a method here, and the wire names and argument keys of this table are theirs", () => {
    // `commands` is the Rust side's list (`cargo test -p yellowmoon-app bindings` keeps the file current); `tauriApi` calls through it.
    expect(Object.keys(tauriApi).filter((k) => k !== "on").sort()).toEqual(Object.keys(commands).sort());
    for (const [name, , command, args] of calls) {
      const found = new RegExp(`\\("${command}"(?:, \\{ ([^}]*) \\})?\\)`).exec(generatedBindings);
      expect(found, `${name}: bindings.ts has the command ${command}`).not.toBeNull();
      const keys = found?.[1]?.split(", ").sort() ?? [];
      expect(keys, `${name}: argument keys of ${command}`).toEqual(Object.keys(args ?? {}).sort());
    }
  });

  it("passes a rejection from the core through untouched", async () => {
    const error = { scope: "export", message: "The folder isn't writable. Choose another folder." };
    invoke.mockRejectedValue(error);
    await expect(tauriApi.exportStart(job)).rejects.toBe(error);
    expect(isAppError(error)).toBe(true);
  });

  it("a unit command that is rejected rejects with the same error", async () => {
    const error = { scope: "viewport", message: "That point is outside the photo." };
    invoke.mockRejectedValue(error);
    await expect(tauriApi.viewportInput({ kind: "pick_white", x: 1, y: 1 })).rejects.toBe(error);
  });
});

describe("tauriApi.on", () => {
  const EVENTS = [
    "photo_progress",
    "thumbnail_ready",
    "render_stats",
    "export_progress",
    "looks_changed",
    "update_state",
    "error",
    "looks_import_progress",
    "download_found",
    "menu_action",
    "viewport_frame",
    "settings_changed",
    "export_busy",
  ] as const;

  it.each(EVENTS)("listens to %s by its wire name, which the shell registers", (name) => {
    listen.mockResolvedValue(() => {});
    tauriApi.on(name, () => {});
    expect(listen.mock.calls[0]?.[0]).toBe(name);
    expect(generatedBindings).toContain(`makeEvent<`);
    expect(generatedBindings).toContain(`("${name}")`);
  });

  it("delivers the frames of the fallback viewport and the news that the core changed a photo's settings", async () => {
    const handlers = new Map<string, (e: { payload: unknown }) => void>();
    listen.mockImplementation((name: string, cb: (e: { payload: unknown }) => void) => {
      handlers.set(name, cb);
      return Promise.resolve(() => {});
    });
    const seen: unknown[] = [];
    tauriApi.on("viewport_frame", (frame) => seen.push(frame));
    tauriApi.on("settings_changed", (e) => seen.push(e));
    tauriApi.on("export_busy", (e) => seen.push(e));
    await Promise.resolve();
    handlers.get("viewport_frame")?.({ payload: { seq: 3, width: 2040, height: 1360, sharp: true, url: "ym-frame://localhost/frame-3.png" } });
    handlers.get("settings_changed")?.({ payload: { photo_id: 4 } });
    handlers.get("export_busy")?.({ payload: { busy: true } });
    expect(seen).toEqual([{ seq: 3, width: 2040, height: 1360, sharp: true, url: "ym-frame://localhost/frame-3.png" }, { photo_id: 4 }, { busy: true }]);
  });

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
    deliver({ payload: { ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null } });
    expect(seen).toEqual([{ ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null }]);
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
