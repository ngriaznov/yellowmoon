import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import type { ResolvedSettings, SettingsPatch } from "../api/types";
import { applyDirect, createStore, NO_ARW_NOTICE } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
});

/** A store on a fast mock, connected, with nothing open. */
async function connected(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, ...options });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  return { api, store };
}

/** A store with the twelve photos open and developed, and the current photo's settings loaded. */
async function opened(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
  const ctx = await connected(options);
  await ctx.store.openDialog("files");
  ctx.api.flush();
  await vi.waitFor(() => {
    const photos = ctx.store.getState().session?.photos ?? [];
    expect(photos).toHaveLength(12);
    expect(photos.every((p) => p.status.kind === "ready" || p.status.kind === "skipped")).toBe(true);
  });
  await vi.waitFor(() => expect(ctx.store.currentSettings()).not.toBeNull());
  return ctx;
}

const currentId = (store: AppStore): number => store.currentPhoto()?.id ?? -1;
const settingsOf = (store: AppStore): ResolvedSettings => {
  const rs = store.currentSettings();
  if (!rs) throw new Error("no settings");
  return rs;
};

describe("connecting", () => {
  it("loads the look library", async () => {
    const { store } = await connected();
    const looks = store.getState().looks;
    expect(looks?.map((l) => l.name).slice(0, 2)).toEqual(["Neutral", "Signature"]);
    expect(looks).toHaveLength(9);
  });

  it("starts on the Start screen with nothing open", async () => {
    const { store } = await connected();
    const s = store.getState();
    expect([s.screen, s.session, s.before, s.zoom, s.applyScope]).toEqual(["start", null, false, { kind: "fit" }, "this"]);
  });

  it("goes straight to Photos when the core already has a session (a reloaded window)", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    await api.openDialog("files");
    api.flush();
    const store = createStore(api, 0);
    stops.push(store.connect());
    await vi.waitFor(() => expect(store.getState().screen).toBe("photos"));
    expect(store.getState().session?.photos).toHaveLength(12);
  });

  it("gets the thumbnails again when the core already has a session (a reloaded window)", async () => {
    // The core announces each thumbnail once; a window that reloaded would show an empty filmstrip without asking.
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const before = createStore(api, 0);
    const stopBefore = before.connect();
    await before.load();
    await before.openDialog("files");
    api.flush();
    await vi.waitFor(() => expect(Object.keys(before.getState().thumbnails)).toHaveLength(12));
    const urls = { ...before.getState().thumbnails };
    stopBefore();

    const after = createStore(api, 0);
    stops.push(after.connect());
    await vi.waitFor(() => expect(Object.keys(after.getState().thumbnails)).toHaveLength(12));
    expect(after.getState().thumbnails).toEqual(urls);
    // Only addresses came back: no picture crossed the bridge for any of them.
    for (const url of Object.values(urls)) expect(url).toMatch(/^(blob:|data:image\/png;base64,)/);
  });

  it("keeps a thumbnail an event brought over the answer to a request that was sent earlier", async () => {
    const { api, store } = await opened();
    const [id] = Object.keys(store.getState().thumbnails).map(Number);
    api.emitEvent("thumbnail_ready", { id: id ?? 1, url: "ym-frame://localhost/thumb-1-999.png" });
    await vi.waitFor(() => expect(store.getState().thumbnails[id ?? 1]).toBe("ym-frame://localhost/thumb-1-999.png"));
    await store.refreshThumbnails();
    expect(store.getState().thumbnails[id ?? 1]).toBe("ym-frame://localhost/thumb-1-999.png");
  });

  it("reloads the looks when the core says they changed", async () => {
    const { api, store } = await connected({ looksImported: false });
    expect(store.getState().looks).toHaveLength(2);
    await api.looksImport("~/Downloads/ARRI_LookLibrary_LogC4.zip");
    await vi.waitFor(() => expect(store.getState().looks).toHaveLength(89));
  });

  it("shows an error the core reports, until it is dismissed", async () => {
    const { api, store } = await connected();
    api.failNext("looksList", { scope: "looks", message: "The looks folder can't be read. Check its permissions." });
    await store.refreshLooks();
    expect(store.getState().error).toEqual({ scope: "looks", message: "The looks folder can't be read. Check its permissions." });
    store.dismissError();
    expect(store.getState().error).toBeNull();
  });
});

describe("opening photos", () => {
  it("opens the session and follows development through the events", async () => {
    const { api, store } = await connected();
    await store.openDialog("files");
    const first = store.getState().session;
    expect(first?.photos).toHaveLength(12);
    expect(first?.folder_name).toBe("Lisbon");
    api.flush();
    await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
    expect(Object.keys(store.getState().thumbnails)).toHaveLength(12);
  });

  it("draws a burst of events together instead of once per event", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const store = createStore(api, 20);
    stops.push(store.connect());
    await store.load();
    await store.openDialog("files");
    let notifications = 0;
    store.subscribe(() => notifications++);
    let events = 0;
    api.on("photo_progress", () => events++);
    api.on("thumbnail_ready", () => events++);
    api.flush();
    await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
    expect(events).toBeGreaterThan(40);
    expect(notifications).toBeLessThan(events / 5);
    expect(Object.keys(store.getState().thumbnails)).toHaveLength(12);
  });

  it("applies statuses that arrive before the session does", async () => {
    // The mock sends the first photo's first event ahead of the command's own result.
    const { store } = await connected({ speed: 1000 });
    await store.openDialog("files");
    await vi.waitFor(() => expect(store.getState().session?.photos[0]?.status.kind).not.toBe("waiting"));
  });

  it("opens dropped ARW files and folders and ignores anything else", async () => {
    const { api, store } = await connected();
    const spy = vi.spyOn(api, "openPaths");
    await store.openPaths(["/a/DSC1.ARW", "/a/notes.txt", "/a/photo.JPG", "/a/Folder"]);
    expect(spy).toHaveBeenCalledWith(["/a/DSC1.ARW", "/a/Folder"]);
    expect(store.getState().session?.photos).toHaveLength(12);
  });

  it("says so when a drop holds nothing to open, and clears the notice on the next open", async () => {
    const { api, store } = await connected();
    const spy = vi.spyOn(api, "openPaths");
    await store.openPaths(["/a/notes.txt", "/a/photo.jpg"]);
    expect(spy).not.toHaveBeenCalled();
    expect(store.getState().dropNotice).toBe(NO_ARW_NOTICE);
    expect(store.getState().session).toBeNull();
    await store.openPaths(["/a/DSC1.ARW"]);
    expect(store.getState().dropNotice).toBeNull();
  });

  describe("in the app", () => {
    // Inside Tauri the webview gets real paths, so a name says nothing about whether a path is a folder.
    const inApp = (): void => {
      (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {};
    };
    afterEach(() => {
      delete (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__;
    });

    it("hands every dropped path to the core, also a folder whose name has a dot and letters", async () => {
      inApp();
      const { api, store } = await connected();
      const spy = vi.spyOn(api, "openPaths");
      const paths = ["/Users/me/Sony.A7C", "/Users/me/2026.Lisbon", "/Users/me/Photos/Trip.Day2", "/Users/me/2026.10.04_Lisbon", "/Users/me/notes.txt"];
      await store.openPaths(paths);
      expect(spy).toHaveBeenCalledWith(paths);
      expect(store.getState().dropNotice).toBeNull();
    });

    it("shows the core's \"No ARW files\" answer as the drop notice, not as an error", async () => {
      inApp();
      const { api, store } = await connected();
      api.failNext("openPaths", { scope: "open", message: NO_ARW_NOTICE });
      const spy = vi.spyOn(api, "openPaths");
      await store.openPaths(["/Users/me/Trip.Day2"]);
      expect(spy).toHaveBeenCalledTimes(1);
      expect(store.getState().dropNotice).toBe(NO_ARW_NOTICE);
      expect(store.getState().error).toBeNull();
      expect(store.getState().opening).toBe(false);
    });

    it("still shows any other open error in the error bar", async () => {
      inApp();
      const { api, store } = await connected();
      api.failNext("openPaths", { scope: "open", message: "That file isn't there any more. Check that its drive is connected, then open it again." });
      await store.openPaths(["/Volumes/Gone/DSC1.ARW"]);
      expect(store.getState().dropNotice).toBeNull();
      expect(store.getState().error?.message).toContain("isn't there");
    });

    it("says so at once when the drop holds no paths", async () => {
      inApp();
      const { api, store } = await connected();
      const spy = vi.spyOn(api, "openPaths");
      await store.openPaths([]);
      expect(spy).not.toHaveBeenCalled();
      expect(store.getState().dropNotice).toBe(NO_ARW_NOTICE);
    });
  });

  it("changes nothing when the dialog is cancelled", async () => {
    const { api, store } = await connected();
    api.cancelNextDialog();
    await store.openDialog("folder");
    expect(store.getState().session).toBeNull();
    expect(store.getState().opening).toBe(false);
  });

  it("shows an open error and stops waiting", async () => {
    const { api, store } = await connected();
    api.failNext("openDialog", { scope: "open", message: "That folder can't be read. Choose another." });
    await store.openDialog("folder");
    expect(store.getState().error?.message).toBe("That folder can't be read. Choose another.");
    expect(store.getState().opening).toBe(false);
  });

  it("keeps a file that can't be developed in the session with its reason", async () => {
    const { store } = await opened({ skipped: { DSC04207: "This file is from a camera mode we can't read yet." } });
    const bad = store.getState().session?.photos[6];
    expect(bad?.status).toEqual({ kind: "skipped", reason: "This file is from a camera mode we can't read yet." });
    expect(store.getState().session?.photos.filter((p) => p.status.kind === "ready")).toHaveLength(11);
  });

  it("Open photos goes to the first photo that is ready when the current one is not", async () => {
    const { api, store } = await connected({ skipped: { DSC04201: "Can't read this one." } });
    await store.openDialog("files");
    api.flush();
    await vi.waitFor(() => expect(store.getState().session?.photos[1]?.status.kind).toBe("ready"));
    await store.openPhotos();
    expect(store.getState().screen).toBe("photos");
    expect(currentId(store)).toBe(2);
  });
});

describe("the current photo, picks and navigation", () => {
  it("loads the settings of the photo that is shown", async () => {
    const { store } = await opened({ current: 3 });
    expect(currentId(store)).toBe(4);
    expect(settingsOf(store).photo_id).toBe(4);
  });

  it("moves with step() and stops at both ends", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "setCurrent");
    await store.step(-1);
    expect(currentId(store)).toBe(1);
    await store.step(1);
    expect(currentId(store)).toBe(2);
    expect(spy).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(settingsOf(store).photo_id).toBe(2));
  });

  it("toggles a pick and mirrors the picked ids", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "setPicked");
    await store.togglePick();
    expect(store.getState().session?.picked).toEqual([1]);
    expect(spy).toHaveBeenCalledWith(1, true);
    await store.togglePick();
    expect(store.getState().session?.picked).toEqual([]);
  });

  it("picks every photo that is ready with pickAll, and none that failed", async () => {
    const { store } = await opened({ skipped: { DSC04203: "Can't read this one." } });
    await store.pickAll();
    const picked = store.getState().session?.picked ?? [];
    expect(picked).toHaveLength(11);
    expect(picked).not.toContain(3);
  });

  it("does not pick a photo that can't be opened", async () => {
    const { api, store } = await opened({ skipped: { DSC04203: "Can't read this one." } });
    const spy = vi.spyOn(api, "setPicked");
    await store.setPicked(3, true);
    expect(spy).not.toHaveBeenCalled();
  });

  it("goes to the wizard and back to where it came from", async () => {
    const { store } = await opened();
    store.go("photos");
    store.go("import-looks");
    expect(store.getState().screen).toBe("import-looks");
    store.back();
    expect(store.getState().screen).toBe("photos");
  });
});

describe("applyDirect", () => {
  it("shows a patch at once for the fields that need no resolving", async () => {
    const { store } = await opened();
    const rs = settingsOf(store);
    const next = applyDirect(rs, {
      white: 12,
      contrast: { kind: "manual", value: 20 },
      output_sharpening: 35,
      light_source: "tungsten",
      exposure: { kind: "manual", ev: 1 },
    });
    expect(next.settings.white).toBe(12);
    expect(next.values.white).toBe(12);
    // Contrast and saturation are Auto or a value (`docs/19`): the value shows at once, and so does the AUTO tag going.
    expect(next.settings.contrast).toEqual({ kind: "manual", value: 20 });
    expect(next.values.contrast).toBe(20);
    expect(next.auto.contrast).toBe(false);
    // The output sharpening is a plain number: the slider shows it at once, without waiting for the
    // core (whose answer comes after the stage has run on the new amount).
    expect(next.settings.output_sharpening).toBe(35);
    expect(next.values.output_sharpening).toBe(35);
    expect(next.auto.light_source).toBe(false);
    // Exposure has to be resolved by the core.
    expect(next.settings.exposure).toEqual({ kind: "auto" });
    expect(next.values.exposure_ev).toBe(rs.values.exposure_ev);
  });
});

describe("changing settings", () => {
  it("shows an output sharpening change at once, while the core is still working on it", async () => {
    const { api, store } = await opened();
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const real = api.updateSettings.bind(api);
    vi.spyOn(api, "updateSettings").mockImplementation(async (id, patch) => {
      await gate;
      return real(id, patch);
    });
    const done = store.update({ output_sharpening: 35 });
    // Not answered yet: the mirror already holds the new amount, so the slider does not jump back.
    expect(settingsOf(store).values.output_sharpening).toBe(35);
    expect(settingsOf(store).settings.output_sharpening).toBe(35);
    release();
    await done;
    expect(settingsOf(store).values.output_sharpening).toBe(35);
  });

  it("sends the patch for the current photo and mirrors the resolved reply", async () => {
    const { api, store } = await opened({ current: 3 });
    const spy = vi.spyOn(api, "updateSettings");
    await store.update({ exposure: { kind: "manual", ev: 1.25 } });
    expect(spy).toHaveBeenCalledWith(4, { exposure: { kind: "manual", ev: 1.25 } });
    const rs = settingsOf(store);
    expect(rs.values.exposure_ev).toBe(1.25);
    expect(rs.auto.exposure).toBe(false);
  });

  it("does nothing before the settings have arrived", async () => {
    const { api, store } = await connected();
    const spy = vi.spyOn(api, "updateSettings");
    await store.update({ white: 4 });
    expect(spy).not.toHaveBeenCalled();
  });

  it("folds a drag's patches into one while the core is busy, and the last value always arrives", async () => {
    const { api, store } = await opened();
    const calls: SettingsPatch[] = [];
    const real = api.updateSettings.bind(api);
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    vi.spyOn(api, "updateSettings").mockImplementation(async (id, patch) => {
      calls.push(patch);
      if (calls.length === 1) await gate;
      return real(id, patch);
    });
    const first = store.update({ white: 10 }, false);
    const second = store.update({ white: 20 }, false);
    const third = store.update({ white: 30, brightness: 5 }, false);
    const last = store.update({ white: 40 }, true);
    release();
    await Promise.all([first, second, third, last]);
    // One in flight, then everything that came after it, as one patch.
    expect(calls).toEqual([{ white: 10 }, { white: 40, brightness: 5 }]);
    expect(settingsOf(store).values.white).toBe(40);
    expect(settingsOf(store).values.brightness).toBe(5);
  });

  it("does not fold a new gesture into the patch that ends the last one", async () => {
    const { api, store } = await opened();
    const calls: SettingsPatch[] = [];
    const real = api.updateSettings.bind(api);
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    vi.spyOn(api, "updateSettings").mockImplementation(async (id, patch) => {
      calls.push(patch);
      if (calls.length === 1) await gate;
      return real(id, patch);
    });
    const a = store.update({ white: 10 }, true);
    const b = store.update({ white: 20 }, true);
    const c = store.update({ brightness: 7 }, false);
    release();
    await Promise.all([a, b, c]);
    expect(calls).toEqual([{ white: 10 }, { white: 20 }, { brightness: 7 }]);
  });

  it("shows the error and asks the core again when a patch fails", async () => {
    const { api, store } = await opened();
    api.failNext("updateSettings", { scope: "render", message: "The photo couldn't be drawn. Try the change again." });
    await store.update({ white: 33 });
    expect(store.getState().error?.message).toBe("The photo couldn't be drawn. Try the change again.");
    // The optimistic value was taken back by asking the core.
    await vi.waitFor(() => expect(settingsOf(store).values.white).toBe(0));
  });

  it("keeps each photo's settings apart", async () => {
    const { store } = await opened();
    await store.update({ white: 9 });
    await store.step(1);
    await vi.waitFor(() => expect(settingsOf(store).photo_id).toBe(2));
    expect(settingsOf(store).values.white).toBe(0);
    await store.step(-1);
    expect(settingsOf(store).values.white).toBe(9);
  });
});

describe("undo and redo", () => {
  it("takes back a change and puts it back, for the current photo", async () => {
    const { api, store } = await opened();
    await store.update({ white: 25 });
    expect(store.getState().canUndo).toBe(true);
    const spy = vi.spyOn(api, "updateSettings");
    await store.undo();
    expect(spy).toHaveBeenLastCalledWith(1, { white: 0 });
    expect(settingsOf(store).values.white).toBe(0);
    expect(store.getState().canRedo).toBe(true);
    await store.redo();
    expect(spy).toHaveBeenLastCalledWith(1, { white: 25 });
    expect(settingsOf(store).values.white).toBe(25);
    expect(store.getState().canRedo).toBe(false);
  });

  it("undoes the output sharpening like any other field, back to 100", async () => {
    const { api, store } = await opened();
    await store.update({ output_sharpening: 35 });
    expect(settingsOf(store).values.output_sharpening).toBe(35);
    const spy = vi.spyOn(api, "updateSettings");
    await store.undo();
    expect(spy).toHaveBeenLastCalledWith(1, { output_sharpening: 100 });
    expect(settingsOf(store).values.output_sharpening).toBe(100);
    await store.redo();
    expect(spy).toHaveBeenLastCalledWith(1, { output_sharpening: 35 });
    expect(settingsOf(store).values.output_sharpening).toBe(35);
    // A whole drag of the slider is one step.
    await store.update({ output_sharpening: 60 }, false);
    await store.update({ output_sharpening: 80 }, false);
    await store.update({ output_sharpening: 150 }, true);
    await store.undo();
    expect(settingsOf(store).values.output_sharpening).toBe(35);
  });

  it("records a whole drag as one step", async () => {
    const { store } = await opened();
    await store.update({ white: -5 }, false);
    await store.update({ white: -12 }, false);
    await store.update({ white: -18 }, true);
    await store.undo();
    expect(settingsOf(store).values.white).toBe(0);
    expect(store.getState().canUndo).toBe(false);
  });

  it("takes an Auto field back to Auto, not to the number it resolved to", async () => {
    const { store } = await opened();
    await store.update({ exposure: { kind: "manual", ev: 1 } });
    await store.undo();
    expect(settingsOf(store).settings.exposure).toEqual({ kind: "auto" });
    expect(settingsOf(store).auto.exposure).toBe(true);
  });

  it("has a history for each photo", async () => {
    const { store } = await opened();
    await store.update({ white: 25 });
    await store.step(1);
    await vi.waitFor(() => expect(settingsOf(store).photo_id).toBe(2));
    expect(store.getState().canUndo).toBe(false);
    await store.undo();
    await store.step(-1);
    expect(settingsOf(store).values.white).toBe(25);
    expect(store.getState().canUndo).toBe(true);
  });

  it("a new change ends the redo chain", async () => {
    const { store } = await opened();
    await store.update({ white: 5 });
    await store.undo();
    await store.update({ brightness: 6 });
    expect(store.getState().canRedo).toBe(false);
  });

  it("does nothing when there is nothing to undo", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "updateSettings");
    await store.undo();
    await store.redo();
    expect(spy).not.toHaveBeenCalled();
  });

  it("keeps 100 steps", async () => {
    const { store } = await opened();
    // Values cycle between 2 and 90, so every change differs from the one before.
    const value = (i: number): number => (i % 90) + 1;
    for (let i = 1; i <= 105; i++) await store.update({ white: value(i) });
    for (let i = 0; i < 120; i++) await store.undo();
    // 105 changes, the oldest five forgotten: the oldest step left takes white back to what change 5 made it.
    expect(settingsOf(store).values.white).toBe(value(5));
  });

  it("undoes Re-run", async () => {
    const { store } = await opened();
    await store.update({ exposure: { kind: "manual", ev: 2 }, noise_reduction: { kind: "manual", luminance: 80, color: 50, detail: 50 } });
    await store.rerunAuto();
    expect(settingsOf(store).auto.exposure).toBe(true);
    expect(settingsOf(store).auto.noise_reduction).toBe(true);
    await store.undo();
    expect(settingsOf(store).values.exposure_ev).toBe(2);
    expect(settingsOf(store).values.noise_reduction.luminance).toBe(80);
  });

  it("waits for a patch in flight before it undoes", async () => {
    const { store } = await opened();
    void store.update({ white: 14 }, true);
    await store.undo();
    expect(settingsOf(store).values.white).toBe(0);
  });
});

describe("Re-run", () => {
  it("calls rerun_auto for the current photo and mirrors the answer", async () => {
    const { api, store } = await opened({ current: 2 });
    await store.update({ exposure: { kind: "manual", ev: 2 } });
    const spy = vi.spyOn(api, "rerunAuto");
    await store.rerunAuto();
    expect(spy).toHaveBeenCalledWith(3);
    expect(settingsOf(store).auto.exposure).toBe(true);
  });
});

describe("the folded sections of the Look tab", () => {
  it("start empty, fold and open one section at a time, and keep the order they were folded in", async () => {
    const { store } = await connected();
    expect(store.getState().collapsedLookLibraries).toEqual([]);
    store.toggleLookLibrary("arri_logc3");
    store.toggleLookLibrary("user");
    expect(store.getState().collapsedLookLibraries).toEqual(["arri_logc3", "user"]);
    store.toggleLookLibrary("arri_logc3");
    expect(store.getState().collapsedLookLibraries).toEqual(["user"]);
    store.toggleLookLibrary("user");
    expect(store.getState().collapsedLookLibraries).toEqual([]);
  });

  it("tell the subscribers, and stay when a session is closed (they belong to the app, not the photos)", async () => {
    const { store } = await opened();
    const seen = vi.fn();
    const stop = store.subscribe(seen);
    store.toggleLookLibrary("arri_alf2");
    expect(seen).toHaveBeenCalledTimes(1);
    stop();
    await store.closeSession();
    expect(store.getState().session).toBeNull();
    expect(store.getState().collapsedLookLibraries).toEqual(["arri_alf2"]);
  });

  it("are not in the preferences, which are the only thing the core keeps", async () => {
    const { api, store } = await connected();
    store.toggleLookLibrary("arri_logc4");
    expect(JSON.stringify(await api.preferencesGet())).not.toMatch(/collapsed|arri_logc4/);
  });
});

describe("looks", () => {
  it("picks a look for this photo only", async () => {
    const { api, store } = await opened();
    store.setApplyScope("this");
    const update = vi.spyOn(api, "updateSettings");
    const apply = vi.spyOn(api, "applyLook");
    await store.chooseLook("arri:3000:teal-and-orange");
    expect(update).toHaveBeenCalledWith(1, { look: "arri:3000:teal-and-orange", look_intensity: 100 });
    expect(apply).not.toHaveBeenCalled();
    expect(settingsOf(store).values.look).toBe("arri:3000:teal-and-orange");
  });

  it.each([["picked"], ["all"]] as const)("copies the look to the %s photos through apply_look", async (scope) => {
    const { api, store } = await opened({ picked: [2, 3] });
    store.setApplyScope(scope);
    const apply = vi.spyOn(api, "applyLook");
    await store.chooseLook("builtin:signature", 60);
    expect(apply).toHaveBeenCalledWith(scope, "builtin:signature", 60);
    const state = await api.sessionState();
    expect(state.photos).toHaveLength(12);
    // The other photos now carry it, and the store asks again when one is shown.
    await store.step(1);
    await vi.waitFor(() => expect(settingsOf(store).photo_id).toBe(2));
    if (scope === "all") expect(settingsOf(store).values.look).toBe("builtin:signature");
  });

  it("does not copy a look to the other photos when the current photo could not take it", async () => {
    const { api, store } = await opened();
    store.setApplyScope("all");
    const apply = vi.spyOn(api, "applyLook");
    api.failNext("updateSettings", { scope: "render", message: "The photo couldn't be drawn. Try the change again." });
    await store.chooseLook("arri:3000:teal-and-orange");
    expect(store.getState().error).not.toBeNull();
    expect(apply).not.toHaveBeenCalled();
    // The same for the end of an Intensity drag.
    store.dismissError();
    api.failNext("updateSettings", { scope: "render", message: "The photo couldn't be drawn. Try the change again." });
    await store.setLookIntensity(40, true);
    expect(apply).not.toHaveBeenCalled();
    // And a later change goes through.
    await store.chooseLook("builtin:signature");
    expect(apply).toHaveBeenCalledWith("all", "builtin:signature", 100);
  });

  it("refuses a look that is missing", async () => {
    const { api, store } = await opened({ missingLooks: ["film-warm"] });
    const spy = vi.spyOn(api, "updateSettings");
    await store.chooseLook("arri:3000:film-warm");
    expect(spy).not.toHaveBeenCalled();
    expect(settingsOf(store).values.look).toBe("builtin:neutral");
  });

  it("changes the intensity while dragging and copies it when the drag ends", async () => {
    const { api, store } = await opened();
    store.setApplyScope("all");
    const update = vi.spyOn(api, "updateSettings");
    const apply = vi.spyOn(api, "applyLook");
    await store.setLookIntensity(80, false);
    await store.setLookIntensity(65, false);
    expect(apply).not.toHaveBeenCalled();
    await store.setLookIntensity(60, true);
    expect(update).toHaveBeenLastCalledWith(1, { look_intensity: 60 });
    expect(apply).toHaveBeenCalledWith("all", "builtin:neutral", 60);
  });

  it("makes a look the default and reloads the library", async () => {
    const { store } = await opened();
    await store.makeDefault("builtin:signature");
    expect(store.getState().looks?.find((l) => l.is_default)?.id).toBe("builtin:signature");
  });

  it("makes look thumbnails once per photo, look and settings", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "lookThumbnail");
    const a = await store.lookThumbnail(1, "builtin:signature", "k1");
    const b = await store.lookThumbnail(1, "builtin:signature", "k1");
    expect(a).toBe(b);
    expect(spy).toHaveBeenCalledTimes(1);
    await store.lookThumbnail(1, "builtin:neutral", "k1");
    expect(spy).toHaveBeenCalledTimes(2);
    await store.lookThumbnail(1, "builtin:signature", "k2");
    expect(spy).toHaveBeenCalledTimes(3);
  });

  it("forgets a failed look thumbnail so it can be asked for again", async () => {
    const { api, store } = await opened({ missingLooks: ["film-warm"] });
    await expect(store.lookThumbnail(1, "arri:3000:film-warm", "k")).rejects.toMatchObject({ message: "File missing or changed. Import again." });
    const spy = vi.spyOn(api, "lookThumbnail");
    await expect(store.lookThumbnail(1, "arri:3000:film-warm", "k")).rejects.toBeDefined();
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("hands a tile that asks again after a cancelled request a new request, not the cancelled one", async () => {
    // Photo A, then B, then A again before A's tile was made: the core cancelled A's tile when the person left, and the tile that
    // comes back is handed the request that is still pending.
    const { api, store } = await opened();
    const cancelled = { scope: "render", message: "The render was cancelled." };
    let reject: (error: unknown) => void = () => {};
    const spy = vi.spyOn(api, "lookThumbnail").mockImplementationOnce(() => new Promise<string>((_, no) => (reject = no)));
    const first = store.lookThumbnail(1, "builtin:signature", "k");
    const second = store.lookThumbnail(1, "builtin:signature", "k");
    expect(second).toBe(first);
    reject(cancelled);
    await expect(second).rejects.toEqual(cancelled);
    expect(spy).toHaveBeenCalledTimes(1);
    const again = store.lookThumbnail(1, "builtin:signature", "k");
    expect(again).not.toBe(first);
    expect(await again).toMatch(/^blob:|^data:|^ym-frame:/);
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("does not let an old failure remove a newer request", async () => {
    const { api, store } = await opened();
    let reject: (error: unknown) => void = () => {};
    vi.spyOn(api, "lookThumbnail").mockImplementationOnce(() => new Promise<string>((_, no) => (reject = no)));
    const old = store.lookThumbnail(1, "builtin:signature", "k");
    // The cache is emptied by other means (the picture was found not to load) and a newer request takes its place.
    const oldCache = (store as unknown as { lookThumbs: Map<number, { urls: Map<string, unknown> }> }).lookThumbs.get(1)!.urls;
    oldCache.clear();
    const newer = store.lookThumbnail(1, "builtin:signature", "k");
    reject({ scope: "render", message: "The render was cancelled." });
    await expect(old).rejects.toBeDefined();
    expect(store.lookThumbnail(1, "builtin:signature", "k")).toBe(newer);
  });

  it("forgets the address of a tile that did not load, so the next ask makes a new picture", async () => {
    // The shell dropped the tiles of photos used least recently and answers 404 for their addresses.
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "lookThumbnail");
    const url = await store.lookThumbnail(1, "builtin:signature", "k");
    store.forgetLookThumbnail(1, "builtin:signature", "k", url);
    const fresh = await store.lookThumbnail(1, "builtin:signature", "k");
    expect(spy).toHaveBeenCalledTimes(2);
    expect(fresh).not.toBe(url);
  });

  it("forgets only the address it was told about", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "lookThumbnail");
    const url = await store.lookThumbnail(1, "builtin:signature", "k");
    store.forgetLookThumbnail(1, "builtin:signature", "k", "ym-frame://localhost/look-1-0.png");
    store.forgetLookThumbnail(1, "builtin:signature", "other settings", url);
    store.forgetLookThumbnail(2, "builtin:signature", "k", url);
    expect(await store.lookThumbnail(1, "builtin:signature", "k")).toBe(url);
    expect(spy).toHaveBeenCalledTimes(1);
  });
});

describe("the viewport", () => {
  it("flips Before, and takes it back when the core refuses", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "setBefore");
    await store.toggleBefore();
    expect(store.getState().before).toBe(true);
    expect(spy).toHaveBeenCalledWith(true);
    api.failNext("setBefore", { scope: "viewport", message: "The viewport isn't ready." });
    await store.toggleBefore();
    expect(store.getState().before).toBe(true);
    expect(store.getState().error?.scope).toBe("viewport");
  });

  it("sets the zoom and mirrors a double click toggle", async () => {
    const { api, store } = await opened();
    await store.setZoom({ kind: "percent", value: 50 });
    expect(api.viewport.mode).toEqual({ kind: "percent", value: 50 });
    await store.setZoom({ kind: "fit" });
    await store.viewportInput({ kind: "toggle_zoom", x: 10, y: 10 });
    expect(store.getState().zoom).toEqual({ kind: "percent", value: 100 });
    await store.viewportInput({ kind: "toggle_zoom", x: 10, y: 10 });
    expect(store.getState().zoom).toEqual({ kind: "fit" });
  });

  it("reports the cell's rectangle to the core and samples a point of the cell", async () => {
    const { api, store } = await opened();
    store.setViewportRect({ x: 0, y: 75, width: 1000, height: 666, scale: 2 });
    await vi.waitFor(() => expect(api.viewport.rect?.width).toBe(1000));
    const spy = vi.spyOn(api, "samplePixel");
    await store.samplePixelAt(0.5, 0.25);
    expect(spy).toHaveBeenCalledWith(500, 166.5);
  });

  it("takes the Fit zoom from the core once a frame has reported it, and estimates it until then", async () => {
    const { api, store } = await opened();
    store.setViewportRect({ x: 0, y: 75, width: 1000, height: 666, scale: 2 });
    await vi.waitFor(() => expect(api.viewport.rect?.width).toBe(1000));
    // Until a viewport frame has reported its Fit zoom (the mock never does) the screen works it out from the photo and the cell.
    const estimate = store.fitPercentExact();
    expect(estimate).not.toBeNull();
    // The core's figure counts what the screen cannot see (lens crop, straighten angle); it wins from then on.
    api.emitEvent("render_stats", { ms: 5, path: "classical", space: "display_p3", zoom_percent: 31.2, fit_percent: 31.2 });
    await vi.waitFor(() => expect(store.fitPercentExact()).toBe(31.2));
    expect(store.fitPercent()).toBe(31);
    // A fixed zoom no larger than it shows the whole picture (the core draws Fit for it).
    await store.setZoom({ kind: "percent", value: 30 });
    expect(store.showsWholePhoto()).toBe(true);
    await store.setZoom({ kind: "percent", value: 32 });
    expect(store.showsWholePhoto()).toBe(false);
    // A sender that has no figure (null) leaves the estimate in charge.
    api.emitEvent("render_stats", { ms: 5, path: "classical", space: "display_p3", zoom_percent: null, fit_percent: null });
    await vi.waitFor(() => expect(store.fitPercentExact()).toBe(estimate));
  });

  it("drops the zoom figures of the last frame when another photo opens, and keeps the status text", async () => {
    const { api, store } = await opened();
    store.setViewportRect({ x: 0, y: 75, width: 1000, height: 666, scale: 2 });
    await vi.waitFor(() => expect(api.viewport.rect?.width).toBe(1000));
    api.emitEvent("render_stats", { ms: 5, path: "classical", space: "display_p3", zoom_percent: 31.2, fit_percent: 31.2 });
    await vi.waitFor(() => expect(store.fitPercentExact()).toBe(31.2));
    // The mock answers a navigation with a render_stats of its own (no zoom figures); the core's frame comes later.
    vi.spyOn(api, "setCurrent").mockResolvedValue(undefined);
    await store.step(1);
    // A portrait after a landscape has another Fit zoom: the figure of the photo that was left is not it, whatever the next
    // frame says. Until that frame reports, the screen estimates; the status text of the last frame stays.
    expect(store.getState().stats).toEqual({ ms: 5, path: "classical", space: "display_p3", zoom_percent: null, fit_percent: null });
    expect(store.fitPercentExact()).not.toBe(31.2);
    api.emitEvent("render_stats", { ms: 7, path: "classical", space: "display_p3", zoom_percent: 24, fit_percent: 24 });
    await vi.waitFor(() => expect(store.fitPercentExact()).toBe(24));
  });

  it("updates the status line from render_stats", async () => {
    const { store } = await opened();
    await store.update({ white: 3 });
    await vi.waitFor(() => expect(store.getState().stats).toEqual({ ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null }));
  });

  it("collapses requests for the histogram", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "histogram");
    await Promise.all([store.requestHistogram(), store.requestHistogram(), store.requestHistogram()]);
    expect(spy.mock.calls.length).toBeLessThanOrEqual(2);
    expect(store.getState().histogram?.red.length).toBeGreaterThan(8);
  });
});

describe("closing the session", () => {
  it("returns to a clean Start and keeps the looks", async () => {
    const { api, store } = await opened({ picked: [1] });
    await store.update({ white: 4 });
    await store.setZoom({ kind: "percent", value: 100 });
    await store.toggleBefore();
    await store.closeSession();
    const s = store.getState();
    expect([s.screen, s.session, s.zoom, s.before, s.canUndo]).toEqual(["start", null, { kind: "fit" }, false, false]);
    expect(Object.keys(s.thumbnails)).toHaveLength(0);
    expect(Object.keys(s.settings)).toHaveLength(0);
    expect(s.looks).toHaveLength(9);
    expect((await api.sessionState()).photos).toHaveLength(0);
  });

  it("can open again afterwards, with fresh history", async () => {
    const { api, store } = await opened();
    await store.update({ white: 4 });
    await store.closeSession();
    await store.openDialog("files");
    api.flush();
    await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
    expect(store.getState().canUndo).toBe(false);
    expect(settingsOf(store).values.white).toBe(0);
  });
});

describe("Apple Auto (docs/19)", () => {
  it("asks whether Apple Auto is available each time Preferences opens", async () => {
    const { api, store } = await connected({ appleAuto: "answered" });
    expect(store.getState().appleAuto).toBeNull();
    const spy = vi.spyOn(api, "appleAutoStatus");
    store.openOverlay("preferences");
    await vi.waitFor(() => expect(store.getState().appleAuto?.available).toBe(true));
    expect(store.getState().appleAuto).toMatchObject({ preference: true, active: true, reason: null });
    store.closeOverlay();
    store.openOverlay("preferences");
    await vi.waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
  });

  it("is not available in the browser build's mock", async () => {
    const { store } = await connected();
    store.openOverlay("preferences");
    await vi.waitFor(() => expect(store.getState().appleAuto).not.toBeNull());
    expect(store.getState().appleAuto).toMatchObject({ available: false, reason: null, active: false });
  });

  it("reads the photo's settings again when the answer arrives, so the Auto values follow", async () => {
    const { api, store } = await opened({ appleAuto: "pending" });
    expect(settingsOf(store).estimates.apple).toEqual({ state: "pending" });
    expect(settingsOf(store).values.contrast).toBe(0);
    api.setAppleAuto("answered");
    await vi.waitFor(() => expect(settingsOf(store).estimates.apple.state).toBe("answered"));
    const rs = settingsOf(store);
    expect([rs.estimates.contrast, rs.estimates.saturation, rs.values.contrast, rs.values.saturation]).toEqual([15, -15, 15, -15]);
    expect(rs.estimates.source).toBe("apple");
    expect(rs.auto.contrast && rs.auto.saturation).toBe(true);
    // Nothing the person did: no undo step.
    expect(store.getState().canUndo).toBe(false);
  });

  it("does not let a re-read that a drag overtook move the slider back", async () => {
    // The core's news (an answer, `settings_changed`) makes the store ask for the photo's settings; a drag's patch is sent and
    // answered while that request is still out, and its reply, made before the patch, arrives last.
    const { api, store } = await opened({ appleAuto: "pending" });
    const id = currentId(store);
    const real = api.getSettings.bind(api);
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    let asked = 0;
    let replied = 0;
    vi.spyOn(api, "getSettings").mockImplementation(async (photoId) => {
      if (photoId !== id) return real(photoId);
      asked += 1;
      // The settings as the core has them when it is asked.
      const reply = await real(photoId);
      if (asked === 1) await gate;
      replied += 1;
      return reply;
    });
    api.setAppleAuto("answered");
    await vi.waitFor(() => expect(asked).toBe(1));
    await store.update({ white: 40 }, false);
    expect(settingsOf(store).values.white).toBe(40);
    const shown: number[] = [];
    const stop = store.subscribe(() => shown.push(settingsOf(store).values.white));
    release();
    // The late reply is not shown; the store asks again once the patch has landed, and keeps the drag's value.
    await vi.waitFor(() => expect(replied).toBe(2));
    await new Promise((resolve) => setTimeout(resolve, 0));
    stop();
    expect(asked).toBe(2);
    expect(shown.filter((v) => v !== 40)).toEqual([]);
    expect(settingsOf(store).estimates.apple.state).toBe("answered");
    expect(settingsOf(store).values.white).toBe(40);
    expect(settingsOf(store).values.contrast).toBe(15);
  });

  it("follows the preference: off puts every photo back to the rule, on brings the answers back", async () => {
    const { store } = await opened({ appleAuto: "answered" });
    expect(settingsOf(store).values.contrast).toBe(15);
    await store.setPreferences({ apple_auto: false });
    await vi.waitFor(() => expect(settingsOf(store).estimates.apple).toEqual({ state: "off" }));
    expect(settingsOf(store).values.contrast).toBe(0);
    await vi.waitFor(() => expect(store.getState().appleAuto?.active).toBe(false));
    await store.setPreferences({ apple_auto: true });
    await vi.waitFor(() => expect(settingsOf(store).values.contrast).toBe(15));
    await vi.waitFor(() => expect(store.getState().appleAuto?.active).toBe(true));
  });

  it("sends a value equal to the Auto value as Auto, and undoes a manual contrast back to Auto", async () => {
    const { api, store } = await opened({ appleAuto: "answered" });
    const spy = vi.spyOn(api, "updateSettings");
    await store.update({ contrast: { kind: "manual", value: -20 } });
    expect(settingsOf(store).values.contrast).toBe(-20);
    expect(settingsOf(store).auto.contrast).toBe(false);
    await store.undo();
    expect(spy).toHaveBeenLastCalledWith(1, { contrast: { kind: "auto" } });
    expect(settingsOf(store).values.contrast).toBe(15);
    expect(settingsOf(store).auto.contrast).toBe(true);
    await store.redo();
    expect(spy).toHaveBeenLastCalledWith(1, { contrast: { kind: "manual", value: -20 } });
  });

  it("Re-run puts contrast and saturation back to Auto", async () => {
    const { store } = await opened({ appleAuto: "answered" });
    await store.update({ contrast: { kind: "manual", value: 40 }, saturation: { kind: "manual", value: -40 } });
    await store.rerunAuto();
    const rs = settingsOf(store);
    expect([rs.settings.contrast, rs.settings.saturation]).toEqual([{ kind: "auto" }, { kind: "auto" }]);
    expect([rs.values.contrast, rs.values.saturation]).toEqual([15, -15]);
  });
});
