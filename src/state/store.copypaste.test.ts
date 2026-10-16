/**
 * Adding photos on the Photos screen, and Copy settings and Paste settings (the owner's request of 2026-10-16, `docs/11` §Copy and paste
 * settings): the store against the mock, which plays the core's rules.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import type { ResolvedSettings } from "../api/types";
import { ALREADY_OPEN_NOTICE, createStore, NO_ARW_NOTICE, openedNotice, pastedNotice } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
  localStorage.clear();
  sessionStorage.clear();
});

/** A store on a fast mock with the twelve photos developed, Photos on screen and the current photo's settings loaded. */
async function onPhotos(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
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
  await store.openPhotos();
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  return { api, store };
}

const settingsOf = (store: AppStore): ResolvedSettings => {
  const rs = store.currentSettings();
  if (!rs) throw new Error("no settings");
  return rs;
};
const idOf = (store: AppStore): number => store.currentPhoto()?.id ?? -1;
const names = (store: AppStore): string[] => (store.getState().session?.photos ?? []).map((p) => p.file_name);

/** Edits the current photo with a patch of every kind: a manual exposure, contrast, a look, a crop and a turn. */
async function editCurrent(store: AppStore): Promise<void> {
  await store.update({
    exposure: { kind: "manual", ev: 0.8 },
    contrast: 20,
    look: "builtin:signature",
    look_intensity: 70,
    crop: { rect: { x: 0.1, y: 0.1, width: 0.6, height: 0.6 }, angle_deg: 2, aspect: "free" },
    orientation: 90,
  });
}

describe("the notices", () => {
  it("say what an open did on Photos", () => {
    expect(openedNotice({ added: 3, already_open: 0 })).toBe("Added 3 photos.");
    expect(openedNotice({ added: 1, already_open: 0 })).toBe("Added 1 photo.");
    expect(openedNotice({ added: 3, already_open: 2 })).toBe("Added 3 photos. 2 were already open.");
    expect(openedNotice({ added: 1, already_open: 1 })).toBe("Added 1 photo. 1 was already open.");
    expect(openedNotice({ added: 0, already_open: 4 })).toBe(ALREADY_OPEN_NOTICE);
    expect(ALREADY_OPEN_NOTICE).toBe("These photos are already open.");
    expect(openedNotice({ added: 0, already_open: 0 })).toBe(NO_ARW_NOTICE);
  });

  it("say what a paste did, and how many photos were left out", () => {
    expect(pastedNotice("DSC04213", 1, 0)).toBe("Pasted the settings of DSC04213.");
    expect(pastedNotice("DSC04213", 5, 0)).toBe("Pasted the settings of DSC04213 to 5 photos.");
    expect(pastedNotice("DSC04213", 4, 1)).toBe("Pasted the settings of DSC04213 to 4 photos. 1 photo was skipped because it isn't developed.");
    expect(pastedNotice("DSC04213", 1, 3)).toBe("Pasted the settings of DSC04213. 3 photos were skipped because they aren't developed.");
    expect(pastedNotice("DSC04213", 0, 1)).toBe("Nothing was pasted: 1 photo isn't developed.");
    expect(pastedNotice("DSC04213", 0, 2)).toBe("Nothing was pasted: 2 photos aren't developed.");
  });

  it("use no exclamation marks", () => {
    const all = [openedNotice({ added: 2, already_open: 1 }), ALREADY_OPEN_NOTICE, pastedNotice("A", 2, 1), pastedNotice("A", 0, 1)];
    for (const text of all) expect(text).not.toContain("!");
  });
});

describe("adding photos on the Photos screen", () => {
  it("adds new files after the others, keeps the current photo and says what was added", async () => {
    const { api, store } = await onPhotos({ current: 3 });
    await store.setCurrent(5);
    const before = store.getState().session!;
    const current = idOf(store);
    api.queueChoice("open", ["~/Pictures/Lisbon/DSC09002.ARW", "~/Pictures/Lisbon/DSC09001.ARW", "~/Pictures/Lisbon/DSC04201.ARW"]);
    await store.openDialog("files");

    const session = store.getState().session!;
    expect(names(store)).toEqual([...before.photos.map((p) => p.file_name), "DSC09001.ARW", "DSC09002.ARW"]);
    expect(session.photos.slice(0, 12).map((p) => p.id)).toEqual(before.photos.map((p) => p.id));
    expect(idOf(store)).toBe(current);
    expect(session.current_index).toBe(before.current_index);
    expect(store.getState().notice).toBe("Added 2 photos. 1 was already open.");
    expect(store.getState().screen).toBe("photos");
    // The new photos develop and get thumbnails while the others stay as they were.
    expect(session.photos.slice(12).map((p) => p.status.kind)).not.toContain("ready");
    api.flush();
    await vi.waitFor(() => expect(store.getState().session!.photos.slice(12).every((p) => p.status.kind === "ready")).toBe(true));
    await vi.waitFor(() => expect(Object.keys(store.getState().thumbnails)).toHaveLength(14));
    expect(store.getState().session!.photos.slice(0, 12).every((p) => p.status.kind === "ready")).toBe(true);
  });

  it("says so when every chosen file is open already, and changes nothing", async () => {
    const { store } = await onPhotos();
    const before = names(store);
    await store.openDialog("files");
    expect(store.getState().notice).toBe("These photos are already open.");
    expect(names(store)).toEqual(before);
    await store.openDialog("folder");
    expect(store.getState().notice).toBe("These photos are already open.");
    expect(store.getState().error).toBeNull();
  });

  it("adds dropped files the same way, and says so when the drop holds no ARW files", async () => {
    const { store } = await onPhotos();
    await store.openPaths(["DSC09100.ARW", "DSC04202.ARW", "notes.txt"]);
    expect(names(store)).toHaveLength(13);
    expect(store.getState().notice).toBe("Added 1 photo. 1 was already open.");
    await store.openPaths(["holiday.jpg", "notes.txt"]);
    // On Photos there is no drop zone to say it in: the notice bar does.
    expect(store.getState().notice).toBe(NO_ARW_NOTICE);
    expect(store.getState().dropNotice).toBeNull();
    expect(names(store)).toHaveLength(13);
  });

  it("is quiet on Start, where the list of developing photos says it all", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const store = createStore(api, 0);
    stops.push(store.connect());
    await store.load();
    await store.openDialog("files");
    expect(store.getState().screen).toBe("start");
    expect(store.getState().notice).toBeNull();
    await store.openDialog("files");
    expect(store.getState().notice).toBeNull();
  });

  it("leaves the settings of the photos alone", async () => {
    const { store } = await onPhotos();
    await editCurrent(store);
    const was = settingsOf(store).settings;
    const id = idOf(store);
    await store.openPaths(["DSC09100.ARW"]);
    expect(idOf(store)).toBe(id);
    expect(settingsOf(store).settings).toEqual(was);
  });
});

describe("Copy settings", () => {
  it("keeps the current photo's settings, says so, and leaves out the crop, the turn and the white pick", async () => {
    const { store } = await onPhotos();
    await editCurrent(store);
    await store.copySettings();
    const copied = store.getState().copied!;
    expect(copied.source).toBe("DSC04201");
    expect(store.getState().notice).toBe("Copied the settings of DSC04201.");
    expect(copied.patch).toMatchObject({ contrast: 20, look: "builtin:signature", look_intensity: 70, exposure: { kind: "manual", ev: 0.8 } });
    for (const key of ["crop", "orientation", "white_pick"]) expect(copied.patch).not.toHaveProperty(key);
    // Auto stays Auto.
    expect(copied.patch.white_balance).toEqual({ kind: "auto" });
    expect(copied.patch.sharpen).toEqual({ kind: "auto" });
  });

  it("waits for the patches still on their way, so the copy has the last value", async () => {
    const { store } = await onPhotos();
    const sending = store.update({ contrast: 33 });
    await store.copySettings();
    await sending;
    expect(store.getState().copied?.patch.contrast).toBe(33);
  });

  it("does nothing for a photo that is not developed", async () => {
    // The mock holds development after the third photo, so the eighth is still waiting.
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY, holdAt: 3 });
    const store = createStore(api, 0);
    stops.push(store.connect());
    await store.load();
    await store.openDialog("files");
    await vi.waitFor(() => expect(store.getState().session?.photos.filter((p) => p.status.kind === "ready")).toHaveLength(3));
    await store.setCurrent(8);
    expect(store.currentPhoto()?.status.kind).not.toBe("ready");
    const spy = vi.spyOn(api, "copySettings");
    await store.copySettings();
    expect(spy).not.toHaveBeenCalled();
    expect(store.getState().copied).toBeNull();
  });
});

describe("Paste settings", () => {
  it("applies them to the current photo, except its crop and turn, as one undo step", async () => {
    const { store } = await onPhotos();
    await editCurrent(store);
    await store.copySettings();
    await store.setCurrent(2);
    await vi.waitFor(() => expect(store.currentSettings()?.photo_id).toBe(2));
    // This photo has a crop of its own.
    await store.update({ crop: { rect: { x: 0, y: 0.2, width: 0.9, height: 0.5 }, angle_deg: -1, aspect: "original" } });
    await store.pasteSettings("this");

    const rs = settingsOf(store);
    expect(store.getState().notice).toBe("Pasted the settings of DSC04201.");
    expect(rs.settings).toMatchObject({ contrast: 20, look: "builtin:signature", look_intensity: 70, exposure: { kind: "manual", ev: 0.8 }, orientation: 0 });
    expect(rs.settings.crop?.angle_deg).toBe(-1);
    expect(rs.values.exposure_ev).toBe(0.8);

    // One Undo takes the whole paste back and leaves the crop alone; Redo does it again.
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    await vi.waitFor(() => expect(settingsOf(store).settings.contrast).toBe(0));
    expect(settingsOf(store).settings).toMatchObject({ exposure: { kind: "auto" }, look_intensity: 100 });
    expect(settingsOf(store).settings.crop?.angle_deg).toBe(-1);
    await store.redo();
    await vi.waitFor(() => expect(settingsOf(store).settings.contrast).toBe(20));
    expect(settingsOf(store).values.exposure_ev).toBe(0.8);
  });

  it("applies them to every picked photo, the current one too, one undo step in each photo's own history", async () => {
    const { api, store } = await onPhotos();
    await editCurrent(store);
    await store.copySettings();
    await store.setCurrent(3);
    await vi.waitFor(() => expect(store.currentSettings()?.photo_id).toBe(3));
    for (const id of [3, 5, 7]) await store.setPicked(id, true);
    const report = vi.spyOn(api, "pasteSettings");
    await store.pasteSettings("picked");

    // One command for all of them.
    expect(report).toHaveBeenCalledTimes(1);
    expect(report.mock.calls[0]![0]).toEqual([3, 5, 7]);
    expect(store.getState().notice).toBe("Pasted the settings of DSC04201 to 3 photos.");
    for (const id of [3, 5, 7]) expect((await api.getSettings(id)).settings.contrast).toBe(20);
    expect((await api.getSettings(1)).settings.contrast).toBe(20);
    expect((await api.getSettings(2)).settings.contrast).toBe(0);
    // The photos that were not shown have no settings in the mirror until they are.
    expect(store.getState().settings[5]).toBeUndefined();
    expect(settingsOf(store).settings.contrast).toBe(20);

    // Each photo undoes its own paste.
    for (const id of [5, 7, 3]) {
      await store.setCurrent(id);
      await vi.waitFor(() => expect(store.currentSettings()?.photo_id).toBe(id));
      expect(store.getState().canUndo, `photo ${id}`).toBe(true);
      await store.undo();
      await vi.waitFor(() => expect(settingsOf(store).settings.contrast).toBe(0));
    }
    expect((await api.getSettings(5)).settings.contrast).toBe(0);
    expect((await api.getSettings(1)).settings.contrast).toBe(20);
  });

  it("counts the photos it had to skip", async () => {
    const { store } = await onPhotos({ picked: [1, 2], skipped: { DSC04203: "This file is not a raw file." } });
    await store.update({ contrast: 15 });
    await store.copySettings();
    await store.pasteSettings("picked");
    // DSC04202 took it; DSC04203 could not be opened.
    expect(store.getState().notice).toBe("Pasted the settings of DSC04201. 1 photo was skipped because it isn't developed.");
    expect(store.getState().session!.photos[2]!.status.kind).toBe("skipped");
  });

  it("says why nothing happened when nothing is copied or nothing is picked", async () => {
    const { api, store } = await onPhotos();
    const spy = vi.spyOn(api, "pasteSettings");
    await store.pasteSettings("this");
    expect(store.getState().notice).toBe("Copy the settings of a photo first.");
    await store.copySettings();
    await store.pasteSettings("picked");
    expect(store.getState().notice).toBe("No photos are picked.");
    expect(spy).not.toHaveBeenCalled();
  });

  it("shows the core's refusal and records nothing", async () => {
    const { api, store } = await onPhotos();
    await store.update({ contrast: 9 });
    await store.copySettings();
    await store.setCurrent(2);
    await vi.waitFor(() => expect(store.currentSettings()?.photo_id).toBe(2));
    api.failNext("pasteSettings", { scope: "looks", message: "That look isn't in the library any more. Import it again." });
    await store.pasteSettings("this");
    expect(store.getState().error?.message).toBe("That look isn't in the library any more. Import it again.");
    expect(store.getState().canUndo).toBe(false);
    expect(settingsOf(store).settings.contrast).toBe(0);
  });

  it("does not undo a paste on the photo it came from when there was nothing to change", async () => {
    const { store } = await onPhotos();
    await store.update({ contrast: 9 });
    await store.copySettings();
    const undoDepth = store.getState().canUndo;
    await store.pasteSettings("this");
    expect(store.getState().notice).toBe("Pasted the settings of DSC04201.");
    // The photo already had them: no new step (the one for the contrast edit is still there).
    await store.undo();
    await vi.waitFor(() => expect(settingsOf(store).settings.contrast).toBe(0));
    expect(undoDepth).toBe(true);
  });
});

describe("what is copied stays in memory", () => {
  it("is kept when the session closes, so it can go to another folder", async () => {
    const { store } = await onPhotos();
    await store.update({ contrast: 12 });
    await store.copySettings();
    await store.closeSession();
    expect(store.getState().session).toBeNull();
    expect(store.getState().copied?.source).toBe("DSC04201");
    await store.openDialog("files");
    await vi.waitFor(() => expect(store.getState().session?.photos).toHaveLength(12));
    await store.openPhotos();
    await store.pasteSettings("this");
  });

  it("is never written to disk or put on the system clipboard", async () => {
    const { api, store } = await onPhotos();
    const clipboard = { writeText: vi.fn(), readText: vi.fn(), write: vi.fn(), read: vi.fn() };
    Object.defineProperty(navigator, "clipboard", { value: clipboard, configurable: true });
    const preferences = JSON.stringify(await api.preferencesGet());
    const setPreferences = vi.spyOn(api, "preferencesSet");
    await store.update({ contrast: 12 });
    await store.copySettings();
    await store.pasteSettings("this");
    expect(JSON.stringify(await api.preferencesGet())).toBe(preferences);
    expect(setPreferences).not.toHaveBeenCalled();
    expect(localStorage.length + sessionStorage.length).toBe(0);
    for (const fn of Object.values(clipboard)) expect(fn).not.toHaveBeenCalled();
    expect(api.system).toEqual({ opened: [], revealed: [], trashed: [], restarts: 0 });
  });
});

describe("the native menu's items", () => {
  it("follow what can be done: a developed photo, something copied, photos picked", async () => {
    const { api, store } = await onPhotos();
    await vi.waitFor(() => expect(api.menuState).toEqual({ copy_settings: true, paste_settings: false, paste_settings_to_picked: false }));
    await store.copySettings();
    await vi.waitFor(() => expect(api.menuState).toEqual({ copy_settings: true, paste_settings: true, paste_settings_to_picked: false }));
    await store.setPicked(4, true);
    await vi.waitFor(() => expect(api.menuState).toEqual({ copy_settings: true, paste_settings: true, paste_settings_to_picked: true }));
    await store.setPicked(4, false);
    await vi.waitFor(() => expect(api.menuState?.paste_settings_to_picked).toBe(false));
    // Nothing open: nothing to copy or paste onto, though what was copied is kept.
    await store.closeSession();
    await vi.waitFor(() => expect(api.menuState).toEqual({ copy_settings: false, paste_settings: false, paste_settings_to_picked: false }));
  });

  it("is told only when it changes", async () => {
    const { api, store } = await onPhotos();
    const spy = vi.spyOn(api, "setMenuState");
    await vi.waitFor(() => expect(api.menuState?.copy_settings).toBe(true));
    spy.mockClear();
    await store.update({ contrast: 5 });
    await store.setZoom({ kind: "percent", value: 100 });
    await store.toggleBefore();
    expect(spy).not.toHaveBeenCalled();
  });
});
