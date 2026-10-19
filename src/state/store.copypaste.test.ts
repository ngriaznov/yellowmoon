/**
 * Adding photos on the Photos screen, and Copy settings and Paste settings (the owner's request of 2026-10-16, `docs/11` §Copy and paste
 * settings): the store against the mock, which plays the core's rules.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import type { PasteReport, PhotoSummary, ResolvedSettings, SessionState, SettingsPatch } from "../api/types";
import { ALREADY_OPEN_NOTICE, createStore, NO_ARW_NOTICE, openedNotice, pastedNotice } from "./store";
import type { AppStore } from "./store";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

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
    contrast: manual(20),
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

  it("keeps what the page learned after the shell took its snapshot: a status, the current photo and the picks", async () => {
    const { api, store } = await onPhotos({ current: 3 });
    await store.setCurrent(5);
    await store.setPicked(2, true);
    const page = store.getState().session!;
    const added: PhotoSummary = { ...page.photos[0]!, id: 99, file_name: "DSC09001.ARW", status: { kind: "waiting" }, picked: false };
    // The snapshot is older than the page: photo 7 was still developing in it, photo 2 was not picked yet, photo 1 was the current one.
    const snapshot: SessionState = {
      ...page,
      folder_name: "Lisbon and more",
      photos: [...page.photos.map((p): PhotoSummary => (p.id === 7 ? { ...p, status: { kind: "developing", stage: "thumbnail" } } : { ...p, picked: false })), added],
      current_index: 0,
      picked: [],
      opened: { added: 1, already_open: 0 },
    };
    vi.spyOn(api, "openPaths").mockResolvedValue(snapshot);
    await store.openPaths(["DSC09001.ARW"]);

    const session = store.getState().session!;
    expect(session.photos.map((p) => p.id)).toEqual([...page.photos.map((p) => p.id), 99]);
    // The cell that became ready after the snapshot stays ready: no later event would say so again.
    expect(session.photos.find((p) => p.id === 7)?.status.kind).toBe("ready");
    expect(session.photos.find((p) => p.id === 99)?.status.kind).toBe("waiting");
    expect(session.current_index).toBe(page.current_index);
    expect(session.photos.find((p) => p.id === 2)?.picked).toBe(true);
    expect(session.picked).toEqual([2]);
    // What only the snapshot knows is taken from it.
    expect(session.folder_name).toBe("Lisbon and more");
    expect(store.getState().notice).toBe("Added 1 photo.");
    // A picked photo that is ready can still be picked and shown: nothing is stuck.
    await store.setPicked(7, true);
    expect(store.getState().session!.picked).toEqual([2, 7]);
  });

  it("takes the first photo as current when the page had none", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const store = createStore(api, 0);
    stops.push(store.connect());
    await store.load();
    await store.openDialog("files");
    expect(store.getState().session!.current_index).toBe(0);
    await store.closeSession();
    await store.openDialog("files");
    expect(store.getState().session!.current_index).toBe(0);
    expect(store.getState().session!.photos).toHaveLength(12);
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
    expect(copied.patch).toMatchObject({ contrast: manual(20), look: "builtin:signature", look_intensity: 70, exposure: { kind: "manual", ev: 0.8 } });
    for (const key of ["crop", "orientation", "white_pick"]) expect(copied.patch).not.toHaveProperty(key);
    // Auto stays Auto.
    expect(copied.patch.white_balance).toEqual({ kind: "auto" });
    expect(copied.patch.sharpen).toEqual({ kind: "auto" });
  });

  it("waits for the patches still on their way, so the copy has the last value", async () => {
    const { store } = await onPhotos();
    const sending = store.update({ contrast: manual(33) });
    await store.copySettings();
    await sending;
    expect(store.getState().copied?.patch.contrast).toEqual(manual(33));
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
    expect(rs.settings).toMatchObject({ contrast: manual(20), look: "builtin:signature", look_intensity: 70, exposure: { kind: "manual", ev: 0.8 }, orientation: 0 });
    expect(rs.settings.crop?.angle_deg).toBe(-1);
    expect(rs.values.exposure_ev).toBe(0.8);

    // One Undo takes the whole paste back and leaves the crop alone; Redo does it again.
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    await vi.waitFor(() => expect(settingsOf(store).settings.contrast).toEqual({ kind: "auto" }));
    expect(settingsOf(store).settings).toMatchObject({ exposure: { kind: "auto" }, look_intensity: 100 });
    expect(settingsOf(store).settings.crop?.angle_deg).toBe(-1);
    await store.redo();
    await vi.waitFor(() => expect(settingsOf(store).settings.contrast).toEqual(manual(20)));
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
    for (const id of [3, 5, 7]) expect((await api.getSettings(id)).settings.contrast).toEqual(manual(20));
    expect((await api.getSettings(1)).settings.contrast).toEqual(manual(20));
    expect((await api.getSettings(2)).settings.contrast).toEqual({ kind: "auto" });
    // The photos that were not shown have no settings in the mirror until they are.
    expect(store.getState().settings[5]).toBeUndefined();
    expect(settingsOf(store).settings.contrast).toEqual(manual(20));

    // Each photo undoes its own paste.
    for (const id of [5, 7, 3]) {
      await store.setCurrent(id);
      await vi.waitFor(() => expect(store.currentSettings()?.photo_id).toBe(id));
      expect(store.getState().canUndo, `photo ${id}`).toBe(true);
      await store.undo();
      await vi.waitFor(() => expect(settingsOf(store).settings.contrast).toEqual({ kind: "auto" }));
    }
    expect((await api.getSettings(5)).settings.contrast).toEqual({ kind: "auto" });
    expect((await api.getSettings(1)).settings.contrast).toEqual(manual(20));
  });

  it("counts the photos it had to skip", async () => {
    const { store } = await onPhotos({ picked: [1, 2], skipped: { DSC04203: "This file is not a raw file." } });
    await store.update({ contrast: manual(15) });
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
    await store.update({ contrast: manual(9) });
    await store.copySettings();
    await store.setCurrent(2);
    await vi.waitFor(() => expect(store.currentSettings()?.photo_id).toBe(2));
    api.failNext("pasteSettings", { scope: "looks", message: "That look isn't in the library any more. Import it again." });
    await store.pasteSettings("this");
    expect(store.getState().error?.message).toBe("That look isn't in the library any more. Import it again.");
    expect(store.getState().canUndo).toBe(false);
    expect(settingsOf(store).settings.contrast).toEqual({ kind: "auto" });
  });

  it("does not undo a paste on the photo it came from when there was nothing to change", async () => {
    const { store } = await onPhotos();
    await store.update({ contrast: manual(9) });
    await store.copySettings();
    const undoDepth = store.getState().canUndo;
    await store.pasteSettings("this");
    expect(store.getState().notice).toBe("Pasted the settings of DSC04201.");
    // The photo already had them: no new step (the one for the contrast edit is still there).
    await store.undo();
    await vi.waitFor(() => expect(settingsOf(store).settings.contrast).toEqual({ kind: "auto" }));
    expect(undoDepth).toBe(true);
  });
});

describe("Paste settings, the edge cases", () => {
  /** A white picked on the current photo with the white balance back on Auto: its white is a picked one. */
  async function pickWhiteOnCurrent(store: AppStore): Promise<void> {
    await store.update({ white_pick: [0.5, 0.5] });
    await store.update({ white_balance: { kind: "auto" } });
    const s = settingsOf(store).settings;
    expect(s.white_pick).not.toBeNull();
    expect(s.white_balance).toEqual({ kind: "auto" });
    expect(s.light_source).toBe("auto");
  }

  it("leaves the photo a picked white came from alone when pasting to picked, and gives the others the manual white", async () => {
    const { api, store } = await onPhotos();
    await pickWhiteOnCurrent(store);
    await store.copySettings();
    const manual = store.getState().copied!.patch.white_balance;
    expect(manual?.kind).toBe("manual");
    await store.pickAll();
    const spy = vi.spyOn(api, "pasteSettings");
    await store.pasteSettings("picked");

    // The source is not in the call, and the notice counts the others only.
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0]![0]).toEqual([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(store.getState().notice).toBe("Pasted the settings of DSC04201 to 11 photos.");
    const source = await api.getSettings(1);
    expect(source.settings.white_balance).toEqual({ kind: "auto" });
    expect(source.settings.white_pick).not.toBeNull();
    for (const id of [2, 7, 12]) expect((await api.getSettings(id)).settings.white_balance).toEqual(manual);
    // No new step on the source: one Undo takes back the last edit it had (the white balance going back to Auto), not a paste.
    await store.undo();
    await vi.waitFor(() => expect(settingsOf(store).settings.white_balance.kind).toBe("manual"));
  });

  it("says so when the source is the only picked photo, and sends nothing", async () => {
    const { api, store } = await onPhotos();
    await pickWhiteOnCurrent(store);
    await store.copySettings();
    await store.setPicked(1, true);
    const spy = vi.spyOn(api, "pasteSettings");
    await store.pasteSettings("picked");
    expect(spy).not.toHaveBeenCalled();
    expect(store.getState().notice).toBe("No other photos are picked.");
    expect(settingsOf(store).settings.white_balance).toEqual({ kind: "auto" });
  });

  it("still pastes onto the source when its white was not a picked one", async () => {
    const { api, store } = await onPhotos();
    await store.update({ contrast: manual(15) });
    await store.copySettings();
    await store.update({ contrast: manual(40) });
    await store.setPicked(1, true);
    await store.setPicked(2, true);
    const spy = vi.spyOn(api, "pasteSettings");
    await store.pasteSettings("picked");
    // Pasting back onto the source restores it to what was copied.
    expect(spy.mock.calls[0]![0]).toEqual([1, 2]);
    expect(settingsOf(store).settings.contrast).toEqual(manual(15));
  });

  it("forgets which photo the settings came from when the session closes, so another session's photo with that id is not left out", async () => {
    const { store } = await onPhotos();
    await pickWhiteOnCurrent(store);
    await store.copySettings();
    expect(store.getState().copied?.sourceId).toBe(1);
    await store.closeSession();
    expect(store.getState().copied?.sourceId).toBeNull();
    expect(store.getState().copied?.source).toBe("DSC04201");
  });

  it("gives the photo on screen when the paste lands its new settings, even if the person moved there meanwhile", async () => {
    const { api, store } = await onPhotos();
    await editCurrent(store);
    await store.copySettings();
    await store.setCurrent(4);
    await vi.waitFor(() => expect(store.currentSettings()?.photo_id).toBe(4));
    for (const id of [3, 4, 5]) await store.setPicked(id, true);
    // Hold the paste in the core, as a slow one over many photos would be.
    const real = api.pasteSettings;
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    const spy = vi.spyOn(api, "pasteSettings").mockImplementation(async (ids: number[], patch: SettingsPatch): Promise<PasteReport> => {
      await gate;
      return real(ids, patch);
    });
    const pasting = store.pasteSettings("picked");
    await vi.waitFor(() => expect(spy).toHaveBeenCalled());
    await store.setCurrent(5);
    await vi.waitFor(() => expect(store.currentSettings()?.photo_id).toBe(5));
    expect(settingsOf(store).settings.contrast).toEqual({ kind: "auto" });
    release();
    await pasting;

    // The tool panel of photo 5 has its settings, and they are the pasted ones, not the ones read before the paste.
    await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
    expect(settingsOf(store).photo_id).toBe(5);
    expect(settingsOf(store).settings.contrast).toEqual(manual(20));
    expect(store.getState().notice).toBe("Pasted the settings of DSC04201 to 3 photos.");
  });

  it("keeps the paste a step of its own when a crop was open on the photo", async () => {
    const { store } = await onPhotos();
    await store.update({ contrast: manual(20), look_intensity: 60 });
    await store.copySettings();
    await store.setCurrent(2);
    await vi.waitFor(() => expect(store.currentSettings()?.photo_id).toBe(2));
    const crop = { rect: { x: 0.1, y: 0.1, width: 0.6, height: 0.6 }, angle_deg: 2, aspect: "free" as const };
    // The crop is being dragged (not final), the paste comes in, and Done ends the crop.
    await store.update({ crop, orientation: 90 }, false);
    await store.pasteSettings("this");
    expect(settingsOf(store).settings).toMatchObject({ contrast: manual(20), look_intensity: 60 });
    await store.update({ crop, orientation: 90 }, true);

    // Two steps: the crop and the paste. The first Undo takes back the crop and the turn only.
    await store.undo();
    await vi.waitFor(() => expect(settingsOf(store).settings.orientation).toBe(0));
    expect(settingsOf(store).settings).toMatchObject({ contrast: manual(20), look_intensity: 60, crop: null });
    // The second takes back the paste.
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    await vi.waitFor(() => expect(settingsOf(store).settings.contrast).toEqual({ kind: "auto" }));
    expect(settingsOf(store).settings.look_intensity).toBe(100);
    expect(store.getState().canUndo).toBe(false);
  });
});

describe("what is copied stays in memory", () => {
  it("is kept when the session closes, so it can go to another folder", async () => {
    const { store } = await onPhotos();
    await store.update({ contrast: manual(12) });
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
    await store.update({ contrast: manual(12) });
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

  it("are all off while a dialog is open, and back when it closes", async () => {
    const { api, store } = await onPhotos();
    await store.copySettings();
    await store.setPicked(4, true);
    const on = { copy_settings: true, paste_settings: true, paste_settings_to_picked: true };
    const off = { copy_settings: false, paste_settings: false, paste_settings_to_picked: false };
    await vi.waitFor(() => expect(api.menuState).toEqual(on));
    // The Photos keys are switched off under a dialog, and the menu must not do what they would not.
    for (const overlay of ["preferences", "about", "confirm-close", "import-cube"] as const) {
      store.openOverlay(overlay);
      await vi.waitFor(() => expect(api.menuState).toEqual(off));
      store.closeOverlay();
      await vi.waitFor(() => expect(api.menuState).toEqual(on));
    }
  });

  it("is told only when it changes", async () => {
    const { api, store } = await onPhotos();
    const spy = vi.spyOn(api, "setMenuState");
    await vi.waitFor(() => expect(api.menuState?.copy_settings).toBe(true));
    spy.mockClear();
    await store.update({ contrast: manual(5) });
    await store.setZoom({ kind: "percent", value: 100 });
    await store.toggleBefore();
    expect(spy).not.toHaveBeenCalled();
  });
});
