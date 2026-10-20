/**
 * Remove from filmstrip and its Undo (the owner's request "I also want to remove photos from the film strip", `docs/11` §Removing
 * photos): the store against the mock, which plays the core's rules (the current photo hands over to the next one, the last removal is
 * kept until the next one or the end of the session).
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import { photoName, removedNotice, restoredNotice } from "./photo";
import { createStore } from "./store";
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

const photos = (store: AppStore) => store.getState().session?.photos ?? [];
const ids = (store: AppStore): number[] => photos(store).map((p) => p.id);
const names = (store: AppStore): string[] => photos(store).map((p) => photoName(p.file_name));
const idOf = (store: AppStore): number => store.currentPhoto()?.id ?? -1;
const nameOf = (store: AppStore): string => photoName(store.currentPhoto()?.file_name ?? "");

describe("the notices", () => {
  it("say what was removed, that the files are still there, and what came back", () => {
    expect(removedNotice(["DSC04204"])).toBe("Removed DSC04204. The file is still on disk.");
    expect(removedNotice(["a", "b", "c"])).toBe("Removed 3 photos. The files are still on disk.");
    expect(restoredNotice(["DSC04204"], 0)).toBe("Put back DSC04204.");
    expect(restoredNotice(["a", "b", "c"], 0)).toBe("Put back 3 photos.");
    expect(restoredNotice(["a", "b"], 1)).toBe("Put back 2 photos. 1 was already open.");
    expect(restoredNotice(["a"], 2)).toBe("Put back a. 2 were already open.");
    for (const text of [removedNotice(["a"]), removedNotice(["a", "b"]), restoredNotice(["a"], 1)]) expect(text).not.toContain("!");
  });
});

describe("removing photos", () => {
  it("takes the photo on screen out and shows the next one, and says the file is still on disk", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    await store.setCurrent(ids(store)[2] as number);
    await store.removeCurrentPhoto();
    expect(names(store)).toEqual(all.filter((_, i) => i !== 2));
    // The photo that followed the removed one is on screen, with its place moved up.
    expect(nameOf(store)).toBe(all[3]);
    expect(store.getState().session?.current_index).toBe(2);
    const state = store.getState();
    expect(state.notice).toBe(`Removed ${all[2]}. The file is still on disk.`);
    expect(state.noticeAction).toBe("undo-removal");
    expect(state.error).toBeNull();
    // The settings of the new photo load (it is the one on screen).
    await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
    expect(store.currentSettings()?.photo_id).toBe(idOf(store));
  });

  it("shows the previous photo when the last one was removed", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    await store.setCurrent(ids(store)[11] as number);
    await store.removeCurrentPhoto();
    expect(names(store)).toEqual(all.slice(0, 11));
    expect(nameOf(store)).toBe(all[10]);
    expect(store.getState().session?.current_index).toBe(10);
  });

  it("keeps the photo on screen when another one is removed, whatever its place becomes", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    await store.setCurrent(ids(store)[5] as number);
    await store.removePhotos([ids(store)[1] as number, ids(store)[2] as number]);
    expect(nameOf(store)).toBe(all[5]);
    expect(store.getState().session?.current_index).toBe(3);
    expect(store.getState().notice).toBe("Removed 2 photos. The files are still on disk.");
    // What depends on the photo on screen was not reset: the zoom is the person's.
    await store.setZoom({ kind: "percent", value: 100 });
    await store.removePhotos([ids(store)[0] as number]);
    expect(store.getState().zoom).toEqual({ kind: "percent", value: 100 });
  });

  it("resets what belongs to the photo on screen when that photo left", async () => {
    const { store } = await onPhotos();
    await store.setZoom({ kind: "percent", value: 100 });
    await store.setBefore(true);
    await store.removeCurrentPhoto();
    const state = store.getState();
    expect(state.zoom).toEqual({ kind: "fit" });
    expect(state.before).toBe(false);
    expect(state.histogram).toBeNull();
  });

  it("removes the picked photos, and the picks go with them", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    const list = ids(store);
    for (const i of [1, 4, 6]) await store.setPicked(list[i] as number, true);
    await store.setPicked(list[8] as number, true);
    await store.removePickedPhotos();
    expect(names(store)).toEqual(all.filter((_, i) => ![1, 4, 6, 8].includes(i)));
    expect(store.getState().session?.picked).toEqual([]);
    expect(store.getState().notice).toBe("Removed 4 photos. The files are still on disk.");
    // Nothing picked: nothing to do, no new notice.
    store.dismissNotice();
    await store.removePickedPhotos();
    expect(store.getState().notice).toBeNull();
    expect(names(store)).toHaveLength(8);
  });

  it("ignores ids that are not in the session and an empty list", async () => {
    const { store, api } = await onPhotos();
    const spy = vi.spyOn(api, "removePhotos");
    await store.removePhotos([]);
    await store.removePhotos([9999]);
    expect(spy).not.toHaveBeenCalled();
    expect(names(store)).toHaveLength(12);
    expect(store.getState().notice).toBeNull();
    // An id twice counts once.
    const first = ids(store)[0] as number;
    await store.removePhotos([first, first]);
    expect(spy).toHaveBeenCalledWith([first]);
    expect(names(store)).toHaveLength(11);
  });

  it("drops what the page knew about the photos that left", async () => {
    const { store } = await onPhotos();
    const gone = ids(store)[0] as number;
    await vi.waitFor(() => expect(store.getState().thumbnails[gone]).toBeTruthy());
    expect(store.getState().settings[gone]).toBeDefined();
    await store.removePhotos([gone]);
    expect(store.getState().thumbnails[gone]).toBeUndefined();
    expect(store.getState().settings[gone]).toBeUndefined();
    // A late event about a removed photo does not bring it back into the session.
    expect(ids(store)).not.toContain(gone);
  });

  it("shows the Start screen when the last photo is removed, with the notice that offers Undo", async () => {
    const { store, api } = await onPhotos();
    const viewportReset = vi.spyOn(api, "viewportSetMode");
    await store.removePhotos(ids(store));
    const state = store.getState();
    expect(state.session).toBeNull();
    expect(state.screen).toBe("start");
    expect(state.thumbnails).toEqual({});
    expect(state.notice).toBe("Removed 12 photos. The files are still on disk.");
    expect(state.noticeAction).toBe("undo-removal");
    expect(store.hasRemovalToUndo()).toBe(true);
    expect(viewportReset).toHaveBeenCalledWith({ kind: "fit" });
    // The core's session is empty too (it was not closed, so Undo still has the photos).
    expect((await api.sessionState()).photos).toEqual([]);
  });

  it("waits for the patches on their way to a photo before it takes the photo out, and shows no error", async () => {
    const { store } = await onPhotos();
    const sent = store.update({ contrast: { kind: "manual", value: 25 } }, false);
    const gone = store.removeCurrentPhoto();
    await Promise.all([sent, gone]);
    expect(store.getState().error).toBeNull();
    expect(names(store)).toHaveLength(11);
  });

  it("shows an error and changes nothing when the core refuses", async () => {
    const { store, api } = await onPhotos();
    api.failNext("removePhotos", { scope: "app", message: "That photo is no longer open. Open it again." });
    await store.removeCurrentPhoto();
    expect(names(store)).toHaveLength(12);
    expect(store.getState().error?.message).toBe("That photo is no longer open. Open it again.");
    expect(store.getState().noticeAction).toBeNull();
    expect(store.hasRemovalToUndo()).toBe(false);
  });

  it("takes a photo that is still developing out, and development goes on for the others", async () => {
    const api = createMockApi({ speed: 4, holdAt: 5 });
    const store = createStore(api, 0);
    stops.push(store.connect());
    await store.load();
    await store.openDialog("files");
    await vi.waitFor(() => expect(photos(store).filter((p) => p.status.kind === "ready").length).toBeGreaterThanOrEqual(2));
    const waiting = photos(store).find((p) => p.status.kind === "waiting");
    expect(waiting).toBeDefined();
    await store.removePhotos([waiting?.id as number]);
    expect(ids(store)).not.toContain(waiting?.id);
    expect(photos(store)).toHaveLength(11);
  });
});

describe("Undo", () => {
  it("puts the photos back where they were, the first of them on screen again", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    await store.setCurrent(ids(store)[3] as number);
    await store.removePhotos([ids(store)[3] as number, ids(store)[7] as number]);
    expect(names(store)).toHaveLength(10);
    await store.undoRemoval();
    expect(names(store)).toEqual(all);
    expect(nameOf(store)).toBe(all[3]);
    const state = store.getState();
    expect(state.notice).toBe(`Put back 2 photos.`);
    expect(state.noticeAction).toBeNull();
    expect(store.hasRemovalToUndo()).toBe(false);
    await vi.waitFor(() => expect(store.currentSettings()?.photo_id).toBe(idOf(store)));
    // Their thumbnails come back (the shell announces them again).
    await vi.waitFor(() => expect(store.getState().thumbnails[ids(store)[7] as number]).toBeTruthy());
  });

  it("keeps the photo the person went on to, and the picks come back", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    const list = ids(store);
    await store.setPicked(list[2] as number, true);
    await store.setPicked(list[9] as number, true);
    await store.setCurrent(list[2] as number);
    await store.removePhotos([list[2] as number]);
    expect(store.getState().session?.picked).toEqual([list[9]]);
    await store.setCurrent(list[10] as number);
    await store.undoRemoval();
    expect(names(store)).toEqual(all);
    expect(nameOf(store)).toBe(all[10]);
    expect(store.getState().session?.picked).toEqual([list[2], list[9]]);
    expect(store.getState().session?.current_index).toBe(10);
  });

  it("brings the settings back exactly, and the undo steps with them", async () => {
    const { store } = await onPhotos();
    const edited = idOf(store);
    await store.update({ exposure: { kind: "manual", ev: 0.8 }, contrast: { kind: "manual", value: 20 } }, true);
    const before = store.currentSettings();
    await store.removeCurrentPhoto();
    expect(store.getState().settings[edited]).toBeUndefined();
    await store.undoRemoval();
    expect(idOf(store)).toBe(edited);
    await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
    expect(store.currentSettings()?.settings).toEqual(before?.settings);
    expect(store.currentSettings()?.values.exposure_ev).toBe(0.8);
    // The change made before the photo left can still be undone (⌘Z on the photo).
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    expect(store.currentSettings()?.settings.contrast).toEqual({ kind: "auto" });
    expect(store.currentSettings()?.auto.exposure).toBe(true);
  });

  it("goes back to Photos from the Start screen after the last photo was removed", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    await store.setCurrent(ids(store)[4] as number);
    await store.removePhotos(ids(store));
    expect(store.getState().screen).toBe("start");
    await store.undoRemoval();
    const state = store.getState();
    expect(state.screen).toBe("photos");
    expect(names(store)).toEqual(all);
    expect(nameOf(store)).toBe(all[4]);
    expect(state.notice).toBe("Put back 12 photos.");
    await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
    await vi.waitFor(() => expect(Object.keys(store.getState().thumbnails)).toHaveLength(12));
  });

  it("is for the last removal only", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    await store.removePhotos([ids(store)[0] as number]);
    await store.removePhotos([ids(store)[0] as number]);
    expect(names(store)).toHaveLength(10);
    await store.undoRemoval();
    expect(names(store)).toEqual(all.filter((_, i) => i !== 0));
    // Nothing more to take back.
    expect(store.hasRemovalToUndo()).toBe(false);
    await store.undoRemoval();
    expect(names(store)).toHaveLength(11);
  });

  it("says so when the core has nothing to put back", async () => {
    const { store, api } = await onPhotos();
    await store.removeCurrentPhoto();
    // The core's buffer is used up behind the page's back (a second window, a reload).
    await api.restorePhotos();
    await store.undoRemoval();
    expect(store.getState().notice).toBe("There is nothing to put back.");
    expect(store.getState().noticeAction).toBeNull();
    expect(store.hasRemovalToUndo()).toBe(false);
  });

  it("is forgotten, with its notice, when the session is closed", async () => {
    const { store } = await onPhotos();
    await store.removeCurrentPhoto();
    expect(store.getState().noticeAction).toBe("undo-removal");
    await store.closeSession();
    expect(store.hasRemovalToUndo()).toBe(false);
    expect(store.getState().notice).toBeNull();
    expect(store.getState().noticeAction).toBeNull();
    await store.undoRemoval();
    expect(store.getState().session).toBeNull();
  });

  it("is forgotten, with its notice, when files are opened on the Start screen: that is a new session", async () => {
    const { store, api } = await onPhotos();
    await store.removePhotos(ids(store));
    expect(store.getState().screen).toBe("start");
    expect(store.hasRemovalToUndo()).toBe(true);
    await store.openDialog("files");
    api.flush();
    expect(store.hasRemovalToUndo()).toBe(false);
    expect(store.getState().notice).toBeNull();
    expect(store.getState().noticeAction).toBeNull();
    await vi.waitFor(() => expect(photos(store)).toHaveLength(12));
    await store.undoRemoval();
    expect(photos(store)).toHaveLength(12);
  });

  it("is replaced by any other notice, and still works with ⌘Z", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    await store.removeCurrentPhoto();
    store.showNotice("Yellowmoon is up to date.");
    expect(store.getState().noticeAction).toBeNull();
    expect(store.hasRemovalToUndo()).toBe(true);
    await store.undo();
    expect(names(store)).toEqual(all);
  });
});

describe("⌘Z after a removal", () => {
  it("takes the removal back while nothing was changed since, and the photo's edits after that", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    await store.update({ contrast: { kind: "manual", value: 30 } }, true);
    expect(store.getState().canUndo).toBe(true);
    await store.setCurrent(ids(store)[1] as number);
    await store.removePhotos([ids(store)[1] as number]);
    // The photo that is current now has no steps of its own, but the removal can be undone.
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    expect(names(store)).toEqual(all);
    // The next ⌘Z is for the settings of the photo on screen (the second photo has none).
    expect(store.hasRemovalToUndo()).toBe(false);
    expect(store.getState().canUndo).toBe(false);
    await store.setCurrent(ids(store)[0] as number);
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    await vi.waitFor(() => expect(store.currentSettings()?.settings.contrast).toEqual({ kind: "auto" }));
  });

  it("undoes a change made after the removal first, then the removal", async () => {
    const { store } = await onPhotos();
    const all = names(store);
    const list = ids(store);
    await store.removePhotos([list[0] as number]);
    // A change on the photo now on screen, once its settings have arrived.
    await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
    await store.update({ contrast: { kind: "manual", value: 40 } }, true);
    expect(store.currentSettings()?.settings.contrast).toEqual({ kind: "manual", value: 40 });
    await store.undo();
    expect(store.currentSettings()?.settings.contrast).toEqual({ kind: "auto" });
    expect(names(store)).toHaveLength(11);
    expect(store.hasRemovalToUndo()).toBe(true);
    // Nothing is left to undo but the removal.
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    expect(names(store)).toEqual(all);
  });
});

describe("the core's news about a photo that is being removed (Apple Auto's answers arrive all the time)", () => {
  /** Holds every read of one photo's settings until the test lets it go; the others go through. */
  function holdReadsOf(api: MockApi, photo: number): { held: () => number; resolve: (rs: unknown) => void; reject: (error: unknown) => void } {
    const real = api.getSettings.bind(api);
    const waiting: Array<{ resolve: (rs: unknown) => void; reject: (error: unknown) => void }> = [];
    vi.spyOn(api, "getSettings").mockImplementation((id) =>
      id === photo ? new Promise((resolve, reject) => waiting.push({ resolve: resolve as (rs: unknown) => void, reject })) : real(id),
    );
    return {
      held: () => waiting.length,
      resolve: (rs) => waiting.splice(0).forEach((w) => w.resolve(rs)),
      reject: (error) => waiting.splice(0).forEach((w) => w.reject(error)),
    };
  }

  it("a read of the settings that was on its way when the person removed the photo fails quietly", async () => {
    const { api, store } = await onPhotos({ appleAuto: "answered" });
    const victim = idOf(store);
    const read = holdReadsOf(api, victim);
    api.emitEvent("settings_changed", { photo_id: victim });
    await vi.waitFor(() => expect(read.held()).toBe(1));

    const removal = store.removePhotos([victim]);
    // The core has removed the photo by the time it hears the question.
    read.reject({ scope: "app", message: "That photo isn't open." });
    await removal;
    await vi.waitFor(() => expect(ids(store)).not.toContain(victim));
    expect(store.getState().error).toBeNull();
    expect(store.getState().settings[victim]).toBeUndefined();
  });

  it("a read that answers after the photo left is not put back into the mirror", async () => {
    const { api, store } = await onPhotos({ appleAuto: "answered" });
    const victim = idOf(store);
    const rs = await api.getSettings(victim);
    const read = holdReadsOf(api, victim);
    api.emitEvent("settings_changed", { photo_id: victim });
    await vi.waitFor(() => expect(read.held()).toBe(1));

    await store.removePhotos([victim]);
    read.resolve(rs);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(store.getState().settings[victim]).toBeUndefined();
    expect(store.getState().error).toBeNull();
  });

  it("news that arrives while the removal is on its way is not read at all, and a removal the core refused reads again", async () => {
    const { api, store } = await onPhotos({ appleAuto: "answered" });
    const victim = idOf(store);
    const spy = vi.spyOn(api, "getSettings");
    // The core refuses the removal.
    let refuse: (error: unknown) => void = () => {};
    vi.spyOn(api, "removePhotos").mockImplementation(() => new Promise((_, reject) => (refuse = reject)));
    const removal = store.removePhotos([victim]);
    await vi.waitFor(() => expect(api.removePhotos).toHaveBeenCalled());
    api.emitEvent("settings_changed", { photo_id: victim });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(spy).not.toHaveBeenCalledWith(victim);
    refuse({ scope: "app", message: "Yellowmoon could not remove that photo." });
    await removal;
    // The photo is still here, and the news it missed is read now.
    expect(ids(store)).toContain(victim);
    await vi.waitFor(() => expect(spy).toHaveBeenCalledWith(victim));
  });
});
