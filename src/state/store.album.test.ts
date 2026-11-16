/**
 * Albums in the store (`docs/31` sections 4 to 6): opening one replaces the session, the edited state follows every change and undo does not
 * clear it, Save writes in place and Save As asks, a missing photo is a failed photo, and the question "Save changes to the album …?".
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import { createStore } from "./store";
import type { AppStore } from "./store";
import { mainGradePatch } from "./settingsGraph";
import { folderSubline, folderTitle } from "./photo";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
  localStorage.clear();
  sessionStorage.clear();
});

const LISBON = "~/Pictures/Lisbon/Lisbon.ymalbum";

async function started(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, albums: ["Lisbon"], ...options });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  return { api, store };
}

async function withAlbum(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
  const both = await started(options);
  await both.store.openPaths([LISBON]);
  both.api.flush();
  await vi.waitFor(() => expect(both.store.getState().session?.photos.every((p) => p.status.kind !== "waiting" && p.status.kind !== "developing")).toBe(true));
  await both.store.openPhotos();
  await vi.waitFor(() => expect(both.store.currentSettings()).not.toBeNull());
  return both;
}

describe("opening an album", () => {
  it("replaces the session with the album's photos, picks and current photo, and not edited", async () => {
    const { store } = await withAlbum();
    const session = store.getState().session!;
    expect(session.album).toEqual({ name: "Lisbon", path: LISBON, edited: false });
    expect(session.photos).toHaveLength(12);
    expect(session.current_index).toBe(3);
    expect(session.picked).toHaveLength(3);
    expect(store.getState().screen).toBe("photos");
    expect(folderTitle(session)).toBe("Lisbon");
  });

  it("is refused with a plain sentence together with photos, or with another album, and the session is not replaced", async () => {
    const { store } = await started({ albums: ["Lisbon", "Porto"] });
    await store.openPaths([LISBON, "/x/DSC04201.ARW"]);
    expect(store.getState().dropNotice).toBe("Open an album on its own, or open photos. Yellowmoon can't open both in one go.");
    expect(store.getState().session).toBeNull();
    await store.openPaths([LISBON, "~/Pictures/Porto/Porto.ymalbum"]);
    expect(store.getState().dropNotice).toBe("Open one album at a time.");
    expect(store.getState().session).toBeNull();
  });

  it("says in the notice bar on Photos, and leaves the session as it was, when the file is not an album", async () => {
    const { store } = await withAlbum();
    await store.openPaths(["~/Pictures/Gone/Gone.ymalbum"]);
    expect(store.getState().notice).toBe("Yellowmoon couldn't read this file. Check that it is still there, then try again.");
    expect(store.getState().session?.album?.name).toBe("Lisbon");
  });

  it("keeps a photo it cannot find as a failed photo that says where it was", async () => {
    const { store } = await withAlbum({ albumMissing: ["DSC04206"] });
    const gone = store.getState().session!.photos.find((p) => p.file_name === "DSC04206.ARW")!;
    expect(gone.status).toEqual({ kind: "skipped", reason: "Not found: ~/Pictures/Lisbon/DSC04206.ARW" });
    expect(store.getState().session!.photos).toHaveLength(12);
  });
});

describe("the edited state", () => {
  it("follows a setting, a pick, and the removal and return of a photo, and undo does not clear it", async () => {
    const { store } = await withAlbum();
    const edited = () => store.getState().session?.album?.edited;
    await vi.waitFor(() => expect(edited()).toBe(false));
    await store.update(mainGradePatch({ contrast: { kind: "manual", value: 5 } }));
    await vi.waitFor(() => expect(edited()).toBe(true));
    expect(folderSubline(store.getState().session!)).toBe("12 photos · edited");
    await store.undo();
    await Promise.resolve();
    expect(edited()).toBe(true);
  });

  it("a pick alone is a change, and so is a removal", async () => {
    const one = await withAlbum();
    await one.store.togglePick();
    await vi.waitFor(() => expect(one.store.getState().session?.album?.edited).toBe(true));
    stops.splice(0).forEach((stop) => stop());
    const two = await withAlbum();
    await two.store.removeCurrentPhoto();
    await vi.waitFor(() => expect(two.store.getState().session?.album?.edited).toBe(true));
  });

  it("has no state with no album: a session opened from photos is not edited and asks nothing on close", async () => {
    const { api, store } = await started();
    await store.openDialog("files");
    api.flush();
    await store.openPhotos();
    await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
    await store.update(mainGradePatch({ contrast: { kind: "manual", value: 5 } }));
    expect(store.getState().session?.album ?? null).toBeNull();
    await store.requestCloseSession();
    expect(store.getState().session).toBeNull();
    expect(store.getState().overlay).toBeNull();
  });
});

describe("saving", () => {
  it("Save writes in place with no question, clears the state and says it", async () => {
    const { api, store } = await withAlbum();
    await store.togglePick();
    await vi.waitFor(() => expect(store.getState().session?.album?.edited).toBe(true));
    const spy = vi.spyOn(api, "albumSave");
    expect(await store.saveAlbum(false)).toBe(true);
    expect(spy).toHaveBeenCalledWith(false);
    expect(store.getState().notice).toBe("Saved the album Lisbon.");
    await vi.waitFor(() => expect(store.getState().session?.album?.edited).toBe(false));
  });

  it("the first Save of a session opened from photos asks where, and the session then is an album", async () => {
    const { api, store } = await started();
    await store.openDialog("files");
    api.flush();
    await store.openPhotos();
    api.queueChoice("save", "~/Desktop/Trip.ymalbum");
    expect(await store.saveAlbum(false)).toBe(true);
    await vi.waitFor(() => expect(store.getState().session?.album).toEqual({ name: "Trip", path: "~/Desktop/Trip.ymalbum", edited: false }));
    expect(api.savedAlbums.get("~/Desktop/Trip.ymalbum")).toEqual({ photos: 12 });
  });

  it("Save As always asks, and a cancelled panel changes nothing and says nothing", async () => {
    const { api, store } = await withAlbum();
    api.queueChoice("save", null);
    store.dismissNotice();
    expect(await store.saveAlbum(true)).toBe(false);
    expect(store.getState().notice).toBeNull();
    api.queueChoice("save", "~/Desktop/Copy.ymalbum");
    expect(await store.saveAlbum(true)).toBe(true);
    await vi.waitFor(() => expect(store.getState().session?.album?.name).toBe("Copy"));
  });

  it("shows the error of the core when the write fails, and stays edited", async () => {
    const { api, store } = await withAlbum();
    await store.togglePick();
    await vi.waitFor(() => expect(store.getState().session?.album?.edited).toBe(true));
    api.failNext("albumSave", { scope: "album", message: "Yellowmoon couldn't save the album to ~/Pictures/Lisbon. Check that you can save there and that the disk has free space, then try again. Nothing was changed." });
    expect(await store.saveAlbum(false)).toBe(false);
    expect(store.getState().error?.scope).toBe("album");
    expect(store.getState().session?.album?.edited).toBe(true);
  });
});

describe("the question before an album is closed or replaced", () => {
  async function edited(): Promise<{ api: MockApi; store: AppStore }> {
    const both = await withAlbum();
    await both.store.togglePick();
    await vi.waitFor(() => expect(both.store.getState().session?.album?.edited).toBe(true));
    return both;
  }

  it("asks on Close Session, and Cancel stays", async () => {
    const { store } = await edited();
    await store.requestCloseSession();
    expect(store.getState().overlay).toBe("album-question");
    expect(store.getState().session).not.toBeNull();
    await store.answerAlbumQuestion("cancel");
    expect(store.getState().overlay).toBeNull();
    expect(store.getState().session).not.toBeNull();
  });

  it("Don't Save closes without writing", async () => {
    const { api, store } = await edited();
    const spy = vi.spyOn(api, "albumSave");
    await store.requestCloseSession();
    await store.answerAlbumQuestion("discard");
    expect(spy).not.toHaveBeenCalled();
    expect(store.getState().session).toBeNull();
  });

  it("Save writes and then closes; a Save that did not happen stays", async () => {
    const one = await edited();
    await one.store.requestCloseSession();
    await one.store.answerAlbumQuestion("save");
    expect(one.api.savedAlbums.size).toBeGreaterThan(0);
    expect(one.store.getState().session).toBeNull();
    stops.splice(0).forEach((stop) => stop());
    const two = await edited();
    two.api.failNext("albumSave", { scope: "album", message: "Yellowmoon couldn't save the album to ~/Pictures/Lisbon. Check that you can save there and that the disk has free space, then try again. Nothing was changed." });
    await two.store.requestCloseSession();
    await two.store.answerAlbumQuestion("save");
    expect(two.store.getState().session).not.toBeNull();
    expect(two.store.getState().error?.scope).toBe("album");
  });

  it("comes before the export question", async () => {
    const { store } = await edited();
    vi.spyOn(store, "exportRunning").mockReturnValue(true);
    await store.requestCloseSession();
    expect(store.getState().overlay).toBe("album-question");
    await store.answerAlbumQuestion("discard");
    expect(store.getState().overlay).toBe("confirm-close");
  });

  it("asks before another album replaces the edited one, and Don't Save opens it", async () => {
    const { api, store } = await edited();
    api.configure({ albums: ["Lisbon", "Porto"] });
    await store.openAlbum("~/Pictures/Lisbon/Lisbon.ymalbum");
    expect(store.getState().overlay).toBe("album-question");
    await store.answerAlbumQuestion("discard");
    expect(store.getState().session?.album?.edited).toBe(false);
    expect(store.getState().overlay).toBeNull();
  });
});
