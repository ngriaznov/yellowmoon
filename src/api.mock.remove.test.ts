/**
 * Remove from filmstrip on the mock API: it plays the core's rules (`Session::remove`, `Session::restore_removed`) so that the screens,
 * the store tests and the end-to-end suite can drive the feature without Tauri.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "./api.mock";
import type { MockApi } from "./api.mock";
import type { ApiEvents, PhotoId, SessionState } from "./api/types";

let api: MockApi;
let thumbnails: Array<ApiEvents["thumbnail_ready"]>;

beforeEach(async () => {
  api = createMockApi({ speed: Number.POSITIVE_INFINITY });
  thumbnails = [];
  api.on("thumbnail_ready", (e) => thumbnails.push(e));
  await api.openDialog("files");
  api.flush();
});
afterEach(() => {
  api.reset();
  vi.useRealTimers();
});

const ids = (s: SessionState): PhotoId[] => s.photos.map((p) => p.id);
const currentId = (s: SessionState): PhotoId | undefined => (s.current_index === null ? undefined : s.photos[s.current_index]?.id);

describe("removePhotos", () => {
  it("takes the photos out, in filmstrip order, and hands the current photo over to the next one", async () => {
    await api.setCurrent(4);
    const change = await api.removePhotos([6, 4, 99, 4]);
    expect(change.photo_ids).toEqual([4, 6]);
    expect(change.already_open).toBe(0);
    expect(ids(change.session)).toEqual([1, 2, 3, 5, 7, 8, 9, 10, 11, 12]);
    expect(currentId(change.session)).toBe(5);
    expect(await api.sessionState()).toEqual(change.session);
    await expect(api.getSettings(4)).rejects.toMatchObject({ scope: "app" });
  });

  it("hands over to the previous photo when the last was removed, and to none when none is left", async () => {
    await api.setCurrent(12);
    expect(currentId((await api.removePhotos([12])).session)).toBe(11);
    const everything = await api.removePhotos(ids(await api.sessionState()));
    expect(everything.session.photos).toEqual([]);
    expect(everything.session.current_index).toBeNull();
    expect(everything.session.folder_name).toBeNull();
  });

  it("keeps the current photo when another one is removed, and takes the picks away with the photos", async () => {
    await api.setCurrent(8);
    await api.setPicked(2, true);
    await api.setPicked(9, true);
    const change = await api.removePhotos([2, 3]);
    expect(currentId(change.session)).toBe(8);
    expect(change.session.picked).toEqual([9]);
  });

  it("changes nothing for ids that are not in the session, and keeps the removal before it for Undo", async () => {
    await api.removePhotos([1]);
    const nothing = await api.removePhotos([500]);
    expect(nothing.photo_ids).toEqual([]);
    expect((await api.restorePhotos()).photo_ids).toEqual([1]);
  });

  it("does not use an id twice: a photo added later gets a new one", async () => {
    await api.removePhotos([12]);
    const added = await api.openPaths(["~/Pictures/Lisbon/Extra01.ARW"]);
    expect(added.opened).toEqual({ added: 1, already_open: 0 });
    expect(added.photos.map((p) => p.id)).toContain(13);
    expect(ids(added)).not.toContain(12);
  });
});

describe("restorePhotos", () => {
  it("puts the photos back where they were, with their settings and picks", async () => {
    await api.updateSettings(3, { contrast: { kind: "manual", value: 22 } });
    await api.setPicked(3, true);
    await api.setCurrent(3);
    await api.removePhotos([3, 7]);
    const back = await api.restorePhotos();
    expect(back.photo_ids).toEqual([3, 7]);
    expect(ids(back.session)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(currentId(back.session)).toBe(3);
    expect(back.session.picked).toEqual([3]);
    expect((await api.getSettings(3)).settings.contrast).toEqual({ kind: "manual", value: 22 });
  });

  it("announces the thumbnails of the photos again", async () => {
    await api.removePhotos([2]);
    thumbnails.length = 0;
    await api.restorePhotos();
    expect(thumbnails.map((t) => t.id)).toEqual([2]);
    expect(await api.thumbnails()).toEqual(expect.arrayContaining([{ id: 2, url: thumbnails[0]?.url }]));
  });

  it("leaves the photo on screen when the person has moved on", async () => {
    await api.setCurrent(5);
    await api.removePhotos([5]);
    await api.setCurrent(9);
    expect(currentId((await api.restorePhotos()).session)).toBe(9);
    await api.setCurrent(5);
    await api.removePhotos([5]);
    expect(currentId((await api.restorePhotos()).session)).toBe(5);
    // A removal that did not take the photo on screen leaves it on screen.
    await api.removePhotos([2]);
    expect(currentId((await api.restorePhotos()).session)).toBe(5);
  });

  it("is for the last removal, once, and the end of the session forgets it", async () => {
    await api.removePhotos([1]);
    await api.removePhotos([2]);
    expect((await api.restorePhotos()).photo_ids).toEqual([2]);
    expect((await api.restorePhotos()).photo_ids).toEqual([]);
    await api.removePhotos([3]);
    await api.closeSession();
    expect((await api.restorePhotos()).photo_ids).toEqual([]);
  });

  it("leaves a photo out when its file was added again meanwhile", async () => {
    await api.removePhotos([4]);
    const added = await api.openPaths(["~/Pictures/Lisbon/DSC04204.ARW"]);
    expect(added.opened).toEqual({ added: 1, already_open: 0 });
    const back = await api.restorePhotos();
    expect(back.photo_ids).toEqual([]);
    expect(back.already_open).toBe(1);
    expect(back.session.photos.filter((p) => p.file_name === "DSC04204.ARW")).toHaveLength(1);
  });

  it("brings every photo back from an empty session, with the first on screen as it was", async () => {
    await api.setCurrent(6);
    await api.removePhotos(ids(await api.sessionState()));
    const back = await api.restorePhotos();
    expect(ids(back.session)).toHaveLength(12);
    expect(currentId(back.session)).toBe(6);
    expect(back.session.folder_name).toBe("Lisbon");
  });
});

describe("a photo that is still developing", () => {
  it("starts again from the beginning when it is put back", async () => {
    const slow = createMockApi({ speed: 8, holdAt: 3 });
    await slow.openDialog("files");
    // Held at three photos: the others wait.
    await vi.waitFor(async () => expect((await slow.sessionState()).photos.filter((p) => p.status.kind === "ready").length).toBe(3));
    const before = await slow.sessionState();
    const waiting = before.photos.find((p) => p.status.kind === "waiting");
    expect(waiting).toBeDefined();
    await slow.removePhotos([waiting?.id as number]);
    const back = await slow.restorePhotos();
    expect(back.photo_ids).toEqual([waiting?.id]);
    expect(back.session.photos.find((p) => p.id === waiting?.id)?.status.kind).toBe("waiting");
    slow.reset();
  });
});
