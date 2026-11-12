/**
 * The shell keeps the filmstrip's thumbnails under a byte budget (`docs/26` H-11, P3-2). A photo whose thumbnail it let go of gets a
 * new one when the filmstrip draws the photo again: the store tells the shell which photos are in view, drops the address that would
 * answer 404, and takes the new one from `thumbnail_ready`.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import { createStore, THUMBNAILS_IN_VIEW_DELAY_MS } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
  localStorage.clear();
  sessionStorage.clear();
});

async function developed(budget: number) {
  const api = createMockApi({ speed: Infinity, thumbnailBudget: budget });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  await vi.waitFor(() => expect(Object.keys(store.getState().thumbnails).length).toBeGreaterThan(0), { timeout: 5000 });
  return { api, store };
}

describe("thumbnailsInView", () => {
  it("brings back a thumbnail the shell let go of when its photo comes into view", async () => {
    const { api, store } = await developed(3);
    const ids = store.getState().session!.photos.map((p) => p.id);
    // Let every photo's thumbnail arrive; the shell's budget of 3 keeps three of the twelve.
    await vi.waitFor(async () => expect((await api.thumbnails()).length).toBe(3), { timeout: 5000 });
    const held = new Set((await api.thumbnails()).map((t) => t.id));
    const lost = ids.find((id) => !held.has(id))!;
    expect(lost).toBeDefined();

    const stale = store.getState().thumbnails[lost];
    expect(stale).toBeDefined();
    store.thumbnailsInView([lost]);
    store.thumbnailsInView([lost]);
    // The ask waits for a pause; then the address that would answer 404 goes and a new one comes.
    await vi.waitFor(() => expect(store.getState().thumbnails[lost]).not.toBe(stale), { timeout: 5000 });
    await vi.waitFor(() => expect(store.getState().thumbnails[lost]).toBeDefined(), { timeout: 5000 });
    expect(THUMBNAILS_IN_VIEW_DELAY_MS).toBeGreaterThan(0);
    expect((await api.thumbnails()).some((t) => t.id === lost)).toBe(true);
    expect((await api.thumbnails()).length).toBe(3);
  });

  it("leaves a photo's address alone when the shell still holds it", async () => {
    const { api, store } = await developed(0);
    const id = store.getState().session!.photos[0]!.id;
    const before = store.getState().thumbnails[id];
    const asked = vi.spyOn(api, "thumbnailsInView");
    store.thumbnailsInView([id]);
    await vi.waitFor(() => expect(asked).toHaveBeenCalledTimes(1), { timeout: 2000 });
    await Promise.resolve();
    expect(store.getState().thumbnails[id]).toBe(before);
  });
});
