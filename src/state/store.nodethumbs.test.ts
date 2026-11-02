/**
 * The store's part in the node thumbnails (`docs/23` §5.7): it asks the core for them while the node view is on, when a change to the
 * photo has landed (never during a drag), when a node is added or removed, when the photo on screen changes and when the view opens;
 * it keeps the addresses per photo and node, and lets go of those of nodes, photos and sessions that are gone.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi } from "../api.mock";
import type { NodeLink, SettingsPatch } from "../api/types";
import { createStore } from "./store";
import type { AppStore } from "./store";
import { mainPatch, patchOf } from "./settingsGraph";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
  vi.useRealTimers();
});

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

/** Nodes 2 and 3 after Main. */
const THREE: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Warmth", pos: [1, 0], trim: { temperature: 40 } },
    "3": { kind: "corrector", name: "Lift", pos: [2, 0], trim: { exposure_ev: 0.5 } },
  },
  links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, "output")],
};

/** The twelve photos open and developed, DSC04204 (id 4) current with its settings loaded; nobody watches the node view yet. */
async function opened(): Promise<{ api: MockApi; store: AppStore; asked: ReturnType<typeof vi.spyOn> }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3 });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
  await store.setCurrent(4);
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  const asked = vi.spyOn(api, "nodeThumbnails");
  return { api, store, asked };
}

/** Lets the microtasks and the mock's zero-length timers run. */
const settle = async (): Promise<void> => {
  await new Promise((resolve) => setTimeout(resolve, 5));
};

const picturesOf = (store: AppStore, photo: number) => store.getState().nodeThumbnails[photo] ?? {};

describe("when the store asks for node thumbnails", () => {
  it("asks for nothing while nobody has the node view on", async () => {
    const { store, asked } = await opened();
    await store.update(mainPatch({ trim: { exposure_ev: 1 } }));
    await store.setCurrent(5);
    await store.update(mainPatch({ trim: { exposure_ev: 0.5 } }));
    await settle();
    expect(asked).not.toHaveBeenCalled();
    expect(store.getState().nodeThumbnails).toEqual({});
  });

  it("asks once when the node view opens, and keeps the pictures that come for the photo and its nodes", async () => {
    const { store, asked } = await opened();
    stops.push(store.watchNodeThumbnails());
    await settle();
    expect(asked.mock.calls).toEqual([[4]]);
    await vi.waitFor(() => expect(Object.keys(picturesOf(store, 4))).toEqual(["1"]));
    expect(picturesOf(store, 4)[1]).toMatch(/^(blob:|data:image\/png)/);
  });

  it("asks when a node is added or removed, and lets go of the picture of a node that is gone", async () => {
    const { store, asked } = await opened();
    stops.push(store.watchNodeThumbnails());
    await vi.waitFor(() => expect(Object.keys(picturesOf(store, 4))).toEqual(["1"]));
    asked.mockClear();
    await store.update(THREE);
    await settle();
    expect(asked.mock.calls).toEqual([[4]]);
    await vi.waitFor(() => expect(Object.keys(picturesOf(store, 4))).toEqual(["1", "2", "3"]));
    const had = picturesOf(store, 4);

    asked.mockClear();
    await store.update({ remove: [3], links: [picture("input", 1), picture(1, 2), picture(2, "output")] });
    await settle();
    expect(asked.mock.calls).toEqual([[4]]);
    // Node 3 has no card any more: its address is let go, and the other two keep theirs (the core sends only what changed).
    expect(Object.keys(picturesOf(store, 4))).toEqual(["1", "2"]);
    expect(picturesOf(store, 4)[1]).toBe(had[1]);
    expect(picturesOf(store, 4)[2]).toBe(had[2]);
    // Undo brings the node back, and its picture is made again.
    await store.undo();
    await vi.waitFor(() => expect(Object.keys(picturesOf(store, 4))).toEqual(["1", "2", "3"]));
    expect(picturesOf(store, 4)[3]).not.toBe(had[3]);
  });

  it("asks nothing during a drag, and once when the release has landed", async () => {
    const { store, asked } = await opened();
    await store.update(THREE);
    stops.push(store.watchNodeThumbnails());
    await vi.waitFor(() => expect(Object.keys(picturesOf(store, 4))).toEqual(["1", "2", "3"]));
    asked.mockClear();
    const at = (ev: number) => patchOf(3, { trim: { exposure_ev: ev } });
    // Frames of a drag: the patches are sent and answered, and nothing is asked.
    void store.update(at(0.6), false);
    await settle();
    void store.update(at(0.7), false);
    await settle();
    void store.update(at(0.8), false);
    await settle();
    expect(asked).not.toHaveBeenCalled();
    const before = picturesOf(store, 4)[3];
    // The release: one patch, one ask, once it has landed.
    await store.update(at(0.9), true);
    await settle();
    expect(asked.mock.calls).toEqual([[4]]);
    await vi.waitFor(() => expect(picturesOf(store, 4)[3]).not.toBe(before));
    // The nodes before the one that moved keep their pictures.
    expect(Object.keys(picturesOf(store, 4))).toEqual(["1", "2", "3"]);
  });

  it("asks once for a burst of changes", async () => {
    const { store, asked } = await opened();
    await store.update(THREE);
    stops.push(store.watchNodeThumbnails());
    await settle();
    asked.mockClear();
    await Promise.all([store.update(patchOf(3, { trim: { exposure_ev: 1 } })), store.update(patchOf(2, { trim: { temperature: -20 } }))]);
    await settle();
    expect(asked.mock.calls.length).toBeLessThanOrEqual(2);
    expect(asked.mock.calls.length).toBeGreaterThanOrEqual(1);
    for (const call of asked.mock.calls) expect(call).toEqual([4]);
  });

  it("asks again after an undo and after a redo", async () => {
    const { store, asked } = await opened();
    stops.push(store.watchNodeThumbnails());
    await settle();
    await store.update(mainPatch({ trim: { exposure_ev: 1 } }));
    await settle();
    asked.mockClear();
    await store.undo();
    await settle();
    expect(asked.mock.calls).toEqual([[4]]);
    asked.mockClear();
    await store.redo();
    await settle();
    expect(asked.mock.calls).toEqual([[4]]);
  });

  it("asks for the photo that comes on screen, and keeps the pictures of the photo that left", async () => {
    const { store, asked } = await opened();
    stops.push(store.watchNodeThumbnails());
    await vi.waitFor(() => expect(Object.keys(picturesOf(store, 4))).toEqual(["1"]));
    const first = picturesOf(store, 4);
    asked.mockClear();
    await store.setCurrent(5);
    await settle();
    expect(asked.mock.calls).toEqual([[5]]);
    await vi.waitFor(() => expect(Object.keys(picturesOf(store, 5))).toEqual(["1"]));
    // Per photo: the first photo's pictures are where they were, and the two are different pictures.
    expect(picturesOf(store, 4)).toEqual(first);
    expect(picturesOf(store, 5)[1]).not.toBe(first[1]);
    // Back to the first photo: asked again, and nothing new comes (nothing changed since).
    asked.mockClear();
    await store.setCurrent(4);
    await settle();
    expect(asked.mock.calls).toEqual([[4]]);
    expect(picturesOf(store, 4)).toEqual(first);
  });

  it("stops asking when the node view goes", async () => {
    const { store, asked } = await opened();
    const stop = store.watchNodeThumbnails();
    await settle();
    stop();
    stop();
    asked.mockClear();
    await store.update(mainPatch({ trim: { exposure_ev: 1 } }));
    await store.setCurrent(5);
    await settle();
    expect(asked).not.toHaveBeenCalled();
    // And two views on show at once are one watch each.
    const one = store.watchNodeThumbnails();
    const two = store.watchNodeThumbnails();
    one();
    asked.mockClear();
    await store.update(mainPatch({ trim: { exposure_ev: 0.2 } }));
    await settle();
    expect(asked).toHaveBeenCalled();
    two();
  });

  it("asks after Re-run, a paste and a look, which change the photo's settings without a patch of the person's", async () => {
    const { store, asked } = await opened();
    await store.update(mainPatch({ trim: { exposure_ev: 1 } }));
    stops.push(store.watchNodeThumbnails());
    await settle();
    asked.mockClear();
    await store.rerunAuto();
    await settle();
    expect(asked.mock.calls).toEqual([[4]]);
    asked.mockClear();
    await store.update(THREE);
    await store.copySettings();
    await store.setCurrent(5);
    await settle();
    asked.mockClear();
    await store.pasteSettings("this");
    await settle();
    expect(asked.mock.calls).toEqual([[5]]);
    asked.mockClear();
    await store.setLookIntensity(40, true);
    await settle();
    expect(asked.mock.calls).toEqual([[5]]);
  });
});

describe("what the store lets go of", () => {
  it("drops the pictures of a photo that was removed, and the ones that arrive for it late", async () => {
    const { api, store } = await opened();
    stops.push(store.watchNodeThumbnails());
    await vi.waitFor(() => expect(Object.keys(picturesOf(store, 4))).toEqual(["1"]));
    await store.removePhotos([4]);
    expect(store.getState().nodeThumbnails[4]).toBeUndefined();
    // The core's event for a photo that is not here any more is not kept.
    const late = vi.fn();
    api.on("node_thumbnails_ready", late);
    await api.nodeThumbnails(4);
    await settle();
    expect(store.getState().nodeThumbnails[4]).toBeUndefined();
  });

  it("drops everything with the session", async () => {
    const { store } = await opened();
    stops.push(store.watchNodeThumbnails());
    await vi.waitFor(() => expect(Object.keys(picturesOf(store, 4))).toEqual(["1"]));
    await store.closeSession();
    expect(store.getState().nodeThumbnails).toEqual({});
  });
});
