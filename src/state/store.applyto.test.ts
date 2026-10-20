/**
 * "Look selection changes the look of another photo" (the owner's report): a patch, and an "Apply to" copy, belong to the photo the
 * person was on when they made the input. These tests drive the races that used to break that with calls held back by hand, so each
 * order is the same on every run: the person picks a look on one photo and moves to another before the core has answered, picks on two
 * photos in turn, and a read of a photo's settings is on its way when a copy lands.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import type { PhotoId, SettingsPatch } from "../api/types";
import { createStore } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  vi.restoreAllMocks();
  for (const stop of stops.splice(0)) stop();
});

/** A store with the twelve photos open and developed, the first one on screen and its settings loaded. */
async function opened(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
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
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  return { api, store };
}

/** A promise that is settled from outside, to hold a command back until the test says so. */
function gate(): { open: () => void; opened: Promise<void> } {
  let open: () => void = () => {};
  const opened = new Promise<void>((resolve) => (open = resolve));
  return { open, opened };
}

const NEUTRAL = "builtin:neutral";
const SIGNATURE = "builtin:signature";
const TEAL = "arri:3000:teal-and-orange";

/** The look the core holds for a photo. */
const coreLook = async (api: MockApi, id: PhotoId): Promise<string> => (await api.getSettings(id)).values.look;
/** The look the store believes a photo has, or undefined when it holds nothing for it. */
const mirrorLook = (store: AppStore, id: PhotoId): string | undefined => store.getState().settings[id]?.values.look;
const currentId = (store: AppStore): PhotoId => store.currentPhoto()?.id ?? -1;

/** Opens a photo and waits for its settings to be in the store. */
async function visit(store: AppStore, id: PhotoId): Promise<void> {
  await store.setCurrent(id);
  await vi.waitFor(() => expect(store.getState().settings[id]).toBeDefined());
}

/** Holds back the answer of `update_settings` for the photos in `ids` until the returned gate is opened. */
function holdUpdates(api: MockApi, ids: PhotoId[]): { open: () => void } {
  const g = gate();
  const real = api.updateSettings.bind(api);
  vi.spyOn(api, "updateSettings").mockImplementation(async (id: PhotoId, patch: SettingsPatch) => {
    if (ids.includes(id)) await g.opened;
    return real(id, patch);
  });
  return g;
}

describe("a look picked on one photo, with the person moving to another before the core answers", () => {
  it("changes that photo only when Apply to is This photo", async () => {
    const { api, store } = await opened();
    store.setApplyScope("this");
    const first = currentId(store);
    await visit(store, first + 1);
    await visit(store, first);
    const held = holdUpdates(api, [first]);
    const apply = vi.spyOn(api, "applyLook");
    const picked = store.chooseLook(SIGNATURE);
    await store.setCurrent(first + 1);
    held.open();
    await picked;
    expect(await coreLook(api, first)).toBe(SIGNATURE);
    expect(await coreLook(api, first + 1)).toBe(NEUTRAL);
    expect(mirrorLook(store, first + 1)).toBe(NEUTRAL);
    expect(apply).not.toHaveBeenCalled();
  });

  it("leaves the photo on screen showing the look the core gave it when Apply to copies the look", async () => {
    // Photo 1 was picked on, photo 2 came on screen meanwhile with the settings it had before the copy landed. The copy changes photo 2
    // in the core, so what the Look tab shows for it has to be read again; it used to keep the old look selected while the photo was
    // drawn (and would be exported) with the new one.
    const { api, store } = await opened();
    store.setApplyScope("all");
    const first = currentId(store);
    await visit(store, first + 1);
    await visit(store, first);
    const held = holdUpdates(api, [first]);
    const picked = store.chooseLook(SIGNATURE);
    await store.setCurrent(first + 1);
    expect(mirrorLook(store, first + 1)).toBe(NEUTRAL);
    held.open();
    await picked;
    expect(await coreLook(api, first + 1)).toBe(SIGNATURE);
    await vi.waitFor(() => expect(mirrorLook(store, first + 1)).toBe(SIGNATURE));
    expect(mirrorLook(store, first)).toBe(SIGNATURE);
  });

  it("is not undone by a read of the photo that was on its way when the copy landed", async () => {
    const { api, store } = await opened();
    store.setApplyScope("all");
    const first = currentId(store);
    const second = first + 1;
    // Photo 2's settings are asked for as the person moves there, before the copy; the answer comes after it.
    const real = api.getSettings.bind(api);
    const late = gate();
    let held = true;
    vi.spyOn(api, "getSettings").mockImplementation(async (id: PhotoId) => {
      const rs = await real(id);
      if (id === second && held) {
        held = false;
        await late.opened;
      }
      return rs;
    });
    const updates = holdUpdates(api, [first]);
    const picked = store.chooseLook(TEAL);
    await store.setCurrent(second);
    updates.open();
    await vi.waitFor(async () => expect(await coreLook(api, second)).toBe(TEAL));
    late.open();
    await picked;
    await vi.waitFor(() => expect(store.getState().settings[second]).toBeDefined());
    expect(mirrorLook(store, second)).toBe(TEAL);
  });
});

describe("two looks picked in turn with Apply to All", () => {
  it("leave the look that was picked last on every photo, whichever photo's patch the core answers first", async () => {
    const { api, store } = await opened();
    store.setApplyScope("all");
    const first = currentId(store);
    const second = first + 1;
    await visit(store, second);
    await visit(store, first);
    // Photo 1's patch is slow: the pick made on photo 2 afterwards is answered first and its copy reaches the core first.
    const held = holdUpdates(api, [first]);
    const calls: string[] = [];
    const real = api.applyLook.bind(api);
    vi.spyOn(api, "applyLook").mockImplementation((scope, look, intensity) => {
      calls.push(look);
      return real(scope, look, intensity);
    });
    const one = store.chooseLook(SIGNATURE);
    await store.setCurrent(second);
    const two = store.chooseLook(TEAL);
    // Photo 2 has its look at once; the copy waits for the copy of the pick before it.
    expect(mirrorLook(store, second)).toBe(TEAL);
    held.open();
    await Promise.all([one, two]);
    expect(calls).toEqual([SIGNATURE, TEAL]);
    for (const id of [first, second, second + 1]) expect(await coreLook(api, id)).toBe(TEAL);
  });
});

describe("a patch that names its photo", () => {
  it("goes to that photo even when another one is on screen by the time it is sent", async () => {
    const { api, store } = await opened();
    const first = currentId(store);
    const second = first + 1;
    await visit(store, second);
    await store.update({ contrast: { kind: "manual", value: 25 } }, false, first);
    await store.update({ contrast: { kind: "manual", value: 30 } }, true, first);
    expect((await api.getSettings(first)).settings.contrast).toEqual({ kind: "manual", value: 30 });
    expect((await api.getSettings(second)).settings.contrast).toEqual({ kind: "auto" });
    expect(store.getState().settings[second]?.settings.contrast).toEqual({ kind: "auto" });
    // The drag is one undo step of the photo it was made on.
    await store.setCurrent(first);
    expect(store.getState().canUndo).toBe(true);
  });

  it("is what the end of an Intensity drag copies, with the look of the photo it began on", async () => {
    const { api, store } = await opened();
    const first = currentId(store);
    const second = first + 1;
    store.setApplyScope("this");
    await store.chooseLook(SIGNATURE);
    await visit(store, second);
    await store.chooseLook(TEAL);
    await store.setCurrent(first);
    store.setApplyScope("all");
    const apply = vi.spyOn(api, "applyLook");
    // The drag began on photo 1; photo 2 is on screen when it ends.
    await store.setLookIntensity(70, false, first);
    await store.setCurrent(second);
    await store.setLookIntensity(60, true, first);
    expect(apply).toHaveBeenCalledWith("all", SIGNATURE, 60);
    expect((await api.getSettings(first)).values.look_intensity).toBe(60);
    expect((await api.getSettings(second)).values).toMatchObject({ look: SIGNATURE, look_intensity: 60 });
    await vi.waitFor(() => expect(store.getState().settings[second]?.values).toMatchObject({ look: SIGNATURE, look_intensity: 60 }));
  });
});

describe("Paste settings to Picked", () => {
  it("pastes onto the photos that were picked when it was asked for", async () => {
    const { api, store } = await opened({ picked: [1, 2] }); // photos 2 and 3
    // Copied from photo 1 with a manual white balance, so the paste first has to look at the source's own settings.
    await store.update({ white_balance: { kind: "manual", kelvin: 4300, tint: 2 }, look: TEAL });
    await store.copySettings();
    const pasted = store.pasteSettings("picked");
    // The picks change while the paste is still finding out what to leave out.
    await store.setPicked(4, true);
    await store.setPicked(2, false);
    await pasted;
    const looks = await Promise.all([2, 3, 4].map((id) => coreLook(api, id)));
    expect(looks).toEqual([TEAL, TEAL, NEUTRAL]);
  });
});
