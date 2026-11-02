/**
 * The store's part in the Object mask (`docs/23` §4.4, §7.6, task N-18): the tool and the card's mode are window state, a click or a box is a `mask_find`, a
 * found answer is **one** `masks.edit` row patch (one undo step) and the pill "Object found · 0.4 s" for three seconds, nothing found changes nothing and
 * says so in the error bar, finds are answered one at a time, and `object_state` tells which photo's encoder is working.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import type { MaskItem, SettingsPatch } from "../api/types";
import { newMask } from "./maskRows";
import { DEFAULT_OBJECT_MODE, FOUND_PILL_MS, NOTHING_FOUND } from "./objectMask";
import { nodeOf } from "./settingsGraph";
import type { CorrectorItem } from "./settingsGraph";
import { createStore } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  vi.restoreAllMocks();
  for (const stop of stops.splice(0)) stop();
});

const picture = (from: number | "input", to: number | "output") => ({ from: { node: from, port: "picture" as const }, to: { node: to, port: "picture" as const } });
const OBJECT = (more: Record<string, unknown> = {}): MaskItem => ({ ...newMask("object"), ...more }) as MaskItem;
const settingsWith = (rows: MaskItem[]): SettingsPatch => ({
  nodes: { "2": { kind: "corrector", name: "Mountain", pos: [1, 0], masks: { invert: false, items: rows } } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

async function opened(rows: MaskItem[] = [OBJECT()], options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3, initialSettings: { DSC04204: settingsWith(rows) }, ...options });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
  await store.setCurrent((await api.sessionState()).photos[3]!.id);
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  return { api, store };
}

const row = (store: AppStore, index = 0): Extract<MaskItem, { kind: "object" }> => (nodeOf(store.currentSettings()!.settings, 2) as CorrectorItem).masks.items[index] as never;
/** A click on the mock's mountain (the far hill of the flat scene) and one on the sky above it. */
const ON_HILL = { kind: "point", x: 0.5, y: 0.7, positive: true } as const;
const ON_SKY = { kind: "point", x: 0.2, y: 0.1, positive: true } as const;

describe("the tool and the card's mode", () => {
  it("start at Add point with the tool off, and are window state: not a setting, not an undo step", async () => {
    const { store } = await opened();
    const before = store.currentSettings();
    expect(store.getState().objectMode).toBe(DEFAULT_OBJECT_MODE);
    expect(store.getState().objectTool).toBe(false);
    store.setObjectMode("box");
    store.setObjectTool(true);
    expect(store.getState().objectMode).toBe("box");
    expect(store.getState().objectTool).toBe(true);
    expect(store.currentSettings()).toBe(before);
    expect(store.getState().canUndo).toBe(false);
  });

  it("the tool ends with a row change, no row and a new photo", async () => {
    const { api, store } = await opened([OBJECT(), OBJECT()]);
    store.selectMaskRow(0);
    store.setObjectTool(true);
    store.selectMaskRow(1);
    expect(store.getState().objectTool).toBe(false);
    store.setObjectTool(true);
    store.selectMaskRow(null);
    expect(store.getState().objectTool).toBe(false);
    store.selectMaskRow(0);
    store.setObjectTool(true);
    await store.setCurrent((await api.sessionState()).photos[4]!.id);
    expect(store.getState().objectTool).toBe(false);
  });
});

describe("findObject", () => {
  it("applies a found answer as one masks.edit patch and one undo step, and shows the pill for three seconds", async () => {
    const { api, store } = await opened();
    const sent = vi.spyOn(api, "updateSettings");
    const timeout = vi.spyOn(globalThis, "setTimeout");
    expect(await store.findObject(2, 0, ON_HILL)).toBe(true);
    expect(sent).toHaveBeenCalledTimes(1);
    expect(sent.mock.calls[0]![1]).toEqual({ nodes: { "2": { masks: { edit: { "0": { points: [[0.5, 0.7, true]], box: null } } } } } });
    expect(row(store).points).toEqual([[0.5, 0.7, true]]);
    expect(store.getState().objectPill?.text).toBe("Object found · 0.4 s");
    // It goes after three seconds.
    const hide = timeout.mock.calls.find((c) => c[1] === FOUND_PILL_MS);
    expect(hide).toBeDefined();
    (hide![0] as () => void)();
    expect(store.getState().objectPill).toBeNull();
    // One step takes it back.
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    expect(row(store).points).toEqual([]);
    expect(store.getState().canUndo).toBe(false);
  });

  it("a second click adds to the row, a negative one takes a part away, and a box sets the box", async () => {
    const { store } = await opened();
    expect(await store.findObject(2, 0, ON_HILL)).toBe(true);
    expect(await store.findObject(2, 0, { kind: "point", x: 0.6, y: 0.75, positive: false })).toBe(true);
    expect(row(store).points).toEqual([[0.5, 0.7, true], [0.6, 0.75, false]]);
    expect(await store.findObject(2, 0, { kind: "box", x0: 0.7, y0: 0.5, x1: 0.3, y1: 0.8 })).toBe(true);
    expect(row(store).box).toEqual([0.3, 0.5, 0.7, 0.8]);
    expect(row(store).points.length).toBe(2);
  });

  it("nothing found adds no point, changes nothing, and says so in the error bar", async () => {
    const { api, store } = await opened();
    const sent = vi.spyOn(api, "updateSettings");
    const before = store.currentSettings();
    expect(await store.findObject(2, 0, ON_SKY)).toBe(false);
    expect(sent).not.toHaveBeenCalled();
    expect(store.currentSettings()).toBe(before);
    expect(store.getState().canUndo).toBe(false);
    expect(store.getState().objectPill).toBeNull();
    expect(store.getState().error).toEqual({ scope: "mask", message: NOTHING_FOUND });
    expect(NOTHING_FOUND).toBe("Nothing found there. Click another part of the object, or draw a box.");
  });

  it("answers one find at a time: a click that comes while another is being answered reads the row as that one left it", async () => {
    const { store } = await opened();
    const both = await Promise.all([store.findObject(2, 0, ON_HILL), store.findObject(2, 0, { kind: "point", x: 0.55, y: 0.72, positive: true })]);
    expect(both).toEqual([true, true]);
    expect(row(store).points).toEqual([[0.5, 0.7, true], [0.55, 0.72, true]]);
  });

  it("says Finding… while the core works: objectFinding names the photo and ends with the answer", async () => {
    const { api, store } = await opened();
    const id = store.currentSettings()!.photo_id;
    // A find takes the mock a moment (`FIND_MS`, 120 ms at speed 1).
    api.configure({ speed: 1 });
    const done = store.findObject(2, 0, ON_HILL);
    await vi.waitFor(() => expect(store.getState().objectFinding).toBe(id));
    await done;
    expect(store.getState().objectFinding).toBeNull();
  });

  it("is refused for a row that is not an Object row, and a refusal of the core goes to the error bar with its sentence", async () => {
    const { api, store } = await opened([newMask("radial"), OBJECT()]);
    expect(await store.findObject(2, 0, ON_HILL)).toBe(false);
    expect(store.getState().error).toBeNull();
    const full = OBJECT({ points: Array.from({ length: 256 }, () => [0.5, 0.7, true]) });
    const again = await opened([full]);
    expect(await again.store.findObject(2, 0, ON_HILL)).toBe(false);
    expect(again.store.getState().error).toEqual({ scope: "mask", message: "An Object mask can have up to 256 points." });
    void api;
  });

  it("?mockMaskFind=none answers nothing found everywhere, and the box finds the hill by its middle", async () => {
    const none = await opened([OBJECT()], { maskFind: "none" });
    expect(await none.store.findObject(2, 0, ON_HILL)).toBe(false);
    const { store } = await opened();
    expect(await store.findObject(2, 0, { kind: "box", x0: 0.1, y0: 0.05, x1: 0.3, y1: 0.2 })).toBe(false);
    expect(await store.findObject(2, 0, { kind: "box", x0: 0.3, y0: 0.5, x1: 0.7, y1: 0.8 })).toBe(true);
  });
});

describe("object_state", () => {
  it("lists the photos whose encoder is working, from the events: an Object row on the photo on screen starts it, and a find waits for it", async () => {
    const { api, store } = await opened([]);
    const id = store.currentSettings()!.photo_id;
    expect(store.getState().objectPreparing).toEqual([]);
    // The encoder takes a moment (the mock's `ENCODER_MS`, 0.9 s at speed 1).
    api.configure({ speed: 1 });
    await store.update({ nodes: { "2": { masks: { items: [OBJECT()] } } } }, true);
    await vi.waitFor(() => expect(store.getState().objectPreparing).toEqual([id]));
    // A click that comes meanwhile is answered after the encoder has finished.
    expect(await store.findObject(2, 0, ON_HILL)).toBe(true);
    expect(store.getState().objectPreparing).toEqual([]);
    // Once for each photo: another row does not start it again.
    const seen: boolean[] = [];
    const stop = store.subscribe(() => seen.push(store.getState().objectPreparing.includes(id)));
    await store.update({ nodes: { "2": { masks: { items: [OBJECT(), OBJECT()] } } } }, true);
    stop();
    expect(seen).not.toContain(true);
  });
});
