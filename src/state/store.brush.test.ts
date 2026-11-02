/**
 * The store's part in the Brush tool (`docs/23` §3.7, §7.6, task N-14): the brush for the next strokes and the tool are window state, a stroke is one
 * `masks.edit` row patch with `strokes_append` and one undo step whose inverse is `strokes_keep`, and a stroke that would take the row over 100,000
 * points is refused with the core's sentence before anything is sent.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi } from "../api.mock";
import type { BrushStroke, MaskItem, SettingsPatch } from "../api/types";
import { BRUSH_FULL, DEFAULT_BRUSH, MAX_BRUSH_POINTS } from "./brushStrokes";
import { newMask } from "./maskRows";
import { createStore } from "./store";
import type { AppStore } from "./store";
import { nodeOf } from "./settingsGraph";
import type { CorrectorItem } from "./settingsGraph";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
});

const picture = (from: number | "input", to: number | "output") => ({ from: { node: from, port: "picture" as const }, to: { node: to, port: "picture" as const } });
const stroke = (x: number, more: Partial<BrushStroke> = {}): BrushStroke => ({ size: 0.03, hardness: 0.5, erase: false, points: [[x, x], [x, 0.9]], ...more });
const BRUSH = (strokes: BrushStroke[]): MaskItem => ({ ...newMask("brush"), strokes }) as MaskItem;
const settingsWith = (strokes: BrushStroke[]): SettingsPatch => ({
  nodes: { "2": { kind: "corrector", name: "Skin", pos: [1, 0], masks: { invert: false, items: [newMask("radial"), BRUSH(strokes)] } } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

async function opened(strokes: BrushStroke[] = []): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3, initialSettings: { DSC04204: settingsWith(strokes) } });
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

const rowsOf = (store: AppStore): MaskItem[] => (nodeOf(store.currentSettings()!.settings, 2) as CorrectorItem).masks.items;
const strokesOf = (store: AppStore): BrushStroke[] => (rowsOf(store)[1] as Extract<MaskItem, { kind: "brush" }>).strokes;
const sentPatches = (api: MockApi): ReturnType<typeof vi.spyOn> => vi.spyOn(api, "updateSettings") as never;

describe("the brush and the tool", () => {
  it("start at the default brush with the tool off, and are window state: not a setting, not an undo step", async () => {
    const { store } = await opened();
    const before = store.currentSettings();
    expect(store.getState().brush).toEqual(DEFAULT_BRUSH);
    expect(store.getState().brushTool).toBe(false);
    store.setBrush({ size: 0.1 });
    store.setBrush({ hardness: 0.9, erase: true });
    store.setBrushTool(true);
    expect(store.getState().brush).toEqual({ size: 0.1, hardness: 0.9, erase: true });
    expect(store.getState().brushTool).toBe(true);
    expect(store.currentSettings()).toBe(before);
    expect(store.getState().canUndo).toBe(false);
  });

  it("the tool ends with a row change, a node's none, and a new photo, and a row chosen again starts over", async () => {
    const { api, store } = await opened();
    store.selectMaskRow(1);
    store.setBrushTool(true);
    store.selectMaskRow(0);
    expect(store.getState().brushTool).toBe(false);
    store.selectMaskRow(1);
    store.setBrushTool(true);
    // Choosing the same row again ends it, and the caller that wants it back turns it on.
    store.selectMaskRow(1);
    expect(store.getState().brushTool).toBe(false);
    store.setBrushTool(true);
    store.selectMaskRow(null);
    expect(store.getState().brushTool).toBe(false);
    store.selectMaskRow(1);
    store.setBrushTool(true);
    await store.setCurrent((await api.sessionState()).photos[4]!.id);
    expect(store.getState().brushTool).toBe(false);
    expect(store.getState().maskRow).toBeNull();
  });

  it("keeps the brush when the tool or the photo changes", async () => {
    const { api, store } = await opened();
    store.setBrush({ size: 0.2, erase: true });
    store.setBrushTool(true);
    await store.setCurrent((await api.sessionState()).photos[4]!.id);
    expect(store.getState().brush).toMatchObject({ size: 0.2, erase: true });
  });
});

describe("a stroke (strokes_append)", () => {
  it("is one row patch with the stroke appended, shown at once, and sends no other stroke of the row", async () => {
    const { api, store } = await opened([stroke(0.1), stroke(0.2)]);
    const spy = sentPatches(api);
    const sent = await store.addStroke(2, 1, stroke(0.3));
    expect(sent).toBe(true);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0]?.[1]).toEqual({ nodes: { "2": { masks: { edit: { "1": { strokes_append: [stroke(0.3)] } } } } } });
    expect(strokesOf(store).map((s) => s.points[0]![0])).toEqual([0.1, 0.2, 0.3]);
    // The core has it too.
    const rs = await api.getSettings(store.currentSettings()!.photo_id);
    expect(((nodeOf(rs.settings, 2) as CorrectorItem).masks.items[1] as { strokes: BrushStroke[] }).strokes).toHaveLength(3);
  });

  it("is one undo step, whose inverse keeps the strokes there were, and redo appends it again", async () => {
    const { store } = await opened([stroke(0.1)]);
    await store.addStroke(2, 1, stroke(0.5, { erase: true }));
    expect(store.getState().canUndo).toBe(true);
    await store.addStroke(2, 1, stroke(0.6));
    expect(strokesOf(store)).toHaveLength(3);
    await store.undo();
    expect(strokesOf(store).map((s) => s.points[0]![0])).toEqual([0.1, 0.5]);
    await store.undo();
    expect(strokesOf(store).map((s) => s.points[0]![0])).toEqual([0.1]);
    expect(store.getState().canUndo).toBe(false);
    await store.redo();
    expect(strokesOf(store)).toHaveLength(2);
    expect(strokesOf(store)[1]).toMatchObject({ erase: true });
    await store.redo();
    expect(strokesOf(store)).toHaveLength(3);
  });

  it("an undo sends strokes_keep with the old count, and a redo the stroke", async () => {
    const { api, store } = await opened([stroke(0.1), stroke(0.2)]);
    await store.addStroke(2, 1, stroke(0.3));
    const spy = sentPatches(api);
    await store.undo();
    await vi.waitFor(() => expect(spy).toHaveBeenCalled());
    expect(spy.mock.calls.at(-1)?.[1]).toEqual({ nodes: { "2": { masks: { edit: { "1": { strokes_keep: 2 } } } } } });
    await store.redo();
    await vi.waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
    expect(spy.mock.calls.at(-1)?.[1]).toEqual({ nodes: { "2": { masks: { edit: { "1": { strokes_append: [stroke(0.3)] } } } } } });
  });

  it("takes nothing for a row that is not a Brush row, an empty stroke, or a row the node does not have", async () => {
    const { api, store } = await opened();
    const spy = sentPatches(api);
    expect(await store.addStroke(2, 0, stroke(0.3))).toBe(false);
    expect(await store.addStroke(2, 1, { ...stroke(0.3), points: [] })).toBe(false);
    expect(await store.addStroke(2, 7, stroke(0.3))).toBe(false);
    expect(await store.addStroke(9, 0, stroke(0.3))).toBe(false);
    expect(spy).not.toHaveBeenCalled();
    expect(store.getState().error).toBeNull();
  });

  it("is refused at 100,000 points with the core's sentence in the error bar, before anything is sent, and changes nothing", async () => {
    // 99,998 points in all; a stroke of two more is 100,000 (allowed), of three is one over.
    const many: BrushStroke[] = Array.from({ length: 99 }, (_, i) => ({ size: 0.03, hardness: 0.5, erase: false, points: Array.from({ length: 1000 }, (_, k): [number, number] => [(k + 1) / 1001, (i + 1) / 100]) }));
    const rest: BrushStroke = { size: 0.03, hardness: 0.5, erase: false, points: Array.from({ length: 998 }, (_, k): [number, number] => [(k + 1) / 999, 0.995]) };
    const { api, store } = await opened([...many, rest]);
    const spy = sentPatches(api);
    expect(strokesOf(store).reduce((n, s) => n + s.points.length, 0)).toBe(MAX_BRUSH_POINTS - 2);
    expect(await store.addStroke(2, 1, { ...stroke(0.4), points: [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]] })).toBe(false);
    expect(spy).not.toHaveBeenCalled();
    expect(store.getState().error).toEqual({ scope: "render", message: BRUSH_FULL });
    expect(BRUSH_FULL).toBe("A brush mask can have up to 100,000 points.");
    expect(strokesOf(store)).toHaveLength(100);
    expect(store.getState().canUndo).toBe(false);
    // Exactly at the limit is allowed.
    store.dismissError();
    expect(await store.addStroke(2, 1, { ...stroke(0.4), points: [[0.1, 0.1], [0.2, 0.2]] })).toBe(true);
    expect(strokesOf(store)).toHaveLength(101);
    expect(store.getState().error).toBeNull();
  });

  it("the core refuses the same patch with the same sentence (the page's check is not the only one)", async () => {
    const many: BrushStroke[] = Array.from({ length: 100 }, (_, i) => ({ size: 0.03, hardness: 0.5, erase: false, points: Array.from({ length: 1000 }, (_, k): [number, number] => [(k + 1) / 1001, (i + 1) / 101]) }));
    const { api, store } = await opened(many);
    const id = store.currentSettings()!.photo_id;
    await expect(api.updateSettings(id, { nodes: { "2": { masks: { edit: { "1": { strokes_append: [stroke(0.3)] } } } } } })).rejects.toMatchObject({ message: BRUSH_FULL });
  });
});
