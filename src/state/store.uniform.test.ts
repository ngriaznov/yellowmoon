/**
 * The store's part in Colour Uniform, on the mock (`docs/32` sections 2.1, 2.3 and 3.11): Add node, Pick colour, one undo step for the first pick,
 * and a later pick that changes only the target. The whole path of CU-10 that the page owns, down to the `sample_chroma` call it makes.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi } from "../api.mock";
import { UNIFORM_PICK_RADIUS_PX } from "../components/uniformMath";
import { insertAfter } from "./graphOps";
import { createStore } from "./store";
import type { AppStore } from "./store";
import { inputPatch, nodeOf } from "./settingsGraph";
import type { ColorUniformItem } from "./settingsGraph";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
  vi.useRealTimers();
});

/** The twelve photos open and developed, DSC04204 current, a Colour Uniform node (2) after Main, the viewport on the whole cell. */
async function withNode(): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3 });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
  await store.setCurrent(4);
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  store.setViewportRect({ x: 0, y: 0, width: 1500, height: 1000, scale: 1 });
  const outcome = insertAfter(store.currentSettings()!.settings, 1, { kind: "color_uniform" });
  if (!outcome.ok) throw new Error(outcome.reason);
  await store.update(outcome.patch, true);
  return { api, store };
}

const uniform = (store: AppStore): ColorUniformItem => nodeOf(store.currentSettings()!.settings, 2) as ColorUniformItem;

describe("Pick colour of a Colour Uniform node", () => {
  it("asks the core for a 7 x 7 mean at the node's input, and the first pick is one patch: target, Compress and a Colour range row", async () => {
    const { api, store } = await withNode();
    expect(uniform(store).target).toBeNull();
    const sample = vi.spyOn(api, "sampleChroma");
    await store.pickUniform(20, 20, 2);
    expect(sample).toHaveBeenCalledWith(20, 20, 2, "node_input", UNIFORM_PICK_RADIUS_PX);
    expect(UNIFORM_PICK_RADIUS_PX).toBe(3);
    const picked = uniform(store);
    expect(picked.target).not.toBeNull();
    expect(picked.compress.saturation).toBeGreaterThan(0);
    expect(picked.masks.items).toHaveLength(1);
    expect(picked.masks.items[0]!.kind).toBe("color_range");
    // The row is narrower than the Colour range's own default of 0.05 (docs/32 §2.3).
    expect(picked.masks.items[0]).toMatchObject({ width: 0.02, softness: 0.5 });
  });

  it("undoes the first pick in one step, back to the node as it was added", async () => {
    const { store } = await withNode();
    await store.pickUniform(20, 20, 2);
    await store.undo();
    const back = uniform(store);
    expect(back.target).toBeNull();
    expect(back.compress).toEqual({ hue: 0, saturation: 0, lightness: 0 });
    expect(back.masks.items).toEqual([]);
    // One more undo takes the node itself away: the pick was not two steps.
    await store.undo();
    expect(nodeOf(store.currentSettings()!.settings, 2)).toBeUndefined();
  });

  it("changes only the target on a later pick, and that is one undo step too", async () => {
    const { store } = await withNode();
    await store.pickUniform(20, 20, 2);
    const first = uniform(store);
    await store.pickUniform(750, 950, 2);
    const second = uniform(store);
    expect(second.target).not.toEqual(first.target);
    expect(second.compress).toEqual(first.compress);
    expect(second.masks).toEqual(first.masks);
    await store.undo();
    expect(uniform(store)).toEqual(first);
  });

  it("does nothing on the paper around a letterboxed photo", async () => {
    const { store } = await withNode();
    await store.update(inputPatch({ orientation: 90 }));
    store.setViewportRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    await store.pickUniform(20, 300, 2);
    expect(uniform(store).target).toBeNull();
  });
});
