/**
 * The store's part in a Reference Grade fit (`docs/34` §3.3, §2.2, RG-12): a fit takes seconds and the panel stays live, so an edit made while it runs keeps its
 * own undo step, the fit is judged under the look it started with, and a cancel leaves no step. The core's fit is held at a gate so that the edits can be made
 * in the middle of it.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi } from "../api.mock";
import { insertAfter } from "./graphOps";
import { cardKey, referenceStatus } from "./reference";
import { nodeOf, outputPatch, patchOf } from "./settingsGraph";
import type { ReferenceGradeItem } from "./settingsGraph";
import { createStore } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
});

/** DSC04204 current with a Reference Grade node (2) after Main and a reference chosen; `fit` is held until `release()`. */
async function setup(): Promise<{ api: MockApi; store: AppStore; release: () => void }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3 });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
  await store.setCurrent(4);
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  const outcome = insertAfter(store.currentSettings()!.settings, 1, { kind: "reference_grade" });
  if (!outcome.ok) throw new Error(outcome.reason);
  await store.update(outcome.patch, true);
  await store.chooseReferenceFile(2);
  let release = (): void => undefined;
  const gate = new Promise<void>((resolve) => (release = resolve));
  const real = api.fitReference.bind(api);
  vi.spyOn(api, "fitReference").mockImplementation(async (...args) => {
    await gate;
    return real(...args);
  });
  return { api, store, release };
}

const node = (store: AppStore): ReferenceGradeItem => nodeOf(store.currentSettings()!.settings, 2) as ReferenceGradeItem;
const photo = (store: AppStore): number => store.currentSettings()!.photo_id;

describe("an edit made while a fit runs", () => {
  it("keeps its own undo step: one undo takes back the fit's fields and leaves Strength", async () => {
    const { store, release } = await setup();
    const running = store.fitReference(2);
    await vi.waitFor(() => expect(store.getState().references[cardKey(photo(store), 2)]?.fit).not.toBeNull());
    await store.update(patchOf(2, { strength: 40 }), true);
    release();
    await running;
    expect(node(store).hue_curves.hue_vs_hue).toHaveLength(6);
    expect(node(store).strength).toBe(40);
    await store.undo();
    expect(node(store).hue_curves.hue_vs_hue).toEqual([]);
    expect(node(store).elements.sky.hue_deg).toBe(0);
    expect(node(store).strength).toBe(40);
    // The next undo takes Strength back.
    await store.undo();
    expect(node(store).strength).toBe(100);
  });
});

describe("the look changing while a fit runs", () => {
  it("makes the fit stale: the status line says the look changed, not that it was fitted", async () => {
    const { store, release } = await setup();
    const running = store.fitReference(2);
    await vi.waitFor(() => expect(store.getState().references[cardKey(photo(store), 2)]?.fit).not.toBeNull());
    await store.update(outcomeLook(), true);
    release();
    await running;
    const settings = store.currentSettings()!.settings;
    const status = referenceStatus(store.getState().references[cardKey(photo(store), 2)], node(store), settings);
    expect(status.state).toBe("look_changed");
  });

  it("does not, when it did not change", async () => {
    const { store, release } = await setup();
    const running = store.fitReference(2);
    release();
    await running;
    const status = referenceStatus(store.getState().references[cardKey(photo(store), 2)], node(store), store.currentSettings()!.settings);
    expect(status.state).toBe("fitted");
  });
});

/** A change of the Output's intensity, which is part of the look the fit depends on. */
const outcomeLook = () => outputPatch({ look_intensity: 50 });

describe("a cancelled fit", () => {
  it("adds no undo step: the next undo takes back the node's own addition", async () => {
    const { api, store, release } = await setup();
    vi.mocked(api.fitReference).mockImplementation(async () => ({
      photo_id: photo(store),
      node_id: 2,
      status: "cancelled",
      notes: [],
      skin_found: false,
      shares: [],
      settings: await api.getSettings(photo(store)),
    }));
    release();
    await store.fitReference(2);
    expect(store.getState().references[cardKey(photo(store), 2)]?.fit).toBeNull();
    await store.undo();
    expect(nodeOf(store.currentSettings()!.settings, 2)).toBeUndefined();
  });
});
