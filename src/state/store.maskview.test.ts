/**
 * The store's part in Show mask (`docs/23` §5.8, §9): it keeps `{ on, style }` (the style starts as Overlay) and sends the core
 * `viewport_set_mode(mode, mask_view)` whenever the corrector the node view shows, the view or the style changes: `{ node, style }` with a
 * corrector and the view on, `null` otherwise, and again for a new photo (the core starts a photo without one).
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi } from "../api.mock";
import type { MaskView } from "../api/types";
import { createStore } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
});

async function opened(): Promise<{ api: MockApi; store: AppStore; sent: ReturnType<typeof vi.spyOn> }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3 });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
  await store.setCurrent(4);
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  const sent = vi.spyOn(api, "viewportSetMode");
  return { api, store, sent };
}

/** The mask views the store sent, in order (the second argument of each call). */
const views = (sent: ReturnType<typeof vi.spyOn>): Array<MaskView | null> => sent.mock.calls.map((c: unknown[]) => (c[1] ?? null) as MaskView | null);

describe("the store's mask view", () => {
  it("starts off, with the Overlay style, and sends nothing until there is a corrector to show", async () => {
    const { store, sent } = await opened();
    expect(store.getState().maskView).toEqual({ on: false, style: "overlay" });
    store.setMaskNode(1);
    store.toggleMaskView();
    expect(views(sent)).toEqual([{ node: 1, style: "overlay" }]);
  });

  it("sends the node and the style whenever the node, the view or the style changes, and nothing for what does not change them", async () => {
    const { store, sent } = await opened();
    store.setMaskNode(1);
    store.setMaskView({ on: true });
    store.setMaskNode(1);
    store.setMaskNode(3);
    store.setMaskView({ style: "mask_only" });
    store.setMaskView({ style: "mask_only" });
    store.toggleMaskView();
    store.toggleMaskView();
    expect(views(sent)).toEqual([
      { node: 1, style: "overlay" },
      { node: 3, style: "overlay" },
      { node: 3, style: "mask_only" },
      null,
      { node: 3, style: "mask_only" },
    ]);
    // The mode the page mirrors goes with it.
    expect(sent.mock.calls.every((c: unknown[]) => (c[0] as { kind: string }).kind === "fit")).toBe(true);
  });

  it("choosing a style turns the view on, with the last style kept while it is off", async () => {
    const { store, sent } = await opened();
    store.setMaskNode(1);
    store.setMaskView({ style: "mask_only" });
    expect(store.getState().maskView).toEqual({ on: true, style: "mask_only" });
    store.toggleMaskView();
    store.toggleMaskView();
    expect(views(sent)).toEqual([{ node: 1, style: "mask_only" }, null, { node: 1, style: "mask_only" }]);
  });

  it("ends the view when there is no corrector (the Input, the Output, the simple view, Crop, Export) and brings it back with the next one", async () => {
    const { store, sent } = await opened();
    store.setMaskNode(2);
    store.toggleMaskView();
    store.setMaskNode(null);
    store.setMaskNode(null);
    expect(store.getState().maskView.on).toBe(true);
    store.setMaskNode(2);
    expect(views(sent)).toEqual([{ node: 2, style: "overlay" }, null, { node: 2, style: "overlay" }]);
  });

  it("tells the core again for a new photo, which starts without a mask view, and nothing when there is none to restore", async () => {
    const { store, sent } = await opened();
    store.setMaskNode(1);
    store.toggleMaskView();
    expect(views(sent)).toHaveLength(1);
    await store.setCurrent(5);
    expect(views(sent)).toEqual([
      { node: 1, style: "overlay" },
      { node: 1, style: "overlay" },
    ]);
    // With the view off a new photo needs nothing sent.
    store.toggleMaskView();
    const before = sent.mock.calls.length;
    await store.setCurrent(6);
    expect(sent.mock.calls.length).toBe(before);
  });

  it("asks for the mask view only through viewport_set_mode (the mock's frame carries it)", async () => {
    const { api, store } = await opened();
    store.setMaskNode(1);
    store.setMaskView({ style: "mask_only" });
    await vi.waitFor(() => expect(api.viewport.maskView).toEqual({ node: 1, style: "mask_only" }));
    store.setMaskNode(null);
    await vi.waitFor(() => expect(api.viewport.maskView).toBeNull());
  });
});

describe("the store's mask row eye (docs/23 §5.8, §7.6)", () => {
  it("shows one row of the node on show, in the View card's style, and writes no settings and no undo step", async () => {
    const { api, store, sent } = await opened();
    const updates = vi.spyOn(api, "updateSettings");
    store.setMaskNode(2);
    store.setMaskEye(1);
    expect(store.getState().maskEye).toBe(1);
    expect(views(sent)).toEqual([{ node: 2, style: "overlay", row: 1 }]);
    store.setMaskView({ style: "mask_only" });
    expect(views(sent).at(-1)).toEqual({ node: 2, style: "mask_only", row: 1 });
    // Choosing a style with the eye on does not turn Show mask on.
    expect(store.getState().maskView).toEqual({ on: false, style: "mask_only" });
    expect(updates).not.toHaveBeenCalled();
    expect(store.getState().canUndo).toBe(false);
    await vi.waitFor(() => expect(api.viewport.maskView).toEqual({ node: 2, style: "mask_only", row: 1 }));
  });

  it("is one row at a time: another row's eye moves it, null turns it off, the same row again changes nothing", async () => {
    const { store, sent } = await opened();
    store.setMaskNode(2);
    store.setMaskEye(0);
    store.setMaskEye(0);
    store.setMaskEye(3);
    store.setMaskEye(null);
    store.setMaskEye(null);
    expect(views(sent)).toEqual([{ node: 2, style: "overlay", row: 0 }, { node: 2, style: "overlay", row: 3 }, null]);
  });

  it("is exclusive with Show mask: the eye turns it off, and turning it on (the toggle, M) turns the eye off", async () => {
    const { store, sent } = await opened();
    store.setMaskNode(2);
    store.setMaskView({ on: true });
    store.setMaskEye(1);
    expect(store.getState().maskView.on).toBe(false);
    expect(views(sent)).toEqual([{ node: 2, style: "overlay" }, { node: 2, style: "overlay", row: 1 }]);
    store.toggleMaskView();
    expect(store.getState().maskEye).toBeNull();
    expect(store.getState().maskView.on).toBe(true);
    expect(views(sent).at(-1)).toEqual({ node: 2, style: "overlay" });
    // The style segment turns Show mask on, as it did, when no eye is on, and ends an eye that is on.
    store.setMaskEye(0);
    store.setMaskView({ on: true });
    expect(store.getState().maskEye).toBeNull();
  });

  it("ends with the node: another node, none, a new photo, the Before view; and it cannot start without a node", async () => {
    const { store, sent } = await opened();
    store.setMaskEye(0);
    expect(store.getState().maskEye).toBeNull();
    expect(sent).not.toHaveBeenCalled();
    store.setMaskNode(2);
    store.setMaskEye(0);
    store.setMaskNode(2);
    expect(store.getState().maskEye).toBe(0);
    store.setMaskNode(3);
    expect(store.getState().maskEye).toBeNull();
    store.setMaskEye(2);
    store.setMaskNode(null);
    expect(store.getState().maskEye).toBeNull();
    store.setMaskNode(3);
    store.setMaskEye(2);
    await store.setCurrent(5);
    expect(store.getState().maskEye).toBeNull();
    store.setMaskNode(3);
    store.setMaskEye(2);
    await store.setBefore(true);
    expect(store.getState().maskEye).toBeNull();
    expect(views(sent).at(-1)).toBeNull();
  });
});

describe("Show depth (docs/23 §4.8)", () => {
  it("sends the Depth row's own view with the depth flag, in Mask only, and is one view at a time with Show mask and the eyes", async () => {
    const { store, sent } = await opened();
    store.setMaskNode(1);
    store.setShowDepth(0);
    expect(store.getState().showDepth).toBe(0);
    store.setMaskEye(2);
    expect(store.getState().showDepth).toBeNull();
    store.setShowDepth(1);
    expect(store.getState().maskEye).toBeNull();
    store.setMaskView({ on: true });
    expect(store.getState().showDepth).toBeNull();
    store.setShowDepth(0);
    expect(store.getState().maskView.on).toBe(false);
    store.setMaskEye(null);
    expect(store.getState().showDepth).toBeNull();
    expect(views(sent)).toEqual([
      { node: 1, style: "mask_only", row: 0, depth: true },
      { node: 1, style: "overlay", row: 2 },
      { node: 1, style: "mask_only", row: 1, depth: true },
      { node: 1, style: "overlay" },
      { node: 1, style: "mask_only", row: 0, depth: true },
      null,
    ]);
  });

  it("ends with the node, and is ignored where there is no corrector to show", async () => {
    const { store, sent } = await opened();
    store.setShowDepth(0);
    expect(store.getState().showDepth).toBeNull();
    store.setMaskNode(1);
    store.setShowDepth(0);
    store.setMaskNode(3);
    expect(store.getState().showDepth).toBeNull();
    expect(views(sent)).toEqual([{ node: 1, style: "mask_only", row: 0, depth: true }, null]);
  });
});
