/**
 * The store against the real core's behaviour (M3-02, M3-03 frontend side): the viewport backend and its frames, Pick white through the
 * core, `settings_changed`, `export_busy`, and a new photo opening at Fit with Before off. The mock stands in for the core and sends
 * the same events.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import type { ViewportFrame } from "../api/types";
import { createStore } from "./store";
import type { AppStore } from "./store";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
});

async function connected(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, ...options });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  return { api, store };
}

async function opened(options: MockOptions = {}): Promise<{ api: MockApi; store: AppStore }> {
  const ctx = await connected(options);
  await ctx.store.openDialog("files");
  ctx.api.flush();
  await vi.waitFor(() => {
    const photos = ctx.store.getState().session?.photos ?? [];
    expect(photos).toHaveLength(12);
    expect(photos.every((p) => p.status.kind === "ready" || p.status.kind === "skipped")).toBe(true);
  });
  await vi.waitFor(() => expect(ctx.store.currentSettings()).not.toBeNull());
  return ctx;
}

const frame = (seq: number, url = `ym-frame://localhost/frame-${seq}.png`): ViewportFrame => ({ seq, width: 2040, height: 1360, sharp: true, url });

describe("the viewport backend", () => {
  it("is unknown, and null, for an API that has no viewport (the browser mock)", async () => {
    const { store } = await connected();
    expect(store.getState().viewportBackend).toBeNull();
  });

  it.each(["native", "fallback"] as const)("asks viewport_info once connected: %s", async (backend) => {
    const { store } = await connected({ viewport: backend });
    expect(store.getState().viewportBackend).toBe(backend);
  });

  it("treats a core that can't answer as no backend", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY, viewport: "native" });
    api.failNext("viewportInfo", { scope: "app", message: "Not now." });
    const store = createStore(api, 0);
    await store.load();
    expect(store.getState().viewportBackend).toBeNull();
    expect(store.getState().error).toBeNull();
  });

  it("keeps the backend when the session closes", async () => {
    const { store } = await opened({ viewport: "fallback" });
    await store.closeSession();
    expect(store.getState().viewportBackend).toBe("fallback");
  });
});

describe("viewport_frame", () => {
  it("keeps the newest frame, at once", async () => {
    const { api, store } = await connected({ viewport: "fallback" });
    api.emitEvent("viewport_frame", frame(1));
    expect(store.getState().viewportFrame).toEqual(frame(1));
    api.emitEvent("viewport_frame", frame(2));
    expect(store.getState().viewportFrame?.seq).toBe(2);
  });

  it("ignores a frame that arrives after a newer one", async () => {
    const { api, store } = await connected({ viewport: "fallback" });
    api.emitEvent("viewport_frame", frame(5));
    api.emitEvent("viewport_frame", frame(4));
    expect(store.getState().viewportFrame?.seq).toBe(5);
  });

  it("an empty address takes the picture away, and the next frame is shown whatever its number", async () => {
    const { api, store } = await connected({ viewport: "fallback" });
    api.emitEvent("viewport_frame", frame(7));
    api.emitEvent("viewport_frame", { seq: 8, width: 0, height: 0, sharp: true, url: "" });
    expect(store.getState().viewportFrame).toBeNull();
    api.emitEvent("viewport_frame", frame(9));
    expect(store.getState().viewportFrame?.seq).toBe(9);
  });

  it("a frame of a cell that has no size yet (the shell draws 1 × 1 before the webview reports the cell) is no picture", async () => {
    const { api, store } = await connected({ viewport: "fallback" });
    api.emitEvent("viewport_frame", { seq: 1, width: 1, height: 1, sharp: true, url: "ym-frame://localhost/frame-1.png" });
    expect(store.getState().viewportFrame).toBeNull();
    api.emitEvent("viewport_frame", frame(2));
    expect(store.getState().viewportFrame?.seq).toBe(2);
    // A tiny frame after a real one takes the picture away, like an empty address: the cell is being drawn again.
    api.emitEvent("viewport_frame", { seq: 3, width: 1, height: 1, sharp: true, url: "ym-frame://localhost/frame-3.png" });
    expect(store.getState().viewportFrame).toBeNull();
  });

  it("a frame that was numbered before the picture was taken away, and arrives after it, does not bring it back", async () => {
    // The shell numbers a frame before it encodes it, and sends the news that the picture is gone from another thread.
    const { api, store } = await connected({ viewport: "fallback" });
    api.emitEvent("viewport_frame", frame(5));
    api.emitEvent("viewport_frame", { seq: 7, width: 0, height: 0, sharp: true, url: "" });
    expect(store.getState().viewportFrame).toBeNull();
    api.emitEvent("viewport_frame", frame(6));
    expect(store.getState().viewportFrame).toBeNull();
    api.emitEvent("viewport_frame", frame(8));
    expect(store.getState().viewportFrame?.seq).toBe(8);
  });

  it("closing the session forgets the frame", async () => {
    const { api, store } = await opened({ viewport: "fallback" });
    api.emitEvent("viewport_frame", frame(3));
    await store.closeSession();
    expect(store.getState().viewportFrame).toBeNull();
  });
});

describe("a new photo", () => {
  it("opens at Fit with the camera view off, in the core and in the mirror", async () => {
    const { api, store } = await opened();
    await store.setZoom({ kind: "percent", value: 200 });
    await store.toggleBefore();
    const next = store.getState().session!.photos[1]!;
    await store.setCurrent(next.id);
    expect(store.getState().zoom).toEqual({ kind: "fit" });
    expect(store.getState().before).toBe(false);
    await vi.waitFor(() => expect(api.viewport).toMatchObject({ mode: { kind: "fit" }, before: false }));
  });
});

describe("a wheel zoom", () => {
  it("moves the mirrored zoom as the core's moves, so the header and the drag to pan follow it", async () => {
    const { store } = await opened({ viewport: "native" });
    // A 3:2 photo of 8640 px in a cell of 1020 px at 2×: Fit is 24 %.
    store.setViewportRect({ x: 0, y: 75, width: 1020, height: 680, scale: 2 });
    expect(store.fitPercent()).toBe(24);
    await store.viewportInput({ kind: "zoom", factor: 2, x: 500, y: 300 });
    // From the Fit zoom as the core has it (23.61 %, not the 24 % the header rounds it to).
    expect(store.getState().zoom).toEqual({ kind: "percent", value: 47.22 });
    await store.viewportInput({ kind: "zoom", factor: 1.5, x: 500, y: 300 });
    expect(store.getState().zoom.kind).toBe("percent");
    // Zooming out past Fit stays at Fit.
    await store.viewportInput({ kind: "zoom", factor: 0.1, x: 500, y: 300 });
    expect(store.getState().zoom).toEqual({ kind: "fit" });
    await store.viewportInput({ kind: "zoom", factor: 0.5, x: 500, y: 300 });
    expect(store.getState().zoom).toEqual({ kind: "fit" });
  });

  it("is forwarded to the core as it was", async () => {
    const { api, store } = await opened();
    await store.viewportInput({ kind: "zoom", factor: 1.2, x: 10, y: 20 });
    expect(api.viewport.inputs).toContainEqual({ kind: "zoom", factor: 1.2, x: 10, y: 20 });
  });
});

describe("Pick white", () => {
  it("sends the click to the core as a point of the cell, reads the settings back, and is one undo step", async () => {
    const { api, store } = await opened();
    store.setViewportRect({ x: 0, y: 75, width: 900, height: 600, scale: 2 });
    await vi.waitFor(() => expect(api.viewport.rect?.width).toBe(900));
    const update = vi.spyOn(api, "updateSettings");
    const input = vi.spyOn(api, "viewportInput");
    await store.pickWhite(450, 150);
    expect(input).toHaveBeenCalledWith({ kind: "pick_white", x: 450, y: 150 });
    expect(update).not.toHaveBeenCalled();
    const rs = store.currentSettings()!;
    expect(rs.settings.white_pick).toEqual([0.5, 0.25]);
    expect(rs.settings.white_balance.kind).toBe("manual");
    expect(store.getState().canUndo).toBe(true);
    await store.undo();
    expect(store.currentSettings()!.settings.white_pick).toBeNull();
    expect(store.currentSettings()!.auto.white_balance).toBe(true);
    expect(store.getState().canRedo).toBe(true);
  });

  it("a click on the paper changes nothing and records nothing", async () => {
    const { api, store } = await opened();
    store.setViewportRect({ x: 0, y: 75, width: 900, height: 600, scale: 2 });
    await vi.waitFor(() => expect(api.viewport.rect?.width).toBe(900));
    await store.update({ orientation: 90 });
    const undoDepth = store.getState().canUndo;
    await store.pickWhite(10, 300);
    expect(store.currentSettings()!.settings.white_pick).toBeNull();
    expect(store.getState().canUndo).toBe(undoDepth);
  });

  it("shows the core's words when it refuses the spot, and keeps the settings", async () => {
    const { api, store } = await opened();
    api.failNext("viewportInput", { scope: "render", message: "That spot is too dark to be a white. Click something lighter." });
    await store.pickWhite(450, 300);
    expect(store.getState().error).toEqual({ scope: "render", message: "That spot is too dark to be a white. Click something lighter." });
    expect(store.currentSettings()!.settings.white_pick).toBeNull();
  });

  it("waits for the patches in flight, so the undo step starts from the settings the core has", async () => {
    const { store } = await opened();
    const done = store.update({ contrast: manual(10) }, true);
    const pick = store.pickWhite(450, 300);
    await Promise.all([done, pick]);
    await store.undo();
    // The pick is undone; the contrast change before it is still there.
    expect(store.currentSettings()!.settings.white_pick).toBeNull();
    expect(store.currentSettings()!.settings.contrast).toEqual(manual(10));
  });
});

describe("settings_changed", () => {
  it("reads a photo's settings again when the core changed them by itself", async () => {
    const { api, store } = await opened();
    const id = store.currentPhoto()!.id;
    // The core sets the pick on its own (another window, a menu): the mirror asks.
    await api.updateSettings(id, { contrast: manual(25) });
    expect(store.currentSettings()!.values.contrast).toBe(0);
    api.emitEvent("settings_changed", { photo_id: id });
    await vi.waitFor(() => expect(store.currentSettings()!.values.contrast).toBe(25));
  });

  it("is ignored for a photo the screen has not asked about", async () => {
    const { api, store } = await opened();
    const spy = vi.spyOn(api, "getSettings");
    api.emitEvent("settings_changed", { photo_id: 9 });
    await Promise.resolve();
    await Promise.resolve();
    expect(spy).not.toHaveBeenCalled();
    expect(Object.keys(store.getState().settings)).toHaveLength(1);
  });
});

describe("export_busy", () => {
  it("follows the event, and the command after a reload", async () => {
    const { api, store } = await connected();
    expect(store.getState().exportBusy).toBe(false);
    api.emitEvent("export_busy", { busy: true });
    expect(store.getState().exportBusy).toBe(true);
    api.emitEvent("export_busy", { busy: false });
    expect(store.getState().exportBusy).toBe(false);

    // A window that was reloaded while an export runs: nothing was heard, the command says.
    const running = createMockApi({ speed: Number.POSITIVE_INFINITY });
    await running.openDialog("files");
    running.flush();
    await running.exportStart({ scope: "all", options: (await running.preferencesGet()).export_defaults });
    const reloaded = createStore(running, 0);
    stops.push(reloaded.connect());
    await reloaded.load();
    expect(reloaded.getState().exportBusy).toBe(true);
  });

  it("is asked again on demand, and says so", async () => {
    const { api, store } = await opened();
    await expect(store.refreshExportBusy()).resolves.toBe(false);
    await api.exportStart({ scope: "this", options: (await api.preferencesGet()).export_defaults });
    await expect(store.refreshExportBusy()).resolves.toBe(true);
    expect(store.getState().exportBusy).toBe(true);
  });

  it("keeps what it knew when the core can't say", async () => {
    const { api, store } = await connected();
    api.emitEvent("export_busy", { busy: true });
    api.failNext("exportBusy", { scope: "app", message: "Not now." });
    await expect(store.refreshExportBusy()).resolves.toBe(true);
    expect(store.getState().error).toBeNull();
  });

  it("survives closing the session", async () => {
    const { api, store } = await opened();
    api.emitEvent("export_busy", { busy: true });
    await store.closeSession();
    expect(store.getState().exportBusy).toBe(true);
  });
});
