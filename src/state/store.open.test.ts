/**
 * Opening many files (`docs/26` H-10, P3-3): the store says how far the core has read ("Reading {done} of {total} files…") once the
 * open has taken a moment, Cancel stops it and keeps the files read so far, and the line goes when the open ends.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import { createStore, OPEN_PROGRESS_DELAY_MS, stoppedNotice } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
  localStorage.clear();
  sessionStorage.clear();
});

async function started(slowOpen: number): Promise<{ store: AppStore; seen: Array<{ done: number; total: number } | null> }> {
  const api = createMockApi({ speed: 1, slowOpen });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  const seen: Array<{ done: number; total: number } | null> = [];
  stops.push(store.subscribe(() => seen.push(store.getState().openProgress)));
  return { store, seen };
}

describe("stoppedNotice", () => {
  it("says how many files were opened, in plain words", () => {
    expect(stoppedNotice(0)).toBe("Stopped. No files were opened.");
    expect(stoppedNotice(1)).toBe("Stopped. 1 file was opened.");
    expect(stoppedNotice(40)).toBe("Stopped. 40 files were opened.");
  });
});

describe("the line while files are read", () => {
  it("is not shown for an open that ends before anyone could read it", async () => {
    const { store, seen } = await started(100);
    await store.openDialog("files");
    expect(OPEN_PROGRESS_DELAY_MS).toBeGreaterThan(200);
    expect(seen.every((p) => p === null)).toBe(true);
    expect(store.getState().opening).toBe(false);
  });

  it("shows done and total once the open has taken a moment, and goes when it ends", async () => {
    const { store } = await started(2000);
    const opening = store.openDialog("files");
    await vi.waitFor(() => expect(store.getState().openProgress).not.toBeNull(), { timeout: 5000 });
    const shown = store.getState().openProgress;
    expect(shown?.total).toBe(2000);
    expect(shown?.done).toBeGreaterThanOrEqual(0);
    await store.cancelOpen();
    await opening;
    expect(store.getState().openProgress).toBeNull();
    expect(store.getState().opening).toBe(false);
  });

  it("keeps the files read before Cancel and says how many (Start)", async () => {
    const { store } = await started(2000);
    const opening = store.openDialog("files");
    await vi.waitFor(() => expect(store.getState().openProgress).not.toBeNull(), { timeout: 5000 });
    await store.cancelOpen();
    await opening;
    const photos = store.getState().session?.photos ?? [];
    expect(photos.length).toBeGreaterThan(0);
    expect(store.getState().dropNotice).toBe(stoppedNotice(photos.length));
  });

  it("keeps nothing when Cancel comes before the first file was read", async () => {
    const { store } = await started(2000);
    const opening = store.openDialog("files");
    // The mock reports the start, 0 of 2000, at once; the first 50 files take 100 ms.
    await vi.waitFor(() => expect(store.getState().opening).toBe(true));
    await store.cancelOpen();
    await opening;
    expect(store.getState().session?.photos ?? []).toHaveLength(0);
    expect(store.getState().dropNotice).toBe("Stopped. No files were opened.");
  });
});
