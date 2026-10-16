/**
 * What redraws the Photos screen (docs/15 M3-18, M3-19). A frame drawn by the viewport, a wheel zoom or a tile finished by an export
 * must not redraw the screen and the hundreds of look tiles under it; a slider moving redraws what shows the settings and nothing more.
 * The counts are the ones `src/perf.ts` takes (`perfCount`), replaced here by counters of the test.
 */
import { act, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../api.mock";
import type { RenderStats } from "../api/types";
import { openPhotos, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";

const counts = vi.hoisted(() => new Map<string, number>());
vi.mock("../perf", async (importOriginal) => {
  const real = await importOriginal<typeof import("../perf")>();
  return { ...real, perfCount: (name: string): void => void counts.set(name, (counts.get(name) ?? 0) + 1) };
});

const n = (name: string): number => counts.get(name) ?? 0;

let h: Harness | undefined;
beforeEach(() => counts.clear());
afterEach(() => {
  vi.useRealTimers();
  h?.unmount();
  h = undefined;
});

const STATS: RenderStats = { ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null };
const zoomButton = (): HTMLElement => screen.getByRole("button", { name: /^(Fit|\d+%)/, description: /Zoom/ });
const tiles = (): number => document.querySelectorAll(".ym-look-tile").length;

/** The app on the Look tab with every library's looks, every tile's picture in, and the settle timer of the tiles held. */
async function setup(options: MockOptions = {}): Promise<Harness> {
  h = renderApp({ current: 3, picked: [2, 3, 4], allLibraries: true, ...options });
  await openPhotos(h);
  await waitFor(() => expect(document.querySelectorAll(".ym-look-tile img")).toHaveLength(tiles()));
  expect(tiles()).toBeGreaterThan(20);
  h.store.setViewportRect({ x: 0, y: 75, width: 1020, height: 680, scale: 2 });
  // Timers stand still from here, so the tiles' 400 ms settle only runs when a test says so.
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  return h;
}

/** Waits for the microtasks of the store's batches and the mock's answers. */
const quiet = (): Promise<void> =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(0);
    await Promise.resolve();
  });

describe("a frame drawn by the viewport", () => {
  it("redraws the status text and not the screen, the Look tab or its tiles", async () => {
    const { api } = await setup();
    await quiet();
    const before = { photos: n("Photos"), tab: n("LookTab"), tile: n("LazyLookTile"), notified: n("store.notify") };
    await act(async () => {
      for (let i = 0; i < 50; i++) api.emitEvent("render_stats", { ...STATS, ms: 100 + i });
    });
    await quiet();
    expect(screen.getByText(/149 ms/)).toBeInTheDocument();
    expect(n("store.notify")).toBeGreaterThan(before.notified);
    expect(n("Photos")).toBe(before.photos);
    expect(n("LookTab")).toBe(before.tab);
    expect(n("LazyLookTile")).toBe(before.tile);
  });
});

describe("a wheel zoom", () => {
  it("redraws the zoom button once per frame and not the screen, the Look tab or its tiles", async () => {
    const { store } = await setup();
    await quiet();
    const before = { photos: n("Photos"), tab: n("LookTab"), tile: n("LazyLookTile") };
    const label = zoomButton().textContent;
    for (let step = 0; step < 20; step++) {
      await act(async () => {
        store.queueViewportInput({ kind: "zoom", factor: 1.05, x: 300, y: 200 });
        await store.flushViewportInput();
      });
    }
    expect(zoomButton().textContent).not.toBe(label);
    expect(zoomButton()).toHaveTextContent(/^\d+%/);
    expect(n("Photos")).toBe(before.photos);
    expect(n("LookTab")).toBe(before.tab);
    expect(n("LazyLookTile")).toBe(before.tile);
  });
});

describe("an export that is running", () => {
  it("redraws the Export cell and not the screen", async () => {
    const { api, store } = await setup();
    await quiet();
    const names = ["DSC04203_neutral.jpg", "DSC04204_neutral.jpg", "DSC04205_neutral.jpg"];
    await act(async () => {
      await store.startExport(names, "~/x");
    });
    await quiet();
    const before = { photos: n("Photos"), tab: n("LookTab"), tile: n("LazyLookTile") };
    await act(async () => {
      api.flush();
      await vi.advanceTimersByTimeAsync(50);
    });
    await quiet();
    expect(store.getState().exportRun?.phase).toBe("finished");
    expect(screen.getByRole("button", { name: /^Exported/ })).toBeInTheDocument();
    expect(n("Photos")).toBe(before.photos);
    expect(n("LookTab")).toBe(before.tab);
    expect(n("LazyLookTile")).toBe(before.tile);
  });

  it("is not followed by the screen while the Export panel is closed: its options changing redraw nothing here", async () => {
    const { store } = await setup();
    await quiet();
    const before = n("Photos");
    await act(async () => store.setExportOptions({ jpeg_quality: 61 }, false));
    await quiet();
    expect(n("Photos")).toBe(before);
  });
});

describe("a slider that moves", () => {
  it("redraws the screen for the settings it shows, and the tiles not at all until the settings have rested for 400 ms", async () => {
    const { api, store } = await setup();
    await quiet();
    const asked = vi.spyOn(api, "lookThumbnail");
    const before = { photos: n("Photos"), tile: n("LazyLookTile") };
    await act(async () => {
      await store.update({ exposure: { kind: "manual", ev: 1.2 } }, false);
    });
    await quiet();
    expect(n("Photos")).toBeGreaterThan(before.photos);
    expect(n("LazyLookTile")).toBe(before.tile);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(399);
    });
    expect(n("LazyLookTile")).toBe(before.tile);
    expect(asked).not.toHaveBeenCalled();

    // At 400 ms the tiles are made again, for the tiles that are in view (all of them, with no layout in a test).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    await quiet();
    expect(n("LazyLookTile")).toBeGreaterThan(before.tile);
    expect(asked.mock.calls.length).toBe(tiles());
  });

  it("restarts the 400 ms with every move, so a drag asks for nothing until it rests", async () => {
    const { api, store } = await setup();
    await quiet();
    const asked = vi.spyOn(api, "lookThumbnail");
    for (let step = 1; step <= 10; step++) {
      await act(async () => {
        await store.update({ exposure: { kind: "manual", ev: step / 10 } }, false);
        await vi.advanceTimersByTimeAsync(300);
      });
    }
    expect(asked).not.toHaveBeenCalled();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(400);
    });
    await quiet();
    expect(asked.mock.calls.length).toBe(tiles());
  });
});

describe("another look", () => {
  it("redraws the tile that was the look and the tile that is, and no other", async () => {
    const { store } = await setup();
    await quiet();
    const before = n("LazyLookTile");
    await act(async () => {
      await store.chooseLook("builtin:signature");
    });
    await quiet();
    expect(store.currentSettings()?.values.look).toBe("builtin:signature");
    expect(n("LazyLookTile") - before).toBeLessThanOrEqual(2);
    expect(n("LazyLookTile")).toBeGreaterThan(before);
  });
});
