/**
 * The look tiles of the Look tab with a real-looking IntersectionObserver (docs/15 M3-19): one observer for the whole grid, a tile asks
 * for its picture only while it is in view, keeps the old picture when it leaves, and asks again only when it is back and the settings
 * changed.
 */
import { act, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import { openPhotos, renderApp } from "../../test/harness";
import type { Harness } from "../../test/harness";
import { createTileWatch } from "./LookTab";

/** An IntersectionObserver the test drives. */
class FakeObserver {
  static all: FakeObserver[] = [];
  targets = new Set<Element>();
  disconnected = false;
  constructor(private readonly callback: IntersectionObserverCallback) {
    FakeObserver.all.push(this);
  }
  observe(el: Element): void {
    this.targets.add(el);
  }
  unobserve(el: Element): void {
    this.targets.delete(el);
  }
  disconnect(): void {
    this.targets.clear();
    this.disconnected = true;
  }
  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }
  /** Reports `inView` for each element. */
  report(changes: Array<[Element, boolean]>): void {
    this.callback(
      changes.map(([target, isIntersecting]) => ({ target, isIntersecting }) as IntersectionObserverEntry),
      this as unknown as IntersectionObserver,
    );
  }
}

let h: Harness | undefined;
beforeEach(() => {
  FakeObserver.all = [];
  vi.stubGlobal("IntersectionObserver", FakeObserver);
});
afterEach(() => {
  vi.useRealTimers();
  h?.unmount();
  h = undefined;
  vi.unstubAllGlobals();
});

describe("createTileWatch", () => {
  it("makes one observer for any number of elements, and tells each one both ways", () => {
    const watch = createTileWatch();
    const [a, b, c] = [document.createElement("div"), document.createElement("div"), document.createElement("div")] as const;
    const seen: Record<string, boolean[]> = { a: [], b: [], c: [] };
    watch.observe(a, (v) => seen.a!.push(v));
    watch.observe(b, (v) => seen.b!.push(v));
    const stopC = watch.observe(c, (v) => seen.c!.push(v));
    expect(FakeObserver.all).toHaveLength(1);
    const [observer] = FakeObserver.all as [FakeObserver];
    expect(observer.targets.size).toBe(3);
    observer.report([[a, true], [b, false]]);
    observer.report([[a, false], [b, true], [c, true]]);
    expect(seen).toEqual({ a: [true, false], b: [false, true], c: [true] });
    stopC();
    expect(observer.targets.has(c)).toBe(false);
    observer.report([[c, false]]);
    expect(seen.c).toEqual([true]);
  });

  it("starts again after disconnect, and calls back at once where there is no IntersectionObserver", () => {
    const watch = createTileWatch();
    watch.observe(document.createElement("div"), () => undefined);
    watch.disconnect();
    expect(FakeObserver.all[0]?.disconnected).toBe(true);
    watch.observe(document.createElement("div"), () => undefined);
    expect(FakeObserver.all).toHaveLength(2);

    vi.stubGlobal("IntersectionObserver", undefined);
    const seen: boolean[] = [];
    createTileWatch().observe(document.createElement("div"), (v) => seen.push(v));
    expect(seen).toEqual([true]);
  });
});

const ALL = { current: 3, picked: [2, 3, 4], allLibraries: true } satisfies MockOptions;

async function setup(): Promise<Harness> {
  h = renderApp(ALL);
  await openPhotos(h);
  return h;
}

const tileEls = (): HTMLElement[] => [...document.querySelectorAll<HTMLElement>(".ym-look-tile")];
const observer = (): FakeObserver => {
  const found = FakeObserver.all.at(-1);
  if (!found) throw new Error("no observer was made");
  return found;
};
const pictureOf = (tile: HTMLElement): string | null => tile.querySelector("img")?.getAttribute("src") ?? null;

describe("the tiles of the Look tab", () => {
  it("share one observer, and ask for nothing until they are in view", async () => {
    const { api } = await setup();
    const ask = vi.spyOn(api, "lookThumbnail");
    expect(FakeObserver.all).toHaveLength(1);
    expect(observer().targets.size).toBe(tileEls().length);
    expect(tileEls().length).toBeGreaterThan(20);
    expect(ask).not.toHaveBeenCalled();
    expect(document.querySelectorAll(".ym-look-tile img")).toHaveLength(0);

    act(() => observer().report(tileEls().slice(0, 3).map((t): [Element, boolean] => [t, true])));
    await waitFor(() => expect(document.querySelectorAll(".ym-look-tile img")).toHaveLength(3));
    expect(ask).toHaveBeenCalledTimes(3);
  });

  it("do not observe the tiles of a section that is folded, and the observer lets go when the tab goes", async () => {
    const { user } = await setup();
    const all = tileEls().length;
    await user.click(screen.getByRole("button", { name: /^Built in, 2 looks/ }));
    await waitFor(() => expect(observer().targets.size).toBe(all - 2));
    await user.click(screen.getByRole("tab", { name: "Light" }));
    await waitFor(() => expect(observer().disconnected).toBe(true));
  });

  it("refresh only while in view: one that left keeps its old picture, and asks again when it is back with other settings", async () => {
    const { api, store } = await setup();
    const [first, second, third] = tileEls() as [HTMLElement, HTMLElement, HTMLElement];
    act(() => observer().report([[first, true], [second, true], [third, true]]));
    await waitFor(() => expect(document.querySelectorAll(".ym-look-tile img")).toHaveLength(3));
    const oldFirst = pictureOf(first);
    const oldThird = pictureOf(third);

    // The first tile scrolls out of view; the settings change and rest.
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    act(() => observer().report([[first, false]]));
    const ask = vi.spyOn(api, "lookThumbnail");
    await act(async () => {
      await store.update({ exposure: { kind: "manual", ev: 1.1 } });
      await vi.advanceTimersByTimeAsync(400);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Only the two in view were made again; the one out of view kept the picture it had.
    expect(ask.mock.calls.map((c) => c[1]).sort()).toEqual([lookOf(second), lookOf(third)].sort());
    expect(pictureOf(first)).toBe(oldFirst);
    expect(pictureOf(third)).not.toBe(oldThird);

    // It comes back: now it asks, for the settings that are.
    act(() => observer().report([[first, true]]));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(ask.mock.calls.map((c) => c[1])).toContain(lookOf(first));
    expect(pictureOf(first)).not.toBe(oldFirst);
  });

  it("leave a tile that went out of view and came back with the same settings alone", async () => {
    const { api } = await setup();
    const [first] = tileEls() as [HTMLElement];
    act(() => observer().report([[first, true]]));
    await waitFor(() => expect(document.querySelectorAll(".ym-look-tile img")).toHaveLength(1));
    const picture = pictureOf(first);
    const ask = vi.spyOn(api, "lookThumbnail");
    act(() => observer().report([[first, false]]));
    act(() => observer().report([[first, true]]));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(ask).not.toHaveBeenCalled();
    expect(pictureOf(first)).toBe(picture);
  });
});

/** The id of the look a tile shows: the tiles are in display order, as `looksInDisplayOrder` has them. */
function lookOf(tile: HTMLElement): string {
  const index = tileEls().indexOf(tile);
  const ordered = h?.store.getState().looks ?? [];
  // The grid shows the libraries in order and each in its own order; the first section holds the first looks.
  const ids = tileEls().map((t) => t.getAttribute("aria-label") ?? "");
  const name = ids[index]?.replace(/ look(, key \d)?(,.*)?$/, "") ?? "";
  const found = ordered.find((l) => l.name === name);
  if (!found) throw new Error(`No look is called ${name}`);
  return found.id;
}
