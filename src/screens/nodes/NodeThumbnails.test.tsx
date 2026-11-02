/**
 * The cards' pictures (`docs/23` §5.7, §7.2): each card shows the photo after its node and the Output, asked for from the core while the
 * node view is on. Until a node's first picture arrives its card shows the photo's thumbnail; a change to a node's grade gives new
 * pictures to the cards it reaches and leaves the ones before it alone. The screen runs on the mock.
 */
import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { NodeLink, SettingsPatch } from "../../api/types";
import { patchOf } from "../../state/settingsGraph";
import { openPhotos, renderApp } from "../../test/harness";
import type { Harness } from "../../test/harness";

let h: Harness | undefined;
let windowHeight = window.innerHeight;
beforeEach(() => {
  windowHeight = window.innerHeight;
  Object.defineProperty(window, "innerHeight", { value: 900, configurable: true, writable: true });
});
afterEach(() => {
  h?.unmount();
  h = undefined;
  vi.restoreAllMocks();
  Object.defineProperty(window, "innerHeight", { value: windowHeight, configurable: true, writable: true });
});

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

/** Nodes 2, 3 and 4 after Main: four cards. */
const FOUR: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Mountain", pos: [1, 0], trim: { exposure_ev: 0.4 } },
    "3": { kind: "corrector", name: "Sky", pos: [2, 0], trim: { temperature: 30 } },
    "4": { kind: "corrector", name: "Warmth", pos: [3, 0], trim: { exposure_ev: -0.3 } },
  },
  links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, 4), picture(4, "output")],
};

async function setup(): Promise<Harness> {
  h = renderApp({ current: 3, initialSettings: { DSC04204: FOUR } });
  await openPhotos(h);
  return h;
}

/** The picture of each card, by node. */
function pictures(): Record<number, string | null> {
  const out: Record<number, string | null> = {};
  for (const id of [1, 2, 3, 4]) {
    const card = h!.container.querySelector<HTMLElement>(`[data-node-card="${id}"]`);
    out[id] = card?.querySelector("img")?.getAttribute("src") ?? null;
  }
  return out;
}

async function openNodeView(): Promise<void> {
  await h!.user.click(within(screen.getByRole("navigation", { name: "Actions" })).getByRole("button", { name: "Nodes" }));
  await waitFor(() => expect(h!.container.querySelector('[data-node-card="4"]')).not.toBeNull());
}

describe("the pictures on the cards", () => {
  it("are the photo's thumbnail until a node's first picture arrives, then the node's own", async () => {
    await setup();
    // The core answers `node_thumbnails` and has not made them yet.
    const held = vi.spyOn(h!.api, "nodeThumbnails").mockResolvedValue(undefined);
    await openNodeView();
    const photo = h!.store.getState().thumbnails[4];
    expect(photo).toBeTruthy();
    expect(held).toHaveBeenCalledWith(4);
    expect(pictures()).toEqual({ 1: photo, 2: photo, 3: photo, 4: photo });

    // The pictures come: every card has its own, none the photo's.
    held.mockRestore();
    await h!.api.nodeThumbnails(4);
    await waitFor(() => {
      const now = pictures();
      expect(Object.values(now).every((src) => src && src !== photo)).toBe(true);
    });
    expect(new Set(Object.values(pictures())).size).toBe(4);
  });

  it("change after a node's grade changes: the node's card and the cards after it, not the ones before it", async () => {
    await setup();
    await openNodeView();
    await waitFor(() => expect(Object.values(pictures()).every((src) => src && src !== h!.store.getState().thumbnails[4])).toBe(true));
    const before = pictures();
    await h!.store.update(patchOf(3, { trim: { exposure_ev: 1.2 } }));
    await waitFor(() => {
      const now = pictures();
      expect(now[3]).not.toBe(before[3]);
      expect(now[4]).not.toBe(before[4]);
    });
    const after = pictures();
    expect(after[1]).toBe(before[1]);
    expect(after[2]).toBe(before[2]);
  });

  it("are not asked for in the simple view", async () => {
    await setup();
    const asked = vi.spyOn(h!.api, "nodeThumbnails");
    await h!.store.update(patchOf(3, { trim: { exposure_ev: 0.9 } }));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(asked).not.toHaveBeenCalled();
    // Opening the view asks, and closing it stops.
    await openNodeView();
    expect(asked).toHaveBeenCalledTimes(1);
    await h!.user.click(within(screen.getByRole("navigation", { name: "Actions" })).getByRole("button", { name: "Nodes" }));
    await waitFor(() => expect(h!.container.querySelector('[data-node-card="4"]')).toBeNull());
    asked.mockClear();
    await h!.store.update(patchOf(3, { trim: { exposure_ev: 0.1 } }));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(asked).not.toHaveBeenCalled();
  });
});
