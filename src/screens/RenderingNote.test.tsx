/**
 * "Rendering on the CPU" (`docs/11` §Photos, the photo cell): shown once the core has said for 250 ms that the viewport renders on the
 * CPU (`viewport_busy` with `busy` and `cpu`), gone the moment it is not busy, and never for a frame on the graphics card. Also where it
 * sits in the photo cell of the Photos screen: outside the picture, and in Before too.
 */
import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi } from "../api.mock";
import { StoreProvider } from "../state/context";
import { createStore } from "../state/store";
import { openPhotos, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";
import { RENDERING_NOTE_DELAY_MS, RENDERING_NOTE_TEXT, RenderingNote } from "./RenderingNote";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
  vi.useRealTimers();
});

/** The note alone, over a store on the mock, with the clock in the test's hands. */
function note(): { api: MockApi; unmount: () => void } {
  vi.useFakeTimers();
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
  const store = createStore(api, 0);
  const view = render(
    <StoreProvider store={store}>
      <RenderingNote />
    </StoreProvider>,
  );
  return { api, unmount: view.unmount };
}

const shown = (): HTMLElement | null => screen.queryByRole("status");
const busy = (api: MockApi, value: boolean, cpu: boolean): void => act(() => api.emitEvent("viewport_busy", { busy: value, cpu }));
const wait = (ms: number): void =>
  act(() => {
    vi.advanceTimersByTime(ms);
  });

describe("RenderingNote", () => {
  it("waits 250 ms before it shows, then says so politely with a spinner", () => {
    const { api, unmount } = note();
    expect(RENDERING_NOTE_DELAY_MS).toBe(250);
    busy(api, true, true);
    expect(shown()).toBeNull();
    wait(249);
    expect(shown()).toBeNull();
    wait(1);
    const status = shown();
    expect(status).toHaveTextContent(RENDERING_NOTE_TEXT);
    expect(RENDERING_NOTE_TEXT).toBe("Rendering on the CPU");
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status).toHaveClass("ym-photo-cell__busy");
    // The spinner is decoration: the words are the status, and there is only one.
    const spinner = status?.querySelector('[data-slot="spinner"]');
    expect(spinner).toHaveAttribute("aria-hidden", "true");
    expect(spinner).not.toHaveAttribute("role");
    expect(screen.getAllByRole("status")).toHaveLength(1);
    unmount();
  });

  it("goes the moment the core is not busy, and waits again for the next frame", () => {
    const { api, unmount } = note();
    busy(api, true, true);
    wait(400);
    expect(shown()).not.toBeNull();
    busy(api, false, true);
    expect(shown()).toBeNull();
    busy(api, true, true);
    wait(200);
    expect(shown()).toBeNull();
    wait(50);
    expect(shown()).not.toBeNull();
    unmount();
  });

  it("never flashes for a frame shorter than 250 ms", () => {
    const { api, unmount } = note();
    for (let i = 0; i < 5; i++) {
      busy(api, true, true);
      wait(200);
      busy(api, false, true);
      wait(100);
    }
    wait(1000);
    expect(shown()).toBeNull();
    unmount();
  });

  it("is not shown while the graphics card renders, however long it takes", () => {
    const { api, unmount } = note();
    busy(api, true, false);
    wait(5000);
    expect(shown()).toBeNull();
    // The route moves to the CPU in the middle of the work (a device error): the wait starts then.
    busy(api, true, true);
    wait(249);
    expect(shown()).toBeNull();
    wait(1);
    expect(shown()).not.toBeNull();
    // And back on the device: gone at once.
    busy(api, true, false);
    expect(shown()).toBeNull();
    unmount();
  });
});

describe("in the photo cell", () => {
  // That it takes no pointer events is checked in a browser (`e2e/cpu-indicator.spec.ts`): Vitest loads no CSS.
  it("sits outside the picture and shows in Before too", async () => {
    h = renderApp({ current: 3 });
    await openPhotos(h);
    const cell = screen.getByRole("main", { name: "Photo" });
    expect(shown()).toBeNull();
    act(() => h!.api.emitEvent("viewport_busy", { busy: true, cpu: true }));
    const status = await screen.findByRole("status", {}, { timeout: 2000 });
    expect(status).toHaveTextContent("Rendering on the CPU");
    expect(cell).toContainElement(status);
    // Not inside the `role="img"` view, whose content a screen reader does not read out.
    expect(cell.querySelector('[role="img"]')).not.toContainElement(status);
    await h.user.click(screen.getByRole("button", { name: "Before" }));
    await waitFor(() => expect(cell.querySelector(".ym-photo-cell__before")).not.toBeNull());
    expect(screen.getByRole("status")).toHaveTextContent("Rendering on the CPU");
    act(() => h!.api.emitEvent("viewport_busy", { busy: false, cpu: true }));
    expect(shown()).toBeNull();
  });
});
