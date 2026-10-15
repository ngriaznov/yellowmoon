/**
 * The photo cell on the Photos screen under each viewport backend (M3-04 frontend side): the browser build draws the mock's picture, the
 * native viewport leaves the cell transparent, the fallback viewport shows the shell's pictures; in every one the cell reports its
 * rectangle and forwards the pointer.
 */
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mockRect } from "../test/raf";
import { openPhotos, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";
import type { MockOptions } from "../api.mock";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
  vi.unstubAllGlobals();
});

async function setup(options: MockOptions = {}): Promise<HTMLElement> {
  h = renderApp({ current: 3, ...options });
  await openPhotos(h);
  return screen.getByRole("main", { name: "Photo" });
}

const frameEvent = (seq: number) => ({ seq, width: 2040, height: 1360, sharp: true, url: `ym-frame://localhost/frame-${seq}.png` });

describe("without a viewport backend (the browser build)", () => {
  it("draws the mock's picture, on paper", async () => {
    const cell = await setup();
    expect(cell.querySelector("img.ym-photo-cell__image")).not.toBeNull();
    expect(cell).not.toHaveAttribute("data-native");
    expect(cell).not.toHaveAttribute("data-viewport-backend");
  });
});

describe("the native viewport", () => {
  it("leaves the cell transparent and draws nothing", async () => {
    const cell = await setup({ viewport: "native" });
    await waitFor(() => expect(cell).toHaveAttribute("data-native", "true"));
    expect(cell).toHaveAttribute("data-viewport-backend", "native");
    expect(cell.querySelector("img")).toBeNull();
    // Pictures the core sends in the other mode are not shown.
    act(() => h!.api.emitEvent("viewport_frame", frameEvent(1)));
    expect(cell.querySelector("img")).toBeNull();
  });

  it("lets the layer behind the page show through: the root says so while the photo shows, and nothing between the cell and it paints", async () => {
    const cell = await setup({ viewport: "native" });
    await waitFor(() => expect(document.documentElement).toHaveAttribute("data-native-viewport"));
    // screens.css makes the root, the body, #root and the Photos screen transparent; a new box between the cell and the screen would
    // paint paper over the photo.
    expect(cell.parentElement).toHaveAttribute("data-screen", "photos");
    h!.unmount();
    h = undefined;
    expect(document.documentElement).not.toHaveAttribute("data-native-viewport");
  });

  it("still carries the look chip and the camera line over the transparent cell", async () => {
    const cell = await setup({ viewport: "native" });
    expect(cell.querySelector(".ym-photo-cell__chip")).toHaveTextContent("Neutral · 100%");
    expect(cell.querySelector(".ym-photo-cell__camera")).toHaveTextContent("Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8");
  });
});

describe("the fallback viewport", () => {
  it("shows no picture until the shell sends one, then each newer frame, then none when the shell takes it away", async () => {
    const cell = await setup({ viewport: "fallback" });
    await waitFor(() => expect(cell).toHaveAttribute("data-viewport-backend", "fallback"));
    expect(cell).not.toHaveAttribute("data-native");
    // The page stays opaque: the pictures are in it.
    expect(document.documentElement).not.toHaveAttribute("data-native-viewport");
    // The mock sends the photo's frame too; use the shell's own address from here on.
    act(() => h!.api.emitEvent("viewport_frame", frameEvent(100)));
    const img = (): HTMLImageElement | null => cell.querySelector("img.ym-photo-cell__frame");
    expect(img()).toHaveAttribute("src", "ym-frame://localhost/frame-100.png");
    act(() => h!.api.emitEvent("viewport_frame", frameEvent(101)));
    expect(img()).toHaveAttribute("src", "ym-frame://localhost/frame-101.png");
    expect(cell.querySelectorAll("img")).toHaveLength(1);
    act(() => h!.api.emitEvent("viewport_frame", { seq: 102, width: 0, height: 0, sharp: true, url: "" }));
    expect(img()).toBeNull();
  });

  it("draws the mock's own frames of the photo as they come", async () => {
    const cell = await setup({ viewport: "fallback" });
    await h!.api.viewportSetRect({ x: 0, y: 75, width: 1020, height: 680, scale: 2 });
    await waitFor(() => expect(cell.querySelector("img.ym-photo-cell__frame")).not.toBeNull());
    const first = cell.querySelector("img.ym-photo-cell__frame")?.getAttribute("src");
    await h!.store.update({ exposure: { kind: "manual", ev: 1.5 } });
    await waitFor(() => expect(cell.querySelector("img.ym-photo-cell__frame")?.getAttribute("src")).not.toBe(first));
  });

  it("covers a picture that is not ready with the note, and shows the frame again when it is", async () => {
    const cell = await setup({ viewport: "fallback", skipped: { DSC04202: "This file is from a camera mode we can't read yet." } });
    act(() => h!.api.emitEvent("viewport_frame", frameEvent(1)));
    expect(cell.querySelector("img.ym-photo-cell__frame")).not.toBeNull();
    await h!.store.setCurrent(2);
    await waitFor(() => expect(screen.getByText("This file can't be opened.")).toBeInTheDocument());
    expect(cell.querySelector("img.ym-photo-cell__frame")).toBeNull();
    await h!.store.setCurrent(3);
    act(() => h!.api.emitEvent("viewport_frame", frameEvent(2)));
    await waitFor(() => expect(cell.querySelector("img.ym-photo-cell__frame")).not.toBeNull());
    expect(screen.queryByText("This file can't be opened.")).toBeNull();
  });
});

describe.each([undefined, "native", "fallback"] as const)("the pointer over the photo (%s)", (viewport) => {
  it("reports the cell's rectangle to the core when it is placed", async () => {
    h = renderApp({ current: 3, ...(viewport ? { viewport } : {}) });
    const placed = vi.spyOn(h.api, "viewportSetRect");
    await openPhotos(h);
    expect(placed).toHaveBeenCalled();
    expect(placed.mock.calls.at(-1)?.[0]).toMatchObject({ x: 0, y: 0, scale: 1 });
    await waitFor(() => expect(h!.api.viewport.rect).not.toBeNull());
  });

  it("sends a double click as toggle_zoom at the point in the cell, and mirrors the zoom", async () => {
    const cell = await setup(viewport ? { viewport } : {});
    mockRect(cell, { left: 190, top: 75, width: 900, height: 600 });
    fireEvent.doubleClick(cell, { clientX: 640, clientY: 375 });
    await waitFor(() => expect(h!.api.viewport.inputs).toContainEqual({ kind: "toggle_zoom", x: 450, y: 300 }));
    expect(h!.store.getState().zoom).toEqual({ kind: "percent", value: 100 });
  });

  it("zooms with ⌘ and the wheel about the pointer, and pans with the wheel once zoomed", async () => {
    const cell = await setup(viewport ? { viewport } : {});
    mockRect(cell, { left: 0, top: 0, width: 900, height: 600 });
    fireEvent.wheel(cell, { deltaY: 40, clientX: 100, clientY: 50 });
    expect(h!.api.viewport.inputs).toEqual([]);
    fireEvent.wheel(cell, { deltaY: -100, ctrlKey: true, clientX: 100, clientY: 50 });
    await waitFor(() => expect(h!.api.viewport.inputs).toHaveLength(1));
    const zoom = h!.api.viewport.inputs[0]!;
    expect(zoom).toMatchObject({ kind: "zoom", x: 100, y: 50 });
    expect(zoom.kind === "zoom" && zoom.factor).toBeCloseTo(Math.exp(1), 6);
    // The mirror followed the core, so the wheel pans now.
    expect(h!.store.getState().zoom.kind).toBe("percent");
    fireEvent.wheel(cell, { deltaX: 3, deltaY: 12, clientX: 100, clientY: 50 });
    await waitFor(() => expect(h!.api.viewport.inputs).toHaveLength(2));
    expect(h!.api.viewport.inputs[1]).toEqual({ kind: "pan", dx: -3, dy: -12 });
  });

  it("pans by dragging once zoomed, but not at Fit", async () => {
    const cell = await setup(viewport ? { viewport } : {});
    mockRect(cell, { left: 0, top: 0, width: 900, height: 600 });
    fireEvent.pointerDown(cell, { clientX: 10, clientY: 10, button: 0, pointerId: 1 });
    fireEvent.pointerMove(cell, { clientX: 30, clientY: 30, pointerId: 1 });
    expect(h!.api.viewport.inputs).toEqual([]);
    await h!.store.setZoom({ kind: "percent", value: 100 });
    fireEvent.pointerDown(cell, { clientX: 10, clientY: 10, button: 0, pointerId: 1 });
    fireEvent.pointerMove(cell, { clientX: 30, clientY: 25, pointerId: 1 });
    await waitFor(() => expect(h!.api.viewport.inputs).toContainEqual({ kind: "pan", dx: 20, dy: 15 }));
  });

  it("reads the pixel under the pointer with ⌥ held, in CSS pixels of the cell, and drops the readout on the paper", async () => {
    const cell = await setup(viewport ? { viewport } : {});
    mockRect(cell, { left: 0, top: 0, width: 900, height: 600 });
    h!.store.setViewportRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    await waitFor(() => expect(h!.api.viewport.rect?.width).toBe(900));
    const sample = vi.spyOn(h!.api, "samplePixel");
    fireEvent.pointerMove(cell, { clientX: 450, clientY: 60, altKey: true });
    await waitFor(() => expect(cell.querySelector(".ym-photo-cell__readout")).toHaveTextContent(/^R \d+ G \d+ B \d+$/));
    expect(sample).toHaveBeenCalledWith(450, 60);
    // A portrait photo is letterboxed: the left edge is paper and has nothing to read.
    await h!.store.update({ orientation: 90 });
    fireEvent.pointerMove(cell, { clientX: 5, clientY: 300, altKey: true });
    await waitFor(() => expect(cell.querySelector(".ym-photo-cell__readout")).toBeNull());
    expect(h!.store.getState().error).toBeNull();
  });
});

describe("the zoom next to the Fit zoom", () => {
  /** A cell big enough that Fit is 56 % for the mock's 8640 px photos: 2400 CSS px at 2×. */
  async function bigCell(): Promise<HTMLElement> {
    vi.stubGlobal("devicePixelRatio", 2);
    const cell = await setup({ viewport: "native" });
    mockRect(cell, { left: 0, top: 75, width: 2400, height: 1600 });
    fireEvent(window, new Event("resize"));
    await waitFor(() => expect(h!.store.fitPercent()).toBe(56));
    return cell;
  }
  const zoomButton = (): HTMLElement => screen.getByRole("button", { name: /^(Fit|\d+%)/, description: /Zoom/ });

  it("says Fit, not 50 %, when the menu's 50 % is no larger than the Fit zoom (the core draws Fit then)", async () => {
    await bigCell();
    expect(zoomButton()).toHaveTextContent("Fit 56%");
    await act(async () => h!.store.setZoom({ kind: "percent", value: 50 }));
    expect(zoomButton()).toHaveTextContent("Fit 56%");
    expect(h!.store.showsWholePhoto()).toBe(true);
    await act(async () => h!.store.setZoom({ kind: "percent", value: 100 }));
    expect(zoomButton()).toHaveTextContent("100%");
    expect(h!.store.showsWholePhoto()).toBe(false);
  });

  it("Zoom In from Fit goes to 100 % where 50 % would only show Fit again, and Z and a double click flip Fit and 100 %", async () => {
    const cell = await bigCell();
    await h!.user.keyboard("{Meta>}={/Meta}");
    await waitFor(() => expect(zoomButton()).toHaveTextContent("100%"));
    await h!.user.keyboard("{Meta>}-{/Meta}");
    await waitFor(() => expect(zoomButton()).toHaveTextContent("Fit 56%"));
    await h!.user.keyboard("z");
    await waitFor(() => expect(zoomButton()).toHaveTextContent("100%"));
    await h!.user.keyboard("z");
    await waitFor(() => expect(zoomButton()).toHaveTextContent("Fit 56%"));
    fireEvent.doubleClick(cell, { clientX: 300, clientY: 300 });
    await waitFor(() => expect(zoomButton()).toHaveTextContent("100%"));
  });

  it("Zoom In and Zoom Out after a wheel zoom step from where the photo is, not from the start of the list", async () => {
    vi.stubGlobal("devicePixelRatio", 2);
    const cell = await setup({ viewport: "native" });
    mockRect(cell, { left: 0, top: 75, width: 1020, height: 680 });
    fireEvent(window, new Event("resize"));
    await waitFor(() => expect(h!.store.fitPercent()).toBe(24));
    // Fit is 23.6 %: a wheel zoom of 5.8 lands on about 137 %.
    await h!.store.viewportInput({ kind: "zoom", factor: 5.8, x: 500, y: 300 });
    expect(zoomButton()).toHaveTextContent("137%");
    await h!.user.keyboard("{Meta>}={/Meta}");
    await waitFor(() => expect(zoomButton()).toHaveTextContent("200%"));
    await h!.store.viewportInput({ kind: "zoom", factor: 0.685, x: 500, y: 300 });
    await h!.user.keyboard("{Meta>}-{/Meta}");
    await waitFor(() => expect(zoomButton()).toHaveTextContent("100%"));
  });

  it("does not pan by dragging while a fixed zoom below the Fit zoom shows the whole photo", async () => {
    const cell = await bigCell();
    await h!.store.setZoom({ kind: "percent", value: 50 });
    fireEvent.pointerDown(cell, { clientX: 10, clientY: 10, button: 0, pointerId: 1 });
    fireEvent.pointerMove(cell, { clientX: 30, clientY: 25, pointerId: 1 });
    expect(h!.api.viewport.inputs).toEqual([]);
  });
});

describe("the pixel readout", () => {
  it("reads where the pointer is when the reading before it comes back, and not after ⌥ is released", async () => {
    const cell = await setup();
    mockRect(cell, { left: 0, top: 0, width: 900, height: 600 });
    h!.store.setViewportRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    // The core is slow: each reading waits until the test lets it go.
    const pending: Array<{ at: [number, number]; answer: (v: { r: number; g: number; b: number }) => void }> = [];
    const sample = vi.spyOn(h!.api, "samplePixel").mockImplementation(
      (x, y) => new Promise((resolve) => pending.push({ at: [x, y], answer: resolve })),
    );
    fireEvent.pointerMove(cell, { clientX: 100, clientY: 100, altKey: true });
    await waitFor(() => expect(pending).toHaveLength(1));
    // The pointer moves on twice while the first reading is out; only the last place is asked for.
    fireEvent.pointerMove(cell, { clientX: 200, clientY: 150, altKey: true });
    fireEvent.pointerMove(cell, { clientX: 300, clientY: 200, altKey: true });
    expect(sample).toHaveBeenCalledTimes(1);
    await act(async () => pending[0]!.answer({ r: 1, g: 2, b: 3 }));
    await waitFor(() => expect(pending).toHaveLength(2));
    expect(pending[1]!.at).toEqual([300, 200]);
    await act(async () => pending[1]!.answer({ r: 4, g: 5, b: 6 }));
    await waitFor(() => expect(cell.querySelector(".ym-photo-cell__readout")).toHaveTextContent("R 4 G 5 B 6"));

    // ⌥ is released while a reading is out: its answer must not bring the readout back.
    fireEvent.pointerMove(cell, { clientX: 400, clientY: 250, altKey: true });
    await waitFor(() => expect(pending).toHaveLength(3));
    fireEvent.pointerMove(cell, { clientX: 400, clientY: 250, altKey: false });
    await waitFor(() => expect(cell.querySelector(".ym-photo-cell__readout")).toBeNull());
    await act(async () => pending[2]!.answer({ r: 7, g: 8, b: 9 }));
    expect(cell.querySelector(".ym-photo-cell__readout")).toBeNull();
    expect(sample).toHaveBeenCalledTimes(3);
  });
});
