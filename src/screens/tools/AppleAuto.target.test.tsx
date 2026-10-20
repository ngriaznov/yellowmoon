/**
 * The controls Apple Auto sets (Highlights, Shadows, Black, the shadows and highlights wheels) with the person moving to another photo
 * while one is being dragged. They are Auto-or-manual settings and send through the same `store.update` as every other control, so a
 * drag, its frames and its release go to the photo it began on (`useGestureCallback`, `rs.photo_id`), and Reset (Auto) goes to the photo
 * whose settings the control was drawn from. Without this the Auto value of one photo, which is its own answer, could be sent to the
 * next one as a Manual value.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { capturingPointer } from "../../test/pointer";
import { installManualRaf, mockRect } from "../../test/raf";
import { openPhotos, openTab, renderApp, sliderOf, valueOf } from "../../test/harness";
import type { Harness } from "../../test/harness";

let h: Harness;
afterEach(() => h?.unmount());

/** The twelve photos open with Apple Auto's answers, photo 4 on screen and photo 5's settings already known, so the move to it is instant. */
async function setup(): Promise<{ from: number; to: number }> {
  h = renderApp({ current: 3, appleAuto: "answered" });
  await openPhotos(h);
  const from = h.store.currentPhoto()?.id ?? -1;
  await act(async () => {
    await h.store.step(1);
  });
  await waitFor(() => expect(h.store.currentSettings()).not.toBeNull());
  const to = h.store.currentPhoto()?.id ?? -1;
  await act(async () => {
    await h.store.setCurrent(from);
  });
  await waitFor(() => expect(h.store.currentSettings()?.photo_id).toBe(from));
  return { from, to };
}

describe("the controls Apple Auto sets while the person moves to another photo", () => {
  it("keeps a Highlights drag on the photo it began on: its frames and its release", async () => {
    const { from, to } = await setup();
    await openTab(h, "Light");
    expect(valueOf("Highlights")).toBe("−20");
    const spy = vi.spyOn(h.api, "updateSettings");
    const raf = installManualRaf();
    try {
      const root = sliderOf("Highlights").closest('[data-slot="slider"]') as HTMLElement;
      mockRect(root, { left: 100, top: 0, width: 200, height: 38 });
      const pointer = capturingPointer(root);
      pointer.down(200);
      pointer.move(230);
      // The next photo comes on screen before the frame is drawn.
      await act(async () => {
        await h.store.setCurrent(to);
      });
      pointer.move(250);
      await act(async () => {
        raf.flush();
      });
      pointer.up(250);
      await waitFor(() => expect(spy.mock.calls.length).toBeGreaterThanOrEqual(2));
    } finally {
      raf.restore();
    }
    expect(spy.mock.calls.map((c) => c[0])).toEqual(spy.mock.calls.map(() => from));
    expect(spy.mock.calls.at(-1)?.[1]).toEqual({ highlights: { kind: "manual", value: 50 } });
    // The photo moved to keeps its own Auto highlights; the one dragged has the manual value.
    expect((await h.api.getSettings(to)).settings.highlights).toEqual({ kind: "auto" });
    expect((await h.api.getSettings(from)).settings.highlights).toEqual({ kind: "manual", value: 50 });
  });

  it("keeps a shadows wheel drag on the photo it began on, and sends it as Manual there", async () => {
    const { from, to } = await setup();
    await openTab(h, "Colour");
    const spy = vi.spyOn(h.api, "updateSettings");
    const raf = installManualRaf();
    try {
      const disc = screen.getByRole("slider", { name: "Shadows colour balance" });
      mockRect(disc, { left: 0, top: 0, width: 92, height: 92 });
      fireEvent.pointerDown(disc, { button: 0, clientX: 46 + 10, clientY: 46 });
      fireEvent.pointerMove(window, { clientX: 46 + 20, clientY: 46 });
      await act(async () => {
        await h.store.setCurrent(to);
      });
      fireEvent.pointerMove(window, { clientX: 46 + 30, clientY: 46 });
      await act(async () => {
        raf.flush();
      });
      fireEvent.pointerUp(window);
      await waitFor(() => expect(spy.mock.calls.length).toBeGreaterThanOrEqual(2));
    } finally {
      raf.restore();
    }
    expect(spy.mock.calls.map((c) => c[0])).toEqual(spy.mock.calls.map(() => from));
    expect(spy.mock.calls.at(-1)?.[1].color_balance?.shadows).toMatchObject({ kind: "manual" });
    expect((await h.api.getSettings(to)).settings.color_balance.shadows).toEqual({ kind: "auto" });
    expect((await h.api.getSettings(from)).settings.color_balance.shadows).toMatchObject({ kind: "manual" });
  });

  it("sends Reset (Auto) to the photo whose controls it is on, and a typed value too", async () => {
    const { from } = await setup();
    await openTab(h, "Light");
    const spy = vi.spyOn(h.api, "updateSettings");
    const box = screen.getByRole("textbox", { name: "Black value" });
    await h.user.click(box);
    await h.user.clear(box);
    await h.user.type(box, "5{Enter}");
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    await h.user.click(await screen.findByRole("button", { name: "Reset Black" }));
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
    expect(spy.mock.calls.map((c) => [c[0], c[1]])).toEqual([
      [from, { black: { kind: "manual", value: 5 } }],
      [from, { black: { kind: "auto" } }],
    ]);
    // The card's Reset puts Highlights, Shadows and Black back to Auto, on the same photo.
    spy.mockClear();
    await h.user.click(within(screen.getByRole("region", { name: "High dynamic range" })).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(spy.mock.calls.at(-1)?.[0]).toBe(from);
    expect(spy.mock.calls.at(-1)?.[1]).toMatchObject({ highlights: { kind: "auto" }, shadows: { kind: "auto" }, black: { kind: "auto" } });
  });
});
