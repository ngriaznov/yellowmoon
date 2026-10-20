/**
 * The Look tab with the person moving to another photo while a look is being changed (the owner's report: choosing a look on one photo
 * sometimes changed another one). The Intensity slider collects a drag over frames; a tile click is answered by the core a moment
 * later; either may be overtaken by a move to the next photo, and neither may follow it there.
 */
import { act, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { capturingPointer } from "../../test/pointer";
import { installManualRaf, mockRect } from "../../test/raf";
import { openPhotos, renderApp, sliderOf } from "../../test/harness";
import type { Harness } from "../../test/harness";

let h: Harness;
afterEach(() => h?.unmount());

/** The twelve photos open with photo 4 on screen and photo 5's settings already known, so the move to it is instant. */
async function setup(): Promise<{ from: number; to: number }> {
  h = renderApp({ current: 3 });
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

describe("the Look tab while the person moves to another photo", () => {
  it("keeps an Intensity drag on the photo it began on: its frames and its release", async () => {
    const { from, to } = await setup();
    h.store.setApplyScope("this");
    const spy = vi.spyOn(h.api, "updateSettings");
    const raf = installManualRaf();
    try {
      const root = sliderOf("Intensity").closest('[data-slot="slider"]') as HTMLElement;
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
    expect(spy.mock.calls.at(-1)?.[1]).toEqual({ look_intensity: 75 });
    expect((await h.api.getSettings(to)).values.look_intensity).toBe(100);
    expect((await h.api.getSettings(from)).values.look_intensity).toBe(75);
  });

  it("puts a look picked from a tile on the photo whose tab it was, and on no other when Apply to is This photo", async () => {
    const { from, to } = await setup();
    h.store.setApplyScope("this");
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => (release = resolve));
    const real = h.api.updateSettings.bind(h.api);
    vi.spyOn(h.api, "updateSettings").mockImplementation(async (id, patch) => {
      await held;
      return real(id, patch);
    });
    const tile = await screen.findByRole("button", { name: /^Signature look/ });
    await h.user.click(tile);
    // Moved on before the core answered.
    await act(async () => {
      await h.store.setCurrent(to);
    });
    release();
    await waitFor(async () => expect((await h.api.getSettings(from)).values.look).toBe("builtin:signature"));
    expect((await h.api.getSettings(to)).values.look).toBe("builtin:neutral");
    await waitFor(() => expect(screen.getByRole("button", { name: /^Neutral look/ })).toHaveAttribute("aria-pressed", "true"));
    expect(screen.getByRole("button", { name: /^Signature look/ })).toHaveAttribute("aria-pressed", "false");
  });
});
