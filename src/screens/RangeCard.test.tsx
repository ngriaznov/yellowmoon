/**
 * The RAW tab's Range card (owner's decision of 2026-10-17, `docs/02` C7, `docs/11`): the card that was Roll-off, with the same
 * two sliders on the same `rolloff` settings. Highlights is the highlight range (0 keeps the brightest tones where the camera
 * recorded them, 1 takes them to white); Low light is the toe near black. A note under the sliders says so in plain words.
 */
import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SettingsPatch } from "../api/types";
import { capturingPointer } from "../test/pointer";
import { installManualRaf, mockRect } from "../test/raf";
import { openPhotos, openTab, renderApp, sliderName, sliderOf, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";

let h: Harness;
afterEach(() => h?.unmount());

/** `docs/11` §Photos, "Range card", word for word. */
const NOTE =
  "Highlights moves the brightest tones the camera recorded: 0 keeps them where they were, 1 takes them to white. Low light softens the noise near black.";

async function setup(): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4] });
  await openPhotos(h);
  await openTab(h, "RAW");
  return vi.spyOn(h.api, "updateSettings");
}

/** Types a value into a slider's box and presses Return. */
async function type(label: string, text: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: `${label} value` });
  await h.user.click(box);
  await h.user.clear(box);
  await h.user.type(box, `${text}{Enter}`);
}

/** The patch of the newest `update_settings` call, once there is one. */
async function lastPatch(spy: UpdateSpy): Promise<SettingsPatch> {
  await waitFor(() => expect(spy).toHaveBeenCalled());
  return spy.mock.calls.at(-1)?.[1] as SettingsPatch;
}

const range = (): ReturnType<typeof within> => within(screen.getByRole("region", { name: "Range" }));

describe("the Range card on the RAW tab", () => {
  it("is the third card, after Camera and Decode, and no card is called Roll-off any more", async () => {
    await setup();
    const regions = screen.getAllByRole("region");
    expect(regions.map((r) => within(r).getAllByRole("heading", { level: 3 })[0]?.textContent)).toEqual(["Camera", "Decode", "Range"]);
    expect(screen.queryByRole("region", { name: "Roll-off" })).toBeNull();
  });

  it("has Highlights and Low light, 0 to 1, at 1.00 and 0.50 out of the box", async () => {
    await setup();
    const sliders = range().getAllByRole("slider");
    expect(sliders.map((s: HTMLElement) => sliderName(s))).toEqual(["Highlights", "Low light"]);
    for (const label of ["Highlights", "Low light"]) {
      const slider = sliderOf(label);
      expect([slider.getAttribute("aria-valuemin"), slider.getAttribute("aria-valuemax")], label).toEqual(["0", "1"]);
    }
    expect(valueOf("Highlights")).toBe("1.00");
    expect(valueOf("Low light")).toBe("0.50");
  });

  it("says under the sliders what they do, in the words of docs/11", async () => {
    await setup();
    const note = range().getByText(NOTE);
    expect(note).toBeInTheDocument();
    // Under the two sliders, not between them.
    const sliders = range().getAllByRole("slider");
    for (const slider of sliders) {
      expect(slider.compareDocumentPosition(note) & Node.DOCUMENT_POSITION_FOLLOWING, sliderName(slider)).toBeTruthy();
    }
    // The labels and the note name no camera maker (AGENTS rule 3).
    expect(screen.getByRole("region", { name: "Range" }).textContent).not.toMatch(/ARRI|ALEXA/i);
  });

  it("a typed value is a rolloff patch that keeps the other slider, and Reset sends them back to 1 and 0.5", async () => {
    const spy = await setup();
    await type("Highlights", "0");
    expect(await lastPatch(spy)).toEqual({ rolloff: { highlights: 0, low_light: 0.5 } });
    await waitFor(() => expect(valueOf("Highlights")).toBe("0.00"));
    spy.mockClear();
    await type("Low light", "0.8");
    expect(await lastPatch(spy)).toEqual({ rolloff: { highlights: 0, low_light: 0.8 } });
    await waitFor(() => expect(valueOf("Low light")).toBe("0.80"));
    expect(h.store.currentSettings()?.settings.rolloff).toEqual({ highlights: 0, low_light: 0.8 });

    spy.mockClear();
    await h.user.click(range().getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy)).toEqual({ rolloff: { highlights: 1, low_light: 0.5 } });
    await waitFor(() => expect(valueOf("Highlights")).toBe("1.00"));
    expect(valueOf("Low light")).toBe("0.50");
  });

  it("a drag of Highlights sends one rolloff patch per frame and a final one on release, as one undo step", async () => {
    const spy = await setup();
    const raf = installManualRaf();
    try {
      const root = sliderOf("Highlights").closest('[data-slot="slider"]') as HTMLElement;
      mockRect(root, { left: 100, top: 0, width: 200, height: 38 });
      const pointer = capturingPointer(root);
      spy.mockClear();
      pointer.down(200); // the middle: 0.5
      pointer.move(230);
      pointer.move(250);
      pointer.move(260); // 0.8
      // Three moves before the frame: one patch, the whole `rolloff` with the newest value and Low light as it was.
      await act(async () => {
        raf.flush();
      });
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
      expect(spy.mock.calls[0]?.[1]).toEqual({ rolloff: { highlights: 0.8, low_light: 0.5 } });
      pointer.up(260);
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
      expect(spy.mock.calls[1]?.[1]).toEqual({ rolloff: { highlights: 0.8, low_light: 0.5 } });
      await waitFor(() => expect(valueOf("Highlights")).toBe("0.80"));
      expect(h.store.getState().canUndo).toBe(true);
    } finally {
      raf.restore();
    }
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(valueOf("Highlights")).toBe("1.00"));
    expect(h.store.getState().canUndo).toBe(false);
  });
});

describe("the white lift is back beside the Range card (owner's decision of 2026-10-17)", () => {
  it("the Light tab has its three cards again: Exposure, White and High dynamic range, with the Offset slider and its AUTO button", async () => {
    await setup();
    await openTab(h, "Light");
    const regions = screen.getAllByRole("region");
    expect(regions.map((r) => within(r).getAllByRole("heading", { level: 3 })[0]?.textContent)).toEqual(["Exposure", "White", "High dynamic range"]);
    const white = within(screen.getByRole("region", { name: "White" }));
    expect(white.getAllByRole("slider").map((s: HTMLElement) => sliderName(s))).toEqual(["Offset"]);
    expect(valueOf("Offset")).toBe("0.00");
    expect(white.getByRole("button", { name: "Auto white" })).toHaveAttribute("aria-pressed", "true");
  });

  it("the Automatic card has its six rows: Exposure, Contrast, White, White balance, Noise and Camera", async () => {
    await setup();
    await openTab(h, "Look");
    const rows = within(screen.getByRole("region", { name: "Automatic" })).getAllByRole("button", { name: /Open / });
    expect(rows.map((r) => r.getAttribute("aria-label")?.replace(/ .*$/, ""))).toEqual(["Exposure", "Contrast", "White", "White", "Noise", "Camera"]);
    expect(rows[2]).toHaveAccessibleName("White +0.80 EV, Auto. Open Light");
    expect(rows[3]).toHaveAccessibleName(/^White balance /);
  });

  it("the settings carry both: the white lift's offset at 0 and the range at 1 and 0.5", async () => {
    await setup();
    const rs = h.store.currentSettings();
    expect(rs?.settings.white_offset_ev).toBe(0);
    expect(rs?.settings.rolloff).toEqual({ highlights: 1, low_light: 0.5 });
    expect(rs?.values.white_ev).toBe(0.8);
    expect(rs?.estimates.white_ev).toBe(0.8);
  });
});
