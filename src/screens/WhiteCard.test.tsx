/**
 * The white lift (owner's decision of 2026-10-14, `docs/02` C12 step 1b, `docs/11`): the Light tab's
 * White card (one slider, Offset, in stops, default 0 = the automatic amount) and the Automatic
 * card's White row (the lift in effect, AUTO while the offset is 0). The mock's automatic lift is
 * +0.80 EV (`MOCK_ESTIMATES.white_ev`).
 */
import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SettingsPatch } from "../api/types";
import { openPhotos, openTab, renderApp, sliderName, sliderOf, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";

let h: Harness;
afterEach(() => h?.unmount());

async function setup(): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4] });
  await openPhotos(h);
  return vi.spyOn(h.api, "updateSettings");
}

async function type(label: string, text: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: `${label} value` });
  await h.user.click(box);
  await h.user.clear(box);
  await h.user.type(box, `${text}{Enter}`);
}

async function lastPatch(spy: UpdateSpy): Promise<SettingsPatch> {
  await waitFor(() => expect(spy).toHaveBeenCalled());
  return spy.mock.calls.at(-1)?.[1] as SettingsPatch;
}

const card = (): ReturnType<typeof within> => within(screen.getByRole("region", { name: "White" }));
const autoButton = (): HTMLElement => card().getByRole("button", { name: "Auto white" });
const whiteRow = (name: RegExp): HTMLElement => screen.getByRole("button", { name });

describe("the White card on the Light tab", () => {
  it("sits between Exposure and High dynamic range with one slider, Offset, at 0 and AUTO", async () => {
    await setup();
    await openTab(h, "Light");
    const regions = screen.getAllByRole("region");
    expect(regions.map((r) => within(r).getAllByRole("heading", { level: 3 })[0]?.textContent)).toEqual(["Exposure", "White", "High dynamic range"]);
    const labels = card()
      .getAllByRole("slider")
      .map((s: HTMLElement) => sliderName(s));
    expect(labels).toEqual(["Offset"]);
    expect(valueOf("Offset")).toBe("0.00");
    expect(autoButton()).toHaveAttribute("aria-pressed", "true");
    // It is a stop offset from −3 to +3 in steps of 0.05; the C9 White slider next door is unchanged.
    const slider = sliderOf("Offset");
    expect([slider.getAttribute("aria-valuemin"), slider.getAttribute("aria-valuemax")]).toEqual(["-3", "3"]);
    expect(valueOf("White")).toBe("0");
    // One arrow key is one step of 0.05 (Radix's slider has no step attribute to read).
    const spy = vi.spyOn(h.api, "updateSettings");
    slider.focus();
    await h.user.keyboard("{ArrowRight}");
    expect(await lastPatch(spy)).toEqual({ white_offset_ev: 0.05 });
  });

  it("an offset is a plain patch of white_offset_ev, shown as signed stops, and the AUTO button follows it", async () => {
    const spy = await setup();
    await openTab(h, "Light");
    await type("Offset", "-0.5");
    expect(await lastPatch(spy)).toEqual({ white_offset_ev: -0.5 });
    await waitFor(() => expect(valueOf("Offset")).toBe("−0.50"));
    expect(autoButton()).toHaveAttribute("aria-pressed", "false");
    expect(h.store.currentSettings()?.settings.white_offset_ev).toBe(-0.5);
    // The C9 White slider is another setting and did not move.
    expect(h.store.currentSettings()?.settings.white).toBe(0);
    spy.mockClear();
    await type("Offset", "0.3");
    expect(await lastPatch(spy)).toEqual({ white_offset_ev: 0.3 });
    await waitFor(() => expect(valueOf("Offset")).toBe("+0.30"));
  });

  it("Reset and the AUTO button put the offset back to 0", async () => {
    const spy = await setup();
    await openTab(h, "Light");
    await type("Offset", "1");
    await waitFor(() => expect(autoButton()).toHaveAttribute("aria-pressed", "false"));
    spy.mockClear();
    await h.user.click(autoButton());
    expect(await lastPatch(spy)).toEqual({ white_offset_ev: 0 });
    await waitFor(() => expect(valueOf("Offset")).toBe("0.00"));
    expect(autoButton()).toHaveAttribute("aria-pressed", "true");

    await type("Offset", "-1");
    await waitFor(() => expect(valueOf("Offset")).toBe("−1.00"));
    spy.mockClear();
    await h.user.click(card().getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy)).toEqual({ white_offset_ev: 0 });
    await waitFor(() => expect(valueOf("Offset")).toBe("0.00"));
  });

  it("undo and redo take it like any field", async () => {
    await setup();
    await openTab(h, "Light");
    await type("Offset", "0.6");
    await waitFor(() => expect(valueOf("Offset")).toBe("+0.60"));
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(valueOf("Offset")).toBe("0.00"));
    await h.user.keyboard("{Meta>}{Shift>}z{/Shift}{/Meta}");
    await waitFor(() => expect(valueOf("Offset")).toBe("+0.60"));
  });
});

describe("the White row of the Automatic card", () => {
  it("shows the automatic lift with the AUTO tag and opens the Light tab", async () => {
    await setup();
    // Between Exposure and White balance, as the owner asked.
    const rows = within(screen.getByRole("region", { name: "Automatic" })).getAllByRole("button", { name: /Open / });
    expect(rows.map((r) => r.getAttribute("aria-label")?.split(" ")[0])).toEqual(["Exposure", "Contrast", "White", "White", "Noise", "Camera"]);
    const row = whiteRow(/^White \+0\.80 EV, Auto\. Open Light$/);
    await h.user.click(row);
    await waitFor(() => expect(screen.getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true"));
  });

  it("follows the offset: the lift in effect, EDITED, and None when it is taken to zero", async () => {
    await setup();
    await openTab(h, "Light");
    await type("Offset", "-0.5");
    await openTab(h, "Look");
    await waitFor(() => expect(whiteRow(/^White \+0\.30 EV, Edited\. Open Light$/)).toBeInTheDocument());
    await openTab(h, "Light");
    await type("Offset", "-1.5");
    await openTab(h, "Look");
    await waitFor(() => expect(whiteRow(/^White None, Edited\. Open Light$/)).toBeInTheDocument());
    await openTab(h, "Light");
    await type("Offset", "0.7");
    await openTab(h, "Look");
    await waitFor(() => expect(whiteRow(/^White \+1\.50 EV, Edited\. Open Light$/)).toBeInTheDocument());
  });

  it("Re-run brings the offset back to 0, like the other rows", async () => {
    await setup();
    await openTab(h, "Light");
    await type("Offset", "0.9");
    await openTab(h, "Look");
    await waitFor(() => expect(whiteRow(/^White \+1\.70 EV, Edited\. Open Light$/)).toBeInTheDocument());
    await h.user.click(screen.getByRole("button", { name: "Re-run" }));
    await waitFor(() => expect(whiteRow(/^White \+0\.80 EV, Auto\. Open Light$/)).toBeInTheDocument());
    expect(h.store.currentSettings()?.settings.white_offset_ev).toBe(0);
  });
});
