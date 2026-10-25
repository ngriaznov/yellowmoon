/**
 * The Look tab's Tone card (owner's decision of 2026-10-13, `docs/16`): Temperature, Contrast and
 * Saturation are the same settings as Colour › White balance › Kelvin, Light › Exposure › Contrast
 * and Colour › Colour balance › Saturation. Moving either copy moves the other, with undo, redo and
 * Reset, and Temperature keeps the Kelvin control's Auto semantics.
 */
import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SettingsPatch } from "../api/types";
import { openPhotos, openTab, renderApp, sliderName, sliderOf, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

let h: Harness;
afterEach(() => h?.unmount());

async function setup(): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4] });
  await openPhotos(h);
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

const tone = (): ReturnType<typeof within> => within(screen.getByRole("region", { name: "Tone" }));
const autoButton = (): HTMLElement => tone().getByRole("button", { name: "Auto temperature" });

describe("the Tone card on the Look tab", () => {
  it("sits under the Look card with Temperature, Contrast and Saturation in that order, and no Tint", async () => {
    await setup();
    const regions = screen.getAllByRole("region");
    expect(regions.map((r) => within(r).getAllByRole("heading", { level: 3 })[0]?.textContent)).toEqual(["Automatic", "Look", "Tone"]);
    const labels = tone()
      .getAllByRole("slider")
      .map((s: HTMLElement) => sliderName(s));
    expect(labels).toEqual(["Temperature", "Contrast", "Saturation"]);
    expect(tone().queryByRole("slider", { name: "Tint" })).toBeNull();
    // The Auto Kelvin with the AUTO tag; Contrast and Saturation start at 0.
    expect(valueOf("Temperature")).toBe("5450 K");
    expect(valueOf("Contrast")).toBe("0");
    expect(valueOf("Saturation")).toBe("0");
    expect(autoButton()).toHaveAttribute("aria-pressed", "true");
  });

  it("leaves Contrast on Light and Kelvin, Tint and Saturation on Colour", async () => {
    await setup();
    await openTab(h, "Light");
    expect(sliderOf("Contrast")).toBeInTheDocument();
    await openTab(h, "Colour");
    for (const label of ["Kelvin", "Tint", "Saturation"]) expect(sliderOf(label)).toBeInTheDocument();
  });

  it("Saturation: moving either copy moves the other", async () => {
    const spy = await setup();
    await type("Saturation", "40");
    expect(await lastPatch(spy)).toEqual({ saturation: manual(40) });
    await waitFor(() => expect(valueOf("Saturation")).toBe("+40"));
    await openTab(h, "Colour");
    expect(valueOf("Saturation")).toBe("+40");
    spy.mockClear();
    await type("Saturation", "-25");
    expect(await lastPatch(spy)).toEqual({ saturation: manual(-25) });
    await openTab(h, "Look");
    await waitFor(() => expect(valueOf("Saturation")).toBe("−25"));
    // The key steps the slider by one, as it does on the Colour tab.
    sliderOf("Saturation").focus();
    await h.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(valueOf("Saturation")).toBe("−24"));
    expect(h.store.currentSettings()?.settings.saturation).toEqual(manual(-24));
  });

  it("Contrast: moving either copy moves the other", async () => {
    const spy = await setup();
    await type("Contrast", "35");
    expect(await lastPatch(spy)).toEqual({ contrast: manual(35) });
    await openTab(h, "Light");
    await waitFor(() => expect(valueOf("Contrast")).toBe("+35"));
    spy.mockClear();
    await type("Contrast", "-60");
    expect(await lastPatch(spy)).toEqual({ contrast: manual(-60) });
    await openTab(h, "Look");
    await waitFor(() => expect(valueOf("Contrast")).toBe("−60"));
  });

  it("Temperature: moving it makes the white balance Manual at that Kelvin with the Auto tint, on both tabs", async () => {
    const spy = await setup();
    await type("Temperature", "6500");
    expect(await lastPatch(spy)).toEqual({ white_balance: { kind: "manual", kelvin: 6500, tint: 0 } });
    await waitFor(() => expect(valueOf("Temperature")).toBe("6500 K"));
    expect(autoButton()).toHaveAttribute("aria-pressed", "false");
    // The Automatic card above reads the same light and says it was edited.
    expect(screen.getByRole("button", { name: "White balance 6500 K · tint 0, Edited. Open Colour" })).toBeInTheDocument();

    await openTab(h, "Colour");
    expect(valueOf("Kelvin")).toBe("6500 K");
    expect(valueOf("Tint")).toBe("0");
    expect(within(screen.getByRole("region", { name: "White balance" })).getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "false");

    // And the other way round: Kelvin and Tint on the Colour tab show on the Look tab (Tint has no row there).
    spy.mockClear();
    await type("Kelvin", "4200");
    expect(await lastPatch(spy)).toEqual({ white_balance: { kind: "manual", kelvin: 4200, tint: 0 } });
    await type("Tint", "9");
    await waitFor(() => expect(valueOf("Tint")).toBe("+9"));
    await openTab(h, "Look");
    await waitFor(() => expect(valueOf("Temperature")).toBe("4200 K"));
    // Moving Temperature keeps the tint that is set.
    spy.mockClear();
    await type("Temperature", "5000");
    expect(await lastPatch(spy)).toEqual({ white_balance: { kind: "manual", kelvin: 5000, tint: 9 } });
  });

  it("Temperature: Reset goes back to Auto, and the Colour tab is Auto again", async () => {
    const spy = await setup();
    await type("Temperature", "7000");
    await waitFor(() => expect(valueOf("Temperature")).toBe("7000 K"));
    spy.mockClear();
    await h.user.click(screen.getByRole("button", { name: "Reset Temperature" }));
    expect(await lastPatch(spy)).toEqual({ white_balance: { kind: "auto" }, light_source: "auto", white_pick: null });
    await waitFor(() => expect(valueOf("Temperature")).toBe("5450 K"));
    expect(autoButton()).toHaveAttribute("aria-pressed", "true");
    expect(h.store.currentSettings()?.settings.white_balance).toEqual({ kind: "auto" });
    await openTab(h, "Colour");
    expect(valueOf("Kelvin")).toBe("5450 K");
    expect(within(screen.getByRole("region", { name: "White balance" })).getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "true");
  });

  it("Temperature: dropping on the Auto Kelvin is Auto, and the AUTO button restores it", async () => {
    const spy = await setup();
    await type("Temperature", "6100");
    await waitFor(() => expect(autoButton()).toHaveAttribute("aria-pressed", "false"));
    spy.mockClear();
    await type("Temperature", "5450");
    expect(await lastPatch(spy)).toEqual({ white_balance: { kind: "auto" } });
    await waitFor(() => expect(autoButton()).toHaveAttribute("aria-pressed", "true"));

    await type("Temperature", "6100");
    await waitFor(() => expect(autoButton()).toHaveAttribute("aria-pressed", "false"));
    spy.mockClear();
    await h.user.click(autoButton());
    expect(await lastPatch(spy)).toEqual({ white_balance: { kind: "auto" }, light_source: "auto", white_pick: null });
    await waitFor(() => expect(autoButton()).toHaveAttribute("aria-pressed", "true"));
    expect(valueOf("Temperature")).toBe("5450 K");
  });

  it("a light preset chosen on the Colour tab shows as Temperature, which is then not Auto", async () => {
    await setup();
    await openTab(h, "Colour");
    await h.user.click(within(screen.getByRole("radiogroup", { name: "Light type" })).getByRole("radio", { name: "Tungsten" }));
    await waitFor(() => expect(valueOf("Kelvin")).toBe("3200 K"));
    await openTab(h, "Look");
    expect(valueOf("Temperature")).toBe("3200 K");
    expect(autoButton()).toHaveAttribute("aria-pressed", "false");
  });

  it("undo and redo take a change made on the Look tab, and show it on the other tab", async () => {
    await setup();
    await type("Saturation", "30");
    await waitFor(() => expect(valueOf("Saturation")).toBe("+30"));
    await type("Temperature", "6800");
    await waitFor(() => expect(valueOf("Temperature")).toBe("6800 K"));

    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(valueOf("Temperature")).toBe("5450 K"));
    expect(valueOf("Saturation")).toBe("+30");
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(valueOf("Saturation")).toBe("0"));
    await openTab(h, "Colour");
    expect(valueOf("Saturation")).toBe("0");
    expect(valueOf("Kelvin")).toBe("5450 K");

    await h.user.keyboard("{Meta>}{Shift>}z{/Shift}{/Meta}");
    await waitFor(() => expect(valueOf("Saturation")).toBe("+30"));
    await h.user.keyboard("{Meta>}{Shift>}z{/Shift}{/Meta}");
    await waitFor(() => expect(valueOf("Kelvin")).toBe("6800 K"));
  });

  it("undo takes a change made on the Colour tab, and the Look tab shows it", async () => {
    await setup();
    await openTab(h, "Colour");
    await type("Saturation", "55");
    await waitFor(() => expect(valueOf("Saturation")).toBe("+55"));
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(valueOf("Saturation")).toBe("0"));
    await openTab(h, "Look");
    expect(valueOf("Saturation")).toBe("0");
  });

  it("each slider's Reset puts it back: Contrast and Saturation to Auto, which is 0 without Apple Auto", async () => {
    const spy = await setup();
    await type("Contrast", "20");
    await type("Saturation", "-45");
    await waitFor(() => expect(valueOf("Saturation")).toBe("−45"));
    spy.mockClear();
    await h.user.click(screen.getByRole("button", { name: "Reset Saturation" }));
    expect(await lastPatch(spy)).toEqual({ saturation: { kind: "auto" } });
    await waitFor(() => expect(valueOf("Saturation")).toBe("0"));
    expect(valueOf("Contrast")).toBe("+20");
    await h.user.click(screen.getByRole("button", { name: "Reset Contrast" }));
    await waitFor(() => expect(valueOf("Contrast")).toBe("0"));
    await openTab(h, "Light");
    expect(valueOf("Contrast")).toBe("0");
  });

  it("the card's Reset puts Temperature, Contrast and Saturation on Auto", async () => {
    const spy = await setup();
    await type("Temperature", "4000");
    await type("Contrast", "12");
    await type("Saturation", "18");
    await waitFor(() => expect(valueOf("Saturation")).toBe("+18"));
    spy.mockClear();
    await h.user.click(tone().getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy)).toEqual({ white_balance: { kind: "auto" }, light_source: "auto", white_pick: null, contrast: { kind: "auto" }, saturation: { kind: "auto" } });
    await waitFor(() => expect(valueOf("Temperature")).toBe("5450 K"));
    expect(valueOf("Contrast")).toBe("0");
    expect(valueOf("Saturation")).toBe("0");
    expect(autoButton()).toHaveAttribute("aria-pressed", "true");
  });

  it("is also shown with the EDITED tag after a white was picked", async () => {
    await setup();
    expect(tone().queryByText("Edited")).toBeNull();
    await h.store.update({ white_pick: [0.4, 0.5] });
    await waitFor(() => expect(tone().getByText("Edited")).toBeInTheDocument());
    expect(autoButton()).toHaveAttribute("aria-pressed", "false");
  });
});
