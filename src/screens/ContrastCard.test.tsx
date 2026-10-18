/**
 * Contrast is Auto or Manual, like Exposure (`docs/18` §7, `docs/11`): the Light tab's Contrast row (the AUTO toggle of its row while it
 * is Auto, Reset once it is moved), the Exposure card's Reset, and the Contrast row of the Automatic card (value, AUTO or EDITED, the
 * arrow to Light). The mockup's current photo (photo 4) resolves Auto contrast to +1.5 (`estimatesFor`); photo 3 to 0.
 */
import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SettingsPatch } from "../api/types";
import { openPhotos, openTab, renderApp, sliderOf, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";

let h: Harness;
afterEach(() => h?.unmount());

async function setup(current = 3): Promise<UpdateSpy> {
  h = renderApp({ current, picked: [2, 3, 4] });
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

const exposureCard = (): ReturnType<typeof within> => within(screen.getByRole("region", { name: "Exposure" }));
const autoContrast = (): HTMLElement => exposureCard().getByRole("button", { name: "Auto contrast" });
const automaticRow = (name: RegExp): HTMLElement => within(screen.getByRole("region", { name: "Automatic" })).getByRole("button", { name });

describe("the Contrast row on the Light tab", () => {
  it("is Auto at the photo's own value: +1.5 with its AUTO toggle pressed, and no Reset", async () => {
    await setup();
    await openTab(h, "Light");
    expect(valueOf("Contrast")).toBe("+1.5");
    expect(autoContrast()).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("button", { name: "Reset Contrast" })).toBeNull();
    // The slider is named by its row and reads the value to a tenth.
    expect(sliderOf("Contrast")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Contrast value" })).toHaveValue("+1.5");
  });

  it("moving it makes it Manual, and its Reset returns it to Auto", async () => {
    const spy = await setup();
    await openTab(h, "Light");
    await type("Contrast", "12");
    expect(await lastPatch(spy)).toEqual({ contrast: { kind: "manual", value: 12 } });
    await waitFor(() => expect(valueOf("Contrast")).toBe("+12.0"));
    expect(screen.queryByRole("button", { name: "Auto contrast" })).toBeNull();
    expect(screen.getByRole("textbox", { name: "Contrast value" })).toHaveAttribute("data-changed", "true");
    spy.mockClear();
    await h.user.click(screen.getByRole("button", { name: "Reset Contrast" }));
    expect(await lastPatch(spy)).toEqual({ contrast: { kind: "auto" } });
    await waitFor(() => expect(valueOf("Contrast")).toBe("+1.5"));
    expect(autoContrast()).toHaveAttribute("aria-pressed", "true");
  });

  it("a key step from the automatic value is Manual, and a step back onto it is Auto again", async () => {
    // Photo 3 resolves Auto contrast to 0.
    const spy = await setup(2);
    await openTab(h, "Light");
    expect(valueOf("Contrast")).toBe("0.0");
    sliderOf("Contrast").focus();
    await h.user.keyboard("{ArrowRight}");
    expect(await lastPatch(spy)).toEqual({ contrast: { kind: "manual", value: 1 } });
    await waitFor(() => expect(valueOf("Contrast")).toBe("+1.0"));
    spy.mockClear();
    await h.user.keyboard("{ArrowLeft}");
    expect(await lastPatch(spy)).toEqual({ contrast: { kind: "auto" } });
    await waitFor(() => expect(valueOf("Contrast")).toBe("0.0"));
    expect(autoContrast()).toHaveAttribute("aria-pressed", "true");
  });

  it("the Exposure card's Reset puts Exposure and Contrast on Auto and Brightness at 0, in one update", async () => {
    const spy = await setup();
    await openTab(h, "Light");
    await type("Contrast", "12");
    await type("Brightness", "5");
    await waitFor(() => expect(valueOf("Brightness")).toBe("+5"));
    spy.mockClear();
    await h.user.click(exposureCard().getByRole("button", { name: "Reset" }));
    expect(spy).toHaveBeenCalledTimes(1);
    expect(await lastPatch(spy)).toEqual({ exposure: { kind: "auto" }, contrast: { kind: "auto" }, brightness: 0 });
    await waitFor(() => expect(valueOf("Contrast")).toBe("+1.5"));
    expect(valueOf("Brightness")).toBe("0");
  });

  it("the AUTO toggle of the row is pressed while Auto, so pressing it does nothing", async () => {
    const spy = await setup();
    await openTab(h, "Light");
    await h.user.click(autoContrast());
    expect(spy).not.toHaveBeenCalled();
  });
});

describe("the Contrast row of the Automatic card", () => {
  it("sits after Exposure, shows the value with a sign and a tenth, AUTO while Auto, and opens the Light tab", async () => {
    await setup();
    const rows = within(screen.getByRole("region", { name: "Automatic" })).getAllByRole("button", { name: /Open / });
    expect(rows[0]).toHaveAccessibleName(/^Exposure /);
    expect(rows[1]).toHaveAccessibleName("Contrast +1.5, Auto. Open Light");
    expect(rows[1]!.textContent).toContain("+1.5");
    await h.user.click(automaticRow(/^Contrast \+1\.5, Auto\. Open Light$/));
    await waitFor(() => expect(screen.getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true"));
  });

  it("follows the value: EDITED with the value on screen once it is moved, and AUTO again after Reset", async () => {
    await setup();
    await openTab(h, "Light");
    await type("Contrast", "-12");
    await openTab(h, "Look");
    await waitFor(() => expect(automaticRow(/^Contrast −12\.0, Edited\. Open Light$/)).toBeInTheDocument());
    await openTab(h, "Light");
    await h.user.click(screen.getByRole("button", { name: "Reset Contrast" }));
    await openTab(h, "Look");
    await waitFor(() => expect(automaticRow(/^Contrast \+1\.5, Auto\. Open Light$/)).toBeInTheDocument());
  });

  it("shows a zero automatic contrast as 0.0", async () => {
    await setup(2);
    expect(automaticRow(/^Contrast 0\.0, Auto\. Open Light$/)).toBeInTheDocument();
  });
});
