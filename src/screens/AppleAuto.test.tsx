/**
 * Apple Auto in the app (`docs/19` sections 2 and 10, `docs/11-ui.md`): the Preferences row "Auto with Apple Intelligence", shown only
 * while Apple Auto is available; the Automatic card's Contrast and Saturation rows and its line while Apple Auto is on for the session;
 * and the Contrast and Saturation sliders, which behave like Exposure (the detent and Reset at the Auto value, a value equal to it sent
 * as Auto). The mock plays each state (`MockOptions.appleAuto`).
 */
import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MockAppleAuto, MockOptions } from "../api.mock";
import type { SettingsPatch } from "../api/types";
import { openPhotos, openTab, renderApp, sliderOf, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";
import { APPLE_AUTO_HINT, APPLE_AUTO_TITLE } from "./dialogs/Preferences";
import { APPLE_AUTO_LINES, appleAutoLine } from "./tools/LookTab";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

async function onPhotos(appleAuto: MockAppleAuto, options: MockOptions = {}): Promise<UpdateSpy> {
  h = renderApp({ current: 3, appleAuto, ...options });
  await openPhotos(h);
  return vi.spyOn(h.api, "updateSettings");
}

async function openPreferences(appleAuto: MockAppleAuto): Promise<HTMLElement> {
  h = renderApp({ appleAuto });
  await screen.findByRole("heading", { level: 1, name: "Drop Sony ARW files." });
  await h.user.keyboard("{Meta>},{/Meta}");
  const dialog = await screen.findByRole("dialog", { name: "Preferences" });
  await waitFor(() => expect(within(dialog).queryByText("Loading your preferences…")).toBeNull());
  await waitFor(() => expect(h!.store.getState().appleAuto).not.toBeNull());
  return dialog;
}

const automatic = () => within(screen.getByRole("region", { name: "Automatic" }));
/** The Automatic card's rows, by the label each starts with. */
const rowLabels = (): string[] =>
  automatic()
    .getAllByRole("button")
    .map((b) => b.querySelector(".ym-auto-row__label")?.textContent ?? "")
    .filter(Boolean);

/** Types a value into a slider's box and presses Return. */
async function type(label: string, text: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: `${label} value` });
  await h!.user.click(box);
  await h!.user.clear(box);
  await h!.user.type(box, `${text}{Enter}`);
}

async function lastPatch(spy: UpdateSpy): Promise<SettingsPatch> {
  await waitFor(() => expect(spy).toHaveBeenCalled());
  return spy.mock.calls.at(-1)?.[1] as SettingsPatch;
}

describe("Preferences › Auto with Apple Intelligence", () => {
  it("is there while Apple Auto is available, with the hint of docs/19, after Display rendering", async () => {
    const dialog = await openPreferences("answered");
    const row = within(dialog).getByRole("group", { name: APPLE_AUTO_TITLE });
    expect(row).toHaveTextContent(APPLE_AUTO_HINT);
    expect(APPLE_AUTO_HINT).toBe(
      "Apple's on-device model looks at each photo and sets its exposure, contrast and saturation, and makes the white warmer or cooler. It runs on this Mac and sends nothing anywhere. Turn it off to use Yellowmoon's own Auto, the same on every computer.",
    );
    // The whole hint shows: it is not cut after two lines like the others.
    expect(row.querySelector(".ym-pref__hint")).toHaveClass("line-clamp-none");
    const groups = within(dialog)
      .getAllByRole("group")
      .map((g) => g.getAttribute("aria-label"));
    expect(groups.indexOf(APPLE_AUTO_TITLE)).toBe(groups.indexOf("Display rendering") + 1);
    expect(within(dialog).getByRole("switch", { name: APPLE_AUTO_TITLE })).toHaveAttribute("aria-checked", "true");
  });

  it("is absent where Apple Auto is not available, and in the browser build", async () => {
    for (const mode of ["none", "unavailable"] as const) {
      const dialog = await openPreferences(mode);
      expect(within(dialog).getByRole("group", { name: "Display rendering" })).toBeInTheDocument();
      expect(within(dialog).queryByRole("group", { name: APPLE_AUTO_TITLE })).toBeNull();
      expect(within(dialog).queryByRole("switch", { name: APPLE_AUTO_TITLE })).toBeNull();
      h!.unmount();
      h = undefined;
    }
  });

  it("writes the preference when the switch is flipped, both ways", async () => {
    const dialog = await openPreferences("answered");
    const toggle = within(dialog).getByRole("switch", { name: APPLE_AUTO_TITLE });
    await h!.user.click(toggle);
    await waitFor(async () => expect((await h!.api.preferencesGet()).apple_auto).toBe(false));
    expect(toggle).toHaveAttribute("aria-checked", "false");
    await waitFor(() => expect(h!.store.getState().appleAuto?.active).toBe(false));
    await h!.user.click(toggle);
    await waitFor(async () => expect((await h!.api.preferencesGet()).apple_auto).toBe(true));
    expect(toggle).toHaveAttribute("aria-checked", "true");
  });

  it("asks again each time Preferences opens, so a Mac where Apple Intelligence was switched on shows the row", async () => {
    let dialog = await openPreferences("unavailable");
    expect(within(dialog).queryByRole("group", { name: APPLE_AUTO_TITLE })).toBeNull();
    await h!.user.click(within(dialog).getByRole("button", { name: "Done" }));
    h!.api.configure({ appleAuto: "pending" });
    await h!.user.keyboard("{Meta>},{/Meta}");
    dialog = await screen.findByRole("dialog", { name: "Preferences" });
    expect(await within(dialog).findByRole("group", { name: APPLE_AUTO_TITLE })).toBeInTheDocument();
  });
});

describe("the Automatic card with Apple Auto", () => {
  it("is today's card while Apple Auto is off or not available", async () => {
    for (const mode of ["none", "unavailable"] as const) {
      await onPhotos(mode);
      expect(rowLabels()).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]);
      for (const line of Object.values(APPLE_AUTO_LINES)) expect(screen.queryByText(line)).toBeNull();
      h!.unmount();
      h = undefined;
    }
  });

  it("is today's card when the preference is off", async () => {
    await onPhotos("answered");
    await h!.store.setPreferences({ apple_auto: false });
    await waitFor(() => expect(rowLabels()).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]));
    expect(screen.queryByText(APPLE_AUTO_LINES.answered)).toBeNull();
  });

  it("answered: Contrast and Saturation between White balance and Noise, signed with no unit, AUTO, and the line", async () => {
    await onPhotos("answered");
    expect(rowLabels()).toEqual(["Exposure", "White", "White balance", "Contrast", "Saturation", "Noise", "Camera"]);
    expect(screen.getByRole("button", { name: "Contrast +15, Auto. Open Tone" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Saturation −15, Auto. Open Tone" })).toBeInTheDocument();
    // The answer's exposure and warmer white are in the rows above.
    expect(screen.getByRole("button", { name: "Exposure +0.60 EV, Auto. Open Light" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "White balance 5750 K · tint 0, Auto. Open Colour" })).toBeInTheDocument();
    expect(automatic().getByText("Set with Apple Intelligence.")).toHaveClass("ym-row-note");
  });

  it("pending: the rule's 0 and the line that says Apple Intelligence is looking; the answer replaces both when it arrives", async () => {
    await onPhotos("pending");
    expect(screen.getByRole("button", { name: "Contrast 0, Auto. Open Tone" })).toBeInTheDocument();
    expect(automatic().getByText("Apple Intelligence is looking at this photo…")).toBeInTheDocument();
    h!.api.setAppleAuto("answered");
    await waitFor(() => expect(screen.getByRole("button", { name: "Contrast +15, Auto. Open Tone" })).toBeInTheDocument());
    expect(automatic().getByText("Set with Apple Intelligence.")).toBeInTheDocument();
    expect(screen.queryByText("Apple Intelligence is looking at this photo…")).toBeNull();
    // The sliders follow: the detent and the box are at the new Auto value.
    expect(valueOf("Contrast")).toBe("+15");
    expect(valueOf("Saturation")).toBe("−15");
  });

  it("the look tiles are made again when the answer changes the picture", async () => {
    await onPhotos("pending");
    const spy = vi.spyOn(h!.api, "lookThumbnail");
    // The tiles of the rule's picture have been asked for by now (the settings rest for 400 ms before a remake).
    await new Promise((resolve) => setTimeout(resolve, 600));
    const before = spy.mock.calls.length;
    h!.api.setAppleAuto("answered");
    await waitFor(() => expect(spy.mock.calls.length).toBeGreaterThan(before), { timeout: 3000 });
  });

  it("failed: the rule's values and the line that says so", async () => {
    await onPhotos("failed");
    expect(screen.getByRole("button", { name: "Saturation 0, Auto. Open Tone" })).toBeInTheDocument();
    expect(automatic().getByText("Apple Intelligence couldn't look at this photo, so Auto uses Yellowmoon's own settings.")).toBeInTheDocument();
  });

  it("says EDITED for a manual contrast, and Re-run gives it back to Auto", async () => {
    await onPhotos("answered");
    await type("Contrast", "40");
    await waitFor(() => expect(screen.getByRole("button", { name: "Contrast +40, Edited. Open Tone" })).toBeInTheDocument());
    await h!.user.click(automatic().getByRole("button", { name: "Re-run" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Contrast +15, Auto. Open Tone" })).toBeInTheDocument());
  });

  it("the arrow of Contrast and Saturation goes to the Tone card's slider on the same tab", async () => {
    await onPhotos("answered");
    await h!.user.click(screen.getByRole("button", { name: "Saturation −15, Auto. Open Tone" }));
    expect(screen.getByRole("tab", { name: "Look" })).toHaveAttribute("aria-selected", "true");
    const tone = screen.getByRole("region", { name: "Tone" });
    await waitFor(() => expect(document.activeElement).toBe(within(tone).getByRole("slider", { name: "Saturation" })));
    await h!.user.click(screen.getByRole("button", { name: "Contrast +15, Auto. Open Tone" }));
    await waitFor(() => expect(document.activeElement).toBe(within(tone).getByRole("slider", { name: "Contrast" })));
  });

  it("the line for each state, and none while Apple Auto is off or not available", () => {
    expect(appleAutoLine({ state: "pending" })).toBe("Apple Intelligence is looking at this photo…");
    expect(appleAutoLine({ state: "failed", kind: "refused", message: "", info: { model: "m", os: "o" } })).toBe(
      "Apple Intelligence couldn't look at this photo, so Auto uses Yellowmoon's own settings.",
    );
    expect(appleAutoLine({ state: "answered", answer: { steps: { exposure: 0, white: 0, contrast: 0, saturation: 0 }, scene: "" }, info: { model: "m", os: "o" } })).toBe(
      "Set with Apple Intelligence.",
    );
    expect(appleAutoLine({ state: "off" })).toBeNull();
    expect(appleAutoLine({ state: "unavailable", reason: "not_a_mac" })).toBeNull();
  });
});

describe("the Contrast and Saturation sliders with Apple Auto", () => {
  it("have their detent and Reset at the Auto value, on the Look, Light and Colour tabs", async () => {
    await onPhotos("answered");
    expect(valueOf("Contrast")).toBe("+15");
    expect(valueOf("Saturation")).toBe("−15");
    // At the Auto value there is nothing to reset.
    expect(screen.queryByRole("button", { name: "Reset Contrast" })).toBeNull();
    await openTab(h!, "Light");
    expect(valueOf("Contrast")).toBe("+15");
    expect(sliderOf("Contrast")).toHaveAttribute("aria-valuenow", "15");
    await openTab(h!, "Colour");
    expect(valueOf("Saturation")).toBe("−15");
  });

  it("send a value equal to the Auto value as Auto, any other value as Manual, and Reset as Auto", async () => {
    const spy = await onPhotos("answered");
    await type("Contrast", "25");
    expect(await lastPatch(spy)).toEqual({ contrast: { kind: "manual", value: 25 } });
    spy.mockClear();
    await type("Contrast", "15");
    expect(await lastPatch(spy)).toEqual({ contrast: { kind: "auto" } });
    await waitFor(() => expect(h!.store.currentSettings()?.auto.contrast).toBe(true));
    spy.mockClear();
    await type("Saturation", "0");
    expect(await lastPatch(spy)).toEqual({ saturation: { kind: "manual", value: 0 } });
    spy.mockClear();
    await h!.user.click(await screen.findByRole("button", { name: "Reset Saturation" }));
    expect(await lastPatch(spy)).toEqual({ saturation: { kind: "auto" } });
    await waitFor(() => expect(valueOf("Saturation")).toBe("−15"));
  });

  it("a key on the knob is one Manual step from the Auto value, and undo takes it back to Auto", async () => {
    const spy = await onPhotos("answered");
    sliderOf("Contrast").focus();
    await h!.user.keyboard("{ArrowRight}");
    expect(await lastPatch(spy)).toEqual({ contrast: { kind: "manual", value: 16 } });
    await waitFor(() => expect(valueOf("Contrast")).toBe("+16"));
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(valueOf("Contrast")).toBe("+15"));
    expect(h!.store.currentSettings()?.settings.contrast).toEqual({ kind: "auto" });
  });

  it("a whole drag is one undo step", async () => {
    await onPhotos("answered");
    await h!.store.update({ saturation: { kind: "manual", value: -5 } }, false);
    await h!.store.update({ saturation: { kind: "manual", value: 10 } }, false);
    await h!.store.update({ saturation: { kind: "manual", value: 30 } }, true);
    await waitFor(() => expect(valueOf("Saturation")).toBe("+30"));
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(valueOf("Saturation")).toBe("−15"));
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("copy and paste carry Auto as Auto: the photo pasted onto keeps its own answer", async () => {
    await onPhotos("answered");
    await type("Saturation", "20");
    await waitFor(() => expect(valueOf("Saturation")).toBe("+20"));
    await h!.store.copySettings();
    await waitFor(() => expect(h!.store.getState().copied).not.toBeNull());
    expect(h!.store.getState().copied?.patch).toMatchObject({ contrast: { kind: "auto" }, saturation: { kind: "manual", value: 20 } });
    await h!.store.step(1);
    await waitFor(() => expect(h!.store.currentSettings()?.photo_id).toBe(5));
    await h!.store.pasteSettings("this");
    await waitFor(() => expect(h!.store.currentSettings()?.settings.saturation).toEqual({ kind: "manual", value: 20 }));
    expect(h!.store.currentSettings()?.settings.contrast).toEqual({ kind: "auto" });
    expect(valueOf("Contrast")).toBe("+15");
  });
});
