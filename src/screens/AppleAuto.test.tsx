/**
 * Apple Intelligence's Auto in the app (`docs/19` sections 2 and 10, `docs/22` section 12, `docs/11-ui.md`): the Preferences row "Auto
 * grading" with Apple Intelligence chosen, shown only while it is available, and under it the instructions for Apple's model; the
 * Automatic card's rows for the values the answer set, its line and the model's plan while a provider grades the photo; and the
 * sliders and wheels the provider sets, which behave like Exposure (the detent and Reset at the Auto value, a value equal to it sent as
 * Auto). The mock plays each state (`MockOptions.appleAuto`). The other providers are in `AutoProviders.test.tsx`.
 */
import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MockAppleAuto, MockOptions } from "../api.mock";
import type { SettingsPatch } from "../api/types";
import { openPhotos, openTab, renderApp, sliderOf, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";
import { MOCK_APPLE_INFO, MOCK_DEFAULT_INSTRUCTIONS } from "../api.mock";
import { AUTO_GRADING_TITLE, PROVIDER_HINTS } from "../state/autoProviders";
import {
  APPLE_INSTRUCTIONS_HINT,
  APPLE_INSTRUCTIONS_LABEL,
  APPLE_INSTRUCTIONS_RESET,
  APPLE_INSTRUCTIONS_SAVE_MS,
} from "./dialogs/Preferences";
import { appleAutoLine, appleColorText, appleRows } from "./tools/LookTab";

/** The Automatic card's lines with Apple Intelligence grading (`docs/22` section 12.5). */
const APPLE_AUTO_LINES = {
  answered: "Set with Apple Intelligence.",
  pending: "Apple Intelligence is looking at this photo…",
  failed: "Apple Intelligence couldn't look at this photo, so Auto uses Yellowmoon's own settings.",
} as const;

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
  await waitFor(() => expect(h!.store.getState().auto).not.toBeNull());
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

describe("Preferences › Auto grading with Apple Intelligence", () => {
  it("is there while Apple Intelligence is available, with the hint of docs/22, after Display rendering", async () => {
    const dialog = await openPreferences("answered");
    const row = within(dialog).getByRole("group", { name: AUTO_GRADING_TITLE });
    expect(row).toHaveTextContent(PROVIDER_HINTS.apple);
    expect(PROVIDER_HINTS.apple).toBe(
      "Apple's on-device model looks at each photo and grades it: exposure, highlights, shadows, black level, contrast, warmth, tint, saturation and the colours of the shadows and highlights. It runs on this Mac and sends nothing anywhere.",
    );
    // The whole hint shows: it is not cut after two lines like the others.
    expect(row.querySelector(".ym-pref__hint")).toHaveClass("line-clamp-none");
    const groups = within(dialog)
      .getAllByRole("group")
      .map((g) => g.getAttribute("aria-label"));
    expect(groups.indexOf(AUTO_GRADING_TITLE)).toBe(groups.indexOf("Display rendering") + 1);
    expect(within(dialog).getByRole("combobox", { name: AUTO_GRADING_TITLE })).toHaveTextContent("Apple Intelligence");
  });

  it("is absent where Apple Intelligence is not available, and in the browser build", async () => {
    for (const mode of ["none", "unavailable"] as const) {
      const dialog = await openPreferences(mode);
      expect(within(dialog).getByRole("group", { name: "Display rendering" })).toBeInTheDocument();
      expect(within(dialog).queryByRole("group", { name: AUTO_GRADING_TITLE })).toBeNull();
      expect(within(dialog).queryByRole("combobox", { name: AUTO_GRADING_TITLE })).toBeNull();
      expect(within(dialog).queryByRole("textbox", { name: APPLE_INSTRUCTIONS_LABEL })).toBeNull();
      h!.unmount();
      h = undefined;
    }
  });

  it("writes the preference when Default Auto is chosen, and when Apple Intelligence is chosen again", async () => {
    const dialog = await openPreferences("answered");
    const select = within(dialog).getByRole("combobox", { name: AUTO_GRADING_TITLE });
    await h!.user.click(select);
    await h!.user.click(await screen.findByRole("option", { name: "Default Auto" }));
    await waitFor(async () => expect((await h!.api.preferencesGet()).auto_provider).toBe("yellowmoon"));
    await waitFor(() => expect(select).toHaveTextContent("Default Auto"));
    await waitFor(() => expect(h!.store.getState().auto?.active).toBe(false));
    await h!.user.click(select);
    await h!.user.click(await screen.findByRole("option", { name: "Apple Intelligence" }));
    await waitFor(async () => expect((await h!.api.preferencesGet()).auto_provider).toBe("apple"));
    await waitFor(() => expect(select).toHaveTextContent("Apple Intelligence"));
    expect(h!.store.getState().consentFor).toBeNull();
  });

  it("shows the instructions for Apple's model under the row while Apple grades, the default when none are saved", async () => {
    const dialog = await openPreferences("answered");
    const field = within(dialog).getByRole("textbox", { name: APPLE_INSTRUCTIONS_LABEL });
    expect(field).toHaveValue(MOCK_DEFAULT_INSTRUCTIONS);
    expect(field).toHaveAttribute("maxlength", "2000");
    expect(within(dialog).getByText(APPLE_INSTRUCTIONS_HINT, { exact: false })).toBeInTheDocument();
    expect(APPLE_INSTRUCTIONS_HINT).toBe("Apple's model reads these before each photo. The answer form stays the same.");
    // At the default there is nothing to reset: the button is there, and disabled.
    const reset = within(dialog).getByRole("button", { name: APPLE_INSTRUCTIONS_RESET });
    expect(reset).toHaveAttribute("aria-disabled", "true");
    // Default Auto chosen, the field goes; Apple Intelligence again, it is back.
    const select = within(dialog).getByRole("combobox", { name: AUTO_GRADING_TITLE });
    await h!.user.click(select);
    await h!.user.click(await screen.findByRole("option", { name: "Default Auto" }));
    await waitFor(() => expect(within(dialog).queryByRole("textbox", { name: APPLE_INSTRUCTIONS_LABEL })).toBeNull());
    await h!.user.click(select);
    await h!.user.click(await screen.findByRole("option", { name: "Apple Intelligence" }));
    expect(await within(dialog).findByRole("textbox", { name: APPLE_INSTRUCTIONS_LABEL })).toHaveValue(MOCK_DEFAULT_INSTRUCTIONS);
  });

  it("saves the instructions when typing rests or the field loses focus, not per keystroke, and Reset to default clears them", async () => {
    const dialog = await openPreferences("answered");
    const spy = vi.spyOn(h!.api, "preferencesSet");
    const field = within(dialog).getByRole("textbox", { name: APPLE_INSTRUCTIONS_LABEL });
    await h!.user.clear(field);
    await h!.user.type(field, "Moody night grade.");
    // Nothing yet: the save waits for the typing to rest.
    expect(spy.mock.calls.filter(([p]) => p.apple_auto_instructions !== undefined)).toHaveLength(0);
    await waitFor(() => expect(spy).toHaveBeenCalledWith({ apple_auto_instructions: "Moody night grade." }), {
      timeout: APPLE_INSTRUCTIONS_SAVE_MS + 2000,
    });
    expect(spy.mock.calls.filter(([p]) => p.apple_auto_instructions !== undefined)).toHaveLength(1);
    await waitFor(async () => expect((await h!.api.preferencesGet()).apple_auto_instructions).toBe("Moody night grade."));
    await waitFor(() => expect(h!.store.getState().auto?.instructions).toBe("edited"));
    // A blur saves at once.
    spy.mockClear();
    await h!.user.type(field, " Cool.");
    await h!.user.tab();
    await waitFor(() => expect(spy).toHaveBeenCalledWith({ apple_auto_instructions: "Moody night grade. Cool." }));
    // Reset to default: enabled now, saves "" at once (the default) and puts the default text back.
    spy.mockClear();
    const reset = within(dialog).getByRole("button", { name: APPLE_INSTRUCTIONS_RESET });
    expect(reset).not.toHaveAttribute("aria-disabled", "true");
    await h!.user.click(reset);
    expect(spy).toHaveBeenCalledWith({ apple_auto_instructions: "" });
    await waitFor(() => expect(field).toHaveValue(MOCK_DEFAULT_INSTRUCTIONS));
    await waitFor(async () => expect((await h!.api.preferencesGet()).apple_auto_instructions).toBe(""));
    await waitFor(() => expect(reset).toHaveAttribute("aria-disabled", "true"));
    await waitFor(() => expect(h!.store.getState().auto?.instructions).toBe("default"));
  });

  it("reads the cached status after each save of the instructions, never the full one that makes the programs check (docs/22 D7)", async () => {
    const dialog = await openPreferences("answered");
    const full = vi.spyOn(h!.api, "autoStatus");
    const cached = vi.spyOn(h!.api, "autoStatusCached");
    const field = within(dialog).getByRole("textbox", { name: APPLE_INSTRUCTIONS_LABEL });
    await h!.user.clear(field);
    await h!.user.type(field, "Moody night grade.");
    await waitFor(() => expect(cached).toHaveBeenCalledTimes(1), { timeout: APPLE_INSTRUCTIONS_SAVE_MS + 2000 });
    await waitFor(() => expect(h!.store.getState().auto?.instructions).toBe("edited"));
    await h!.user.type(field, " Cool.");
    await h!.user.tab();
    await waitFor(() => expect(cached).toHaveBeenCalledTimes(2));
    expect(full).not.toHaveBeenCalled();
  });

  it("asks again each time Preferences opens, so a Mac where Apple Intelligence was switched on shows the row", async () => {
    let dialog = await openPreferences("unavailable");
    expect(within(dialog).queryByRole("group", { name: AUTO_GRADING_TITLE })).toBeNull();
    await h!.user.click(within(dialog).getByRole("button", { name: "Done" }));
    h!.api.configure({ appleAuto: "pending" });
    await h!.user.keyboard("{Meta>},{/Meta}");
    dialog = await screen.findByRole("dialog", { name: "Preferences" });
    expect(await within(dialog).findByRole("group", { name: AUTO_GRADING_TITLE })).toBeInTheDocument();
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

  it("is today's card when Default Auto is chosen", async () => {
    await onPhotos("answered");
    await h!.store.setPreferences({ auto_provider: "yellowmoon" });
    await waitFor(() => expect(rowLabels()).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]));
    expect(screen.queryByText(APPLE_AUTO_LINES.answered)).toBeNull();
  });

  it("answered: a row for each value the answer set, in the table's order between White balance and Noise, the line and the plan", async () => {
    await onPhotos("answered");
    // The mock's answer leaves the shadows and the tint empty: no row for them.
    expect(rowLabels()).toEqual([
      "Exposure",
      "White",
      "White balance",
      "Highlights",
      "Black",
      "Contrast",
      "Saturation",
      "Shadows colour",
      "Highlights colour",
      "Noise",
      "Camera",
    ]);
    expect(screen.getByRole("button", { name: "Highlights −20, Auto. Open Light" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Black −15, Auto. Open Light" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Contrast +15, Auto. Open Tone" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Saturation −15, Auto. Open Tone" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Shadows colour Teal · medium, Auto. Open Colour" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Highlights colour Warm · subtle, Auto. Open Colour" })).toBeInTheDocument();
    // The answer's exposure and warmer white are in the rows above.
    expect(screen.getByRole("button", { name: "Exposure +0.60 EV, Auto. Open Light" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "White balance 5750 K · tint 0, Auto. Open Colour" })).toBeInTheDocument();
    expect(automatic().getByText("Set with Apple Intelligence.")).toHaveClass("ym-row-note");
    expect(automatic().getByText("\u201cTeal shadows and warm highlights for a cinematic late-sun street\u201d")).toHaveClass("ym-row-note");
  });

  it("pending: no rows of Apple's yet and the line that says Apple Intelligence is looking; the answer brings them", async () => {
    await onPhotos("pending");
    expect(rowLabels()).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]);
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

  it("failed: the rule's values, no rows of Apple's, and the line that says so", async () => {
    await onPhotos("failed");
    expect(rowLabels()).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]);
    expect(valueOf("Saturation")).toBe("0");
    expect(automatic().getByText("Apple Intelligence couldn't look at this photo, so Auto uses Yellowmoon's own settings.")).toBeInTheDocument();
  });

  it("lists only the values that were set, and names the colours with their strengths", () => {
    const empty = {
      exposure: null,
      highlights: null,
      shadows: null,
      blacks: null,
      contrast: null,
      white: null,
      tint: null,
      saturation: null,
      shadow_color: null,
      shadow_color_strength: null,
      highlight_color: null,
      highlight_color_strength: null,
    } as const;
    expect(appleRows(null)).toEqual({
      highlights: false,
      shadows: false,
      black: false,
      contrast: false,
      saturation: false,
      shadowColor: null,
      highlightColor: null,
    });
    const some = appleRows({ ...empty, shadows: 0, blacks: 1, shadow_color: "none", highlight_color: "gold" });
    expect(some).toEqual({
      highlights: false,
      shadows: true,
      black: true,
      contrast: false,
      saturation: false,
      shadowColor: "None",
      // A colour without a strength sets nothing.
      highlightColor: null,
    });
    expect(appleColorText("magenta", 3)).toBe("Magenta · strong");
    expect(appleColorText(null, 2)).toBeNull();
  });

  it("an edited Highlights slider says EDITED on its row, and Re-run gives it back to Auto", async () => {
    await onPhotos("answered");
    await openTab(h!, "Light");
    await type("Highlights", "10");
    await openTab(h!, "Look");
    await waitFor(() => expect(screen.getByRole("button", { name: "Highlights +10, Edited. Open Light" })).toBeInTheDocument());
    await h!.user.click(automatic().getByRole("button", { name: "Re-run" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Highlights −20, Auto. Open Light" })).toBeInTheDocument());
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
    expect(appleAutoLine({ state: "pending" }, "apple")).toBe("Apple Intelligence is looking at this photo…");
    expect(appleAutoLine({ state: "failed", kind: "refused", message: "", info: MOCK_APPLE_INFO }, "apple")).toBe(
      "Apple Intelligence couldn't look at this photo, so Auto uses Yellowmoon's own settings.",
    );
    const steps = {
      exposure: 0,
      highlights: null,
      shadows: null,
      blacks: null,
      contrast: 0,
      white: 0,
      tint: null,
      saturation: 0,
      shadow_color: null,
      shadow_color_strength: null,
      highlight_color: null,
      highlight_color_strength: null,
    };
    expect(appleAutoLine({ state: "answered", answer: { steps, scene: "", plan: "" }, info: MOCK_APPLE_INFO }, "apple")).toBe(
      "Set with Apple Intelligence.",
    );
    expect(appleAutoLine({ state: "off" }, "apple")).toBeNull();
    expect(appleAutoLine({ state: "unavailable", reason: "not_a_mac" }, "apple")).toBeNull();
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

describe("the Highlights, Shadows and Black sliders and the wheels with Apple Auto", () => {
  it("have their detent and Reset at the Auto value, and send it as Auto", async () => {
    const spy = await onPhotos("answered");
    await openTab(h!, "Light");
    expect(valueOf("Highlights")).toBe("−20");
    expect(valueOf("Shadows")).toBe("0");
    expect(valueOf("Black")).toBe("−15");
    await type("Highlights", "-25");
    expect(await lastPatch(spy)).toEqual({ highlights: { kind: "manual", value: -25 } });
    spy.mockClear();
    await type("Highlights", "-20");
    expect(await lastPatch(spy)).toEqual({ highlights: { kind: "auto" } });
    spy.mockClear();
    await type("Black", "5");
    expect(await lastPatch(spy)).toEqual({ black: { kind: "manual", value: 5 } });
    spy.mockClear();
    await h!.user.click(await screen.findByRole("button", { name: "Reset Black" }));
    expect(await lastPatch(spy)).toEqual({ black: { kind: "auto" } });
    await waitFor(() => expect(valueOf("Black")).toBe("−15"));
  });

  it("the shadows and highlights wheels show Apple's colours while Auto, and Reset sends Auto", async () => {
    const spy = await onPhotos("answered");
    await openTab(h!, "Colour");
    const shadows = screen.getByRole("slider", { name: "Shadows colour balance" });
    // Teal at medium: radius 0.55, at 240 degrees clockwise from 12 o'clock (the wheel's own angle).
    expect(shadows).toHaveAttribute("aria-valuenow", "55");
    expect(shadows).toHaveAttribute("aria-valuetext", "Strength 55%, hue 240 degrees");
    const highlights = screen.getByRole("slider", { name: "Highlights colour balance" });
    expect(highlights).toHaveAttribute("aria-valuetext", "Strength 30%, hue 60 degrees");
    // A move makes it Manual; Reset (Home) sends Auto and the puck goes back to Apple's colour.
    shadows.focus();
    await h!.user.keyboard("{ArrowUp}");
    expect((await lastPatch(spy)).color_balance?.shadows).toMatchObject({ kind: "manual" });
    spy.mockClear();
    await h!.user.keyboard("{Home}");
    expect((await lastPatch(spy)).color_balance?.shadows).toEqual({ kind: "auto" });
    await waitFor(() => expect(shadows).toHaveAttribute("aria-valuetext", "Strength 55%, hue 240 degrees"));
  });
});
