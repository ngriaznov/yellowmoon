/**
 * M3-07, "every control round-trips through update_settings": each control of the six tabs is
 * driven the way a person would (keys, typed values, clicks), the patch that reaches the mock is
 * checked, and so is what the screen shows once the core's resolved settings come back.
 */
import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SettingsPatch } from "../api/types";
import { capturingPointer } from "../test/pointer";
import { installManualRaf, mockRect } from "../test/raf";
import { openPhotos, openTab, renderApp, sliderOf, tools, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

let h: Harness;
afterEach(() => h?.unmount());

const PHOTO = 4;

async function setup(): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4] });
  await openPhotos(h);
  return vi.spyOn(h.api, "updateSettings");
}

type Spy = UpdateSpy;

/** The patch of the newest `update_settings` call, once there is one. */
async function lastPatch(spy: Spy, label: string): Promise<SettingsPatch> {
  await waitFor(() => expect(spy, `${label}: update_settings was called`).toHaveBeenCalled());
  const call = spy.mock.calls.at(-1);
  expect(call?.[0], `${label}: for the current photo`).toBe(PHOTO);
  return call?.[1] as SettingsPatch;
}

/** Types a value into a slider's box and presses Return. */
async function type(label: string, text: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: `${label} value` });
  await h.user.click(box);
  await h.user.clear(box);
  await h.user.type(box, `${text}{Enter}`);
}

interface TypedCase {
  label: string;
  typed: string;
  /** What the core must receive. */
  patch: SettingsPatch;
  /** What the box shows afterwards. */
  shows: string;
}

/** Types each value, and checks the patch, the box and the core's stored state. */
async function driveTyped(spy: Spy, cases: TypedCase[]): Promise<void> {
  for (const c of cases) {
    spy.mockClear();
    await type(c.label, c.typed);
    expect(await lastPatch(spy, c.label), c.label).toEqual(c.patch);
    await waitFor(() => expect(valueOf(c.label), `${c.label} shows the resolved value`).toBe(c.shows));
    const stored = await h.api.getSettings(PHOTO);
    for (const [key, value] of Object.entries(c.patch)) {
      if (typeof value === "object" && value !== null && "kind" in value) expect(stored.settings[key as keyof typeof stored.settings], c.label).toEqual(value);
      else expect(stored.settings[key as keyof typeof stored.settings], c.label).toEqual(value);
    }
  }
}

describe("every control round-trips through update_settings", () => {
  it("Look: intensity, the tiles and Apply to", async () => {
    const spy = await setup();
    const apply = vi.spyOn(h.api, "applyLook");
    await type("Intensity", "60");
    expect(await lastPatch(spy, "Intensity")).toEqual({ look_intensity: 60 });
    await waitFor(() => expect(valueOf("Intensity")).toBe("60%"));
    // Apply to starts on This photo: nothing is copied to the others until the person says so.
    expect(apply).not.toHaveBeenCalled();
    expect(within(screen.getByRole("main", { name: "Photo" })).getByText("Neutral · 60%")).toBeInTheDocument();

    spy.mockClear();
    await h.user.click(screen.getByRole("radio", { name: "All · 12" }));
    await h.user.click(screen.getByRole("button", { name: "Film warm look, key 3" }));
    expect(await lastPatch(spy, "Film warm")).toEqual({ look: "arri:3000:film-warm", look_intensity: 60 });
    await waitFor(() => expect(apply).toHaveBeenLastCalledWith("all", "arri:3000:film-warm", 60));
    expect((await h.api.getSettings(PHOTO)).settings.look).toBe("arri:3000:film-warm");

    // The slider is a real range input too: a key press steps it by one.
    spy.mockClear();
    sliderOf("Intensity").focus();
    await h.user.keyboard("{ArrowRight}");
    expect(await lastPatch(spy, "Intensity key")).toEqual({ look_intensity: 61 });
    await h.user.keyboard("{Shift>}{ArrowLeft}{/Shift}");
    await waitFor(() => expect(valueOf("Intensity")).toBe("51%"));

    // Reset and double click: back to the full strength.
    await h.user.click(screen.getByRole("button", { name: "Reset Intensity" }));
    await waitFor(() => expect(valueOf("Intensity")).toBe("100%"));
  });

  it("Light: Exposure with Auto, Contrast, Brightness and the four HDR sliders", async () => {
    const spy = await setup();
    await openTab(h, "Light");
    const exposureCard = within(screen.getByRole("region", { name: "Exposure" }));
    expect(exposureCard.getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "true");
    expect(valueOf("Exposure")).toBe("+0.30");

    await driveTyped(spy, [
      { label: "Exposure", typed: "1.5", patch: { exposure: { kind: "manual", ev: 1.5 } }, shows: "+1.50" },
      { label: "Contrast", typed: "-25", patch: { contrast: manual(-25) }, shows: "−25" },
      { label: "Brightness", typed: "10", patch: { brightness: 10 }, shows: "+10" },
      { label: "Highlights", typed: "-18", patch: { highlights: manual(-18) }, shows: "−18" },
      { label: "Shadows", typed: "12", patch: { shadows: manual(12) }, shows: "+12" },
      { label: "White", typed: "5", patch: { white: 5 }, shows: "+5" },
      { label: "Black", typed: "-7", patch: { black: manual(-7) }, shows: "−7" },
    ]);
    // A manual exposure turns the AUTO button off, and the Automatic card says EDITED.
    expect(exposureCard.getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "false");

    // Keys step the slider and clamp at its ends.
    spy.mockClear();
    sliderOf("Contrast").focus();
    await h.user.keyboard("{End}");
    expect(await lastPatch(spy, "Contrast End")).toEqual({ contrast: manual(100) });
    await h.user.keyboard("{Home}");
    await waitFor(() => expect(valueOf("Contrast")).toBe("−100"));

    // Reset on Exposure goes back to Auto, not to a manual +0.30.
    spy.mockClear();
    await h.user.click(screen.getByRole("button", { name: "Reset Exposure" }));
    expect(await lastPatch(spy, "Exposure reset")).toEqual({ exposure: { kind: "auto" } });
    await waitFor(() => expect(valueOf("Exposure")).toBe("+0.30"));
    expect(exposureCard.getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "true");

    // The card's Reset puts the three sliders back at once.
    spy.mockClear();
    await h.user.click(exposureCard.getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy, "Exposure card reset")).toEqual({ exposure: { kind: "auto" }, contrast: { kind: "auto" }, brightness: 0 });
    await waitFor(() => expect(valueOf("Contrast")).toBe("0"));

    spy.mockClear();
    await h.user.click(within(screen.getByRole("region", { name: "High dynamic range" })).getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy, "HDR reset")).toEqual({ highlights: { kind: "auto" }, shadows: { kind: "auto" }, white: 0, black: { kind: "auto" } });
  });

  it("Light: the AUTO button restores Auto exposure, and a slider dropped on its detent is Auto again", async () => {
    const spy = await setup();
    await openTab(h, "Light");
    await type("Exposure", "2");
    await waitFor(() => expect(valueOf("Exposure")).toBe("+2.00"));
    spy.mockClear();
    await h.user.click(within(screen.getByRole("region", { name: "Exposure" })).getByRole("button", { name: "Auto" }));
    expect(await lastPatch(spy, "AUTO")).toEqual({ exposure: { kind: "auto" } });
    await waitFor(() => expect(valueOf("Exposure")).toBe("+0.30"));
    // Typing the number Automatic resolves to is Automatic, not a manual value that happens to match.
    await type("Exposure", "1");
    await waitFor(() => expect(valueOf("Exposure")).toBe("+1.00"));
    spy.mockClear();
    await type("Exposure", "0.3");
    expect(await lastPatch(spy, "detent")).toEqual({ exposure: { kind: "auto" } });
  });

  it("Colour: light presets, Kelvin, Tint, the three wheels and Saturation", async () => {
    const spy = await setup();
    await openTab(h, "Colour");
    const light = screen.getByRole("radiogroup", { name: "Light type" });
    expect(within(light).getAllByRole("radio").map((r) => r.textContent)).toEqual(["Auto", "Day", "Shade", "Tung.", "LED", "Fluor."]);

    // Each preset is sent as light_source, and the core resolves a kelvin for it.
    const presets: Array<[string, string, string]> = [
      ["Daylight", "day", "5500 K"],
      ["Shade", "shade", "7500 K"],
      ["Tungsten", "tungsten", "3200 K"],
      ["LED", "led", "4800 K"],
      ["Fluorescent", "fluorescent", "4100 K"],
      ["Auto", "auto", "5450 K"],
    ];
    for (const [name, value, kelvin] of presets) {
      spy.mockClear();
      await h.user.click(within(light).getByRole("radio", { name }));
      expect(await lastPatch(spy, name), name).toEqual({ light_source: value });
      await waitFor(() => expect(within(light).getByRole("radio", { name }), name).toHaveAttribute("aria-checked", "true"));
      await waitFor(() => expect(valueOf("Kelvin"), name).toBe(kelvin));
    }

    await driveTyped(spy, [
      { label: "Kelvin", typed: "6500", patch: { white_balance: { kind: "manual", kelvin: 6500, tint: 0 } }, shows: "6500 K" },
      { label: "Tint", typed: "12", patch: { white_balance: { kind: "manual", kelvin: 6500, tint: 12 } }, shows: "+12" },
      { label: "Saturation", typed: "6", patch: { saturation: manual(6) }, shows: "+6" },
    ]);
    expect(within(screen.getByRole("region", { name: "White balance" })).getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "false");

    // Kelvin's own Reset goes back to the estimated light, and so does the AUTO button.
    spy.mockClear();
    await h.user.click(screen.getByRole("button", { name: "Reset Kelvin" }));
    expect(await lastPatch(spy, "Kelvin reset")).toEqual({ white_balance: { kind: "manual", kelvin: 5450, tint: 12 }, light_source: "auto", white_pick: null });
    spy.mockClear();
    await h.user.click(within(screen.getByRole("region", { name: "White balance" })).getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy, "WB reset")).toEqual({ white_balance: { kind: "auto" }, light_source: "auto", white_pick: null });
    await waitFor(() => expect(valueOf("Tint")).toBe("0"));
    expect(valueOf("Kelvin")).toBe("5450 K");
  });

  it("Colour: each wheel moves with the arrow keys, and the card's Reset centres them", async () => {
    const spy = await setup();
    await openTab(h, "Colour");
    const wheels = ["Shadows", "Midtones", "Highlights"] as const;
    // The shadows and highlights wheels are Auto or a point (`docs/19` section 5); the midtones wheel is a point.
    const moved = (name: string) => (name === "Midtones" ? [0.02, 0.02] : { kind: "manual", point: [0.02, 0.02] });
    let expected = { shadows: { kind: "auto" }, midtones: [0, 0], highlights: { kind: "auto" } } as Record<string, unknown>;
    for (const name of wheels) {
      spy.mockClear();
      const wheel = screen.getByRole("slider", { name: `${name} colour balance` });
      wheel.focus();
      await h.user.keyboard("{ArrowRight}{ArrowUp}");
      expected = { ...expected, [name.toLowerCase()]: moved(name) };
      await waitFor(() => expect(spy.mock.calls.length).toBeGreaterThanOrEqual(2));
      expect(await lastPatch(spy, name)).toEqual({ color_balance: expected });
      await waitFor(() => expect(wheel, name).toHaveAttribute("aria-valuetext", expect.stringMatching(/^Strength \d+%, hue 45 degrees$/)));
    }
    expect((await h.api.getSettings(PHOTO)).settings.color_balance).toEqual(expected);
    // Reset on a wheel (Home) sends Auto for the shadows and highlights wheels.
    spy.mockClear();
    screen.getByRole("slider", { name: "Highlights colour balance" }).focus();
    await h.user.keyboard("{Home}");
    expect(await lastPatch(spy, "Highlights wheel reset")).toEqual({ color_balance: { ...expected, highlights: { kind: "auto" } } });
    // The Colour balance card's Reset puts the shadows and highlights wheels back to Auto, centres the midtones and resets Saturation.
    spy.mockClear();
    await h.user.click(within(screen.getByRole("region", { name: "Colour balance" })).getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy, "Colour balance reset")).toEqual({
      color_balance: { shadows: { kind: "auto" }, midtones: [0, 0], highlights: { kind: "auto" } },
      saturation: { kind: "auto" },
    });
    await waitFor(() => expect(screen.getByRole("slider", { name: "Shadows colour balance" })).toHaveAttribute("aria-valuetext", "Centred"));
  });

  it("Detail: Amount, Radius, Output and Luminance, Colour, Detail, with Auto and Reset", async () => {
    const spy = await setup();
    await openTab(h, "Detail");
    expect(valueOf("Amount")).toBe("120");
    expect(valueOf("Radius")).toBe("0.8 px");
    // The Output slider (the owner's decision of 2026-10-13): a plain percentage, 100 by default.
    expect(valueOf("Output")).toBe("100%");
    expect(valueOf("Luminance")).toBe("30");
    const nr = within(screen.getByRole("region", { name: "Noise reduction" }));
    expect(nr.getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "true");

    await driveTyped(spy, [
      { label: "Amount", typed: "200", patch: { sharpen: { kind: "manual", amount: 200, radius_px: 0.8 } }, shows: "200" },
      { label: "Radius", typed: "1.5", patch: { sharpen: { kind: "manual", amount: 200, radius_px: 1.5 } }, shows: "1.5 px" },
      // Output is its own field: a number, not an Auto or Manual pair, and it lands on the grid of 5.
      { label: "Output", typed: "35", patch: { output_sharpening: 35 }, shows: "35%" },
      { label: "Output", typed: "52", patch: { output_sharpening: 50 }, shows: "50%" },
      { label: "Output", typed: "250", patch: { output_sharpening: 200 }, shows: "200%" },
      { label: "Output", typed: "0", patch: { output_sharpening: 0 }, shows: "0%" },
      { label: "Luminance", typed: "60", patch: { noise_reduction: { kind: "manual", luminance: 60, color: 50, detail: 50 } }, shows: "60" },
      { label: "Colour", typed: "20", patch: { noise_reduction: { kind: "manual", luminance: 60, color: 20, detail: 50 } }, shows: "20" },
      { label: "Detail", typed: "70", patch: { noise_reduction: { kind: "manual", luminance: 60, color: 20, detail: 70 } }, shows: "70" },
    ]);
    expect(nr.getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "false");

    spy.mockClear();
    await h.user.click(nr.getByRole("button", { name: "Auto" }));
    expect(await lastPatch(spy, "NR auto")).toEqual({ noise_reduction: { kind: "auto" } });
    await waitFor(() => expect(valueOf("Luminance")).toBe("30"));

    // Output's own Reset goes back to 100, as a number (it has no Auto).
    spy.mockClear();
    await h.user.click(screen.getByRole("button", { name: "Reset Output" }));
    expect(await lastPatch(spy, "Output reset")).toEqual({ output_sharpening: 100 });
    await waitFor(() => expect(valueOf("Output")).toBe("100%"));
    // The arrow keys step by 5, Shift by 50, and the ends are 0 and 200.
    spy.mockClear();
    sliderOf("Output").focus();
    await h.user.keyboard("{ArrowRight}");
    expect(await lastPatch(spy, "Output key")).toEqual({ output_sharpening: 105 });
    await h.user.keyboard("{End}");
    await waitFor(() => expect(valueOf("Output")).toBe("200%"));
    await h.user.keyboard("{Home}");
    await waitFor(() => expect(valueOf("Output")).toBe("0%"));

    // The card's Reset puts the whole card back: Amount and Radius to Auto, Output to 100.
    spy.mockClear();
    await h.user.click(within(screen.getByRole("region", { name: "Sharpening" })).getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy, "Sharpening reset")).toEqual({ sharpen: { kind: "auto" }, output_sharpening: 100 });
    await waitFor(() => expect(valueOf("Amount")).toBe("120"));
    expect(valueOf("Radius")).toBe("0.8 px");
    expect(valueOf("Output")).toBe("100%");

    // One slider's own Reset moves only that value.
    await type("Radius", "2");
    await waitFor(() => expect(valueOf("Radius")).toBe("2.0 px"));
    spy.mockClear();
    await h.user.click(screen.getByRole("button", { name: "Reset Radius" }));
    expect(await lastPatch(spy, "Radius reset")).toEqual({ sharpen: { kind: "auto" } });
  });

  it("Curve: adding points to each channel, and Reset", async () => {
    const spy = await setup();
    await openTab(h, "Curve");
    const channels = [["Luma", "luma"], ["Red", "red"], ["Green", "green"], ["Blue", "blue"]] as const;
    expect(within(screen.getByRole("radiogroup", { name: "Curve channel" })).getAllByRole("radio").map((r) => r.textContent)).toEqual(["Luma", "Red", "Green", "Blue"]);
    const identity = [[0, 0], [1, 1]];
    const three = [[0, 0], [0.5, 0.5], [1, 1]];
    const edited = new Set<string>();
    for (const [label, key] of channels) {
      spy.mockClear();
      await h.user.click(within(screen.getByRole("radiogroup", { name: "Curve channel" })).getByRole("radio", { name: label }));
      const plot = await screen.findByRole("group", { name: `${label} curve` });
      plot.focus();
      // Return adds a point in the widest gap: the middle of an identity curve.
      await h.user.keyboard("{Enter}");
      const patch = (await lastPatch(spy, label)) as unknown as { curve: Record<string, number[][]> };
      expect(patch.curve[key], label).toEqual(three);
      // The channels edited before keep their points; the rest are still straight.
      for (const other of channels) if (other[1] !== key) expect(patch.curve[other[1]], `${label}: others unchanged`).toEqual(edited.has(other[1]) ? three : identity);
      edited.add(key);
      await waitFor(() => expect(within(screen.getByRole("group", { name: `${label} curve` })).getAllByRole("slider"), label).toHaveLength(3));
    }
    spy.mockClear();
    await h.user.click(within(screen.getByRole("region", { name: "Curve" })).getByRole("button", { name: "Reset" }));
    const reset = (await lastPatch(spy, "Curve reset")) as unknown as { curve: Record<string, number[][]> };
    expect(reset.curve).toEqual({ luma: identity, red: identity, green: identity, blue: identity });
    await waitFor(() => expect(within(screen.getByRole("group", { name: "Blue curve" })).getAllByRole("slider")).toHaveLength(2));
  });

  it("Curve: moving a point with the arrow keys sends the new curve", async () => {
    const spy = await setup();
    await openTab(h, "Curve");
    const plot = await screen.findByRole("group", { name: "Luma curve" });
    plot.focus();
    await h.user.keyboard("{Enter}");
    await waitFor(() => expect(within(plot).getAllByRole("slider")).toHaveLength(3));
    const middle = within(plot).getByRole("slider", { name: "Point 2 of 3" });
    middle.focus();
    spy.mockClear();
    await h.user.keyboard("{ArrowUp}");
    const patch = (await lastPatch(spy, "point")) as { curve: { luma: number[][] } };
    expect(patch.curve.luma[1]?.[0]).toBeCloseTo(0.5, 2);
    expect(patch.curve.luma[1]?.[1]).toBeCloseTo(0.5 + 1 / 255, 3);
  });

  it("RAW: the four decode switches and the two range sliders", async () => {
    const spy = await setup();
    await openTab(h, "RAW");
    const all = { neural_demosaic: true, highlight_recovery: true, lens_corrections: true, chromatic_aberration: true };
    const switches: Array<[string, keyof typeof all]> = [
      ["Neural demosaic", "neural_demosaic"],
      ["Highlight recovery", "highlight_recovery"],
      ["Lens corrections from camera", "lens_corrections"],
      ["Chromatic aberration", "chromatic_aberration"],
    ];
    let state = { ...all };
    for (const [label, key] of switches) {
      spy.mockClear();
      const toggle = screen.getByRole("switch", { name: label });
      expect(toggle).toHaveAttribute("aria-checked", "true");
      await h.user.click(toggle);
      state = { ...state, [key]: false };
      expect(await lastPatch(spy, label)).toEqual({ decode: state });
      await waitFor(() => expect(screen.getByRole("switch", { name: label })).toHaveAttribute("aria-checked", "false"));
    }
    expect((await h.api.getSettings(PHOTO)).settings.decode).toEqual({ ...all, neural_demosaic: false, highlight_recovery: false, lens_corrections: false, chromatic_aberration: false });
    spy.mockClear();
    await h.user.click(screen.getByRole("switch", { name: "Neural demosaic" }));
    expect(await lastPatch(spy, "on again")).toEqual({ decode: { ...state, neural_demosaic: true } });

    expect(valueOf("Highlights")).toBe("1.00");
    await driveTyped(spy, [
      { label: "Highlights", typed: "0.7", patch: { rolloff: { highlights: 0.7, low_light: 0.5 } }, shows: "0.70" },
      { label: "Low light", typed: "0.25", patch: { rolloff: { highlights: 0.7, low_light: 0.25 } }, shows: "0.25" },
    ]);
    spy.mockClear();
    await h.user.click(within(screen.getByRole("region", { name: "Range" })).getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy, "Range reset")).toEqual({ rolloff: { highlights: 1, low_light: 0.5 } });
    await waitFor(() => expect(valueOf("Low light")).toBe("0.50"));
  });

  it("changes made on one tab show on the others and in the Automatic card", async () => {
    const spy = await setup();
    await openTab(h, "Colour");
    await type("Kelvin", "7000");
    await lastPatch(spy, "Kelvin");
    await openTab(h, "Look");
    expect(await within(tools()).findByRole("button", { name: "White balance 7000 K · tint 0, Edited. Open Colour" })).toBeInTheDocument();
    await openTab(h, "Detail");
    await type("Luminance", "55");
    await openTab(h, "Look");
    expect(await within(tools()).findByRole("button", { name: "Noise Luminance 55, Edited. Open Detail" })).toBeInTheDocument();
  });

  it("every change is for the photo on screen, and the next photo starts at Auto", async () => {
    const spy = await setup();
    await openTab(h, "Light");
    await type("Contrast", "20");
    expect(await lastPatch(spy, "Contrast")).toEqual({ contrast: manual(20) });
    await h.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(h.store.currentSettings()?.photo_id).toBe(5));
    await waitFor(() => expect(valueOf("Contrast")).toBe("0"));
    spy.mockClear();
    await type("Contrast", "9");
    expect(spy.mock.calls.at(-1)?.[0]).toBe(5);
  });

  it("shows the RAW tab's camera card for each way a profile was chosen", async () => {
    await setup();
    await openTab(h, "RAW");
    const camera = within(screen.getByRole("region", { name: "Camera" }));
    expect(camera.getByText("Sony A1 (ILCE-1)")).toBeInTheDocument();
    expect(camera.getByText("A7R IV")).toBeInTheDocument();
    expect(camera.getByText("Stand-in")).toBeInTheDocument();
    expect(camera.getByText("The A1 has no measured profile yet, so it borrows the A7R IV. Eleven Sony cameras have their own, and cameras that share their sensors use them.")).toBeInTheDocument();
  });

  it("a slider drag sends one patch per animation frame and a final one on release, as one undo step", async () => {
    const spy = await setup();
    await openTab(h, "Light");
    const raf = installManualRaf();
    try {
      const root = sliderOf("Highlights").closest('[data-slot="slider"]') as HTMLElement;
      mockRect(root, { left: 100, top: 0, width: 200, height: 38 });
      const pointer = capturingPointer(root);
      spy.mockClear();
      pointer.down(200); // the middle: 0
      pointer.move(190);
      pointer.move(180);
      pointer.move(170);
      // Three moves before the frame: one patch, with the newest value.
      await act(async () => {
        raf.flush();
      });
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
      expect(spy.mock.calls[0]?.[1]).toEqual({ highlights: manual(-30) });
      pointer.move(150);
      await act(async () => {
        raf.flush();
      });
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
      expect(spy.mock.calls[1]?.[1]).toEqual({ highlights: manual(-50) });
      pointer.up(150);
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(3));
      expect(spy.mock.calls[2]?.[1]).toEqual({ highlights: manual(-50) });
      await waitFor(() => expect(valueOf("Highlights")).toBe("\u221250"));
      expect(h.store.getState().canUndo).toBe(true);
    } finally {
      raf.restore();
    }
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(valueOf("Highlights")).toBe("0"));
    expect(h.store.getState().canUndo).toBe(false);
  });
});
