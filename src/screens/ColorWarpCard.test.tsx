/**
 * Colour › Colour warp on the Photos screen (`docs/11` §Colour warp): the card's tools, the plane, the selected pin's sliders, Reset,
 * Delete pin, the sixteen-pin hint, and Pick colour on the photo through the pick layer it shares with Pick white.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ColorWarpPin, SettingsPatch } from "../api/types";
import { lockedPin, movingPin, planeToPx, WARP_MAX_PINS } from "../components/warpMath";
import { hexChroma } from "../fixtures/warp";
import { SCENES } from "../fixtures/scenes";
import { openPhotos, openTab, renderApp, sliderName, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";
import { mockRect } from "../test/raf";
import { WARP_HINT, WARP_LIMIT_HINT } from "./tools/ColorWarpCard";
import { inputPatch, mainGradeOf, mainGradePatch } from "../state/settingsGraph";

let h: Harness;
afterEach(() => h?.unmount());

/** DSC04204 (the mockup's photo) on the Colour tab. */
async function setup(initialPins: ColorWarpPin[] = []): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: mainGradePatch({ color_warp: { pins: initialPins } }) } });
  await openPhotos(h);
  await openTab(h, "Colour");
  return vi.spyOn(h.api, "updateSettings");
}

const card = () => within(screen.getByRole("region", { name: "Colour warp" }));
const plane = (): HTMLElement => {
  const el = screen.getByRole("group", { name: "Colour warp plane" });
  mockRect(el, { left: 0, top: 0, width: 352, height: 352 });
  return el;
};
const at = (a: number, b: number) => {
  const p = planeToPx([a, b], 352);
  return { clientX: p.x, clientY: p.y };
};
/** The text of a slider's value box in the Colour warp card (the Colour balance card has a Saturation of its own). */
const cardValue = (label: string): string => (card().getByRole("textbox", { name: `${label} value` }) as HTMLInputElement).value;
const pins = (): ColorWarpPin[] => mainGradeOf(h.store.currentSettings()!.settings).color_warp.pins ?? [];
const photo = (): HTMLElement => screen.getByRole("main", { name: "Photo" });
const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);

async function type(label: string, text: string): Promise<void> {
  const box = card().getByRole("textbox", { name: `${label} value` });
  await h.user.click(box);
  await h.user.clear(box);
  await h.user.type(box, `${text}{Enter}`);
}

/** A press and release on the plane, at a chroma, and a drag to another one. */
async function press(from: [number, number], to?: [number, number]): Promise<void> {
  const el = plane();
  fireEvent.pointerDown(el, { button: 0, ...at(...from) });
  if (to) fireEvent.pointerMove(window, at(...to));
  fireEvent.pointerUp(window);
  await waitFor(() => expect(h.store.getState().canUndo).toBe(true));
}

describe("the Colour warp card", () => {
  it("sits under Colour balance with its tools, the plane and the line of help, and no Reset while there are no pins", async () => {
    await setup();
    const titles = screen.getAllByRole("region").map((r) => within(r).getAllByRole("heading", { level: 3 })[0]?.textContent);
    expect(titles).toEqual(["White balance", "Colour balance", "Colour warp"]);
    const tool = card().getByRole("radiogroup", { name: "Pin tool" });
    expect(within(tool).getByRole("radio", { name: "Move" })).toHaveAttribute("aria-checked", "true");
    expect(within(tool).getByRole("radio", { name: "Lock" })).toHaveAttribute("aria-checked", "false");
    expect(card().getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "false");
    expect(card().getByRole("button", { name: "Delete pin" })).toHaveAttribute("aria-disabled", "true");
    expect(card().getByRole("group", { name: "Colour warp plane" })).toBeInTheDocument();
    expect(card().getByText(WARP_HINT)).toBeInTheDocument();
    expect(card().queryByRole("button", { name: "Reset" })).toBeNull();
    // The plane's colours: the photo's scatter has arrived.
    await waitFor(() => expect(h.store.getState().chromaScatter?.points.length).toBeGreaterThan(0));
  });

  it("a drag on the plane adds a moving pin as one undo step and shows its six sliders at their defaults", async () => {
    const spy = await setup();
    await press([0.1, 0], [0.1, 0.05]);
    await waitFor(() => expect(pins()).toHaveLength(1));
    const [pin] = pins();
    expect(pin?.source[0]).toBeCloseTo(0.1, 6);
    expect(pin?.target[1]).toBeCloseTo(0.05, 6);
    expect(patches(spy).every((p) => Object.keys(p.nodes ?? {}).join() === "1" && Object.keys(p.nodes?.["1"]?.grade ?? {}).join() === "color_warp")).toBe(true);
    const labels = card()
      .getAllByRole("slider")
      .filter((s) => s.getAttribute("aria-labelledby"))
      .map((s) => sliderName(s));
    expect(labels).toEqual(["Range", "Shadows", "Highlights", "Pivot", "Exposure", "Saturation"]);
    expect([valueOf("Range"), valueOf("Shadows"), valueOf("Highlights"), valueOf("Pivot"), valueOf("Exposure"), cardValue("Saturation")]).toEqual([
      "2",
      "100%",
      "100%",
      "0.0 EV",
      "0.00 EV",
      "0%",
    ]);
    expect(card().getByRole("slider", { name: "Pin 1" })).toHaveAttribute("aria-valuetext", "moves red toward orange");
    expect(card().getByRole("button", { name: "Delete pin" })).not.toHaveAttribute("aria-disabled");
    // One step: undo takes the pin away.
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(pins()).toEqual([]));
    expect(card().getByText(WARP_HINT)).toBeInTheDocument();
  });

  it("the sliders change the selected pin: Range ×100, Shadows and Highlights in percent, Pivot and Exposure in stops, Saturation in signed percent", async () => {
    const spy = await setup([movingPin([0.1, 0], [0.1, 0.05]), movingPin([-0.1, 0])]);
    act(() => card().getByRole("slider", { name: "Pin 2" }).focus());
    await type("Range", "10");
    await type("Shadows", "40");
    await type("Highlights", "75");
    await type("Pivot", "1");
    await type("Exposure", "0.3");
    await type("Saturation", "-35");
    await waitFor(() => expect(pins()[1]).toMatchObject({ range: 0.1, shadows: 0.4, highlights: 0.75, pivot_ev: 1, exposure_ev: 0.3, saturation: -0.35 }));
    expect(pins()[0]).toEqual(movingPin([0.1, 0], [0.1, 0.05]));
    expect(valueOf("Pivot")).toBe("+1.0 EV");
    expect(valueOf("Exposure")).toBe("+0.30 EV");
    expect(cardValue("Saturation")).toBe("−35%");
    await type("Saturation", "+20");
    await waitFor(() => expect(pins()[1]?.saturation).toBeCloseTo(0.2, 12));
    expect(cardValue("Saturation")).toBe("+20%");
    // Each typed value is one whole list.
    expect(patches(spy).at(-1)?.nodes?.["1"]?.grade?.color_warp?.pins).toHaveLength(2);
    // The reset column goes back to the default.
    await h.user.click(card().getByRole("button", { name: "Reset Pivot" }));
    await waitFor(() => expect(pins()[1]?.pivot_ev).toBe(0));
    await h.user.click(card().getByRole("button", { name: "Reset Range" }));
    await waitFor(() => expect(pins()[1]?.range).toBe(0.02));
    await h.user.click(card().getByRole("button", { name: "Reset Saturation" }));
    await waitFor(() => expect(pins()[1]?.saturation).toBe(0));
    expect(cardValue("Saturation")).toBe("0%");
  });

  it("Saturation runs from −100 % to +100 %, starts at 0 and is one undo step, and a locked pin has no such row", async () => {
    await setup([movingPin([0.1, 0], [0.1, 0.05]), lockedPin([-0.1, 0])]);
    act(() => card().getByRole("slider", { name: "Pin 1" }).focus());
    const knob = card().getByRole("slider", { name: "Saturation" });
    expect([knob.getAttribute("aria-valuemin"), knob.getAttribute("aria-valuemax"), knob.getAttribute("aria-valuenow")]).toEqual(["-100", "100", "0"]);
    await type("Saturation", "250");
    await waitFor(() => expect(pins()[0]?.saturation).toBe(1));
    expect(cardValue("Saturation")).toBe("+100%");
    await type("Saturation", "-250");
    await waitFor(() => expect(pins()[0]?.saturation).toBe(-1));
    expect(cardValue("Saturation")).toBe("−100%");
    // Each typed value is one undo step.
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(pins()[0]?.saturation).toBe(1));
    // The pin stays what it was in every other field.
    expect(pins()[0]).toEqual({ ...movingPin([0.1, 0], [0.1, 0.05]), saturation: 1 });
    act(() => card().getByRole("slider", { name: "Pin 2" }).focus());
    await waitFor(() => expect(card().queryByRole("slider", { name: "Saturation" })).toBeNull());
    expect(pins()[1]?.saturation).toBe(0);
  });

  it("with Lock a click adds a locked pin, which shows Range only", async () => {
    await setup();
    await h.user.click(card().getByRole("radio", { name: "Lock" }));
    expect(card().getByRole("radio", { name: "Lock" })).toHaveAttribute("aria-checked", "true");
    await press([-0.1, 0.05]);
    await waitFor(() => expect(pins()).toHaveLength(1));
    expect(pins()[0]?.locked).toBe(true);
    const labels = card()
      .getAllByRole("slider")
      .filter((s) => s.getAttribute("aria-labelledby"))
      .map((s) => sliderName(s));
    expect(labels).toEqual(["Range"]);
    expect(card().getByRole("slider", { name: "Pin 1" })).toHaveAttribute("aria-valuetext", "holds cyan");
  });

  it("Delete pin removes the selected pin and Reset removes them all", async () => {
    const spy = await setup([movingPin([0.1, 0], [0.1, 0.05]), lockedPin([-0.1, 0]), movingPin([0, 0.1])]);
    act(() => card().getByRole("slider", { name: "Pin 2" }).focus());
    await h.user.click(card().getByRole("button", { name: "Delete pin" }));
    await waitFor(() => expect(pins()).toEqual([movingPin([0.1, 0], [0.1, 0.05]), movingPin([0, 0.1])]));
    expect(card().getByText(WARP_HINT)).toBeInTheDocument();
    spy.mockClear();
    await h.user.click(card().getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(pins()).toEqual([]));
    expect(patches(spy)).toEqual([mainGradePatch({ color_warp: { pins: [] } })]);
    expect(card().queryByRole("button", { name: "Reset" })).toBeNull();
  });

  it("at sixteen pins a press on empty plane adds nothing and the line says so", async () => {
    const full = Array.from({ length: WARP_MAX_PINS }, (_, i) => movingPin([0.15 * Math.cos(i * 0.39), 0.15 * Math.sin(i * 0.39)]));
    const spy = await setup(full);
    act(() => card().getByRole("slider", { name: "Pin 1" }).focus());
    fireEvent.pointerDown(plane(), { button: 0, ...at(-0.25, -0.25) });
    fireEvent.pointerUp(window);
    expect(await card().findByText(WARP_LIMIT_HINT)).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();
    expect(pins()).toHaveLength(WARP_MAX_PINS);
  });

  it("the selection goes with the photo", async () => {
    await setup([movingPin([0.1, 0], [0.1, 0.05])]);
    act(() => card().getByRole("slider", { name: "Pin 1" }).focus());
    expect(card().getByRole("slider", { name: "Range" })).toBeInTheDocument();
    await act(() => h.store.setCurrent(5));
    await waitFor(() => expect(h.store.getState().warpPin).toBeNull());
  });
});

describe("Pick colour", () => {
  async function startPicking(): Promise<HTMLElement> {
    await h.user.click(card().getByRole("button", { name: "Pick colour" }));
    const layer = await within(photo()).findByRole("application", { name: /^Pick colour\./ });
    mockRect(layer, { left: 0, top: 0, width: 900, height: 600 });
    await h.api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    return layer;
  }

  it("turns on the crosshair layer with its hint; a click adds a moving pin at that colour, selected, and the mode stays on", async () => {
    await setup();
    const sample = vi.spyOn(h.api, "sampleChroma");
    const layer = await startPicking();
    expect(card().getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Click a colour in the photo.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Pick white" })).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(layer, { clientX: 450, clientY: 100 });
    await waitFor(() => expect(pins()).toHaveLength(1));
    expect(sample).toHaveBeenCalledWith(450, 100);
    expect(pins()[0]).toEqual(movingPin(hexChroma(SCENES[3]!.sky)));
    expect(h.store.getState().warpPin).toBe(0);
    expect(card().getByRole("slider", { name: "Range" })).toBeInTheDocument();
    // Still on: a second click adds a second pin, from the ground.
    expect(within(photo()).getByRole("application")).toBeInTheDocument();
    fireEvent.click(layer, { clientX: 450, clientY: 580 });
    await waitFor(() => expect(pins()).toHaveLength(2));
    expect(pins()[1]?.source).toEqual(hexChroma(SCENES[3]!.near));
    expect(h.store.getState().warpPin).toBe(1);
    // Undo takes the last one back.
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(pins()).toHaveLength(1));
  });

  it("ends with Esc or the toggle, and Pick white takes over from it", async () => {
    await setup();
    await startPicking();
    await h.user.keyboard("{Escape}");
    await waitFor(() => expect(within(photo()).queryByRole("application")).toBeNull());
    expect(card().getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "false");
    await startPicking();
    await h.user.click(card().getByRole("button", { name: "Pick colour" }));
    await waitFor(() => expect(within(photo()).queryByRole("application")).toBeNull());
    await startPicking();
    await h.user.click(screen.getByRole("button", { name: "Pick white" }));
    expect(await within(photo()).findByRole("application", { name: /^Pick white\./ })).toBeInTheDocument();
    expect(card().getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("Click something neutral: grey or white.")).toBeInTheDocument();
  });

  it("ends as soon as another tab is shown, by a click or by ⌘1 to ⌘7: the layer goes at once, a click on the photo adds no pin, and Colour comes back with it off", async () => {
    await setup();
    const sample = vi.spyOn(h.api, "sampleChroma");
    const toggle = () => card().getByRole("button", { name: "Pick colour" });
    // Look by a click on the tab; every other tab by its key (⌘4 is Colour itself).
    const elsewhere: Array<[string, () => Promise<void>]> = [
      ["Look, by a click", () => h.user.click(screen.getByRole("tab", { name: "Look" }))],
      ["Look, by ⌘1", () => h.user.keyboard("{Meta>}1{/Meta}")],
      ["Lens, by ⌘2", () => h.user.keyboard("{Meta>}2{/Meta}")],
      ["Light, by ⌘3", () => h.user.keyboard("{Meta>}3{/Meta}")],
      ["Detail, by ⌘5", () => h.user.keyboard("{Meta>}5{/Meta}")],
      ["Curve, by ⌘6", () => h.user.keyboard("{Meta>}6{/Meta}")],
      ["RAW, by ⌘7", () => h.user.keyboard("{Meta>}7{/Meta}")],
    ];
    for (const [what, go] of elsewhere) {
      await startPicking();
      expect(screen.getByText("Click a colour in the photo.")).toBeInTheDocument();
      await go();
      // At once: no waiting for anything.
      expect(within(photo()).queryByRole("application"), what).toBeNull();
      expect(screen.queryByText("Click a colour in the photo."), what).toBeNull();
      // The photo takes the click as a click on a photo: no colour is asked for, no pin is made.
      fireEvent.click(photo(), { clientX: 450, clientY: 100 });
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(sample, what).not.toHaveBeenCalled();
      expect(pins(), what).toEqual([]);
      // Back on Colour (⌘4): Pick colour is off.
      await h.user.keyboard("{Meta>}4{/Meta}");
      expect(toggle(), what).toHaveAttribute("aria-pressed", "false");
      expect(within(photo()).queryByRole("application"), what).toBeNull();
    }
    // Staying on Colour keeps it on: ⌘4 and a click on the Colour tab do nothing to it.
    await startPicking();
    await h.user.keyboard("{Meta>}4{/Meta}");
    await h.user.click(screen.getByRole("tab", { name: "Colour" }));
    expect(toggle()).toHaveAttribute("aria-pressed", "true");
    expect(within(photo()).getByRole("application")).toBeInTheDocument();
  });

  it("ends when the Export panel replaces the tools, and when the photo changes by any road", async () => {
    await setup();
    const toggle = () => card().getByRole("button", { name: "Pick colour" });
    await startPicking();
    act(() => h.store.openExport());
    await waitFor(() => expect(within(photo()).queryByRole("application")).toBeNull());
    act(() => h.store.closeExport());
    await waitFor(() => expect(toggle()).toHaveAttribute("aria-pressed", "false"));
    expect(within(photo()).queryByRole("application")).toBeNull();
    // Another photo, from the store (the filmstrip, the arrows and Delete go this way or through the screen).
    await startPicking();
    await act(() => h.store.setCurrent(5));
    await waitFor(() => expect(within(photo()).queryByRole("application")).toBeNull());
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
  });

  it("leaves Pick white as it was: it outlives a tab, and another photo still ends it", async () => {
    await setup();
    await h.user.click(screen.getByRole("button", { name: "Pick white" }));
    expect(await within(photo()).findByRole("application", { name: /^Pick white\./ })).toBeInTheDocument();
    await h.user.keyboard("{Meta>}1{/Meta}");
    expect(within(photo()).getByRole("application", { name: /^Pick white\./ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Pick white" })).toHaveAttribute("aria-pressed", "true");
    // As before, a click on another photo in the filmstrip ends it.
    await h.user.click(within(screen.getByRole("navigation", { name: "Photos" })).getByRole("button", { name: /^Show DSC04206/ }));
    await waitFor(() => expect(within(photo()).queryByRole("application")).toBeNull());
    expect(screen.getByRole("button", { name: "Pick white" })).toHaveAttribute("aria-pressed", "false");
  });

  it("is off again after the Start screen: leaving the photo view ends it", async () => {
    await setup();
    await startPicking();
    act(() => h.store.go("start"));
    await waitFor(() => expect(screen.queryByRole("main", { name: "Photo" })).toBeNull());
    act(() => h.store.go("photos"));
    await screen.findByRole("main", { name: "Photo" });
    expect(within(photo()).queryByRole("application")).toBeNull();
    await openTab(h, "Colour");
    expect(card().getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "false");
  });

  it("a click on the paper around a letterboxed photo adds nothing", async () => {
    const spy = await setup();
    await h.store.update(inputPatch({ orientation: 90 }));
    spy.mockClear();
    const sample = vi.spyOn(h.api, "sampleChroma");
    const layer = await startPicking();
    fireEvent.click(layer, { clientX: 20, clientY: 300 });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(sample).not.toHaveBeenCalled();
    expect(pins()).toEqual([]);
  });

  it("copy and paste carry the pins to another photo", async () => {
    await setup([movingPin([0.1, 0], [0.1, 0.05]), lockedPin([-0.1, 0])]);
    await act(() => h.store.copySettings());
    await act(() => h.store.setCurrent(5));
    await waitFor(() => expect(h.store.currentSettings()?.photo_id).toBe(5));
    await act(() => h.store.pasteSettings("this"));
    await waitFor(() => expect(pins()).toEqual([movingPin([0.1, 0], [0.1, 0.05]), lockedPin([-0.1, 0])]));
  });
});
