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

let h: Harness;
afterEach(() => h?.unmount());

/** DSC04204 (the mockup's photo) on the Colour tab. */
async function setup(initialPins: ColorWarpPin[] = []): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: { color_warp: { pins: initialPins } } } });
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
const pins = (): ColorWarpPin[] => h.store.currentSettings()?.settings.color_warp.pins ?? [];
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

  it("a drag on the plane adds a moving pin as one undo step and shows its five sliders at their defaults", async () => {
    const spy = await setup();
    await press([0.1, 0], [0.1, 0.05]);
    await waitFor(() => expect(pins()).toHaveLength(1));
    const [pin] = pins();
    expect(pin?.source[0]).toBeCloseTo(0.1, 6);
    expect(pin?.target[1]).toBeCloseTo(0.05, 6);
    expect(patches(spy).every((p) => Object.keys(p).join() === "color_warp")).toBe(true);
    const labels = card()
      .getAllByRole("slider")
      .filter((s) => s.getAttribute("aria-labelledby"))
      .map((s) => sliderName(s));
    expect(labels).toEqual(["Range", "Shadows", "Highlights", "Pivot", "Exposure"]);
    expect([valueOf("Range"), valueOf("Shadows"), valueOf("Highlights"), valueOf("Pivot"), valueOf("Exposure")]).toEqual(["4", "100%", "100%", "0.0 EV", "0.00 EV"]);
    expect(card().getByRole("slider", { name: "Pin 1" })).toHaveAttribute("aria-valuetext", "moves red toward orange");
    expect(card().getByRole("button", { name: "Delete pin" })).not.toHaveAttribute("aria-disabled");
    // One step: undo takes the pin away.
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(pins()).toEqual([]));
    expect(card().getByText(WARP_HINT)).toBeInTheDocument();
  });

  it("the sliders change the selected pin: Range ×100, Shadows and Highlights in percent, Pivot and Exposure in stops", async () => {
    const spy = await setup([movingPin([0.1, 0], [0.1, 0.05]), movingPin([-0.1, 0])]);
    act(() => card().getByRole("slider", { name: "Pin 2" }).focus());
    await type("Range", "10");
    await type("Shadows", "40");
    await type("Highlights", "75");
    await type("Pivot", "1");
    await type("Exposure", "0.3");
    await waitFor(() => expect(pins()[1]).toMatchObject({ range: 0.1, shadows: 0.4, highlights: 0.75, pivot_ev: 1, exposure_ev: 0.3 }));
    expect(pins()[0]).toEqual(movingPin([0.1, 0], [0.1, 0.05]));
    expect(valueOf("Pivot")).toBe("+1.0 EV");
    expect(valueOf("Exposure")).toBe("+0.30 EV");
    // Each typed value is one whole list.
    expect(patches(spy).at(-1)?.color_warp?.pins).toHaveLength(2);
    // The reset column goes back to the default.
    await h.user.click(card().getByRole("button", { name: "Reset Pivot" }));
    await waitFor(() => expect(pins()[1]?.pivot_ev).toBe(0));
    await h.user.click(card().getByRole("button", { name: "Reset Range" }));
    await waitFor(() => expect(pins()[1]?.range).toBe(0.04));
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
    expect(patches(spy)).toEqual([{ color_warp: { pins: [] } }]);
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

  it("a click on the paper around a letterboxed photo adds nothing", async () => {
    const spy = await setup();
    await h.store.update({ orientation: 90 });
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
