import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SettingsPatch } from "../api/types";
import { mockRect } from "../test/raf";
import { openPhotos, renderApp, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

async function setup(): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4] });
  await openPhotos(h);
  return vi.spyOn(h.api, "updateSettings");
}

type Spy = UpdateSpy;
const crop = (): HTMLElement => screen.getByRole("complementary", { name: "Crop" });
const photo = (): HTMLElement => screen.getByRole("main", { name: "Photo" });
const settings = () => h!.store.currentSettings()!;
const patches = (spy: Spy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const frame = (): HTMLElement => screen.getByRole("group", { name: /Crop frame/ });
const pct = (el: HTMLElement, prop: "left" | "top" | "width" | "height"): number => Number.parseFloat(el.style[prop]);

async function startCrop(): Promise<void> {
  await h!.user.click(screen.getByRole("button", { name: "Crop" }));
  await screen.findByRole("complementary", { name: "Crop" });
}

describe("the Crop panel", () => {
  it("replaces the tool panel with a black title block, Aspect, Straighten, Rotate, Reset, Done and Cancel", async () => {
    await setup();
    await startCrop();
    expect(screen.queryByRole("complementary", { name: "Tools" })).toBeNull();
    expect(screen.queryByRole("tab", { name: "Look" })).toBeNull();
    const panel = within(crop());
    expect(panel.getByRole("heading", { level: 2, name: "Crop" }).closest(".ym-title-block")).toHaveAttribute("data-tone", "ink");
    expect(within(panel.getByRole("radiogroup", { name: "Aspect" })).getAllByRole("radio").map((r) => r.textContent)).toEqual(["Original", "1:1", "4:5", "16:9", "Free"]);
    expect(panel.getByRole("radio", { name: "Original" })).toHaveAttribute("aria-checked", "true");
    expect(panel.getByRole("slider", { name: "Straighten" })).toBeInTheDocument();
    expect(valueOf("Straighten")).toBe("0°");
    expect(panel.getByRole("button", { name: "Rotate 90°" })).toBeInTheDocument();
    expect(panel.getByRole("button", { name: "Reset" })).toBeInTheDocument();
    expect(panel.getByRole("button", { name: /^Done/ })).toBeInTheDocument();
    expect(panel.getByRole("button", { name: /^Cancel/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Crop" })).toHaveAttribute("aria-pressed", "true");
  });

  it("draws the frame with eight handles over the photo", async () => {
    await setup();
    await startCrop();
    expect(within(photo()).getAllByRole("button", { name: /^Crop .* (corner|edge)/ })).toHaveLength(8);
    expect([pct(frame(), "left"), pct(frame(), "top"), pct(frame(), "width"), pct(frame(), "height")]).toEqual([0, 0, 100, 100]);
  });

  it("draws a handle on the cell's edge back inside it by the handle's own size, so the cell does not cut it", async () => {
    await setup();
    await startCrop();
    // The frame is the whole photo and the photo fills the cell: every handle but the middle ones stands on an edge of the cell.
    const handle = (name: RegExp): HTMLElement => screen.getByRole("button", { name });
    const back = "calc(var(--crop-handle) * -1)";
    // Left and top: the handle starts at the edge, so no margin pulls it back. Right and bottom: it ends at the edge.
    expect(handle(/Crop top left corner/).style.marginLeft).toBe("0px");
    expect(handle(/Crop top left corner/).style.marginTop).toBe("0px");
    expect(handle(/Crop bottom right corner/).style.marginLeft).toBe(back);
    expect(handle(/Crop bottom right corner/).style.marginTop).toBe(back);
    expect(handle(/Crop right edge/).style.marginLeft).toBe(back);
    expect(handle(/Crop bottom edge/).style.marginTop).toBe(back);
    // The size is the stylesheet's (`--crop-handle`), not a number here: the offset used to be a fixed 10 px, and the glass handle is 14.
    expect(handle(/Crop bottom right corner/).style.marginLeft).not.toBe("-10px");
    // A handle on the cell's edge, the middle ones of the edges included, draws its focus ring inside itself, where the cell does not clip.
    expect(handle(/Crop bottom right corner/)).toHaveAttribute("data-at-cell-edge", "true");
    expect(handle(/Crop top edge/)).toHaveAttribute("data-at-cell-edge", "true");
    expect(handle(/Crop left edge/)).toHaveAttribute("data-at-cell-edge", "true");
  });

  it("draws the handles as they were once the frame is smaller than the photo", async () => {
    await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "Free" }));
    const right = screen.getByRole("button", { name: /Crop right edge/ });
    right.focus();
    await h!.user.keyboard("{Shift>}{ArrowLeft}{/Shift}");
    expect(pct(frame(), "width")).toBeLessThan(100);
    // The frame no longer reaches the right edge: its right handles sit on the border, centred, as the stylesheet draws them.
    expect(right.style.marginLeft).toBe("");
    expect(right).not.toHaveAttribute("data-at-cell-edge");
    // The left edge still stands on the cell's edge.
    expect(screen.getByRole("button", { name: /Crop left edge/ })).toHaveAttribute("data-at-cell-edge", "true");
  });

  it("Return on a focused button presses that button, not Done", async () => {
    const spy = await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "Free" }));
    within(crop()).getByRole("button", { name: /^Cancel/ }).focus();
    spy.mockClear();
    await h!.user.keyboard("{Enter}");
    await screen.findByRole("complementary", { name: "Tools" });
    // Cancel ran (no crop was kept); Done did not also run.
    expect(settings().values.crop).toBeNull();
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("Return on one of the frame's handles finishes the crop", async () => {
    await setup();
    await startCrop();
    screen.getByRole("button", { name: /Crop top left corner/ }).focus();
    await h!.user.keyboard("{Enter}");
    await screen.findByRole("complementary", { name: "Tools" });
  });

  it("C starts a crop, Esc cancels it, and Return finishes it", async () => {
    await setup();
    await h!.user.keyboard("c");
    await screen.findByRole("complementary", { name: "Crop" });
    await h!.user.keyboard("{Escape}");
    await screen.findByRole("complementary", { name: "Tools" });
    await h!.user.keyboard("c");
    await screen.findByRole("complementary", { name: "Crop" });
    await h!.user.keyboard("{Enter}");
    await screen.findByRole("complementary", { name: "Tools" });
  });

  it("changes the straighten angle live without a final change, and Done writes the crop as one undo step", async () => {
    const spy = await setup();
    await startCrop();
    const box = screen.getByRole("textbox", { name: "Straighten value" });
    await h!.user.click(box);
    await h!.user.clear(box);
    await h!.user.type(box, "3.5{Enter}");
    await waitFor(() => expect(settings().values.crop?.angle_deg).toBe(3.5));
    // While the panel is open the core is told the angle with the frame set to the whole photo.
    expect(patches(spy).at(-1)).toEqual({ crop: { rect: { x: 0, y: 0, width: 1, height: 1 }, angle_deg: 3.5, aspect: "original" }, orientation: 0 });
    expect(valueOf("Straighten")).toBe("+3.5°");
    expect(h!.store.getState().canUndo).toBe(false);
    // The photo in the browser shows the straightened picture.
    expect(photo().querySelector("img")?.getAttribute("src")).toContain("rotate(3.5");
    await h!.user.click(within(crop()).getByRole("button", { name: /^Done/ }));
    await screen.findByRole("complementary", { name: "Tools" });
    expect(settings().values.crop).toEqual({ rect: { x: 0, y: 0, width: 1, height: 1 }, angle_deg: 3.5, aspect: "original" });
    expect(h!.store.getState().canUndo).toBe(true);
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(settings().values.crop).toBeNull());
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("an aspect preset reshapes the frame, locks it, and is sent with the crop", async () => {
    const spy = await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "1:1" }));
    // 8640 × 5760: a square is two thirds as wide as the photo.
    await waitFor(() => expect(pct(frame(), "width")).toBeCloseTo(66.67, 1));
    expect(pct(frame(), "height")).toBeCloseTo(100, 1);
    expect(pct(frame(), "left")).toBeCloseTo(16.67, 1);
    expect(patches(spy).at(-1)).toMatchObject({ crop: { aspect: "square" } });
    // 16:9 keeps the shape landscape.
    await h!.user.click(within(crop()).getByRole("radio", { name: "16:9" }));
    await waitFor(() => expect(pct(frame(), "width")).toBeGreaterThan(pct(frame(), "height")));
    await h!.user.click(within(crop()).getByRole("radio", { name: "Free" }));
    await h!.user.click(within(crop()).getByRole("radio", { name: "Original" }));
    await waitFor(() => expect((pct(frame(), "width") / pct(frame(), "height")) * 1.5).toBeCloseTo(1.5, 1));
  });

  it("Rotate 90° turns the photo and its orientation, and the frame follows", async () => {
    const spy = await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "Free" }));
    await h!.user.click(within(crop()).getByRole("button", { name: "Rotate 90°" }));
    await waitFor(() => expect(settings().values.orientation).toBe(90));
    expect(patches(spy).at(-1)).toMatchObject({ orientation: 90 });
    // The portrait picture is drawn from a portrait frame.
    expect(photo().querySelector("img")?.getAttribute("src")).toContain("viewBox%3D%220%200%201000%201500%22");
    await h!.user.click(within(crop()).getByRole("button", { name: "Rotate 90°" }));
    await h!.user.click(within(crop()).getByRole("button", { name: "Rotate 90°" }));
    await h!.user.click(within(crop()).getByRole("button", { name: "Rotate 90°" }));
    await waitFor(() => expect(settings().values.orientation).toBe(0));
  });

  it("Done sends the frame, the angle and the orientation together", async () => {
    const spy = await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "Free" }));
    // The right edge, moved in with the arrow keys: 0.5% a press, Shift ten times as far.
    const right = screen.getByRole("button", { name: /Crop right edge/ });
    right.focus();
    await h!.user.keyboard("{Shift>}{ArrowLeft}{ArrowLeft}{/Shift}");
    await waitFor(() => expect(pct(frame(), "width")).toBeCloseTo(90, 1));
    spy.mockClear();
    await h!.user.click(within(crop()).getByRole("button", { name: /^Done/ }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    const sent = patches(spy).at(-1) as { crop: { rect: { x: number; y: number; width: number; height: number }; angle_deg: number; aspect: string }; orientation: number };
    expect(sent.orientation).toBe(0);
    expect(sent.crop.angle_deg).toBe(0);
    expect(sent.crop.aspect).toBe("free");
    expect(sent.crop.rect.x).toBeCloseTo(0, 5);
    expect(sent.crop.rect.width).toBeCloseTo(0.9, 5);
    expect(sent.crop.rect.height).toBeCloseTo(1, 5);
    await waitFor(() => expect(settings().values.crop?.rect.width).toBeCloseTo(0.9, 5));
    // The photo is now shown cropped: a 0.9 × 1 piece of the frame.
    expect(photo().querySelector("img")?.getAttribute("src")).toContain("width%3D%221350%22");
  });

  it("moves the frame with the arrow keys and keeps it inside the photo", async () => {
    await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "Free" }));
    screen.getByRole("button", { name: /Crop bottom right corner/ }).focus();
    await h!.user.keyboard("{Shift>}{ArrowLeft}{ArrowUp}{/Shift}");
    await waitFor(() => expect(pct(frame(), "width")).toBeCloseTo(95, 1));
    frame().focus();
    await h!.user.keyboard("{ArrowRight}{ArrowRight}{ArrowRight}");
    await waitFor(() => expect(pct(frame(), "left")).toBeCloseTo(1.5, 1));
    await h!.user.keyboard("{Shift>}{ArrowRight}{ArrowRight}{/Shift}");
    // It stops at the right edge: left + width = 100.
    await waitFor(() => expect(pct(frame(), "left") + pct(frame(), "width")).toBeCloseTo(100, 1));
  });

  it("drags a handle with the pointer", async () => {
    await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "Free" }));
    const image = document.querySelector(".ym-crop__image") as HTMLElement;
    mockRect(image, { left: 0, top: 0, width: 1000, height: 667 });
    const handle = screen.getByRole("button", { name: /Crop bottom right corner/ });
    fireEvent.pointerDown(handle, { button: 0, clientX: 1000, clientY: 667 });
    fireEvent.pointerMove(window, { clientX: 800, clientY: 500 });
    fireEvent.pointerUp(window, { clientX: 800, clientY: 500 });
    await waitFor(() => expect(pct(frame(), "width")).toBeCloseTo(80, 0));
    expect(pct(frame(), "height")).toBeCloseTo(75, 0);
    // The thirds lines show while it is dragged.
    expect(document.querySelector(".ym-crop")).not.toHaveAttribute("data-dragging");
    // Drag the whole frame.
    fireEvent.pointerDown(frame(), { button: 0, clientX: 400, clientY: 300 });
    fireEvent.pointerMove(window, { clientX: 500, clientY: 300 });
    fireEvent.pointerUp(window, { clientX: 500, clientY: 300 });
    await waitFor(() => expect(pct(frame(), "left")).toBeCloseTo(10, 0));
  });

  it("locks the frame to the aspect while a corner is dragged", async () => {
    await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "1:1" }));
    await waitFor(() => expect(pct(frame(), "width")).toBeCloseTo(66.67, 1));
    const image = document.querySelector(".ym-crop__image") as HTMLElement;
    mockRect(image, { left: 0, top: 0, width: 1500, height: 1000 });
    const handle = screen.getByRole("button", { name: /Crop top left corner/ });
    fireEvent.pointerDown(handle, { button: 0, clientX: 250, clientY: 0 });
    fireEvent.pointerMove(window, { clientX: 450, clientY: 100 });
    fireEvent.pointerUp(window);
    await waitFor(() => {
      const w = pct(frame(), "width") * 15; // percent of 1500 px
      const hgt = pct(frame(), "height") * 10;
      expect(w / hgt).toBeCloseTo(1, 1);
    });
  });

  it("Cancel puts the old crop and orientation back and records nothing", async () => {
    const spy = await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("button", { name: "Rotate 90°" }));
    await waitFor(() => expect(settings().values.orientation).toBe(90));
    spy.mockClear();
    await h!.user.click(within(crop()).getByRole("button", { name: /^Cancel/ }));
    await screen.findByRole("complementary", { name: "Tools" });
    expect(patches(spy).at(-1)).toEqual({ crop: null, orientation: 0 });
    await waitFor(() => expect(settings().values.orientation).toBe(0));
    expect(settings().values.crop).toBeNull();
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("Cancel with nothing touched sends nothing", async () => {
    const spy = await setup();
    await startCrop();
    spy.mockClear();
    await h!.user.keyboard("{Escape}");
    await screen.findByRole("complementary", { name: "Tools" });
    expect(spy).not.toHaveBeenCalled();
  });

  it("Reset goes back to the whole, level, unturned photo", async () => {
    await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "1:1" }));
    await h!.user.click(within(crop()).getByRole("button", { name: "Rotate 90°" }));
    await h!.user.click(within(crop()).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(pct(frame(), "width")).toBe(100));
    expect(within(crop()).getByRole("radio", { name: "Original" })).toHaveAttribute("aria-checked", "true");
    await waitFor(() => expect(settings().values.orientation).toBe(0));
  });

  it("the arrows and the photo strip don't move to another photo mid-crop, and picking one ends the crop", async () => {
    await setup();
    await startCrop();
    await h!.user.keyboard("{ArrowRight}");
    expect(h!.store.currentPhoto()?.id).toBe(4);
    await h!.user.click(screen.getByRole("button", { name: "Show DSC04206" }));
    await screen.findByRole("complementary", { name: "Tools" });
    await waitFor(() => expect(h!.store.currentPhoto()?.id).toBe(6));
  });

  it("a crop applied earlier is where the frame starts, and the picture shows it cropped", async () => {
    await setup();
    await h!.store.update({ crop: { rect: { x: 0.1, y: 0.1, width: 0.5, height: 0.5 }, angle_deg: 0, aspect: "free" } });
    await waitFor(() => expect(photo().querySelector("img")?.getAttribute("src")).toContain("viewBox%3D%22150%20100%20750%20500%22"));
    await startCrop();
    expect(within(crop()).getByRole("radio", { name: "Free" })).toHaveAttribute("aria-checked", "true");
    await waitFor(() => expect(pct(frame(), "left")).toBeCloseTo(10, 1));
    expect(pct(frame(), "width")).toBeCloseTo(50, 1);
  });
});

describe("Pick white", () => {
  it("shows the hint in the header, a crosshair layer over the photo, and presses the button", async () => {
    await setup();
    await h!.user.click(screen.getByRole("button", { name: "Pick white" }));
    expect(screen.getByText("Click something neutral: grey or white.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Pick white" })).toHaveAttribute("aria-pressed", "true");
    const layer = within(photo()).getByRole("application");
    expect(layer).toHaveFocus();
    expect(screen.queryByText("Neural · 42 ms · Display P3")).toBeNull();
  });

  it("W starts it and Esc cancels it without changing anything", async () => {
    const spy = await setup();
    await h!.user.keyboard("w");
    expect(await screen.findByText("Click something neutral: grey or white.")).toBeInTheDocument();
    spy.mockClear();
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByText("Click something neutral: grey or white.")).toBeNull());
    expect(within(photo()).queryByRole("application")).toBeNull();
    expect(spy).not.toHaveBeenCalled();
  });

  it("a click goes to the core as a point of the cell, which finds the spot; the mode ends and the result shows as EDITED on the Colour tab", async () => {
    const spy = await setup();
    const input = vi.spyOn(h!.api, "viewportInput");
    await h!.user.keyboard("w");
    const layer = await within(photo()).findByRole("application");
    mockRect(layer, { left: 0, top: 0, width: 900, height: 600 });
    await h!.api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    spy.mockClear();
    fireEvent.click(layer, { clientX: 450, clientY: 150 });
    await waitFor(() => expect(input).toHaveBeenCalled());
    expect(input.mock.calls.at(-1)).toEqual([{ kind: "pick_white", x: 450, y: 150 }]);
    await waitFor(() => expect(within(photo()).queryByRole("application")).toBeNull());
    expect(screen.getByRole("tab", { name: "Colour" })).toHaveAttribute("aria-selected", "true");
    await waitFor(() => expect(settings().settings.white_pick).toEqual([0.5, 0.25]));
    // The screen did not work out the spot itself, and did not send it as a setting: the core set it and the screen read it back.
    expect(spy).not.toHaveBeenCalled();
    const card = within(screen.getByRole("region", { name: "White balance" }));
    expect(card.getByText("Edited")).toBeInTheDocument();
    expect(card.getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "false");
    // The core resolved a kelvin from the sky under the pick: bluer than the estimated light.
    expect(valueOf("Kelvin")).not.toBe("5450 K");
    // Reset clears the pick.
    spy.mockClear();
    await h!.user.click(card.getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(settings().settings.white_pick).toBeNull());
    expect(valueOf("Kelvin")).toBe("5450 K");
  });

  it("works from the keyboard: arrows move the crosshair, Return picks", async () => {
    await setup();
    const input = vi.spyOn(h!.api, "viewportInput");
    await h!.user.keyboard("w");
    const layer = await within(photo()).findByRole("application");
    mockRect(layer, { left: 0, top: 0, width: 900, height: 600 });
    expect(layer).toHaveFocus();
    await h!.user.keyboard("{ArrowLeft}{ArrowUp}");
    await h!.user.keyboard("{Shift>}{ArrowRight}{/Shift}");
    await h!.user.keyboard("{Enter}");
    await waitFor(() => expect(input).toHaveBeenCalled());
    const [event] = input.mock.calls.at(-1) as [{ kind: string; x: number; y: number }];
    expect(event.kind).toBe("pick_white");
    expect(event.x).toBeCloseTo((0.5 - 0.01 + 0.1) * 900, 5);
    expect(event.y).toBeCloseTo((0.5 - 0.01) * 600, 5);
  });

  it("ignores a click on the paper around a letterboxed photo", async () => {
    await setup();
    const input = vi.spyOn(h!.api, "viewportInput");
    await h!.store.update({ orientation: 90 });
    await h!.user.keyboard("w");
    const layer = await within(photo()).findByRole("application");
    mockRect(layer, { left: 0, top: 0, width: 900, height: 600 });
    await h!.api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    input.mockClear();
    // A portrait photo in a 3:2 cell: the picture spans the middle 44 % across. The mode stays on.
    fireEvent.click(layer, { clientX: 20, clientY: 300 });
    expect(input).not.toHaveBeenCalled();
    expect(within(photo()).getByRole("application")).toBeInTheDocument();
    fireEvent.click(layer, { clientX: 450, clientY: 300 });
    await waitFor(() => expect(input).toHaveBeenCalledWith({ kind: "pick_white", x: 450, y: 300 }));
    await waitFor(() => expect(settings().settings.white_pick).not.toBeNull());
    expect(settings().settings.white_pick?.map((v) => Math.round(v * 1000) / 1000)).toEqual([0.5, 0.5]);
  });

  it("at a fixed zoom the picture fills the cell and every click counts", async () => {
    await setup();
    const input = vi.spyOn(h!.api, "viewportInput");
    await h!.store.update({ orientation: 90 });
    await h!.store.setZoom({ kind: "percent", value: 100 });
    await h!.user.keyboard("w");
    const layer = await within(photo()).findByRole("application");
    mockRect(layer, { left: 0, top: 0, width: 900, height: 600 });
    input.mockClear();
    fireEvent.click(layer, { clientX: 20, clientY: 300 });
    await waitFor(() => expect(input).toHaveBeenCalledWith({ kind: "pick_white", x: 20, y: 300 }));
  });

  it("a spot the core can't use shows its words, ends the mode and changes nothing", async () => {
    await setup();
    await h!.user.keyboard("w");
    const layer = await within(photo()).findByRole("application");
    mockRect(layer, { left: 0, top: 0, width: 900, height: 600 });
    h!.api.failNext("viewportInput", { scope: "render", message: "That spot is too bright to be a white. Click something grey." });
    fireEvent.click(layer, { clientX: 450, clientY: 300 });
    expect(await screen.findByText("That spot is too bright to be a white. Click something grey.")).toBeInTheDocument();
    expect(within(photo()).queryByRole("application")).toBeNull();
    expect(settings().settings.white_pick).toBeNull();
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("can't start while the photo is being cropped, and the button is off", async () => {
    await setup();
    await startCrop();
    expect(screen.getByRole("button", { name: "Pick white" })).toHaveAttribute("aria-disabled", "true");
    await h!.user.keyboard("w");
    expect(screen.queryByText("Click something neutral: grey or white.")).toBeNull();
  });

  it("undo takes a pick back", async () => {
    await setup();
    await h!.user.keyboard("w");
    const layer = await within(photo()).findByRole("application");
    mockRect(layer, { left: 0, top: 0, width: 900, height: 600 });
    fireEvent.click(layer, { clientX: 450, clientY: 300 });
    await waitFor(() => expect(settings().settings.white_pick).not.toBeNull());
    expect(h!.store.getState().canUndo).toBe(true);
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(settings().settings.white_pick).toBeNull());
    expect(settings().auto.white_balance).toBe(true);
    // Redo asks the core for the same pick again.
    await h!.user.keyboard("{Meta>}{Shift>}z{/Shift}{/Meta}");
    await waitFor(() => expect(settings().settings.white_pick).not.toBeNull());
  });
});
