import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CropSuggestion, NormRect, SettingsPatch } from "../api/types";
import { mockRect } from "../test/raf";
import { openPhotos, renderApp, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";
import { canvasOf, FIT_MARGIN_PX, fitInside, fitRatio, FULL_RECT, isInside, rectRatioPx } from "./crop/cropMath";

// No border in photos (the owner's decision of 2026-10-22, docs/11-ui.md §Crop): the Crop panel's frame always lies inside the
// straightened photo. The panel keeps the frame the person set (the base) and shows, draws and sends it fitted inside; drags are kept
// inside. The mock's photos are 8640 × 5760.

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

const LANDSCAPE = canvasOf(8640, 5760, 0);
const PORTRAIT = canvasOf(8640, 5760, 90);
const crop = (): HTMLElement => screen.getByRole("complementary", { name: "Crop" });
const settings = () => h!.store.currentSettings()!;
const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const frame = (): HTMLElement => screen.getByRole("group", { name: /Crop frame/ });
const pct = (el: HTMLElement, prop: "left" | "top" | "width" | "height"): number => Number.parseFloat(el.style[prop]);

/** The frame drawn over the photo is `rect` (the overlay writes it to a hundredth of a percent). */
function expectFrame(rect: NormRect): void {
  const shown = [pct(frame(), "left"), pct(frame(), "top"), pct(frame(), "width"), pct(frame(), "height")].map((v) => v / 100);
  const wanted = [rect.x, rect.y, rect.width, rect.height];
  shown.forEach((v, i) => expect(v, `frame ${["x", "y", "width", "height"][i]}`).toBeCloseTo(wanted[i]!, 3));
}

async function startCrop(): Promise<void> {
  await h!.user.click(screen.getByRole("button", { name: "Crop" }));
  await screen.findByRole("complementary", { name: "Crop" });
}

async function straighten(degrees: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: "Straighten value" });
  await h!.user.click(box);
  await h!.user.clear(box);
  await h!.user.type(box, `${degrees}{Enter}`);
}

/** Presses Done and returns the crop it sent. */
async function done(spy: UpdateSpy): Promise<SettingsPatch["crop"]> {
  spy.mockClear();
  await h!.user.click(within(crop()).getByRole("button", { name: /^Done/ }));
  await screen.findByRole("complementary", { name: "Tools" });
  return patches(spy).at(-1)?.crop;
}

describe("the frame never shows an empty corner", () => {
  it("Straighten 5° on the whole photo shows the largest frame inside; back to 0° it is the whole photo, and Done sends no crop", async () => {
    const spy = await setup();
    await startCrop();
    await straighten("5");
    await waitFor(() => expect(valueOf("Straighten")).toBe("+5.0°"));
    const largest = fitInside(FULL_RECT, 5, LANDSCAPE);
    expectFrame(largest);
    // h = min(W / (r|cos| + |sin|), H / (r|sin| + |cos|)) for r = 3:2, less the 2 px margin on each side.
    const t = (5 * Math.PI) / 180;
    const want = Math.min(8640 / (1.5 * Math.cos(t) + Math.sin(t)), 5760 / (1.5 * Math.sin(t) + Math.cos(t)));
    expect(Math.abs(largest.height * 5760 - want)).toBeLessThanOrEqual(2 * FIT_MARGIN_PX);
    // The core is still told the whole photo with the angle, so the viewport shows the whole straightened photo under the frame.
    expect(patches(spy).at(-1)).toEqual({ crop: { rect: FULL_RECT, angle_deg: 5, aspect: "original" }, orientation: 0 });
    // Halfway back the frame grows; at 0° it is the whole photo again.
    await straighten("2");
    await waitFor(() => expect(valueOf("Straighten")).toBe("+2.0°"));
    expectFrame(fitInside(FULL_RECT, 2, LANDSCAPE));
    expect(pct(frame(), "width") / 100).toBeGreaterThan(largest.width);
    await straighten("0");
    await waitFor(() => expect(valueOf("Straighten")).toBe("0°"));
    expectFrame(FULL_RECT);
    expect(await done(spy)).toBeNull();
    expect(settings().values.crop).toBeNull();
  });

  it("Done sends the frame shown: the one set, fitted inside the straightened photo", async () => {
    const spy = await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "Free" }));
    // A frame touching the photo's top left corner, which straightening by 2.5° leaves empty.
    screen.getByRole("button", { name: /Crop bottom right corner/ }).focus();
    await h!.user.keyboard("{Shift>}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{/Shift}");
    expectFrame({ x: 0, y: 0, width: 0.85, height: 0.8 });
    await straighten("2.5");
    await waitFor(() => expect(valueOf("Straighten")).toBe("+2.5°"));
    const fitted = fitInside({ x: 0, y: 0, width: 0.85, height: 0.8 }, 2.5, LANDSCAPE);
    expect(fitted.x).toBeGreaterThan(0);
    expectFrame(fitted);
    const sent = await done(spy);
    expect(sent).toMatchObject({ angle_deg: 2.5, aspect: "free" });
    // The key steps leave a little floating-point noise in the frame set, so the comparison is to 1e-12.
    for (const key of ["x", "y", "width", "height"] as const) expect(sent!.rect[key], key).toBeCloseTo(fitted[key], 12);
    expect(isInside(sent!.rect, 2.5, LANDSCAPE)).toBe(true);
    // Its shape is the frame's own.
    expect(rectRatioPx(sent!.rect, 1.5)).toBeCloseTo(rectRatioPx({ x: 0, y: 0, width: 0.85, height: 0.8 }, 1.5), 9);
  });

  it("a drag of the frame stops at the straightened photo's edge, and a handle pulled out stays inside", async () => {
    const spy = await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "Free" }));
    await straighten("5");
    await waitFor(() => expect(valueOf("Straighten")).toBe("+5.0°"));
    // Smaller first: the bottom right corner in by 0.3 both ways (inward is always inside).
    screen.getByRole("button", { name: /Crop bottom right corner/ }).focus();
    await h!.user.keyboard("{Shift>}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowLeft}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{/Shift}");
    const largest = fitInside(FULL_RECT, 5, LANDSCAPE);
    const before = { x: largest.x, y: largest.y, width: largest.width - 0.3, height: largest.height - 0.3 };
    expectFrame(before);
    const image = document.querySelector(".ym-crop__image") as HTMLElement;
    mockRect(image, { left: 0, top: 0, width: 1000, height: 1000 });
    // Drag the frame half the photo to the right: it would leave the turned photo, so it stops where its corner meets the edge.
    fireEvent.pointerDown(frame(), { button: 0, clientX: 300, clientY: 300 });
    fireEvent.pointerMove(window, { clientX: 800, clientY: 300 });
    fireEvent.pointerUp(window, { clientX: 800, clientY: 300 });
    const left = pct(frame(), "left") / 100;
    expect(left).toBeGreaterThan(before.x + 0.1);
    // Not as far as the unturned photo's edge would let it go.
    expect(left + before.width).toBeLessThan(1 - 0.01);
    // A handle pulled far out of the photo keeps the frame inside too.
    const corner = screen.getByRole("button", { name: /Crop bottom right corner/ });
    const [x, y] = [(left + before.width) * 1000, (before.y + before.height) * 1000];
    fireEvent.pointerDown(corner, { button: 0, clientX: x, clientY: y });
    fireEvent.pointerMove(window, { clientX: 1400, clientY: 1400 });
    fireEvent.pointerUp(window, { clientX: 1400, clientY: 1400 });
    const sent = await done(spy);
    expect(sent?.angle_deg).toBe(5);
    expect(isInside(sent!.rect, 5, LANDSCAPE)).toBe(true);
    // It went as far as it could: a thousandth more across or down shows an empty corner.
    expect(isInside({ ...sent!.rect, width: sent!.rect.width + 0.001 }, 5, LANDSCAPE)).toBe(false);
    expect(isInside({ ...sent!.rect, height: sent!.rect.height + 0.001 }, 5, LANDSCAPE)).toBe(false);
  });

  it("the arrow keys can't push the largest frame out", async () => {
    const spy = await setup();
    await startCrop();
    await straighten("4");
    await waitFor(() => expect(valueOf("Straighten")).toBe("+4.0°"));
    const largest = fitInside(FULL_RECT, 4, LANDSCAPE);
    frame().focus();
    await h!.user.keyboard("{Shift>}{ArrowRight}{ArrowDown}{/Shift}");
    expectFrame(largest);
    const sent = await done(spy);
    expect(isInside(sent!.rect, 4, LANDSCAPE)).toBe(true);
    expect(sent!.rect.x).toBeCloseTo(largest.x, 9);
    expect(sent!.rect.y).toBeCloseTo(largest.y, 9);
  });

  it("an aspect at an angle shows the largest frame of that shape inside; back at 0° it is the aspect's frame of the whole photo", async () => {
    const spy = await setup();
    await startCrop();
    await straighten("6");
    await waitFor(() => expect(valueOf("Straighten")).toBe("+6.0°"));
    await h!.user.click(within(crop()).getByRole("radio", { name: "1:1" }));
    const square = fitRatio(FULL_RECT, 1, 1.5);
    const shown = fitInside(square, 6, LANDSCAPE);
    await waitFor(() => expectFrame(shown));
    expect(rectRatioPx(shown, 1.5)).toBeCloseTo(1, 9);
    expect(shown.height).toBeLessThan(1);
    expect(patches(spy).at(-1)).toEqual({ crop: { rect: FULL_RECT, angle_deg: 6, aspect: "square" }, orientation: 0 });
    await straighten("0");
    await waitFor(() => expect(valueOf("Straighten")).toBe("0°"));
    expectFrame(square);
  });

  it("Rotate 90° at an angle shows the largest frame inside the turned photo", async () => {
    await setup();
    await startCrop();
    await straighten("3");
    await waitFor(() => expect(valueOf("Straighten")).toBe("+3.0°"));
    await h!.user.click(within(crop()).getByRole("button", { name: "Rotate 90°" }));
    await waitFor(() => expect(settings().values.orientation).toBe(90));
    // The canvas is now 5760 × 8640: the largest portrait frame at 3°.
    const shown = fitInside(FULL_RECT, 3, PORTRAIT);
    expectFrame(shown);
    expect(rectRatioPx(shown, 2 / 3)).toBeCloseTo(2 / 3, 9);
    expect(isInside(shown, 3, PORTRAIT)).toBe(true);
  });

  it("opens a stored crop that pokes outside as its fit; Cancel leaves the stored crop as it was, Done sends the fit", async () => {
    const spy = await setup();
    const stored = { rect: { x: 0.02, y: 0.03, width: 0.9, height: 0.9 }, angle_deg: 4, aspect: "free" as const };
    await h!.store.update({ crop: stored });
    expect(isInside(stored.rect, 4, LANDSCAPE)).toBe(false);
    const fitted = fitInside(stored.rect, 4, LANDSCAPE);
    await startCrop();
    expectFrame(fitted);
    expect(valueOf("Straighten")).toBe("+4.0°");
    // Cancel with nothing touched sends nothing: the stored crop is as it was.
    spy.mockClear();
    await h!.user.click(within(crop()).getByRole("button", { name: /^Cancel/ }));
    await screen.findByRole("complementary", { name: "Tools" });
    expect(spy).not.toHaveBeenCalled();
    expect(settings().values.crop).toEqual(stored);
    // Touched (Straighten moved and back), Cancel puts the stored crop back unchanged.
    await startCrop();
    await straighten("1");
    await waitFor(() => expect(settings().values.crop?.angle_deg).toBe(1));
    await h!.user.click(within(crop()).getByRole("button", { name: /^Cancel/ }));
    await screen.findByRole("complementary", { name: "Tools" });
    await waitFor(() => expect(settings().values.crop).toEqual(stored));
    // Done without a change sends the frame shown.
    await startCrop();
    expect(await done(spy)).toEqual({ ...stored, rect: fitted });
  });

  it("Auto crop's frame is fitted all the same, and becomes the frame the person set", async () => {
    await setup();
    await startCrop();
    // A proposal whose frame pokes outside at its angle (the core's never does; the panel guards against it).
    const answer: CropSuggestion = { straighten: "rotate", angle_deg: 5, frames: [{ rect: FULL_RECT, score: 0.1 }], keep: false, subject: "picture" };
    vi.spyOn(h!.api, "suggestCrop").mockResolvedValueOnce(answer);
    await h!.user.click(within(crop()).getByRole("button", { name: "Auto crop" }));
    await waitFor(() => expect(valueOf("Straighten")).toBe("+5.0°"));
    const fitted = fitInside(FULL_RECT, 5, LANDSCAPE);
    expectFrame(fitted);
    // Straighten back to 0°: Auto turns off, and the frame stays the fitted one, not the whole photo.
    await straighten("0");
    await waitFor(() => expect(valueOf("Straighten")).toBe("0°"));
    expectFrame(fitted);
  });
});
