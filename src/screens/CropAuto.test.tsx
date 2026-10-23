import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CropAspect, CropSuggestion, NormRect, Orientation, PhotoId, SettingsPatch } from "../api/types";
import { mockCropSuggestion } from "../api.mock";
import type { MockOptions } from "../api.mock";
import { mockRect } from "../test/raf";
import { openPhotos, renderApp, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";
import { AUTO_CROP_LINES, autoCropLines, autoCropNoteLines, levelledLine } from "./crop/CropPanel";
import { canvasOf, fitInside } from "./crop/cropMath";
import { AUTO_CROP_FAILED_MS, draftToCrop } from "./crop/useCropSession";

// Auto crop in the Crop panel (docs/20-auto-crop.md §1): the Frame card's AUTO toggle, what it does to the draft, its note, and the
// answers it must not use. The mock proposes a 1.4° turn and a 90 % frame a little left of centre, or keeps a "keep" photo whole.

let h: Harness | undefined;
afterEach(() => {
  vi.useRealTimers();
  h?.unmount();
  h = undefined;
});

async function setup(options: MockOptions = {}): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], ...options });
  await openPhotos(h);
  return vi.spyOn(h.api, "updateSettings");
}

const FULL: NormRect = { x: 0, y: 0, width: 1, height: 1 };
const crop = (): HTMLElement => screen.getByRole("complementary", { name: "Crop" });
const photo = (): HTMLElement => screen.getByRole("main", { name: "Photo" });
const settings = () => h!.store.currentSettings()!;
const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const frame = (): HTMLElement => screen.getByRole("group", { name: /Crop frame/ });
const pct = (el: HTMLElement, prop: "left" | "top" | "width" | "height"): number => Number.parseFloat(el.style[prop]);
const autoToggle = (): HTMLElement => within(crop()).getByRole("button", { name: "Auto crop" });
const isOn = (): boolean => autoToggle().getAttribute("aria-pressed") === "true";
const noteLines = (): string[] => [...within(crop()).getByRole("status").querySelectorAll("p")].map((p) => p.textContent ?? "");
const currentId = (): PhotoId => h!.store.currentPhoto()!.id;

/** What the mock proposes for the current photo. */
const proposal = (aspect: CropAspect, orientation: Orientation): CropSuggestion => mockCropSuggestion(h!.store.currentPhoto()!, aspect, orientation);
const firstFrame = (s: CropSuggestion): NormRect => s.frames[0]!.rect;

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

/** Presses Auto while it is off and waits for the proposal to be in effect. */
async function autoOn(): Promise<void> {
  await h!.user.click(autoToggle());
  await waitFor(() => expect(noteLines()[0]).not.toBe(AUTO_CROP_LINES.asking));
  expect(isOn()).toBe(true);
}

interface Held {
  aspect: CropAspect;
  orientation: Orientation;
  /** Answers with the mock's proposal for what was asked, or with `answer`. */
  answer: (answer?: CropSuggestion) => Promise<void>;
  fail: () => Promise<void>;
}

/** Holds every request for a proposal until the test answers it. */
function holdSuggestions(): Held[] {
  const held: Held[] = [];
  vi.spyOn(h!.api, "suggestCrop").mockImplementation(
    (id, aspect, orientation) =>
      new Promise<CropSuggestion>((resolve, reject) => {
        const summary = h!.store.getState().session!.photos.find((p) => p.id === id)!;
        held.push({
          aspect,
          orientation,
          answer: (answer) => act(async () => resolve(answer ?? mockCropSuggestion(summary, aspect, orientation))),
          fail: () => act(async () => reject({ scope: "render", message: "The picture could not be made." })),
        });
      }),
  );
  return held;
}

describe("the Auto toggle of the Frame card", () => {
  it("sits in the card's header before Reset, is named Auto crop, reads Auto and starts off with no note", async () => {
    await setup();
    await startCrop();
    const toggle = autoToggle();
    const header = toggle.closest(".ym-card__header") as HTMLElement;
    expect(within(header).getByRole("heading", { name: "Frame" })).toBeInTheDocument();
    expect(toggle).toHaveTextContent("Auto");
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    // As in the other cards' headers: the AUTO button, then Reset, in the order they are drawn and reached with Tab.
    const reset = within(header).getByRole("button", { name: "Reset" });
    expect(toggle.compareDocumentPosition(reset) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(noteLines()).toEqual([]);
  });

  it("pressed, it is on at once and the note says it is finding a frame; the proposal then sets the angle and the frame", async () => {
    const spy = await setup();
    await startCrop();
    const held = holdSuggestions();
    await h!.user.click(autoToggle());
    expect(isOn()).toBe(true);
    expect(noteLines()).toEqual(["Finding a frame…"]);
    // The draft's aspect and orientation are what is asked for; nothing moved yet.
    expect(held.map((r) => [r.aspect, r.orientation])).toEqual([["original", 0]]);
    expectFrame(FULL);
    expect(valueOf("Straighten")).toBe("0°");
    spy.mockClear();
    await held[0]!.answer();
    expectFrame(firstFrame(proposal("original", 0)));
    expect(valueOf("Straighten")).toBe("+1.4°");
    expect(noteLines()).toEqual(["Cropped to frame the subject.", "Levelled by 1.4°."]);
    // The core is told the angle as Straighten tells it: the whole photo, so the viewport shows it straightened under the frame.
    expect(patches(spy).at(-1)).toEqual({ crop: { rect: FULL, angle_deg: 1.4, aspect: "original" }, orientation: 0 });
    await waitFor(() => expect(settings().values.crop?.angle_deg).toBe(1.4));
    expect(photo().querySelector("img")?.getAttribute("src")).toContain("rotate(1.4");
    // Nothing is a step of the history until Done.
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("asks again when the aspect changes while it is on, and the frame jumps to the new proposal; the angle stays", async () => {
    await setup();
    await startCrop();
    const ask = vi.spyOn(h!.api, "suggestCrop");
    await autoOn();
    await h!.user.click(within(crop()).getByRole("radio", { name: "1:1" }));
    await waitFor(() => expect(ask).toHaveBeenLastCalledWith(currentId(), "square", 0));
    await waitFor(() => expectFrame(firstFrame(proposal("square", 0))));
    expect(isOn()).toBe(true);
    expect(valueOf("Straighten")).toBe("+1.4°");
    expect(noteLines()).toEqual(["Cropped to frame the subject.", "Levelled by 1.4°."]);
    await h!.user.click(within(crop()).getByRole("radio", { name: "16:9" }));
    await waitFor(() => expectFrame(firstFrame(proposal("sixteen_by_nine", 0))));
  });

  it("asks again for the new orientation after Rotate 90° while it is on", async () => {
    await setup();
    await startCrop();
    const ask = vi.spyOn(h!.api, "suggestCrop");
    await autoOn();
    await h!.user.click(within(crop()).getByRole("button", { name: "Rotate 90°" }));
    await waitFor(() => expect(ask).toHaveBeenLastCalledWith(currentId(), "original", 90));
    await waitFor(() => expectFrame(firstFrame(proposal("original", 90))));
    await waitFor(() => expect(settings().values.orientation).toBe(90));
    expect(isOn()).toBe(true);
  });

  it("turns off when an arrow key moves the frame, and the moved frame stays", async () => {
    await setup();
    await startCrop();
    await autoOn();
    const auto = firstFrame(proposal("original", 0));
    frame().focus();
    await h!.user.keyboard("{ArrowRight}");
    expect(isOn()).toBe(false);
    expect(noteLines()).toEqual([]);
    expectFrame({ ...auto, x: auto.x + 0.005 });
    // The angle Auto set stays too: only the frame was moved.
    expect(valueOf("Straighten")).toBe("+1.4°");
  });

  it("turns off when a handle or the frame is dragged with the pointer", async () => {
    await setup();
    await startCrop();
    await autoOn();
    const image = document.querySelector(".ym-crop__image") as HTMLElement;
    mockRect(image, { left: 0, top: 0, width: 1000, height: 1000 });
    const auto = firstFrame(proposal("original", 0));
    const handle = screen.getByRole("button", { name: /Crop bottom right corner/ });
    const [x, y] = [(auto.x + auto.width) * 1000, (auto.y + auto.height) * 1000];
    fireEvent.pointerDown(handle, { button: 0, clientX: x, clientY: y });
    fireEvent.pointerMove(window, { clientX: x - 100, clientY: y - 100 });
    fireEvent.pointerUp(window, { clientX: x - 100, clientY: y - 100 });
    await waitFor(() => expect(isOn()).toBe(false));
    expect(pct(frame(), "width") / 100).toBeLessThan(auto.width - 0.05);
    // On again, then a drag of the whole frame.
    await autoOn();
    fireEvent.pointerDown(frame(), { button: 0, clientX: 500, clientY: 500 });
    fireEvent.pointerMove(window, { clientX: 520, clientY: 500 });
    fireEvent.pointerUp(window, { clientX: 520, clientY: 500 });
    await waitFor(() => expect(isOn()).toBe(false));
    expect(pct(frame(), "left") / 100).toBeCloseTo(auto.x + 0.02, 3);
  });

  it("turns off when Straighten moves, and the frame Auto chose stays", async () => {
    const spy = await setup();
    await startCrop();
    await autoOn();
    const box = screen.getByRole("textbox", { name: "Straighten value" });
    await h!.user.click(box);
    await h!.user.clear(box);
    await h!.user.type(box, "-2{Enter}");
    await waitFor(() => expect(isOn()).toBe(false));
    expect(valueOf("Straighten")).toBe("−2.0°");
    expectFrame(firstFrame(proposal("original", 0)));
    expect(noteLines()).toEqual([]);
    await waitFor(() => expect(patches(spy).at(-1)).toMatchObject({ crop: { angle_deg: -2 } }));
  });

  it("turns off with Reset, which goes back to the whole, level, unturned photo in Original", async () => {
    await setup();
    await startCrop();
    await h!.user.click(within(crop()).getByRole("radio", { name: "1:1" }));
    await autoOn();
    await h!.user.click(within(crop()).getByRole("button", { name: "Reset" }));
    expect(isOn()).toBe(false);
    expect(noteLines()).toEqual([]);
    expectFrame(FULL);
    expect(valueOf("Straighten")).toBe("0°");
    expect(within(crop()).getByRole("radio", { name: "Original" })).toHaveAttribute("aria-checked", "true");
  });

  it("pressed while on, it turns off and puts back the draft from just before it was turned on", async () => {
    const spy = await setup();
    await startCrop();
    // A draft of the person's own: Free, the right edge in by a tenth, turned 2°.
    await h!.user.click(within(crop()).getByRole("radio", { name: "Free" }));
    screen.getByRole("button", { name: /Crop right edge/ }).focus();
    await h!.user.keyboard("{Shift>}{ArrowLeft}{ArrowLeft}{/Shift}");
    const box = screen.getByRole("textbox", { name: "Straighten value" });
    await h!.user.click(box);
    await h!.user.clear(box);
    await h!.user.type(box, "2{Enter}");
    await waitFor(() => expect(valueOf("Straighten")).toBe("+2.0°"));
    await autoOn();
    expect(valueOf("Straighten")).toBe("+1.4°");
    spy.mockClear();
    await h!.user.click(autoToggle());
    expect(isOn()).toBe(false);
    expect(noteLines()).toEqual([]);
    // The frame set before (the right edge in by a tenth), shown inside the photo straightened by 2° (no empty corner, docs/11 §Crop).
    expectFrame(fitInside({ x: 0, y: 0, width: 0.9, height: 1 }, 2, canvasOf(8640, 5760, 0)));
    expect(valueOf("Straighten")).toBe("+2.0°");
    expect(within(crop()).getByRole("radio", { name: "Free" })).toHaveAttribute("aria-checked", "true");
    // The core is told the angle from before, as Straighten would tell it.
    expect(patches(spy).at(-1)).toEqual({ crop: { rect: FULL, angle_deg: 2, aspect: "free" }, orientation: 0 });
  });

  it("Done sends the crop with auto: true, one undo step; with Auto off the field is left out", async () => {
    const spy = await setup();
    await startCrop();
    await autoOn();
    spy.mockClear();
    await h!.user.click(within(crop()).getByRole("button", { name: /^Done/ }));
    await screen.findByRole("complementary", { name: "Tools" });
    const auto = firstFrame(proposal("original", 0));
    expect(patches(spy).at(-1)).toEqual({ crop: { rect: auto, angle_deg: 1.4, aspect: "original", auto: true }, orientation: 0 });
    await waitFor(() => expect(settings().settings.crop?.auto).toBe(true));
    expect(h!.store.getState().canUndo).toBe(true);
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(settings().settings.crop).toBeNull());
    // Auto on, then the frame moved by hand: the crop is the person's.
    await startCrop();
    await autoOn();
    frame().focus();
    await h!.user.keyboard("{ArrowLeft}");
    spy.mockClear();
    await h!.user.click(within(crop()).getByRole("button", { name: /^Done/ }));
    await screen.findByRole("complementary", { name: "Tools" });
    const sent = patches(spy).at(-1)?.crop;
    expect(sent).not.toBeNull();
    expect(sent).not.toHaveProperty("auto");
  });

  it("Cancel puts back the crop from before the panel opened, whatever Auto did", async () => {
    const spy = await setup();
    await startCrop();
    await autoOn();
    spy.mockClear();
    await h!.user.click(within(crop()).getByRole("button", { name: /^Cancel/ }));
    await screen.findByRole("complementary", { name: "Tools" });
    expect(patches(spy).at(-1)).toEqual({ crop: null, orientation: 0 });
    await waitFor(() => expect(settings().values.crop).toBeNull());
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("opens on with an Auto crop, asks again to fill the note, and off goes to the whole photo at 0° in the aspect chosen", async () => {
    await setup();
    const auto = proposal("square", 0);
    await h!.store.update({ crop: { rect: firstFrame(auto), angle_deg: auto.angle_deg, aspect: "square", auto: true } });
    const ask = vi.spyOn(h!.api, "suggestCrop");
    await startCrop();
    expect(isOn()).toBe(true);
    expect(ask).toHaveBeenCalledWith(currentId(), "square", 0);
    await waitFor(() => expect(noteLines()).toEqual(["Cropped to frame the subject.", "Levelled by 1.4°."]));
    // The answer is the same: the frame and the angle are where they were.
    expectFrame(firstFrame(auto));
    expect(valueOf("Straighten")).toBe("+1.4°");
    await h!.user.click(autoToggle());
    expect(isOn()).toBe(false);
    // The largest square of the 3:2 photo, straight.
    expectFrame({ x: 1 / 6, y: 0, width: 2 / 3, height: 1 });
    expect(valueOf("Straighten")).toBe("0°");
    expect(within(crop()).getByRole("radio", { name: "1:1" })).toHaveAttribute("aria-checked", "true");
  });

  it("A toggles it while the Crop panel is open, and Space and Return press it", async () => {
    await setup();
    // Outside the panel A does nothing.
    await h!.user.keyboard("a");
    expect(screen.queryByRole("complementary", { name: "Crop" })).toBeNull();
    await startCrop();
    await h!.user.keyboard("a");
    expect(isOn()).toBe(true);
    await waitFor(() => expect(noteLines()).toEqual(["Cropped to frame the subject.", "Levelled by 1.4°."]));
    await h!.user.keyboard("a");
    expect(isOn()).toBe(false);
    expectFrame(FULL);
    autoToggle().focus();
    await h!.user.keyboard(" ");
    expect(isOn()).toBe(true);
    await waitFor(() => expect(valueOf("Straighten")).toBe("+1.4°"));
    // Return on the focused toggle presses it, not Done.
    await h!.user.keyboard("{Enter}");
    expect(isOn()).toBe(false);
    expect(screen.getByRole("complementary", { name: "Crop" })).toBeInTheDocument();
  });

  it("is locked while the photo is developing", async () => {
    await setup();
    await startCrop();
    const ask = vi.spyOn(h!.api, "suggestCrop");
    act(() => h!.api.emitEvent("photo_progress", { id: currentId(), stage: "estimate", done: false, error: null }));
    await waitFor(() => expect(autoToggle()).toHaveAttribute("aria-disabled", "true"));
    // A locked toggle ignores the pointer (`pointer-events: none`); a click that reaches it all the same does nothing.
    fireEvent.click(autoToggle());
    await h!.user.keyboard("a");
    expect(isOn()).toBe(false);
    expect(ask).not.toHaveBeenCalled();
  });
});

describe("the note under the Frame card", () => {
  it("a photo kept whole and already level says so, and Done sends no crop", async () => {
    const spy = await setup({ photoOverrides: { DSC04204: { file_name: "DSC04204_keep.ARW" } } });
    await startCrop();
    await autoOn();
    expect(noteLines()).toEqual(["No better frame found, so the whole photo is kept.", "Already level."]);
    expectFrame(FULL);
    expect(valueOf("Straighten")).toBe("0°");
    spy.mockClear();
    await h!.user.click(within(crop()).getByRole("button", { name: /^Done/ }));
    await screen.findByRole("complementary", { name: "Tools" });
    // The whole photo, straight, is no crop, Auto or not.
    expect(patches(spy).at(-1)).toEqual({ crop: null, orientation: 0 });
  });

  it("a photo without a horizon or straight lines is not turned, and the note says so", async () => {
    await setup();
    await startCrop();
    const frameOnly = proposal("original", 0).frames[1]!;
    vi.spyOn(h!.api, "suggestCrop").mockResolvedValueOnce({ straighten: "unsure", angle_deg: 0, frames: [{ rect: { x: 0.1, y: 0.1, width: 0.8, height: 0.8 }, score: 0.12 }, frameOnly], keep: false, subject: "picture" });
    await autoOn();
    expect(noteLines()).toEqual(["Cropped to frame the subject.", "No horizon or straight lines found, so the photo is not turned."]);
    expectFrame({ x: 0.1, y: 0.1, width: 0.8, height: 0.8 });
    expect(valueOf("Straighten")).toBe("0°");
  });

  it("a request that fails turns Auto off and says so for 4 s, without the error bar", async () => {
    await setup();
    await startCrop();
    h!.api.failNext("suggestCrop", { scope: "render", message: "The picture could not be made." });
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    fireEvent.click(autoToggle());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(isOn()).toBe(false);
    expect(noteLines()).toEqual(["Couldn't suggest a frame for this photo."]);
    expectFrame(FULL);
    expect(AUTO_CROP_FAILED_MS).toBe(4000);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(AUTO_CROP_FAILED_MS - 1);
    });
    expect(noteLines()).toEqual(["Couldn't suggest a frame for this photo."]);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(noteLines()).toEqual([]);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("a press within the 4 s asks again and the failure line goes", async () => {
    await setup();
    await startCrop();
    const held = holdSuggestions();
    await h!.user.click(autoToggle());
    await held[0]!.fail();
    expect(noteLines()).toEqual(["Couldn't suggest a frame for this photo."]);
    await h!.user.click(autoToggle());
    expect(noteLines()).toEqual(["Finding a frame…"]);
    await held[1]!.answer();
    expect(noteLines()).toEqual(["Cropped to frame the subject.", "Levelled by 1.4°."]);
  });

  it("writes every line word for word", () => {
    const rotate: CropSuggestion = { straighten: "rotate", angle_deg: -1.4, frames: [{ rect: FULL, score: 0.9 }], keep: false, subject: "picture" };
    expect(autoCropLines(rotate)).toEqual(["Cropped to frame the subject.", "Levelled by 1.4°."]);
    expect(autoCropLines({ ...rotate, angle_deg: 7.25 })[1]).toBe("Levelled by 7.3°.");
    expect(autoCropLines({ ...rotate, straighten: "level", angle_deg: 0, keep: true })).toEqual([
      "No better frame found, so the whole photo is kept.",
      "Already level.",
    ]);
    expect(autoCropLines({ ...rotate, straighten: "unsure", angle_deg: 0 })[1]).toBe("No horizon or straight lines found, so the photo is not turned.");
    expect(levelledLine(0.25)).toBe("Levelled by 0.3°.");
    expect(autoCropNoteLines(null)).toEqual([]);
    expect(autoCropNoteLines({ kind: "asking" })).toEqual(["Finding a frame…"]);
    expect(autoCropNoteLines({ kind: "failed" })).toEqual(["Couldn't suggest a frame for this photo."]);
    expect(autoCropNoteLines({ kind: "answer", suggestion: rotate })).toEqual(autoCropLines(rotate));
  });

  it("draftToCrop says auto only while Auto is on, and the whole photo straight is no crop", () => {
    const rect = { x: 0.1, y: 0.1, width: 0.5, height: 0.5 };
    expect(draftToCrop({ rect, angle_deg: 1, aspect: "free", orientation: 0, auto: true })).toEqual({ rect, angle_deg: 1, aspect: "free", auto: true });
    expect(draftToCrop({ rect, angle_deg: 1, aspect: "free", orientation: 0, auto: false })).toEqual({ rect, angle_deg: 1, aspect: "free" });
    expect(draftToCrop({ rect, angle_deg: 1, aspect: "free", orientation: 0, auto: false })).not.toHaveProperty("auto");
    expect(draftToCrop({ rect: FULL, angle_deg: 0, aspect: "original", orientation: 0, auto: true })).toBeNull();
  });
});

describe("answers that come too late", () => {
  it("an answer after Auto was turned off is not used", async () => {
    const spy = await setup();
    await startCrop();
    const held = holdSuggestions();
    await h!.user.click(autoToggle());
    await h!.user.click(autoToggle());
    expect(isOn()).toBe(false);
    spy.mockClear();
    await held[0]!.answer();
    expect(isOn()).toBe(false);
    expect(noteLines()).toEqual([]);
    expectFrame(FULL);
    expect(valueOf("Straighten")).toBe("0°");
    expect(patches(spy).some((p) => p.crop?.angle_deg === 1.4)).toBe(false);
  });

  it("an answer after the frame was moved by hand is not used", async () => {
    await setup();
    await startCrop();
    const held = holdSuggestions();
    await h!.user.click(autoToggle());
    frame().focus();
    await h!.user.keyboard("{ArrowDown}");
    expect(isOn()).toBe(false);
    await held[0]!.answer();
    expect(isOn()).toBe(false);
    expect(valueOf("Straighten")).toBe("0°");
  });

  it("only the answer for the latest aspect counts, whatever order they arrive in", async () => {
    await setup();
    await startCrop();
    const held = holdSuggestions();
    await h!.user.click(autoToggle());
    await held[0]!.answer();
    await h!.user.click(within(crop()).getByRole("radio", { name: "1:1" }));
    await h!.user.click(within(crop()).getByRole("radio", { name: "16:9" }));
    expect(held.map((r) => r.aspect)).toEqual(["original", "square", "sixteen_by_nine"]);
    expect(noteLines()).toEqual(["Finding a frame…"]);
    await held[2]!.answer();
    expectFrame(firstFrame(proposal("sixteen_by_nine", 0)));
    await held[1]!.answer();
    expectFrame(firstFrame(proposal("sixteen_by_nine", 0)));
    expect(noteLines()).toEqual(["Cropped to frame the subject.", "Levelled by 1.4°."]);
    expect(isOn()).toBe(true);
  });

  it("an answer after the panel closed is not used, and the next panel starts clean", async () => {
    const spy = await setup();
    await startCrop();
    const held = holdSuggestions();
    await h!.user.click(autoToggle());
    await h!.user.click(within(crop()).getByRole("button", { name: /^Cancel/ }));
    await screen.findByRole("complementary", { name: "Tools" });
    spy.mockClear();
    await held[0]!.answer();
    expect(spy).not.toHaveBeenCalled();
    expect(settings().values.crop).toBeNull();
    await startCrop();
    expect(isOn()).toBe(false);
    expect(noteLines()).toEqual([]);
    expectFrame(FULL);
  });
});
