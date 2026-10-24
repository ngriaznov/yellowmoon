/**
 * The Look tab's Lens card (`docs/21` §1, `docs/11` *Lens card*): a lens choice, Flares, Edges and Anamorphic, and a note with the
 * lens's line. Every change sends the whole `lens` object; choosing a lens sends its starting Flares and Edges with it, in one
 * patch, and Anamorphic stays.
 */
import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { LensSettings, SettingsPatch } from "../api/types";
import { LENS_MISSING_NOTE, NO_LENS } from "./tools/LensCard";
import { capturingPointer } from "../test/pointer";
import { installManualRaf, mockRect } from "../test/raf";
import { openPhotos, renderApp, sliderName, sliderOf, valueOf } from "../test/harness";
import type { Harness, UpdateSpy } from "../test/harness";
import type { MockOptions } from "../api.mock";

let h: Harness;
afterEach(() => h?.unmount());

/** The seven lenses with their lines and starting amounts, word for word from `docs/21` §1 and `data/lenses/`. */
const LENSES = [
  { kind: "default", name: "Default", line: "Clean and modern: even sharpness, faint glow.", flares: 0, edges: 0 },
  { kind: "classic", name: "Classic", line: "Warm and gentle: soft glow, amber flares, smooth falloff.", flares: 35, edges: 30 },
  { kind: "vintage", name: "Vintage", line: "Glows wide open, lower contrast, soft edges, a little swirl.", flares: 45, edges: 40 },
  { kind: "petzval", name: "Petzval", line: "Very sharp centre, strong swirl, fast falloff.", flares: 30, edges: 70 },
  { kind: "triplet", name: "Triplet", line: "Soft corners, busy flares, a crisp centre.", flares: 50, edges: 35 },
  { kind: "sonnar", name: "Sonnar", line: "Smooth and glowing, with gentle edges.", flares: 40, edges: 35 },
  { kind: "detuned", name: "Detuned", line: "Sharp only in the centre, with smeared flares.", flares: 45, edges: 65 },
] as const;

async function setup(options: MockOptions = {}): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], ...options });
  await openPhotos(h);
  // The card enables its controls once the list has arrived.
  if (!options.lensesMissing) await waitFor(() => expect(lensBox()).toBeEnabled());
  else await waitFor(() => expect(h.store.getState().lenses).toEqual([]));
  return vi.spyOn(h.api, "updateSettings");
}

const card = (): ReturnType<typeof within> => within(screen.getByRole("region", { name: "Lens" }));
const lensBox = (): HTMLElement => screen.getByRole("combobox", { name: "Lens" });
const currentLens = (): LensSettings => h.store.currentSettings()!.settings.lens;

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

/** Opens the menu and clicks the lens called `name`. */
async function choose(name: string): Promise<void> {
  await h.user.click(lensBox());
  await h.user.click(await screen.findByRole("option", { name: new RegExp(`^${name}\\b`) }));
}

describe("the Lens card on the Look tab", () => {
  it("is the fourth card, under Tone, with a Lens choice and Flares, Edges and Anamorphic in that order", async () => {
    await setup();
    const regions = screen.getAllByRole("region");
    expect(regions.map((r) => within(r).getAllByRole("heading", { level: 3 })[0]?.textContent)).toEqual(["Automatic", "Look", "Tone", "Lens"]);
    const lens = screen.getByRole("region", { name: "Lens" });
    const tone = screen.getByRole("region", { name: "Tone" });
    expect(tone.compareDocumentPosition(lens) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // The choice sits above the sliders.
    const sliders = card().getAllByRole("slider");
    expect(sliders.map((s: HTMLElement) => sliderName(s))).toEqual(["Flares", "Edges", "Anamorphic"]);
    expect(lensBox().compareDocumentPosition(sliders[0]!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(card().getByRole("button", { name: "Reset" })).toBeInTheDocument();
  });

  it("starts as no lens: Default, with the three sliders at 0 from 0 to 100", async () => {
    await setup();
    expect(lensBox()).toHaveTextContent("Default");
    for (const label of ["Flares", "Edges", "Anamorphic"]) {
      const slider = sliderOf(label);
      expect([slider.getAttribute("aria-valuemin"), slider.getAttribute("aria-valuemax"), slider.getAttribute("aria-valuenow")], label).toEqual(["0", "100", "0"]);
      expect(valueOf(label)).toBe("0");
    }
    // Nothing is changed yet, so there is no Reset on a row.
    expect(card().queryByRole("button", { name: /^Reset (Flares|Edges|Anamorphic)$/ })).toBeNull();
  });

  it("lists the seven lenses in the menu, each with its line under its name, and shows only the name in the box", async () => {
    await setup();
    const box = lensBox();
    await h.user.click(box);
    const options = await screen.findAllByRole("option");
    expect(options).toHaveLength(7);
    expect(options.map((o) => o.querySelector(".ym-lens-option__name")?.textContent)).toEqual(LENSES.map((l) => l.name));
    expect(options.map((o) => o.querySelector(".ym-lens-option__line")?.textContent)).toEqual(LENSES.map((l) => l.line));
    // The chosen one carries the check; the box itself reads just "Default".
    expect(options[0]).toHaveAttribute("aria-selected", "true");
    expect(box).toHaveTextContent(/^Default$/);
  });

  it("shows the chosen lens's line in a note under the sliders", async () => {
    await setup();
    const note = (): HTMLElement => card().getByText(LENSES[0].line);
    expect(note()).toHaveClass("ym-row-note");
    for (const slider of card().getAllByRole("slider")) {
      expect(slider.compareDocumentPosition(note()) & Node.DOCUMENT_POSITION_FOLLOWING, sliderName(slider)).toBeTruthy();
    }
    await choose("Petzval");
    await waitFor(() => expect(card().getByText("Very sharp centre, strong swirl, fast falloff.")).toHaveClass("ym-row-note"));
    expect(card().queryByText(LENSES[0].line)).toBeNull();
    // The labels and notes name no camera maker (AGENTS rule 3).
    expect(screen.getByRole("region", { name: "Lens" }).textContent).not.toMatch(/ARRI|ALEXA/i);
  });

  it("choosing Vintage sends one patch with the kind and Vintage's starting amounts, and keeps Anamorphic", async () => {
    const spy = await setup();
    await type("Anamorphic", "25");
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "default", flares: 0, edges: 0, anamorphic: 25 } });
    spy.mockClear();
    await choose("Vintage");
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    expect(spy.mock.calls[0]?.[1]).toEqual({ lens: { kind: "vintage", flares: 45, edges: 40, anamorphic: 25 } });
    // The card follows at once: the box, the amounts and the note.
    await waitFor(() => expect(lensBox()).toHaveTextContent("Vintage"));
    expect(valueOf("Flares")).toBe("45");
    expect(valueOf("Edges")).toBe("40");
    expect(valueOf("Anamorphic")).toBe("25");
    expect(card().getByText("Glows wide open, lower contrast, soft edges, a little swirl.")).toBeInTheDocument();
    expect(currentLens()).toEqual({ kind: "vintage", flares: 45, edges: 40, anamorphic: 25 });
    // Every other lens sends its own starting amounts the same way.
    spy.mockClear();
    for (const { kind, name, flares, edges } of LENSES) {
      await choose(name);
      await waitFor(() => expect(spy.mock.calls.at(-1)?.[1]).toEqual({ lens: { kind, flares, edges, anamorphic: 25 } }));
    }
    expect(spy).toHaveBeenCalledTimes(LENSES.length);
  });

  it("a lens choice is one undo step, and one redo", async () => {
    const spy = await setup();
    await type("Anamorphic", "25");
    await waitFor(() => expect(valueOf("Anamorphic")).toBe("25"));
    spy.mockClear();
    await choose("Triplet");
    await waitFor(() => expect(currentLens()).toEqual({ kind: "triplet", flares: 50, edges: 35, anamorphic: 25 }));
    await h.user.keyboard("{Meta>}z{/Meta}");
    // One step back: the lens and both amounts it moved, with the earlier Anamorphic still in place.
    await waitFor(() => expect(currentLens()).toEqual({ kind: "default", flares: 0, edges: 0, anamorphic: 25 }));
    expect(lensBox()).toHaveTextContent("Default");
    expect(valueOf("Flares")).toBe("0");
    await h.user.keyboard("{Meta>}{Shift>}z{/Shift}{/Meta}");
    await waitFor(() => expect(currentLens()).toEqual({ kind: "triplet", flares: 50, edges: 35, anamorphic: 25 }));
    // And the step before it is the Anamorphic value, nothing in between.
    await h.user.keyboard("{Meta>}z{/Meta}");
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(currentLens()).toEqual(NO_LENS));
    expect(h.store.getState().canUndo).toBe(false);
  });

  it("a typed value is a patch with the whole lens object, the other parts as they are", async () => {
    const spy = await setup();
    await choose("Sonnar");
    await waitFor(() => expect(currentLens().kind).toBe("sonnar"));
    spy.mockClear();
    await type("Edges", "80");
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "sonnar", flares: 40, edges: 80, anamorphic: 0 } });
    await waitFor(() => expect(valueOf("Edges")).toBe("80"));
    spy.mockClear();
    await type("Flares", "100");
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "sonnar", flares: 100, edges: 80, anamorphic: 0 } });
    // A key steps by one.
    spy.mockClear();
    sliderOf("Anamorphic").focus();
    await h.user.keyboard("{ArrowRight}");
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "sonnar", flares: 100, edges: 80, anamorphic: 1 } });
    expect(valueOf("Anamorphic")).toBe("1");
  });

  it("a drag sends the whole lens object once per frame and again on release, as one undo step", async () => {
    const spy = await setup();
    await choose("Classic");
    await waitFor(() => expect(currentLens()).toEqual({ kind: "classic", flares: 35, edges: 30, anamorphic: 0 }));
    const raf = installManualRaf();
    try {
      const root = sliderOf("Flares").closest('[data-slot="slider"]') as HTMLElement;
      mockRect(root, { left: 100, top: 0, width: 200, height: 38 });
      const pointer = capturingPointer(root);
      spy.mockClear();
      pointer.down(170); // 35
      pointer.move(200);
      pointer.move(230);
      pointer.move(260); // 80
      await act(async () => {
        raf.flush();
      });
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
      expect(spy.mock.calls[0]?.[1]).toEqual({ lens: { kind: "classic", flares: 80, edges: 30, anamorphic: 0 } });
      pointer.up(260);
      await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
      expect(spy.mock.calls[1]?.[1]).toEqual({ lens: { kind: "classic", flares: 80, edges: 30, anamorphic: 0 } });
      await waitFor(() => expect(valueOf("Flares")).toBe("80"));
    } finally {
      raf.restore();
    }
    // The drag is one step: undo goes back to Classic's start, not to the middle of the drag.
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(currentLens()).toEqual({ kind: "classic", flares: 35, edges: 30, anamorphic: 0 }));
    expect(valueOf("Flares")).toBe("35");
  });

  it("a row's Reset puts that amount back to 0 and sends the whole lens object", async () => {
    const spy = await setup();
    await choose("Petzval");
    await waitFor(() => expect(currentLens()).toEqual({ kind: "petzval", flares: 30, edges: 70, anamorphic: 0 }));
    spy.mockClear();
    await h.user.click(card().getByRole("button", { name: "Reset Edges" }));
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "petzval", flares: 30, edges: 0, anamorphic: 0 } });
    await waitFor(() => expect(valueOf("Edges")).toBe("0"));
    expect(valueOf("Flares")).toBe("30");
    expect(lensBox()).toHaveTextContent("Petzval");
  });

  it("the card's Reset goes to Default with the three sliders at 0", async () => {
    const spy = await setup();
    await choose("Detuned");
    await type("Anamorphic", "60");
    await waitFor(() => expect(currentLens()).toEqual({ kind: "detuned", flares: 45, edges: 65, anamorphic: 60 }));
    spy.mockClear();
    await h.user.click(card().getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "default", flares: 0, edges: 0, anamorphic: 0 } });
    await waitFor(() => expect(currentLens()).toEqual(NO_LENS));
    expect(lensBox()).toHaveTextContent("Default");
    for (const label of ["Flares", "Edges", "Anamorphic"]) expect(valueOf(label)).toBe("0");
    expect(card().getByText(LENSES[0].line)).toBeInTheDocument();
    // One undo step brings the whole lens back.
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(currentLens()).toEqual({ kind: "detuned", flares: 45, edges: 65, anamorphic: 60 }));
  });

  it("shows the lens of the photo on screen, and a drag or a choice reaches only that photo", async () => {
    const spy = await setup();
    const { session } = h.store.getState();
    const shown = session!.photos[session!.current_index!]!;
    const other = session!.photos.find((p) => p.id !== shown.id)!;
    await choose("Triplet");
    await waitFor(() => expect(currentLens().kind).toBe("triplet"));
    expect(spy.mock.calls.at(-1)?.[0]).toBe(shown.id);
    // Another photo has its own lens (none), and the first keeps Triplet.
    await act(async () => {
      await h.store.setCurrent(other.id);
    });
    await waitFor(() => expect(lensBox()).toHaveTextContent("Default"));
    expect(valueOf("Flares")).toBe("0");
    await choose("Sonnar");
    await waitFor(() => expect(spy.mock.calls.at(-1)?.[0]).toBe(other.id));
    await act(async () => {
      await h.store.setCurrent(shown.id);
    });
    await waitFor(() => expect(lensBox()).toHaveTextContent("Triplet"));
    expect(valueOf("Flares")).toBe("50");
  });
});

describe("the Lens card while the lens data are missing", () => {
  it("shows the card with the menu and the three sliders off, and says why", async () => {
    await setup({ lensesMissing: true });
    expect(screen.getByRole("region", { name: "Lens" })).toBeInTheDocument();
    expect(lensBox()).toBeDisabled();
    for (const label of ["Flares", "Edges", "Anamorphic"]) expect(sliderOf(label)).toHaveAttribute("data-disabled");
    expect(card().getByText(LENS_MISSING_NOTE)).toBeInTheDocument();
    expect(LENS_MISSING_NOTE).toBe("Lens data are missing, so lenses can't be used.");
    // The box still reads the photo's lens.
    expect(lensBox()).toHaveTextContent("Default");
  });

  it("sends nothing from the sliders, and the card's Reset still works", async () => {
    const spy = await setup({ lensesMissing: true });
    const box = screen.getByRole("textbox", { name: "Flares value" });
    expect(box).toBeDisabled();
    spy.mockClear();
    await h.user.click(card().getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "default", flares: 0, edges: 0, anamorphic: 0 } });
  });

  it("does not ask for the list again when the card is drawn again", async () => {
    h = renderApp({ current: 3, picked: [2, 3, 4] });
    const spy = vi.spyOn(h.api, "lensList");
    await openPhotos(h);
    await waitFor(() => expect(lensBox()).toBeEnabled());
    await h.user.click(screen.getByRole("tab", { name: "Light" }));
    await h.user.click(screen.getByRole("tab", { name: "Look" }));
    await waitFor(() => expect(lensBox()).toBeEnabled());
    expect(spy).not.toHaveBeenCalled();
    expect(h.store.getState().lenses).toHaveLength(7);
  });
});
