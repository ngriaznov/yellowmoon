/**
 * The Lens tab (`docs/21` §1, `docs/11` *Lens card*): the Lens card (the seven lenses as a list of rows, each with its line) and the
 * Amounts card (Flares, Edges and Anamorphic, and the card's Reset). Every change sends the whole `lens` object; choosing a lens sends its
 * starting Flares and Edges with it, in one patch, and Anamorphic stays.
 */
import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { LensSettings, SettingsPatch } from "../api/types";
import { LENS_MISSING_NOTE, NO_LENS } from "./tools/LensCard";
import type { LensKind } from "../api/types";
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
  await h.user.click(screen.getByRole("tab", { name: "Lens" }));
  // The card enables its rows once the list has arrived.
  if (!options.lensesMissing) await waitFor(() => expect(lensGroup()).toBeEnabled());
  else await waitFor(() => expect(h.store.getState().lenses).toEqual([]));
  return vi.spyOn(h.api, "updateSettings");
}

const lensCard = (): HTMLElement => screen.getByRole("region", { name: "Lens" });
const card = (): ReturnType<typeof within> => within(lensCard());
const amounts = (): ReturnType<typeof within> => within(screen.getByRole("region", { name: "Amounts" }));
const lensGroup = (): HTMLElement => screen.getByRole("radiogroup", { name: "Lens" });
const lensRows = (): HTMLElement[] => within(lensGroup()).getAllByRole("radio");
/** The radio of the lens called `name`; its accessible name is the name followed by its line. */
const lensRadio = (name: string): HTMLElement => within(lensGroup()).getByRole("radio", { name: new RegExp(`^${name}\\b`) });
/** The name of the lens that is chosen. */
const chosen = (): string | undefined =>
  lensRows()
    .find((r) => r.getAttribute("aria-checked") === "true")
    ?.closest("label")
    ?.querySelector(".ym-choice-row__name")?.textContent ?? undefined;
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

/** Clicks the row of the lens called `name`. */
async function choose(name: string): Promise<void> {
  await h.user.click(lensRadio(name));
}

describe("the Lens tab", () => {
  it("has two cards, Lens and Amounts, and the Look tab has no Lens card any more", async () => {
    await setup();
    const regions = screen.getAllByRole("region");
    expect(regions.map((r) => within(r).getAllByRole("heading", { level: 3 })[0]?.textContent)).toEqual(["Lens", "Amounts"]);
    expect(screen.getByRole("heading", { level: 2, name: "Lens" })).toBeInTheDocument();
    expect(screen.getByRole("tabpanel")).toHaveAttribute("data-tab", "lens");
    // The list comes first, the sliders after it.
    const sliders = amounts().getAllByRole("slider");
    expect(sliders.map((s: HTMLElement) => sliderName(s))).toEqual(["Flares", "Edges", "Anamorphic"]);
    expect(lensGroup().compareDocumentPosition(sliders[0]!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // The card's Reset is the Amounts card's; the Lens card has none.
    expect(amounts().getByRole("button", { name: "Reset" })).toBeInTheDocument();
    expect(card().queryByRole("button", { name: "Reset" })).toBeNull();
    await h.user.click(screen.getByRole("tab", { name: "Look" }));
    expect(await screen.findByRole("region", { name: "Tone" })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Lens" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Amounts" })).toBeNull();
    expect(screen.queryByRole("radiogroup", { name: "Lens" })).toBeNull();
  });

  it("starts as no lens: Default chosen, with the three sliders at 0 from 0 to 100", async () => {
    await setup();
    expect(chosen()).toBe("Default");
    for (const label of ["Flares", "Edges", "Anamorphic"]) {
      const slider = sliderOf(label);
      expect([slider.getAttribute("aria-valuemin"), slider.getAttribute("aria-valuemax"), slider.getAttribute("aria-valuenow")], label).toEqual(["0", "100", "0"]);
      expect(valueOf(label)).toBe("0");
    }
    // Nothing is changed yet, so there is no Reset on a row.
    expect(amounts().queryByRole("button", { name: /^Reset (Flares|Edges|Anamorphic)$/ })).toBeNull();
  });

  it("lists the seven lenses as rows in the order of lens_list, each with its line under its name", async () => {
    await setup();
    const rows = lensRows();
    expect(rows).toHaveLength(7);
    const labels = rows.map((r) => r.closest("label")!);
    expect(labels.map((l) => l.querySelector(".ym-choice-row__name")?.textContent)).toEqual(LENSES.map((l) => l.name));
    expect(labels.map((l) => l.querySelector(".ym-choice-row__line")?.textContent)).toEqual(LENSES.map((l) => l.line));
    // A row's accessible name is the name and then the line.
    expect(rows.map((r) => r.getAttribute("aria-checked"))).toEqual(["true", "false", "false", "false", "false", "false", "false"]);
    expect(lensRadio("Vintage")).toHaveAccessibleName("Vintage Glows wide open, lower contrast, soft edges, a little swirl.");
    // The line is the quieter text, under the name.
    const line = labels[0]!.querySelector(".ym-choice-row__line")!;
    const name = labels[0]!.querySelector(".ym-choice-row__name")!;
    expect(name.compareDocumentPosition(line) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // The labels name no camera maker (AGENTS rule 3).
    expect(lensCard().textContent).not.toMatch(/ARRI|ALEXA/i);
  });

  it("has one Tab stop in the list, on the chosen row, and the arrows choose the next lens", async () => {
    const spy = await setup();
    // From the tab, one Tab reaches the list (on the chosen row), and the next one leaves it for the Amounts card's Reset.
    screen.getByRole("tab", { name: "Lens" }).focus();
    await h.user.tab();
    expect(lensRadio("Default")).toHaveFocus();
    await h.user.tab();
    expect(amounts().getByRole("button", { name: "Reset" })).toHaveFocus();
    await h.user.tab({ shift: true });
    expect(lensRadio("Default")).toHaveFocus();
    spy.mockClear();
    // Radix chooses the row an arrow focused while the key is down, and moves the focus a tick after the key goes down.
    await h.user.keyboard("{ArrowDown>}");
    await waitFor(() => expect(lensRadio("Classic")).toHaveFocus());
    await waitFor(() => expect(currentLens().kind).toBe("classic"));
    await h.user.keyboard("{/ArrowDown}");
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0]?.[1]).toEqual({ lens: { kind: "classic", flares: 35, edges: 30, anamorphic: 0 } });
    expect(chosen()).toBe("Classic");
    expect(lensRadio("Classic")).toHaveAttribute("tabindex", "0");
  });

  it("shows each lens's line in its row only: the Amounts card has no copy of it", async () => {
    await setup();
    expect(card().getByText(LENSES[0].line)).toBeInTheDocument();
    expect(amounts().queryByText(LENSES[0].line)).toBeNull();
    await choose("Petzval");
    await waitFor(() => expect(chosen()).toBe("Petzval"));
    expect(amounts().queryByText("Very sharp centre, strong swirl, fast falloff.")).toBeNull();
    expect(screen.getAllByText("Very sharp centre, strong swirl, fast falloff.")).toHaveLength(1);
    expect(amounts().queryByText(/./, { selector: ".ym-row-note" })).toBeNull();
  });

  it("choosing Vintage sends one patch with the kind and Vintage's starting amounts, and keeps Anamorphic", async () => {
    const spy = await setup();
    await type("Anamorphic", "25");
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "default", flares: 0, edges: 0, anamorphic: 25 } });
    spy.mockClear();
    await choose("Vintage");
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    expect(spy.mock.calls[0]?.[1]).toEqual({ lens: { kind: "vintage", flares: 45, edges: 40, anamorphic: 25 } });
    // The tab follows at once: the chosen row and the amounts.
    await waitFor(() => expect(chosen()).toBe("Vintage"));
    expect(lensRadio("Vintage")).toHaveAttribute("aria-checked", "true");
    expect(lensRadio("Default")).toHaveAttribute("aria-checked", "false");
    expect(valueOf("Flares")).toBe("45");
    expect(valueOf("Edges")).toBe("40");
    expect(valueOf("Anamorphic")).toBe("25");
    expect(currentLens()).toEqual({ kind: "vintage", flares: 45, edges: 40, anamorphic: 25 });
    // Every other lens sends its own starting amounts the same way.
    spy.mockClear();
    for (const { kind, name, flares, edges } of LENSES) {
      await choose(name);
      await waitFor(() => expect(spy.mock.calls.at(-1)?.[1]).toEqual({ lens: { kind, flares, edges, anamorphic: 25 } }));
    }
    // Vintage was chosen already, so its click was not a change: six patches for the other six.
    expect(spy).toHaveBeenCalledTimes(LENSES.length);
  });

  it("choosing the lens that is chosen sends nothing", async () => {
    const spy = await setup();
    await choose("Triplet");
    await waitFor(() => expect(currentLens().kind).toBe("triplet"));
    spy.mockClear();
    await choose("Triplet");
    expect(spy).not.toHaveBeenCalled();
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
    expect(chosen()).toBe("Default");
    expect(valueOf("Flares")).toBe("0");
    await h.user.keyboard("{Meta>}{Shift>}z{/Shift}{/Meta}");
    await waitFor(() => expect(currentLens()).toEqual({ kind: "triplet", flares: 50, edges: 35, anamorphic: 25 }));
    expect(chosen()).toBe("Triplet");
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
    await h.user.click(amounts().getByRole("button", { name: "Reset Edges" }));
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "petzval", flares: 30, edges: 0, anamorphic: 0 } });
    await waitFor(() => expect(valueOf("Edges")).toBe("0"));
    expect(valueOf("Flares")).toBe("30");
    expect(chosen()).toBe("Petzval");
  });

  it("the Amounts card's Reset goes to Default with the three sliders at 0", async () => {
    const spy = await setup();
    await choose("Detuned");
    await type("Anamorphic", "60");
    await waitFor(() => expect(currentLens()).toEqual({ kind: "detuned", flares: 45, edges: 65, anamorphic: 60 }));
    spy.mockClear();
    await h.user.click(amounts().getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "default", flares: 0, edges: 0, anamorphic: 0 } });
    await waitFor(() => expect(currentLens()).toEqual(NO_LENS));
    expect(chosen()).toBe("Default");
    for (const label of ["Flares", "Edges", "Anamorphic"]) expect(valueOf(label)).toBe("0");
    // One undo step brings the whole lens back.
    await h.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(currentLens()).toEqual({ kind: "detuned", flares: 45, edges: 65, anamorphic: 60 }));
    expect(chosen()).toBe("Detuned");
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
    await waitFor(() => expect(chosen()).toBe("Default"));
    expect(valueOf("Flares")).toBe("0");
    await choose("Sonnar");
    await waitFor(() => expect(spy.mock.calls.at(-1)?.[0]).toBe(other.id));
    await act(async () => {
      await h.store.setCurrent(shown.id);
    });
    await waitFor(() => expect(chosen()).toBe("Triplet"));
    expect(valueOf("Flares")).toBe("50");
  });

  it("every lens kind of lens_list has a row, in the order of the list", async () => {
    await setup();
    const kinds = h.store.getState().lenses!.map((l) => l.kind) satisfies LensKind[];
    expect(kinds).toEqual(LENSES.map((l) => l.kind));
    expect(lensRows().map((r) => r.getAttribute("value"))).toEqual(kinds);
  });
});

describe("the Lens tab while the lens data are missing", () => {
  it("shows both cards with one row off for the photo's lens and the sliders off, and says why", async () => {
    await setup({ lensesMissing: true });
    expect(screen.getByRole("region", { name: "Lens" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Amounts" })).toBeInTheDocument();
    // The row names the photo's own lens, with no line, and nothing can be chosen.
    expect(lensRows()).toHaveLength(1);
    expect(lensRows()[0]).toBeDisabled();
    expect(chosen()).toBe("Default");
    for (const label of ["Flares", "Edges", "Anamorphic"]) expect(sliderOf(label)).toHaveAttribute("data-disabled");
    expect(card().getByText(LENS_MISSING_NOTE)).toBeInTheDocument();
    expect(LENS_MISSING_NOTE).toBe("Lens data are missing, so lenses can't be used.");
  });

  it("sends nothing from the sliders, and the Amounts card's Reset still works", async () => {
    const spy = await setup({ lensesMissing: true });
    const box = screen.getByRole("textbox", { name: "Flares value" });
    expect(box).toBeDisabled();
    spy.mockClear();
    await h.user.click(amounts().getByRole("button", { name: "Reset" }));
    expect(await lastPatch(spy)).toEqual({ lens: { kind: "default", flares: 0, edges: 0, anamorphic: 0 } });
  });

  it("does not ask for the list again when the tab is drawn again", async () => {
    h = renderApp({ current: 3, picked: [2, 3, 4] });
    const spy = vi.spyOn(h.api, "lensList");
    await openPhotos(h);
    await h.user.click(screen.getByRole("tab", { name: "Lens" }));
    await waitFor(() => expect(lensGroup()).toBeEnabled());
    await h.user.click(screen.getByRole("tab", { name: "Light" }));
    await h.user.click(screen.getByRole("tab", { name: "Lens" }));
    await waitFor(() => expect(lensGroup()).toBeEnabled());
    expect(spy).not.toHaveBeenCalled();
    expect(h.store.getState().lenses).toHaveLength(7);
  });
});
