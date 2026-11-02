/**
 * Pick white in the Colour tab (`docs/11` §Pick white, `docs/23` §8): the header has no button for it, the White balance card has a row
 * of its own, the hint is a pill on the photo, W opens the Colour tab and starts it (W again or Esc ends it), and it is off the moment the
 * Colour tab is left, as Pick colour is, and exclusive with Pick colour. What a click does (the point goes to the core) is in
 * `CropAndPickWhite.test.tsx`.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mockRect } from "../test/raf";
import { openPhotos, openTab, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";
import { PICK_WHITE_HINT } from "./PickWhiteHint";

let h: Harness;
afterEach(() => h?.unmount());

/** DSC04204 (the mockup's photo) on the Look tab, with Pick white off. */
async function setup(): Promise<void> {
  h = renderApp({ current: 3, picked: [2, 3, 4] });
  await openPhotos(h);
}

const photo = (): HTMLElement => screen.getByRole("main", { name: "Photo" });
const card = () => within(screen.getByRole("region", { name: "White balance" }));
const toggle = (): HTMLElement => card().getByRole("button", { name: "Pick white" });
const layer = (): HTMLElement | null => within(photo()).queryByRole("application", { name: /^Pick white\./ });
const pill = (): HTMLElement | null => within(photo()).queryByRole("status");

/** Pick white on, from the card's toggle on the Colour tab, with the layer given a size. */
async function startPicking(): Promise<HTMLElement> {
  if (screen.getByRole("tab", { name: "Colour" }).getAttribute("aria-selected") !== "true") await openTab(h, "Colour");
  await h.user.click(toggle());
  const el = await within(photo()).findByRole("application", { name: /^Pick white\./ });
  mockRect(el, { left: 0, top: 0, width: 900, height: 600 });
  return el;
}

describe("the row in the White balance card", () => {
  it("sits between the light presets and the Kelvin slider, with an eyedropper, a name, a description and the key as a chip", async () => {
    await setup();
    await openTab(h, "Colour");
    const white = screen.getByRole("region", { name: "White balance" });
    const lights = within(white).getByRole("radiogroup", { name: "Light type" });
    const kelvin = within(white).getByRole("slider", { name: "Kelvin" });
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    expect(toggle()).toHaveAccessibleDescription("Click something neutral on the photo (W)");
    expect(toggle().querySelector("svg")).toHaveAttribute("aria-hidden", "true");
    expect(toggle()).toHaveAttribute("data-slot", "toggle");
    // Document order: presets, the row, Kelvin.
    expect(lights.compareDocumentPosition(toggle()) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(toggle().compareDocumentPosition(kelvin) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // "or press W": text and a glasscn key chip, hidden from assistive technology (the key is in the description above).
    const keys = white.querySelector(".ym-pick-row__keys");
    expect(keys).toHaveTextContent("or press W");
    expect(keys).toHaveAttribute("aria-hidden", "true");
    const chip = keys?.querySelector("kbd");
    expect(chip).toHaveAttribute("data-slot", "kbd");
    expect(chip).toHaveTextContent(/^W$/);
    // The card header keeps its Auto and Reset.
    expect(within(white).getByRole("button", { name: "Auto", pressed: true })).toBeInTheDocument();
    expect(within(white).getByRole("button", { name: "Reset" })).toBeInTheDocument();
  });

  it("is not a header button: the header's cells are Before, Crop, Nodes, Add photos and Fit, on every tab", async () => {
    await setup();
    for (const name of ["Look", "Colour", "RAW"]) {
      await openTab(h, name);
      const labels = within(screen.getByRole("navigation", { name: "Actions" })).getAllByRole("button").map((b) => b.textContent);
      expect(labels, name).toEqual(["Before", "Crop", "Nodes", "Add photos", expect.stringMatching(/^Fit/)]);
    }
  });

  it("is pressed while picking, and the toggle, W and Esc end it", async () => {
    await setup();
    await startPicking();
    expect(toggle()).toHaveAttribute("aria-pressed", "true");
    expect(layer()).not.toBeNull();
    await h.user.click(toggle());
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    expect(layer()).toBeNull();
    // W starts it on the Colour tab itself, without leaving the tab, and W again ends it.
    await h.user.keyboard("w");
    expect(toggle()).toHaveAttribute("aria-pressed", "true");
    expect(layer()).not.toBeNull();
    await h.user.keyboard("w");
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    expect(layer()).toBeNull();
    expect(screen.getByRole("tab", { name: "Colour" })).toHaveAttribute("aria-selected", "true");
    await h.user.keyboard("w");
    await h.user.keyboard("{Escape}");
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    expect(layer()).toBeNull();
  });
});

describe("the hint on the photo", () => {
  it("is a polite status pill with an eyedropper and the sentence, only while Pick white is on", async () => {
    await setup();
    await openTab(h, "Colour");
    expect(pill()).toBeNull();
    await startPicking();
    const status = pill()!;
    expect(PICK_WHITE_HINT).toBe("Click something neutral: grey or white. Esc cancels.");
    expect(status).toHaveTextContent(PICK_WHITE_HINT);
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status).toHaveAttribute("data-slot", "badge");
    expect(status.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
    // A pill of the cell, not a header text, and out of the picture that a screen reader describes.
    expect(status.closest("main")).toBe(photo());
    expect(status.closest('[role="img"]')).toBeNull();
    await h.user.keyboard("{Escape}");
    expect(pill()).toBeNull();
  });

  it("is not Pick colour's: that one says its words in the header, and no pill shows", async () => {
    await setup();
    await openTab(h, "Colour");
    await h.user.click(within(screen.getByRole("region", { name: "Colour warp" })).getByRole("button", { name: "Pick colour" }));
    expect(await screen.findByText("Click a colour in the photo.")).toBeInTheDocument();
    expect(pill()).toBeNull();
  });
});

describe("the crosshair", () => {
  it("is white with a dark outline under it, centred on the pointer", async () => {
    await setup();
    const el = await startPicking();
    expect(el.querySelector("svg")).toBeNull();
    fireEvent.pointerMove(el, { clientX: 450, clientY: 150 });
    const mark = el.querySelector("svg.ym-pick-layer__mark")!;
    expect(mark).toHaveAttribute("aria-hidden", "true");
    expect((mark as SVGElement).style.left).toBe("50%");
    expect((mark as SVGElement).style.top).toBe("25%");
    const outline = mark.querySelector('[data-part="outline"]')!;
    const line = mark.querySelector('[data-part="line"]')!;
    expect(line.getAttribute("stroke")).toBe("#fff");
    expect(outline.getAttribute("stroke")).toMatch(/^rgb\(0 0 0/);
    // The outline is the wider stroke and is drawn first, so the white line lies over it.
    expect(Number(outline.getAttribute("stroke-width"))).toBeGreaterThan(Number(line.getAttribute("stroke-width")));
    expect(outline.compareDocumentPosition(line) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // The cursor itself is the crosshair (the stylesheet's rule is read in `e2e`; here the layer says which mode it is).
    expect(el).toHaveAttribute("data-mode", "white");
  });
});

describe("it turns off when the Colour tab is left", () => {
  it("by a click on a tab or by ⌘1 to ⌘7: the layer and the pill go at once, a click on the photo picks nothing, and Colour comes back with it off", async () => {
    await setup();
    const input = vi.spyOn(h.api, "viewportInput");
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
      expect(pill(), what).not.toBeNull();
      await go();
      // At once: no waiting for anything.
      expect(layer(), what).toBeNull();
      expect(within(photo()).queryByRole("application"), what).toBeNull();
      expect(pill(), what).toBeNull();
      fireEvent.click(photo(), { clientX: 450, clientY: 100 });
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(input, what).not.toHaveBeenCalled();
      // Back on Colour (⌘4): off.
      await h.user.keyboard("{Meta>}4{/Meta}");
      expect(toggle(), what).toHaveAttribute("aria-pressed", "false");
      expect(layer(), what).toBeNull();
      expect(pill(), what).toBeNull();
    }
    // Staying on Colour keeps it on: ⌘4 and a click on the Colour tab do nothing to it.
    await startPicking();
    await h.user.keyboard("{Meta>}4{/Meta}");
    await h.user.click(screen.getByRole("tab", { name: "Colour" }));
    expect(toggle()).toHaveAttribute("aria-pressed", "true");
    expect(layer()).not.toBeNull();
  });

  it("by Crop (the button, or C): the crop starts, and Colour is back with it off once the crop is over", async () => {
    await setup();
    await startPicking();
    await h.user.click(screen.getByRole("button", { name: "Crop" }));
    await screen.findByRole("complementary", { name: "Crop" });
    expect(within(photo()).queryByRole("application", { name: /^Pick white\./ })).toBeNull();
    expect(pill()).toBeNull();
    await h.user.keyboard("{Escape}");
    await screen.findByRole("complementary", { name: "Tools" });
    expect(screen.getByRole("tab", { name: "Colour" })).toHaveAttribute("aria-selected", "true");
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    expect(layer()).toBeNull();
    await startPicking();
    await h.user.keyboard("c");
    await screen.findByRole("complementary", { name: "Crop" });
    await h.user.keyboard("{Escape}");
    await screen.findByRole("complementary", { name: "Tools" });
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
  });

  it("by the Export panel, and back with it off", async () => {
    await setup();
    await startPicking();
    act(() => h.store.openExport());
    await waitFor(() => expect(within(photo()).queryByRole("application")).toBeNull());
    expect(pill()).toBeNull();
    act(() => h.store.closeExport());
    await waitFor(() => expect(toggle()).toHaveAttribute("aria-pressed", "false"));
    expect(layer()).toBeNull();
  });

  it("by another photo, whichever way: the filmstrip, the store, and the arrow keys with the focus on the toggle", async () => {
    await setup();
    await startPicking();
    await h.user.click(within(screen.getByRole("navigation", { name: "Photos" })).getByRole("button", { name: /^Show DSC04206/ }));
    await waitFor(() => expect(layer()).toBeNull());
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    expect(pill()).toBeNull();
    await startPicking();
    await act(() => h.store.setCurrent(5));
    await waitFor(() => expect(layer()).toBeNull());
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    // The arrows move the crosshair while the layer has the focus; from the toggle they go to the next photo.
    await startPicking();
    act(() => toggle().focus());
    const id = h.store.currentPhoto()?.id;
    await h.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(h.store.currentPhoto()?.id).not.toBe(id));
    expect(layer()).toBeNull();
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
  });

  it("by leaving the Photos screen: Start, and back", async () => {
    await setup();
    await startPicking();
    act(() => h.store.go("start"));
    await waitFor(() => expect(screen.queryByRole("main", { name: "Photo" })).toBeNull());
    act(() => h.store.go("photos"));
    await screen.findByRole("main", { name: "Photo" });
    expect(layer()).toBeNull();
    await openTab(h, "Colour");
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
  });
});

describe("W", () => {
  it("opens the Colour tab from any other tab and starts picking", async () => {
    await setup();
    for (const name of ["Look", "Lens", "Light", "Detail", "Curve", "RAW"]) {
      await openTab(h, name);
      await h.user.keyboard("w");
      expect(screen.getByRole("tab", { name: "Colour" }), name).toHaveAttribute("aria-selected", "true");
      expect(toggle(), name).toHaveAttribute("aria-pressed", "true");
      expect(layer(), name).not.toBeNull();
      expect(pill(), name).not.toBeNull();
      // Leaving ends it (the next lap starts from a tab other than Colour).
      await h.user.keyboard("{Escape}");
    }
  });

  it("opens the Colour tab from the Export panel too, as ⌘4 does", async () => {
    await setup();
    act(() => h.store.openExport());
    await screen.findByRole("complementary", { name: "Export" });
    await h.user.keyboard("w");
    await screen.findByRole("complementary", { name: "Tools" });
    expect(screen.getByRole("tab", { name: "Colour" })).toHaveAttribute("aria-selected", "true");
    expect(toggle()).toHaveAttribute("aria-pressed", "true");
    expect(layer()).not.toBeNull();
  });

  it("is not a letter typed into a text box, or held down", async () => {
    await setup();
    await openTab(h, "Look");
    await h.user.click(screen.getByRole("searchbox", { name: "Search looks" }));
    await h.user.keyboard("w");
    expect(layer()).toBeNull();
    expect(screen.getByRole("tab", { name: "Look" })).toHaveAttribute("aria-selected", "true");
    // A held key repeats: one press, one change.
    act(() => (document.activeElement as HTMLElement | null)?.blur());
    fireEvent.keyDown(document.body, { key: "w" });
    await within(photo()).findByRole("application", { name: /^Pick white\./ });
    fireEvent.keyDown(document.body, { key: "w", repeat: true });
    fireEvent.keyDown(document.body, { key: "w", repeat: true });
    expect(layer()).not.toBeNull();
  });

  it("does not start on a file that can't be opened: there is nothing to adjust", async () => {
    h = renderApp({ current: 1, picked: [], skipped: { DSC04202: "This file is from a camera mode we can't read yet." } });
    await openPhotos(h);
    // The current photo is the first one that is ready, so go to the broken one by hand.
    await h.user.click(await screen.findByRole("button", { name: "Show DSC04202, can't open" }));
    expect(await screen.findByText("This file can't be opened.")).toBeInTheDocument();
    await h.user.keyboard("w");
    expect(within(photo()).queryByRole("application")).toBeNull();
    expect(screen.queryByText(PICK_WHITE_HINT)).toBeNull();
    expect(screen.getByRole("tab", { name: "Look" })).toHaveAttribute("aria-selected", "true");
  });
});

describe("Pick white and Pick colour are exclusive", () => {
  it("W while Pick colour is on switches to Pick white; Pick colour's toggle then switches back", async () => {
    await setup();
    await openTab(h, "Colour");
    const colour = () => within(screen.getByRole("region", { name: "Colour warp" })).getByRole("button", { name: "Pick colour" });
    await h.user.click(colour());
    expect(await within(photo()).findByRole("application", { name: /^Pick colour\./ })).toBeInTheDocument();
    await h.user.keyboard("w");
    expect(within(photo()).getByRole("application", { name: /^Pick white\./ })).toBeInTheDocument();
    expect(colour()).toHaveAttribute("aria-pressed", "false");
    expect(toggle()).toHaveAttribute("aria-pressed", "true");
    expect(pill()).not.toBeNull();
    await h.user.click(colour());
    expect(within(photo()).getByRole("application", { name: /^Pick colour\./ })).toBeInTheDocument();
    expect(toggle()).toHaveAttribute("aria-pressed", "false");
    expect(pill()).toBeNull();
  });
});
