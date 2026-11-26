/**
 * The Reference Grade node's interface (`docs/34` §2.1, §9, RG-12): the Add node item, the seven tabs, the Reference tab in each state of §9.3 (the status
 * line, the notes, the errors, Fit with progress and Cancel, Strength, Skin protection), the strip chooser, the Hue tab and the Elements tab. The screen runs on
 * the mock, as in a browser; the mock's fit is a fixed one.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import type { NodeLink, SettingsPatch } from "../../api/types";
import { outputPatch, nodeOf } from "../../state/settingsGraph";
import type { ReferenceGradeItem } from "../../state/settingsGraph";
import { REFERENCE_TEXT, cardKey } from "../../state/reference";
import { openPhotos, openTab, renderApp, tools } from "../../test/harness";
import type { Harness, UpdateSpy } from "../../test/harness";

let h: Harness | undefined;
let windowHeight = window.innerHeight;
beforeEach(() => {
  windowHeight = window.innerHeight;
  Object.defineProperty(window, "innerHeight", { value: 900, configurable: true, writable: true });
});
afterEach(() => {
  h?.unmount();
  h = undefined;
  Object.defineProperty(window, "innerHeight", { value: windowHeight, configurable: true, writable: true });
});

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

/** Main, then a Reference Grade node (2), then the Output. */
const WITH_MATCH = (more: Record<string, unknown> = {}): SettingsPatch => ({
  nodes: { "2": { kind: "reference_grade", name: "Match", pos: [1, 0], ...more } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

/** DSC04204 in the node view with the Reference Grade node selected, on its Reference tab. */
async function setup(settings: SettingsPatch = WITH_MATCH(), options: MockOptions = {}): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: settings }, ...options });
  await openPhotos(h);
  await openTab(h, "Light");
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  fireEvent.click(screen.getByRole("button", { name: /^Node 2, Match/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /^Node 2, Match/ })).toHaveAttribute("aria-pressed", "true"));
  return vi.spyOn(h.api, "updateSettings");
}

const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const node = (): ReferenceGradeItem => nodeOf(h!.store.currentSettings()!.settings, 2) as ReferenceGradeItem;
const photoId = (): number => h!.store.currentSettings()!.photo_id;
const cardOf = () => h!.store.getState().references[cardKey(photoId(), 2)];
const statusGroup = (): HTMLElement => tools().querySelector<HTMLElement>(".ym-ref__status")!;
const fitButton = (): HTMLElement => within(tools()).getByRole("button", { name: /^Fit/ });
const chooseButton = (): HTMLElement => within(tools()).getByRole("button", { name: "Choose reference…" });
const tabs = (): string[] => within(tools()).getAllByRole("tab").map((t) => t.textContent ?? "");

/** Chooses a file: the mock answers with `path` (queued), standing in for the system's open panel. */
async function chooseFile(path?: string): Promise<void> {
  if (path !== undefined) h!.api.queueChoice("reference", path);
  await h!.user.click(chooseButton());
  await h!.user.click(await screen.findByRole("menuitem", { name: "Image file…" }));
}

describe("Add node > Reference Grade", () => {
  it("adds a node named Match, selects it and opens its Reference tab, and asks for no file", async () => {
    h = renderApp({ current: 3, picked: [2, 3, 4] });
    const dialog = vi.spyOn(h.api, "referenceChooseFile");
    await openPhotos(h);
    await openTab(h, "Light");
    await h.user.keyboard("n");
    await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
    await h.user.click(screen.getByRole("button", { name: "Add node" }));
    const menu = await screen.findByRole("menu", { name: "Add node" });
    await h.user.click(within(menu).getByRole("menuitem", { name: /^Reference Grade/ }));
    await waitFor(() => expect(screen.getByRole("button", { name: /^Node 2, Match/ })).toHaveAttribute("aria-pressed", "true"));
    expect(tabs()).toEqual(["Reference", "Light", "Colour", "Curve", "Hue", "Elements", "Mask"]);
    expect(within(tools()).getByRole("tab", { name: "Reference" })).toHaveAttribute("aria-selected", "true");
    expect(dialog).not.toHaveBeenCalled();
    const head = within(tools()).getByRole("group", { name: "Node" });
    expect(head.querySelector(".ym-nodehead__line")?.textContent).toBe("Reference Grade");
  });

  it("⌘1 to ⌘7 select the seven tabs in the order of the row", async () => {
    await setup();
    const order = ["Reference", "Light", "Colour", "Curve", "Hue", "Elements", "Mask"];
    for (const [i, name] of order.entries()) {
      await h!.user.keyboard(`{Meta>}${i + 1}{/Meta}`);
      await waitFor(() => expect(within(tools()).getByRole("tab", { name })).toHaveAttribute("aria-selected", "true"));
    }
  });
});

describe("the Reference tab with no reference", () => {
  it("has no card title, a placeholder with its name, Fit off with its tooltip, Strength at 100 %, Skin protection on, and the first status line", async () => {
    await setup();
    expect(within(tools()).queryAllByRole("heading", { level: 3 })).toEqual([]);
    expect(within(tools()).getByRole("region", { name: "Reference" })).toBeInTheDocument();
    expect(within(tools()).getByTestId("reference-name")).toHaveTextContent("No reference");
    expect(fitButton()).toHaveAttribute("aria-disabled", "true");
    expect(fitButton()).toHaveTextContent("Fit");
    expect(screen.getByRole("textbox", { name: "Strength value" })).toHaveValue("100%");
    expect(within(tools()).getByRole("switch", { name: "Skin protection" })).toBeChecked();
    expect(within(tools()).getByText(REFERENCE_TEXT.strengthLine)).toBeInTheDocument();
    expect(within(tools()).getByText(REFERENCE_TEXT.skinLine)).toBeInTheDocument();
    expect(statusGroup()).toHaveTextContent("Choose a picture to match.");
    expect(statusGroup()).toHaveAttribute("role", "status");
    expect(statusGroup()).toHaveAttribute("aria-live", "polite");
    // A press on the disabled Fit does nothing.
    const spy = vi.spyOn(h!.api, "fitReference");
    await h!.user.click(fitButton());
    expect(spy).not.toHaveBeenCalled();
  });

  it("explains the disabled Fit with its tooltip", async () => {
    await setup();
    await h!.user.hover(fitButton());
    expect(await screen.findAllByText("Choose a reference first.")).not.toHaveLength(0);
  });
});

describe("choosing a file and fitting", () => {
  it("shows the file's name and picture and makes Fit the prominent button; nothing about the file is in the settings", async () => {
    await setup();
    await chooseFile("~/Pictures/harbour-dusk.jpg");
    await waitFor(() => expect(within(tools()).getByTestId("reference-name")).toHaveTextContent("harbour-dusk.jpg"));
    expect(within(tools()).getByTestId("reference-name").previousElementSibling?.querySelector("img")).toBeTruthy();
    expect(statusGroup()).toHaveTextContent("Press Fit to match the photo to this picture.");
    expect(fitButton()).not.toHaveAttribute("aria-disabled", "true");
    expect(fitButton()).toHaveAttribute("data-variant", "primary");
    expect(JSON.stringify(h!.store.currentSettings()!.settings)).not.toContain("harbour");
  });

  it("does nothing when the open panel is cancelled", async () => {
    await setup();
    h!.api.queueChoice("reference", null);
    await chooseFile();
    expect(within(tools()).getByTestId("reference-name")).toHaveTextContent("No reference");
    expect(statusGroup()).toHaveTextContent("Choose a picture to match.");
    expect(within(tools()).queryByTestId("reference-error")).toBeNull();
  });

  it("fits: the progress bar and Cancel take the Fit row, the step is the status line, the picture is unchanged until the end, and Fit again follows", async () => {
    await setup(WITH_MATCH(), { speed: 0.02 });
    await chooseFile();
    await waitFor(() => expect(fitButton()).not.toHaveAttribute("aria-disabled", "true"));
    const before = JSON.stringify(h!.store.currentSettings()!.settings);
    await h!.user.click(fitButton());
    await waitFor(() => expect(within(tools()).getByRole("progressbar", { name: "Fitting" })).toBeInTheDocument());
    expect(chooseButton()).toHaveAttribute("aria-disabled", "true");
    expect(within(tools()).getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(within(tools()).queryByRole("button", { name: /^Fit/ })).toBeNull();
    await waitFor(() => expect(statusGroup()).toHaveTextContent("Finding what is in the pictures…"));
    for (const k of [1, 2, 3]) {
      act(() => h!.api.emitEvent("fit_progress", { photo_id: photoId(), node_id: 2, step: "search", search: k, of: 3, fraction: k / 3 }));
      await waitFor(() => expect(statusGroup()).toHaveTextContent(`Fitting… ${k} of 3`));
    }
    expect(JSON.stringify(h!.store.currentSettings()!.settings)).toBe(before);
    h!.api.configure({ speed: Number.POSITIVE_INFINITY });
    h!.api.flush();
  });

  it("ends with the fit written as one patch, 'Fitted to the reference.' and Fit again, which one undo takes back", async () => {
    await setup();
    await chooseFile();
    await waitFor(() => expect(fitButton()).not.toHaveAttribute("aria-disabled", "true"));
    const before = JSON.stringify(node());
    await h!.user.click(fitButton());
    await waitFor(() => expect(statusGroup()).toHaveTextContent("Fitted to the reference."));
    expect(fitButton()).toHaveTextContent("Fit again");
    expect(fitButton()).not.toHaveAttribute("data-accent");
    expect(fitButton()).toHaveAttribute("data-variant", "header");
    expect(node().hue_curves.hue_vs_hue).toHaveLength(6);
    expect(node().elements.sky.hue_deg).toBe(-14);
    expect(within(tools()).queryByRole("progressbar")).toBeNull();
    // One undo step for the whole fit.
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(JSON.stringify(node())).toBe(before));
  });

  it("Cancel changes nothing, and so does Esc", async () => {
    await setup(WITH_MATCH(), { speed: 0.02 });
    for (const how of ["button", "esc"] as const) {
      await chooseFile();
      await waitFor(() => expect(fitButton()).not.toHaveAttribute("aria-disabled", "true"));
      const before = JSON.stringify(h!.store.currentSettings()!.settings);
      await h!.user.click(fitButton());
      await waitFor(() => expect(within(tools()).getByRole("progressbar", { name: "Fitting" })).toBeInTheDocument());
      if (how === "button") await h!.user.click(within(tools()).getByRole("button", { name: "Cancel" }));
      else await h!.user.keyboard("{Escape}");
      await waitFor(() => expect(within(tools()).queryByRole("progressbar")).toBeNull());
      expect(JSON.stringify(h!.store.currentSettings()!.settings)).toBe(before);
      expect(statusGroup()).toHaveTextContent("Press Fit to match the photo to this picture.");
      expect(within(tools()).queryByTestId("reference-error")).toBeNull();
    }
  });
});

describe("the status line after a fit", () => {
  async function fitted(): Promise<void> {
    await setup();
    await chooseFile();
    await waitFor(() => expect(fitButton()).not.toHaveAttribute("aria-disabled", "true"));
    await h!.user.click(fitButton());
    await waitFor(() => expect(statusGroup()).toHaveTextContent("Fitted to the reference."));
  }

  it("says the look changed when the Output's look or its intensity changed, and gives Fit again the accent; nothing is re-fitted by itself", async () => {
    await fitted();
    const before = JSON.stringify(node());
    await act(async () => {
      await h!.store.update(outputPatch({ look_intensity: 50 }), true);
    });
    await waitFor(() => expect(statusGroup()).toHaveTextContent("The look changed since the fit. Fit again to match it with the new look."));
    expect(fitButton()).toHaveAttribute("data-accent");
    expect(fitButton()).toHaveAttribute("data-variant", "primary");
    expect(JSON.stringify(node())).toBe(before);
    // Fit again clears it.
    await h!.user.click(fitButton());
    await waitFor(() => expect(statusGroup()).toHaveTextContent("Fitted to the reference."));
    expect(fitButton()).not.toHaveAttribute("data-accent");
  });

  it("says it was edited by hand when a fitted value moved, with no accent; Strength is not an edit", async () => {
    await fitted();
    await act(async () => {
      await h!.store.update({ nodes: { "2": { strength: 60 } } }, true);
    });
    expect(statusGroup()).toHaveTextContent("Fitted to the reference.");
    await act(async () => {
      await h!.store.update({ nodes: { "2": { elements: { vegetation: { hue_deg: 32, chroma: 0.6 } } } } }, true);
    });
    await waitFor(() => expect(statusGroup()).toHaveTextContent("Edited by hand since the fit. Fit again to start over."));
    expect(fitButton()).not.toHaveAttribute("data-accent");
  });

  it("a node that holds values and no reference says so", async () => {
    await setup(WITH_MATCH({ hue_curves: { hue_vs_hue: [{ hue_deg: 120, value: 12 }] } }));
    expect(statusGroup()).toHaveTextContent("The values are kept. Choose a reference to fit again.");
    expect(fitButton()).toHaveAttribute("aria-disabled", "true");
  });

  it("choosing another reference starts again at 'Press Fit', and the values stay", async () => {
    await fitted();
    await chooseFile("~/Pictures/other.png");
    await waitFor(() => expect(statusGroup()).toHaveTextContent("Press Fit to match the photo to this picture."));
    expect(node().hue_curves.hue_vs_hue).toHaveLength(6);
  });
});

describe("the notes and the errors", () => {
  it("show the notes of the reference, each on its own line, under the status", async () => {
    await setup();
    await chooseFile("~/Pictures/bars-profile-grey.png");
    await waitFor(() => expect(within(tools()).getAllByRole("note")).toHaveLength(3));
    expect(within(tools()).getAllByRole("note").map((n) => n.textContent)).toEqual([REFERENCE_TEXT.blackBars, REFERENCE_TEXT.profileUnreadable, REFERENCE_TEXT.noColour]);
  });

  it("says that Film and Blur nodes after this one are not part of the fit", async () => {
    await setup({ ...WITH_MATCH(), nodes: { "2": { kind: "reference_grade", name: "Match", pos: [1, 0] }, "3": { kind: "blur", pos: [2, 0] } }, links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, "output")] });
    expect(within(tools()).getByRole("note")).toHaveTextContent("Nodes after this one that blur or add film aren't part of the fit.");
  });

  it.each([
    ["damaged.png", "This file can't be read. It may be damaged. Choose another picture."],
    ["photo.heic", "Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again."],
    ["tiny.png", "This picture is too small to match. Use one that is at least 256 pixels on its short side."],
    ["empty-bars.png", "There's no picture inside the black bars. Choose another picture."],
  ])("a file named %s is refused with its sentence, in the card, with no code", async (file, text) => {
    await setup();
    await chooseFile(`~/Pictures/${file}`);
    await waitFor(() => expect(within(tools()).getByTestId("reference-error")).toHaveTextContent(text));
    expect(within(tools()).getByTestId("reference-error").textContent).toBe(text);
    expect(statusGroup()).toHaveTextContent("Choose a picture to match.");
    expect(fitButton()).toHaveAttribute("aria-disabled", "true");
  });

  it("a fit that failed says so, and the reference stays", async () => {
    await setup();
    await chooseFile();
    await waitFor(() => expect(fitButton()).not.toHaveAttribute("aria-disabled", "true"));
    h!.api.failNext("fitReference", { scope: "render", message: REFERENCE_TEXT.fitFailed });
    await h!.user.click(fitButton());
    await waitFor(() => expect(within(tools()).getByTestId("reference-error")).toHaveTextContent("The grade couldn't be fitted. Try another reference."));
    expect(within(tools()).getByTestId("reference-name")).toHaveTextContent("harbour-dusk.jpg");
    expect(fitButton()).not.toHaveAttribute("aria-disabled", "true");
  });

  it("a missing scene model is an error at Fit, and then a note while element grades are set", async () => {
    await setup(WITH_MATCH({ elements: { sky: { exposure_ev: -0.3 } } }));
    await chooseFile();
    await waitFor(() => expect(fitButton()).not.toHaveAttribute("aria-disabled", "true"));
    h!.api.failNext("fitReference", { scope: "render", message: REFERENCE_TEXT.sceneMissing });
    await h!.user.click(fitButton());
    await waitFor(() => expect(within(tools()).getByTestId("reference-error")).toHaveTextContent("The scene model is missing from this install. Reinstall Yellowmoon."));
    expect(within(tools()).getByRole("note")).toHaveTextContent("The scene model is missing from this install, so element grades aren't applied. Reinstall Yellowmoon.");
  });

  it("an error goes at the next choice", async () => {
    await setup();
    await chooseFile("~/Pictures/damaged.png");
    await waitFor(() => expect(within(tools()).getByTestId("reference-error")).toBeInTheDocument());
    await chooseFile("~/Pictures/ok.png");
    await waitFor(() => expect(within(tools()).queryByTestId("reference-error")).toBeNull());
  });
});

describe("Strength and Skin protection", () => {
  it("Strength sends the node's strength, has a Reset below 100 and none at 100", async () => {
    const spy = await setup();
    const box = screen.getByRole("textbox", { name: "Strength value" });
    expect(within(tools()).queryByRole("button", { name: "Reset Strength" })).toBeNull();
    await h!.user.click(box);
    await h!.user.clear(box);
    await h!.user.type(box, "40{Enter}");
    await waitFor(() => expect(node().strength).toBe(40));
    expect(patches(spy).at(-1)).toEqual({ nodes: { "2": { strength: 40 } } });
    await waitFor(() => expect(within(tools()).getByRole("button", { name: "Reset Strength" })).toBeInTheDocument());
    await h!.user.click(within(tools()).getByRole("button", { name: "Reset Strength" }));
    await waitFor(() => expect(node().strength).toBe(100));
  });

  it("Skin protection is a switch that sends the node's skin_protection and changes no pixel by itself", async () => {
    const spy = await setup();
    await h!.user.click(within(tools()).getByRole("switch", { name: "Skin protection" }));
    await waitFor(() => expect(node().skin_protection).toBe(false));
    expect(patches(spy).at(-1)).toEqual({ nodes: { "2": { skin_protection: false } } });
    expect(within(tools()).getByRole("switch", { name: "Skin protection" })).not.toBeChecked();
  });

  it("Reset this node puts every field back (Strength 100, Skin protection on) in one undo step", async () => {
    await setup(WITH_MATCH({ strength: 30, skin_protection: false, hue_curves: { hue_vs_sat: [{ hue_deg: 10, value: 1.2 }] } }));
    await h!.user.click(within(tools()).getByRole("button", { name: "Reset this node" }));
    await waitFor(() => expect(node().strength).toBe(100));
    expect(node().skin_protection).toBe(true);
    expect(node().hue_curves.hue_vs_sat).toEqual([]);
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(node().strength).toBe(30));
  });
});

describe("the strip as a chooser", () => {
  const strip = (): HTMLElement => screen.getByRole("navigation", { name: "Choose a reference" });
  async function open(options: MockOptions = {}): Promise<void> {
    await setup(WITH_MATCH(), options);
    await h!.user.click(chooseButton());
    await h!.user.click(await screen.findByRole("menuitem", { name: "Photo in the strip" }));
  }

  it("takes the graph's place, says what to do, and puts the graded photo off with 'This photo'", async () => {
    await open();
    expect(strip()).toBeInTheDocument();
    expect(screen.getByText("Click a photo in the strip. Esc cancels.")).toBeInTheDocument();
    expect(document.querySelector<HTMLElement>(".ym-graph__canvas")).toHaveAttribute("hidden");
    expect(chooseButton()).toHaveAttribute("aria-pressed", "true");
    const tiles = within(strip()).getAllByRole("button");
    expect(tiles).toHaveLength(12);
    const own = tiles.find((t) => /DSC04204/.test(t.getAttribute("aria-label") ?? ""))!;
    expect(own).toHaveAttribute("aria-disabled", "true");
    expect(own).toHaveTextContent("This photo");
    // The graph's toolbar is dimmed and inert.
    expect(screen.getByRole("group", { name: "Graph tools" })).toHaveAttribute("inert");
  });

  it("takes the keyboard: the focus is on the first photo that can be chosen, and the arrows move it without choosing", async () => {
    await open();
    const tiles = within(strip()).getAllByRole("button");
    await waitFor(() => expect(tiles[0]).toHaveFocus());
    await h!.user.keyboard("{ArrowRight}");
    expect(tiles[1]).toHaveFocus();
    expect(h!.store.getState().choosing).not.toBeNull();
  });

  it("chooses a photo with a click or with Return: its place in the strip is its name, and the graph comes back", async () => {
    await open();
    await h!.user.click(within(strip()).getAllByRole("button")[4]!);
    await waitFor(() => expect(within(tools()).getByTestId("reference-name")).toHaveTextContent("Photo 5 of 12"));
    expect(screen.queryByRole("navigation", { name: "Choose a reference" })).toBeNull();
    expect(document.querySelector<HTMLElement>(".ym-graph__canvas")).not.toHaveAttribute("hidden");
    expect(statusGroup()).toHaveTextContent("Press Fit to match the photo to this picture.");
    // With the keyboard.
    await h!.user.click(chooseButton());
    await h!.user.click(await screen.findByRole("menuitem", { name: "Photo in the strip" }));
    await waitFor(() => expect(within(strip()).getAllByRole("button")[0]).toHaveFocus());
    await h!.user.keyboard("{Enter}");
    await waitFor(() => expect(within(tools()).getByTestId("reference-name")).toHaveTextContent("Photo 1 of 12"));
  });

  it("is ended by Esc and chooses nothing; a click on the photo being graded or on one that can't be opened chooses nothing", async () => {
    await open({ skipped: { DSC04207: "This file can't be opened." } });
    const tiles = within(strip()).getAllByRole("button");
    const own = tiles.find((t) => /DSC04204/.test(t.getAttribute("aria-label") ?? ""))!;
    const broken = tiles.find((t) => /DSC04207/.test(t.getAttribute("aria-label") ?? ""))!;
    expect(broken).toHaveAttribute("aria-disabled", "true");
    expect(broken.getAttribute("aria-label")).toContain("This photo isn't ready yet.");
    await h!.user.click(own);
    await h!.user.click(broken);
    expect(h!.store.getState().choosing).not.toBeNull();
    expect(within(tools()).getByTestId("reference-name")).toHaveTextContent("No reference");
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("navigation", { name: "Choose a reference" })).toBeNull());
    expect(h!.store.getState().choosing).toBeNull();
    expect(within(tools()).getByTestId("reference-name")).toHaveTextContent("No reference");
    expect(chooseButton()).toHaveAttribute("aria-pressed", "false");
  });

  it("ends with another tab or another node", async () => {
    await open();
    await openTab(h!, "Light");
    await waitFor(() => expect(h!.store.getState().choosing).toBeNull());
    expect(screen.queryByRole("navigation", { name: "Choose a reference" })).toBeNull();
  });

  it("the photo being graded is refused by the core with its sentence (a file or a photo that is the photo)", async () => {
    await setup();
    h!.store.startChoosingReference(2);
    await h!.store.chooseReferencePhoto(photoId());
    expect(cardOf()?.error).toBe("That's the photo you're grading. Choose another.");
    expect(cardOf()?.info).toBeNull();
  });
});

describe("the Hue tab", () => {
  const FIT_CURVES = {
    hue_curves: {
      hue_vs_hue: [0, 60, 120, 180, 240, 300].map((hue_deg, i) => ({ hue_deg, value: [-4, 6, 12, 4, -3, 2][i] as number })),
      hue_vs_sat: [0, 60, 120, 180, 240, 300].map((hue_deg, i) => ({ hue_deg, value: [1.05, 1.08, 0.85, 1.12, 1.06, 0.92][i] as number })),
    },
  };
  const plot = (name: string): HTMLElement => within(tools()).getByRole("group", { name });
  const points = (name: string): HTMLElement[] => within(plot(name)).getAllByRole("slider");

  it("has two cards on the Curve plot, each with its own Reset, and opens zoomed on the fitted values", async () => {
    await setup(WITH_MATCH(FIT_CURVES));
    await openTab(h!, "Hue");
    expect(within(tools()).getAllByRole("heading", { level: 3 }).map((e) => e.textContent)).toEqual(["Hue vs hue", "Hue vs saturation"]);
    expect(points("Hue vs hue curve")).toHaveLength(6);
    expect(points("Hue vs saturation curve")).toHaveLength(6);
    // Zoomed: the top of the turn axis is +15°, not +180°; the full range is one press away.
    const first = within(tools()).getAllByRole("heading", { level: 3 })[0]!.closest("section") as HTMLElement;
    expect(first.querySelector(".ym-hue__yaxis")?.textContent).toContain("+15°");
    expect(first.querySelector(".ym-hue__yaxis")?.textContent).not.toContain("+180°");
    await h!.user.click(within(first).getByRole("button", { name: "Full range" }));
    expect(first.querySelector(".ym-hue__yaxis")?.textContent).toContain("+180°");
    expect(within(first).getByRole("button", { name: "Full range" })).toHaveAttribute("aria-pressed", "true");
    expect(within(tools()).getByRole("button", { name: "Reset Hue vs hue" })).toBeInTheDocument();
    expect(within(tools()).getByRole("button", { name: "Reset Hue vs saturation" })).toBeInTheDocument();
  });

  it("moves a point with the arrow keys (a degree of hue, a tenth of a degree of turn), Shift ×10, as one patch of one curve", async () => {
    const spy = await setup(WITH_MATCH(FIT_CURVES));
    await openTab(h!, "Hue");
    const third = points("Hue vs hue curve")[2]!;
    third.focus();
    spy.mockClear();
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(node().hue_curves.hue_vs_hue[2]).toEqual({ hue_deg: 121, value: 12 }));
    await h!.user.keyboard("{Shift>}{ArrowUp}{/Shift}");
    await waitFor(() => expect(node().hue_curves.hue_vs_hue[2]).toEqual({ hue_deg: 121, value: 13 }));
    const last = patches(spy).at(-1)!.nodes!["2"]!;
    expect(Object.keys(last)).toEqual(["hue_curves"]);
    expect(Object.keys(last.hue_curves!)).toEqual(["hue_vs_hue"]);
    expect(node().hue_curves.hue_vs_sat).toHaveLength(6);
  });

  it("names a point for a screen reader with its hue and its turn", async () => {
    await setup(WITH_MATCH(FIT_CURVES));
    await openTab(h!, "Hue");
    expect(points("Hue vs hue curve")[2]).toHaveAttribute("aria-valuetext", "Hue 120°, turn +12°");
    expect(points("Hue vs saturation curve")[2]).toHaveAttribute("aria-valuetext", "Hue 120°, saturation 85%");
  });

  it("removes a point with Delete and with a double click, adds one with Return on the plot, and says at 16 points that a curve can have no more", async () => {
    await setup(WITH_MATCH(FIT_CURVES));
    await openTab(h!, "Hue");
    points("Hue vs hue curve")[1]!.focus();
    await h!.user.keyboard("{Delete}");
    await waitFor(() => expect(node().hue_curves.hue_vs_hue).toHaveLength(5));
    await h!.user.dblClick(points("Hue vs hue curve")[0]!);
    await waitFor(() => expect(node().hue_curves.hue_vs_hue).toHaveLength(4));
    plot("Hue vs hue curve").focus();
    await h!.user.keyboard("{Enter}");
    await waitFor(() => expect(node().hue_curves.hue_vs_hue).toHaveLength(5));
    // Fill to 16, then ask for one more.
    const sixteen = Array.from({ length: 16 }, (_, i) => ({ hue_deg: i * 22, value: 0 }));
    await act(async () => {
      await h!.store.update({ nodes: { "2": { hue_curves: { hue_vs_hue: sixteen } } } }, true);
    });
    await waitFor(() => expect(points("Hue vs hue curve")).toHaveLength(16));
    plot("Hue vs hue curve").focus();
    await h!.user.keyboard("{Enter}");
    expect(await within(tools()).findByText("A curve can have up to 16 points.")).toBeInTheDocument();
    expect(node().hue_curves.hue_vs_hue).toHaveLength(16);
  });

  it("Reset empties one curve and leaves the other", async () => {
    await setup(WITH_MATCH(FIT_CURVES));
    await openTab(h!, "Hue");
    await h!.user.click(within(tools()).getByRole("button", { name: "Reset Hue vs hue" }));
    await waitFor(() => expect(node().hue_curves.hue_vs_hue).toEqual([]));
    expect(node().hue_curves.hue_vs_sat).toHaveLength(6);
  });
});

describe("the Elements tab", () => {
  const SKY = { elements: { sky: { exposure_ev: -0.45, hue_deg: -14, chroma: 1.25 }, subject: { exposure_ev: 0.2 } } };
  const card = (name: string): HTMLElement => within(tools()).getByRole("region", { name });

  it("has seven cards in the order the grades run; the Subject has Exposure only", async () => {
    await setup(WITH_MATCH(SKY));
    await openTab(h!, "Elements");
    expect(within(tools()).getAllByRole("heading", { level: 3 }).map((e) => e.textContent)).toEqual(["Sky", "Vegetation", "Water", "Ground", "Built", "Skin", "Subject"]);
    for (const name of ["Sky", "Vegetation", "Water", "Ground", "Built", "Skin"]) {
      expect(within(card(name)).getAllByRole("slider").map((s) => s.getAttribute("aria-label") ?? s.getAttribute("aria-labelledby") ? true : false)).toHaveLength(3);
    }
    expect(within(card("Subject")).getAllByRole("slider")).toHaveLength(1);
    expect(within(card("Sky")).getByRole("textbox", { name: "Exposure (EV) value" })).toHaveValue("−0.45");
    expect(within(card("Sky")).getByRole("textbox", { name: "Hue value" })).toHaveValue("−14°");
    expect(within(card("Sky")).getByRole("textbox", { name: "Chroma value" })).toHaveValue("125%");
    expect(within(card("Subject")).getByRole("textbox", { name: "Exposure (EV) value" })).toHaveValue("+0.20");
  });

  it("sends one element's field as a partial of the elements, chroma as a ratio, and has a Reset for the card and for each row", async () => {
    const spy = await setup(WITH_MATCH(SKY));
    await openTab(h!, "Elements");
    const chroma = within(card("Sky")).getByRole("textbox", { name: "Chroma value" });
    spy.mockClear();
    await h!.user.click(chroma);
    await h!.user.clear(chroma);
    await h!.user.type(chroma, "150{Enter}");
    await waitFor(() => expect(node().elements.sky.chroma).toBe(1.5));
    expect(patches(spy).at(-1)).toEqual({ nodes: { "2": { elements: { sky: { chroma: 1.5 } } } } });
    await h!.user.click(within(card("Sky")).getByRole("button", { name: "Reset Sky" }));
    await waitFor(() => expect(node().elements.sky).toEqual({ exposure_ev: 0, hue_deg: 0, chroma: 1 }));
    expect(within(card("Sky")).queryByRole("button", { name: "Reset Sky" })).toBeNull();
    expect(node().elements.subject.exposure_ev).toBe(0.2);
  });

  it("holds Hue to the element's own cap", async () => {
    await setup(WITH_MATCH());
    await openTab(h!, "Elements");
    const skin = within(card("Skin")).getByRole("slider", { name: "Hue" });
    expect(skin).toHaveAttribute("aria-valuemax", "15");
    expect(within(card("Sky")).getByRole("slider", { name: "Hue" })).toHaveAttribute("aria-valuemax", "40");
    expect(within(card("Built")).getByRole("slider", { name: "Hue" })).toHaveAttribute("aria-valuemax", "25");
  });

  it("says nothing of shares before a fit; after one it shows each element's share and greys the one that was not found, its sliders still working", async () => {
    await setup();
    await openTab(h!, "Elements");
    expect(within(tools()).queryByText(/% of the photo/)).toBeNull();
    await openTab(h!, "Reference");
    await chooseFile();
    await waitFor(() => expect(fitButton()).not.toHaveAttribute("aria-disabled", "true"));
    await h!.user.click(fitButton());
    await waitFor(() => expect(statusGroup()).toHaveTextContent("Fitted to the reference."));
    await openTab(h!, "Elements");
    expect(within(card("Sky")).getByText("26 % of the photo")).toBeInTheDocument();
    expect(within(card("Vegetation")).getByText("29 % of the photo")).toBeInTheDocument();
    expect(within(card("Water")).getByText("Not found in this photo.")).toBeInTheDocument();
    expect(within(card("Water")).queryByText(/% of the photo/)).toBeNull();
    const slider = within(card("Water")).getByRole("slider", { name: "Hue" });
    expect(slider).not.toHaveAttribute("data-disabled");
  });

  it("says 'Looking at the photo…' on every card while the scene is made", async () => {
    await setup(WITH_MATCH(), { speed: 0.02 });
    await chooseFile();
    await waitFor(() => expect(fitButton()).not.toHaveAttribute("aria-disabled", "true"));
    await h!.user.click(fitButton());
    await waitFor(() => expect(cardOf()?.fit).toEqual({ step: "scene" }));
    await openTab(h!, "Elements");
    // Every card says it, the Subject's too.
    expect(within(tools()).getAllByText("Looking at the photo…")).toHaveLength(7);
    h!.api.configure({ speed: Number.POSITIVE_INFINITY });
    h!.api.flush();
  });
});

describe("the eye of an Elements card (docs/34 §9.6)", () => {
  const eyeOf = (title: string): HTMLElement => within(tools()).getByRole("button", { name: new RegExp(`^(Show|Hide) mask, ${title}$`) });
  const views = (): unknown[] => vi.mocked(h!.api.viewportSetMode).mock.calls.map((c) => c[1] ?? null);
  const NAMES = ["Sky", "Vegetation", "Water", "Ground", "Built", "Skin", "Subject"];

  async function withEyes(): Promise<void> {
    await setup(WITH_MATCH());
    await openTab(h!, "Elements");
    vi.spyOn(h!.api, "viewportSetMode");
  }

  it("is on each of the seven cards, off", async () => {
    await withEyes();
    for (const name of NAMES) expect(eyeOf(name)).toHaveAttribute("aria-pressed", "false");
  });

  it("sends the overlay view of the card's element, numbered as the settings number them, and ends it with a second click", async () => {
    await withEyes();
    await h!.user.click(eyeOf("Sky"));
    await waitFor(() => expect(views().at(-1)).toEqual({ node: 2, style: "overlay", element: 0 }));
    expect(eyeOf("Sky")).toHaveAttribute("aria-pressed", "true");
    expect(eyeOf("Sky")).toHaveAccessibleName("Hide mask, Sky");
    await h!.user.click(eyeOf("Sky"));
    await waitFor(() => expect(views().at(-1)).toBeNull());
    expect(h!.store.getState().elementEye).toBeNull();
    // The numbers are the order of the cards.
    for (const [i, name] of NAMES.entries()) {
      await h!.user.click(eyeOf(name));
      await waitFor(() => expect(views().at(-1)).toEqual({ node: 2, style: "overlay", element: i }));
    }
  });

  it("is one view at a time: another eye, Show mask, a mask row's eye and the skin areas turn it off, and it turns them off", async () => {
    await withEyes();
    await h!.user.click(eyeOf("Sky"));
    await h!.user.click(eyeOf("Skin"));
    expect(h!.store.getState().elementEye).toBe(5);
    expect(eyeOf("Sky")).toHaveAttribute("aria-pressed", "false");
    expect(eyeOf("Skin")).toHaveAttribute("aria-pressed", "true");
    h!.store.setMaskView({ on: true });
    expect(h!.store.getState().elementEye).toBeNull();
    await waitFor(() => expect(views().at(-1)).toEqual({ node: 2, style: "overlay" }));
    await h!.user.click(eyeOf("Ground"));
    expect(h!.store.getState().maskView.on).toBe(false);
    h!.store.setMaskEye(0);
    expect(h!.store.getState().elementEye).toBeNull();
    await h!.user.click(eyeOf("Ground"));
    expect(h!.store.getState().maskEye).toBeNull();
  });

  it("keeps the style of the View card, and keeps the eye when the style changes", async () => {
    await withEyes();
    h!.store.setMaskView({ style: "mask_only", on: false });
    await h!.user.click(eyeOf("Water"));
    await waitFor(() => expect(views().at(-1)).toEqual({ node: 2, style: "mask_only", element: 2 }));
    h!.store.setMaskView({ style: "overlay" });
    expect(h!.store.getState().elementEye).toBe(2);
    await waitFor(() => expect(views().at(-1)).toEqual({ node: 2, style: "overlay", element: 2 }));
  });

  it("ends with another node, with the tab, with the photo and in the Before view", async () => {
    await withEyes();
    await h!.user.click(eyeOf("Sky"));
    await waitFor(() => expect(h!.store.getState().elementEye).toBe(0));
    await openTab(h!, "Hue");
    await waitFor(() => expect(h!.store.getState().elementEye).toBeNull());
    expect(views().at(-1)).toBeNull();

    await openTab(h!, "Elements");
    await h!.user.click(eyeOf("Sky"));
    await waitFor(() => expect(h!.store.getState().elementEye).toBe(0));
    fireEvent.click(screen.getByRole("button", { name: /^Node 1/ }));
    await waitFor(() => expect(h!.store.getState().elementEye).toBeNull());
    expect(views().at(-1)).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /^Node 2, Match/ }));
    await openTab(h!, "Elements");
    await h!.user.click(eyeOf("Built"));
    await waitFor(() => expect(h!.store.getState().elementEye).toBe(4));
    const first = photoId();
    await act(async () => {
      await h!.store.setCurrent(first + 1);
    });
    await waitFor(() => expect(h!.store.getState().elementEye).toBeNull());

    await act(async () => {
      await h!.store.setCurrent(first);
    });
    h!.store.setElementEye(1);
    expect(h!.store.getState().elementEye).toBe(1);
    await act(async () => {
      await h!.store.setBefore(true);
    });
    expect(h!.store.getState().elementEye).toBeNull();
  });

  it("is off, not gone, on an element that is not in the photo", async () => {
    await setup();
    await chooseFile();
    await waitFor(() => expect(fitButton()).not.toHaveAttribute("aria-disabled", "true"));
    await h!.user.click(fitButton());
    await waitFor(() => expect(statusGroup()).toHaveTextContent("Fitted to the reference."));
    await openTab(h!, "Elements");
    expect(eyeOf("Water")).toHaveAttribute("aria-disabled", "true");
    expect(eyeOf("Sky")).not.toHaveAttribute("aria-disabled", "true");
  });
});

describe("Copy and Paste", () => {
  it("carry the node with its fitted values to another photo, which then says the values are kept", async () => {
    await setup(WITH_MATCH({ hue_curves: { hue_vs_hue: [{ hue_deg: 120, value: 12 }] } }));
    await h!.user.click(within(tools()).getByRole("button", { name: /Copy settings/ }));
    const first = photoId();
    await act(async () => {
      await h!.store.setCurrent(first + 1);
    });
    await waitFor(() => expect(h!.store.currentSettings()!.photo_id).not.toBe(first));
    await h!.user.click(within(tools()).getByRole("button", { name: /^Paste/ }));
    await waitFor(() => expect(nodeOf(h!.store.currentSettings()!.settings, 2)?.kind).toBe("reference_grade"));
    const pasted = nodeOf(h!.store.currentSettings()!.settings, 2) as ReferenceGradeItem;
    expect(pasted.hue_curves.hue_vs_hue).toEqual([{ hue_deg: 120, value: 12 }]);
    // The reference is not in the clipboard: nothing says it, and the node says its values are kept.
    fireEvent.click(screen.getByRole("button", { name: /^Node 2, Match/ }));
    await waitFor(() => expect(statusGroup()).toHaveTextContent("The values are kept. Choose a reference to fit again."));
  });
});
