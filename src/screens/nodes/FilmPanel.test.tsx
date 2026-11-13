/**
 * The panel of a Film node (`docs/24` section 4, task F-09): the tabs, the Film tab with its Stock tiles, the Print tab and the slide case,
 * Halation and Grain, the Output's note and About's credit. The screen runs on the mock, as in a browser.
 */
import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SettingsPatch } from "../../api/types";
import { FILM_CREDIT, FILM_CREDIT_NAME, FILM_CREDIT_URL } from "../../fixtures/filmCredit";
import { nodeOf } from "../../state/settingsGraph";
import type { FilmItem } from "../../state/settingsGraph";
import { openPhotos, openTab, renderApp, sliderOf, tools, valueOf } from "../../test/harness";
import type { Harness, UpdateSpy } from "../../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

const picture = (from: number | "input", to: number | "output") => ({ from: { node: from, port: "picture" as const }, to: { node: to, port: "picture" as const } });

/** Main, then a Film node (2, Portra 400 on its matching paper), then the Output. */
const WITH_FILM = (stock = "kodak_portra_400"): SettingsPatch => ({
  nodes: { "2": { kind: "film", name: "Film", pos: [1, 0], film: { stock } } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

async function setup(settings: SettingsPatch = WITH_FILM()): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: settings } });
  await openPhotos(h);
  await openTab(h, "Light");
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  fireEventClick(screen.getByRole("button", { name: /^Node 2, Film/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /^Node 2, Film/ })).toHaveAttribute("aria-pressed", "true"));
  return vi.spyOn(h.api, "updateSettings");
}
function fireEventClick(el: HTMLElement): void {
  act(() => el.click());
}
const head = (): HTMLElement => within(tools()).getByRole("group", { name: "Node" });
const tabsOf = (): string[] => within(tools()).getAllByRole("tab").map((t) => t.textContent ?? "");
const film = (): FilmItem => nodeOf(h!.store.currentSettings()!.settings, 2) as FilmItem;
const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
async function openTabNamed(name: string): Promise<void> {
  await h!.user.click(within(tools()).getByRole("tab", { name }));
  await waitFor(() => expect(within(tools()).getByRole("tab", { name })).toHaveAttribute("aria-selected", "true"));
}
async function type(label: string, text: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: `${label} value` });
  await h!.user.click(box);
  await h!.user.clear(box);
  await h!.user.type(box, `${text}{Enter}`);
}

describe("the head and the tabs", () => {
  it("is the node's own head, and the tabs are Film, Print, Halation, Grain and Mask", async () => {
    await setup();
    expect(within(head()).getByRole("heading", { level: 2, name: "Film" })).toBeInTheDocument();
    expect(head().querySelector(".ym-nodehead__chip")).toHaveTextContent("02");
    expect(head().querySelector(".ym-nodehead__line")).toHaveTextContent("Film · whole photo");
    expect(tabsOf()).toEqual(["Film", "Print", "Halation", "Grain", "Mask"]);
    expect(within(tools()).getByRole("tab", { name: "Film" })).toHaveAttribute("aria-selected", "true");
    const buttons = within(within(head()).getByRole("group", { name: "Node actions" })).getAllByRole("button");
    expect(buttons.map((b) => b.getAttribute("aria-label"))).toEqual(["Bypass this node", "Reset this node", "More"]);
  });

  it("opens the Mask tab of a corrector for a Film node", async () => {
    await setup();
    await openTabNamed("Mask");
    expect(within(tools()).getByRole("button", { name: /Add mask/ })).toBeInTheDocument();
  });
});

describe("the Film tab", () => {
  it("has the Stock with its kind and ISO, Exposure and Strength at their defaults, and the credit line at the end", async () => {
    await setup();
    expect(await screen.findByRole("button", { name: "Stock, Kodak Portra 400" })).toBeInTheDocument();
    expect(await within(tools()).findByText("Colour negative · ISO 400")).toBeInTheDocument();
    expect(valueOf("Exposure")).toBe("0.00");
    expect(valueOf("Strength")).toBe("100%");
    expect(sliderOf("Exposure")).toHaveAttribute("aria-valuemin", "-4");
    expect(sliderOf("Exposure")).toHaveAttribute("aria-valuemax", "4");
    expect(sliderOf("Strength")).toHaveAttribute("aria-valuemax", "100");
    expect(tools().querySelector(".ym-film-credit")).toHaveTextContent(FILM_CREDIT);
    // There is no Brightness option (the owner chose "as the app" only).
    expect(within(tools()).queryByText(/As printed/)).toBeNull();
  });

  it("opens the project's page in the browser from the credit line", async () => {
    await setup();
    await h!.user.click(within(tools()).getByRole("button", { name: FILM_CREDIT_NAME }));
    expect(h!.api.system.opened).toEqual([FILM_CREDIT_URL]);
  });

  it("sends the node's own patch for Exposure and Strength, and Reset puts the card back", async () => {
    const spy = await setup();
    await type("Exposure", "1.5");
    await waitFor(() => expect(film().film.exposure_ev).toBe(1.5));
    expect(patches(spy).at(-1)).toEqual({ nodes: { "2": { film: { exposure_ev: 1.5 } } } });
    await type("Strength", "60");
    await waitFor(() => expect(film().strength).toBe(60));
    await h!.user.click(within(screen.getByRole("region", { name: "Film" })).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(film().strength).toBe(100));
    expect(film().film).toEqual({ stock: "kodak_portra_400", exposure_ev: 0 });
  });
});

describe("the Stock picker", () => {
  async function openPicker(): Promise<HTMLElement> {
    await h!.user.click(await screen.findByRole("button", { name: /^Stock, / }));
    return screen.findByRole("dialog", { name: "Film stocks" });
  }

  it("has no aria-controls, open or closed: axe cannot decide it on an open aria-haspopup trigger (aria-valid-attr-value)", async () => {
    await setup();
    const field = await screen.findByRole("button", { name: /^Stock, / });
    expect(field).toHaveAttribute("aria-haspopup", "dialog");
    expect(field).toHaveAttribute("aria-expanded", "false");
    expect(field).not.toHaveAttribute("aria-controls");
    await openPicker();
    expect(field).toHaveAttribute("aria-expanded", "true");
    expect(field).not.toHaveAttribute("aria-controls");
  });

  it("is tiles in three groups: 20 stocks, a small render of the photo in each, the chosen one marked", async () => {
    await setup();
    const picker = await openPicker();
    expect(within(picker).getAllByRole("heading", { level: 3 }).map((e) => e.textContent)).toEqual(["Colour negative", "Cine negative", "Slide"]);
    const tiles = within(picker).getAllByRole("button");
    expect(tiles).toHaveLength(20);
    expect(within(within(picker).getByRole("region", { name: "Slide" })).getAllByRole("button")).toHaveLength(4);
    expect(within(picker).getByRole("button", { name: "Kodak Portra 400, ISO 400" })).toHaveAttribute("aria-pressed", "true");
    expect(within(picker).getByRole("button", { name: "Kodak Portra 800 (Push 1), ISO 800" })).toHaveAttribute("aria-pressed", "false");
    // The pictures come one at a time as they are made; the picker has not waited for them.
    await waitFor(() => expect(picker.querySelectorAll("img")).toHaveLength(20), { timeout: 5000 });
    for (const img of picker.querySelectorAll("img")) expect(img.getAttribute("src")).toMatch(/^blob:/);
  });

  it("asks the core for one tile per stock, for this node", async () => {
    await setup();
    const spy = vi.spyOn(h!.api, "filmTile");
    await openPicker();
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(20), { timeout: 5000 });
    expect(new Set(spy.mock.calls.map((c) => c[2])).size).toBe(20);
    for (const call of spy.mock.calls) expect(call[1]).toBe(2);
  });

  it("chooses a stock with a click and closes", async () => {
    const spy = await setup();
    const picker = await openPicker();
    await h!.user.click(within(picker).getByRole("button", { name: "Kodak Ektar 100, ISO 100" }));
    await waitFor(() => expect(film().film.stock).toBe("kodak_ektar_100"));
    expect(patches(spy).at(-1)).toEqual({ nodes: { "2": { film: { stock: "kodak_ektar_100" } } } });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Film stocks" })).toBeNull());
    expect(await screen.findByRole("button", { name: "Stock, Kodak Ektar 100" })).toBeInTheDocument();
    expect(await within(tools()).findByText("Colour negative · ISO 100")).toBeInTheDocument();
  });

  it("shows the name of a stock on a plain card until its picture is made, and the picker never waits for it", async () => {
    await setup();
    // No picture is made: every request waits.
    vi.spyOn(h!.api, "filmTile").mockImplementation(() => new Promise<string>(() => {}));
    const picker = await openPicker();
    const tile = within(picker).getByRole("button", { name: "Fujifilm Velvia 100, ISO 100" });
    expect(tile.querySelector(".ym-stock-tile__card")).toHaveTextContent("Fujifilm Velvia 100");
    expect(tile.querySelector("img")).toBeNull();
    // It can be chosen all the same.
    await h!.user.click(tile);
    await waitFor(() => expect(film().film.stock).toBe("fujifilm_velvia_100"));
  });
});

describe("the Print tab", () => {
  it("has Paper (Matching paper, the matching paper's name under it), Print exposure, the Lights with their hint, and Cyan under More", async () => {
    await setup();
    await openTabNamed("Print");
    expect(await screen.findByRole("combobox", { name: "Paper" })).toHaveTextContent("Matching paper");
    expect(within(tools()).getByText("Kodak Professional Portra Endura")).toBeInTheDocument();
    expect(valueOf("Print exposure")).toBe("0.00");
    expect(valueOf("Yellow")).toBe("0");
    expect(valueOf("Magenta")).toBe("0");
    expect(sliderOf("Yellow")).toHaveAttribute("aria-valuemin", "-50");
    expect(within(tools()).getByText("0 is neutral for this film and paper.")).toBeInTheDocument();
    expect(screen.queryByRole("slider", { name: "Cyan" })).toBeNull();
    await h!.user.click(within(screen.getByRole("region", { name: "Lights" })).getByRole("button", { name: "More" }));
    expect(sliderOf("Cyan")).toHaveAttribute("aria-valuemax", "50");
    expect(valueOf("Cyan")).toBe("0");
  });

  it("lists Matching paper and the six papers, and not the cine print films", async () => {
    await setup();
    await openTabNamed("Print");
    await h!.user.click(await screen.findByRole("combobox", { name: "Paper" }));
    const options = (await screen.findAllByRole("option")).map((o) => o.textContent);
    expect(options).toEqual([
      "Matching paper",
      "Fujifilm Crystal Archive Type II",
      "Kodak Ektacolor Edge",
      "Kodak Professional Endura Premier",
      "Kodak Professional Portra Endura",
      "Kodak Professional Supra Endura",
      "Kodak Professional Ultra Endura",
    ]);
    expect(options.join(" ")).not.toMatch(/Vision/);
  });

  it("sends the paper chosen, and the lights in halves", async () => {
    const spy = await setup();
    await openTabNamed("Print");
    await h!.user.click(await screen.findByRole("combobox", { name: "Paper" }));
    await h!.user.click(await screen.findByRole("option", { name: "Kodak Ektacolor Edge" }));
    await waitFor(() => expect(film().print.paper).toBe("kodak_ektacolor_edge"));
    expect(screen.queryByText("Kodak Professional Portra Endura")).toBeNull();
    await type("Yellow", "7.5");
    await waitFor(() => expect(film().print.lights.y).toBe(7.5));
    expect(patches(spy).at(-1)).toEqual({ nodes: { "2": { print: { lights: { y: 7.5 } } } } });
    await type("Magenta", "-3");
    await waitFor(() => expect(film().print.lights).toEqual({ c: 0, m: -3, y: 7.5 }));
  });

  it("is dimmed for a slide, which is scanned as it is, and says there is no print", async () => {
    await setup(WITH_FILM("fujifilm_velvia_100"));
    await openTabNamed("Print");
    expect(await screen.findByText("Slide film is scanned as it is. There is no print.")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Paper" })).toHaveTextContent("None");
    expect(screen.getByRole("combobox", { name: "Paper" })).toBeDisabled();
    expect(sliderOf("Yellow")).toHaveAttribute("data-disabled");
    expect(sliderOf("Print exposure")).toHaveAttribute("data-disabled");
    expect(tools().querySelector(".ym-film-print")).toHaveAttribute("data-slide");
  });
});

describe("the Halation and Grain tabs", () => {
  it("have Amount and Size at their ranges and defaults", async () => {
    const spy = await setup();
    await openTabNamed("Halation");
    expect(valueOf("Amount")).toBe("100%");
    expect(valueOf("Size")).toBe("100%");
    expect(sliderOf("Amount")).toHaveAttribute("aria-valuemax", "300");
    expect(sliderOf("Size")).toHaveAttribute("aria-valuemin", "50");
    expect(sliderOf("Size")).toHaveAttribute("aria-valuemax", "200");
    await type("Amount", "250");
    await waitFor(() => expect(film().halation.amount).toBe(250));
    expect(patches(spy).at(-1)).toEqual({ nodes: { "2": { halation: { amount: 250 } } } });
    await openTabNamed("Grain");
    expect(sliderOf("Amount")).toHaveAttribute("aria-valuemax", "200");
    expect(sliderOf("Size")).toHaveAttribute("aria-valuemin", "50");
    await type("Size", "300");
    await waitFor(() => expect(film().grain.size).toBe(200));
    expect(film().grain.seed).toBe(0);
  });
});

describe("the Output's note", () => {
  it("is on the Look tab while a Film node is live, and not without one", async () => {
    await setup();
    fireEventClick(screen.getByRole("button", { name: "Output" }));
    await waitFor(() => expect(within(tools()).getByRole("tab", { name: "Look" })).toHaveAttribute("aria-selected", "true"));
    expect(within(tools()).getByText("The look acts after the Film node. Choose Neutral to see the print as scanned.")).toBeInTheDocument();
  });

  it("is not there for a photo without a Film node", async () => {
    h = renderApp({ current: 3, picked: [2, 3, 4] });
    await openPhotos(h);
    await openTab(h, "Look");
    await h.user.keyboard("n");
    await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
    fireEventClick(screen.getByRole("button", { name: "Output" }));
    await waitFor(() => expect(within(tools()).getByRole("tab", { name: "Look" })).toBeInTheDocument());
    expect(within(tools()).queryByText(/The look acts after the Film node/)).toBeNull();
  });
});

describe("Add node › Film", () => {
  it("adds a Film node after the selected node, Portra 400 on its matching paper, and selects it", async () => {
    h = renderApp({ current: 3, picked: [2, 3, 4] });
    await openPhotos(h);
    await openTab(h, "Light");
    await h.user.keyboard("n");
    await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
    await h.user.click(screen.getByRole("button", { name: "Add node" }));
    await h.user.click(await screen.findByRole("menuitem", { name: /Film/ }));
    await waitFor(() => expect(within(head()).getByRole("heading", { level: 2, name: "Film" })).toBeInTheDocument());
    const added = nodeOf(h.store.currentSettings()!.settings, 2) as FilmItem;
    expect(added.kind).toBe("film");
    expect(added.film).toEqual({ stock: "kodak_portra_400", exposure_ev: 0 });
    expect(added.print).toEqual({ paper: null, exposure_ev: 0, lights: { c: 0, m: 0, y: 0 } });
    expect(added.strength).toBe(100);
    expect(tabsOf()).toEqual(["Film", "Print", "Halation", "Grain", "Mask"]);
  });
});
