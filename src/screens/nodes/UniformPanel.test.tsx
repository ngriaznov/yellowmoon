/**
 * The Colour Uniform node's interface (`docs/32` sections 2.1 to 2.6, 4 and 5.3): the Add node item, the Compress tab in each state, Pick colour and
 * the one patch of the first pick, the target fields, the two Resets and the head's line. The screen runs on the mock, as in a browser.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import type { NodeLink, SettingsPatch } from "../../api/types";
import { nodeOf } from "../../state/settingsGraph";
import type { ColorUniformItem } from "../../state/settingsGraph";
import { openPhotos, openTab, renderApp, tools } from "../../test/harness";
import type { Harness, UpdateSpy } from "../../test/harness";
import { mockRect } from "../../test/raf";
import { UNIFORM_TEXT } from "./UniformTab";

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

/** Main, then a Colour Uniform node (2), then the Output. */
const WITH_UNIFORM = (more: Record<string, unknown> = {}): SettingsPatch => ({
  nodes: { "2": { kind: "color_uniform", name: "Uniform", pos: [1, 0], ...more } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

const PICKED = { hue_deg: 40, saturation: 12, lightness_ev: 0.2 };

/** DSC04204 in the node view with the Colour Uniform node selected, the photo cell 600 x 400. */
async function setup(settings: SettingsPatch = WITH_UNIFORM(), options: MockOptions = {}): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: settings }, ...options });
  await openPhotos(h);
  await openTab(h, "Light");
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  fireEvent.click(screen.getByRole("button", { name: /^Node 2, Uniform/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /^Node 2, Uniform/ })).toHaveAttribute("aria-pressed", "true"));
  const cell = document.querySelector<HTMLElement>("[data-viewport]")!;
  mockRect(cell, { left: 0, top: 0, width: 600, height: 400 });
  await h.api.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
  await act(async () => {
    fireEvent(window, new Event("resize"));
  });
  return vi.spyOn(h.api, "updateSettings");
}

const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const node = (): ColorUniformItem => nodeOf(h!.store.currentSettings()!.settings, 2) as ColorUniformItem;
const card = (name: string): HTMLElement => within(tools()).getByRole("region", { name });
const pickButton = (): HTMLElement => within(card("Target")).getByRole("button", { name: "Pick colour" });
const pickLayer = (): HTMLElement => screen.getByRole("application", { name: /Pick colour for the target of Colour Uniform/ });
const head = (): HTMLElement => within(tools()).getByRole("group", { name: "Node" });
/** A field of the Target card: the Compress card has a Hue and a Saturation of its own, so the cards are told apart by their region. */
const field = (label: string): HTMLInputElement => within(card("Target")).getByRole("textbox", { name: `${label} value` }) as HTMLInputElement;
const valueOf = (label: string): string => field(label).value;
async function type(label: string, text: string): Promise<void> {
  const box = field(label);
  await h!.user.click(box);
  await h!.user.clear(box);
  await h!.user.type(box, `${text}{Enter}`);
}

describe("Add node > Colour Uniform", () => {
  it("adds a node named Uniform, selects it and opens its Compress tab with Pick colour on", async () => {
    h = renderApp({ current: 3, picked: [2, 3, 4] });
    await openPhotos(h);
    await openTab(h, "Light");
    await h.user.keyboard("n");
    await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
    await h.user.click(screen.getByRole("button", { name: "Add node" }));
    const menu = await screen.findByRole("menu", { name: "Add node" });
    await h.user.click(within(menu).getByRole("menuitem", { name: /^Colour Uniform/ }));
    await waitFor(() => expect(screen.getByRole("button", { name: /^Node 2, Uniform/ })).toHaveAttribute("aria-pressed", "true"));
    expect(within(tools()).getByRole("tab", { name: "Compress" })).toHaveAttribute("aria-selected", "true");
    expect(within(tools()).getAllByRole("tab").map((t) => t.textContent)).toEqual(["Compress", "Light", "Colour", "Curve", "Mask"]);
    await waitFor(() => expect(pickButton()).toHaveAttribute("aria-pressed", "true"));
    expect(pickLayer()).toBeInTheDocument();
  });
});

describe("the Compress tab with no target", () => {
  it("says so, shows dashes in the three fields, and turns the three amounts off with their line", async () => {
    await setup();
    expect(within(tools()).getAllByRole("heading", { level: 3 }).map((e) => e.textContent)).toEqual(["Target", "Compress"]);
    expect(within(card("Target")).getByText(UNIFORM_TEXT.noTarget)).toBeInTheDocument();
    expect(within(card("Target")).getByText(UNIFORM_TEXT.noTargetLine)).toBeInTheDocument();
    expect(within(card("Target")).getByRole("img", { name: "No target colour yet" })).toBeInTheDocument();
    for (const label of ["Hue", "Saturation", "Lightness (EV)"]) expect(valueOf(label)).toBe("–");
    for (const label of ["Hue", "Saturation", "Lightness"]) {
      const sliders = within(card("Compress")).getAllByRole("slider", { name: label });
      expect(sliders[0]).toHaveAttribute("data-disabled");
    }
    expect(within(card("Compress")).getByText(UNIFORM_TEXT.needTarget)).toBeInTheDocument();
    expect(within(head()).queryByRole("button", { name: "Reset target" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Reset target" })).toBeNull();
    expect(head().querySelector(".ym-nodehead__line")?.textContent).toBe("Colour Uniform");
  });
});

describe("Pick colour", () => {
  it("is a toggle: a click on the photo sets the target, Compress 50 / 50 / 0 and one Colour range row in one patch, and one undo takes them all back", async () => {
    const spy = await setup();
    await h!.user.click(pickButton());
    expect(pickButton()).toHaveAttribute("aria-pressed", "true");
    mockRect(pickLayer(), { left: 0, top: 0, width: 600, height: 400 });
    spy.mockClear();
    await h!.user.pointer({ target: pickLayer(), coords: { clientX: 300, clientY: 20 }, keys: "[MouseLeft]" });
    await waitFor(() => expect(patches(spy)).toHaveLength(1));
    const only = patches(spy)[0]!.nodes!["2"]!;
    expect(only.target).toBeTruthy();
    expect(only.compress).toEqual({ hue: 50, saturation: 50, lightness: 0 });
    expect(only.masks).toBeTruthy();
    expect(node().target).not.toBeNull();
    expect(node().masks.items).toHaveLength(1);
    // The swatch is named by its colour, the card says how to go on, the amounts are on and the pick is off.
    expect(within(card("Target")).getByRole("img", { name: /^Target colour: / })).toBeInTheDocument();
    expect(within(card("Target")).getByText(UNIFORM_TEXT.picked)).toBeInTheDocument();
    expect(within(card("Compress")).getAllByRole("slider", { name: "Hue" })[0]).not.toHaveAttribute("data-disabled");
    // Pick colour turns off after the pick (the owner, 2026-11-21): the button shows unpressed and the layer is gone.
    await waitFor(() => expect(pickButton()).toHaveAttribute("aria-pressed", "false"));
    expect(pickButton()).toHaveAttribute("data-state", "off");
    expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull();
    // The row the pick added is the selected one in the Mask tab.
    expect(h!.store.getState().maskRow).toBe(0);
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(node().target).toBeNull());
    expect(node().masks.items).toEqual([]);
    expect(node().compress).toEqual({ hue: 0, saturation: 0, lightness: 0 });
  });

  it("turns off after every pick, not only the first, and a click on the button turns it on again", async () => {
    const spy = await setup();
    for (const clientX of [300, 200]) {
      await h!.user.click(pickButton());
      expect(pickButton()).toHaveAttribute("data-state", "on");
      mockRect(pickLayer(), { left: 0, top: 0, width: 600, height: 400 });
      spy.mockClear();
      await h!.user.pointer({ target: pickLayer(), coords: { clientX, clientY: 20 }, keys: "[MouseLeft]" });
      await waitFor(() => expect(patches(spy)).toHaveLength(1));
      await waitFor(() => expect(pickButton()).toHaveAttribute("aria-pressed", "false"));
      expect(pickButton()).toHaveAttribute("data-state", "off");
      expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull();
    }
  });

  it("ends with Esc and with the toggle", async () => {
    await setup();
    await h!.user.click(pickButton());
    expect(pickLayer()).toBeInTheDocument();
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull());
    expect(pickButton()).toHaveAttribute("aria-pressed", "false");
    await h!.user.click(pickButton());
    await h!.user.click(pickButton());
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour/ })).toBeNull());
  });
});

describe("the target fields and the Resets", () => {
  it("show the target, and a change to one sends the node's target with the other two kept", async () => {
    const spy = await setup(WITH_UNIFORM({ target: PICKED, compress: { hue: 50, saturation: 50, lightness: 0 } }));
    expect(valueOf("Hue")).toBe("40°");
    expect(valueOf("Saturation")).toBe("12");
    spy.mockClear();
    await type("Saturation", "20");
    await waitFor(() => expect(node().target).toEqual({ ...PICKED, saturation: 20 }));
    expect(patches(spy).at(-1)!.nodes!["2"]).not.toHaveProperty("compress");
  });

  it("sets a target from the resting values when a field is changed with none", async () => {
    await setup();
    await type("Saturation", "15");
    await waitFor(() => expect(node().target).toEqual({ hue_deg: 0, saturation: 15, lightness_ev: 0 }));
  });

  it("have a Reset on each card: the target's clears the target only, Compress's sets the three amounts to 0", async () => {
    await setup(WITH_UNIFORM({ target: PICKED, compress: { hue: 50, saturation: 50, lightness: 10 } }));
    await h!.user.click(screen.getByRole("button", { name: "Reset Compress" }));
    await waitFor(() => expect(node().compress).toEqual({ hue: 0, saturation: 0, lightness: 0 }));
    expect(node().target).toEqual(PICKED);
    await h!.user.click(screen.getByRole("button", { name: "Reset target" }));
    await waitFor(() => expect(node().target).toBeNull());
  });

  it("move a Compress amount with the keyboard", async () => {
    await setup(WITH_UNIFORM({ target: PICKED, compress: { hue: 50, saturation: 50, lightness: 0 } }));
    const slider = within(card("Compress")).getAllByRole("slider", { name: "Hue" })[0]!;
    slider.focus();
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(node().compress.hue).toBe(51));
  });
});
