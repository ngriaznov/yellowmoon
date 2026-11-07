/**
 * The Depth mask's interface (`docs/23` §4.8 and §7.6, `docs/11` *The Depth card*, task D-05): the Add mask entry and where it is hidden, the row and its states
 * ("Finding…", "Can't be made", the missing model), the card (Pick depth on for a new row, Focus plane and its reasons, Near, Far and Softness with their stops, Show depth
 * and its exclusivity, Try again), all on the mock as the browser runs it. The core's side is `crates/ym-core/tests/mask_depth.rs`.
 */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import { DEPTH_FAILED } from "../../api.mock";
import type { MaskItem, NodeLink, SettingsPatch } from "../../api/types";
import { newMask } from "../../state/maskRows";
import { nodeOf } from "../../state/settingsGraph";
import type { CorrectorItem } from "../../state/settingsGraph";
import { openPhotos, openTab, renderApp, sliderName, sliderOf, tools, valueOf } from "../../test/harness";
import type { Harness, UpdateSpy } from "../../test/harness";
import { FAILED_ADVICE, FOCUS_AT_CENTRE, NO_FOCUS_POINT } from "./DepthCard";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const DEPTH: MaskItem = newMask("depth");
/** Main, Haze (a Depth row) and Warmth (no masks), as the mockup. */
const GRAPH = (rows: MaskItem[] = [DEPTH]): SettingsPatch => ({
  nodes: {
    "2": { kind: "corrector", name: "Haze", pos: [1, 0], masks: { invert: false, items: rows } },
    "3": { kind: "corrector", name: "Warmth", pos: [2, 0] },
  },
  links: [picture("input", 1), picture(1, 2), picture(2, 3), picture(3, "output")],
});

/** The photo (DSC04204 has a focus point, DSC04201 none) in the node view, on the Mask tab of the node `onNode`. */
async function setup(settings: SettingsPatch, options: MockOptions = {}, onNode: RegExp = /^Node 2, Haze/, current = 3): Promise<UpdateSpy> {
  const photo = `DSC042${String(current + 1).padStart(2, "0")}`;
  h = renderApp({ current, picked: [2, 3, 4], initialSettings: { [photo]: settings }, ...options });
  await openPhotos(h);
  await openTab(h, "Light");
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  fireEvent.click(screen.getByRole("button", { name: onNode }));
  await waitFor(() => expect(screen.getByRole("button", { name: onNode })).toHaveAttribute("aria-pressed", "true"));
  await h.user.click(within(tools()).getByRole("tab", { name: "Mask" }));
  await waitFor(() => expect(within(tools()).getByRole("tab", { name: "Mask" })).toHaveAttribute("aria-selected", "true"));
  return vi.spyOn(h.api, "updateSettings");
}

const rowList = (): HTMLElement => within(tools()).getByRole("list", { name: "Masks of this node" });
const rowButton = (name: string | RegExp): HTMLElement => within(rowList()).getByRole("button", { name });
const card = (): HTMLElement => within(tools()).getByRole("region", { name: "Depth" });
const rows = (node: number): MaskItem[] => (nodeOf(h!.store.currentSettings()!.settings, node) as CorrectorItem).masks.items;
const mask = () => h!.api.viewport.maskView;
async function type(label: string, text: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: `${label} value` });
  await h!.user.click(box);
  await h!.user.clear(box);
  await h!.user.type(box, `${text}{Enter}`);
}
/** Waits for the mock's depth map (`depth_state`) to be made. */
const mapMade = (): Promise<void> => waitFor(() => expect(h!.store.getState().viewportBusy.finding).toBe(false));

describe("the Add mask menu", () => {
  it("lists Depth after Background where the weights are there, and hides it where they are missing", async () => {
    await setup(GRAPH([]), {}, /^Node 3, Warmth/);
    await h!.user.click(within(tools()).getByRole("button", { name: "Add mask" }));
    const menu = await screen.findByRole("menu", { name: "Add mask" });
    const names = within(menu).getAllByRole("menuitem").map((m) => within(m).getAllByText(/./)[0]?.textContent);
    expect(names.indexOf("Depth")).toBe(names.indexOf("Background") + 1);
    expect(within(menu).getByText("Near or far, from depth")).toBeInTheDocument();
    h!.unmount();
    h = undefined;
    await setup(GRAPH([]), { maskKinds: ["object", "background", "linear"] }, /^Node 3, Warmth/);
    await h!.user.click(within(tools()).getByRole("button", { name: "Add mask" }));
    const without = await screen.findByRole("menu", { name: "Add mask" });
    expect(within(without).queryByRole("menuitem", { name: /^Depth/ })).toBeNull();
  });

  it("adds a Depth row with its defaults, selects it, starts Pick depth and asks for the map: the row says Finding…", async () => {
    const spy = await setup(GRAPH([]), { speed: 1 }, /^Node 3, Warmth/);
    await h!.user.click(within(tools()).getByRole("button", { name: "Add mask" }));
    await h!.user.click(within(await screen.findByRole("menu", { name: "Add mask" })).getByRole("menuitem", { name: /^Depth/ }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(rows(3)).toEqual([{ ...DEPTH }]);
    expect(rows(3)[0]).toMatchObject({ kind: "depth", near: 0, far: 0.5, softness: 0.5, feather: 0, grow: 0, refine: 0.5 });
    // The row says it is finding, and the card is there with Pick depth on.
    expect(await within(rowList()).findByText("Finding…")).toBeInTheDocument();
    expect(card()).toBeInTheDocument();
    expect(within(card()).getByRole("button", { name: "Pick depth" })).toHaveAttribute("aria-pressed", "true");
    // The pill over the photo waits for the map, then says what to do.
    expect(await screen.findByText("Getting ready…")).toBeInTheDocument();
    expect(await screen.findByText("Click the part of the photo to select. Esc stops.")).toBeInTheDocument();
    await waitFor(() => expect(within(rowList()).queryByText("Finding…")).toBeNull());
    expect(rowButton("Depth mask, 0 to 50")).toBeInTheDocument();
  });
});

describe("the card", () => {
  it("is DEPTH with Pick depth, Focus plane, Near, Far, Softness, Show depth and the raster sliders, in that order", async () => {
    await setup(GRAPH());
    await mapMade();
    await h!.user.click(rowButton(/^Depth mask/));
    const c = card();
    expect(within(c).getAllByRole("button").map((b) => b.textContent).filter(Boolean)).toEqual(["Reset", "Pick depth", "Focus plane", "Show depth"]);
    expect(within(c).getAllByRole("slider").map((s) => sliderName(s))).toEqual(["Near", "Far", "Softness", "Refine edges", "Feather", "Grow", "Opacity"]);
    expect([valueOf("Near"), valueOf("Far"), valueOf("Softness"), valueOf("Refine edges")]).toEqual(["0", "50", "50", "50%"]);
    // Pick depth is off for a row that was not just added.
    expect(within(c).getByRole("button", { name: "Pick depth" })).toHaveAttribute("aria-pressed", "false");
  });

  it("Near never passes Far and Far never passes Near, and each slider writes one field", async () => {
    const spy = await setup(GRAPH());
    await mapMade();
    await h!.user.click(rowButton(/^Depth mask/));
    await type("Far", "80");
    await waitFor(() => expect(rows(2)[0]).toMatchObject({ far: 0.8 }));
    await type("Near", "20");
    await waitFor(() => expect(rows(2)[0]).toMatchObject({ near: 0.2, far: 0.8 }));
    expect(rowButton("Depth mask, 20 to 80")).toBeInTheDocument();
    expect(sliderOf("Near")).toHaveAttribute("aria-valuemax", "80");
    expect(sliderOf("Far")).toHaveAttribute("aria-valuemin", "20");
    await type("Near", "95");
    await waitFor(() => expect(rows(2)[0]).toMatchObject({ near: 0.8 }));
    await type("Softness", "10");
    await waitFor(() => expect(rows(2)[0]).toMatchObject({ softness: 0.1 }));
    const last = spy.mock.calls.at(-1)?.[1] as SettingsPatch;
    expect(last).toEqual({ nodes: { "2": { masks: { edit: { "0": { softness: 0.1 } } } } } });
  });

  it("Reset puts the row back as a new Depth row", async () => {
    await setup(GRAPH([{ ...DEPTH, near: 0.3, far: 0.6, softness: 0.1 } as MaskItem]));
    await mapMade();
    await h!.user.click(rowButton(/^Depth mask/));
    await h!.user.click(within(card()).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(rows(2)[0]).toMatchObject({ near: 0, far: 0.5, softness: 0.5 }));
  });
});

describe("Pick depth", () => {
  it("is a toggle of the card that turns the crosshair on the photo on and off, and one at a time with Pick colour", async () => {
    await setup(GRAPH());
    await mapMade();
    await h!.user.click(rowButton(/^Depth mask/));
    const pick = within(card()).getByRole("button", { name: "Pick depth" });
    await h!.user.click(pick);
    expect(pick).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("application", { name: /^Pick depth/ })).toBeInTheDocument();
    expect(screen.getByText("Click the part of the photo to select. Esc stops.")).toBeInTheDocument();
    await h!.user.click(pick);
    expect(pick).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("application", { name: /^Pick depth/ })).toBeNull();
  });

  it("a click on the photo sets Near and Far around the depth there, as one undo step, and the tool stays on", async () => {
    const spy = await setup(GRAPH());
    await mapMade();
    await h!.user.click(rowButton(/^Depth mask/));
    await h!.user.click(within(card()).getByRole("button", { name: "Pick depth" }));
    const layer = screen.getByRole("application", { name: /^Pick depth/ });
    // jsdom has no layout: a 300 x 200 cell, and a click at the middle of the bottom (the near ground of the mock's flat scene).
    layer.getBoundingClientRect = () => ({ x: 0, y: 0, left: 0, top: 0, right: 300, bottom: 200, width: 300, height: 200, toJSON: () => ({}) });
    spy.mockClear();
    fireEvent.click(layer, { clientX: 150, clientY: 190 });
    await waitFor(() => expect(rows(2)[0]).not.toMatchObject({ near: 0, far: 0.5 }));
    const row = rows(2)[0] as Extract<MaskItem, { kind: "depth" }>;
    expect(row.far - row.near).toBeCloseTo(0.2, 2);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(within(card()).getByRole("button", { name: "Pick depth" })).toHaveAttribute("aria-pressed", "true");
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(rows(2)[0]).toMatchObject({ near: 0, far: 0.5 }));
  });
});

describe("Focus plane", () => {
  it("sets Near and Far around the plane the camera focused on, as one undo step", async () => {
    const spy = await setup(GRAPH());
    await mapMade();
    await h!.user.click(rowButton(/^Depth mask/));
    const button = within(card()).getByRole("button", { name: "Focus plane" });
    expect(button).not.toHaveAttribute("aria-disabled", "true");
    spy.mockClear();
    await h!.user.click(button);
    await waitFor(() => expect(rows(2)[0]).toMatchObject({ near: 0.05, far: 0.25 }));
    expect(spy).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(FOCUS_AT_CENTRE)).toBeNull();
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(rows(2)[0]).toMatchObject({ near: 0, far: 0.5 }));
  });

  it("is off, with the reason, for a photo that has no focus point, and writes nothing", async () => {
    const spy = await setup(GRAPH(), {}, /^Node 2, Haze/, 0);
    await mapMade();
    await h!.user.click(rowButton(/^Depth mask/));
    await waitFor(() => expect(within(card()).getByRole("button", { name: "Focus plane" })).toHaveAttribute("aria-disabled", "true"));
    // The line in the card (the button's own tooltip says it too).
    expect(card().querySelector("p.ym-row-note")).toHaveTextContent(NO_FOCUS_POINT);
    spy.mockClear();
    await h!.user.click(within(card()).getByRole("button", { name: "Focus plane" }));
    expect(spy).not.toHaveBeenCalled();
  });

  it("says the camera may not have recorded where it focused when the point is the exact centre, until the band is changed", async () => {
    await setup(GRAPH(), {}, /^Node 2, Haze/, 11);
    await mapMade();
    await h!.user.click(rowButton(/^Depth mask/));
    await h!.user.click(within(card()).getByRole("button", { name: "Focus plane" }));
    expect(await within(card()).findByText(FOCUS_AT_CENTRE)).toBeInTheDocument();
    await type("Far", "60");
    await waitFor(() => expect(within(card()).queryByText(FOCUS_AT_CENTRE)).toBeNull());
  });
});

describe("Show depth", () => {
  it("shows the depth map in the viewer (no patch), is exclusive with Show mask and the eyes, and ends with the card", async () => {
    const spy = await setup(GRAPH());
    await mapMade();
    await h!.user.click(rowButton(/^Depth mask/));
    const show = within(card()).getByRole("button", { name: "Show depth" });
    spy.mockClear();
    await h!.user.click(show);
    expect(show).toHaveAttribute("aria-pressed", "true");
    expect(mask()).toEqual({ node: 2, style: "mask_only", row: 0, depth: true });
    expect(spy).not.toHaveBeenCalled();
    // The row's eye turns it off, and it turns the eye off.
    await h!.user.click(within(rowList()).getByRole("button", { name: "Show this mask" }));
    expect(show).toHaveAttribute("aria-pressed", "false");
    expect(mask()).toEqual({ node: 2, style: "overlay", row: 0 });
    await h!.user.click(show);
    expect(mask()).toMatchObject({ depth: true });
    expect(within(rowList()).getByRole("button", { name: "Show this mask" })).toBeInTheDocument();
    // Show mask (M) turns it off.
    await h!.user.keyboard("m");
    expect(show).toHaveAttribute("aria-pressed", "false");
    expect(mask()).toEqual({ node: 2, style: "overlay" });
    // Esc ends it, and so does choosing another node.
    await h!.user.click(show);
    expect(mask()).toMatchObject({ depth: true });
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(mask()).toBeNull());
    await h!.user.click(show);
    fireEvent.click(screen.getByRole("button", { name: /^Node 3, Warmth/ }));
    await waitFor(() => expect(h!.store.getState().showDepth).toBeNull());
  });
});

describe("a map that cannot be made", () => {
  it("says Can't be made in the row, the sentence and Try again in the card, and Try again starts the map again", async () => {
    await setup(GRAPH(), { depth: "fail" });
    await waitFor(() => expect(h!.store.getState().depthFailed[h!.store.getState().session!.photos[3]!.id]).toBe(DEPTH_FAILED));
    expect(rowButton("Depth mask, Can't be made")).toBeInTheDocument();
    await h!.user.click(rowButton(/^Depth mask/));
    const c = card();
    expect(within(c).getByText(`${DEPTH_FAILED} ${FAILED_ADVICE}`)).toBeInTheDocument();
    expect(within(c).queryByRole("slider")).toBeNull();
    const retry = vi.spyOn(h!.api, "retryDepth");
    await h!.user.click(within(c).getByRole("button", { name: "Try again" }));
    expect(retry).toHaveBeenCalledTimes(1);
    // The map is made this time: the card is the normal one, the row has its band back.
    await waitFor(() => expect(within(card()).getAllByRole("slider").length).toBeGreaterThan(3));
    expect(rowButton("Depth mask, 0 to 50")).toBeInTheDocument();
  });

  it("fails again when the model does, and the card says so again", async () => {
    await setup(GRAPH(), { depth: "fail-always" });
    await h!.user.click(rowButton(/^Depth mask/));
    await h!.user.click(await within(tools()).findByRole("button", { name: "Try again" }));
    expect(await within(card()).findByRole("button", { name: "Try again" })).toBeInTheDocument();
    expect(within(card()).getByText(`${DEPTH_FAILED} ${FAILED_ADVICE}`)).toBeInTheDocument();
  });
});

describe("where the depth weights are missing", () => {
  it("the row a file brought says so, selects nothing and has no card", async () => {
    await setup(GRAPH(), { maskKinds: ["object", "background", "linear"] });
    const name = "Depth mask, Depth masks need the depth model. Reinstall the app.";
    expect(rowButton(name)).toBeInTheDocument();
    await h!.user.click(rowButton(name));
    expect(within(tools()).queryByRole("region", { name: "Depth" })).toBeNull();
    expect(within(rowList()).queryByText("Finding…")).toBeNull();
    // The row is still in the settings.
    expect(rows(2)).toHaveLength(1);
  });
});
