/**
 * The Blur node's panel (`docs/33` §2.1 to §2.6, §5.3; BL-13): the Type control, each type's controls, the Mask edges card, the card's Reset
 * and the undo steps. The screen runs on the mock, as in a browser.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { NodeLink, SettingsPatch } from "../../api/types";
import { nodeOf } from "../../state/settingsGraph";
import type { BlurItem } from "../../state/settingsGraph";
import { openPhotos, openTab, renderApp, tools } from "../../test/harness";
import { mockRect } from "../../test/raf";
import type { Harness, UpdateSpy } from "../../test/harness";
import { BLUR_TEXT } from "./blurParts";

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

/** Main, then a Blur node (2), then the Output. */
const WITH_BLUR = (more: Record<string, unknown> = {}): SettingsPatch => ({
  nodes: { "2": { kind: "blur", name: "Blur", pos: [1, 0], ...more } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

async function setup(settings: SettingsPatch = WITH_BLUR()): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: settings } });
  await openPhotos(h);
  await openTab(h, "Light");
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  fireEvent.click(screen.getByRole("button", { name: /^Node 2, Blur/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /^Node 2, Blur/ })).toHaveAttribute("aria-pressed", "true"));
  return vi.spyOn(h.api, "updateSettings");
}

const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const node = (): BlurItem => nodeOf(h!.store.currentSettings()!.settings, 2) as BlurItem;
const card = (name: string): HTMLElement => within(tools()).getByRole("region", { name });
const typeControl = (): HTMLElement => within(card("Blur")).getByRole("radiogroup", { name: "Blur type" });
const typeRadio = (name: string): HTMLElement => within(typeControl()).getByRole("radio", { name });
const field = (label: string): HTMLInputElement => within(card("Blur")).getByRole("textbox", { name: `${label} value` }) as HTMLInputElement;
async function type(label: string, text: string): Promise<void> {
  const box = field(label);
  await h!.user.click(box);
  await h!.user.clear(box);
  await h!.user.type(box, `${text}{Enter}`);
}

describe("the Blur tab", () => {
  it("opens with the Type control on Soft, Radius 20, its line and the Mask edges card", async () => {
    await setup();
    expect(typeRadio("Soft")).toHaveAttribute("aria-checked", "true");
    expect(within(typeControl()).getAllByRole("radio").map((r) => r.textContent)).toEqual(["Soft", "Lens", "Motion", "Zoom"]);
    expect(field("Radius").value).toBe("20");
    expect(within(card("Blur")).getByText(BLUR_TEXT.softLine)).toBeInTheDocument();
    expect(within(card("Blur")).queryByRole("button", { name: "Reset Blur" })).toBeNull();
    const edges = card("Mask edges");
    expect(within(edges).getByRole("switch", { name: "Respect mask edges" })).toBeChecked();
    expect(within(edges).getByText(BLUR_TEXT.edgesOn)).toBeInTheDocument();
    expect(within(edges).getByText(BLUR_TEXT.edgesNoMask)).toBeInTheDocument();
  });

  it("writes Radius as one patch and a typed value lands on the node", async () => {
    const spy = await setup();
    await type("Radius", "50");
    await waitFor(() => expect(node().soft.radius).toBe(50));
    expect(patches(spy).at(-1)!.nodes!["2"]).toEqual({ soft: { radius: 50 } });
    expect(within(card("Blur")).getByRole("button", { name: "Reset Blur" })).toBeInTheDocument();
  });

  it("keeps each type's values when the type changes and changes only the type", async () => {
    const spy = await setup();
    await type("Radius", "50");
    spy.mockClear();
    await h!.user.click(typeRadio("Lens"));
    await waitFor(() => expect(node().type).toBe("lens"));
    expect(patches(spy)).toHaveLength(1);
    expect(patches(spy)[0]!.nodes!["2"]).toEqual({ type: "lens" });
    expect(field("Radius").value).toBe("20");
    await type("Radius", "30");
    await h!.user.click(typeRadio("Motion"));
    await waitFor(() => expect(within(card("Blur")).getByText(BLUR_TEXT.motionLine)).toBeInTheDocument());
    await h!.user.click(typeRadio("Soft"));
    await waitFor(() => expect(field("Radius").value).toBe("50"));
    await h!.user.click(typeRadio("Lens"));
    await waitFor(() => expect(field("Radius").value).toBe("30"));
    expect(node().soft.radius).toBe(50);
    expect(node().lens.radius).toBe(30);
  });

  it("moves between types with the arrow keys", async () => {
    await setup();
    typeRadio("Soft").focus();
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(node().type).toBe("lens"));
    await h!.user.keyboard("{End}");
    await waitFor(() => expect(node().type).toBe("zoom"));
  });
});

describe("Lens", () => {
  const lens = { type: "lens" };

  it("has Radius, Shape, Rotation and Highlights, with Rotation off while the shape is round", async () => {
    await setup(WITH_BLUR(lens));
    expect(field("Radius").value).toBe("20");
    expect(within(card("Blur")).getByRole("radiogroup", { name: "Shape" })).toBeInTheDocument();
    expect(within(card("Blur")).getAllByRole("radio", { name: /^(Round|[5-9] blades)$/ })).toHaveLength(6);
    expect(within(card("Blur")).getByRole("radio", { name: "Round" })).toHaveAttribute("aria-checked", "true");
    expect(within(card("Blur")).getAllByRole("slider", { name: "Rotation" })[0]).toHaveAttribute("data-disabled");
    expect(field("Highlights").value).toBe("0");
    expect(within(card("Blur")).getByText(BLUR_TEXT.lensLine)).toBeInTheDocument();
  });

  it("turns Rotation on with a polygon and writes blades, rotation and highlights", async () => {
    const spy = await setup(WITH_BLUR(lens));
    await h!.user.click(within(card("Blur")).getByRole("radio", { name: "6 blades" }));
    await waitFor(() => expect(node().lens.blades).toBe(6));
    expect(patches(spy).at(-1)!.nodes!["2"]).toEqual({ lens: { blades: 6 } });
    expect(within(card("Blur")).getAllByRole("slider", { name: "Rotation" })[0]).not.toHaveAttribute("data-disabled");
    await type("Rotation", "15");
    await waitFor(() => expect(node().lens.rotation_deg).toBe(15));
    await type("Highlights", "60");
    await waitFor(() => expect(node().lens.highlights).toBe(60));
    expect(field("Rotation").value).toBe("15°");
  });
});

describe("Motion", () => {
  it("has Length and Angle and writes the motion group", async () => {
    const spy = await setup(WITH_BLUR({ type: "motion" }));
    expect(field("Length").value).toBe("20");
    expect(field("Angle").value).toBe("0°");
    await type("Angle", "45");
    await waitFor(() => expect(node().motion.angle_deg).toBe(45));
    expect(patches(spy).at(-1)!.nodes!["2"]).toEqual({ motion: { angle_deg: 45 } });
    await type("Length", "70");
    await waitFor(() => expect(node().motion.length).toBe(70));
    expect(within(card("Blur")).getByText(BLUR_TEXT.motionLine)).toBeInTheDocument();
  });
});

describe("Zoom", () => {
  it("has Amount, a Centre readout and a Pick centre toggle that is off", async () => {
    const spy = await setup(WITH_BLUR({ type: "zoom" }));
    expect(field("Amount").value).toBe("20");
    expect(within(card("Blur")).getByRole("button", { name: "Pick centre" })).toHaveAttribute("aria-pressed", "false");
    expect(within(card("Blur")).getByText(BLUR_TEXT.centreAtMiddle)).toBeInTheDocument();
    expect(within(card("Blur")).queryByRole("button", { name: "Reset centre" })).toBeNull();
    await type("Amount", "80");
    await waitFor(() => expect(node().zoom.amount).toBe(80));
    expect(patches(spy).at(-1)!.nodes!["2"]).toEqual({ zoom: { amount: 80 } });
  });

  it("says when the centre is moved and Reset puts it back as one undo step", async () => {
    const spy = await setup(WITH_BLUR({ type: "zoom", zoom: { center: [0.3, 0.6] } }));
    expect(within(card("Blur")).getByText(BLUR_TEXT.centreMoved)).toBeInTheDocument();
    spy.mockClear();
    await h!.user.click(within(card("Blur")).getByRole("button", { name: "Reset centre" }));
    await waitFor(() => expect(node().zoom.center).toEqual([0.5, 0.5]));
    expect(patches(spy)).toHaveLength(1);
    expect(patches(spy)[0]!.nodes!["2"]).toEqual({ zoom: { center: [0.5, 0.5] } });
    expect(within(card("Blur")).getByText(BLUR_TEXT.centreAtMiddle)).toBeInTheDocument();
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(node().zoom.center).toEqual([0.3, 0.6]));
  });
});

describe("the Zoom centre on the photo", () => {
  /** The photo cell, 600 x 400, and the mock told so: its own mapping answers for it. */
  async function placeCell(): Promise<void> {
    const cell = document.querySelector<HTMLElement>("[data-viewport]")!;
    mockRect(cell, { left: 0, top: 0, width: 600, height: 400 });
    await h!.api.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
    await act(async () => {
      fireEvent(window, new Event("resize"));
    });
  }
  const handle = (): HTMLElement => screen.getByRole("button", { name: /^Zoom centre/ });
  const layer = (): HTMLElement | null => document.querySelector<HTMLElement>("[data-zoom-centre]");
  const pickButton = (): HTMLElement => within(card("Blur")).getByRole("button", { name: "Pick centre" });

  it("is drawn at the centre the core maps, with eight rays, for a Zoom node only", async () => {
    await setup(WITH_BLUR({ type: "zoom", zoom: { center: [0.25, 0.5] } }));
    await placeCell();
    await waitFor(() => expect(layer()).not.toBeNull());
    await waitFor(() => expect(parseFloat(handle().style.left)).toBeCloseTo(150, 3));
    expect(parseFloat(handle().style.top)).toBeCloseTo(200, 3);
    expect(layer()!.querySelectorAll(".ym-zoom-centre__ray")).toHaveLength(8);
    // Another type has no handle, and the type again has it.
    await h!.user.click(within(card("Blur")).getByRole("radio", { name: "Soft" }));
    await waitFor(() => expect(layer()).toBeNull());
  });

  it("a drag of the handle writes the centre where the pointer is, a patch per move, and is one undo step", async () => {
    const spy = await setup(WITH_BLUR({ type: "zoom" }));
    await placeCell();
    await waitFor(() => expect(handle()).toBeInTheDocument());
    spy.mockClear();
    fireEvent.pointerDown(handle(), { clientX: 300, clientY: 200, button: 0, buttons: 1, pointerId: 1 });
    fireEvent.pointerMove(window, { clientX: 360, clientY: 240, pointerId: 1 });
    await waitFor(() => expect(node().zoom.center).toEqual([0.6, 0.6]));
    fireEvent.pointerMove(window, { clientX: 420, clientY: 300, pointerId: 1 });
    await waitFor(() => expect(node().zoom.center).toEqual([0.7, 0.75]));
    fireEvent.pointerUp(window, { clientX: 420, clientY: 300, pointerId: 1 });
    await waitFor(() => expect(patches(spy).at(-1)!.nodes!["2"]).toEqual({ zoom: { center: [0.7, 0.75] } }));
    await waitFor(() => expect(h!.store.getState().canUndo).toBe(true));
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(node().zoom.center).toEqual([0.5, 0.5]));
    expect(h!.store.getState().canUndo).toBe(false);
  });

  it("the arrow keys nudge the handle 1 px, Shift 10 px, each one undo step", async () => {
    const spy = await setup(WITH_BLUR({ type: "zoom" }));
    await placeCell();
    await waitFor(() => expect(handle()).toBeInTheDocument());
    act(() => handle().focus());
    spy.mockClear();
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(node().zoom.center).toEqual([0.5017, 0.5]));
    await h!.user.keyboard("{Shift>}{ArrowDown}{/Shift}");
    await waitFor(() => expect(node().zoom.center).toEqual([0.5017, 0.525]));
    expect(patches(spy)).toHaveLength(2);
  });

  it("Pick centre is a toggle: on, a click on the photo sets the centre as one patch and ends the pick", async () => {
    const spy = await setup(WITH_BLUR({ type: "zoom" }));
    await placeCell();
    await h!.user.click(pickButton());
    expect(pickButton()).toHaveAttribute("aria-pressed", "true");
    expect(within(card("Blur")).getByText(BLUR_TEXT.centrePicking)).toBeInTheDocument();
    const catcher = await waitFor(() => {
      const el = document.querySelector<HTMLElement>("[data-zoom-catch]");
      expect(el).not.toBeNull();
      return el!;
    });
    mockRect(catcher.parentElement!, { left: 0, top: 0, width: 600, height: 400 });
    spy.mockClear();
    fireEvent.click(catcher, { clientX: 150, clientY: 100 });
    await waitFor(() => expect(node().zoom.center).toEqual([0.25, 0.25]));
    expect(patches(spy)).toHaveLength(1);
    expect(patches(spy)[0]!.nodes!["2"]).toEqual({ zoom: { center: [0.25, 0.25] } });
    await waitFor(() => expect(pickButton()).toHaveAttribute("aria-pressed", "false"));
    expect(document.querySelector("[data-zoom-catch]")).toBeNull();
    expect(within(card("Blur")).getByText(BLUR_TEXT.centreMoved)).toBeInTheDocument();
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(node().zoom.center).toEqual([0.5, 0.5]));
  });

  it("Esc cancels a pick and changes nothing; so does the toggle, and a change of type", async () => {
    const spy = await setup(WITH_BLUR({ type: "zoom" }));
    await placeCell();
    spy.mockClear();
    await h!.user.click(pickButton());
    await waitFor(() => expect(document.querySelector("[data-zoom-catch]")).not.toBeNull());
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(pickButton()).toHaveAttribute("aria-pressed", "false"));
    expect(document.querySelector("[data-zoom-catch]")).toBeNull();
    await h!.user.click(pickButton());
    await h!.user.click(pickButton());
    expect(pickButton()).toHaveAttribute("aria-pressed", "false");
    await h!.user.click(pickButton());
    await h!.user.click(within(card("Blur")).getByRole("radio", { name: "Lens" }));
    await waitFor(() => expect(document.querySelector("[data-zoom-catch]")).toBeNull());
    expect(patches(spy).filter((p) => (p.nodes?.["2"] as { zoom?: unknown } | undefined)?.zoom)).toHaveLength(0);
    expect(node().zoom.center).toEqual([0.5, 0.5]);
  });

  it("a click on the paper around the photo sets nothing and the pick stays on", async () => {
    await setup(WITH_BLUR({ type: "zoom" }));
    // A cell of 3:4 has paper above and below the 3:2 photo.
    const cell = document.querySelector<HTMLElement>("[data-viewport]")!;
    mockRect(cell, { left: 0, top: 0, width: 300, height: 400 });
    await h!.api.viewportSetRect({ x: 0, y: 0, width: 300, height: 400, scale: 1 });
    await act(async () => {
      fireEvent(window, new Event("resize"));
    });
    await h!.user.click(pickButton());
    const catcher = await waitFor(() => {
      const el = document.querySelector<HTMLElement>("[data-zoom-catch]");
      expect(el).not.toBeNull();
      return el!;
    });
    mockRect(catcher.parentElement!, { left: 0, top: 0, width: 300, height: 400 });
    fireEvent.click(catcher, { clientX: 150, clientY: 5 });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 30));
    });
    expect(node().zoom.center).toEqual([0.5, 0.5]);
    expect(pickButton()).toHaveAttribute("aria-pressed", "true");
  });
});

describe("Mask edges", () => {
  it("turns the switch off with its other line, and one undo turns it back on", async () => {
    const spy = await setup();
    const edges = card("Mask edges");
    await h!.user.click(within(edges).getByRole("switch", { name: "Respect mask edges" }));
    await waitFor(() => expect(node().respect_mask_edges).toBe(false));
    expect(patches(spy).at(-1)!.nodes!["2"]).toEqual({ respect_mask_edges: false });
    expect(within(edges).getByText(BLUR_TEXT.edgesOff)).toBeInTheDocument();
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(node().respect_mask_edges).toBe(true));
  });

  it("drops the no-mask line once the node has a mask row", async () => {
    await setup(WITH_BLUR({ masks: { invert: false, items: [{ kind: "linear" }] } }));
    expect(within(card("Mask edges")).queryByText(BLUR_TEXT.edgesNoMask)).toBeNull();
  });
});

describe("Reset", () => {
  it("the card's Reset sends the type and every value in one patch, keeps the masks, and one undo brings them back", async () => {
    const spy = await setup(WITH_BLUR({ type: "lens", soft: { radius: 70 }, lens: { radius: 40, blades: 6, rotation_deg: 15, highlights: 60 }, respect_mask_edges: false }));
    spy.mockClear();
    await h!.user.click(within(card("Blur")).getByRole("button", { name: "Reset Blur" }));
    await waitFor(() => expect(node().type).toBe("soft"));
    expect(patches(spy)).toHaveLength(1);
    expect(node().soft.radius).toBe(20);
    expect(node().lens).toEqual({ radius: 20, blades: 0, rotation_deg: 0, highlights: 0 });
    expect(node().respect_mask_edges).toBe(true);
    expect(within(card("Blur")).queryByRole("button", { name: "Reset Blur" })).toBeNull();
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(node().type).toBe("lens"));
    expect(node().lens.blades).toBe(6);
    expect(node().soft.radius).toBe(70);
  });

  it("a slider's own Reset goes back to that control's default", async () => {
    await setup(WITH_BLUR({ soft: { radius: 70 } }));
    const row = [within(card("Blur")).getByRole("button", { name: "Reset Radius" })];
    await h!.user.click(row[0]!);
    await waitFor(() => expect(node().soft.radius).toBe(20));
  });
});
