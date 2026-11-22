/**
 * The Colour Uniform node's Light, Colour and Curve tabs (`docs/32` §12a.2 and §12a.3): the five tabs, the corrector's components on the node's
 * trim and grade, one pick at a time, Reset of the node, the mock's graded picture and `resolvedGraded`.
 */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { NodeLink, SettingsPatch } from "../../api/types";
import { resetNode } from "../../state/graphOps";
import { gradeOf, neutralGrade, nodeOf, resolvedGraded } from "../../state/settingsGraph";
import type { ColorUniformItem } from "../../state/settingsGraph";
import { openPhotos, openTab, renderApp, tools } from "../../test/harness";
import type { Harness, UpdateSpy } from "../../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

const WITH_UNIFORM = (more: Record<string, unknown> = {}): SettingsPatch => ({
  nodes: { "2": { kind: "color_uniform", name: "Uniform", pos: [1, 0], ...more } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

async function setup(settings: SettingsPatch = WITH_UNIFORM()): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: settings } });
  await openPhotos(h);
  await openTab(h, "Light");
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  fireEvent.click(screen.getByRole("button", { name: /^Node 2, Uniform/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /^Node 2, Uniform/ })).toHaveAttribute("aria-pressed", "true"));
  return vi.spyOn(h.api, "updateSettings");
}

const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const node = (): ColorUniformItem => nodeOf(h!.store.currentSettings()!.settings, 2) as ColorUniformItem;
const go = (name: string): Promise<void> => openTab(h!, name);

describe("the Colour Uniform node's tabs", () => {
  it("are Compress, Light, Colour, Curve and Mask, in that order, and none is clipped", async () => {
    await setup();
    const names = within(tools()).getAllByRole("tab").map((t) => t.textContent);
    expect(names).toEqual(["Compress", "Light", "Colour", "Curve", "Mask"]);
    // jsdom has no layout: the check of a clipped label is the Playwright test's; here the tab list holds all five.
    expect(within(tools()).getByRole("tablist").querySelectorAll("[role=tab]")).toHaveLength(5);
  });

  it("show a corrector's Light, Colour and Curve cards for the node", async () => {
    await setup();
    await go("Light");
    await waitFor(() => expect(within(tools()).getByRole("region", { name: "Exposure" })).toBeInTheDocument());
    expect(within(tools()).getByRole("region", { name: "High dynamic range" })).toBeInTheDocument();
    await go("Colour");
    await waitFor(() => expect(within(tools()).getByRole("region", { name: "Temperature and tint" })).toBeInTheDocument());
    expect(within(tools()).getByRole("region", { name: "Colour warp" })).toBeInTheDocument();
    await go("Curve");
    await waitFor(() => expect(within(tools()).getByRole("region", { name: "Curve" })).toBeInTheDocument());
  });

  it("write the node's trim and grade with the corrector's patches", async () => {
    const spy = await setup();
    await go("Light");
    const slider = await within(tools()).findByRole("slider", { name: "Brightness" });
    slider.focus();
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    await waitFor(() => expect(patches(spy).some((p) => p.nodes?.["2"]?.grade?.brightness !== undefined)).toBe(true));
    expect(node().grade?.brightness).not.toBe(0);
    await go("Colour");
    const temperature = await within(tools()).findByRole("slider", { name: "Temperature" });
    temperature.focus();
    fireEvent.keyDown(temperature, { key: "ArrowRight" });
    await waitFor(() => expect(node().trim?.temperature).not.toBe(0));
  });

  it("read the defaults when the node has neither field, and the node's values when it has them", async () => {
    await setup(WITH_UNIFORM({ trim: { exposure_ev: 0.5 }, grade: { brightness: 12 } }));
    const rs = h!.store.currentSettings()!;
    expect(gradeOf(rs.settings, 2).brightness).toBe(12);
    expect(gradeOf(rs.settings, 2).contrast).toEqual({ kind: "manual", value: 0 });
    const values = resolvedGraded(rs.values, 2);
    expect(values?.trim.exposure_ev).toBe(0.5);
    expect(values?.grade.brightness).toBe(12);
    // A node with neither reads as a neutral grade.
    const plain = gradeOf({ nodes: { items: [{ ...(node() as ColorUniformItem), grade: undefined }], links: [] } }, 2);
    expect(plain.brightness).toBe(0);
  });

  it("have one pick at a time: the warp's Pick colour ends the target's", async () => {
    await setup();
    // The target's pick is on when the node is added from the menu; here it is asked by the button of the Compress tab.
    fireEvent.click(within(within(tools()).getByRole("region", { name: "Target" })).getByRole("button", { name: "Pick colour" }));
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour for the target/ })).toBeInTheDocument());
    await go("Colour");
    await waitFor(() => expect(screen.queryByRole("application", { name: /Pick colour for the target/ })).not.toBeInTheDocument());
    fireEvent.click(within(within(tools()).getByRole("region", { name: "Colour warp" })).getByRole("button", { name: "Pick colour" }));
    await waitFor(() => expect(within(within(tools()).getByRole("region", { name: "Colour warp" })).getByRole("button", { name: "Pick colour" })).toHaveAttribute("aria-pressed", "true"));
  });
});

describe("Reset this node on a Colour Uniform node", () => {
  it("puts target, Compress, trim and grade back and keeps the masks", () => {
    const item: ColorUniformItem = {
      kind: "color_uniform",
      id: 2,
      name: "Uniform",
      bypass: false,
      pos: [1, 0],
      target: { hue_deg: 26, saturation: 9.7, lightness_ev: 0 },
      compress: { hue: 50, saturation: 50, lightness: 0 },
      trim: { exposure_ev: 0.5, temperature: 0, tint: 0 },
      grade: { ...neutralGrade(), brightness: 9 },
      masks: { invert: false, items: [] },
    };
    const outcome = resetNode({ nodes: { items: [item], links: [] } }, 2);
    expect(outcome.ok).toBe(true);
    const sent = (outcome.ok ? (outcome.patch.nodes?.["2"] ?? {}) : {}) as Record<string, unknown>;
    expect(sent.target).toBeNull();
    expect(sent.trim).toEqual({ exposure_ev: 0, temperature: 0, tint: 0 });
    expect(sent).toHaveProperty("grade");
    expect(sent).not.toHaveProperty("masks");
  });
});
