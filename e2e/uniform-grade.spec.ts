import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, watchErrors } from "./helpers/app";

// The Colour Uniform node's Light, Colour and Curve tabs in a browser on the mock API (`docs/32` §12a.2, milestone CG, CG-06): the five tabs and
// none clipped, the corrector's cards on the node, their sliders writing the node's trim and grade, one pick at a time, Reset of a card and of
// the node, Show mask, Copy and Paste, and undo of a Reset.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;
const picture = (from: number | "input", to: number | "output") => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const GRAPH = (more: Record<string, unknown> = {}) => ({
  nodes: { "2": { kind: "color_uniform", name: "Uniform", pos: [1, 0], target: { hue_deg: 40, saturation: 12, lightness_ev: 0.2 }, compress: { hue: 50, saturation: 50, lightness: 0 }, ...more } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

const tools = (page: Page): Locator => page.getByRole("complementary", { name: "Tools" });
const tab = (page: Page, name: string): Locator => tools(page).getByRole("tab", { name, exact: true });
const card = (page: Page, name: string): Locator => tools(page).getByRole("region", { name });

async function uniformOf(page: Page, id = 4): Promise<Record<string, any> | undefined> {
  const s = await page.evaluate(async (photo) => (await window.__yellowmoon!.mockApi.getSettings(photo)).settings as never, id);
  return (s as { nodes: { items: Array<Record<string, unknown>> } }).nodes.items.find((i) => i.kind === "color_uniform") as Record<string, any> | undefined;
}

async function view(page: Page, more: Record<string, unknown> = {}): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, { initialSettings: { DSC04204: GRAPH(more) } as never });
  await openPhotos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await page.getByRole("button", { name: /^Node 2, Uniform/ }).click();
  await expect(page.getByRole("button", { name: /^Node 2, Uniform/ })).toHaveAttribute("aria-pressed", "true");
}

test.describe("the Colour Uniform node's five tabs", () => {
  test("are Compress, Light, Colour, Curve and Mask, and no tab is clipped at three window sizes", async ({ page }) => {
    const problems = watchErrors(page);
    for (const size of [{ width: 1440, height: 900 }, { width: 1180, height: 760 }, { width: 1800, height: 1100 }]) {
      await page.setViewportSize(size);
      await view(page);
      await expect(tools(page).getByRole("tab")).toHaveText(["Compress", "Light", "Colour", "Curve", "Mask"]);
      // Every label is whole: the tab list's content is no wider than the list, and each tab's text fits its own box.
      const clipped = await tools(page).getByRole("tablist").evaluate((list) => {
        const wide = list.scrollWidth > list.clientWidth + 1;
        const cut = [...list.querySelectorAll<HTMLElement>("[role=tab]")].filter((t) => t.scrollWidth > t.clientWidth + 1).map((t) => t.textContent);
        return { wide, cut };
      });
      expect(clipped, `${size.width} x ${size.height}`).toEqual({ wide: false, cut: [] });
    }
    // React Flow's dev-mode attribution warning (`docs/14`: the notice is in THIRD-PARTY.md) fires once, a second after the graph mounts, so
    // whether it lands before this read depends on how fast the run is; nothing else may be said.
    expect(problems.filter((p) => !/React Flow: It seems like you are hiding the attribution/.test(p))).toEqual([]);
  });

  test("Light, Colour and Curve are a corrector's cards, and ⌘1 to ⌘5 select the five", async ({ page }) => {
    await view(page);
    await tab(page, "Light").click();
    await expect(card(page, "Exposure")).toBeVisible();
    await expect(card(page, "High dynamic range")).toBeVisible();
    await tab(page, "Colour").click();
    await expect(card(page, "Temperature and tint")).toBeVisible();
    await expect(card(page, "Colour balance")).toBeVisible();
    await expect(card(page, "Colour warp")).toBeVisible();
    await tab(page, "Curve").click();
    await expect(card(page, "Curve")).toBeVisible();
    for (const [key, name] of [["1", "Compress"], ["2", "Light"], ["3", "Colour"], ["4", "Curve"], ["5", "Mask"]] as const) {
      await page.keyboard.press(`Meta+${key}`);
      await expect(tab(page, name)).toHaveAttribute("aria-selected", "true");
    }
  });

  test("their sliders write the node's trim and grade and leave the pull alone", async ({ page }) => {
    await view(page);
    await tab(page, "Light").click();
    const brightness = card(page, "Exposure").getByRole("slider", { name: "Brightness" });
    await brightness.focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => (await uniformOf(page))?.grade?.brightness).toBeGreaterThan(0);
    await tab(page, "Colour").click();
    const temperature = card(page, "Temperature and tint").getByRole("slider", { name: "Temperature" });
    await temperature.focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => (await uniformOf(page))?.trim?.temperature).toBeGreaterThan(0);
    // The pull's own values are untouched by all of it.
    expect((await uniformOf(page))?.compress).toEqual({ hue: 50, saturation: 50, lightness: 0 });
  });

  test("Reset of a card and of the node put the grade back, and the node's Reset keeps the masks", async ({ page }) => {
    await view(page, { trim: { exposure_ev: 0.5, temperature: 0, tint: 0 }, grade: { brightness: 20 }, masks: { invert: false, items: [{ kind: "radial", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], radii: [0.3, 0.2], angle_deg: 0, feather: 0.3 }] } });
    await tab(page, "Light").click();
    await card(page, "Exposure").getByRole("button", { name: "Reset", exact: true }).click();
    await expect.poll(async () => (await uniformOf(page))?.trim?.exposure_ev).toBe(0);
    expect((await uniformOf(page))?.grade?.brightness).toBe(0);
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await uniformOf(page))?.trim?.exposure_ev).toBe(0.5);
    await page.getByRole("button", { name: "Reset this node" }).click();
    await expect.poll(async () => (await uniformOf(page))?.target).toBeNull();
    const node = await uniformOf(page);
    expect(node?.trim?.exposure_ev).toBe(0);
    expect(node?.grade?.brightness).toBe(0);
    expect(node?.masks.items).toHaveLength(1);
  });

  test("have one pick at a time: the warp's Pick colour ends the target's, and Esc ends the warp's", async ({ page }) => {
    await view(page);
    const target = card(page, "Target").getByRole("button", { name: "Pick colour" });
    await target.click();
    await expect(target).toHaveAttribute("aria-pressed", "true");
    await tab(page, "Colour").click();
    const warp = card(page, "Colour warp").getByRole("button", { name: "Pick colour" });
    await expect(warp).toHaveAttribute("aria-pressed", "false");
    await warp.click();
    await expect(warp).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("application", { name: /Pick colour for the target/ })).toHaveCount(0);
    await page.keyboard.press("Escape");
    await expect(warp).toHaveAttribute("aria-pressed", "false");
  });

  test("Show mask works on the node and Copy then Paste carry the trim and grade to another photo", async ({ page }) => {
    await view(page, { trim: { exposure_ev: 0.5, temperature: 0, tint: 0 }, grade: { brightness: 20 } });
    await page.keyboard.press("m");
    await expect(page.getByRole("button", { name: "Show mask" })).toHaveAttribute("aria-pressed", "true");
    await page.keyboard.press("m");
    await page.getByRole("button", { name: "Copy settings" }).click();
    await page.getByRole("button", { name: "Next photo" }).click();
    await page.getByRole("button", { name: "Paste" }).click();
    await expect.poll(async () => (await uniformOf(page, 5))?.grade?.brightness).toBe(20);
    expect((await uniformOf(page, 5))?.trim?.exposure_ev).toBe(0.5);
  });
});
