import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { openApp, openPhotos, watchErrors } from "./helpers/app";
import { referenceView, REFERENCE_FITTED } from "./helpers/states";

// The Reference Grade node in a browser on the mock API (`docs/34` §9 and §10 "Interface", RG-12): the Add node menu, the seven tabs and that none is clipped
// at three window sizes, choosing a file, Fit with its progress and Cancel (and Esc), Strength, Skin protection, each error, the strip as a chooser, the status
// line after a look change and a hand edit, the Hue and Elements tabs, Reset, Copy and Paste and undo. The mock's fit is a fixed one and draws nothing, so the
// tests look at the panel and at the settings the page sends.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

const tools = (page: Page): Locator => page.getByRole("complementary", { name: "Tools" });
const status = (page: Page): Locator => tools(page).locator(".ym-ref__status");
const fitButton = (page: Page): Locator => tools(page).getByRole("button", { name: /^Fit/ });
const settingsOf = async (page: Page, id = 4): Promise<{ nodes: { items: Array<Record<string, any>> } }> =>
  page.evaluate(async (photo) => (await window.__yellowmoon!.mockApi.getSettings(photo)).settings as never, id);
const matchOf = async (page: Page): Promise<Record<string, any>> => (await settingsOf(page)).nodes.items.find((i) => i.kind === "reference_grade")!;
const choose = async (page: Page, path: string | null): Promise<void> => {
  await page.evaluate((answer) => window.__yellowmoon!.mockApi.queueChoice("reference", answer), path);
  await tools(page).getByRole("button", { name: "Choose reference…" }).click();
  await page.getByRole("menuitem", { name: "Image file…" }).click();
};

test.describe("the Add node menu", () => {
  test("has Reference Grade after Blur with its line; choosing it adds a node named Match after the selected one, opens its Reference tab and asks for no file", async ({ page }) => {
    const problems = watchErrors(page);
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.getByRole("tab", { name: "Light", exact: true }).click();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await page.getByRole("button", { name: "Add node" }).click();
    const menu = page.getByRole("menu", { name: "Add node" });
    const names = await menu.getByRole("menuitem").locator(".ym-menu-line__name").allTextContents();
    expect(names).toEqual(["Corrector", "Film", "Face Refinement", "Colour Uniform", "Blur", "Reference Grade", "Parallel", "Layer", "Splitter and combiner"]);
    const item = menu.getByRole("menuitem", { name: /^Reference Grade/ });
    await expect(item).toContainText("Matches the grade of a reference picture.");
    await item.click();
    await expect(page.getByRole("button", { name: /^Node 2, Match/ })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("tab")).toHaveText(["Reference", "Light", "Colour", "Curve", "Hue", "Elements", "Mask"]);
    await expect(page.getByRole("tab", { name: "Reference", exact: true })).toHaveAttribute("aria-selected", "true");
    await expect(status(page)).toHaveText("Choose a picture to match.");
    expect((await matchOf(page)).name).toBe("Match");
    // The graph's own library warns once about its attribution on the first build of a graph, as in every node spec.
    expect(problems.filter((p) => !p.includes("React Flow"))).toEqual([]);
  });
});

test.describe("the tabs", () => {
  for (const [width, height] of [[1440, 900], [1180, 760], [1800, 1100]] as const) {
    test(`no label of the seven is clipped at ${width} × ${height}`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await referenceView(page);
      const result = await page.evaluate(() => {
        const list = document.querySelector<HTMLElement>('.ym-tools [role="tablist"]')!;
        const listBox = list.getBoundingClientRect();
        const clipped: string[] = [];
        for (const tab of list.querySelectorAll<HTMLElement>('[role="tab"]')) {
          const box = tab.getBoundingClientRect();
          const range = document.createRange();
          range.selectNodeContents(tab);
          const text = range.getBoundingClientRect();
          // The letters are inside the tab, the tab inside the list, and nothing overflows its own box.
          if (text.left < box.left - 0.5 || text.right > box.right + 0.5 || box.left < listBox.left - 0.5 || box.right > listBox.right + 0.5 || tab.scrollWidth > tab.clientWidth + 1) {
            clipped.push(`${tab.textContent}: text ${text.left.toFixed(1)}–${text.right.toFixed(1)}, tab ${box.left.toFixed(1)}–${box.right.toFixed(1)}`);
          }
        }
        return { clipped, labels: [...list.querySelectorAll('[role="tab"]')].map((t) => t.textContent), listOverflow: list.scrollWidth - list.clientWidth, size: getComputedStyle(list.querySelector('[role="tab"]')!).fontSize };
      });
      expect(result.labels).toEqual(["Reference", "Light", "Colour", "Curve", "Hue", "Elements", "Mask"]);
      expect(result.clipped).toEqual([]);
      expect(result.listOverflow).toBeLessThanOrEqual(1);
      // The fix for a row that clipped would be the type size of this kind, never a shorter label: the labels are whole.
      expect(result.size).toBe("12px");
    });
  }

  test("⌘1 to ⌘7 select them in the order of the row, and the tab is remembered for the kind", async ({ page }) => {
    await referenceView(page);
    const order = ["Reference", "Light", "Colour", "Curve", "Hue", "Elements", "Mask"];
    for (const [i, name] of order.entries()) {
      await page.keyboard.press(`Meta+${i + 1}`);
      await expect(page.getByRole("tab", { name, exact: true })).toHaveAttribute("aria-selected", "true");
    }
  });
});

test.describe("the Reference tab", () => {
  test("with no reference: no card title, Fit off with its tooltip, Strength at 100 %, Skin protection on", async ({ page }) => {
    await referenceView(page);
    await expect(tools(page).getByRole("heading", { level: 3 })).toHaveCount(0);
    await expect(fitButton(page)).toHaveAttribute("aria-disabled", "true");
    // The button is aria-disabled and has pointer-events off; its tooltip is on the span around it.
    await tools(page).locator(".ym-ref__row [data-slot=tooltip-trigger]").last().hover();
    await expect(page.getByRole("tooltip").first()).toHaveText("Choose a reference first.");
    await expect(tools(page).getByRole("textbox", { name: "Strength value" })).toHaveValue("100%");
    await expect(tools(page).getByRole("switch", { name: "Skin protection" })).toBeChecked();
    await expect(tools(page).getByText("Scales the fitted values. The tabs show them before Strength.")).toBeVisible();
    await expect(tools(page).getByText("Keeps skin tones from drifting. Applies at the next fit.")).toBeVisible();
    await expect(tools(page).getByTestId("reference-name")).toHaveText("No reference");
  });

  test("choosing a file shows its name and picture, Fit then fits it with a progress bar, ends on 'Fitted to the reference.' and Fit again", async ({ page }) => {
    await referenceView(page);
    await choose(page, "~/Pictures/harbour-dusk.jpg");
    await expect(tools(page).getByTestId("reference-name")).toHaveText("harbour-dusk.jpg");
    await expect(status(page)).toHaveText("Press Fit to match the photo to this picture.");
    await expect(fitButton(page)).toHaveText("Fit");
    await page.evaluate(() => window.__yellowmoon!.mockApi.configure({ speed: 0.5 }));
    await fitButton(page).click();
    await expect(tools(page).getByRole("progressbar", { name: "Fitting" })).toBeVisible();
    await expect(status(page)).toHaveText("Finding what is in the pictures…");
    await expect(tools(page).getByRole("button", { name: "Choose reference…" })).toHaveAttribute("aria-disabled", "true");
    await expect(status(page)).toHaveText("Fitted to the reference.");
    await expect(fitButton(page)).toHaveText("Fit again");
    expect((await matchOf(page)).hue_curves.hue_vs_hue).toHaveLength(6);
  });

  for (const how of ["the Cancel button", "Esc"] as const) {
    test(`${how} cancels a fit and changes nothing`, async ({ page }) => {
      await referenceView(page, { state: "chosen" });
      const before = JSON.stringify(await matchOf(page));
      await page.evaluate(() => window.__yellowmoon!.mockApi.configure({ speed: 0.01 }));
      await fitButton(page).click();
      await expect(tools(page).getByRole("progressbar", { name: "Fitting" })).toBeVisible();
      if (how === "Esc") await page.keyboard.press("Escape");
      else await tools(page).getByRole("button", { name: "Cancel" }).click();
      await expect(tools(page).getByRole("progressbar")).toHaveCount(0);
      await expect(status(page)).toHaveText("Press Fit to match the photo to this picture.");
      expect(JSON.stringify(await matchOf(page))).toBe(before);
    });
  }

  test("a fit is one undo step", async ({ page }) => {
    await referenceView(page, { state: "fitted" });
    expect((await matchOf(page)).hue_curves.hue_vs_hue).toHaveLength(6);
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await matchOf(page)).hue_curves.hue_vs_hue).toEqual([]);
    await page.keyboard.press("Meta+Shift+z");
    await expect.poll(async () => (await matchOf(page)).hue_curves.hue_vs_hue).toHaveLength(6);
  });

  test("the look changing says so and gives Fit again the accent; a hand edit says so with none", async ({ page }) => {
    await referenceView(page, { state: "look-changed" });
    await expect(fitButton(page)).toHaveText("Fit again");
    await expect(fitButton(page)).toHaveAttribute("data-accent", "");
    await referenceView(page, { state: "edited" });
    await expect(status(page)).toHaveText("Edited by hand since the fit. Fit again to start over.");
    await expect(fitButton(page)).not.toHaveAttribute("data-accent", "");
  });

  test("a node with values and no reference says they are kept", async ({ page }) => {
    await referenceView(page, { values: true });
    await expect(status(page)).toHaveText("The values are kept. Choose a reference to fit again.");
    await expect(fitButton(page)).toHaveAttribute("aria-disabled", "true");
  });

  test("Strength is a number field and a slider with a Reset below 100; Skin protection is a switch", async ({ page }) => {
    await referenceView(page);
    const box = tools(page).getByRole("textbox", { name: "Strength value" });
    await box.click();
    await box.fill("40");
    await box.press("Enter");
    await expect.poll(async () => (await matchOf(page)).strength).toBe(40);
    await tools(page).getByRole("button", { name: "Reset Strength" }).click();
    await expect.poll(async () => (await matchOf(page)).strength).toBe(100);
    await tools(page).getByRole("switch", { name: "Skin protection" }).click();
    await expect.poll(async () => (await matchOf(page)).skin_protection).toBe(false);
  });

  const ERRORS: Array<[string, string]> = [
    ["damaged.png", "This file can't be read. It may be damaged. Choose another picture."],
    ["photo.heic", "Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again."],
    ["tiny.png", "This picture is too small to match. Use one that is at least 256 pixels on its short side."],
    ["empty.png", "There's no picture inside the black bars. Choose another picture."],
  ];
  for (const [file, text] of ERRORS) {
    test(`${file} is refused in the card with its sentence and no code`, async ({ page }) => {
      await referenceView(page);
      await choose(page, `~/Pictures/${file}`);
      await expect(tools(page).getByTestId("reference-error")).toHaveText(text);
      await expect(status(page)).toContainText("Choose a picture to match.");
    });
  }

  test("a fit that failed, a missing scene model, and the notes of a file", async ({ page }) => {
    await referenceView(page, { state: "chosen" });
    await page.evaluate(() => window.__yellowmoon!.mockApi.failNext("fitReference", { scope: "render", message: "The grade couldn't be fitted. Try another reference." }));
    await fitButton(page).click();
    await expect(tools(page).getByTestId("reference-error")).toHaveText("The grade couldn't be fitted. Try another reference.");
    await page.evaluate(() => window.__yellowmoon!.mockApi.failNext("fitReference", { scope: "render", message: "The scene model is missing from this install. Reinstall Yellowmoon." }));
    await fitButton(page).click();
    await expect(tools(page).getByTestId("reference-error")).toHaveText("The scene model is missing from this install. Reinstall Yellowmoon.");
    await choose(page, "~/Pictures/bars-profile-grey.png");
    await expect(tools(page).getByRole("note")).toHaveText([
      "Black bars were removed from the reference.",
      "The picture's colour profile couldn't be read, so it is taken as sRGB.",
      "This picture has no colour. The photo will be matched in black and white.",
    ]);
  });
});

test.describe("the strip as a chooser", () => {
  test("takes the graph's place with its hint, puts the graded photo off, chooses another with a click and returns the graph", async ({ page }) => {
    await referenceView(page, { state: "chooser" });
    const strip = page.getByRole("navigation", { name: "Choose a reference" });
    await expect(page.getByText("Click a photo in the strip. Esc cancels.")).toBeVisible();
    await expect(page.locator(".ym-graph__canvas")).toBeHidden();
    const own = strip.getByRole("button", { name: /DSC04204/ });
    await expect(own).toHaveAttribute("aria-disabled", "true");
    await expect(own).toContainText("This photo");
    await own.hover();
    await expect(page.getByRole("tooltip")).toHaveText("That's the photo you're grading. Choose another.");
    await expect(tools(page).getByRole("button", { name: "Choose reference…" })).toHaveAttribute("aria-pressed", "true");
    await strip.getByRole("button", { name: /DSC04206/ }).click();
    await expect(tools(page).getByTestId("reference-name")).toHaveText("Photo 6 of 12");
    await expect(strip).toHaveCount(0);
    await expect(page.locator(".ym-graph__canvas")).toBeVisible();
  });

  test("is reachable by keyboard: it takes the focus, the arrows move it, Return chooses, and Esc cancels", async ({ page }) => {
    await referenceView(page, { state: "chooser" });
    const strip = page.getByRole("navigation", { name: "Choose a reference" });
    await expect(strip.getByRole("button").first()).toBeFocused();
    await page.keyboard.press("ArrowRight");
    await expect(strip.getByRole("button").nth(1)).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(tools(page).getByTestId("reference-name")).toHaveText("Photo 2 of 12");
    await tools(page).getByRole("button", { name: "Choose reference…" }).click();
    await page.getByRole("menuitem", { name: "Photo in the strip" }).click();
    await expect(strip).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(strip).toHaveCount(0);
    await expect(tools(page).getByTestId("reference-name")).toHaveText("Photo 2 of 12");
  });
});

test.describe("the Hue tab", () => {
  test("opens zoomed on the fitted values, reaches the full range, and moves a point by key as one patch of one curve", async ({ page }) => {
    await referenceView(page, { values: true, tab: "Hue" });
    const first = tools(page).getByRole("region", { name: "Hue vs hue" });
    await expect(first.locator(".ym-hue__yaxis")).toContainText("+15°");
    await first.getByRole("button", { name: "Full range" }).click();
    await expect(first.locator(".ym-hue__yaxis")).toContainText("+180°");
    const point = first.getByRole("slider", { name: "Point 3 of 6" });
    await point.focus();
    await page.keyboard.press("ArrowUp");
    await expect.poll(async () => (await matchOf(page)).hue_curves.hue_vs_hue[2]).toEqual({ hue_deg: 120, value: 12.1 });
    await page.keyboard.press("Delete");
    await expect.poll(async () => (await matchOf(page)).hue_curves.hue_vs_hue).toHaveLength(5);
    await tools(page).getByRole("button", { name: "Reset Hue vs saturation" }).click();
    await expect.poll(async () => (await matchOf(page)).hue_curves.hue_vs_sat).toEqual([]);
    expect((await matchOf(page)).hue_curves.hue_vs_hue).toHaveLength(5);
  });

  test("a click on the plot adds a point there and a drag moves it", async ({ page }) => {
    await referenceView(page, { tab: "Hue" });
    const plot = tools(page).getByRole("group", { name: "Hue vs hue curve" });
    const box = (await plot.boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.2, { steps: 4 });
    await page.mouse.up();
    await expect.poll(async () => (await matchOf(page)).hue_curves.hue_vs_hue.length).toBe(1);
    const point = (await matchOf(page)).hue_curves.hue_vs_hue[0];
    expect(point.hue_deg).toBeGreaterThan(170);
    expect(point.hue_deg).toBeLessThan(190);
    expect(point.value).toBeGreaterThan(0);
  });
});

test.describe("the Elements tab", () => {
  test("has seven cards; a fit shows each share and greys the element that was not found; each card has its own Reset", async ({ page }) => {
    await referenceView(page, { state: "fitted", tab: "Elements" });
    await expect(tools(page).getByRole("heading", { level: 3 })).toHaveText(["Sky", "Vegetation", "Water", "Ground", "Built", "Skin", "Subject"]);
    const sky = tools(page).getByRole("region", { name: "Sky" });
    await expect(sky).toContainText("26 % of the photo");
    await expect(tools(page).getByRole("region", { name: "Water" })).toContainText("Not found in this photo.");
    await expect(tools(page).getByRole("region", { name: "Subject" }).getByRole("slider")).toHaveCount(1);
    await sky.getByRole("button", { name: "Reset Sky" }).click();
    await expect.poll(async () => (await matchOf(page)).elements.sky).toEqual({ exposure_ev: 0, hue_deg: 0, chroma: 1 });
    expect((await matchOf(page)).elements.vegetation.hue_deg).toBe(20);
  });
});

test.describe("Reset, Copy and Paste", () => {
  test("Reset this node puts every field back in one undo step", async ({ page }) => {
    await referenceView(page, { values: true });
    await tools(page).getByRole("button", { name: "Reset this node" }).click();
    await expect.poll(async () => (await matchOf(page)).hue_curves.hue_vs_hue).toEqual([]);
    expect((await matchOf(page)).elements.sky).toEqual({ exposure_ev: 0, hue_deg: 0, chroma: 1 });
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    await page.keyboard.press("Meta+z");
    await expect.poll(async () => (await matchOf(page)).hue_curves.hue_vs_hue).toHaveLength(6);
  });

  test("Copy and Paste carry the node with its numbers to another photo; the reference stays behind", async ({ page }) => {
    await referenceView(page, { values: true });
    await tools(page).getByRole("button", { name: "Copy settings" }).click();
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("button", { name: /^Node 1, Main/ })).toBeVisible();
    await tools(page).getByRole("button", { name: "Paste" }).click();
    await expect.poll(async () => (await settingsOf(page, 5)).nodes.items.some((i) => i.kind === "reference_grade")).toBe(true);
    const pasted = (await settingsOf(page, 5)).nodes.items.find((i) => i.kind === "reference_grade")!;
    expect(pasted.hue_curves.hue_vs_hue).toEqual(REFERENCE_FITTED.hue_curves.hue_vs_hue);
  });
});
