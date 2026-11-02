import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { failNext, openApp, openPhotos } from "./helpers/app";

// Pick white in the Colour tab (`docs/23` §8, `docs/11` §Pick white; N-06), in a real browser against the mocked API: the header has
// nothing of it, the White balance card has a row of its own, the hint is a pill on the photo, W opens the Colour tab and starts it, and it
// is off the moment the Colour tab is left.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;
const HINT = "Click something neutral: grey or white. Esc cancels.";

async function photos(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await openPhotos(page);
}

const header = (page: Page) => page.getByRole("navigation", { name: "Actions" });
const card = (page: Page) => page.getByRole("region", { name: "White balance" });
const toggle = (page: Page) => card(page).getByRole("button", { name: "Pick white" });
const layer = (page: Page) => page.getByRole("application", { name: /^Pick white\./ });
const pill = (page: Page) => page.getByRole("status").filter({ hasText: HINT });
const cell = (page: Page) => page.locator("[data-viewport]");
const tab = (page: Page, name: string) => page.getByRole("tab", { name });

/** A click at a share of the photo cell, which is where the mock puts the white. */
async function clickPhoto(page: Page, u: number, v: number): Promise<void> {
  const box = (await layer(page).boundingBox())!;
  await page.mouse.move(box.x + box.width * u, box.y + box.height * v);
  await page.mouse.click(box.x + box.width * u, box.y + box.height * v);
}

test.describe("the row in the White balance card", () => {
  test("is a row of its own between the light presets and Kelvin, with the eyedropper, the name and, at the right, 'or press W'", async ({ page }) => {
    await photos(page);
    await tab(page, "Colour").click();
    await expect(header(page).getByRole("button", { name: "Pick white" })).toHaveCount(0);
    const presets = (await card(page).getByRole("radiogroup", { name: "Light type" }).boundingBox())!;
    const kelvin = (await card(page).getByRole("slider", { name: "Kelvin" }).boundingBox())!;
    const button = (await toggle(page).boundingBox())!;
    const keys = (await card(page).locator(".ym-pick-row__keys").boundingBox())!;
    const chip = (await card(page).locator(".ym-pick-row kbd").boundingBox())!;
    // Its own row, below the presets and above Kelvin: neither the presets nor Kelvin share its height.
    expect(button.y).toBeGreaterThanOrEqual(presets.y + presets.height);
    expect(button.y + button.height).toBeLessThanOrEqual(kelvin.y);
    expect(keys.y + keys.height).toBeLessThanOrEqual(kelvin.y);
    // The toggle at the left (glasscn's small control, 32 px), the key at the right of the card, level with it.
    expect(button.height).toBe(32);
    expect(button.x).toBeLessThan(presets.x + 8);
    expect(keys.x).toBeGreaterThan(button.x + button.width);
    expect(Math.abs(presets.x + presets.width - (keys.x + keys.width))).toBeLessThanOrEqual(1);
    expect(Math.abs(button.y + button.height / 2 - (chip.y + chip.height / 2))).toBeLessThanOrEqual(1.5);
    await expect(card(page).locator(".ym-pick-row__keys")).toHaveText("or press W");
    await expect(toggle(page)).toHaveAttribute("data-slot", "toggle");
    await expect(toggle(page).locator("svg")).toBeVisible();
    // The card's header keeps AUTO and Reset.
    await expect(card(page).getByRole("button", { name: "Auto", pressed: true })).toBeVisible();
    await expect(card(page).getByRole("button", { name: "Reset" })).toBeVisible();
  });

  test("is yellow when pressed, as the header's toggles are, though the Colour tab's accent is green", async ({ page }) => {
    await photos(page);
    await tab(page, "Colour").click();
    await expect(toggle(page)).not.toHaveCSS("background-color", "rgb(255, 194, 14)");
    await toggle(page).click();
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "true");
    await expect(toggle(page)).toHaveCSS("background-color", "rgb(255, 194, 14)");
    await expect(toggle(page)).toHaveCSS("color", "rgb(0, 0, 0)");
    // The header's Crop, pressed, is the same yellow.
    await page.keyboard.press("Escape");
    await header(page).getByRole("button", { name: "Crop", exact: true }).click();
    await expect(header(page).getByRole("button", { name: "Crop", exact: true })).toHaveCSS("background-color", "rgb(255, 194, 14)");
  });
});

test.describe("picking from the card", () => {
  test("the pill, the crosshair, a click that sets the white and ends the mode with EDITED on the card", async ({ page }) => {
    await photos(page);
    await tab(page, "Colour").click();
    await expect(pill(page)).toHaveCount(0);
    await toggle(page).click();
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "true");
    // The pill: at the top centre of the photo, with the sentence, and it takes no click.
    await expect(pill(page)).toBeVisible();
    const photo = (await cell(page).boundingBox())!;
    const box = (await pill(page).boundingBox())!;
    expect(Math.abs(box.x + box.width / 2 - (photo.x + photo.width / 2))).toBeLessThanOrEqual(1);
    expect(box.y - photo.y).toBeGreaterThanOrEqual(8);
    expect(box.y - photo.y).toBeLessThanOrEqual(16);
    await expect(pill(page)).toHaveCSS("pointer-events", "none");
    await expect(pill(page).locator("svg")).toBeVisible();
    // The cursor is a crosshair, and the crosshair drawn on the photo is white with a dark outline.
    await expect(layer(page)).toHaveCSS("cursor", "crosshair");
    const at = (await layer(page).boundingBox())!;
    await page.mouse.move(at.x + at.width * 0.3, at.y + at.height * 0.5);
    const mark = layer(page).locator("svg.ym-pick-layer__mark");
    await expect(mark).toBeVisible();
    await expect(mark.locator('[data-part="line"]')).toHaveAttribute("stroke", "#fff");
    expect(((await mark.boundingBox())!).width).toBe(48);
    // A click on the pill's place goes to the photo under it.
    await clickPhoto(page, 0.5, (box.y + box.height / 2 - at.y) / at.height);
    await expect(card(page).getByText("Edited")).toBeVisible();
    // The mode has ended: no layer, no pill, the toggle is up, and the Colour tab is still the one shown.
    await expect(layer(page)).toHaveCount(0);
    await expect(pill(page)).toHaveCount(0);
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "false");
    await expect(tab(page, "Colour")).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("textbox", { name: "Kelvin value" })).not.toHaveValue("5450 K");
  });

  test("a spot the core refuses shows its words in the error bar, changes nothing and ends the mode", async ({ page }) => {
    await photos(page);
    await tab(page, "Colour").click();
    await toggle(page).click();
    await failNext(page, "viewportInput", { scope: "render", message: "That spot is too bright to be a white. Click something grey." });
    await clickPhoto(page, 0.4, 0.4);
    await expect(page.locator(".ym-error-bar")).toContainText("That spot is too bright to be a white. Click something grey.");
    await expect(card(page).getByText("Edited")).toHaveCount(0);
    await expect(layer(page)).toHaveCount(0);
    await expect(page.getByRole("textbox", { name: "Kelvin value" })).toHaveValue("5450 K");
  });

  test("the keyboard path of the layer is unchanged: it takes the focus, the arrows move the crosshair, Return picks", async ({ page }) => {
    await photos(page);
    await tab(page, "Colour").click();
    await toggle(page).click();
    await expect(layer(page)).toBeFocused();
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("ArrowUp");
    await expect(layer(page).locator("svg.ym-pick-layer__mark")).toBeVisible();
    await page.keyboard.press("Enter");
    await expect(card(page).getByText("Edited")).toBeVisible();
    await expect(layer(page)).toHaveCount(0);
  });
});

test.describe("the pill beside the CPU note", () => {
  test("'Rendering on the CPU' goes down one pill while the hint is up, and takes its place again when the hint goes", async ({ page }) => {
    await photos(page);
    await page.keyboard.press("w");
    await expect(pill(page)).toBeVisible();
    await page.evaluate(() => window.__yellowmoon!.mockApi.emitEvent("viewport_busy", { busy: true, cpu: true }));
    const note = page.getByRole("status").filter({ hasText: "Rendering on the CPU" });
    await expect(note).toBeVisible();
    const hint = (await pill(page).boundingBox())!;
    const cpu = (await note.boundingBox())!;
    const photo = (await cell(page).boundingBox())!;
    // The same centre, the note under the hint without touching it, and neither takes a pointer event.
    expect(Math.abs(hint.x + hint.width / 2 - (cpu.x + cpu.width / 2))).toBeLessThanOrEqual(1);
    expect(cpu.y).toBeGreaterThanOrEqual(hint.y + hint.height);
    await expect(note).toHaveCSS("pointer-events", "none");
    await page.keyboard.press("Escape");
    await expect(pill(page)).toHaveCount(0);
    await expect(note).toBeVisible();
    expect(Math.round(((await note.boundingBox())!).y - photo.y)).toBe(12);
  });
});

test.describe("W", () => {
  test("opens the Colour tab and starts picking from another tab; a click picks; Esc and W end it", async ({ page }) => {
    await photos(page);
    await expect(tab(page, "Look")).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("w");
    await expect(tab(page, "Colour")).toHaveAttribute("aria-selected", "true");
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "true");
    await expect(pill(page)).toBeVisible();
    await expect(layer(page)).toBeVisible();
    // Esc ends it and stays on Colour; W starts it again and W ends it.
    await page.keyboard.press("Escape");
    await expect(layer(page)).toHaveCount(0);
    await expect(tab(page, "Colour")).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("w");
    await expect(layer(page)).toBeVisible();
    await page.keyboard.press("w");
    await expect(layer(page)).toHaveCount(0);
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "false");
    // And a click picks.
    await page.keyboard.press("w");
    await clickPhoto(page, 0.3, 0.15);
    await expect(card(page).getByText("Edited")).toBeVisible();
    await expect(layer(page)).toHaveCount(0);
  });

  test("from the keyboard alone: Tab to the toggle, Return starts it, and the layer takes the focus", async ({ page }) => {
    await photos(page);
    await tab(page, "Colour").click();
    await toggle(page).focus();
    await page.keyboard.press("Enter");
    await expect(layer(page)).toBeFocused();
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "true");
  });
});

test.describe("it turns off when the Colour tab is left", () => {
  test("by a click on a tab, by ⌘1 to ⌘7, by Crop, by the Export panel and by another photo", async ({ page }) => {
    await photos(page);
    const start = async (): Promise<void> => {
      if ((await tab(page, "Colour").getAttribute("aria-selected")) !== "true") await tab(page, "Colour").click();
      await toggle(page).click();
      await expect(layer(page)).toBeVisible();
      await expect(pill(page)).toBeVisible();
    };
    const off = async (what: string): Promise<void> => {
      await expect(layer(page), what).toHaveCount(0);
      await expect(pill(page), what).toHaveCount(0);
    };
    // Tabs: by a click, and by the keys.
    for (const name of ["Look", "Lens", "Light", "Detail", "Curve", "RAW"]) {
      await start();
      await tab(page, name).click();
      await off(`${name}, by a click`);
      await tab(page, "Colour").click();
      await expect(toggle(page), `${name}: back on Colour`).toHaveAttribute("aria-pressed", "false");
    }
    for (const [key, name] of [["1", "Look"], ["2", "Lens"], ["3", "Light"], ["5", "Detail"], ["6", "Curve"], ["7", "RAW"]] as const) {
      await start();
      await page.keyboard.press(`Meta+${key}`);
      await expect(tab(page, name)).toHaveAttribute("aria-selected", "true");
      await off(`${name}, by ⌘${key}`);
      await page.keyboard.press("Meta+4");
      await expect(toggle(page), `⌘${key}: back on Colour`).toHaveAttribute("aria-pressed", "false");
    }
    // Crop.
    await start();
    await page.getByRole("button", { name: "Crop" }).click();
    await expect(page.getByRole("complementary", { name: "Crop" })).toBeVisible();
    await off("Crop");
    await page.keyboard.press("Escape");
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "false");
    // The Export panel.
    await start();
    await page.keyboard.press("Meta+e");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    await off("Export");
    await page.keyboard.press("Escape");
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "false");
    // Another photo: the filmstrip.
    await start();
    await page.getByRole("navigation", { name: "Photos" }).getByRole("button", { name: /^Show DSC04206/ }).click();
    await off("another photo");
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "false");
  });

  test("and Pick white and Pick colour are exclusive", async ({ page }) => {
    await photos(page);
    await tab(page, "Colour").click();
    const colour = page.getByRole("region", { name: "Colour warp" }).getByRole("button", { name: "Pick colour" });
    await toggle(page).click();
    await expect(pill(page)).toBeVisible();
    await colour.click();
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "false");
    await expect(colour).toHaveAttribute("aria-pressed", "true");
    await expect(pill(page)).toHaveCount(0);
    await expect(page.getByText("Click a colour in the photo.")).toBeVisible();
    await page.keyboard.press("w");
    await expect(colour).toHaveAttribute("aria-pressed", "false");
    await expect(page.getByText("Click a colour in the photo.")).toHaveCount(0);
    await expect(toggle(page)).toHaveAttribute("aria-pressed", "true");
    await expect(pill(page)).toBeVisible();
  });
});
