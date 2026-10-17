import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, setValue, watchErrors } from "./helpers/app";

// Colour › Colour warp (the owner's request of 2026-10-17, docs/11 §Colour warp, docs/02 C9b) in a real browser on the mocked API: a
// drag on the plane adds a pin and the photo follows it, Pick colour adds a pin from a click on the photo, undo takes it back, the keys
// move a pin, and Copy settings and Paste carry the pins to another photo.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

type Pin = { source: [number, number]; target: [number, number]; locked: boolean };

/** The pins a photo has in the mock (the core's copy of the settings). */
const pinsOf = (page: Page, id: number): Promise<Pin[]> =>
  page.evaluate(`window.__yellowmoon.mockApi.getSettings(${id}).then((rs) => rs.settings.color_warp.pins)`) as Promise<Pin[]>;

async function colourTab(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await openPhotos(page);
  await page.getByRole("tab", { name: "Colour", exact: true }).click();
  await page.getByRole("region", { name: "Colour warp" }).scrollIntoViewIfNeeded();
}

const card = (page: Page) => page.getByRole("region", { name: "Colour warp" });
const plane = (page: Page) => page.getByRole("group", { name: "Colour warp plane" });
const photoSrc = (page: Page) => page.locator(".ym-photo-cell__image").getAttribute("src");

/** A point of the plane in page pixels, from a chroma: |c| = 0.30 at the middle of each edge, +b up. */
async function planePoint(page: Page, a: number, b: number): Promise<{ x: number; y: number }> {
  const box = (await plane(page).boundingBox())!;
  const scale = box.width / 2 / 0.3;
  return { x: box.x + box.width / 2 + a * scale, y: box.y + box.height / 2 - b * scale };
}

test("a drag on the plane adds a pin, the photo follows it, and undo takes it back", async ({ page }) => {
  const problems = watchErrors(page);
  await colourTab(page);
  await expect(card(page).getByText("Drag a colour to move it. Lock holds a colour in place.")).toBeVisible();
  const before = await photoSrc(page);
  // From the photo's blue (the sky of DSC04204 sits near (−0.09, −0.06)) toward green.
  const from = await planePoint(page, -0.09, -0.065);
  const to = await planePoint(page, -0.12, 0.05);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 8 });
  await page.mouse.up();
  await expect(card(page).getByRole("slider", { name: "Pin 1" })).toHaveAttribute("aria-valuetext", /^moves blue toward /);
  await expect.poll(() => pinsOf(page, 4).then((p) => p.length)).toBe(1);
  await expect.poll(() => photoSrc(page)).not.toBe(before);
  // The pin's own sliders, at their defaults.
  for (const [label, value] of [["Range", "4"], ["Shadows", "100%"], ["Highlights", "100%"], ["Pivot", "0.0 EV"], ["Exposure", "0.00 EV"]] as const) {
    await expect(card(page).getByRole("textbox", { name: `${label} value` })).toHaveValue(value);
  }
  // One step back is the whole drag.
  await page.keyboard.press("Meta+z");
  await expect.poll(() => pinsOf(page, 4).then((p) => p.length)).toBe(0);
  await expect.poll(() => photoSrc(page)).toBe(before);
  await expect(card(page).getByRole("slider", { name: "Pin 1" })).toHaveCount(0);
  expect(problems).toEqual([]);
});

test("a pin moves from the keyboard, its sliders change it, and Delete removes it", async ({ page }) => {
  await colourTab(page);
  const at = await planePoint(page, 0.1, 0.05);
  await page.mouse.click(at.x, at.y);
  const pin = card(page).getByRole("slider", { name: "Pin 1" });
  await expect(pin).toBeFocused();
  await expect(pin).toHaveAttribute("aria-valuetext", "orange, not moved");
  await page.keyboard.press("Shift+ArrowUp");
  await page.keyboard.press("Shift+ArrowUp");
  await expect.poll(() => pinsOf(page, 4).then((p) => p[0]?.target[1] ?? 0)).toBeCloseTo(0.09, 4);
  await setValue(page, "Exposure", "0.5");
  await expect(card(page).getByRole("textbox", { name: "Exposure value" })).toHaveValue("+0.50 EV");
  await pin.focus();
  await page.keyboard.press("Delete");
  await expect.poll(() => pinsOf(page, 4).then((p) => p.length)).toBe(0);
  await expect(card(page).getByText("Drag a colour to move it. Lock holds a colour in place.")).toBeVisible();
});

test("Lock adds a locked pin that shows Range only", async ({ page }) => {
  await colourTab(page);
  await card(page).getByRole("radio", { name: "Lock" }).click();
  const at = await planePoint(page, 0.12, -0.1);
  await page.mouse.click(at.x, at.y);
  await expect(card(page).getByRole("slider", { name: "Pin 1" })).toHaveAttribute("aria-valuetext", "holds magenta");
  await expect(card(page).getByRole("textbox", { name: "Range value" })).toBeVisible();
  await expect(card(page).getByRole("textbox", { name: "Exposure value" })).toHaveCount(0);
});

test("Pick colour adds a pin from a click on the photo, stays on for the next, and Esc ends it", async ({ page }) => {
  await colourTab(page);
  const toggle = card(page).getByRole("button", { name: "Pick colour" });
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Click a colour in the photo.")).toBeVisible();
  const layer = page.getByRole("application", { name: /^Pick colour\./ });
  const box = (await layer.boundingBox())!;
  // The sky, then the ground.
  await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.15);
  await expect(card(page).getByRole("slider", { name: "Pin 1" })).toHaveAttribute("aria-valuetext", "blue, not moved");
  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.95);
  await expect(card(page).getByRole("slider", { name: "Pin 2" })).toBeVisible();
  const pins = await pinsOf(page, 4);
  expect(pins).toHaveLength(2);
  expect(pins.every((p) => p.target[0] === p.source[0] && p.target[1] === p.source[1] && !p.locked)).toBe(true);
  // The selected pin is the last one picked: its rows show.
  await expect(card(page).getByRole("textbox", { name: "Range value" })).toBeVisible();
  // Undo takes the last pick back; the mode stays on until Esc.
  await page.keyboard.press("Meta+z");
  await expect.poll(() => pinsOf(page, 4).then((p) => p.length)).toBe(1);
  await expect(layer).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(layer).toHaveCount(0);
  await expect(toggle).toHaveAttribute("aria-pressed", "false");
});

test("Copy settings and Paste carry the pins to another photo", async ({ page }) => {
  await colourTab(page);
  const from = await planePoint(page, 0.05, 0.06);
  const to = await planePoint(page, 0.12, 0.02);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 5 });
  await page.mouse.up();
  await expect.poll(() => pinsOf(page, 4).then((p) => p.length)).toBe(1);
  const copied = await pinsOf(page, 4);
  await page.getByRole("button", { name: "Copy settings" }).click();
  await page.getByRole("navigation", { name: "Photos" }).getByRole("button", { name: /^Show DSC04206/ }).click();
  await expect(page.getByRole("img", { name: /^Photo DSC04206/ })).toBeVisible();
  await expect(card(page).getByRole("slider", { name: "Pin 1" })).toHaveCount(0);
  await page.getByRole("button", { name: "Paste", exact: true }).click();
  await expect(card(page).getByRole("slider", { name: "Pin 1" })).toBeVisible();
  expect(await pinsOf(page, 6)).toEqual(copied);
});
