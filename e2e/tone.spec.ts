import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, recordCalls, setValue } from "./helpers/app";

// The Look tab's Tone card (owner's decision of 2026-10-13, docs/16): Temperature, Contrast and
// Saturation are the same settings as Colour › Kelvin, Light › Contrast and Colour › Saturation,
// in a real browser against the mocked API.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

async function photos(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await openPhotos(page);
}

const tabs = (page: Page) => page.getByRole("tablist", { name: "Tool tabs" });
const valueBox = (page: Page, label: string) => page.getByRole("textbox", { name: `${label} value` });
const tone = (page: Page) => page.getByRole("region", { name: "Tone" });
const openTab = (page: Page, name: string) => tabs(page).getByRole("tab", { name }).click();
const whiteBalanceAuto = (page: Page) => page.getByRole("region", { name: "White balance" }).getByRole("button", { name: "Auto" });

/** Drags a slider's thumb to a share of its track with the real mouse. */
async function dragSlider(page: Page, label: string, to: number, steps = 6): Promise<void> {
  const control = page.getByRole("slider", { name: label }).locator("xpath=ancestor::*[@data-slot='slider'][1]");
  const box = (await control.boundingBox())!;
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width * 0.5, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * to, y, { steps });
  await page.mouse.up();
}

test.describe("the Tone card", () => {
  test("sits under the Look card with Temperature, Contrast and Saturation in that order", async ({ page }) => {
    await photos(page);
    await tone(page).scrollIntoViewIfNeeded();
    const look = (await page.getByRole("region", { name: "Look" }).boundingBox())!;
    const card = (await tone(page).boundingBox())!;
    expect(card.y).toBeGreaterThanOrEqual(look.y + look.height - 1);
    const ys: number[] = [];
    for (const label of ["Temperature", "Contrast", "Saturation"]) {
      await expect(tone(page).getByRole("slider", { name: label })).toBeVisible();
      ys.push((await valueBox(page, label).boundingBox())!.y);
    }
    expect(ys).toEqual([...ys].sort((a, b) => a - b));
    await expect(tone(page).getByRole("slider", { name: "Tint" })).toHaveCount(0);
    await expect(valueBox(page, "Temperature")).toHaveValue("5450 K");
    await expect(page.getByRole("button", { name: "Auto temperature" })).toHaveAttribute("aria-pressed", "true");
    // Nothing was taken from the other tabs.
    await openTab(page, "Light");
    await expect(page.getByRole("slider", { name: "Contrast" })).toBeVisible();
    await openTab(page, "Colour");
    for (const label of ["Kelvin", "Tint", "Saturation"]) await expect(page.getByRole("slider", { name: label })).toBeVisible();
  });

  test("Saturation, Contrast and Temperature moved on the Look tab show on the Colour and Light tabs", async ({ page }) => {
    await photos(page);
    await setValue(page, "Saturation", "35");
    await expect(valueBox(page, "Saturation")).toHaveValue("+35");
    await setValue(page, "Contrast", "-20");
    await setValue(page, "Temperature", "6500");
    await expect(valueBox(page, "Temperature")).toHaveValue("6500 K");
    await expect(page.getByRole("button", { name: "Auto temperature" })).toHaveAttribute("aria-pressed", "false");

    await openTab(page, "Colour");
    await expect(valueBox(page, "Saturation")).toHaveValue("+35");
    await expect(valueBox(page, "Kelvin")).toHaveValue("6500 K");
    await expect(valueBox(page, "Tint")).toHaveValue("0");
    await expect(whiteBalanceAuto(page)).toHaveAttribute("aria-pressed", "false");
    await openTab(page, "Light");
    await expect(valueBox(page, "Contrast")).toHaveValue("−20");

    // And back: the Colour and Light tabs move the Look tab's rows.
    await setValue(page, "Contrast", "45");
    await openTab(page, "Colour");
    await setValue(page, "Saturation", "-10");
    await setValue(page, "Kelvin", "4300");
    await openTab(page, "Look");
    await expect(valueBox(page, "Saturation")).toHaveValue("−10");
    await expect(valueBox(page, "Contrast")).toHaveValue("+45");
    await expect(valueBox(page, "Temperature")).toHaveValue("4300 K");
  });

  test("dragging Saturation with the mouse sends patches and the Colour tab shows the last one", async ({ page }) => {
    await photos(page);
    // The card is under the look grid: scroll the panel to it, so the thumb is where the mouse can reach it.
    await page.locator(".ym-tools__scroll").evaluate((el) => {
      el.scrollTop = el.scrollHeight;
    });
    await expect(tone(page)).toBeInViewport({ ratio: 1 });
    const calls = await recordCalls(page, "updateSettings");
    await dragSlider(page, "Saturation", 0.8);
    // Auto or a value (docs/19 section 5): a manual value, or Auto where the drag passes the Auto value (0 here).
    type Percent = { kind: "auto" } | { kind: "manual"; value: number };
    const patches = (await calls()).map((c) => {
      const saturation = (c.args[1] as { saturation: Percent }).saturation;
      return saturation.kind === "manual" ? saturation.value : 0;
    });
    expect(patches.length).toBeGreaterThanOrEqual(2);
    expect(patches.every((p) => Number.isInteger(p) && p >= -100 && p <= 100)).toBe(true);
    const last = patches.at(-1)!;
    expect(last).toBeGreaterThan(40);
    await expect(valueBox(page, "Saturation")).toHaveValue(`+${last}`);
    await openTab(page, "Colour");
    await expect(valueBox(page, "Saturation")).toHaveValue(`+${last}`);
  });

  test("Reset on the Look tab's Temperature returns the Colour tab to Auto", async ({ page }) => {
    await photos(page);
    await setValue(page, "Temperature", "7200");
    await expect(valueBox(page, "Temperature")).toHaveValue("7200 K");
    await openTab(page, "Colour");
    await expect(valueBox(page, "Kelvin")).toHaveValue("7200 K");
    await expect(whiteBalanceAuto(page)).toHaveAttribute("aria-pressed", "false");

    await openTab(page, "Look");
    const calls = await recordCalls(page, "updateSettings");
    await page.getByRole("button", { name: "Reset Temperature" }).click();
    await expect(valueBox(page, "Temperature")).toHaveValue("5450 K");
    expect((await calls()).at(-1)!.args[1]).toEqual({ white_balance: { kind: "auto" }, light_source: "auto", white_pick: null });
    await expect(page.getByRole("button", { name: "Auto temperature" })).toHaveAttribute("aria-pressed", "true");
    await openTab(page, "Colour");
    await expect(valueBox(page, "Kelvin")).toHaveValue("5450 K");
    await expect(whiteBalanceAuto(page)).toHaveAttribute("aria-pressed", "true");
    // The Automatic card on the Look tab says Auto again.
    await openTab(page, "Look");
    await expect(page.getByRole("button", { name: "White balance 5450 K · tint 0, Auto. Open Colour" })).toBeVisible();
  });

  test("undo takes a change made on the Look tab", async ({ page }) => {
    await photos(page);
    await setValue(page, "Saturation", "50");
    await expect(valueBox(page, "Saturation")).toHaveValue("+50");
    await page.keyboard.press("Meta+z");
    await expect(valueBox(page, "Saturation")).toHaveValue("0");
    await openTab(page, "Colour");
    await expect(valueBox(page, "Saturation")).toHaveValue("0");
  });
});
