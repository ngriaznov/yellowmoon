import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, recordCalls, setValue } from "./helpers/app";

// Contrast is Auto or Manual, like Exposure (`docs/18` §7, `docs/11`), in a real browser against the mocked API. The mockup's current
// photo (photo 4) resolves Auto contrast to +1.5 (`estimatesFor` in `src/api.mock.ts`).

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

async function photos(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await openPhotos(page);
}

const tabs = (page: Page) => page.getByRole("tablist", { name: "Tool tabs" });
const valueBox = (page: Page, label: string) => page.getByRole("textbox", { name: `${label} value` });
const exposure = (page: Page) => page.getByRole("region", { name: "Exposure" });
const automatic = (page: Page) => page.getByRole("region", { name: "Automatic" });
const tone = (page: Page) => page.getByRole("region", { name: "Tone" });

/** Drags a slider's thumb to a share of its track with the real mouse (as `tone.spec.ts` does). */
async function dragSlider(page: Page, label: string, to: number, steps = 6): Promise<void> {
  const control = page.getByRole("slider", { name: label }).locator("xpath=ancestor::*[@data-slot='slider'][1]");
  const box = (await control.boundingBox())!;
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width * 0.5, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * to, y, { steps });
  await page.mouse.up();
}

test.describe("automatic contrast", () => {
  test("on the Light tab Contrast is AUTO at the photo's own value; a typed value makes it EDITED, and Reset returns it to AUTO", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "Light" }).click();
    await expect(valueBox(page, "Contrast")).toHaveValue("+1.5");
    await expect(exposure(page).getByRole("button", { name: "Auto contrast" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "Reset Contrast" })).toHaveCount(0);

    await setValue(page, "Contrast", "25");
    await expect(valueBox(page, "Contrast")).toHaveValue("+25.0");
    await expect(valueBox(page, "Contrast")).toHaveAttribute("data-changed", "true");
    await expect(exposure(page).getByRole("button", { name: "Auto contrast" })).toHaveCount(0);

    await page.getByRole("button", { name: "Reset Contrast" }).click();
    await expect(valueBox(page, "Contrast")).toHaveValue("+1.5");
    await expect(exposure(page).getByRole("button", { name: "Auto contrast" })).toHaveAttribute("aria-pressed", "true");
  });

  test("a drag with the mouse makes Contrast Manual, sends patches, and the box shows the last one", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "Light" }).click();
    const calls = await recordCalls(page, "updateSettings");
    await dragSlider(page, "Contrast", 0.85);
    const patches = (await calls()).map((c) => c.args[1] as { contrast: { kind: "auto" } | { kind: "manual"; value: number } });
    expect(patches.length).toBeGreaterThanOrEqual(2);
    expect(patches.every((p) => p.contrast.kind === "manual" && p.contrast.value >= -100 && p.contrast.value <= 100)).toBe(true);
    const last = patches.at(-1)!.contrast;
    expect(last.kind).toBe("manual");
    if (last.kind !== "manual") throw new Error("a drag ends on a manual value");
    const text = last.value > 0 ? `+${last.value}.0` : last.value < 0 ? `−${-last.value}.0` : "0.0";
    await expect(valueBox(page, "Contrast")).toHaveValue(text);
    await expect(page.getByRole("button", { name: "Reset Contrast" })).toBeVisible();
    await expect(exposure(page).getByRole("button", { name: "Auto contrast" })).toHaveCount(0);
  });

  test("the keyboard moves it to EDITED, and Reset returns it to AUTO", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "Light" }).click();
    await page.getByRole("slider", { name: "Contrast" }).focus();
    await page.keyboard.press("End");
    await expect(valueBox(page, "Contrast")).toHaveValue("+100.0");
    await expect(page.getByRole("button", { name: "Reset Contrast" })).toBeVisible();
    await page.getByRole("button", { name: "Reset Contrast" }).click();
    await expect(valueBox(page, "Contrast")).toHaveValue("+1.5");
    await expect(exposure(page).getByRole("button", { name: "Auto contrast" })).toHaveAttribute("aria-pressed", "true");
  });

  test("the Exposure card's Reset puts Contrast back on AUTO with Exposure, and Brightness to 0", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "Light" }).click();
    await setValue(page, "Contrast", "40");
    await setValue(page, "Brightness", "7");
    await exposure(page).getByRole("button", { name: "Reset", exact: true }).click();
    await expect(valueBox(page, "Contrast")).toHaveValue("+1.5");
    await expect(valueBox(page, "Brightness")).toHaveValue("0");
    await expect(exposure(page).getByRole("button", { name: "Auto contrast" })).toHaveAttribute("aria-pressed", "true");
  });

  test("the Automatic card's Contrast row shows AUTO, then EDITED with the value, and its arrow opens Light", async ({ page }) => {
    await photos(page);
    await expect(automatic(page).getByRole("button", { name: "Contrast +1.5, Auto. Open Light" })).toBeVisible();
    await tabs(page).getByRole("tab", { name: "Light" }).click();
    await setValue(page, "Contrast", "-12");
    await tabs(page).getByRole("tab", { name: "Look" }).click();
    await expect(automatic(page).getByRole("button", { name: "Contrast −12.0, Edited. Open Light" })).toBeVisible();
    await automatic(page).getByRole("button", { name: "Contrast −12.0, Edited. Open Light" }).click();
    await expect(tabs(page).getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true");
    await page.getByRole("button", { name: "Reset Contrast" }).click();
    await tabs(page).getByRole("tab", { name: "Look" }).click();
    await expect(automatic(page).getByRole("button", { name: "Contrast +1.5, Auto. Open Light" })).toBeVisible();
  });

  test("the Tone card's AUTO covers Temperature and Contrast: pressed only while both are Auto, and one press sends both", async ({ page }) => {
    await photos(page);
    const calls = await recordCalls(page, "updateSettings");
    const combined = tone(page).getByRole("button", { name: "Auto temperature and contrast" });
    await expect(combined).toHaveAttribute("aria-pressed", "true");
    await setValue(page, "Contrast", "10");
    await expect(combined).toHaveAttribute("aria-pressed", "false");

    await calls(true);
    await combined.click();
    await expect(combined).toHaveAttribute("aria-pressed", "true");
    const sent = await calls(true);
    expect(sent).toHaveLength(1);
    expect(sent[0]!.args[1]).toEqual({ white_balance: { kind: "auto" }, light_source: "auto", white_pick: null, contrast: { kind: "auto" } });
    await expect(valueBox(page, "Contrast")).toHaveValue("+1.5");
  });
});
