import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos } from "./helpers/app";

// A look picked on one photo belongs to that photo (the owner's report: choosing a look sometimes changed the look of another opened
// photo). The core's answer to the pick is held back by hand, so the person moves to the next photo before it comes, which is the
// order a slow machine produces by itself.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

async function photos(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await openPhotos(page);
}

/** Holds back the core's answer to `update_settings` until the returned function is called. */
async function holdUpdates(page: Page): Promise<() => Promise<void>> {
  await page.evaluate(() => {
    const api = window.__yellowmoon!.mockApi;
    const real = api.updateSettings.bind(api);
    let open: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      open = resolve;
    });
    (window as unknown as { __release: () => void }).__release = open;
    api.updateSettings = async (id, patch) => {
      await gate;
      return real(id, patch);
    };
  });
  return () => page.evaluate(() => (window as unknown as { __release: () => void }).__release());
}

const chip = (page: Page) => page.locator(".ym-photo-cell__chip");
const tile = (page: Page, name: string) => page.getByRole("button", { name: `${name} look`, exact: false });

test.describe("a look picked while the person moves to the next photo", () => {
  test("stays on the photo it was picked on when Apply to is This photo", async ({ page }) => {
    await photos(page);
    // Apply to starts on This photo (the owner's decision of 2026-10-20).
    await expect(page.getByRole("radio", { name: "This photo" })).toHaveAttribute("aria-checked", "true");
    const release = await holdUpdates(page);
    await tile(page, "Film warm").click();
    await expect(chip(page)).toHaveText("Film warm · 100%");
    await page.keyboard.press("ArrowRight");
    await expect(chip(page)).toHaveText("Neutral · 100%");
    await release();
    // Give the held answer time to arrive; the photo on screen keeps its own look.
    await page.waitForTimeout(150);
    await expect(chip(page)).toHaveText("Neutral · 100%");
    await expect(tile(page, "Neutral")).toHaveAttribute("aria-pressed", "true");
    await expect(tile(page, "Film warm")).toHaveAttribute("aria-pressed", "false");
    await page.keyboard.press("ArrowLeft");
    await expect(chip(page)).toHaveText("Film warm · 100%");
    await expect(tile(page, "Film warm")).toHaveAttribute("aria-pressed", "true");
  });

  test("shows the look the photo has been given when Apply to is All", async ({ page }) => {
    await photos(page);
    await page.getByRole("radio", { name: /^All/ }).click();
    const release = await holdUpdates(page);
    await tile(page, "Film warm").click();
    await page.keyboard.press("ArrowRight");
    // Not copied yet: the photo on screen still has the look it had.
    await expect(chip(page)).toHaveText("Neutral · 100%");
    await release();
    // The copy lands in the core; what the tab shows for the photo follows it, not what it knew before.
    await expect(chip(page)).toHaveText("Film warm · 100%");
    await expect(tile(page, "Film warm")).toHaveAttribute("aria-pressed", "true");
    await expect(tile(page, "Neutral")).toHaveAttribute("aria-pressed", "false");
  });
});
