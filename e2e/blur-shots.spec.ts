import { expect, test } from "@playwright/test";
import { compareScreen } from "./helpers/compare";
import { blurView } from "./helpers/states";

// The reference pictures of the Blur node (`design/screenshots-c2/photos-blur-*.png`, BL-15), the states of `design/blur/states.jpg`: the Blur
// tab of each type, the Zoom centre handle on the photo, Zoom with Pick centre on, and the Add node menu with Blur.
// The owner looks at the pictures before they are committed (docs/15, BL-15); until then the pictures and this file are not in the repository.

test.describe("Blur", () => {
  test("Soft (photos-blur-soft.png)", async ({ page }) => {
    await blurView(page);
    await compareScreen(page, "photos-blur-soft");
  });

  test("Lens, six blades, with a mask (photos-blur-lens.png)", async ({ page }) => {
    await blurView(page, { type: "lens", masked: true, more: { lens: { radius: 30, blades: 6, rotation_deg: 15, highlights: 40 } } });
    await compareScreen(page, "photos-blur-lens");
  });

  test("Motion (photos-blur-motion.png)", async ({ page }) => {
    await blurView(page, { type: "motion", more: { motion: { length: 30, angle_deg: 20 } } });
    await compareScreen(page, "photos-blur-motion");
  });

  test("Zoom with its centre handle on the photo (photos-blur-zoom.png)", async ({ page }) => {
    await blurView(page, { type: "zoom", more: { zoom: { amount: 40, center: [0.35, 0.4] } } });
    await expect(page.getByRole("button", { name: /^Zoom centre/ })).toBeVisible();
    await compareScreen(page, "photos-blur-zoom");
  });

  test("Zoom with Pick centre on (photos-blur-zoom-pick.png)", async ({ page }) => {
    await blurView(page, { type: "zoom", picking: true });
    await expect(page.locator("[data-zoom-catch]")).toBeVisible();
    await compareScreen(page, "photos-blur-zoom-pick", { pointer: { x: 520, y: 330 } });
  });

  test("the Add node menu with Blur (photos-blur-add-node.png)", async ({ page }) => {
    await blurView(page);
    await page.getByRole("button", { name: "Add node" }).click();
    await expect(page.getByRole("menu", { name: "Add node" }).getByRole("menuitem", { name: /^Blur/ })).toBeVisible();
    await page.waitForTimeout(500);
    await compareScreen(page, "photos-blur-add-node");
  });
});
