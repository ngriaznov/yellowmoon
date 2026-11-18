import { expect, test } from "@playwright/test";
import { compareScreen } from "./helpers/compare";
import { uniformView } from "./helpers/states";

// The reference pictures of the Colour Uniform node (`design/screenshots-c2/photos-uniform-*.png`, CU-11), the states of
// `design/colour-uniform/01-node.jpg`, `02-pick-target.jpg`, `03-show-mask.jpg`, `03b-mask-tab.jpg` and `04-add-node-menu.jpg`.
// The owner looks at the pictures before they are committed (docs/15, CU-11); until then this file is not in the repository.

test.describe("Colour Uniform", () => {
  test("the node, picked (photos-uniform-node.png)", async ({ page }) => {
    await uniformView(page);
    await compareScreen(page, "photos-uniform-node");
  });

  test("Pick colour on, no target yet (photos-uniform-pick.png)", async ({ page }) => {
    await uniformView(page, { picked: false, picking: true });
    await expect(page.getByRole("application", { name: /Pick colour for the target/ })).toBeVisible();
    await compareScreen(page, "photos-uniform-pick", { pointer: { x: 520, y: 330 } });
  });

  test("Show mask on (photos-uniform-show-mask.png)", async ({ page }) => {
    await uniformView(page, { show: "Overlay" });
    await compareScreen(page, "photos-uniform-show-mask");
  });

  test("the Mask tab with the Colour range row (photos-uniform-mask-tab.png)", async ({ page }) => {
    await uniformView(page, { tab: "Mask" });
    await compareScreen(page, "photos-uniform-mask-tab");
  });

  test("the Add node menu with Colour Uniform (photos-uniform-add-node.png)", async ({ page }) => {
    await uniformView(page);
    await page.getByRole("button", { name: "Add node" }).click();
    await expect(page.getByRole("menu", { name: "Add node" }).getByRole("menuitem", { name: /^Colour Uniform/ })).toBeVisible();
    await page.waitForTimeout(500);
    await compareScreen(page, "photos-uniform-add-node");
  });
});
