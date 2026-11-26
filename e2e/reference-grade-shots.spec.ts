import { expect, test } from "@playwright/test";
import { compareScreen } from "./helpers/compare";
import { referenceView } from "./helpers/states";

// The reference pictures of the Reference Grade node (`design/screenshots-c2/photos-reference-grade-*.png`, RG-12), the states of
// `design/reference-grade/`: the Reference tab with no reference, with a file chosen, fitting, fitted, after the look changed, after a hand edit and
// with an error, the strip as a chooser, the Hue tab, the Elements tab and the Add node menu with Reference Grade. The owner looks at the pictures
// before they are committed (docs/15, RG-12); until then the pictures and this file are not in the repository.

test.describe("Reference Grade", () => {
  test("no reference (photos-reference-grade-empty.png)", async ({ page }) => {
    await referenceView(page);
    await compareScreen(page, "photos-reference-grade-empty");
  });

  test("a file chosen (photos-reference-grade-chosen.png)", async ({ page }) => {
    await referenceView(page, { state: "chosen" });
    await compareScreen(page, "photos-reference-grade-chosen");
  });

  test("fitting (photos-reference-grade-fitting.png)", async ({ page }) => {
    await referenceView(page, { state: "fitting" });
    await compareScreen(page, "photos-reference-grade-fitting");
  });

  test("fitted (photos-reference-grade-fitted.png)", async ({ page }) => {
    await referenceView(page, { state: "fitted" });
    await compareScreen(page, "photos-reference-grade-fitted");
  });

  test("the look changed since the fit (photos-reference-grade-look-changed.png)", async ({ page }) => {
    await referenceView(page, { state: "look-changed" });
    await compareScreen(page, "photos-reference-grade-look-changed");
  });

  test("edited by hand since the fit (photos-reference-grade-edited.png)", async ({ page }) => {
    await referenceView(page, { state: "edited" });
    await compareScreen(page, "photos-reference-grade-edited");
  });

  test("an error (photos-reference-grade-error.png)", async ({ page }) => {
    await referenceView(page, { state: "error" });
    await compareScreen(page, "photos-reference-grade-error");
  });

  test("the strip as a chooser (photos-reference-grade-chooser.png)", async ({ page }) => {
    await referenceView(page, { state: "chooser" });
    await compareScreen(page, "photos-reference-grade-chooser");
  });

  test("the Hue tab (photos-reference-grade-hue.png)", async ({ page }) => {
    await referenceView(page, { state: "fitted", tab: "Hue" });
    await compareScreen(page, "photos-reference-grade-hue");
  });

  test("the Elements tab (photos-reference-grade-elements.png)", async ({ page }) => {
    await referenceView(page, { state: "fitted", tab: "Elements" });
    await compareScreen(page, "photos-reference-grade-elements");
  });

  test("the Add node menu with Reference Grade (photos-reference-grade-add-node.png)", async ({ page }) => {
    await referenceView(page);
    await page.getByRole("button", { name: "Add node" }).click();
    await expect(page.getByRole("menu", { name: "Add node" }).getByRole("menuitem", { name: /^Reference Grade/ })).toBeVisible();
    await page.waitForTimeout(500);
    await compareScreen(page, "photos-reference-grade-add-node");
  });
});
