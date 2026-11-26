import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { settle } from "./helpers/app";
import { compareScreen } from "./helpers/compare";
import { referenceView } from "./helpers/states";

// The reference pictures of the Reference Grade node (`design/screenshots-c2/photos-reference-grade-*.png`, RG-12), the states of
// `design/reference-grade/`: the Reference tab with no reference, with a file chosen, fitting, fitted, after the look changed, after a hand edit and
// with an error, the strip as a chooser, the Hue tab, the Elements tab and the Add node menu with Reference Grade. The owner looks at the pictures
// before they are committed (docs/15, RG-12); until then the pictures and this file are not in the repository.

/**
 * Where a picture goes that has no reference yet: the owner looks at it before it is committed (`design/screenshots-c2/` is not written by a run).
 * `YM_NEW_SHOTS` names the folder; the default is inside the ignored `node_modules`.
 */
const NEW_SHOTS = process.env.YM_NEW_SHOTS ?? resolve(import.meta.dirname, "../node_modules/.cache/yellowmoon-e2e/new-shots");

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

  test("the eye of the Sky card on (photos-reference-grade-elements-eye.png)", async ({ page }) => {
    await referenceView(page, { state: "fitted", tab: "Elements" });
    const tools = page.getByRole("complementary", { name: "Tools" });
    await tools.getByRole("button", { name: "Show mask, Sky" }).click();
    await expect(tools.getByRole("button", { name: "Hide mask, Sky" })).toHaveAttribute("aria-pressed", "true");
    // The view is the store's: it reached the viewport as the node's element 0, and no setting moved.
    await expect(page.locator("[data-mask-view], img[alt]").first()).toBeVisible();
    await settle(page);
    await page.mouse.move(1000, 400);
    await page.waitForTimeout(100);
    mkdirSync(NEW_SHOTS, { recursive: true });
    writeFileSync(resolve(NEW_SHOTS, "photos-reference-grade-elements-eye.png"), await page.screenshot({ animations: "disabled" }));
    // A second click ends it.
    await tools.getByRole("button", { name: "Hide mask, Sky" }).click();
    await expect(tools.getByRole("button", { name: "Show mask, Sky" })).toHaveAttribute("aria-pressed", "false");
  });

  test("the Add node menu with Reference Grade (photos-reference-grade-add-node.png)", async ({ page }) => {
    await referenceView(page);
    await page.getByRole("button", { name: "Add node" }).click();
    await expect(page.getByRole("menu", { name: "Add node" }).getByRole("menuitem", { name: /^Reference Grade/ })).toBeVisible();
    await page.waitForTimeout(500);
    await compareScreen(page, "photos-reference-grade-add-node");
  });
});
