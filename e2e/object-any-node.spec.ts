import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos } from "./helpers/app";

// An Object mask on a node that is not a corrector (0.2.4-beta.4: "That mask is not an Object mask" on a Colour Uniform node). Every node that has masks
// can hold an Object row, so a click on the photo finds the object on any of them (`docs/23` §9, `mask_find`). Also: Pick colour of a Colour Uniform node
// (on when the node is added) and the Object tool are never on together.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

const pickButton = (page: Page) => page.getByRole("complementary", { name: "Tools" }).getByRole("region", { name: "Target" }).getByRole("button", { name: "Pick colour" });
const pickLayer = (page: Page) => page.getByRole("application", { name: /Pick colour for the target of Colour Uniform/ });
const objectLayer = (page: Page) => page.locator("[data-object-layer]");

/** The node view of a photo with a new Colour Uniform node (2), selected: Add node › Colour Uniform, which starts Pick colour. */
async function addUniform(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await openPhotos(page);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await page.getByRole("button", { name: "Add node" }).click();
  await page.getByRole("menuitem", { name: /^Colour Uniform/ }).click();
  await expect(page.getByRole("button", { name: /^Node 2, Uniform/ })).toHaveAttribute("aria-pressed", "true");
  await expect(pickButton(page)).toHaveAttribute("aria-pressed", "true");
}

async function addObjectMask(page: Page): Promise<void> {
  await page.getByRole("tab", { name: "Mask", exact: true }).click();
  await page.getByRole("button", { name: "Add mask" }).click();
  await page.getByRole("menuitem", { name: /^Object/ }).click();
}

test.describe("an Object mask on a Colour Uniform node", () => {
  test("a click on the photo finds the object", async ({ page }) => {
    await addUniform(page);
    await addObjectMask(page);
    await expect(objectLayer(page)).toHaveAttribute("data-tool", /.*/);
    // The Pick colour the node was added with is off: the Object tool has the click.
    await expect(pickLayer(page)).toHaveCount(0);
    const cell = (await page.locator("[data-viewport]").boundingBox())!;
    await page.mouse.click(cell.x + cell.width * 0.5, cell.y + cell.height * 0.7);
    await expect(page.getByText("Object found · 0.4 s")).toBeVisible();
    await expect(page.getByRole("alert")).toHaveCount(0);
    await expect(page.locator("[data-object-point]")).toHaveCount(1);
  });

  test("Pick colour and the Object tool never both take the click: the pick's layer has it while the pick is on, and the tool is back after", async ({ page }) => {
    await addUniform(page);
    await addObjectMask(page);
    await expect(objectLayer(page)).toHaveAttribute("data-tool", /.*/);
    // Back on the Compress tab, Pick colour on: its layer takes the click and the Object layer is gone.
    await page.getByRole("tab", { name: "Compress", exact: true }).click();
    await pickButton(page).click();
    await expect(pickButton(page)).toHaveAttribute("data-state", "on");
    await expect(pickLayer(page)).toBeVisible();
    await expect(objectLayer(page)).toHaveCount(0);
    await page.keyboard.press("Escape");
    await expect(pickLayer(page)).toHaveCount(0);
    // On the Mask tab the pick is off and the Object tool has the click again.
    await page.getByRole("tab", { name: "Mask", exact: true }).click();
    await expect(pickLayer(page)).toHaveCount(0);
    await expect(objectLayer(page)).toHaveAttribute("data-tool", /.*/);
  });
});
