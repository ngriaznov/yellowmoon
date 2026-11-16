import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openState } from "./helpers/states";

// The tree of the photo cell (`docs/28` A-03, finding A-P3 and A-P8). The picture is `role="img"`, and an `img` has no accessible children,
// so the chips, the camera line and the readout stand in a group beside it, and the crop frame says what the arrow keys do.
// `a11y.spec.ts` keeps its own checks of names and roles.

const cell = (page: Page) => page.getByRole("main", { name: "Photo" });

test.describe("the photo cell's accessibility tree", () => {
  test("a photo is an image with a label and no children, and its texts are a group beside it", async ({ browser }) => {
    const { page, context } = await openState(browser, "10-tab-look");
    try {
      await expect(cell(page)).toMatchAriaSnapshot(`
        - main "Photo":
          - img "Photo DSC04204, Neutral · 100%"
          - group "Photo information": Neutral · 100% Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8
      `);
      // Nothing is inside the image, in the tree or in the page.
      await expect(page.locator(".ym-photo-cell__view")).toBeEmpty();
    } finally {
      await context.close();
    }
  });

  test("Before shows its label in the group, in place of the look", async ({ browser }) => {
    const { page, context } = await openState(browser, "18-before-on");
    try {
      await expect(cell(page)).toMatchAriaSnapshot(`
        - main "Photo":
          - img "Photo DSC04204, Neutral · 100%"
          - group "Photo information": Before · camera default Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8
      `);
    } finally {
      await context.close();
    }
  });

  test("the crop frame is a group with its arrow-key description, and its handles are buttons", async ({ browser }) => {
    const { page, context } = await openState(browser, "16-mode-crop");
    try {
      await expect(cell(page)).toMatchAriaSnapshot(`
        - group "Crop frame":
          - button /Crop top left corner/
          - button /Crop top edge/
          - button /Crop top right corner/
          - button /Crop right edge/
          - button /Crop bottom right corner/
          - button /Crop bottom edge/
          - button /Crop bottom left corner/
          - button /Crop left edge/
      `);
      await expect(page.getByRole("group", { name: "Crop frame" })).toHaveAccessibleDescription("Arrow keys move the frame. Hold Shift to move it faster.");
    } finally {
      await context.close();
    }
  });
});

