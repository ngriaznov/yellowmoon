import { test } from "@playwright/test";
import { compareScreen } from "./helpers/compare";
import { uniformView } from "./helpers/states";

// The reference picture of the Colour Uniform node's Light tab (`design/screenshots-c2/photos-uniform-graded.png`, CG-06). The owner looks at the
// picture before it is committed (docs/15, CG-06); until then this file and the picture are not in the repository.

test.describe("Colour Uniform, graded", () => {
  test("the Light tab of a graded node (photos-uniform-graded.png)", async ({ page }) => {
    await uniformView(page, { tab: "Light", graded: true });
    await compareScreen(page, "photos-uniform-graded");
  });
});
