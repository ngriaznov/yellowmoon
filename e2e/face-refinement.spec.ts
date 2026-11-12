import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { ResolvedSettings } from "../src/api/types";
import { nodeOf } from "../src/state/settingsGraph";
import type { FaceRefinementItem } from "../src/state/settingsGraph";
import { configureMock, openApp, openPhotos, setValue, showUpdateReady } from "./helpers/app";

// The Face Refinement node's interface (docs/27 S-11c, design/skin/), in a real browser on the mock API: Add node › Face Refinement, the faces found, a
// slider, Same for all faces, the choice of a face from the list and from the photo, and Protect. The screenshots of the panel in each state go to the folder
// named by YM_FACE_SHOTS when it is set (they are not committed).

const SHOTS = process.env.YM_FACE_SHOTS ?? "";
test.use({ deviceScaleFactor: 1 });

type Mock = { getSettings(id: number): Promise<ResolvedSettings> };
const mock = <T>(page: Page, run: (api: Mock, ...args: never[]) => T | Promise<T>): Promise<T> =>
  page.evaluate(`(${run.toString()})(window.__yellowmoon.mockApi)`) as Promise<T>;
const faceNode = async (page: Page): Promise<FaceRefinementItem> => nodeOf((await mock(page, (api) => api.getSettings(4))).settings, 2) as FaceRefinementItem;
const panel = (page: Page) => page.getByRole("complementary", { name: "Tools" });

const picture = (from: number | "input", to: number | "output") => ({ from: { node: from, port: "picture" as const }, to: { node: to, port: "picture" as const } });
const WITH_NODE = {
  nodes: { "2": { kind: "face_refinement", name: "Face Refinement", pos: [1, 0] } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
};

async function shot(page: Page, name: string): Promise<void> {
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png` });
}

/** The node view of the mockup's photo, with the mock finding `faces` faces, and the Face Refinement node (2) selected. */
async function faceView(page: Page, faces: "one" | "five" | "twelve" | "none" | "fail" | "missing" = "five", settings: Record<string, unknown> = {}): Promise<void> {
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4", mockFaces: faces });
  await configureMock(page, { initialSettings: { DSC04204: { nodes: { "2": { ...WITH_NODE.nodes["2"], ...settings } }, links: WITH_NODE.links } } as never });
  await openPhotos(page);
  await showUpdateReady(page);
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await page.locator('[data-node-card="2"]').click();
  await expect(page.locator('[data-node-card="2"]')).toHaveAttribute("aria-pressed", "true");
}

test.describe("Add node › Face Refinement", () => {
  test("adds the node, finds the faces, and the Face tab shows them", async ({ page }) => {
    await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4", mockFaces: "five" });
    await openPhotos(page);
    await showUpdateReady(page);
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await configureMock(page, { speed: 0.1 });
    await page.getByRole("button", { name: "Add node" }).click();
    await shot(page, "01-add-node-menu");
    await page.getByRole("menuitem", { name: /^Face Refinement/ }).click();
    await expect(panel(page).getByRole("heading", { level: 2, name: "Face Refinement" })).toBeVisible();
    await expect(panel(page).locator(".ym-nodehead__line")).toHaveText("Finding faces");
    await expect(panel(page).getByText("Finding faces…")).toBeVisible();
    await shot(page, "02-finding");
    // The mock finds them after 4 s at this speed.
    await expect(panel(page).getByText("5 found")).toBeVisible({ timeout: 20_000 });
    await expect(panel(page).locator(".ym-nodehead__line")).toHaveText("5 faces");
    await expect(panel(page).getByRole("tab")).toHaveText(["Face", "Mask"]);
    await expect(page.locator("[data-face-number]")).toHaveCount(5);
  });
});

test.describe("the Face tab", () => {
  test("moves a slider, switches Same for all faces and sets one face on its own", async ({ page }) => {
    await faceView(page);
    await expect(panel(page).getByText("5 found")).toBeVisible();
    await page.waitForTimeout(400);
    await shot(page, "03-five-faces-same-for-all");
    await setValue(page, "Blemish", "55");
    await expect.poll(async () => (await faceNode(page)).all.blemish).toBe(55);
    await panel(page).getByRole("switch", { name: "Same for all faces" }).click();
    await expect.poll(async () => (await faceNode(page)).faces.length).toBe(5);
    expect((await faceNode(page)).faces.every((f) => f.blemish === 55)).toBe(true);
    await panel(page).getByRole("button", { name: "Face 2", exact: true }).click();
    await expect(panel(page).getByRole("heading", { level: 3, name: "Face 2" })).toBeVisible();
    await setValue(page, "Blemish", "45");
    await expect.poll(async () => (await faceNode(page)).faces[1]?.blemish).toBe(45);
    expect((await faceNode(page)).faces[0]?.blemish).toBe(55);
    await page.waitForTimeout(400);
    await shot(page, "04-five-faces-per-face");
    await expect(page.locator('[data-face-number="2"]')).toHaveAttribute("data-selected", "true");
    await expect(page.locator('[data-face-number="3"]')).not.toHaveAttribute("data-selected", "true");
    await page.locator('[data-face-number="4"] [data-face-hit]').click({ force: true });
    await expect(panel(page).getByRole("heading", { level: 3, name: "Face 4" })).toBeVisible();
    await panel(page).getByRole("button", { name: "Copy to other faces" }).click();
    await expect.poll(async () => (await faceNode(page)).faces.map((f) => f.blemish)).toEqual([55, 55, 55, 55, 55]);
  });

  test("has one face with no list switch, twelve with a scrolling list", async ({ page }) => {
    await faceView(page, "one");
    await expect(panel(page).getByText("1 found")).toBeVisible();
    await expect(panel(page).getByRole("switch", { name: "Same for all faces" })).toHaveCount(0);
    await page.waitForTimeout(300);
    await shot(page, "05-one-face");
    await faceView(page, "twelve");
    await expect(panel(page).getByText("12 found")).toBeVisible();
    await expect(panel(page).locator(".ym-faces")).toHaveAttribute("data-scroll", "true");
    await page.waitForTimeout(300);
    await shot(page, "06-twelve-faces");
  });

  test("Protect paints on the photo and a stroke is one patch", async ({ page }) => {
    await faceView(page, "one");
    await expect(panel(page).getByText("1 found")).toBeVisible();
    await panel(page).getByRole("radio", { name: "Paint" }).click();
    const layer = page.locator("[data-brush-layer]");
    await expect(layer).toBeVisible();
    await expect(panel(page).getByText("Paint on the photo. Hold ⌥ to erase.")).toBeVisible();
    const box = (await layer.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 40, box.y + box.height / 2 + 20, { steps: 4 });
    await page.mouse.up();
    await expect.poll(async () => (await faceNode(page)).protect.strokes.length).toBe(1);
    await page.waitForTimeout(700);
    await shot(page, "07-protect");
    await panel(page).getByRole("button", { name: "Clear" }).click();
    await expect.poll(async () => (await faceNode(page)).protect.strokes.length).toBe(0);
  });
});

test.describe("the states", () => {
  test("none found, failed and models missing say so, with nothing to set", async ({ page }) => {
    await faceView(page, "none");
    await expect(panel(page).getByText("No faces found in this photo.")).toBeVisible();
    await expect(panel(page).locator(".ym-nodehead__line")).toHaveText("No faces found");
    await expect(panel(page).getByRole("slider")).toHaveCount(0);
    await page.waitForTimeout(300);
    await shot(page, "08-no-faces");
    await faceView(page, "fail");
    await expect(panel(page).getByText("Faces can't be found: the model stopped before it finished.")).toBeVisible();
    await shot(page, "09-failed");
    await faceView(page, "missing");
    await expect(panel(page).getByText(/needs the face models/)).toBeVisible();
    await shot(page, "10-models-missing");
  });

  test("the simple view shows its note and nothing of the node", async ({ page }) => {
    await faceView(page, "five");
    await expect(panel(page).getByText("5 found")).toBeVisible();
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await expect(panel(page).getByRole("tab", { name: "Face" })).toHaveCount(0);
    await expect(panel(page).getByText(/These controls change/)).toBeVisible();
    await shot(page, "11-simple-view");
  });
});
