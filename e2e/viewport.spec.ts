import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, settle, showUpdateReady, watchErrors } from "./helpers/app";
import { pixelDiff } from "./helpers/pixelDiff";

// The photo cell under each viewport backend (M3-04, M0-06 frontend side) in a real browser against the mocked API. The mock plays the
// app's two backends: `?mockViewport=native` (a transparent cell, the picture is behind the webview) and `?mockViewport=fallback` (the
// shell sends a picture of the cell as `viewport_frame` events). Without it the page draws the mock's fixture, as a browser build does.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

type Viewport = { rect: { x: number; y: number; width: number; height: number; scale: number } | null; inputs: Array<Record<string, unknown>>; mode: { kind: string } };

/** What the mock has recorded of the viewport commands. */
const viewportOf = (page: Page): Promise<Viewport> =>
  page.evaluate(() => JSON.parse(JSON.stringify((window as unknown as { __yellowmoon: { mockApi: { viewport: unknown } } }).__yellowmoon.mockApi.viewport)) as Viewport);

async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await openPhotos(page);
}

const cell = (page: Page) => page.locator("[data-viewport]");

test.describe("the native viewport", () => {
  test("leaves the cell transparent with nothing drawn in it, and tells the core where the cell is", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page, { mockViewport: "native" });
    await expect(cell(page)).toHaveAttribute("data-native", "true");
    await expect(cell(page)).toHaveCSS("background-color", "rgba(0, 0, 0, 0)");
    await expect(cell(page).locator("img")).toHaveCount(0);
    // The chip and the camera line are the webview's own and stay.
    await expect(cell(page).locator(".ym-photo-cell__chip")).toHaveText("Neutral · 100%");
    const box = (await cell(page).boundingBox())!;
    await expect.poll(async () => (await viewportOf(page)).rect).toMatchObject({ scale: 2 });
    const rect = (await viewportOf(page)).rect!;
    expect(rect.x).toBeCloseTo(box.x, 1);
    expect(rect.y).toBeCloseTo(box.y, 1);
    expect(rect.width).toBeCloseTo(box.width, 1);
    expect(rect.height).toBeCloseTo(box.height, 1);
    expect(problems).toEqual([]);
  });

  test("follows the cell when the window is resized", async ({ page }) => {
    await photos(page, { mockViewport: "native" });
    await expect.poll(async () => (await viewportOf(page)).rect?.width).toBeCloseTo((await cell(page).boundingBox())!.width, 1);
    await page.setViewportSize({ width: 1800, height: 1100 });
    await expect.poll(async () => (await viewportOf(page)).rect?.width).toBeCloseTo((await cell(page).boundingBox())!.width, 1);
    const wide = (await viewportOf(page)).rect!;
    expect(wide.width / wide.height).toBeCloseTo(1.5, 2);
    await page.setViewportSize({ width: 1180, height: 760 });
    await expect.poll(async () => (await viewportOf(page)).rect?.width).toBeCloseTo((await cell(page).boundingBox())!.width, 1);
    expect((await viewportOf(page)).rect!.width).toBeLessThan(wide.width);
  });
});

test.describe("the fallback viewport", () => {
  test("shows the shell's picture of the cell, one picture pixel to one device pixel", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page, { mockViewport: "fallback" });
    const frame = cell(page).locator("img.ym-photo-cell__frame");
    await expect(frame).toHaveCount(1);
    await expect(cell(page)).not.toHaveAttribute("data-native", "true");
    await settle(page);
    const box = (await cell(page).boundingBox())!;
    const shown = (await frame.boundingBox())!;
    expect(shown.width).toBeCloseTo(box.width, 0);
    expect(shown.height).toBeCloseTo(box.height, 0);
    const natural = await frame.evaluate((img: HTMLImageElement) => ({ w: img.naturalWidth, h: img.naturalHeight }));
    expect(natural.w).toBe(Math.round(box.width * 2));
    expect(natural.h).toBe(Math.round(box.height * 2));
    expect(problems).toEqual([]);
  });

  test("draws the same picture as the browser build does (the mock's frame and its fixture agree)", async ({ page, browser }) => {
    await photos(page);
    await settle(page);
    const fixture = await cell(page).screenshot({ animations: "disabled" });
    await photos(page, { mockViewport: "fallback" });
    await expect(cell(page).locator("img.ym-photo-cell__frame")).toHaveCount(1);
    await settle(page);
    const frame = await cell(page).screenshot({ animations: "disabled" });
    const diff = await pixelDiff(browser, fixture, frame, { threshold: 24 });
    console.log(`[viewport] fixture against frame: ${(diff.mismatch * 100).toFixed(3)} % of pixels differ by more than 24`);
    expect(diff.mismatch).toBeLessThan(0.005);
  });

  test("sends a new picture for a new photo, for a change of the look, and for a new size of the cell", async ({ page }) => {
    await photos(page, { mockViewport: "fallback" });
    const frame = cell(page).locator("img.ym-photo-cell__frame");
    const src = async (): Promise<string> => (await frame.getAttribute("src")) ?? "";
    const first = await src();
    await page.keyboard.press("ArrowRight");
    await expect.poll(src).not.toBe(first);
    const second = await src();
    await page.getByRole("button", { name: /Signature/ }).first().click();
    await expect.poll(src).not.toBe(second);
    const before = (await frame.evaluate((img: HTMLImageElement) => img.naturalWidth)) as number;
    await page.setViewportSize({ width: 1800, height: 1100 });
    await expect.poll(() => frame.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(before);
    const box = (await cell(page).boundingBox())!;
    await expect.poll(() => frame.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBe(Math.round(box.width * 2));
  });

  test("takes the picture away for a file that can't be opened, and brings it back for the next photo", async ({ page }) => {
    await photos(page, { mockViewport: "fallback", mockSkipped: "DSC04205:This file is from a camera mode we can't read yet." });
    const frame = cell(page).locator("img.ym-photo-cell__frame");
    await expect(frame).toHaveCount(1);
    await page.keyboard.press("ArrowRight");
    await expect(page.getByText("This file can't be opened.", { exact: true })).toBeVisible();
    await expect(frame).toHaveCount(0);
    await page.keyboard.press("ArrowRight");
    await expect(frame).toHaveCount(1);
    await expect(page.getByText("This file can't be opened.", { exact: true })).toHaveCount(0);
  });
});

for (const backend of ["native", "fallback"] as const) {
  test.describe(`the pointer over the photo (${backend})`, () => {
    test("Pick white sends the click to the core as a point of the cell and shows what the core set", async ({ page }) => {
      await photos(page, { mockViewport: backend });
      await page.getByRole("button", { name: "Pick white" }).click();
      const layer = page.getByRole("application");
      const box = (await layer.boundingBox())!;
      await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.15);
      await expect(page.getByRole("tab", { name: "Colour" })).toHaveAttribute("aria-selected", "true");
      await expect(page.getByRole("region", { name: "White balance" }).getByText("Edited")).toBeVisible();
      const input = (await viewportOf(page)).inputs.at(-1)!;
      expect(input.kind).toBe("pick_white");
      // The mouse works in whole pixels.
      expect(Math.abs((input.x as number) - box.width * 0.3)).toBeLessThanOrEqual(1);
      expect(Math.abs((input.y as number) - box.height * 0.15)).toBeLessThanOrEqual(1);
      // One undo step takes it back, through the core.
      await page.keyboard.press("Meta+z");
      await expect(page.getByRole("region", { name: "White balance" }).getByText("Edited")).toHaveCount(0);
    });

    test("a double click toggles Fit and 100 %, ⌘ and the wheel zoom about the pointer, and a drag pans once zoomed", async ({ page }) => {
      await photos(page, { mockViewport: backend });
      const box = (await cell(page).boundingBox())!;
      await page.mouse.dblclick(box.x + 100, box.y + 50);
      await expect(page.getByRole("button", { name: /^100%/ })).toBeVisible();
      expect((await viewportOf(page)).inputs.at(-1)).toMatchObject({ kind: "toggle_zoom" });
      expect((await viewportOf(page)).inputs.at(-1)!.x as number).toBeCloseTo(100, 0);
      await page.mouse.dblclick(box.x + 100, box.y + 50);
      await expect(page.getByRole("button", { name: /^Fit/ })).toBeVisible();
      // A wheel zoom from Fit: the header follows, and the drag now pans.
      await page.mouse.move(box.x + 300, box.y + 200);
      await page.keyboard.down("Control");
      await page.mouse.wheel(0, -300);
      await page.keyboard.up("Control");
      await expect(page.getByRole("button", { name: /^\d+%/ })).toBeVisible();
      const inputs = (await viewportOf(page)).inputs.filter((i) => i.kind === "zoom");
      expect(inputs.length).toBeGreaterThan(0);
      expect(inputs[0]!.x as number).toBeCloseTo(300, 0);
      await page.mouse.move(box.x + 300, box.y + 200);
      await page.mouse.down();
      await page.mouse.move(box.x + 340, box.y + 230, { steps: 4 });
      await page.mouse.up();
      expect((await viewportOf(page)).inputs.filter((i) => i.kind === "pan").length).toBeGreaterThan(0);
    });

    test("⌥ reads the pixel under the pointer in CSS pixels of the cell", async ({ page }) => {
      await photos(page, { mockViewport: backend });
      const box = (await cell(page).boundingBox())!;
      await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.1);
      await page.keyboard.down("Alt");
      await expect(page.locator(".ym-photo-cell__readout")).toHaveText(/^R \d+ G \d+ B \d+$/);
      await page.keyboard.up("Alt");
      await expect(page.locator(".ym-photo-cell__readout")).toHaveCount(0);
    });
  });
}

test.describe("Update ready while the core exports", () => {
  test("is held back while the core says an export is busy, and comes back when it is quiet", async ({ page }) => {
    await photos(page);
    await showUpdateReady(page);
    const pill = page.getByRole("button", { name: /Update ready/ });
    await expect(pill).not.toHaveAttribute("aria-disabled", "true");
    await page.evaluate(() => (window as unknown as { __yellowmoon: { mockApi: { emitEvent: (e: string, p: unknown) => void } } }).__yellowmoon.mockApi.emitEvent("export_busy", { busy: true }));
    await expect(pill).toHaveAttribute("aria-disabled", "true");
    await expect(pill).toHaveAttribute("title", "Restarts after the export finishes");
    await page.evaluate(() => (window as unknown as { __yellowmoon: { mockApi: { emitEvent: (e: string, p: unknown) => void } } }).__yellowmoon.mockApi.emitEvent("export_busy", { busy: false }));
    await expect(pill).not.toHaveAttribute("aria-disabled", "true");
  });
});
