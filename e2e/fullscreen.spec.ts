import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, watchErrors } from "./helpers/app";

// Full screen (docs/11 §Full screen, the owner's request of 2026-10-20) in a real browser against the mocked API: F shows the photo
// alone, filling the window; F or Esc brings the layout back. The photo cell's rectangle is then the window's, and the core is told so
// (`viewport_set_rect`), as after a window resize. The window's own macOS full screen is the app's and is not played by the mock.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

type Rect = { x: number; y: number; width: number; height: number; scale: number };

/** The last rectangle the page sent the core (`viewport_set_rect`), as the mock recorded it. */
const sentRect = (page: Page): Promise<Rect | null> =>
  page.evaluate(() => JSON.parse(JSON.stringify((window as unknown as { __yellowmoon: { mockApi: { viewport: { rect: unknown } } } }).__yellowmoon.mockApi.viewport.rect)) as Rect | null);

const screenEl = (page: Page) => page.locator('.ym-screen[data-screen="photos"]');
const cell = (page: Page) => page.locator("[data-viewport]");
const header = (page: Page) => page.locator(".ym-screen > .ym-header");
const strip = (page: Page) => page.locator(".ym-screen > .ym-filmstrip");
const tools = (page: Page) => page.locator('.ym-screen > aside[aria-label="Tools"]');
/** The zoom button's label ("Fit 24%"): it is in the header, read even while the header is hidden. */
const zoomLabel = async (page: Page): Promise<string> => ((await page.locator(".ym-fit").textContent()) ?? "").trim();

async function photos(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await openPhotos(page);
  await expect(page.getByText("4 of 12 · 3 picked")).toBeVisible();
}

/** The cell's box is within `px` of the rectangle on every side. */
function expectBox(box: { x: number; y: number; width: number; height: number } | null, want: { x: number; y: number; width: number; height: number }, px = 2): void {
  expect(box).not.toBeNull();
  expect(Math.abs(box!.x - want.x)).toBeLessThanOrEqual(px);
  expect(Math.abs(box!.y - want.y)).toBeLessThanOrEqual(px);
  expect(Math.abs(box!.width - want.width)).toBeLessThanOrEqual(px);
  expect(Math.abs(box!.height - want.height)).toBeLessThanOrEqual(px);
}

test("F fills the window with the photo, tells the core, and F again puts everything back as it was", async ({ page }) => {
  const problems = watchErrors(page);
  await photos(page);
  const window = page.viewportSize()!;
  const before = (await cell(page).boundingBox())!;
  expect(before.width / before.height).toBeCloseTo(1.5, 2);
  await expect.poll(async () => {
    const r = await sentRect(page);
    return r && [r.width, r.height].map((v) => Math.round(v));
  }).toEqual([before.width, before.height].map((v) => Math.round(v)));
  const fitBefore = await zoomLabel(page);
  expect(fitBefore).toMatch(/^Fit \d+%$/);

  await page.keyboard.press("f");
  await expect(screenEl(page)).toHaveAttribute("data-fullscreen", "true");
  // The photo cell is the whole window.
  expectBox(await cell(page).boundingBox(), { x: 0, y: 0, width: window.width, height: window.height });
  await expect(header(page)).toBeHidden();
  await expect(strip(page)).toBeHidden();
  await expect(tools(page)).toBeHidden();
  // The camera line goes with them; the look chip stays.
  await expect(cell(page).locator(".ym-photo-cell__camera")).toBeHidden();
  await expect(cell(page).locator(".ym-photo-cell__chip")).toBeVisible();
  // Nothing scrolls, and the core and the aurora's cut-out follow the new rectangle.
  expect(await page.evaluate(() => document.documentElement.scrollHeight - document.documentElement.clientHeight)).toBeLessThanOrEqual(0);
  await expect.poll(async () => {
    const r = await sentRect(page);
    return r && [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height), r.scale];
  }).toEqual([0, 0, window.width, window.height, 2]);
  expect(await page.evaluate(() => document.documentElement.style.getPropertyValue("--cell-w"))).toBe(`${window.width}px`);
  // Fit fits the bigger cell, as after a window resize.
  const fitFull = await zoomLabel(page);
  expect(Number(/\d+/.exec(fitFull)?.[0])).toBeGreaterThan(Number(/\d+/.exec(fitBefore)?.[0]));
  // A screen reader hears it once.
  await expect(page.getByText("Full screen. Press F or Esc to leave.")).toBeAttached();

  await page.keyboard.press("f");
  await expect(screenEl(page)).not.toHaveAttribute("data-fullscreen");
  await expect(header(page)).toBeVisible();
  await expect(strip(page)).toBeVisible();
  await expect(tools(page)).toBeVisible();
  await expect(cell(page).locator(".ym-photo-cell__camera")).toBeVisible();
  expectBox(await cell(page).boundingBox(), before, 1);
  await expect.poll(async () => {
    const r = await sentRect(page);
    return r && [r.x, r.y, r.width, r.height].map((v) => Math.round(v));
  }).toEqual([before.x, before.y, before.width, before.height].map((v) => Math.round(v)));
  await expect.poll(() => zoomLabel(page)).toBe(fitBefore);
  expect(problems).toEqual([]);
});

test("Esc leaves full screen, after ending Pick white first", async ({ page }) => {
  await photos(page);
  await page.keyboard.press("w");
  await expect(page.getByRole("application", { name: /^Pick white/ })).toBeVisible();
  await page.keyboard.press("f");
  await expect(screenEl(page)).toHaveAttribute("data-fullscreen", "true");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("application", { name: /^Pick white/ })).toHaveCount(0);
  await expect(screenEl(page)).toHaveAttribute("data-fullscreen", "true");
  await page.keyboard.press("Escape");
  await expect(screenEl(page)).not.toHaveAttribute("data-fullscreen");
  await expect(header(page)).toBeVisible();
});

test("in full screen the arrows move between photos; back in the layout the strip marks the photo on screen", async ({ page }) => {
  await photos(page);
  await page.keyboard.press("f");
  for (let i = 0; i < 6; i += 1) await page.keyboard.press("ArrowRight");
  await page.keyboard.press("f");
  await expect(page.getByText("10 of 12 · 3 picked")).toBeVisible();
  const thumb = page.getByRole("button", { name: "Show DSC04210" });
  await expect(thumb).toHaveAttribute("aria-current", "true");
  await expect(thumb).toBeInViewport({ ratio: 1 });
});

test("full screen keeps the photo's main landmark, hides the header, strip and panel from a screen reader, and takes the focus off them", async ({ page }) => {
  await photos(page);
  // The focus was on a tab, which is hidden now: it does not stay there.
  await page.getByRole("tab", { name: "Light" }).click();
  await page.keyboard.press("f");
  await expect(screenEl(page)).toHaveAttribute("data-fullscreen", "true");
  expect(await page.evaluate(() => document.activeElement === document.body)).toBe(true);
  const snapshot = await page.locator("body").ariaSnapshot();
  expect(snapshot).toMatch(/- main "Photo"/);
  expect(snapshot).not.toMatch(/- banner\b/);
  expect(snapshot).not.toMatch(/- navigation\b/);
  expect(snapshot).not.toMatch(/- complementary\b/);
  expect(snapshot).not.toMatch(/- tab\b/);
});
