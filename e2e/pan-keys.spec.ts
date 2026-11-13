import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { openApp, openPhotos } from "./helpers/app";
import { ringContrast } from "./helpers/contrast";

// Pan and the pixel readout from the keyboard (`docs/28` A-10, A-P10): a zoomed photo is a Tab stop named "Photo, zoomed. The arrow keys
// move it.", its arrows pan it (5 % of the view, 25 % with Shift), Home centres it and I says the pixel at the middle in the polite region of
// A-08. ← and → still step photos from anywhere else. The mock keeps the sum of the pans as `viewport.offset`.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;
const NAME = "Photo, zoomed. The arrow keys move it.";

const view = (page: Page): Locator => page.locator(".ym-photo-cell__view");
const region = (page: Page): Locator => page.locator('[data-screen="photos"] > [aria-live="polite"]');
const offset = (page: Page): Promise<{ x: number; y: number }> =>
  page.evaluate(() => (window as unknown as { __yellowmoon: { mockApi: { viewport: { offset: { x: number; y: number } } } } }).__yellowmoon.mockApi.viewport.offset);
const said = async (page: Page): Promise<string> => ((await region(page).textContent()) ?? "").replace(/\u00a0/g, "").trim();

/** The zoom button of the header: "Fit 24%" at Fit, "200%" once a step is chosen. */
const zoomButton = (page: Page): Locator => page.getByRole("button", { name: /^(Fit|\d+%)/ });

async function zoomTo200(page: Page): Promise<void> {
  await zoomButton(page).click();
  await page.getByRole("menu").getByText("200%", { exact: true }).click();
  await expect(page.getByRole("img", { name: NAME })).toBeVisible();
}

test.describe("the zoomed photo from the keyboard", () => {
  test("at Fit the photo is not a Tab stop; at 200 % it is, named, with a ring", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await expect(view(page)).not.toHaveAttribute("tabindex", /.*/);
    await zoomTo200(page);
    const photo = page.getByRole("img", { name: NAME });
    await expect(photo).toHaveAttribute("tabindex", "0");
    // Tab until the photo has the focus (it is reached from the page's start, whatever is before it).
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    let reached = false;
    for (let i = 0; i < 60 && !reached; i++) {
      await page.keyboard.press("Tab");
      reached = await photo.evaluate((el) => el === document.activeElement);
    }
    expect(reached, "Tab reaches the zoomed photo").toBe(true);
    expect(await photo.evaluate((el) => el.matches(":focus-visible"))).toBe(true);
    const ring = await photo.evaluate((el) => {
      const outline = getComputedStyle(el);
      const after = getComputedStyle(el, "::after");
      return { style: outline.outlineStyle, width: outline.outlineWidth, shadow: after.boxShadow };
    });
    expect(ring.style).toBe("solid");
    expect(parseFloat(ring.width)).toBeGreaterThanOrEqual(3);
    expect(ring.shadow).not.toBe("none");
    // Back to Fit: it is no stop again.
    await zoomButton(page).click();
    await page.getByRole("menu").getByText("Fit", { exact: true }).click();
    await expect(view(page)).not.toHaveAttribute("tabindex", /.*/);
  });

  test("the ring reads on a white, a grey and a black photo (3:1, A-P5)", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await zoomTo200(page);
    const photo = page.getByRole("img", { name: NAME });
    for (const colour of ["#fff", "#808080", "#000"]) {
      const style = await page.addStyleTag({
        content: `.ym-photo-cell { background: ${colour} !important; } .ym-photo-cell__image, .ym-photo-cell__frame { visibility: hidden !important; }`,
      });
      await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
      await page.evaluate(() => new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))));
      const before = await page.screenshot({ animations: "disabled" });
      await page.keyboard.press("Shift");
      await photo.focus();
      expect(await photo.evaluate((el) => el.matches(":focus-visible"))).toBe(true);
      await page.evaluate(() => new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))));
      const after = await page.screenshot({ animations: "disabled" });
      const box = (await photo.boundingBox())!;
      const { ratio, pixels } = await ringContrast(page, before, after, { x: box.x - 4, y: box.y - 4, w: box.width + 8, h: box.height + 8 });
      expect(pixels, `${colour}: the ring is drawn`).toBeGreaterThan(0);
      expect(ratio, `${colour}: the ring's contrast`).toBeGreaterThanOrEqual(3);
      await style.evaluate((node) => (node as Element).remove());
    }
  });

  test("the arrows pan by 5 %, Shift by 25 %, and Home centres", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await zoomTo200(page);
    const photo = page.getByRole("img", { name: NAME });
    await photo.focus();
    const width = (await photo.boundingBox())!.width;
    expect(await offset(page)).toEqual({ x: 0, y: 0 });

    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => (await offset(page)).x).toBeCloseTo(-0.05 * width, 1);
    const small = (await offset(page)).x;
    await page.keyboard.press("Shift+ArrowRight");
    await expect.poll(async () => (await offset(page)).x).toBeCloseTo(small - 0.25 * width, 1);
    expect(Math.abs((await offset(page)).x)).toBeGreaterThan(Math.abs(small));
    const height = (await photo.boundingBox())!.height;
    await page.keyboard.press("ArrowDown");
    await expect.poll(async () => (await offset(page)).y).toBeCloseTo(-0.05 * height, 1);
    // The arrows are used up there: the photo did not change.
    await expect(page.getByText(/^4 of 12 · /)).toBeVisible();

    await page.keyboard.press("Home");
    await expect.poll(() => offset(page)).toEqual({ x: 0, y: 0 });
    // Home keeps the zoom.
    await expect(zoomButton(page)).toHaveText(/^200%/);
    expect(await page.evaluate(() => (window as unknown as { __yellowmoon: { mockApi: { viewport: { mode: unknown } } } }).__yellowmoon.mockApi.viewport.mode)).toEqual({ kind: "percent", value: 200 });
  });

  test("with the focus on the filmstrip, → and ← still step photos, zoomed or not", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await zoomTo200(page);
    // The filmstrip steps from the thumbnail that has the focus: the one of the photo on screen (photo 4).
    await page.locator('[data-photo-id][tabindex="0"]').focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByText(/^5 of 12 · /)).toBeVisible();
    // A button, and the page itself, do the same.
    await zoomButton(page).focus();
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByText(/^4 of 12 · /)).toBeVisible();
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    await page.keyboard.press("ArrowRight");
    await expect(page.getByText(/^5 of 12 · /)).toBeVisible();
  });

  test("a click on the zoomed photo does not take the focus, so ← and → keep stepping photos", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await zoomTo200(page);
    const box = (await view(page).boundingBox())!;
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    expect(await view(page).evaluate((el) => el === document.activeElement)).toBe(false);
    await page.keyboard.press("ArrowRight");
    await expect(page.getByText(/^5 of 12 · /)).toBeVisible();
  });

  test("I says the pixel at the middle in the live region, in the form of the ⌥ readout", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await zoomTo200(page);
    await page.getByRole("img", { name: NAME }).focus();
    expect(await said(page)).toBe("");
    await page.keyboard.press("i");
    await expect.poll(() => said(page)).toMatch(/^R \d+ G \d+ B \d+\.$/);
    // The same pixel the ⌥ readout reads at the middle of the cell.
    const spoken = (await said(page)).replace(/\.$/, "");
    const box = (await view(page).boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.keyboard.down("Alt");
    await expect(page.locator(".ym-photo-cell__readout")).toHaveText(spoken);
    await page.keyboard.up("Alt");
  });

  test("I does nothing in a text box", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await zoomTo200(page);
    const search = page.getByRole("searchbox").first();
    await search.focus();
    await page.keyboard.type("i");
    await expect(search).toHaveValue(/i/);
    expect(await said(page)).toBe("");
  });
});
