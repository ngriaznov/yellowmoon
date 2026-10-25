import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { describeClips, frameClipping } from "./helpers/frames";
import type { FrameTarget } from "./helpers/frames";
import { arrowInList, openApp, openPhotos } from "./helpers/app";

// Selection frames, drawn whole (the owner, 2026-10-20: "Selection frame is cut off"). A ring is drawn outside the box it belongs to, and a
// box that clips its content (overflow hidden, auto, scroll) cuts it off when the item stands closer to the edge than the ring is wide. These
// tests measure it in the browser: where each ring is (`helpers/frames.ts`), and which ancestor clips which side of it, for
//
// - the look tile in use, and the one that has keyboard focus (the grid is inside a Collapsible that clips: the first row, the first column
//   and the last column lost 4 px of their ring);
// - the photo on screen in the filmstrip, at the start, the middle and the end of the strip, and in the Export panel;
// - the crop frame and its eight handles (the frame is the whole photo when Crop opens, so its right and bottom handles stood on the edge of
//   the cell and lost 2 to 4 px of their 14 px).
// - the chosen lens row of the Lens tab and the focus ring of the row that has the keyboard (the row's tab-colour ring is an inset ring, so
//   nothing of it is outside the row; the circle's focus ring is 5 px out and stands 14 px in from the row's edge; the list is longer than
//   the panel in the smaller windows, so the arrows scroll it).
//
// Screenshot comparisons are in screenshots.spec.ts.

const SIZES = [
  [1440, 900],
  [1180, 760],
  [1800, 1100],
] as const;

const LOOK_IN_USE: FrameTarget[] = [{ selector: '.ym-look-tile[aria-pressed="true"]', ring: ".ym-look-tile__picture" }];
const LOOK_FOCUS: FrameTarget[] = [{ selector: ".ym-look-tile:focus-visible", ring: ".ym-look-tile__picture" }];
const PHOTO_ON_SCREEN: FrameTarget[] = [{ selector: '.ym-filmstrip__thumb[aria-current="true"]' }];
const THUMB_FOCUS: FrameTarget[] = [{ selector: ".ym-filmstrip__thumb:focus-visible" }];
const LENS_CHOSEN: FrameTarget[] = [{ selector: '.ym-choice-row:has([data-state="checked"])' }];
const LENS_FOCUS: FrameTarget[] = [{ selector: '.ym-choice-row [data-slot="radio-group-item"]:focus-visible' }, { selector: '.ym-choice-row:has(:focus-visible)' }];
const CROP_HANDLES: FrameTarget[] = [{ selector: ".ym-crop__handle" }];
const CROP_FOCUS: FrameTarget[] = [{ selector: ".ym-crop__frame:focus-visible" }, { selector: ".ym-crop__handle:focus-visible" }];

async function photos(page: Page, size: readonly [number, number] = SIZES[0], extra: Record<string, string | number> = {}): Promise<void> {
  await page.setViewportSize({ width: size[0], height: size[1] });
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4", ...extra });
  await openPhotos(page);
}

/** Asserts that `count` rings were measured and none is cut. The count keeps a selector that matches nothing from passing. */
async function expectWhole(page: Page, targets: FrameTarget[], count = 1): Promise<void> {
  const reports = await frameClipping(page, targets);
  expect(reports, "the rings that were measured").toHaveLength(count);
  expect(describeClips(reports)).toEqual([]);
}

/** Makes the page believe the last input was the keyboard, so that `focus()` shows a focus ring (`:focus-visible`). */
const byKeyboard = (page: Page): Promise<void> => page.keyboard.press("Shift");

test.describe("look tiles", () => {
  for (const size of SIZES) {
    test(`the ring of the look in use is whole, whichever tile it is, at ${size[0]} × ${size[1]}`, async ({ page }) => {
      await photos(page, size);
      for (const key of ["1", "2", "3", "4", "5", "6", "7", "8", "9"]) {
        await page.keyboard.press(key);
        const tile = page.locator('.ym-look-tile[aria-pressed="true"]');
        await expect(tile).toHaveAttribute("aria-label", new RegExp(`key ${key}$`));
        // Bring the tile to the middle of the panel, so that only what clips the ring itself is measured, not a scroll that hides the tile.
        await tile.evaluate((el) => el.scrollIntoView({ block: "center" }));
        await expectWhole(page, LOOK_IN_USE);
      }
    });
  }

  test("a clicked tile, in the first row and the first column, shows its whole ring", async ({ page }) => {
    await photos(page);
    await page.getByRole("button", { name: /^Signature look/ }).click();
    await expect(page.getByRole("button", { name: /^Signature look/ })).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("button", { name: /^Neutral look/ }).click();
    await expect(page.getByRole("button", { name: /^Neutral look/ })).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("button", { name: /^Neutral look/ }).evaluate((el) => el.scrollIntoView({ block: "center" }));
    await expectWhole(page, LOOK_IN_USE);
  });

  test("a section that is folded and opened again still shows its tiles' rings whole", async ({ page }) => {
    await photos(page);
    const header = page.getByRole("button", { name: /^ARRI looks · LogC4/ });
    await header.click();
    await expect(page.getByRole("button", { name: /^Film warm look/ })).toHaveCount(0);
    await header.click();
    await page.keyboard.press("5");
    await expect(page.locator('.ym-look-tile[aria-pressed="true"]')).toHaveAttribute("aria-label", /key 5$/);
    // The height animates while it opens; the ring is measured when it has stopped.
    await page.evaluate(() => Promise.all(document.getAnimations().map((animation) => animation.finished.catch(() => undefined))));
    await page.locator('.ym-look-tile[aria-pressed="true"]').evaluate((el) => el.scrollIntoView({ block: "center" }));
    await expectWhole(page, LOOK_IN_USE);
  });

  test("a search that leaves one tile shows its ring whole in the corner of the grid", async ({ page }) => {
    await photos(page);
    await page.getByRole("button", { name: /^Neutral look/ }).click();
    await page.getByRole("searchbox", { name: "Search looks" }).fill("neutral");
    await expect(page.locator(".ym-look-tile")).toHaveCount(1);
    await page.locator('.ym-look-tile[aria-pressed="true"]').evaluate((el) => el.scrollIntoView({ block: "center" }));
    await expectWhole(page, LOOK_IN_USE);
  });

  test("the focus ring of every tile is whole, including the first row and the outer columns", async ({ page }) => {
    await photos(page);
    await byKeyboard(page);
    const tiles = page.locator(".ym-look-tile");
    const count = await tiles.count();
    expect(count).toBeGreaterThanOrEqual(9);
    for (let i = 0; i < count; i++) {
      await tiles.nth(i).focus();
      await tiles.nth(i).evaluate((el) => el.scrollIntoView({ block: "center" }));
      await expectWhole(page, LOOK_FOCUS);
    }
  });

  test("a tile that focus brings in from under the panel's top edge shows its ring whole", async ({ page }) => {
    // At 1180 × 760, the smallest window: the panel is 590 px high, so it scrolls far enough (to 841 px) for a tile of the first row to stand
    // partly above its top edge. (In a taller window, or for a tile of a later row, the panel cannot scroll that far: the Look tab ends
    // with the Tone card, and the Lens card, which used to make it long enough, has its own tab.)
    await photos(page, SIZES[1]);
    await byKeyboard(page);
    const scroller = page.locator(".ym-tools__scroll");
    const tile = page.locator(".ym-look-tile").first();
    // Scroll so that the tile starts 10 px above the top of the panel: partly in view, which focus then brings fully in. (The browser
    // centres an element that is partly out of view, so the ring ends up clear of the edge without any scroll padding.)
    await scroller.evaluate((el) => {
      const target = el.querySelector(".ym-look-tile")!;
      el.scrollTop += target.getBoundingClientRect().top - el.getBoundingClientRect().top + 10;
    });
    const placed = await page.evaluate(() => {
      const box = document.querySelector(".ym-tools__scroll")!.getBoundingClientRect();
      const t = document.querySelector(".ym-look-tile")!.getBoundingClientRect();
      return { above: box.top - t.top, belowTop: t.bottom - box.top };
    });
    expect(placed.above, "the tile starts above the panel's top edge").toBeGreaterThan(0);
    expect(placed.belowTop, "and reaches below it: it is partly in view").toBeGreaterThan(20);
    await tile.focus();
    await expectWhole(page, LOOK_FOCUS);
  });

  test("a tile that focus brings in from under the panel's bottom edge shows its ring whole", async ({ page }) => {
    // The same at the other edge, in every window: the last row of the grid, scrolled until 10 px of it show above the panel's bottom edge.
    for (const size of SIZES) {
      await photos(page, size);
      await byKeyboard(page);
      const tile = page.locator(".ym-look-tile").last();
      await page.locator(".ym-tools__scroll").evaluate((el) => {
        const target = [...el.querySelectorAll(".ym-look-tile")].at(-1)!;
        el.scrollTop += target.getBoundingClientRect().top - (el.getBoundingClientRect().bottom - 10);
      });
      const placed = await page.evaluate(() => {
        const box = document.querySelector(".ym-tools__scroll")!.getBoundingClientRect();
        const t = [...document.querySelectorAll(".ym-look-tile")].at(-1)!.getBoundingClientRect();
        return { insideBottom: box.bottom - t.top, belowBottom: t.bottom - box.bottom };
      });
      expect(placed.insideBottom, `${size.join(" × ")}: the tile starts above the panel's bottom edge`).toBeGreaterThan(0);
      expect(placed.belowBottom, `${size.join(" × ")}: and ends below it: it is partly in view`).toBeGreaterThan(20);
      await tile.focus();
      await expectWhole(page, LOOK_FOCUS);
    }
  });
});

test.describe("lens rows", () => {
  for (const size of SIZES) {
    test(`the chosen row and the focus ring are whole for each of the seven lenses, first row to last, at ${size[0]} × ${size[1]}`, async ({ page }) => {
      await photos(page, size);
      await page.getByRole("tab", { name: "Lens" }).click();
      await byKeyboard(page);
      const radios = page.getByRole("radio");
      await expect(radios).toHaveCount(7);
      await radios.first().focus();
      for (let i = 0; i < 7; i++) {
        await expect(radios.nth(i)).toBeFocused();
        await expect(radios.nth(i)).toHaveAttribute("aria-checked", "true");
        // The row that has the keyboard is the chosen one: one ring measured for each, the focus ring of the circle and of the row.
        await expectWhole(page, LENS_CHOSEN);
        await expectWhole(page, LENS_FOCUS, 2);
        if (i < 6) await arrowInList(page, "ArrowDown");
      }
      // And back up to the first row, which the panel scrolls to its top edge, below its scroll padding.
      for (let i = 5; i >= 0; i--) {
        await arrowInList(page, "ArrowUp");
        await expect(radios.nth(i)).toBeFocused();
        await expectWhole(page, LENS_FOCUS, 2);
      }
    });
  }

  test("a row that focus brings in from under the panel's top edge shows its focus ring whole", async ({ page }) => {
    // At 1180 × 760 the list is longer than the 590 px panel, which can scroll to put the first row partly above its top edge (in a taller
    // window the whole tab fits or nearly: the panel cannot scroll that far).
    await photos(page, SIZES[1]);
    await page.getByRole("tab", { name: "Lens" }).click();
    await byKeyboard(page);
    await page.locator(".ym-tools__scroll").evaluate((el) => {
      const row = el.querySelector(".ym-choice-row")!;
      el.scrollTop += row.getBoundingClientRect().top - el.getBoundingClientRect().top + 10;
    });
    const placed = await page.evaluate(() => {
      const box = document.querySelector(".ym-tools__scroll")!.getBoundingClientRect();
      const r = document.querySelector(".ym-choice-row")!.getBoundingClientRect();
      return { above: box.top - r.top, belowTop: r.bottom - box.top };
    });
    expect(placed.above, "the row starts above the panel's top edge").toBeGreaterThan(0);
    expect(placed.belowTop, "and reaches below it: it is partly in view").toBeGreaterThan(20);
    await page.getByRole("radio").first().focus();
    await expectWhole(page, LENS_FOCUS, 2);
  });
});

test.describe("filmstrip", () => {
  for (const size of SIZES) {
    test(`the photo on screen keeps its whole ring from the start of the strip to the end, at ${size[0]} × ${size[1]}`, async ({ page }) => {
      // 24 photos: even in the widest window the strip is a few screens long, so it has a start, a middle and an end.
      await photos(page, size, { mockPhotos: 24 });
      const strip = page.locator(".ym-filmstrip__scroll");
      const thumbs = page.locator(".ym-filmstrip__thumb");
      const count = await thumbs.count();
      expect(count).toBe(24);
      expect(await strip.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeGreaterThan(500);

      await thumbs.first().click();
      await expect(thumbs.first()).toHaveAttribute("aria-current", "true");
      expect(await strip.evaluate((el) => el.scrollLeft)).toBe(0);
      await expectWhole(page, PHOTO_ON_SCREEN);

      // Walk to the end with the arrow keys: each step selects the next photo and the strip scrolls it in.
      for (let i = 1; i < count; i++) {
        await page.keyboard.press("ArrowRight");
        await expect(thumbs.nth(i)).toHaveAttribute("aria-current", "true");
        await expectWhole(page, PHOTO_ON_SCREEN);
        // Focus follows the selection, and its ring (3 px, 3 px off) is drawn around the thumbnail as well.
        await expect(thumbs.nth(i)).toBeFocused();
        await expectWhole(page, THUMB_FOCUS);
      }
      expect(await strip.evaluate((el) => Math.round(el.scrollWidth - el.clientWidth - el.scrollLeft))).toBeLessThanOrEqual(1);

      // And back to the start.
      for (let i = count - 2; i >= 0; i--) {
        await page.keyboard.press("ArrowLeft");
        await expect(thumbs.nth(i)).toHaveAttribute("aria-current", "true");
        await expectWhole(page, PHOTO_ON_SCREEN);
      }
      expect(await strip.evaluate((el) => el.scrollLeft)).toBe(0);
    });
  }

  test("a thumbnail clicked half way in at the edge of the strip is brought in whole", async ({ page }) => {
    await photos(page);
    const strip = page.locator(".ym-filmstrip__scroll");
    const thumbs = page.locator(".ym-filmstrip__thumb");
    // At the start, the photo that runs off the right edge is partly in view.
    const edge = await strip.evaluate((el) => {
      const right = el.getBoundingClientRect().right;
      const list = [...el.querySelectorAll(".ym-filmstrip__thumb")];
      return list.findIndex((t) => t.getBoundingClientRect().right > right);
    });
    expect(edge).toBeGreaterThan(0);
    // `click()` of the element itself: Playwright's own click would first scroll the thumbnail in, and the strip's own scroll is what is under test.
    await thumbs.nth(edge).evaluate((el) => (el as HTMLElement).click());
    await expect(thumbs.nth(edge)).toHaveAttribute("aria-current", "true");
    await expectWhole(page, PHOTO_ON_SCREEN);
  });

  test("the photo on screen is whole in the Export panel's strip too", async ({ page }) => {
    await photos(page);
    await page.keyboard.press("Meta+e");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    await expect(page.locator(".ym-filmstrip__thumb img")).toHaveCount(12);
    await expectWhole(page, PHOTO_ON_SCREEN);
  });
});

test.describe("crop", () => {
  for (const size of SIZES) {
    test(`the frame's eight handles are whole when it is the whole photo, at ${size[0]} × ${size[1]}`, async ({ page }) => {
      await photos(page, size);
      await page.getByRole("button", { name: "Crop" }).click();
      await expect(page.getByRole("complementary", { name: "Crop" })).toBeVisible();
      await expectWhole(page, CROP_HANDLES, 8);
    });
  }

  test("the focus ring of the frame and of each handle is whole when the frame is the whole photo", async ({ page }) => {
    await photos(page);
    await page.getByRole("button", { name: "Crop" }).click();
    await expect(page.getByRole("complementary", { name: "Crop" })).toBeVisible();
    await byKeyboard(page);
    await page.getByRole("group", { name: /Crop frame/ }).focus();
    await expectWhole(page, CROP_FOCUS);
    const handles = page.locator(".ym-crop__handle");
    for (let i = 0; i < 8; i++) {
      await handles.nth(i).focus();
      await expectWhole(page, CROP_FOCUS);
    }
  });

  test("the handles stay whole when a corner is dragged in, and again when it is dragged back", async ({ page }) => {
    await photos(page);
    await page.getByRole("button", { name: "Crop" }).click();
    await page.getByRole("complementary", { name: "Crop" }).getByRole("radio", { name: "Free" }).click();
    const image = (await page.locator(".ym-crop__image").boundingBox())!;
    const corner = (await page.getByRole("button", { name: /Crop bottom right corner/ }).boundingBox())!;
    await page.mouse.move(corner.x + corner.width / 2, corner.y + corner.height / 2);
    await page.mouse.down();
    await page.mouse.move(image.x + image.width * 0.6, image.y + image.height * 0.6, { steps: 4 });
    await expectWhole(page, CROP_HANDLES, 8);
    await page.mouse.move(image.x + image.width + 20, image.y + image.height + 20, { steps: 4 });
    await page.mouse.up();
    await expectWhole(page, CROP_HANDLES, 8);
  });
});
