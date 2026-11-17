import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { dropFiles, openApp, openPhotos } from "./helpers/app";

// The owner's report on 0.2.4-beta.1: after going to Export and back to the Photos screen the filmstrip looked empty. The Export panel
// draws the same strip in its "export" mode; coming back must leave the cells, their pictures, the scroll offset and the photo on screen
// as they were (`docs/11` §Filmstrip). 60 photos are drawn whole, 150 are windowed (`docs/26` H-11).

const strip = (page: Page) => page.getByRole("navigation", { name: /^Photos/ });
const scroller = (page: Page) => page.locator(".ym-filmstrip__scroll");
// The photo on screen, from the caption ("21 of 150 · 0 picked"): its thumbnail is not drawn while the strip is scrolled far from it.
const current = (page: Page) => strip(page).locator(".ym-filmstrip__caption");

/** Opens the photos (or the album Lisbon) and waits for the strip. */
async function open(page: Page, options: Record<string, string | number>, album = false): Promise<void> {
  if (!album) {
    await openApp(page, options);
    await openPhotos(page);
  } else {
    await openApp(page, { mockAlbum: "Lisbon", ...options });
    await dropFiles(page, ["Lisbon.ymalbum"]);
    await expect(page.getByRole("heading", { level: 2, name: /^\d+ photos? ready$/ })).toBeVisible();
    await page.getByRole("button", { name: /Open photos/ }).click();
    await expect(page.getByRole("tab", { name: "Look" })).toBeVisible();
    await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();
  }
  await expect(strip(page).getByRole("button", { name: /^Show / }).first()).toBeVisible();
}

/** Scrolls the strip by arithmetic so that the thumbnail at `index` stands at its left edge. */
async function scrollTo(page: Page, index: number): Promise<void> {
  await scroller(page).evaluate((el, i) => {
    const thumbs = el.querySelectorAll<HTMLElement>(".ym-filmstrip__thumb");
    const pitch = thumbs[1]!.getBoundingClientRect().left - thumbs[0]!.getBoundingClientRect().left;
    el.scrollLeft = pitch * i;
  }, index);
  await page.waitForTimeout(300);
}

const scrollLeft = (page: Page): Promise<number> => scroller(page).evaluate((el) => el.scrollLeft);

/** The thumbnails inside the strip's view: how many, and how many of those have a picture that has loaded. */
async function inView(page: Page): Promise<{ cells: number; pictures: number; first: string }> {
  return scroller(page).evaluate((el) => {
    const box = el.getBoundingClientRect();
    const cells = [...el.querySelectorAll<HTMLElement>(".ym-filmstrip__thumb")].filter((t) => {
      const r = t.getBoundingClientRect();
      return r.right > box.left && r.left < box.right;
    });
    const pictures = cells.filter((t) => {
      const img = t.querySelector("img");
      return img !== null && img.complete && img.naturalWidth > 0;
    });
    return { cells: cells.length, pictures: pictures.length, first: cells[0]?.getAttribute("aria-label") ?? "" };
  });
}

async function toExport(page: Page): Promise<void> {
  await page.keyboard.press("Meta+e");
  await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Photos to export" })).toBeVisible();
}

async function toPhotos(page: Page): Promise<void> {
  await page.getByRole("button", { name: "← Back to photos" }).click();
  await expect(strip(page).getByRole("button", { name: /^Show / }).first()).toBeVisible();
}

/**
 * The drawn cells' spacing: every neighbouring pair is one pitch apart (a thumbnail's width plus the CSS gap), in the whole list, and
 * the pitch is the same as before. A pitch measured with another size of cell (the Export panel's) leaves a cell-sized gap between them.
 */
async function spacing(page: Page): Promise<{ pitches: number[]; width: number; gap: number; items: number }> {
  return scroller(page).evaluate((el) => {
    const thumbs = [...el.querySelectorAll<HTMLElement>(".ym-filmstrip__thumb")];
    const items = [...el.querySelectorAll<HTMLElement>(".ym-filmstrip__item")];
    const gap = parseFloat(getComputedStyle(el.querySelector(".ym-filmstrip__list")!).columnGap);
    const lefts = thumbs.map((t) => t.getBoundingClientRect().left);
    // The first and last drawn items carry the stand-in space as a margin; neighbours inside are what is compared.
    // Each list item is exactly as wide as its thumbnail: a wider one is a gap after every cell.
    for (const item of items) {
      const t = item.querySelector<HTMLElement>(".ym-filmstrip__thumb")!;
      if (Math.abs(item.getBoundingClientRect().width - t.getBoundingClientRect().width) > 0.5) throw new Error("a list item is wider than its thumbnail");
    }
    return {
      pitches: lefts.slice(1).map((l, i) => Math.round((l - lefts[i]!) * 100) / 100),
      width: thumbs[0]!.getBoundingClientRect().width,
      gap,
      items: items.length,
    };
  });
}

/** Every pair of drawn neighbours is one thumbnail and one gap apart, and the drawn cells fill the view without a hole. */
async function expectEvenSpacing(page: Page, width?: number): Promise<number> {
  const s = await spacing(page);
  const pitch = s.width + s.gap;
  expect(s.items).toBeGreaterThan(3);
  // The first pair and the last pair have the stand-in space as margin on one side, the rest are exact.
  for (const p of s.pitches.slice(1, -1)) expect(Math.abs(p - pitch)).toBeLessThan(0.6);
  if (width !== undefined) expect(Math.abs(s.width - width)).toBeLessThan(0.6);
  const v = await scroller(page).evaluate((el) => {
    const box = el.getBoundingClientRect();
    const lefts = [...el.querySelectorAll<HTMLElement>(".ym-filmstrip__thumb")]
      .map((t) => t.getBoundingClientRect())
      .filter((r) => r.right > box.left && r.left < box.right)
      .map((r) => r.left);
    return lefts;
  });
  // The view holds at least as many cells as fit in it (none missing between those drawn).
  expect(v.length).toBeGreaterThan(Math.floor((await scroller(page).evaluate((el) => el.clientWidth)) / pitch) - 1);
  return s.width;
}

/** The round trip, and what must hold after it. */
async function roundTrip(page: Page, middle: number, trips = 1): Promise<void> {
  await scrollTo(page, middle);
  const before = await scrollLeft(page);
  expect(before).toBeGreaterThan(100);
  const first = (await inView(page)).first;
  const width = await expectEvenSpacing(page);
  const on = await current(page).textContent();
  for (let i = 0; i < trips; i++) {
    await toExport(page);
    await toPhotos(page);
  }
  // The cells are drawn, each has its picture, and the strip stands where it stood.
  await expect.poll(async () => (await inView(page)).cells, { timeout: 5000 }).toBeGreaterThan(0);
  await expect.poll(async () => {
    const v = await inView(page);
    return v.pictures === v.cells;
  }, { timeout: 5000 }).toBe(true);
  expect(Math.abs((await scrollLeft(page)) - before)).toBeLessThan(2);
  expect((await inView(page)).first).toBe(first);
  expect(await current(page).textContent()).toBe(on);
  await expectEvenSpacing(page, width);
  // And it goes on following the scroll: a strip that lost its listener draws nothing new when scrolled further.
  await scrollTo(page, middle + 25);
  await expect.poll(async () => {
    const v = await inView(page);
    return v.cells > 0 && v.pictures === v.cells;
  }, { timeout: 5000 }).toBe(true);
}

for (const count of [60, 150, 202]) {
  test.describe(`${count} photos`, () => {
    test("photos opened: the strip keeps its cells, pictures and place after Export", async ({ page }) => {
      await open(page, { mockPhotos: count, mockCurrent: count === 202 ? 170 : 20 });
      await roundTrip(page, count === 202 ? 171 : 30);
    });

    test("an album: the strip keeps its cells, pictures and place after Export", async ({ page }) => {
      await open(page, { mockPhotos: count }, true);
      await roundTrip(page, count === 202 ? 171 : 30);
    });

    test("with the shell's thumbnail budget on, the pictures are asked for again", async ({ page }) => {
      await open(page, { mockPhotos: count, mockCurrent: count === 202 ? 170 : 20, mockThumbnailBudget: 14 });
      await roundTrip(page, count === 202 ? 171 : 30);
    });

    test("in a small window", async ({ page }) => {
      await page.setViewportSize({ width: 760, height: 640 });
      await open(page, { mockPhotos: count, mockCurrent: count === 202 ? 170 : 20 });
      await roundTrip(page, count === 202 ? 171 : 30);
    });

    test("the strip's height changes while Export is up: the cells are re-measured", async ({ page }) => {
      await open(page, { mockPhotos: count, mockCurrent: count === 202 ? 170 : 20 });
      await scrollTo(page, count === 202 ? 171 : 30);
      const before = await expectEvenSpacing(page);
      await toExport(page);
      await page.setViewportSize({ width: 1440, height: 1100 });
      await toPhotos(page);
      await expect.poll(async () => (await spacing(page)).width, { timeout: 5000 }).not.toBe(before);
      await expectEvenSpacing(page);
      await scrollTo(page, 40);
      await expect.poll(async () => (await inView(page)).cells > 0, { timeout: 5000 }).toBe(true);
      await expectEvenSpacing(page);
    });

    test("going to Export and back twice", async ({ page }) => {
      await open(page, { mockPhotos: count, mockCurrent: count === 202 ? 170 : 20 });
      await roundTrip(page, count === 202 ? 171 : 30, 2);
    });
  });
}
