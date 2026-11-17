import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos } from "./helpers/app";

// Removing a photo keeps the filmstrip where it was and hands the photo on screen to the next one that stays (the previous one when the
// last went), `docs/11` §Removing photos. The folders here are long enough for the strip to scroll: 60 photos are drawn whole, 150 are windowed.

const strip = (page: Page) => page.getByRole("navigation", { name: "Photos" });
const scroller = (page: Page) => page.locator(".ym-filmstrip__scroll");

/** Opens `count` photos with photo `current` (0-based) on screen and the strip scrolled so that photo `middle` is at its left edge. */
async function open(page: Page, count: number, current: number, middle: number): Promise<void> {
  await openApp(page, { mockPhotos: count, mockCurrent: current, mockPicked: "" });
  await openPhotos(page);
  await expect(strip(page).getByRole("button", { name: /^Show / }).first()).toBeVisible();
  await scrollTo(page, middle);
}

/** The thumbnail names in view, with each one's left edge relative to the strip, in order. */
async function inView(page: Page): Promise<{ name: string; left: number }[]> {
  return scroller(page).evaluate((el) => {
    const box = el.getBoundingClientRect();
    return [...el.querySelectorAll<HTMLElement>(".ym-filmstrip__thumb")]
      .map((t) => ({ name: t.getAttribute("aria-label") ?? "", left: Math.round(t.getBoundingClientRect().left - box.left), right: t.getBoundingClientRect().right - box.left }))
      .filter((t) => t.right > 0 && t.left < box.width)
      .map(({ name, left }) => ({ name, left }));
  });
}

/** Scrolls the strip by arithmetic so that the thumbnail at `index` stands at its left edge (the pitch is measured from two thumbnails). */
async function scrollTo(page: Page, index: number): Promise<void> {
  await scroller(page).evaluate((el, i) => {
    const thumbs = el.querySelectorAll<HTMLElement>(".ym-filmstrip__thumb");
    const pitch = thumbs[1]!.getBoundingClientRect().left - thumbs[0]!.getBoundingClientRect().left;
    el.scrollLeft = pitch * i;
  }, index);
  await page.waitForTimeout(200);
}

const scrollLeft = (page: Page): Promise<number> => scroller(page).evaluate((el) => el.scrollLeft);
/** Whether the thumbnail called `name` lies wholly inside the strip. */
const visible = (page: Page, name: string): Promise<boolean> =>
  scroller(page).evaluate((el, n) => {
    const t = el.querySelector<HTMLElement>(`.ym-filmstrip__thumb[aria-label^="${n}"]`);
    if (!t) return false;
    const a = el.getBoundingClientRect();
    const b = t.getBoundingClientRect();
    return b.left >= a.left && b.right <= a.right;
  }, name);
const name = (index: number): string => `DSC0${4201 + index}`;
const current = (page: Page) => strip(page).locator('.ym-filmstrip__thumb[aria-current="true"]');

for (const count of [60, 150]) {
  test.describe(`${count} photos`, () => {
    test("removing the photo on screen keeps the strip's place and selects the next photo", async ({ page }) => {
      // The strip shows photos 30 to about 40; photo 40 is on screen, at its right.
      await open(page, count, 20, 36);
      // The person clicks the photo in the strip, so the keyboard focus is on its thumbnail.
      await strip(page).getByRole("button", { name: new RegExp(`^Show ${name(40)}`) }).click();
      await expect(current(page)).toHaveAttribute("aria-label", new RegExp(`^Show ${name(40)}`));
      await page.waitForTimeout(300);
      const before = await scrollLeft(page);
      const first = (await inView(page))[0]?.name;
      await page.keyboard.press("Delete");
      await expect(page.getByText(/^Removed /)).toBeVisible();
      await expect(current(page)).toHaveAttribute("aria-label", new RegExp(`^Show ${name(41)}`));
      await expect(strip(page).getByText(`41 of ${count - 1} · 0 picked`)).toBeVisible();
      await page.waitForTimeout(300);
      // The thumbnails in view stay where they were, closing the gap: the same one is at the start.
      expect((await inView(page))[0]?.name).toBe(first);
      expect(Math.abs((await scrollLeft(page)) - before)).toBeLessThan(2);
      // The focus is on the strip still (on the photo on screen), so the arrows go on working.
      await expect(page.locator(":focus")).toHaveAttribute("aria-label", new RegExp(`^Show ${name(41)}`));
      expect(await visible(page, `Show ${name(41)}`)).toBe(true);
    });

    test("the menu's Remove on the photo on screen keeps the strip's place too", async ({ page }) => {
      await open(page, count, 20, 36);
      await strip(page).getByRole("button", { name: new RegExp(`^Show ${name(40)}`) }).click();
      await expect(current(page)).toHaveAttribute("aria-label", new RegExp(`^Show ${name(40)}`));
      await page.waitForTimeout(300);
      const before = await scrollLeft(page);
      await strip(page).getByRole("button", { name: new RegExp(`^Show ${name(40)}`) }).click({ button: "right" });
      await page.getByRole("menuitem", { name: "Remove from filmstrip" }).click();
      await expect(page.getByText(/^Removed /)).toBeVisible();
      await expect(current(page)).toHaveAttribute("aria-label", new RegExp(`^Show ${name(41)}`));
      await page.waitForTimeout(300);
      expect(Math.abs((await scrollLeft(page)) - before)).toBeLessThan(2);
      expect(await visible(page, `Show ${name(41)}`)).toBe(true);
    });

    test("removing a photo that is not on screen leaves the photo and the scroll alone", async ({ page }) => {
      await open(page, count, 40, 36);
      const before = await scrollLeft(page);
      await strip(page).getByRole("button", { name: new RegExp(`^Show ${name(38)}`) }).click({ button: "right" });
      await page.getByRole("menuitem", { name: "Remove from filmstrip" }).click();
      await expect(page.getByText(/^Removed /)).toBeVisible();
      await expect(current(page)).toHaveAttribute("aria-label", new RegExp(`^Show ${name(40)}`));
      await expect(strip(page).getByText(`40 of ${count - 1} · 0 picked`)).toBeVisible();
      await page.waitForTimeout(300);
      expect(Math.abs((await scrollLeft(page)) - before)).toBeLessThan(2);
    });

    test("removing the last photo selects the one before it", async ({ page }) => {
      await open(page, count, count - 1, count - 6);
      await page.keyboard.press("Delete");
      await expect(page.getByText(/^Removed /)).toBeVisible();
      await expect(current(page)).toHaveAttribute("aria-label", new RegExp(`^Show ${name(count - 2)}`));
      await page.waitForTimeout(300);
      expect(await visible(page, `Show ${name(count - 2)}`)).toBe(true);
      expect(await scrollLeft(page)).toBeGreaterThan(100);
    });

    test("removing picked photos hands over to the next that stays, and Undo does not jump to the start", async ({ page }) => {
      await openApp(page, { mockPhotos: count, mockCurrent: 40, mockPicked: "38,39,40,41,45" });
      await openPhotos(page);
      await expect(strip(page).getByRole("button", { name: /^Show / }).first()).toBeVisible();
      await scrollTo(page, 36);
      const before = await scrollLeft(page);
      await strip(page).getByRole("button", { name: new RegExp(`^Show ${name(39)}`) }).click({ button: "right" });
      await page.getByRole("menuitem", { name: /^Remove picked photos/ }).click();
      await expect(page.getByText(/^Removed 5 photos/)).toBeVisible();
      // 40 and 41 went too: 42 is next, and it is the 38th now (38, 39, 40, 41 and 45 left).
      await expect(current(page)).toHaveAttribute("aria-label", new RegExp(`^Show ${name(42)}`));
      await page.waitForTimeout(300);
      expect(await visible(page, `Show ${name(42)}`)).toBe(true);
      expect(await scrollLeft(page)).toBeGreaterThan(before / 2);

      await page.getByRole("button", { name: "Undo" }).click();
      await expect(page.getByText(/^Put back 5 photos/)).toBeVisible();
      await expect(current(page)).toHaveAttribute("aria-label", new RegExp(`^Show ${name(40)}`));
      await page.waitForTimeout(300);
      expect(await scrollLeft(page)).toBeGreaterThan(before / 2);
      expect(await visible(page, `Show ${name(40)}`)).toBe(true);
    });
  });
}
