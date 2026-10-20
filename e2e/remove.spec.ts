import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, watchErrors } from "./helpers/app";

// Remove from filmstrip and its Undo (the owner's request "I also want to remove photos from the film strip", docs/11 §Removing
// photos): the right-click menu of a thumbnail, Delete and Backspace, the notice with its Undo button, ⌘Z, and the Start screen when
// no photo is left, in a real browser against the mocked API.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await openPhotos(page);
}

const strip = (page: Page) => page.getByRole("navigation", { name: "Photos" });
const thumbnails = (page: Page) => strip(page).getByRole("button");
/** The words of the notice bar (not its buttons). */
const notice = (page: Page) => page.locator(".ym-error-bar[data-kind=notice] [data-slot=item-content] > span");
const noticeBar = (page: Page) => page.locator(".ym-error-bar[data-kind=notice]");
const caption = (page: Page, text: string | RegExp) => strip(page).getByText(text);

test.describe("the thumbnail's menu", () => {
  test("removes the photo that was right-clicked, says the file is still on disk, and Undo puts it back", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page);
    await expect(thumbnails(page)).toHaveCount(12);

    await strip(page).getByRole("button", { name: "Show DSC04207" }).click({ button: "right" });
    const menu = page.getByRole("menu", { name: "DSC04207 actions" });
    await expect(menu).toBeVisible();
    // glasscn's context menu: the glass panel, one item for the photo and one for the picked photos.
    await expect(menu).toHaveAttribute("data-slot", "context-menu-content");
    await expect(menu.getByRole("menuitem")).toHaveText(["Remove from filmstrip", "Remove picked photos3"]);
    // The thumbnail the menu is about is ringed meanwhile.
    // (Radix hides the page behind an open menu from the accessibility tree, so the thumbnail is found by its label.)
    await expect(page.locator('nav.ym-filmstrip button[aria-label="Show DSC04207"]')).toHaveAttribute("data-menu", "open");
    await menu.getByRole("menuitem", { name: "Remove from filmstrip" }).click();

    await expect(thumbnails(page)).toHaveCount(11);
    await expect(strip(page).getByRole("button", { name: /^Show DSC04207/ })).toHaveCount(0);
    // The photo on screen did not change.
    await expect(caption(page, "4 of 11 · 3 picked")).toBeVisible();
    await expect(page.getByRole("img", { name: /^Photo DSC04204/ })).toBeVisible();
    await expect(notice(page)).toHaveText("Removed DSC04207. The file is still on disk.");
    await expect(noticeBar(page).getByRole("button", { name: "Undo" })).toBeVisible();

    await noticeBar(page).getByRole("button", { name: "Undo" }).click();
    await expect(thumbnails(page)).toHaveCount(12);
    // Back between DSC04206 and DSC04208, with its picture.
    const names = await thumbnails(page).evaluateAll((els) => els.map((el) => el.getAttribute("aria-label")));
    expect(names.indexOf("Show DSC04207")).toBe(6);
    await expect(strip(page).getByRole("button", { name: "Show DSC04207" }).locator("img")).toBeVisible();
    await expect(notice(page)).toHaveText("Put back DSC04207.");
    await expect(noticeBar(page).getByRole("button", { name: "Undo" })).toHaveCount(0);
    expect(problems).toEqual([]);
  });

  test("removes the picked photos, and ⌘Z puts them back", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page);
    await strip(page).getByRole("button", { name: "Show DSC04201" }).click({ button: "right" });
    await page.getByRole("menuitem", { name: /^Remove picked photos/ }).click();

    await expect(thumbnails(page)).toHaveCount(9);
    // DSC04204 was on screen and went with the others: DSC04206 follows it.
    await expect(page.getByRole("img", { name: /^Photo DSC04206/ })).toBeVisible();
    await expect(caption(page, "3 of 9 · 0 picked")).toBeVisible();
    await expect(notice(page)).toHaveText("Removed 3 photos. The files are still on disk.");
    await expect(page.getByRole("button", { name: /^Export\s?⌘E$/ })).toBeVisible();

    await page.keyboard.press("Meta+z");
    await expect(thumbnails(page)).toHaveCount(12);
    await expect(page.getByRole("img", { name: /^Photo DSC04204/ })).toBeVisible();
    await expect(caption(page, "4 of 12 · 3 picked")).toBeVisible();
    await expect(notice(page)).toHaveText("Put back 3 photos.");
    expect(problems).toEqual([]);
  });

  test("has Escape to close it, and no menu on the Export panel's strip", async ({ page }) => {
    await photos(page);
    await strip(page).getByRole("button", { name: "Show DSC04203, picked" }).click({ button: "right" });
    await expect(page.getByRole("menu")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("menu")).toHaveCount(0);
    await expect(thumbnails(page)).toHaveCount(12);

    await page.getByRole("button", { name: /^Export/ }).click();
    await page.getByRole("navigation", { name: "Photos to export" }).getByRole("img", { name: /DSC04203/ }).click({ button: "right" });
    await expect(page.getByRole("menu")).toHaveCount(0);
  });
});

test.describe("Delete and Backspace", () => {
  test("remove the photo on screen, one after the other", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page);
    await page.keyboard.press("Delete");
    await expect(thumbnails(page)).toHaveCount(11);
    await expect(page.getByRole("img", { name: /^Photo DSC04205/ })).toBeVisible();
    await expect(notice(page)).toHaveText("Removed DSC04204. The file is still on disk.");
    await page.keyboard.press("Backspace");
    await expect(thumbnails(page)).toHaveCount(10);
    await expect(page.getByRole("img", { name: /^Photo DSC04206/ })).toBeVisible();
    await expect(caption(page, "4 of 10 · 1 picked")).toBeVisible();
    expect(problems).toEqual([]);
  });

  test("leave a text box and a slider alone", async ({ page }) => {
    await photos(page);
    const search = page.getByRole("searchbox");
    await search.fill("sig");
    await search.press("Backspace");
    await search.press("Delete");
    await expect(search).toHaveValue("si");
    await expect(thumbnails(page)).toHaveCount(12);

    await page.getByRole("tab", { name: "Light" }).click();
    const box = page.getByRole("textbox", { name: "Exposure value" });
    await box.click();
    await box.press("Backspace");
    await page.getByRole("slider", { name: "Exposure" }).focus();
    await page.keyboard.press("Delete");
    await page.keyboard.press("Backspace");
    await expect(thumbnails(page)).toHaveCount(12);
  });

  test("remove the last photo of the strip and show the one before it", async ({ page }) => {
    await photos(page, { mockCurrent: 11 });
    await page.keyboard.press("Delete");
    await expect(thumbnails(page)).toHaveCount(11);
    await expect(page.getByRole("img", { name: /^Photo DSC04211/ })).toBeVisible();
    await expect(caption(page, "11 of 11 · 3 picked")).toBeVisible();
  });
});

test.describe("the last photo", () => {
  test("shows the Start screen with the notice, and Undo goes back to the photo", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page, { mockPhotos: 2, mockCurrent: 0, mockPicked: "" });
    await page.keyboard.press("Delete");
    await expect(thumbnails(page)).toHaveCount(1);
    await page.keyboard.press("Delete");
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Look" })).toHaveCount(0);
    await expect(notice(page)).toHaveText("Removed DSC04202. The file is still on disk.");

    await noticeBar(page).getByRole("button", { name: "Undo" }).click();
    await expect(page.getByRole("tab", { name: "Look" })).toBeVisible();
    await expect(thumbnails(page)).toHaveCount(1);
    await expect(page.getByRole("img", { name: /^Photo DSC04202/ })).toBeVisible();
    expect(problems).toEqual([]);
  });

  test("remove every photo with the picked ones, and open files again", async ({ page }) => {
    await photos(page, { mockPhotos: 3, mockCurrent: 0, mockPicked: "0,1,2" });
    await strip(page).getByRole("button", { name: /^Show DSC04202/ }).click({ button: "right" });
    await page.getByRole("menuitem", { name: /^Remove picked photos/ }).click();
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
    await expect(notice(page)).toHaveText("Removed 3 photos. The files are still on disk.");
    // The files are not "already open": they open again.
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("heading", { level: 2, name: /^\d+ photos? ready$/ })).toBeVisible();
  });
});
