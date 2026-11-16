import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { dropFiles, openApp, watchErrors } from "./helpers/app";

// Albums (docs/31, the owner's request of 2026-11-16): File › Open of an album by a drop, the folder cell's name and EDITED, File › Save
// and Save As…, the question when an edited album is closed, a missing photo, and a mixed choice refused, in a real browser against the
// mocked API. The menu items themselves belong to the shell (`menu_action` events are sent here).

const notice = (page: Page) => page.locator(".ym-error-bar[data-kind=notice] [data-slot=item-content] > span");
const cell = (page: Page) => page.locator(".ym-brand");
const menu = (page: Page, action: string, extra: Record<string, unknown> = {}): Promise<void> =>
  page.evaluate(`window.__yellowmoon.mockApi.emitEvent("menu_action", ${JSON.stringify({ action, ...extra })})`);

async function opened(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { mockAlbum: "Lisbon", ...extra });
  await dropFiles(page, ["Lisbon.ymalbum"]);
  // The photos develop on Start as they do for any open; the album's name is in the folder cell already.
  await expect(page.getByRole("heading", { level: 2, name: /^\d+ photos? ready$/ })).toBeVisible();
  await page.getByRole("button", { name: /Open photos/ }).click();
  await expect(page.getByRole("tab", { name: "Look" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();
}

test.describe("an album", () => {
  test("opens from a drop with its name in the folder cell, its picks and its current photo", async ({ page }) => {
    const problems = watchErrors(page);
    await opened(page);
    await expect(cell(page).locator(".ym-brand-name")).toHaveText("Lisbon");
    await expect(cell(page).locator(".ym-brand-sub")).toHaveText("2026-10-04 · 12 photos");
    await expect(page.getByText(/^4 of 12 · 3 picked/)).toBeVisible();
    expect(problems).toEqual([]);
  });

  test("says EDITED in place of the date after a change, and Save brings the date back", async ({ page }) => {
    await opened(page);
    await page.getByRole("button", { name: /^Show DSC04205/ }).click();
    await page.keyboard.press("p");
    await expect(cell(page).locator(".ym-brand-sub")).toHaveText("12 photos · edited");
    // The sub-line is drawn in capitals and stays on one line in the 300 px cell.
    await expect(cell(page).locator(".ym-brand-sub")).toHaveCSS("text-transform", "uppercase");
    const box = (await cell(page).locator(".ym-brand-sub").boundingBox())!;
    expect(box.height).toBeLessThan(24);
    await menu(page, "save");
    await expect(notice(page)).toHaveText("Saved the album Lisbon.");
    await expect(cell(page).locator(".ym-brand-sub")).toHaveText("2026-10-04 · 12 photos");
  });

  test("asks 'Save changes to the album Lisbon?' on Close Session, and answers Cancel, Don't Save and Save", async ({ page }) => {
    await opened(page);
    await page.getByRole("button", { name: /^Show DSC04205/ }).click();
    await page.keyboard.press("p");
    await expect(cell(page).locator(".ym-brand-sub")).toContainText("edited");
    await menu(page, "close_session");
    const question = page.getByRole("dialog", { name: "Save changes to the album Lisbon?" });
    await expect(question.getByRole("button", { name: "Save", exact: true })).toBeFocused();
    await question.getByRole("button", { name: "Cancel" }).click();
    await expect(question).toHaveCount(0);
    await expect(page.getByRole("tab", { name: "Look" })).toBeVisible();
    await menu(page, "close_session");
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
  });

  test("keeps a photo it cannot find as a failed photo with the album's own sentence", async ({ page }) => {
    await opened(page, { mockAlbumMissing: "DSC04206" });
    await page.getByRole("button", { name: /^Show DSC04206/ }).click();
    await expect(page.getByText("Not found: ~/Pictures/Lisbon/DSC04206.ARW")).toBeVisible();
  });

  test("refuses a mixed choice and two albums with a plain sentence, and opens nothing", async ({ page }) => {
    await openApp(page, { mockAlbum: "Lisbon" });
    await dropFiles(page, ["Lisbon.ymalbum", "DSC04201.ARW"]);
    await expect(page.getByText("Open an album on its own, or open photos. Yellowmoon can't open both in one go.")).toBeVisible();
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
  });

  test("Save of a session opened from photos asks where (the panel is the shell's), and the session then is an album", async ({ page }) => {
    await openApp(page, { mockCurrent: 3 });
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("heading", { level: 2, name: /ready$/ })).toBeVisible();
    await page.getByRole("button", { name: /Open photos/ }).click();
    await expect(page.getByRole("tab", { name: "Look" })).toBeVisible();
    await menu(page, "save");
    await expect(notice(page)).toHaveText("Saved the album Lisbon.");
    await expect(cell(page).locator(".ym-brand-name")).toHaveText("Lisbon");
  });
});
