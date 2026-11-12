import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, watchErrors } from "./helpers/app";

// The memory notice (`docs/26` H-14): the core says it once on the way into the critical level (`memory_notice` with the sentence) and
// says it is gone when the level is normal (`null`). Shown in the error bar's notice style, in a real browser against the mocked API.

const SENTENCE = "Your Mac is short of memory, so Yellowmoon paused background work. Close other apps to speed it up again.";

const say = (page: Page, message: string | null) =>
  page.evaluate((m) => window.__yellowmoon!.mockApi.emitEvent("memory_notice", { message: m }), message);
const bar = (page: Page) => page.locator(".ym-error-bar[data-kind=notice]");

test("shows the sentence in the notice bar, goes when the level is normal, and Dismiss hides it", async ({ page }) => {
  const problems = watchErrors(page);
  await openApp(page, { mockCurrent: 3 });
  await openPhotos(page);
  await expect(bar(page)).toHaveCount(0);

  await say(page, SENTENCE);
  await expect(bar(page)).toHaveCount(1);
  await expect(bar(page).locator("[data-slot=item-content] > span")).toHaveText(SENTENCE);
  await expect(bar(page)).toHaveAttribute("role", "status");

  await say(page, null);
  await expect(bar(page)).toHaveCount(0);

  await say(page, SENTENCE);
  await bar(page).getByRole("button", { name: "Dismiss" }).click();
  await expect(bar(page)).toHaveCount(0);
  expect(problems).toEqual([]);
});
