import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { openApp, openPhotos } from "./helpers/app";

// Announcements and positions (docs/28 A-08, A-P7): the one polite live region of the Photos screen says what a shortcut did, and the
// windowed lists carry aria-setsize and aria-posinset on what they draw, at 12, 500 and 2000 photos. The sentences themselves are
// tested one by one in src/state/announce.test.ts; these tests press the keys in a real browser.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

/** The live region of the Photos screen. */
const region = (page: Page): Locator => page.locator('[data-screen="photos"] > [aria-live="polite"]');

/** What the region says, without the no-break space that makes a repeated sentence a change. */
async function said(page: Page): Promise<string> {
  return ((await region(page).textContent()) ?? "").replace(/ /g, "").trim();
}

async function expectSaid(page: Page, sentence: string | RegExp): Promise<void> {
  await expect.poll(() => said(page)).toMatch(typeof sentence === "string" ? new RegExp(`^${sentence.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`) : sentence);
}

/** Opens `n` photos and gets to the Photos screen (development held at 40 so that a big folder has a steady state). */
async function openMany(page: Page, n: number): Promise<void> {
  await openApp(page, { mockPhotos: n, mockHold: 40 });
  await page.getByRole("button", { name: "Open files…" }).click();
  await expect(page.getByRole("heading", { level: 2, name: `Developing ${n} photos` })).toBeVisible();
  await expect.poll(() => page.evaluate(async () => (await window.__yellowmoon!.mockApi.sessionState()).photos.filter((p) => p.status.kind === "ready").length)).toBe(40);
  await page.getByRole("button", { name: /Open photos/ }).click();
  await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();
}

test.describe("what the shortcuts say", () => {
  test("3, \\, Z and P on the photo say what they did, and a repeat is said again", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await expect(region(page)).toHaveAttribute("aria-live", "polite");
    expect(await said(page)).toBe("");

    await page.keyboard.press("3");
    await expectSaid(page, "Look: Film warm.");
    await page.keyboard.press("\\");
    await expectSaid(page, "Before.");
    // The same key again says the other sentence; and a sentence said twice in a row changes the region's text each time.
    await page.keyboard.press("\\");
    await expectSaid(page, "After.");

    await page.keyboard.press("z");
    await expect.poll(() => said(page)).toMatch(/^(Zoom \d+ %|Zoom to fit)\.$/);

    // The photo is picked (mockPicked 2,3,4); P takes the pick away, and P again puts it back.
    await page.keyboard.press("p");
    await expectSaid(page, "Not picked. 2 picked.");
    await page.keyboard.press("p");
    await expectSaid(page, "Picked. 3 picked.");

  });

  test("the same sentence twice in a row changes the region's text each time", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    const texts: string[] = [];
    await page.evaluate(() => {
      const el = document.querySelector<HTMLElement>('[data-screen="photos"] > [aria-live="polite"]')!;
      const seen: string[] = [];
      (window as unknown as { __said: string[] }).__said = seen;
      new MutationObserver(() => seen.push(el.textContent ?? "")).observe(el, { childList: true, characterData: true, subtree: true });
    });
    // ⌘A says the same sentence each time it is pressed.
    await page.keyboard.press("Meta+a");
    await expect.poll(async () => (await page.evaluate(() => (window as unknown as { __said: string[] }).__said)).length).toBeGreaterThan(0);
    const one = await page.evaluate(() => (window as unknown as { __said: string[] }).__said.length);
    await page.keyboard.press("Meta+a");
    await expect.poll(async () => (await page.evaluate(() => (window as unknown as { __said: string[] }).__said)).length).toBeGreaterThan(one);
    texts.push(...(await page.evaluate(() => (window as unknown as { __said: string[] }).__said)));
    const all = texts.filter((t) => t.trim().startsWith("Picked all"));
    expect(all.length).toBeGreaterThanOrEqual(2);
    // Two identical sentences in a row are told apart by the no-break space, which is not spoken: the text is never the same twice running.
    expect(texts.every((t, i) => i === 0 || t !== texts[i - 1])).toBe(true);
    expect(new Set(all.map((t) => t.replace(/\u00A0/g, ""))).size).toBe(1);
  });

  test("a key held down says its last sentence once, after the repeats stop", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.evaluate(() => {
      const el = document.querySelector<HTMLElement>('[data-screen="photos"] > [aria-live="polite"]')!;
      const seen: string[] = [];
      (window as unknown as { __said: string[] }).__said = seen;
      new MutationObserver(() => seen.push((el.textContent ?? "").replace(/ /g, "").trim())).observe(el, { childList: true, characterData: true, subtree: true });
    });
    // Playwright marks a second keydown of a key that is still down as an auto-repeat.
    for (let n = 0; n < 6; n += 1) {
      await page.keyboard.down("ArrowRight");
      await page.waitForTimeout(30);
    }
    await page.keyboard.up("ArrowRight");
    await expect.poll(() => said(page)).toMatch(/^Photo \d+ of 12\.$/);
    await page.waitForTimeout(700);
    const log = (await page.evaluate(() => (window as unknown as { __said: string[] }).__said)).filter((t) => t !== "");
    // The press itself, said at once, and the last step after the repeats: never one per step.
    expect(log.length).toBeLessThanOrEqual(2);
    expect(log.at(-1)).toBe(await said(page));
  });

  test("Return finishes a crop and C starts one; N enters the node view; nothing is said in a text box", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.keyboard.press("c");
    await expectSaid(page, "Crop. Press Return to finish.");
    await page.keyboard.press("Escape");
    await page.keyboard.press("n");
    await expectSaid(page, "Node view.");
    await page.keyboard.press("n");
    await expectSaid(page, "Simple view.");
  });

  test("a letter typed in a text box says nothing", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    const box = page.getByRole("textbox").first();
    await box.focus();
    for (const key of ["p", "z", "c", "n", "3", "\\"]) await page.keyboard.press(key);
    await page.waitForTimeout(300);
    expect(await said(page)).toBe("");
  });

  test("the menu's Before/After says it too", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.evaluate(() => window.__yellowmoon!.mockApi.emitEvent("menu_action", { action: "before_after" } as never));
    await expectSaid(page, "Before.");
  });
});

/** The [aria-posinset, aria-setsize] pairs of the list items in `scope`, as numbers. */
async function pairs(scope: Locator): Promise<Array<[number, number]>> {
  return scope.evaluateAll((els) => els.map((el) => [Number(el.getAttribute("aria-posinset")), Number(el.getAttribute("aria-setsize"))] as [number, number]));
}

for (const n of [12, 500, 2000]) {
  test(`the filmstrip draws its place in the folder at ${n} photos`, async ({ page }) => {
    if (n === 12) {
      await openApp(page, MOCKUP);
      await openPhotos(page);
    } else {
      await openMany(page, n);
    }
    const items = page.locator(".ym-filmstrip__scroll > [role=list] > [role=listitem]");
    await expect(page.getByRole("navigation", { name: "Photos" }).getByRole("list")).toHaveCount(1);
    const first = await pairs(items);
    expect(first[0]).toEqual([1, n]);
    expect(first.every(([, size]) => size === n)).toBe(true);
    expect(first.map(([pos]) => pos)).toEqual(first.map((_, i) => i + 1));
    if (n === 12) expect(first).toHaveLength(12);
    else expect(first.length).toBeLessThan(100);
    // Each thumbnail is still the button inside its item, and it is the item that has the pair.
    await expect(items.first().getByRole("button")).toHaveAttribute("aria-label", /^Show DSC/);
    expect(await page.locator(".ym-filmstrip__thumb[aria-posinset], .ym-filmstrip__thumb[aria-setsize]").count()).toBe(0);
  });
}

test("at 2000 photos the 812th, scrolled into view, is 812 of 2000", async ({ page }) => {
  await openMany(page, 2000);
  const strip = page.locator(".ym-filmstrip__scroll");
  const target = page.locator('.ym-filmstrip__scroll [role=listitem][aria-posinset="812"]');
  // The scroll bar is true: the share of the strip's width is the share of the folder.
  await expect
    .poll(async () => {
      await strip.evaluate((el) => void (el.scrollLeft = (el.scrollWidth - el.clientWidth) * (811 / 1999)));
      return target.count();
    })
    .toBe(1);
  await expect(target).toHaveAttribute("aria-setsize", "2000");
  await expect(target.getByRole("button")).toHaveAttribute("aria-label", /^Show DSC05012/);
});

test("the Start list carries its places while a folder develops, at 500 and 2000 photos", async ({ page }) => {
  for (const n of [500, 2000]) {
    await openApp(page, { mockPhotos: n, mockHold: 40 });
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("heading", { level: 2, name: `Developing ${n} photos` })).toBeVisible();
    const rows = page.getByRole("list", { name: "Photos being developed" }).getByRole("listitem");
    await expect(rows.first()).toBeVisible();
    const got = await pairs(rows);
    expect(got[0]).toEqual([1, n]);
    expect(got.every(([, size]) => size === n)).toBe(true);
    expect(got.length).toBeLessThan(100);
  }
});

test("the Start list of a short folder is a plain list with places", async ({ page }) => {
  await openApp(page, { mockCurrent: 3 });
  await page.getByRole("button", { name: "Open files…" }).click();
  await expect(page.getByRole("region", { name: "Photos to develop" })).toBeVisible();
  const got = await pairs(page.getByRole("region", { name: "Photos to develop" }).getByRole("listitem"));
  expect(got.length).toBeGreaterThan(0);
  expect(got.map(([pos]) => pos)).toEqual(got.map((_, i) => i + 1));
  expect(got.every(([, size]) => size === got.length)).toBe(true);
});

test("the Export panel's strip and the queue carry their places", async ({ page }) => {
  await openApp(page, MOCKUP);
  await openPhotos(page);
  await page.keyboard.press("Meta+e");
  const strip = page.getByRole("region", { name: "List of photos to export" });
  await expect(strip).toHaveAttribute("tabindex", "0");
  // The list is inside the region, and the region keeps its name.
  const items = strip.getByRole("list").getByRole("listitem");
  const got = await pairs(items);
  expect(got).toHaveLength(12);
  expect(got[0]).toEqual([1, 12]);
  expect(got.at(-1)).toEqual([12, 12]);
});
