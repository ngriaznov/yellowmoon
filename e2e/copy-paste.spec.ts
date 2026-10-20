import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, dropFiles, openApp, openPhotos, setValue, watchErrors } from "./helpers/app";

// "Add photos" in the header, drops on the Photos screen, and Copy settings, Paste and Paste Settings to Picked (the owner's request of
// 2026-10-16, docs/11 §Photos and §Copy and paste settings), in a real browser against the mocked API.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

type Mock = {
  queueChoice(kind: "open", paths: string[]): void;
  getSettings(id: number): Promise<{ settings: { crop: { rect: { width: number } } | null; orientation: number; exposure: unknown; look: string }; values: { contrast: number } }>;
  menuState: unknown;
};

const mock = <T>(page: Page, run: (api: Mock) => T | Promise<T>): Promise<T> =>
  page.evaluate(`(${run.toString()})(window.__yellowmoon.mockApi)`) as Promise<T>;

async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await openPhotos(page);
}

const header = (page: Page) => page.getByRole("navigation", { name: "Actions" });
const strip = (page: Page) => page.getByRole("navigation", { name: "Photos" });
const foot = (page: Page) => page.getByRole("group", { name: "Copy and paste settings" });
/** The words of the notice bar (not its Dismiss link). */
const notice = (page: Page) => page.locator(".ym-error-bar[data-kind=notice] [data-slot=item-content] > span");

test.describe("Add photos", () => {
  test("adds to the filmstrip from the Photos screen and keeps the current photo", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page);
    await expect(strip(page).getByRole("button")).toHaveCount(12);
    await expect(page.getByText(/^4 of 12 · /)).toBeVisible();

    await mock(page, (api) => api.queueChoice("open", ["~/Pictures/Lisbon/DSC09001.ARW", "~/Pictures/Lisbon/DSC09002.ARW", "~/Pictures/Lisbon/DSC04201.ARW"]));
    await header(page).getByRole("button", { name: "Add photos" }).click();

    await expect(strip(page).getByRole("button")).toHaveCount(14);
    await expect(strip(page).getByRole("button", { name: /^Show DSC09002/ })).toBeVisible();
    // The current photo stays where it was: DSC04204, now 4 of 14, and the other photos are untouched.
    await expect(page.getByText(/^4 of 14 · /)).toBeVisible();
    await expect(strip(page).getByRole("button", { name: "Show DSC04204, picked" })).toHaveAttribute("aria-current", "true");
    await expect(page.getByRole("img", { name: /^Photo DSC04204/ })).toBeVisible();
    await expect(page.getByText("3 picked")).toBeVisible();
    await expect(notice(page)).toHaveText("Added 2 photos. 1 was already open.");
    // The new photos develop (their thumbnails arrive) without the others changing.
    await expect(strip(page).getByRole("button", { name: "Show DSC09001" })).toBeVisible();
    await expect(strip(page).locator("img")).toHaveCount(14);
    // Export counts them.
    await page.getByRole("button", { name: /^Export/ }).click();
    await expect(page.getByRole("radio", { name: /All · 14/ })).toBeVisible();
    // The button is not there while the Export panel shows.
    await expect(header(page).getByRole("button", { name: "Add photos" })).toHaveCount(0);
    expect(problems).toEqual([]);
  });

  test("says so when everything chosen is open already", async ({ page }) => {
    await photos(page);
    await header(page).getByRole("button", { name: "Add photos" }).click();
    await expect(notice(page)).toHaveText("These photos are already open.");
    await expect(strip(page).getByRole("button")).toHaveCount(12);
  });

  test("takes files dropped on the Photos screen", async ({ page }) => {
    await photos(page);
    await dropFiles(page, ["DSC09100.ARW", "notes.txt"]);
    await expect(strip(page).getByRole("button")).toHaveCount(13);
    await expect(notice(page)).toHaveText("Added 1 photo.");
    await expect(page.getByText(/^4 of 13 · /)).toBeVisible();
    await dropFiles(page, ["holiday.jpg"]);
    await expect(notice(page)).toHaveText("No ARW files in what you dropped.");
    await expect(strip(page).getByRole("button")).toHaveCount(13);
  });

  test("the header keeps its labels in one piece at 1440 and gives way to the status text at 1180", async ({ page }) => {
    await photos(page);
    const nav = header(page);
    // At 1440 nothing wraps: every button is one line (32 px, glasscn's small control) and the status text is there, inside its cell.
    for (const name of ["Before", "Crop", "Pick white", "Add photos"]) {
      expect((await nav.getByRole("button", { name }).boundingBox())!.height).toBe(32);
    }
    const status = page.getByText("Neural · 42 ms · Display P3");
    await expect(status).toBeVisible();
    const cell = (await page.locator(".ym-header__right").boundingBox())!;
    expect((await status.boundingBox())!.x + (await status.boundingBox())!.width).toBeLessThanOrEqual(cell.x);
    // At the narrowest window nothing runs into the Export cell: the status text is left out.
    await page.setViewportSize({ width: 1180, height: 760 });
    await expect(status).toBeHidden();
    const scrolls = await page.locator(".ym-header__nav").evaluate((el) => el.scrollWidth - el.clientWidth);
    expect(scrolls).toBeLessThanOrEqual(0);
    await expect(nav.getByRole("button", { name: "Add photos" })).toBeVisible();
  });

  test("the Pick white hint never runs into the Export cell, at any width from 1180 up", async ({ page }) => {
    await photos(page);
    await header(page).getByRole("button", { name: "Pick white" }).click();
    const hint = page.getByText("Click something neutral: grey or white.");
    await expect(hint).toBeVisible();
    for (const width of [1180, 1273, 1300, 1359, 1360, 1440]) {
      await page.setViewportSize({ width, height: 760 });
      const cell = (await page.locator(".ym-header__right").boundingBox())!;
      const box = (await hint.boundingBox())!;
      expect(box.x + box.width, `the hint's right edge at ${width}`).toBeLessThanOrEqual(cell.x);
      const scrolls = await page.locator(".ym-header__nav").evaluate((el) => el.scrollWidth - el.clientWidth);
      expect(scrolls, `the header's overflow at ${width}`).toBeLessThanOrEqual(0);
    }
  });
});

test.describe("Copy settings and Paste", () => {
  test("copy on one photo, paste on another: the sliders match except the crop, and Undo restores the photo", async ({ page }) => {
    await openApp(page, MOCKUP);
    await configureMock(page, { initialSettings: { DSC04206: { crop: { rect: { x: 0.1, y: 0.1, width: 0.5, height: 0.5 }, angle_deg: 0, aspect: "free" }, orientation: 180 } } });
    await openPhotos(page);

    // Photo A: a contrast, an exposure and a look.
    await page.getByRole("tab", { name: "Light" }).click();
    await setValue(page, "Contrast", "30");
    await setValue(page, "Exposure", "0.8");
    await expect(foot(page).getByRole("button", { name: "Paste" })).toHaveAttribute("aria-disabled", "true");
    await foot(page).getByRole("button", { name: "Copy settings" }).click();
    await expect(notice(page)).toHaveText("Copied the settings of DSC04204.");
    await expect(foot(page).getByRole("button", { name: "Paste" })).toHaveAccessibleDescription("Paste the settings of DSC04204 (⇧⌘V)");

    // Photo B has a crop and a turn of its own.
    await strip(page).getByRole("button", { name: "Show DSC04206" }).click();
    await expect(page.getByText(/^6 of 12 · /)).toBeVisible();
    await expect(page.getByRole("textbox", { name: "Contrast value" })).toHaveValue("0");
    await foot(page).getByRole("button", { name: "Paste" }).click();
    await expect(notice(page)).toHaveText("Pasted the settings of DSC04204.");
    await expect(page.getByRole("textbox", { name: "Contrast value" })).toHaveValue("+30");
    await expect(page.getByRole("textbox", { name: "Exposure value" })).toHaveValue("+0.80");
    const b = await mock(page, (api) => api.getSettings(6));
    expect(b.values.contrast).toBe(30);
    expect(b.settings.crop?.rect.width).toBe(0.5);
    expect(b.settings.orientation).toBe(180);

    // One undo step: ⌘Z restores B, and A keeps its settings.
    await page.keyboard.press("Meta+z");
    await expect(page.getByRole("textbox", { name: "Contrast value" })).toHaveValue("0");
    await expect(page.getByRole("textbox", { name: "Exposure value" })).toHaveValue("+0.30");
    const after = await mock(page, (api) => api.getSettings(6));
    expect(after.settings.crop?.rect.width).toBe(0.5);
    expect((await mock(page, (api) => api.getSettings(4))).values.contrast).toBe(30);
  });

  test("⇧⌘C copies and ⇧⌘V pastes from the keyboard, not from a text box", async ({ page }) => {
    await photos(page);
    await page.getByRole("tab", { name: "Light" }).click();
    await setValue(page, "Contrast", "12");
    await page.keyboard.press("Meta+Shift+KeyC");
    await expect(notice(page)).toHaveText("Copied the settings of DSC04204.");
    await strip(page).getByRole("button", { name: "Show DSC04207" }).click();
    await expect(page.getByText(/^7 of 12 · /)).toBeVisible();
    // Plain ⌘V is the system's: nothing is pasted.
    await page.keyboard.press("Meta+KeyV");
    expect((await mock(page, (api) => api.getSettings(7))).values.contrast).toBe(0);
    // In a text box ⇧⌘V is left alone.
    await page.getByRole("tab", { name: "Look" }).click();
    await page.getByRole("searchbox", { name: "Search looks" }).click();
    await page.keyboard.press("Meta+Shift+KeyV");
    expect((await mock(page, (api) => api.getSettings(7))).values.contrast).toBe(0);
    await page.locator("body").click({ position: { x: 700, y: 300 } });
    await page.keyboard.press("Meta+Shift+KeyV");
    await expect(notice(page)).toHaveText("Pasted the settings of DSC04204.");
    expect((await mock(page, (api) => api.getSettings(7))).values.contrast).toBe(12);
  });

  test("Paste Settings to Picked (⌥⇧⌘V) changes every picked photo, each with its own undo", async ({ page }) => {
    await photos(page);
    await page.getByRole("tab", { name: "Light" }).click();
    await setValue(page, "Contrast", "21");
    await page.keyboard.press("Meta+Shift+KeyC");
    await expect(notice(page)).toHaveText("Copied the settings of DSC04204.");
    // Photos 3, 4 and 5 are picked; 4 is the current one.
    await page.keyboard.press("Meta+Alt+Shift+KeyV");
    await expect(notice(page)).toHaveText("Pasted the settings of DSC04204 to 3 photos.");
    const contrasts = await mock(page, async (api) => Promise.all([1, 2, 3, 4, 5, 6].map(async (id) => (await api.getSettings(id)).values.contrast)));
    expect(contrasts).toEqual([0, 0, 21, 21, 21, 0]);
    // Go to DSC04205 (picked) and undo there: only that photo goes back.
    await strip(page).getByRole("button", { name: "Show DSC04205, picked" }).click();
    await expect(page.getByText(/^5 of 12 · /)).toBeVisible();
    await expect(page.getByRole("textbox", { name: "Contrast value" })).toHaveValue("+21");
    await page.keyboard.press("Meta+z");
    await expect(page.getByRole("textbox", { name: "Contrast value" })).toHaveValue("0");
    const after = await mock(page, async (api) => Promise.all([3, 4, 5].map(async (id) => (await api.getSettings(id)).values.contrast)));
    expect(after).toEqual([21, 21, 0]);
  });

  test("the buttons are named, the menu items follow what can be done, and nothing is saved", async ({ page }) => {
    await photos(page);
    await expect(foot(page).getByRole("button", { name: "Copy settings" })).toHaveAccessibleName("Copy settings");
    await expect(foot(page).getByRole("button", { name: "Paste" })).toHaveAccessibleName("Paste");
    await expect(header(page).getByRole("button", { name: "Add photos" })).toHaveAccessibleName("Add photos");
    await foot(page).getByRole("button", { name: "Copy settings" }).click();
    await expect.poll(() => mock(page, (api) => api.menuState)).toEqual({ copy_settings: true, paste_settings: true, paste_settings_to_picked: true });
    // The notice that says so does not cover the row: Paste can be pressed while it shows.
    const bar = (await page.locator(".ym-error-bar").boundingBox())!;
    const panel = (await page.getByRole("complementary", { name: "Tools" }).boundingBox())!;
    expect(bar.x + bar.width).toBeLessThanOrEqual(panel.x);
    expect(await page.evaluate(() => localStorage.length + sessionStorage.length)).toBe(0);
  });
});
