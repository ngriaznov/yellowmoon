import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { dropFiles, openApp, openPhotos, setValue, watchErrors } from "./helpers/app";

// Presets (docs/30, the owner's decision of 2026-11-16): the Presets card as the last card of the Look tab, the Save dialog, applying to
// the picked photos, update, rename, delete, the notices, a dropped file and the keyboard, in a real browser against the mocked API.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await openPhotos(page);
}

const card = (page: Page) => page.getByRole("region", { name: "Presets" });
const tile = (page: Page, name: string) => card(page).getByRole("button", { name: `${name} preset` });
const notice = (page: Page) => page.locator(".ym-error-bar[data-kind=notice] [data-slot=item-content] > span");
const mock = <T>(page: Page, run: (api: { getSettings(id: number): Promise<{ values: { main: { grade: { contrast: number } } } }> }) => T | Promise<T>): Promise<T> =>
  page.evaluate(`(${run.toString()})(window.__yellowmoon.mockApi)`) as Promise<T>;

test.describe("the Presets card", () => {
  test("is the last card of the Look tab, empty at first, and there is no Tone card", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page);
    const titles = await page.getByRole("complementary", { name: "Tools" }).getByRole("heading", { level: 3 }).allTextContents();
    expect(titles.filter((t) => ["Automatic", "Look", "Presets"].includes(t))).toEqual(["Automatic", "Look", "Presets"]);
    expect(titles.at(-1)).toBe("Presets");
    await expect(page.getByRole("region", { name: "Tone" })).toHaveCount(0);
    await expect(card(page).getByText("Set a photo the way you like it, then save it as a preset.")).toBeVisible();
    await expect(card(page).getByRole("button", { name: "Save preset…" })).toBeVisible();
    expect(problems).toEqual([]);
  });

  test("shows six tiles of fourteen, Show all and Show fewer, with the ring on the preset the photo has", async ({ page }) => {
    await photos(page, { mockPresets: 14 });
    await card(page).scrollIntoViewIfNeeded();
    await expect(card(page).getByRole("button", { name: / preset$/ })).toHaveCount(6);
    await expect(card(page).getByText("14", { exact: true })).toBeVisible();
    await card(page).getByRole("button", { name: "Show all 14" }).click();
    await expect(card(page).getByRole("button", { name: / preset$/ })).toHaveCount(14);
    await card(page).getByRole("button", { name: "Show fewer" }).click();
    await expect(card(page).getByRole("button", { name: / preset$/ })).toHaveCount(6);
    // The list is sorted by name, ignoring case (docs/30 section 6): the first six of the fourteen.
    await expect(card(page).getByRole("button", { name: / preset$/ }).first()).toHaveAccessibleName("Cool morning preset");
    await tile(page, "Cool morning").click();
    await expect(notice(page)).toHaveText("Applied Cool morning to DSC04204.");
    await expect(tile(page, "Cool morning")).toHaveAttribute("aria-pressed", "true");
    await expect(tile(page, "Faded film")).toHaveAttribute("aria-pressed", "false");
  });

  test("saves a preset from the photo, with the changed groups ticked, and says it", async ({ page }) => {
    await photos(page);
    await page.getByRole("tab", { name: "Light" }).click();
    await setValue(page, "Contrast", "30");
    await page.getByRole("tab", { name: "Look" }).click();
    await card(page).getByRole("button", { name: "Save preset…" }).click();
    const dialog = page.getByRole("dialog", { name: "Save preset" });
    await expect(dialog.getByRole("textbox", { name: "Name" })).toBeFocused();
    await expect(dialog.getByRole("checkbox", { name: "Light" })).toBeChecked();
    await expect(dialog.getByRole("checkbox", { name: "Look" })).not.toBeChecked();
    await dialog.getByRole("textbox", { name: "Name" }).fill("Warm evening");
    await dialog.getByRole("button", { name: "Save" }).click();
    await expect(notice(page)).toHaveText("Saved the preset Warm evening.");
    await expect(tile(page, "Warm evening")).toBeVisible();
    await expect(tile(page, "Warm evening")).toHaveAttribute("aria-pressed", "true");
    // The same name again asks first.
    await card(page).getByRole("button", { name: "Save…" }).click();
    await page.getByRole("textbox", { name: "Name" }).fill("warm evening");
    await page.getByRole("checkbox", { name: "Light" }).check();
    await page.keyboard.press("Enter");
    const question = page.getByRole("dialog", { name: "Replace Warm evening?" });
    await expect(question.getByRole("button", { name: "Cancel" })).toBeFocused();
    await question.getByRole("button", { name: "Replace" }).click();
    await expect(notice(page)).toHaveText("Replaced the preset warm evening.");
  });

  test("applies to the picked photos from the tile's menu, one undo step each", async ({ page }) => {
    await photos(page, { mockPresets: 3 });
    await tile(page, "Warm evening").click({ button: "right" });
    const menu = page.getByRole("menu", { name: "Warm evening actions" });
    await expect(menu.getByRole("menuitem")).toHaveText(["Apply to Picked Photos", "Update from This Photo", "Rename…", "Export…", "Show in Finder", "Delete"]);
    await menu.getByRole("menuitem", { name: "Apply to Picked Photos" }).click();
    await expect(notice(page)).toHaveText("Applied Warm evening to 3 photos.");
    const contrasts = await mock(page, async (api) => Promise.all([2, 3, 4, 5, 6].map(async (id) => (await api.getSettings(id)).values.main.grade.contrast)));
    expect(contrasts).toEqual([0, 12, 12, 12, 0]);
    await page.keyboard.press("Meta+z");
    const after = await mock(page, async (api) => Promise.all([3, 4, 5].map(async (id) => (await api.getSettings(id)).values.main.grade.contrast)));
    expect(after).toEqual([12, 0, 12]);
  });

  test("updates after a question, renames with the Name field alone, and deletes with no question", async ({ page }) => {
    await photos(page, { mockPresets: 3 });
    await tile(page, "Warm evening").click({ button: "right" });
    await page.getByRole("menuitem", { name: "Update from This Photo" }).click();
    const update = page.getByRole("dialog", { name: "Update Warm evening?" });
    await expect(update).toContainText("It keeps its groups, Look, Light and Colour, and takes their values from DSC04204.");
    await update.getByRole("button", { name: "Update" }).click();
    await expect(notice(page)).toHaveText("Updated Warm evening from DSC04204.");

    await tile(page, "Warm evening").click({ button: "right" });
    await page.getByRole("menuitem", { name: "Rename…" }).click();
    const rename = page.getByRole("dialog", { name: "Rename preset" });
    await rename.getByRole("textbox", { name: "Name" }).fill("cool morning");
    await rename.getByRole("button", { name: "Rename" }).click();
    await expect(rename.getByText("A preset called cool morning already exists.")).toBeVisible();
    await rename.getByRole("textbox", { name: "Name" }).fill("Late sun");
    await rename.getByRole("button", { name: "Rename" }).click();
    await expect(notice(page)).toHaveText("Renamed the preset Warm evening to Late sun.");

    await tile(page, "Late sun").click({ button: "right" });
    await page.getByRole("menuitem", { name: "Delete" }).click();
    await expect(notice(page)).toHaveText("Moved the preset Late sun to the Trash.");
    await expect(tile(page, "Late sun")).toHaveCount(0);
  });

  test("a missing look: the rest is applied and the notice names the look", async ({ page }) => {
    await photos(page, { mockPresets: 3, mockMissing: "film-warm" });
    await expect(tile(page, "Warm evening")).toHaveAccessibleDescription("Look, Light and Colour. The look Film warm isn't imported.");
    await tile(page, "Warm evening").click();
    await expect(notice(page)).toHaveText("Warm evening uses the look Film warm, which isn't imported. Everything else was applied.");
  });

  test("lists a damaged file with the reason and Delete", async ({ page }) => {
    await photos(page, { mockPresets: 2, mockPresetsDamaged: 1 });
    await expect(card(page).getByText("This preset was made by a newer version of Yellowmoon. Update to use it.")).toBeVisible();
    await card(page).getByRole("button", { name: "Delete Old.ympreset" }).click();
    await expect(card(page).getByText("Old.ympreset")).toHaveCount(0);
  });

  test("takes a dropped .ympreset on Photos and says it", async ({ page }) => {
    await photos(page);
    await dropFiles(page, ["Warm evening.ympreset"]);
    await expect(notice(page)).toHaveText("Added the preset Warm evening.");
    await expect(tile(page, "Warm evening")).toBeVisible();
  });

  test("takes a dropped .ympreset on Start, under the drop zone, and starts no session", async ({ page }) => {
    await openApp(page, MOCKUP);
    await dropFiles(page, ["Warm evening.ympreset"]);
    await expect(page.getByText("Added the preset Warm evening.")).toBeVisible();
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
    await expect(page.getByText("No ARW files in what you dropped.")).toHaveCount(0);
  });

  test("a tile is a tab stop and Return applies it", async ({ page }) => {
    await photos(page, { mockPresets: 3 });
    await tile(page, "Cool morning").focus();
    await expect(tile(page, "Cool morning")).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(notice(page)).toHaveText("Applied Cool morning to DSC04204.");
  });
});
