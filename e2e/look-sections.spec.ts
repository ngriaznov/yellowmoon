import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { openApp, openPhotos } from "./helpers/app";

// The Look tab's sections (owner's decision of 2026-10-16, docs/11-ui.md §Look tab): one section per look library, each a disclosure
// button that folds away. The mock has looks in every library with `mockLibraries=all`: the same titles recur from library to library
// as they do in ARRI's, so the sections and the qualified names can be seen in a real browser.

const ALL = { mockCurrent: 3, mockPicked: "2,3,4", mockLibraries: "all" } as const;

async function photos(page: Page): Promise<void> {
  await openApp(page, ALL);
  await openPhotos(page);
}

const card = (page: Page): Locator => page.getByRole("region", { name: "Look" });
/** The header button of a section, by the start of its accessible name ("ARRI looks · LogC3, "). */
const header = (page: Page, start: string): Locator => page.getByRole("button", { name: new RegExp(`^${start.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`) });
/** The tiles inside a section. */
const tilesOf = (page: Page, start: string): Locator => page.locator(`[data-library]`).filter({ has: header(page, start) }).locator(".ym-look-tile");
const photoChip = (page: Page): Locator => page.getByRole("main", { name: "Photo" });

test.describe("the sections of the Look tab", () => {
  test("are one per library, in order, with their labels and counts", async ({ page }) => {
    await photos(page);
    const headers = card(page).locator("button[aria-expanded]");
    await expect(headers).toHaveCount(6);
    expect(await headers.evaluateAll((els) => els.map((el) => el.getAttribute("aria-expanded")))).toEqual(["true", "true", "true", "true", "true", "true"]);
    expect(await headers.evaluateAll((els) => els.map((el) => el.getAttribute("aria-label")))).toEqual([
      "Built in, 2 looks, collapse",
      "ARRI looks · LogC4, 7 looks, collapse",
      "ARRI looks · LogC3, 5 looks, collapse",
      "ARRI looks · LogC3 to Rec.709, 4 looks, collapse",
      "ARRI looks · LogC3 to Rec.709 · ALF2, 6 looks, collapse",
      "Your LUTs, 5 looks, collapse",
    ]);
    expect(await headers.evaluateAll((els) => els.map((el) => el.textContent))).toEqual([
      "Built in2",
      "ARRI looks · LogC47",
      "ARRI looks · LogC35",
      "ARRI looks · LogC3 to Rec.7094",
      "ARRI looks · LogC3 to Rec.709 · ALF26",
      "Your LUTs5",
    ]);
    // Down the panel in the same order.
    const ys = await headers.evaluateAll((els) => els.map((el) => el.getBoundingClientRect().top + window.scrollY));
    expect(ys).toEqual([...ys].sort((a, b) => a - b));
  });

  test("draw each header across the whole row with its count at the right end, with or without a search", async ({ page }) => {
    // A button whose title comes and goes sits in the span of its tooltip; the span has to be as wide as the row, or the header
    // shrinks to its text and the count moves next to the label.
    await photos(page);
    const rows = async (): Promise<Array<{ header: number; group: number; gap: number }>> =>
      card(page)
        .locator("button[aria-expanded]")
        .evaluateAll((els) =>
          els.map((el) => {
            const group = (el.closest("[data-library]") as HTMLElement).getBoundingClientRect();
            const count = (el.querySelector(".ym-look-group__count") as HTMLElement).getBoundingClientRect();
            const box = el.getBoundingClientRect();
            return { header: Math.round(box.width), group: Math.round(group.width), gap: Math.round(box.right - count.right) };
          }),
        );
    for (const stage of ["no search", "a search"]) {
      if (stage === "a search") await page.getByRole("searchbox", { name: "Search looks" }).fill("film");
      const found = await rows();
      expect(found.length, stage).toBeGreaterThan(0);
      for (const row of found) {
        expect(row.header, `${stage}: header width`).toBe(row.group);
        expect(row.gap, `${stage}: space right of the count`).toBeLessThan(12);
      }
    }
  });

  test("keep the old call to action while no ARRI looks are imported", async ({ page }) => {
    await openApp(page, { mockCurrent: 3, mockLooks: 0 });
    await openPhotos(page);
    await expect(card(page).locator("button[aria-expanded]")).toHaveCount(1);
    await expect(page.getByText("The free ARRI Look Library is not imported yet.")).toBeVisible();
    await expect(card(page).getByText("ARRI looks", { exact: true })).toBeVisible();
    await expect(card(page).getByRole("button", { name: "Import ARRI looks →" })).toBeVisible();
  });

  test("fold away and open again on a click, and name the action", async ({ page }) => {
    await photos(page);
    const logc3 = header(page, "ARRI looks · LogC3, ");
    await expect(tilesOf(page, "ARRI looks · LogC3, ")).toHaveCount(5);

    await logc3.click();
    await expect(logc3).toHaveAttribute("aria-expanded", "false");
    await expect(logc3).toHaveAccessibleName("ARRI looks · LogC3, 5 looks, expand");
    await expect(tilesOf(page, "ARRI looks · LogC3, ")).toHaveCount(0);
    // The neighbours stay as they were.
    await expect(tilesOf(page, "ARRI looks · LogC4, ")).toHaveCount(7);
    await expect(tilesOf(page, "ARRI looks · LogC3 to Rec.709, ")).toHaveCount(4);

    await logc3.click();
    await expect(logc3).toHaveAttribute("aria-expanded", "true");
    await expect(logc3).toHaveAccessibleName("ARRI looks · LogC3, 5 looks, collapse");
    await expect(tilesOf(page, "ARRI looks · LogC3, ")).toHaveCount(5);
  });

  test("fold from the keyboard and keep their state when the tab is left", async ({ page }) => {
    await photos(page);
    const mine = header(page, "Your LUTs, ");
    await mine.focus();
    await page.keyboard.press("Enter");
    await expect(mine).toHaveAttribute("aria-expanded", "false");
    await page.getByRole("tablist", { name: "Tool tabs" }).getByRole("tab", { name: "Light" }).click();
    await page.getByRole("tablist", { name: "Tool tabs" }).getByRole("tab", { name: "Look" }).click();
    await expect(header(page, "Your LUTs, ")).toHaveAttribute("aria-expanded", "false");
    await expect(tilesOf(page, "Your LUTs, ")).toHaveCount(0);
    await header(page, "Your LUTs, ").press("Space");
    await expect(header(page, "Your LUTs, ")).toHaveAttribute("aria-expanded", "true");
  });

  test("open for a search, whatever was folded, and go back as they were", async ({ page }) => {
    await photos(page);
    await header(page, "ARRI looks · LogC3, ").click();
    await header(page, "Your LUTs, ").click();
    await expect(tilesOf(page, "ARRI looks · LogC3, ")).toHaveCount(0);

    await page.getByRole("searchbox", { name: "Search looks" }).fill("film");
    const open = card(page).locator("button[aria-expanded]");
    await expect(open).toHaveCount(5);
    expect(await open.evaluateAll((els) => els.map((el) => el.getAttribute("aria-expanded")))).toEqual(["true", "true", "true", "true", "true"]);
    expect(await open.evaluateAll((els) => els.map((el) => el.querySelector(".ym-look-group__count")?.textContent))).toEqual(["2 of 7", "2 of 5", "2 of 4", "3 of 6", "2 of 5"]);
    await expect(tilesOf(page, "ARRI looks · LogC3, ")).toHaveCount(2);
    await expect(tilesOf(page, "Your LUTs, ")).toHaveCount(2);
    // Built in has no match, so it is not listed.
    await expect(card(page).getByText("Built in", { exact: true })).toHaveCount(0);

    await page.getByRole("searchbox", { name: "Search looks" }).fill("");
    await expect(header(page, "ARRI looks · LogC3, ")).toHaveAttribute("aria-expanded", "false");
    await expect(header(page, "Your LUTs, ")).toHaveAttribute("aria-expanded", "false");
    await expect(header(page, "ARRI looks · LogC4, ")).toHaveAttribute("aria-expanded", "true");
    await expect(card(page).getByText("Built in", { exact: true })).toBeVisible();
  });

  test("show a small square on a folded section that holds the current look", async ({ page }) => {
    await photos(page);
    await tilesOf(page, "ARRI looks · LogC3, ").filter({ hasText: "Film cool" }).click();
    const logc3 = header(page, "ARRI looks · LogC3, ");
    await expect(logc3.locator(".ym-look-group__marker")).toHaveCount(0);
    await logc3.click();
    await expect(logc3.locator(".ym-look-group__marker")).toHaveCount(1);
    await expect(logc3).toHaveAccessibleName("ARRI looks · LogC3, 5 looks, holds the current look, expand");
    // The square is a filled box a few pixels wide.
    const box = (await logc3.locator(".ym-look-group__marker").boundingBox())!;
    expect(box.width).toBeGreaterThanOrEqual(6);
    expect(box.width).toBeLessThanOrEqual(10);
    expect(Math.abs(box.width - box.height)).toBeLessThan(0.5);
    await logc3.click();
    await expect(logc3.locator(".ym-look-group__marker")).toHaveCount(0);
  });

  test("keep the numbers of the whole list: keys 1 to 9 pick a look in a folded section", async ({ page }) => {
    await photos(page);
    await header(page, "ARRI looks · LogC4, ").click();
    await page.keyboard.press("3");
    await expect(photoChip(page).getByText("Film warm · LogC4 · 100%")).toBeVisible();
    await page.keyboard.press("2");
    await expect(photoChip(page).getByText("Signature · 100%")).toBeVisible();
  });

  test("tell what a look is in a tooltip", async ({ page }) => {
    await photos(page);
    const title = (start: string, name: RegExp): Promise<string | null> => tilesOf(page, start).filter({ hasText: name }).first().getAttribute("title");
    expect(await title("ARRI looks · LogC4, ", /Film warm/)).toBe("ARRI looks · LogC4 · 3D LUT, 65 points");
    expect(await title("ARRI looks · LogC3, ", /Film warm/)).toBe("ARRI looks · LogC3 · 3D LUT, 33 points");
    expect(await title("ARRI looks · LogC3 to Rec.709 · ALF2, ", /Film warm · Low/)).toBe("ARRI looks · LogC3 to Rec.709 · ALF2 · ALF2 look file, 33 points");
    expect(await title("Your LUTs, ", /Warm Pull/)).toBe("Your LUTs · ALF4 look file, 33 points");
    expect(await title("Your LUTs, ", /Custom Look/)).toBe("Your LUTs · ALF4C look file, 33 points");
    expect(await title("Built in, ", /Neutral/)).toBeNull();
  });
});

test.describe("a name that two libraries share", () => {
  test("says which library on the look chip of the photo", async ({ page }) => {
    await photos(page);
    await tilesOf(page, "ARRI looks · LogC3, ").filter({ hasText: "Film warm" }).click();
    await expect(photoChip(page).getByText("Film warm · LogC3 · 100%")).toBeVisible();
    await tilesOf(page, "Your LUTs, ").filter({ hasText: "Film cool" }).click();
    await expect(photoChip(page).getByText("Film cool · Your LUTs · 100%")).toBeVisible();
    // A title that is only in one library keeps its plain name.
    await page.keyboard.press("9");
    await expect(photoChip(page).getByText("Monochrome · 100%")).toBeVisible();
  });

  test("says which library in the default look choices", async ({ page }) => {
    await photos(page);
    await page.keyboard.press("Meta+,");
    await page.getByRole("dialog", { name: "Preferences" }).getByRole("combobox", { name: "Default look" }).click();
    const labels = await page.getByRole("option").allTextContents();
    expect(labels).toContain("Neutral");
    expect(labels).toContain("Film warm · LogC4");
    expect(labels).toContain("Film warm · LogC3 to Rec.709");
    expect(labels).toContain("Film cool · Your LUTs");
    expect(labels).not.toContain("Film warm");
  });
});
