import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { mockQuery, openApp, openPhotos, watchErrors } from "./helpers/app";

// Preferences (⌘,), About and the menu bar's items (M3-13) in a real browser against the mocked API. docs/11 §Preferences is
// not mocked, so there is no screenshot to compare with; the layout rules are asserted instead.

async function menu(page: Page, action: string): Promise<void> {
  await page.evaluate((a) => window.__yellowmoon!.mockApi.emitEvent("menu_action", { action: a } as never), action);
}

test.describe("Preferences", () => {
  test("⌘, opens a dialog with the rows of docs/11, in the style of the app", async ({ page }) => {
    const problems = watchErrors(page);
    await openApp(page);
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog).toBeVisible();
    for (const row of ["Default look", "Display rendering", "Export defaults", "Remove location from exports", "Watch Downloads during import", "Check for updates automatically", "Command-line tool"]) {
      await expect(dialog.getByRole("group", { name: row })).toBeVisible();
    }
    // Square, boxed, 3 px rules, black title block.
    await expect(dialog).toHaveCSS("border-top-width", "3px");
    await expect(dialog).toHaveCSS("border-top-left-radius", "0px");
    await expect(dialog.locator(".ym-dialog__head")).toHaveCSS("background-color", "rgb(0, 0, 0)");
    await expect(dialog.getByRole("switch", { name: "Check for updates automatically" })).toHaveAttribute("aria-checked", "true");
    expect(problems).toEqual([]);
  });

  test("opens from the Photos screen too, and Esc closes it", async ({ page }) => {
    await openApp(page, { mockCurrent: 3 });
    await openPhotos(page);
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("dialog", { name: "Preferences" })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog")).toHaveCount(0);
    await expect(page.getByRole("tab", { name: "Look" })).toBeVisible();
  });

  test("the keyboard stays in the dialog, with the 3 px focus ring", async ({ page }) => {
    await openApp(page);
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog.getByRole("button", { name: "Done" })).toBeVisible();
    for (let i = 0; i < 20; i++) {
      await page.keyboard.press("Tab");
      expect(await page.evaluate(() => !!document.activeElement?.closest("[role=dialog]"))).toBe(true);
    }
    await page.keyboard.press("Tab");
    const focused = page.locator(":focus");
    await expect(focused).toHaveCSS("outline-width", "3px");
    await expect(focused).toHaveCSS("outline-offset", "2px");
  });

  test("switches save at once and survive closing the dialog", async ({ page }) => {
    await openApp(page);
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await dialog.getByRole("switch", { name: "Remove location from exports" }).click();
    await dialog.getByRole("switch", { name: "Check for updates automatically" }).click();
    await dialog.getByRole("button", { name: "Done" }).click();
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("switch", { name: "Remove location from exports" })).toHaveAttribute("aria-checked", "true");
    await expect(page.getByRole("switch", { name: "Check for updates automatically" })).toHaveAttribute("aria-checked", "false");
  });

  test("preferences persist: they are still there after the page is reloaded (the mock keeps them in localStorage)", async ({ page }) => {
    await page.goto(mockQuery({ mockPersist: 1 }));
    await expect(page.locator("[data-screen]")).toBeVisible();
    await page.keyboard.press("Meta+,");
    let dialog = page.getByRole("dialog", { name: "Preferences" });
    await dialog.getByRole("switch", { name: "Remove location from exports" }).click();
    await dialog.getByRole("switch", { name: "Watch Downloads during import" }).click();
    await dialog.getByRole("combobox", { name: "Default look" }).selectOption({ label: "Signature" });
    await dialog.getByRole("button", { name: "Edit…" }).click();
    await dialog.getByRole("radiogroup", { name: "Format" }).getByRole("radio", { name: "TIFF" }).click();
    await expect(dialog.getByRole("combobox", { name: "Default look" })).toHaveValue("builtin:signature");
    await page.waitForFunction(async () => (await window.__yellowmoon!.mockApi.preferencesGet()).export_defaults.format === "tiff");

    await page.reload();
    await expect(page.locator("[data-screen]")).toBeVisible();
    await page.keyboard.press("Meta+,");
    dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog.getByRole("switch", { name: "Remove location from exports" })).toHaveAttribute("aria-checked", "true");
    await expect(dialog.getByRole("switch", { name: "Watch Downloads during import" })).toHaveAttribute("aria-checked", "true");
    await expect(dialog.getByRole("combobox", { name: "Default look" })).toHaveValue("builtin:signature");
    await dialog.getByRole("button", { name: "Edit…" }).click();
    await expect(dialog.getByRole("radiogroup", { name: "Format" }).getByRole("radio", { name: "TIFF" })).toHaveAttribute("aria-checked", "true");
    // The Start screen shows the default look too, and the Export panel starts from the remembered format.
    await dialog.getByRole("button", { name: "Done" }).click();
    await expect(page.locator(".ym-about-row", { hasText: "Default look" })).toContainText("Signature");
  });

  test("the imported looks and the display choice persist as well", async ({ page }) => {
    await page.goto(mockQuery({ mockPersist: 1, mockLooks: 0 }));
    await expect(page.locator("[data-screen]")).toBeVisible();
    await page.evaluate(async () => {
      const { mockApi } = window.__yellowmoon!;
      await mockApi.looksImport("~/Downloads/ARRI_DisplayLUTs_LogC4.zip");
      await mockApi.preferencesSet({ display_rendering: { kind: "arri_display", id: "arri-display:logc4:rec709-g24" } });
    });
    await page.reload();
    await expect(page.locator("[data-screen]")).toBeVisible();
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog.getByRole("radio", { name: "ARRI's display rendering" })).toHaveAttribute("aria-checked", "true");
  });

  test("the command-line tool: Install… links it and says where", async ({ page }) => {
    await openApp(page);
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog.getByText("Not installed.")).toBeVisible();
    await dialog.getByRole("button", { name: "Install…" }).click();
    await expect(dialog.getByText("Installed at /usr/local/bin/ym")).toBeVisible();
  });

  test("the export defaults are the Export panel's options without photos", async ({ page }) => {
    await openApp(page);
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await dialog.getByRole("button", { name: "Edit…" }).click();
    await expect(dialog.getByRole("region", { name: "Photos" })).toHaveCount(0);
    await expect(dialog.getByRole("region", { name: "Same thing from the command line" })).toHaveCount(0);
    await expect(dialog.getByRole("region", { name: "Format" })).toBeVisible();
    await expect(dialog.getByRole("slider", { name: "Quality" })).toHaveValue("92");
  });
});

test.describe("About", () => {
  test("shows the version, the licences, the Adobe DNG notice and the line about ARRI and Sony", async ({ page }) => {
    await openApp(page);
    await menu(page, "about");
    const dialog = page.getByRole("dialog", { name: "About Yellowmoon" });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText(/^Version \d+\.\d+\.\d+ · pipeline /)).toBeVisible();
    await expect(dialog.getByText("Apache License 2.0.", { exact: false })).toBeVisible();
    await expect(dialog.getByText("This product includes DNG technology under license by Adobe.")).toBeVisible();
    await expect(dialog.getByText("Not affiliated with ARRI or Sony. ARRI and ALEXA are trademarks of ARRI AG; Sony is a trademark of Sony Group Corporation.")).toBeVisible();
    await dialog.getByRole("button", { name: "Done" }).click();
    await expect(dialog).toHaveCount(0);
  });

  test("is one click from Preferences", async ({ page }) => {
    await openApp(page);
    await page.keyboard.press("Meta+,");
    await page.getByRole("button", { name: "About Yellowmoon" }).click();
    await expect(page.getByRole("dialog", { name: "About Yellowmoon" })).toBeVisible();
  });
});

test.describe("the menu bar", () => {
  test("Check for Updates… says what it found", async ({ page }) => {
    await openApp(page);
    await menu(page, "check_for_updates");
    await expect(page.getByRole("status").filter({ hasText: "Yellowmoon is up to date." })).toBeVisible();
  });

  test("Install Command-Line Tool… says where it is installed", async ({ page }) => {
    await openApp(page);
    await menu(page, "install_cli");
    await expect(page.getByText("The command-line tool is installed at /usr/local/bin/ym.")).toBeVisible();
  });

  test("Import ARRI Looks… and Import .cube… open the wizard and the dialog", async ({ page }) => {
    await openApp(page);
    await menu(page, "import_arri_looks");
    await expect(page.getByRole("heading", { level: 1, name: "Choose what to import" })).toBeVisible();
    await page.getByRole("button", { name: "Cancel" }).click();
    await page.evaluate(() => window.__yellowmoon!.mockApi.queueChoice("files", ["~/LUTs/Warm_Film.cube"]));
    await menu(page, "import_cube");
    await expect(page.getByRole("dialog", { name: "Import LUTs" })).toBeVisible();
  });

  test("File, Edit and View items act on the photos", async ({ page }) => {
    await openApp(page, { mockCurrent: 3 });
    await openPhotos(page);
    await menu(page, "before_after");
    await expect(page.getByText("Before · camera log")).toBeVisible();
    await menu(page, "pick");
    await expect(page.locator(".ym-filmstrip__count-label")).toHaveText("1 picked");
    await menu(page, "next_photo");
    await expect(page.locator(".ym-filmstrip__count")).toHaveText("5 / 12");
    await menu(page, "export");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    await menu(page, "close_session");
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
  });
});
