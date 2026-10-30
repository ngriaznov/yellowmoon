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
    for (const row of ["Default look", "Export defaults", "Remove location from exports", "Watch Downloads during import", "Check for updates automatically", "Command-line tool"]) {
      await expect(dialog.getByRole("group", { name: row })).toBeVisible();
    }
    // A glasscn dialog: a glass pane with a 1 px rim and rounded corners, titled by an h2, over a dimmed window.
    await expect(dialog).toHaveAttribute("data-slot", "dialog-content");
    await expect(dialog).toHaveCSS("border-top-width", "1px");
    expect(await dialog.evaluate((el) => parseFloat(getComputedStyle(el).borderTopLeftRadius))).toBeGreaterThan(8);
    await expect(dialog.getByRole("heading", { level: 2, name: "Preferences" })).toBeVisible();
    await expect(page.locator('[data-slot="dialog-overlay"]')).toBeVisible();
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

  test("Import LUTs keeps the keyboard inside while its question is unanswered, and Esc still closes it", async ({ page }) => {
    await openApp(page);
    await page.evaluate(() => window.__yellowmoon!.mockApi.queueChoice("files", ["~/LUTs/Warm_Film.cube"]));
    await menu(page, "import_cube");
    const dialog = page.getByRole("dialog", { name: "Import LUTs" });
    await expect(dialog).toBeVisible();
    // Nothing says what Warm_Film.cube expects, so Import is locked (aria-disabled, still in the tab order).
    await expect(dialog.getByRole("button", { name: "Import" })).toHaveAttribute("aria-disabled", "true");
    const inside = (): Promise<boolean> => page.evaluate(() => !!document.activeElement?.closest("[role=dialog]"));
    expect(await inside()).toBe(true);
    for (let i = 0; i < 12; i++) {
      await page.keyboard.press("Tab");
      expect(await inside()).toBe(true);
    }
    for (let i = 0; i < 12; i++) {
      await page.keyboard.press("Shift+Tab");
      expect(await inside()).toBe(true);
    }
    // The first Tab used to leave the dialog, and Esc did nothing from there. Focus on the locked Import opens no tooltip (a title is
    // shown on hover only: an open tooltip is the topmost layer and would take the first Esc), so one Esc closes the dialog.
    await dialog.getByRole("button", { name: "Import" }).focus();
    await expect(dialog.getByRole("button", { name: "Import" })).toBeFocused();
    await expect(page.getByRole("tooltip")).toHaveCount(0);
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog")).toHaveCount(0);
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
    await dialog.getByRole("combobox", { name: "Default look" }).click();
    await page.getByRole("option", { name: "Signature" }).click();
    await dialog.getByRole("button", { name: "Edit…" }).click();
    await dialog.getByRole("radiogroup", { name: "Format" }).getByRole("radio", { name: "TIFF" }).click();
    await expect(dialog.getByRole("combobox", { name: "Default look" })).toHaveText("Signature");
    await page.waitForFunction(async () => (await window.__yellowmoon!.mockApi.preferencesGet()).export_defaults.format === "tiff");

    await page.reload();
    await expect(page.locator("[data-screen]")).toBeVisible();
    await page.keyboard.press("Meta+,");
    dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog.getByRole("switch", { name: "Remove location from exports" })).toHaveAttribute("aria-checked", "true");
    await expect(dialog.getByRole("switch", { name: "Watch Downloads during import" })).toHaveAttribute("aria-checked", "true");
    await expect(dialog.getByRole("combobox", { name: "Default look" })).toHaveText("Signature");
    await dialog.getByRole("button", { name: "Edit…" }).click();
    await expect(dialog.getByRole("radiogroup", { name: "Format" }).getByRole("radio", { name: "TIFF" })).toHaveAttribute("aria-checked", "true");
    // The Start screen shows the default look too, and the Export panel starts from the remembered format.
    await dialog.getByRole("button", { name: "Done" }).click();
    await expect(page.locator(".ym-about-row", { hasText: "Default look" })).toContainText("Signature");
  });

  test("the imported looks persist as well, and a display choice kept by an earlier version is ignored without a word", async ({ page }) => {
    await page.goto(mockQuery({ mockPersist: 1, mockLooks: 0 }));
    await expect(page.locator("[data-screen]")).toBeVisible();
    await page.evaluate(async () => {
      await window.__yellowmoon!.mockApi.looksImport("~/Downloads/ARRI_LookLibrary_LogC4.zip");
      // What an earlier version left in the page's storage: the choice of ARRI's display LUT, and the display package it imported.
      const key = "yellowmoon.mock.state";
      const stored = JSON.parse(localStorage.getItem(key) ?? "{}");
      stored.preferences = { ...stored.preferences, display_rendering: { kind: "arri_display", id: "arri-display:logc4:rec709-g24" } };
      stored.importedPackages = [...stored.importedPackages, "display_luts_logc4"];
      localStorage.setItem(key, JSON.stringify(stored));
    });
    await page.reload();
    await expect(page.locator("[data-screen]")).toBeVisible();
    await expect(page.getByRole("alert")).toHaveCount(0);
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog.getByRole("group", { name: "Default look" })).toBeVisible();
    await expect(dialog.getByRole("group", { name: "Display rendering" })).toHaveCount(0);
    await expect(dialog.getByRole("radio", { name: /display rendering/i })).toHaveCount(0);
    const result = await page.evaluate(async () => {
      const { mockApi } = window.__yellowmoon!;
      return { preferences: await mockApi.preferencesGet(), arri: (await mockApi.looksList()).filter((l) => l.group.kind === "arri").length };
    });
    expect(result.preferences).not.toHaveProperty("display_rendering");
    expect(result.arri).toBe(87);
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
    await expect(dialog.getByRole("slider", { name: "Quality" })).toHaveAttribute("aria-valuenow", "92");
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
    await expect(page.getByText("Before · camera default")).toBeVisible();
    await menu(page, "pick");
    await expect(page.locator(".ym-filmstrip__caption")).toHaveText("4 of 12 · 1 picked");
    await menu(page, "next_photo");
    await expect(page.locator(".ym-filmstrip__caption")).toHaveText("5 of 12 · 1 picked");
    await menu(page, "export");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    await menu(page, "close_session");
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
  });
});
