import { expect, test } from "@playwright/test";
import { dropFiles, failNext, openApp, showUpdateReady, watchErrors } from "./helpers/app";

// Start screen (task M3-05) in a real browser against the mocked API: layout, the whole-window drop,
// every state of the drop zone, and the way into Photos. Screenshot comparisons are in screenshots.spec.ts.

test.describe("idle", () => {
  test("lays the screen out as the mockup does at 1440 × 900", async ({ page }) => {
    const problems = watchErrors(page);
    await openApp(page, { mockLooks: 0 });
    await expect(page).toHaveTitle("Yellowmoon");
    const geometry = await page.evaluate(() => {
      const box = (selector: string) => {
        const r = (document.querySelector(selector) as HTMLElement).getBoundingClientRect();
        return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) };
      };
      return { header: box(".ym-header"), about: box(".ym-about"), drop: box(".ym-drop"), steps: box(".ym-steps"), brand: box(".ym-brand") };
    });
    // 75 px header, 124 px steps, two equal halves between them; the folder cell is 300 px wide.
    expect(geometry.header).toMatchObject({ y: 0, h: 75 });
    expect(geometry.steps).toMatchObject({ y: 776, h: 124 });
    expect(geometry.about).toMatchObject({ x: 0, y: 75, w: 720, h: 701 });
    expect(geometry.drop).toMatchObject({ x: 720, y: 75, w: 720, h: 701 });
    expect(geometry.brand.w).toBe(300);
    expect(problems).toEqual([]);
  });

  test("shows the copy of the mockup", async ({ page }) => {
    await openApp(page, { mockLooks: 0 });
    await expect(page.getByText("Drop ARW files to start")).toBeVisible();
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
    await expect(page.getByText("Every photo is developed automatically: exposure, white balance, noise and cinema colour.")).toBeVisible();
    await expect(page.getByText("Any Sony camera · every kind of ARW")).toBeVisible();
    await expect(page.getByText("Not imported yet")).toBeVisible();
    await expect(page.getByRole("button", { name: "Import →" })).toBeVisible();
    await expect(page.getByText("Drop files or a folder")).toBeVisible();
    await expect(page.getByText("ARW only, as many as you like")).toBeVisible();
    for (const text of ["Open", "Adjust, if you want", "Export"]) await expect(page.getByRole("region", { name: "How it works" }).getByText(text, { exact: true })).toBeVisible();
    // Export is there but disabled until photos are open.
    await expect(page.getByRole("button", { name: /Export/ })).toHaveAttribute("aria-disabled", "true");
  });

  test("draws the drop zone as a glasscn Card with a 2 px dashed rim, the header with a 1 px rim, and keeps room for the traffic lights", async ({ page }) => {
    await openApp(page);
    const zone = page.getByRole("group", { name: "Drop files or a folder" });
    await expect(zone).toHaveAttribute("data-slot", "card");
    await expect(zone).toHaveCSS("border-top-style", "dashed");
    await expect(zone).toHaveCSS("border-top-width", "2px");
    await expect(page.locator(".ym-header")).toHaveCSS("border-bottom-width", "1px");
    await expect(page.locator(".ym-brand")).toHaveCSS("padding-left", "84px");
  });

  test("shows the update cell in the header when there is news", async ({ page }) => {
    await openApp(page);
    await expect(page.getByText("Update ready")).toHaveCount(0);
    await showUpdateReady(page);
    await expect(page.locator(".ym-update-pill")).toBeVisible();
  });

  test("the default look can be changed from the Start screen", async ({ page }) => {
    await openApp(page);
    const row = page.locator(".ym-about-row", { hasText: "Default look" });
    await expect(row).toContainText("Neutral");
    await row.getByRole("button", { name: "Change" }).click();
    await page.getByRole("menuitemradio", { name: "Signature" }).click();
    await expect(row).toContainText("Signature");
  });

  test("the keyboard reaches every action in tab order, with the 3 px focus ring", async ({ page }) => {
    await openApp(page);
    await page.keyboard.press("Tab");
    await expect(page.getByRole("button", { name: "Open files ⌘O" })).toBeFocused();
    await expect(page.getByRole("button", { name: "Open files ⌘O" })).toHaveCSS("outline-width", "3px");
    await expect(page.getByRole("button", { name: "Open files ⌘O" })).toHaveCSS("outline-offset", "2px");
    await page.keyboard.press("Tab");
    await expect(page.getByRole("button", { name: "Open folder ⇧⌘O" })).toBeFocused();
  });
});

test.describe("opening photos", () => {
  test("Open files develops twelve photos, lists them and opens the first photo", async ({ page }) => {
    const problems = watchErrors(page);
    await openApp(page);
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    await expect(page.getByText("Everything is developed and ready to export.")).toBeVisible();
    await expect(page.getByRole("list", { name: "Photos being developed" }).getByRole("listitem")).toHaveCount(12);
    await expect(page.getByText("Lisbon")).toBeVisible();
    await expect(page.getByText("2026-10-04 · 12 photos")).toBeVisible();
    await page.getByRole("button", { name: /Open photos/ }).click();
    await expect(page.locator("[data-screen=photos]")).toBeVisible();
    expect(problems).toEqual([]);
  });

  test("shows development under way: Ready, Developing and Waiting, and Open photos works at once", async ({ page }) => {
    await openApp(page, { mockHold: 5.4 });
    await page.getByRole("button", { name: "Open files…" }).click();
    const list = page.getByRole("list", { name: "Photos being developed" });
    await expect(list.getByText("Ready", { exact: true })).toHaveCount(5);
    await expect(list.getByText("Developing", { exact: true })).toHaveCount(1);
    await expect(list.getByText("Waiting", { exact: true })).toHaveCount(6);
    await expect(page.getByRole("heading", { level: 2, name: "Developing 12 photos" })).toBeVisible();
    await expect(page.getByRole("progressbar", { name: "Developing photos" })).toHaveAttribute("aria-valuenow", "45");
    await expect(page.getByText("You can start looking while the rest finishes.")).toBeVisible();
    await page.getByRole("button", { name: /Open photos/ }).click();
    await expect(page.locator("[data-screen=photos]")).toBeVisible();
    // The photos that are still developing show as placeholders in the filmstrip.
    await expect(page.locator(".ym-filmstrip__thumb[data-status=developing]")).not.toHaveCount(0);
  });

  test("the progress bar fills as photos finish", async ({ page }) => {
    // At the speed of the mockups: about four seconds for twelve photos.
    await page.goto("/");
    await page.evaluate(() => document.fonts.ready);
    await page.getByRole("button", { name: "Open files…" }).click();
    const bar = page.getByRole("progressbar", { name: "Developing photos" });
    await expect(bar).toHaveAttribute("aria-valuenow", /^(?!0$)\d+$/);
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible({ timeout: 10_000 });
    await expect(bar).toHaveAttribute("aria-valuenow", "100");
  });

  test("a file that can't be developed is Skipped, with the reason, and the summary counts it", async ({ page }) => {
    await openApp(page, { mockSkipped: "DSC04207:This file is from a camera mode we can't read yet." });
    await page.getByRole("button", { name: "Open files…" }).click();
    const list = page.getByRole("list", { name: "Photos being developed" });
    await expect(page.getByRole("heading", { level: 2, name: "11 photos ready" })).toBeVisible();
    await expect(list.getByText("Skipped", { exact: true })).toHaveCount(1);
    await expect(list.getByText("This file is from a camera mode we can't read yet.")).toBeVisible();
    await expect(page.getByText("1 file skipped. Everything else is developed and ready to export.")).toBeVisible();
    await expect(list.getByText("Skipped", { exact: true })).toHaveAttribute("data-variant", "skipped");
  });

  test("Start over returns to the drop zone", async ({ page }) => {
    await openApp(page);
    await page.getByRole("button", { name: "Open files…" }).click();
    await page.getByRole("button", { name: "Start over" }).click();
    await expect(page.getByRole("group", { name: "Drop files or a folder" })).toBeVisible();
    await expect(page.getByText("Drop ARW files to start")).toBeVisible();
  });

  test("⌘O opens files and ⇧⌘O a folder", async ({ page }) => {
    await openApp(page);
    await page.keyboard.press("Meta+o");
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    await page.getByRole("button", { name: "Start over" }).click();
    await page.keyboard.press("Meta+Shift+O");
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
  });
});

test.describe("dropping files on the window", () => {
  test("ARW files dropped anywhere start the session, and the zone lights up while the drag is over", async ({ page }) => {
    await openApp(page);
    await page.evaluate(() => {
      const data = new DataTransfer();
      data.items.add(new File([new Uint8Array(4)], "DSC04201.ARW"));
      window.dispatchEvent(new DragEvent("dragenter", { dataTransfer: data, bubbles: true, cancelable: true }));
    });
    await expect(page.getByRole("group", { name: "Drop files or a folder" })).toHaveAttribute("data-active", "true");
    // The zone lights up: a solid ring in the accent in place of the dashed rim.
    await expect(page.getByRole("group", { name: "Drop files or a folder" })).toHaveCSS("border-top-style", "solid");
    await expect(page.getByRole("group", { name: "Drop files or a folder" })).toHaveCSS("border-top-color", "rgb(255, 194, 14)");
    await dropFiles(page, ["DSC04201.ARW", "DSC04202.ARW"]);
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
  });

  test("a drop of non-ARW files says so and opens nothing", async ({ page }) => {
    await openApp(page);
    await dropFiles(page, ["holiday.jpg", "notes.txt"]);
    await expect(page.getByRole("alert")).toHaveText("No ARW files in what you dropped.");
    await expect(page.getByRole("group", { name: "Drop files or a folder" })).toBeVisible();
    // Non-ARW files in a mixed drop are ignored and not listed.
    await dropFiles(page, ["holiday.jpg", "DSC04201.ARW"]);
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    await expect(page.getByText("holiday.jpg")).toHaveCount(0);
  });
});

test.describe("errors", () => {
  test("an error from the core shows in a bar that can be dismissed", async ({ page }) => {
    await openApp(page);
    await failNext(page, "openDialog", { scope: "open", message: "That folder can't be read. Choose another." });
    await page.getByRole("button", { name: "Open folder…" }).click();
    const bar = page.getByRole("alert");
    await expect(bar).toHaveText(/That folder can't be read\. Choose another\./);
    await bar.getByRole("button", { name: "Dismiss" }).click();
    await expect(bar).toHaveCount(0);
  });
});
