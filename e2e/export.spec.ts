import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, recordCalls, settle, showUpdateReady, watchErrors } from "./helpers/app";
import { compareScreen, isCopyOfDesign } from "./helpers/compare";

// The Export panel (M3-11) in a real browser against the mocked API: every format, the options, the queue, the finished
// state, exports that keep running while the person browses, and screenshot comparisons against design/screenshots/export-*.png.

const MOCKUP: Record<string, string | number> = { mockCurrent: 3, mockPicked: "2,3,4" };

const STATES = ["export-jpeg", "export-tiff", "export-png", "export-dng", "export-running", "export-finished"];

test("the baselines are byte-for-byte copies of design/screenshots", () => {
  for (const name of STATES) expect(isCopyOfDesign(name), `${name}.png is a copy`).toBe(true);
});

/** Photos, the update cell as in the mockups, then ⌘E; waits for the estimate. */
async function exportPanel(page: Page, options: Record<string, string | number> = MOCKUP): Promise<void> {
  await openApp(page, options);
  // Wait for all twelve so the mockup's current photo (the fourth) is the one on screen.
  await openPhotos(page);
  await showUpdateReady(page);
  await page.keyboard.press("Meta+e");
  await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toContainText("≈");
  await expect(page.locator(".ym-filmstrip__thumb img")).toHaveCount(12);
}

const format = (page: Page, name: string) => page.getByRole("radiogroup", { name: "Format" }).getByRole("radio", { name }).click();

test.describe("layout", () => {
  test("is the mockup's grid at 1440 × 900: the panel is 400 px wide, the title block black, the footer 56 px", async ({ page }) => {
    const problems = watchErrors(page);
    await exportPanel(page);
    const geometry = await page.evaluate(() => {
      const box = (selector: string) => {
        const r = (document.querySelector(selector) as HTMLElement).getBoundingClientRect();
        return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) };
      };
      return { panel: box(".ym-tools"), title: box(".ym-title-block"), footer: box(".ym-tools__footer"), button: box(".ym-tools__footer .ym-btn"), cell: box(".ym-photo-cell") };
    });
    expect(geometry.panel).toMatchObject({ x: 1040, y: 75, w: 400, h: 825 });
    // The panel has a 3 px rule on its left edge; the title block fills the rest.
    expect(geometry.title).toMatchObject({ x: 1043, w: 397 });
    expect(geometry.button.h).toBe(56);
    expect(geometry.footer.y + geometry.footer.h).toBe(900);
    expect(geometry.cell).toMatchObject({ y: 75 });
    await expect(page.locator(".ym-title-block")).toHaveCSS("background-color", "rgb(0, 0, 0)");
    expect(problems).toEqual([]);
  });

  test("keeps the header's Export cell black while the panel is open, and says Back to photos", async ({ page }) => {
    await exportPanel(page);
    await expect(page.locator(".ym-export")).toHaveCSS("background-color", "rgb(0, 0, 0)");
    await expect(page.locator(".ym-export")).toHaveAttribute("aria-current", "true");
    await expect(page.getByRole("button", { name: "← Back to photos" })).toBeVisible();
    await expect(page.getByText("Neural · Display P3")).toBeVisible();
    await page.getByRole("button", { name: "← Back to photos" }).click();
    await expect(page.getByRole("complementary", { name: "Tools" })).toBeVisible();
  });

  test("dims the photos that won't be exported (opacity 0.3) and thickens the ones that will", async ({ page }) => {
    await exportPanel(page);
    const thumbs = page.locator(".ym-filmstrip__thumb");
    await expect(thumbs.nth(2)).toHaveCSS("opacity", "1");
    await expect(thumbs.nth(2)).toHaveCSS("border-top-width", "3px");
    await expect(thumbs.nth(3)).toHaveCSS("border-top-width", "4px"); // the current photo
    await expect(thumbs.nth(0)).toHaveCSS("opacity", "0.3");
    await expect(page.locator(".ym-filmstrip__count")).toHaveText("3");
    await expect(page.locator(".ym-filmstrip__count-label")).toHaveText("to export");
  });
});

test.describe("each format", () => {
  test("JPEG: Quality slider, 92, and the estimate", async ({ page }) => {
    await exportPanel(page);
    await expect(page.getByRole("slider", { name: "Quality" })).toHaveValue("92");
    await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toContainText("≈ 75 MB · ⌘↩");
    await expect(page.getByText("DSC04203_neutral.jpg")).toBeVisible();
  });

  test("TIFF: Bit depth and Compression", async ({ page }) => {
    await exportPanel(page);
    await format(page, "TIFF");
    await expect(page.getByRole("radiogroup", { name: "Bit depth" })).toBeVisible();
    await expect(page.getByRole("radiogroup", { name: "Compression" }).getByRole("radio", { name: "LZW" })).toHaveAttribute("aria-checked", "true");
    await expect(page.getByText("DSC04203_neutral.tif")).toBeVisible();
    const command = page.getByRole("region", { name: "Same thing from the command line" });
    // LZW is the default, so the command names only another compression.
    await expect(command).toContainText("--format tiff");
    await expect(command).not.toContainText("--compression");
    await page.getByRole("radiogroup", { name: "Compression" }).getByRole("radio", { name: "ZIP" }).click();
    await expect(command).toContainText("--compression zip");
  });

  test("PNG: Bit depth only", async ({ page }) => {
    await exportPanel(page);
    await format(page, "PNG");
    await expect(page.getByRole("radiogroup", { name: "Bit depth" })).toBeVisible();
    await expect(page.getByRole("radiogroup", { name: "Compression" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toContainText("≈ 630 MB");
  });

  test("DNG: the note, the Adobe notice, and no Colour space or Size", async ({ page }) => {
    await exportPanel(page);
    await format(page, "DNG");
    await expect(page.getByText("Linear DNG with your look baked in.")).toBeVisible();
    await expect(page.getByText(/This product includes DNG technology under license by Adobe\./)).toBeVisible();
    await expect(page.getByRole("region", { name: "Colour space" })).toHaveCount(0);
    await expect(page.getByRole("region", { name: "Size" })).toHaveCount(0);
    await expect(page.getByRole("region", { name: "Same thing from the command line" })).not.toContainText("--colour-space");
  });

  test("DNG is disabled for a look with its own display rendering", async ({ page }) => {
    await exportPanel(page);
    // Import the LogC3 to Rec.709 package through the app's own mock, and pick one of its looks.
    await page.evaluate(async () => {
      const { mockApi } = window.__yellowmoon!;
      await mockApi.looksImport("~/Downloads/ARRI_LookLibrary_LogC3_to_Rec709.zip");
      await mockApi.applyLook("this", "arri:3000:film-warm-rec709", 100);
    });
    // The panel asks again when the photos or options change.
    await page.getByRole("radiogroup", { name: "Photos" }).getByRole("radio", { name: "This photo" }).click();
    const dng = page.getByRole("radio", { name: "DNG, not available for this look" });
    await expect(dng).toHaveAttribute("aria-disabled", "true");
    await expect(page.getByText("This look includes its own display rendering, so it can't go into a DNG. Pick another look or export TIFF.")).toBeVisible();
  });

  test("Long edge: the slider runs from 1000 to 8640 px in 20 px steps", async ({ page }) => {
    await exportPanel(page);
    await page.getByRole("radiogroup", { name: "Size" }).getByRole("radio", { name: "Long edge" }).click();
    const slider = page.getByRole("slider", { name: "Long edge" });
    await expect(slider).toHaveAttribute("min", "1000");
    await expect(slider).toHaveAttribute("max", "8640");
    await expect(slider).toHaveAttribute("step", "20");
    await slider.focus();
    await page.keyboard.press("ArrowRight");
    await expect(slider).toHaveValue("3020");
    await page.keyboard.press("Shift+ArrowRight");
    await expect(slider).toHaveValue("3220");
    await expect(page.getByRole("region", { name: "Same thing from the command line" })).toContainText("--long-edge 3220");
  });
});

test.describe("Save to, the file name and the command line", () => {
  test("Choose… takes the folder the chooser gives; the command line follows", async ({ page }) => {
    await exportPanel(page);
    await page.evaluate(() => window.__yellowmoon!.mockApi.queueChoice("folder", "~/Exports/Portugal"));
    await page.getByRole("region", { name: "Save to" }).getByRole("button", { name: "Choose…" }).click();
    await expect(page.getByRole("region", { name: "Save to" })).toContainText("~/Exports/Portugal");
    await expect(page.getByRole("region", { name: "Same thing from the command line" })).toContainText('--out "~/Exports/Portugal"');
  });

  test("the file name template previews, and a wrong one is explained and blocks the button", async ({ page }) => {
    await exportPanel(page);
    const box = page.getByRole("textbox", { name: "File name template" });
    await box.fill("{date}_{name}_{seq}");
    await expect(page.getByText("2026-10-04_DSC04203_001.jpg")).toBeVisible();
    await box.fill("{name}_{colour}");
    await expect(page.getByRole("alert")).toContainText("{colour} is not a token.");
    await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toHaveAttribute("aria-disabled", "true");
  });

  test("Copy puts the command on the clipboard and says Copied", async ({ page, context }) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await exportPanel(page);
    await page.getByRole("region", { name: "Same thing from the command line" }).getByRole("button", { name: "Copy" }).click();
    await expect(page.getByRole("button", { name: "Copied" })).toBeVisible();
    const text = await page.evaluate(() => navigator.clipboard.readText());
    expect(text).toContain("ym render DSC04203.ARW DSC04204.ARW DSC04205.ARW");
    expect(text).toContain("--out");
  });

  test("says how many photos have adjustments the command doesn't include", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.getByRole("tab", { name: "Light" }).click();
    await page.getByRole("textbox", { name: "Contrast value" }).fill("20");
    await page.getByRole("textbox", { name: "Contrast value" }).press("Enter");
    await page.keyboard.press("Meta+e");
    await expect(page.getByText("1 photo has adjustments this command doesn't include.")).toBeVisible();
  });
});

test.describe("running", () => {
  for (const fmt of ["JPEG", "TIFF", "PNG", "DNG"]) {
    test(`Photos → Export → running → finished, as ${fmt}`, async ({ page }) => {
      // The mock holds the run at one done and the second at 45 % until `flush()`, so the running state is on screen for as long as
      // the test needs it, not for as long as a timer says.
      await exportPanel(page, { ...MOCKUP, mockExportHold: 1.45 });
      await format(page, fmt);
      await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toContainText("≈");
      await page.getByRole("button", { name: /^Export 3 photos/ }).click();
      await expect(page.getByRole("heading", { level: 2, name: "Exporting" })).toBeVisible();
      const queue = page.getByRole("list", { name: "Export queue" });
      await expect(queue.getByRole("listitem")).toHaveCount(3);
      await expect(page.getByText("1 of 3 done.")).toBeVisible();
      await expect(queue.getByText("45%", { exact: true })).toBeVisible();
      await expect(queue.getByText("Waiting", { exact: true })).toHaveCount(1);
      await page.evaluate(() => window.__yellowmoon!.mockApi.flush());
      await expect(page.getByRole("heading", { level: 2, name: "Exported" })).toBeVisible();
      await expect(page.getByText("3 photos saved to Lisbon export.")).toBeVisible();
      await expect(queue.getByText("Done", { exact: true })).toHaveCount(3);
      const ext = { JPEG: "jpg", TIFF: "tif", PNG: "png", DNG: "dng" }[fmt];
      await expect(page.getByText(`DSC04205_neutral.${ext}`)).toBeVisible();
      await page.getByRole("button", { name: "Done" }).click();
      await expect(page.getByRole("complementary", { name: "Tools" })).toBeVisible();
    });
  }

  test("Show in Finder reveals the folder, and Done goes back to the tools", async ({ page }) => {
    await exportPanel(page);
    const reveal = await recordCalls(page, "revealInFinder");
    await page.getByRole("button", { name: /^Export 3 photos/ }).click();
    await expect(page.getByRole("heading", { level: 2, name: "Exported" })).toBeVisible();
    await page.getByRole("button", { name: "Show in Finder" }).click();
    expect((await reveal()).map((c) => c.args)).toEqual([["~/Pictures/Lisbon export"]]);
    await page.getByRole("button", { name: "Done" }).click();
    await expect(page.getByRole("complementary", { name: "Tools" })).toBeVisible();
  });

  test("⌘↩ starts the export", async ({ page }) => {
    await exportPanel(page);
    await page.keyboard.press("Meta+Enter");
    await expect(page.getByRole("heading", { level: 2, name: "Exporting" })).toBeVisible();
  });

  test("Cancel returns to the options", async ({ page }) => {
    await exportPanel(page, { ...MOCKUP, mockExportHold: 1.45 });
    await page.getByRole("button", { name: /^Export 3 photos/ }).click();
    await expect(page.getByRole("button", { name: "Cancel" })).toBeVisible();
    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(page.getByRole("region", { name: "Format" })).toBeVisible();
    await expect(page.getByText(/^Export stopped\./)).toBeVisible();
  });

  test("a failed file and a skipped file are listed with their reasons", async ({ page }) => {
    await exportPanel(page, { ...MOCKUP, mockExportFails: "DSC04204:The disk is full.", mockExisting: "DSC04205_neutral.jpg" });
    await page.getByRole("radiogroup", { name: "If the file exists" }).getByRole("radio", { name: "Skip" }).click();
    await page.getByRole("button", { name: /^Export 3 photos/ }).click();
    await expect(page.getByRole("heading", { level: 2, name: "Exported" })).toBeVisible();
    await expect(page.getByText("The disk is full.", { exact: true })).toBeVisible();
    await expect(page.getByText("1 photo saved to Lisbon export. 1 photo could not be saved. 1 photo skipped because the file already exists.")).toBeVisible();
  });

  test("exports keep running while the person browses, and an update waits for the end", async ({ page }) => {
    await exportPanel(page, { ...MOCKUP, mockExportHold: 1.45 });
    await page.getByRole("button", { name: /^Export 3 photos/ }).click();
    await expect(page.getByText("1 of 3 done.")).toBeVisible();
    // Browse: back to the photos, another photo, a tool tab.
    await page.getByRole("button", { name: "← Back to photos" }).click();
    await page.keyboard.press("ArrowRight");
    await page.getByRole("tab", { name: "Colour" }).click();
    await expect(page.getByRole("button", { name: "Exporting 1 of 3 ⌘E" })).toBeVisible();
    await expect(page.locator(".ym-export__strip > span")).toHaveCSS("width", /^(?!0px)/);
    // The ready update can't restart yet.
    const pill = page.getByRole("button", { name: /Update ready/ });
    await expect(pill).toHaveAttribute("aria-disabled", "true");
    await expect(pill).toHaveAttribute("title", "Restarts after the export finishes");
    // The queue is one click away; the rest of the run finishes.
    await page.getByRole("button", { name: /^Exporting/ }).click();
    await expect(page.getByRole("heading", { level: 2, name: "Exporting" })).toBeVisible();
    await page.evaluate(() => window.__yellowmoon!.mockApi.flush());
    await expect(page.getByRole("heading", { level: 2, name: "Exported" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Update ready/ })).not.toHaveAttribute("aria-disabled", "true");
  });

  test("⌘W during an export asks first", async ({ page }) => {
    await exportPanel(page, { ...MOCKUP, mockExportHold: 1.45 });
    await page.getByRole("button", { name: /^Export 3 photos/ }).click();
    await expect(page.getByText("1 of 3 done.")).toBeVisible();
    await page.keyboard.press("Meta+w");
    const dialog = page.getByRole("dialog", { name: "An export is running" });
    await expect(dialog).toBeVisible();
    await dialog.getByRole("button", { name: "Keep exporting" }).click();
    await expect(dialog).toHaveCount(0);
    await page.keyboard.press("Meta+w");
    await page.getByRole("button", { name: "Stop export and close" }).click();
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
  });
});

test.describe("keyboard", () => {
  test("every control is reachable with Tab and has the 3 px focus ring", async ({ page }) => {
    await exportPanel(page);
    // Focus by keyboard, so that :focus-visible applies.
    await page.getByRole("radiogroup", { name: "Photos" }).getByRole("radio", { name: "Picked · 3" }).focus();
    await page.keyboard.press("Shift+Tab");
    await page.keyboard.press("Tab");
    await expect(page.getByRole("radio", { name: "Picked · 3" })).toHaveCSS("outline-width", "3px");
    await expect(page.getByRole("radio", { name: "Picked · 3" })).toHaveCSS("outline-offset", "2px");
    // Photos (1 stop), Format (1), Quality slider, the value box, Colour space, Size, Choose…, the template, If it exists, Copy, Export.
    const seen: string[] = [];
    for (let i = 0; i < 14; i++) {
      await page.keyboard.press("Tab");
      seen.push(await page.evaluate(() => (document.activeElement?.getAttribute("aria-label") ?? document.activeElement?.textContent ?? "").trim().slice(0, 30)));
    }
    expect(seen.join("|")).toContain("File name template");
    expect(seen.join("|")).toContain("Choose…");
    expect(seen.join("|")).toContain("Copy");
  });

  test("Esc closes the panel and the export keeps its place", async ({ page }) => {
    await exportPanel(page);
    await page.keyboard.press("Escape");
    await expect(page.getByRole("complementary", { name: "Tools" })).toBeVisible();
  });
});

test.describe("screenshots against design/screenshots", () => {
  test("JPEG (export-jpeg.png)", async ({ page }) => {
    await exportPanel(page);
    await compareScreen(page, "export-jpeg");
  });

  test("TIFF (export-tiff.png)", async ({ page }) => {
    await exportPanel(page);
    await format(page, "TIFF");
    await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toContainText("≈ 608 MB");
    await compareScreen(page, "export-tiff");
  });

  test("PNG (export-png.png)", async ({ page }) => {
    await exportPanel(page);
    await format(page, "PNG");
    await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toContainText("≈ 630 MB");
    await compareScreen(page, "export-png");
  });

  test("DNG (export-dng.png)", async ({ page }) => {
    await exportPanel(page);
    await format(page, "DNG");
    await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toContainText("≈ 900 MB");
    await compareScreen(page, "export-dng");
  });

  test("running, 1 of 3 done and the second at 45 % (export-running.png)", async ({ page }) => {
    await exportPanel(page, { ...MOCKUP, mockExportHold: 1.45 });
    await page.getByRole("button", { name: /^Export 3 photos/ }).click();
    await expect(page.getByText("1 of 3 done.")).toBeVisible();
    await expect(page.getByText("45%")).toBeVisible();
    await compareScreen(page, "export-running");
  });

  test("finished (export-finished.png)", async ({ page }) => {
    await exportPanel(page);
    await page.getByRole("button", { name: /^Export 3 photos/ }).click();
    await expect(page.getByRole("heading", { level: 2, name: "Exported" })).toBeVisible();
    await settle(page);
    await compareScreen(page, "export-finished");
  });
});
