import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { dropFiles, openApp, openPhotos, recordCalls, showUpdateReady, watchErrors } from "./helpers/app";
import { compareScreen, isCopyOfDesign } from "./helpers/compare";

// The ARRI import wizard (M3-09) in a real browser against the mocked API: the happy path through the five steps, the failure
// paths with synthetic packages (a wrong package, an ALF4-only zip, a missing file), the Downloads watcher, the generic .cube
// question, and screenshot comparisons against design/screenshots/import-*.png.

const STATES = ["import-1-choose", "import-2-get-files", "import-3-recognise", "import-4-importing", "import-4-imported", "import-5-finish"];

test("the baselines are byte-for-byte copies of design/screenshots", () => {
  for (const name of STATES) expect(isCopyOfDesign(name), `${name}.png is a copy`).toBe(true);
});

const LL4 = "ARRI_LookLibrary_LogC4.zip";
const DL4 = "ARRI_DisplayLUTs_LogC4.zip";
const LL3 = "ARRI_LookLibrary_LogC3.zip";
const ALF4 = "ARRI_LookFiles_ALF4.zip";

/** From Start, with no ARRI looks imported, into the wizard. */
async function wizard(page: Page, options: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { mockLooks: 0, ...options });
  await page.getByRole("button", { name: "Import →" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Choose what to import" })).toBeVisible();
}

const next = (page: Page) => page.locator(".ym-wizard__footer").getByRole("button", { name: /^(Next|Import|Done)/ });
const step = (page: Page) => page.getByText(/^Import · step \d of 5$/);

/** Choose, drop the zips, Recognise. */
async function toRecognise(page: Page, ...names: string[]): Promise<void> {
  await next(page).click();
  await expect(page.getByRole("heading", { level: 1, name: "Download from ARRI" })).toBeVisible();
  await dropFiles(page, names);
  await expect(next(page)).not.toHaveAttribute("aria-disabled", "true");
  await next(page).click();
  await expect(page.getByRole("table", { name: "What arrived" })).toBeVisible();
  await expect(page.getByText("Checking", { exact: true })).toHaveCount(0);
}

test.describe("the happy path", () => {
  test("Start → Choose → Get the files → Recognise → Import → Finish → Done", async ({ page }) => {
    const problems = watchErrors(page);
    await wizard(page);
    await expect(step(page)).toHaveText("Import · step 1 of 5");
    await expect(page.getByText("Download total: 181 MB")).toBeVisible();
    await next(page).click();
    await expect(step(page)).toHaveText("Import · step 2 of 5");
    await expect(page.getByText("Waiting for Look Library · LogC4…")).toBeVisible();
    await expect(next(page)).toHaveAttribute("aria-disabled", "true");
    await dropFiles(page, [LL4]);
    await expect(page.getByText("Everything has arrived.", { exact: true })).toBeVisible();
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "Found it" })).toBeVisible();
    await expect(page.getByText("Match", { exact: true })).toBeVisible();
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "Importing" })).toBeVisible();
    await expect(page.getByRole("heading", { level: 1, name: "Imported" })).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("87", { exact: true })).toBeVisible();
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "87 looks ready" })).toBeVisible();
    const trash = await recordCalls(page, "trashPath");
    await next(page).click();
    // Back on Start, which now says how many looks are imported; the zip went to the Trash.
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
    await expect(page.getByText("87 looks imported")).toBeVisible();
    expect((await trash()).map((c) => c.args)).toEqual([[LL4]]);
    expect(problems).toEqual([]);
  });

  test("a wizard opened from the Look tab returns to the photos, with the looks in the grid", async ({ page }) => {
    await openApp(page, { mockLooks: 0, mockCurrent: 3 });
    await openPhotos(page);
    await page.getByRole("button", { name: "Import ARRI looks →" }).click();
    await toRecognise(page, LL4);
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "Imported" })).toBeVisible({ timeout: 10_000 });
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "87 looks ready" })).toBeVisible();
    // The tiles are the current photo through each look.
    await expect(page.locator(".ym-wizard__tile-picture img")).toHaveCount(7);
    await next(page).click();
    await expect(page.getByRole("tab", { name: "Look" })).toBeVisible();
    await expect(page.getByText("ARRI looks · imported")).toBeVisible();
    await expect(page.locator(".ym-look-group__count").nth(1)).toHaveText("87");
  });

  test("display LUTs bring the display rendering choice, which is saved", async ({ page }) => {
    await wizard(page);
    await page.getByRole("checkbox", { name: "Display LUTs · LogC4" }).click();
    await toRecognise(page, LL4, DL4);
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "Imported" })).toBeVisible({ timeout: 15_000 });
    await next(page).click();
    await page.getByRole("radiogroup", { name: "Display rendering" }).getByRole("radio", { name: "ARRI's" }).click();
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();
    const display = await page.evaluate(async () => (await window.__yellowmoon!.mockApi.preferencesGet()).display_rendering);
    expect(display).toEqual({ kind: "arri_display", id: "arri-display:logc4:rec709-g24" });
  });
});

test.describe("steps are locked until their turn", () => {
  test("later steps and Next are locked, earlier ones can be revisited", async ({ page }) => {
    await wizard(page);
    const rail = page.getByRole("navigation", { name: "Steps" });
    await expect(rail.getByRole("button", { name: /^Get the files/ })).toHaveAttribute("aria-disabled", "true");
    await page.getByRole("checkbox", { name: "Look Library · LogC4" }).click();
    await expect(page.getByText("Choose at least one package")).toBeVisible();
    await expect(next(page)).toHaveAttribute("aria-disabled", "true");
    await next(page).click({ force: true });
    await expect(step(page)).toHaveText("Import · step 1 of 5");
    await page.getByRole("checkbox", { name: "Look Library · LogC4" }).click();
    await toRecognise(page, LL4);
    await rail.getByRole("button", { name: /^Choose[,.]/ }).click();
    await expect(step(page)).toHaveText("Import · step 1 of 5");
    await expect(rail.getByRole("button", { name: /^Recognise/ })).toHaveAttribute("aria-disabled", "true");
  });

  test("Cancel is not possible while files are copied", async ({ page }) => {
    await wizard(page, { mockImportHold: 40 });
    await toRecognise(page, LL4);
    await next(page).click();
    await expect(page.getByText("of 87 files")).toBeVisible();
    await expect(page.getByRole("button", { name: "Cancel" })).toHaveAttribute("aria-disabled", "true");
    await expect(page.getByRole("button", { name: "← Back" })).toHaveAttribute("aria-disabled", "true");
    await page.evaluate(() => window.__yellowmoon!.mockApi.flush());
    await expect(page.getByRole("heading", { level: 1, name: "Imported" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Cancel" })).not.toHaveAttribute("aria-disabled", "true");
  });
});

test.describe("failure paths", () => {
  test("a wrong package: it says what it is and what to download instead", async ({ page }) => {
    await wizard(page);
    await toRecognise(page, LL3);
    await expect(page.getByRole("heading", { level: 1, name: "Check what arrived" })).toBeVisible();
    await expect(page.getByText("Not chosen", { exact: true })).toBeVisible();
    await expect(page.getByText("This is Look Library · LogC3, not Look Library · LogC4, which you chose.", { exact: false })).toBeVisible();
    await expect(next(page)).toHaveAttribute("aria-disabled", "true");
    await expect(page.getByText("Look Library · LogC4 is missing. Go back and add it.")).toBeVisible();
    // Back, drop the right one, and carry on.
    await page.getByRole("button", { name: "← Back" }).click();
    await dropFiles(page, [LL4]);
    await next(page).click();
    await expect(page.getByText("Match", { exact: true })).toBeVisible();
  });

  test("an ALF4-only zip: the explanation of docs/08", async ({ page }) => {
    await wizard(page);
    await toRecognise(page, ALF4);
    await expect(page.getByText("These are ARRI look files for the camera (.alf4). Yellowmoon needs the 3D-LUT version: download 'Look Library for LogC4 (3D-LUTs)' from ARRI's look page.")).toBeVisible();
    await expect(page.getByText("Can't use", { exact: true })).toBeVisible();
    await expect(next(page)).toHaveAttribute("aria-disabled", "true");
  });

  test("a missing file: it went away after it arrived", async ({ page }) => {
    await wizard(page);
    await next(page).click();
    await dropFiles(page, [LL4]);
    await expect(next(page)).not.toHaveAttribute("aria-disabled", "true");
    await page.evaluate((name) => window.__yellowmoon!.mockApi.removeSource(name), LL4);
    await next(page).click();
    await expect(page.getByText("Missing", { exact: true })).toBeVisible();
    await expect(page.getByText("That file is not there any more. Download it again, or choose it again.")).toBeVisible();
    await expect(next(page)).toHaveAttribute("aria-disabled", "true");
  });

  test("an incomplete download names the missing files' count", async ({ page }) => {
    await wizard(page);
    await toRecognise(page, "ARRI_LookLibrary_LogC4_incomplete.zip");
    await expect(page.getByText(/should have 87 files, but this zip has 79\. The download may be incomplete\./)).toBeVisible();
  });

  test("an import that fails says why and can be tried again", async ({ page }) => {
    await wizard(page);
    await toRecognise(page, LL4);
    await page.evaluate(() => window.__yellowmoon!.mockApi.failNext("looksImport", { scope: "looks", message: "The looks folder can't be written to. Check its permissions." }));
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "Not imported" })).toBeVisible();
    await expect(page.getByRole("alert").filter({ hasText: "The looks folder can't be written to." })).toBeVisible();
    await page.getByRole("button", { name: "Try again" }).click();
    await expect(page.getByRole("heading", { level: 1, name: "Imported" })).toBeVisible({ timeout: 10_000 });
  });
});

test.describe("the generic .cube question", () => {
  test("a .cube of unknown origin asks what it expects as input, and is imported with the answer", async ({ page }) => {
    await wizard(page);
    await toRecognise(page, LL4, "Warm_Film.cube");
    await expect(page.getByText("What does this LUT expect as input?")).toBeVisible();
    await expect(next(page)).toHaveAttribute("aria-disabled", "true");
    await expect(page.getByText("Say what the other LUTs expect as input.")).toBeVisible();
    const group = page.getByRole("radiogroup", { name: "Input of Warm_Film.cube" });
    await expect(group.getByRole("radio")).toHaveText(["LogC4", "LogC3", "LogC3 · Rec.709"]);
    await group.getByRole("radio", { name: "LogC3 · Rec.709" }).click();
    await expect(page.getByText("LogC3 in, Rec.709 out. It replaces our display rendering.")).toBeVisible();
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "Imported" })).toBeVisible({ timeout: 10_000 });
    const mine = await page.evaluate(async () => (await window.__yellowmoon!.mockApi.looksList()).find((l) => l.id === "user:warm-film"));
    expect(mine).toMatchObject({ name: "Warm Film", input: "logc3_to_rec709" });
  });

  test("Looks › Import .cube… is offered in the Look tab", async ({ page }) => {
    await openApp(page, { mockCurrent: 3 });
    await openPhotos(page);
    await page.evaluate(() => window.__yellowmoon!.mockApi.queueChoice("files", ["~/LUTs/Warm_Film.cube"]));
    await page.getByRole("button", { name: "Import .cube…" }).click();
    const dialog = page.getByRole("dialog", { name: "Import LUTs" });
    await expect(dialog).toBeVisible();
    await dialog.getByRole("radio", { name: "LogC4" }).click();
    await dialog.getByRole("button", { name: "Import" }).click();
    await expect(page.getByText("1 LUT added to Your LUTs.")).toBeVisible();
    await expect(page.locator(".ym-look-group__header", { hasText: "Your LUTs" })).toBeVisible();
  });
});

test.describe("Get the files", () => {
  test("Open ARRI's page opens the page in the browser", async ({ page }) => {
    await wizard(page);
    await next(page).click();
    const opened = await recordCalls(page, "openUrl");
    await page.getByRole("button", { name: /Open ARRI's page/ }).click();
    expect((await opened()).map((c) => c.args)).toEqual([["https://www.arri.com/en/learn/camera-systems/image-science/look-files"]]);
  });

  test("Watch my Downloads folder picks up a zip that lands", async ({ page }) => {
    await wizard(page, { mockDownloads: LL4 });
    await next(page).click();
    await page.getByRole("switch", { name: "Watch my Downloads folder" }).click();
    // The mock lets the zip "land" a moment after the watcher starts.
    await expect(page.getByRole("list", { name: "Files that arrived" })).toContainText(LL4, { timeout: 10_000 });
    await expect(next(page)).not.toHaveAttribute("aria-disabled", "true");
    // The switch is a preference: it is still on in Preferences.
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("switch", { name: "Watch Downloads during import" })).toHaveAttribute("aria-checked", "true");
  });

  test("macOS refusing the Downloads folder is said in words and the switch goes back", async ({ page }) => {
    await wizard(page, { mockDownloadsDenied: 1 });
    await next(page).click();
    await page.getByRole("switch", { name: "Watch my Downloads folder" }).click();
    await expect(page.getByRole("alert")).toContainText("Yellowmoon can't open your Downloads folder.");
    await expect(page.getByRole("switch", { name: "Watch my Downloads folder" })).toHaveAttribute("aria-checked", "false");
  });

  test("Choose file… takes the file the chooser gives", async ({ page }) => {
    await wizard(page);
    await next(page).click();
    await page.evaluate((name) => window.__yellowmoon!.mockApi.queueChoice("files", [`~/Downloads/${name}`]), LL4);
    await page.getByRole("button", { name: "Choose file…" }).click();
    await expect(page.getByRole("list", { name: "Files that arrived" })).toContainText(LL4);
  });

  test("highlights the drop zone while a drag is over the window", async ({ page }) => {
    await wizard(page);
    await next(page).click();
    await page.evaluate(() => {
      const data = new DataTransfer();
      data.items.add(new File([new Uint8Array(4)], "x.zip"));
      window.dispatchEvent(new DragEvent("dragenter", { dataTransfer: data, bubbles: true, cancelable: true }));
    });
    await expect(page.getByRole("group", { name: "Or drop the zip here" })).toHaveAttribute("data-active", "true");
  });
});

test.describe("layout", () => {
  test("is the mockup's grid: a 360 px rail, a 75 px header, a 96 px footer", async ({ page }) => {
    await wizard(page);
    const geometry = await page.evaluate(() => {
      const box = (selector: string) => {
        const r = (document.querySelector(selector) as HTMLElement).getBoundingClientRect();
        return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) };
      };
      return { header: box(".ym-wizard__header"), rail: box(".ym-wizard__rail"), main: box(".ym-wizard__main"), footer: box(".ym-wizard__footer"), brand: box(".ym-brand") };
    });
    expect(geometry.header).toMatchObject({ y: 0, h: 75 });
    expect(geometry.rail).toMatchObject({ x: 0, y: 75, w: 360 });
    expect(geometry.footer).toMatchObject({ x: 360, y: 804, h: 96 });
    expect(geometry.brand.w).toBe(360);
    await expect(page.locator(".ym-wizard__rail button[aria-current='step']")).toHaveCSS("background-color", "rgb(255, 194, 14)");
  });
});

test.describe("screenshots against design/screenshots", () => {
  /** The mockup's states: Photos with the fourth photo current (the tiles of the last step are that photo), then the wizard. */
  async function mockupWizard(page: Page, options: Record<string, string | number> = {}): Promise<void> {
    await openApp(page, { mockLooks: 0, mockCurrent: 3, mockPicked: "2,3,4", ...options });
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    await page.getByRole("button", { name: /Open photos/ }).click();
    await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();
    await page.getByRole("button", { name: "Import ARRI looks →" }).click();
    await expect(page.getByRole("heading", { level: 1, name: "Choose what to import" })).toBeVisible();
  }

  test("1 · Choose (import-1-choose.png)", async ({ page }) => {
    await mockupWizard(page);
    await compareScreen(page, "import-1-choose");
  });

  test("2 · Get the files, watching Downloads (import-2-get-files.png)", async ({ page }) => {
    await mockupWizard(page);
    await next(page).click();
    await page.getByRole("switch", { name: "Watch my Downloads folder" }).click();
    await expect(page.getByText("Watching Downloads for Look Library · LogC4…")).toBeVisible();
    await compareScreen(page, "import-2-get-files");
  });

  test("3 · Recognise (import-3-recognise.png)", async ({ page }) => {
    await mockupWizard(page);
    await toRecognise(page, LL4);
    await compareScreen(page, "import-3-recognise");
  });

  test("4 · Importing, 40 of 87 (import-4-importing.png)", async ({ page }) => {
    await mockupWizard(page, { mockImportHold: 40 });
    await toRecognise(page, LL4);
    await next(page).click();
    await expect(page.getByText("of 87 files")).toBeVisible();
    await expect(page.getByRole("progressbar", { name: "Importing looks" })).toHaveAttribute("aria-valuenow", "46");
    await compareScreen(page, "import-4-importing");
  });

  test("4 · Imported (import-4-imported.png)", async ({ page }) => {
    await mockupWizard(page, { mockImportHold: 40 });
    await toRecognise(page, LL4);
    await next(page).click();
    await expect(page.getByText("of 87 files")).toBeVisible();
    await page.evaluate(() => window.__yellowmoon!.mockApi.flush());
    await expect(page.getByRole("heading", { level: 1, name: "Imported" })).toBeVisible();
    await compareScreen(page, "import-4-imported");
  });

  test("5 · Finish (import-5-finish.png)", async ({ page }) => {
    await mockupWizard(page);
    await toRecognise(page, LL4);
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "Imported" })).toBeVisible({ timeout: 10_000 });
    await next(page).click();
    await expect(page.getByRole("heading", { level: 1, name: "87 looks ready" })).toBeVisible();
    await expect(page.locator(".ym-wizard__tile-picture img")).toHaveCount(7);
    await compareScreen(page, "import-5-finish");
  });
});

test("the update cell is not drawn in the wizard (the mockup's header has none)", async ({ page }) => {
  await wizard(page);
  await showUpdateReady(page).catch(() => undefined);
  await expect(page.locator(".ym-wizard__header .ym-update-pill")).toHaveCount(0);
});
