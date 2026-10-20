import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, openApp, openPhotos, recordCalls, setValue, watchErrors } from "./helpers/app";

// "Rendering on the CPU" (docs/11 §Photos, the photo cell) in a real browser against the mocked API. With `?mockCpu=1` the mock
// renders like a core whose raw stages run on the CPU: `viewport_busy { busy: true, cpu: true }` before every frame it "renders" and
// `{ busy: false }` 600 ms divided by its speed later. The photos open at full speed, then the mock is slowed to 0.25 so that a frame
// takes 2.4 s: long enough to see the note come (after 250 ms) and go. Without the option the mock is a graphics card and says nothing.

/** The photos of the mockup, open on the Photos screen; frames take 2.4 s from then on. */
async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { mockCurrent: 3, ...extra });
  await openPhotos(page);
  await configureMock(page, { speed: 0.25 });
}

const cell = (page: Page) => page.locator("[data-viewport]");
const note = (page: Page) => page.getByRole("status").filter({ hasText: "Rendering on the CPU" });

/** The note is at the top centre of the photo cell, 12 px in from its top edge. */
async function expectTopCentre(page: Page): Promise<void> {
  const box = (await cell(page).boundingBox())!;
  const at = (await note(page).boundingBox())!;
  expect(Math.abs(at.x + at.width / 2 - (box.x + box.width / 2))).toBeLessThan(1.5);
  expect(at.y - box.y).toBeCloseTo(12, 0);
}

/** Notes, from now on, whether the words "Rendering on the CPU" were ever on the page. */
async function watchForTheNote(page: Page): Promise<() => Promise<boolean>> {
  await page.evaluate(() => {
    const w = window as unknown as { __sawCpuNote: boolean };
    w.__sawCpuNote = false;
    const look = (): void => {
      if (document.body.textContent?.includes("Rendering on the CPU")) w.__sawCpuNote = true;
    };
    new MutationObserver(look).observe(document.body, { subtree: true, childList: true, characterData: true });
  });
  return () => page.evaluate(() => (window as unknown as { __sawCpuNote: boolean }).__sawCpuNote);
}

test.describe("with the raw stages on the CPU (mockCpu)", () => {
  test("a slider change shows the note at the top centre of the photo, and it goes when the frame is done", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page, { mockCpu: 1 });
    await expect(note(page)).toHaveCount(0);
    await setValue(page, "Intensity", "80");
    await expect(note(page)).toBeVisible();
    await expect(note(page)).toHaveText("Rendering on the CPU");
    await expect(note(page)).toHaveAttribute("aria-live", "polite");
    await expectTopCentre(page);
    // The glass pill of the cell's other badges, with the ring spinner.
    await expect(note(page)).toHaveCSS("background-color", "rgba(14, 14, 18, 0.74)");
    await expect(note(page).locator('[data-slot="spinner"]')).toBeVisible();
    await expect(note(page)).toBeHidden({ timeout: 15_000 });
    expect(problems).toEqual([]);
  });

  test("another photo shows it too", async ({ page }) => {
    await photos(page, { mockCpu: 1 });
    await page.keyboard.press("ArrowRight");
    await expect(page.getByText(/^5 of 12 · /)).toBeVisible();
    await expect(note(page)).toBeVisible();
    await expect(note(page)).toBeHidden({ timeout: 15_000 });
  });

  test("it is not in the pointer's way: a double click on it reaches the photo", async ({ page }) => {
    await photos(page, { mockCpu: 1 });
    await setValue(page, "Intensity", "70");
    await expect(note(page)).toBeVisible();
    await expect(note(page)).toHaveCSS("pointer-events", "none");
    const inputs = await recordCalls(page, "viewportInput");
    const at = (await note(page).boundingBox())!;
    await page.mouse.dblclick(at.x + at.width / 2, at.y + at.height / 2);
    await expect.poll(async () => (await inputs()).map((c) => (c.args[0] as { kind: string }).kind)).toContain("toggle_zoom");
  });

  test("in Before and in full screen it stays at the top centre", async ({ page }) => {
    await photos(page, { mockCpu: 1 });
    await page.keyboard.press("\\");
    await expect(page.locator(".ym-photo-cell__before")).toBeVisible();
    await expect(note(page)).toBeVisible();
    await expectTopCentre(page);
    // Clear of the Before label at the top left.
    const label = (await page.locator(".ym-photo-cell__before").boundingBox())!;
    expect(label.x + label.width).toBeLessThan((await note(page).boundingBox())!.x);
    await expect(note(page)).toBeHidden({ timeout: 15_000 });

    await page.keyboard.press("f");
    await expect(page.locator('.ym-screen[data-screen="photos"]')).toHaveAttribute("data-fullscreen", "true");
    await expect(note(page)).toBeVisible();
    await expectTopCentre(page);
    await expect(note(page)).toBeHidden({ timeout: 15_000 });
  });
});

test.describe("on the graphics card (no mockCpu)", () => {
  test("the note never appears, whatever is changed", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page);
    const saw = await watchForTheNote(page);
    await setValue(page, "Intensity", "80");
    await page.keyboard.press("ArrowRight");
    await expect(page.getByText(/^5 of 12 · /)).toBeVisible();
    await page.keyboard.press("\\");
    const box = (await cell(page).boundingBox())!;
    await page.mouse.dblclick(box.x + box.width / 2, box.y + box.height / 2);
    // Four times the note's wait: had the mock said it was busy on the CPU, the note would be on the page by now.
    await page.waitForTimeout(1000);
    expect(await saw()).toBe(false);
    await expect(note(page)).toHaveCount(0);
    expect(problems).toEqual([]);
  });
});
