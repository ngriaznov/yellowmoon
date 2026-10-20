import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, watchErrors } from "./helpers/app";

// Apple Auto in the app (docs/19 sections 2 and 10, docs/11) in a real browser against the mocked API. The mock plays a Mac where Apple
// Auto is available (`mockAppleAuto=answered|pending|failed`) or not (`unavailable`, and the default, which is the browser build): the
// Preferences row is there only while it is available, with the instructions for Apple's model under it while the switch is on, and
// the Automatic card has a row for each value Apple's answer set, its line and the answer's plan only while Apple Auto is on for the
// session.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;
const TITLE = "Auto with Apple Intelligence";
const HINT =
  "Apple's on-device model looks at each photo and grades it: exposure, highlights, shadows, black level, contrast, warmth, tint, saturation and the colours of the shadows and highlights. It runs on this Mac and sends nothing anywhere. Turn it off to use Yellowmoon's own Auto, the same on every computer.";
const INSTRUCTIONS = "Instructions for Apple Intelligence";
const INSTRUCTIONS_HINT = "Apple's model reads these before each photo. The answer form stays the same.";
/** The rows of the Automatic card with the mock's answer (it leaves the shadows and the tint empty: no row for them). */
const ANSWERED_ROWS = [
  "Exposure",
  "White",
  "White balance",
  "Highlights",
  "Black",
  "Contrast",
  "Saturation",
  "Shadows colour",
  "Highlights colour",
  "Noise",
  "Camera",
];
const PLAN = "\u201cTeal shadows and warm highlights for a cinematic late-sun street\u201d";
const LINES = {
  answered: "Set with Apple Intelligence.",
  pending: "Apple Intelligence is looking at this photo…",
  failed: "Apple Intelligence couldn't look at this photo, so Auto uses Yellowmoon's own settings.",
} as const;

async function preferences(page: Page, options: Record<string, string | number>) {
  await openApp(page, options);
  await page.keyboard.press("Meta+,");
  const dialog = page.getByRole("dialog", { name: "Preferences" });
  await expect(dialog.getByRole("group", { name: "Display rendering" })).toBeVisible();
  return dialog;
}

/** The labels of the Automatic card's rows, in order. */
const rows = (page: Page): Promise<string[]> => page.getByRole("region", { name: "Automatic" }).locator(".ym-auto-row__label").allTextContents();

test.describe("Preferences", () => {
  test("with Apple Auto available: the row after Display rendering, its whole hint and a glasscn switch", async ({ page }) => {
    const problems = watchErrors(page);
    const dialog = await preferences(page, { mockAppleAuto: "answered" });
    const row = dialog.getByRole("group", { name: TITLE });
    await expect(row).toBeVisible();
    await expect(row).toContainText(HINT);
    const groups = await dialog.getByRole("group").evaluateAll((els) => els.map((el) => el.getAttribute("aria-label")));
    expect(groups.indexOf(TITLE)).toBe(groups.indexOf("Display rendering") + 1);
    const toggle = dialog.getByRole("switch", { name: TITLE });
    await expect(toggle).toHaveAttribute("aria-checked", "true");
    await expect(toggle).toHaveAttribute("data-slot", "switch");
    // The hint is not cut: its paragraph is as tall as its lines.
    const hint = row.locator(".ym-pref__hint");
    expect(await hint.evaluate((el) => el.scrollHeight <= el.clientHeight + 1)).toBe(true);
    // The switch writes the preference, and the Automatic card follows it.
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-checked", "false");
    await expect.poll(() => page.evaluate(async () => (await window.__yellowmoon!.mockApi.preferencesGet()).apple_auto)).toBe(false);
    expect(problems).toEqual([]);
  });

  test("the instructions for Apple's model: under the switch while it is on, saved on blur, and Reset to default", async ({ page }) => {
    const problems = watchErrors(page);
    const dialog = await preferences(page, { mockAppleAuto: "answered" });
    const row = dialog.getByRole("group", { name: TITLE });
    const field = dialog.getByRole("textbox", { name: INSTRUCTIONS });
    const reset = dialog.getByRole("button", { name: "Reset to default" });
    const saved = () => page.evaluate(async () => (await window.__yellowmoon!.mockApi.preferencesGet()).apple_auto_instructions);
    const defaults = await page.evaluate(async () => (await window.__yellowmoon!.mockApi.appleAutoStatus()).default_instructions);
    // The default instructions, whole, with the hint and the count; Reset to default is there and disabled.
    await expect(field).toBeVisible();
    await expect(field).toHaveValue(defaults);
    await expect(field).toHaveAttribute("data-slot", "textarea");
    await expect(dialog.getByRole("group", { name: INSTRUCTIONS })).toContainText(INSTRUCTIONS_HINT);
    await expect(dialog.getByRole("group", { name: INSTRUCTIONS })).toContainText(`${[...defaults].length} of 2000 characters.`);
    await expect(reset).toBeVisible();
    await expect(reset).toBeDisabled();
    await expect(reset).toHaveAttribute("data-slot", "button");
    // The button stands next to the field: just under its bottom edge, inside the field's group, under the switch's row.
    const group = dialog.getByRole("group", { name: INSTRUCTIONS });
    const [fieldBox, resetBox, rowBox, switchRowBox] = await Promise.all([field.boundingBox(), reset.boundingBox(), group.boundingBox(), row.boundingBox()]);
    expect(fieldBox!.y).toBeGreaterThanOrEqual(switchRowBox!.y + switchRowBox!.height - 1);
    expect(resetBox!.y).toBeGreaterThanOrEqual(fieldBox!.y + fieldBox!.height - 1);
    expect(resetBox!.y - (fieldBox!.y + fieldBox!.height)).toBeLessThan(48);
    expect(resetBox!.x + resetBox!.width).toBeLessThanOrEqual(rowBox!.x + rowBox!.width + 1);
    // Edited text is saved when the field loses focus, and every photo is asked again with it.
    await field.fill("Grade it like a warm documentary.");
    await field.blur();
    await expect.poll(saved).toBe("Grade it like a warm documentary.");
    await expect.poll(() => page.evaluate(async () => (await window.__yellowmoon!.mockApi.appleAutoStatus()).info?.instructions)).toBe("edited");
    await expect(reset).toBeEnabled();
    // Reset to default saves the default (no own instructions) at once and puts the text back.
    await reset.click();
    await expect.poll(saved).toBe("");
    await expect(field).toHaveValue(defaults);
    await expect(reset).toBeDisabled();
    await expect.poll(() => page.evaluate(async () => (await window.__yellowmoon!.mockApi.appleAutoStatus()).info?.instructions)).toBe("default");
    // The field goes away with the switch.
    await dialog.getByRole("switch", { name: TITLE }).click();
    await expect(field).toHaveCount(0);
    await expect(reset).toHaveCount(0);
    expect(problems).toEqual([]);
  });

  test("without Apple Auto: no row, in the browser build and on a Mac where it is not available", async ({ page }) => {
    for (const options of [{}, { mockAppleAuto: "unavailable" }] as Array<Record<string, string>>) {
      const dialog = await preferences(page, options);
      for (const row of ["Default look", "Display rendering", "Export defaults", "Remove location from exports", "Watch Downloads during import", "Check for updates automatically", "Command-line tool"]) {
        await expect(dialog.getByRole("group", { name: row })).toBeVisible();
      }
      // The status has been asked for by now: the row stays away.
      await expect.poll(() => page.evaluate(() => window.__yellowmoon!.mockApi.appleAutoStatus().then((s) => s.available))).toBe(false);
      await expect(dialog.getByRole("group", { name: TITLE })).toHaveCount(0);
      await expect(dialog.getByRole("switch", { name: TITLE })).toHaveCount(0);
      await expect(dialog.getByRole("textbox", { name: INSTRUCTIONS })).toHaveCount(0);
    }
  });
});

test.describe("the Automatic card", () => {
  test("is today's card without Apple Auto", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    expect(await rows(page)).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]);
    for (const line of Object.values(LINES)) await expect(page.getByText(line)).toHaveCount(0);
  });

  test("answered: a row for each value the answer set between White balance and Noise, the line and the plan", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockAppleAuto: "answered" });
    await openPhotos(page);
    expect(await rows(page)).toEqual(ANSWERED_ROWS);
    await expect(page.getByRole("button", { name: "Highlights −20, Auto. Open Light" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Black −15, Auto. Open Light" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Contrast +15, Auto. Open Tone" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Saturation −15, Auto. Open Tone" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Shadows colour Teal · medium, Auto. Open Colour" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Highlights colour Warm · subtle, Auto. Open Colour" })).toBeVisible();
    const line = page.getByRole("region", { name: "Automatic" }).getByText(LINES.answered);
    await expect(line).toBeVisible();
    await expect(page.getByRole("region", { name: "Automatic" }).getByText(PLAN)).toBeVisible();
    // Muted, as the Camera card's note.
    const [lineColour, rowColour] = await Promise.all([
      line.evaluate((el) => getComputedStyle(el).color),
      page.getByRole("region", { name: "Automatic" }).locator(".ym-auto-row__label").first().evaluate((el) => getComputedStyle(el).color),
    ]);
    expect(lineColour).not.toBe(rowColour);
    // The arrow goes to the Tone card's slider, on the same tab.
    await page.getByRole("button", { name: "Saturation −15, Auto. Open Tone" }).click();
    const slider = page.getByRole("region", { name: "Tone" }).getByRole("slider", { name: "Saturation" });
    await expect(slider).toBeFocused();
    await expect(slider).toBeInViewport();
    await expect(page.getByRole("tab", { name: "Look" })).toHaveAttribute("aria-selected", "true");
  });

  test("pending, then the answer: the rule's values and no rows of Apple's until it arrives, and the line follows", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockAppleAuto: "pending" });
    await openPhotos(page);
    expect(await rows(page)).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]);
    await expect(page.getByRole("textbox", { name: "Contrast value" })).toHaveValue("0");
    await expect(page.getByText(LINES.pending)).toBeVisible();
    await page.evaluate(() => window.__yellowmoon!.mockApi.setAppleAuto("answered"));
    await expect(page.getByRole("button", { name: "Contrast +15, Auto. Open Tone" })).toBeVisible();
    await expect(page.getByText(LINES.answered)).toBeVisible();
    await expect(page.getByText(LINES.pending)).toHaveCount(0);
    expect(await rows(page)).toEqual(ANSWERED_ROWS);
    await expect(page.getByRole("textbox", { name: "Contrast value" })).toHaveValue("+15");
  });

  test("failed: Yellowmoon's own values and the line that says so", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockAppleAuto: "failed" });
    await openPhotos(page);
    expect(await rows(page)).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]);
    await expect(page.getByRole("textbox", { name: "Saturation value" })).toHaveValue("0");
    await expect(page.getByText(LINES.failed)).toBeVisible();
  });

  test("turning the preference off gives back today's card at once", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockAppleAuto: "answered" });
    await openPhotos(page);
    await page.keyboard.press("Meta+,");
    await page.getByRole("switch", { name: TITLE }).click();
    await page.keyboard.press("Escape");
    await expect.poll(() => rows(page)).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]);
    await expect(page.getByText(LINES.answered)).toHaveCount(0);
    await expect(page.getByRole("textbox", { name: "Contrast value" })).toHaveValue("0");
  });
});

test.describe("the sliders", () => {
  test("Contrast: the detent at the Auto value; a typed value is Manual, the Auto value is Auto, Reset is Auto", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockAppleAuto: "answered" });
    await openPhotos(page);
    await page.getByRole("tab", { name: "Light" }).click();
    const box = page.getByRole("textbox", { name: "Contrast value" });
    await expect(box).toHaveValue("+15");
    const settings = () => page.evaluate(async () => (await window.__yellowmoon!.mockApi.getSettings(4)).settings.contrast);
    await box.fill("30");
    await box.press("Enter");
    await expect.poll(settings).toEqual({ kind: "manual", value: 30 });
    await box.fill("15");
    await box.press("Enter");
    await expect.poll(settings).toEqual({ kind: "auto" });
    await box.fill("-5");
    await box.press("Enter");
    await expect.poll(settings).toEqual({ kind: "manual", value: -5 });
    await page.getByRole("button", { name: "Reset Contrast" }).click();
    await expect.poll(settings).toEqual({ kind: "auto" });
    await expect(box).toHaveValue("+15");
  });
});
