import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, recordCalls, setValue, watchErrors } from "./helpers/app";

// The Look tab's Lens card (docs/21 section 1, docs/11 "Lens card"): a lens choice, Flares, Edges and Anamorphic, and a note with the
// lens's line, in a real browser against the mocked API.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await openPhotos(page);
}

const lens = (page: Page) => page.getByRole("region", { name: "Lens" });
const lensBox = (page: Page) => lens(page).getByRole("combobox", { name: "Lens" });
const valueBox = (page: Page, label: string) => page.getByRole("textbox", { name: `${label} value` });

/** The card lies under the look grid: scrolls the panel to its end, so the card and its thumbs are where the mouse can reach them. */
async function scrollToCard(page: Page): Promise<void> {
  await page.locator(".ym-tools__scroll").evaluate((el) => {
    el.scrollTop = el.scrollHeight;
  });
  await expect(lens(page)).toBeInViewport({ ratio: 0.98 });
}

/** Drags a slider's thumb to a share of its track with the real mouse. */
async function dragSlider(page: Page, label: string, to: number, steps = 6): Promise<void> {
  const control = page.getByRole("slider", { name: label }).locator("xpath=ancestor::*[@data-slot='slider'][1]");
  const box = (await control.boundingBox())!;
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width * 0.5, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * to, y, { steps });
  await page.mouse.up();
}

type Lens = { kind: string; flares: number; edges: number; anamorphic: number };

/** The `lens` of every `update_settings` call recorded so far. */
async function lensPatches(calls: () => Promise<Array<{ args: unknown[] }>>): Promise<Lens[]> {
  return (await calls()).map((c) => (c.args[1] as { lens: Lens }).lens);
}

test.describe("the Lens card", () => {
  test("sits under the Tone card: the Lens choice, then Flares, Edges and Anamorphic, then the lens's line", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page);
    await scrollToCard(page);
    const tone = (await page.getByRole("region", { name: "Tone" }).boundingBox())!;
    const card = (await lens(page).boundingBox())!;
    expect(card.y).toBeGreaterThanOrEqual(tone.y + tone.height - 1);
    await expect(lens(page).getByRole("heading", { level: 3, name: "Lens" })).toBeVisible();
    await expect(lensBox(page)).toHaveText("Default");
    const ys: number[] = [(await lensBox(page).boundingBox())!.y];
    for (const label of ["Flares", "Edges", "Anamorphic"]) {
      await expect(lens(page).getByRole("slider", { name: label })).toBeVisible();
      await expect(valueBox(page, label)).toHaveValue("0");
      ys.push((await valueBox(page, label).boundingBox())!.y);
    }
    const note = lens(page).getByText("Clean and modern: even sharpness, faint glow.");
    await expect(note).toBeVisible();
    ys.push((await note.boundingBox())!.y);
    expect(ys).toEqual([...ys].sort((a, b) => a - b));
    expect(problems).toEqual([]);
  });

  test("the menu lists the seven lenses with their lines, and choosing one sends it with its starting amounts in one patch", async ({ page }) => {
    await photos(page);
    await scrollToCard(page);
    await setValue(page, "Anamorphic", "25");
    await expect(valueBox(page, "Anamorphic")).toHaveValue("25");
    const calls = await recordCalls(page, "updateSettings");

    await lensBox(page).click();
    const options = page.getByRole("option");
    await expect(options).toHaveCount(7);
    const names = ["Default", "Classic", "Vintage", "Petzval", "Triplet", "Sonnar", "Detuned"];
    const lines = [
      "Clean and modern: even sharpness, faint glow.",
      "Warm and gentle: soft glow, amber flares, smooth falloff.",
      "Glows wide open, lower contrast, soft edges, a little swirl.",
      "Very sharp centre, strong swirl, fast falloff.",
      "Soft corners, busy flares, a crisp centre.",
      "Smooth and glowing, with gentle edges.",
      "Sharp only in the centre, with smeared flares.",
    ];
    for (const [i, name] of names.entries()) {
      await expect(options.nth(i).locator(".ym-lens-option__name")).toHaveText(name);
      await expect(options.nth(i).locator(".ym-lens-option__line")).toHaveText(lines[i]!);
    }
    await options.nth(2).click();

    await expect(lensBox(page)).toHaveText("Vintage");
    expect(await lensPatches(calls)).toEqual([{ kind: "vintage", flares: 45, edges: 40, anamorphic: 25 }]);
    await expect(valueBox(page, "Flares")).toHaveValue("45");
    await expect(valueBox(page, "Edges")).toHaveValue("40");
    await expect(valueBox(page, "Anamorphic")).toHaveValue("25");
    await expect(lens(page).getByText(lines[2]!)).toBeVisible();
    await expect(lens(page).getByText(lines[0]!)).toHaveCount(0);

    // One undo step: the lens and the amounts it moved go back, the earlier Anamorphic stays.
    await page.keyboard.press("Meta+z");
    await expect(lensBox(page)).toHaveText("Default");
    await expect(valueBox(page, "Flares")).toHaveValue("0");
    await expect(valueBox(page, "Anamorphic")).toHaveValue("25");
  });

  test("a lens can be chosen with the keyboard alone", async ({ page }) => {
    await photos(page);
    await scrollToCard(page);
    const calls = await recordCalls(page, "updateSettings");
    await lensBox(page).focus();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("option")).toHaveCount(7);
    // Radix moves the focus a tick after the key, so each step waits for it.
    for (const next of [/^Classic/, /^Vintage/, /^Petzval/]) {
      await page.keyboard.press("ArrowDown");
      await expect(page.getByRole("option", { name: next })).toBeFocused();
    }
    await page.keyboard.press("Enter");
    await expect(lensBox(page)).toHaveText("Petzval");
    expect(await lensPatches(calls)).toEqual([{ kind: "petzval", flares: 30, edges: 70, anamorphic: 0 }]);
    // The focus comes back to the box.
    await expect(lensBox(page)).toBeFocused();
  });

  test("dragging a slider sends the whole lens object and is one undo step", async ({ page }) => {
    await photos(page);
    await scrollToCard(page);
    await lensBox(page).click();
    await page.getByRole("option", { name: /^Classic/ }).click();
    await expect(valueBox(page, "Flares")).toHaveValue("35");
    const calls = await recordCalls(page, "updateSettings");
    await dragSlider(page, "Edges", 0.9);
    const patches = await lensPatches(calls);
    expect(patches.length).toBeGreaterThanOrEqual(2);
    for (const p of patches) {
      expect(p.kind).toBe("classic");
      expect(p.flares).toBe(35);
      expect(p.anamorphic).toBe(0);
      expect(Number.isInteger(p.edges) && p.edges >= 0 && p.edges <= 100).toBe(true);
    }
    const last = patches.at(-1)!.edges;
    expect(last).toBeGreaterThan(60);
    await expect(valueBox(page, "Edges")).toHaveValue(String(last));
    // The whole drag is one step.
    await page.keyboard.press("Meta+z");
    await expect(valueBox(page, "Edges")).toHaveValue("30");
    await expect(lensBox(page)).toHaveText("Classic");
  });

  test("a row's Reset sends that amount at 0, and the card's Reset goes to Default with the sliders at 0", async ({ page }) => {
    await photos(page);
    await scrollToCard(page);
    await lensBox(page).click();
    await page.getByRole("option", { name: /^Detuned/ }).click();
    await setValue(page, "Anamorphic", "60");
    await expect(valueBox(page, "Anamorphic")).toHaveValue("60");
    const calls = await recordCalls(page, "updateSettings");

    await lens(page).getByRole("button", { name: "Reset Edges" }).click();
    await expect(valueBox(page, "Edges")).toHaveValue("0");
    expect((await lensPatches(calls)).at(-1)).toEqual({ kind: "detuned", flares: 45, edges: 0, anamorphic: 60 });

    await lens(page).getByRole("button", { name: "Reset", exact: true }).click();
    await expect(lensBox(page)).toHaveText("Default");
    for (const label of ["Flares", "Edges", "Anamorphic"]) await expect(valueBox(page, label)).toHaveValue("0");
    expect((await lensPatches(calls)).at(-1)).toEqual({ kind: "default", flares: 0, edges: 0, anamorphic: 0 });
    await expect(lens(page).getByText("Clean and modern: even sharpness, faint glow.")).toBeVisible();
  });

  test("the lens belongs to the photo: another photo has its own", async ({ page }) => {
    await photos(page);
    await scrollToCard(page);
    await lensBox(page).click();
    await page.getByRole("option", { name: /^Triplet/ }).click();
    await expect(lensBox(page)).toHaveText("Triplet");
    // The filmstrip: the photo next to it has no lens; coming back, Triplet is still on the first.
    const thumb = (n: number) => page.locator(".ym-filmstrip__thumb").nth(n);
    await thumb(4).click();
    await expect(page.locator(".ym-filmstrip__caption")).toContainText("5 of 12");
    await expect(lensBox(page)).toHaveText("Default");
    await expect(valueBox(page, "Flares")).toHaveValue("0");
    await thumb(3).click();
    await expect(page.locator(".ym-filmstrip__caption")).toContainText("4 of 12");
    await expect(lensBox(page)).toHaveText("Triplet");
    await expect(valueBox(page, "Flares")).toHaveValue("50");
  });

  test("without lens data the card stays, with the choice and the sliders off and a note that says why", async ({ page }) => {
    await photos(page, { mockLenses: 0 });
    await scrollToCard(page);
    await expect(lensBox(page)).toBeDisabled();
    for (const label of ["Flares", "Edges", "Anamorphic"]) await expect(valueBox(page, label)).toBeDisabled();
    await expect(lens(page).getByText("Lens data are missing, so lenses can't be used.")).toBeVisible();
    // Reset still works.
    await expect(lens(page).getByRole("button", { name: "Reset", exact: true })).toBeEnabled();
  });
});
