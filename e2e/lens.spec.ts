import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { arrowInList, openApp, openPhotos, recordCalls, setValue, watchErrors } from "./helpers/app";

// The Lens tab (docs/21 section 1, docs/11 "Lens card"): the Lens card (the seven lenses as rows, each with its line) and the Amounts card
// (Flares, Edges and Anamorphic), in a real browser against the mocked API.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await openPhotos(page);
  await page.getByRole("tab", { name: "Lens" }).click();
  await expect(page.getByRole("tab", { name: "Lens" })).toHaveAttribute("aria-selected", "true");
}

const lens = (page: Page) => page.getByRole("region", { name: "Lens" });
const amounts = (page: Page) => page.getByRole("region", { name: "Amounts" });
const list = (page: Page) => lens(page).getByRole("radiogroup", { name: "Lens" });
/** The row of the lens called `name`: a radio named by the name and then the line. */
const row = (page: Page, name: string) => list(page).getByRole("radio", { name: new RegExp(`^${name} `) });
const valueBox = (page: Page, label: string) => page.getByRole("textbox", { name: `${label} value` });

const NAMES = ["Default", "Classic", "Vintage", "Petzval", "Triplet", "Sonnar", "Detuned"];
const LINES = [
  "Clean and modern: even sharpness, faint glow.",
  "Warm and gentle: soft glow, amber flares, smooth falloff.",
  "Glows wide open, lower contrast, soft edges, a little swirl.",
  "Very sharp centre, strong swirl, fast falloff.",
  "Soft corners, busy flares, a crisp centre.",
  "Smooth and glowing, with gentle edges.",
  "Sharp only in the centre, with smeared flares.",
];

/**
 * The sliders lie under the list: at 1440 × 900 the Amounts card is below the visible part of the panel, so the panel is scrolled to its
 * end, where the mouse can reach them (a real mouse drag does not scroll).
 */
async function scrollToAmounts(page: Page): Promise<void> {
  await page.locator(".ym-tools__scroll").evaluate((el) => {
    el.scrollTop = el.scrollHeight;
  });
  await expect(amounts(page).getByRole("slider", { name: "Anamorphic" })).toBeInViewport({ ratio: 1 });
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

test.describe("the Lens tab", () => {
  test("is the second tab, ⌘2 opens it, and it has the Lens card (seven rows) and the Amounts card", async ({ page }) => {
    const problems = watchErrors(page);
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await expect(page.getByRole("tab")).toHaveText(["Look", "Lens", "Light", "Colour", "Detail", "Curve", "RAW"]);
    await page.keyboard.press("Meta+2");
    await expect(page.getByRole("tab", { name: "Lens" })).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("complementary", { name: "Tools" }).getByRole("heading", { level: 2, name: "Lens" })).toBeVisible();
    await expect(page.getByText("How a lens draws light: flares, softer edges, the shape of blur. Off until you raise a slider.")).toBeVisible();
    await expect(lens(page).getByRole("heading", { level: 3, name: "Lens" })).toBeVisible();
    await expect(amounts(page).getByRole("heading", { level: 3, name: "Amounts" })).toBeVisible();
    await expect(list(page).getByRole("radio")).toHaveCount(7);
    // The list is above the sliders.
    const listBox = (await list(page).boundingBox())!;
    const flares = (await valueBox(page, "Flares").boundingBox())!;
    expect(flares.y).toBeGreaterThan(listBox.y + listBox.height);
    expect(problems).toEqual([]);
  });

  test("the Look tab has no Lens card, and Light is the third tab (⌘3)", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await expect(page.getByRole("region", { name: "Tone" })).toBeVisible();
    await expect(page.getByRole("region", { name: "Lens" })).toHaveCount(0);
    await expect(page.getByRole("slider", { name: "Flares" })).toHaveCount(0);
    await page.keyboard.press("Meta+3");
    await expect(page.getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true");
  });

  test("the tab has its own colour: a soft violet on the chosen row, the underline and a changed value", async ({ page }) => {
    await photos(page);
    // #bd8cff; black on it.
    const violet = "rgb(189, 140, 255)";
    await expect(row(page, "Default")).toHaveCSS("background-color", violet);
    await expect(page.locator('[data-slot="tabs-indicator"]')).toHaveCSS("background-color", violet);
    await expect(page.getByRole("tabpanel")).toHaveAttribute("data-tab", "lens");
    const chosenRow = row(page, "Default").locator("xpath=ancestor::label[1]");
    // The ring of the row is the tab colour too (an inset ring of 2 px).
    expect(await chosenRow.evaluate((el) => getComputedStyle(el).boxShadow)).toContain(violet);
    await setValue(page, "Flares", "60");
    await expect(valueBox(page, "Flares")).toHaveCSS("border-color", violet);
  });

  test("the list has the seven lenses, each with its line under its name, in the order of the card", async ({ page }) => {
    await photos(page);
    const rows = list(page).locator("label");
    await expect(rows).toHaveCount(7);
    for (const [i, name] of NAMES.entries()) {
      await expect(rows.nth(i).locator(".ym-choice-row__name")).toHaveText(name);
      await expect(rows.nth(i).locator(".ym-choice-row__line")).toHaveText(LINES[i]!);
      // The line is under the name and in the quieter colour.
      const nameBox = (await rows.nth(i).locator(".ym-choice-row__name").boundingBox())!;
      const lineBox = (await rows.nth(i).locator(".ym-choice-row__line").boundingBox())!;
      expect(lineBox.y).toBeGreaterThanOrEqual(nameBox.y + nameBox.height - 1);
    }
    await expect(row(page, "Default")).toHaveAttribute("aria-checked", "true");
    // A row has the name and then the line as its accessible name.
    await expect(row(page, "Vintage")).toHaveAccessibleName(`Vintage ${LINES[2]}`);
    // The line of the chosen lens is in its row only: the Amounts card has none.
    await expect(page.getByText(LINES[0]!)).toHaveCount(1);
    await expect(amounts(page).locator(".ym-row-note")).toHaveCount(0);
  });

  test("choosing a lens with a click sends it with its starting amounts in one patch, keeps Anamorphic and is one undo step", async ({ page }) => {
    await photos(page);
    await setValue(page, "Anamorphic", "25");
    await expect(valueBox(page, "Anamorphic")).toHaveValue("25");
    const calls = await recordCalls(page, "updateSettings");

    // A click on the line of a row chooses it, as a click on the name or the circle does.
    await lens(page).getByText(LINES[2]!).click();
    await expect(row(page, "Vintage")).toHaveAttribute("aria-checked", "true");
    await expect(row(page, "Default")).toHaveAttribute("aria-checked", "false");
    expect(await lensPatches(calls)).toEqual([{ kind: "vintage", flares: 45, edges: 40, anamorphic: 25 }]);
    await expect(valueBox(page, "Flares")).toHaveValue("45");
    await expect(valueBox(page, "Edges")).toHaveValue("40");
    await expect(valueBox(page, "Anamorphic")).toHaveValue("25");

    // One undo step: the lens and the amounts it moved go back, the earlier Anamorphic stays.
    await page.keyboard.press("Meta+z");
    await expect(row(page, "Default")).toHaveAttribute("aria-checked", "true");
    await expect(valueBox(page, "Flares")).toHaveValue("0");
    await expect(valueBox(page, "Anamorphic")).toHaveValue("25");
  });

  test("a lens can be chosen with the keyboard alone: Tab reaches the list on the chosen row, the arrows choose, one patch each", async ({ page }) => {
    await photos(page);
    const calls = await recordCalls(page, "updateSettings");
    await page.getByRole("tab", { name: "Lens" }).focus();
    await page.keyboard.press("Tab");
    await expect(row(page, "Default")).toBeFocused();
    await expect(row(page, "Default")).toHaveCSS("outline-width", "3px");
    await arrowInList(page, "ArrowDown");
    await expect(row(page, "Classic")).toBeFocused();
    await expect(row(page, "Classic")).toHaveAttribute("aria-checked", "true");
    await arrowInList(page, "ArrowDown");
    await expect(row(page, "Vintage")).toBeFocused();
    await expect(row(page, "Vintage")).toHaveAttribute("aria-checked", "true");
    expect(await lensPatches(calls)).toEqual([
      { kind: "classic", flares: 35, edges: 30, anamorphic: 0 },
      { kind: "vintage", flares: 45, edges: 40, anamorphic: 0 },
    ]);
    // The arrows wrap at the ends, and only the chosen row is a Tab stop: the next Tab leaves the list for the Amounts card's Reset.
    await arrowInList(page, "ArrowUp");
    await arrowInList(page, "ArrowUp");
    await arrowInList(page, "ArrowUp");
    await expect(row(page, "Detuned")).toBeFocused();
    await expect(row(page, "Detuned")).toHaveAttribute("aria-checked", "true");
    await page.keyboard.press("Tab");
    await expect(amounts(page).getByRole("button", { name: "Reset", exact: true })).toBeFocused();
    await page.keyboard.press("Shift+Tab");
    await expect(row(page, "Detuned")).toBeFocused();
  });

  test("dragging a slider sends the whole lens object and is one undo step", async ({ page }) => {
    await photos(page);
    await row(page, "Classic").click();
    await expect(valueBox(page, "Flares")).toHaveValue("35");
    await scrollToAmounts(page);
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
    await expect(row(page, "Classic")).toHaveAttribute("aria-checked", "true");
  });

  test("a row's Reset sends that amount at 0, and the Amounts card's Reset goes to Default with the sliders at 0", async ({ page }) => {
    await photos(page);
    await row(page, "Detuned").click();
    await setValue(page, "Anamorphic", "60");
    await expect(valueBox(page, "Anamorphic")).toHaveValue("60");
    const calls = await recordCalls(page, "updateSettings");

    await amounts(page).getByRole("button", { name: "Reset Edges" }).click();
    await expect(valueBox(page, "Edges")).toHaveValue("0");
    expect((await lensPatches(calls)).at(-1)).toEqual({ kind: "detuned", flares: 45, edges: 0, anamorphic: 60 });

    await amounts(page).getByRole("button", { name: "Reset", exact: true }).click();
    await expect(row(page, "Default")).toHaveAttribute("aria-checked", "true");
    for (const label of ["Flares", "Edges", "Anamorphic"]) await expect(valueBox(page, label)).toHaveValue("0");
    expect((await lensPatches(calls)).at(-1)).toEqual({ kind: "default", flares: 0, edges: 0, anamorphic: 0 });
  });

  test("the lens belongs to the photo: another photo has its own", async ({ page }) => {
    await photos(page);
    await row(page, "Triplet").click();
    await expect(row(page, "Triplet")).toHaveAttribute("aria-checked", "true");
    // The filmstrip: the photo next to it has no lens; coming back, Triplet is still on the first.
    const thumb = (n: number) => page.locator(".ym-filmstrip__thumb").nth(n);
    await thumb(4).click();
    await expect(page.locator(".ym-filmstrip__caption")).toContainText("5 of 12");
    await expect(row(page, "Default")).toHaveAttribute("aria-checked", "true");
    await expect(valueBox(page, "Flares")).toHaveValue("0");
    await thumb(3).click();
    await expect(page.locator(".ym-filmstrip__caption")).toContainText("4 of 12");
    await expect(row(page, "Triplet")).toHaveAttribute("aria-checked", "true");
    await expect(valueBox(page, "Flares")).toHaveValue("50");
  });

  test("without lens data both cards stay: one row off for the photo's lens, the sliders off, and a note that says why", async ({ page }) => {
    await photos(page, { mockLenses: 0 });
    await expect(list(page).getByRole("radio")).toHaveCount(1);
    await expect(list(page).getByRole("radio")).toBeDisabled();
    await expect(list(page).getByRole("radio")).toHaveAttribute("aria-checked", "true");
    for (const label of ["Flares", "Edges", "Anamorphic"]) await expect(valueBox(page, label)).toBeDisabled();
    await expect(lens(page).getByText("Lens data are missing, so lenses can't be used.")).toBeVisible();
    // Reset still works.
    await expect(amounts(page).getByRole("button", { name: "Reset", exact: true })).toBeEnabled();
  });
});
