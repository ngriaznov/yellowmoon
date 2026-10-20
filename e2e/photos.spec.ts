import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, failNext, lookTilesLoaded, openApp, openPhotos, recordCalls, setUpdateState, setValue, settle, showUpdateReady, watchErrors } from "./helpers/app";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

// Photos screen (tasks M3-06, M3-07, M3-08 UI) in a real browser against the mocked API: layout, the
// flow Start → open → look → every tab, keyboard shortcuts, Crop and Pick white by mouse, and the error
// states. Screenshot comparisons are in screenshots.spec.ts.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await openPhotos(page);
}

const tabs = (page: Page) => page.getByRole("tablist", { name: "Tool tabs" });
const valueBox = (page: Page, label: string) => page.getByRole("textbox", { name: `${label} value` });

/** Drags a slider's thumb to a share of its track with the real mouse. */
async function dragSlider(page: Page, label: string, to: number, steps = 6): Promise<void> {
  const control = page.getByRole("slider", { name: label }).locator("xpath=ancestor::*[@data-slot='slider'][1]");
  const box = (await control.boundingBox())!;
  const y = box.y + box.height / 2;
  const fromX = box.x + box.width * 0.5;
  await page.mouse.move(fromX, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * to, y, { steps });
  await page.mouse.up();
}

test.describe("layout", () => {
  for (const [width, height] of [[1440, 900], [1180, 760], [1800, 1100]] as const) {
    test(`keeps the photo cell exactly 3:2 at ${width} × ${height}`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await photos(page);
      const cell = page.locator("[data-viewport]");
      const box = (await cell.boundingBox())!;
      expect(box.width / box.height).toBeCloseTo(1.5, 2);
      const expected = Math.min(width - 400, (Math.max(height, 760) - 199) * 1.5);
      expect(box.width).toBeCloseTo(expected, 0);
      // No scrolling of the page, and the panel keeps 400 px.
      expect(await page.evaluate(() => document.documentElement.scrollHeight - document.documentElement.clientHeight)).toBeLessThanOrEqual(0);
      expect((await page.getByRole("complementary", { name: "Tools" }).boundingBox())!.width).toBe(400);
      expect((await page.locator(".ym-header").boundingBox())!.height).toBe(75);
    });
  }

  test("places the grid as the mockup does: header 75 px, columns 1fr and 400 px, filmstrip at least 124 px", async ({ page }) => {
    await photos(page);
    const grid = await page.locator("[data-screen=photos]").evaluate((el) => {
      const s = getComputedStyle(el);
      return { columns: s.gridTemplateColumns, rows: s.gridTemplateRows };
    });
    expect(grid.columns).toBe("1040px 400px");
    expect(grid.rows.split(" ")[0]).toBe("75px");
    const strip = (await page.getByRole("navigation", { name: "Photos" }).boundingBox())!;
    expect(strip.height).toBeGreaterThanOrEqual(124);
    expect(await page.locator(".ym-brand").evaluate((el) => getComputedStyle(el).paddingLeft)).toBe("84px");
  });

  test("tells the native layer where the photo cell is", async ({ page }) => {
    await openApp(page, MOCKUP);
    const calls = await recordCalls(page, "viewportSetRect");
    await openPhotos(page);
    await expect.poll(async () => (await calls()).length).toBeGreaterThan(0);
    const [rect] = (await calls())[0]!.args as [{ width: number; height: number; scale: number }];
    expect(rect.scale).toBe(2);
    expect(rect.width / rect.height).toBeCloseTo(1.5, 2);
    await page.setViewportSize({ width: 1300, height: 800 });
    await expect.poll(async () => (await calls()).length).toBeGreaterThan(1);
  });
});

test.describe("the flow", () => {
  test("Start → open → Photos → change the look → visit every tab", async ({ page }) => {
    const problems = watchErrors(page);
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await expect(page.getByText("2026-10-04 · 12 photos")).toBeVisible();
    await expect(page.getByText("Neural · 42 ms · Display P3")).toBeVisible();
    await expect(page.getByRole("img", { name: "Photo DSC04204, Neutral · 100%" })).toBeVisible();

    // The look: a tile click changes the chip, the tile's ring, and the photo.
    await page.getByRole("button", { name: "Teal and orange look, key 6" }).click();
    await expect(page.getByRole("img", { name: "Photo DSC04204, Teal and orange · 100%" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Teal and orange look, key 6" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator(".ym-photo-cell__chip")).toHaveText("Teal and orange · 100%");

    for (const [name, title] of [["Light", "Light"], ["Colour", "Colour"], ["Detail", "Detail"], ["Curve", "Curve"], ["RAW", "RAW"], ["Look", "Look"]] as const) {
      await tabs(page).getByRole("tab", { name }).click();
      await expect(page.getByRole("complementary", { name: "Tools" }).getByRole("heading", { level: 2, name: title })).toBeVisible();
      await expect(tabs(page).getByRole("tab", { name })).toHaveAttribute("aria-selected", "true");
    }
    expect(problems).toEqual([]);
  });

  test("every tab has its own accent, which its sliders, switches and changed values take, and the title block stays neutral glass", async ({ page }) => {
    await photos(page);
    // `--primary` is what glasscn paints a slider's range, a switch, the selected tab's underline and the focus ring from.
    const accents: Array<[string, string]> = [
      ["Look", "#ffc20e"],
      ["Light", "#2ec4f4"],
      ["Colour", "#7be35f"],
      ["Detail", "#f4b9ac"],
      ["Curve", "#ff7340"],
      ["RAW", "#5b9bff"],
    ];
    for (const [name, accent] of accents) {
      await tabs(page).getByRole("tab", { name }).click();
      await expect(tabs(page).getByRole("tab", { name })).toHaveAttribute("aria-selected", "true");
      await expect.poll(() => page.getByRole("tabpanel").evaluate((el) => getComputedStyle(el).getPropertyValue("--primary").trim())).toBe(accent);
      // Yellow is for buttons: no tab paints its title block in its colour.
      expect(await page.locator(".ym-title-block").evaluate((el) => getComputedStyle(el).backgroundColor)).not.toMatch(/^rgb\(/);
    }
  });

  test("a look picked keeps the other photos' thumbnails (Apply to starts on This photo); All reaches every photo's thumbnail", async ({ page }) => {
    await photos(page);
    const thumb = (n: number) => page.locator(".ym-filmstrip__thumb img").nth(n);
    await expect(page.getByRole("radio", { name: "This photo" })).toHaveAttribute("aria-checked", "true");
    const second = await thumb(1).getAttribute("src");
    await page.getByRole("button", { name: "Film warm look, key 3" }).click();
    await expect(page.getByRole("img", { name: /Film warm/ })).toBeVisible();
    await page.waitForTimeout(150);
    expect(await thumb(1).getAttribute("src")).toBe(second);
    await page.getByRole("radio", { name: /^All/ }).click();
    const before = await thumb(0).getAttribute("src");
    await page.getByRole("button", { name: "Night blue look, key 8" }).click();
    await expect.poll(async () => thumb(0).getAttribute("src")).not.toBe(before);
  });

  test("look thumbnails load lazily: the tiles in view first, the rest as they scroll into view, all decodable", async ({ page }) => {
    await photos(page);
    await settle(page);
    const sizes = () => page.locator(".ym-look-tile img").evaluateAll((images) => (images as HTMLImageElement[]).map((i) => [i.naturalWidth, i.naturalHeight]));
    const first = await sizes();
    expect(first.length).toBeGreaterThanOrEqual(2);
    expect(first.length).toBeLessThan(9);
    await page.locator(".ym-tools__scroll").evaluate((el) => {
      el.scrollTop = el.scrollHeight;
    });
    await expect.poll(async () => (await sizes()).length).toBe(9);
    await settle(page);
    expect((await sizes()).every(([w, h]) => w === 240 && h === 160)).toBe(true);
  });

  test("search filters the grid and counts matches", async ({ page }) => {
    await photos(page);
    await page.getByRole("searchbox", { name: "Search looks" }).fill("film");
    await expect(page.getByText("2 of 7")).toBeVisible();
    await expect(page.getByRole("button", { name: / look, key \d/ })).toHaveCount(2);
    await page.getByRole("searchbox", { name: "Search looks" }).fill("nothing like this");
    await expect(page.getByText("No looks match “nothing like this”.")).toBeVisible();
  });

  test("a look tile whose request the core cancelled is asked for again", async ({ page }) => {
    // The core cancels the tiles of a photo when the person moves to another one; a tile that comes back meanwhile gets that failure.
    await openApp(page, MOCKUP);
    const calls = await recordCalls(page, "lookThumbnail");
    await failNext(page, "lookThumbnail", { scope: "render", message: "The render was cancelled." });
    await openPhotos(page);
    await lookTilesLoaded(page);
    const asked = (await calls()).map((c) => `${c.args[0]} ${c.args[1]}`);
    expect(asked.length).toBeGreaterThan(new Set(asked).size);
  });

  test("a look tile whose picture the shell dropped is made again when the photo is visited again", async ({ page }) => {
    // The shell keeps 256 MiB of tiles and answers 404 for the addresses of the photos used least recently. The mock's addresses are
    // `blob:` URLs, so revoking them makes the next load of the same address fail the same way.
    await photos(page);
    await lookTilesLoaded(page);
    const srcs = () => page.locator(".ym-look-tile img").evaluateAll((images) => (images as HTMLImageElement[]).map((i) => i.src));
    const first = await srcs();
    expect(first.length).toBeGreaterThanOrEqual(2);
    await page.getByRole("button", { name: "Show DSC04206" }).click();
    await expect(page.getByRole("img", { name: /^Photo DSC04206/ })).toBeVisible();
    await expect.poll(async () => (await srcs()).some((src) => first.includes(src))).toBe(false);
    await lookTilesLoaded(page);
    await page.evaluate((urls) => urls.forEach((u) => URL.revokeObjectURL(u)), first);
    await page.getByRole("button", { name: /^Show DSC04204/ }).click();
    await expect(page.getByRole("img", { name: /^Photo DSC04204/ })).toBeVisible();
    await lookTilesLoaded(page);
    const again = await srcs();
    expect(again.length).toBeGreaterThanOrEqual(first.length);
    expect(again.filter((src) => first.includes(src))).toEqual([]);
  });
});

test.describe("controls with the real mouse", () => {
  test("dragging a slider sends patches while it moves and a final one on release, each a valid exposure", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "Light" }).click();
    const calls = await recordCalls(page, "updateSettings");
    await dragSlider(page, "Exposure", 0.8, 8);
    await expect(valueBox(page, "Exposure")).not.toHaveValue("+0.30");
    const patches = (await calls()).map((c) => c.args[1] as { exposure: { kind: "auto" } | { kind: "manual"; ev: number } });
    expect(patches.length).toBeGreaterThanOrEqual(2);
    // A value dropped on the detent (+0.30) is Automatic again; every other one is manual and in range.
    expect(patches.every((p) => p.exposure.kind === "auto" || (p.exposure.ev >= -4 && p.exposure.ev <= 4))).toBe(true);
    // The box shows what the core resolved: the last patch is the value on screen.
    const shown = Number((await valueBox(page, "Exposure").inputValue()).replace("\u2212", "-"));
    const last = patches.at(-1)!.exposure;
    expect(last.kind === "manual" ? last.ev : 0.3).toBeCloseTo(shown, 2);
    // Fewer patches than pointer moves is not guaranteed, but never more than one per frame.
    expect(patches.length).toBeLessThanOrEqual(12);
  });

  test("typing a value, pressing Reset and the AUTO button", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "Light" }).click();
    await setValue(page, "Exposure", "1.5");
    await expect(valueBox(page, "Exposure")).toHaveValue("+1.50");
    await expect(valueBox(page, "Exposure")).toHaveAttribute("data-changed", "true");
    // A changed value is drawn in its tab's colour (Light's #2ec4f4): the text and a ring.
    await expect(valueBox(page, "Exposure")).toHaveCSS("color", "rgb(46, 196, 244)");
    await expect(page.getByRole("region", { name: "Exposure" }).getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "false");
    await page.getByRole("button", { name: "Reset Exposure" }).click();
    await expect(valueBox(page, "Exposure")).toHaveValue("+0.30");
    await expect(page.getByRole("region", { name: "Exposure" }).getByRole("button", { name: "Auto" })).toHaveAttribute("aria-pressed", "true");
    await setValue(page, "Exposure", "-2");
    await page.getByRole("region", { name: "Exposure" }).getByRole("button", { name: "Auto" }).click();
    await expect(valueBox(page, "Exposure")).toHaveValue("+0.30");
  });

  test("Detail: the Output slider sits in the Sharpening card, snaps to steps of 5 and resets to 100, not to Auto", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "Detail" }).click();
    const card = page.getByRole("region", { name: "Sharpening" });
    for (const name of ["Amount", "Radius", "Output"]) await expect(card.getByRole("slider", { name })).toBeVisible();
    await expect(valueBox(page, "Output")).toHaveValue("100%");
    await expect(valueBox(page, "Output")).not.toHaveAttribute("data-changed", "true");
    const calls = await recordCalls(page, "updateSettings");
    await dragSlider(page, "Output", 0.2, 6);
    const patches = (await calls()).map((c) => c.args[1] as { output_sharpening: number });
    expect(patches.length).toBeGreaterThanOrEqual(2);
    // Every patch is a plain number on the grid of 5 inside 0 to 200, and the last one is on screen.
    expect(patches.every((p) => Number.isInteger(p.output_sharpening / 5) && p.output_sharpening >= 0 && p.output_sharpening <= 200)).toBe(true);
    await expect(valueBox(page, "Output")).toHaveValue(`${patches.at(-1)!.output_sharpening}%`);
    await expect(valueBox(page, "Output")).toHaveAttribute("data-changed", "true");
    // Typing, then the row's own Reset: back to 100 as a number.
    await setValue(page, "Output", "35");
    await expect(valueBox(page, "Output")).toHaveValue("35%");
    await page.getByRole("button", { name: "Reset Output" }).click();
    await expect(valueBox(page, "Output")).toHaveValue("100%");
    expect((await calls()).at(-1)!.args[1]).toEqual({ output_sharpening: 100 });
    // The card's Reset puts the whole card back, Output included.
    await setValue(page, "Output", "60");
    await card.getByRole("button", { name: "Reset" }).first().click();
    await expect(valueBox(page, "Output")).toHaveValue("100%");
    expect((await calls()).at(-1)!.args[1]).toEqual({ sharpen: { kind: "auto" }, output_sharpening: 100 });
  });

  test("a change on one tab is shown on the Look tab's Automatic card as EDITED", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "Colour" }).click();
    await page.getByRole("radio", { name: "Tungsten" }).click();
    await expect(valueBox(page, "Kelvin")).toHaveValue("3200 K");
    await tabs(page).getByRole("tab", { name: "Look" }).click();
    await expect(page.getByRole("button", { name: "White balance 3200 K · tint 0, Edited. Open Colour" })).toBeVisible();
    await page.getByRole("button", { name: "Re-run" }).click();
    await expect(page.getByRole("button", { name: "White balance 5450 K · tint 0, Auto. Open Colour" })).toBeVisible();
  });

  test("drags a colour wheel puck and resets it with a double click", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "Colour" }).click();
    const wheel = page.getByRole("slider", { name: "Midtones colour balance" });
    const box = (await wheel.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.85, box.y + box.height * 0.3, { steps: 5 });
    await page.mouse.up();
    await expect(wheel).not.toHaveAttribute("aria-valuetext", "Centred");
    await wheel.dblclick();
    await expect(wheel).toHaveAttribute("aria-valuetext", "Centred");
  });

  test("adds and drags a curve point, and the readout follows", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "Curve" }).click();
    const plot = page.getByRole("group", { name: "Luma curve" });
    const box = (await plot.boundingBox())!;
    const calls = await recordCalls(page, "updateSettings");
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height * 0.3, { steps: 6 });
    await page.mouse.up();
    await expect(plot.getByRole("slider")).toHaveCount(3);
    const last = (await calls()).at(-1)!.args[1] as { curve: { luma: number[][] } };
    expect(last.curve.luma).toHaveLength(3);
    expect(last.curve.luma[1]![1]).toBeGreaterThan(0.6);
    // Dragging the point far off the plot removes it.
    const point = plot.getByRole("slider", { name: "Point 2 of 3" });
    const p = (await point.boundingBox())!;
    await page.mouse.move(p.x + p.width / 2, p.y + p.height / 2);
    await page.mouse.down();
    await page.mouse.move(p.x + p.width / 2, box.y + box.height + 80, { steps: 5 });
    await page.mouse.up();
    await expect(plot.getByRole("slider")).toHaveCount(2);
  });

  test("the RAW switches toggle and the range sliders move", async ({ page }) => {
    await photos(page);
    await tabs(page).getByRole("tab", { name: "RAW" }).click();
    const sw = page.getByRole("switch", { name: "Neural demosaic" });
    await expect(sw).toHaveAttribute("aria-checked", "true");
    await sw.click();
    await expect(sw).toHaveAttribute("aria-checked", "false");
    // Off, the switch is neutral; on, it is the tab's colour (RAW's #5b9bff).
    await expect(sw).not.toHaveCSS("background-color", "rgb(91, 155, 255)");
    await sw.click();
    await expect(sw).toHaveCSS("background-color", "rgb(91, 155, 255)");
    await sw.click();
    await setValue(page, "Highlights", "0.8");
    await expect(valueBox(page, "Highlights")).toHaveValue("0.80");
  });

  test("Crop, Pick white and Before sit in the header with the shortcuts in their tooltips", async ({ page }) => {
    await photos(page);
    // A title is a glass tooltip and the button's description.
    await expect(page.getByRole("button", { name: "Before" })).toHaveAccessibleDescription(/\\/);
    await expect(page.getByRole("button", { name: "Crop" })).toHaveAccessibleDescription("Crop (C)");
    await expect(page.getByRole("button", { name: "Pick white" })).toHaveAccessibleDescription("Pick white (W)");
    await page.getByRole("button", { name: "Crop" }).hover();
    await expect(page.getByRole("tooltip").first()).toHaveText("Crop (C)");
  });
});

test.describe("keyboard shortcuts", () => {
  test("the shortcuts of docs/11", async ({ page }) => {
    await photos(page);
    // ⌘1 to ⌘6
    for (const [key, name] of [["2", "Light"], ["3", "Colour"], ["4", "Detail"], ["5", "Curve"], ["6", "RAW"], ["1", "Look"]] as const) {
      await page.keyboard.press(`Meta+${key}`);
      await expect(tabs(page).getByRole("tab", { name })).toHaveAttribute("aria-selected", "true");
    }
    // 1 to 9 pick looks on the Look tab.
    await page.keyboard.press("6");
    await expect(page.locator(".ym-photo-cell__chip")).toHaveText("Teal and orange · 100%");
    // ← →
    await page.keyboard.press("ArrowRight");
    await expect(page.getByText(/^5 of 12 · /)).toBeVisible();
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByText(/^3 of 12 · /)).toBeVisible();
    // P picks, ⌘A picks all
    await page.keyboard.press("p");
    await expect(page.getByText("2 picked")).toBeVisible();
    await page.keyboard.press("Meta+a");
    await expect(page.getByText("12 picked")).toBeVisible();
    await expect(page.getByRole("button", { name: /Export 12/ })).toBeVisible();
    // \ flips Before
    await page.keyboard.press("\\");
    await expect(page.getByRole("button", { name: "Before" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByText("Before · camera default")).toBeVisible();
    await page.keyboard.press("\\");
    await expect(page.getByText("Before · camera default")).toHaveCount(0);
    // Z toggles Fit and 100 %
    await page.keyboard.press("z");
    await expect(page.getByRole("button", { name: /^100%/ })).toBeVisible();
    await page.keyboard.press("z");
    await expect(page.getByRole("button", { name: /^Fit/ })).toBeVisible();
  });

  test("⌘Z and ⇧⌘Z undo and redo changes to the photo on screen", async ({ page }) => {
    await photos(page);
    await page.keyboard.press("Meta+2");
    await setValue(page, "Contrast", "30");
    await expect(valueBox(page, "Contrast")).toHaveValue("+30");
    await setValue(page, "Brightness", "-10");
    await page.keyboard.press("Meta+z");
    await expect(valueBox(page, "Brightness")).toHaveValue("0");
    await expect(valueBox(page, "Contrast")).toHaveValue("+30");
    await page.keyboard.press("Meta+z");
    await expect(valueBox(page, "Contrast")).toHaveValue("0");
    await page.keyboard.press("Meta+Shift+z");
    await expect(valueBox(page, "Contrast")).toHaveValue("+30");
  });

  test("keys that type don't act as shortcuts in a text box", async ({ page }) => {
    await photos(page);
    await page.getByRole("searchbox", { name: "Search looks" }).fill("");
    await page.getByRole("searchbox", { name: "Search looks" }).press("3");
    await page.getByRole("searchbox", { name: "Search looks" }).press("p");
    await expect(page.getByRole("searchbox", { name: "Search looks" })).toHaveValue("3p");
    await expect(page.locator(".ym-photo-cell__chip")).toHaveText("Neutral · 100%");
    await expect(page.getByText("3 picked")).toBeVisible();
  });

  test("⌘W closes the session and shows Start", async ({ page }) => {
    await photos(page);
    await page.keyboard.press("Meta+w");
    await expect(page.getByRole("group", { name: "Drop files or a folder" })).toBeVisible();
  });

  test("the zoom menu opens with the keyboard, moves with the arrows and Esc closes it", async ({ page }) => {
    await photos(page);
    const trigger = page.getByRole("button", { name: /^Fit/ });
    await trigger.focus();
    await page.keyboard.press("ArrowDown");
    const menu = page.getByRole("menu", { name: "Zoom" });
    await expect(menu).toBeVisible();
    await expect(menu.getByRole("menuitemradio", { name: "Fit" })).toBeFocused();
    // Radix moves the focus a tick after the key, so each step is waited for.
    await page.keyboard.press("ArrowDown");
    await expect(menu.getByRole("menuitemradio", { name: "50%" })).toBeFocused();
    await page.keyboard.press("ArrowDown");
    await expect(menu.getByRole("menuitemradio", { name: "100%" })).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(menu).toHaveCount(0);
    await expect(page.getByRole("button", { name: /^100%/ })).toBeFocused();
    await page.getByRole("button", { name: /^100%/ }).press("ArrowDown");
    await page.keyboard.press("Escape");
    await expect(page.getByRole("menu")).toHaveCount(0);
  });

  test("a focused look tile shows the 3 px focus ring", async ({ page }) => {
    await photos(page);
    await page.getByRole("button", { name: "Film warm look, key 3" }).focus();
    await page.keyboard.press("Shift+Tab");
    await page.keyboard.press("Tab");
    const tile = page.getByRole("button", { name: "Film warm look, key 3" });
    await expect(tile).toBeFocused();
    await expect(tile).toHaveCSS("outline-width", "3px");
  });
});

test.describe("crop and pick white by mouse", () => {
  test("Crop: aspect, straighten, a handle drag, Done; the photo is shown cropped and Undo takes it back", async ({ page }) => {
    await photos(page);
    await page.getByRole("button", { name: "Crop" }).click();
    const panel = page.getByRole("complementary", { name: "Crop" });
    await expect(panel).toBeVisible();
    await panel.getByRole("radio", { name: "Free" }).click();
    const image = page.locator(".ym-crop__image");
    const box = (await image.boundingBox())!;
    const handle = page.getByRole("button", { name: /Crop bottom right corner/ });
    const h = (await handle.boundingBox())!;
    await page.mouse.move(h.x + h.width / 2, h.y + h.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.6, { steps: 6 });
    // The thirds lines show while dragging.
    await expect(page.locator(".ym-crop")).toHaveAttribute("data-dragging", "true");
    await page.mouse.up();
    const frame = page.getByRole("group", { name: /Crop frame/ });
    await expect.poll(async () => (await frame.boundingBox())!.width / box.width).toBeCloseTo(0.7, 1);
    await setValue(page, "Straighten", "2.5");
    await expect(valueBox(page, "Straighten")).toHaveValue("+2.5°");
    await panel.getByRole("button", { name: /^Done/ }).click();
    await expect(page.getByRole("complementary", { name: "Tools" })).toBeVisible();
    // The browser build shows the cropped picture: the frame is now 0.7 × 0.6 of the photo.
    const src = await page.locator(".ym-photo-cell__image").getAttribute("src");
    expect(decodeURIComponent(src!)).toMatch(/viewBox="0 0 1050 600"/);
    await page.keyboard.press("Meta+z");
    const undone = await page.locator(".ym-photo-cell__image").getAttribute("src");
    expect(decodeURIComponent(undone!)).toMatch(/viewBox="0 0 1500 1000"/);
  });

  test("Crop: Rotate 90° and Cancel", async ({ page }) => {
    await photos(page);
    await page.keyboard.press("c");
    const panel = page.getByRole("complementary", { name: "Crop" });
    await panel.getByRole("button", { name: "Rotate 90°" }).click();
    expect(decodeURIComponent((await page.locator(".ym-photo-cell__image").getAttribute("src"))!)).toMatch(/viewBox="0 0 1000 1500"/);
    await page.keyboard.press("Escape");
    await expect(page.getByRole("complementary", { name: "Tools" })).toBeVisible();
    expect(decodeURIComponent((await page.locator(".ym-photo-cell__image").getAttribute("src"))!)).toMatch(/viewBox="0 0 1500 1000"/);
  });

  test("Pick white: crosshair cursor, header hint, a click sets the white balance, Esc cancels", async ({ page }) => {
    await photos(page);
    await page.keyboard.press("w");
    await expect(page.getByText("Click something neutral: grey or white.")).toBeVisible();
    const layer = page.getByRole("application");
    await expect(layer).toHaveCSS("cursor", "crosshair");
    await page.keyboard.press("Escape");
    await expect(layer).toHaveCount(0);
    await page.getByRole("button", { name: "Pick white" }).click();
    const box = (await layer.boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.15);
    await expect(layer.locator("svg")).toBeVisible();
    await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.15);
    await expect(page.getByRole("tab", { name: "Colour" })).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("region", { name: "White balance" }).getByText("Edited")).toBeVisible();
    await expect(valueBox(page, "Kelvin")).not.toHaveValue("5450 K");
  });

  test("holding ⌥ over the photo reads the pixel under the pointer", async ({ page }) => {
    await photos(page);
    const box = (await page.locator("[data-viewport]").boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.1);
    await page.keyboard.down("Alt");
    await expect(page.locator(".ym-photo-cell__readout")).toHaveText(/^R \d+ G \d+ B \d+$/);
    await page.keyboard.up("Alt");
    await expect(page.locator(".ym-photo-cell__readout")).toHaveCount(0);
  });
});

test.describe("error states", () => {
  test("a file that can't be opened is tagged CAN'T OPEN in the filmstrip and explained over the photo", async ({ page }) => {
    await openApp(page, { mockSkipped: "DSC04202:This file is from a camera mode we can't read yet.", mockCurrent: 0 });
    await openPhotos(page);
    const thumb = page.getByRole("button", { name: "Show DSC04202, can't open" });
    await expect(thumb).toContainText("Can't open");
    await expect(thumb.locator('[data-slot="badge"]')).toHaveAttribute("data-variant", "cant-open");
    await thumb.click();
    await expect(page.getByText("This file can't be opened.", { exact: true })).toBeVisible();
    await expect(page.locator(".ym-photo-note").getByText("This file is from a camera mode we can't read yet.")).toBeVisible();
    // Nothing to adjust, and it can't be picked.
    await expect(page.getByRole("region", { name: "Automatic" })).toHaveCount(0);
    await page.keyboard.press("p");
    await expect(page.getByText("3 picked")).toHaveCount(0);
    // Other photos still work.
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();
  });

  test("a look whose file is missing is disabled, says why, and can't be picked", async ({ page }) => {
    await photos(page, { mockMissing: "film-warm" });
    const tile = page.getByRole("button", { name: /Film warm look, key 3, file missing or changed/ });
    await expect(tile).toHaveAttribute("aria-disabled", "true");
    await expect(tile).toContainText("File missing or changed. Import again.");
    // Playwright won't click a disabled control, so press it anyway.
    await tile.click({ force: true });
    await page.keyboard.press("3");
    await expect(page.locator(".ym-photo-cell__chip")).toHaveText("Neutral · 100%");
  });

  test("no ARRI looks imported: the group offers the wizard instead", async ({ page }) => {
    await photos(page, { mockLooks: 0 });
    await expect(page.getByText("The free ARRI Look Library is not imported yet.")).toBeVisible();
    await page.getByRole("button", { name: "Import ARRI looks →" }).click();
    await expect(page.getByRole("heading", { name: "Choose what to import" })).toBeVisible();
  });

  test("an error from the core shows over the photos and can be dismissed", async ({ page }) => {
    await photos(page);
    await failNext(page, "updateSettings", { scope: "render", message: "The photo couldn't be drawn. Try the change again." });
    await page.keyboard.press("Meta+2");
    await setValue(page, "Contrast", "40");
    await expect(page.getByRole("alert")).toHaveText(/The photo couldn't be drawn\. Try the change again\./);
    // The control goes back to what the core has.
    await expect(valueBox(page, "Contrast")).toHaveValue("0");
    await page.getByRole("alert").getByRole("button", { name: "Dismiss" }).click();
    await expect(page.getByRole("alert")).toHaveCount(0);
  });

  test("the update cell shows each state of the updater, and a ready update waits for an export", async ({ page }) => {
    await photos(page);
    await setUpdateState(page, { phase: "available", version: "0.1.1" });
    await expect(page.getByRole("button", { name: "Update available" })).toBeVisible();
    await setUpdateState(page, { phase: "downloading", version: "0.1.1", percent: 42 });
    await expect(page.getByText("Downloading 42%")).toBeVisible();
    await setUpdateState(page, { phase: "error", message: "No network." });
    await expect(page.getByText("Update failed")).toHaveAttribute("title", "No network.");
    await showUpdateReady(page);
  });
});

test.describe("export cell", () => {
  test("says Export, or Export N, and opens the export panel placeholder with ⌘E", async ({ page }) => {
    await photos(page);
    await expect(page.getByRole("button", { name: /Export 3/ })).toBeVisible();
    await page.keyboard.press("Meta+e");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("complementary", { name: "Tools" })).toBeVisible();
  });
});

test("a folder of 500 photos: the first one opens while the rest develop, and the arrows walk the whole folder", async ({ page }) => {
  // Development stops after the first 40 so the test has a steady state; the session still holds all 500.
  await openApp(page, { mockPhotos: 500, mockHold: 40 });
  await page.getByRole("button", { name: "Open files…" }).click();
  await expect(page.getByRole("heading", { level: 2, name: "Developing 500 photos" })).toBeVisible();
  await expect(page.getByRole("list", { name: "Photos being developed" }).getByText("Ready", { exact: true })).toHaveCount(40);
  await page.getByRole("button", { name: /Open photos/ }).click();
  await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();
  await expect(page.getByText("2026-10-04 · 500 photos")).toBeVisible();
  await expect(page.locator(".ym-filmstrip__caption")).toHaveText("1 of 500 · 0 picked");
  await page.keyboard.press("ArrowRight");
  await expect(page.locator(".ym-filmstrip__caption")).toHaveText("2 of 500 · 0 picked");
  await page.keyboard.press("End");
  // A photo that is still developing says so and has nothing to adjust.
  await page.getByRole("button", { name: "Show DSC04700, developing" }).click();
  await expect(page.locator(".ym-filmstrip__caption")).toHaveText("500 of 500 · 0 picked");
  await expect(page.getByText("Still developing.")).toBeVisible();
  await expect(page.getByRole("region", { name: "Automatic" })).toHaveCount(0);
  // The window still takes keys: ⌘A picks the photos that are ready, not the ones still developing.
  await page.keyboard.press("Meta+a");
  await expect(page.getByText("40 picked")).toBeVisible();
});

test("settings made before the session opens come through the mock (initialSettings)", async ({ page }) => {
  await openApp(page, MOCKUP);
  await configureMock(page, { initialSettings: { DSC04204: { highlights: manual(-18), shadows: manual(12), saturation: manual(6) } } });
  await openPhotos(page);
  await page.keyboard.press("Meta+2");
  await expect(valueBox(page, "Highlights")).toHaveValue("−18");
  await expect(valueBox(page, "Shadows")).toHaveValue("+12");
  await page.keyboard.press("Meta+3");
  await expect(valueBox(page, "Saturation")).toHaveValue("+6");
});
