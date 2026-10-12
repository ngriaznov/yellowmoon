import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { MockOptions } from "../src/api.mock";
import { configureMock, lookTilesLoaded, openApp, openPhotos, showUpdateReady } from "./helpers/app";
import { compareScreen, isCopyOfDesign } from "./helpers/compare";

// Screenshot comparisons of the screens against the mockups (docs/11 §Tests): the app at 1440 × 900 at 2×
// compared with `design/screenshots/*.png`, whose copies sit in `e2e/__screenshots__/`. `toHaveScreenshot` is the
// assertion (default threshold 0.2, at most 2 % of the pixels allowed to differ, see playwright.config.ts). The
// share of pixels that differ by more than 24 on a channel, the measure of the M3-01 report, is printed and
// kept in `node_modules/.cache/yellowmoon-e2e/screenshot-diffs/<state>.json` for each state.
//
// The mock photos are the mockups' flat-shape photos, graded with the mockup's own grading function, so the
// comparison is meaningful. What stays different: the photo drawn as SVG (anti-aliasing), the thumbnails as PNGs,
// "Fit 24%" (the mockup says 34%), the histogram and the pixel readout (made-up numbers in the mockup), and,
// on the Curve tab, the histogram behind the plot.

const MOCKUP: Record<string, string | number> = { mockCurrent: 3, mockPicked: "2,3,4" };

/** The mockup's curves (`src/dev/demoData.ts` has the same numbers; that file needs Vite, so they are repeated here). */
const CURVE_DEMO_POINTS = {
  luma: [[0, 0], [0.25, 0.191], [0.5, 141 / 255], [0.75, 0.845], [1, 1]],
  red: [[0, 0], [0.25, 0.2], [0.5, 0.57], [0.75, 0.86], [1, 1]],
  green: [[0, 0], [0.5, 0.5], [1, 1]],
  blue: [[0, 0], [0.25, 0.1], [0.5, 0.45], [0.75, 0.82], [1, 1]],
} satisfies Record<string, Array<[number, number]>>;

/** What the mockup's 4th photo has been given: Light's Highlights and Shadows, Colour's wheels and Saturation, a curve. */
const MOCKUP_SETTINGS: MockOptions["initialSettings"] = {
  DSC04204: {
    highlights: -18,
    shadows: 12,
    saturation: 6,
    color_balance: { shadows: [-0.08, -0.12], midtones: [0, 0], highlights: [0.12, 0.12] },
    curve: { luma: CURVE_DEMO_POINTS.luma, red: CURVE_DEMO_POINTS.red, green: CURVE_DEMO_POINTS.green, blue: CURVE_DEMO_POINTS.blue },
  },
};

test("the baselines are byte-for-byte copies of design/screenshots", () => {
  for (const name of ["start-idle", "start-busy", "start-done", "photos-look", "photos-light", "photos-color", "photos-detail", "photos-curve", "photos-raw", "photos-before", "photos-look-teal", "photos-look-search"]) {
    expect(isCopyOfDesign(name), `${name}.png is a copy`).toBe(true);
  }
});

test.describe("Start", () => {
  test("idle (start-idle.png)", async ({ page }) => {
    await openApp(page, { mockLooks: 0 });
    await showUpdateReady(page);
    await compareScreen(page, "start-idle");
  });

  test("developing, 5 of 12 ready (start-busy.png)", async ({ page }) => {
    await openApp(page, { mockHold: 5.4, mockLooks: 0 });
    await showUpdateReady(page);
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("list", { name: "Photos being developed" }).getByText("Ready", { exact: true })).toHaveCount(5);
    await compareScreen(page, "start-busy");
  });

  test("all 12 ready (start-done.png)", async ({ page }) => {
    await openApp(page, { mockLooks: 0 });
    await showUpdateReady(page);
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    await compareScreen(page, "start-done");
  });
});

test.describe("Photos", () => {
  async function photos(page: Page): Promise<void> {
    await openApp(page, MOCKUP);
    await configureMock(page, { initialSettings: MOCKUP_SETTINGS });
    await openPhotos(page);
    await showUpdateReady(page);
    await expect(page.getByText("Neural · 42 ms · Display P3")).toBeVisible();
    // The look tiles ask for their thumbnails as they scroll into view; wait for every one that shows.
    await lookTilesLoaded(page);
    await expect(page.locator(".ym-filmstrip__thumb img")).toHaveCount(12);
  }

  const tab = (page: Page, name: string) => page.getByRole("tab", { name }).click();

  test("Look (photos-look.png)", async ({ page }) => {
    await photos(page);
    await compareScreen(page, "photos-look");
  });

  test("Light (photos-light.png)", async ({ page }) => {
    await photos(page);
    await tab(page, "Light");
    await expect(page.locator(".ym-histogram path").first()).toHaveAttribute("d", /L/);
    await compareScreen(page, "photos-light");
  });

  test("Colour (photos-color.png)", async ({ page }) => {
    await photos(page);
    await tab(page, "Colour");
    await compareScreen(page, "photos-color");
  });

  test("Detail (photos-detail.png)", async ({ page }) => {
    await photos(page);
    await tab(page, "Detail");
    await compareScreen(page, "photos-detail");
  });

  test("Curve (photos-curve.png)", async ({ page }) => {
    await photos(page);
    await tab(page, "Curve");
    await expect(page.getByRole("group", { name: "Luma curve" }).getByRole("slider")).toHaveCount(5);
    await compareScreen(page, "photos-curve");
  });

  test("RAW (photos-raw.png)", async ({ page }) => {
    await photos(page);
    await tab(page, "RAW");
    await compareScreen(page, "photos-raw");
  });

  test("Before (photos-before.png)", async ({ page }) => {
    await photos(page);
    await page.getByRole("button", { name: "Before" }).click();
    await expect(page.getByText("Before · camera log")).toBeVisible();
    await compareScreen(page, "photos-before");
  });

  test("Look: Teal and orange picked, panel scrolled to the library (photos-look-teal.png)", async ({ page }) => {
    await photos(page);
    const thumbs = () => page.locator(".ym-filmstrip__thumb img").evaluateAll((imgs) => imgs.map((img) => (img as HTMLImageElement).src));
    const before = await thumbs();
    await page.getByRole("button", { name: "Teal and orange look, key 6" }).click();
    await expect(page.locator(".ym-photo-cell__chip")).toHaveText("Teal and orange · 100%");
    // The mockup was captured scrolled by 357 px, 3 px short of the very bottom of its panel (360 px of travel).
    await page.locator(".ym-tools__scroll").evaluate((el) => {
      el.scrollTop = 357;
    });
    await expect.poll(async () => page.locator(".ym-look-tile img").count()).toBe(9);
    await lookTilesLoaded(page);
    // The filmstrip follows the look, as "Apply to: All" says: every thumbnail is a new picture.
    await expect.poll(async () => {
      const now = await thumbs();
      return now.length === before.length && now.every((src, i) => src !== before[i]);
    }).toBe(true);
    await compareScreen(page, "photos-look-teal");
  });

  test("Look: search \"film\" (photos-look-search.png)", async ({ page }) => {
    await photos(page);
    await page.getByRole("searchbox", { name: "Search looks" }).fill("film");
    await expect(page.getByText("2 of 7")).toBeVisible();
    await expect.poll(async () => page.locator(".ym-look-tile img").count()).toBe(2);
    await lookTilesLoaded(page);
    await compareScreen(page, "photos-look-search");
  });
});
