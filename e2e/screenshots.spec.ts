import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { MockOptions } from "../src/api.mock";
import { configureMock, lookTilesLoaded, openApp, openPhotos, showUpdateReady } from "./helpers/app";
import { compareScreen } from "./helpers/compare";
import { WARP_PINS } from "./helpers/states";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

// Screenshot comparisons of the screens (docs/11 §Tests): the app at 1440 × 900 at 2× compared with `design/screenshots-c2/*.png`,
// the references of the dark glass interface (C2). They are the app's own pictures on the mock API, taken with the browser flags of
// `playwright.config.ts` (`npx playwright test --update-snapshots` retakes them; a person looks at them before they are committed).
// `toHaveScreenshot` is the assertion (threshold 0.2, a share of pixels allowed to differ, see playwright.config.ts and the budgets in
// `helpers/compare.ts`). The share of pixels that differ by more than 24 on a channel is printed and kept in
// `node_modules/.cache/yellowmoon-e2e/screenshot-diffs/<state>.json` for each state.
//
// The mock photos are flat-shape photos graded with the mock's own grading function, so the picture is the same on every run.
// Nothing moves while the picture is taken: animations are off and the aurora is still.

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
    highlights: manual(-18),
    shadows: manual(12),
    saturation: manual(6),
    color_balance: {
      shadows: { kind: "manual", point: [-0.08, -0.12] },
      midtones: [0, 0],
      highlights: { kind: "manual", point: [0.12, 0.12] },
    },
    curve: { luma: CURVE_DEMO_POINTS.luma, red: CURVE_DEMO_POINTS.red, green: CURVE_DEMO_POINTS.green, blue: CURVE_DEMO_POINTS.blue },
  },
};

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
  async function photos(page: Page, settings: MockOptions["initialSettings"] = MOCKUP_SETTINGS): Promise<void> {
    await openApp(page, MOCKUP);
    await configureMock(page, { initialSettings: settings });
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

  test("Colour warp: two moving pins and a lock, the card in view, its first pin selected (photos-color-warp.png)", async ({ page }) => {
    // The owner's request of 2026-10-17 (docs/11 §Colour warp): the sky's blue moved toward teal, the sun's warm white toward orange, a
    // lock in the magentas. The photo shows the move (the mock warps its flat colours, src/fixtures/warp.ts).
    await photos(page, { DSC04204: { ...MOCKUP_SETTINGS?.DSC04204, color_warp: { pins: WARP_PINS as never } } });
    await tab(page, "Colour");
    // The photo's colours have been drawn behind the plane.
    await expect(page.locator('.ym-warp-plane:not([data-scatter-points="0"])')).toBeVisible();
    await page.getByRole("slider", { name: "Pin 1" }).focus();
    await page.locator(".ym-tools__scroll").evaluate((el) => {
      el.scrollTop = el.scrollHeight;
    });
    await expect(page.getByRole("textbox", { name: "Range value" })).toBeInViewport();
    await compareScreen(page, "photos-color-warp");
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
    await expect(page.getByText("Before · camera default")).toBeVisible();
    await compareScreen(page, "photos-before");
  });

  test("Look: Teal and orange picked, panel scrolled to the library (photos-look-teal.png)", async ({ page }) => {
    await photos(page);
    const thumbs = () => page.locator(".ym-filmstrip__thumb img").evaluateAll((imgs) => imgs.map((img) => (img as HTMLImageElement).src));
    const before = await thumbs();
    await page.getByRole("button", { name: "Teal and orange look, key 6" }).click();
    await expect(page.locator(".ym-photo-cell__chip")).toHaveText("Teal and orange · 100%");
    // Scrolled to 357 px, 3 px short of the very bottom of the panel's travel (the state the pre-glass screenshot was taken in).
    await page.locator(".ym-tools__scroll").evaluate((el) => {
      el.scrollTop = 357;
    });
    await expect.poll(async () => page.locator(".ym-look-tile img").count()).toBe(9);
    await lookTilesLoaded(page);
    // Apply to starts on This photo (the owner's decision of 2026-10-20): only the photo on screen has a new thumbnail.
    await expect.poll(async () => {
      const now = await thumbs();
      return now.length === before.length && now.filter((src, i) => src !== before[i]).length === 1;
    }).toBe(true);
    await compareScreen(page, "photos-look-teal");
  });

  test("Look with Apple Auto's answer: the Automatic card's rows for the values it set, its line and plan (photos-look-apple-auto.png)", async ({ page }) => {
    // docs/19 section 10, not in the mockup: the state of a Mac where Apple Auto is available and has answered (the mock's answer is a
    // little brighter and warmer, darker highlights, deeper blacks, more contrast, less saturation, teal shadows and warm highlights).
    // The photo keeps only the mockup's curve, so every row shows Apple's value as Auto. Without Apple Auto the Look tab is
    // photos-look.png, unchanged.
    await openApp(page, { ...MOCKUP, mockAppleAuto: "answered" });
    await configureMock(page, { initialSettings: { DSC04204: { curve: MOCKUP_SETTINGS?.DSC04204?.curve } } });
    await openPhotos(page);
    await showUpdateReady(page);
    await expect(page.getByText("Neural · 42 ms · Display P3")).toBeVisible();
    await expect(page.getByText("Set with Apple Intelligence.")).toBeVisible();
    // The card is six rows and two lines taller, so the look tiles start below the fold: wait only for the ones that show.
    await page.waitForFunction(() => {
      const box = document.querySelector(".ym-tools__scroll")?.getBoundingClientRect();
      if (!box) return false;
      return [...document.querySelectorAll<HTMLElement>(".ym-look-tile")]
        .filter((tile) => tile.getBoundingClientRect().top < box.bottom && tile.getBoundingClientRect().bottom > box.top)
        .every((tile) => {
          const img = tile.querySelector("img");
          return !!img && img.complete && img.naturalWidth > 0;
        });
    });
    await expect(page.locator(".ym-filmstrip__thumb img")).toHaveCount(12);
    await compareScreen(page, "photos-look-apple-auto");
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
