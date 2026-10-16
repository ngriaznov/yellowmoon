import { expect, test } from "@playwright/test";

// Smoke test of the app shell: it boots in a browser on the mocked API and shows the Start screen
// with the moon mark, in Figtree. The screens themselves are tested in start.spec.ts and photos.spec.ts.

test("shows the Start screen", async ({ page }) => {
  const problems: string[] = [];
  page.on("pageerror", (error) => problems.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(message.text());
  });

  await page.goto("/");

  await expect(page).toHaveTitle("Yellowmoon");
  const title = page.locator(".ym-brand-title");
  await expect(title).toHaveText("Yellowmoon");
  await expect(title).toHaveCSS("font-weight", "800");

  // The sub-line is written in sentence case and drawn in capitals, as in the mockup.
  const subline = page.getByText("Drop ARW files to start");
  await expect(subline).toBeVisible();
  await expect(subline).toHaveCSS("text-transform", "uppercase");

  await expect(page.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeVisible();

  // The moon: a yellow circle with a 2 px ink border.
  const moon = title.locator(".ym-moon");
  await expect(moon).toBeVisible();
  await expect(moon).toHaveCSS("background-color", "rgb(255, 194, 14)");
  await expect(moon).toHaveCSS("border-top-width", "2px");
  await expect(moon).toHaveCSS("border-radius", "50%");

  expect(problems).toEqual([]);
});

test("uses the bundled Figtree font on a dark page", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("[data-screen]")).toBeVisible();
  // The page is dark (`index.html`): the colour scheme says so, the ground is the aurora's near-black, and the text is near-white.
  await expect(page.locator("html")).toHaveCSS("color-scheme", "dark");
  const channels = (css: string) =>
    page.evaluate((color) => {
      const ctx = document.createElement("canvas").getContext("2d", { willReadFrequently: true })!;
      ctx.fillStyle = color;
      ctx.fillRect(0, 0, 1, 1);
      return [...ctx.getImageData(0, 0, 1, 1).data].slice(0, 3);
    }, css);
  const ground = await channels(await page.locator('[data-slot="aurora"]').evaluate((el) => getComputedStyle(el).backgroundColor));
  const text = await channels(await page.locator("body").evaluate((el) => getComputedStyle(el).color));
  expect(Math.max(...ground), "the ground's brightest channel").toBeLessThan(40);
  expect(Math.min(...text), "the text's dimmest channel").toBeGreaterThan(235);

  const loaded = await page.evaluate(async () => {
    const faces = await document.fonts.load('800 24px "Figtree"');
    return faces.map((face) => `${face.family}:${face.status}`);
  });
  expect(loaded.length).toBeGreaterThan(0);
  expect(loaded.every((face) => face.endsWith(":loaded"))).toBe(true);
});

test("keyboard focus shows the 3 px focus ring", async ({ page }) => {
  await page.goto("/");
  // The global rule applies to any button: check it on a probe that comes first in the tab order.
  await page.evaluate(() => {
    const button = document.createElement("button");
    button.textContent = "Probe";
    document.body.prepend(button);
  });
  await page.keyboard.press("Tab");
  const probe = page.getByRole("button", { name: "Probe" });
  await expect(probe).toBeFocused();
  await expect(probe).toHaveCSS("outline-width", "3px");
  await expect(probe).toHaveCSS("outline-offset", "2px");
  // The focus blue of the dark theme (`--focus` in tokens.css).
  await expect(probe).toHaveCSS("outline-color", "rgb(13, 183, 232)");
});
