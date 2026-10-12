import { expect, test } from "@playwright/test";

// The in-memory API, running in a real browser: development timers, PNG thumbnails and SVG
// fixtures are only convincing once Chromium has decoded them.

const MOCK = "/src/api.mock.ts";

test.beforeEach(async ({ page }) => {
  // Fifty times faster than the mockups: 12 photos develop in about 80 ms.
  await page.goto("/?mockSpeed=50");
  await expect(page.getByRole("heading", { level: 1, name: "Yellowmoon" })).toBeVisible();
});

test("develops twelve photos and sends thumbnails Chromium can decode", async ({ page }) => {
  const result = await page.evaluate(async (path) => {
    const { mockApi } = (await import(/* @vite-ignore */ path)) as typeof import("../src/api.mock");
    const sizes: Array<[number, number]> = [];
    const decoded: Promise<void>[] = [];
    mockApi.on("thumbnail_ready", ({ png_bytes }) => {
      decoded.push(
        createImageBitmap(new Blob([new Uint8Array(png_bytes)], { type: "image/png" })).then((bitmap) => {
          sizes.push([bitmap.width, bitmap.height]);
        }),
      );
    });
    const state = await mockApi.openDialog("folder");
    await new Promise<void>((resolve) => {
      const stop = mockApi.on("photo_progress", (e) => {
        if (e.id === 12 && e.stage === "thumbnail" && e.done) {
          stop();
          resolve();
        }
      });
    });
    await Promise.all(decoded);
    const after = await mockApi.sessionState();
    return {
      count: state?.photos.length,
      statuses: [...new Set(after.photos.map((p) => p.status.kind))],
      sizes,
    };
  }, MOCK);

  expect(result.count).toBe(12);
  expect(result.statuses).toEqual(["ready"]);
  expect(result.sizes).toHaveLength(12);
  expect(result.sizes.every(([w, h]) => w === 240 && h === 160)).toBe(true);
});

test("serves a 3:2 SVG fixture for every photo", async ({ page }) => {
  const sizes = await page.evaluate(async (path) => {
    const { mockApi } = (await import(/* @vite-ignore */ path)) as typeof import("../src/api.mock");
    await mockApi.openDialog("files");
    const out: Array<[number, number]> = [];
    for (let id = 1; id <= 12; id++) {
      const url = mockApi.fixtureUrl(id);
      if (!url) throw new Error(`no fixture for photo ${id}`);
      const image = new Image();
      image.src = url;
      await image.decode();
      out.push([image.naturalWidth, image.naturalHeight]);
    }
    return out;
  }, MOCK);
  expect(sizes).toHaveLength(12);
  expect(sizes.every(([w, h]) => w === 1500 && h === 1000)).toBe(true);
});

test("runs an export to completion", async ({ page }) => {
  const last = await page.evaluate(async (path) => {
    const { mockApi } = (await import(/* @vite-ignore */ path)) as typeof import("../src/api.mock");
    await mockApi.openDialog("files");
    await mockApi.setPicked(3, true);
    await mockApi.setPicked(4, true);
    await mockApi.setPicked(5, true);
    const job = (await mockApi.preferencesGet()).export_defaults;
    return await new Promise<{ file: string; done: number; total: number }>((resolve) => {
      const stop = mockApi.on("export_progress", (e) => {
        if (e.done === e.total) {
          stop();
          resolve({ file: e.file, done: e.done, total: e.total });
        }
      });
      void mockApi.exportStart({ scope: "picked", options: job });
    });
  }, MOCK);
  expect(last).toEqual({ file: "DSC04205_neutral.jpg", done: 3, total: 3 });
});
