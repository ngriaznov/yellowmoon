import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, settle, showUpdateReady, watchErrors } from "./helpers/app";
import { pixelDiff } from "./helpers/pixelDiff";

// The photo cell under each viewport backend (M3-04, M0-06 frontend side) in a real browser against the mocked API. The mock plays the
// app's two backends: `?mockViewport=native` (a transparent cell, the picture is behind the webview) and `?mockViewport=fallback` (the
// shell sends a picture of the cell as `viewport_frame` events). Without it the page draws the mock's fixture, as a browser build does.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

type Viewport = { rect: { x: number; y: number; width: number; height: number; scale: number } | null; inputs: Array<Record<string, unknown>>; mode: { kind: string } };

/** What the mock has recorded of the viewport commands. */
const viewportOf = (page: Page): Promise<Viewport> =>
  page.evaluate(() => JSON.parse(JSON.stringify((window as unknown as { __yellowmoon: { mockApi: { viewport: unknown } } }).__yellowmoon.mockApi.viewport)) as Viewport);

async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await openPhotos(page);
}

const cell = (page: Page) => page.locator("[data-viewport]");

test.describe("the native viewport", () => {
  test("leaves the cell transparent with nothing drawn in it, and tells the core where the cell is", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page, { mockViewport: "native" });
    await expect(cell(page)).toHaveAttribute("data-native", "true");
    await expect(cell(page)).toHaveCSS("background-color", "rgba(0, 0, 0, 0)");
    await expect(cell(page).locator("img")).toHaveCount(0);
    // The chip and the camera line are the webview's own and stay.
    await expect(cell(page).locator(".ym-photo-cell__chip")).toHaveText("Neutral · 100%");
    const box = (await cell(page).boundingBox())!;
    await expect.poll(async () => (await viewportOf(page)).rect).toMatchObject({ scale: 2 });
    const rect = (await viewportOf(page)).rect!;
    expect(rect.x).toBeCloseTo(box.x, 1);
    expect(rect.y).toBeCloseTo(box.y, 1);
    expect(rect.width).toBeCloseTo(box.width, 1);
    expect(rect.height).toBeCloseTo(box.height, 1);
    expect(problems).toEqual([]);
  });

  test("the page paints nothing over the cell: every box under it is clear, and the aurora is cut out of it", async ({ page }) => {
    // The photo is a layer behind the webview (docs/01 §Native viewport), so anything the page paints where the cell is would cover
    // it. Since the dark glass interface the page has more that paints (an aurora under the panes, glass surfaces, a blurred backdrop);
    // none of it may reach the cell. Checked two ways: the boxes at five points of the cell, and the pixels of a capture taken
    // without the page's own background.
    await photos(page, { mockViewport: "native" });
    await expect(page.locator("html")).toHaveAttribute("data-native-viewport", "");
    const box = (await cell(page).boundingBox())!;
    const points = [
      [0.5, 0.5],
      [0.25, 0.5],
      [0.75, 0.5],
      [0.5, 0.3],
      [0.5, 0.7],
    ].map(([fx, fy]) => ({ x: Math.round(box.x + box.width * fx!), y: Math.round(box.y + box.height * fy!) }));
    const boxes = await page.evaluate((at) => {
      const name = (e: Element) => e.tagName.toLowerCase() + (e.id ? `#${e.id}` : "") + [...e.classList].map((c) => `.${c}`).join("");
      return at.flatMap(({ x, y }) =>
        document.elementsFromPoint(x, y).map((e) => {
          const css = getComputedStyle(e);
          return { at: `${x},${y}`, element: name(e), color: css.backgroundColor, image: css.backgroundImage, backdrop: css.backdropFilter || css.getPropertyValue("-webkit-backdrop-filter") || "none" };
        }),
      );
    }, points);
    expect(boxes.length).toBeGreaterThan(points.length * 3);
    for (const b of boxes) {
      expect(b, `${b.element} at ${b.at} paints a background`).toMatchObject({ color: "rgba(0, 0, 0, 0)", image: "none" });
      expect(b.backdrop, `${b.element} at ${b.at} blurs what is behind it`).toBe("none");
    }
    // In pixels: transparent inside the cell (nothing painted), opaque in the header above it (the glass paints there).
    const png = await page.screenshot({ omitBackground: true, animations: "disabled" });
    const alpha = await page.evaluate(
      async ({ data, at, scale }) => {
        const bitmap = await createImageBitmap(await (await fetch(`data:image/png;base64,${data}`)).blob());
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(bitmap, 0, 0);
        return at.map(({ x, y }) => ctx.getImageData(Math.round(x * scale), Math.round(y * scale), 1, 1).data[3]!);
      },
      { data: png.toString("base64"), at: [...points, { x: 720, y: 24 }], scale: 2 },
    );
    expect(alpha.slice(0, points.length), "alpha inside the cell").toEqual(points.map(() => 0));
    expect(alpha[points.length], "alpha in the header").toBeGreaterThan(0);
  });

  test("the Mac job's description of the page sees the aurora cut out at the cell (SMOKE_DESCRIBE_PAGE, run here as it runs there)", async ({ page }) => {
    // The `viewport` job of build.yml is the only guard on a Mac. It runs this script in the app's own web view and reads the result in
    // src-tauri/src/app.rs (`page_paints_over_cell`). The aurora takes no pointer events, so the boxes under the cell never include it:
    // the script reads its mask and the cell's variables itself. Run here, the script must see what the pixel check above sees.
    await photos(page, { mockViewport: "native" });
    const source = readFileSync(resolve(import.meta.dirname, "../src-tauri/src/app.rs"), "utf8");
    const script = /const SMOKE_DESCRIBE_PAGE: &str = r##"([\s\S]*?)"##;/.exec(source)?.[1];
    expect(script, "SMOKE_DESCRIBE_PAGE in src-tauri/src/app.rs").toBeDefined();
    const described = JSON.parse((await page.evaluate(script as string)) as string) as {
      cell: [number, number, number, number];
      stack: string[];
      paints: Array<{ element: string; image: string; backdrop: string }>;
      aurora: { composite: string; size: string; position: string } | null;
      cell_vars: string[];
    };
    expect(described.stack.length).toBeGreaterThan(2);
    for (const entry of described.stack) expect(entry, "a box over the cell").toMatch(/rgba\(0, 0, 0, 0\)$/);
    expect(described.paints.length).toBe(described.stack.length);
    for (const paint of described.paints) {
      expect(paint, `${paint.element} paints an image`).toMatchObject({ image: "none", backdrop: "none" });
    }
    expect(described.cell_vars, "--cell-x/y/w/h").toHaveLength(4);
    for (const value of described.cell_vars) expect(value).toMatch(/^[\d.]+px$/);
    expect(described.aurora, "the aurora").not.toBeNull();
    // One entry for each of the two layers ("exclude, exclude"); the first is composited with the layer below it.
    expect(["exclude", "xor"]).toContain(described.aurora!.composite.split(",")[0]!.trim());
    // The second layer of the mask is the cell, in pixels.
    const px = (layer: string | undefined): number[] => (layer ?? "").trim().split(/\s+/).map((n) => parseFloat(n));
    const [, size] = described.aurora!.size.split(",");
    const [, position] = described.aurora!.position.split(",");
    const [left, top, width, height] = described.cell;
    px(size).forEach((n, i) => expect(n).toBeCloseTo([width, height][i]!, 0));
    px(position).forEach((n, i) => expect(n).toBeCloseTo([left, top][i]!, 0));
    expect(px(size)).toHaveLength(2);
    expect(px(position)).toHaveLength(2);
  });

  test("follows the cell when the window is resized", async ({ page }) => {
    await photos(page, { mockViewport: "native" });
    await expect.poll(async () => (await viewportOf(page)).rect?.width).toBeCloseTo((await cell(page).boundingBox())!.width, 1);
    await page.setViewportSize({ width: 1800, height: 1100 });
    await expect.poll(async () => (await viewportOf(page)).rect?.width).toBeCloseTo((await cell(page).boundingBox())!.width, 1);
    const wide = (await viewportOf(page)).rect!;
    expect(wide.width / wide.height).toBeCloseTo(1.5, 2);
    await page.setViewportSize({ width: 1180, height: 760 });
    await expect.poll(async () => (await viewportOf(page)).rect?.width).toBeCloseTo((await cell(page).boundingBox())!.width, 1);
    expect((await viewportOf(page)).rect!.width).toBeLessThan(wide.width);
  });
});

test.describe("the fallback viewport", () => {
  test("shows the shell's picture of the cell, one picture pixel to one device pixel", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page, { mockViewport: "fallback" });
    const frame = cell(page).locator("img.ym-photo-cell__frame");
    await expect(frame).toHaveCount(1);
    await expect(cell(page)).not.toHaveAttribute("data-native", "true");
    await settle(page);
    const box = (await cell(page).boundingBox())!;
    const shown = (await frame.boundingBox())!;
    expect(shown.width).toBeCloseTo(box.width, 0);
    expect(shown.height).toBeCloseTo(box.height, 0);
    const natural = await frame.evaluate((img: HTMLImageElement) => ({ w: img.naturalWidth, h: img.naturalHeight }));
    expect(natural.w).toBe(Math.round(box.width * 2));
    expect(natural.h).toBe(Math.round(box.height * 2));
    expect(problems).toEqual([]);
  });

  test("draws the same picture as the browser build does (the mock's frame and its fixture agree)", async ({ page, browser }) => {
    await photos(page);
    await settle(page);
    const fixture = await cell(page).screenshot({ animations: "disabled" });
    await photos(page, { mockViewport: "fallback" });
    await expect(cell(page).locator("img.ym-photo-cell__frame")).toHaveCount(1);
    await settle(page);
    const frame = await cell(page).screenshot({ animations: "disabled" });
    const diff = await pixelDiff(browser, fixture, frame, { threshold: 24 });
    console.log(`[viewport] fixture against frame: ${(diff.mismatch * 100).toFixed(3)} % of pixels differ by more than 24`);
    expect(diff.mismatch).toBeLessThan(0.005);
  });

  test("sends a new picture for a new photo, for a change of the look, and for a new size of the cell", async ({ page }) => {
    await photos(page, { mockViewport: "fallback" });
    const frame = cell(page).locator("img.ym-photo-cell__frame");
    const src = async (): Promise<string> => (await frame.getAttribute("src")) ?? "";
    const first = await src();
    await page.keyboard.press("ArrowRight");
    await expect.poll(src).not.toBe(first);
    const second = await src();
    await page.getByRole("button", { name: /Signature/ }).first().click();
    await expect.poll(src).not.toBe(second);
    const before = (await frame.evaluate((img: HTMLImageElement) => img.naturalWidth)) as number;
    await page.setViewportSize({ width: 1800, height: 1100 });
    await expect.poll(() => frame.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(before);
    const box = (await cell(page).boundingBox())!;
    await expect.poll(() => frame.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBe(Math.round(box.width * 2));
  });

  test("takes the picture away for a file that can't be opened, and brings it back for the next photo", async ({ page }) => {
    await photos(page, { mockViewport: "fallback", mockSkipped: "DSC04205:This file is from a camera mode we can't read yet." });
    const frame = cell(page).locator("img.ym-photo-cell__frame");
    await expect(frame).toHaveCount(1);
    await page.keyboard.press("ArrowRight");
    await expect(page.getByText("This file can't be opened.", { exact: true })).toBeVisible();
    await expect(frame).toHaveCount(0);
    await page.keyboard.press("ArrowRight");
    await expect(frame).toHaveCount(1);
    await expect(page.getByText("This file can't be opened.", { exact: true })).toHaveCount(0);
  });
});

for (const backend of ["native", "fallback"] as const) {
  test.describe(`the pointer over the photo (${backend})`, () => {
    test("Pick white sends the click to the core as a point of the cell and shows what the core set", async ({ page }) => {
      await photos(page, { mockViewport: backend });
      await page.getByRole("button", { name: "Pick white" }).click();
      const layer = page.getByRole("application");
      const box = (await layer.boundingBox())!;
      await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.15);
      await expect(page.getByRole("tab", { name: "Colour" })).toHaveAttribute("aria-selected", "true");
      await expect(page.getByRole("region", { name: "White balance" }).getByText("Edited")).toBeVisible();
      const input = (await viewportOf(page)).inputs.at(-1)!;
      expect(input.kind).toBe("pick_white");
      // The mouse works in whole pixels.
      expect(Math.abs((input.x as number) - box.width * 0.3)).toBeLessThanOrEqual(1);
      expect(Math.abs((input.y as number) - box.height * 0.15)).toBeLessThanOrEqual(1);
      // One undo step takes it back, through the core.
      await page.keyboard.press("Meta+z");
      await expect(page.getByRole("region", { name: "White balance" }).getByText("Edited")).toHaveCount(0);
    });

    test("a double click toggles Fit and 100 %, ⌘ and the wheel zoom about the pointer, and a drag pans once zoomed", async ({ page }) => {
      await photos(page, { mockViewport: backend });
      const box = (await cell(page).boundingBox())!;
      await page.mouse.dblclick(box.x + 100, box.y + 50);
      await expect(page.getByRole("button", { name: /^100%/ })).toBeVisible();
      expect((await viewportOf(page)).inputs.at(-1)).toMatchObject({ kind: "toggle_zoom" });
      expect((await viewportOf(page)).inputs.at(-1)!.x as number).toBeCloseTo(100, 0);
      await page.mouse.dblclick(box.x + 100, box.y + 50);
      await expect(page.getByRole("button", { name: /^Fit/ })).toBeVisible();
      // A wheel zoom from Fit: the header follows, and the drag now pans.
      await page.mouse.move(box.x + 300, box.y + 200);
      await page.keyboard.down("Control");
      await page.mouse.wheel(0, -300);
      await page.keyboard.up("Control");
      await expect(page.getByRole("button", { name: /^\d+%/ })).toBeVisible();
      const inputs = (await viewportOf(page)).inputs.filter((i) => i.kind === "zoom");
      expect(inputs.length).toBeGreaterThan(0);
      expect(inputs[0]!.x as number).toBeCloseTo(300, 0);
      await page.mouse.move(box.x + 300, box.y + 200);
      await page.mouse.down();
      await page.mouse.move(box.x + 340, box.y + 230, { steps: 4 });
      await page.mouse.up();
      expect((await viewportOf(page)).inputs.filter((i) => i.kind === "pan").length).toBeGreaterThan(0);
    });

    test("⌥ reads the pixel under the pointer in CSS pixels of the cell", async ({ page }) => {
      await photos(page, { mockViewport: backend });
      const box = (await cell(page).boundingBox())!;
      await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.1);
      await page.keyboard.down("Alt");
      await expect(page.locator(".ym-photo-cell__readout")).toHaveText(/^R \d+ G \d+ B \d+$/);
      await page.keyboard.up("Alt");
      await expect(page.locator(".ym-photo-cell__readout")).toHaveCount(0);
    });
  });
}

test.describe("Update ready while the core exports", () => {
  test("is held back while the core says an export is busy, and comes back when it is quiet", async ({ page }) => {
    await photos(page);
    await showUpdateReady(page);
    const pill = page.getByRole("button", { name: /Update ready/ });
    await expect(pill).not.toHaveAttribute("aria-disabled", "true");
    await page.evaluate(() => (window as unknown as { __yellowmoon: { mockApi: { emitEvent: (e: string, p: unknown) => void } } }).__yellowmoon.mockApi.emitEvent("export_busy", { busy: true }));
    await expect(pill).toHaveAttribute("aria-disabled", "true");
    await expect(pill).toHaveAccessibleDescription("Restarts after the export finishes");
    // A locked button takes no pointer events itself, so the tooltip is on the span around it: hovering the dead cell explains it.
    await page.locator('[data-slot="tooltip-trigger"]', { has: pill }).hover();
    await expect(page.getByRole("tooltip").first()).toHaveText("Restarts after the export finishes");
    await page.evaluate(() => (window as unknown as { __yellowmoon: { mockApi: { emitEvent: (e: string, p: unknown) => void } } }).__yellowmoon.mockApi.emitEvent("export_busy", { busy: false }));
    await expect(pill).not.toHaveAttribute("aria-disabled", "true");
  });
});
