import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, setValue } from "./helpers/app";

// Desktop feel (docs/11 §Desktop feel, M3-16 to M3-18) in a real browser against the mocked API. The owner, 2026-10-16: "UI in Yellowmoon
// is sluggish. I can select UI text, when I pan photo with two fingers on trackpad - all app moves." What cannot be shown here is WebKit's
// own scroll path (the rubber band) and whether WKWebView delivers pinch events: those are checks on a Mac (docs/STATUS.md).

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

type Viewport = { inputs: Array<Record<string, unknown>> };
const viewportOf = (page: Page): Promise<Viewport> =>
  page.evaluate(() => JSON.parse(JSON.stringify((window as unknown as { __yellowmoon: { mockApi: { viewport: unknown } } }).__yellowmoon.mockApi.viewport)) as Viewport);

async function photos(page: Page, extra: Record<string, string | number> = {}): Promise<void> {
  await openApp(page, { ...MOCKUP, ...extra });
  await openPhotos(page);
}

const cell = (page: Page) => page.locator("[data-viewport]");
const selection = (page: Page): Promise<string> => page.evaluate(() => window.getSelection()?.toString() ?? "");
const frames = (page: Page, n = 3): Promise<void> =>
  page.evaluate(async (count) => {
    for (let i = 0; i < count; i++) await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }, n);

/** Dispatches a bubbling, cancelable wheel event and says whether anything cancelled it. */
const wheelOn = (page: Page, selector: string, init: { deltaX?: number; deltaY?: number; ctrlKey?: boolean }): Promise<boolean> =>
  page.evaluate(
    ([sel, opts]) => {
      const el = document.querySelector(sel as string);
      if (!el) throw new Error(`no ${sel}`);
      const event = new WheelEvent("wheel", { bubbles: true, cancelable: true, ...(opts as object) });
      el.dispatchEvent(event);
      return event.defaultPrevented;
    },
    [selector, init] as const,
  );

test.describe("the page does not move as a whole", () => {
  test("locks the root and turns overscroll off, and keeps each scroller's own scrolling to itself", async ({ page }) => {
    await photos(page);
    const root = await page.evaluate(() => {
      const style = (el: Element) => {
        const s = getComputedStyle(el);
        return { overflow: s.overflow, overscroll: s.overscrollBehavior };
      };
      return { html: style(document.documentElement), body: style(document.body), root: style(document.getElementById("root")!) };
    });
    for (const box of [root.html, root.body, root.root]) {
      expect(box.overflow).toBe("hidden");
      expect(box.overscroll).toBe("none");
    }
    for (const selector of [".ym-tools__scroll", ".ym-filmstrip__scroll"]) {
      expect(await page.locator(selector).evaluate((el) => getComputedStyle(el).overscrollBehavior), selector).toBe("contain");
    }
    // The filmstrip is a row: it never scrolls up and down.
    expect(await page.locator(".ym-filmstrip__scroll").evaluate((el) => getComputedStyle(el).overflowY)).toBe("hidden");
  });

  test("a wheel over the photo is the photo's (cancelled), a wheel over the tool panel is the browser's (not)", async ({ page }) => {
    await photos(page);
    // At Fit there is nothing to pan, and the wheel is the photo's all the same: that is what stops the page from bouncing.
    expect(await wheelOn(page, "[data-viewport]", { deltaY: 10 })).toBe(true);
    expect(await wheelOn(page, "[data-viewport]", { deltaY: 10, ctrlKey: true })).toBe(true);
    expect(await wheelOn(page, ".ym-tools__scroll", { deltaY: 10 })).toBe(false);
    expect(await wheelOn(page, ".ym-filmstrip__scroll", { deltaY: 10 })).toBe(false);
    expect((await viewportOf(page)).inputs.filter((i) => i.kind === "pan")).toEqual([]);
  });

  test("the tool panel still scrolls with the mouse wheel, and nothing else moves", async ({ page }) => {
    await page.setViewportSize({ width: 1180, height: 760 });
    await photos(page, { mockLibraries: "all" });
    const scroller = page.locator(".ym-tools__scroll");
    expect(await scroller.evaluate((el) => el.scrollHeight - el.clientHeight)).toBeGreaterThan(100);
    const box = (await scroller.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, 400);
    await expect.poll(() => scroller.evaluate((el) => el.scrollTop)).toBeGreaterThan(100);
    expect(await page.evaluate(() => [window.scrollX, window.scrollY, document.documentElement.scrollTop, document.body.scrollTop])).toEqual([0, 0, 0, 0]);
    // The photo did not get it.
    expect((await viewportOf(page)).inputs).toEqual([]);
  });

  test("60 wheel events over a zoomed photo reach the core as at most two pans with the summed distance", async ({ page }) => {
    await photos(page, { mockViewport: "native" });
    const box = (await cell(page).boundingBox())!;
    await page.mouse.dblclick(box.x + 100, box.y + 80);
    await expect(page.getByRole("button", { name: /^100%/ })).toBeVisible();
    const before = (await viewportOf(page)).inputs.length;
    await page.evaluate(() => {
      const el = document.querySelector("[data-viewport]")!;
      for (let i = 0; i < 60; i++) el.dispatchEvent(new WheelEvent("wheel", { bubbles: true, cancelable: true, deltaX: 2, deltaY: 3 }));
    });
    await frames(page, 4);
    const pans = (await viewportOf(page)).inputs.slice(before).filter((i) => i.kind === "pan");
    expect(pans.length).toBeGreaterThan(0);
    expect(pans.length).toBeLessThanOrEqual(2);
    expect(pans.reduce((sum, p) => sum + (p.dx as number), 0)).toBe(-120);
    expect(pans.reduce((sum, p) => sum + (p.dy as number), 0)).toBe(-180);
  });

  test("60 ⌘ wheel events about one point are one zoom, by the product of their factors", async ({ page }) => {
    await photos(page, { mockViewport: "native" });
    await page.evaluate(() => {
      const el = document.querySelector("[data-viewport]")!;
      const box = el.getBoundingClientRect();
      for (let i = 0; i < 60; i++) {
        el.dispatchEvent(new WheelEvent("wheel", { bubbles: true, cancelable: true, ctrlKey: true, deltaY: -5, clientX: box.left + 300, clientY: box.top + 200 }));
      }
    });
    await frames(page, 4);
    const zooms = (await viewportOf(page)).inputs.filter((i) => i.kind === "zoom");
    expect(zooms.length).toBeGreaterThan(0);
    expect(zooms.length).toBeLessThanOrEqual(2);
    expect(zooms.reduce((product, z) => product * (z.factor as number), 1)).toBeCloseTo(Math.exp(0.01 * 5 * 60), 6);
    expect(zooms[0]!.x as number).toBeCloseTo(300, 0);
    // The header followed, once.
    await expect(page.getByRole("button", { name: /^\d+%/ })).toBeVisible();
  });

  test("a trackpad pinch (WebKit's gesture events) zooms the photo about the fingers, and nothing else zooms", async ({ page }) => {
    await photos(page, { mockViewport: "native" });
    const prevented = await page.evaluate(() => {
      const el = document.querySelector("[data-viewport]")!;
      const box = el.getBoundingClientRect();
      const fire = (type: string, fields: object): boolean => {
        const event = Object.assign(new Event(type, { bubbles: true, cancelable: true }), fields);
        el.dispatchEvent(event);
        return event.defaultPrevented;
      };
      return [
        fire("gesturestart", {}),
        fire("gesturechange", { scale: 1.5, clientX: box.left + 250, clientY: box.top + 150 }),
        fire("gesturechange", { scale: 2.5, clientX: box.left + 250, clientY: box.top + 150 }),
        fire("gestureend", {}),
      ];
    });
    expect(prevented).toEqual([true, true, true, true]);
    await frames(page, 4);
    const zooms = (await viewportOf(page)).inputs.filter((i) => i.kind === "zoom");
    expect(zooms.length).toBeGreaterThan(0);
    expect(zooms.reduce((product, z) => product * (z.factor as number), 1)).toBeCloseTo(2.5, 6);
    expect(zooms[0]!.x as number).toBeCloseTo(250, 0);
    expect(zooms[0]!.y as number).toBeCloseTo(150, 0);
    await expect(page.getByRole("button", { name: /^\d+%/ })).toBeVisible();
    // A pinch anywhere else is the page's to zoom, and the page does not: it is cancelled at the window.
    const elsewhere = await page.evaluate(() => {
      const event = new Event("gesturechange", { bubbles: true, cancelable: true });
      document.querySelector(".ym-header")!.dispatchEvent(event);
      return event.defaultPrevented;
    });
    expect(elsewhere).toBe(true);
  });

  test("a drag pans once per frame and ends with the pointer", async ({ page }) => {
    await photos(page, { mockViewport: "native" });
    const box = (await cell(page).boundingBox())!;
    await page.mouse.dblclick(box.x + 100, box.y + 80);
    await expect(page.getByRole("button", { name: /^100%/ })).toBeVisible();
    const before = (await viewportOf(page)).inputs.length;
    await page.mouse.move(box.x + 300, box.y + 200);
    await page.mouse.down();
    await page.mouse.move(box.x + 400, box.y + 260, { steps: 20 });
    await page.mouse.up();
    await frames(page, 2);
    const pans = (await viewportOf(page)).inputs.slice(before).filter((i) => i.kind === "pan");
    expect(pans.length).toBeGreaterThan(0);
    expect(pans.length).toBeLessThanOrEqual(20);
    expect(pans.reduce((sum, p) => sum + (p.dx as number), 0)).toBe(100);
    expect(pans.reduce((sum, p) => sum + (p.dy as number), 0)).toBe(60);
  });
});

test.describe("text is not selectable, except where people copy it", () => {
  test("turns selection off on the page", async ({ page }) => {
    await photos(page);
    expect(await page.locator(".ym-header").evaluate((el) => getComputedStyle(el).userSelect)).toBe("none");
    expect(await page.locator("html").evaluate((el) => getComputedStyle(el).cursor)).toBe("default");
  });

  test("a triple click in the header, a double click on the photo, ⌘A and a drag across the window select nothing", async ({ page }) => {
    await photos(page, { mockViewport: "native" });
    await page.locator(".ym-brand-name").click({ clickCount: 3 });
    expect(await selection(page)).toBe("");
    const box = (await cell(page).boundingBox())!;
    await page.mouse.dblclick(box.x + box.width / 2, box.y + box.height - 20);
    expect(await selection(page)).toBe("");
    await page.keyboard.press("Control+a");
    expect(await selection(page)).toBe("");
    await page.mouse.move(box.x + 20, 20);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width - 20, box.y + box.height - 20, { steps: 8 });
    await page.mouse.up();
    expect(await selection(page)).toBe("");
  });

  test("the command line in the Export panel can be selected and copied", async ({ page }) => {
    await photos(page);
    await page.keyboard.press("Meta+e");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    const code = page.locator(".ym-export__code");
    await expect(code).toContainText("ym render");
    expect(await code.evaluate((el) => getComputedStyle(el).userSelect)).toBe("text");
    // A triple click takes the line it lands on; the command is several, so aim at the first.
    await code.click({ clickCount: 3, position: { x: 12, y: 8 } });
    expect(await selection(page)).toContain("ym render");
    // The folder it saves to is text too.
    expect(await page.locator(".ym-export__folder-text").evaluate((el) => getComputedStyle(el).userSelect)).toBe("text");
  });

  test("an error and a path in Preferences can be selected", async ({ page }) => {
    await photos(page);
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("dialog", { name: "Preferences" })).toBeVisible();
    expect(await page.locator(".ym-pref__hint").first().evaluate((el) => getComputedStyle(el).userSelect)).toBe("text");
    await page.keyboard.press("Escape");
  });

  test("typing in the search field and in a value box still works", async ({ page }) => {
    await photos(page);
    const search = page.getByRole("searchbox", { name: "Search looks" });
    await search.click();
    await page.keyboard.type("neutral");
    await expect(search).toHaveValue("neutral");
    expect(await search.evaluate((el) => getComputedStyle(el).userSelect)).toBe("text");
    // A box that selects its text on focus still does, and takes a typed value.
    await search.fill("");
    await setValue(page, "Intensity", "55");
    await expect(page.getByRole("textbox", { name: "Intensity value" })).toHaveValue("55%");
    const box = page.getByRole("textbox", { name: "Intensity value" });
    await box.click();
    expect(await selection(page)).not.toBeNull();
  });

  test("keeps the page menu away over the page and gives it back in a field and in selectable text", async ({ page }) => {
    await photos(page);
    const menu = (selector: string): Promise<boolean> =>
      page.evaluate((sel) => {
        const el = document.querySelector(sel);
        if (!el) throw new Error(`no ${sel}`);
        const event = new MouseEvent("contextmenu", { bubbles: true, cancelable: true });
        el.dispatchEvent(event);
        return event.defaultPrevented;
      }, selector);
    expect(await menu(".ym-header")).toBe(true);
    expect(await menu("[data-viewport]")).toBe(true);
    expect(await menu(".ym-filmstrip")).toBe(true);
    expect(await menu('input[type="search"]')).toBe(false);
    await page.keyboard.press("Meta+e");
    await expect(page.locator(".ym-export__code")).toBeVisible();
    expect(await menu(".ym-export__code")).toBe(false);
    expect(await menu(".ym-export__template")).toBe(false);
  });

  test("a button is an arrow and a link is a hand", async ({ page }) => {
    await photos(page);
    expect(await page.getByRole("button", { name: "Before" }).evaluate((el) => getComputedStyle(el).cursor)).toBe("default");
    expect(await page.getByRole("tab", { name: "Light" }).evaluate((el) => getComputedStyle(el).cursor)).toBe("default");
    expect(await page.getByRole("button", { name: "Import .cube…" }).evaluate((el) => getComputedStyle(el).cursor)).toBe("pointer");
    expect(await page.getByRole("searchbox", { name: "Search looks" }).evaluate((el) => getComputedStyle(el).cursor)).toBe("text");
  });
});

test.describe("measuring", () => {
  test("is off unless the page asks, and on with ?perf: counts, late frames and the api's round trips in one JSON object", async ({ page }) => {
    await photos(page);
    expect(await page.evaluate(() => (window as unknown as { __ymPerf?: unknown }).__ymPerf)).toBeUndefined();

    await openApp(page, { ...MOCKUP, perf: 1 });
    await openPhotos(page);
    await frames(page, 5);
    const snapshot = await page.evaluate(() =>
      (window as unknown as { __ymPerf: { snapshot: () => Record<string, unknown> } }).__ymPerf.snapshot(),
    );
    const counts = snapshot.counts as Record<string, { total: number }>;
    expect(counts.Photos!.total).toBeGreaterThan(0);
    expect(counts.LookTab!.total).toBeGreaterThan(0);
    expect(counts.LazyLookTile!.total).toBeGreaterThan(0);
    expect(counts["store.notify"]!.total).toBeGreaterThan(0);
    expect(counts["event.render_stats"]!.total).toBeGreaterThan(0);
    const commands = snapshot.commands as Record<string, { calls: number }>;
    expect(commands.openDialog!.calls).toBeGreaterThan(0);
    expect(commands.lookThumbnail!.calls).toBeGreaterThan(0);
    expect((snapshot.frames as { watching: boolean }).watching).toBe(true);
  });
});
