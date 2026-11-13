import { expect, test } from "@playwright/test";
import type { BrowserContext, Page } from "@playwright/test";
import { openApp, openPhotos } from "./helpers/app";

// The system's display options (`docs/28` A-07, A-P12; `docs/11` *Display options*). On a Mac the shell reads Reduce Transparency, Reduce Motion
// and Increase Contrast and the page sets `data-reduce-transparency`, `data-reduce-motion` and `data-increase-contrast` on `<html>`; the CSS
// answers the attributes and the media queries alike, because WebKit may not match every query. Here the attributes come through the mock's
// `?mockDisplay=` (the real wiring: the command `display_options`, the store, `src/displayOptions.ts`), and once more the media queries are
// emulated through the DevTools protocol with no attribute at all. Both paths must give the same computed results:
//
// * Reduce Transparency: every pane is opaque (the alpha of its fill is 1) and no element, pseudo-elements included, has a `backdrop-filter`;
//   on the Photos screen, in Preferences, in an open menu and in a tooltip.
// * Increase Contrast: the tertiary label is the secondary's, and the focus ring is 4 px.
// * Reduce Motion: nothing runs: no transition, and no animation longer than a millisecond.
//
// What the rendered contrast test says with the attributes set is in `e2e/contrast.spec.ts` (its variants).

const ALL = "reduce-transparency,reduce-motion,increase-contrast";

/** What the page says about itself: the elements with a backdrop filter and the panes that let the aurora through. */
interface Scan {
  filters: string[];
  translucent: string[];
  panes: number;
}

/**
 * Walks every element of the page (and its `::before` and `::after`). A pane is a glass surface (`.glass`, `.glass-strong`, `.glass-subtle`,
 * `.glass-liquid`, `.glass-liquid-clear`), the content of a dialog, menu, popover, select, tooltip or sheet, and the app's header, tool panel,
 * filmstrip and error bar. It is opaque when the alpha of its fill is 1, or when every colour of its background image is (the filmstrip is a
 * gradient on a clear colour). A fully clear fill is no fill (a pane with a backdrop and no tint is judged by the image alone).
 */
function scan(): Scan {
  const alphaOf = (color: string): number => {
    const slash = /\/\s*([\d.]+%?)\s*\)$/.exec(color);
    if (slash) return slash[1]!.endsWith("%") ? Number.parseFloat(slash[1]!) / 100 : Number(slash[1]);
    const rgba = /^rgba\([^,]+,[^,]+,[^,]+,\s*([\d.]+)\)$/.exec(color);
    return rgba ? Number(rgba[1]) : 1;
  };
  const imageAlphas = (image: string): number[] =>
    [...image.matchAll(/(?:oklch|oklab|rgba?|hsla?|color)\([^)]*\)/g)].map((m) => alphaOf(m[0])).filter((a) => a > 0);
  const paneClass = /(^|\s)glass(-strong|-subtle|-liquid|-liquid-clear)?(\s|$)/;
  const paneSlot = /^(dialog|alert-dialog|dropdown-menu|context-menu|popover|select|tooltip|hover-card|sheet|menubar)-content$/;
  const paneApp = /(^|\s)ym-(header|tools|filmstrip|error-bar)(\s|$)/;
  const out: Scan = { filters: [], translucent: [], panes: 0 };
  const name = (el: Element, pseudo = ""): string => `${el.tagName.toLowerCase()}.${String((el as HTMLElement).className).trim().split(/\s+/).slice(0, 3).join(".")}${pseudo}`;
  for (const el of document.querySelectorAll<HTMLElement>("*")) {
    for (const pseudo of ["", "::before", "::after"]) {
      const cs = getComputedStyle(el, pseudo || null);
      const filter = cs.backdropFilter || (cs as unknown as { webkitBackdropFilter?: string }).webkitBackdropFilter || "none";
      if (filter !== "none") out.filters.push(`${name(el, pseudo)} ${filter}`);
    }
    const cls = String(el.className);
    const slot = el.getAttribute("data-slot") ?? "";
    if (!(paneClass.test(cls) || paneSlot.test(slot) || paneApp.test(cls))) continue;
    const cs = getComputedStyle(el);
    if (cs.display === "none") continue;
    out.panes += 1;
    const fill = alphaOf(cs.backgroundColor);
    const stops = imageAlphas(cs.backgroundImage);
    const solidImage = stops.length > 0 && stops.every((a) => a === 1);
    const fillClear = cs.backgroundColor === "rgba(0, 0, 0, 0)";
    // The glass's sheen is a gradient from a faint white to nothing: it is not the fill, and the fill is judged by its colour.
    const opaque = fill === 1 || (fillClear && solidImage);
    if (!opaque) out.translucent.push(`${name(el)} fill ${cs.backgroundColor} image ${cs.backgroundImage.slice(0, 80)}`);
  }
  return out;
}

/** The three display options on, as attributes (through the mock) or as emulated media queries (through DevTools). */
type Path = "attributes" | "media";

const MEDIA = [
  { name: "prefers-reduced-transparency", value: "reduce" },
  { name: "prefers-contrast", value: "more" },
  { name: "prefers-reduced-motion", value: "reduce" },
];

/** Opens the Photos screen with the options on, by `path`. Returns the context (for DevTools). */
async function openWith(page: Page, context: BrowserContext, path: Path, display = ALL): Promise<void> {
  if (path === "media") {
    const session = await context.newCDPSession(page);
    await session.send("Emulation.setEmulatedMedia", { features: MEDIA });
  }
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4", ...(path === "attributes" ? { mockDisplay: display } : {}) });
  await openPhotos(page);
  const attributes = await page.evaluate(() => ["data-reduce-transparency", "data-reduce-motion", "data-increase-contrast"].map((a) => document.documentElement.hasAttribute(a)));
  expect(attributes, path === "attributes" ? "the three attributes are set on <html>" : "no attribute is set: the media queries alone answer").toEqual(path === "attributes" ? [true, true, true] : [false, false, false]);
}

for (const path of ["attributes", "media"] as const) {
  test.describe(`display options, from the ${path === "attributes" ? "attributes of the shell" : "media queries"}`, () => {
    test("Reduce Transparency: every pane is opaque and nothing has a backdrop filter, on the Photos screen, in Preferences, in a menu and in a tooltip", async ({ page, context }) => {
      await openWith(page, context, path);
      const where: Record<string, Scan> = {};

      where.photos = await page.evaluate(scan);

      await page.getByRole("button", { name: /^Fit/ }).click();
      await expect(page.getByRole("menu")).toBeVisible();
      await page.waitForTimeout(300);
      where.menu = await page.evaluate(scan);
      expect(where.menu.panes, "the menu is a pane the scan found").toBeGreaterThan(where.photos.panes);
      await page.keyboard.press("Escape");
      await expect(page.getByRole("menu")).toHaveCount(0);

      await page.getByRole("button", { name: /^Fit/ }).hover();
      await expect(page.locator("[data-slot=tooltip-content]").first()).toBeVisible();
      await page.waitForTimeout(300);
      where.tooltip = await page.evaluate(scan);
      expect(where.tooltip.panes, "the tooltip is a pane the scan found").toBeGreaterThan(where.photos.panes);
      await page.mouse.move(5, 5);

      await page.keyboard.press("Meta+,");
      await expect(page.getByRole("dialog", { name: "Preferences" })).toBeVisible();
      await page.waitForTimeout(300);
      where.preferences = await page.evaluate(scan);
      expect(where.preferences.panes, "the dialog is a pane the scan found").toBeGreaterThan(where.photos.panes);

      for (const [scene, result] of Object.entries(where)) {
        expect(result.filters, `${scene}: elements with a backdrop-filter`).toEqual([]);
        expect(result.translucent, `${scene}: panes whose fill lets the page through`).toEqual([]);
      }
    });

    test("Increase Contrast: the tertiary label is the secondary's and the focus ring is 4 px", async ({ page, context }) => {
      await openWith(page, context, path);
      const labels = await page.evaluate(() => {
        const host = document.querySelector(".ym-tools") ?? document.body;
        const colour = (token: string): string => {
          const span = document.createElement("span");
          span.style.color = `var(${token})`;
          host.appendChild(span);
          const value = getComputedStyle(span).color;
          span.remove();
          return value;
        };
        return { secondary: colour("--glass-label-secondary"), tertiary: colour("--glass-label-tertiary") };
      });
      expect(labels.tertiary).toBe(labels.secondary);

      // The keyboard's focus ring on the first controls reached by Tab: glasscn's buttons and the app's own.
      const widths: string[] = [];
      for (let i = 0; i < 6; i += 1) {
        await page.keyboard.press("Tab");
        widths.push(
          await page.evaluate(() => {
            const el = document.activeElement as HTMLElement;
            return el.matches(":focus-visible") ? getComputedStyle(el).outlineWidth : "not focus-visible";
          }),
        );
      }
      expect(widths, "the outline of each element Tab reaches").toEqual(Array(6).fill("4px"));
    });

    test("Reduce Motion: no transition and no animation runs", async ({ page, context }) => {
      await openWith(page, context, path);
      const running = () =>
        page.evaluate(() =>
          document
            .getAnimations()
            .filter((a) => a.playState === "running")
            .map((a) => `${a.constructor.name} ${(a as CSSTransition).transitionProperty ?? (a as CSSAnimation).animationName} ${a.effect?.getComputedTiming().duration}`)
            .filter((line) => !/ (0|0\.001)$/.test(line)),
        );
      expect(await running(), "at rest").toEqual([]);
      // What moves in the app: a menu opening (the glass's materialise), a tab (the pill that slides), a pressed button.
      await page.getByRole("button", { name: /^Fit/ }).click();
      await expect(page.getByRole("menu")).toBeVisible();
      expect(await running(), "a menu opening").toEqual([]);
      await page.keyboard.press("Escape");
      await page.getByRole("tab", { name: "Light", exact: true }).click();
      expect(await running(), "a tab changing").toEqual([]);
      const style = await page.evaluate(() => {
        const cs = getComputedStyle(document.querySelector("[data-slot=tabs-indicator]") ?? document.body);
        return { property: cs.transitionProperty, duration: getComputedStyle(document.documentElement).getPropertyValue("--glass-duration").trim() };
      });
      expect(style.property).toBe("none");
      expect(style.duration).toBe("0ms");
    });
  });
}

test("with no display option on, the page has no attribute and keeps its blur, its translucent panes and its 3 px ring", async ({ page }) => {
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4" });
  await openPhotos(page);
  expect(await page.evaluate(() => ["data-reduce-transparency", "data-reduce-motion", "data-increase-contrast"].filter((a) => document.documentElement.hasAttribute(a)))).toEqual([]);
  const result = await page.evaluate(scan);
  expect(result.filters.length, "the glass blurs what is behind it").toBeGreaterThan(0);
  expect(result.translucent.length, "the panes let the aurora through").toBeGreaterThan(0);
  await page.keyboard.press("Tab");
  expect(await page.evaluate(() => getComputedStyle(document.activeElement as HTMLElement).outlineWidth)).toBe("3px");
  const labels = await page.evaluate(() => {
    const colour = (token: string): string => {
      const span = document.createElement("span");
      span.style.color = `var(${token})`;
      (document.querySelector(".ym-tools") ?? document.body).appendChild(span);
      const value = getComputedStyle(span).color;
      span.remove();
      return value;
    };
    return { secondary: colour("--glass-label-secondary"), tertiary: colour("--glass-label-tertiary") };
  });
  expect(labels.tertiary, "the tertiary label is dimmer than the secondary").not.toBe(labels.secondary);
});

test("each option alone changes only its own things, and an event from the shell follows a change of the system at once", async ({ page }) => {
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4", mockDisplay: "reduce-motion" });
  await openPhotos(page);
  const blurs = () => page.evaluate(scan).then((s) => s.filters.length);
  expect(await blurs(), "Reduce Motion alone keeps the blur").toBeGreaterThan(0);
  const say = (options: { reduce_transparency: boolean; reduce_motion: boolean; increase_contrast: boolean }) =>
    page.evaluate((o) => window.__yellowmoon!.mockApi.emitEvent("display_options", o), options);
  await say({ reduce_transparency: true, reduce_motion: false, increase_contrast: false });
  await expect.poll(blurs, "the system turned Reduce Transparency on").toBe(0);
  expect(await page.evaluate(() => document.documentElement.hasAttribute("data-reduce-motion")), "and Reduce Motion off").toBe(false);
  await say({ reduce_transparency: false, reduce_motion: false, increase_contrast: true });
  await expect.poll(() => page.evaluate(() => document.documentElement.hasAttribute("data-increase-contrast"))).toBe(true);
  expect(await blurs(), "Increase Contrast alone keeps the blur (it lightens the tint and the rim)").toBeGreaterThan(0);
  expect((await page.evaluate(scan)).translucent.length, "and the panes at 94 %").toBeGreaterThan(0);
  await say({ reduce_transparency: false, reduce_motion: false, increase_contrast: false });
  await expect.poll(blurs, "the system turned it off again").toBeGreaterThan(0);
  expect(await page.evaluate(() => ["data-reduce-transparency", "data-reduce-motion", "data-increase-contrast"].filter((a) => document.documentElement.hasAttribute(a)))).toEqual([]);
});
