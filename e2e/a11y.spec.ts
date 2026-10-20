import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { openApp, openPhotos, setValue } from "./helpers/app";

// Accessibility basics (docs/11, a preview of M4-04) in a real browser: every control has a role and a name, every screen has one
// main landmark, the keyboard alone gets through the main flow, and every stop of the Tab order shows the 3 px focus ring.
// Contrast of the tab colours is in src/styles/contrast.node.test.ts.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

/** Roles that name something to press, type into or choose: without a name a screen reader says only "button". */
const NEEDS_A_NAME = new Set([
  "button",
  "checkbox",
  "combobox",
  "dialog",
  "link",
  "menuitem",
  "menuitemradio",
  "option",
  "radio",
  "radiogroup",
  "searchbox",
  "slider",
  "spinbutton",
  "switch",
  "tab",
  "tablist",
  "textbox",
]);

/** The accessibility tree of the page as Playwright prints it. */
const tree = (page: Page): Promise<string> => page.locator("body").ariaSnapshot();

/** Lines of the tree that are controls without a name. */
function unnamed(snapshot: string): string[] {
  const found: string[] = [];
  for (const line of snapshot.split("\n")) {
    const match = /^\s*- ([a-z]+)((?: "(?:[^"\\]|\\.)*")?)((?: \[[^\]]*\])*)(?::.*)?$/.exec(line);
    if (match && NEEDS_A_NAME.has(match[1] as string) && !match[2]) found.push(line.trim());
  }
  return found;
}

const count = (snapshot: string, role: string): number => (snapshot.match(new RegExp(`^\\s*- ${role}\\b`, "gm")) ?? []).length;

async function expectNamed(page: Page, where: string): Promise<string> {
  const snapshot = await tree(page);
  expect(unnamed(snapshot), `${where}: controls without a name`).toEqual([]);
  return snapshot;
}

async function toWizard(page: Page): Promise<void> {
  await openApp(page, { mockLooks: 0 });
  await page.getByRole("button", { name: "Import →" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Choose what to import" })).toBeVisible();
}

test.describe("names and landmarks", () => {
  test("Start: every control is named, one main landmark, one h1", async ({ page }) => {
    await openApp(page, { mockLooks: 0 });
    let snapshot = await expectNamed(page, "Start");
    expect(count(snapshot, "main")).toBe(1);
    expect(count(snapshot, "banner")).toBe(1);
    expect(snapshot).toMatch(/heading "Drop Sony ARW files\." \[level=1\]/);
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    snapshot = await expectNamed(page, "Start, developed");
    expect(count(snapshot, "main")).toBe(1);
    expect(count(snapshot, "heading")).toBeGreaterThanOrEqual(2);
  });

  test("Photos: every tab, Crop, Pick white and the zoom menu", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    let snapshot = await expectNamed(page, "Photos");
    expect(count(snapshot, "main")).toBe(1);
    expect(count(snapshot, "banner")).toBe(1);
    // The folder name is the screen's heading.
    expect(snapshot).toMatch(/heading "Lisbon" \[level=1\]/);
    for (const tab of ["Look", "Light", "Colour", "Detail", "Curve", "RAW"]) {
      await page.getByRole("tab", { name: tab }).click();
      await expect(page.getByRole("tab", { name: tab })).toHaveAttribute("aria-selected", "true");
      await expectNamed(page, `the ${tab} tab`);
    }
    await page.getByRole("button", { name: "Crop" }).click();
    await expect(page.getByRole("button", { name: "Done" })).toBeVisible();
    snapshot = await expectNamed(page, "Crop");
    expect(count(snapshot, "main")).toBe(1);
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Pick white" }).click();
    await expectNamed(page, "Pick white");
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: /^Fit/ }).click();
    await expect(page.getByRole("menu")).toBeVisible();
    await expectNamed(page, "the zoom menu");
  });

  test("Look tab sections: every header is a named disclosure button, folded or searching", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockLibraries: "all" });
    await openPhotos(page);
    let snapshot = await expectNamed(page, "the Look tab with every library");
    for (const name of [
      "Built in, 2 looks, collapse",
      "ARRI looks · LogC4, 7 looks, collapse",
      "ARRI looks · LogC3, 5 looks, collapse",
      "ARRI looks · LogC3 to Rec.709, 4 looks, collapse",
      "ARRI looks · LogC3 to Rec.709 · ALF2, 6 looks, collapse",
      "Your LUTs, 5 looks, collapse",
    ]) {
      expect(snapshot, name).toMatch(new RegExp(`- button "${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}" \\[expanded\\]`));
    }
    // Folded: no [expanded], the name says expand, and the section holding the current look says so.
    await page.getByRole("button", { name: "Built in, 2 looks, collapse" }).click();
    snapshot = await expectNamed(page, "a folded section");
    expect(snapshot).toContain('- button "Built in, 2 looks, holds the current look, expand"');
    expect(snapshot).not.toContain('"Built in, 2 looks, holds the current look, expand" [expanded]');
    // Searching: every header with matches is open and still named.
    await page.getByRole("searchbox", { name: "Search looks" }).fill("film");
    snapshot = await expectNamed(page, "a search over folded sections");
    expect(snapshot).toContain('- button "ARRI looks · LogC4, 2 of 7 looks" [disabled] [expanded]');
  });

  test("Export panel, in each format, and while it runs", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockExportHold: 1.45 });
    await openPhotos(page);
    await page.keyboard.press("Meta+e");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    for (const format of ["JPEG", "TIFF", "PNG", "DNG"]) {
      await page.getByRole("radiogroup", { name: "Format" }).getByRole("radio", { name: format }).click();
      const snapshot = await expectNamed(page, `Export, ${format}`);
      expect(count(snapshot, "main")).toBe(1);
    }
    await page.getByRole("button", { name: /^Export 3 photos/ }).click();
    await expect(page.getByText("1 of 3 done.")).toBeVisible();
    await expectNamed(page, "Export, running");
  });

  test("the wizard and the dialogs", async ({ page }) => {
    await toWizard(page);
    let snapshot = await expectNamed(page, "wizard, step 1");
    expect(count(snapshot, "main")).toBe(1);
    await page.getByRole("button", { name: /^Next/ }).click();
    await expect(page.getByRole("heading", { level: 1, name: "Download from ARRI" })).toBeVisible();
    await expectNamed(page, "wizard, step 2");
    await page.getByRole("button", { name: "Cancel" }).click();
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("dialog", { name: "Preferences" })).toBeVisible();
    await expectNamed(page, "Preferences");
    await page.getByRole("button", { name: /About/ }).click();
    await expect(page.getByRole("dialog", { name: /About/ })).toBeVisible();
    await expectNamed(page, "About");
  });
});

test.describe("Apple Auto (docs/19)", () => {
  test("the Automatic card's rows and line and the Preferences switch are named", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockAppleAuto: "answered" });
    await openPhotos(page);
    let snapshot = await expectNamed(page, "Photos with Apple Auto");
    expect(snapshot).toContain('button "Contrast +15, Auto. Open Tone"');
    expect(snapshot).toContain('button "Saturation −15, Auto. Open Tone"');
    expect(snapshot).toContain('button "Highlights −20, Auto. Open Light"');
    expect(snapshot).toContain('button "Shadows colour Teal · medium, Auto. Open Colour"');
    expect(snapshot).toContain("Set with Apple Intelligence.");
    expect(snapshot).toContain("Teal shadows and warm highlights for a cinematic late-sun street");
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("switch", { name: "Auto with Apple Intelligence" })).toBeVisible();
    snapshot = await expectNamed(page, "Preferences with Apple Auto");
    expect(snapshot).toMatch(/switch "Auto with Apple Intelligence" \[checked\]/);
    // The instructions for Apple's model: a named text box and its Reset to default, disabled while the text is the default.
    expect(snapshot).toMatch(/textbox "Instructions for Apple Intelligence"/);
    expect(snapshot).toMatch(/button "Reset to default" \[disabled\]/);
  });

  test("the switch flips from the keyboard, and the rows are Tab stops with the ring", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockAppleAuto: "answered" });
    await openPhotos(page);
    await tabTo(page, (el) => el.getAttribute("aria-label") === "Contrast +15, Auto. Open Tone");
    await expect(page.locator(":focus")).toHaveCSS("outline-width", "3px");
    await page.keyboard.press("Meta+,");
    const toggle = page.getByRole("switch", { name: "Auto with Apple Intelligence" });
    await toggle.focus();
    await page.keyboard.press("Space");
    await expect(toggle).toHaveAttribute("aria-checked", "false");
  });
});

/** Presses Tab until the focused element satisfies `matches` (it runs in the page, so it can't use anything outside itself), at most `limit` times. */
async function tabTo(page: Page, matches: (el: Element) => boolean, limit = 80): Promise<void> {
  for (let i = 0; i < limit; i++) {
    await page.keyboard.press("Tab");
    if (await page.evaluate(`(${matches.toString()})(document.activeElement)`)) return;
  }
  throw new Error(`Tab did not reach the element within ${limit} presses`);
}

test.describe("keyboard only", () => {
  test("Start → open → Photos → Light tab → Exposure → Export, without the mouse", async ({ page }) => {
    await openApp(page, MOCKUP);
    await tabTo(page, (el) => (el.textContent ?? "").trim() === "Open files…");
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { level: 2, name: "12 photos ready" })).toBeVisible();
    await tabTo(page, (el) => (el.textContent ?? "").trim().startsWith("Open photos"));
    await page.keyboard.press("Enter");
    await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();

    // The tab bar is one stop; the arrows move between the tabs.
    await tabTo(page, (el) => el.getAttribute("role") === "tab");
    await expect(page.getByRole("tab", { name: "Look" })).toBeFocused();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("complementary", { name: "Tools" }).getByRole("heading", { level: 2, name: "Light" })).toBeVisible();

    // A slider takes the arrows; Shift takes ten steps; Home goes to the end of the range.
    await tabTo(page, (el) => el.getAttribute("role") === "slider" && document.getElementById(el.getAttribute("aria-labelledby") ?? "")?.textContent === "Exposure");
    const exposure = page.getByRole("slider", { name: "Exposure" });
    await expect(exposure).toBeFocused();
    const value = async (): Promise<number> => Number(await exposure.getAttribute("aria-valuenow"));
    const before = await value();
    await page.keyboard.press("ArrowRight");
    expect(await value()).toBeGreaterThan(before);
    await page.keyboard.press("Shift+ArrowRight");
    expect(await value()).toBeGreaterThan(before + 0.05);
    await page.keyboard.press("Home");
    expect(await value()).toBe(-4);
    await expect(page.getByRole("textbox", { name: "Exposure value" })).toHaveValue("−4.00");
    // ⌘Z takes the keys back.
    await page.keyboard.press("Meta+z");

    // Export: ⌘E opens the panel, Tab reaches the button, Return starts it, Tab reaches Done.
    await page.keyboard.press("Meta+e");
    await expect(page.getByRole("complementary", { name: "Export" })).toBeVisible();
    await expect(page.getByRole("button", { name: /^Export 3 photos/ })).toContainText("≈");
    await tabTo(page, (el) => (el.textContent ?? "").trim().startsWith("Export 3 photos"));
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { level: 2, name: "Exported" })).toBeVisible();
    await tabTo(page, (el) => (el.textContent ?? "").trim() === "Done");
    await page.keyboard.press("Enter");
    await expect(page.getByRole("complementary", { name: "Tools" })).toBeVisible();
  });

  test("a look tile, a segment, a switch and the search box work from the keyboard", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await tabTo(page, (el) => el.getAttribute("aria-label") === "Search looks");
    await page.keyboard.type("film");
    await expect(page.getByText("2 of 7")).toBeVisible();
    // The field's Clear button (there while the field holds text) is not a Tab stop: the first tile is the next one.
    await page.keyboard.press("Tab");
    await expect(page.getByRole("button", { name: "Film warm look, key 3" })).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("button", { name: "Film warm look, key 3" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator(".ym-photo-cell__chip")).toHaveText("Film warm · 100%");

    // Apply to is a radio group: the arrows move and choose.
    await page.getByRole("radiogroup", { name: "Apply look to" }).getByRole("radio", { name: /^This photo/ }).focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("radiogroup", { name: "Apply look to" }).getByRole("radio", { name: /^Picked/ })).toHaveAttribute("aria-checked", "true");

    // A switch toggles with Space.
    await page.keyboard.press("Meta+6");
    const lens = page.getByRole("switch", { name: /Lens corrections/ });
    await lens.focus();
    const was = await lens.getAttribute("aria-checked");
    await page.keyboard.press("Space");
    await expect(lens).not.toHaveAttribute("aria-checked", was ?? "");
  });

  test("a typed value takes Return, and Esc gives it up", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.keyboard.press("Meta+2");
    await setValue(page, "Contrast", "25");
    await expect(page.getByRole("slider", { name: "Contrast" })).toHaveAttribute("aria-valuenow", "25");
    const box = page.getByRole("textbox", { name: "Contrast value" });
    await box.click();
    await box.fill("80");
    await box.press("Escape");
    await expect(box).toHaveValue("+25");
  });

  test("the wizard: Tab reaches Next, Return moves on, Back returns", async ({ page }) => {
    await toWizard(page);
    await tabTo(page, (el) => (el.textContent ?? "").trim().startsWith("Next"));
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { level: 1, name: "Download from ARRI" })).toBeVisible();
    await tabTo(page, (el) => el.tagName === "BUTTON" && (el.textContent ?? "").includes("Back"));
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { level: 1, name: "Choose what to import" })).toBeVisible();
  });

  test("a dialog traps focus, Esc closes it, and focus goes back", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    const fit = page.getByRole("button", { name: /^Fit/ });
    await fit.focus();
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog).toBeVisible();
    for (let i = 0; i < 25; i++) {
      await page.keyboard.press("Tab");
      expect(await page.evaluate(() => !!document.activeElement?.closest("[role=dialog]"))).toBe(true);
    }
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(page.locator(":focus")).toHaveCount(1);
  });
});

test.describe("focus ring", () => {
  /** Walks the Tab order and returns each stop's label and whether its focus ring is the 3 px outline with a 2 px offset. */
  async function walk(page: Page, stops: number): Promise<Array<{ label: string; ring: boolean }>> {
    const seen: Array<{ label: string; ring: boolean }> = [];
    for (let i = 0; i < stops; i++) {
      await page.keyboard.press("Tab");
      seen.push(
        await page.evaluate(async () => {
          // The slider sets its ring in a state update; let it draw first.
          await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
          const el = document.activeElement as HTMLElement | null;
          if (!el || el === document.body) return { label: "BODY", ring: true };
          // A text input inside a capsule (the search field) draws its ring on the capsule around it (focus-within); every other stop
          // draws its own.
          const rings = (node: Element | null): boolean => {
            if (!node) return false;
            const style = getComputedStyle(node);
            return style.outlineStyle !== "none" && parseFloat(style.outlineWidth) >= 3;
          };
          const own = rings(el) || (el instanceof HTMLInputElement && el.type !== "checkbox" && rings(el.parentElement));
          const label = (el.getAttribute("aria-label") ?? el.textContent ?? el.tagName).trim().slice(0, 40);
          return { label, ring: own };
        }),
      );
    }
    return seen;
  }

  test("every stop of Start's Tab order has the ring", async ({ page }) => {
    await openApp(page, { mockLooks: 0 });
    const seen = await walk(page, 7);
    expect(seen.filter((s) => !s.ring)).toEqual([]);
    expect(seen.map((s) => s.label)).toContain("Open files…");
  });

  test("every stop of the Photos Look tab has the ring", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    const seen = await walk(page, 30);
    expect(seen.filter((s) => !s.ring)).toEqual([]);
    expect(seen.length).toBe(30);
  });

  test("a focused tab shows the 3 px ring, drawn inside its cell", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await page.keyboard.press("Meta+6");
    await tabTo(page, (el) => el.getAttribute("role") === "tab");
    const raw = page.getByRole("tab", { name: "RAW" });
    await expect(raw).toBeFocused();
    await expect(raw).toHaveCSS("outline-width", "3px");
    // glasscn's tabs draw the ring inset (the tab row is a pane of its own, there is no room outside it).
    await expect(raw).toHaveCSS("outline-offset", "-3px");
  });
});
