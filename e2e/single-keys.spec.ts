import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { configureMock, mockQuery, openApp, openPhotos } from "./helpers/app";
import { GRADED_NODES } from "./helpers/states";

// The single-key switch (docs/28 A-09, A-P11, WCAG 2.1.4): Preferences › "Single-key shortcuts" is on by default; with it off none of the
// 21 keys (P Z C W N F B M A \ 1 to 9 Delete Backspace) acts outside a text box, every menu item still does what its key did, ⌘-keys, the
// arrows and Esc stay, and the value is kept across a reload (the mock keeps preferences with `mockPersist=1`, as the core keeps preferences.json).

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;
const RADIAL = { kind: "radial", mode: "add", invert: false, enabled: true, opacity: 1, center: [0.5, 0.5], radii: [0.3, 0.2], angle_deg: 0, feather: 0.4 };

/** Everything a key of the list changes, read the way a person or a screen reader would find it. */
async function snapshot(page: Page): Promise<string> {
  return page.evaluate(async () => {
    const screen = document.querySelector('[data-screen="photos"]');
    const said = (screen?.querySelector(':scope > [aria-live="polite"]')?.textContent ?? "").replace(/ /g, "").trim();
    const mock = window.__yellowmoon!.mockApi;
    const session = await mock.sessionState();
    return JSON.stringify({
      said,
      nodes: screen?.getAttribute("data-nodes"),
      fullscreen: screen?.getAttribute("data-fullscreen"),
      panel: screen?.getAttribute("data-panel"),
      crop: document.querySelector(".ym-crop") !== null,
      tab: [...document.querySelectorAll('[role="tab"][aria-selected="true"]')].map((t) => t.textContent).join("|"),
      auto: document.querySelector('[role="switch"][aria-label*="Auto crop" i], [aria-label*="Auto crop" i][aria-pressed]')?.getAttribute("aria-pressed") ?? null,
      photos: session.photos.length,
      current: session.current_index,
      picked: session.picked.length,
      mode: mock.viewport.mode,
      before: mock.viewport.before,
      maskView: mock.viewport.maskView,
      cards: [...document.querySelectorAll("[data-node-card]")].map((c) => c.getAttribute("aria-label")),
      pickWhite: document.querySelector("[data-hint]")?.textContent ?? null,
    });
  });
}

async function menu(page: Page, action: string): Promise<void> {
  await page.evaluate((a) => window.__yellowmoon!.mockApi.emitEvent("menu_action", { action: a } as never), action);
}

/** Turns the switch off in Preferences, the way a person does, and closes the dialog. */
async function switchOff(page: Page): Promise<void> {
  await page.keyboard.press("Meta+,");
  const dialog = page.getByRole("dialog", { name: "Preferences" });
  const sw = dialog.getByRole("switch", { name: "Single-key shortcuts" });
  await expect(sw).toHaveAttribute("aria-checked", "true");
  await sw.click();
  await expect(sw).toHaveAttribute("aria-checked", "false");
  await page.waitForFunction(async () => (await window.__yellowmoon!.mockApi.preferencesGet()).single_key_shortcuts === false);
  await dialog.getByRole("button", { name: "Done" }).click();
  await expect(dialog).toBeHidden();
}

async function nodeViewWithMask(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await configureMock(page, {
    initialSettings: { DSC04204: { nodes: { "1": { masks: { items: [RADIAL] } }, ...GRADED_NODES.nodes }, links: GRADED_NODES.links } } as never,
  });
  await openPhotos(page);
}

interface Case {
  /** The key as Playwright names it. */
  key: string;
  /** The menu item (`menu_action`) that does the same; absent for F, whose item is the system's Full Screen. */
  item?: string;
  /** Gets the screen ready, with the menu where a key would do (so it is the same with the switch on or off). */
  setup?: (page: Page) => Promise<void>;
}

const inNodeView = async (page: Page): Promise<void> => {
  await menu(page, "nodes");
  await expect(page.locator('[data-screen="photos"]')).toHaveAttribute("data-nodes", "true");
  await page.locator('[data-node-card="3"]').click();
};
const inCrop = async (page: Page): Promise<void> => {
  await menu(page, "crop");
  await expect(page.locator(".ym-crop")).toBeVisible();
};

/** A fresh page on the state the key needs, with the switch on or turned off in Preferences. */
async function prepare(page: Page, c: Case, off: boolean): Promise<void> {
  const nodes = c.key === "b" || c.key === "m";
  if (nodes) await nodeViewWithMask(page);
  else {
    await openApp(page, MOCKUP);
    await openPhotos(page);
  }
  if (off) await switchOff(page);
  if (!nodes) await page.getByRole("tab", { name: "Look", exact: true }).click();
  if (c.setup) await c.setup(page);
}

const CASES: Case[] = [
  { key: "p", item: "pick" },
  { key: "z", item: "zoom_100" },
  { key: "c", item: "crop" },
  { key: "w", item: "pick_white" },
  { key: "n", item: "nodes" },
  { key: "f" },
  { key: "b", item: "bypass_node", setup: inNodeView },
  { key: "m", item: "mask_view", setup: inNodeView },
  { key: "a", item: "auto_crop", setup: inCrop },
  { key: "\\", item: "before_after" },
  ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n): Case => ({ key: String(n), item: `look_${n}` })),
  { key: "Delete", item: "remove_photo" },
  { key: "Backspace", item: "remove_photo" },
];

test.describe("Single-key shortcuts off", () => {
  test("the list is 21 keys", () => {
    expect(CASES).toHaveLength(21);
  });

  for (const c of CASES) {
    test(`${c.key}: nothing with the switch off${c.item ? `, and the menu item ${c.item} acts` : ""}`, async ({ page }) => {
      // On: the key acts (the control of the test, so "nothing" below is the switch and not a key that never worked).
      await prepare(page, c, false);
      const start = await snapshot(page);
      await page.keyboard.press(c.key);
      await expect.poll(() => snapshot(page), { message: `${c.key} acts while the switch is on` }).not.toBe(start);

      // Off: a fresh page, the same state.
      await prepare(page, c, true);
      const before = await snapshot(page);
      await page.keyboard.press(c.key);
      // Wait for anything the key could have started, then compare.
      await page.waitForTimeout(400);
      expect(await snapshot(page), `${c.key} does nothing`).toBe(before);

      if (c.item) {
        await menu(page, c.item);
        await expect.poll(() => snapshot(page), { message: `${c.item} acts` }).not.toBe(before);
      }
    });
  }

  test("a text box keeps its letters with the switch on or off, and ⌘-keys, the arrows and Esc still act", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await switchOff(page);
    const before = await snapshot(page);
    await page.keyboard.press("ArrowRight");
    await expect.poll(async () => JSON.parse(await snapshot(page)).current).not.toBe(JSON.parse(before).current);
    await page.keyboard.press("Meta+=");
    await expect.poll(async () => JSON.parse(await snapshot(page)).mode).not.toEqual(JSON.parse(before).mode);
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(dialog).toBeHidden();
    // Crop's Return is a layer's key, not a shortcut: it still finishes the crop, and Esc ends it.
    await menu(page, "crop");
    await expect(page.locator(".ym-crop")).toBeVisible();
    await page.keyboard.press("Enter");
    await expect(page.locator(".ym-crop")).toHaveCount(0);
    await menu(page, "crop");
    await expect(page.locator(".ym-crop")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.locator(".ym-crop")).toHaveCount(0);
  });

  test("turning the switch on again gives the keys back", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await switchOff(page);
    await page.keyboard.press("p");
    await page.waitForTimeout(300);
    expect(JSON.parse(await snapshot(page)).picked).toBe(3);
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await dialog.getByRole("switch", { name: "Single-key shortcuts" }).click();
    await expect(dialog.getByRole("switch", { name: "Single-key shortcuts" })).toHaveAttribute("aria-checked", "true");
    await dialog.getByRole("button", { name: "Done" }).click();
    await expect(dialog).toBeHidden();
    await page.keyboard.press("p");
    await expect.poll(async () => JSON.parse(await snapshot(page)).picked).toBe(2);
  });

  test("the choice survives a restart: the page is reloaded against the mock's stored preferences and the switch is still off", async ({ page }) => {
    await page.goto(mockQuery({ mockPersist: 1, ...MOCKUP }));
    await expect(page.locator("[data-screen]")).toBeVisible();
    await switchOff(page);
    await page.reload();
    await expect(page.locator("[data-screen]")).toBeVisible();
    expect((await page.evaluate(() => window.__yellowmoon!.mockApi.preferencesGet())).single_key_shortcuts).toBe(false);
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("dialog", { name: "Preferences" }).getByRole("switch", { name: "Single-key shortcuts" })).toHaveAttribute("aria-checked", "false");
  });

  test("I on the zoomed photo is a single key too: off, it reads nothing and the description stops saying it; Home still centres", async ({ page }) => {
    await openApp(page, MOCKUP);
    await openPhotos(page);
    await switchOff(page);
    await page.getByRole("button", { name: /^(Fit|\d+%)/ }).click();
    await page.getByRole("menu").getByText("200%", { exact: true }).click();
    const photo = page.getByRole("img", { name: "Photo, zoomed. The arrow keys move it." });
    await expect(photo).toBeVisible();
    await expect(photo).not.toHaveAccessibleDescription(/I reads the pixel/);
    await photo.focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(() => page.evaluate(() => window.__yellowmoon!.mockApi.viewport.offset.x)).not.toBe(0);
    await page.keyboard.press("i");
    await page.waitForTimeout(400);
    expect(JSON.parse(await snapshot(page)).said).toBe("");
    await page.keyboard.press("Home");
    await expect.poll(() => page.evaluate(() => window.__yellowmoon!.mockApi.viewport.offset.x)).toBe(0);
  });
});
