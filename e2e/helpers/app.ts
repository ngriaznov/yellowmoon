import type { Page } from "@playwright/test";
import { expect } from "@playwright/test";
import type { MockOptions } from "../../src/api.mock";
import "../../src/testHooks";

/** Options added to every page this process opens, under the options a test gives (see `useMockDefaults`). */
let mockDefaults: Record<string, string | number> = {};

/**
 * Adds `options` to every page `openApp` opens (and every `mockQuery`) until the returned function is called, so that a spec can run the states
 * of `helpers/states.ts` with an option on (the contrast test's display options, `?mockDisplay=`). A worker runs one test at a time, so the
 * setting is the test's own; always restore it in a `finally`.
 */
export function useMockDefaults(options: Record<string, string | number>): () => void {
  const before = mockDefaults;
  mockDefaults = { ...before, ...options };
  return () => {
    mockDefaults = before;
  };
}

/** The mock's query string for a state: `mockSpeed=Infinity` runs development at once. */
export function mockQuery(options: Record<string, string | number> = {}): string {
  const params = new URLSearchParams({ mockSpeed: "Infinity" });
  for (const [key, value] of Object.entries({ ...mockDefaults, ...options })) params.set(key, String(value));
  return `/?${params.toString()}`;
}

/** Loads every weight the screens use, so a capture never shows a fallback face (`document.fonts.ready` alone resolves before the first request). */
async function loadFonts(page: Page): Promise<void> {
  await page.evaluate(async () => {
    await Promise.all(["400", "600", "700", "800"].map((weight) => document.fonts.load(`${weight} 14px Figtree`)));
    await document.fonts.ready;
  });
}

/** Loads the app on the mock, and waits for the fonts. */
export async function openApp(page: Page, options: Record<string, string | number> = {}): Promise<void> {
  await page.goto(mockQuery(options));
  await loadFonts(page);
  await expect(page.locator("[data-screen]")).toBeVisible();
}

/** The hooks the app publishes in a browser (`src/testHooks.ts`): the very mock and updater it runs on. */
type Hooks = NonNullable<Window["__yellowmoon"]>;

/**
 * Imports a look package through the mock and waits until the import has finished and the page has settled on the result.
 *
 * Not `await mockApi.looksImport(...)` inside one `page.evaluate`: Chromium's DevTools holds the promise that `page.evaluate` awaits
 * only weakly, and a look import ends with the store refreshing its looks, a long run of microtasks that allocate. If a garbage
 * collection comes after the evaluated promise is settled and before DevTools has read it, DevTools answers "Promise was collected" and
 * Playwright reports it as "Execution context was destroyed, most likely because of a navigation" (a page that has not navigated at
 * all), about 1 run in 50 on a busy machine. Here the evaluate returns at once, and the wait is a poll on a flag.
 */
export async function importLooks(page: Page, path: string): Promise<void> {
  await page.evaluate((zip) => {
    const job: LooksImportJob = (window as unknown as { __looksImport: LooksImportJob }).__looksImport = { done: false };
    void window.__yellowmoon!.mockApi.looksImport(zip).then(
      () => {
        job.done = true;
      },
      (error: unknown) => {
        job.error = JSON.stringify(error);
        job.done = true;
      },
    );
  }, path);
  await page.waitForFunction(() => (window as unknown as { __looksImport: LooksImportJob }).__looksImport.done);
  const error = await page.evaluate(() => (window as unknown as { __looksImport: LooksImportJob }).__looksImport.error);
  if (error !== undefined) throw new Error(`Importing ${path} failed: ${error}`);
}

/** What `importLooks` leaves on the page. */
interface LooksImportJob {
  done: boolean;
  error?: string;
}

/** Changes the mock's options in the page, before a session is open. */
export async function configureMock(page: Page, options: MockOptions): Promise<void> {
  await page.evaluate((opts) => (window as unknown as { __yellowmoon: Hooks }).__yellowmoon.mockApi.configure(opts), options);
}

/** Makes the next call of a mock command reject with an error of the core. */
export async function failNext(page: Page, command: string, error: { scope: string; message: string }): Promise<void> {
  await page.evaluate(
    ([name, err]) => (window as unknown as { __yellowmoon: Hooks }).__yellowmoon.mockApi.failNext(name as never, err as never),
    [command, error] as const,
  );
}

/** Sends the updater a state, as the core's `update_state` event does. */
export async function setUpdateState(page: Page, state: Record<string, unknown>): Promise<void> {
  await page.evaluate((next) => (window as unknown as { __yellowmoon: Hooks }).__yellowmoon.updater.applyExternal(next as never), state);
}

/** Shows the update cell as "Update ready", as every mockup screenshot does. */
export async function showUpdateReady(page: Page): Promise<void> {
  await setUpdateState(page, { phase: "ready", version: "0.1.1" });
  await expect(page.getByText("Update ready")).toBeVisible();
}

/**
 * Opens the photos from Start and goes to the Photos screen, once every photo is developed. "Open photos →" is enabled as soon as
 * the first one is ready and then opens the first ready photo, so clicking early opens a different photo from the one a test asked
 * for with `mockCurrent`; waiting for the end makes it the same on a slow machine and a fast one.
 */
export async function openPhotos(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Open files…" }).click();
  // Developed; with the mock's provider `pending` the photos stay "Grading N photos with Apple Intelligence" (or Claude Code, or Antigravity; `docs/11` §Start).
  await expect(
    page.getByRole("heading", { level: 2, name: /^(\d+ photos? ready|Grading \d+ photos? with (Apple Intelligence|Claude Code|Antigravity))$/ }),
  ).toBeVisible();
  await page.getByRole("button", { name: /Open photos/ }).click();
  await expect(page.getByRole("tab", { name: "Look" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();
}

/**
 * Waits until every look tile that shows in the tool panel, even by a few pixels, has its picture. A tile asks for its picture when
 * it first scrolls into view, so a capture taken after "two tiles have loaded" can still catch the ones along the bottom edge half
 * drawn.
 */
export async function lookTilesLoaded(page: Page): Promise<void> {
  await page.waitForFunction(() => {
    const scroller = document.querySelector(".ym-tools__scroll");
    if (!scroller) return false;
    const box = scroller.getBoundingClientRect();
    const shown = [...document.querySelectorAll<HTMLElement>(".ym-look-tile")].filter((tile) => {
      const r = tile.getBoundingClientRect();
      return r.bottom > box.top && r.top < box.bottom;
    });
    return shown.length > 0 && shown.every((tile) => {
      const img = tile.querySelector("img");
      return !!img && img.complete && img.naturalWidth > 0;
    });
  });
}

/** Waits until every picture on the page has loaded, and two frames have been drawn. */
export async function settle(page: Page): Promise<void> {
  await loadFonts(page);
  await page.evaluate(async () => {
    const images = [...document.images];
    await Promise.all(images.map((img) => (img.complete ? Promise.resolve() : img.decode().catch(() => undefined))));
    await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
  });
}

/** Types a value into a slider's box. */
export async function setValue(page: Page, label: string, text: string): Promise<void> {
  const box = page.getByRole("textbox", { name: `${label} value` });
  await box.click();
  await box.fill(text);
  await box.press("Enter");
}

/** Collects console errors and page errors, so a test can assert there were none. */
export function watchErrors(page: Page): string[] {
  const problems: string[] = [];
  page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning") problems.push(`${message.type()}: ${message.text()}`);
  });
  return problems;
}

/** Drops files on the window, as the browser delivers them: a `drop` event with a `DataTransfer` of files. */
export async function dropFiles(page: Page, names: string[]): Promise<void> {
  await page.evaluate((list) => {
    const data = new DataTransfer();
    for (const name of list) data.items.add(new File([new Uint8Array(4)], name));
    for (const type of ["dragenter", "dragover", "drop"]) {
      window.dispatchEvent(new DragEvent(type, { dataTransfer: data, bubbles: true, cancelable: true }));
    }
  }, names);
}

/** What `recordCalls` keeps for each call of a command. */
export interface RecordedCall {
  args: unknown[];
}

/**
 * An arrow key in a radio list (the Lens tab's lenses), held until the focus has moved to the next row and let go after. Radix focuses the
 * next row a tick after `keydown` and chooses it only if the key is still down at that moment (`isArrowKeyPressedRef`, cleared on `keyup`),
 * so a key pressed and released at once, or held for a fixed 60 ms on a machine that is busy enough to run the tick after the release,
 * moves the focus and chooses nothing. A hand holds the key for 80 ms or more with an idle page behind it; this does the same without
 * counting on either.
 */
export async function arrowInList(page: Page, key: "ArrowDown" | "ArrowUp"): Promise<void> {
  const was = await page.evaluateHandle(() => document.activeElement);
  await page.keyboard.down(key);
  try {
    await page.waitForFunction((before) => document.activeElement !== before, was);
  } finally {
    await page.keyboard.up(key);
  }
}

/**
 * Wraps a command of the mock in the page so every call is recorded, and returns a function that
 * reads (and optionally clears) the record. The app looks the command up when it calls it, so the
 * wrapper sees every call.
 */
export async function recordCalls(page: Page, command: string): Promise<(clear?: boolean) => Promise<RecordedCall[]>> {
  await page.evaluate((name) => {
    const { mockApi } = (window as unknown as { __yellowmoon: Hooks }).__yellowmoon;
    const w = window as unknown as Record<string, unknown>;
    const key = `__calls_${name}`;
    w[key] = [];
    const api = mockApi as unknown as Record<string, (...a: unknown[]) => unknown>;
    const original = api[name as string]!.bind(mockApi);
    api[name as string] = (...args: unknown[]) => {
      (w[key] as unknown[][]).push(args);
      return original(...args);
    };
  }, command);
  return async (clear = false) => {
    const calls = await page.evaluate(
      ([name, wipe]) => {
        const w = window as unknown as Record<string, unknown[][]>;
        const key = `__calls_${name}`;
        const list = w[key] ?? [];
        if (wipe) w[key] = [];
        return list;
      },
      [command, clear] as const,
    );
    return calls.map((args) => ({ args }));
  };
}
