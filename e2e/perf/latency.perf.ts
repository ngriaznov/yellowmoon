import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { openApp } from "../helpers/app";

// The browser-side half of M3-03 ("opening 500 files keeps the UI responsive", docs/11 Large folders, docs/13): the app on the mock
// with a folder of 500 photos, built for production, while the mock develops the photos in the background (about 2,500 progress
// events and 500 thumbnails), keys pressed meanwhile. The latency of a key is the time from the moment the browser made the key
// event (`KeyboardEvent.timeStamp`, so the wait behind whatever the page was busy with counts) to two frames after the filmstrip's
// counter shows the next photo (the first frame carries the change, the second callback runs when that frame is out).
//
// The numbers are a report (test-results/perf/latency.json and latency.md, also attached to the test); the only assertions are that
// the measurement measured what it says (every key got an answer; photos were still developing at the last key) and a generous
// bound on the 95th percentile, so that a loaded runner does not fail it. The core's half is the ignored Rust test
// `five_hundred_files_leave_the_commands_responsive` (src-tauri/src/api/tests.rs). Run it with e2e/perf/run.sh.

const KEYS = Number(process.env.YM_PERF_KEYS ?? 80);
const SPEED = Number(process.env.YM_PERF_SPEED ?? 10);
const BOUND_P95_MS = Number(process.env.YM_PERF_P95_MS ?? 150);
const GAP_MS = 100;
const OUT = process.env.YM_PERF_OUT ?? "test-results/perf";

/** Nearest-rank percentile of a sorted list. */
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return Number.NaN;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1))]!;
}

const round1 = (n: number): number => Math.round(n * 10) / 10;

interface InPage {
  samples: number[];
  longTasks: number[];
}

declare global {
  interface Window {
    __perf?: InPage;
  }
}

test("500 photos developing: a key is on screen two frames later, p95 well inside the bound", async ({ page, browser }) => {
  await openApp(page, { mockPhotos: 500, mockSpeed: SPEED });
  await page.getByRole("button", { name: "Open files…" }).click();
  await expect(page.getByRole("heading", { level: 2, name: "Developing 500 photos" })).toBeVisible();
  await page.getByRole("button", { name: /Open photos/ }).click();
  await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();
  await expect(page.locator(".ym-filmstrip__count")).toHaveText("1 / 500");

  const developing = (): Promise<number> =>
    page.locator(".ym-filmstrip button").evaluateAll((buttons) => buttons.filter((b) => /, developing$/.test(b.getAttribute("aria-label") ?? "")).length);

  // The instruments: the time of every keydown, the long tasks, and one sample per change of the counter.
  await page.evaluate(() => {
    const perf: InPage = { samples: [], longTasks: [] };
    window.__perf = perf;
    let pressedAt: number | null = null;
    window.addEventListener(
      "keydown",
      (event) => {
        if (event.key === "ArrowRight") pressedAt = event.timeStamp;
      },
      true,
    );
    const counter = document.querySelector(".ym-filmstrip__count");
    if (!counter) throw new Error("no filmstrip counter");
    let shown = counter.textContent;
    new MutationObserver(() => {
      const now = counter.textContent;
      if (now === shown) return;
      shown = now;
      const since = pressedAt;
      pressedAt = null;
      if (since === null) return;
      requestAnimationFrame(() => requestAnimationFrame(() => perf.samples.push(performance.now() - since)));
    }).observe(counter, { childList: true, characterData: true, subtree: true });
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) perf.longTasks.push(entry.duration);
    }).observe({ type: "longtask", buffered: true });
  });

  const developingAtFirstKey = await developing();
  for (let i = 0; i < KEYS; i++) {
    await page.keyboard.press("ArrowRight");
    await page.waitForTimeout(GAP_MS);
  }
  await expect.poll(() => page.evaluate(() => window.__perf?.samples.length ?? 0), { timeout: 30_000 }).toBe(KEYS);
  const developingAtLastKey = await developing();
  const { samples, longTasks } = await page.evaluate(() => window.__perf as InPage);

  const sorted = [...samples].sort((a, b) => a - b);
  const stats = {
    chromium: browser.version(),
    photos: 500,
    mock_speed: SPEED,
    keys: KEYS,
    gap_ms: GAP_MS,
    min_ms: round1(sorted[0] ?? Number.NaN),
    median_ms: round1(percentile(sorted, 0.5)),
    p95_ms: round1(percentile(sorted, 0.95)),
    max_ms: round1(sorted.at(-1) ?? Number.NaN),
    bound_p95_ms: BOUND_P95_MS,
    long_tasks: longTasks.length,
    longest_task_ms: round1(Math.max(0, ...longTasks)),
    long_tasks_total_ms: round1(longTasks.reduce((a, b) => a + b, 0)),
    developing_at_first_key: developingAtFirstKey,
    developing_at_last_key: developingAtLastKey,
    samples_ms: samples.map(round1),
  };

  const markdown = [
    "### Browser-side latency, 500 photos developing (production build)",
    "",
    `Chromium ${stats.chromium}, ${stats.keys} presses of the right arrow ${stats.gap_ms} ms apart while the mock developed the photos at speed ${stats.mock_speed}.`,
    "Time from the key event to two frames after the filmstrip shows the next photo.",
    "",
    "| min | median | p95 | max | bound on p95 |",
    "|---:|---:|---:|---:|---:|",
    `| ${stats.min_ms} ms | ${stats.median_ms} ms | ${stats.p95_ms} ms | ${stats.max_ms} ms | ${stats.bound_p95_ms} ms |`,
    "",
    `Long tasks (50 ms or more) during the run: ${stats.long_tasks}, the longest ${stats.longest_task_ms} ms, ${stats.long_tasks_total_ms} ms in all.`,
    `Photos still developing at the first key: ${stats.developing_at_first_key}; at the last: ${stats.developing_at_last_key}.`,
    "",
  ].join("\n");
  mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, "latency.json"), `${JSON.stringify(stats, null, 2)}\n`);
  writeFileSync(join(OUT, "latency.md"), markdown);
  await test.info().attach("latency.md", { body: markdown, contentType: "text/markdown" });
  console.log(markdown);

  // The measurement measured what it says: the photos were still being developed while the keys were pressed.
  expect(developingAtFirstKey).toBeGreaterThan(100);
  expect(developingAtLastKey).toBeGreaterThan(0);
  // A generous bound, for a runner that is busy with something else. The figure to watch is in the report.
  expect(stats.p95_ms).toBeLessThan(BOUND_P95_MS);
});
