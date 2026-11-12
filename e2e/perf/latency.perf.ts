import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { openApp } from "../helpers/app";

// The browser-side half of M3-03 ("opening 500 files keeps the UI responsive", docs/11 Large folders, docs/13): the app on the mock
// with a folder of 500 photos, built for production, while the mock develops the photos in the background (about 2,500 progress
// events and 500 thumbnails), keys pressed meanwhile. The latency of a key is the time from the moment the browser made the key
// event (`KeyboardEvent.timeStamp`, so the wait behind whatever the page was busy with counts) to two frames after the filmstrip's
// caption ("4 of 12 · 3 picked") shows the next photo (the first frame carries the change, the second callback runs when that frame is out).
//
// The numbers are a report (test-results/perf/latency-N.json and latency-N.md per folder size, also attached to the test); the only assertions are that
// the measurement measured what it says (every key got an answer; photos were still developing at the last key) and a generous
// bound on the 95th percentile, so that a loaded runner does not fail it. The core's half is the ignored Rust test
// `five_hundred_files_leave_the_commands_responsive` (src-tauri/src/api/tests.rs). Run it with e2e/perf/run.sh.

// YM_PERF_PHOTOS: the folder sizes to measure, one number or a comma list ("100,500,2000"); default 500. With both 500 and 2000 in the
// list the run also asserts that the p95 at 2000 is at most RATIO_MAX times the p95 at 500 of the same run (docs/26 H-09).
const SIZES = (process.env.YM_PERF_PHOTOS ?? "500")
  .split(",")
  .map((n) => Number(n.trim()))
  .filter((n) => Number.isInteger(n) && n > 0);
const RATIO_MAX = Number(process.env.YM_PERF_RATIO ?? 1.25);
/** The p95 a size measured in this run, read from its report (a failed test restarts the worker, so nothing is kept in memory). */
const p95Of = (photos: number): number | undefined => {
  try {
    return (JSON.parse(readFileSync(join(OUT, `latency-${photos}.json`), "utf8")) as { p95_ms?: number }).p95_ms;
  } catch {
    return undefined;
  }
};
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

for (const PHOTOS of SIZES) {
test(`${PHOTOS} photos developing: a key is on screen two frames later, p95 well inside the bound`, async ({ page, browser }) => {
  await openApp(page, { mockPhotos: PHOTOS, mockSpeed: SPEED });
  await page.getByRole("button", { name: "Open files…" }).click();
  await expect(page.getByRole("heading", { level: 2, name: `Developing ${PHOTOS} photos` })).toBeVisible();
  await page.getByRole("button", { name: /Open photos/ }).click();
  await expect(page.getByRole("region", { name: "Automatic" })).toBeVisible();
  await expect(page.locator(".ym-filmstrip__caption")).toHaveText(`1 of ${PHOTOS} · 0 picked`);

  // How many photos are still developing, asked of the mock itself: the filmstrip draws only the thumbnails in view.
  const developing = (): Promise<number> =>
    page.evaluate(async () => {
      const hooks = (window as unknown as { __yellowmoon: { mockApi: { sessionState(): Promise<{ photos: { status: { kind: string } }[] }> } } }).__yellowmoon;
      return (await hooks.mockApi.sessionState()).photos.filter((p) => p.status.kind === "waiting" || p.status.kind === "developing").length;
    });

  // The instruments: the time of every keydown, the long tasks, and one sample per change of the caption.
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
    const caption = document.querySelector(".ym-filmstrip__caption");
    if (!caption) throw new Error("no filmstrip caption");
    let shown = caption.textContent;
    new MutationObserver(() => {
      const now = caption.textContent;
      if (now === shown) return;
      shown = now;
      const since = pressedAt;
      pressedAt = null;
      if (since === null) return;
      requestAnimationFrame(() => requestAnimationFrame(() => perf.samples.push(performance.now() - since)));
    }).observe(caption, { childList: true, characterData: true, subtree: true });
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
    photos: PHOTOS,
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
    `### Browser-side latency, ${PHOTOS} photos developing (production build)`,
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
  writeFileSync(join(OUT, `latency-${PHOTOS}.json`), `${JSON.stringify(stats, null, 2)}\n`);
  writeFileSync(join(OUT, `latency-${PHOTOS}.md`), markdown);
  await test.info().attach(`latency-${PHOTOS}.md`, { body: markdown, contentType: "text/markdown" });
  console.log(markdown);

  // The measurement measured what it says: the photos were still being developed while the keys were pressed.
  expect(developingAtFirstKey).toBeGreaterThan(PHOTOS >= 500 ? 100 : 0);
  if (PHOTOS >= 500) expect(developingAtLastKey).toBeGreaterThan(0);
  // A generous bound, for a runner that is busy with something else. The figure to watch is in the report.
  expect(stats.p95_ms).toBeLessThan(BOUND_P95_MS);
});
}

// The point of H-09: the cost of a key must not grow with the folder. A ratio of two figures of one run, so a slow runner does not matter.
if (SIZES.includes(500) && SIZES.includes(2000)) {
  test("a key at 2000 photos costs about what it costs at 500 (p95 ratio)", () => {
    const small = p95Of(500);
    const large = p95Of(2000);
    expect(small, "the 500-photo test ran first").toBeDefined();
    expect(large, "the 2000-photo test ran").toBeDefined();
    const ratio = large! / small!;
    const markdown = `### Key latency, 2000 against 500 photos\n\np95 ${large} ms at 2000, ${small} ms at 500: ratio ${round1(ratio)} (at most ${RATIO_MAX}).\n`;
    mkdirSync(OUT, { recursive: true });
    writeFileSync(join(OUT, "latency-ratio.md"), markdown);
    console.log(markdown);
    expect(ratio).toBeLessThanOrEqual(RATIO_MAX);
  });
}
