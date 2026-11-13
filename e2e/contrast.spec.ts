import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { collectText, groundsOf, measureText, ringContrast } from "./helpers/contrast";
import type { Box, Measured } from "./helpers/contrast";
import { useMockDefaults } from "./helpers/app";
import { openState } from "./helpers/states";

// Contrast as drawn (`docs/28` A-05, M4-04; the method of its section 2.5). axe cannot decide contrast through the glass's `backdrop-filter`
// (it reports every text node as "incomplete"), so its `color-contrast` rule is off (`e2e/axe.spec.ts`) and this is what replaces it: for each
// of 25 states, every visible text element is found, its colour is read from the page (the alpha recovered through a canvas), and what is
// behind it is read from a screenshot (the commonest colour of its box). The ratio is WCAG 2.x: 4.5:1 for text, 3:1 when it is large (24 px, or
// 18.66 px and bold).
//
// Backdrops. The panes are glass over the app's aurora, so the text of the panes is measured on the aurora as it is, and on black as the
// stress case. What floats over the photo is measured with the photo cell painted white, mid grey (#808080) and black and the picture hidden, and
// only the texts that overlap the cell count there (a photo is any colour; the cell is cut out of the aurora, so a white photo is never behind a
// pane). The Fit menu is opened, and the Fit button's tooltip shown, and measured over the same three. The crop frame's ring and its handles are measured the same way (the ring's
// pixels, the ones the focus changes, against the pixels they replace).
//
// Asserts. 0 non-disabled text under its limit, on every backdrop; 0 rings under 3:1; a disabled button's key hint at 3:1 (disabled text is
// exempt from 1.4.3 and is only listed, but the hint of a disabled Export is the one way to learn the shortcut). What fails and cannot
// be fixed at once is in `EXPECTED`, each with the task that fixes it (A-06 emptied it: every text over the photo reads at 4.5:1 and every ring at 3:1).
// The test is green while an entry fails and red when an entry stops failing (so a fix has to remove its line, the rule of the allowlist of
// `axe.spec.ts`) and red for anything that fails and is not listed.
//
// Variants (`docs/28` A-07). The whole file runs three times: as it is, with Reduce Transparency set (`?mockDisplay=reduce-transparency`: the shell's
// attribute, so the real wiring runs), and with the three display options set. Under Reduce Transparency every menu, popover, dialog and tooltip is
// solid, and since A-06 they are never under 88 % tint, so they pass in every variant; the camera line has its own pill, the crop ring its dark
// lines and the key hint of a disabled Export is not dimmed. Each `EXPECTED` entry says in which
// variants it fails, and in each variant the test is red for an entry that stopped failing, as it is red for a failure that is not listed.
//
// The last test of a variant prints the table of 2.5 (the worst ratio per kind of text and where it is), and the disabled texts under 4.5.

/** A way to run the whole file: the options of the mock that turn the display options on. */
interface Variant {
  name: "default" | "reduce-transparency" | "all-three";
  options: Record<string, string>;
}
const VARIANTS: readonly Variant[] = [
  { name: "default", options: {} },
  { name: "reduce-transparency", options: { mockDisplay: "reduce-transparency" } },
  { name: "all-three", options: { mockDisplay: "reduce-transparency,reduce-motion,increase-contrast" } },
];
type VariantName = Variant["name"];

type Backdrop = "aurora" | "black" | "cell-white" | "cell-grey" | "cell-black";
const PANE_BACKDROPS: readonly Backdrop[] = ["aurora", "black"];
const CELL_BACKDROPS: readonly Backdrop[] = ["cell-white", "cell-grey", "cell-black"];

const CELL = (colour: string): string =>
  `.ym-photo-cell { background: ${colour} !important; } .ym-photo-cell__image, .ym-photo-cell__frame { visibility: hidden !important; }`;
const BLACK_GROUND =
  ".ym-aurora, [data-slot=aurora], [data-slot=aurora-blob] { background: #000 !important; } .ym-aurora * { background: transparent !important; } html, body, #root { background: #000 !important; }";
const CSS: Record<Backdrop, string> = {
  aurora: "",
  black: BLACK_GROUND,
  "cell-white": CELL("#fff"),
  "cell-grey": CELL("#808080"),
  "cell-black": CELL("#000"),
};

/** The 25 states of the audit: the panes on every tab, the modes, the dialogs, the Export panel, the node view and the wizard. */
const STATE_NAMES = [
  "01-start-empty", "03-start-ready", "10-tab-look", "11-tab-light", "12-tab-colour", "13-tab-detail", "14-tab-curve", "15-tab-raw", "37-tab-lens",
  "16-mode-crop", "17-mode-pick-white", "20-dialog-preferences", "22-dialog-about", "24-error-bar", "25-export-jpeg", "29-export-running",
  "31-colour-warp", "42-nodes-default", "54-nodes-corrector-light", "58-nodes-masks-two-rows", "05-wizard-1-choose", "67-film-tab",
  "78-mixers-parallel-inputs", "19-pixel-readout", "18-before-on",
] as const;
const FIT_MENU = "fit-menu";
const TOOLTIP = "tooltip";
const CROP_RING = "crop-ring";

/** Where the results of the states are kept for the table (the tests run in several workers). */
const RESULTS = resolve(import.meta.dirname, "../test-results/contrast-rendered");

/** What a measurement is: text of a pane, of the photo cell, of a menu, a key hint, a ring. */
type Kind = "body" | "secondary" | "tertiary" | "other" | "placeholder" | "large" | "disabled" | "chip" | "camera" | "menu" | "key-hint" | "ring";

/** One thing the test judges. */
interface Finding {
  kind: Kind;
  /** The text, or the name of the ring. */
  label: string;
  cls: string;
  ratio: number;
  need: number;
}

function kindOf(m: Measured, inMenu: boolean): Kind {
  if (m.tag === "ring") return "ring";
  if (m.keycap && m.disabled) return "key-hint";
  if (m.disabled) return "disabled";
  if (inMenu) return "menu";
  if (m.placeholder) return "placeholder";
  if (m.cls.includes("ym-photo-cell__camera")) return "camera";
  if (m.cls.includes("ym-photo-cell__")) return "chip";
  if (m.need === 3) return "large";
  return m.token === "label" ? "body" : m.token;
}

/** A thing that fails and is allowed to, until its task lands. Each names the task that fixes it, and in which variants, where and over what it fails. */
interface Expected {
  /** The variants in which it fails (the others must pass: an entry that fails nowhere it is listed is stale). */
  variants: readonly VariantName[];
  /** The element, in words. */
  element: string;
  /** The task that fixes it (`docs/28`). */
  task: string;
  /** The states (or `fit-menu`, `crop-ring`) in which it must fail, and the backdrops it must fail on in each. */
  scopes: readonly string[];
  backdrops: readonly Backdrop[];
  match: (f: Finding) => boolean;
  /** The ratios it had when the entry was written, for whoever reads the failure. */
  measured: string;
  reason: string;
}

/**
 * What fails today: nothing. `docs/28` A-05 listed five things here, each with the task that fixes it, and A-06 fixed them and removed their lines:
 * the camera line (now the chip's dark pill), the Fit menu and the dialogs' secondary text over a white photo (the floor of 88 % for floating
 * layers), the crop frame's ring (the dark line on each side) and the keycap of a disabled Export (no longer dimmed). The list stays, empty, so
 * that a failure that cannot be fixed at once can be named here, with its task and its variants, instead of being left out of the test: the test is
 * red when an entry stops failing and for anything that fails and is not here.
 */
export const EXPECTED: readonly Expected[] = [];

/** Finds what is judged wrong in one scope on one backdrop: the findings under their limit, and which of the expected failures cover them. */
function judge(variant: VariantName, scope: string, backdrop: Backdrop, findings: readonly Finding[]): { unexpected: string[]; stale: string[] } {
  const covering = EXPECTED.filter((e) => e.variants.includes(variant) && e.scopes.includes(scope) && e.backdrops.includes(backdrop));
  const failing = findings.filter((f) => f.ratio < f.need);
  const unexpected = failing.filter((f) => !covering.some((e) => e.match(f))).map((f) => `${backdrop}: ${f.kind} "${f.label}" ${f.ratio.toFixed(2)}:1 (needs ${f.need}) ${f.cls.split(" ").slice(0, 3).join(" ")}`);
  const stale = covering.filter((e) => !failing.some((f) => e.match(f))).map((e) => `${backdrop}: "${e.element}" (${e.task}) no longer fails`);
  return { unexpected, stale };
}

/** Two animation frames: what a style change needs to be drawn, with no timer. */
async function frames(page: Page): Promise<void> {
  await page.evaluate(() => new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))));
}

/** Where the cell is, so that only what overlaps it is judged over the photo. */
async function cellBox(page: Page): Promise<Box | null> {
  return page.evaluate(() => {
    const cell = document.querySelector(".ym-photo-cell");
    if (!cell) return null;
    const r = cell.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height };
  });
}

const overlaps = (a: Box, b: Box): boolean => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

/** Draws one backdrop, waits for it to be drawn, and measures the texts (those overlapping `only` when it is given, and inside `within` when it is given). */
async function measureBackdrop(page: Page, backdrop: Backdrop, only?: { cell?: Box; within?: Box }): Promise<Array<{ m: Measured; inMenu: boolean }>> {
  const style = CSS[backdrop] ? await page.addStyleTag({ content: CSS[backdrop] }) : null;
  await frames(page);
  const png = await page.screenshot({ animations: "disabled" });
  const { items, labels } = await collectText(page);
  const kept = items.filter((it) => (!only?.cell || overlaps(it, only.cell)) && (!only?.within || overlaps(it, only.within)));
  const measured = measureText(kept, await groundsOf(page, png, kept), labels);
  await style?.evaluate((el) => (el as Element).remove());
  return measured.map((m) => ({ m, inMenu: only?.within !== undefined }));
}

const findingOf = (m: Measured, inMenu: boolean): Finding => ({ kind: kindOf(m, inMenu), label: m.text, cls: m.cls, ratio: m.ratio, need: m.keycap && m.disabled ? 3 : m.need });

/** The findings that the asserts are made of: non-disabled text, and the disabled key hints. The other disabled texts are listed, not judged. */
const judged = (rows: ReadonlyArray<{ m: Measured; inMenu: boolean }>): Finding[] =>
  rows.map(({ m, inMenu }) => findingOf(m, inMenu)).filter((f) => f.kind !== "disabled");

interface Saved {
  variant: VariantName;
  scope: string;
  run: number;
  backdrops: Record<string, Array<{ kind: Kind; text: string; cls: string; ratio: number; need: number; size: number; disabled: boolean; bg: number[] }>>;
}

function save(variant: VariantName, scope: string, backdrops: Record<string, ReadonlyArray<{ m: Measured; inMenu: boolean }>>): void {
  mkdirSync(RESULTS, { recursive: true });
  const out: Saved = { variant, scope, run: process.ppid, backdrops: {} };
  for (const [name, rows] of Object.entries(backdrops)) {
    out.backdrops[name] = rows.map(({ m, inMenu }) => ({ kind: kindOf(m, inMenu), text: m.text, cls: m.cls.split(" ").slice(0, 4).join(" "), ratio: +m.ratio.toFixed(2), need: m.keycap && m.disabled ? 3 : m.need, size: m.size, disabled: m.disabled, bg: m.bg.map(Math.round) }));
  }
  writeFileSync(resolve(RESULTS, `${variant}__${scope}.json`), JSON.stringify(out));
}

for (const variant of VARIANTS) {
test.describe(variant.name === "default" ? "contrast as drawn" : `contrast as drawn, ${variant.name}`, () => {
  // One page per state, so they run side by side like the axe states.
  test.describe.configure({ mode: "parallel" });

  for (const name of STATE_NAMES) {
    test(name, async ({ browser }) => {
      test.setTimeout(90_000);
      const restore = useMockDefaults(variant.options);
      const { page, context } = await openState(browser, name).finally(restore);
      try {
        // The Start screen, the wizard and the error bar have no photo cell: only the panes are measured there.
        const cell = await cellBox(page);
        const backdrops = cell ? [...PANE_BACKDROPS, ...CELL_BACKDROPS] : PANE_BACKDROPS;
        const result: Record<string, Array<{ m: Measured; inMenu: boolean }>> = {};
        const problems: string[] = [];
        for (const backdrop of backdrops) {
          result[backdrop] = await measureBackdrop(page, backdrop, CELL_BACKDROPS.includes(backdrop) ? { cell: cell as Box } : undefined);
          const { unexpected, stale } = judge(variant.name, name, backdrop, judged(result[backdrop]!));
          problems.push(...unexpected, ...stale);
        }
        save(variant.name, name, result);
        expect(problems, `${name}: contrast that is not listed in EXPECTED (fix it, or list it with its task), or listed and fixed (remove the entry)`).toEqual([]);

        // The disabled texts are listed (not asserted), the way the audit listed them.
        const disabled = [...PANE_BACKDROPS].flatMap((b) => result[b]!.filter(({ m }) => m.disabled && !m.keycap && m.ratio < m.need).map(({ m }) => `${m.text} ${m.ratio.toFixed(2)}`));
        if (disabled.length > 0) console.log(`${name}: disabled text under its limit (exempt): ${[...new Set(disabled)].join("; ")}`);
      } finally {
        await context.close();
      }
    });
  }

  test(FIT_MENU, async ({ browser }) => {
    test.setTimeout(90_000);
    const restore = useMockDefaults(variant.options);
    const { page, context } = await openState(browser, "10-tab-look").finally(restore);
    try {
      await page.getByRole("button", { name: /^Fit/ }).click();
      const menu = page.getByRole("menu");
      await menu.waitFor();
      const box = (await menu.boundingBox()) as { x: number; y: number; width: number; height: number };
      const cell = (await cellBox(page)) as Box;
      const within: Box = { x: box.x, y: box.y, w: box.width, h: box.height };
      const result: Record<string, Array<{ m: Measured; inMenu: boolean }>> = {};
      const problems: string[] = [];
      for (const backdrop of CELL_BACKDROPS) {
        // The part of the menu over the cell: what the photo is behind.
        result[backdrop] = await measureBackdrop(page, backdrop, { cell, within });
        expect(result[backdrop]!.length, "the menu's items are found").toBeGreaterThan(0);
        const { unexpected, stale } = judge(variant.name, FIT_MENU, backdrop, judged(result[backdrop]!));
        problems.push(...unexpected, ...stale);
      }
      save(variant.name, FIT_MENU, result);
      expect(problems, "the Fit menu: contrast that is not listed in EXPECTED, or listed and fixed").toEqual([]);
    } finally {
      await context.close();
    }
  });

  test(TOOLTIP, async ({ browser }) => {
    test.setTimeout(90_000);
    const restore = useMockDefaults(variant.options);
    const { page, context } = await openState(browser, "10-tab-look").finally(restore);
    try {
      // The Fit button's tooltip opens under the header, over the top of the photo.
      await page.getByRole("button", { name: /^Fit/ }).hover();
      const tip = page.locator('[data-slot="tooltip-content"]');
      await tip.waitFor();
      const box = (await tip.boundingBox()) as { x: number; y: number; width: number; height: number };
      const cell = (await cellBox(page)) as Box;
      const within: Box = { x: box.x, y: box.y, w: box.width, h: box.height };
      const result: Record<string, Array<{ m: Measured; inMenu: boolean }>> = {};
      const problems: string[] = [];
      for (const backdrop of CELL_BACKDROPS) {
        result[backdrop] = await measureBackdrop(page, backdrop, { cell, within });
        const { unexpected, stale } = judge(variant.name, TOOLTIP, backdrop, judged(result[backdrop]!));
        problems.push(...unexpected, ...stale);
      }
      // Only the part of the tooltip that is over the cell has a photo behind it; the rest is over the header's glass and the aurora.
      expect(overlaps(within, cell), "the tooltip is over the cell").toBe(true);
      expect(Object.values(result).flat().length, "the tooltip's text is found").toBeGreaterThan(0);
      save(variant.name, TOOLTIP, result);
      expect(problems, "the tooltip: contrast that is not listed in EXPECTED, or listed and fixed").toEqual([]);
    } finally {
      await context.close();
    }
  });

  test(CROP_RING, async ({ browser }) => {
    test.setTimeout(90_000);
    const restore = useMockDefaults(variant.options);
    const { page, context } = await openState(browser, "16-mode-crop").finally(restore);
    try {
      const targets = [".ym-crop__frame", ...Array.from({ length: 8 }, (_, i) => `.ym-crop__handle >> nth=${i}`)];
      const problems: string[] = [];
      const rows: Record<string, Array<{ m: Measured; inMenu: boolean }>> = {};
      for (const backdrop of CELL_BACKDROPS) {
        const style = await page.addStyleTag({ content: CSS[backdrop] });
        const findings: Finding[] = [];
        for (const [index, target] of targets.entries()) {
          const el = page.locator(target);
          await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
          await frames(page);
          const before = await page.screenshot({ animations: "disabled" });
          // A key first, so that the focus is drawn as it is for the keyboard (`:focus-visible`), then the focus.
          await page.keyboard.press("Shift");
          await el.focus();
          expect(await el.evaluate((node) => node.matches(":focus-visible")), `${target} shows the keyboard ring`).toBe(true);
          await frames(page);
          const after = await page.screenshot({ animations: "disabled" });
          const rect = (await el.boundingBox()) as { x: number; y: number; width: number; height: number };
          // 12 px around the element, enough for the 3 px ring, its offset and a line outside it.
          const { ratio, pixels } = await ringContrast(page, before, after, { x: rect.x - 12, y: rect.y - 12, w: rect.width + 24, h: rect.height + 24 });
          findings.push({ kind: "ring", label: index === 0 ? "crop frame" : `crop handle ${index}`, cls: "", ratio: pixels === 0 ? 1 : ratio, need: 3 });
        }
        await style.evaluate((node) => (node as Element).remove());
        const { unexpected, stale } = judge(variant.name, CROP_RING, backdrop, findings);
        problems.push(...unexpected, ...stale);
        rows[backdrop] = findings.map((f) => ({
          m: { text: f.label, cls: "", tag: "ring", size: 0, disabled: false, keycap: false, placeholder: false, fg: [0, 0, 0], bg: [0, 0, 0], ratio: f.ratio, need: 3, token: "other" },
          inMenu: false,
        }));
      }
      save(variant.name, CROP_RING, rows);
      expect(problems, "the crop frame's rings: contrast that is not listed in EXPECTED, or listed and fixed").toEqual([]);
    } finally {
      await context.close();
    }
  });

  test(`the table of docs/28 section 2.5 (${variant.name})`, async ({}, testInfo) => {
    test.setTimeout(150_000);
    const scopes = [...STATE_NAMES, FIT_MENU, TOOLTIP, CROP_RING];
    // The states run in other workers: wait for their results of this run (the runner is every worker's parent), until all are in or no new one has come for 20 s.
    const mine = (): Saved[] =>
      existsSync(RESULTS)
        ? readdirSync(RESULTS)
            .filter((f) => f.endsWith(".json"))
            .map((f) => JSON.parse(readFileSync(resolve(RESULTS, f), "utf8")) as Saved)
            .filter((s) => s.run === process.ppid && s.variant === variant.name && scopes.includes(s.scope))
        : [];
    let seen = 0;
    let idle = 0;
    while (mine().length < scopes.length && idle < 20) {
      await new Promise((done) => setTimeout(done, 1000));
      const now = mine().length;
      idle = now === seen ? idle + 1 : 0;
      seen = now;
    }
    const table = renderTable(variant.name, mine());
    console.log(table);
    await testInfo.attach(`contrast-table-${variant.name}.txt`, { body: table, contentType: "text/plain" });
  });
});
}

/** The table of `docs/28` 2.5: per kind of text, the worst ratio, where it is, and over what. */
function renderTable(variant: VariantName, saved: readonly Saved[]): string {
  type Row = Saved["backdrops"][string][number];
  const worst = (kind: Kind, backdrops: readonly string[], keep: (r: Row) => boolean = () => true): string => {
    let best: { row: Row; scope: string; backdrop: string } | null = null;
    for (const s of saved) {
      for (const b of backdrops) {
        for (const row of s.backdrops[b] ?? []) {
          if (row.kind !== kind || !keep(row)) continue;
          if (!best || row.ratio < best.row.ratio) best = { row, scope: s.scope, backdrop: b };
        }
      }
    }
    return best ? `${best.row.ratio.toFixed(2)} (limit ${best.row.need}) "${best.row.text}" in ${best.scope} on ${best.backdrop}` : "none";
  };
  const lines = ["", `Contrast as drawn (${variant}), the worst ratio per kind of text (docs/28 section 2.5)`, `${saved.length} scopes, ${saved.reduce((n, s) => n + Object.values(s.backdrops).reduce((k, rows) => k + rows.length, 0), 0)} measurements`, ""];
  const panes = [["Body and labels", "body"], ["Secondary", "secondary"], ["Tertiary", "tertiary"], ["Other colours", "other"], ["Placeholder", "placeholder"], ["Large (48 px step numbers)", "large"], ["Disabled", "disabled"], ["Key hint of a disabled button", "key-hint"]] as const;
  lines.push("On the panes (the aurora and black):");
  for (const [title, kind] of panes) lines.push(`  ${title.padEnd(32)} ${worst(kind, PANE_BACKDROPS)}`);
  lines.push("", "Over the photo (the cell white, grey and black; the worst on each):");
  for (const [title, kind] of [["Chips and the readout", "chip"], ["The camera line", "camera"], ["The Fit menu", "menu"]] as const) {
    for (const backdrop of CELL_BACKDROPS) lines.push(`  ${(backdrop === "cell-white" ? title : "").padEnd(32)} ${backdrop.padEnd(11)} ${worst(kind, [backdrop])}`);
  }
  for (const backdrop of CELL_BACKDROPS) {
    const rings = saved.find((s) => s.scope === CROP_RING)?.backdrops[backdrop] ?? [];
    const low = [...rings].sort((a, b) => a.ratio - b.ratio)[0];
    lines.push(`  ${(backdrop === "cell-white" ? "Crop frame and handles (ring)" : "").padEnd(32)} ${backdrop.padEnd(11)} ${low ? `${low.ratio.toFixed(2)} (limit 3) "${low.text}"` : "none"}`);
  }
  const disabled = new Map<string, number>();
  for (const s of saved) for (const b of PANE_BACKDROPS) for (const row of s.backdrops[b] ?? []) if (row.kind === "disabled" && row.ratio < row.need) disabled.set(row.text, Math.min(row.ratio, disabled.get(row.text) ?? 99));
  lines.push("", "Disabled text under its limit (exempt from 1.4.3, listed):", `  ${[...disabled].sort((a, b) => a[1] - b[1]).map(([t, r]) => `${t} ${r.toFixed(2)}`).join("; ") || "none"}`);
  lines.push("", "Non-disabled text under its limit on the aurora and on black, in all states:");
  const under = saved.flatMap((s) => PANE_BACKDROPS.flatMap((b) => (s.backdrops[b] ?? []).filter((r) => !r.disabled && r.ratio < r.need).map((r) => `${s.scope}/${b}: ${r.text} ${r.ratio.toFixed(2)}`)));
  lines.push(`  ${under.length === 0 ? "0" : under.join("; ")}`, "");
  return lines.join("\n");
}
