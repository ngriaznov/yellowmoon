import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import type { Browser, BrowserContext, Page } from "@playwright/test";
import { openApp, settle } from "./helpers/app";
import { openState, STATES } from "./helpers/states";

// axe-core over every page state of the app (`docs/28` A-01, M4-04): each state of `STATES`, and the Photos screen with 500 photos (the
// filmstrip is windowed there). The rule sets are WCAG 2.0, 2.1 and 2.2 at A and AA, and axe's best practices. A finding is a violation, or
// an "incomplete" result (axe could not decide), because the audit's two attribute findings were of that kind.
//
// Every finding that is left is in `ALLOWED`, with the state it is allowed in and the reason. The test fails on a finding that is not
// listed, and also on a listed one that axe no longer reports in that state, so a fix removes its line and an entry cannot go stale.
// Colour contrast is the one rule switched off for all states (below).

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

/** The 500-photo Photos screen, as the mockup's current and picked photos on a 500-photo mock folder. */
const MANY = "many-photos-500";

/**
 * Colour contrast is switched off, in every state: the app's text sits on translucent glass (`backdrop-filter`), and axe cannot decide
 * contrast through it ("incomplete" for every text node of all 79 states). `docs/28` A-05 replaces it with a measurement on the pixels
 * (`e2e/contrast.spec.ts`).
 */
const OFF = ["color-contrast"];

/**
 * "Incomplete" results that are not counted: axe could not decide, and what it could not decide is Radix's own pattern. A modal dialog
 * or menu puts `aria-hidden` on the rest of the page (`hideOthers`), and axe asks a person to check that what is hidden cannot be
 * reached; Radix keeps Tab and the pointer out of it, which the focus-order test of A-04 pins. Every other incomplete result counts as
 * a finding, because the audit's `aria-prohibited-attr` and `aria-valid-attr-value` findings came as incomplete results.
 */
const UNDECIDED = ["aria-hidden-focus", "bypass"];

interface Allowed {
  rule: string;
  /** The states the finding is allowed in; each must report the rule, or the entry is stale. */
  states: readonly string[];
  reason: string;
}

/** The seven states with a menu open. */
const MENUS = [
  "43-nodes-add-node",
  "52-nodes-link-selected",
  "59-nodes-add-mask-menu",
  "66-film-add-node-menu",
  "70-film-paper-list",
  "77-mixers-add-node-menu",
  "80-mixers-layer-mode-menu",
] as const;

const RADIX_MENU =
  "A Radix menu is open. Radix's modal menus put aria-hidden on everything else (hideOthers), so the page looks empty to axe, and the header, graph and divider hold focusable content under it. Radix keeps Tab and the pointer out of them while the menu is open. A known pattern of the library, not ours.";

/** The node view's states with no menu open: the divider stands outside every landmark (A-02 fixes it). */
const NODE_VIEW = [
  "42-nodes-default",
  "51-nodes-graph-states",
  "53-nodes-input-colour",
  "54-nodes-corrector-light",
  "55-nodes-corrector-colour",
  "56-nodes-output-look",
  "57-nodes-mask-view",
  "58-nodes-masks-two-rows",
  "60-nodes-linear-card-handles",
  "61-nodes-colour-range-picking",
  "62-nodes-brush-card",
  "63-nodes-background-card",
  "64-nodes-object-card",
  "65-nodes-subject-card",
  "67-film-tab",
  "68-film-stock-picker",
  "69-film-print-tab",
  "71-film-slide-print-tab",
  "72-film-halation-tab",
  "73-film-grain-tab",
  "74-film-mask-tab",
  "75-film-output-note",
  "78-mixers-parallel-inputs",
  "79-mixers-layer-blend",
  "81-mixers-splitter-channels",
  "82-mixers-combiner-channels",
  "83-mixers-link-to-plus",
  "84-mixers-card-states",
] as const;

export const ALLOWED: readonly Allowed[] = [
  // Radix, an open menu (the audit's 2.2).
  { rule: "aria-hidden-focus", states: MENUS, reason: RADIX_MENU },
  { rule: "landmark-one-main", states: MENUS, reason: RADIX_MENU },
  { rule: "page-has-heading-one", states: MENUS, reason: RADIX_MENU },
  { rule: "region", states: MENUS, reason: RADIX_MENU + " The portal's wrapper is outside every landmark." },
  {
    rule: "scrollable-region-focusable",
    states: ["43-nodes-add-node", "52-nodes-link-selected", "59-nodes-add-mask-menu"],
    reason: RADIX_MENU + " The menu's own list scrolls; its items take the focus (roving), and the list itself is a role=menu that Radix manages.",
  },
  {
    rule: "region",
    states: ["40-dialog-preferences-providers"],
    reason: "The open Select's viewport is a Radix portal child of the dialog; it is not ours to put in a landmark.",
  },
  // Known, fixed by A-02 (docs/28).
  {
    rule: "scrollable-region-focusable",
    states: ["02-start-developing", "03-start-ready", "25-export-jpeg", "26-export-tiff", "27-export-png", "28-export-dng", "29-export-running"],
    reason: "Known, fixed by A-02: Start's list and the Export panel's strip scroll and have nothing to focus, so they are no keyboard stop.",
  },
  { rule: "region", states: NODE_VIEW, reason: "Known, fixed by A-02: the node view's divider is outside every landmark." },
  {
    rule: "aria-prohibited-attr",
    states: ["25-export-jpeg", "26-export-tiff", "27-export-png", "28-export-dng"],
    reason: 'Known, fixed by A-02: `<code aria-label="Command line">`, a label on an element that may not carry one.',
  },
  {
    rule: "aria-valid-attr-value",
    states: ["68-film-stock-picker"],
    reason: "Known, fixed by A-02: the Stock field's aria-controls names a popover that is not in the page yet.",
  },
];

type Finding = { rule: string; nodes: string[] };

/** Runs axe on the page as it stands. Violations and incomplete results, one entry per rule, with where each hit. */
async function scan(page: Page): Promise<Finding[]> {
  const result = await new AxeBuilder({ page }).withTags(TAGS).disableRules(OFF).analyze();
  const byRule = new Map<string, string[]>();
  for (const found of [...result.violations, ...result.incomplete.filter((r) => !UNDECIDED.includes(r.id))]) {
    const nodes = byRule.get(found.id) ?? [];
    for (const node of found.nodes) nodes.push(node.target.join(" "));
    byRule.set(found.id, nodes);
  }
  return [...byRule].map(([rule, nodes]) => ({ rule, nodes }));
}

/** Checks the findings of one state against the allowlist: nothing new, and nothing allowed that has gone. */
function check(name: string, findings: Finding[]): void {
  const allowed = ALLOWED.filter((entry) => entry.states.includes(name));
  const unlisted = findings.filter((f) => !allowed.some((a) => a.rule === f.rule));
  expect(
    unlisted.map((f) => `${f.rule}: ${f.nodes.slice(0, 4).join(" | ")}`),
    `axe findings in ${name} that are not in ALLOWED (fix them, or add an entry with a reason)`,
  ).toEqual([]);
  const stale = allowed.filter((a) => !findings.some((f) => f.rule === a.rule));
  expect(
    stale.map((a) => a.rule),
    `entries of ALLOWED that axe no longer reports in ${name} (remove them)`,
  ).toEqual([]);
}

/** The Photos screen with 500 photos: the click on Open photos waits longer than `openPhotos` does, because 500 photos take a while to develop. */
async function manyPhotos(browser: Browser): Promise<{ page: Page; context: BrowserContext }> {
  const context = await browser.newContext({ baseURL: "http://localhost:1420", viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4", mockPhotos: 500 });
  await page.getByRole("button", { name: "Open files…" }).click();
  await page.getByRole("button", { name: /Open photos/ }).click({ timeout: 30_000 });
  await expect(page.getByRole("tab", { name: "Look" })).toBeVisible();
  await expect(page.locator(".ym-filmstrip__thumb").first()).toBeVisible();
  await settle(page);
  return { page, context };
}

test.describe("axe-core over every page state", () => {
  // Each state opens its own page, so they run side by side: one after the other they would take about six minutes.
  test.describe.configure({ mode: "parallel" });
  for (const [name] of STATES) {
    test(name, async ({ browser }) => {
      const { page, context } = await openState(browser, name);
      try {
        check(name, await scan(page));
      } finally {
        await context.close();
      }
    });
  }

  test(MANY, async ({ browser }) => {
    test.setTimeout(90_000);
    const { page, context } = await manyPhotos(browser);
    try {
      check(MANY, await scan(page));
    } finally {
      await context.close();
    }
  });
});
