import type { Page } from "@playwright/test";

// The Tab walk of `e2e/focus-order.spec.ts` (`docs/28` A-04): press Tab from the start of the page and write down every stop, until the
// focus wraps (a stop comes round again, as in a dialog that traps it) or leaves the document (the browser's own controls; `body` here).

/** How a focused control shows that it has the focus. */
export type RingKind =
  /** An `outline` of at least 2 px: what glasscn's `glass-focus` and our own rules draw. */
  | "outline"
  /** No outline, but a hard `box-shadow` of at least 2 px (no blur, no offset), which a ring can be drawn with as well. */
  | "shadow"
  /** The control draws none itself but one of its two parents does (glasscn's search field puts `focus-within:outline` on its capsule). */
  | "parent"
  | "none";

/** One Tab stop. */
export interface Stop {
  /** The role Playwright's accessibility tree gives it (`button`, `tab`, `slider`, `region`, `generic` when it has none). */
  role: string;
  /** Its accessible name, as that tree computes it. */
  name: string;
  /** The computed `outline-style` while it has the focus: `solid` for every ring we draw. */
  outline: string;
  /** How the focus shows. `none` is a stop the person cannot see. */
  ring: RingKind;
}

/** The stops of one page state, in Tab order, and how the walk ended. */
export interface Walk {
  /** `wraps`: the first stop came round again (a dialog keeps Tab inside); `leaves`: the focus left the document. */
  end: "wraps" | "leaves";
  stops: Stop[];
}

/** More stops than any screen has (the largest is the Photos screen's 43); reaching it is a keyboard trap or a runaway list. */
export const MAX_STOPS = 120;

/**
 * How light a computed colour is, 0 to 1: the `L` of `oklab(…)` and `oklch(…)`, the mean of the channels of `rgb(…)`. A ring has to stand out
 * from the dark glass; glasscn's `glass-focus` also draws a 2 px box-shadow in the glass tint, as a gap between the control and its outline,
 * which is dark and must not be counted as a ring.
 */
function lightness(colour: string): number {
  const numbers = (/\(([^)]*)\)/.exec(colour)?.[1] ?? "").split(/[\s,/]+/).map(Number.parseFloat);
  if (/^ok(?:lab|lch)/.test(colour)) return numbers[0] ?? 0;
  if (/^rgba?/.test(colour)) return ((numbers[0] ?? 0) + (numbers[1] ?? 0) + (numbers[2] ?? 0)) / 3 / 255;
  return 1;
}

/**
 * How a control with this computed style shows the focus. `outlineStyle`, `outlineWidth` and `boxShadow` are the strings
 * `getComputedStyle` gives. Pure, so the spec can test it on strings without a browser.
 */
export function classifyRing(outlineStyle: string, outlineWidth: string, boxShadow: string): RingKind {
  if (outlineStyle !== "none" && outlineStyle !== "hidden" && Number.parseFloat(outlineWidth) >= 2) return "outline";
  // A layer of a computed box-shadow reads `<colour> <x> <y> <blur> <spread> [inset]`, the colour being `rgb(…)`, `oklab(…)` and so on.
  const layer = /((?:rgba?|oklab|oklch|lab|lch|hsla?|color)\([^)]*\))\s+(-?[\d.]+)px\s+(-?[\d.]+)px\s+(-?[\d.]+)px\s+(-?[\d.]+)px(\s+inset)?/g;
  for (const match of boxShadow.matchAll(layer)) {
    const [, colour, x, y, blur, spread, inset] = match;
    if (inset !== undefined || colour === undefined) continue;
    const alpha = /\/\s*([\d.]+)\)$/.exec(colour)?.[1] ?? /,\s*([\d.]+)\)$/.exec(colour)?.[1] ?? "1";
    const hard = Number.parseFloat(x ?? "1") === 0 && Number.parseFloat(y ?? "1") === 0 && Number.parseFloat(blur ?? "1") === 0;
    if (hard && Number.parseFloat(spread ?? "0") >= 2 && Number.parseFloat(alpha) > 0.3 && lightness(colour) >= 0.5) return "shadow";
  }
  return "none";
}

/** The role and name on the first line of a Playwright aria snapshot (`- button "Fit"`, `- 'tab "A: b"'`, `- region "Photos":`). */
export function roleAndName(snapshot: string): { role: string; name: string } {
  let line = (snapshot.split("\n")[0] ?? "").trim().replace(/^-\s*/, "");
  // YAML quotes a whole line that has a colon in the name: 'tab "A: b"' (an apostrophe inside is doubled).
  if (line.startsWith("'")) {
    const end = line.lastIndexOf("'");
    line = line.slice(1, end > 0 ? end : undefined).replaceAll("''", "'");
  }
  const parsed = /^([\w-]+)(?:\s+"((?:[^"\\]|\\.)*)")?/.exec(line);
  return { role: parsed?.[1] ?? "generic", name: (parsed?.[2] ?? "").replace(/\\(["\\])/g, "$1") };
}

/** What the page says about the focused element: whether it is new in this walk, and the computed style that draws the ring. */
interface Style {
  outlineStyle: string;
  outlineWidth: string;
  boxShadow: string;
}
type Probe = { body: true } | { body: false; seen: boolean; self: Style; parents: Style[] };

/** Runs in the page. Waits for a transition on the element to end, so the style read is the ring's, not a step on the way to it. */
const probe = async (): Promise<Probe> => {
  const el = document.activeElement;
  if (!el || el === document.body || el === document.documentElement) return { body: true };
  const w = window as unknown as { __ymSeen?: WeakSet<Element> };
  const seenSet = (w.__ymSeen ??= new WeakSet<Element>());
  const seen = seenSet.has(el);
  seenSet.add(el);
  await Promise.all(el.getAnimations().filter((a) => a instanceof CSSTransition).map((a) => a.finished.catch(() => undefined)));
  const styleOf = (e: Element): Style => {
    const cs = getComputedStyle(e);
    return { outlineStyle: cs.outlineStyle, outlineWidth: cs.outlineWidth, boxShadow: cs.boxShadow };
  };
  const parents: Style[] = [];
  for (let up = el.parentElement, n = 0; up && n < 2; up = up.parentElement, n++) parents.push(styleOf(up));
  return { body: false, seen, self: styleOf(el), parents };
};

/** Puts the sequential focus starting point at the top of the page, with nothing focused. */
const toStart = (): void => {
  (document.activeElement as HTMLElement | null)?.blur?.();
  const body = document.body;
  body.tabIndex = -1;
  body.focus({ preventScroll: true });
  body.removeAttribute("tabindex");
  window.scrollTo(0, 0);
};

/** Presses Tab from the start of the page and records each stop. Throws when the walk does not end (a keyboard trap). */
export async function walkTabs(page: Page): Promise<Walk> {
  await page.evaluate(toStart);
  const stops: Stop[] = [];
  for (let n = 0; n < MAX_STOPS; n++) {
    await page.keyboard.press("Tab");
    const at = await page.evaluate(probe);
    if (at.body) return { end: "leaves", stops };
    if (at.seen) return { end: "wraps", stops };
    const snapshot = await page.locator(":focus").first().ariaSnapshot({ timeout: 5000 }).catch(() => "");
    let ring = classifyRing(at.self.outlineStyle, at.self.outlineWidth, at.self.boxShadow);
    if (ring === "none" && at.parents.some((p) => classifyRing(p.outlineStyle, p.outlineWidth, p.boxShadow) !== "none")) ring = "parent";
    stops.push({ ...roleAndName(snapshot), outline: at.self.outlineStyle, ring });
  }
  throw new Error(`the Tab walk did not end in ${MAX_STOPS} stops: a keyboard trap, or a list that never stops. The last stops: ${stops.slice(-4).map((s) => `${s.role} "${s.name}"`).join(", ")}`);
}

/** One line for a stop, for a readable diff: `12. button "Fit" (outline: solid, ring)`. */
export function describeStop(index: number, stop: Stop): string {
  return `${String(index + 1).padStart(2, "0")}. ${stop.role} "${stop.name}" (outline-style ${stop.outline}, ring: ${stop.ring})`;
}
