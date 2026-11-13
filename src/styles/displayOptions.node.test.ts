import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { DISPLAY_ATTRIBUTES } from "../displayAttributes";

// The system's display options (docs/28 A-P12). WebKit may not match every media query, so each rule written for `prefers-reduced-motion`,
// `prefers-reduced-transparency` or `prefers-contrast: more` has a twin under the attribute the shell sets on <html>. A media query cannot be
// OR-ed with a selector, so the twin is a second block; this test reads the CSS sources and fails when one block has no twin, or the twin
// says something else, so the two cannot drift. (Writing a rule once would need a build step that is not worth its risk: lightningcss and
// Tailwind 4 pass an `@media` with a selector inside through, and cannot merge the two forms.)

const DIR = import.meta.dirname;
const SHEETS = readdirSync(DIR).filter((f) => f.endsWith(".css"));
const read = (name: string): string => readFileSync(resolve(DIR, name), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");

interface Rule {
  selector: string;
  body: string;
}
interface Block {
  /** The prelude: a selector list for a rule, `@media (…)` for an at-rule. */
  prelude: string;
  body: string;
}

/** The blocks of a stylesheet's level: `prelude { body }`, with the braces balanced. Declarations without a block (`@import …;`) are skipped. */
function blocks(css: string): Block[] {
  const out: Block[] = [];
  let depth = 0;
  let start = 0;
  let open = -1;
  for (let i = 0; i < css.length; i += 1) {
    const ch = css[i];
    if (ch === "{") {
      if (depth === 0) open = i;
      depth += 1;
    } else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        const prelude = css.slice(start, open).split(";").pop() ?? "";
        out.push({ prelude: prelude.trim(), body: css.slice(open + 1, i) });
        start = i + 1;
      }
    }
  }
  return out;
}

const rulesOf = (bs: readonly Block[]): Rule[] => bs.map((b) => ({ selector: b.prelude, body: b.body }));

/** A selector or a body in one line with one space, so that a different indentation is not a difference. */
const squash = (text: string): string => text.replace(/\s+/g, " ").replace(/\s*([,;:{}()!])\s*/g, "$1").trim();

/** The selector of a twin as the media block writes it: `:root[data-x] .a` is `.a`, `:root[data-x]` is `:root`. */
const untwin = (selector: string, attribute: string): string => {
  const escaped = attribute.replace(/[-]/g, "\\-");
  return squash(selector.replace(new RegExp(`:root\\[${escaped}\\]\\s+`, "g"), "").replace(new RegExp(`:root\\[${escaped}\\]`, "g"), ":root"));
};

const QUERIES = [
  { media: "(prefers-reduced-motion: reduce)", attribute: DISPLAY_ATTRIBUTES.reduce_motion },
  { media: "(prefers-reduced-transparency: reduce)", attribute: DISPLAY_ATTRIBUTES.reduce_transparency },
  { media: "(prefers-contrast: more)", attribute: DISPLAY_ATTRIBUTES.increase_contrast },
] as const;

describe("the display options of the system", () => {
  for (const query of QUERIES) {
    describe(`${query.media} and :root[${query.attribute}]`, () => {
      for (const sheet of SHEETS) {
        const top = blocks(read(sheet));
        const inMedia = top.filter((b) => b.prelude === `@media ${query.media}`).flatMap((b) => rulesOf(blocks(b.body)));
        const twins = rulesOf(top.filter((b) => !b.prelude.startsWith("@") && b.prelude.includes(`[${query.attribute}]`)));
        if (inMedia.length === 0 && twins.length === 0) continue;
        it(`${sheet}: every rule of the query has its twin under the attribute, in the same order and with the same declarations`, () => {
          expect(inMedia.length, "the query has rules (a twin with no query is a rule written for nothing)").toBeGreaterThan(0);
          expect(twins.map((r) => untwin(r.selector, query.attribute))).toEqual(inMedia.map((r) => squash(r.selector)));
          expect(twins.map((r) => squash(r.body))).toEqual(inMedia.map((r) => squash(r.body)));
        });
      }
    });
  }

  it("has the rules where the glass lives: Reduce Transparency and Increase Contrast in glass.css, Reduce Motion in glass.css and global.css", () => {
    const has = (sheet: string, media: string): boolean => blocks(read(sheet)).some((b) => b.prelude === `@media ${media}`);
    expect(has("glass.css", "(prefers-reduced-transparency: reduce)")).toBe(true);
    expect(has("glass.css", "(prefers-contrast: more)")).toBe(true);
    expect(has("glass.css", "(prefers-reduced-motion: reduce)")).toBe(true);
    expect(has("global.css", "(prefers-reduced-motion: reduce)")).toBe(true);
  });

  it("uses no attribute that the page does not set, and sets no attribute that the CSS does not answer", () => {
    const used = new Set<string>();
    for (const sheet of SHEETS) for (const m of read(sheet).matchAll(/:root\[(data-(?:reduce-[a-z]+|increase-[a-z]+))\]/g)) used.add(m[1]!);
    expect([...used].sort()).toEqual(Object.values(DISPLAY_ATTRIBUTES).sort());
  });

  it("makes every pane opaque and takes away every backdrop filter under Reduce Transparency, whichever form answers", () => {
    const glass = blocks(read("glass.css"));
    const media = rulesOf(blocks(glass.find((b) => b.prelude === "@media (prefers-reduced-transparency: reduce)")?.body ?? ""));
    const text = media.map((r) => `${r.selector}{${squash(r.body)}}`).join("");
    for (const token of ["--glass-opacity:100%!important", "--glass-opacity-strong:100%!important", "--glass-opacity-subtle:100%!important", "--glass-liquid-opacity:100%!important", "--glass-liquid-clear-opacity:100%!important", "backdrop-filter:none!important"]) {
      expect(text, token).toContain(token);
    }
  });

  it("puts the tertiary label on the secondary's value and the focus ring at 4 px under Increase Contrast", () => {
    const glass = blocks(read("glass.css"));
    const media = rulesOf(blocks(glass.find((b) => b.prelude === "@media (prefers-contrast: more)")?.body ?? ""));
    const text = media.map((r) => squash(r.body)).join("");
    expect(text).toContain("--glass-label-tertiary:var(--glass-label-secondary)!important");
    expect(text).toContain("--glass-ring-width:4px!important");
    expect(text).toContain("--ym-focus-width:4px!important");
  });

  it("draws the focus rings the app writes by hand with the width the options change", () => {
    for (const sheet of SHEETS.filter((s) => s !== "glass.css")) {
      for (const m of read(sheet).matchAll(/:focus-visible[^{}]*\{[^{}]*outline:\s*(\d+px) solid/g)) {
        expect(m[0], `${sheet}: a focus ring with a fixed ${m[1]}, not var(--ym-focus-width, …)`).toBe("never");
      }
    }
  });
});
