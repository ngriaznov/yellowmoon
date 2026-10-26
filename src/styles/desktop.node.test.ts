import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

// Desktop feel (docs/11 §Desktop feel, M3-16). The owner, 2026-10-16: "UI in Yellowmoon is sluggish. I can select UI text, when I pan
// photo with two fingers on trackpad - all app moves." These rules keep the web view from behaving like a web page: nothing scrolls or
// bounces as a whole, text is not selectable except where people copy it, and a button is a button (arrow), not a link (hand).

const STYLES = import.meta.dirname;
const SRC = resolve(STYLES, "..");

interface Rule {
  file: string;
  selectors: string[];
  decls: Map<string, string>;
}

const strip = (css: string): string => css.replace(/\/\*[\s\S]*?\*\//g, "");

/** Every rule of src/styles/*.css, flat (the innermost blocks, so a rule inside @media or @supports counts). */
function rules(): Rule[] {
  const out: Rule[] = [];
  for (const file of readdirSync(STYLES).filter((f) => f.endsWith(".css"))) {
    const css = strip(readFileSync(join(STYLES, file), "utf8"));
    for (const match of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      const prelude = (match[1] as string).trim();
      if (prelude.startsWith("@")) continue;
      const decls = new Map<string, string>();
      for (const part of (match[2] as string).split(";")) {
        const colon = part.indexOf(":");
        if (colon > 0) decls.set(part.slice(0, colon).trim().toLowerCase(), part.slice(colon + 1).trim().replace(/\s*!important$/, ""));
      }
      out.push({ file, selectors: prelude.split(",").map((s) => s.trim().replace(/\s+/g, " ")), decls });
    }
  }
  return out;
}

const ALL = rules();
const withSelector = (selector: string): Rule[] => ALL.filter((r) => r.selectors.includes(selector));
const declared = (selector: string, property: string): string[] =>
  withSelector(selector)
    .map((r) => r.decls.get(property))
    .filter((v): v is string => v !== undefined);

/** The text of every non-test source file under src/, for what the markup and the listeners are. */
function sources(): Array<{ file: string; text: string }> {
  const found: Array<{ file: string; text: string }> = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (/\.(ts|tsx)$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name)) found.push({ file: path, text: readFileSync(path, "utf8") });
    }
  };
  walk(SRC);
  return found;
}

/** Boxes that scroll on their own: each must keep its scrolling to itself. */
const SCROLLERS = [".ym-tools__scroll", ".ym-dialog__body", ".ym-menu__list", ".ym-wizard__main", ".ym-develop-list", ".ym-step-rail", ".ym-filmstrip__scroll"];

describe("the page does not scroll or bounce as a whole", () => {
  it("locks the root, the body and #root and turns overscroll off", () => {
    const rule = ALL.find((r) => r.selectors.includes("html") && r.selectors.includes("body") && r.selectors.includes("#root") && r.file === "global.css");
    expect(rule, "a rule for html, body, #root in global.css").toBeDefined();
    expect(rule?.decls.get("overflow")).toBe("hidden");
    expect(rule?.decls.get("overscroll-behavior")).toBe("none");
  });

  it.each(SCROLLERS)("%s scrolls on its own and does not hand a gesture on to the page", (selector) => {
    expect(declared(selector, "overscroll-behavior")).toContain("contain");
  });

  it("covers every box that scrolls, now and later: a rule that sets overflow to auto or scroll names only boxes that contain it", () => {
    const contained = new Set(ALL.filter((r) => r.decls.get("overscroll-behavior") === "contain").flatMap((r) => r.selectors));
    const missing: string[] = [];
    for (const rule of ALL) {
      for (const property of ["overflow", "overflow-x", "overflow-y"]) {
        const value = rule.decls.get(property);
        if (value !== "auto" && value !== "scroll") continue;
        for (const selector of rule.selectors) if (!contained.has(selector)) missing.push(`${rule.file}: ${selector} { ${property}: ${value} }`);
      }
    }
    expect(missing, `scrollers without overscroll-behavior: contain in global.css\n${missing.join("\n")}`).toEqual([]);
  });

  it("keeps the filmstrip a row: it scrolls along and never up and down", () => {
    expect(declared(".ym-filmstrip__scroll", "overflow-y")).toContain("hidden");
    expect(declared(".ym-filmstrip__scroll", "overflow-x")).toContain("auto");
  });

  it("gives the photo cell's wheel and pinch to the cell: nothing under it chains", () => {
    expect(declared(".ym-photo-cell", "overscroll-behavior")).toContain("none");
  });

  it("puts no wheel listener on the document or the window: only the photo cell's own, which is not passive", () => {
    const listeners: string[] = [];
    for (const { file, text } of sources()) {
      for (const match of text.matchAll(/(\b[\w.]+)\.addEventListener\(\s*["'`]wheel["'`]([^)]*)/g)) listeners.push(`${file.slice(SRC.length + 1)}: ${match[1]}${match[2]}`);
    }
    expect(listeners).toHaveLength(1);
    expect(listeners[0]).toMatch(/^components\/usePhotoGestures\.ts: el, onWheel, \{ passive: false \}/);
    // …and the page has no inline wheel handler on a React element either (a React onWheel is passive and could not cancel anything).
    const jsx = sources().filter(({ text }) => /\sonWheel=\{/.test(text)).map(({ file }) => file.slice(SRC.length + 1));
    expect(jsx).toEqual([]);
  });
});

describe("the text of the app is not selectable, except where people copy it", () => {
  it("turns selection, the touch callout and the I-beam off for the whole page", () => {
    expect(declared("html", "user-select")).toContain("none");
    expect(declared("html", "-webkit-user-select")).toContain("none");
    expect(declared("html", "-webkit-touch-callout")).toContain("none");
    expect(declared("html", "cursor")).toContain("default");
  });

  /** What people copy: fields, the command line and its example, paths, errors, file names, version and legal text. */
  const SELECTABLE = [
    "input",
    "textarea",
    ".ym-export__code",
    ".ym-export__example",
    ".ym-export__folder-text",
    ".ym-pref__hint",
    ".ym-pref__note",
    ".ym-about-dialog__version",
    ".ym-about-dialog__section",
    ".ym-about-dialog__adobe",
    ".ym-about-dialog__legal",
    '.ym-error-bar [data-slot="item-content"]',
    ".ym-wizard__problem",
    '.ym-wizard__callout--error [data-slot="item-content"]',
    ".ym-drop-zone__notice",
    ".ym-develop__notice",
    ".ym-wizard__received-name",
    ".ym-queue__file",
    ".ym-queue__note",
    ".ym-info-row__value",
  ];

  it.each(SELECTABLE)("%s is selectable again, with the prefixed property WebKit needs", (selector) => {
    expect(declared(selector, "user-select")).toContain("text");
    expect(declared(selector, "-webkit-user-select")).toContain("text");
  });

  it("names only classes the markup uses (a renamed class must not quietly stop being selectable)", () => {
    const text = sources()
      .map((s) => s.text)
      .join("\n");
    const classes = [...new Set(ALL.filter((r) => r.decls.get("user-select") === "text").flatMap((r) => r.selectors))].flatMap(
      (s) => s.match(/\.ym-[\w-]+/g) ?? [],
    );
    expect(classes.length).toBeGreaterThan(10);
    const unused = classes.filter((c) => !text.includes(c.slice(1)));
    expect(unused).toEqual([]);
  });

  it("keeps a field a field: the I-beam over text fields and over the read-only text people copy, not over sliders", () => {
    const beam = ALL.filter((r) => r.decls.get("cursor") === "text").flatMap((r) => r.selectors);
    expect(beam).toEqual(
      expect.arrayContaining(["textarea", ".ym-export__code", ".ym-pref__hint", ".ym-pref__note", '.ym-error-bar [data-slot="item-content"]', ".ym-develop__notice"]),
    );
    expect(beam).not.toContain("input");
  });

  it("does not let a picture or a link be dragged out of the window", () => {
    expect(declared("img", "-webkit-user-drag")).toContain("none");
    expect(declared("a", "-webkit-user-drag")).toContain("none");
  });
});

describe("a button is an arrow, a link is a hand", () => {
  it("makes the arrow the cursor of every button", () => {
    expect(declared("button", "cursor")).toContain("default");
  });

  it("leaves the pointing hand to link-styled buttons (LinkButton, ResetLink) and nothing else", () => {
    const hands = ALL.filter((r) => r.decls.get("cursor") === "pointer").flatMap((r) => r.selectors);
    // glasscn's `link` variant carries data-variant="link"; a link that cannot be used (aria-disabled) is an arrow again.
    expect(hands.sort()).toEqual(['button[data-slot="button"][data-variant="link"]:not([aria-disabled="true"])']);
  });

  it("draws LinkButton and ResetLink as glasscn's link variant, the one that gets the hand", () => {
    const text = (file: string): string => sources().find((s) => s.file.endsWith(file))?.text ?? "";
    expect(text("components/Button.tsx")).toMatch(/<GlassButton[^>]*variant="link"/);
    expect(text("components/ResetLink.tsx")).toMatch(/<GlassButton[^>]*variant="link"/);
  });
});
