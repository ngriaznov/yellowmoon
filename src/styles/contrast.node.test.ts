import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterAll, describe, expect, it } from "vitest";

// Contrast of the colours of design/tokens.css (a preview of M4-04: "contrast check of tab colours"), WCAG 2.x. Text needs 4.5:1,
// the focus ring and other parts of a control need 3:1. The numbers are in the failure messages and in docs/11.

const css = readFileSync(resolve(import.meta.dirname, "tokens.css"), "utf8");
const token = (name: string): string => {
  const match = new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`).exec(css);
  if (!match) throw new Error(`token --${name} not found`);
  return (match[1] as string).toLowerCase();
};

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255) as [number, number, number];
  const lin = (c: number): number => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (hi + 0.05) / (lo + 0.05);
}

const TABS = ["look", "light", "colour", "detail", "curve", "raw", "lens"] as const;

describe("contrast of the design tokens", () => {
  it("is computed right: black on white is 21:1", () => {
    expect(contrast("#000000", "#ffffff")).toBeCloseTo(21, 5);
    expect(contrast("#777777", "#ffffff")).toBeCloseTo(4.48, 2);
  });

  it.each(TABS)("the %s tab's text is readable on its colour (4.5:1)", (tab) => {
    const ratio = contrast(token(`tab-${tab}`), token(`tab-${tab}-ink`));
    expect(ratio, `${tab}: ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(4.5);
  });

  it("ink is readable on the brand yellow and on the status colours", () => {
    for (const name of ["brand", "status-ok", "status-error", "tab-curve"]) {
      const ratio = contrast(token(name), token("ink"));
      expect(ratio, `ink on ${name}: ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("secondary text (muted) is readable on paper", () => {
    expect(contrast(token("muted"), token("paper"))).toBeGreaterThanOrEqual(4.5);
  });

  it("the focus ring shows against paper and ink (3:1)", () => {
    for (const name of ["paper", "ink"]) {
      const ratio = contrast(token("focus"), token(name));
      expect(ratio, `focus on ${name}: ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(3);
    }
  });

  it("the focus ring shows against the brand yellow of the Start screen (3:1)", () => {
    expect(contrast(token("focus"), token("brand"))).toBeGreaterThanOrEqual(3);
  });

  it("the three curve colours show on paper (3:1 for graphics)", () => {
    for (const name of ["curve-r", "curve-g", "curve-b"]) {
      const ratio = contrast(token(name), token("paper"));
      expect(ratio, `${name}: ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(3);
    }
  });
});

// The seven tab colours of the dark glass interface (`src/styles/glass-c2.css`, the ones the app shows; `tokens.css` keeps the pre-glass
// values for the first six). The ground is `--paper` (#0f0f10); a glass pane over the aurora is a little lighter, about #2a2a2a to #303030
// (glasscn's tint oklch(0.24) over blobs of 0.31 to 0.34), so the pane is checked at #303030, the lighter and so the worse case.
const glass = readFileSync(resolve(import.meta.dirname, "glass-c2.css"), "utf8");
const glassToken = (name: string): string => {
  const match = new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`).exec(glass);
  if (!match) throw new Error(`token --${name} not found in glass-c2.css`);
  return (match[1] as string).toLowerCase();
};
const PANE = "#303030";
const GROUND = "#0f0f10";

/** The sRGB hex colour in OKLab (Björn Ottosson, "A perceptual color space for image processing", 2020; the forward matrices of the linear sRGB to LMS and of LMS' to Lab). */
function oklabOf(hex: string): [number, number, number] {
  const lin = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const [r, g, bl] = [1, 3, 5].map((i) => lin(parseInt(hex.slice(i, i + 2), 16) / 255)) as [number, number, number];
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * bl);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * bl);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * bl);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}

/** OKLab distance of two colours: about 0.1 is "plainly another colour" for flat accents (the closest pair of the six is Light and RAW, 0.115). */
function oklabDistance(a: string, b: string): number {
  const [x, y] = [oklabOf(a), oklabOf(b)];
  return Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]);
}

describe("contrast of the tab colours of the dark glass interface", () => {
  it.each(TABS)("the %s tab's black text is readable on its colour (4.5:1)", (tab) => {
    const ratio = contrast(glassToken(`tab-${tab}`), glassToken(`tab-${tab}-ink`));
    expect(ratio, `${tab}: ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(4.5);
  });

  it.each(TABS)("the %s tab's colour shows on the glass pane and on the ground (3:1 for graphics)", (tab) => {
    for (const [name, ground] of [["pane", PANE], ["ground", GROUND]] as const) {
      const ratio = contrast(glassToken(`tab-${tab}`), ground);
      expect(ratio, `${tab} on the ${name}: ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(3);
    }
  });

  it("the Lens violet takes black text at 7:1 (AAA) and shows on the ground at 7:1", () => {
    // #bd8cff: 8.36:1 black on it, 7.63:1 on #0f0f10, 5.25:1 on the lighter pane estimate of #303030.
    expect(glassToken("tab-lens")).toBe("#bd8cff");
    expect(contrast(glassToken("tab-lens"), glassToken("tab-lens-ink"))).toBeGreaterThanOrEqual(7);
    expect(contrast(glassToken("tab-lens"), GROUND)).toBeGreaterThanOrEqual(7);
  });

  it("the Lens violet is no closer to another tab colour than the closest two of the others are to each other", () => {
    const others = TABS.filter((t) => t !== "lens");
    let closestOthers = Infinity;
    for (const [i, a] of others.entries()) for (const b of others.slice(i + 1)) closestOthers = Math.min(closestOthers, oklabDistance(glassToken(`tab-${a}`), glassToken(`tab-${b}`)));
    for (const other of others) {
      const distance = oklabDistance(glassToken("tab-lens"), glassToken(`tab-${other}`));
      expect(distance, `lens to ${other}: ${distance.toFixed(3)}, the closest pair of the six is ${closestOthers.toFixed(3)}`).toBeGreaterThanOrEqual(closestOthers);
    }
  });

  it("tokens.css and tokens.json carry the same Lens colour as the interface", () => {
    expect(token("tab-lens")).toBe(glassToken("tab-lens"));
    expect(token("tab-lens-ink")).toBe(glassToken("tab-lens-ink"));
    const json = JSON.parse(readFileSync(resolve(import.meta.dirname, "../../design/tokens.json"), "utf8")) as { light: Record<string, string> };
    expect(json.light["tab-lens"]).toBe(glassToken("tab-lens"));
    expect(json.light["tab-lens-ink"]).toBe(glassToken("tab-lens-ink"));
  });
});

// The mask accent (`docs/23` section 7): orchid pink, #FF6FD2, black text on it. It is the colour of the Mask tab, the mask tint over the photo, the
// key ports and key links, so it is checked as the tab colours are: text on it, and it against the pane and the ground it is drawn on; and it must
// not be mistaken for a tab (the Detail peach is the closest, and the Lens violet).
describe("contrast of the mask accent", () => {
  it("is orchid pink in the tokens of the design and of the app, and in tokens.json", () => {
    expect(token("mask")).toBe("#ff6fd2");
    expect(token("mask-ink")).toBe("#000000");
    const design = readFileSync(resolve(import.meta.dirname, "../../design/tokens.css"), "utf8");
    expect(/--mask:\s*#ff6fd2/i.test(design), "--mask in design/tokens.css").toBe(true);
    expect(/--mask-ink:\s*#000000/i.test(design), "--mask-ink in design/tokens.css").toBe(true);
    const json = JSON.parse(readFileSync(resolve(import.meta.dirname, "../../design/tokens.json"), "utf8")) as { light: Record<string, string> };
    expect(json.light.mask).toBe("#ff6fd2");
    expect(json.light["mask-ink"]).toBe("#000000");
  });

  it("takes black text at 4.5:1 and shows on the glass pane and on the ground at 3:1 (the numbers of design/nodes/mockup-notes.md: 8.5, 5.3 and 7.7)", () => {
    const ink = contrast(token("mask"), token("mask-ink"));
    const pane = contrast(token("mask"), PANE);
    const ground = contrast(token("mask"), GROUND);
    expect(ink, `black on the mask accent: ${ink.toFixed(2)}:1`).toBeGreaterThanOrEqual(4.5);
    expect(pane, `mask accent on the pane: ${pane.toFixed(2)}:1`).toBeGreaterThanOrEqual(3);
    expect(ground, `mask accent on the ground: ${ground.toFixed(2)}:1`).toBeGreaterThanOrEqual(3);
    expect(ink).toBeCloseTo(8.5, 1);
    expect(ground).toBeCloseTo(7.7, 1);
    expect(pane).toBeCloseTo(5.3, 1);
  });

  it("the Mask tab takes it as its accent, as the tabs take theirs", () => {
    expect(/\[data-tab="mask"\]\s*\{\s*--primary:\s*var\(--mask\);\s*--primary-foreground:\s*var\(--mask-ink\);\s*--ring:\s*var\(--mask\);/.test(glass)).toBe(true);
  });

  it("is no closer to a tab colour than the closest two tab colours are to each other", () => {
    let closestTabs = Infinity;
    for (const [i, a] of TABS.entries()) for (const b of TABS.slice(i + 1)) closestTabs = Math.min(closestTabs, oklabDistance(glassToken(`tab-${a}`), glassToken(`tab-${b}`)));
    for (const tab of TABS) {
      const distance = oklabDistance(token("mask"), glassToken(`tab-${tab}`));
      expect(distance, `mask to ${tab}: ${distance.toFixed(3)}, the closest pair of the tabs is ${closestTabs.toFixed(3)}`).toBeGreaterThanOrEqual(closestTabs);
    }
  });
});

// The text and focus colours of the glass (`docs/28` A-05, M4-04). The label colours are `oklch()` values in the CSS of glasscn (`glass.css`, the
// `.dark` block) and in the app's overrides (`glass-c2.css`, `html.dark`), the focus colours are in `tokens.css` and the tab rules, so each is read
// where it is defined and converted. The numbers are on the pane estimate and on the ground; what is drawn (the real panes, menus, photos) is
// measured by `e2e/contrast.spec.ts`, which prints the table of `docs/28` 2.5. A pane's real colour varies, and the text colours are used on the
// darker places of it, which is why the two tests are both needed.

/** The text of a CSS rule's block: from the line that matches `opener` to the next line that is only `}`. */
function block(text: string, opener: RegExp, what: string): string {
  const start = opener.exec(text);
  if (!start) throw new Error(`${what} not found`);
  const end = text.indexOf("\n}", start.index);
  return text.slice(start.index, end < 0 ? undefined : end);
}
const glasscnCss = readFileSync(resolve(import.meta.dirname, "glass.css"), "utf8");
const darkGlasscn = block(glasscnCss, /^\.dark\s*\{/m, "the .dark block of glass.css");
const darkApp = block(glass, /^html\.dark\s*\{/m, "the html.dark block of glass-c2.css");
const darkTokens = block(css, /^:root\[data-theme="dark"\]\s*\{/m, "the dark block of tokens.css");

/**
 * An `oklch(L C H)` colour as an sRGB hex colour, rounded to 8 bits as a browser draws it (out of gamut channels are clamped). The inverse
 * OKLab matrices and the oklch polar form are Björn Ottosson's (https://bottosson.github.io/posts/oklab/); the transfer function is sRGB's.
 */
function oklchToHex(value: string): string {
  const match = /^oklch\(\s*([\d.]+)(%?)\s+([\d.]+)\s+([\d.]+)(?:deg)?\s*\)$/.exec(value.trim());
  if (!match) throw new Error(`not an oklch() colour: ${value}`);
  const lightness = parseFloat(match[1] as string) / (match[2] ? 100 : 1);
  const chroma = parseFloat(match[3] as string);
  const hue = (parseFloat(match[4] as string) * Math.PI) / 180;
  const a = chroma * Math.cos(hue);
  const b = chroma * Math.sin(hue);
  const l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const sCone = (lightness - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const linear = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * sCone, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * sCone, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * sCone];
  const encode = (c: number): number => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
  return `#${linear.map((c) => Math.round(Math.min(1, Math.max(0, encode(c))) * 255).toString(16).padStart(2, "0")).join("")}`;
}

/** A custom property of the dark glass as a hex colour: the app's override first, then glasscn's own value. */
function darkColour(name: string): string {
  for (const text of [darkApp, darkGlasscn]) {
    const match = new RegExp(`--${name}:\\s*(oklch\\([^)]*\\)|#[0-9a-fA-F]{6})\\s*;`).exec(text);
    if (match) return (match[1] as string).startsWith("#") ? (match[1] as string).toLowerCase() : oklchToHex(match[1] as string);
  }
  throw new Error(`--${name} not found in the dark glass`);
}

describe("oklch to sRGB", () => {
  it.each([
    ["oklch(1 0 0)", "#ffffff"],
    ["oklch(0 0 0)", "#000000"],
    ["oklch(0.5 0 0)", "#636363"],
    ["oklch(0.985 0 0)", "#fafafa"],
    ["oklch(0.88 0 0)", "#d7d7d7"],
    ["oklch(98.5% 0 0)", "#fafafa"],
  ])("%s is %s", (value, hex) => {
    expect(oklchToHex(value)).toBe(hex);
  });

  it("gives sRGB red for the oklch of red (the CSS Color 4 values, to one level)", () => {
    const hex = oklchToHex("oklch(0.62796 0.25768 29.234)");
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) as [number, number, number];
    expect([r, g, b], hex).toEqual([expect.closeTo(255, -0.5), expect.closeTo(0, -0.5), expect.closeTo(0, -0.5)]);
  });
});

/** What is checked, for the table printed once at the end. */
const printed: string[] = [];

describe("contrast of the text and focus colours of the glass (A-05)", () => {
  const LABELS = [
    ["glass-label", "body text", 7],
    ["glass-label-secondary", "secondary text", 4.5],
    ["glass-label-tertiary", "tertiary text", 4.5],
  ] as const;

  it("the pane estimate is within what the glass is made of: the tint, and the lightest blob of the aurora", () => {
    // The pane is the tint (an alpha of it) over the blobs, so it lies between the tint's lightness and the lightest blob's.
    const tint = oklabOf(darkColour("glass-tint"))[0];
    const blobs = ["aurora-1", "aurora-2", "aurora-3"].map((name) => oklabOf(darkColour(name))[0]);
    const pane = oklabOf(PANE)[0];
    expect(pane, `pane L ${pane.toFixed(3)}, tint ${tint.toFixed(3)}, blobs ${blobs.map((l) => l.toFixed(3)).join(" ")}`).toBeGreaterThanOrEqual(tint);
    expect(pane).toBeLessThanOrEqual(Math.max(...blobs));
  });

  it("the ground is the paper of the dark theme", () => {
    const paper = /--paper:\s*(#[0-9a-fA-F]{6})/.exec(darkTokens);
    expect(paper?.[1]?.toLowerCase()).toBe(GROUND);
  });

  it.each(LABELS)("%s (%s) is at its limit on the ground", (name, _what, limit) => {
    const ratio = contrast(darkColour(name), GROUND);
    expect(ratio, `${name} ${darkColour(name)} on the ground: ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(limit);
  });

  it.each(LABELS.filter(([name]) => name !== "glass-label-tertiary"))("%s (%s) is at its limit on the pane estimate", (name, _what, limit) => {
    const ratio = contrast(darkColour(name), PANE);
    expect(ratio, `${name} ${darkColour(name)} on the pane: ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(limit);
  });

  // The tertiary grey (#868686) is 4.5:1 only on a ground darker than about #1f1f1f. The pane estimate (#303030) is the lighter of the two the
  // glass can be, so it is 3.6:1 there; the tertiary text of the app is the hints and the large step numbers, drawn where the pane is darker
  // (the rendered test measures 4.6:1 at worst on the aurora), and nothing but the hint of a disabled button is below. This is the one limit of
  // the task that the estimate does not meet, so it is stated, with the number, and the test fails if it ever does (then the line goes).
  it("the tertiary text is under 4.5:1 on the lightest pane estimate, by design of the estimate: the rendered test is what proves it where it is drawn", () => {
    const ratio = contrast(darkColour("glass-label-tertiary"), PANE);
    expect(ratio, `tertiary on the pane estimate ${PANE}: ${ratio.toFixed(2)}:1 (if this is 4.5 or more, delete this test and assert it on the pane)`).toBeLessThan(4.5);
    expect(ratio).toBeGreaterThanOrEqual(3);
  });

  /** The colours a focus ring is drawn in: `--focus`, and the accent each rule of `glass-c2.css` gives `--ring` (the brand yellow, the tabs', the mask's). */
  const ringColours = (): Array<[string, string]> => {
    const out: Array<[string, string]> = [["--focus", /--focus:\s*(#[0-9a-fA-F]{6})/.exec(darkTokens)?.[1]?.toLowerCase() ?? ""]];
    for (const name of new Set([...glass.matchAll(/--ring:\s*var\(--([a-z-]+)\)/g)].map((m) => m[1] as string))) {
      const hex = new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`).exec(glass + css)?.[1];
      out.push([`--ring: var(--${name})`, (hex ?? "").toLowerCase()]);
    }
    return out;
  };

  it("finds the focus colours: --focus, the brand yellow, the seven tabs and the mask", () => {
    const names = ringColours().map(([name]) => name);
    expect(names).toContain("--focus");
    for (const name of ["brand", "mask", ...TABS.map((t) => `tab-${t}`)]) expect(names, name).toContain(`--ring: var(--${name})`);
    for (const [name, hex] of ringColours()) expect(hex, name).toMatch(/^#[0-9a-f]{6}$/);
  });

  it("every focus ring colour shows on the pane estimate and on the ground (3:1)", () => {
    for (const [name, hex] of ringColours()) {
      for (const [where, ground] of [["pane", PANE], ["ground", GROUND]] as const) {
        const ratio = contrast(hex, ground);
        expect(ratio, `${name} ${hex} on the ${where}: ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(3);
      }
    }
  });

  it("collects the numbers of the table", () => {
    printed.push("Contrast of the glass tokens (WCAG 2.x; the pane estimate is " + PANE + ", the ground " + GROUND + ")");
    printed.push("  token                         colour   pane   ground  limit");
    for (const [name, _what, limit] of LABELS) {
      const hex = darkColour(name);
      printed.push(`  --${name.padEnd(26)} ${hex}  ${contrast(hex, PANE).toFixed(2).padStart(5)}  ${contrast(hex, GROUND).toFixed(2).padStart(6)}  ${limit}${name === "glass-label-tertiary" ? " (pane: see the test)" : ""}`);
    }
    for (const [name, hex] of ringColours()) printed.push(`  ${`${name} (ring)`.padEnd(29)} ${hex}  ${contrast(hex, PANE).toFixed(2).padStart(5)}  ${contrast(hex, GROUND).toFixed(2).padStart(6)}  3`);
    expect(printed.length).toBeGreaterThan(5);
  });

  afterAll(() => {
    console.log(`\n${printed.join("\n")}\n`);
  });
});
