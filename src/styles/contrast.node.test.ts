import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

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

/** OKLab distance of two colours: about 0.1 is "plainly another colour" for flat accents (the closest pair of the six is Light and RAW, 0.115). */
function oklabDistance(a: string, b: string): number {
  const lin = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const lab = (hex: string): [number, number, number] => {
    const [r, g, bl] = [1, 3, 5].map((i) => lin(parseInt(hex.slice(i, i + 2), 16) / 255)) as [number, number, number];
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * bl);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * bl);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * bl);
    return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
  };
  const [x, y] = [lab(a), lab(b)];
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
