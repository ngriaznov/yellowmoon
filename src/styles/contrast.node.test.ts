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

const TABS = ["look", "light", "colour", "detail", "curve", "raw"] as const;

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
