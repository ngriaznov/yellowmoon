import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

// The native viewport (docs/01 §Native viewport) draws the photo behind the webview. The page lets it through while the Photos screen
// shows (screens.css), and the window behind the page is paper (src-tauri/src/native.rs), so the two must agree on the colour.

const read = (path: string): string => readFileSync(resolve(import.meta.dirname, path), "utf8");

describe("the page over the native viewport", () => {
  it("is transparent from the root down to the photo cell", () => {
    const css = read("screens.css");
    const rule = /([^{}]*:root\[data-native-viewport\][^{}]*)\{\s*background:\s*transparent( !important)?;[^{}]*\}/.exec(css);
    expect(rule, "a transparent rule for :root[data-viewport=native]").not.toBeNull();
    const selectors = (rule?.[1] ?? "")
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .split(",")
      .map((s) => s.trim());
    expect(selectors).toEqual([
      ':root[data-native-viewport]',
      ':root[data-native-viewport] body',
      ':root[data-native-viewport] #root',
      ':root[data-native-viewport] .ym-screen[data-screen="photos"]',
    ]);
  });

  it("beats the inline background of the root in index.html", () => {
    // The first app on a Mac (0.1.0) showed no photo: index.html paints <html> paper inline, which a plain rule cannot override.
    const html = readFileSync(resolve(import.meta.dirname, "../../index.html"), "utf8");
    const inline = /<html[^>]*style="[^"]*background/.test(html);
    const important = /:root\[data-native-viewport\][^{}]*\{\s*background:\s*transparent !important;/.test(read("screens.css"));
    expect(!inline || important, "index.html paints the root inline, so the transparent rule needs !important").toBe(true);
  });

  it("never animates its background: a transition is above !important, and WebKit can leave one at the old paper", () => {
    // 0.1.4's viewport job: the reduced-motion rule turned every property change into a 0.001 ms transition, and WebKit on the Mac
    // runner kept the root, the body and the screen paper with the transparent rule and an inline !important both in force.
    expect(/:root\[data-native-viewport\][^{}]*\{[^}]*transition:\s*none !important;/.test(read("screens.css"))).toBe(true);
    const global = read("global.css").replace(/\/\*[\s\S]*?\*\//g, "");
    const reduce = /@media \(prefers-reduced-motion: reduce\)\s*\{([\s\S]*?)\n\}/.exec(global)?.[1] ?? "";
    expect(reduce, "the reduced-motion block").not.toBe("");
    expect(reduce).not.toMatch(/transition-duration/);
    expect(reduce).toMatch(/transition-property:\s*none !important/);
  });

  /** `--paper` of the dark theme in tokens.css: since the dark glass interface (docs/16, 2026-10-16) the page is always on it (index.html sets `data-theme="dark"`). */
  const darkPaper = (): string => {
    const block = /:root\[data-theme="dark"\]\s*\{([^}]*)\}/.exec(read("tokens.css"))?.[1] ?? "";
    const paper = /--paper:\s*#([0-9a-fA-F]{6})/.exec(block)?.[1];
    expect(paper, "--paper in the dark theme block of tokens.css").toBeDefined();
    return (paper as string).toLowerCase();
  };
  const html = (): string => readFileSync(resolve(import.meta.dirname, "../../index.html"), "utf8");

  it("is on the dark theme from the first paint: the root says so, and paints the paper inline", () => {
    const root = /<html[^>]*>/.exec(html())?.[0] ?? "";
    expect(root).toContain('data-theme="dark"');
    expect(root).toContain('class="dark"');
    expect(root).toMatch(new RegExp(`style="[^"]*background:\\s*#${darkPaper()}[^"]*color-scheme:\\s*dark`, "i"));
    expect(html()).toMatch(/<meta name="color-scheme" content="dark"/);
  });

  it("sits on a window of the same paper colour", () => {
    const paper = darkPaper();
    const want = [0, 2, 4].map((i) => parseInt(paper.slice(i, i + 2), 16) / 255);
    const native = /const PAPER_SRGB: \[f64; 3\] = \[([^\]]+)\];/.exec(read("../../src-tauri/src/native.rs"))?.[1];
    expect(native, "PAPER_SRGB in src-tauri/src/native.rs").toBeDefined();
    // rustfmt puts a long array on one line per value, with a comma after the last: read the numbers, not the commas.
    const got = (native as string).split(",").map((v) => v.trim()).filter((v) => v !== "").map(Number);
    expect(got).toHaveLength(3);
    got.forEach((v, i) => expect(v).toBeCloseTo(want[i] as number, 3));
  });

  it("is the colour of the window before the shell sets it, and of the core's letterbox", () => {
    const paper = darkPaper();
    // tauri.conf.json: the window is not clear (and the desktop behind it not visible) between its creation and native.rs setting its colour.
    const conf = JSON.parse(readFileSync(resolve(import.meta.dirname, "../../src-tauri/tauri.conf.json"), "utf8")) as { app: { windows: Array<{ backgroundColor?: string }> } };
    expect(conf.app.windows[0]?.backgroundColor?.toLowerCase()).toBe(`#${paper}`);
    // api.rs: the letterbox of the core's viewport frames, thumbnails and look tiles, as 8-bit codes.
    const rgb8 = /const PAPER_RGB8: \[u8; 3\] = \[([^\]]+)\];/.exec(read("../../src-tauri/src/api.rs"))?.[1];
    expect(rgb8, "PAPER_RGB8 in src-tauri/src/api.rs").toBeDefined();
    const codes = (rgb8 as string).split(",").map((v) => v.trim()).filter((v) => v !== "").map(Number);
    expect(codes.map((c) => c.toString(16).padStart(2, "0")).join("")).toBe(paper);
  });

  it("cuts the aurora out of the photo cell, whose rectangle the Photos screen reports", () => {
    // The aurora is painted by the page, in front of the photo layer; the mask leaves a hole at the cell (--cell-x/y/w/h).
    const css = read("glass-c2.css").replace(/\/\*[\s\S]*?\*\//g, "");
    const rule = /\.ym-aurora\s*\{([^}]*)\}/.exec(css)?.[1] ?? "";
    expect(rule).toMatch(/mask-composite:\s*exclude/);
    for (const v of ["--cell-x", "--cell-y", "--cell-w", "--cell-h"]) expect(rule, `the mask reads ${v}`).toContain(v);
    const photos = readFileSync(resolve(import.meta.dirname, "../screens/Photos.tsx"), "utf8");
    for (const v of ["--cell-x", "--cell-y", "--cell-w", "--cell-h"]) expect(photos, `Photos sets ${v}`).toContain(v);
    // …and the aurora cannot be hit by the pointer or read by a screen reader.
    const aurora = readFileSync(resolve(import.meta.dirname, "../components/glass/aurora.tsx"), "utf8");
    expect(aurora).toContain("pointer-events-none");
    expect(aurora).toContain("aria-hidden");
  });
});
