import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

// The native viewport (docs/01 §Native viewport) draws the photo behind the webview. The page lets it through while the Photos screen
// shows (screens.css), and the window behind the page is paper (src-tauri/src/native.rs), so the two must agree on the colour.

const read = (path: string): string => readFileSync(resolve(import.meta.dirname, path), "utf8");

describe("the page over the native viewport", () => {
  it("is transparent from the root down to the photo cell", () => {
    const css = read("screens.css");
    const rule = /([^{}]*:root\[data-native-viewport\][^{}]*)\{\s*background:\s*transparent( !important)?;\s*\}/.exec(css);
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

  it("sits on a window of the same paper colour", () => {
    const paper = /--paper:\s*#([0-9a-fA-F]{6})/.exec(read("tokens.css"))?.[1];
    expect(paper).toBeDefined();
    const want = [0, 2, 4].map((i) => parseInt((paper as string).slice(i, i + 2), 16) / 255);
    const native = /const PAPER_SRGB: \[f64; 3\] = \[([^\]]+)\];/.exec(read("../../src-tauri/src/native.rs"))?.[1];
    expect(native, "PAPER_SRGB in src-tauri/src/native.rs").toBeDefined();
    const got = (native as string).split(",").map((v) => Number(v.trim()));
    expect(got).toHaveLength(3);
    got.forEach((v, i) => expect(v).toBeCloseTo(want[i] as number, 3));
  });
});
