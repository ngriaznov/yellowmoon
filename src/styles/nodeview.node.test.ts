import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { DIVIDER_H, HEADER_H, MIN_WINDOW_H } from "../screens/nodes/layout";

// The CSS of the node view (`docs/23` §7.1) and the numbers the TypeScript side repeats from it.

const read = (file: string): string => readFileSync(resolve(import.meta.dirname, file), "utf8");
const glass = read("glass-c2.css");
const tokens = read("tokens.css");

/** The body of the first rule whose selector is exactly `selector` (not the last line of a longer list of selectors). */
function rule(css: string, selector: string): string {
  for (let at = css.indexOf(`\n${selector} {`); at >= 0; at = css.indexOf(`\n${selector} {`, at + 1)) {
    if (css[at - 1] === ",") continue;
    return css.slice(css.indexOf("{", at) + 1, css.indexOf("}", at));
  }
  throw new Error(`no rule for ${selector}`);
}

describe("the node view's grid", () => {
  const screen = rule(glass, '.ym-screen[data-screen="photos"][data-nodes="true"]');

  it("is five areas: the header, the photo over the divider over the graph, and the panel full height at the right", () => {
    const areas = /grid-template-areas:\s*((?:"[^"]+"\s*)+);/.exec(screen)?.[1];
    expect(areas, "grid-template-areas").toBeDefined();
    const rows = [...areas!.matchAll(/"([^"]+)"/g)].map((m) => m[1]!.trim().split(/\s+/));
    expect(rows).toEqual([
      ["header", "header"],
      ["photo", "tools"],
      ["divider", "tools"],
      ["graph", "tools"],
    ]);
  });

  it("has the header's row, a photo row that takes what is left, the divider's 10 px and the editor's height, and the 400 px panel", () => {
    expect(screen).toMatch(/grid-template-columns:\s*minmax\(0, 1fr\) var\(--panel-w\)/);
    expect(screen).toMatch(/grid-template-rows:\s*var\(--header-h\) minmax\(0, 1fr\) var\(--divider-h\) var\(--editor-h\)/);
    expect(screen).toMatch(/--divider-h:\s*10px/);
    expect(tokens).toMatch(/--panel-w:\s*400px/);
  });

  it("puts each part in its area, the filmstrip is not one of them, and the photo's wrapper makes no box in the simple view", () => {
    for (const [part, area] of [
      [".ym-header", "header"],
      [".ym-viewer", "photo"],
      [".ym-divider", "divider"],
      [".ym-graph", "graph"],
      [".ym-tools", "tools"],
    ] as const) {
      expect(rule(glass, `.ym-screen[data-nodes="true"] ${part}`), part).toMatch(new RegExp(`grid-area:\\s*${area};`));
    }
    expect(rule(glass, ".ym-viewer")).toMatch(/display:\s*contents/);
    expect(glass).not.toMatch(/\.ym-filmstrip[^{]*\{[^}]*grid-area/);
  });

  it("keeps the cell 3:2 and centred: its width is the row's height times 1.5, or the whole column when that is less, and paper lies beside it", () => {
    expect(rule(glass, ".ym-screen[data-nodes=\"true\"] .ym-viewer > .ym-photo-cell")).toMatch(/flex:\s*0 1 calc\(var\(--viewer-h\) \* 1\.5\)/);
    expect(rule(glass, ".ym-screen[data-nodes=\"true\"] .ym-viewer")).toMatch(/justify-content:\s*center/);
    expect(rule(glass, ".ym-screen[data-nodes=\"true\"] .ym-viewer::before,\n.ym-screen[data-nodes=\"true\"] .ym-viewer::after")).toMatch(/background:\s*var\(--paper\)/);
    // No height of its own: the 3:2 of the cell's own rule gives it.
    expect(rule(glass, ".ym-screen[data-nodes=\"true\"] .ym-viewer > .ym-photo-cell")).not.toMatch(/\bheight:/);
  });

  it("lets the photo's wrapper paint nothing of its own: the paper is beside the cell, never under it (the native viewport shows through)", () => {
    expect(rule(glass, ".ym-viewer")).not.toMatch(/background/);
    expect(rule(glass, '.ym-screen[data-nodes="true"] .ym-viewer')).not.toMatch(/background/);
    expect(rule(glass, '.ym-screen[data-nodes="true"] .ym-viewer > .ym-photo-cell')).not.toMatch(/background/);
  });

  it("is 44 px of toolbar over a canvas, in the strip's ground, with a 10 px divider", () => {
    expect(rule(glass, ".ym-graph__bar")).toMatch(/height:\s*44px/);
    expect(rule(glass, ".ym-graph")).toMatch(/background:\s*var\(--strip-neutral\)/);
    expect(rule(glass, ".ym-divider")).toMatch(/cursor:\s*row-resize/);
  });
});

describe("the numbers layout.ts repeats from the tokens", () => {
  it("are the header's height and the smallest window's", () => {
    expect(tokens).toMatch(new RegExp(`--header-h:\\s*${HEADER_H}px`));
    expect(tokens).toMatch(new RegExp(`--min-window-h:\\s*${MIN_WINDOW_H}px`));
    expect(DIVIDER_H).toBe(10);
  });
});
