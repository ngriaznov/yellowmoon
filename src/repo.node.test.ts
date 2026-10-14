// @vitest-environment node
// Checks that read files of the repository. They run in Node, so they sit outside the browser
// tsconfig (see tsconfig.node.json).
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const root = new URL("../", import.meta.url);
const read = (path: string): string => readFileSync(new URL(path, root), "utf8");

/** Version under `[workspace.package]` in the root Cargo.toml. */
function workspaceVersion(cargoToml: string): string {
  const lines = cargoToml.split(/\r?\n/);
  const start = lines.findIndex((line) => line.trim() === "[workspace.package]");
  if (start < 0) throw new Error("Cargo.toml has no [workspace.package] section");
  for (const line of lines.slice(start + 1)) {
    if (/^\s*\[/.test(line)) break;
    const match = /^\s*version\s*=\s*"([^"]+)"/.exec(line);
    if (match?.[1]) return match[1];
  }
  throw new Error("[workspace.package] has no version");
}

describe("versions (docs/12 §Versioning)", () => {
  it("package.json mirrors the workspace version in Cargo.toml", () => {
    const pkg = JSON.parse(read("package.json")) as { version: string };
    expect(pkg.version).toBe(workspaceVersion(read("Cargo.toml")));
  });

  it("package-lock.json carries the same version", () => {
    const pkg = JSON.parse(read("package.json")) as { version: string };
    const lock = JSON.parse(read("package-lock.json")) as { version: string; packages: Record<string, { version?: string }> };
    expect(lock.version).toBe(pkg.version);
    expect(lock.packages[""]?.version).toBe(pkg.version);
  });
});

describe("design tokens", () => {
  /** Everything from the first @font-face on; the header comment differs on purpose. */
  const body = (css: string): string => css.slice(css.indexOf("@font-face"));

  it("src/styles/tokens.css matches design/tokens.css", () => {
    expect(body(read("src/styles/tokens.css"))).toBe(body(read("design/tokens.css")));
  });

  it("bundles the Figtree fonts and their licence", () => {
    for (const file of ["figtree-latin-wght-normal.woff2", "figtree-latin-ext-wght-normal.woff2", "OFL.txt"]) {
      expect(read(`src/styles/fonts/${file}`).length).toBeGreaterThan(0);
    }
  });

  it("global.css sets the base font and the focus ring from docs/11", () => {
    const css = read("src/styles/global.css");
    expect(css).toMatch(/font-family:\s*var\(--font\)/);
    expect(css).toMatch(/background:\s*var\(--paper\)/);
    expect(css).toMatch(/color:\s*var\(--ink\)/);
    expect(css).toMatch(/outline:\s*3px solid var\(--focus\)/);
    expect(css).toMatch(/outline-offset:\s*2px/);
    expect(css).not.toMatch(/outline:\s*(none|0)\b/);
  });
});

describe("copy (docs/11 tab blurbs)", () => {
  // Since 0.2.0 Auto exposure is the camera's own, so the Light tab no longer says it is set for each photo.
  const LIGHT = "Exposure starts at your camera's own. Adjust it from there.";
  const OLD = "Exposure is set for each photo automatically.";

  it("the Light blurb is the same in the component, the mockup and docs/11", () => {
    expect(read("src/components/tabs.ts")).toContain(`blurb: "${LIGHT}"`);
    expect(read("design/mockups/photos.html")).toContain(`blurb: '${LIGHT.replace("'", "\\'")}'`);
    expect(read("docs/11-ui.md")).toContain(`Light "${LIGHT}"`);
  });

  it("the old Light blurb is gone from the component, the mockup and docs/11", () => {
    for (const file of ["src/components/tabs.ts", "design/mockups/photos.html", "docs/11-ui.md"]) {
      expect(read(file), file).not.toContain(OLD);
    }
  });
});
