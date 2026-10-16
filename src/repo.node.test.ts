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

describe("the web engine the interface needs (docs/STATUS.md question 30)", () => {
  /** macOS 13.3 is the first release to ship Safari 16.4 (`@property`, with `color-mix()`, `:has()` and `@layer`: what Tailwind 4 needs). */
  const FIRST_WITH_16_4 = [13, 3];

  const minimum = (): number[] => {
    const conf = JSON.parse(read("src-tauri/tauri.conf.json")) as { bundle: { macOS: { minimumSystemVersion: string } } };
    return conf.bundle.macOS.minimumSystemVersion.split(".").map(Number);
  };
  const older = (a: number[], b: number[]): boolean => (a[0] ?? 0) < (b[0] ?? 0) || ((a[0] ?? 0) === (b[0] ?? 0) && (a[1] ?? 0) < (b[1] ?? 0));

  it("does not call Safari 15 the oldest WebKit the app supports: only its syntax is lowered to that", () => {
    const vite = read("vite.config.ts");
    expect(vite).not.toMatch(/oldest WebKit the app supports/);
    expect(vite).toMatch(/16\.4/);
  });

  it("tells people on a macOS that ships an older engine to update, while the minimum allows one", () => {
    if (!older(minimum(), FIRST_WITH_16_4)) return; // the owner raised the minimum: nothing left to say
    expect(read("README.md")).toMatch(/macOS 12 and 13, install the latest system and Safari updates/);
    // The note goes out with the first release of the glass interface (0.1.6); a later release moves Unreleased on.
    expect(read("CHANGELOG.md")).toMatch(/macOS 12 and 13, install the latest system and Safari updates/);
  });
});

describe("the docs agree on the dark glass decision (owner, 2026-10-16)", () => {
  it("docs/00 settled decisions call the interface dark glass, not the white grid, and point at the C2 references", () => {
    const row = /^\| UI \|.*$/m.exec(read("docs/00-overview.md"))?.[0] ?? "";
    expect(row).toMatch(/Dark glass/);
    expect(row).toMatch(/design\/screenshots-c2\//);
    expect(row).not.toMatch(/not V1/);
  });

  it("AGENTS.md sends a UI task to the C2 screenshots for the look, and keeps the mockups for layout and copy", () => {
    const agents = read("AGENTS.md");
    expect(agents).not.toMatch(/mockups\/` are the visual and copy reference/);
    expect(agents).toMatch(/mockups\/` are the layout, size and copy reference/);
    expect(agents).toMatch(/docs\/11-ui\.md` wins/);
  });

  it("STATUS does not call PAPER_SRGB a clear colour: the native pass clears to black and the colour is the window's", () => {
    expect(read("docs/STATUS.md")).not.toMatch(/native layer's clear colour/);
    expect(read("src-tauri/src/native.rs")).toMatch(/wgpu::Color::BLACK/);
  });
});
