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

describe("copy (docs/11 Range card, owner's decision of 2026-10-17)", () => {
  const NOTE =
    "Highlights moves the brightest tones the camera recorded: 0 keeps them where they were, 1 takes them to white. Low light softens the noise near black.";

  it("the note under the RAW tab's Range sliders is word for word the same in the component and docs/11", () => {
    expect(read("src/screens/tools/RawTab.tsx")).toContain(`"${NOTE}"`);
    expect(read("docs/11-ui.md")).toContain(`"${NOTE}"`);
    expect(read("src/screens/tools/RawTab.tsx")).toContain('<Card title="Range"');
  });
});

describe("the oldest macOS the app runs on (owner's decision of 2026-10-17, docs/16; STATUS question 30)", () => {
  /** macOS 14 Sonoma ships Safari 17: Tailwind 4's CSS (`@property`, `color-mix()`, `:has()`, `@layer`) needs 16.4. */
  const MINIMUM = "14.0";

  const minimumOf = (path: string): string =>
    (JSON.parse(read(path)) as { bundle: { macOS: { minimumSystemVersion: string } } }).bundle.macOS.minimumSystemVersion;

  it("is the same in the bundle, its template, the Homebrew cask and the build target", () => {
    expect(minimumOf("src-tauri/tauri.conf.json")).toBe(MINIMUM);
    expect(minimumOf("templates/tauri.conf.json")).toBe(MINIMUM);
    for (const cask of ["Casks/yellowmoon.rb", "templates/Casks/yellowmoon.rb"]) {
      expect(read(cask)).toMatch(/depends_on macos: ">= :sonoma"/);
    }
    // The build lowers its syntax to the Safari that the oldest supported macOS ships.
    expect(read("vite.config.ts")).toMatch(/target: "safari17"/);
  });

  it("is what the README and the overview say", () => {
    expect(read("README.md")).toMatch(/Yellowmoon needs macOS 14 Sonoma or later/);
    expect(read("README.md")).not.toMatch(/macOS 12|macOS 13/);
    expect(read("docs/00-overview.md")).toMatch(/Platform: macOS 14 Sonoma or later/);
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
    // The native viewport's pass is recorded in present.rs (shared with its test since 2026-10-19).
    expect(read("src-tauri/src/present.rs")).toMatch(/wgpu::Color::BLACK/);
  });
});
