// @vitest-environment node
// Checks that read files of the repository. They run in Node, so they sit outside the browser
// tsconfig (see tsconfig.node.json).
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { GRAPH_TEXT } from "./fixtures/graph";

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
    // 3 px, or 4 px under Increase Contrast (docs/28 A-P12): the width is --ym-focus-width, 3 px by default.
    expect(css).toMatch(/outline:\s*var\(--ym-focus-width, 3px\) solid var\(--focus\)/);
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

describe("the Lens card (docs/21 section 1)", () => {
  const ids = ["default", "classic", "vintage", "petzval", "triplet", "sonnar", "detuned"];
  const lens = (id: string): { name: string; line: string; start: { flares: number; edges: number } } =>
    JSON.parse(read(`data/lenses/${id}/lens.json`)) as { name: string; line: string; start: { flares: number; edges: number } };

  it("lists the lenses of data/lenses/ in the browser mock with the names, lines and starting amounts of the spec and the data", () => {
    const spec = read("docs/21-lens-character.md");
    // The mock's table is wrapped by the formatter where a row is long (Detuned's).
    const mock = read("src/api.mock.ts").replace(/\s+/g, " ");
    for (const id of ids) {
      const { name, line, start } = lens(id);
      expect(spec, id).toContain(`| ${name} | "${line}" |`);
      expect(mock, id).toContain(`"${name}", "${line}", ${start.flares}, ${start.edges},`);
    }
  });

  it("says in the same words in the component and docs/11 that the lens data are missing", () => {
    const NOTE = "Lens data are missing, so lenses can't be used.";
    expect(read("src/screens/tools/LensCard.tsx")).toContain(`"${NOTE}"`);
    expect(read("docs/11-ui.md")).toContain(`"${NOTE}"`);
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

describe("the mock's rules for the graph (docs/23 section 9)", () => {
  const rust = read("crates/ym-core/src/settings/graph.rs");
  /** The sentence of a `#[error("...")]` of the core, as the source writes it. */
  const sentence = (text: string): string => `#[error("${text}")]`;

  it("say what the core says, word for word", () => {
    for (const text of [
      GRAPH_TEXT.missingFixed,
      GRAPH_TEXT.linkDirection,
      GRAPH_TEXT.linkPortTypes,
      GRAPH_TEXT.outputLink,
      GRAPH_TEXT.mainInput,
      GRAPH_TEXT.cycle,
      GRAPH_TEXT.tooManyNodes,
      GRAPH_TEXT.tooManyMasks,
      GRAPH_TEXT.tooManyBrushPoints,
      GRAPH_TEXT.tooManyObjectPoints,
      GRAPH_TEXT.cannotRemove,
      GRAPH_TEXT.tooManyMixerInputs,
    ]) {
      expect(rust, text).toContain(sentence(text));
    }
    // The sentences with a hole in them: the mock fills the hole as the core does.
    expect(rust).toContain(sentence(GRAPH_TEXT.unknownNode("{0}" as never)));
    expect(rust).toContain(sentence(GRAPH_TEXT.duplicateId("{0}" as never)));
    expect(rust).toContain(sentence(GRAPH_TEXT.patchUnknownNode("{0}" as never)));
    expect(rust).toContain(sentence("{node} has no port called “{port}”."));
    expect(GRAPH_TEXT.unknownPort(7, "mask")).toBe("Node 7 has no port called “mask”.");
    expect(rust).toContain(sentence("An input can have one link. {node} has two on its {port} input."));
    expect(GRAPH_TEXT.twoLinks(3, "picture")).toBe("An input can have one link. Node 3 has two on its picture input.");
    expect(rust).toContain(sentence("{node} is {actual}, not {given}."));
    expect(GRAPH_TEXT.patchWrongKind(7, "corrector", "input")).toBe("Node 7 is a corrector, not an input.");
    expect(rust).toContain(sentence("{kind} has no field called “{field}”."));
    expect(GRAPH_TEXT.unknownField("corrector", "exposure")).toBe("A corrector has no field called “exposure”.");
    // The mixers' rules (`docs/25` section 2.3): the gap sentence is written over three lines in the source, so its text is looked up whole.
    expect(rust).toContain('"{node} has a link on input {linked} and none on input {missing}. A mixer\'s inputs are numbered from 1 with no gap."');
    expect(GRAPH_TEXT.mixerGap(7, 3, 2)).toBe("Node 7 has a link on input 3 and none on input 2. A mixer's inputs are numbered from 1 with no gap.");
    // The kinds in sentences (`NodeKind::with_article`).
    expect(GRAPH_TEXT.unknownField("parallel", "masks")).toBe("A parallel mixer has no field called “masks”.");
    expect(GRAPH_TEXT.unknownField("splitter", "bypass")).toBe("A splitter has no field called “bypass”.");
    expect(GRAPH_TEXT.patchWrongKind(7, "layer", "corrector")).toBe("Node 7 is a layer mixer, not a corrector.");
    expect(GRAPH_TEXT.patchWrongKind(7, "combiner", "film")).toBe("Node 7 is a combiner, not a film node.");
    for (const text of ['"a parallel mixer"', '"a layer mixer"', '"a splitter"', '"a combiner"', '"A parallel mixer"', '"A layer mixer"', '"A splitter"', '"A combiner"']) {
      expect(rust, text).toContain(`=> ${text}`);
    }
  });

  it("are the sentences of docs/23 section 3.5", () => {
    const doc = read("docs/23-nodes.md");
    for (const text of [GRAPH_TEXT.missingFixed, GRAPH_TEXT.linkDirection, GRAPH_TEXT.linkPortTypes, GRAPH_TEXT.outputLink, GRAPH_TEXT.mainInput, GRAPH_TEXT.cycle, GRAPH_TEXT.tooManyNodes]) {
      expect(doc, text).toContain(`"${text}"`);
    }
  });

  it("are the sentences of docs/25 section 2.3", () => {
    const doc = read("docs/25-mixers.md");
    for (const text of [GRAPH_TEXT.tooManyMixerInputs, GRAPH_TEXT.mixerGap(7, 3, 2), GRAPH_TEXT.unknownPort(7, "picture"), GRAPH_TEXT.tooManyNodes]) {
      expect(doc, text).toContain(`"${text}"`);
    }
  });
});
