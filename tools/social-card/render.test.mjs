// @vitest-environment node
// Tests of the social card's parts that need no browser: the command line and the page with the font put in. (The picture itself is
// checked by render.mjs when it runs: the font, the margins and the screenshot frame; see tools/social-card/README.md.)
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { parseArgs, writeCard } from "./render.mjs";

let work;
beforeAll(() => {
  work = mkdtempSync(join(tmpdir(), "ym-social-card-test-"));
});
afterAll(() => {
  rmSync(work, { recursive: true, force: true });
});

describe("parseArgs", () => {
  it("reads the variant, the screenshot and the output", () => {
    const o = parseArgs(["--variant", "b", "--shot", "look.png", "--out", "card.png"]);
    expect(o).toMatchObject({ variant: "B", shot: "look.png", out: "card.png", colors: 256, quant: true });
  });

  it("takes --jpeg with or without a quality", () => {
    expect(parseArgs(["--jpeg"]).jpeg).toBe(92);
    expect(parseArgs(["--jpeg", "88"]).jpeg).toBe(88);
  });

  it("refuses an option it does not know", () => {
    expect(() => parseArgs(["--size", "2"])).toThrow(/unknown option --size/);
  });
});

describe("writeCard", () => {
  it("puts Figtree in as a data URI and nothing else changes", () => {
    const page = readFileSync(writeCard(work), "utf8");
    const template = readFileSync(new URL("./card.html", import.meta.url), "utf8");
    const font = readFileSync(new URL("../../design/fonts/figtree-latin-wght-normal.woff2", import.meta.url)).toString("base64");
    expect(page).not.toContain("__FIGTREE_WOFF2__");
    expect(page).toContain(`url(data:font/woff2;base64,${font})`);
    expect(page.replace(`data:font/woff2;base64,${font}`, "__FIGTREE_WOFF2__")).toBe(template);
  });

  it("names the product only as Yellowmoon", () => {
    const template = readFileSync(new URL("./card.html", import.meta.url), "utf8");
    expect(template).not.toMatch(/\bARRI\b|\bALEXA\b/i);
    expect(template).toContain("Cinema colour for Sony raw files.");
  });
});
