// @vitest-environment node
// The mock's Apple Auto (`src/api.mock.ts`, its constants in `src/api/appleAutoMock.ts`) copies the core's default instructions, their limit and the table of `docs/19` section 5
// (`crates/ym-core/src/apple_auto.rs`): this reads the Rust file and fails when either side changes without the other.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { APPLE_TABLE, MOCK_DEFAULT_INSTRUCTIONS, MOCK_DEFAULT_INSTRUCTIONS_SHA256, MOCK_INSTRUCTIONS_MAX_CHARS } from "./api/appleAutoMock";

const source = readFileSync(new URL("../crates/ym-core/src/apple_auto.rs", import.meta.url), "utf8");

/** The value of `pub const NAME: type = <value>;` in the Rust source, as written: a whole string literal (which may hold `;`), or what comes before the `;`. */
function rustConst(name: string): string {
  const match = new RegExp(`pub const ${name}: [\\w&' ]+ = ("(?:[^"\\\\]|\\\\.)*"|[^;]+);`).exec(source);
  if (!match?.[1]) throw new Error(`${name} is not a constant of apple_auto.rs`);
  return match[1].trim();
}

/** The numbers of one row of `TABLE` (`name: [a, b, …],`). */
function tableRow(name: string): number[] {
  const table = /pub const TABLE: Table = Table \{([\s\S]*?)\n\};/.exec(source)?.[1] ?? "";
  const row = new RegExp(`\\b${name}: \\[([^\\]]+)\\]`).exec(table)?.[1];
  if (!row) throw new Error(`TABLE has no row ${name}`);
  return row.split(",").map((n) => n.trim()).filter(Boolean).map(Number);
}

/** The values of a row of the mock's table, from its lowest step to its highest. */
const byStep = (row: Record<number, number>): number[] =>
  Object.keys(row)
    .map(Number)
    .sort((a, b) => a - b)
    .map((step) => row[step]!);

describe("the mock's Apple Auto matches crates/ym-core/src/apple_auto.rs", () => {
  it("has the default instructions word for word, their limit and their digest", () => {
    expect(JSON.parse(rustConst("DEFAULT_INSTRUCTIONS"))).toBe(MOCK_DEFAULT_INSTRUCTIONS);
    expect(Number(rustConst("INSTRUCTIONS_MAX_CHARS"))).toBe(MOCK_INSTRUCTIONS_MAX_CHARS);
    const sha = createHash("sha256").update(MOCK_DEFAULT_INSTRUCTIONS, "utf8").digest("hex").slice(0, 12);
    expect(MOCK_DEFAULT_INSTRUCTIONS_SHA256).toBe(sha);
  });

  it("has the table of section 5, step for step", () => {
    expect(byStep(APPLE_TABLE.exposure_ev)).toEqual(tableRow("exposure_ev"));
    expect(byStep(APPLE_TABLE.highlights)).toEqual(tableRow("highlights"));
    expect(byStep(APPLE_TABLE.warmer_mired)).toEqual(tableRow("warmer_mired"));
    expect(byStep(APPLE_TABLE.tint)).toEqual(tableRow("tint"));
    for (const row of ["shadows", "black", "contrast", "saturation"]) expect(byStep(APPLE_TABLE.percent)).toEqual(tableRow(row));
    expect(byStep(APPLE_TABLE.strength_radius)).toEqual(tableRow("strength_radius"));
  });

  it("points each colour the way the core does", () => {
    const cos30 = Number(/const COS_30: f64 = ([^;]+);/.exec(source)?.[1]);
    expect(cos30).toBeCloseTo(Math.cos(Math.PI / 6), 3);
    const table = /pub const TABLE: Table = Table \{([\s\S]*?)\n\};/.exec(source)?.[1] ?? "";
    const points = (name: string): Array<[number, number]> => {
      const block = new RegExp(`${name}: \\[([\\s\\S]*?)\\n    \\],`).exec(table)?.[1] ?? "";
      const value = (n: string): number => {
        const sign = n.startsWith("-") ? -1 : 1;
        const bare = n.replace(/^-/, "");
        return sign * (bare === "COS_30" ? cos30 : Number(bare));
      };
      return [...block.matchAll(/\[([^\]]+)\]/g)].map((m) => {
        const [a, b] = m[1]!.split(",").map((n) => n.trim());
        return [value(a!), value(b!)];
      });
    };
    expect(points("shadow_directions")).toHaveLength(6);
    expect(Object.values(APPLE_TABLE.shadow_directions)).toEqual(points("shadow_directions"));
    expect(Object.values(APPLE_TABLE.highlight_directions)).toEqual(points("highlight_directions"));
  });
});
