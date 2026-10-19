// @vitest-environment node
// The colour warp's constants in TypeScript are copies of the core's (`crates/ym-color/src/warp.rs`, `consts.rs`): this reads the Rust
// files and fails when a number on either side changes without the other.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  LOGC4,
  WARP_COORD_LIMIT,
  WARP_EXP_LIMIT,
  WARP_EXPOSURE_LIMIT_EV,
  WARP_MAX_PINS,
  WARP_MAX_STEPS,
  WARP_NEUTRAL_GUARD,
  WARP_PIVOT_LIMIT_EV,
  WARP_PLANE_EXTENT,
  WARP_RANGE_DEFAULT,
  WARP_RANGE_MAX,
  WARP_RANGE_MIN,
  WARP_STEP_SAFETY,
  WARP_TONE_HALF_WIDTH_STOPS,
} from "./warpMath";

const root = new URL("../../", import.meta.url);
const read = (path: string): string => readFileSync(new URL(path, root), "utf8");

/** The value of `pub const NAME: type = <value>;` in a Rust source, as written. */
function rustConst(source: string, name: string): string {
  const match = new RegExp(`pub const ${name}: \\w+ = ([^;]+);`).exec(source);
  if (!match?.[1]) throw new Error(`${name} is not a constant of the Rust file`);
  return match[1].trim();
}

describe("the colour warp's constants match crates/ym-color/src/warp.rs", () => {
  const warp = read("crates/ym-color/src/warp.rs");

  it.each([
    ["WARP_MAX_PINS", WARP_MAX_PINS],
    ["WARP_PLANE_EXTENT", WARP_PLANE_EXTENT],
    ["WARP_COORD_LIMIT", WARP_COORD_LIMIT],
    ["WARP_RANGE_MIN", WARP_RANGE_MIN],
    ["WARP_RANGE_MAX", WARP_RANGE_MAX],
    ["WARP_RANGE_DEFAULT", WARP_RANGE_DEFAULT],
    ["WARP_STEP_SAFETY", WARP_STEP_SAFETY],
    ["WARP_MAX_STEPS", WARP_MAX_STEPS],
    ["WARP_EXP_LIMIT", WARP_EXP_LIMIT],
    ["WARP_NEUTRAL_GUARD", WARP_NEUTRAL_GUARD],
    ["WARP_PIVOT_LIMIT_EV", WARP_PIVOT_LIMIT_EV],
    ["WARP_EXPOSURE_LIMIT_EV", WARP_EXPOSURE_LIMIT_EV],
  ])("%s", (name, value) => {
    expect(Number(rustConst(warp, name))).toBe(value);
  });

  it("WARP_TONE_HALF_WIDTH is two LogC4 stops", () => {
    const text = rustConst(warp, "WARP_TONE_HALF_WIDTH");
    const match = /^([\d.]+) \* LOGC4_STOP$/.exec(text);
    expect(match, text).not.toBeNull();
    expect(Number(match?.[1])).toBe(WARP_TONE_HALF_WIDTH_STOPS);
  });

  it("the LogC4 constants are the core's", () => {
    const consts = read("crates/ym-color/src/consts.rs");
    const block = /pub const LOGC4: LogC4Constants = LogC4Constants \{([^}]*)\}/.exec(consts)?.[1] ?? "";
    for (const key of ["a", "b", "c", "s", "t"] as const) {
      const value = new RegExp(`\\b${key}: (-?[\\d.e-]+),`).exec(block)?.[1];
      expect(Number(value), key).toBe(LOGC4[key]);
    }
    expect(/pub const LOGC4_STOP: f64 = LOGC4\.b \/ 14\.0;/.test(consts)).toBe(true);
  });
});
