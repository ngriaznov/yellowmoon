import { describe, expect, it } from "vitest";
import { share } from "./share";

describe("share", () => {
  it("returns the old object when the new one holds the same values", () => {
    const prev = { a: 1, b: { c: [1, 2, { d: "x" }] }, e: null };
    expect(share(prev, JSON.parse(JSON.stringify(prev)))).toBe(prev);
  });

  it("keeps the parts that did not change and takes the ones that did", () => {
    const prev = { a: { x: 1 }, b: { y: [1, 2] }, c: 3 };
    const next = { a: { x: 1 }, b: { y: [1, 5] }, c: 3 };
    const merged = share(prev, next);
    expect(merged).toEqual(next);
    expect(merged).not.toBe(prev);
    expect(merged.a).toBe(prev.a);
    expect(merged.b).not.toBe(prev.b);
    expect(merged.b.y).not.toBe(prev.b.y);
  });

  it("notices added, removed and retyped fields and a longer list", () => {
    expect(share({ a: 1 } as Record<string, unknown>, { a: 1, b: 2 })).toEqual({ a: 1, b: 2 });
    expect(share({ a: 1, b: 2 } as Record<string, unknown>, { a: 1 })).toEqual({ a: 1 });
    expect(share({ a: [1] } as Record<string, unknown>, { a: { 0: 1 } })).toEqual({ a: { 0: 1 } });
    const prev = { l: [{ k: 1 }] };
    const merged = share(prev, { l: [{ k: 1 }, { k: 2 }] });
    expect(merged.l).toHaveLength(2);
    expect(merged.l[0]).toBe(prev.l[0]);
  });
});
