import { describe, expect, it } from "vitest";
import { MINUS } from "../../components/format";
import { tintLabel } from "./shared";

describe("tintLabel", () => {
  it("writes whole tints with a plus or a real minus", () => {
    expect(tintLabel(0)).toBe("0");
    expect(tintLabel(4)).toBe("+4");
    expect(tintLabel(-3)).toBe(`${MINUS}3`);
  });

  it("reads 0 for an automatic tint that rounds to zero, never +0 or −0", () => {
    // The automatic estimate is a continuous number: |tint| < 0.5 happens on ordinary photos.
    for (const tint of [0.3, -0.3, 0.49, -0.49, -0.5]) expect(tintLabel(tint)).toBe("0");
  });

  it("rounds a fractional tint to the nearest whole one", () => {
    expect(tintLabel(3.6)).toBe("+4");
    expect(tintLabel(-2.6)).toBe(`${MINUS}3`);
  });
});
