import { describe, expect, it } from "vitest";
import { clamp, clampSnap, editText, formats, MINUS, parseNumber, roundToStep, sameValue, stepDecimals } from "./format";

describe("stepDecimals and roundToStep", () => {
  it("counts the decimals of a step", () => {
    expect(stepDecimals(1)).toBe(0);
    expect(stepDecimals(50)).toBe(0);
    expect(stepDecimals(0.1)).toBe(1);
    expect(stepDecimals(0.01)).toBe(2);
    expect(stepDecimals(1e-7)).toBe(7);
    expect(stepDecimals(0)).toBe(0);
  });

  it("removes float noise", () => {
    expect(roundToStep(0.1 + 0.2, 0.01)).toBe(0.3);
    expect(roundToStep(2.4000000000000004, 0.1)).toBe(2.4);
  });
});

describe("clamp and clampSnap", () => {
  it("clamps into the range", () => {
    expect(clamp(5, 0, 3)).toBe(3);
    expect(clamp(-5, 0, 3)).toBe(0);
    expect(clamp(2, 0, 3)).toBe(2);
  });

  it("snaps to the step grid that starts at min", () => {
    expect(clampSnap(5437, 2000, 11000, 50)).toBe(5450);
    expect(clampSnap(0.304, -4, 4, 0.01)).toBe(0.3);
    expect(clampSnap(99999, 2000, 11000, 50)).toBe(11000);
    expect(clampSnap(-3, 0.3, 2.5, 0.1)).toBe(0.3);
  });

  it("never snaps past max when the range is not whole steps", () => {
    expect(clampSnap(10, 0, 10, 3)).toBe(9);
  });

  it("compares values within a thousandth of a step", () => {
    expect(sameValue(0.3, 0.3000001, 0.01)).toBe(true);
    expect(sameValue(0.3, 0.31, 0.01)).toBe(false);
  });
});

describe("parseNumber", () => {
  it("reads plain numbers, signs and a real minus", () => {
    expect(parseNumber("12")).toBe(12);
    expect(parseNumber("+12")).toBe(12);
    expect(parseNumber("-18")).toBe(-18);
    expect(parseNumber(`${MINUS}18`)).toBe(-18);
    expect(parseNumber("0.30")).toBe(0.3);
    expect(parseNumber(".5")).toBe(0.5);
  });

  it("accepts a decimal comma and ignores units", () => {
    expect(parseNumber("0,5")).toBe(0.5);
    expect(parseNumber("5450 K")).toBe(5450);
    expect(parseNumber("0.8 px")).toBe(0.8);
    expect(parseNumber("100%")).toBe(100);
  });

  it("returns null when there is nothing to read", () => {
    expect(parseNumber("")).toBeNull();
    expect(parseNumber("abc")).toBeNull();
    expect(parseNumber("-")).toBeNull();
  });
});

describe("editText", () => {
  it("shows the number with the step's decimals and a plain hyphen", () => {
    expect(editText(0.3, 0.01)).toBe("0.30");
    expect(editText(-18, 1)).toBe("-18");
    expect(editText(5450, 50)).toBe("5450");
  });
});

describe("formats", () => {
  it("signed puts a plus on positives and a real minus on negatives", () => {
    expect(formats.signed(0)).toBe("0");
    expect(formats.signed(12)).toBe("+12");
    expect(formats.signed(-18)).toBe(`${MINUS}18`);
  });

  it("signed takes the sign from the rounded value, so a value that rounds to zero is plain 0", () => {
    for (const value of [0.3, -0.3, 0.49, -0.49, -0.5, -0, 1e-9]) expect(formats.signed(value)).toBe("0");
    expect(formats.signed(0.5)).toBe("+1");
    expect(formats.signed(-0.51)).toBe(`${MINUS}1`);
    expect(formats.signed(11.6)).toBe("+12");
    expect(formats.signed(-17.5)).toBe(`${MINUS}17`);
  });

  it("signed1 has a tenth, a plus or a real minus, and no sign at zero (the automatic contrast)", () => {
    expect(formats.signed1(4.2)).toBe("+4.2");
    expect(formats.signed1(-2)).toBe(`${MINUS}2.0`);
    expect(formats.signed1(0)).toBe("0.0");
    expect(formats.signed1(-0.04)).toBe("0.0");
    expect(formats.signed1(25)).toBe("+25.0");
  });

  it("ev has two decimals and no sign at zero", () => {
    expect(formats.ev(0.3)).toBe("+0.30");
    expect(formats.ev(-1.25)).toBe(`${MINUS}1.25`);
    expect(formats.ev(0)).toBe("0.00");
    expect(formats.ev(-0.001)).toBe("0.00");
  });

  it("formats units", () => {
    expect(formats.kelvin(5450)).toBe("5450 K");
    expect(formats.percent(100)).toBe("100%");
    expect(formats.px1(0.8)).toBe("0.8 px");
    expect(formats.px0(3000)).toBe("3000 px");
    expect(formats.fixed2(0.5)).toBe("0.50");
    expect(formats.integer(120.4)).toBe("120");
  });
});
