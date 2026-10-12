import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Histogram, histogramMax, histogramPath } from "./Histogram";

const DATA = { red: [1, 4, 2], green: [2, 8, 4], blue: [0, 2, 1] };

describe("histogramPath", () => {
  it("closes a filled path along the bottom", () => {
    expect(histogramPath([0, 2, 0], 2, 100, 50)).toBe("M0 50 L0 50 L50 0 L100 50 L100 50 Z");
  });

  it("normalises by the maximum and caps at the top", () => {
    expect(histogramPath([1, 1], 2, 100, 100)).toBe("M0 100 L0 50 L100 50 L100 100 Z");
    expect(histogramPath([5, 5], 2, 100, 100)).toBe("M0 100 L0 0 L100 0 L100 100 Z");
  });

  it("draws nothing for fewer than two bins or an empty histogram", () => {
    expect(histogramPath([3], 3)).toBe("");
    expect(histogramPath([0, 0], 0)).toBe("");
  });

  it("finds the maximum across channels", () => {
    expect(histogramMax([1, 2], [3, 9], [4])).toBe(9);
    expect(histogramMax()).toBe(0);
  });
});

describe("Histogram", () => {
  it("is an image named Histogram with three channel fills", () => {
    const { container } = render(<Histogram data={DATA} />);
    expect(screen.getByRole("img", { name: "Histogram" })).toBeInTheDocument();
    expect(container.querySelectorAll("path")).toHaveLength(3);
  });

  it("scales all channels together", () => {
    const { container } = render(<Histogram data={DATA} />);
    const paths = [...container.querySelectorAll("path")].map((p) => p.getAttribute("d"));
    // The green peak (8) reaches the top of the 84-unit box; the red peak (4) reaches half way.
    expect(paths[1]).toContain("L160 0");
    expect(paths[0]).toContain("L160 42");
  });

  it("shows the caption row when given text", () => {
    render(<Histogram data={DATA} left="R 182  G 164  B 140" right="ISO 400 · 1/125 · f/2.8" />);
    expect(screen.getByText("R 182 G 164 B 140")).toBeInTheDocument();
    expect(screen.getByText("ISO 400 · 1/125 · f/2.8")).toBeInTheDocument();
  });

  it("has no caption row without text", () => {
    const { container } = render(<Histogram data={DATA} />);
    expect(container.querySelector(".ym-histogram__caption")).toBeNull();
  });
});
