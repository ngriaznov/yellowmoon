import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MoonMark } from "./MoonMark";

describe("MoonMark", () => {
  it("is decorative", () => {
    const { container } = render(<MoonMark />);
    const mark = container.firstElementChild;
    expect(mark).toHaveAttribute("aria-hidden", "true");
    expect(mark).toHaveClass("ym-moon");
  });

  it("takes an explicit size", () => {
    const { container } = render(<MoonMark size={24} />);
    expect(container.firstElementChild).toHaveStyle({ width: "24px", height: "24px" });
  });
});
