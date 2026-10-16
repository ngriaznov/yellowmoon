import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProgressBar } from "./ProgressBar";

describe("ProgressBar", () => {
  it("is a progressbar with a name and the 0–100 range", () => {
    render(<ProgressBar value={40} label="Developing photos" />);
    const bar = screen.getByRole("progressbar", { name: "Developing photos" });
    expect(bar).toHaveAttribute("aria-valuemin", "0");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
    expect(bar).toHaveAttribute("aria-valuenow", "40");
  });

  it("fills in percent", () => {
    const { container } = render(<ProgressBar value={42.4} label="x" />);
    expect((container.querySelector('[data-slot="progress-indicator"]') as HTMLElement).style.width).toBe("42%");
  });

  it("keeps the value inside 0–100", () => {
    const { rerender } = render(<ProgressBar value={-5} label="x" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
    rerender(<ProgressBar value={250} label="x" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
  });

  it("draws no fill at 0, part of the bar between and all of it at 100 per cent", () => {
    const width = (): string => (document.querySelector('[data-slot="progress-indicator"]') as HTMLElement).style.width;
    const { rerender } = render(<ProgressBar value={0} label="x" />);
    expect(width()).toBe("0%");
    rerender(<ProgressBar value={50} label="x" />);
    expect(width()).toBe("50%");
    rerender(<ProgressBar value={100} label="x" />);
    expect(width()).toBe("100%");
  });

  it("has a large size for the wizard", () => {
    render(<ProgressBar value={10} label="x" size="lg" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("data-size", "lg");
  });
});
