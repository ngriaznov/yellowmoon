import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LookTile } from "./LookTile";

describe("LookTile", () => {
  it("is a toggle button named after the look and its key", () => {
    render(<LookTile name="Film warm" src="data:," index={3} onSelect={() => undefined} />);
    const tile = screen.getByRole("button", { name: "Film warm look, key 3" });
    expect(tile).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("3")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByText("Film warm")).toBeInTheDocument();
  });

  it("selects on click", async () => {
    const onSelect = vi.fn();
    render(<LookTile name="Neutral" onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button", { name: "Neutral look" }));
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it("has a tooltip when it is given one, and the note is the tooltip of a disabled tile", () => {
    const { rerender } = render(<LookTile name="Film" title="ARRI looks · LogC4 · 3D LUT, 65 points" onSelect={() => undefined} />);
    expect(screen.getByRole("button", { name: "Film look" })).toHaveAttribute("title", "ARRI looks · LogC4 · 3D LUT, 65 points");
    rerender(<LookTile name="Film" onSelect={() => undefined} />);
    expect(screen.getByRole("button", { name: "Film look" })).not.toHaveAttribute("title");
    rerender(<LookTile name="Film" title="ARRI looks · LogC4 · 3D LUT, 65 points" disabled note="File missing or changed. Import again." onSelect={() => undefined} />);
    expect(screen.getByRole("button", { name: /^Film look/ })).toHaveAttribute("title", "File missing or changed. Import again.");
  });

  it("shows the selected state", () => {
    render(<LookTile name="Neutral" selected onSelect={() => undefined} />);
    expect(screen.getByRole("button", { name: "Neutral look" })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows the DEFAULT tag", () => {
    render(<LookTile name="Neutral" isDefault onSelect={() => undefined} />);
    expect(screen.getByText("Default")).toHaveClass("ym-look-tile__default");
  });

  it("has a badge only for keys 1 to 9", () => {
    const { rerender } = render(<LookTile name="A" index={9} onSelect={() => undefined} />);
    expect(screen.getByText("9")).toBeInTheDocument();
    rerender(<LookTile name="A" index={10} onSelect={() => undefined} />);
    expect(screen.queryByText("10")).toBeNull();
    expect(screen.getByRole("button", { name: "A look" })).toBeInTheDocument();
    rerender(<LookTile name="A" index={0} onSelect={() => undefined} />);
    expect(screen.getByRole("button", { name: "A look" })).toBeInTheDocument();
  });

  it("draws the thumbnail as a decorative image, or custom content", () => {
    const { container, rerender } = render(<LookTile name="A" src="thumb.png" onSelect={() => undefined} />);
    const img = container.querySelector("img");
    expect(img).toHaveAttribute("src", "thumb.png");
    expect(img).toHaveAttribute("alt", "");
    rerender(<LookTile name="A" thumbnail={<canvas data-testid="c" />} onSelect={() => undefined} />);
    expect(screen.getByTestId("c")).toBeInTheDocument();
    expect(container.querySelector("img")).toBeNull();
  });

  it("is disabled with a reason when its file is missing, and cannot be selected", async () => {
    const onSelect = vi.fn();
    render(<LookTile name="Gone" disabled note="File missing or changed" onSelect={onSelect} />);
    const tile = screen.getByRole("button", { name: "Gone look, file missing or changed" });
    expect(tile).toHaveAttribute("aria-disabled", "true");
    expect(tile).toHaveAttribute("title", "File missing or changed");
    await userEvent.click(tile);
    expect(onSelect).not.toHaveBeenCalled();
  });
});
