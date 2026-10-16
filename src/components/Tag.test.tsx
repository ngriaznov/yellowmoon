import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AutoToggle, Tag, TAG_LABELS } from "./Tag";
import type { TagVariant } from "./Tag";

describe("Tag", () => {
  it("has every variant of docs/11 and the start list", () => {
    const required: TagVariant[] = ["auto", "edited", "on", "stand-in", "reduced", "default", "cant-open", "waiting", "developing", "ready", "skipped"];
    for (const variant of required) expect(Object.keys(TAG_LABELS)).toContain(variant);
  });

  it("shows its own label and carries the variant", () => {
    render(<Tag variant="stand-in" />);
    const tag = screen.getByText("Stand-in");
    expect(tag).toHaveAttribute("data-slot", "badge");
    expect(tag).toHaveAttribute("data-variant", "stand-in");
  });

  it("uses the apostrophe of the mockups for CAN'T OPEN", () => {
    render(<Tag variant="cant-open" />);
    expect(screen.getByText("Can't open")).toBeInTheDocument();
  });

  it("takes its own text, such as a percentage", () => {
    render(<Tag variant="active">45%</Tag>);
    expect(screen.getByText("45%")).toHaveAttribute("data-variant", "active");
  });

  it("can be small, for a thumbnail", () => {
    render(<Tag variant="cant-open" size="sm" />);
    expect(screen.getByText("Can't open")).toHaveAttribute("data-size", "sm");
  });

  it("renders each variant with its text", () => {
    for (const [variant, label] of Object.entries(TAG_LABELS) as Array<[TagVariant, string]>) {
      const { unmount } = render(<Tag variant={variant} />);
      expect(screen.getByText(label)).toHaveAttribute("data-variant", variant);
      unmount();
    }
  });
});

describe("AutoToggle", () => {
  it("is pressed while the card is automatic, and pressing it does nothing", async () => {
    const onAuto = vi.fn();
    render(<AutoToggle isAuto onAuto={onAuto} />);
    const button = screen.getByRole("button", { name: "Auto" });
    expect(button).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(button);
    expect(onAuto).not.toHaveBeenCalled();
  });

  it("is not pressed once edited, and pressing it restores Auto", async () => {
    const onAuto = vi.fn();
    render(<AutoToggle isAuto={false} onAuto={onAuto} />);
    const button = screen.getByRole("button", { name: "Auto" });
    expect(button).toHaveAttribute("aria-pressed", "false");
    await userEvent.click(button);
    expect(onAuto).toHaveBeenCalledTimes(1);
  });
});
