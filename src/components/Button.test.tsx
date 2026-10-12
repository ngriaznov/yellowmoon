import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Button, KeyHint, LinkButton } from "./Button";

describe("Button", () => {
  it("is a plain button that runs onClick", async () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Open files…</Button>);
    const button = screen.getByRole("button", { name: "Open files…" });
    expect(button).toHaveAttribute("type", "button");
    await userEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("carries its variant and size for the styles", () => {
    render(
      <>
        <Button>One</Button>
        <Button variant="secondary" size="lg" block>
          Two
        </Button>
        <Button variant="header">Three</Button>
      </>,
    );
    expect(screen.getByRole("button", { name: "One" })).toHaveAttribute("data-variant", "primary");
    expect(screen.getByRole("button", { name: "One" })).toHaveAttribute("data-size", "md");
    const two = screen.getByRole("button", { name: "Two" });
    expect(two).toHaveAttribute("data-variant", "secondary");
    expect(two).toHaveAttribute("data-size", "lg");
    expect(two).toHaveAttribute("data-block", "true");
    expect(screen.getByRole("button", { name: "Three" })).toHaveAttribute("data-variant", "header");
  });

  it("stays focusable when disabled, and ignores clicks", async () => {
    const onClick = vi.fn();
    render(
      <Button disabled onClick={onClick} title="Next unlocks when the zip arrives.">
        Next
      </Button>,
    );
    const button = screen.getByRole("button", { name: "Next" });
    expect(button).toHaveAttribute("aria-disabled", "true");
    expect(button).not.toBeDisabled();
    await userEvent.click(button);
    button.focus();
    await userEvent.keyboard("{Enter}");
    expect(onClick).not.toHaveBeenCalled();
    expect(button).toHaveFocus();
  });

  it("reports a toggle state for header buttons", () => {
    render(
      <>
        <Button variant="header" pressed>
          Before
        </Button>
        <Button variant="header" pressed={false}>
          Crop
        </Button>
        <Button variant="header">Plain</Button>
      </>,
    );
    expect(screen.getByRole("button", { name: "Before" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Crop" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "Plain" })).not.toHaveAttribute("aria-pressed");
  });

  it("draws a shortcut hint and a split aside", () => {
    render(
      <>
        <Button variant="header" shortcut="⌘O">
          Open files
        </Button>
        <Button aside="≈ 450 MB · ⌘↩">Export 3 photos</Button>
      </>,
    );
    expect(screen.getByRole("button", { name: "Open files ⌘O" })).toBeInTheDocument();
    const split = screen.getByRole("button", { name: /Export 3 photos/ });
    expect(split).toHaveAttribute("data-split", "true");
    expect(split).toHaveTextContent("≈ 450 MB · ⌘↩");
  });
});

describe("KeyHint and LinkButton", () => {
  it("draws a key hint", () => {
    render(<KeyHint>⌘E</KeyHint>);
    expect(screen.getByText("⌘E")).toHaveClass("ym-key-hint");
  });

  it("runs onClick, and does nothing when disabled", async () => {
    const onClick = vi.fn();
    const { rerender } = render(<LinkButton onClick={onClick}>Change</LinkButton>);
    await userEvent.click(screen.getByRole("button", { name: "Change" }));
    expect(onClick).toHaveBeenCalledTimes(1);
    rerender(
      <LinkButton onClick={onClick} disabled>
        Change
      </LinkButton>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Change" }));
    expect(onClick).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Change" })).toHaveAttribute("aria-disabled", "true");
  });
});
