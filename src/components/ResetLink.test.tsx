import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ResetLink } from "./ResetLink";

describe("ResetLink", () => {
  it("reads Reset by default and runs onClick", async () => {
    const onClick = vi.fn();
    render(<ResetLink onClick={onClick} />);
    await userEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("can read Re-run, take an accessible name and a card size", () => {
    render(<ResetLink label="Re-run" size="card" aria-label="Re-run automatic estimates" />);
    const link = screen.getByRole("button", { name: "Re-run automatic estimates" });
    expect(link).toHaveTextContent("Re-run");
    expect(link).toHaveAttribute("data-size", "card");
  });

  it("can be reached and used with the keyboard", async () => {
    const onClick = vi.fn();
    render(<ResetLink onClick={onClick} />);
    await userEvent.tab();
    expect(screen.getByRole("button", { name: "Reset" })).toHaveFocus();
    await userEvent.keyboard(" ");
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
