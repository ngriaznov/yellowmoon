import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { StepRail } from "./StepRail";

const STEPS = [
  { title: "Choose", text: "Pick the ARRI packages you want." },
  { title: "Get the files", text: "Download them from ARRI's site." },
  { title: "Recognise", text: "We check what each package is." },
  { title: "Import", text: "Copy the looks into the app." },
  { title: "Finish", text: "Pick a default and clean up." },
];

describe("StepRail", () => {
  it("is a navigation with an ordered list of the five steps", () => {
    render(<StepRail steps={STEPS} current={1} />);
    expect(screen.getByRole("navigation", { name: "Steps" })).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(5);
  });

  it("marks the current step with aria-current", () => {
    render(<StepRail steps={STEPS} current={1} />);
    const current = screen.getByRole("button", { name: /^Get the files\./ });
    expect(current).toHaveAttribute("aria-current", "step");
    expect(current).toHaveAttribute("data-state", "current");
    expect(screen.getByRole("button", { name: /^Choose/ })).not.toHaveAttribute("aria-current");
  });

  it("shows numbers for the current and later steps and a tick for finished ones", () => {
    const { container } = render(<StepRail steps={STEPS} current={2} />);
    const marks = [...container.querySelectorAll(".ym-step-rail__mark")];
    expect(marks[0]?.querySelector("svg")).not.toBeNull();
    expect(marks[1]?.querySelector("svg")).not.toBeNull();
    expect(marks[2]?.textContent).toBe("3");
    expect(marks[4]?.textContent).toBe("5");
  });

  it("says in the name which steps are done", () => {
    render(<StepRail steps={STEPS} current={2} />);
    expect(screen.getByRole("button", { name: "Choose, done. Pick the ARRI packages you want." })).toBeInTheDocument();
  });

  it("locks later steps", () => {
    render(<StepRail steps={STEPS} current={1} />);
    const locked = screen.getByRole("button", { name: /^Import/ });
    expect(locked).toHaveAttribute("aria-disabled", "true");
    expect(locked).toHaveAttribute("data-state", "locked");
  });

  it("goes back to a finished step on click, and ignores current and locked ones", async () => {
    const onSelect = vi.fn();
    render(<StepRail steps={STEPS} current={2} onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button", { name: /^Choose/ }));
    expect(onSelect).toHaveBeenCalledWith(0);
    await userEvent.click(screen.getByRole("button", { name: /^Recognise/ }));
    await userEvent.click(screen.getByRole("button", { name: /^Finish/ }));
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it("draws each step as a glasscn Item", () => {
    render(<StepRail steps={STEPS} current={1} />);
    for (const button of screen.getAllByRole("button")) expect(button).toHaveAttribute("data-slot", "item");
  });
});
