import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { formats } from "./format";
import { ValueBox } from "./ValueBox";

function setup(props: Partial<Parameters<typeof ValueBox>[0]> = {}) {
  const onCommit = vi.fn();
  const user = userEvent.setup();
  render(<ValueBox value={0.3} min={-4} max={4} step={0.01} label="Exposure value" format={formats.ev} onCommit={onCommit} {...props} />);
  const box = screen.getByRole("textbox", { name: "Exposure value" });
  return { box, onCommit, user };
}

describe("ValueBox", () => {
  it("shows the formatted value and is a labelled text box", () => {
    const { box } = setup();
    expect(box).toHaveValue("+0.30");
    expect(box).toHaveAttribute("inputmode", "decimal");
  });

  it("is paper at the default and takes the tab colour when changed", () => {
    const { box } = setup();
    expect(box).not.toHaveAttribute("data-changed");
    render(<ValueBox value={1} min={-4} max={4} label="Other" changed tab="light" onCommit={() => undefined} />);
    expect(screen.getByRole("textbox", { name: "Other" })).toHaveAttribute("data-changed", "true");
    expect(screen.getByRole("textbox", { name: "Other" })).toHaveAttribute("data-tab", "light");
  });

  it("switches to editing text on focus and selects it", async () => {
    const { box, user } = setup();
    await user.click(box);
    expect(box).toHaveValue("0.30");
    expect(box).toHaveAttribute("data-editing", "true");
    expect(box).toHaveFocus();
  });

  it("commits on Return, clamps and snaps", async () => {
    const { box, onCommit, user } = setup();
    await user.click(box);
    await user.clear(box);
    await user.type(box, "1.234{Enter}");
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenCalledWith(1.23);
    expect(box).not.toHaveFocus();
    expect(box).toHaveValue("+0.30"); // controlled: still shows the prop until it changes
  });

  it("clamps out-of-range values", async () => {
    const { box, onCommit, user } = setup();
    await user.click(box);
    await user.clear(box);
    await user.type(box, "99{Enter}");
    expect(onCommit).toHaveBeenLastCalledWith(4);
    await user.click(box);
    await user.clear(box);
    await user.type(box, "-99{Enter}");
    expect(onCommit).toHaveBeenLastCalledWith(-4);
  });

  it("reads a real minus sign and units", async () => {
    const { box, onCommit, user } = setup({ value: 5450, min: 2000, max: 11000, step: 50, format: formats.kelvin, label: "Exposure value" });
    await user.click(box);
    await user.clear(box);
    await user.type(box, "6200 K{Enter}");
    expect(onCommit).toHaveBeenLastCalledWith(6200);
  });

  it("cancels on Escape", async () => {
    const { box, onCommit, user } = setup();
    await user.click(box);
    await user.clear(box);
    await user.type(box, "2{Escape}");
    expect(onCommit).not.toHaveBeenCalled();
    expect(box).toHaveValue("+0.30");
    expect(box).not.toHaveFocus();
  });

  it("commits when the box loses focus", async () => {
    const { box, onCommit, user } = setup();
    await user.click(box);
    await user.clear(box);
    await user.type(box, "-1");
    await user.tab();
    expect(onCommit).toHaveBeenCalledWith(-1);
  });

  it("does nothing for text it cannot read, or for the same value", async () => {
    const { box, onCommit, user } = setup();
    await user.click(box);
    await user.clear(box);
    await user.type(box, "abc{Enter}");
    await user.click(box);
    await user.clear(box);
    await user.type(box, "0.3{Enter}");
    expect(onCommit).not.toHaveBeenCalled();
    expect(box).toHaveValue("+0.30");
  });

  it("nudges the typed value with the arrow keys, ten steps with Shift", async () => {
    const { box, onCommit, user } = setup({ step: 1, value: 10, min: 0, max: 100, format: undefined });
    await user.click(box);
    await user.keyboard("{ArrowUp}{ArrowUp}");
    expect(box).toHaveValue("12");
    await user.keyboard("{Shift>}{ArrowDown}{/Shift}");
    expect(box).toHaveValue("2");
    await user.keyboard("{Enter}");
    expect(onCommit).toHaveBeenCalledWith(2);
  });

  it("is inert when disabled", () => {
    const { box } = setup({ disabled: true });
    expect(box).toBeDisabled();
  });
});
