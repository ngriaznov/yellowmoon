import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Segmented } from "./Segmented";
import type { SegmentedOption } from "./Segmented";

const OPTIONS: SegmentedOption<string>[] = [
  { value: "this", label: "This photo" },
  { value: "picked", label: "Picked · 3" },
  { value: "all", label: "All · 12" },
];

function Harness({ options = OPTIONS, initial = "this", onChange }: { options?: SegmentedOption<string>[]; initial?: string; onChange?: (v: string) => void }) {
  const [value, setValue] = useState(initial);
  return (
    <Segmented
      label="Apply look to"
      options={options}
      value={value}
      onChange={(v) => {
        setValue(v);
        onChange?.(v);
      }}
    />
  );
}

describe("Segmented", () => {
  it("is a radiogroup of radios with the active one checked", () => {
    render(<Harness initial="picked" />);
    expect(screen.getByRole("radiogroup", { name: "Apply look to" })).toBeInTheDocument();
    const radios = screen.getAllByRole("radio");
    expect(radios).toHaveLength(3);
    expect(screen.getByRole("radio", { name: "Picked · 3" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "This photo" })).not.toBeChecked();
  });

  it("selects on click and reports only real changes", async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await userEvent.click(screen.getByRole("radio", { name: "All · 12" }));
    expect(onChange).toHaveBeenCalledWith("all");
    await userEvent.click(screen.getByRole("radio", { name: "All · 12" }));
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("puts only the active option in the tab order", async () => {
    const user = userEvent.setup();
    render(<Harness initial="picked" />);
    // Radix's roving focus: the group takes the Tab and hands the focus to the active option; the others are left out (tabindex -1).
    await user.tab();
    expect(screen.getByRole("radio", { name: "Picked · 3" })).toHaveFocus();
    expect(screen.getByRole("radio", { name: "This photo" })).toHaveAttribute("tabindex", "-1");
    expect(screen.getByRole("radio", { name: "All · 12" })).toHaveAttribute("tabindex", "-1");
    await user.tab();
    expect(screen.getAllByRole("radio")).not.toContain(document.activeElement);
  });

  it("moves the selection and focus with the arrow keys, and wraps", async () => {
    const user = userEvent.setup();
    render(<Harness initial="all" />);
    await user.tab();
    expect(screen.getByRole("radio", { name: "All · 12" })).toHaveFocus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("radio", { name: "This photo" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "This photo" })).toHaveFocus();
    await user.keyboard("{ArrowLeft}");
    expect(screen.getByRole("radio", { name: "All · 12" })).toBeChecked();
    await user.keyboard("{ArrowDown}{ArrowUp}");
    expect(screen.getByRole("radio", { name: "All · 12" })).toBeChecked();
  });

  it("jumps to the ends with Home and End", async () => {
    const user = userEvent.setup();
    render(<Harness initial="picked" />);
    await user.tab();
    await user.keyboard("{End}");
    expect(screen.getByRole("radio", { name: "All · 12" })).toBeChecked();
    await user.keyboard("{Home}");
    expect(screen.getByRole("radio", { name: "This photo" })).toBeChecked();
  });

  it("skips disabled options", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <Harness
        initial="ours"
        onChange={onChange}
        options={[
          { value: "ours", label: "Ours" },
          { value: "arri", label: "ARRI's", disabled: true },
          { value: "other", label: "Other" },
        ]}
      />,
    );
    await user.tab();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("radio", { name: "Other" })).toBeChecked();
    await user.click(screen.getByRole("radio", { name: "ARRI's" }));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("radio", { name: "ARRI's" })).toBeDisabled();
  });

  it("sets one column per option, or as many as asked", () => {
    const { rerender } = render(<Harness />);
    // The segments share the row evenly (each one is `flex: 1`); `columns` is only recorded, for a caller that wants a grid.
    expect(screen.getByRole("radiogroup")).not.toHaveAttribute("data-columns");
    expect(screen.getAllByRole("radio")).toHaveLength(3);
    rerender(<Segmented label="x" options={OPTIONS} value="this" onChange={() => undefined} columns={6} size="lg" />);
    expect(screen.getByRole("radiogroup")).toHaveAttribute("data-columns", "6");
    expect(screen.getByRole("radiogroup")).toHaveAttribute("data-size", "lg");
  });

  it("leaves modified arrow keys to the app", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await user.tab();
    await user.keyboard("{Meta>}{ArrowRight}{/Meta}");
    expect(onChange).not.toHaveBeenCalled();
  });
});
