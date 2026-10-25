import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { ChoiceList } from "./ChoiceList";
import type { ChoiceOption } from "./ChoiceList";

const OPTIONS: ChoiceOption<string>[] = [
  { value: "one", name: "One", line: "The first line." },
  { value: "two", name: "Two", line: "The second line." },
  { value: "three", name: "Three" },
];

function Harness({ initial = "one", disabled, onChange }: { initial?: string; disabled?: boolean; onChange?: (v: string) => void }) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <button type="button">Before</button>
      <ChoiceList
        label="Number"
        options={OPTIONS}
        value={value}
        disabled={disabled}
        onChange={(next) => {
          setValue(next);
          onChange?.(next);
        }}
      />
      <button type="button">After</button>
    </>
  );
}

/**
 * Presses an arrow the way a hand does: Radix chooses the row that the arrow focused only while the key is still down (its `keyup` ends the
 * arrow), and it moves the focus a tick after `keydown`, so `user.keyboard("{ArrowDown}")`, which releases at once, would only move the focus.
 */
async function arrow(user: ReturnType<typeof userEvent.setup>, key: "ArrowDown" | "ArrowUp", to: RegExp | string): Promise<void> {
  await user.keyboard(`{${key}>}`);
  await waitFor(() => expect(screen.getByRole("radio", { name: to })).toHaveFocus());
  await waitFor(() => expect(screen.getByRole("radio", { name: to })).toHaveAttribute("aria-checked", "true"));
  await user.keyboard(`{/${key}}`);
}

describe("ChoiceList", () => {
  it("is a radiogroup of radios, the chosen one checked, each named by its name and then its line", () => {
    render(<Harness initial="two" />);
    expect(screen.getByRole("radiogroup", { name: "Number" })).toBeInTheDocument();
    const radios = screen.getAllByRole("radio");
    expect(radios).toHaveLength(3);
    expect(screen.getByRole("radio", { name: "Two The second line." })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("radio", { name: "One The first line." })).toHaveAttribute("aria-checked", "false");
    // A choice with no line is named by its name alone.
    expect(screen.getByRole("radio", { name: "Three" })).toBeInTheDocument();
  });

  it("draws the name and the line of every row, the line under the name", () => {
    const { container } = render(<Harness />);
    const names = [...container.querySelectorAll(".ym-choice-row__name")].map((el) => el.textContent);
    const lines = [...container.querySelectorAll(".ym-choice-row__line")].map((el) => el.textContent);
    expect(names).toEqual(["One", "Two", "Three"]);
    expect(lines).toEqual(["The first line.", "The second line."]);
  });

  it("chooses on a click on the name, on the line and on the row, and reports only real changes", async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await userEvent.click(screen.getByText("The second line."));
    expect(onChange).toHaveBeenLastCalledWith("two");
    expect(screen.getByRole("radio", { name: /^Two/ })).toHaveAttribute("aria-checked", "true");
    await userEvent.click(screen.getByText("Three"));
    expect(onChange).toHaveBeenLastCalledWith("three");
    onChange.mockClear();
    await userEvent.click(screen.getByText("Three"));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("has one Tab stop, on the chosen row, and the arrows move and choose, wrapping at the ends", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness initial="two" onChange={onChange} />);
    await user.tab();
    await user.tab();
    expect(screen.getByRole("radio", { name: /^Two/ })).toHaveFocus();
    await arrow(user, "ArrowDown", "Three");
    await arrow(user, "ArrowDown", /^One/);
    expect(onChange).toHaveBeenLastCalledWith("one");
    await arrow(user, "ArrowUp", "Three");
    expect(onChange).toHaveBeenCalledTimes(3);
    // The list is one stop: Tab leaves it.
    await user.tab();
    expect(screen.getByRole("button", { name: "After" })).toHaveFocus();
    await user.tab({ shift: true });
    expect(screen.getByRole("radio", { name: "Three" })).toHaveFocus();
  });

  it("is off as a whole: nothing takes the focus or a click, and nothing is reported", async () => {
    const onChange = vi.fn();
    render(<Harness disabled onChange={onChange} />);
    for (const radio of screen.getAllByRole("radio")) expect(radio).toBeDisabled();
    await userEvent.click(screen.getByText("The second line."));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("rings the chosen row in the tab colour (an inset ring, so nothing is drawn outside the row)", () => {
    const { container } = render(<Harness initial="two" />);
    const rows = [...container.querySelectorAll<HTMLElement>(".ym-choice-row")];
    expect(rows).toHaveLength(3);
    for (const row of rows) expect(row.className).toContain("has-data-[state=checked]:inset-ring-primary");
    expect(rows[1]!.querySelector('[data-state="checked"]')).not.toBeNull();
    expect(rows[0]!.querySelector('[data-state="checked"]')).toBeNull();
  });
});
