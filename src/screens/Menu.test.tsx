import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Menu } from "./Menu";

const OPTIONS = [
  { value: "fit", label: "Fit" },
  { value: "50", label: "50%" },
  { value: "100", label: "100%" },
  { value: "200", label: "200%" },
];

function setup(value: string | null = "fit") {
  const onSelect = vi.fn();
  const view = render(
    <>
      <Menu
        label="Zoom"
        options={OPTIONS}
        value={value}
        onSelect={onSelect}
        trigger={({ ref, ...props }) => (
          <button ref={ref} type="button" {...props}>
            Open
          </button>
        )}
      />
      <button type="button">Elsewhere</button>
    </>,
  );
  return { ...view, onSelect, user: userEvent.setup(), trigger: screen.getByRole("button", { name: "Open" }) };
}

describe("Menu", () => {
  it("is closed until the trigger is pressed, and says it has a pop-up", async () => {
    const { user, trigger } = setup();
    expect(trigger).toHaveAttribute("aria-haspopup", "menu");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("menu")).toBeNull();
    await user.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("menu", { name: "Zoom" })).toBeInTheDocument();
    expect(screen.getAllByRole("menuitemradio")).toHaveLength(4);
  });

  it("checks the current option and puts focus on it", async () => {
    const { user, trigger } = setup("100");
    await user.click(trigger);
    expect(screen.getByRole("menuitemradio", { name: "100%" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("menuitemradio", { name: "100%" })).toHaveFocus();
  });

  it("focuses the first item when nothing is checked", async () => {
    const { user, trigger } = setup(null);
    await user.click(trigger);
    expect(screen.getByRole("menuitemradio", { name: "Fit" })).toHaveFocus();
  });

  it("opens with the down arrow on the trigger and moves with the arrows, wrapping", async () => {
    const { user, trigger } = setup();
    trigger.focus();
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("menuitemradio", { name: "Fit" })).toHaveFocus();
    await user.keyboard("{ArrowUp}");
    expect(screen.getByRole("menuitemradio", { name: "200%" })).toHaveFocus();
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("menuitemradio", { name: "Fit" })).toHaveFocus();
    await user.keyboard("{End}");
    expect(screen.getByRole("menuitemradio", { name: "200%" })).toHaveFocus();
    await user.keyboard("{Home}");
    expect(screen.getByRole("menuitemradio", { name: "Fit" })).toHaveFocus();
  });

  it("chooses with Return, closes and gives focus back to the trigger", async () => {
    const { user, trigger, onSelect } = setup();
    trigger.focus();
    await user.keyboard("{ArrowDown}{ArrowDown}{ArrowDown}{Enter}");
    expect(onSelect).toHaveBeenCalledWith("100");
    expect(screen.queryByRole("menu")).toBeNull();
    expect(trigger).toHaveFocus();
  });

  it("chooses with a click", async () => {
    const { user, trigger, onSelect } = setup();
    await user.click(trigger);
    await user.click(screen.getByRole("menuitemradio", { name: "200%" }));
    expect(onSelect).toHaveBeenCalledWith("200");
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("closes on Esc, returning focus, and on a click or Tab outside", async () => {
    const { user, trigger } = setup();
    await user.click(trigger);
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).toBeNull();
    expect(trigger).toHaveFocus();
    await user.click(trigger);
    await user.click(screen.getByRole("button", { name: "Elsewhere" }));
    expect(screen.queryByRole("menu")).toBeNull();
    await user.click(trigger);
    await user.keyboard("{Tab}");
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("keeps focus on the item the arrows chose when the screen draws again", async () => {
    const { user, trigger, rerender, onSelect } = setup();
    await user.click(trigger);
    await user.keyboard("{ArrowDown}{ArrowDown}");
    expect(screen.getByRole("menuitemradio", { name: "100%" })).toHaveFocus();
    // The parent passes new option objects on every render, as a screen does.
    rerender(
      <>
        <Menu
          label="Zoom"
          options={OPTIONS.map((o) => ({ ...o }))}
          value="fit"
          onSelect={onSelect}
          trigger={({ ref, ...props }) => (
            <button ref={ref} type="button" {...props}>
              Open
            </button>
          )}
        />
        <button type="button">Elsewhere</button>
      </>,
    );
    expect(screen.getByRole("menuitemradio", { name: "100%" })).toHaveFocus();
  });
});
