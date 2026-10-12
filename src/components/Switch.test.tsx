import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Switch } from "./Switch";

describe("Switch", () => {
  it("is a switch named by its label, with its state", () => {
    render(<Switch label="Neural demosaic" checked onChange={() => undefined} />);
    const sw = screen.getByRole("switch", { name: "Neural demosaic" });
    expect(sw).toBeChecked();
    expect(sw).toHaveAttribute("aria-checked", "true");
  });

  it("toggles on click, Space and Return", async () => {
    const user = userEvent.setup();
    function Harness() {
      const [on, setOn] = useState(false);
      return <Switch label="Lens corrections" checked={on} onChange={setOn} />;
    }
    render(<Harness />);
    const sw = screen.getByRole("switch", { name: "Lens corrections" });
    await user.click(sw);
    expect(sw).toBeChecked();
    await user.keyboard(" ");
    expect(sw).not.toBeChecked();
    await user.keyboard("{Enter}");
    expect(sw).toBeChecked();
  });

  it("reports the new state", async () => {
    const onChange = vi.fn();
    const { rerender } = render(<Switch label="x" checked={false} onChange={onChange} />);
    await userEvent.click(screen.getByRole("switch"));
    expect(onChange).toHaveBeenLastCalledWith(true);
    rerender(<Switch label="x" checked onChange={onChange} />);
    await userEvent.click(screen.getByRole("switch"));
    expect(onChange).toHaveBeenLastCalledWith(false);
  });

  it("does nothing when disabled", async () => {
    const onChange = vi.fn();
    render(<Switch label="x" checked={false} onChange={onChange} disabled />);
    await userEvent.click(screen.getByRole("switch"));
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByRole("switch")).toHaveAttribute("aria-disabled", "true");
  });

  it("can name its tab for the colour", () => {
    render(<Switch label="x" checked onChange={() => undefined} tab="raw" />);
    expect(screen.getByRole("switch")).toHaveAttribute("data-tab", "raw");
  });
});
