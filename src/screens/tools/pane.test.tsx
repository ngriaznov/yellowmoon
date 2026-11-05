import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TabPane, usePaneActive } from "./pane";

let draws = 0;
function Probe() {
  draws += 1;
  return <p>{usePaneActive() ? "active" : "hidden"}</p>;
}
const pane = (active: boolean, n: number) => (
  <TabPane id="light" active={active}>
    <Probe key={n} />
  </TabPane>
);

describe("a tab pane", () => {
  it("is hidden, not removed, when another tab is shown, and tells what is inside", () => {
    const view = render(pane(true, 0));
    expect(screen.getByText("active")).toBeVisible();
    view.rerender(pane(false, 0));
    const kept = view.container.querySelector("[data-pane]");
    expect(kept).toHaveAttribute("data-hidden");
    expect(kept).toHaveAttribute("aria-hidden", "true");
    expect(kept).toHaveTextContent("hidden");
    view.rerender(pane(true, 0));
    expect(screen.getByText("active")).toBeVisible();
  });

  it("is not drawn again while it stays hidden, and catches up when it is shown", () => {
    draws = 0;
    const view = render(pane(false, 0));
    expect(draws).toBe(1);
    view.rerender(pane(false, 1));
    view.rerender(pane(false, 2));
    expect(draws, "a screen that draws at every tick of a drag does not draw the tabs that are not shown").toBe(1);
    view.rerender(pane(true, 3));
    expect(draws).toBe(2);
    expect(screen.getByText("active")).toBeVisible();
  });
});
