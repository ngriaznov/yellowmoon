import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Divider } from "./Divider";
import { DEFAULT_FRACTION, EXPANDED_FRACTION, layoutOf, MIN_PHOTO_H } from "./layout";
import { useNodeLayout } from "./useNodeLayout";

/** The divider and the layout it drives, as the Photos screen wires them, with what the layout says drawn as text. */
function Harness() {
  const layout = useNodeLayout(true);
  return (
    <>
      <Divider editor={layout.editor} area={layout.area} min={layout.min} max={layout.max} onResize={layout.setEditor} onNudge={layout.nudge} onReset={layout.reset} />
      <button type="button" onClick={layout.toggleExpanded} aria-pressed={layout.expanded}>
        Expand
      </button>
      <output aria-label="editor">{layout.editor}</output>
      <output aria-label="photo">{layout.viewer}</output>
    </>
  );
}

const editor = (): number => Number(screen.getByLabelText("editor").textContent);
const viewer = (): number => Number(screen.getByLabelText("photo").textContent);
const divider = (): HTMLElement => screen.getByRole("separator", { name: "Resize the graph" });

function setWindowHeight(height: number): void {
  Object.defineProperty(window, "innerHeight", { value: height, configurable: true, writable: true });
  act(() => {
    window.dispatchEvent(new Event("resize"));
  });
}

let before = window.innerHeight;
beforeEach(() => {
  before = window.innerHeight;
  Object.defineProperty(window, "innerHeight", { value: 900, configurable: true, writable: true });
});
afterEach(() => {
  Object.defineProperty(window, "innerHeight", { value: before, configurable: true, writable: true });
});

describe("the divider", () => {
  it("is a separator named 'Resize the graph', horizontal, focusable, with its place as a share of the height under the header", () => {
    render(<Harness />);
    const el = divider();
    expect(el).toHaveAttribute("aria-orientation", "horizontal");
    expect(el).toHaveAttribute("tabindex", "0");
    const l = layoutOf(900, DEFAULT_FRACTION);
    expect(el).toHaveAttribute("aria-valuenow", String(Math.round((l.editor / l.area) * 100)));
    expect(el).toHaveAttribute("aria-valuenow", "45");
    expect(el).toHaveAttribute("aria-valuemin", "25");
    expect(Number(el.getAttribute("aria-valuemax"))).toBeLessThanOrEqual(70);
    expect(editor()).toBe(371);
    expect(viewer()).toBe(444);
  });

  it("moves 10 px with ↑ and ↓ (↑ takes the divider up: the graph grows) and 50 with ⇧", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    divider().focus();
    await user.keyboard("{ArrowUp}");
    expect(editor()).toBe(381);
    expect(viewer()).toBe(434);
    await user.keyboard("{ArrowDown}{ArrowDown}");
    expect(editor()).toBe(361);
    await user.keyboard("{Shift>}{ArrowUp}{/Shift}");
    expect(editor()).toBe(411);
    await user.keyboard("{Shift>}{ArrowDown}{ArrowDown}{/Shift}");
    expect(editor()).toBe(311);
  });

  it("stops at 25 % and at 70 % (or where the photo has 200 px), and moves off a bound at once", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const l = layoutOf(900, DEFAULT_FRACTION);
    divider().focus();
    await user.keyboard("{Shift>}" + "{ArrowUp}".repeat(12) + "{/Shift}");
    expect(editor()).toBe(l.max);
    expect(viewer()).toBeGreaterThanOrEqual(MIN_PHOTO_H);
    // Nothing is stored past the bound: one press down moves 10 px from it.
    await user.keyboard("{ArrowDown}");
    expect(editor()).toBe(l.max - 10);
    await user.keyboard("{Shift>}" + "{ArrowDown}".repeat(12) + "{/Shift}");
    expect(editor()).toBe(l.min);
    await user.keyboard("{ArrowUp}");
    expect(editor()).toBe(l.min + 10);
  });

  it("leaves ←, →, other keys and ⌘, ⌃ and ⌥ with the arrows alone, and takes the arrows it uses from the page", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    divider().focus();
    await user.keyboard("{ArrowLeft}{ArrowRight}a{Home}{End}");
    await user.keyboard("{Meta>}{ArrowUp}{/Meta}{Control>}{ArrowUp}{/Control}{Alt>}{ArrowUp}{/Alt}");
    expect(editor()).toBe(371);
    expect(fireEvent.keyDown(divider(), { key: "ArrowUp" }), "the arrow is used: the page does not scroll").toBe(false);
    expect(fireEvent.keyDown(divider(), { key: "ArrowLeft" }), "← is not used: the shortcut for the previous photo gets it").toBe(true);
  });

  it("is dragged with the pointer: up makes the graph taller, down shorter, the photo keeping its 200 px and the graph its 25 %", () => {
    render(<Harness />);
    const l = layoutOf(900, DEFAULT_FRACTION);
    fireEvent.pointerDown(divider(), { clientY: 500, button: 0, pointerId: 1 });
    expect(divider()).toHaveAttribute("data-dragging", "true");
    fireEvent.pointerMove(window, { clientY: 450, pointerId: 1 });
    expect(editor()).toBe(421);
    expect(viewer()).toBe(394);
    fireEvent.pointerMove(window, { clientY: 520, pointerId: 1 });
    expect(editor()).toBe(351);
    // Far past either end.
    fireEvent.pointerMove(window, { clientY: -4000, pointerId: 1 });
    expect(editor()).toBe(l.max);
    expect(viewer()).toBeGreaterThanOrEqual(MIN_PHOTO_H);
    fireEvent.pointerMove(window, { clientY: 4000, pointerId: 1 });
    expect(editor()).toBe(l.min);
    // And back to the pointer's own place from the other bound: the drag has no dead zone.
    fireEvent.pointerMove(window, { clientY: 500, pointerId: 1 });
    expect(editor()).toBe(371);
    fireEvent.pointerUp(window, { clientY: 500, pointerId: 1 });
    expect(divider()).not.toHaveAttribute("data-dragging");
    // A move after the release is not a drag.
    fireEvent.pointerMove(window, { clientY: 100, pointerId: 1 });
    expect(editor()).toBe(371);
  });

  it("is not dragged by any button but the primary one", () => {
    render(<Harness />);
    fireEvent.pointerDown(divider(), { clientY: 500, button: 2, pointerId: 1 });
    fireEvent.pointerMove(window, { clientY: 300, pointerId: 1 });
    expect(editor()).toBe(371);
  });

  it("returns to 45 % on a double click, from wherever it was, and ends Expand", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    divider().focus();
    await user.keyboard("{Shift>}{ArrowUp}{ArrowUp}{/Shift}");
    expect(editor()).toBe(471);
    await user.dblClick(divider());
    expect(editor()).toBe(371);
    await user.click(screen.getByRole("button", { name: "Expand" }));
    expect(editor()).toBe(545);
    await user.dblClick(divider());
    expect(editor()).toBe(371);
    expect(screen.getByRole("button", { name: "Expand" })).toHaveAttribute("aria-pressed", "false");
  });

  it("is Expand's too: 66 %, and back to where the divider was; a drag or an arrow while it is on ends it", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    divider().focus();
    await user.keyboard("{Shift>}{ArrowDown}{/Shift}");
    expect(editor()).toBe(321);
    await user.click(screen.getByRole("button", { name: "Expand" }));
    expect(editor()).toBe(layoutOf(900, EXPANDED_FRACTION).editor);
    expect(editor()).toBe(545);
    expect(viewer()).toBe(270);
    await user.click(screen.getByRole("button", { name: "Expand" }));
    expect(editor()).toBe(321);
    // With Expand on, an arrow moves from the 545 it shows, and Expand is off.
    await user.click(screen.getByRole("button", { name: "Expand" }));
    divider().focus();
    await user.keyboard("{ArrowDown}");
    expect(editor()).toBe(535);
    expect(screen.getByRole("button", { name: "Expand" })).toHaveAttribute("aria-pressed", "false");
    // A drag does the same.
    await user.click(screen.getByRole("button", { name: "Expand" }));
    expect(editor()).toBe(545);
    fireEvent.pointerDown(divider(), { clientY: 500, button: 0, pointerId: 1 });
    fireEvent.pointerMove(window, { clientY: 540, pointerId: 1 });
    fireEvent.pointerUp(window, { clientY: 540, pointerId: 1 });
    expect(editor()).toBe(505);
    expect(screen.getByRole("button", { name: "Expand" })).toHaveAttribute("aria-pressed", "false");
  });

  it("keeps its proportion when the window changes height, and never gives the photo less than 200 px at the smallest window", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    expect(editor()).toBe(371);
    setWindowHeight(1117);
    expect(editor()).toBe(469);
    expect(viewer()).toBe(563);
    divider().focus();
    await user.keyboard("{Shift>}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{ArrowUp}{/Shift}");
    setWindowHeight(700);
    // 700 px is shorter than the smallest window: the screen is 760 px.
    expect(viewer()).toBeGreaterThanOrEqual(MIN_PHOTO_H);
    expect(editor() + 10 + viewer()).toBe(760 - 75);
  });
});
