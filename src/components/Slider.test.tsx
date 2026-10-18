import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { capturingPointer } from "../test/pointer";
import { installManualRaf, mockRect } from "../test/raf";
import type { ManualRaf } from "../test/raf";
import { formats } from "./format";
import { Slider } from "./Slider";
import type { SliderProps } from "./Slider";

let raf: ManualRaf;
beforeEach(() => {
  raf = installManualRaf();
});
afterEach(() => {
  // Undone in the order they were done: the fake timers sit on top of the stubbed frame functions. (Radix's size observer calls
  // `cancelAnimationFrame` when the slider unmounts, so the real one has to be back by then.)
  vi.useRealTimers();
  raf.restore();
});

/** The Radix slider root: the element that measures the pointer and takes the pointer events. */
function rootOf(container: HTMLElement): HTMLElement {
  return container.querySelector('[data-slot="slider"]') as HTMLElement;
}

function setup(props: Partial<SliderProps> = {}) {
  const onChange = vi.fn();
  const view = render(
    <Slider label="Exposure" value={0.3} min={-4} max={4} step={0.01} defaultValue={0.3} format={formats.ev} onChange={onChange} {...props} />,
  );
  const slider = screen.getByRole("slider", { name: "Exposure" });
  const root = rootOf(view.container);
  mockRect(root, { left: 100, top: 0, width: 200, height: 38 });
  return { ...view, onChange, slider, root, pointer: capturingPointer(root) };
}

/** Where the knob sits, in percent of the track: Radix places it with `left: calc(P% + offset)`. */
function knobAt(slider: HTMLElement): string {
  const left = (slider.parentElement as HTMLElement).style.left;
  return `${/calc\(([\d.]+)%/.exec(left)?.[1]}%`;
}

/** The accent range of a slider with a default: it runs from the default to the knob (`inset-inline-start` and `width`). */
function rangeOf(container: HTMLElement): { start: string; width: string } {
  const range = container.querySelector('[data-slot="slider-range"]') as HTMLElement;
  return { start: range.style.getPropertyValue("inset-inline-start"), width: range.style.width };
}

describe("Slider anatomy and ARIA", () => {
  it("is Radix's slider named by its label, with the limits and a readable value", () => {
    const { slider } = setup();
    expect(slider).toHaveAttribute("aria-valuemin", "-4");
    expect(slider).toHaveAttribute("aria-valuemax", "4");
    expect(slider).toHaveAttribute("aria-valuenow", "0.3");
    expect(slider).toHaveAttribute("aria-valuetext", "+0.30");
    expect(slider).toHaveAttribute("aria-orientation", "horizontal");
    expect(slider).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("textbox", { name: "Exposure value" })).toHaveValue("+0.30");
  });

  it("keeps the readable value in step with the value", () => {
    const { rerender, slider, onChange } = setup({ value: 0.3 });
    rerender(<Slider label="Exposure" value={-1.5} min={-4} max={4} step={0.01} defaultValue={0.3} format={formats.ev} onChange={onChange} />);
    expect(slider).toHaveAttribute("aria-valuenow", "-1.5");
    expect(slider).toHaveAttribute("aria-valuetext", "−1.50");
  });

  it("draws the track and the knob, and no accent range or Reset at the default", () => {
    const { container, slider } = setup();
    expect(container.querySelector('[data-slot="slider-track"]')).not.toBeNull();
    expect(slider).toHaveAttribute("data-slot", "slider-thumb");
    expect(rangeOf(container).width).toBe("0%");
    expect(screen.queryByRole("button", { name: /reset/i })).toBeNull();
  });

  it("positions the knob and the accent range in percent of the range", () => {
    const { container, slider } = setup({ value: -2, min: -4, max: 4, defaultValue: 0 });
    expect(knobAt(slider)).toBe("25%");
    // From the default (50 %) back to the knob (25 %).
    expect(rangeOf(container)).toEqual({ start: "25%", width: "25%" });
  });

  it("runs the accent range up from the default when the value is above it", () => {
    const { container, slider } = setup({ value: 2, min: -4, max: 4, defaultValue: 0 });
    expect(knobAt(slider)).toBe("75%");
    expect(rangeOf(container)).toEqual({ start: "50%", width: "25%" });
  });

  it("shows Reset and marks the box once changed", () => {
    setup({ value: -2, min: -4, max: 4, defaultValue: 0 });
    expect(screen.getByRole("textbox", { name: "Exposure value" })).toHaveAttribute("data-changed", "true");
    expect(screen.getByRole("button", { name: "Reset Exposure" })).toBeInTheDocument();
  });

  it("fills from the left end when there is no default, and has no Reset", () => {
    const { container } = setup({ value: 75, min: 50, max: 100, step: 1, defaultValue: undefined, resetColumn: false });
    const range = container.querySelector('[data-slot="slider-range"]') as HTMLElement;
    expect(range.style.left).toBe("0%");
    expect(range.style.right).toBe("50%");
    expect(screen.queryByRole("button", { name: /reset/i })).toBeNull();
  });

  it("can force the changed look with `changed`", () => {
    setup({ changed: true });
    expect(screen.getByRole("button", { name: "Reset Exposure" })).toBeInTheDocument();
  });

  it("is disabled as a whole", () => {
    const { slider, onChange, pointer: drag, container } = setup({ disabled: true, value: 1 });
    expect(slider).toHaveAttribute("data-disabled");
    expect(rootOf(container)).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("textbox", { name: "Exposure value" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: /reset/i })).toBeNull();
    drag.down(150);
    drag.up();
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("Slider keyboard", () => {
  it("steps with the arrow keys", async () => {
    const user = userEvent.setup();
    const { slider, onChange } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0, format: formats.signed });
    slider.focus();
    await user.keyboard("{ArrowRight}");
    expect(onChange).toHaveBeenLastCalledWith(1, true);
    await user.keyboard("{ArrowUp}");
    expect(onChange).toHaveBeenLastCalledWith(2, true);
  });

  it("steps down with left and down arrows", async () => {
    const user = userEvent.setup();
    const { slider, onChange } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0 });
    slider.focus();
    await user.keyboard("{ArrowLeft}");
    expect(onChange).toHaveBeenLastCalledWith(-1, true);
    // The component is controlled and `value` did not change, but it shows its own draft: the
    // next key continues from -1, as it must while a patch is still in flight.
    await user.keyboard("{ArrowDown}");
    expect(onChange).toHaveBeenLastCalledWith(-2, true);
  });

  it("moves ten steps with Shift and PageUp/PageDown", async () => {
    const user = userEvent.setup();
    const { slider, onChange } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0 });
    slider.focus();
    await user.keyboard("{Shift>}{ArrowRight}{/Shift}");
    expect(onChange).toHaveBeenLastCalledWith(10, true);
    await user.keyboard("{PageUp}");
    expect(onChange).toHaveBeenLastCalledWith(20, true);
    await user.keyboard("{PageDown}{PageDown}");
    expect(onChange).toHaveBeenLastCalledWith(0, true);
  });

  it("jumps to the limits with Home and End", async () => {
    const user = userEvent.setup();
    const { slider, onChange } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0 });
    slider.focus();
    await user.keyboard("{Home}");
    expect(onChange).toHaveBeenLastCalledWith(-100, true);
    await user.keyboard("{End}");
    expect(onChange).toHaveBeenLastCalledWith(100, true);
  });

  it("clamps at the limits and does not report a change that is not one", async () => {
    const user = userEvent.setup();
    const { slider, onChange } = setup({ value: 100, min: -100, max: 100, step: 1, defaultValue: 0 });
    slider.focus();
    await user.keyboard("{ArrowRight}{End}");
    expect(onChange).not.toHaveBeenCalled();
    await user.keyboard("{Shift>}{ArrowLeft}{/Shift}");
    expect(onChange).toHaveBeenCalledWith(90, true);
  });

  it("keeps the step grid for fractional steps", async () => {
    const user = userEvent.setup();
    const { slider, onChange } = setup({ value: 0.3, step: 0.01 });
    slider.focus();
    await user.keyboard("{ArrowRight}{ArrowRight}");
    expect(onChange).toHaveBeenNthCalledWith(1, 0.31, true);
    // The value is controlled and did not change, but the draft shows the first step: the next lands on 0.32.
    expect(onChange).toHaveBeenNthCalledWith(2, 0.32, true);
  });

  it("leaves key combinations with modifiers to the app", async () => {
    const user = userEvent.setup();
    const { slider, onChange } = setup({ value: 0, min: -100, max: 100, step: 1 });
    slider.focus();
    await user.keyboard("{Meta>}{ArrowRight}{/Meta}");
    await user.keyboard("{Control>}{ArrowUp}{/Control}");
    await user.keyboard("{Alt>}{Home}{/Alt}");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("is adjusted by the arrow keys a screen reader sends, with nothing but a focus on the knob", async () => {
    const user = userEvent.setup();
    const { slider, onChange } = setup({ value: 0, min: -100, max: 100, step: 1 });
    // VoiceOver's adjust action arrives as ArrowUp and ArrowDown on the focused slider.
    slider.focus();
    await user.keyboard("{ArrowUp}");
    expect(onChange).toHaveBeenLastCalledWith(1, true);
    await user.keyboard("{ArrowDown}{ArrowDown}");
    expect(onChange).toHaveBeenLastCalledWith(-1, true);
  });
});

describe("Slider reset", () => {
  it("goes back to the default from the Reset link", async () => {
    const user = userEvent.setup();
    const { onChange } = setup({ value: -1.2 });
    await user.click(screen.getByRole("button", { name: "Reset Exposure" }));
    expect(onChange).toHaveBeenCalledWith(0.3, true);
  });

  it("goes back to the default on a double click of the track", () => {
    const { root, onChange } = setup({ value: -1.2 });
    fireEvent.doubleClick(root);
    expect(onChange).toHaveBeenCalledWith(0.3, true);
  });

  it("calls onReset instead when the caller has one (Exposure goes back to Auto)", async () => {
    const user = userEvent.setup();
    const onReset = vi.fn();
    const { onChange, root } = setup({ value: -1.2, onReset });
    await user.click(screen.getByRole("button", { name: "Reset Exposure" }));
    fireEvent.doubleClick(root);
    expect(onReset).toHaveBeenCalledTimes(2);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("shows its status in the Reset column while the value is the default, and Reset instead once it changes", async () => {
    const user = userEvent.setup();
    const { rerender, onChange } = setup({ status: <span>AUTO</span> });
    expect(screen.getByText("AUTO")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reset Exposure" })).toBeNull();
    rerender(<Slider label="Exposure" value={-1.2} min={-4} max={4} step={0.01} defaultValue={0.3} format={formats.ev} onChange={onChange} status={<span>AUTO</span>} />);
    expect(screen.queryByText("AUTO")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Reset Exposure" }));
    expect(onChange).toHaveBeenCalledWith(0.3, true);
  });

  it("does nothing on a double click when there is no default", () => {
    const { root, onChange } = setup({ defaultValue: undefined, value: 1 });
    fireEvent.doubleClick(root);
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("Slider dragging", () => {
  it("sends one patch per animation frame while dragging and a final one on release", () => {
    const { onChange, slider, pointer: drag } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0 });
    drag.down(200); // middle: 0
    drag.move(250); // +50
    drag.move(260); // +60
    drag.move(270); // +70
    expect(onChange).not.toHaveBeenCalled();
    raf.flush();
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenLastCalledWith(70, false);
    drag.move(280); // +80
    drag.up(280);
    expect(onChange).toHaveBeenLastCalledWith(80, true);
    raf.flush();
    expect(onChange).toHaveBeenCalledTimes(2); // the waiting frame was dropped; the release was final
    expect(knobAt(slider)).toBe("90%");
  });

  it("ignores a press of any button but the primary one (a right click on the track does not move the knob)", () => {
    const { onChange, slider, pointer: drag } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0 });
    for (const button of [2, 1]) {
      drag.down(250, button);
      drag.move(280);
      drag.up(280);
      raf.flush();
    }
    expect(onChange).not.toHaveBeenCalled();
    expect(knobAt(slider)).toBe("50%");
    expect(slider).toHaveAttribute("aria-valuenow", "0");
    // The primary button still drags.
    drag.down(250);
    drag.up(250);
    expect(onChange).toHaveBeenLastCalledWith(50, true);
  });

  it("draws its own value while dragging, before the parent answers", () => {
    const { slider, pointer: drag } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0 });
    drag.down(100);
    expect(knobAt(slider)).toBe("0%");
    expect(slider).toHaveAttribute("aria-valuenow", "-100");
    expect(screen.getByRole("textbox", { name: "Exposure value" })).toHaveValue("−100.00");
    drag.up();
  });

  it("snaps to the step and stays inside the range when the pointer leaves the track", () => {
    const { onChange, pointer: drag } = setup({ value: 5450, min: 2000, max: 11000, step: 50, defaultValue: 5450 });
    drag.down(1000);
    drag.up();
    expect(onChange).toHaveBeenLastCalledWith(11000, true);
    drag.down(100 + 0.2 * 200 + 0.5);
    drag.up();
    expect(onChange).toHaveBeenLastCalledWith(3800, true);
  });

  it("ends the drag on pointercancel and ignores later movement", () => {
    const { onChange, pointer: drag } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0 });
    drag.down(250);
    drag.cancel();
    expect(onChange).toHaveBeenLastCalledWith(50, true);
    const calls = onChange.mock.calls.length;
    drag.move(290);
    raf.flush();
    expect(onChange).toHaveBeenCalledTimes(calls);
  });

  it("holds the released value until the controlled value catches up", () => {
    const onChange = vi.fn();
    const { rerender, container } = render(
      <Slider label="Exposure" value={0} min={-100} max={100} step={1} defaultValue={0} onChange={onChange} />,
    );
    const root = rootOf(container);
    mockRect(root, { left: 0, width: 200, height: 38 });
    const drag = capturingPointer(root);
    const slider = screen.getByRole("slider", { name: "Exposure" });
    drag.down(150);
    drag.up();
    expect(knobAt(slider)).toBe("75%"); // the parent has not answered yet
    rerender(<Slider label="Exposure" value={50} min={-100} max={100} step={1} defaultValue={0} onChange={onChange} />);
    expect(knobAt(slider)).toBe("75%"); // it has: 50 is 75 % of the way from -100 to 100
    rerender(<Slider label="Exposure" value={-40} min={-100} max={100} step={1} defaultValue={0} onChange={onChange} />);
    expect(knobAt(slider)).toBe("30%"); // later, someone else changes it
  });

  it("does not jump back when an older reply arrives, and gives up on a value that never does", () => {
    vi.useFakeTimers();
    const onChange = vi.fn();
    const { rerender, container } = render(
      <Slider label="Exposure" value={0} min={-100} max={100} step={1} defaultValue={0} onChange={onChange} />,
    );
    const root = rootOf(container);
    mockRect(root, { left: 0, width: 200, height: 38 });
    const drag = capturingPointer(root);
    const slider = screen.getByRole("slider", { name: "Exposure" });
    drag.down(150);
    drag.up();
    rerender(<Slider label="Exposure" value={20} min={-100} max={100} step={1} defaultValue={0} onChange={onChange} />);
    expect(knobAt(slider)).toBe("75%"); // the reply is for an earlier patch
    act(() => {
      vi.advanceTimersByTime(1100);
    });
    expect(knobAt(slider)).toBe("60%"); // the core never agreed to 50: show what it says
  });

  it("takes typed values from the box and reports them as final", async () => {
    const user = userEvent.setup();
    const { onChange } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0, format: formats.signed });
    const box = screen.getByRole("textbox", { name: "Exposure value" });
    await user.click(box);
    await user.clear(box);
    await user.type(box, "-18{Enter}");
    expect(onChange).toHaveBeenCalledWith(-18, true);
  });
});
