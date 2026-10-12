import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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
  raf.restore();
  vi.useRealTimers();
});

function setup(props: Partial<SliderProps> = {}) {
  const onChange = vi.fn();
  const view = render(
    <Slider label="Exposure" value={0.3} min={-4} max={4} step={0.01} defaultValue={0.3} format={formats.ev} onChange={onChange} {...props} />,
  );
  const slider = screen.getByRole("slider", { name: "Exposure" }) as HTMLInputElement;
  const control = view.container.querySelector(".ym-slider__control") as HTMLElement;
  mockRect(control, { left: 100, top: 0, width: 200, height: 38 });
  return { ...view, onChange, slider, control };
}

describe("Slider anatomy and ARIA", () => {
  it("has a native range input named by its label, with the limits and a readable value", () => {
    const { slider } = setup();
    expect(slider.type).toBe("range");
    expect(slider).toHaveAttribute("min", "-4");
    expect(slider).toHaveAttribute("max", "4");
    expect(slider).toHaveAttribute("step", "0.01");
    expect(slider).toHaveValue("0.3");
    expect(slider).toHaveAttribute("aria-valuetext", "+0.30");
    expect(screen.getByRole("textbox", { name: "Exposure value" })).toHaveValue("+0.30");
  });

  it("draws the track, detent and thumb, and no fill or Reset at the default", () => {
    const { container } = setup();
    expect(container.querySelector(".ym-slider__track")).not.toBeNull();
    expect(container.querySelector(".ym-slider__detent")).not.toBeNull();
    expect(container.querySelector(".ym-slider__thumb")).not.toBeNull();
    expect(container.querySelector(".ym-slider__fill")).toBeNull();
    expect(screen.queryByRole("button", { name: /reset/i })).toBeNull();
  });

  it("positions thumb and detent in percent of the range", () => {
    const { container } = setup({ value: -2, min: -4, max: 4, defaultValue: 0 });
    expect((container.querySelector(".ym-slider__thumb") as HTMLElement).style.left).toBe("25%");
    expect((container.querySelector(".ym-slider__detent") as HTMLElement).style.left).toBe("50%");
  });

  it("fills between the detent and the thumb once changed, and shows Reset", () => {
    const { container } = setup({ value: -2, min: -4, max: 4, defaultValue: 0 });
    const fill = container.querySelector(".ym-slider__fill") as HTMLElement;
    expect(fill.style.left).toBe("25%");
    expect(fill.style.width).toBe("25%");
    expect(screen.getByRole("textbox", { name: "Exposure value" })).toHaveAttribute("data-changed", "true");
    expect(screen.getByRole("button", { name: "Reset Exposure" })).toBeInTheDocument();
  });

  it("fills from the left end when there is no default, and has no detent", () => {
    const { container } = setup({ value: 75, min: 50, max: 100, step: 1, defaultValue: undefined, resetColumn: false });
    const fill = container.querySelector(".ym-slider__fill") as HTMLElement;
    expect(fill.style.left).toBe("0%");
    expect(fill.style.width).toBe("50%");
    expect(container.querySelector(".ym-slider__detent")).toBeNull();
    expect(container.querySelector(".ym-slider__reset")).toBeNull();
  });

  it("can force the changed look with `changed`", () => {
    setup({ changed: true });
    expect(screen.getByRole("button", { name: "Reset Exposure" })).toBeInTheDocument();
  });

  it("is disabled as a whole", async () => {
    const { slider, onChange, control } = setup({ disabled: true, value: 1 });
    expect(slider).toBeDisabled();
    expect(screen.queryByRole("button", { name: /reset/i })).toBeNull();
    fireEvent.pointerDown(control, { button: 0, clientX: 150 });
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
    expect(onChange).not.toHaveBeenCalled();
  });

  it("reports changes from the native input (assistive technology)", () => {
    const { slider, onChange } = setup({ value: 0, min: -100, max: 100, step: 1 });
    fireEvent.change(slider, { target: { value: "25" } });
    expect(onChange).toHaveBeenCalledWith(25, true);
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
    const { control, onChange } = setup({ value: -1.2 });
    fireEvent.doubleClick(control);
    expect(onChange).toHaveBeenCalledWith(0.3, true);
  });

  it("calls onReset instead when the caller has one (Exposure goes back to Auto)", async () => {
    const user = userEvent.setup();
    const onReset = vi.fn();
    const { onChange, control } = setup({ value: -1.2, onReset });
    await user.click(screen.getByRole("button", { name: "Reset Exposure" }));
    fireEvent.doubleClick(control);
    expect(onReset).toHaveBeenCalledTimes(2);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("does nothing on a double click when there is no default", () => {
    const { control, onChange } = setup({ defaultValue: undefined, value: 1 });
    fireEvent.doubleClick(control);
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("Slider dragging", () => {
  it("sends one patch per animation frame while dragging and a final one on release", () => {
    const { control, onChange, container } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0 });
    fireEvent.pointerDown(control, { button: 0, clientX: 200 }); // middle: 0
    fireEvent.pointerMove(window, { clientX: 250 }); // +50
    fireEvent.pointerMove(window, { clientX: 260 }); // +60
    fireEvent.pointerMove(window, { clientX: 270 }); // +70
    expect(onChange).not.toHaveBeenCalled();
    raf.flush();
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenLastCalledWith(70, false);
    fireEvent.pointerMove(window, { clientX: 280 }); // +80
    fireEvent.pointerUp(window, { clientX: 280 });
    expect(onChange).toHaveBeenLastCalledWith(80, true);
    raf.flush();
    expect(onChange).toHaveBeenCalledTimes(2); // the waiting frame was dropped; the release was final
    expect((container.querySelector(".ym-slider__thumb") as HTMLElement).style.left).toBe("90%");
  });

  it("draws its own value while dragging, before the parent answers", () => {
    const { control, container } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0 });
    fireEvent.pointerDown(control, { button: 0, clientX: 100 });
    expect((container.querySelector(".ym-slider__thumb") as HTMLElement).style.left).toBe("0%");
    expect(screen.getByRole("textbox", { name: "Exposure value" })).toHaveValue("−100.00");
  });

  it("snaps to the step and stays inside the range when the pointer leaves the track", () => {
    const { control, onChange } = setup({ value: 5450, min: 2000, max: 11000, step: 50, defaultValue: 5450 });
    fireEvent.pointerDown(control, { button: 0, clientX: 1000 });
    fireEvent.pointerUp(window);
    expect(onChange).toHaveBeenLastCalledWith(11000, true);
    fireEvent.pointerDown(control, { button: 0, clientX: 100 + 0.2 * 200 + 0.5 });
    fireEvent.pointerUp(window);
    expect(onChange).toHaveBeenLastCalledWith(3800, true);
  });

  it("ends the drag on pointercancel and ignores later movement", () => {
    const { control, onChange } = setup({ value: 0, min: -100, max: 100, step: 1, defaultValue: 0 });
    fireEvent.pointerDown(control, { button: 0, clientX: 250 });
    fireEvent.pointerCancel(window);
    expect(onChange).toHaveBeenLastCalledWith(50, true);
    const calls = onChange.mock.calls.length;
    fireEvent.pointerMove(window, { clientX: 290 });
    raf.flush();
    expect(onChange).toHaveBeenCalledTimes(calls);
  });

  it("holds the released value until the controlled value catches up", () => {
    const onChange = vi.fn();
    const { rerender, container } = render(
      <Slider label="Exposure" value={0} min={-100} max={100} step={1} defaultValue={0} onChange={onChange} />,
    );
    const control = container.querySelector(".ym-slider__control") as HTMLElement;
    mockRect(control, { left: 0, width: 200, height: 38 });
    fireEvent.pointerDown(control, { button: 0, clientX: 150 });
    fireEvent.pointerUp(window);
    const thumb = (): string => container.querySelector<HTMLElement>(".ym-slider__thumb")?.style.left ?? "";
    expect(thumb()).toBe("75%"); // the parent has not answered yet
    rerender(<Slider label="Exposure" value={50} min={-100} max={100} step={1} defaultValue={0} onChange={onChange} />);
    expect(thumb()).toBe("75%"); // it has: 50 is 75 % of the way from -100 to 100
    rerender(<Slider label="Exposure" value={-40} min={-100} max={100} step={1} defaultValue={0} onChange={onChange} />);
    expect(thumb()).toBe("30%"); // later, someone else changes it
  });

  it("does not jump back when an older reply arrives, and gives up on a value that never does", () => {
    vi.useFakeTimers();
    const onChange = vi.fn();
    const { rerender, container } = render(
      <Slider label="Exposure" value={0} min={-100} max={100} step={1} defaultValue={0} onChange={onChange} />,
    );
    const control = container.querySelector(".ym-slider__control") as HTMLElement;
    mockRect(control, { left: 0, width: 200, height: 38 });
    fireEvent.pointerDown(control, { button: 0, clientX: 150 });
    fireEvent.pointerUp(window);
    const thumb = (): string => container.querySelector<HTMLElement>(".ym-slider__thumb")?.style.left ?? "";
    rerender(<Slider label="Exposure" value={20} min={-100} max={100} step={1} defaultValue={0} onChange={onChange} />);
    expect(thumb()).toBe("75%"); // the reply is for an earlier patch
    act(() => {
      vi.advanceTimersByTime(1100);
    });
    expect(thumb()).toBe("60%"); // the core never agreed to 50: show what it says
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
