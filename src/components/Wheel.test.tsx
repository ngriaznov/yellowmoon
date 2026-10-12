import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installManualRaf, mockRect } from "../test/raf";
import type { ManualRaf } from "../test/raf";
import { Wheel, WheelGroup } from "./Wheel";
import type { WheelProps } from "./Wheel";

let raf: ManualRaf;
beforeEach(() => {
  raf = installManualRaf();
});
afterEach(() => raf.restore());

function setup(props: Partial<WheelProps> = {}) {
  const onChange = vi.fn();
  const view = render(<Wheel label="Shadows" value={[0, 0]} onChange={onChange} {...props} />);
  const disc = screen.getByRole("slider", { name: "Shadows colour balance" });
  mockRect(disc, { left: 0, top: 0, width: 92, height: 92 });
  return { ...view, disc, onChange };
}

describe("Wheel", () => {
  it("is a named slider whose value is the strength", () => {
    const { disc } = setup({ value: [0.3, 0.4] });
    expect(disc).toHaveAttribute("aria-valuemin", "0");
    expect(disc).toHaveAttribute("aria-valuemax", "100");
    expect(disc).toHaveAttribute("aria-valuenow", "50");
    expect(disc).toHaveAttribute("aria-valuetext", "Strength 50%, hue 37 degrees");
    expect(screen.getByText("Shadows")).toHaveClass("ym-wheel__label");
  });

  it("says Centred at the centre and paints the puck paper", () => {
    const { disc, container } = setup();
    expect(disc).toHaveAttribute("aria-valuetext", "Centred");
    expect((container.querySelector(".ym-wheel__puck") as HTMLElement).style.background).toBe("rgb(255, 255, 255)");
  });

  it("takes the hue of the direction it points in", () => {
    const { container } = setup({ value: [1, 0] });
    expect((container.querySelector(".ym-wheel__puck") as HTMLElement).style.background).toBe("rgb(255, 90, 31)");
  });

  it("puts the puck inside the rim: 88 px inside the border, the puck 14 px", () => {
    const { container, rerender } = setup({ value: [0, 0] });
    const puck = container.querySelector(".ym-wheel__puck") as HTMLElement;
    expect([puck.style.left, puck.style.top]).toEqual(["37px", "37px"]);
    rerender(<Wheel label="Shadows" value={[1, 0]} onChange={() => undefined} />);
    expect(puck.style.left).toBe("74px");
    expect(puck.style.top).toBe("37px");
  });

  it("drags: one patch per frame, a final one on release, kept inside the unit disc", () => {
    const { disc, onChange } = setup();
    fireEvent.pointerDown(disc, { button: 0, clientX: 46 + 18.5, clientY: 46 });
    fireEvent.pointerMove(window, { clientX: 46 + 30, clientY: 46 - 10 });
    fireEvent.pointerMove(window, { clientX: 46 + 100, clientY: 46 });
    expect(onChange).not.toHaveBeenCalled();
    raf.flush();
    expect(onChange).toHaveBeenCalledTimes(1);
    const [point, final] = onChange.mock.calls[0] as [[number, number], boolean];
    expect(final).toBe(false);
    expect(point[0]).toBeCloseTo(1, 12);
    expect(point[1]).toBeCloseTo(0, 12);
    fireEvent.pointerUp(window);
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(onChange.mock.calls[1]?.[1]).toBe(true);
  });

  it("measures the pointer up as positive b", () => {
    const { disc, onChange } = setup();
    fireEvent.pointerDown(disc, { button: 0, clientX: 46, clientY: 46 - 18.5 });
    fireEvent.pointerUp(window);
    const [point] = onChange.mock.calls.at(-1) as [[number, number], boolean];
    expect(point[0]).toBeCloseTo(0, 12);
    expect(point[1]).toBeCloseTo(0.5, 12);
  });

  it("resets to the centre on a double click", () => {
    const { disc, onChange } = setup({ value: [0.3, 0.2] });
    fireEvent.doubleClick(disc);
    expect(onChange).toHaveBeenCalledWith([0, 0], true);
  });

  it("does not report a reset when already centred", () => {
    const { disc, onChange } = setup();
    fireEvent.doubleClick(disc);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("moves with the arrow keys, ten steps with Shift, and centres with Home, Delete or Backspace", async () => {
    const user = userEvent.setup();
    const { disc, onChange } = setup({ value: [0, 0] });
    disc.focus();
    await user.keyboard("{ArrowRight}");
    expect(onChange).toHaveBeenLastCalledWith([0.02, 0], true);
    await user.keyboard("{ArrowUp}");
    expect(onChange).toHaveBeenLastCalledWith([0.02, 0.02], true);
    await user.keyboard("{Shift>}{ArrowLeft}{/Shift}");
    const [point] = onChange.mock.calls.at(-1) as [[number, number], boolean];
    expect(point[0]).toBeCloseTo(-0.18, 12);
    await user.keyboard("{Home}");
    expect(onChange).toHaveBeenLastCalledWith([0, 0], true);
  });

  it("stays inside the disc on the keyboard", async () => {
    const user = userEvent.setup();
    const { disc, onChange } = setup({ value: [0.99, 0.1] });
    disc.focus();
    await user.keyboard("{ArrowRight}");
    const [point] = onChange.mock.calls.at(-1) as [[number, number], boolean];
    expect(Math.hypot(point[0], point[1])).toBeCloseTo(1, 12);
  });

  it("reports nothing when pushed outwards from the rim", async () => {
    const user = userEvent.setup();
    const { disc, onChange } = setup({ value: [1, 0] });
    disc.focus();
    await user.keyboard("{ArrowRight}");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("is inert when disabled", async () => {
    const { disc, onChange } = setup({ disabled: true, value: [0.2, 0.2] });
    expect(disc).toHaveAttribute("tabindex", "-1");
    fireEvent.pointerDown(disc, { button: 0, clientX: 80, clientY: 46 });
    fireEvent.doubleClick(disc);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("lays three wheels in a group", () => {
    const { container } = render(
      <WheelGroup>
        <Wheel label="Shadows" value={[0, 0]} onChange={() => undefined} />
        <Wheel label="Midtones" value={[0, 0]} onChange={() => undefined} />
        <Wheel label="Highlights" value={[0, 0]} onChange={() => undefined} />
      </WheelGroup>,
    );
    expect(container.querySelector(".ym-wheels")?.children).toHaveLength(3);
    expect(screen.getAllByRole("slider")).toHaveLength(3);
  });
});
