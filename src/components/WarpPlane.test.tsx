import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ColorWarpPin } from "../api/types";
import { installManualRaf, mockRect } from "../test/raf";
import type { ManualRaf } from "../test/raf";
import { WarpPlane, WARP_KEY_STEP } from "./WarpPlane";
import type { WarpTool } from "./WarpPlane";
import { lockedPin, movingPin, planeToPx, WARP_MAX_PINS } from "./warpMath";

let raf: ManualRaf;
beforeEach(() => {
  raf = installManualRaf();
});
afterEach(() => raf.restore());

/** The plane is 352 px square at (0, 0): the centre is at 176, and 0.3 of chroma is 176 px. */
const SIZE = 352;
const at = (a: number, b: number): { clientX: number; clientY: number } => {
  const p = planeToPx([a, b], SIZE);
  return { clientX: p.x, clientY: p.y };
};

/** The plane with its pins and selection kept as the card keeps them, and every call recorded. */
function setup(initial: ColorWarpPin[] = [], options: { tool?: WarpTool; selected?: number | null } = {}) {
  const onChange = vi.fn<(pins: ColorWarpPin[], final: boolean) => void>();
  const onSelect = vi.fn<(index: number | null) => void>();
  function Host() {
    const [pins, setPins] = useState(initial);
    const [selected, setSelected] = useState<number | null>(options.selected ?? null);
    return (
      <WarpPlane
        pins={pins}
        selected={selected}
        tool={options.tool}
        onSelect={(index) => {
          onSelect(index);
          setSelected(index);
        }}
        onChange={(next, final) => {
          onChange(next, final);
          setPins(next);
        }}
      />
    );
  }
  const view = render(<Host />);
  const plane = screen.getByRole("group", { name: "Colour warp plane" });
  mockRect(plane, { left: 0, top: 0, width: SIZE, height: SIZE });
  const last = (): [ColorWarpPin[], boolean] => onChange.mock.calls.at(-1) as [ColorWarpPin[], boolean];
  return { ...view, plane, onChange, onSelect, last };
}

const pin = (n: number) => screen.getByRole("slider", { name: `Pin ${n}` });

describe("WarpPlane: drawing", () => {
  it("is a group of pins, each a slider named by its number and read as what it does", () => {
    setup([movingPin([0.1, 0], [0.1, 0.06]), lockedPin([-0.1, 0])]);
    expect(screen.getAllByRole("slider")).toHaveLength(2);
    expect(pin(1)).toHaveAttribute("aria-valuetext", "moves red toward orange");
    expect(pin(2)).toHaveAttribute("aria-valuetext", "holds cyan");
    expect(pin(1)).toHaveAttribute("tabindex", "0");
    // The value is the hue angle of the target: atan2(0.06, 0.1) = 31°.
    expect(pin(1)).toHaveAttribute("aria-valuenow", "31");
  });

  it("draws a moving pin as a ring at its source, a line and a dot at its target in the target's colour", () => {
    const { container } = setup([movingPin([0.1, 0], [0.1, 0.06])]);
    const source = container.querySelector(".ym-warp-plane__source") as SVGCircleElement;
    const target = container.querySelector(".ym-warp-plane__target") as SVGCircleElement;
    const line = container.querySelector(".ym-warp-plane__line") as SVGLineElement;
    const s = planeToPx([0.1, 0], SIZE);
    const t = planeToPx([0.1, 0.06], SIZE);
    expect([Number(source.getAttribute("cx")), Number(source.getAttribute("cy"))]).toEqual([s.x, s.y]);
    expect([Number(target.getAttribute("cx")), Number(target.getAttribute("cy"))]).toEqual([t.x, t.y]);
    expect([line.getAttribute("x1"), line.getAttribute("y2")]).toEqual([String(s.x), String(t.y)]);
    // 10 px across: a ring of radius 4 with a 2 px stroke, a dot of 4.25 with a 1.5 px stroke.
    expect(source.getAttribute("r")).toBe("4");
    expect(target.style.fill).toMatch(/^rgb\(/);
  });

  it("draws a lock as a ring with a small square, and the selected pin with a ring of its own", () => {
    const { container } = setup([lockedPin([0.1, 0]), movingPin([-0.1, 0], [-0.1, 0.05])], { selected: 1 });
    expect(container.querySelectorAll(".ym-warp-plane__lock")).toHaveLength(1);
    expect(container.querySelectorAll(".ym-warp-plane__lock-mark")).toHaveLength(1);
    expect(container.querySelectorAll(".ym-warp-plane__halo")).toHaveLength(1);
    expect(pin(2)).toHaveAttribute("data-selected", "true");
  });

  it("draws the rings at 0.1, 0.2 and 0.3, the last touching the middle of each edge", () => {
    const { container } = setup();
    const radii = [...container.querySelectorAll(".ym-warp-plane__ring")].map((c) => Number(c.getAttribute("r")));
    expect(radii[2]).toBeCloseTo(SIZE / 2, 9);
    expect(radii[0]).toBeCloseTo(SIZE / 6, 9);
  });
});

describe("WarpPlane: pointer", () => {
  it("adds a moving pin where Move presses on empty plane, selects it, and the drag moves its target: one patch per frame and a final one", () => {
    const { plane, onChange, onSelect, last } = setup();
    fireEvent.pointerDown(plane, { button: 0, ...at(0.1, 0) });
    expect(onSelect).toHaveBeenLastCalledWith(0);
    // The new pin goes out at once (not a final change), so the card's rows show with it.
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(last()[1]).toBe(false);
    expect(last()[0][0]?.source[0]).toBeCloseTo(0.1, 9);
    fireEvent.pointerMove(window, at(0.12, 0.04));
    fireEvent.pointerMove(window, at(0.1, 0.06));
    expect(onChange).toHaveBeenCalledTimes(1);
    raf.flush();
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(last()[1]).toBe(false);
    fireEvent.pointerUp(window);
    expect(onChange).toHaveBeenCalledTimes(3);
    const [pins, final] = last();
    expect(final).toBe(true);
    expect(pins).toHaveLength(1);
    expect(pins[0]?.locked).toBe(false);
    expect(pins[0]?.source[0]).toBeCloseTo(0.1, 9);
    expect(pins[0]?.source[1]).toBeCloseTo(0, 9);
    expect(pins[0]?.target[0]).toBeCloseTo(0.1, 9);
    expect(pins[0]?.target[1]).toBeCloseTo(0.06, 9);
    expect(pin(1)).toHaveFocus();
  });

  it("adds a locked pin with Lock, and a drag moves the whole lock", () => {
    const { plane, last } = setup([], { tool: "lock" });
    fireEvent.pointerDown(plane, { button: 0, ...at(-0.1, 0.05) });
    fireEvent.pointerMove(window, at(-0.12, 0.05));
    fireEvent.pointerUp(window);
    const [[lock], final] = last();
    expect(final).toBe(true);
    expect(lock?.locked).toBe(true);
    expect(lock?.source[0]).toBeCloseTo(-0.12, 9);
    expect(lock?.target).toEqual(lock?.source);
    expect(pin(1)).toHaveAttribute("aria-valuetext", "holds cyan");
  });

  it("selects a pin pressed on its dot or its ring, and moves its target from where it was taken, the source staying", () => {
    const { plane, onChange, onSelect, last } = setup([movingPin([0.1, 0], [0.1, 0.06]), movingPin([-0.1, 0])]);
    // On the source ring of pin 1: selected, and a press alone changes nothing.
    fireEvent.pointerDown(plane, { button: 0, ...at(0.1, 0) });
    fireEvent.pointerUp(window);
    expect(onSelect).toHaveBeenLastCalledWith(0);
    expect(onChange).not.toHaveBeenCalled();
    // Taken by its dot, 3 px off its centre, and dragged by 0.02 to the right: the target moves by 0.02, not to the pointer.
    const t = planeToPx([0.1, 0.06], SIZE);
    fireEvent.pointerDown(plane, { button: 0, clientX: t.x + 3, clientY: t.y });
    fireEvent.pointerMove(window, { clientX: t.x + 3 + 0.02 * (SIZE / 2 / 0.3), clientY: t.y });
    fireEvent.pointerUp(window);
    const [pins, final] = last();
    expect(final).toBe(true);
    expect(pins).toHaveLength(2);
    expect(pins[0]?.source).toEqual([0.1, 0]);
    expect(pins[0]?.target[0]).toBeCloseTo(0.12, 9);
    expect(pins[0]?.target[1]).toBeCloseTo(0.06, 9);
  });

  it("keeps a dragged target on the drawn square", () => {
    const { plane, last } = setup([movingPin([0.1, 0])]);
    fireEvent.pointerDown(plane, { button: 0, ...at(0.1, 0) });
    fireEvent.pointerMove(window, { clientX: 2000, clientY: -500 });
    fireEvent.pointerUp(window);
    expect(last()[0][0]?.target).toEqual([0.3, 0.3]);
  });

  it("puts a pin's target back on its source on a double click", () => {
    const { plane, last } = setup([movingPin([0.1, 0], [0.1, 0.06])]);
    fireEvent.doubleClick(plane, at(0.1, 0.06));
    expect(last()).toEqual([[movingPin([0.1, 0])], true]);
  });

  it("adds nothing at sixteen pins: a press on empty plane selects nothing", () => {
    const full = Array.from({ length: WARP_MAX_PINS }, (_, i) => movingPin([0.1 * Math.cos(i), 0.1 * Math.sin(i)]));
    const { plane, onChange, onSelect } = setup(full, { selected: 3 });
    fireEvent.pointerDown(plane, { button: 0, ...at(-0.25, -0.25) });
    fireEvent.pointerUp(window);
    expect(onChange).not.toHaveBeenCalled();
    expect(onSelect).toHaveBeenLastCalledWith(null);
    expect(screen.getAllByRole("slider")).toHaveLength(WARP_MAX_PINS);
  });

  it("ignores buttons other than the primary one", () => {
    const { plane, onChange } = setup();
    fireEvent.pointerDown(plane, { button: 2, ...at(0.1, 0) });
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("WarpPlane: keyboard", () => {
  it("selects the pin that takes the focus; Tab moves through the pins", async () => {
    const user = userEvent.setup({ advanceTimers: () => raf.flush() });
    const { onSelect } = setup([movingPin([0.1, 0]), movingPin([-0.1, 0])]);
    await user.tab();
    expect(pin(1)).toHaveFocus();
    expect(onSelect).toHaveBeenLastCalledWith(0);
    await user.tab();
    expect(pin(2)).toHaveFocus();
    expect(onSelect).toHaveBeenLastCalledWith(1);
  });

  it("moves the target by 0.002 with the arrows and ten times that with Shift, each a finished change; Home puts it back", async () => {
    const user = userEvent.setup();
    const { last } = setup([movingPin([0.1, 0])]);
    act(() => pin(1).focus());
    await user.keyboard("{ArrowRight}");
    expect(last()[1]).toBe(true);
    expect(last()[0][0]?.target[0]).toBeCloseTo(0.1 + WARP_KEY_STEP, 12);
    await user.keyboard("{Shift>}{ArrowUp}{/Shift}");
    expect(last()[0][0]?.target[1]).toBeCloseTo(10 * WARP_KEY_STEP, 12);
    await user.keyboard("{ArrowDown}{ArrowLeft}");
    expect(last()[0][0]?.target[0]).toBeCloseTo(0.1, 12);
    expect(last()[0][0]?.target[1]).toBeCloseTo(9 * WARP_KEY_STEP, 12);
    expect(last()[0][0]?.source).toEqual([0.1, 0]);
    await user.keyboard("{Home}");
    expect(last()).toEqual([[movingPin([0.1, 0])], true]);
  });

  it("moves a lock as a whole from the keyboard", async () => {
    const user = userEvent.setup();
    const { last } = setup([lockedPin([0.1, 0])]);
    act(() => pin(1).focus());
    await user.keyboard("{ArrowLeft}");
    const [[lock]] = last();
    expect(lock?.source[0]).toBeCloseTo(0.1 - WARP_KEY_STEP, 12);
    expect(lock?.target).toEqual(lock?.source);
  });

  it("removes the focused pin with Delete or Backspace and gives the focus to the one before", async () => {
    const user = userEvent.setup();
    const { last, onSelect } = setup([movingPin([0.1, 0]), movingPin([-0.1, 0]), lockedPin([0, 0.1])]);
    act(() => pin(2).focus());
    await user.keyboard("{Delete}");
    expect(last()).toEqual([[movingPin([0.1, 0]), lockedPin([0, 0.1])], true]);
    expect(onSelect).toHaveBeenLastCalledWith(0);
    expect(pin(1)).toHaveFocus();
    await user.keyboard("{Backspace}");
    expect(last()).toEqual([[lockedPin([0, 0.1])], true]);
    expect(pin(1)).toHaveFocus();
    await user.keyboard("{Backspace}");
    expect(last()).toEqual([[], true]);
    expect(onSelect).toHaveBeenLastCalledWith(null);
    expect(screen.getByRole("group", { name: "Colour warp plane" })).toHaveFocus();
  });

  it("leaves keys with ⌘, ⌃ or ⌥ alone", async () => {
    const user = userEvent.setup();
    const { onChange } = setup([movingPin([0.1, 0])]);
    act(() => pin(1).focus());
    await user.keyboard("{Meta>}{ArrowRight}{/Meta}{Alt>}{Backspace}{/Alt}");
    expect(onChange).not.toHaveBeenCalled();
  });
});
