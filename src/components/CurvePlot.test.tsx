import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installManualRaf, mockRect } from "../test/raf";
import type { ManualRaf } from "../test/raf";
import type { CurvePoint } from "./curve";
import { MAX_CURVE_POINTS } from "./curve";
import { CurvePlot } from "./CurvePlot";
import type { CurvePlotProps } from "./CurvePlot";

let raf: ManualRaf;
beforeEach(() => {
  raf = installManualRaf();
});
afterEach(() => raf.restore());

const IDENTITY: CurvePoint[] = [[0, 0], [1, 1]];
const THREE: CurvePoint[] = [[0, 0], [0.5, 0.5], [1, 1]];

function setup(props: Partial<CurvePlotProps> = {}) {
  const onChange = vi.fn();
  const view = render(<CurvePlot points={IDENTITY} onChange={onChange} {...props} />);
  const plot = screen.getByRole("group", { name: /curve/i });
  // The viewBox is 320 x 220 and the box is the same size, so client pixels are plot units.
  mockRect(plot, { left: 0, top: 0, width: 320, height: 220 });
  return { ...view, plot, onChange };
}

/** The last call's list of points. */
const lastPoints = (onChange: ReturnType<typeof vi.fn>): CurvePoint[] => (onChange.mock.calls.at(-1) as [CurvePoint[], boolean])[0];
const lastFinal = (onChange: ReturnType<typeof vi.fn>): boolean => (onChange.mock.calls.at(-1) as [CurvePoint[], boolean])[1];

describe("CurvePlot rendering", () => {
  it("is a group named after its channel", () => {
    setup({ channel: "red" });
    expect(screen.getByRole("group", { name: "Red curve" })).toBeInTheDocument();
  });

  it("draws the grid, the identity diagonal, the curve and one handle per point", () => {
    const { container } = setup({ points: THREE });
    expect(container.querySelector(".ym-curve__grid")).not.toBeNull();
    expect(container.querySelector(".ym-curve__identity")).not.toBeNull();
    expect(container.querySelector(".ym-curve__line")?.getAttribute("d")).toMatch(/^M0 220 C/);
    expect(screen.getAllByRole("slider")).toHaveLength(3);
  });

  it("colours the curve by channel", () => {
    const { container } = setup({ channel: "green" });
    expect((container.querySelector(".ym-curve__line") as SVGElement).style.stroke).toBe("var(--curve-g)");
  });

  it("exposes each point's input and output as code values", () => {
    setup({ points: [[0, 0], [0.5, 141 / 255], [1, 1]] });
    const middle = screen.getByRole("slider", { name: "Point 2 of 3" });
    expect(middle).toHaveAttribute("aria-valuenow", "141");
    expect(middle).toHaveAttribute("aria-valuetext", "Input 128, output 141");
  });

  it("draws the selected point filled and bigger", () => {
    setup({ points: THREE, defaultSelected: 1 });
    const middle = screen.getByRole("slider", { name: "Point 2 of 3" });
    expect(middle).toHaveAttribute("data-selected", "true");
    expect(middle).toHaveAttribute("width", "12");
    expect(screen.getByRole("slider", { name: "Point 1 of 3" })).toHaveAttribute("width", "10");
  });

  it("keeps the end handles fully inside the plot", () => {
    setup({ points: IDENTITY });
    const first = screen.getByRole("slider", { name: "Point 1 of 2" });
    const last = screen.getByRole("slider", { name: "Point 2 of 2" });
    expect(Number(first.getAttribute("x"))).toBeGreaterThanOrEqual(0);
    expect(Number(first.getAttribute("y")) + 10).toBeLessThanOrEqual(220);
    expect(Number(last.getAttribute("x")) + 10).toBeLessThanOrEqual(320);
    expect(Number(last.getAttribute("y"))).toBeGreaterThanOrEqual(0);
  });

  it("draws a histogram behind the grid when given one", () => {
    const { container, rerender } = setup({ histogram: [1, 3, 2, 5, 1] });
    expect(container.querySelector(".ym-curve__histogram")?.getAttribute("d")).toMatch(/^M0 220 L0 /);
    rerender(<CurvePlot points={IDENTITY} onChange={() => undefined} histogram={null} />);
    expect(container.querySelector(".ym-curve__histogram")).toBeNull();
  });

  it("reads Input and Output under the plot: the middle by default, the selected point, the pointer", () => {
    const { plot, container } = setup({ points: [[0, 0], [0.5, 141 / 255], [1, 1]] });
    const readout = container.querySelector(".ym-curve__readout") as HTMLElement;
    expect(readout.textContent).toBe("Input 128Output 141");
    fireEvent.pointerMove(plot, { clientX: 80, clientY: 100 });
    expect(readout.textContent).toMatch(/^Input 64Output /);
    fireEvent.pointerLeave(plot);
    expect(readout.textContent).toBe("Input 128Output 141");
  });

  it("can hide the readout", () => {
    const { container } = setup({ hideReadout: true });
    expect(container.querySelector(".ym-curve__readout")).toBeNull();
  });
});

describe("CurvePlot pointer editing", () => {
  it("adds a point on the curve where it is pressed, then drags it", () => {
    const { plot, onChange } = setup();
    fireEvent.pointerDown(plot, { button: 0, clientX: 160, clientY: 110 });
    raf.flush();
    expect(lastPoints(onChange)).toEqual([[0, 0], [0.5, 0.5], [1, 1]]);
    expect(lastFinal(onChange)).toBe(false);
    fireEvent.pointerMove(window, { clientX: 160, clientY: 55 });
    raf.flush();
    expect(lastPoints(onChange)[1]?.[0]).toBeCloseTo(0.5, 12);
    expect(lastPoints(onChange)[1]?.[1]).toBeCloseTo(0.75, 12);
    fireEvent.pointerUp(window);
    expect(lastFinal(onChange)).toBe(true);
    expect(lastPoints(onChange)).toHaveLength(3);
    expect(lastPoints(onChange)[1]?.[1]).toBeCloseTo(0.75, 12);
  });

  it("selects the point it adds", () => {
    const { plot } = setup();
    fireEvent.pointerDown(plot, { button: 0, clientX: 160, clientY: 110 });
    fireEvent.pointerUp(window);
    // The list is controlled and did not change, but the draft shows the new point.
    expect(screen.getByRole("slider", { name: "Point 2 of 3" })).toHaveAttribute("data-selected", "true");
  });

  it("grabs an existing point instead of adding one", () => {
    const { plot, onChange } = setup({ points: THREE });
    fireEvent.pointerDown(plot, { button: 0, clientX: 163, clientY: 106 }); // a few pixels from (0.5, 0.5)
    fireEvent.pointerMove(window, { clientX: 100, clientY: 150 });
    fireEvent.pointerUp(window);
    const points = lastPoints(onChange);
    expect(points).toHaveLength(3);
    expect(points[1]?.[0]).toBeCloseTo(100 / 320, 12);
    expect(points[1]?.[1]).toBeCloseTo(1 - 150 / 220, 12);
  });

  it("keeps an inner point between its neighbours", () => {
    const { plot, onChange } = setup({ points: [[0, 0], [0.3, 0.3], [0.6, 0.6], [1, 1]] });
    fireEvent.pointerDown(plot, { button: 0, clientX: 0.3 * 320, clientY: 0.7 * 220 });
    fireEvent.pointerMove(window, { clientX: 0.9 * 320, clientY: 0.7 * 220 });
    fireEvent.pointerUp(window);
    const points = lastPoints(onChange);
    expect(points[1]?.[0]).toBeLessThan(points[2]?.[0] as number);
  });

  it("moves an end point up and down only", () => {
    const { plot, onChange } = setup();
    fireEvent.pointerDown(plot, { button: 0, clientX: 2, clientY: 218 });
    fireEvent.pointerMove(window, { clientX: 200, clientY: 110 });
    fireEvent.pointerUp(window);
    const points = lastPoints(onChange);
    expect(points).toHaveLength(2);
    expect(points[0]?.[0]).toBe(0);
    expect(points[0]?.[1]).toBeCloseTo(0.5, 12);
  });

  it("deletes an inner point dragged off the plot", () => {
    const { plot, onChange } = setup({ points: THREE });
    fireEvent.pointerDown(plot, { button: 0, clientX: 160, clientY: 110 });
    fireEvent.pointerMove(window, { clientX: 160, clientY: 400 });
    raf.flush();
    expect(lastPoints(onChange)).toHaveLength(2); // the preview already shows it gone
    fireEvent.pointerUp(window);
    expect(lastFinal(onChange)).toBe(true);
    expect(lastPoints(onChange)).toEqual([[0, 0], [1, 1]]);
  });

  it("brings the point back when the pointer returns to the plot before release", () => {
    const { plot, onChange } = setup({ points: THREE });
    fireEvent.pointerDown(plot, { button: 0, clientX: 160, clientY: 110 });
    fireEvent.pointerMove(window, { clientX: 160, clientY: 400 });
    fireEvent.pointerMove(window, { clientX: 200, clientY: 80 });
    fireEvent.pointerUp(window);
    expect(lastPoints(onChange)).toHaveLength(3);
  });

  it("never deletes an end point by dragging it away", () => {
    const { plot, onChange } = setup({ points: THREE });
    fireEvent.pointerDown(plot, { button: 0, clientX: 2, clientY: 218 });
    fireEvent.pointerMove(window, { clientX: -200, clientY: 700 });
    fireEvent.pointerUp(window);
    expect(lastPoints(onChange)).toHaveLength(3);
  });

  it("adds no point once the curve has sixteen", () => {
    const full: CurvePoint[] = Array.from({ length: MAX_CURVE_POINTS }, (_, i) => [i / (MAX_CURVE_POINTS - 1), i / (MAX_CURVE_POINTS - 1)]);
    const { plot, onChange } = setup({ points: full });
    // Between the 8th and 9th points, far from both.
    fireEvent.pointerDown(plot, { button: 0, clientX: 7.5 / 15 * 320, clientY: 20 });
    fireEvent.pointerUp(window);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("sends one patch per frame while dragging", () => {
    const { plot, onChange } = setup({ points: THREE });
    fireEvent.pointerDown(plot, { button: 0, clientX: 160, clientY: 110 });
    for (let i = 0; i < 5; i++) fireEvent.pointerMove(window, { clientX: 160, clientY: 100 - i * 5 });
    raf.flush();
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(lastFinal(onChange)).toBe(false);
  });
});

describe("CurvePlot keyboard", () => {
  it("reaches each point with Tab and selects it", async () => {
    const user = userEvent.setup();
    setup({ points: THREE });
    await user.tab(); // the plot
    await user.tab();
    expect(screen.getByRole("slider", { name: "Point 1 of 3" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("slider", { name: "Point 2 of 3" })).toHaveFocus();
    expect(screen.getByRole("slider", { name: "Point 2 of 3" })).toHaveAttribute("data-selected", "true");
  });

  it("moves a point by one code value with the arrows, ten with Shift", async () => {
    const user = userEvent.setup();
    const { onChange } = setup({ points: THREE });
    screen.getByRole("slider", { name: "Point 2 of 3" }).focus();
    await user.keyboard("{ArrowUp}");
    expect(lastPoints(onChange)[1]?.[1]).toBeCloseTo(0.5 + 1 / 255, 12);
    expect(lastFinal(onChange)).toBe(true);
    await user.keyboard("{Shift>}{ArrowLeft}{/Shift}");
    expect(lastPoints(onChange)[1]?.[0]).toBeCloseTo(0.5 - 10 / 255, 12);
  });

  it("deletes an inner point with Delete and never an end", async () => {
    const user = userEvent.setup();
    const { onChange } = setup({ points: THREE });
    screen.getByRole("slider", { name: "Point 1 of 3" }).focus();
    await user.keyboard("{Delete}");
    expect(onChange).not.toHaveBeenCalled();
    screen.getByRole("slider", { name: "Point 2 of 3" }).focus();
    await user.keyboard("{Backspace}");
    expect(lastPoints(onChange)).toEqual([[0, 0], [1, 1]]);
  });

  it("adds a point in the widest gap with Return on the plot", async () => {
    const user = userEvent.setup();
    const { onChange } = setup({ points: IDENTITY });
    await user.tab();
    expect(screen.getByRole("group", { name: /curve/i })).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(lastPoints(onChange)).toEqual([[0, 0], [0.5, 0.5], [1, 1]]);
    expect(lastFinal(onChange)).toBe(true);
  });

  it("leaves modified keys alone", async () => {
    const user = userEvent.setup();
    const { onChange } = setup({ points: THREE });
    screen.getByRole("slider", { name: "Point 2 of 3" }).focus();
    await user.keyboard("{Meta>}{ArrowUp}{/Meta}");
    expect(onChange).not.toHaveBeenCalled();
  });
});
