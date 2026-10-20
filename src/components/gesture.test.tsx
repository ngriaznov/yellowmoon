/**
 * A drag collects its input over time (one value per frame, then a final one on release). The control it is made on may be given
 * another `onChange` meanwhile: the Photos screen draws the same controls for whichever photo is on screen, and the callback of the new
 * render belongs to that photo. A gesture reports to the callback it began with (`useGestureCallback`), so what the person dragged on
 * one photo is never sent to the next; a key or a typed value, which are handled at once, go to the newest callback.
 */
import { fireEvent, render, renderHook, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ColorWarpPin } from "../api/types";
import { capturingPointer } from "../test/pointer";
import { installManualRaf, mockRect } from "../test/raf";
import type { ManualRaf } from "../test/raf";
import type { CurvePoint } from "./curve";
import { CurvePlot } from "./CurvePlot";
import { useGestureCallback } from "./hooks";
import { Slider } from "./Slider";
import { Wheel } from "./Wheel";
import { WarpPlane } from "./WarpPlane";
import { planeToPx } from "./warpMath";

let raf: ManualRaf;
beforeEach(() => {
  raf = installManualRaf();
});
afterEach(() => raf.restore());

describe("useGestureCallback", () => {
  it("is the newest callback outside a gesture and the one it began with inside it", () => {
    const first = vi.fn();
    const second = vi.fn();
    const { result, rerender } = renderHook(({ callback }) => useGestureCallback(callback), { initialProps: { callback: first } });
    expect(result.current.current).toBe(first);
    result.current.begin();
    rerender({ callback: second });
    expect(result.current.current).toBe(first);
    result.current.end();
    expect(result.current.current).toBe(second);
  });
});

describe("a drag on the controls of the tool panel", () => {
  it("a Slider's frames and release go to the callback it began with", () => {
    const photo1 = vi.fn();
    const photo2 = vi.fn();
    const view = render(<Slider label="Exposure" value={0} min={-4} max={4} step={0.01} defaultValue={0} onChange={photo1} />);
    const root = view.container.querySelector('[data-slot="slider"]') as HTMLElement;
    mockRect(root, { left: 100, top: 0, width: 200, height: 38 });
    const pointer = capturingPointer(root);
    pointer.down(200);
    pointer.move(230);
    // The next photo is on screen before the frame comes: the same slider, another callback and another value.
    view.rerender(<Slider label="Exposure" value={1} min={-4} max={4} step={0.01} defaultValue={0} onChange={photo2} />);
    pointer.move(250);
    raf.flush();
    pointer.up(250);
    expect(photo2).not.toHaveBeenCalled();
    expect(photo1.mock.calls.map((c) => c[1])).toEqual([false, true]);
  });

  it("a Slider goes back to the newest callback when the drag is over", async () => {
    const photo1 = vi.fn();
    const photo2 = vi.fn();
    const view = render(<Slider label="Exposure" value={0} min={-4} max={4} step={0.01} defaultValue={0} onChange={photo1} />);
    const root = view.container.querySelector('[data-slot="slider"]') as HTMLElement;
    mockRect(root, { left: 100, top: 0, width: 200, height: 38 });
    const pointer = capturingPointer(root);
    pointer.down(200);
    pointer.move(230);
    pointer.up(230);
    view.rerender(<Slider label="Exposure" value={0} min={-4} max={4} step={0.01} defaultValue={0} onChange={photo2} />);
    const box = screen.getByRole("textbox", { name: "Exposure value" });
    await userEvent.setup().click(box);
    await userEvent.setup().keyboard("1{Enter}");
    expect(photo2).toHaveBeenCalledWith(1, true);
  });

  it("a Wheel's frames and release go to the callback it began with", () => {
    const photo1 = vi.fn();
    const photo2 = vi.fn();
    const view = render(<Wheel label="Shadows" value={[0, 0]} onChange={photo1} />);
    const disc = screen.getByRole("slider", { name: "Shadows colour balance" });
    mockRect(disc, { left: 0, top: 0, width: 92, height: 92 });
    fireEvent.pointerDown(disc, { button: 0, clientX: 46 + 10, clientY: 46 });
    fireEvent.pointerMove(window, { clientX: 46 + 20, clientY: 46 });
    view.rerender(<Wheel label="Shadows" value={[0.5, 0]} onChange={photo2} />);
    fireEvent.pointerMove(window, { clientX: 46 + 30, clientY: 46 });
    raf.flush();
    fireEvent.pointerUp(window);
    expect(photo2).not.toHaveBeenCalled();
    expect(photo1.mock.calls.map((c) => c[1])).toEqual([false, true]);
  });

  it("a CurvePlot's frames and release go to the callback it began with", () => {
    const photo1 = vi.fn();
    const photo2 = vi.fn();
    const identity: CurvePoint[] = [
      [0, 0],
      [1, 1],
    ];
    const view = render(<CurvePlot points={identity} onChange={photo1} />);
    const plot = screen.getByRole("group", { name: /curve/i });
    mockRect(plot, { left: 0, top: 0, width: 320, height: 220 });
    fireEvent.pointerDown(plot, { button: 0, clientX: 160, clientY: 110 });
    fireEvent.pointerMove(window, { clientX: 170, clientY: 90 });
    view.rerender(<CurvePlot points={identity} onChange={photo2} />);
    fireEvent.pointerMove(window, { clientX: 180, clientY: 80 });
    raf.flush();
    fireEvent.pointerUp(window);
    expect(photo2).not.toHaveBeenCalled();
    expect(photo1.mock.calls.at(-1)?.[1]).toBe(true);
    expect(photo1.mock.calls.slice(0, -1).every((c) => c[1] === false)).toBe(true);
  });

  it("a WarpPlane's pin, frames and release go to the callback it began with", () => {
    const photo1 = vi.fn();
    const photo2 = vi.fn();
    const SIZE = 352;
    const at = (a: number, b: number): { clientX: number; clientY: number } => {
      const p = planeToPx([a, b], SIZE);
      return { clientX: p.x, clientY: p.y };
    };
    const none: ColorWarpPin[] = [];
    const view = render(<WarpPlane pins={none} selected={null} onSelect={() => undefined} onChange={photo1} />);
    const plane = screen.getByRole("group", { name: "Colour warp plane" });
    mockRect(plane, { left: 0, top: 0, width: SIZE, height: SIZE });
    fireEvent.pointerDown(plane, { button: 0, ...at(0.1, 0) });
    fireEvent.pointerMove(window, at(0.12, 0.04));
    view.rerender(<WarpPlane pins={none} selected={null} onSelect={() => undefined} onChange={photo2} />);
    fireEvent.pointerMove(window, at(0.1, 0.06));
    raf.flush();
    fireEvent.pointerUp(window);
    expect(photo2).not.toHaveBeenCalled();
    expect(photo1.mock.calls.at(-1)?.[1]).toBe(true);
    expect(photo1.mock.calls.slice(0, -1).every((c) => c[1] === false)).toBe(true);
  });
});
