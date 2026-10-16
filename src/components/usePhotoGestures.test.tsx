import { render } from "@testing-library/react";
import { useRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { usePhotoGestures } from "./usePhotoGestures";
import type { PhotoGestureHandlers } from "./usePhotoGestures";

function Cell({ handlers, origin }: { handlers?: PhotoGestureHandlers; origin?: () => { left: number; top: number } }) {
  const ref = useRef<HTMLDivElement>(null);
  usePhotoGestures(ref, handlers, origin);
  return <div ref={ref} data-testid="cell" />;
}

function handlers(over: Partial<PhotoGestureHandlers> = {}): PhotoGestureHandlers & { pan: ReturnType<typeof vi.fn>; zoom: ReturnType<typeof vi.fn> } {
  return { enabled: () => true, canPan: () => true, pan: vi.fn(), zoom: vi.fn(), ...over } as PhotoGestureHandlers & { pan: ReturnType<typeof vi.fn>; zoom: ReturnType<typeof vi.fn> };
}

const wheel = (init: WheelEventInit = {}): WheelEvent => new WheelEvent("wheel", { bubbles: true, cancelable: true, ...init });

/** WebKit's pinch events: a plain event with the fields the browser gives it. */
function gesture(type: "gesturestart" | "gesturechange" | "gestureend", fields: { scale?: number; clientX?: number; clientY?: number } = {}): Event {
  return Object.assign(new Event(type, { bubbles: true, cancelable: true }), fields);
}

describe("usePhotoGestures", () => {
  it("claims every wheel event over the cell, so the page never scrolls or bounces under the photo", () => {
    const h = handlers({ canPan: () => false });
    const { getByTestId } = render(<Cell handlers={h} />);
    const cell = getByTestId("cell");
    // At Fit there is nothing to pan, and the wheel is claimed all the same.
    const plain = wheel({ deltaY: 12 });
    cell.dispatchEvent(plain);
    expect(plain.defaultPrevented).toBe(true);
    expect(h.pan).not.toHaveBeenCalled();
    const zoom = wheel({ deltaY: 12, ctrlKey: true });
    cell.dispatchEvent(zoom);
    expect(zoom.defaultPrevented).toBe(true);
  });

  it("pans by the distance the fingers moved when the photo is larger than the cell", () => {
    const h = handlers();
    const { getByTestId } = render(<Cell handlers={h} />);
    getByTestId("cell").dispatchEvent(wheel({ deltaX: 3, deltaY: -12 }));
    expect(h.pan).toHaveBeenCalledWith(-3, 12);
    expect(h.zoom).not.toHaveBeenCalled();
  });

  it("zooms with ⌘ or ctrl and the wheel, about the pointer, by exp(-deltaY / 100)", () => {
    const h = handlers();
    const { getByTestId } = render(<Cell handlers={h} origin={() => ({ left: 10, top: 20 })} />);
    getByTestId("cell").dispatchEvent(wheel({ deltaY: -100, metaKey: true, clientX: 110, clientY: 70 }));
    expect(h.zoom).toHaveBeenCalledTimes(1);
    const [factor, x, y] = h.zoom.mock.calls[0] as [number, number, number];
    expect(factor).toBeCloseTo(Math.exp(1), 9);
    expect([x, y]).toEqual([100, 50]);
    getByTestId("cell").dispatchEvent(wheel({ deltaY: 50, ctrlKey: true, clientX: 10, clientY: 20 }));
    expect(h.zoom.mock.calls[1]?.[0]).toBeCloseTo(Math.exp(-0.5), 9);
  });

  it("claims the wheel but does nothing while Crop or Pick white has the cell", () => {
    const h = handlers({ enabled: () => false });
    const { getByTestId } = render(<Cell handlers={h} />);
    const event = wheel({ deltaY: 30, ctrlKey: true });
    getByTestId("cell").dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(h.pan).not.toHaveBeenCalled();
    expect(h.zoom).not.toHaveBeenCalled();
  });

  it("zooms about the fingers with WebKit's pinch: each change is the step since the one before", () => {
    const h = handlers();
    const { getByTestId } = render(<Cell handlers={h} origin={() => ({ left: 0, top: 0 })} />);
    const cell = getByTestId("cell");
    const start = gesture("gesturestart");
    cell.dispatchEvent(start);
    expect(start.defaultPrevented).toBe(true);
    const first = gesture("gesturechange", { scale: 1.2, clientX: 40, clientY: 30 });
    cell.dispatchEvent(first);
    expect(first.defaultPrevented).toBe(true);
    cell.dispatchEvent(gesture("gesturechange", { scale: 1.5, clientX: 42, clientY: 31 }));
    expect(h.zoom).toHaveBeenCalledTimes(2);
    expect(h.zoom.mock.calls[0]).toEqual([1.2, 40, 30]);
    expect(h.zoom.mock.calls[1]?.[0]).toBeCloseTo(1.25, 12);
    expect(h.zoom.mock.calls[1]?.slice(1)).toEqual([42, 31]);
    const end = gesture("gestureend");
    cell.dispatchEvent(end);
    expect(end.defaultPrevented).toBe(true);
    // A new pinch starts from 1 again.
    cell.dispatchEvent(gesture("gesturestart"));
    cell.dispatchEvent(gesture("gesturechange", { scale: 0.8, clientX: 1, clientY: 2 }));
    expect(h.zoom.mock.calls[2]).toEqual([0.8, 1, 2]);
  });

  it("claims a pinch but ignores it while the cell is not the photo's, and ignores a scale that makes no sense", () => {
    let on = false;
    const h = handlers({ enabled: () => on });
    const { getByTestId } = render(<Cell handlers={h} />);
    const cell = getByTestId("cell");
    const event = gesture("gesturechange", { scale: 2, clientX: 1, clientY: 1 });
    cell.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(h.zoom).not.toHaveBeenCalled();
    on = true;
    cell.dispatchEvent(gesture("gesturestart"));
    cell.dispatchEvent(gesture("gesturechange", { scale: Number.NaN, clientX: 1, clientY: 1 }));
    cell.dispatchEvent(gesture("gesturechange", { scale: 0, clientX: 1, clientY: 1 }));
    expect(h.zoom).not.toHaveBeenCalled();
  });

  it("uses the newest handlers without attaching again", () => {
    const first = handlers();
    const second = handlers();
    const { getByTestId, rerender } = render(<Cell handlers={first} />);
    rerender(<Cell handlers={second} />);
    getByTestId("cell").dispatchEvent(wheel({ deltaY: 5 }));
    expect(first.pan).not.toHaveBeenCalled();
    expect(second.pan).toHaveBeenCalledWith(-0, -5);
  });

  it("leaves the wheel to the page when it has no handlers (a cell on the dev page), and lets go on unmount", () => {
    const { getByTestId, rerender, unmount } = render(<Cell />);
    const cell = getByTestId("cell");
    const free = wheel({ deltaY: 5 });
    cell.dispatchEvent(free);
    expect(free.defaultPrevented).toBe(false);
    rerender(<Cell handlers={handlers()} />);
    const claimed = wheel({ deltaY: 5 });
    cell.dispatchEvent(claimed);
    expect(claimed.defaultPrevented).toBe(true);
    unmount();
    const after = wheel({ deltaY: 5 });
    cell.dispatchEvent(after);
    expect(after.defaultPrevented).toBe(false);
  });
});
