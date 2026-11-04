import { afterEach, describe, expect, it, vi } from "vitest";
import { gestureHeld, holdGesture, onGestureEnd } from "./gestureHold";

afterEach(() => {
  vi.useRealTimers();
});

describe("holding a gesture", () => {
  it("is held from the first to the last, and tells the listener once, at the end of the last", () => {
    const ended = vi.fn();
    const stop = onGestureEnd(ended);
    expect(gestureHeld()).toBe(false);
    const a = holdGesture();
    const b = holdGesture();
    a();
    expect(gestureHeld()).toBe(true);
    expect(ended).not.toHaveBeenCalled();
    b();
    expect(gestureHeld()).toBe(false);
    expect(ended).toHaveBeenCalledTimes(1);
    // Ending twice changes nothing.
    b();
    a();
    expect(ended).toHaveBeenCalledTimes(1);
    stop();
    holdGesture()();
    expect(ended).toHaveBeenCalledTimes(1);
  });

  it("lets go by itself when the end never comes (a pointer released outside the window)", () => {
    vi.useFakeTimers();
    const ended = vi.fn();
    const stop = onGestureEnd(ended);
    holdGesture();
    expect(gestureHeld()).toBe(true);
    vi.advanceTimersByTime(16_000);
    expect(gestureHeld()).toBe(false);
    expect(ended).toHaveBeenCalledTimes(1);
    stop();
  });
});
