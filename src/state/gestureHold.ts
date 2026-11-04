/**
 * Which hand gestures are on (U-02). While a slider's knob, a wheel, a curve point or a warp pin is held, the extras that follow the
 * picture (the histogram, the middle readout, the colour plane's scatter) are not asked for: each is a job in the core's one queue and a
 * redraw of the page, five times a second, in the middle of the drag. The controls call {@link holdGesture} when the pointer goes
 * down and the function it returns when the gesture ends; the store asks once more when the last one ends.
 */
/** The longest a gesture holds the extras back. */
const GUARD_MS = 15_000;
let held = 0;
const listeners = new Set<() => void>();

/** A gesture begins. Returns the function that ends it (calling it twice is harmless). */
export function holdGesture(): () => void {
  held += 1;
  let live = true;
  // A pointer that is released outside the window can leave a gesture without its end; the extras come back after a while regardless.
  const guard = setTimeout(() => end(), GUARD_MS);
  const end = (): void => {
    if (!live) return;
    live = false;
    clearTimeout(guard);
    held -= 1;
    if (held === 0) for (const listener of [...listeners]) listener();
  };
  return end;
}

/** Whether a gesture is on. */
export function gestureHeld(): boolean {
  return held > 0;
}

/** Calls `listener` whenever the last gesture ends. Returns the function that stops listening. */
export function onGestureEnd(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
