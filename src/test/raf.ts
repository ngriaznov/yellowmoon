import { vi } from "vitest";

/** A hand-driven `requestAnimationFrame`, so tests decide when a frame happens. */
export interface ManualRaf {
  /** Runs every callback that is waiting, as one frame. */
  flush: () => void;
  /** How many callbacks are waiting. */
  readonly pending: number;
  /** Puts the real `requestAnimationFrame` back. */
  restore: () => void;
}

export function installManualRaf(): ManualRaf {
  const queue = new Map<number, FrameRequestCallback>();
  let next = 0;
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback): number => {
    next += 1;
    queue.set(next, callback);
    return next;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number): void => {
    queue.delete(id);
  });
  return {
    flush() {
      const callbacks = [...queue.values()];
      queue.clear();
      for (const callback of callbacks) callback(0);
    },
    get pending() {
      return queue.size;
    },
    restore() {
      vi.unstubAllGlobals();
    },
  };
}

/** Gives an element a box, since jsdom lays nothing out. */
export function mockRect(element: Element, rect: { left?: number; top?: number; width: number; height: number }): void {
  const { left = 0, top = 0, width, height } = rect;
  element.getBoundingClientRect = () =>
    ({ left, top, width, height, right: left + width, bottom: top + height, x: left, y: top, toJSON: () => ({}) }) as DOMRect;
}
