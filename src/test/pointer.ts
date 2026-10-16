import { fireEvent } from "@testing-library/react";

/** The pointer on a Radix slider (or any element that captures it): the calls a browser makes, one at a time. */
export interface CapturingPointer {
  /** Presses at `clientX` (with the primary button, or `button`: 1 middle, 2 secondary); the element captures the pointer, as Radix asks the browser to. */
  down: (clientX: number, button?: number) => void;
  /** Moves to `clientX`. Radix follows moves only while it holds the capture. */
  move: (clientX: number) => void;
  /** Lets go (at `clientX`, or where it was) and gives the capture up. */
  up: (clientX?: number) => void;
  /** The browser cancels the gesture and drops the capture. */
  cancel: () => void;
}

/**
 * Drives `element` (the Radix slider root, `[data-slot="slider"]`) like a pointer does. jsdom has no pointer capture, which Radix
 * checks before it follows a move, so the element keeps one: taken when something calls `setPointerCapture` on it (Radix does, on a
 * primary press), dropped by `releasePointerCapture`, `up` or `cancel`.
 * Give the element a box first (`mockRect`): Radix turns `clientX` into a value from it.
 */
export function capturingPointer(element: HTMLElement): CapturingPointer {
  let captured = false;
  // The capture is taken by whoever asks for it, as in a browser: Radix asks on a primary press, so a press that is ignored takes none.
  element.hasPointerCapture = () => captured;
  element.setPointerCapture = () => {
    captured = true;
  };
  element.releasePointerCapture = () => {
    captured = false;
  };
  const at = (clientX: number, button = 0) => ({ clientX, pointerId: 1, button, buttons: button === 0 ? 1 : button === 2 ? 2 : 4 });
  return {
    down: (clientX, button = 0) => {
      fireEvent.pointerDown(element, at(clientX, button));
    },
    move: (clientX) => {
      fireEvent.pointerMove(element, at(clientX));
    },
    up: (clientX) => {
      fireEvent.pointerUp(element, clientX === undefined ? { pointerId: 1 } : at(clientX));
      captured = false;
    },
    cancel: () => {
      captured = false;
      fireEvent.pointerCancel(element, { pointerId: 1 });
    },
  };
}
