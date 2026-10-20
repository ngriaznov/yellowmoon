/**
 * Small hooks shared by the controls: a latest-value ref, a one-per-frame emitter for drag
 * patches, and a window-level pointer drag.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

/** A ref that always holds the newest value, for callbacks that outlive a render. */
export function useLatest<T>(value: T): { readonly current: T } {
  const ref = useRef(value);
  useEffect(() => {
    ref.current = value;
  });
  return ref;
}

/** What `useGestureCallback` returns. */
export interface GestureCallback<F> {
  /** The callback to call now: the one the running gesture began with, else the newest. */
  readonly current: F;
  /** A drag, or any input collected over time, begins: it reports to the callback of the render it began in until {@link end}. */
  begin: () => void;
  /** The gesture is over (released, cancelled): the newest callback is the one to call again. */
  end: () => void;
}

/**
 * The callback a control reports its changes to. A control that collects input (a drag sends one value per frame, then a final one)
 * calls `onChange` some time after the input that made the value; the `onChange` of the newest render may by then belong to another
 * photo (the person moved on between the pointer going down and the frame, or the release), and the value would be sent there. So a
 * gesture holds the callback it began with: `begin()` when it starts, `end()` when it is over, `current` in between. Outside a gesture
 * (a key, a typed value, Reset) `current` is the newest callback, which is the one for the input being handled.
 */
export function useGestureCallback<F>(callback: F): GestureCallback<F> {
  const latest = useLatest(callback);
  const held = useRef<{ callback: F } | null>(null);
  return useMemo(
    () => ({
      get current() {
        return held.current ? held.current.callback : latest.current;
      },
      begin() {
        held.current = { callback: latest.current };
      },
      end() {
        held.current = null;
      },
    }),
    [latest],
  );
}

/** What `useFrameEmitter` returns. */
export interface FrameEmitter<T> {
  /** Remembers `value` and emits it on the next animation frame, once however many are pushed. */
  push: (value: T) => void;
  /** Drops a value that is waiting for its frame. */
  cancel: () => void;
}

/**
 * Throttles a stream of drag values to one emit per animation frame (`docs/11` §Frontend
 * structure: "one patch per animation frame and a final patch on release"). Only the newest
 * pushed value is emitted. The emitter always calls the latest `emit` and cancels on unmount.
 */
export function useFrameEmitter<T>(emit: (value: T) => void): FrameEmitter<T> {
  const emitRef = useLatest(emit);
  const frame = useRef<number | null>(null);
  const pending = useRef<{ value: T } | null>(null);

  const cancel = useCallback(() => {
    if (frame.current !== null) {
      cancelAnimationFrame(frame.current);
      frame.current = null;
    }
    pending.current = null;
  }, []);

  const push = useCallback(
    (value: T) => {
      pending.current = { value };
      if (frame.current !== null) return;
      frame.current = requestAnimationFrame(() => {
        frame.current = null;
        const next = pending.current;
        pending.current = null;
        if (next) emitRef.current(next.value);
      });
    },
    [emitRef],
  );

  useEffect(() => cancel, [cancel]);
  return { push, cancel };
}

/** Callbacks of a pointer drag. */
export interface PointerDragHandlers {
  /** Pointer went down. Return false to refuse the drag. */
  start: (event: ReactPointerEvent<Element>) => boolean | void;
  /** Pointer moved while the drag is on (listens on the window, so it follows outside the element). */
  move: (event: PointerEvent) => void;
  /** Pointer went up or the drag was cancelled. */
  end: (event: PointerEvent | null, cancelled: boolean) => void;
}

/**
 * A drag that listens on the window after `pointerdown`, so it keeps following the pointer
 * outside the element and needs no pointer capture. Returns the `onPointerDown` handler.
 * Only the primary button starts a drag.
 */
export function usePointerDrag(handlers: PointerDragHandlers): (event: ReactPointerEvent<Element>) => void {
  const handlersRef = useLatest(handlers);
  const stop = useRef<(() => void) | null>(null);

  useEffect(
    () => () => {
      stop.current?.();
    },
    [],
  );

  return useCallback(
    (event: ReactPointerEvent<Element>) => {
      if (event.button !== 0) return;
      stop.current?.();
      if (handlersRef.current.start(event) === false) return;
      const onMove = (e: PointerEvent): void => handlersRef.current.move(e);
      const finish = (e: PointerEvent | null, cancelled: boolean): void => {
        stop.current?.();
        handlersRef.current.end(e, cancelled);
      };
      const onUp = (e: PointerEvent): void => finish(e, false);
      const onCancel = (e: PointerEvent): void => finish(e, true);
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onCancel);
      stop.current = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        window.removeEventListener("pointercancel", onCancel);
        stop.current = null;
      };
    },
    [handlersRef],
  );
}

/** What `useDraft` returns. */
export interface Draft<T> {
  /** The value to draw: the draft while one is held, else the controlled value. */
  shown: T;
  /** A drag is on: draw `next` until `settle` is called. */
  drag: (next: T) => void;
  /**
   * The change is finished (pointer released, key pressed, value typed): keep drawing `next`
   * until the controlled value catches up with it, or until `SETTLE_MS` passes (the core
   * refused it).
   */
  settle: (next: T) => void;
  /** Drops the draft now. */
  clear: () => void;
  /** True while a drag is on. */
  readonly dragging: boolean;
}

/** How long a finished change is held before the controlled value takes over again. */
export const SETTLE_MS = 1000;

/**
 * The state behind a control that mirrors a value kept elsewhere (in Rust): while the person
 * drags, or until the patch comes back, the control draws its own value so a thumb never lags
 * behind the pointer or jumps back when the first reply arrives late.
 */
export function useDraft<T>(value: T, same: (a: T, b: T) => boolean): Draft<T> {
  const [draft, setDraft] = useState<{ v: T } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragging = useRef(false);
  const sameRef = useLatest(same);
  const valueRef = useLatest(value);

  const stopTimer = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
  }, []);
  useEffect(() => stopTimer, [stopTimer]);

  useEffect(() => {
    if (dragging.current) return;
    setDraft((current) => (current !== null && sameRef.current(current.v, value) ? null : current));
  }, [value, sameRef]);

  const clear = useCallback(() => {
    stopTimer();
    dragging.current = false;
    setDraft(null);
  }, [stopTimer]);

  const drag = useCallback(
    (next: T) => {
      stopTimer();
      dragging.current = true;
      setDraft({ v: next });
    },
    [stopTimer],
  );

  const settle = useCallback(
    (next: T) => {
      stopTimer();
      dragging.current = false;
      if (sameRef.current(next, valueRef.current)) {
        setDraft(null);
        return;
      }
      setDraft({ v: next });
      timer.current = setTimeout(() => {
        timer.current = null;
        setDraft(null);
      }, SETTLE_MS);
    },
    [stopTimer, sameRef, valueRef],
  );

  return { shown: draft ? draft.v : value, drag, settle, clear, dragging: dragging.current };
}
