import { useEffect } from "react";
import type { RefObject } from "react";
import { useLatest } from "./hooks";

/** WebKit's pinch events; they are not in `lib.dom.d.ts`. `scale` is relative to `gesturestart`. */
interface WebKitGestureEvent extends UIEvent {
  readonly scale: number;
  readonly clientX: number;
  readonly clientY: number;
}

/** What the photo cell's wheel and pinch do. Coordinates are CSS pixels from the cell's corner. */
export interface PhotoGestureHandlers {
  /** False while Crop or Pick white has the cell: the gesture is still claimed (nothing may scroll under the photo) but does nothing. */
  enabled: () => boolean;
  /** True when the photo is larger than the cell, so there is something to pan. */
  canPan: () => boolean;
  /** The picture follows the fingers: the distance in CSS pixels. */
  pan: (dx: number, dy: number) => void;
  /** Multiply the zoom by `factor` about the point (x, y) of the cell. */
  zoom: (factor: number, x: number, y: number) => void;
}

/** How much one wheel step of a ⌘ or ctrl scroll zooms: the factor is `exp(-deltaY * WHEEL_ZOOM_RATE)`. */
export const WHEEL_ZOOM_RATE = 0.01;

/**
 * The photo cell's wheel and pinch, as native listeners (`docs/11` §Photos input).
 *
 * React registers `wheel` as a passive listener on the root (react-dom's `addTrappedEventListener`: touchstart, touchmove and wheel),
 * so an `onWheel` prop can never call `preventDefault`, and the default action of a two-finger scroll that nothing scrolls is the web
 * view's elastic bounce of the whole page. This listener is not passive and claims every wheel event over the cell, in Crop and Pick
 * white and at Fit too: the gesture is the photo's first, and only then does it decide whether to act. ⌘ or ctrl with the wheel zooms
 * about the pointer; the wheel alone pans when the photo is larger than the cell.
 *
 * WebKit sends a trackpad pinch as `gesturestart`, `gesturechange` and `gestureend` (with a `scale` since the start), not as a
 * ctrl+wheel as Chromium does; the pinch zooms about the fingers. No listener is attached on `document`: a non-passive wheel listener
 * there would make the scrolling thread ask the busy web process before each scroll step over the whole window, the tool panel
 * included.
 *
 * Does nothing while `handlers` is undefined (a cell on the dev page keeps the page's scrolling). `origin` says where the cell is
 * (its corner in the window), by default read from the element on each event.
 */
export function usePhotoGestures(
  cell: RefObject<HTMLElement | null>,
  handlers: PhotoGestureHandlers | undefined,
  origin?: () => { left: number; top: number },
): void {
  const latest = useLatest(handlers);
  const originRef = useLatest(origin);
  const active = handlers !== undefined;
  useEffect(() => {
    const el = cell.current;
    if (!el || !active) return;
    const at = (clientX: number, clientY: number): { x: number; y: number } => {
      const box = originRef.current?.() ?? el.getBoundingClientRect();
      return { x: clientX - box.left, y: clientY - box.top };
    };
    const onWheel = (event: WheelEvent): void => {
      event.preventDefault();
      const h = latest.current;
      if (!h?.enabled()) return;
      if (event.ctrlKey || event.metaKey) {
        const p = at(event.clientX, event.clientY);
        h.zoom(Math.exp(-event.deltaY * WHEEL_ZOOM_RATE), p.x, p.y);
      } else if (h.canPan()) {
        h.pan(-event.deltaX, -event.deltaY);
      }
    };
    let previous = 1;
    const onGestureStart = (event: Event): void => {
      event.preventDefault();
      previous = 1;
    };
    const onGestureChange = (event: Event): void => {
      event.preventDefault();
      const g = event as WebKitGestureEvent;
      const factor = g.scale / previous;
      previous = g.scale;
      const h = latest.current;
      if (!h?.enabled() || !(factor > 0) || !Number.isFinite(factor)) return;
      const p = at(g.clientX, g.clientY);
      h.zoom(factor, p.x, p.y);
    };
    const onGestureEnd = (event: Event): void => event.preventDefault();
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("gesturestart", onGestureStart);
    el.addEventListener("gesturechange", onGestureChange);
    el.addEventListener("gestureend", onGestureEnd);
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("gesturestart", onGestureStart);
      el.removeEventListener("gesturechange", onGestureChange);
      el.removeEventListener("gestureend", onGestureEnd);
    };
  }, [cell, active, latest, originRef]);
}
