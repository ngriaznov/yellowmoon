import { useEffect, useRef, useState } from "react";

/**
 * How long the photo's settings must rest before the tiles are made again: a person moving a slider on this tab (the Tone card is
 * here) is not asking to see every look through each step, and each remake is a job for the core's one foreground worker.
 */
export const THUMBNAIL_SETTLE_MS = 400;

/** Waits `ms` after a value stops changing before passing it on; the first value passes at once. */
export function useSettled<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const timer = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return settled;
}

/**
 * Tells the tiles of the tab when they are in view, with one `IntersectionObserver` for all of them (a tab can hold hundreds of
 * tiles; one observer each was a few hundred objects the page had to keep). Without `IntersectionObserver` (a test) every tile is
 * always in view.
 */
export interface TileWatch {
  /** Calls `callback(true)` while `el` is in view and `callback(false)` while it is not, both ways. Returns the function that stops. */
  observe: (el: Element, callback: (inView: boolean) => void) => () => void;
  /** Stops watching everything; `observe` starts again if asked. */
  disconnect: () => void;
}

/** Makes a {@link TileWatch}. */
export function createTileWatch(): TileWatch {
  const callbacks = new Map<Element, (inView: boolean) => void>();
  let observer: IntersectionObserver | null = null;
  return {
    observe(el, callback) {
      if (typeof IntersectionObserver === "undefined") {
        callback(true);
        return () => undefined;
      }
      observer ??= new IntersectionObserver((entries) => {
        for (const entry of entries) callbacks.get(entry.target)?.(entry.isIntersecting);
      });
      callbacks.set(el, callback);
      observer.observe(el);
      return () => {
        callbacks.delete(el);
        observer?.unobserve(el);
      };
    },
    disconnect() {
      observer?.disconnect();
      observer = null;
      callbacks.clear();
    },
  };
}

