import { useCallback, useLayoutEffect, useRef, useState } from "react";
import type { RefObject } from "react";

/**
 * A small windowing hook (`docs/26` H-09, `docs/11` §Large folders): a list of `count` equal items draws only the ones in view and a
 * margin, so the page keeps the same number of DOM nodes whether the folder holds twelve photos or two thousand. There is no wrapper
 * and no absolute positioning: the items that are not drawn are stood in for by space, the first drawn item's leading margin (a
 * sideways strip) or the list's padding (a column), so the scroll bar is true and the items keep their normal layout.
 *
 * The item size is measured from the drawn items (the distance between the first two, gap included), so the CSS stays the only
 * place that says how big a row is. Lists with fewer than `min` items are drawn whole, as before the hook existed.
 */

/** Below this many items a list is drawn whole: nothing to save, and the layout is exactly the one without windowing. */
export const WINDOW_MIN = 100;

/** What {@link useWindowedRange} needs. */
export interface WindowedOptions {
  /** How many items the list has. */
  count: number;
  /** `x` for a strip that scrolls sideways, `y` for a column. */
  axis: "x" | "y";
  /** The element that holds the items, in order and without anything between them. */
  list: RefObject<HTMLElement | null>;
  /** The element that scrolls, for a strip: the list itself. For a column the nearest scrolling ancestor of the list is used. */
  scroller?: RefObject<HTMLElement | null>;
  /** CSS selector of one item, for measuring. */
  itemSelector: string;
  /** The item size, gap included, until the first items have been measured. */
  estimate: number;
  /** Items drawn beyond each edge of the view. Default 8. */
  overscan?: number;
  /** Lists shorter than this are drawn whole. Default {@link WINDOW_MIN}. */
  min?: number;
}

/** The items to draw and the space that stands in for the rest. */
export interface WindowedRange {
  /** Index of the first item to draw. */
  start: number;
  /** Index after the last item to draw. */
  end: number;
  /** Space in CSS pixels for the items before `start` (0 when the list is drawn whole). */
  before: number;
  /** Space in CSS pixels for the items from `end` on. */
  after: number;
  /** False when the list is short and drawn whole: no spacing is needed. */
  windowed: boolean;
  /** Scrolls so that item `index` is fully in view with `margin` to spare (does nothing when it already is), drawn or not. */
  scrollToIndex: (index: number, margin?: number) => void;
}

interface Geometry {
  /** Scroll offset of the scroller along the axis. */
  scroll: number;
  /** Size of the scroller's view along the axis; 0 until measured. */
  view: number;
  /** Distance from one item's start to the next one's. */
  pitch: number;
  /** Size of one item along the axis (without the gap). */
  size: number;
  /** Where item 0 starts, in scroller content coordinates. */
  origin: number;
}

/** The view assumed until the scroller has been measured (a screen's width), so the first paint already has its items. */
const FALLBACK_VIEW = 1600;

/** The first and last-plus-one index to draw for a geometry. */
function rangeOf(count: number, g: Geometry, overscan: number): [number, number] {
  const view = g.view > 0 ? g.view : FALLBACK_VIEW;
  const first = Math.floor((g.scroll - g.origin) / g.pitch) - overscan;
  const last = Math.ceil((g.scroll + view - g.origin) / g.pitch) + overscan;
  const start = Math.max(0, Math.min(count, first));
  return [start, Math.max(start, Math.min(count, last))];
}

/** The nearest ancestor of `el` that scrolls along y, or the document's scrolling element. */
function scrollParent(el: HTMLElement | null): HTMLElement | null {
  for (let node = el?.parentElement ?? null; node; node = node.parentElement) {
    const overflow = getComputedStyle(node).overflowY;
    if (overflow === "auto" || overflow === "scroll") return node;
  }
  return (document.scrollingElement as HTMLElement | null) ?? null;
}

/** The range of a windowed list for its scroll position; see the module comment. */
export function useWindowedRange(options: WindowedOptions): WindowedRange {
  const { count, axis, list, itemSelector, estimate, overscan = 8, min = WINDOW_MIN } = options;
  const windowed = count >= min;
  const horizontal = axis === "x";
  const scrollerRef = useRef<HTMLElement | null>(null);
  const initial: Geometry = { scroll: 0, view: 0, pitch: estimate, size: estimate, origin: 0 };
  const geoRef = useRef<Geometry>(initial);
  const [geo, setGeo] = useState<Geometry>(initial);
  const shown = useRef<[number, number]>([0, 0]);
  const renderedPitch = useRef(estimate);
  // The options a listener needs when it runs, without making the listeners depend on them.
  const latest = useRef({ count, overscan, windowed, horizontal, itemSelector, scroller: options.scroller });
  latest.current = { count, overscan, windowed, horizontal, itemSelector, scroller: options.scroller };

  const [start, end] = windowed ? rangeOf(count, geo, overscan) : [0, count];
  shown.current = [start, end];
  // The pitch the space before the first drawn item was made with: what that space is subtracted with when measuring.
  renderedPitch.current = geo.pitch;

  /** Takes a new geometry; draws again only when it changes what is drawn (or how big the stand-in space is). */
  const accept = useCallback((next: Geometry) => {
    geoRef.current = next;
    const l = latest.current;
    const wanted = rangeOf(l.count, next, l.overscan);
    setGeo((was) => {
      if (was.pitch === next.pitch && was.size === next.size && was.origin === next.origin && was.view === next.view) {
        const have = rangeOf(l.count, was, l.overscan);
        if (have[0] === wanted[0] && have[1] === wanted[1]) return was;
      }
      return next;
    });
  }, []);

  /** Reads the scroller's position and view, and the items' pitch and origin from the DOM. */
  const measure = useCallback((): void => {
    const l = latest.current;
    const el = scrollerRef.current;
    const container = list.current;
    // A list that is not shown (display: none) has no boxes to measure; the resize observer measures it when it is.
    if (!el || !container || el.getClientRects().length === 0) return;
    const x = l.horizontal;
    const scrollerBox = el.getBoundingClientRect();
    const scroll = x ? el.scrollLeft : el.scrollTop;
    const view = x ? el.clientWidth : el.clientHeight;
    const was = geoRef.current;
    let { pitch, size, origin } = was;
    const items = container.querySelectorAll<HTMLElement>(l.itemSelector);
    const a = items[0]?.getBoundingClientRect();
    if (a) {
      const b = items[1]?.getBoundingClientRect();
      const lead = x ? a.left : a.top;
      size = x ? a.width : a.height;
      pitch = b ? (x ? b.left : b.top) - lead : size;
      if (pitch <= 0) pitch = was.pitch;
      const edge = x ? scrollerBox.left + el.clientLeft : scrollerBox.top + el.clientTop;
      origin = lead - edge + scroll - shown.current[0] * renderedPitch.current;
    }
    accept({ scroll, view, pitch, size, origin });
  }, [accept, list]);

  // Which element scrolls, and listening to it. Re-attached when the list starts or stops being windowed.
  useLayoutEffect(() => {
    if (!windowed) return;
    const el = horizontal ? (latest.current.scroller?.current ?? list.current) : scrollParent(list.current);
    scrollerRef.current = el;
    if (!el) return;
    measure();
    let frame = 0;
    const onScroll = (): void => {
      if (frame) return;
      // One look per frame, however many scroll events came.
      frame = requestAnimationFrame(() => {
        frame = 0;
        const g = geoRef.current;
        const scroll = latest.current.horizontal ? el.scrollLeft : el.scrollTop;
        if (scroll !== g.scroll) accept({ ...g, scroll });
      });
    };
    const target: HTMLElement | Document = el === document.scrollingElement ? document : el;
    target.addEventListener("scroll", onScroll, { passive: true });
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => measure());
    observer?.observe(el);
    return () => {
      target.removeEventListener("scroll", onScroll);
      observer?.disconnect();
      if (frame) cancelAnimationFrame(frame);
    };
  }, [windowed, horizontal, list, measure, accept]);

  // The first items drawn (or a first item that changed size) give the pitch; the count changing needs the range worked out again.
  useLayoutEffect(() => {
    if (windowed) measure();
  }, [windowed, count, geo.pitch, measure]);

  const scrollToIndex = useCallback(
    (index: number, margin = 24): void => {
      const el = scrollerRef.current;
      if (!el) return;
      const g = geoRef.current;
      const x = latest.current.horizontal;
      const view = x ? el.clientWidth : el.clientHeight;
      const first = g.origin + index * g.pitch;
      const last = first + g.size;
      let scroll = x ? el.scrollLeft : el.scrollTop;
      if (first < scroll + margin) scroll = first - margin;
      else if (last > scroll + view - margin) scroll = last - view + margin;
      else return;
      if (x) el.scrollLeft = Math.max(0, scroll);
      else el.scrollTop = Math.max(0, scroll);
      accept({ ...g, view, scroll: x ? el.scrollLeft : el.scrollTop });
    },
    [accept],
  );

  return {
    start,
    end,
    before: windowed ? start * geo.pitch : 0,
    after: windowed ? (count - end) * geo.pitch : 0,
    windowed,
    scrollToIndex,
  };
}
