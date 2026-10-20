import { useEffect, useLayoutEffect, useRef } from "react";
import type { HTMLAttributes, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import type { ViewportRect } from "../api/types";
import { Badge } from "@/components/glass/badge";
import { isTauri } from "../env";
import { cx } from "./cx";
import { useLatest } from "./hooks";
import { usePhotoGestures } from "./usePhotoGestures";
import type { PhotoGestureHandlers } from "./usePhotoGestures";

export interface PhotoProbe {
  /** Position in the cell, 0–1 across and down. */
  x: number;
  y: number;
}

/**
 * A picture of the whole cell made by the shell (the fallback viewport, `docs/11` §Shell notes): the photo already letterboxed on paper,
 * `widthPx` × `heightPx` device pixels. It is drawn at that size from the cell's corner, one picture pixel to one device pixel, so
 * a 100 % view is not resampled by the browser.
 */
export interface PhotoCellFrame {
  src: string;
  widthPx: number;
  heightPx: number;
}

export interface PhotoCellProps extends Omit<HTMLAttributes<HTMLElement>, "children" | "className"> {
  /** URL of the photo, drawn as an `<img>` in the browser and the mock. Ignored with `native`, and when there is a `frame`. */
  imageSrc?: string;
  /** The shell's picture of the cell, instead of `imageSrc`, when the fallback viewport draws the photo. Ignored with `native`. */
  frame?: PhotoCellFrame;
  /** What the picture shows, for `role="img"`: "Photo DSC04204, Neutral · 100%". */
  imageLabel?: string;
  /** Bottom-left black chip: the look and its intensity ("Neutral · 100%"). Hidden in Before mode. */
  lookChip?: string;
  /** Bottom-right line: "Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8". */
  cameraLine?: string;
  /** Before mode: shows the yellow "Before · camera default" label and hides the look chip. */
  before?: boolean;
  /** Text of the Before label. Default "Before · camera default". */
  beforeLabel?: string;
  /** Pixel readout shown at the top right while ⌥ is held ("R 182 G 164 B 140"). */
  pixelReadout?: string | null;
  /**
   * A note at the top centre, over everything else in the cell ("Rendering on the CPU"). It sits outside the picture's `role="img"`,
   * so a screen reader hears a status in it, and it should take no pointer events.
   */
  notice?: ReactNode;
  /**
   * True when a native layer draws the photo behind a transparent cell (the app on a Mac): no `<img>` is made. Defaults to
   * `isTauri()`; the Photos screen sets it from `viewport_info`, since the fallback viewport draws in the webview.
   */
  native?: boolean;
  /**
   * The cell's rectangle in CSS pixels with the device scale, on mount and whenever the cell
   * resizes or moves (ResizeObserver and window resize). The app sends it as `viewport_set_rect`.
   */
  onRect?: (rect: ViewportRect) => void;
  /**
   * While ⌥ is held over the cell, the pointer position (0–1) on every move; null when ⌥ is
   * released or the pointer leaves. The screen answers with `sample_pixel` and `pixelReadout`.
   */
  onPixelProbe?: (probe: PhotoProbe | null) => void;
  /**
   * What the wheel and a trackpad pinch do over the cell. With it the cell claims every wheel and gesture event over it (nothing
   * scrolls or bounces under the photo); without it the cell leaves them to the page (`usePhotoGestures`).
   */
  gestures?: PhotoGestureHandlers;
  /** Extra layers over the photo (the crop rectangle). They fill the cell. */
  children?: ReactNode;
  className?: string;
}

const sameRect = (a: ViewportRect, b: ViewportRect): boolean =>
  Math.abs(a.x - b.x) < 0.01 &&
  Math.abs(a.y - b.y) < 0.01 &&
  Math.abs(a.width - b.width) < 0.01 &&
  Math.abs(a.height - b.height) < 0.01 &&
  a.scale === b.scale;

/**
 * The photo cell: always exactly 3:2, `width: min(100%, calc((max(100vh, 760px) − 199px) · 1.5))`,
 * centred, no surround. It carries `data-viewport` and reports its rectangle through `onRect`,
 * which is what the native viewport follows (`docs/01` §Native viewport). In the browser and the
 * mock it draws the photo as an `<img>` letterboxed on paper; with the native viewport the cell is
 * transparent; with the fallback viewport it shows the shell's picture of the cell (`frame`).
 * Overlays: look chip, camera line, the Before label, the ⌥ pixel readout and a notice at the top centre.
 */
export function PhotoCell({
  imageSrc,
  frame,
  imageLabel = "Photo",
  lookChip,
  cameraLine,
  before = false,
  beforeLabel = "Before · camera default",
  pixelReadout,
  notice,
  native = isTauri(),
  onRect,
  onPixelProbe,
  gestures,
  children,
  className,
  ...rest
}: PhotoCellProps) {
  const cell = useRef<HTMLElement>(null);
  // Where the cell is, as `report` last measured it: a pointer move or a wheel event over the cell reads this instead of asking the
  // browser (a forced layout whenever anything on the page is dirty). It is as fresh as what the native layer follows.
  const placed = useRef<{ left: number; top: number; width: number; height: number } | null>(null);
  const onRectRef = useLatest(onRect);
  const probeRef = useLatest(onPixelProbe);
  const pointer = useRef<PhotoProbe | null>(null);
  const probing = useRef(false);

  useLayoutEffect(() => {
    const el = cell.current;
    if (!el) return;
    let previous: ViewportRect | null = null;
    const report = (): void => {
      const box = el.getBoundingClientRect();
      placed.current = { left: box.left, top: box.top, width: box.width, height: box.height };
      const next: ViewportRect = {
        x: box.left,
        y: box.top,
        width: box.width,
        height: box.height,
        scale: window.devicePixelRatio || 1,
      };
      if (previous && sameRect(previous, next)) return;
      previous = next;
      onRectRef.current?.(next);
    };
    report();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(report);
    observer?.observe(el);
    // The cell can move without resizing (the window gets wider and the cell stays centred).
    window.addEventListener("resize", report);
    // Dragging the window to a display of another scale changes `scale` but not the cell's size in CSS pixels, and not every web view
    // sends `resize` for it: the media query for the current scale stops matching.
    let scaleQuery: MediaQueryList | null = null;
    const onScale = (): void => {
      report();
      watchScale();
    };
    const watchScale = (): void => {
      scaleQuery?.removeEventListener("change", onScale);
      scaleQuery = typeof window.matchMedia === "function" ? window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`) : null;
      scaleQuery?.addEventListener?.("change", onScale);
    };
    watchScale();
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", report);
      scaleQuery?.removeEventListener?.("change", onScale);
    };
  }, [onRectRef]);

  // ⌥ pressed or released while the pointer rests on the cell.
  useEffect(() => {
    if (!onPixelProbe) return;
    const onKey = (event: KeyboardEvent): void => {
      if (event.key !== "Alt") return;
      if (event.type === "keydown" && pointer.current) {
        probing.current = true;
        probeRef.current?.(pointer.current);
      } else if (event.type === "keyup" && probing.current) {
        probing.current = false;
        probeRef.current?.(null);
      }
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, [onPixelProbe, probeRef]);

  /** The cell's place: the last measured one, or a fresh measurement before the first (and for a cell that has no size yet). */
  const place = (): { left: number; top: number; width: number; height: number } => {
    const known = placed.current;
    if (known && known.width > 0 && known.height > 0) return known;
    const box = cell.current?.getBoundingClientRect();
    return box ? { left: box.left, top: box.top, width: box.width, height: box.height } : { left: 0, top: 0, width: 0, height: 0 };
  };
  usePhotoGestures(cell, gestures, place);

  const onPointerMove = (event: ReactPointerEvent<HTMLElement>): void => {
    rest.onPointerMove?.(event);
    const box = place();
    if (box.width <= 0 || box.height <= 0) return;
    pointer.current = { x: (event.clientX - box.left) / box.width, y: (event.clientY - box.top) / box.height };
    if (event.altKey) {
      probing.current = true;
      probeRef.current?.(pointer.current);
    } else if (probing.current) {
      probing.current = false;
      probeRef.current?.(null);
    }
  };

  const onPointerLeave = (event: ReactPointerEvent<HTMLElement>): void => {
    rest.onPointerLeave?.(event);
    pointer.current = null;
    if (probing.current) {
      probing.current = false;
      probeRef.current?.(null);
    }
  };

  return (
    <main
      aria-label="Photo"
      {...rest}
      ref={cell}
      data-viewport=""
      data-native={native || undefined}
      className={cx("ym-photo-cell", className)}
      onPointerMove={onPointerMove}
      onPointerLeave={onPointerLeave}
    >
      <div className="ym-photo-cell__view" role="img" aria-label={imageLabel}>
        {!native && frame ? (
          <img
            className="ym-photo-cell__frame"
            src={frame.src}
            alt=""
            draggable={false}
            decoding="sync"
            style={{ width: frame.widthPx / (window.devicePixelRatio || 1), height: frame.heightPx / (window.devicePixelRatio || 1) }}
          />
        ) : !native && imageSrc ? (
          <img className="ym-photo-cell__image" src={imageSrc} alt="" draggable={false} />
        ) : null}
        {before ? (
          <Badge variant="outline" className="ym-photo-cell__before">
            {beforeLabel}
          </Badge>
        ) : null}
        {!before && lookChip ? (
          <Badge variant="outline" className="ym-photo-cell__chip">
            {lookChip}
          </Badge>
        ) : null}
        {cameraLine ? <span className="ym-photo-cell__camera">{cameraLine}</span> : null}
        {pixelReadout ? (
          <Badge variant="outline" className="ym-photo-cell__readout">
            {pixelReadout}
          </Badge>
        ) : null}
      </div>
      {children ? <div className="ym-photo-cell__overlay">{children}</div> : null}
      {notice}
    </main>
  );
}
