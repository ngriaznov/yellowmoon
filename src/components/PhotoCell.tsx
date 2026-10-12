import { useEffect, useLayoutEffect, useRef } from "react";
import type { HTMLAttributes, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import type { ViewportRect } from "../api/types";
import { isTauri } from "../env";
import { cx } from "./cx";
import { useLatest } from "./hooks";

export interface PhotoProbe {
  /** Position in the cell, 0–1 across and down. */
  x: number;
  y: number;
}

export interface PhotoCellProps extends Omit<HTMLAttributes<HTMLElement>, "children" | "className"> {
  /** URL of the photo, drawn as an `<img>` in the browser and the mock. Ignored in the app. */
  imageSrc?: string;
  /** What the picture shows, for `role="img"`: "Photo DSC04204, Neutral · 100%". */
  imageLabel?: string;
  /** Bottom-left black chip: the look and its intensity ("Neutral · 100%"). Hidden in Before mode. */
  lookChip?: string;
  /** Bottom-right line: "Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8". */
  cameraLine?: string;
  /** Before mode: shows the yellow "Before · camera log" label and hides the look chip. */
  before?: boolean;
  /** Text of the Before label. Default "Before · camera log". */
  beforeLabel?: string;
  /** Pixel readout shown at the top right while ⌥ is held ("R 182 G 164 B 140"). */
  pixelReadout?: string | null;
  /**
   * True inside the app: the native layer draws the photo behind a transparent cell and no
   * `<img>` is made. Defaults to `isTauri()`.
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
 * mock it draws the photo as an `<img>` letterboxed on paper; in the app the cell is transparent.
 * Overlays: look chip, camera line, the Before label and the ⌥ pixel readout.
 */
export function PhotoCell({
  imageSrc,
  imageLabel = "Photo",
  lookChip,
  cameraLine,
  before = false,
  beforeLabel = "Before · camera log",
  pixelReadout,
  native = isTauri(),
  onRect,
  onPixelProbe,
  children,
  className,
  ...rest
}: PhotoCellProps) {
  const cell = useRef<HTMLElement>(null);
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
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", report);
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

  const onPointerMove = (event: ReactPointerEvent<HTMLElement>): void => {
    rest.onPointerMove?.(event);
    const box = event.currentTarget.getBoundingClientRect();
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
        {!native && imageSrc ? <img className="ym-photo-cell__image" src={imageSrc} alt="" draggable={false} /> : null}
        {before ? <span className="ym-photo-cell__before">{beforeLabel}</span> : null}
        {!before && lookChip ? <span className="ym-photo-cell__chip">{lookChip}</span> : null}
        {cameraLine ? <span className="ym-photo-cell__camera">{cameraLine}</span> : null}
        {pixelReadout ? <span className="ym-photo-cell__readout">{pixelReadout}</span> : null}
      </div>
      {children ? <div className="ym-photo-cell__overlay">{children}</div> : null}
    </main>
  );
}
