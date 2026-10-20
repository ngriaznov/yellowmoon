import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { cx } from "./cx";
import { clamp } from "./format";
import { useDraft, useFrameEmitter, useGestureCallback, usePointerDrag } from "./hooks";
import {
  evaluateCurve,
  insertPoint,
  movePoint,
  removePoint,
  curvePath,
  toCodeValue,
  widestGapX,
} from "./curve";
import type { CurvePoint } from "./curve";

/** The four curves of the Curve tab. */
export type CurveChannel = "luma" | "red" | "green" | "blue";

/** Stroke colour of each channel's curve (`--curve-r`, `--curve-g`, `--curve-b`; luma is ink). */
export const CURVE_COLORS: Readonly<Record<CurveChannel, string>> = {
  luma: "var(--ink)",
  red: "var(--curve-r)",
  green: "var(--curve-g)",
  blue: "var(--curve-b)",
};

/** Names for the segmented control and the plot's accessible name. */
export const CURVE_CHANNEL_LABELS: Readonly<Record<CurveChannel, string>> = {
  luma: "Luma",
  red: "Red",
  green: "Green",
  blue: "Blue",
};

export interface CurvePlotProps {
  /** The control points, sorted by x, in [0, 1]². At least two. */
  points: readonly CurvePoint[];
  /**
   * Called with the new point list. While a point is dragged it fires at most once per
   * animation frame with `final = false`, then once with `final = true` on release. Keys call it
   * once with `final = true`.
   */
  onChange: (points: CurvePoint[], final: boolean) => void;
  /** Which curve this is: sets the stroke colour and the accessible name. Default `luma`. */
  channel?: CurveChannel;
  /** Histogram of the photo in LogC4, in any number of bins; drawn faintly behind the grid. */
  histogram?: readonly number[] | null;
  /** The point that starts selected (drawn filled). Default none. */
  defaultSelected?: number | null;
  /** Hides the Input and Output readout under the plot. */
  hideReadout?: boolean;
  className?: string;
}

/** Plot size in SVG user units. The element itself is as wide as its column (352 px in the panel). */
export const PLOT_WIDTH = 320;
export const PLOT_HEIGHT = 220;
/** How close, in CSS pixels, a press must be to grab an existing point. */
const GRAB_PX = 12;
/** How far outside the plot, in CSS pixels, a drag must go to delete its point. */
const DELETE_PX = 14;
/** One arrow-key step in unit coordinates: one code value. */
const KEY_STEP = 1 / 255;

const samePoints = (a: readonly CurvePoint[], b: readonly CurvePoint[]): boolean =>
  a.length === b.length && a.every((p, i) => Math.abs(p[0] - (b[i] as CurvePoint)[0]) < 1e-6 && Math.abs(p[1] - (b[i] as CurvePoint)[1]) < 1e-6);

interface DragState {
  index: number;
  /** The list without the deletion applied, as of the last move inside the plot. */
  base: CurvePoint[];
  removing: boolean;
}

/**
 * The curve editor (`docs/11-ui.md` §Components): a 320 × 220 plot with a 2 px border, a
 * quarter grid, the identity diagonal, a monotone cubic spline (Fritsch–Carlson, the maths of
 * `docs/02` §C11) and up to 16 square control points. Press anywhere to add a point on the curve
 * and drag it; press a point to move it; drag a point off the plot to delete it. The two end
 * points only move up and down and cannot be deleted. Below the plot, the readout gives Input and
 * Output as code values (0–255) for the point being dragged, the position under the pointer,
 * or the selected point.
 *
 * Keyboard: Tab reaches the plot, then each point. Arrows move a point by one code value (Shift
 * ×10), Delete or Backspace removes an inner point, Return on the plot adds a point in the widest
 * gap.
 */
export function CurvePlot({
  points,
  onChange,
  channel = "luma",
  histogram,
  defaultSelected = null,
  hideReadout = false,
  className,
}: CurvePlotProps) {
  const svg = useRef<SVGSVGElement>(null);
  const pointRefs = useRef<Array<SVGRectElement | null>>([]);
  const focusAfter = useRef<number | null>(null);
  const drag = useRef<DragState | null>(null);
  const [selected, setSelected] = useState<number | null>(defaultSelected);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);

  const draft = useDraft<readonly CurvePoint[]>(points, samePoints);
  const shown = draft.shown;
  // A drag reports to the `onChange` it began with (`useGestureCallback`).
  const onChangeRef = useGestureCallback(onChange);
  const frames = useFrameEmitter<CurvePoint[]>((list) => onChangeRef.current(list, false));

  const commit = (list: CurvePoint[]): void => {
    draft.settle(list);
    onChangeRef.current(list, true);
  };

  /** Pointer position as unit coordinates inside the plot (y up) plus the box for pixel maths. */
  const locate = (clientX: number, clientY: number) => {
    const el = svg.current;
    const rect = el?.getBoundingClientRect();
    if (!el || !rect || rect.width <= 0 || rect.height <= 0) return null;
    const border = parseFloat(getComputedStyle(el).borderLeftWidth) || 0;
    const w = Math.max(1, rect.width - border * 2);
    const h = Math.max(1, rect.height - border * 2);
    return {
      x: (clientX - rect.left - border) / w,
      y: 1 - (clientY - rect.top - border) / h,
      w,
      h,
      outside:
        clientX < rect.left - DELETE_PX ||
        clientX > rect.right + DELETE_PX ||
        clientY < rect.top - DELETE_PX ||
        clientY > rect.bottom + DELETE_PX,
    };
  };

  const onPointerDown = usePointerDrag({
    start: (event: ReactPointerEvent<Element>) => {
      const at = locate(event.clientX, event.clientY);
      if (!at) return false;
      let list: CurvePoint[] = shown.map((p): CurvePoint => [p[0], p[1]]);
      let index = -1;
      let best = GRAB_PX;
      list.forEach((p, i) => {
        const distance = Math.hypot((p[0] - at.x) * at.w, (p[1] - at.y) * at.h);
        if (distance <= best) {
          best = distance;
          index = i;
        }
      });
      if (index < 0) {
        const inserted = insertPoint(list, clamp(at.x, 0, 1));
        if (!inserted) return false;
        list = inserted.points;
        index = inserted.index;
      }
      svg.current?.focus({ preventScroll: true });
      onChangeRef.begin();
      drag.current = { index, base: list, removing: false };
      setSelected(index);
      setDraggedIndex(index);
      draft.drag(list);
      frames.push(list);
      return true;
    },
    move: (event: PointerEvent) => {
      const state = drag.current;
      const at = locate(event.clientX, event.clientY);
      if (!state || !at) return;
      const interior = state.index > 0 && state.index < state.base.length - 1;
      if (at.outside && interior) {
        state.removing = true;
        const list = removePoint(state.base, state.index);
        draft.drag(list);
        frames.push(list);
        return;
      }
      state.removing = false;
      const list = movePoint(state.base, state.index, clamp(at.x, 0, 1), clamp(at.y, 0, 1));
      state.base = list;
      draft.drag(list);
      frames.push(list);
    },
    end: () => {
      const state = drag.current;
      drag.current = null;
      frames.cancel();
      setDraggedIndex(null);
      if (state?.removing) {
        setSelected(null);
        commit(removePoint(state.base, state.index));
      } else if (state) {
        commit(state.base);
      }
      onChangeRef.end();
    },
  });

  const addAtWidestGap = (): void => {
    const inserted = insertPoint(shown, widestGapX(shown));
    if (!inserted) return;
    focusAfter.current = inserted.index;
    setSelected(inserted.index);
    commit(inserted.points);
  };

  const onPointKeyDown = (event: KeyboardEvent<SVGRectElement>, index: number): void => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const unit = KEY_STEP * (event.shiftKey ? 10 : 1);
    const p = shown[index];
    if (!p) return;
    let next: CurvePoint[] | null = null;
    switch (event.key) {
      case "ArrowUp":
        next = movePoint(shown, index, p[0], p[1] + unit);
        break;
      case "ArrowDown":
        next = movePoint(shown, index, p[0], p[1] - unit);
        break;
      case "ArrowRight":
        next = movePoint(shown, index, p[0] + unit, p[1]);
        break;
      case "ArrowLeft":
        next = movePoint(shown, index, p[0] - unit, p[1]);
        break;
      case "Delete":
      case "Backspace":
        if (index > 0 && index < shown.length - 1) {
          focusAfter.current = Math.max(0, index - 1);
          setSelected(null);
          event.preventDefault();
          event.stopPropagation();
          commit(removePoint(shown, index));
        }
        return;
      default:
        return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (!samePoints(next, shown)) commit(next);
  };

  // Put focus back on a point after the list changed under the keyboard (add or delete).
  useEffect(() => {
    const index = focusAfter.current;
    if (index === null) return;
    focusAfter.current = null;
    pointRefs.current[index]?.focus();
  });

  const color = CURVE_COLORS[channel];
  const name = `${CURVE_CHANNEL_LABELS[channel]} curve`;
  const path = curvePath(shown, PLOT_WIDTH, PLOT_HEIGHT);

  // Readout: the dragged point, else the pointer position, else the selected point, else the middle.
  const dragged = draggedIndex !== null ? shown[draggedIndex] : undefined;
  const chosen = selected !== null ? shown[selected] : undefined;
  const readX = dragged ? dragged[0] : hoverX !== null ? hoverX : chosen ? chosen[0] : 0.5;
  const readY = dragged ? dragged[1] : hoverX !== null || !chosen ? evaluateCurve(shown, readX) : chosen[1];

  let histogramD = "";
  if (histogram && histogram.length >= 2) {
    const top = Math.max(0, ...histogram);
    if (top > 0) {
      histogramD = `M0 ${PLOT_HEIGHT} ${histogram
        .map((v, i) => `L${Math.round((i / (histogram.length - 1)) * PLOT_WIDTH * 100) / 100} ${Math.round((PLOT_HEIGHT - (v / top) * PLOT_HEIGHT * 0.9) * 100) / 100}`)
        .join(" ")} L${PLOT_WIDTH} ${PLOT_HEIGHT} Z`;
    }
  }

  return (
    <div className={cx("ym-curve", className)}>
      <svg
        ref={svg}
        className="ym-curve__plot"
        viewBox={`0 0 ${PLOT_WIDTH} ${PLOT_HEIGHT}`}
        role="group"
        aria-label={name}
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={(event) => {
          if (drag.current) return;
          const at = locate(event.clientX, event.clientY);
          setHoverX(at ? clamp(at.x, 0, 1) : null);
        }}
        onPointerLeave={() => setHoverX(null)}
        onKeyDown={(event) => {
          if (event.target === event.currentTarget && event.key === "Enter") {
            event.preventDefault();
            addAtWidestGap();
          }
        }}
      >
        {histogramD ? <path className="ym-curve__histogram" d={histogramD} /> : null}
        <path className="ym-curve__grid" d="M80 0V220M160 0V220M240 0V220M0 55H320M0 110H320M0 165H320" />
        <path className="ym-curve__identity" d="M0 220 L320 0" />
        <path className="ym-curve__line" d={path} style={{ stroke: color }} />
        {shown.map((p, i) => {
          const isSelected = i === selected;
          const isEnd = i === 0 || i === shown.length - 1;
          const size = isSelected ? 12 : 10;
          // Keep the end handles fully inside the plot instead of half-clipped at the corners.
          const left = clamp(p[0] * PLOT_WIDTH - size / 2, 0, PLOT_WIDTH - size);
          const top = clamp((1 - p[1]) * PLOT_HEIGHT - size / 2, 0, PLOT_HEIGHT - size);
          return (
            <rect
              key={i}
              ref={(node) => {
                pointRefs.current[i] = node;
              }}
              className="ym-curve__point"
              data-selected={isSelected || undefined}
              data-end={isEnd || undefined}
              x={left}
              y={top}
              width={size}
              height={size}
              tabIndex={0}
              role="slider"
              aria-label={`Point ${i + 1} of ${shown.length}`}
              aria-orientation="vertical"
              aria-valuemin={0}
              aria-valuemax={255}
              aria-valuenow={toCodeValue(p[1])}
              aria-valuetext={`Input ${toCodeValue(p[0])}, output ${toCodeValue(p[1])}`}
              onFocus={() => setSelected(i)}
              onKeyDown={(event) => onPointKeyDown(event, i)}
            />
          );
        })}
      </svg>
      {hideReadout ? null : (
        <div className="ym-curve__readout" aria-live="off">
          <span>
            Input <b>{toCodeValue(readX)}</b>
          </span>
          <span>
            Output <b>{toCodeValue(readY)}</b>
          </span>
        </div>
      )}
    </div>
  );
}
