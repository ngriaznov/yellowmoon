import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { cx } from "./cx";
import { clamp } from "./format";
import { useDraft, useFrameEmitter, useGestureCallback, usePointerDrag } from "./hooks";
import {
  addHuePoint,
  axisTicks,
  HUE_CURVES,
  HUE_PERIOD,
  hueCurvePath,
  MAX_HUE_POINTS,
  moveHuePoint,
  pointReadout,
  removeHuePoint,
  roundValue,
  sortedPoints,
  tickUnit,
  unitToValue,
  valueToUnit,
  widestGapHue,
} from "./hueCurve";
import type { HueCurveKind, HuePoint, HueScale } from "./hueCurve";
import { hueGradient } from "./uniformMath";

export interface HuePlotProps {
  kind: HueCurveKind;
  /** The curve's points as stored (any order; read modulo 360 and sorted). */
  points: readonly HuePoint[];
  /** The view: which values the plot's top and bottom are. */
  scale: HueScale;
  /**
   * Called with the new list. While a point is dragged it fires at most once per animation frame with `final = false`, then once with `final = true` on
   * release. Keys, a double click and Delete call it once with `final = true`.
   */
  onChange: (points: HuePoint[], final: boolean) => void;
  /** Said under the plot while the curve is full ("A curve can have up to 16 points.") and a point is asked for. */
  onFull?: () => void;
  /** The accessible name of the plot ("Hue vs hue curve"). */
  label: string;
  className?: string;
}

/** Plot size in SVG user units; the element is as wide as its column. */
export const HUE_PLOT_WIDTH = 320;
export const HUE_PLOT_HEIGHT = 170;
/** How close, in CSS pixels, a press must be to grab an existing point. */
const GRAB_PX = 12;

const samePoints = (a: readonly HuePoint[], b: readonly HuePoint[]): boolean =>
  a.length === b.length && a.every((p, i) => Math.abs(p.hue_deg - (b[i] as HuePoint).hue_deg) < 1e-6 && Math.abs(p.value - (b[i] as HuePoint).value) < 1e-6);

/**
 * The hue curve editor of the Hue tab (`docs/34` §9.4): the Curve card's plot widget (`CurvePlot`: its look, its keyboard path, its drag and its drafts)
 * **taught a periodic x axis**. The x axis is the hue, 0° to 360°, with the rainbow track of Colour Uniform's Hue slider under the plot; the curve is the
 * core's periodic monotone spline, closed at the seam. The y axis is a turn in degrees (a detent at 0) or a multiplier on a log scale (a detent at 1), over
 * the view the card chose (zoomed on the fit's values, or the full range).
 *
 * Press on the plot to add a point there and drag it; press a point to move it; double-click a point or press Delete to remove it. Keyboard: Tab reaches the
 * plot, then each point; the arrows move a point by a degree of hue or a tenth of a degree of turn (a hundredth for a multiplier), Shift ×10; Enter on the plot
 * adds a point in the widest gap. At most 16 points. An empty curve is the identity.
 */
export function HuePlot({ kind, points, scale, onChange, onFull, label, className }: HuePlotProps) {
  const svg = useRef<SVGSVGElement>(null);
  const pointRefs = useRef<Array<SVGRectElement | null>>([]);
  const focusAfter = useRef<number | null>(null);
  const drag = useRef<{ index: number; base: HuePoint[] } | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const draft = useDraft<readonly HuePoint[]>(sortedPoints(points), samePoints);
  const shown = draft.shown;
  const onChangeRef = useGestureCallback(onChange);
  const frames = useFrameEmitter<HuePoint[]>((list) => onChangeRef.current(list, false));
  const { identity } = HUE_CURVES[kind];

  const commit = (list: HuePoint[]): void => {
    draft.settle(list);
    onChangeRef.current(list, true);
  };

  /** The pointer as a hue and a value on the plot, plus the box for pixel maths. */
  const locate = (clientX: number, clientY: number) => {
    const el = svg.current;
    const rect = el?.getBoundingClientRect();
    if (!el || !rect || rect.width <= 0 || rect.height <= 0) return null;
    const border = parseFloat(getComputedStyle(el).borderLeftWidth) || 0;
    const w = Math.max(1, rect.width - border * 2);
    const h = Math.max(1, rect.height - border * 2);
    const ux = clamp((clientX - rect.left - border) / w, 0, 1);
    const uy = clamp(1 - (clientY - rect.top - border) / h, 0, 1);
    return { hue: ux * HUE_PERIOD, value: unitToValue(scale, uy), ux, uy, w, h };
  };

  const onPointerDown = usePointerDrag({
    start: (event: ReactPointerEvent<Element>) => {
      const at = locate(event.clientX, event.clientY);
      if (!at) return false;
      let list = shown.map((p): HuePoint => ({ ...p }));
      let index = -1;
      let best = GRAB_PX;
      list.forEach((p, i) => {
        const dx = (p.hue_deg / HUE_PERIOD - at.ux) * at.w;
        const dy = (valueToUnit(scale, p.value) - at.uy) * at.h;
        const distance = Math.hypot(dx, dy);
        if (distance <= best) {
          best = distance;
          index = i;
        }
      });
      if (index < 0) {
        const added = addHuePoint(list, kind, at.hue);
        if (!added) {
          if (list.length >= MAX_HUE_POINTS) onFull?.();
          return false;
        }
        list = added.points;
        index = added.index;
      }
      svg.current?.focus({ preventScroll: true });
      onChangeRef.begin();
      drag.current = { index, base: list };
      setSelected(index);
      draft.drag(list);
      frames.push(list);
      return true;
    },
    move: (event: PointerEvent) => {
      const state = drag.current;
      const at = locate(event.clientX, event.clientY);
      if (!state || !at) return;
      const list = moveHuePoint(state.base, kind, state.index, at.hue, at.value);
      state.base = list;
      draft.drag(list);
      frames.push(list);
    },
    end: () => {
      const state = drag.current;
      drag.current = null;
      frames.cancel();
      if (state) commit(state.base);
      onChangeRef.end();
    },
  });

  const addAtWidestGap = (): void => {
    const added = addHuePoint(shown, kind, widestGapHue(shown));
    if (!added) {
      if (shown.length >= MAX_HUE_POINTS) onFull?.();
      return;
    }
    focusAfter.current = added.index;
    setSelected(added.index);
    commit(added.points);
  };

  const remove = (index: number): void => {
    focusAfter.current = Math.max(0, index - 1);
    setSelected(null);
    commit(removeHuePoint(shown, index));
  };

  const onPointKeyDown = (event: KeyboardEvent<SVGRectElement>, index: number): void => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const p = shown[index];
    if (!p) return;
    const big = event.shiftKey ? 10 : 1;
    const valueStep = (kind === "hue_vs_hue" ? 0.1 : 0.01) * big;
    let next: HuePoint[] | null = null;
    switch (event.key) {
      case "ArrowUp":
        next = moveHuePoint(shown, kind, index, p.hue_deg, p.value + valueStep);
        break;
      case "ArrowDown":
        next = moveHuePoint(shown, kind, index, p.hue_deg, p.value - valueStep);
        break;
      case "ArrowRight":
        next = moveHuePoint(shown, kind, index, p.hue_deg + big, p.value);
        break;
      case "ArrowLeft":
        next = moveHuePoint(shown, kind, index, p.hue_deg - big, p.value);
        break;
      case "Delete":
      case "Backspace":
        event.preventDefault();
        event.stopPropagation();
        remove(index);
        return;
      default:
        return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (!samePoints(next, shown)) commit(next);
  };

  // Put the focus back on a point after the list changed under the keyboard (a point was added or removed).
  useEffect(() => {
    const index = focusAfter.current;
    if (index === null) return;
    focusAfter.current = null;
    pointRefs.current[index]?.focus();
  });

  const path = hueCurvePath(shown, scale, HUE_PLOT_WIDTH, HUE_PLOT_HEIGHT);
  const ticks = axisTicks(scale);
  const identityY = (1 - valueToUnit(scale, identity)) * HUE_PLOT_HEIGHT;
  const chosen = selected !== null ? shown[selected] : undefined;

  return (
    <div className={cx("ym-curve ym-hue", className)} data-kind={kind}>
      <div className="ym-hue__body">
        <div className="ym-hue__yaxis" aria-hidden="true">
          {ticks.map((t) => (
            <span key={t.label} style={{ bottom: `${clamp(tickUnit(scale, t.value), 0, 1) * 100}%` }}>
              {t.label}
            </span>
          ))}
        </div>
        <svg
          ref={svg}
          className="ym-curve__plot ym-hue__plot"
          viewBox={`0 0 ${HUE_PLOT_WIDTH} ${HUE_PLOT_HEIGHT}`}
          role="group"
          aria-label={label}
          tabIndex={0}
          onPointerDown={onPointerDown}
          onKeyDown={(event) => {
            if (event.target === event.currentTarget && event.key === "Enter") {
              event.preventDefault();
              addAtWidestGap();
            }
          }}
        >
          <path
            className="ym-curve__grid"
            d={`${[1, 2, 3, 4, 5].map((k) => `M${(k * HUE_PLOT_WIDTH) / 6} 0V${HUE_PLOT_HEIGHT}`).join("")}${ticks.map((t) => `M0 ${(1 - clamp(tickUnit(scale, t.value), 0, 1)) * HUE_PLOT_HEIGHT}H${HUE_PLOT_WIDTH}`).join("")}`}
          />
          <path className="ym-hue__detent" d={`M0 ${identityY}H${HUE_PLOT_WIDTH}`} />
          <path className="ym-curve__line" d={path} style={{ stroke: "var(--curve-r)" }} />
          {shown.map((p, i) => {
            const isSelected = i === selected;
            const size = isSelected ? 12 : 10;
            const left = clamp((p.hue_deg / HUE_PERIOD) * HUE_PLOT_WIDTH - size / 2, 0, HUE_PLOT_WIDTH - size);
            const top = clamp((1 - valueToUnit(scale, p.value)) * HUE_PLOT_HEIGHT - size / 2, 0, HUE_PLOT_HEIGHT - size);
            const read = pointReadout(kind, p);
            return (
              <rect
                key={i}
                ref={(node) => {
                  pointRefs.current[i] = node;
                }}
                className="ym-curve__point ym-hue__point"
                data-selected={isSelected || undefined}
                x={left}
                y={top}
                width={size}
                height={size}
                rx={size / 2}
                tabIndex={0}
                role="slider"
                aria-label={`Point ${i + 1} of ${shown.length}`}
                aria-orientation="vertical"
                aria-valuemin={HUE_CURVES[kind].min}
                aria-valuemax={HUE_CURVES[kind].max}
                aria-valuenow={roundValue(kind, p.value)}
                aria-valuetext={`Hue ${read.hue}, ${read.label.toLowerCase()} ${read.value}`}
                onFocus={() => setSelected(i)}
                onDoubleClick={() => remove(i)}
                onKeyDown={(event) => onPointKeyDown(event, i)}
              />
            );
          })}
        </svg>
      </div>
      <div className="ym-hue__rainbow" style={{ background: hueGradient() }} aria-hidden="true" />
      <div className="ym-hue__xaxis" aria-hidden="true">
        {[0, 60, 120, 180, 240, 300, 360].map((hue) => (
          <span key={hue}>{hue}°</span>
        ))}
      </div>
      <div className="ym-hue__readout" aria-live="off">
        {chosen ? (
          <>
            <span>
              Hue <b>{pointReadout(kind, chosen).hue}</b>
            </span>
            <span>
              {pointReadout(kind, chosen).label} <b>{pointReadout(kind, chosen).value}</b>
            </span>
          </>
        ) : (
          <span>{points.length === 0 ? "No points. Click the plot to add one." : "Select a point."}</span>
        )}
      </div>
    </div>
  );
}
