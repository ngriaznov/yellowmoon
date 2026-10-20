import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { KeyboardEvent, MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, RefObject } from "react";
import type { ChromaPoint, ColorWarpPin } from "../api/types";
import { cx } from "./cx";
import { useDraft, useFrameEmitter, useGestureCallback, useLatest, usePointerDrag } from "./hooks";
import {
  clampToPlane,
  hueAngle,
  hueFieldPixels,
  lockedPin,
  movingPin,
  pinValueText,
  planeHex,
  planeToPx,
  pxToPlane,
  WARP_MAX_PINS,
  WARP_PLANE_EXTENT,
} from "./warpMath";
import type { PlaneSpace } from "./warpMath";

/** What a press on empty plane makes: a moving pin (Move) or a locked one (Lock). */
export type WarpTool = "move" | "lock";

export interface WarpPlaneProps {
  /** The pins, in order: "Pin 1" is the first. */
  pins: readonly ColorWarpPin[];
  /**
   * Called with the new pin list. While a pin is dragged it fires at most once per animation frame with `final = false` (and once at
   * the press that adds a pin), then once with `final = true` on release. Keys and a double click call it once with `final = true`.
   */
  onChange: (pins: ColorWarpPin[], final: boolean) => void;
  /** The selected pin (an index into `pins`), or null. An index past the end counts as none. */
  selected: number | null;
  /** Called when a press, a new pin or the focus selects a pin, and with null when a press selects none. */
  onSelect: (index: number | null) => void;
  /** What a press on empty plane adds. Default `move`. */
  tool?: WarpTool;
  /** The photo's colours, `[a0, b0, a1, b1, …]` (`chroma_scatter`), drawn as faint dots. */
  scatter?: readonly number[] | null;
  /** The plane's width in CSS pixels until it has been measured (352 in the 400 px panel). */
  defaultSize?: number;
  className?: string;
}

/** One arrow-key step in chroma units; Shift multiplies it by ten (`docs/11` §Colour warp). */
export const WARP_KEY_STEP = 0.002;
/** How close, in CSS pixels, a press must be to a pin's dot or ring to take hold of it. */
const GRAB_PX = 9;
/** The faint rings of the plane, in chroma units. */
const RINGS = [0.1, 0.2, 0.3] as const;
/** Pixels of the drawn hue field per CSS pixel of the plane, and its bounds (it is a smooth field, scaled up by the browser). */
const FIELD_MIN = 128;
const FIELD_MAX = 384;

const samePoint = (a: ChromaPoint, b: ChromaPoint): boolean => Math.abs(a[0] - b[0]) < 1e-7 && Math.abs(a[1] - b[1]) < 1e-7;
const samePin = (a: ColorWarpPin, b: ColorWarpPin): boolean =>
  a.locked === b.locked &&
  samePoint(a.source, b.source) &&
  samePoint(a.target, b.target) &&
  a.range === b.range &&
  a.shadows === b.shadows &&
  a.highlights === b.highlights &&
  a.pivot_ev === b.pivot_ev &&
  a.exposure_ev === b.exposure_ev;
const samePins = (a: readonly ColorWarpPin[], b: readonly ColorWarpPin[]): boolean => a.length === b.length && a.every((p, i) => samePin(p, b[i] as ColorWarpPin));
const copyPin = (p: ColorWarpPin): ColorWarpPin => ({ ...p, source: [p.source[0], p.source[1]], target: [p.target[0], p.target[1]] });

/** A pin moved to `to`: a moving pin's target, or a locked pin's place (its source and target together). */
const movedTo = (pin: ColorWarpPin, to: ChromaPoint): ColorWarpPin =>
  pin.locked ? { ...pin, source: [to[0], to[1]], target: [to[0], to[1]] } : { ...pin, target: [to[0], to[1]] };

/** The hue fields drawn so far, by resolution and colour space: each is made once. */
const FIELDS = new Map<string, ImageData>();

/** Draws the hue field into its canvas: in Display P3 where the canvas has it, sRGB otherwise. */
function drawField(canvas: HTMLCanvasElement, resolution: number): void {
  const ctx = canvas.getContext("2d", { colorSpace: "display-p3" });
  if (!ctx) return;
  const space: PlaneSpace = ctx.getContextAttributes?.().colorSpace === "display-p3" ? "display-p3" : "srgb";
  const key = `${resolution}:${space}`;
  let image = FIELDS.get(key);
  if (!image) {
    image = new ImageData(hueFieldPixels(resolution, space), resolution, resolution, { colorSpace: space });
    FIELDS.set(key, image);
  }
  ctx.putImageData(image, 0, 0);
}

/** The element's width in CSS pixels, following resizes; `fallback` until it has one. */
function useWidth(ref: RefObject<HTMLElement | null>, fallback: number): number {
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = (): void => {
      const w = el.getBoundingClientRect().width;
      if (w > 0) setWidth(Math.round(w * 100) / 100);
    };
    read();
    const observer = new ResizeObserver(read);
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

interface DragState {
  index: number;
  list: ColorWarpPin[];
  /** Pointer to the pin's dot, in chroma units, so the dot keeps its place under the pointer. */
  grab: ChromaPoint;
  changed: boolean;
}

/**
 * The colour warp's chroma plane (`docs/11-ui.md` §Colour warp, `docs/02` C9b): a square as wide as its column, centred on neutral and
 * reaching `|c| = 0.30` at the middle of each edge, red (`+e1`) to the right and green against blue (`+e2`) up. It draws a hue field
 * (each point's colour at a bright grey, faded toward the dark ground at the centre and past the edge), faint rings at 0.1, 0.2 and 0.3
 * with the two axes, the photo's colours as faint dots, and the pins: a moving pin is a ring at its source, a line and a dot at its
 * target in the target's colour; a locked pin is a ring with a small square inside; the selected pin has a ring in the tab colour.
 *
 * Pointer: a press on a pin's dot or ring selects it and a drag moves its target (a lock: its place); a press on empty plane adds a
 * pin (a moving one with Move, a locked one with Lock) and the drag moves it; a double click puts a pin's target back on its source.
 * Sixteen pins at most: then a press on empty plane selects nothing. Keyboard: each pin is a `role="slider"` ("Pin 3", "moves red
 * toward orange"); the focused pin is the selected one; the arrows move it by 0.002 (Shift ×10), Home puts the target back on the
 * source, Delete or Backspace removes it.
 */
export function WarpPlane({ pins, onChange, selected, onSelect, tool = "move", scatter = null, defaultSize = 352, className }: WarpPlaneProps) {
  const root = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLCanvasElement>(null);
  const dots = useRef<HTMLCanvasElement>(null);
  const pinRefs = useRef<Array<SVGGElement | null>>([]);
  /** A pin to focus once it is drawn (-1: the plane itself). */
  const focusAfter = useRef<number | null>(null);
  const drag = useRef<DragState | null>(null);
  const size = useWidth(root, defaultSize);

  const draft = useDraft<readonly ColorWarpPin[]>(pins, samePins);
  const shown = draft.shown;
  // A drag reports to the `onChange` it began with (`useGestureCallback`).
  const onChangeRef = useGestureCallback(onChange);
  const onSelectRef = useLatest(onSelect);
  const frames = useFrameEmitter<ColorWarpPin[]>((list) => onChangeRef.current(list, false));
  const current = selected !== null && selected >= 0 && selected < shown.length ? selected : null;

  const commit = (list: ColorWarpPin[]): void => {
    draft.settle(list);
    onChangeRef.current(list, true);
  };

  // The hue field: made once per resolution (`FIELDS`), drawn again only when the plane's size changes.
  const resolution = Math.min(FIELD_MAX, Math.max(FIELD_MIN, Math.round(size)));
  useEffect(() => {
    if (field.current) drawField(field.current, resolution);
  }, [resolution]);

  // The photo's colours, at the device's pixels so the dots are sharp.
  const ratio = typeof window === "undefined" ? 1 : Math.min(2, window.devicePixelRatio || 1);
  const dotsPx = Math.round(size * ratio);
  useEffect(() => {
    const ctx = dots.current?.getContext("2d") ?? null;
    if (!ctx) return;
    ctx.clearRect(0, 0, dotsPx, dotsPx);
    if (!scatter || scatter.length < 2) return;
    ctx.fillStyle = "rgba(255, 255, 255, 0.42)";
    const r = 0.9 * ratio;
    for (let i = 0; i + 1 < scatter.length; i += 2) {
      const a = scatter[i] as number;
      const b = scatter[i + 1] as number;
      if (Math.abs(a) > WARP_PLANE_EXTENT || Math.abs(b) > WARP_PLANE_EXTENT) continue;
      const { x, y } = planeToPx([a, b], dotsPx);
      ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
    }
  }, [scatter, dotsPx, ratio]);

  // Put the focus on a pin once it is drawn: one just added or pressed, or the neighbour of one removed from the keyboard.
  useEffect(() => {
    const index = focusAfter.current;
    if (index === null) return;
    if (index < 0) {
      focusAfter.current = null;
      root.current?.focus({ preventScroll: true });
      return;
    }
    const el = pinRefs.current[index];
    if (!el) return;
    focusAfter.current = null;
    if (document.activeElement !== el) el.focus({ preventScroll: true });
  });

  /** The pointer on the plane: its chroma and its place in CSS pixels of the plane. */
  const locate = (event: { clientX: number; clientY: number }): { c: ChromaPoint; x: number; y: number } | null => {
    const rect = root.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    return { c: pxToPlane(x, y, rect.width), x, y };
  };

  /** The pin under a point of the plane: target dots first (the last drawn on top), then source rings. */
  const pinAt = (x: number, y: number, list: readonly ColorWarpPin[]): number | null => {
    const rect = root.current?.getBoundingClientRect();
    const width = rect && rect.width > 0 ? rect.width : size;
    const near = (c: ChromaPoint): boolean => {
      const p = planeToPx(clampToPlane(c), width);
      return Math.hypot(p.x - x, p.y - y) <= GRAB_PX;
    };
    for (let i = list.length - 1; i >= 0; i--) if (near((list[i] as ColorWarpPin).target)) return i;
    for (let i = list.length - 1; i >= 0; i--) if (near((list[i] as ColorWarpPin).source)) return i;
    return null;
  };

  const onPointerDown = usePointerDrag({
    start: (event: ReactPointerEvent<Element>) => {
      const at = locate(event);
      if (!at) return false;
      const list = shown.map(copyPin);
      let index = pinAt(at.x, at.y, list);
      const added = index === null;
      if (index === null) {
        // Sixteen pins at most: a press on empty plane selects nothing, and the card says why.
        if (list.length >= WARP_MAX_PINS) {
          onSelectRef.current(null);
          return false;
        }
        const point = clampToPlane(at.c);
        list.push(tool === "lock" ? lockedPin(point) : movingPin(point));
        index = list.length - 1;
      }
      const pin = list[index] as ColorWarpPin;
      onChangeRef.begin();
      drag.current = { index, list, grab: [pin.target[0] - at.c[0], pin.target[1] - at.c[1]], changed: added };
      focusAfter.current = index;
      onSelectRef.current(index);
      if (added) {
        // The new pin shows at once, and the card's rows with it.
        draft.drag(list);
        onChangeRef.current(list, false);
      }
      return true;
    },
    move: (event: PointerEvent) => {
      const state = drag.current;
      const at = locate(event);
      if (!state || !at) return;
      const pin = state.list[state.index] as ColorWarpPin;
      const to = clampToPlane([at.c[0] + state.grab[0], at.c[1] + state.grab[1]]);
      if (samePoint(to, pin.target)) return;
      const list = state.list.slice();
      list[state.index] = movedTo(pin, to);
      state.list = list;
      state.changed = true;
      draft.drag(list);
      frames.push(list);
    },
    end: () => {
      frames.cancel();
      const state = drag.current;
      drag.current = null;
      if (state?.changed) commit(state.list);
      onChangeRef.end();
    },
  });

  const onDoubleClick = (event: ReactMouseEvent<HTMLDivElement>): void => {
    const at = locate(event);
    if (!at) return;
    const index = pinAt(at.x, at.y, shown);
    const pin = index === null ? undefined : shown[index];
    if (index === null || !pin || pin.locked || samePoint(pin.source, pin.target)) return;
    const list = shown.map(copyPin);
    list[index] = { ...pin, target: [pin.source[0], pin.source[1]] };
    commit(list);
  };

  const onPinKeyDown = (event: KeyboardEvent<SVGGElement>, index: number): void => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const pin = shown[index];
    if (!pin) return;
    const unit = WARP_KEY_STEP * (event.shiftKey ? 10 : 1);
    const [a, b] = pin.target;
    let next: ColorWarpPin | null = null;
    switch (event.key) {
      case "ArrowRight":
        next = movedTo(pin, clampToPlane([a + unit, b]));
        break;
      case "ArrowLeft":
        next = movedTo(pin, clampToPlane([a - unit, b]));
        break;
      case "ArrowUp":
        next = movedTo(pin, clampToPlane([a, b + unit]));
        break;
      case "ArrowDown":
        next = movedTo(pin, clampToPlane([a, b - unit]));
        break;
      case "Home":
        next = pin.locked ? pin : { ...pin, target: [pin.source[0], pin.source[1]] };
        break;
      case "Delete":
      case "Backspace": {
        event.preventDefault();
        event.stopPropagation();
        const list = shown.filter((_, i) => i !== index).map(copyPin);
        const neighbour = list.length === 0 ? null : Math.max(0, index - 1);
        focusAfter.current = neighbour ?? -1;
        onSelectRef.current(neighbour);
        commit(list);
        return;
      }
      default:
        return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (samePin(next, pin)) return;
    const list = shown.map(copyPin);
    list[index] = next;
    commit(list);
  };

  const scale = size / 2 / WARP_PLANE_EXTENT;
  const middle = size / 2;

  return (
    <div
      ref={root}
      className={cx("ym-warp-plane", className)}
      role="group"
      aria-label="Colour warp plane"
      tabIndex={-1}
      data-tool={tool}
      data-scatter-points={scatter ? Math.floor(scatter.length / 2) : 0}
      onPointerDown={onPointerDown}
      // The press must not take the focus to the plane: the pin it selects takes it (`focusAfter`).
      onMouseDown={(event) => event.preventDefault()}
      onDoubleClick={onDoubleClick}
    >
      <canvas ref={field} className="ym-warp-plane__field" width={resolution} height={resolution} aria-hidden="true" />
      <canvas ref={dots} className="ym-warp-plane__scatter" width={dotsPx} height={dotsPx} aria-hidden="true" />
      <svg className="ym-warp-plane__overlay" viewBox={`0 0 ${size} ${size}`}>
        <g aria-hidden="true">
          {RINGS.map((r) => (
            <circle key={r} className="ym-warp-plane__ring" cx={middle} cy={middle} r={r * scale} />
          ))}
          <path className="ym-warp-plane__axis" d={`M${middle} 0V${size}M0 ${middle}H${size}`} />
        </g>
        {shown.map((pin, i) => {
          const s = planeToPx(clampToPlane(pin.source), size);
          const t = planeToPx(clampToPlane(pin.target), size);
          const isSelected = i === current;
          return (
            <g
              key={i}
              ref={(node) => {
                pinRefs.current[i] = node;
              }}
              className="ym-warp-plane__pin"
              data-locked={pin.locked || undefined}
              data-selected={isSelected || undefined}
              role="slider"
              tabIndex={0}
              aria-label={`Pin ${i + 1}`}
              aria-valuemin={0}
              aria-valuemax={360}
              aria-valuenow={Math.round(hueAngle(pin.target)) % 360}
              aria-valuetext={pinValueText(pin)}
              onFocus={() => {
                if (i !== current) onSelectRef.current(i);
              }}
              onKeyDown={(event) => onPinKeyDown(event, i)}
            >
              {pin.locked ? (
                <>
                  {isSelected ? <circle className="ym-warp-plane__halo" cx={s.x} cy={s.y} r={9} /> : null}
                  <circle className="ym-warp-plane__lock" cx={s.x} cy={s.y} r={4} />
                  <rect className="ym-warp-plane__lock-mark" x={s.x - 2} y={s.y - 2} width={4} height={4} />
                </>
              ) : (
                <>
                  <line className="ym-warp-plane__line" x1={s.x} y1={s.y} x2={t.x} y2={t.y} />
                  <circle className="ym-warp-plane__source" cx={s.x} cy={s.y} r={4} />
                  {isSelected ? <circle className="ym-warp-plane__halo" cx={t.x} cy={t.y} r={9} /> : null}
                  <circle className="ym-warp-plane__target" cx={t.x} cy={t.y} r={4.25} style={{ fill: planeHex(pin.target) }} />
                </>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
