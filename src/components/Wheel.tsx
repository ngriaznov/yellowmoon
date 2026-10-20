import { useRef } from "react";
import type { CSSProperties, KeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { cx } from "./cx";
import { useDraft, useFrameEmitter, useGestureCallback, usePointerDrag } from "./hooks";
import type { TabId } from "./tabs";
import { angleOf, clampToDisc, isCentered, pointFromPointer, puckColor, strengthOf, WHEEL_GRADIENT } from "./wheelMath";
import type { ColorBalancePoint } from "./wheelMath";

export interface WheelProps {
  /** "Shadows", "Midtones", "Highlights": drawn under the wheel and in its accessible name. */
  label: string;
  /** The point in the unit disc: `a` to the right, `b` upwards. `[0, 0]` is the default. */
  value: ColorBalancePoint;
  /**
   * Called with a new point. During a drag it fires at most once per animation frame with
   * `final = false`, then once with `final = true` on release. Keys and double click call it
   * once with `final = true`.
   */
  onChange: (value: ColorBalancePoint, final: boolean) => void;
  /**
   * Where Reset (double-click, Home, Delete, Backspace) puts the puck: the wheel's Automatic point (`docs/19` section 5). `[0, 0]`
   * when not given.
   */
  defaultValue?: ColorBalancePoint;
  /** Called by Reset instead of `onChange(defaultValue, true)`: a wheel that is Automatic sends Auto, not a point (`docs/11`). */
  onReset?: () => void;
  /** Outer diameter in CSS pixels, border included. The mockup's wheel is 92 (88 inside the border). */
  size?: number;
  disabled?: boolean;
  tab?: TabId;
  className?: string;
}

/**
 * The puck's size in pixels: a small round point, so it covers little of the wheel and travels further (the owner's request of
 * 2026-10-17; it was 16 px). Set on the element here, so the travel below and the drawn size cannot disagree.
 */
const PUCK = 10;
const PUCK_HALF = PUCK / 2;
/** The disc's rim, 1 px in the glass interface (`glass-c2-pages.css`). */
const BORDER = 1;
const sameSpot = (a: ColorBalancePoint, b: ColorBalancePoint): boolean =>
  Math.abs(a[0] - b[0]) < 1e-4 && Math.abs(a[1] - b[1]) < 1e-4;

/** One step of an arrow key, in unit-disc units; Shift multiplies it by ten. */
export const WHEEL_KEY_STEP = 0.02;

/**
 * Colour balance wheel (`docs/11-ui.md` §Components): a 92 px disc (90 px inside its 1 px
 * rim) with a hue gradient fading to the ground at the centre, faint cross hairs and a small round
 * 10 px puck. Drag the puck; double-click resets it (to the centre, or to `defaultValue`, the
 * Automatic point). The puck takes the colour of its direction. Keyboard: arrows move the puck
 * (Shift ×10), Home, Delete or Backspace reset it.
 * It is exposed as a slider whose value is the strength (0–100 %) and whose text gives
 * the strength and the hue angle.
 */
export function Wheel({ label, value, onChange, defaultValue, onReset, size = 92, disabled = false, tab, className }: WheelProps) {
  const disc = useRef<HTMLDivElement>(null);
  const draft = useDraft<ColorBalancePoint>(value, sameSpot);
  const shown = draft.shown;
  // A drag reports to the `onChange` it began with (`useGestureCallback`).
  const onChangeRef = useGestureCallback(onChange);
  const last = useRef<ColorBalancePoint>(value);
  const frames = useFrameEmitter<ColorBalancePoint>((v) => onChangeRef.current(v, false));

  const inner = size - BORDER * 2;
  const travel = inner / 2 - PUCK_HALF;

  const commit = (next: ColorBalancePoint): void => {
    last.current = next;
    draft.settle(next);
    onChangeRef.current(next, true);
  };

  const home: ColorBalancePoint = defaultValue ?? [0, 0];
  /** Reset: the puck to the default point, sent as Reset when the wheel has one, else as a point. */
  const reset = (): void => {
    if (onReset) {
      last.current = home;
      draft.settle(home);
      onReset();
    } else if (!sameSpot(shown, home)) {
      commit(home);
    }
  };

  const pointAt = (event: { clientX: number; clientY: number }): ColorBalancePoint => {
    const rect = disc.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return last.current;
    return pointFromPointer(event.clientX, event.clientY, rect, travel * (rect.width / size));
  };

  const onPointerDown = usePointerDrag({
    start: (event: ReactPointerEvent<Element>) => {
      if (disabled) return false;
      disc.current?.focus({ preventScroll: true });
      onChangeRef.begin();
      const next = pointAt(event);
      last.current = next;
      draft.drag(next);
      frames.push(next);
      return true;
    },
    move: (event: PointerEvent) => {
      const next = pointAt(event);
      last.current = next;
      draft.drag(next);
      frames.push(next);
    },
    end: () => {
      frames.cancel();
      commit(last.current);
      onChangeRef.end();
    },
  });

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (disabled || event.metaKey || event.ctrlKey || event.altKey) return;
    const unit = WHEEL_KEY_STEP * (event.shiftKey ? 10 : 1);
    const [a, b] = shown;
    let next: ColorBalancePoint;
    switch (event.key) {
      case "ArrowRight":
        next = clampToDisc(a + unit, b);
        break;
      case "ArrowLeft":
        next = clampToDisc(a - unit, b);
        break;
      case "ArrowUp":
        next = clampToDisc(a, b + unit);
        break;
      case "ArrowDown":
        next = clampToDisc(a, b - unit);
        break;
      case "Home":
      case "Delete":
      case "Backspace":
        event.preventDefault();
        reset();
        return;
      default:
        return;
    }
    event.preventDefault();
    if (!sameSpot(next, shown)) commit(next);
  };

  const strengthPct = Math.round(strengthOf(shown) * 100);
  const valueText = isCentered(shown) ? "Centred" : `Strength ${strengthPct}%, hue ${Math.round(angleOf(shown))} degrees`;
  const left = inner / 2 + shown[0] * travel - PUCK_HALF;
  const top = inner / 2 - shown[1] * travel - PUCK_HALF;

  return (
    <div className={cx("ym-wheel", className)} data-disabled={disabled || undefined} data-tab={tab}>
      <div
        ref={disc}
        className="ym-wheel__disc"
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label={`${label} colour balance`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={strengthPct}
        aria-valuetext={valueText}
        aria-disabled={disabled || undefined}
        style={{ "--ym-wheel-size": `${size}px`, background: WHEEL_GRADIENT } as CSSProperties}
        onPointerDown={onPointerDown}
        onDoubleClick={() => {
          if (!disabled) reset();
        }}
        onKeyDown={onKeyDown}
      >
        <div className="ym-wheel__fade" />
        <div className="ym-wheel__axis-v" />
        <div className="ym-wheel__axis-h" />
        <div className="ym-wheel__puck" style={{ left, top, width: PUCK, height: PUCK, background: puckColor(shown) }} />
      </div>
      <span className="ym-wheel__label">{label}</span>
    </div>
  );
}

/** The three-column grid the wheels sit in (Shadows, Midtones, Highlights). */
export function WheelGroup({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx("ym-wheels", className)}>{children}</div>;
}
