import { useId, useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { cx } from "./cx";
import { clamp, clampSnap, roundToStep, sameValue } from "./format";
import { useDraft, useFrameEmitter, useLatest, usePointerDrag } from "./hooks";
import { ResetLink } from "./ResetLink";
import type { TabId } from "./tabs";
import { ValueBox } from "./ValueBox";
import type { ValueBoxProps } from "./ValueBox";

export interface SliderProps {
  /** Row label, 13 px 800 ("Exposure"). Also the slider's accessible name. */
  label: string;
  value: number;
  min: number;
  max: number;
  /** Default 1. Typed values, keys and drags all land on this grid. */
  step?: number;
  /**
   * The default value, which for an automatic setting is the resolved auto value. A 2 × 16
   * detent tick marks it, the fill bar runs from it to the thumb, and Reset and a double
   * click go back to it. Omit it for a slider with no default (Export's Quality): the fill
   * then runs from the left end and there is no reset.
   */
  defaultValue?: number;
  /** How the value reads in the box: "+0.30", "5450 K". */
  format?: (value: number) => string;
  /** Turns typed text into a number (see `ValueBox`). */
  parse?: ValueBoxProps["parse"];
  /**
   * Called with every new value. While the thumb is dragged it fires at most once per
   * animation frame with `final = false`, then once more with `final = true` when the pointer
   * is released. Keys, typing, Reset and double click call it once with `final = true`.
   */
  onChange: (value: number, final: boolean) => void;
  /**
   * What Reset and double click do, when going back to the default is not the same as setting
   * the default value (Exposure goes back to Auto, not to Manual(+0.30)). Without it they call
   * `onChange(defaultValue, true)`.
   */
  onReset?: () => void;
  /** Overrides the "differs from default" test (the fill, the box colour and the Reset link). */
  changed?: boolean;
  /** Tab whose colour fills the bar and the changed box. Inherited from `data-tab` when omitted. */
  tab?: TabId;
  disabled?: boolean;
  /** `false` drops the 40 px Reset column, as in the Export panel. Default true. */
  resetColumn?: boolean;
  id?: string;
  className?: string;
}

/**
 * Slider row (`docs/11-ui.md` §Components): grid `92px 1fr 72px 40px`, 38 px high. A 2 px
 * track, a 10 px fill bar in the tab colour with a 2 px ink border between the detent and the
 * value, a 2 × 16 detent tick at the default, an 8 × 22 ink thumb. A native
 * `<input type="range">` sits on top at opacity 0 for assistive technology and the keyboard.
 *
 * Keys: arrows step, Shift ×10, PageUp/PageDown ×10, Home/End to the limits. Double-click the
 * track to reset. The component is controlled; while a drag or a patch is in flight it shows
 * its own value so the thumb never lags behind the pointer.
 */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  defaultValue,
  format,
  parse,
  onChange,
  onReset,
  changed: changedProp,
  tab,
  disabled = false,
  resetColumn = true,
  id: idProp,
  className,
}: SliderProps) {
  const autoId = useId();
  const id = idProp ?? `ym-slider-${autoId}`;
  const control = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const [focusVisible, setFocusVisible] = useState(false);
  const draft = useDraft<number>(value, (a, b) => sameValue(a, b, step));
  const shown = draft.shown;
  const hasDefault = defaultValue !== undefined;
  const changed = changedProp ?? (hasDefault && !sameValue(shown, defaultValue, step));

  const onChangeRef = useLatest(onChange);
  const last = useRef(value);
  const frames = useFrameEmitter<number>((v) => onChangeRef.current(v, false));

  /** A finished change: keys, typing, Reset, release. */
  const commit = (next: number): void => {
    last.current = next;
    draft.settle(next);
    onChangeRef.current(next, true);
  };

  const valueAt = (clientX: number): number => {
    const rect = control.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return last.current;
    const t = clamp((clientX - rect.left) / rect.width, 0, 1);
    return clampSnap(min + t * (max - min), min, max, step);
  };

  const onPointerDown = usePointerDrag({
    start: (event: ReactPointerEvent<Element>) => {
      if (disabled) return false;
      input.current?.focus({ preventScroll: true });
      const next = valueAt(event.clientX);
      last.current = next;
      draft.drag(next);
      frames.push(next);
      return true;
    },
    move: (event: PointerEvent) => {
      const next = valueAt(event.clientX);
      if (next === last.current) return;
      last.current = next;
      draft.drag(next);
      frames.push(next);
    },
    end: () => {
      frames.cancel();
      commit(last.current);
    },
  });

  const reset = (): void => {
    if (disabled || !hasDefault) return;
    if (onReset) {
      onReset();
      draft.clear();
    } else {
      commit(defaultValue);
    }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (disabled || event.metaKey || event.ctrlKey || event.altKey) return;
    const unit = step * (event.shiftKey ? 10 : 1);
    let next: number;
    switch (event.key) {
      case "ArrowRight":
      case "ArrowUp":
        next = shown + unit;
        break;
      case "ArrowLeft":
      case "ArrowDown":
        next = shown - unit;
        break;
      case "PageUp":
        next = shown + step * 10;
        break;
      case "PageDown":
        next = shown - step * 10;
        break;
      case "Home":
        next = min;
        break;
      case "End":
        next = max;
        break;
      default:
        return;
    }
    event.preventDefault();
    const snapped = clampSnap(next, min, max, step);
    if (!sameValue(snapped, shown, step)) commit(snapped);
  };

  // The thumb and fill positions, in percent of the control.
  const span = max - min || 1;
  const pos = clamp(((shown - min) / span) * 100, 0, 100);
  const detent = hasDefault ? clamp(((defaultValue - min) / span) * 100, 0, 100) : null;
  const fill =
    detent === null
      ? { left: 0, width: pos }
      : { left: Math.min(pos, detent), width: Math.abs(pos - detent) };
  const showFill = detent === null || changed;

  return (
    <div
      className={cx("ym-slider", className)}
      data-tab={tab}
      data-changed={changed || undefined}
      data-disabled={disabled || undefined}
      data-reset-column={resetColumn ? undefined : "false"}
    >
      <label className="ym-slider__label" htmlFor={id}>
        {label}
      </label>
      <div ref={control} className="ym-slider__control" onPointerDown={onPointerDown} onDoubleClick={reset} data-focus-visible={focusVisible || undefined}>
        <div className="ym-slider__track" />
        {showFill ? <div className="ym-slider__fill" style={{ left: `${fill.left}%`, width: `${fill.width}%` }} /> : null}
        {detent !== null ? <div className="ym-slider__detent" style={{ left: `${detent}%` }} /> : null}
        <div className="ym-slider__thumb" style={{ left: `${pos}%` }} />
        <input
          ref={input}
          id={id}
          className="ym-slider__input"
          type="range"
          min={min}
          max={max}
          step={step}
          value={roundToStep(shown, step)}
          disabled={disabled}
          aria-valuetext={format ? format(shown) : undefined}
          onChange={(event: ChangeEvent<HTMLInputElement>) => {
            const next = clampSnap(Number(event.target.value), min, max, step);
            if (!sameValue(next, shown, step)) commit(next);
          }}
          onKeyDown={onKeyDown}
          onFocus={(event) => setFocusVisible(event.target.matches(":focus-visible"))}
          onBlur={() => setFocusVisible(false)}
        />
      </div>
      <ValueBox
        value={shown}
        min={min}
        max={max}
        step={step}
        format={format}
        parse={parse}
        changed={changed}
        onCommit={commit}
        label={`${label} value`}
        disabled={disabled}
      />
      {resetColumn ? (
        <div className="ym-slider__reset">
          {changed && !disabled ? <ResetLink aria-label={`Reset ${label}`} onClick={reset} /> : null}
        </div>
      ) : null}
    </div>
  );
}
