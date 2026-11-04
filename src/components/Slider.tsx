import { memo, useCallback, useEffect, useId, useRef } from "react";
import type { MutableRefObject } from "react";
import { Slider as GlassSlider } from "@/components/glass/slider";
import { cx } from "./cx";
import { clampSnap, roundToStep, sameValue } from "./format";
import { holdGesture } from "../state/gestureHold";
import { useDraft, useFrameEmitter, useGestureCallback } from "./hooks";
import type { GestureCallback } from "./hooks";
import { ResetLink } from "./ResetLink";
import type { TabId } from "./tabs";
import { ValueBox } from "./ValueBox";
import type { ValueBoxProps } from "./ValueBox";

/** The keys Radix's slider acts on. */
const STEP_KEYS: ReadonlySet<string> = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]);

export interface SliderProps {
  /** Row label, 13 px 800 ("Exposure"). Also the slider's accessible name. */
  label: string;
  value: number;
  min: number;
  max: number;
  /** Default 1. Typed values, keys and drags all land on this grid. */
  step?: number;
  /**
   * The default value, which for an automatic setting is the resolved auto value. The accent
   * range runs from it to the knob, and Reset and a double click go back to it. Omit it for a slider with no default (Export's Quality): the fill
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

/** A stable function that calls the newest `fn` (set while rendering, so a call from an effect of the same commit sees it). Undefined stays undefined. */
function useProxy<A extends unknown[], R>(fn: ((...args: A) => R) | undefined): ((...args: A) => R) | undefined {
  const ref: MutableRefObject<typeof fn> = useRef(fn);
  ref.current = fn;
  const proxy = useCallback((...args: A): R => (ref.current as (...a: A) => R)(...args), []);
  return fn ? proxy : undefined;
}

/**
 * Slider row: a 92 px label, glasscn's slider (a soft track, an accent range that grows from the default, its small round 16 px knob that
 * turns into a lens while dragged), a 72 px value box and a 40 px Reset column, 38 px high. The knob is Radix's `role="slider"`, named by the label.
 *
 * Keys (Radix): arrows step, Shift+arrow or PageUp/PageDown ×10, Home/End to the limits; the same keys with ⌘, ⌃ or ⌥ are left alone. Double-click the track to reset. The component is
 * controlled; while a drag or a patch is in flight it shows its own value so the knob never lags behind the pointer.
 *
 * Speed: the callbacks a caller passes are new functions at every render of the caller, and a tab draws again at every tick of a drag.
 * This outer part is cheap and keeps the newest callbacks behind stable ones; the row itself (`SliderRow`) is memoised on the data, so
 * only the row that moves draws (U-02).
 */
export function Slider({ onChange, onReset, format, parse, ...data }: SliderProps) {
  const gesture = useGestureCallback(onChange);
  const resetFn = useProxy(onReset);
  const formatFn = useProxy(format);
  const parseFn = useProxy(parse);
  // The row does not see a new `format`, so what it would print for the ends and the value decides when it draws again.
  const formatKey = format ? `${format(data.value)}|${data.defaultValue === undefined ? "" : format(data.defaultValue)}|${format(data.min)}|${format(data.max)}` : "";
  return <SliderRow {...data} gesture={gesture} resetFn={resetFn} formatFn={formatFn} parseFn={parseFn} formatKey={formatKey} />;
}

type SliderRowProps = Omit<SliderProps, "onChange" | "onReset" | "format" | "parse"> & {
  gesture: GestureCallback<SliderProps["onChange"]>;
  resetFn: (() => void) | undefined;
  formatFn: ((value: number) => string) | undefined;
  parseFn: ValueBoxProps["parse"];
  formatKey: string;
};

const SliderRow = memo(function SliderRow({
  label,
  value,
  min,
  max,
  step = 1,
  defaultValue,
  formatFn: format,
  parseFn: parse,
  gesture: onChangeRef,
  resetFn: onReset,
  formatKey,
  changed: changedProp,
  tab,
  disabled = false,
  resetColumn = true,
  id: idProp,
  className,
}: SliderRowProps) {
  const autoId = useId();
  const id = idProp ?? `ym-slider-${autoId}`;
  const labelId = `${id}-label`;
  const control = useRef<HTMLDivElement>(null);
  const draft = useDraft<number>(value, (a, b) => sameValue(a, b, step));
  const shown = draft.shown;
  const hasDefault = defaultValue !== undefined;
  const changed = changedProp ?? (hasDefault && !sameValue(shown, defaultValue, step));

  // A drag reports to the `onChange` it began with, so the frame and the release after it cannot reach another photo (`useGestureCallback`).
  const last = useRef(value);
  const pointerDown = useRef(false);
  // While the knob is held the histogram, the readout and the scatter are not asked for (`holdGesture`).
  const hold = useRef<(() => void) | null>(null);
  const endHold = (): void => {
    hold.current?.();
    hold.current = null;
  };
  useEffect(() => endHold, []);
  const moved = useRef(false);
  const frames = useFrameEmitter<number>((v) => onChangeRef.current(v, false));

  /** A finished change: keys, typing, Reset, release. */
  const commit = (next: number): void => {
    last.current = next;
    draft.settle(next);
    onChangeRef.current(next, true);
  };

  const reset = (): void => {
    if (disabled || !hasDefault) return;
    if (onReset) {
      onReset();
      draft.clear();
    } else {
      commit(defaultValue);
    }
  };

  // Radix's knob has no `aria-valuetext` prop here, so the readable value ("+0.30 EV") is put on it.
  useEffect(() => {
    if (!format) return;
    control.current?.querySelector('[role="slider"]')?.setAttribute("aria-valuetext", format(shown));
  }, [format, formatKey, shown]);

  const onValueChange = ([raw]: number[]): void => {
    if (disabled || raw === undefined) return;
    const next = clampSnap(raw, min, max, step);
    if (pointerDown.current) {
      // A drag: at most one call per frame, and a final one on release.
      if (next === last.current) return;
      moved.current = true;
      last.current = next;
      draft.drag(next);
      frames.push(next);
    } else if (!sameValue(next, shown, step)) {
      // A key: one finished change.
      commit(next);
    }
  };

  const release = (): void => {
    if (!pointerDown.current) return;
    pointerDown.current = false;
    frames.cancel();
    if (moved.current) commit(last.current);
    moved.current = false;
    onChangeRef.end();
    // After the final change is sent, so the extras the release asks for come after it in the core's queue.
    endHold();
  };

  return (
    <div
      className={cx(
        "grid min-h-[38px] items-center gap-x-3",
        resetColumn ? "grid-cols-[92px_minmax(0,1fr)_72px_40px]" : "grid-cols-[92px_minmax(0,1fr)_72px]",
        className,
      )}
      data-slider=""
      data-tab={tab}
      data-changed={changed || undefined}
      data-disabled={disabled || undefined}
      data-reset-column={resetColumn ? undefined : "false"}
    >
      <span id={labelId} className="text-[0.8125rem] font-semibold leading-tight">
        {label}
      </span>
      <div
        ref={control}
        className="px-1"
        onDoubleClick={reset}
        // Radix steps the value on ⌘, ⌃ and ⌥ with the arrows, Home, End and Page keys as well; those combinations belong to the app and the system.
        onKeyDownCapture={(event) => {
          if ((event.metaKey || event.ctrlKey || event.altKey) && STEP_KEYS.has(event.key)) event.stopPropagation();
        }}
        onPointerDownCapture={(event) => {
          // Only the primary button drags. Radix's slider takes any button, so a right click (or a two-finger click) on the track would
          // move the knob and end as a finished change; stopping the event here keeps Radix's own handler from seeing it.
          if (event.button !== 0) {
            event.stopPropagation();
            return;
          }
          if (disabled) return;
          pointerDown.current = true;
          endHold();
          hold.current = holdGesture();
          moved.current = false;
          last.current = shown;
          onChangeRef.begin();
        }}
        onPointerUp={release}
        onPointerCancel={release}
      >
        <GlassSlider
          aria-labelledby={labelId}
          min={min}
          max={max}
          step={step}
          value={[roundToStep(shown, step)]}
          neutralValue={hasDefault ? defaultValue : undefined}
          size="sm"
          disabled={disabled}
          onValueChange={onValueChange}
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
        <div className="flex justify-end">{changed && !disabled ? <ResetLink aria-label={`Reset ${label}`} onClick={reset} /> : null}</div>
      ) : null}
    </div>
  );
});
