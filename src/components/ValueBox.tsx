import { useLayoutEffect, useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent, MouseEvent } from "react";
import { cx } from "./cx";
import { clampSnap, editText, parseNumber, roundToStep, sameValue } from "./format";
import type { TabId } from "./tabs";

export interface ValueBoxProps {
  /** The value to show. */
  value: number;
  min: number;
  max: number;
  /** Step of the value grid; typed values snap to it. Default 1. */
  step?: number;
  /** How the value reads when not being edited: "+0.30", "5450 K". Default: the number as is. */
  format?: (value: number) => string;
  /** Turns typed text into a number, or null to cancel. Default `parseNumber`. */
  parse?: (text: string) => number | null;
  /** True when the value differs from its default: the box takes the tab colour. */
  changed?: boolean;
  /** Called with the clamped, snapped value when the person commits a different value. */
  onCommit: (value: number) => void;
  /** Accessible name: "Exposure value". */
  label: string;
  /** Tab whose colour a changed box takes. Inherited from `data-tab` above when omitted. */
  tab?: TabId;
  disabled?: boolean;
  id?: string;
  className?: string;
}

/**
 * The 72 × 30 number box beside a slider. 2 px border, 14 px 800 tabular numbers; paper at
 * the default, tab colour when changed. Click to type: the text is selected, Return commits,
 * Esc cancels, leaving the box commits. Out-of-range values clamp and snap to the step. ↑/↓
 * nudge the typed value by one step (Shift: ten).
 */
export function ValueBox({
  value,
  min,
  max,
  step = 1,
  format,
  parse = parseNumber,
  changed = false,
  onCommit,
  label,
  tab,
  disabled = false,
  id,
  className,
}: ValueBoxProps) {
  const [text, setText] = useState<string | null>(null);
  // Mirrors `text` synchronously so a blur that follows Return or Esc sees the finished edit.
  const editing = useRef<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const justFocused = useRef(false);

  const setEditing = (next: string | null): void => {
    editing.current = next;
    setText(next);
  };

  // Select everything once the editing text is on screen.
  const isEditing = text !== null;
  useLayoutEffect(() => {
    if (isEditing && justFocused.current) input.current?.select();
  }, [isEditing]);

  const finish = (apply: boolean): void => {
    const typed = editing.current;
    setEditing(null);
    if (!apply || typed === null) return;
    const parsed = parse(typed);
    if (parsed === null) return;
    const next = clampSnap(parsed, min, max, step);
    if (!sameValue(next, value, step)) onCommit(next);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true);
      event.currentTarget.blur();
    } else if (event.key === "Escape") {
      event.preventDefault();
      finish(false);
      event.currentTarget.blur();
    } else if ((event.key === "ArrowUp" || event.key === "ArrowDown") && editing.current !== null) {
      event.preventDefault();
      const typed = parse(editing.current) ?? value;
      const delta = (event.key === "ArrowUp" ? 1 : -1) * step * (event.shiftKey ? 10 : 1);
      setEditing(editText(clampSnap(typed + delta, min, max, step), step));
      requestAnimationFrame(() => input.current?.select());
    }
  };

  return (
    <input
      ref={input}
      id={id}
      type="text"
      inputMode="decimal"
      autoComplete="off"
      spellCheck={false}
      aria-label={label}
      className={cx("ym-value-box", className)}
      data-tab={tab}
      data-changed={changed || undefined}
      data-editing={isEditing || undefined}
      disabled={disabled}
      value={text ?? (format ? format(value) : String(roundToStep(value, step)))}
      onFocus={() => {
        justFocused.current = true;
        setEditing(editText(value, step));
      }}
      onMouseUp={(event: MouseEvent<HTMLInputElement>) => {
        // Safari and Chromium clear a selection made on focus when the button is released.
        if (justFocused.current) event.preventDefault();
        justFocused.current = false;
      }}
      onChange={(event: ChangeEvent<HTMLInputElement>) => setEditing(event.target.value)}
      onKeyDown={onKeyDown}
      onBlur={() => {
        justFocused.current = false;
        finish(true);
      }}
    />
  );
}
