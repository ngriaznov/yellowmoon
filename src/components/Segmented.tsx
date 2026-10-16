import type { KeyboardEvent } from "react";
import { SegmentedControl, SegmentedControlItem } from "@/components/glass/segmented-control";
import { cx } from "./cx";
import type { TabId } from "./tabs";

/** One option of a `Segmented`. */
export interface SegmentedOption<T extends string> {
  value: T;
  /** Text ("This photo", "Picked · 3"). */
  label: string;
  disabled?: boolean;
  /** Longer accessible name when the label is short ("Tung." for tungsten). */
  ariaLabel?: string;
}

export interface SegmentedProps<T extends string> {
  options: ReadonlyArray<SegmentedOption<T>>;
  value: T;
  onChange: (value: T) => void;
  /** Accessible name of the group ("Light type", "Apply look to"). */
  label: string;
  /** Option height: `sm`, `md` (default) or `lg`. */
  size?: "sm" | "md" | "lg";
  /** Columns; defaults to one per option. */
  columns?: number;
  /** Tab whose colour the thumb follows; normally inherited. */
  tab?: TabId;
  /** Sized by its labels instead of stretched across the row (a control at the end of a Preferences row). */
  fit?: boolean;
  className?: string;
}

/**
 * glasscn's segmented control: a soft track and one raised thumb that slides to the chosen segment (drag it to scrub). The group keeps
 * `role="radiogroup"` with a `role="radio"` per segment; arrows move the selection, Home and End jump to the ends.
 */
export function Segmented<T extends string>({ options, value, onChange, label, size = "md", columns, tab, fit = false, className }: SegmentedProps<T>) {
  // A radio group selects as the arrows move; Radix's toggle group only moves the focus. Caught before Radix sees the key.
  const onKeyDownCapture = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"];
    if (!keys.includes(event.key)) return;
    const enabled = options.map((o, i) => (o.disabled ? -1 : i)).filter((i) => i >= 0);
    if (enabled.length === 0) return;
    event.preventDefault();
    event.stopPropagation();
    const current = enabled.indexOf(Math.max(0, options.findIndex((o) => o.value === value)));
    const at = Math.max(0, current);
    const next =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? enabled.length - 1
          : (at + (event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1) + enabled.length) % enabled.length;
    const target = options[enabled[next] ?? 0];
    if (!target) return;
    const group = event.currentTarget;
    if (target.value !== value) onChange(target.value);
    requestAnimationFrame(() => group.querySelector<HTMLElement>(`[data-value="${target.value}"]`)?.focus());
  };
  return (
    <SegmentedControl
      role="radiogroup"
      aria-label={label}
      value={value}
      onValueChange={(next) => onChange(next as T)}
      size={size === "md" ? "default" : size}
      data-tab={tab}
      data-columns={columns}
      onKeyDownCapture={onKeyDownCapture}
      className={cx(fit ? "w-fit" : "w-full", className)}
    >
      {options.map((option) => (
        <SegmentedControlItem key={option.value} value={option.value} disabled={option.disabled} aria-label={option.ariaLabel} className={cx("px-2 text-[0.8125rem]", fit && "min-w-max flex-none px-3")}>
          {option.label}
        </SegmentedControlItem>
      ))}
    </SegmentedControl>
  );
}
