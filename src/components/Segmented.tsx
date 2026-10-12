import { useRef } from "react";
import type { CSSProperties, KeyboardEvent } from "react";
import { cx } from "./cx";
import type { TabId } from "./tabs";

/** One option of a `Segmented`. */
export interface SegmentedOption<T extends string> {
  value: T;
  /** Text, drawn in capitals ("This photo", "Picked · 3"). */
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
  /** Option height: `sm` 30 px, `md` 32 px (default), `lg` 36 px. */
  size?: "sm" | "md" | "lg";
  /** Columns; defaults to one per option. */
  columns?: number;
  /** Tab whose colour the focus ring follows; normally inherited. */
  tab?: TabId;
  className?: string;
}

/**
 * Segmented control: an ink background with 2 px gaps, 11 px 800 caps options, the active one
 * black with white text. `role="radiogroup"`; one option is in the tab order, arrows move the
 * selection (and wrap), Home and End jump to the ends.
 */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
  size = "md",
  columns,
  tab,
  className,
}: SegmentedProps<T>) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const activeIndex = options.findIndex((o) => o.value === value);
  const tabbable = activeIndex >= 0 ? activeIndex : options.findIndex((o) => !o.disabled);

  const move = (from: number, direction: 1 | -1 | "first" | "last"): void => {
    const n = options.length;
    const enabled = (i: number): boolean => !options[i]?.disabled;
    let index = from;
    if (direction === "first" || direction === "last") {
      index = direction === "first" ? 0 : n - 1;
      const step = direction === "first" ? 1 : -1;
      while (index >= 0 && index < n && !enabled(index)) index += step;
    } else {
      for (let tries = 0; tries < n; tries++) {
        index = (index + direction + n) % n;
        if (enabled(index)) break;
      }
    }
    const target = options[index];
    if (!target || target.disabled || index === from) return;
    onChange(target.value);
    refs.current[index]?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number): void => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    switch (event.key) {
      case "ArrowRight":
      case "ArrowDown":
        event.preventDefault();
        move(index, 1);
        break;
      case "ArrowLeft":
      case "ArrowUp":
        event.preventDefault();
        move(index, -1);
        break;
      case "Home":
        event.preventDefault();
        move(index, "first");
        break;
      case "End":
        event.preventDefault();
        move(index, "last");
        break;
    }
  };

  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cx("ym-segmented", className)}
      data-size={size === "md" ? undefined : size}
      data-tab={tab}
      style={{ "--ym-cols": columns ?? options.length } as CSSProperties}
    >
      {options.map((option, index) => (
        <button
          key={option.value}
          ref={(node) => {
            refs.current[index] = node;
          }}
          type="button"
          role="radio"
          className="ym-segmented__option"
          aria-checked={option.value === value}
          aria-disabled={option.disabled || undefined}
          aria-label={option.ariaLabel}
          tabIndex={index === tabbable ? 0 : -1}
          onClick={() => {
            if (!option.disabled && option.value !== value) onChange(option.value);
          }}
          onKeyDown={(event) => onKeyDown(event, index)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
