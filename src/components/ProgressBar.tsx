import { cx } from "./cx";
import { clamp } from "./format";

export interface ProgressBarProps {
  /** 0–100. */
  value: number;
  /** Accessible name: "Developing photos", "Exporting". */
  label: string;
  /** `md` is 24 px (Start, Export), `lg` is 28 px (the wizard's import step). */
  size?: "md" | "lg";
  className?: string;
}

/**
 * The progress bar of Start, Export and the wizard: a 3 px ink box with a brand yellow fill
 * that has a 3 px ink edge while it is between 0 and 100 %. `role="progressbar"`.
 */
export function ProgressBar({ value, label, size = "md", className }: ProgressBarProps) {
  const pct = clamp(Math.round(value), 0, 100);
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      className={cx("ym-progress", className)}
      data-size={size === "lg" ? "lg" : undefined}
      data-partial={pct > 0 && pct < 100 ? "true" : undefined}
    >
      <div className="ym-progress__fill" style={{ width: `${pct}%` }} />
    </div>
  );
}
