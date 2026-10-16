import { Progress } from "@/components/glass/progress";
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
  return <Progress aria-label={label} value={pct} data-size={size === "lg" ? "lg" : undefined} className={cx(size === "lg" && "h-3.5", className)} />;
}
