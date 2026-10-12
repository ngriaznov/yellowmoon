import type { ReactNode } from "react";
import { cx } from "./cx";

/** Per-channel histogram counts. Any bin count of 2 or more; any scale (they are normalised together). */
export interface HistogramData {
  red: readonly number[];
  green: readonly number[];
  blue: readonly number[];
}

/** Path of a filled histogram in a `width` × `height` box, y up. `max` is the count that reaches the top. */
export function histogramPath(bins: readonly number[], max: number, width = 320, height = 84, headroom = 1): string {
  const n = bins.length;
  if (n < 2 || !(max > 0)) return "";
  const point = (i: number): string => {
    const x = (i / (n - 1)) * width;
    const y = height - Math.min(1, (bins[i] as number) / max) * height * headroom;
    return `${Math.round(x * 100) / 100} ${Math.round(y * 100) / 100}`;
  };
  let d = `M0 ${height} L${point(0)}`;
  for (let i = 1; i < n; i++) d += ` L${point(i)}`;
  return `${d} L${width} ${height} Z`;
}

/** Largest count across the given bins. */
export const histogramMax = (...series: ReadonlyArray<readonly number[]>): number =>
  Math.max(0, ...series.map((bins) => Math.max(0, ...bins)));

export interface HistogramProps {
  data: HistogramData;
  /** Text under the box at the left ("R 182 G 164 B 140", the pixel under the pointer). */
  left?: ReactNode;
  /** Text under the box at the right ("ISO 400 · 1/125 · f/2.8"). */
  right?: ReactNode;
  /** Accessible name. Default "Histogram". */
  label?: string;
  className?: string;
}

/**
 * The Light tab's histogram: an 88 px box with a 2 px border holding three translucent
 * channel fills (orange-red, green, cyan-blue as in the mockup), and a caption row below.
 * `role="img"`; the values are decoration for the eyes, the caption carries the numbers.
 */
export function Histogram({ data, left, right, label = "Histogram", className }: HistogramProps) {
  const max = histogramMax(data.red, data.green, data.blue);
  return (
    <div className={cx("ym-histogram", className)}>
      <div className="ym-histogram__box">
        <svg viewBox="0 0 320 84" preserveAspectRatio="none" role="img" aria-label={label}>
          <path d={histogramPath(data.red, max)} fill="#FF5A1F" fillOpacity="0.55" />
          <path d={histogramPath(data.green, max)} fill="#77E05A" fillOpacity="0.55" />
          <path d={histogramPath(data.blue, max)} fill="#0DB7E8" fillOpacity="0.55" />
        </svg>
      </div>
      {left !== undefined || right !== undefined ? (
        <div className="ym-histogram__caption">
          <span>{left}</span>
          <span>{right}</span>
        </div>
      ) : null}
    </div>
  );
}
