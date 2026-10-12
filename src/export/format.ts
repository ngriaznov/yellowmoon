/** Words and numbers of the Export panel: the size estimate, the folder name, the long edge slider. */
import type { ExportEstimate } from "../api/types";

/**
 * "≈ 120 MB", or "≈ 1.2 GB" from 1000 MB, from the core's byte count (decimal megabytes, as in the mockup).
 * Never less than 1 MB.
 */
export function formatEstimate(bytes: number): string {
  const mb = bytes / 1_000_000;
  return `≈ ${mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.max(1, Math.round(mb))} MB`}`;
}

/** "about 40 s", "about 3 min": the time estimate, for a tooltip. */
export function formatDuration(seconds: number): string {
  if (!(seconds > 0)) return "";
  if (seconds < 90) return `about ${Math.max(1, Math.round(seconds))} s`;
  return `about ${Math.round(seconds / 60)} min`;
}

/** The estimate and the time as one tooltip: "≈ 75 MB, about 4 s". */
export function estimateTitle(estimate: ExportEstimate): string {
  const time = formatDuration(estimate.seconds);
  return time ? `${formatEstimate(estimate.bytes)}, ${time}` : formatEstimate(estimate.bytes);
}

/** The last part of a path: "~/Pictures/Lisbon export" gives "Lisbon export". */
export function folderLabel(path: string): string {
  const parts = path.replace(/[\\/]+$/, "").split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

/** "1 photo", "3 photos". */
export const photoCount = (n: number): string => `${n} ${n === 1 ? "photo" : "photos"}`;

/** The shortest long edge the slider offers, and its step (`docs/09`: "Long edge 1000–full px in 20 px steps"). */
export const LONG_EDGE_MIN = 1000;
export const LONG_EDGE_STEP = 20;

/**
 * The slider's range for photos whose longest side is `fullPx`: 1000 up to the full size, rounded
 * down to the step so the maximum can be reached by the slider. Null when the photo is smaller than the
 * least the slider offers (then Size has no Long edge).
 */
export function longEdgeRange(fullPx: number): { min: number; max: number } | null {
  if (fullPx < LONG_EDGE_MIN + LONG_EDGE_STEP) return null;
  const max = LONG_EDGE_MIN + Math.floor((fullPx - LONG_EDGE_MIN) / LONG_EDGE_STEP) * LONG_EDGE_STEP;
  return { min: LONG_EDGE_MIN, max };
}

/** A long edge pulled onto the slider's grid and into its range. */
export function snapLongEdge(px: number, fullPx: number): number {
  const range = longEdgeRange(fullPx);
  if (!range) return Math.min(px, fullPx);
  const snapped = LONG_EDGE_MIN + Math.round((px - LONG_EDGE_MIN) / LONG_EDGE_STEP) * LONG_EDGE_STEP;
  return Math.min(range.max, Math.max(range.min, snapped));
}
