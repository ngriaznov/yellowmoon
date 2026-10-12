/** The zoom steps of the Fit menu, from smallest to largest (`docs/11-ui.md` §Photos). */
import type { ViewportMode } from "../api/types";

export const ZOOMS: ReadonlyArray<{ key: string; label: string; mode: ViewportMode }> = [
  { key: "fit", label: "Fit", mode: { kind: "fit" } },
  { key: "50", label: "50%", mode: { kind: "percent", value: 50 } },
  { key: "100", label: "100%", mode: { kind: "percent", value: 100 } },
  { key: "200", label: "200%", mode: { kind: "percent", value: 200 } },
];

/** The key of a zoom mode: "fit", "50", "100", "200". */
export const zoomKey = (mode: ViewportMode): string => (mode.kind === "fit" ? "fit" : String(mode.value));

/** The step `delta` places from the current zoom, kept inside the list. */
export function zoomStep(mode: ViewportMode, delta: number): ViewportMode {
  const index = Math.max(0, ZOOMS.findIndex((z) => z.key === zoomKey(mode)));
  return (ZOOMS[Math.min(ZOOMS.length - 1, Math.max(0, index + delta))] ?? ZOOMS[0]!).mode;
}
