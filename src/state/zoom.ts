/** The zoom steps of the Fit menu, from smallest to largest (`docs/11-ui.md` §Photos). */
import type { PhotoSummary, RenderStats, ResolvedValues, ViewportMode, ViewportRect } from "../api/types";

export const ZOOMS: ReadonlyArray<{ key: string; label: string; mode: ViewportMode }> = [
  { key: "fit", label: "Fit", mode: { kind: "fit" } },
  { key: "50", label: "50%", mode: { kind: "percent", value: 50 } },
  { key: "100", label: "100%", mode: { kind: "percent", value: 100 } },
  { key: "200", label: "200%", mode: { kind: "percent", value: 200 } },
];

/** The key of a zoom mode: "fit", "50", "100", "200". */
export const zoomKey = (mode: ViewportMode): string => (mode.kind === "fit" ? "fit" : String(mode.value));

/** What the core calls "no larger than the Fit zoom": it shows the whole picture then, whatever percentage was asked for. */
const FIT_TOLERANCE = 1.0001;

/**
 * True when `mode` shows the whole picture: Fit, or a fixed zoom no larger than the Fit zoom (the core draws Fit then, so a menu
 * choice of 50 % on a big display where Fit is 62 % is Fit, and the screen has to say so). `fit` is `fitZoomExact`; without it only
 * Fit itself counts.
 */
export function showsWhole(mode: ViewportMode, fit: number | null): boolean {
  return mode.kind === "fit" || (fit !== null && mode.value <= fit * FIT_TOLERANCE);
}

/**
 * The step `delta` places from the current zoom on the menu's list (Fit, 50, 100, 200 %), kept inside it. The zoom need not be on the
 * list: a wheel zoom leaves it anywhere (137 %), and the next step up is then 200 % and the next down 100 %. With `fit`, steps at or
 * below the Fit zoom are skipped, since they would show Fit again and a key press would seem to do nothing.
 */
export function zoomStep(mode: ViewportMode, delta: number, fit: number | null = null): ViewportMode {
  const value = (m: ViewportMode): number => (m.kind === "fit" ? 0 : m.value);
  const steps = ZOOMS.map((z) => z.mode).filter((m) => m.kind === "fit" || fit === null || m.value > fit * FIT_TOLERANCE);
  const here = showsWhole(mode, fit) ? 0 : value(mode);
  const count = Math.abs(Math.trunc(delta));
  if (count === 0) return mode;
  const ahead = delta > 0 ? steps.filter((m) => value(m) > here + 1e-9) : steps.filter((m) => value(m) < here - 1e-9).reverse();
  return ahead[Math.min(count, ahead.length) - 1] ?? (showsWhole(mode, fit) ? { kind: "fit" } : mode);
}

/** The zoom limits of the core (`docs/01` §Native viewport): up to 800 %. */
export const MAX_ZOOM_PERCENT = 800;

/**
 * The zoom that fits the whole picture in the photo cell, in percent and not rounded: device pixels per pixel of the final image (after
 * the quarter turn and the crop). A straighten angle and the lens crop are not counted. Null before the cell or the photo has a size.
 */
export function fitZoomExact(
  cell: Pick<ViewportRect, "width" | "height" | "scale">,
  photo: Pick<PhotoSummary, "width_px" | "height_px">,
  values: Pick<ResolvedValues, "orientation" | "crop"> | null,
): number | null {
  const turned = (values?.orientation ?? 0) % 180 !== 0;
  const crop = values?.crop?.rect;
  const width = (turned ? photo.height_px : photo.width_px) * (crop?.width ?? 1);
  const height = (turned ? photo.width_px : photo.height_px) * (crop?.height ?? 1);
  if (!(width > 0 && height > 0 && cell.width > 0 && cell.height > 0)) return null;
  return Math.min((cell.width * cell.scale) / width, (cell.height * cell.scale) / height) * 100;
}

/**
 * The Fit zoom in percent and not rounded: the core's (`render_stats.fit_percent`, which knows the lens crop, the straighten angle and
 * the crop that `fitZoomExact` cannot see) once a viewport frame has reported it, else `estimate`, this screen's own figure from the
 * photo and the cell (`fitZoomExact`), which is all the mock can offer.
 */
export function fitZoomOf(stats: Pick<RenderStats, "fit_percent"> | null | undefined, estimate: number | null): number | null {
  const core = stats?.fit_percent;
  return typeof core === "number" && Number.isFinite(core) && core > 0 ? core : estimate;
}

/** `fitZoomExact` rounded to a whole percent: "Fit 24%" in the header. */
export function fitZoomPercent(
  cell: Pick<ViewportRect, "width" | "height" | "scale">,
  photo: Pick<PhotoSummary, "width_px" | "height_px">,
  values: Pick<ResolvedValues, "orientation" | "crop"> | null,
): number | null {
  const exact = fitZoomExact(cell, photo, values);
  return exact === null ? null : Math.round(exact);
}

/**
 * The zoom after a wheel or pinch step that multiplies the core's zoom by `factor`. A step is mirrored at once, before the core has drawn
 * a frame to report (`render_stats`), so this starts from the mirrored percentage, or from the Fit zoom at Fit (`fit`: the core's,
 * `fitZoomOf`). Zooming out to the Fit zoom or below is Fit; zooming in stops at 800 %. Without `fit` a zoom in from Fit is taken
 * from 25 %.
 */
export function zoomAfter(mode: ViewportMode, factor: number, fit: number | null): ViewportMode {
  if (!(factor > 0) || !Number.isFinite(factor)) return mode;
  const whole = showsWhole(mode, fit);
  // The core never shows less than the whole picture: zooming out from it stays at Fit.
  if (whole && factor <= 1) return mode.kind === "fit" ? mode : { kind: "fit" };
  // A fixed zoom below the Fit zoom is the Fit zoom to the core, and a step starts from what it shows.
  const base = mode.kind === "fit" ? (fit ?? 25) : whole ? (fit ?? mode.value) : mode.value;
  const next = base * factor;
  if (next <= (fit ?? 0) * FIT_TOLERANCE) return { kind: "fit" };
  // Two decimals keep a run of small steps from stalling on rounding; the header rounds for display.
  return { kind: "percent", value: Math.min(MAX_ZOOM_PERCENT, Math.max(1, Math.round(next * 100) / 100)) };
}
