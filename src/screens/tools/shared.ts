import type { AutoOr, BalanceSetting, ColorBalancePoint, PhotoSummary, ResolvedSettings } from "../../api/types";
import { formats } from "../../components/format";
import type { AppStore } from "../../state/store";

/** What every tab needs: the store, the photo and its resolved settings. */
export interface TabProps {
  store: AppStore;
  photo: PhotoSummary;
  rs: ResolvedSettings;
}

const near = (a: number, b: number): boolean => Math.abs(a - b) < 1e-9;

/** True when every field of `a` equals the same field of `b`. */
export function sameNumbers<T extends object>(a: T, b: T): boolean {
  return (Object.keys(a) as Array<keyof T>).every((key) => near(Number(a[key]), Number(b[key])));
}

/**
 * A manual value for a setting that can also be Automatic. A value equal to what Automatic
 * resolves to *is* Automatic: a slider dropped on its detent, or reset by hand, gets the AUTO tag
 * back instead of a manual value that happens to match.
 */
export function manualOrAuto<M extends object>(manual: M, auto: M): AutoOr<M> {
  return sameNumbers(manual, auto) ? { kind: "auto" } : ({ kind: "manual", ...manual } as AutoOr<M>);
}

/** The reset of `docs/11`: Auto for these fields. */
export const AUTO = { kind: "auto" } as const;

/**
 * A shadows or highlights wheel set by hand to `point` (`docs/19` section 5): a point on the Auto point (within the wheel's own
 * tolerance) *is* Automatic, as a slider dropped on its detent is.
 */
export function pointOrAuto(point: ColorBalancePoint, auto: ColorBalancePoint): BalanceSetting {
  return Math.abs(point[0] - auto[0]) < 1e-4 && Math.abs(point[1] - auto[1]) < 1e-4 ? AUTO : { kind: "manual", point };
}

/** The default of Detail › Sharpening › Output, in percent (`DevelopSettings.output_sharpening`). */
export const OUTPUT_SHARPENING_DEFAULT = 100;

/** The default of Light › White › Offset, in stops (`DevelopSettings.white_offset_ev`): the automatic white lift as it is. */
export const WHITE_OFFSET_DEFAULT = 0;

/** The range of Light › White › Offset, in stops (`DevelopSettings.white_offset_ev`). */
export const WHITE_OFFSET_RANGE_EV = [-3, 3] as const;

/** Shows a tint: "0", "+4", "−3" (with a true minus). The automatic tint is not a whole number: 0.3 reads "0", not "+0". */
export const tintLabel = (tint: number): string => formats.signed(tint);
