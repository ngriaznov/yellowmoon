import { cx } from "./cx";
import { tabInfo } from "./tabs";
import type { TabId } from "./tabs";

export interface TitleBlockProps {
  /** Title in 40 px 800 ("Look", "Export"). Defaults to the tab's label. */
  title?: string;
  /** The blurb under it. Defaults to the tab's blurb from `docs/11`. */
  blurb?: string;
  /** Tab whose colour fills the block. Omit it to inherit `data-tab` from above. */
  tab?: TabId;
  /** `ink` is the Export panel's black block with white text. */
  tone?: "tab" | "ink";
  /** The blurb is a status: a screen reader says it again when it changes ("1 of 3 done."). */
  live?: boolean;
  className?: string;
}

/**
 * The title block at the top of a tool tab or the Export panel: a flat colour, a 40 px
 * title and a 15 px blurb, with a 3 px rule below. Pass only `tab` for the stock title and
 * blurb of a tool tab.
 */
export function TitleBlock({ title, blurb, tab, tone = "tab", live = false, className }: TitleBlockProps) {
  const info = tab ? tabInfo(tab) : undefined;
  return (
    <div className={cx("ym-title-block", className)} data-tab={tab} data-tone={tone === "ink" ? "ink" : undefined}>
      <h2 className="ym-title-block__title">{title ?? info?.label}</h2>
      {(blurb ?? info?.blurb) ? (
        <p className="ym-title-block__blurb" role={live ? "status" : undefined} aria-live={live ? "polite" : undefined}>
          {blurb ?? info?.blurb}
        </p>
      ) : null}
    </div>
  );
}
