import type { ReactNode, Ref } from "react";
import { cx } from "./cx";
import type { TabId } from "./tabs";

export interface LookTileProps {
  /** The look's name ("Film warm"), shown under the picture. */
  name: string;
  /** URL of the thumbnail: the current photo through this look. */
  src?: string;
  /** Custom picture content (for example a canvas) instead of `src`. */
  thumbnail?: ReactNode;
  /** The key that picks the look (1–9, Look tab visible): a black badge top-left. Later looks have none. */
  index?: number;
  /** The user's default look: a yellow DEFAULT tag bottom-right. */
  isDefault?: boolean;
  /** The look in use: a 3 px ink ring and the name on a tab-colour label. */
  selected?: boolean;
  /** A look whose file is missing or changed: shown dimmed and not selectable. */
  disabled?: boolean;
  /** Why it is disabled; part of the accessible name. */
  note?: string;
  onSelect: () => void;
  /** The tile's button, for a screen that watches it scroll into view (lazy thumbnails). */
  ref?: Ref<HTMLButtonElement>;
  /** Tab whose colour labels the selected tile; normally inherited. */
  tab?: TabId;
  className?: string;
}

/**
 * A look in the Look tab's grid: a 3:2 thumbnail with a 2 px border, a 1–9 key badge top-left,
 * a DEFAULT tag bottom-right and the name below. Selected is a 3 px ink ring and the name on a
 * tab-colour label. A toggle button (`aria-pressed`) named "<name> look, key <n>".
 */
export function LookTile({
  name,
  src,
  thumbnail,
  index,
  isDefault = false,
  selected = false,
  disabled = false,
  note,
  onSelect,
  ref,
  tab,
  className,
}: LookTileProps) {
  const badge = index !== undefined && index >= 1 && index <= 9 ? index : undefined;
  const label = `${name} look${badge ? `, key ${badge}` : ""}${disabled && note ? `, ${note.toLowerCase()}` : ""}`;
  return (
    <button
      ref={ref}
      type="button"
      className={cx("ym-look-tile", className)}
      aria-pressed={selected}
      aria-disabled={disabled || undefined}
      aria-label={label}
      data-tab={tab}
      title={disabled ? note : undefined}
      onClick={() => {
        if (!disabled) onSelect();
      }}
    >
      <span className="ym-look-tile__picture">
        {thumbnail ?? (src ? <img src={src} alt="" draggable={false} /> : null)}
        {badge ? (
          <span className="ym-look-tile__badge" aria-hidden="true">
            {badge}
          </span>
        ) : null}
        {isDefault ? <span className="ym-look-tile__default">Default</span> : null}
      </span>
      <span className="ym-look-tile__name">{name}</span>
      {disabled && note ? <span className="ym-look-tile__note">{note}</span> : null}
    </button>
  );
}
