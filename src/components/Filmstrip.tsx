import { useEffect, useRef } from "react";
import type { KeyboardEvent, MouseEvent, ReactNode } from "react";
import { cx } from "./cx";
import { Tag } from "./Tag";

/** What a thumbnail shows: a picture, a placeholder while developing, or a "can't open" tag. */
export type FilmstripStatus = "ready" | "developing" | "failed";

/** One photo in the strip. */
export interface FilmstripPhoto {
  /** Stable for the session (the core's `PhotoId`). */
  id: number | string;
  /** File name without extension ("DSC04204"), for the accessible name. */
  name: string;
  /** Thumbnail URL (a PNG of at most 240 px from `thumbnail_ready`). */
  src?: string;
  /** Default `ready`. */
  status?: FilmstripStatus;
  /** Picked for export: a yellow corner square. */
  picked?: boolean;
  /** Export mode only: this photo will be exported. Others are dimmed. */
  included?: boolean;
}

export interface FilmstripProps {
  photos: readonly FilmstripPhoto[];
  /** The photo on screen: the same size as the others, told apart by a 3 px accent ring. Null when none. */
  currentId: number | string | null;
  /** A thumbnail was clicked, or ←/→ moved the selection. */
  onSelect?: (id: number | string) => void;
  /** ⌘-click or P on a thumbnail. */
  onTogglePick?: (id: number | string) => void;
  /**
   * `browse` (default) is the Photos screen: clickable thumbnails, a "4 / 12" counter and
   * "3 picked". `export` is the Export panel: not clickable, included photos have a 3 px border
   * and the rest are dimmed, and the counter reads "3 to export".
   */
  mode?: "browse" | "export";
  /** Replaces the counter's big text. */
  counter?: ReactNode;
  /** Replaces the counter's caption. */
  counterLabel?: ReactNode;
  /** Accessible name of the strip. Defaults to "Photos" (browse) or "Photos to export". */
  label?: string;
  className?: string;
}

/** Space kept between a revealed thumbnail and the edge of the strip, in CSS pixels. */
const REVEAL_MARGIN = 24;

/**
 * How far the strip must scroll sideways (positive: to the right) to bring `item` fully into view,
 * with a margin; 0 when it is already inside. Both rectangles are in viewport coordinates.
 */
export function scrollOffsetToReveal(
  strip: { left: number; right: number },
  item: { left: number; right: number },
  margin = REVEAL_MARGIN,
): number {
  if (item.left < strip.left + margin) return item.left - (strip.left + margin);
  if (item.right > strip.right - margin) return item.right - (strip.right - margin);
  return 0;
}

/**
 * The filmstrip under the photo: 3:2 thumbnails as tall as the row less the panel inset above and
 * below (24 px, the tool panel's own inset), the current one told apart by a 3 px accent ring, a
 * yellow corner dot on picked photos and a black CAN'T OPEN tag on files that failed, then a
 * counter cell. Click selects, ⌘-click or P toggles the pick, ←/→ move the selection (focus
 * follows), the strip scrolls sideways.
 */
export function Filmstrip({
  photos,
  currentId,
  onSelect,
  onTogglePick,
  mode = "browse",
  counter,
  counterLabel,
  label,
  className,
}: FilmstripProps) {
  const refs = useRef(new Map<number | string, HTMLElement>());
  const scroller = useRef<HTMLDivElement>(null);
  const focusAfter = useRef<number | string | null>(null);
  const currentIndex = photos.findIndex((p) => p.id === currentId);
  const tabbableIndex = currentIndex >= 0 ? currentIndex : 0;

  // Keep the current photo in view, and keep focus on it after keyboard navigation. Only the strip
  // scrolls, sideways: `scrollIntoView` would also move the page.
  useEffect(() => {
    const el = currentId === null ? undefined : refs.current.get(currentId);
    if (!el) return;
    const strip = scroller.current;
    if (strip) strip.scrollLeft += scrollOffsetToReveal(strip.getBoundingClientRect(), el.getBoundingClientRect());
    if (focusAfter.current === currentId) el.focus({ preventScroll: true });
    focusAfter.current = null;
  }, [currentId]);

  const onKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    if (mode !== "browse" || event.metaKey || event.ctrlKey || event.altKey) return;
    const focused = (event.target as HTMLElement).closest<HTMLElement>("[data-photo-id]");
    const focusedId = focused?.dataset.photoId;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      const from = focusedId !== undefined ? photos.findIndex((p) => String(p.id) === focusedId) : currentIndex;
      const next = photos[from + (event.key === "ArrowRight" ? 1 : -1)];
      event.preventDefault();
      if (next) {
        focusAfter.current = next.id;
        onSelect?.(next.id);
      }
    } else if (event.key === "p" || event.key === "P") {
      const target = focusedId !== undefined ? photos.find((p) => String(p.id) === focusedId) : photos[currentIndex];
      if (target) {
        event.preventDefault();
        onTogglePick?.(target.id);
      }
    }
  };

  const pickedCount = photos.filter((p) => p.picked).length;
  const includedCount = photos.filter((p) => p.included).length;
  const bigText = counter ?? (mode === "export" ? String(includedCount) : `${currentIndex >= 0 ? currentIndex + 1 : 0} / ${photos.length}`);
  const smallText = counterLabel ?? (mode === "export" ? "to export" : `${pickedCount} picked`);

  const thumbContent = (photo: FilmstripPhoto): ReactNode => (
    <>
      {photo.status !== "failed" && photo.status !== "developing" && photo.src ? <img src={photo.src} alt="" draggable={false} /> : null}
      {photo.status === "failed" ? (
        <span className="ym-filmstrip__failed">
          <Tag variant="cant-open" size="sm" />
        </span>
      ) : null}
      {photo.picked ? <span className="ym-filmstrip__picked" aria-hidden="true" /> : null}
    </>
  );

  const describe = (photo: FilmstripPhoto): string =>
    [
      photo.name,
      photo.status === "failed" ? "can't open" : photo.status === "developing" ? "developing" : null,
      photo.picked ? "picked" : null,
      mode === "export" && photo.included ? "will be exported" : null,
    ]
      .filter(Boolean)
      .join(", ");

  return (
    <nav aria-label={label ?? (mode === "export" ? "Photos to export" : "Photos")} className={cx("ym-filmstrip", className)} onKeyDown={onKeyDown}>
      <div ref={scroller} className="ym-filmstrip__scroll">
        {photos.map((photo, index) => {
          const isCurrent = photo.id === currentId;
          const common = {
            "data-photo-id": String(photo.id),
            "data-status": photo.status === "ready" ? undefined : photo.status,
            ref: (node: HTMLElement | null) => {
              if (node) refs.current.set(photo.id, node);
              else refs.current.delete(photo.id);
            },
          };
          if (mode === "export") {
            return (
              <span
                key={photo.id}
                {...common}
                role="img"
                aria-label={describe(photo)}
                className="ym-filmstrip__thumb"
                aria-current={isCurrent ? "true" : undefined}
                data-in-export={photo.included ? "true" : "false"}
              >
                {thumbContent(photo)}
              </span>
            );
          }
          return (
            <button
              key={photo.id}
              {...common}
              type="button"
              className="ym-filmstrip__thumb"
              aria-label={`Show ${describe(photo)}`}
              aria-current={isCurrent ? "true" : "false"}
              tabIndex={index === tabbableIndex ? 0 : -1}
              onClick={(event: MouseEvent<HTMLButtonElement>) => {
                if (event.metaKey) onTogglePick?.(photo.id);
                else onSelect?.(photo.id);
              }}
            >
              {thumbContent(photo)}
            </button>
          );
        })}
      </div>
      <div className="ym-filmstrip__counter">
        <span className="ym-filmstrip__count">{bigText}</span>
        <span className="ym-filmstrip__count-label">{smallText}</span>
      </div>
    </nav>
  );
}
