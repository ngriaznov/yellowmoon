import { useEffect, useId, useRef, useState } from "react";
import type { FocusEvent, KeyboardEvent, MouseEvent, PointerEvent, ReactNode } from "react";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuShortcut, ContextMenuTrigger } from "@/components/glass/context-menu";
import { cx } from "./cx";
import { Tag } from "./Tag";
import type { TabId } from "./tabs";

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
   * "Remove from filmstrip" in the thumbnail's right-click menu (Control-click, or the Menu key on the focused thumbnail): takes that photo
   * out of the session, the files stay on disk. Without it the thumbnails have no menu.
   */
  onRemove?: (id: number | string) => void;
  /** "Remove picked photos" in the same menu, offered while any photo is picked; the count is shown at its right. */
  onRemovePicked?: () => void;
  /**
   * `browse` (default) is the Photos screen: clickable thumbnails and a caption, "4 of 12 · 3 picked".
   * `export` is the Export panel: not clickable, included photos have a 3 px border and the rest are
   * dimmed, and the caption reads "3 to export".
   */
  mode?: "browse" | "export";
  /** Replaces the caption line ("4 of 12 · 3 picked"). */
  caption?: ReactNode;
  /** Tab whose colour tints the strip, for a surface that follows the tab. Omit it for the brand colour. */
  tab?: TabId;
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
 * The one line that says where you are in the folder: "4 of 12 · 3 picked", or "3 to export" in the Export panel. The strip
 * shows it as its caption; a header that wants to show the same line asks here, so the two never differ.
 */
export function filmstripCaption(photos: readonly FilmstripPhoto[], currentId: number | string | null, mode: "browse" | "export" = "browse"): string {
  if (mode === "export") return `${photos.filter((p) => p.included).length} to export`;
  const index = photos.findIndex((p) => p.id === currentId);
  return `${index >= 0 ? index + 1 : 0} of ${photos.length} · ${photos.filter((p) => p.picked).length} picked`;
}

/**
 * glasscn's context menu around the strip's scroller, or just the scroller when `enabled` is off. `onTarget` is called first, for a
 * right click (and for a touch that may begin a long press); it may call `preventDefault` to stop the menu from opening.
 */
function MenuAround({
  enabled,
  onOpenChange,
  onTarget,
  content,
  children,
}: {
  enabled: boolean;
  onOpenChange: (open: boolean) => void;
  onTarget: (event: MouseEvent<HTMLElement> | PointerEvent<HTMLElement>) => void;
  content: ReactNode;
  children: ReactNode;
}) {
  if (!enabled) return <>{children}</>;
  return (
    <ContextMenu onOpenChange={onOpenChange}>
      <ContextMenuTrigger
        asChild
        onContextMenu={onTarget}
        // A mouse press is no menu; the touch of a long press is (Radix opens it by itself after a moment).
        onPointerDown={(event) => {
          if (event.pointerType !== "mouse") onTarget(event);
        }}
      >
        {children}
      </ContextMenuTrigger>
      {content}
    </ContextMenu>
  );
}

/**
 * The filmstrip under the photo: 3:2 thumbnails as tall as the row less the panel inset above and
 * below (24 px, the tool panel's own inset), the current one told apart by a 3 px accent ring, a
 * yellow corner dot on picked photos and a black CAN'T OPEN tag on files that failed, and a one-line
 * caption ("4 of 12 · 3 picked") in the strip's top inset. Click selects, ⌘-click or P toggles the
 * pick, ←/→ move the selection (focus follows), the strip scrolls sideways. With `onRemove` the strip has glasscn's context menu
 * (right-click a thumbnail): "Remove from filmstrip", and "Remove picked photos" while any photo is picked.
 */
export function Filmstrip({
  photos,
  currentId,
  onSelect,
  onTogglePick,
  onRemove,
  onRemovePicked,
  mode = "browse",
  caption,
  tab,
  label,
  className,
}: FilmstripProps) {
  const captionId = useId();
  const refs = useRef(new Map<number | string, HTMLElement>());
  const scroller = useRef<HTMLDivElement>(null);
  const focusAfter = useRef<number | string | null>(null);
  const currentIndex = photos.findIndex((p) => p.id === currentId);
  const tabbableIndex = currentIndex >= 0 ? currentIndex : 0;
  const pickedCount = photos.filter((p) => p.picked).length;

  // One context menu for the whole strip (a menu per thumbnail would be hundreds of Radix roots to redraw whenever a thumbnail
  // arrives). The thumbnail a menu is about is told when it opens: by the right click, or by the touch that begins a long press.
  const removable = mode === "browse" && onRemove !== undefined;
  const [menu, setMenu] = useState<{ id: number | string | null; open: boolean }>({ id: null, open: false });
  const idUnder = (target: EventTarget | null): number | string | undefined => {
    const id = (target as HTMLElement | null)?.closest?.<HTMLElement>("[data-photo-id]")?.dataset.photoId;
    return id === undefined ? undefined : photos.find((p) => String(p.id) === id)?.id;
  };
  const menuPhoto = menu.id === null ? undefined : photos.find((p) => p.id === menu.id);

  // The keyboard focus was on a thumbnail that is gone now (a photo was removed): it goes to the photo on screen, so that the arrows
  // go on working from the strip. A real blur (the person went elsewhere) is told apart from the removal of the focused element by the
  // element still being in the page.
  const focusInside = useRef(false);
  const onFocusIn = (): void => {
    focusInside.current = true;
  };
  const onFocusOut = (event: FocusEvent<HTMLElement>): void => {
    if ((event.target as HTMLElement).isConnected) focusInside.current = false;
  };
  useEffect(() => {
    if (!focusInside.current || currentId === null) return;
    const strip = scroller.current;
    if (strip && !strip.contains(document.activeElement)) refs.current.get(currentId)?.focus({ preventScroll: true });
  }, [photos, currentId]);

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

  const captionText = caption ?? filmstripCaption(photos, currentId, mode);

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
    <nav
      aria-label={label ?? (mode === "export" ? "Photos to export" : "Photos")}
      aria-describedby={captionId}
      className={cx("ym-filmstrip", className)}
      data-tab={tab}
      onKeyDown={onKeyDown}
      onFocus={onFocusIn}
      onBlur={onFocusOut}
    >
      <MenuAround
        enabled={removable}
        onOpenChange={(open) => setMenu((was) => ({ ...was, open }))}
        onTarget={(event) => {
          // A press or right click between the thumbnails opens no menu.
          const id = idUnder(event.target);
          if (id === undefined) event.preventDefault();
          else setMenu({ id, open: false });
        }}
        content={
          <ContextMenuContent aria-label={`${menuPhoto?.name ?? "Photo"} actions`}>
            <ContextMenuItem onSelect={() => menu.id !== null && onRemove?.(menu.id)}>Remove from filmstrip</ContextMenuItem>
            {pickedCount > 0 && onRemovePicked ? (
              <ContextMenuItem onSelect={() => onRemovePicked()}>
                Remove picked photos
                <ContextMenuShortcut>{pickedCount}</ContextMenuShortcut>
              </ContextMenuItem>
            ) : null}
          </ContextMenuContent>
        }
      >
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
                data-menu={menu.open && menu.id === photo.id ? "open" : undefined}
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
      </MenuAround>
      <span id={captionId} className="ym-filmstrip__caption">
        {captionText}
      </span>
    </nav>
  );
}
