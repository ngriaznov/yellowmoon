import { memo, useCallback, useEffect, useId, useRef, useState } from "react";
import type { CSSProperties, FocusEvent, KeyboardEvent, MouseEvent, PointerEvent, ReactNode } from "react";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuShortcut, ContextMenuTrigger } from "@/components/glass/context-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/glass/tooltip";
import { cx } from "./cx";
import { useStableCallback } from "./hooks";
import { NodesGlyph } from "./NodesGlyph";
import { Tag } from "./Tag";
import type { TabId } from "./tabs";
import { useWindowedRange } from "./windowed";

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
  /**
   * What the photo's graph has that the simple view does not show, in words ("4 nodes", "Node 1 has masks or a trim"): the thumbnail
   * carries the nodes glyph at its bottom left with these words as its tooltip, and its name says them too (`docs/23` §6).
   */
  nodes?: string;
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
  /** Hidden (the Photos screen's full screen, F). The strip still follows the photo on screen and shows it when it is back. */
  hidden?: boolean;
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

/** What `Thumb` draws for one photo. */
interface ThumbProps {
  photo: FilmstripPhoto;
  mode: "browse" | "export";
  isCurrent: boolean;
  /** Browse mode: this thumbnail is the strip's one tab stop. */
  tabbable: boolean;
  /** A right-click menu about this photo is open. */
  menuOpen: boolean;
  /** Space that stands in for the thumbnails that are not drawn, on the first and last one drawn. */
  style?: CSSProperties;
  register: (id: number | string, node: HTMLElement | null) => void;
  onActivate: (id: number | string, pick: boolean) => void;
}

/** The words that name a photo, for its accessible name. */
function describe(photo: FilmstripPhoto, mode: "browse" | "export"): string {
  return [
    photo.name,
    photo.status === "failed" ? "can't open" : photo.status === "developing" ? "developing" : null,
    photo.picked ? "picked" : null,
    photo.nodes ? photo.nodes.charAt(0).toLowerCase() + photo.nodes.slice(1) : null,
    mode === "export" && photo.included ? "will be exported" : null,
  ]
    .filter(Boolean)
    .join(", ");
}

/** Two photos draw the same: the fields `Thumb` reads, by value (the list is made again whenever anything about any photo changes). */
const samePhoto = (a: FilmstripPhoto, b: FilmstripPhoto): boolean =>
  a === b || (a.id === b.id && a.name === b.name && a.src === b.src && a.status === b.status && a.picked === b.picked && a.included === b.included && a.nodes === b.nodes);

/** One thumbnail. Memoised by value, so a key press draws the two thumbnails that change and not the other hundreds. */
const Thumb = memo(
  function Thumb({ photo, mode, isCurrent, tabbable, menuOpen, style, register, onActivate }: ThumbProps) {
    const content = (
      <>
        {photo.status !== "failed" && photo.status !== "developing" && photo.src ? (
          <img src={photo.src} alt="" draggable={false} loading="lazy" decoding="async" />
        ) : null}
        {photo.status === "failed" ? (
          <span className="ym-filmstrip__failed">
            <Tag variant="cant-open" size="sm" />
          </span>
        ) : null}
        {photo.picked ? <span className="ym-filmstrip__picked" aria-hidden="true" /> : null}
        {photo.nodes ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="ym-filmstrip__nodes" data-nodes={photo.nodes}>
                <NodesGlyph />
              </span>
            </TooltipTrigger>
            <TooltipContent>{photo.nodes}</TooltipContent>
          </Tooltip>
        ) : null}
      </>
    );
    const common = {
      "data-photo-id": String(photo.id),
      "data-status": photo.status === "ready" ? undefined : photo.status,
      style,
      ref: (node: HTMLElement | null) => register(photo.id, node),
    };
    if (mode === "export") {
      return (
        <span
          {...common}
          role="img"
          aria-label={describe(photo, mode)}
          className="ym-filmstrip__thumb"
          aria-current={isCurrent ? "true" : undefined}
          data-in-export={photo.included ? "true" : "false"}
        >
          {content}
        </span>
      );
    }
    return (
      <button
        {...common}
        type="button"
        className="ym-filmstrip__thumb"
        aria-label={`Show ${describe(photo, mode)}`}
        aria-current={isCurrent ? "true" : "false"}
        data-menu={menuOpen ? "open" : undefined}
        tabIndex={tabbable ? 0 : -1}
        onClick={(event: MouseEvent<HTMLButtonElement>) => onActivate(photo.id, event.metaKey)}
      >
        {content}
      </button>
    );
  },
  (a, b) =>
    a.mode === b.mode &&
    a.isCurrent === b.isCurrent &&
    a.tabbable === b.tabbable &&
    a.menuOpen === b.menuOpen &&
    a.style?.marginLeft === b.style?.marginLeft &&
    a.style?.marginRight === b.style?.marginRight &&
    a.register === b.register &&
    a.onActivate === b.onActivate &&
    samePhoto(a.photo, b.photo),
);

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
  hidden = false,
  className,
}: FilmstripProps) {
  const captionId = useId();
  const refs = useRef(new Map<number | string, HTMLElement>());
  const scroller = useRef<HTMLDivElement>(null);
  const focusAfter = useRef<number | string | null>(null);
  const currentIndex = photos.findIndex((p) => p.id === currentId);
  // Only the thumbnails in view and a margin are drawn once the folder is large (`docs/11` §Large folders).
  const win = useWindowedRange({ count: photos.length, axis: "x", list: scroller, scroller, itemSelector: ".ym-filmstrip__thumb", estimate: 120 });
  // The strip's one tab stop is the photo on screen; when that one is far off and not drawn, the first drawn thumbnail takes it.
  const tabbableIndex = currentIndex >= win.start && currentIndex < win.end ? currentIndex : win.start;
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
  // scrolls, sideways: `scrollIntoView` would also move the page. A hidden strip has no boxes to measure; it reveals the photo on
  // screen when it shows again (the arrows moved through the photos in full screen).
  // A photo that is not drawn (far off in a large folder) is brought into view by arithmetic, which draws it; the effect runs again then.
  const toReveal = useRef<number | string | null>(null);
  useEffect(() => {
    toReveal.current = currentId === null || hidden ? null : currentId;
  }, [currentId, hidden]);
  const { scrollToIndex } = win;
  useEffect(() => {
    const id = toReveal.current;
    if (id === null) return;
    const el = refs.current.get(id);
    if (!el) {
      if (currentIndex >= 0) scrollToIndex(currentIndex);
      return;
    }
    toReveal.current = null;
    const strip = scroller.current;
    if (strip) strip.scrollLeft += scrollOffsetToReveal(strip.getBoundingClientRect(), el.getBoundingClientRect());
    if (focusAfter.current === id) el.focus({ preventScroll: true });
    focusAfter.current = null;
  }, [currentId, hidden, currentIndex, scrollToIndex, win.start, win.end]);

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

  const register = useCallback((id: number | string, node: HTMLElement | null) => {
    if (node) refs.current.set(id, node);
    else refs.current.delete(id);
  }, []);
  const activate = useStableCallback((id: number | string, pick: boolean) => {
    if (pick) onTogglePick?.(id);
    else onSelect?.(id);
  });

  return (
    <nav
      aria-label={label ?? (mode === "export" ? "Photos to export" : "Photos")}
      aria-describedby={captionId}
      className={cx("ym-filmstrip", className)}
      hidden={hidden || undefined}
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
          {photos.slice(win.start, win.end).map((photo, i) => {
            const index = win.start + i;
            // The space of the thumbnails that are not drawn is the margin of the first and last one that is.
            const style: CSSProperties | undefined =
              win.windowed && (index === win.start || index === win.end - 1)
                ? { marginLeft: index === win.start ? win.before : undefined, marginRight: index === win.end - 1 ? win.after : undefined }
                : undefined;
            return (
              <Thumb
                key={photo.id}
                photo={photo}
                mode={mode}
                isCurrent={photo.id === currentId}
                tabbable={index === tabbableIndex}
                menuOpen={menu.open && menu.id === photo.id}
                style={style}
                register={register}
                onActivate={activate}
              />
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
