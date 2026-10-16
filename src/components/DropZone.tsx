import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { DownloadIcon } from "lucide-react";
import { Card } from "@/components/glass/card";
import { Item, ItemMedia } from "@/components/glass/item";
import { cx } from "./cx";
import { useLatest } from "./hooks";
import { domDropSource } from "./useWindowDrop";

export interface DropZoneProps {
  /** Headline: "Drop files or a folder", "Or drop the zip here". */
  title: string;
  /** Under it: "ARW only, as many as you like". */
  hint?: string;
  /** A problem with the last drop, in a peach box: "No ARW files in what you dropped." */
  notice?: string;
  /**
   * `large` is the Start screen's dashed box with the download icon. `compact` is the
   * wizard's "Or drop the zip here" strip: title and hint at the left, actions at the right.
   */
  variant?: "large" | "compact";
  /** Highlight (a drag is over the window). Use `useWindowDrop().dragging`. */
  active?: boolean;
  /**
   * Handles files dropped on this element (DOM drops: a browser, the mock, tests). In the app
   * use `useWindowDrop` for the whole window instead and leave this out.
   */
  onDropPaths?: (paths: string[]) => void;
  /** Buttons or links ("Open files…", "Choose file…"). */
  children?: ReactNode;
  className?: string;
}

/**
 * A dashed drop target on a glasscn `Card` (`docs/11-ui.md` §Start). It highlights while a drag is over the
 * window (`active`) or over itself, and reports DOM drops through `onDropPaths`. The whole
 * window accepts drops through `useWindowDrop`; this component is where they are shown.
 */
export function DropZone({
  title,
  hint,
  notice,
  variant = "large",
  active = false,
  onDropPaths,
  children,
  className,
}: DropZoneProps) {
  const el = useRef<HTMLDivElement>(null);
  const [over, setOver] = useState(false);
  const hasHandler = onDropPaths !== undefined;
  const latest = useLatest(onDropPaths);

  useEffect(() => {
    const node = el.current;
    if (!node || !hasHandler) return;
    return domDropSource(node).subscribe({
      onEnter: () => setOver(true),
      onLeave: () => setOver(false),
      onDrop: (paths) => latest.current?.(paths),
    });
  }, [hasHandler, latest]);

  return (
    <Card
      ref={el}
      role="group"
      aria-label={title}
      className={cx("ym-drop-zone", className)}
      data-variant={variant === "compact" ? "compact" : undefined}
      data-active={active || over || undefined}
    >
      {variant === "large" ? (
        <ItemMedia variant="icon" className="ym-drop-zone__icon" aria-hidden="true">
          <DownloadIcon />
        </ItemMedia>
      ) : null}
      <span className="ym-drop-zone__text">
        <span className="ym-drop-zone__title">{title}</span>
        {hint ? <span className="ym-drop-zone__hint">{hint}</span> : null}
      </span>
      {notice ? (
        <Item variant="muted" size="sm" role="alert" className="ym-drop-zone__notice">
          {notice}
        </Item>
      ) : null}
      {children ? <div className="ym-drop-zone__actions">{children}</div> : null}
    </Card>
  );
}
