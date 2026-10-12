import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
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
   * `large` is the Start screen's dashed box with the black download icon. `compact` is the
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
 * A dashed 3 px drop target (`docs/11-ui.md` §Start). It highlights while a drag is over the
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
    <div
      ref={el}
      role="group"
      aria-label={title}
      className={cx("ym-drop-zone", className)}
      data-variant={variant === "compact" ? "compact" : undefined}
      data-active={active || over || undefined}
    >
      {variant === "large" ? (
        <span className="ym-drop-zone__icon" aria-hidden="true">
          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="square">
            <path d="M12 3v13" />
            <path d="M6 10.5L12 16.5l6-6" />
            <path d="M4 21h16" />
          </svg>
        </span>
      ) : null}
      <span className="ym-drop-zone__text">
        <span className="ym-drop-zone__title">{title}</span>
        {hint ? <span className="ym-drop-zone__hint">{hint}</span> : null}
      </span>
      {notice ? (
        <span role="alert" className="ym-drop-zone__notice">
          {notice}
        </span>
      ) : null}
      {children ? <div className="ym-drop-zone__actions">{children}</div> : null}
    </div>
  );
}
