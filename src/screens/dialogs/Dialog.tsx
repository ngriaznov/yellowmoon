import { useEffect, useId, useRef } from "react";
import type { KeyboardEvent, ReactNode } from "react";

export interface DialogProps {
  /** The title, in the black block at the top. Also the dialog's accessible name. */
  title: string;
  /** A line under the title. */
  blurb?: string;
  /** Closes the dialog: Esc, a click outside it, and the buttons that dismiss it. */
  onClose: () => void;
  /** A click outside or Esc does nothing while this is false (an import is running). Default true. */
  dismissable?: boolean;
  /** Buttons along the bottom edge. */
  footer?: ReactNode;
  /** Width in CSS pixels. Default 560. */
  width?: number;
  children?: ReactNode;
}

const FOCUSABLE = 'button:not([aria-disabled="true"]), [href], input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])';

/**
 * A square, boxed dialog over a dimmed window (Preferences, About, the question before closing a session during an export).
 * `role="dialog"` and `aria-modal`; focus moves into it when it opens (to the control marked `data-autofocus`, else the first
 * one), Tab stays inside, Esc closes it and focus goes back to where it was.
 */
export function Dialog({ title, blurb, onClose, dismissable = true, footer, width = 560, children }: DialogProps) {
  const titleId = useId();
  const root = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    const node = root.current;
    const first = node?.querySelector<HTMLElement>("[data-autofocus]") ?? node?.querySelector<HTMLElement>(FOCUSABLE) ?? node;
    first?.focus({ preventScroll: true });
    return () => {
      if (before && document.contains(before)) before.focus({ preventScroll: true });
    };
  }, []);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === "Escape") {
      event.stopPropagation();
      event.preventDefault();
      if (dismissable) closeRef.current();
      return;
    }
    if (event.key !== "Tab") return;
    const items = Array.from(root.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []).filter((el) => el.offsetParent !== null || el === document.activeElement);
    if (items.length === 0) {
      event.preventDefault();
      return;
    }
    const firstItem = items[0] as HTMLElement;
    const lastItem = items[items.length - 1] as HTMLElement;
    if (event.shiftKey && (document.activeElement === firstItem || document.activeElement === root.current)) {
      event.preventDefault();
      lastItem.focus();
    } else if (!event.shiftKey && document.activeElement === lastItem) {
      event.preventDefault();
      firstItem.focus();
    }
  };

  return (
    <div
      className="ym-dialog-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && dismissable) closeRef.current();
      }}
    >
      <div
        ref={root}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="ym-dialog"
        style={{ width }}
        tabIndex={-1}
        onKeyDown={onKeyDown}
      >
        <div className="ym-dialog__head">
          <h2 id={titleId} className="ym-dialog__title">
            {title}
          </h2>
          {blurb ? <p className="ym-dialog__blurb">{blurb}</p> : null}
        </div>
        <div className="ym-dialog__body">{children}</div>
        {footer ? <div className="ym-dialog__footer">{footer}</div> : null}
      </div>
    </div>
  );
}
