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

/**
 * What Tab reaches. A button locked with `aria-disabled` is still in the tab order (`Button` keeps it there so it can explain itself), so
 * it counts here: leaving it out made the real last stop look like the one before it, and Tab walked out of the dialog. A radio that is
 * not the one selected has `tabindex="-1"`, and Tab skips it.
 */
const FOCUSABLE = [
  'button:not([disabled]):not([tabindex="-1"])',
  '[href]:not([tabindex="-1"])',
  'input:not([disabled]):not([tabindex="-1"])',
  'select:not([disabled]):not([tabindex="-1"])',
  'textarea:not([disabled]):not([tabindex="-1"])',
  '[tabindex]:not([tabindex="-1"])',
].join(", ");

/**
 * A square, boxed dialog over a dimmed window (Preferences, About, the question before closing a session during an export).
 * `role="dialog"` and `aria-modal`; focus moves into it when it opens (to the control marked `data-autofocus`, else the first
 * one), Tab stays inside, Esc closes it and focus goes back to where it was. If the focus is somewhere behind the dialog anyway
 * (a click on the dimmed window), Tab brings it back in and Esc still closes the dialog.
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

  const dismissableRef = useRef(dismissable);
  dismissableRef.current = dismissable;

  /** Esc closes the dialog (unless it can't be dismissed), Tab wraps. `outside` is true when the focus is not in the dialog. */
  const handleKey = (event: Pick<globalThis.KeyboardEvent, "key" | "shiftKey" | "preventDefault" | "stopPropagation">, outside: boolean): void => {
    if (event.key === "Escape") {
      event.stopPropagation();
      event.preventDefault();
      if (dismissableRef.current) closeRef.current();
      return;
    }
    if (event.key !== "Tab") return;
    const items = Array.from(root.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []).filter((el) => el.offsetParent !== null || el === document.activeElement);
    if (items.length === 0) {
      event.preventDefault();
      root.current?.focus({ preventScroll: true });
      return;
    }
    const firstItem = items[0] as HTMLElement;
    const lastItem = items[items.length - 1] as HTMLElement;
    if (outside) {
      event.preventDefault();
      (event.shiftKey ? lastItem : firstItem).focus();
    } else if (event.shiftKey && (document.activeElement === firstItem || document.activeElement === root.current)) {
      event.preventDefault();
      lastItem.focus();
    } else if (!event.shiftKey && document.activeElement === lastItem) {
      event.preventDefault();
      firstItem.focus();
    }
  };
  const handleKeyRef = useRef(handleKey);
  handleKeyRef.current = handleKey;

  // Keys pressed while the focus is not in the dialog (a click on the dimmed window left it on the page behind): the dialog's own
  // handler never sees them.
  useEffect(() => {
    const onDocumentKeyDown = (event: globalThis.KeyboardEvent): void => {
      const node = root.current;
      if (!node || (event.target instanceof Node && node.contains(event.target))) return;
      handleKeyRef.current(event, true);
    };
    document.addEventListener("keydown", onDocumentKeyDown);
    return () => document.removeEventListener("keydown", onDocumentKeyDown);
  }, []);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => handleKey(event, false);

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
