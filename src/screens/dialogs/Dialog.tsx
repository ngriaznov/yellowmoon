import { useState } from "react";
import type { ReactNode } from "react";
import { Dialog as GlassDialog, DialogBody, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/glass/dialog";

export interface DialogProps {
  /** The title, at the top. Also the dialog's accessible name. */
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
 * A glass dialog over a dimmed window (Preferences, About, the question before closing a session during an export), glasscn's `Dialog`
 * on Radix. `role="dialog"`; focus moves into it when it opens (to the control marked `data-autofocus`, else the first one), Tab stays
 * inside, Esc closes it and focus goes back to where it was. Radix makes everything behind it inert, so a click on the dimmed window
 * cannot leave the focus there.
 */
export function Dialog({ title, blurb, onClose, dismissable = true, footer, width = 560, children }: DialogProps) {
  // Radix gives the focus back to the dialog's trigger; these dialogs open from a shortcut or a menu and have none, so the element that
  // had the focus when the dialog opened is remembered here.
  const [opener] = useState<Element | null>(() => (typeof document === "undefined" ? null : document.activeElement));
  return (
    <GlassDialog
      open
      onOpenChange={(open) => {
        if (!open && dismissable) onClose();
      }}
    >
      <DialogContent
        showCloseButton={false}
        style={{ width, maxWidth: "calc(100% - 2rem)" }}
        // Without a blurb there is no description to point at; saying so keeps Radix from warning about it.
        {...(blurb ? {} : { "aria-describedby": undefined })}
        onOpenAutoFocus={(event) => {
          // The control marked `data-autofocus` takes the focus, as it did before; otherwise Radix focuses the first one.
          const target = (event.currentTarget as HTMLElement).querySelector<HTMLElement>("[data-autofocus]");
          if (target) {
            event.preventDefault();
            target.focus({ preventScroll: true });
          }
        }}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          if (opener instanceof HTMLElement && opener.isConnected && opener !== document.body) opener.focus({ preventScroll: true });
        }}
        onEscapeKeyDown={(event) => {
          if (!dismissable) event.preventDefault();
        }}
        onInteractOutside={(event) => {
          if (!dismissable) event.preventDefault();
        }}
      >
        <DialogHeader>
          <DialogTitle className="text-2xl">{title}</DialogTitle>
          {blurb ? <DialogDescription>{blurb}</DialogDescription> : null}
        </DialogHeader>
        <DialogBody>{children}</DialogBody>
        {footer ? <DialogFooter className="items-center sm:justify-end">{footer}</DialogFooter> : null}
      </DialogContent>
    </GlassDialog>
  );
}
