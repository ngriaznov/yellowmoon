import { useRef } from "react";
import type { Overlay } from "../../state/store";
import { useAppState } from "../../state/context";
import { About } from "./About";
import { AutoConsent } from "./AutoConsent";
import { ConfirmClose } from "./ConfirmClose";
import { CubeImport } from "./CubeImport";
import { DialogOpenerContext } from "./Dialog";
import { Preferences } from "./Preferences";

/**
 * Who opened the dialog that shows: the element that had the focus when the first dialog of a run opened. A dialog that replaces another
 * (About from Preferences' footer, the .cube import from Preferences, the close question over either) inherits it, because the control
 * that opened the dialog being replaced is gone with it: Esc from About returns to what opened Preferences, not to a "body" the About link
 * left behind (`docs/28` F3). The consent question over Preferences is a dialog over a dialog and keeps its own (`DialogOpenerContext`).
 */
export function nextOpener(was: { overlay: Overlay | null; opener: Element | null }, overlay: Overlay | null, focused: Element | null): Element | null {
  if (overlay === null) return null;
  return was.overlay === null ? focused : was.opener;
}

/** The dialog laid over the screen, if any (`AppState.overlay`). */
export function Overlays() {
  const overlay = useAppState((s) => s.overlay);
  // Read while the screen is drawn for the new overlay, before it is committed: the focus is still on the control that opened it. Written
  // once per change of overlay, so a second draw (StrictMode) finds the same answer.
  const opened = useRef<{ overlay: Overlay | null; opener: Element | null }>({ overlay: null, opener: null });
  if (opened.current.overlay !== overlay) {
    opened.current = { overlay, opener: nextOpener(opened.current, overlay, typeof document === "undefined" ? null : document.activeElement) };
  }
  const opener = opened.current.opener;
  switch (overlay) {
    case "preferences":
      // The consent dialog (`docs/22` section 12.3) opens over Preferences, so Esc closes only it.
      return (
        <>
          <DialogOpenerContext.Provider value={opener}>
            <Preferences />
          </DialogOpenerContext.Provider>
          <AutoConsent />
        </>
      );
    case "about":
      return (
        <DialogOpenerContext.Provider value={opener}>
          <About />
        </DialogOpenerContext.Provider>
      );
    case "confirm-close":
      return (
        <DialogOpenerContext.Provider value={opener}>
          <ConfirmClose />
        </DialogOpenerContext.Provider>
      );
    case "import-cube":
      return (
        <DialogOpenerContext.Provider value={opener}>
          <CubeImport />
        </DialogOpenerContext.Provider>
      );
    default:
      return null;
  }
}
