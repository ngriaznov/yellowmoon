import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { isTauri } from "../env";

/** What the full-screen view needs of the app window: Tauri's `Window` has both. */
export interface FullScreenWindow {
  isFullscreen(): Promise<boolean>;
  setFullscreen(fullscreen: boolean): Promise<void>;
}

/** The app's window inside Tauri; null in a browser, in Vitest and in Playwright, where only the page's layout changes. */
async function appWindow(): Promise<FullScreenWindow | null> {
  if (!isTauri()) return null;
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  return getCurrentWindow();
}

/** Puts the window in macOS full screen with the view, and back. */
export interface WindowFollower {
  enter(): Promise<void>;
  leave(): Promise<void>;
}

/**
 * The window follows the full-screen view: entering puts it in macOS full screen unless it already is (the green button, ⌃⌘F), and
 * leaving takes it out only when entering put it there, so a window the person made full screen stays so. The calls run one after
 * the other (F pressed twice quickly cannot leave the window the wrong way round), and a call that fails changes nothing on the page.
 */
export function windowFollower(getWindow: () => Promise<FullScreenWindow | null> = appWindow): WindowFollower {
  let queue: Promise<void> = Promise.resolve();
  let entered = false;
  const run = (step: (win: FullScreenWindow) => Promise<void>): Promise<void> => {
    queue = queue
      .then(async () => {
        const win = await getWindow();
        if (win) await step(win);
      })
      .catch(() => {});
    return queue;
  };
  return {
    enter: () =>
      run(async (win) => {
        if (entered || (await win.isFullscreen())) return;
        await win.setFullscreen(true);
        entered = true;
      }),
    leave: () =>
      run(async (win) => {
        if (!entered) return;
        entered = false;
        await win.setFullscreen(false);
      }),
  };
}

export interface FullScreen {
  /** The photo fills the window. */
  on: boolean;
  /** F: in or out. Does nothing while the view is not `allowed`. */
  toggle(): void;
  /** F, Esc, and anything that needs the panel (Crop, Pick white, a tab, Export). */
  leave(): void;
}

/**
 * The Photos screen's full-screen view (F, `docs/11` §Full screen): the state, and the window, which goes into macOS full screen with
 * it. The view is off while it is not `allowed` (the Export panel or Crop is open: both need the panel), and opening either from full
 * screen (⌘E, the File menu, C) leaves it for good. On entering, a control that keeps the focus and is now hidden gives it up (the page
 * takes it; the photo cell's own layers keep it), so the arrows and the other keys reach the screen. Leaving the screen (⌘W, the
 * wizard) puts the window back as it was.
 */
export function useFullScreen(allowed: boolean, getWindow?: () => Promise<FullScreenWindow | null>): FullScreen {
  const [wanted, setWanted] = useState(false);
  const on = wanted && allowed;
  // One follower for the screen's life: it remembers whether it put the window in full screen.
  const [follower] = useState(() => windowFollower(getWindow));

  // The Export panel or Crop opened while in full screen: it ends, and does not come back when they close.
  useEffect(() => {
    if (wanted && !allowed) setWanted(false);
  }, [wanted, allowed]);

  useEffect(() => {
    if (!on) return;
    void follower.enter();
    return () => void follower.leave();
  }, [on, follower]);

  useLayoutEffect(() => {
    if (!on) return;
    const focused = document.activeElement;
    if (focused instanceof HTMLElement && focused !== document.body && !focused.closest("[data-viewport]")) focused.blur();
  }, [on]);

  const toggle = useCallback(() => {
    if (allowed) setWanted((was) => !was);
  }, [allowed]);
  const leave = useCallback(() => setWanted(false), []);
  return { on, toggle, leave };
}
