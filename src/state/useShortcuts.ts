import { useEffect } from "react";
import { useLatest } from "../components";
import { useAppState } from "./context";
import { shortcutFor } from "./shortcuts";
import type { Shortcut, ShortcutContext } from "./shortcuts";

/** True while "Single-key shortcuts" is on in Preferences (the default; `docs/28` A-P11). Read by `useShortcuts` and by the zoomed photo's I. */
export function useSingleKeys(): boolean {
  return useAppState((s) => s.preferences?.single_key_shortcuts ?? true);
}

/**
 * Listens for the shortcuts of `docs/11` on the window and hands each to `handle`. Return true from
 * `handle` when the shortcut was used, and the browser's own action is cancelled. Shortcuts that
 * a focused control already used (it called `preventDefault`) never arrive.
 */
export function useShortcuts(handle: (shortcut: Shortcut, event: KeyboardEvent) => boolean, context: ShortcutContext = {}, enabled = true): void {
  const handler = useLatest(handle);
  const singleKeys = useSingleKeys();
  const ctx = useLatest({ ...context, singleKeys });
  useEffect(() => {
    if (!enabled) return;
    const onKeyDown = (event: KeyboardEvent): void => {
      const shortcut = shortcutFor(event, ctx.current);
      if (shortcut && handler.current(shortcut, event)) event.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [enabled, handler, ctx]);
}
