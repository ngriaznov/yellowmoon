import { useEffect } from "react";
import { useLatest } from "../components";
import { shortcutFor } from "./shortcuts";
import type { Shortcut, ShortcutContext } from "./shortcuts";

/**
 * Listens for the shortcuts of `docs/11` on the window and hands each to `handle`. Return true from
 * `handle` when the shortcut was used, and the browser's own action is cancelled. Shortcuts that
 * a focused control already used (it called `preventDefault`) never arrive.
 */
export function useShortcuts(handle: (shortcut: Shortcut, event: KeyboardEvent) => boolean, context: ShortcutContext = {}, enabled = true): void {
  const handler = useLatest(handle);
  const ctx = useLatest(context);
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
