import { useEffect } from "react";
import { useLatest } from "./hooks";
import { TABS } from "./tabs";
import type { TabId } from "./tabs";

/** The tab a key event selects: ⌘1 to ⌘7 (no Shift, Option or Control), else null. */
export function tabForShortcut(event: Pick<KeyboardEvent, "metaKey" | "shiftKey" | "altKey" | "ctrlKey" | "code" | "key">): TabId | null {
  if (!event.metaKey || event.shiftKey || event.altKey || event.ctrlKey) return null;
  const digit = /^Digit([1-7])$/.exec(event.code)?.[1] ?? (/^[1-7]$/.test(event.key) ? event.key : undefined);
  if (!digit) return null;
  return TABS[Number(digit) - 1]?.id ?? null;
}

/**
 * ⌘1–⌘7 switch tabs from anywhere in the window, including while a value box has focus. Call
 * it once in the Photos screen.
 */
export function useTabShortcuts(onSelect: (tab: TabId) => void, enabled = true): void {
  const latest = useLatest(onSelect);
  useEffect(() => {
    if (!enabled) return;
    const handler = (event: KeyboardEvent): void => {
      const tab = tabForShortcut(event);
      if (!tab) return;
      event.preventDefault();
      latest.current(tab);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [enabled, latest]);
}
