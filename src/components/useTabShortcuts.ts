import { useEffect } from "react";
import { useLatest } from "./hooks";
import { TAB_IDS } from "./tabs";
import type { TabId } from "./tabs";

/** The tab a key event selects: ⌘1 to ⌘7 (no Shift, Option or Control) in the order of `tabs` (the seven of the simple view, or a node's own: ⌘1 to ⌘5 at most), else null. */
export function tabForShortcut(
  event: Pick<KeyboardEvent, "metaKey" | "shiftKey" | "altKey" | "ctrlKey" | "code" | "key">,
  tabs: readonly TabId[] = TAB_IDS,
): TabId | null {
  if (!event.metaKey || event.shiftKey || event.altKey || event.ctrlKey) return null;
  const digit = /^Digit([1-7])$/.exec(event.code)?.[1] ?? (/^[1-7]$/.test(event.key) ? event.key : undefined);
  if (!digit) return null;
  return tabs[Number(digit) - 1] ?? null;
}

/**
 * ⌘1–⌘7 switch tabs from anywhere in the window, including while a value box has focus; in the node view `tabs` is the selected node's
 * own (⌘1–⌘5: the digits past its last tab are left to the system). Call it once in the Photos screen.
 */
export function useTabShortcuts(onSelect: (tab: TabId) => void, enabled = true, tabs: readonly TabId[] = TAB_IDS): void {
  const latest = useLatest(onSelect);
  const list = useLatest(tabs);
  useEffect(() => {
    if (!enabled) return;
    const handler = (event: KeyboardEvent): void => {
      const tab = tabForShortcut(event, list.current);
      if (!tab) return;
      event.preventDefault();
      latest.current(tab);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [enabled, latest, list]);
}
