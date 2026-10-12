import { useEffect, useRef } from "react";
import type { KeyboardEvent } from "react";
import { cx } from "./cx";
import { tabDomId, tabPanelDomId, TABS } from "./tabs";
import type { TabId } from "./tabs";

export interface TabBarProps {
  value: TabId;
  onChange: (tab: TabId) => void;
  /** Accessible name of the bar. Default "Tool tabs". */
  label?: string;
  className?: string;
}

/**
 * The six tool tabs: equal cells 46 px high, 12 px 800 caps, 3 px separators; the active cell
 * is filled with its tab colour. `role="tablist"`. ←/→ move between the tabs and select them
 * (Home and End jump to the ends); ⌘1–⌘6 work from anywhere through `useTabShortcuts`.
 * Pair each tab with a panel from `tabPanelProps(tab)`.
 */
export function TabBar({ value, onChange, label = "Tool tabs", className }: TabBarProps) {
  const refs = useRef<Partial<Record<TabId, HTMLButtonElement | null>>>({});
  const focusAfter = useRef<TabId | null>(null);

  // Keep focus on the tab that keyboard navigation just selected.
  useEffect(() => {
    if (focusAfter.current === value) refs.current[value]?.focus();
    focusAfter.current = null;
  }, [value]);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const index = TABS.findIndex((t) => t.id === value);
    let next: number;
    switch (event.key) {
      case "ArrowRight":
        next = (index + 1) % TABS.length;
        break;
      case "ArrowLeft":
        next = (index - 1 + TABS.length) % TABS.length;
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = TABS.length - 1;
        break;
      default:
        return;
    }
    event.preventDefault();
    const target = TABS[next];
    if (!target) return;
    focusAfter.current = target.id;
    if (target.id === value) refs.current[value]?.focus();
    else onChange(target.id);
  };

  return (
    <div role="tablist" aria-label={label} className={cx("ym-tab-bar", className)} onKeyDown={onKeyDown}>
      {TABS.map((tab) => (
        <button
          key={tab.id}
          ref={(node) => {
            refs.current[tab.id] = node;
          }}
          type="button"
          role="tab"
          id={tabDomId(tab.id)}
          aria-controls={tabPanelDomId(tab.id)}
          aria-selected={tab.id === value}
          tabIndex={tab.id === value ? 0 : -1}
          className="ym-tab-bar__tab"
          data-tab={tab.id}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
