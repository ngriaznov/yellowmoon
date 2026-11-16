import { memo } from "react";
import { Tabs, TabsList, TabsTrigger } from "@/components/glass/tabs";
import { cx } from "./cx";
import { tabDomId, tabInfo, tabPanelDomId, TAB_IDS } from "./tabs";
import type { TabId } from "./tabs";

export interface TabBarProps {
  value: TabId;
  onChange: (tab: TabId) => void;
  /** Accessible name of the bar. Default "Tool tabs". */
  label?: string;
  /** The tabs, in bar order: the seven of the simple view by default, a node's own in the node view (`NODE_TABS`). */
  tabs?: readonly TabId[];
  className?: string;
}

/**
 * The tool tabs (seven in the simple view, a node's own in the node view), as glasscn's `Tabs`: a glass capsule with a pill that slides to the chosen tab. `role="tablist"`. ←/→ move between
 * the tabs and select them (Home and End jump to the ends); ⌘1–⌘7 (⌘1–⌘5 in the node view) work from anywhere through `useTabShortcuts`. Each tab keeps the ids
 * `tabPanelProps(tab)` expects, so the panel the screen draws stays tied to it.
 *
 * **The list is never a Tab stop** (`tabIndex={-1}`; the selected tab is the one stop, and the arrows rove from it). Radix's roving group
 * takes `tabIndex=0` on its root and sends the focus on to the current tab when the Tab key enters it, but not after a click: Radix's
 * `TabsTrigger` focuses itself in `onMouseDown`, before the group's own `onMouseDown` (which runs later in the bubble) records "this
 * focus is a click", so that note stays set and the next Tab into the list is taken for a click focus and not redirected. The list then
 * keeps the focus itself, with `outline: none` (Radix's inline style), an invisible stop (`docs/28` F1). An explicit `tabIndex` on the list
 * overrides Radix's and the stop is gone for good. Radix gives a tab `tabindex=0` only once one has been focused (the group's "current tab
 * stop" starts empty, and the root's redirect was what made the first Tab work), so the selected tab names its own `tabIndex` too: it is the
 * one stop from the start, whatever was clicked before.
 */
export const TabBar = memo(function TabBar({ value, onChange, label = "Tool tabs", tabs = TAB_IDS, className }: TabBarProps) {
  return (
    <Tabs value={value} onValueChange={(next) => onChange(next as TabId)} className="block">
      <TabsList variant="line" tabIndex={-1} aria-label={label} data-tab={value} className={cx("flex w-full", className)}>
        {tabs.map((id) => (
          <TabsTrigger key={id} value={id} id={tabDomId(id)} tabIndex={id === value ? 0 : -1} aria-controls={tabPanelDomId(id)} data-tab={id} className="px-1 text-[0.8125rem]">
            {tabInfo(id).label}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
});
