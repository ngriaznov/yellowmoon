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
 */
export function TabBar({ value, onChange, label = "Tool tabs", tabs = TAB_IDS, className }: TabBarProps) {
  return (
    <Tabs value={value} onValueChange={(next) => onChange(next as TabId)} className="block">
      <TabsList variant="line" aria-label={label} data-tab={value} className={cx("flex w-full", className)}>
        {tabs.map((id) => (
          <TabsTrigger key={id} value={id} id={tabDomId(id)} aria-controls={tabPanelDomId(id)} data-tab={id} className="px-1 text-[0.8125rem]">
            {tabInfo(id).label}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
