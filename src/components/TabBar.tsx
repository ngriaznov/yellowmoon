import { Tabs, TabsList, TabsTrigger } from "@/components/glass/tabs";
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
 * The six tool tabs, as glasscn's `Tabs`: a glass capsule with a pill that slides to the chosen tab. `role="tablist"`. ←/→ move between
 * the tabs and select them (Home and End jump to the ends); ⌘1–⌘6 work from anywhere through `useTabShortcuts`. Each tab keeps the ids
 * `tabPanelProps(tab)` expects, so the panel the screen draws stays tied to it.
 */
export function TabBar({ value, onChange, label = "Tool tabs", className }: TabBarProps) {
  return (
    <Tabs value={value} onValueChange={(next) => onChange(next as TabId)} className="block">
      <TabsList variant="line" aria-label={label} data-tab={value} className={cx("flex w-full", className)}>
        {TABS.map((tab) => (
          <TabsTrigger
            key={tab.id}
            value={tab.id}
            id={tabDomId(tab.id)}
            aria-controls={tabPanelDomId(tab.id)}
            data-tab={tab.id}
            className="px-1 text-[0.8125rem]"
          >
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
