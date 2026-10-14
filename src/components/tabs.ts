/**
 * The six tool tabs of the Photos screen (`docs/11-ui.md` §Photos). Each tab owns a flat colour
 * that fills its tab, its title block, its slider fills and its changed value boxes. The
 * colours live in `styles/tokens.css` (`--tab-look` …); `components.css` maps a `data-tab`
 * attribute onto `--tab` and `--tab-ink` so every component below an element with
 * `data-tab="light"` picks the Light colour up without a prop.
 */

/** Tab ids. American spelling in identifiers; the label reads "Colour". */
export type TabId = "look" | "light" | "color" | "detail" | "curve" | "raw";

/** One tab: its id, the caps label and the blurb under the title (`design/mockups/photos.html`). */
export interface TabInfo {
  id: TabId;
  label: string;
  blurb: string;
}

/** The tabs in bar order. ⌘1 selects the first, ⌘6 the last. */
export const TABS: readonly TabInfo[] = [
  { id: "look", label: "Look", blurb: "Every photo is developed automatically. Pick a look and how strongly to apply it." },
  { id: "light", label: "Light", blurb: "Exposure starts at your camera's own. Adjust it from there." },
  {
    id: "color",
    label: "Colour",
    blurb: "White balance comes from the light in the scene. Set it by hand when you want a different mood.",
  },
  { id: "detail", label: "Detail", blurb: "Sharpening and noise reduction, tuned for each ISO of your camera." },
  { id: "curve", label: "Curve", blurb: "Fine contrast, for all channels or one at a time." },
  { id: "raw", label: "RAW", blurb: "How the sensor data is decoded before any colour is applied." },
];

/** Tab ids in bar order. */
export const TAB_IDS: readonly TabId[] = TABS.map((tab) => tab.id);

/** The tab info for an id. */
export function tabInfo(id: TabId): TabInfo {
  const info = TABS.find((tab) => tab.id === id);
  // TABS holds every TabId, so this cannot be undefined.
  return info as TabInfo;
}

/** DOM id of a tab button, for `aria-labelledby` and `aria-controls`. */
export const tabDomId = (id: TabId): string => `ym-tab-${id}`;
/** DOM id of the panel a tab controls. */
export const tabPanelDomId = (id: TabId): string => `ym-tabpanel-${id}`;

/**
 * Props for the element that holds a tab's content: the tabpanel role, ids that link it to its
 * tab button, and `data-tab` so everything inside picks up the tab colour.
 */
export function tabPanelProps(id: TabId): {
  role: "tabpanel";
  id: string;
  "aria-labelledby": string;
  "data-tab": TabId;
} {
  return { role: "tabpanel", id: tabPanelDomId(id), "aria-labelledby": tabDomId(id), "data-tab": id };
}
