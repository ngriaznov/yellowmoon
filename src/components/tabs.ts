/**
 * The seven tool tabs of the Photos screen (`docs/11-ui.md` §Photos). Each tab owns a flat colour
 * that fills its tab, its title block, its slider fills and its changed value boxes. The
 * colours live in `styles/tokens.css` (`--tab-look` …); `components.css` maps a `data-tab`
 * attribute onto `--tab` and `--tab-ink` so every component below an element with
 * `data-tab="light"` picks the Light colour up without a prop.
 */

import type { NodeKind } from "../api/types";

/** Tab ids. American spelling in identifiers; the label reads "Colour". `mask` is the corrector's Mask tab of the node view (`docs/23` §7.6). */
export type TabId = "look" | "lens" | "light" | "color" | "detail" | "curve" | "raw" | "mask" | "film" | "print" | "halation" | "grain" | "inputs" | "blend" | "channels";

/** The tabs that only the node view has: the Mask tab, a Film node's four and the one tab of each of the nodes of `docs/25`. */
export type NodeOnlyTabId = "mask" | "film" | "print" | "halation" | "grain" | "inputs" | "blend" | "channels";

/** One tab: its id, the caps label and the blurb under the title (`design/mockups/photos.html`). */
export interface TabInfo {
  id: TabId;
  label: string;
  blurb: string;
}

/** The tabs of the simple view in bar order. ⌘1 selects the first, ⌘7 the last. */
export const TABS: readonly TabInfo[] = [
  { id: "look", label: "Look", blurb: "Every photo is developed automatically. Pick a look and how strongly to apply it." },
  {
    id: "lens",
    label: "Lens",
    blurb: "How a lens draws light: flares, softer edges, the shape of blur. Off until you raise a slider.",
  },
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

/** The Mask tab of a corrector in the node view: the mask accent (`--mask`). It has no place in the simple view's bar. */
export const MASK_TAB: TabInfo = { id: "mask", label: "Mask", blurb: "Where in the photo this node acts." };

/** The tabs of a Film node (`docs/24` section 4.1): they have no place in the simple view's bar, and they take the Look tab's yellow. */
export const FILM_TABS: Readonly<Record<"film" | "print" | "halation" | "grain", TabInfo>> = {
  film: { id: "film", label: "Film", blurb: "The film stock and how much of its look is applied." },
  print: { id: "print", label: "Print", blurb: "The paper the negative is printed on, and the enlarger's light." },
  halation: { id: "halation", label: "Halation", blurb: "The glow around bright edges." },
  grain: { id: "grain", label: "Grain", blurb: "The film's grain." },
};

/** The one tab of a Parallel (Inputs), a Layer (Blend), a Splitter and a Combiner (Channels; `docs/25` §5.5): they take the Look tab's yellow. */
export const MIXER_TABS: Readonly<Record<"inputs" | "blend" | "channels", TabInfo>> = {
  inputs: { id: "inputs", label: "Inputs", blurb: "Adds the change each input makes to the picture they start from." },
  blend: { id: "blend", label: "Blend", blurb: "How the inputs are laid over each other." },
  channels: { id: "channels", label: "Channels", blurb: "The red, green and blue of the picture." },
};

/** Tab ids of the simple view in bar order. */
export const TAB_IDS: readonly TabId[] = TABS.map((tab) => tab.id);

/** The kinds of node a panel can be on: one for each `NodeItem.kind`. */
export type PanelKind = NodeKind;

/** The panel a node of any kind opens: its own (`NODE_TABS`). */
export function panelKindOf(kind: NodeKind): PanelKind {
  return kind;
}

/**
 * The tabs of each kind of node in the node view, in bar order (`docs/23` §7.5): the Input RAW · Lens · Light · Colour · Detail, a corrector
 * (Main included) Light · Colour · Curve · Mask, the Output Look · Detail. ⌘1 selects the first, up to ⌘5 on the Input.
 */
export const NODE_TABS: Readonly<Record<PanelKind, readonly TabId[]>> = {
  input: ["raw", "lens", "light", "color", "detail"],
  corrector: ["light", "color", "curve", "mask"],
  film: ["film", "print", "halation", "grain", "mask"],
  output: ["look", "detail"],
  parallel: ["inputs"],
  layer: ["blend"],
  splitter: ["channels"],
  combiner: ["channels"],
};

/** The tab each kind of node opens on the first time (and when a remembered tab is not its own). */
export const FIRST_NODE_TAB: Readonly<Record<PanelKind, TabId>> = {
  input: "raw",
  corrector: "light",
  film: "film",
  output: "look",
  parallel: "inputs",
  layer: "blend",
  splitter: "channels",
  combiner: "channels",
};

/**
 * The node and the tab the node view opens on when it is entered from a tab of the simple view (`docs/23` §7.5 *Tab memory*, decision 21):
 * Look → Output › Look, Lens → Input › Lens, Light → Main › Light, Colour → Main › Colour, Detail → Input › Detail, Curve → Main › Curve,
 * RAW → Input › RAW.
 */
export const NODE_FOR_SIMPLE_TAB: Readonly<Record<Exclude<TabId, NodeOnlyTabId>, { kind: PanelKind; tab: TabId }>> = {
  look: { kind: "output", tab: "look" },
  lens: { kind: "input", tab: "lens" },
  light: { kind: "corrector", tab: "light" },
  color: { kind: "corrector", tab: "color" },
  detail: { kind: "input", tab: "detail" },
  curve: { kind: "corrector", tab: "curve" },
  raw: { kind: "input", tab: "raw" },
};

/** The simple view's tab that stands for `tab`: itself when it is one of the seven, Light for the node view's own (Mask, a Film node's and the mixers'). */
export function simpleTabOf(tab: TabId): Exclude<TabId, NodeOnlyTabId> {
  return tab === "mask" || tab === "film" || tab === "print" || tab === "halation" || tab === "grain" || tab === "inputs" || tab === "blend" || tab === "channels" ? "light" : tab;
}

/** The tab info for an id. */
export function tabInfo(id: TabId): TabInfo {
  const info =
    id === "mask"
      ? MASK_TAB
      : id === "film" || id === "print" || id === "halation" || id === "grain"
        ? FILM_TABS[id]
        : id === "inputs" || id === "blend" || id === "channels"
          ? MIXER_TABS[id]
          : TABS.find((tab) => tab.id === id);
  // TABS and the Mask tab hold every TabId, so this cannot be undefined.
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
