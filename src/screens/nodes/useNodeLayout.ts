import { useCallback, useEffect, useState } from "react";
import { DEFAULT_FRACTION, EXPANDED_FRACTION, fractionOf, layoutOf } from "./layout";
import type { NodeLayout } from "./layout";

/** The window's height, followed while `enabled` (the node view is the only one that sizes itself from it). */
export function useWindowHeight(enabled: boolean): number {
  const [height, setHeight] = useState(() => (typeof window === "undefined" ? 900 : window.innerHeight));
  useEffect(() => {
    if (!enabled) return;
    const update = (): void => setHeight(window.innerHeight);
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [enabled]);
  return height;
}

/** What the node view's layout offers its divider, its Expand toggle and the grid. */
export interface NodeLayoutState extends NodeLayout {
  /** Expand is on: the editor takes 66 %. */
  expanded: boolean;
  /** Sets the editor's height (a drag), which ends Expand. */
  setEditor: (px: number) => void;
  /** Moves the editor's height by this much (a key), which ends Expand. */
  nudge: (deltaPx: number) => void;
  /** The divider's double click: back to 45 %, Expand off. */
  reset: () => void;
  /** Expand and back: the divider's own place is kept under it. */
  toggleExpanded: () => void;
}

/**
 * The divider's place and Expand, for as long as the Photos screen lives (window state: nothing about them is saved, `docs/23` §7.1).
 * The place is a share of the height under the header, so a window that grows or shrinks keeps the proportion.
 */
export function useNodeLayout(enabled: boolean): NodeLayoutState {
  const [place, setPlace] = useState({ fraction: DEFAULT_FRACTION, expanded: false });
  const windowHeight = useWindowHeight(enabled);
  const layout = layoutOf(windowHeight, place.expanded ? EXPANDED_FRACTION : place.fraction);
  const { area, editor } = layout;
  const setEditor = useCallback((px: number) => setPlace({ fraction: fractionOf(px, area), expanded: false }), [area]);
  const nudge = useCallback((deltaPx: number) => setPlace({ fraction: fractionOf(editor + deltaPx, area), expanded: false }), [area, editor]);
  const reset = useCallback(() => setPlace({ fraction: DEFAULT_FRACTION, expanded: false }), []);
  const toggleExpanded = useCallback(() => setPlace((was) => ({ ...was, expanded: !was.expanded })), []);
  return { ...layout, expanded: place.expanded, setEditor, nudge, reset, toggleExpanded };
}
