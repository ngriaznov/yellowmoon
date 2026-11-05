import { createContext, memo, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import type { TabId } from "../../components";

/** Whether the tab pane a component sits in is the one on show. True outside a pane (the node view's panel, tests). */
const PaneActive = createContext(true);

/** True while the component's tab is on show. A watcher (the histogram, the colour plane's scatter) stops while it is false (U-02). */
export function usePaneActive(): boolean {
  return useContext(PaneActive);
}

/**
 * False for the render that mounts the caller and true from the second animation frame on. A heavy plot (the colour plane's canvases)
 * is drawn when this is true, so the tab appears in its first frame and the plot follows it (U-02). The caller draws a placeholder of the
 * plot's size meanwhile.
 */
export function useAfterFirstFrame(): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let second = 0;
    const first = requestAnimationFrame(() => {
      second = requestAnimationFrame(() => setReady(true));
    });
    return () => {
      cancelAnimationFrame(first);
      cancelAnimationFrame(second);
    };
  }, []);
  return ready;
}

/** What takes a hidden pane out of the sight, the tab order and the accessibility tree (the rest is in `screens.css`). */
const HIDDEN = { visibility: "hidden" } as const;

interface TabPaneProps {
  id: TabId;
  active: boolean;
  children?: ReactNode;
}

/**
 * One tab's content. A tab that was opened once stays mounted and is hidden when another is shown, so coming back to it
 * mounts nothing (U-02). While it is hidden it is not drawn again when the screen draws (a slider's tick gives the screen new
 * settings every frame); it catches up in the render that shows it.
 */
export const TabPane = memo(
  function TabPane({ id, active, children }: TabPaneProps) {
    return (
      // Not `hidden` (display: none): a pane with no box is measured as empty (the sliders' thumbs, the segmented controls' indicators), and
      // showing it again would measure and draw it a second time. The pane keeps its width and has no height (`screens.css`).
      <div data-pane={id} data-hidden={!active || undefined} aria-hidden={!active || undefined} inert={!active} style={active ? undefined : HIDDEN}>
        <PaneActive value={active}>{children}</PaneActive>
      </div>
    );
  },
  (prev, next) => !prev.active && !next.active && prev.id === next.id,
);
