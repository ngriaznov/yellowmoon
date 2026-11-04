import { createContext, memo, useContext } from "react";
import type { ReactNode } from "react";
import type { TabId } from "../../components";

/** Whether the tab pane a component sits in is the one on show. True outside a pane (the node view's panel, tests). */
const PaneActive = createContext(true);

/** True while the component's tab is on show. A watcher (the histogram, the colour plane's scatter) stops while it is false (U-02). */
export function usePaneActive(): boolean {
  return useContext(PaneActive);
}

interface TabPaneProps {
  id: TabId;
  active: boolean;
  children?: ReactNode;
}

/**
 * One tab's content. A tab that was opened once stays mounted and is hidden (`hidden`) when another is shown, so coming back to it
 * mounts nothing (U-02). While it is hidden it is not drawn again when the screen draws (a slider's tick gives the screen new
 * settings every frame); it catches up in the render that shows it.
 */
export const TabPane = memo(
  function TabPane({ id, active, children }: TabPaneProps) {
    return (
      <div data-pane={id} hidden={!active || undefined}>
        <PaneActive value={active}>{children}</PaneActive>
      </div>
    );
  },
  (prev, next) => !prev.active && !next.active && prev.id === next.id,
);
