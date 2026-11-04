import type { ReactNode } from "react";
import type { NodeId, PhotoSummary, ResolvedSettings } from "../../api/types";
import { NODE_TABS, TabBar, tabPanelProps } from "../../components";
import type { TabId } from "../../components";
import { isReady } from "../../state/photo";
import { nodeOf } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { NodeHead } from "./NodeHead";
import { NodeMoreMenu } from "./NodeMoreMenu";
import { NodeTab } from "./NodeTabs";
import type { NodeGraphActions } from "./useNodeGraph";

export interface NodePanelProps {
  store: AppStore;
  photo: PhotoSummary | null;
  rs: ResolvedSettings | null;
  /** The selected node. */
  node: NodeId;
  /** The tab it shows (one of `NODE_TABS` of its kind). */
  tab: TabId;
  onTab: (tab: TabId) => void;
  /** What the head's buttons and the More menu do (`useNodeGraph`). */
  graph: NodeGraphActions;
  /** The Colour tab's picks: Pick white (the Input's) and Pick colour (a corrector's), kept by `Photos`. */
  pickingWhite: boolean;
  onPickWhite: (on: boolean) => void;
  pickingColor: boolean;
  onPickColor: (on: boolean) => void;
  /** The Mask tab's pick: Pick colour of a Colour range. */
  pickingRange: boolean;
  onPickRange: (on: boolean) => void;
  /** The row at the foot: Copy settings and Paste, which act on the whole photo here too. */
  foot: ReactNode;
  /** Full screen: the panel is not shown. */
  hidden?: boolean;
}

/**
 * The tool panel of the node view (`docs/23` §7.5): it edits the selected node. A compact head (`NodeHead`: the number chip, the name, the
 * quiet line, Bypass, Reset and More), the tabs of the node's kind (Input RAW · Lens · Light · Colour · Detail, a corrector Light · Colour
 * · Curve · Mask, the Output Look · Detail), the cards of the tab (`NodeTab`) and the foot row. There is no title block, no Automatic
 * card and no Tone card here, and no notice of what the simple view cannot show. Every control writes the selected node.
 */
export function NodePanel({ store, photo, rs, node, tab, onTab, graph, pickingWhite, onPickWhite, pickingColor, onPickColor, pickingRange, onPickRange, foot, hidden }: NodePanelProps) {
  const item = rs ? nodeOf(rs.settings, node) : undefined;
  if (!photo || !rs || !item || !isReady(photo)) {
    return (
      <aside aria-label="Tools" className="ym-tools" data-node-panel hidden={hidden || undefined}>
        <div className="ym-tools__scroll">
          {photo ? (
            <p className="ym-tools__notice">
              {photo.status.kind === "skipped" ? "There is nothing to adjust: this file can't be opened." : "This photo can be adjusted when it is developed."}
            </p>
          ) : null}
        </div>
        <div className="ym-tools__foot" role="group" aria-label="Copy and paste settings">
          {foot}
        </div>
      </aside>
    );
  }
  const settings = rs.settings;
  const more =
    item.kind === "output" ? undefined : (
      <NodeMoreMenu
        settings={settings}
        node={node}
        side="bottom"
        buttonClassName="ym-nodehead__btn"
        onRename={() => graph.startRename(node, "head")}
        onMoveEarlier={graph.moveEarlier}
        onMoveLater={graph.moveLater}
        onLink={graph.connect}
        onRemoveLink={graph.disconnect}
        onDelete={() => graph.deleteNode(node)}
      />
    );
  return (
    <aside aria-label="Tools" className="ym-tools" data-node-panel data-node-kind={item.kind} hidden={hidden || undefined}>
      <NodeHead
        settings={settings}
        node={node}
        renaming={graph.renaming === node && graph.renamingIn === "head"}
        onStartRename={() => graph.startRename(node, "head")}
        onRename={(name) => graph.finishRename(node, name)}
        onBypass={graph.toggleBypass}
        onReset={graph.resetNode}
        more={more}
      />
      <TabBar value={tab} onChange={onTab} tabs={NODE_TABS[item.kind]} />
      <div className="ym-tools__scroll" {...tabPanelProps(tab)}>
        <NodeTab
          key={String(node)}
          store={store}
          photo={photo}
          rs={rs}
          node={node}
          tab={tab}
          pickingWhite={pickingWhite}
          onPickWhite={onPickWhite}
          pickingColor={pickingColor}
          onPickColor={onPickColor}
          pickingRange={pickingRange}
          onPickRange={onPickRange}
        />
      </div>
      <div className="ym-tools__foot" role="group" aria-label="Copy and paste settings">
        {foot}
      </div>
    </aside>
  );
}
