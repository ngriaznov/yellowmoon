import { Maximize2Icon } from "lucide-react";
import type { DevelopSettings, NodeId } from "../../api/types";
import { Button } from "../../components";
import { GRAPH_OP_TEXT, MAX_CORRECTORS } from "../../state/graphOps";
import { correctorsOf, INPUT_ID, MAIN_ID, nodeOf, OUTPUT_ID } from "../../state/settingsGraph";
import { AddNodeMenu } from "./AddNodeMenu";
import { NodeGraph } from "./NodeGraph";
import { NodeMoreMenu } from "./NodeMoreMenu";
import { PhotoSwitcher } from "./PhotoSwitcher";
import type { PhotoSwitcherProps } from "./PhotoSwitcher";
import type { NodeGraphActions } from "./useNodeGraph";

export interface GraphAreaProps extends PhotoSwitcherProps {
  /** The photo's graph, or null while the photo has no settings yet. */
  settings: DevelopSettings | null;
  /** Changes with the photo. */
  photoKey: string | number | null;
  /** The node the graph and the panel are on. Main until a card is clicked. */
  selected: NodeId;
  /** What the toolbar, the menus, the keys and the canvas do (`useNodeGraph`). */
  graph: NodeGraphActions;
  /** The picture of a node's card (the photo's own for every node until N-05). */
  thumbnailOf: (id: NodeId) => string | undefined;
  /** Expand is on: the editor takes 66 % of the height under the header. */
  expanded: boolean;
  onToggleExpanded: () => void;
  /** The Crop panel or the Export panel is showing: the graph is dimmed and read only (`docs/23` §7.1). */
  dimmed: boolean;
}

/** Why the toolbar's Delete is off for a node, or null when it is on. */
function deleteReason(id: NodeId, link: boolean): string | null {
  if (link) return null;
  if (id === MAIN_ID) return GRAPH_OP_TEXT.mainDelete;
  return id === INPUT_ID || id === OUTPUT_ID ? GRAPH_OP_TEXT.markerDelete : null;
}

/**
 * The graph area of the node view (`docs/23` §7.1 to §7.3): a 44 px toolbar and the canvas under it, in the strip's glass. The toolbar has
 * Add node ▾, Bypass, Delete and More at the left; at the right the photo switcher, Show mask and Expand. The canvas is the graph itself
 * (`NodeGraph`: the cards, the markers, the ports and the links on @xyflow/react). Bypass and Delete act on the selected node (Delete on
 * the selected link first), with the sentence that says why when they are off; Show mask comes with the mask view (N-11). While Crop or
 * Export shows, the whole area is `inert` and dimmed.
 */
export function GraphArea({ settings, photoKey, selected, graph, thumbnailOf, expanded, onToggleExpanded, dimmed, ...switcher }: GraphAreaProps) {
  const node = settings ? nodeOf(settings, selected) : undefined;
  const count = settings ? correctorsOf(settings).length : 0;
  const bypassOff = node?.kind !== "corrector";
  const deleteOff = deleteReason(selected, graph.selectedLink !== null);
  return (
    <section aria-label="Node editor" className="ym-graph" data-dimmed={dimmed || undefined} inert={dimmed || undefined}>
      <div role="group" aria-label="Graph tools" className="ym-graph__bar">
        <AddNodeMenu onAddCorrector={settings ? graph.addCorrector : undefined} blocked={count >= MAX_CORRECTORS ? GRAPH_OP_TEXT.tooMany : null} />
        <Button
          variant="header"
          pressed={node?.kind === "corrector" && node.bypass}
          disabled={bypassOff}
          title={bypassOff ? GRAPH_OP_TEXT.noBypass : "Bypass this node (B)"}
          onClick={graph.toggleBypass}
        >
          Bypass
        </Button>
        <Button variant="header" disabled={deleteOff !== null} title={deleteOff ?? (graph.selectedLink ? "Delete the link" : "Delete this node")} onClick={graph.deleteSelection}>
          Delete
        </Button>
        {settings ? (
          <NodeMoreMenu
            settings={settings}
            node={selected}
            onRename={() => graph.startRename(selected)}
            onMoveEarlier={graph.moveEarlier}
            onMoveLater={graph.moveLater}
            onLink={graph.connect}
            onRemoveLink={graph.disconnect}
            onDelete={() => graph.deleteNode(selected)}
          />
        ) : null}
        <span className="ym-graph__spacer" />
        <PhotoSwitcher {...switcher} />
        <Button variant="header" pressed={false} disabled title="Show mask (M)">
          Show mask
        </Button>
        <Button variant="header" pressed={expanded} icon={<Maximize2Icon aria-hidden="true" />} onClick={onToggleExpanded}>
          Expand
        </Button>
      </div>
      <div className="ym-graph__canvas" data-graph data-selected-node={String(selected)}>
        {settings ? (
          <NodeGraph
            settings={settings}
            photoKey={photoKey}
            selected={selected}
            selectedLink={graph.selectedLink}
            renaming={graph.renaming}
            thumbnailOf={thumbnailOf}
            onSelect={graph.selectNode}
            onSelectLink={graph.selectLink}
            onStartRename={graph.startRename}
            onRename={graph.finishRename}
            onMove={graph.moveCard}
            onReorder={graph.reorderInto}
            onConnect={graph.connect}
            onReconnect={graph.reconnect}
            onDisconnect={graph.disconnect}
            onRefuse={graph.refuse}
          />
        ) : null}
      </div>
    </section>
  );
}
