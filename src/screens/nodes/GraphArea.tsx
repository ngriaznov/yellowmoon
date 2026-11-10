import { Maximize2Icon } from "lucide-react";
import type { DevelopSettings, NodeId } from "../../api/types";
import { Button } from "../../components";
import { GRAPH_OP_TEXT } from "../../state/graphOps";
import { INPUT_ID, isGlyphItem, MAIN_ID, nodeOf, numberedOf, OUTPUT_ID } from "../../state/settingsGraph";
import { AddNodeMenu } from "./AddNodeMenu";
import { NodeGraph } from "./NodeGraph";
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
  /** Show mask is on (`docs/23` §5.8): the toolbar's toggle is pressed while a corrector is selected. */
  maskOn?: boolean;
  /** The toolbar's Show mask: the same as M. */
  onToggleMask?: () => void;
  /** The node view is closed but kept built (U-02): the area takes no room, is not drawn and is not drawn again. */
  hidden?: boolean;
}

/** Why the toolbar's Delete is off for a node, or null when it is on. */
function deleteReason(id: NodeId, link: boolean): string | null {
  if (link) return null;
  if (id === MAIN_ID) return GRAPH_OP_TEXT.mainDelete;
  return id === INPUT_ID || id === OUTPUT_ID ? GRAPH_OP_TEXT.markerDelete : null;
}

/**
 * The graph area of the node view (`docs/23` §7.1 to §7.3): a 44 px toolbar and the canvas under it, in the strip's glass. The toolbar has
 * Add node ▾, Bypass and Delete at the left (More is the third icon of the node's head, `NodeHead`); at the right the photo switcher, Show mask and Expand. The canvas is the graph itself
 * (`NodeGraph`: the cards, the markers, the ports and the links on @xyflow/react). Bypass and Delete act on the selected node (Delete on
 * the selected link first), with the sentence that says why when they are off; Show mask turns the core's mask view on and off (N-11; off on the Input and the Output). While Crop or
 * Export shows, the whole area is `inert` and dimmed.
 */
export function GraphArea({ settings, photoKey, selected, graph, thumbnailOf, expanded, onToggleExpanded, dimmed, maskOn = false, onToggleMask, hidden = false, ...switcher }: GraphAreaProps) {
  const node = settings ? nodeOf(settings, selected) : undefined;
  const count = settings ? numberedOf(settings).length : 0;
  const isCard = node?.kind === "corrector" || node?.kind === "film";
  // A Splitter and a Combiner have no Bypass and no mask; a Parallel and a Layer have Bypass and no mask (`docs/25` §5.5).
  const noBypassKind = node?.kind === "splitter" || node?.kind === "combiner";
  const bypassOff = !isCard && !(isGlyphItem(node) && !noBypassKind);
  const deleteOff = deleteReason(selected, graph.selectedLink !== null);
  // The Input and the Output have no mask: the view's on or off is kept and applies again on a corrector.
  const maskOff = !isCard;
  const maskOffReason = isGlyphItem(node) ? GRAPH_OP_TEXT.glyphNoMask : GRAPH_OP_TEXT.noMask;
  return (
    <section aria-label="Node editor" className="ym-graph" data-dimmed={dimmed || undefined} inert={dimmed || hidden || undefined} hidden={hidden || undefined}>
      <div role="group" aria-label="Graph tools" className="ym-graph__bar">
        <AddNodeMenu
          onAddCorrector={settings ? graph.addCorrector : undefined}
          onAddFilm={settings ? graph.addFilm : undefined}
          onAddParallel={settings ? graph.addParallel : undefined}
          onAddLayer={settings ? graph.addLayer : undefined}
          onAddTrio={settings ? graph.addTrio : undefined}
          count={count}
        />
        <Button
          variant="header"
          pressed={node !== undefined && "bypass" in node && node.bypass}
          disabled={bypassOff}
          title={bypassOff ? (noBypassKind ? GRAPH_OP_TEXT.glyphNoBypass : GRAPH_OP_TEXT.noBypass) : "Bypass this node (B)"}
          onClick={graph.toggleBypass}
        >
          Bypass
        </Button>
        <Button variant="header" disabled={deleteOff !== null} title={deleteOff ?? (graph.selectedLink ? "Delete the link" : "Delete this node")} onClick={graph.deleteSelection}>
          Delete
        </Button>
        <span className="ym-graph__spacer" />
        <PhotoSwitcher {...switcher} />
        <Button
          variant="header"
          pressed={maskOn && !maskOff}
          disabled={maskOff}
          title={maskOff ? maskOffReason : "Show mask (M)"}
          onClick={onToggleMask}
        >
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
            renaming={graph.renamingIn === "card" ? graph.renaming : null}
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
