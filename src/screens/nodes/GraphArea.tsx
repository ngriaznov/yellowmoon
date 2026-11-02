import { Maximize2Icon } from "lucide-react";
import { Button } from "../../components";
import type { NodeId } from "../../api/types";
import { AddNodeMenu } from "./AddNodeMenu";
import { PhotoSwitcher } from "./PhotoSwitcher";
import type { PhotoSwitcherProps } from "./PhotoSwitcher";

export interface GraphAreaProps extends PhotoSwitcherProps {
  /** The node the graph and the panel are on. Main until a card is clicked (N-08). */
  selected: NodeId;
  /** Expand is on: the editor takes 66 % of the height under the header. */
  expanded: boolean;
  onToggleExpanded: () => void;
  /** The Crop panel or the Export panel is showing: the graph is dimmed and read only (`docs/23` §7.1). */
  dimmed: boolean;
}

/**
 * The graph area of the node view (`docs/23` §7.1 and §7.3): a 44 px toolbar and the canvas under it, in the strip's glass. The toolbar
 * has Add node ▾, Bypass and Delete at the left; at the right the photo switcher, Show mask and Expand. **The canvas is empty**: the
 * cards, links, zoom and the keys come with N-08, and until then Bypass and Delete (which act on the selected card) and Show mask (the
 * mask view, N-11) are off, and Add node's Corrector is too, while the menu itself opens. While Crop or Export shows, the whole area is
 * `inert` and dimmed.
 */
export function GraphArea({ selected, expanded, onToggleExpanded, dimmed, ...switcher }: GraphAreaProps) {
  return (
    <section aria-label="Node editor" className="ym-graph" data-dimmed={dimmed || undefined} inert={dimmed || undefined}>
      <div role="group" aria-label="Graph tools" className="ym-graph__bar">
        <AddNodeMenu />
        <Button variant="header" pressed={false} disabled title="Bypass this node (B)">
          Bypass
        </Button>
        <Button variant="header" disabled>
          Delete
        </Button>
        <span className="ym-graph__spacer" />
        <PhotoSwitcher {...switcher} />
        <Button variant="header" pressed={false} disabled title="Show mask (M)">
          Show mask
        </Button>
        <Button variant="header" pressed={expanded} icon={<Maximize2Icon aria-hidden="true" />} onClick={onToggleExpanded}>
          Expand
        </Button>
      </div>
      <div className="ym-graph__canvas" data-selected-node={String(selected)} />
    </section>
  );
}
