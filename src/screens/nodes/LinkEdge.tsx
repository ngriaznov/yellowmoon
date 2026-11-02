import { BaseEdge, getBezierPath, Position } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";
import type { NodeLink } from "../../api/types";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "@/components/glass/context-menu";
import { sameLink } from "../../state/graphOps";
import { useGraphUi } from "./graphUi";

/** The ids of the arrowheads the links end in (`LinkMarkers`). */
export const ARROW = { picture: "ym-arrow-picture", key: "ym-arrow-key", selected: "ym-arrow-selected" } as const;

/**
 * The arrowheads, defined once for every link (an SVG `marker` is found by id in the whole document): a small triangle in the colour of
 * the link, so that the link and the port glyph beside it make the double arrowhead of `design/nodes/05-nodes-default.jpg`.
 */
export function LinkMarkers() {
  return (
    <svg className="ym-link-markers" width="0" height="0" aria-hidden="true" focusable="false">
      <defs>
        {Object.entries(ARROW).map(([kind, id]) => (
          <marker key={id} id={id} markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">
            <path className={`ym-arrow ym-arrow--${kind}`} d="M0 0 L7 3.5 L0 7 Z" />
          </marker>
        ))}
      </defs>
    </svg>
  );
}

/** What xyflow is told about a link: its ends are the link's own. */
export interface LinkEdgeData extends Record<string, unknown> {
  link: NodeLink;
}

/**
 * A link between two ports (`docs/23` §7.2): a grey line with an arrowhead into the next input for a picture, dashed in the mask accent
 * for a key; thick and yellow when it is selected, and lit when a card dragged over it would be put into it. A click selects it; Delete or
 * a right click › Delete link takes it away. The line has a wide invisible stroke so that it is easy to hit.
 */
export function LinkEdge({ sourceX, sourceY, targetX, targetY, data, selected }: EdgeProps) {
  const ui = useGraphUi();
  const link = (data as LinkEdgeData).link;
  const [path] = getBezierPath({ sourceX, sourceY, sourcePosition: Position.Right, targetX, targetY, targetPosition: Position.Left });
  const kind = link.from.port;
  const drop = ui.dropLink !== null && sameLink(ui.dropLink, link);
  const arrow = selected || drop ? ARROW.selected : ARROW[kind];
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <g className="ym-link" data-kind={kind} data-selected={selected || undefined} data-drop={drop || undefined}>
          <BaseEdge path={path} markerEnd={`url(#${arrow})`} interactionWidth={18} />
        </g>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem variant="destructive" onSelect={() => ui.onDeleteLink(link)}>
          Delete link
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
}
