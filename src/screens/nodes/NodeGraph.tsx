import "@xyflow/react/dist/base.css";
import { ReactFlow, ReactFlowProvider, ViewportPortal, useReactFlow } from "@xyflow/react";
import type { Edge, Node, NodeChange } from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { DevelopSettings, NodeId, NodeLink, NodePos, NodeRef } from "../../api/types";
import { cardStates, GRAPH_OP_TEXT, inputStates, sameLink, sameRef } from "../../state/graphOps";
import type { InputState } from "../../state/graphOps";
import { INPUT_ID, MAIN_ID, nodeOf, numberedOf, OUTPUT_ID } from "../../state/settingsGraph";
import { isGlyphKind } from "../../state/ports";
import { CARD_H, glyphPosOf, isMarker, layoutGraph, linkUnder, neighbourIn, portPoint, posOf } from "./graphLayout";
import type { Direction, GraphLayout, XY } from "./graphLayout";
import { GraphUiContext, portKey } from "./graphUi";
import type { GraphUi, LinkDraft } from "./graphUi";
import { GestureLayer } from "./LinkGesture";
import type { GestureEnd } from "./LinkGesture";
import { LinkEdge, LinkMarkers } from "./LinkEdge";
import type { LinkEdgeData } from "./LinkEdge";
import { GlyphCard } from "./GlyphCard";
import { NodeCard } from "./NodeCard";
import { NodeMarker } from "./NodeMarker";
import { MAX_ZOOM, MIN_ZOOM, ZoomControl } from "./ZoomControl";

const nodeTypes = { card: NodeCard, glyph: GlyphCard, marker: NodeMarker };
const edgeTypes = { link: LinkEdge };
const NO_LIT: ReadonlyMap<string, InputState> = new Map();

/** The text under the link of a graph that has Main alone (`docs/23` §7.2). */
export const EMPTY_GRAPH_HINT = "Add a node to change only part of the photo.";

/** The id xyflow knows a node by. */
const flowId = (id: NodeId): string => String(id);
/** The id xyflow knows a link by. */
export const linkId = (link: NodeLink): string => `${link.from.node}.${link.from.port}>${link.to.node}.${link.to.port}`;

export interface NodeGraphProps {
  /** The photo's graph. */
  settings: DevelopSettings;
  /** Changes with the photo: the canvas is fitted again for each. */
  photoKey: string | number | null;
  /** The node the panel is on. */
  selected: NodeId;
  /** The selected link (thick and yellow), if one is. */
  selectedLink: NodeLink | null;
  /** The card being renamed. */
  renaming: NodeId | null;
  /** The picture of a node's card. Until N-05 it is the photo's own thumbnail for every node; N-05 gives each node its own. */
  thumbnailOf: (id: NodeId) => string | undefined;
  onSelect: (id: NodeId) => void;
  onSelectLink: (link: NodeLink | null) => void;
  onStartRename: (id: NodeId) => void;
  /** The name typed, or null when Esc cancels. */
  onRename: (id: NodeId, name: string | null) => void;
  /** A card dropped somewhere: its new `pos`. */
  onMove: (id: NodeId, pos: NodePos) => void;
  /** A card dropped with its centre over a picture link. */
  onReorder: (id: NodeId, link: NodeLink) => void;
  onConnect: (from: NodeRef, to: NodeRef) => void;
  /** A link's input end picked up and dropped on another input. */
  onReconnect: (link: NodeLink, to: NodeRef) => void;
  /** A link removed: its input end dropped on the empty canvas, or Delete link. */
  onDisconnect: (link: NodeLink) => void;
  /** Something the graph cannot do, said in the error bar. */
  onRefuse: (reason: string) => void;
}

/**
 * Moves the viewport as little as it takes to bring a node's box into view, with a margin, when the selection changes (a keyboard move, a
 * node added or deleted, an undo); a node already in view moves nothing. The visible area is the canvas as the page measures it.
 */
function useRevealed(layout: GraphLayout, selected: NodeId, root: React.RefObject<HTMLElement | null>): void {
  const flow = useReactFlow();
  // The node the view last showed: the first one is shown by the fit, and a repeated effect (Strict Mode) finds it unchanged.
  const shown = useRef(selected);
  useEffect(() => {
    if (shown.current === selected) return;
    shown.current = selected;
    const box = layout.boxes.get(selected);
    const area = root.current?.getBoundingClientRect();
    if (!box || !area || area.width === 0) return;
    const { x, y, zoom } = flow.getViewport();
    const margin = 28;
    const left = box.x * zoom + x;
    const top = box.y * zoom + y;
    const right = left + box.w * zoom;
    const bottom = top + box.h * zoom;
    const dx = left < margin ? margin - left : right > area.width - margin ? area.width - margin - right : 0;
    const dy = top < margin ? margin - top : bottom > area.height - margin ? area.height - margin - bottom : 0;
    if (dx !== 0 || dy !== 0) void flow.setViewport({ x: x + dx, y: y + dy, zoom }, { duration: 160 });
    // Only a change of the selection moves the view; the layout is read as it is then.
  }, [selected]);
}

function Canvas(props: NodeGraphProps) {
  const { settings, selected, selectedLink, renaming, thumbnailOf, onSelect, onSelectLink, onStartRename, onRename, onMove, onReorder, onConnect, onReconnect, onDisconnect, onRefuse } = props;
  const rootRef = useRef<HTMLDivElement>(null);
  const [moved, setMoved] = useState<ReadonlyMap<NodeId, XY>>(new Map());
  const [dropLink, setDropLink] = useState<NodeLink | null>(null);
  const [gesture, setGesture] = useState<{ draft: LinkDraft; start: XY } | null>(null);
  const [hover, setHover] = useState<string | null>(null);

  const layout = useMemo(() => layoutGraph(settings, moved), [settings, moved]);
  const states = useMemo(() => cardStates(settings), [settings]);
  const draft = gesture?.draft ?? null;
  const lit = useMemo(
    () => (draft ? new Map(inputStates(settings, draft.from, draft.lifted ?? undefined).map((s) => [portKey(s.to), s])) : NO_LIT),
    [settings, draft],
  );
  useRevealed(layout, selected, rootRef);
  // The links are drawn for the pointer (`docs/23` §7.8): the SVG that holds them is not an image for a screen reader to list (xyflow gives
  // the container no way to say so, and a keyboard user has the More menu).
  useEffect(() => {
    rootRef.current?.querySelector(".react-flow__edges")?.setAttribute("aria-hidden", "true");
  }, []);

  const nodes = useMemo<Node[]>(
    () =>
      [...layout.boxes].map(([id, box]) => ({
        id: flowId(id),
        type: isMarker(id) ? "marker" : isGlyphKind(nodeOf(settings, id)?.kind ?? "corrector") ? "glyph" : "card",
        position: { x: box.x, y: box.y },
        data: { id },
        width: box.w,
        height: box.h,
        // The size is ours (the card is drawn at exactly this size), so it is also what the node measures. Without `measured`, xyflow takes each
        // new node object as a node that was never measured and drops the position of its ports (`parseHandles` in @xyflow/system); the links
        // of that node are then not drawn until the node is measured again, and in the built app that second measurement can fail to come
        // (the node element is only observed again when the render sees it uninitialised), so the links stayed gone after a few drags.
        measured: { width: box.w, height: box.h },
        draggable: !isMarker(id),
        selectable: false,
        focusable: false,
        connectable: false,
        deletable: false,
      })),
    [layout, settings],
  );
  const edges = useMemo<Edge[]>(
    () =>
      settings.nodes.links
        .filter((link) => !(draft?.lifted && sameLink(link, draft.lifted)))
        .map((link) => {
          const on = selectedLink !== null && sameLink(link, selectedLink);
          return {
            id: linkId(link),
            type: "link",
            source: flowId(link.from.node),
            sourceHandle: link.from.port,
            target: flowId(link.to.node),
            targetHandle: link.to.port,
            data: { link } satisfies LinkEdgeData,
            selected: on,
            selectable: true,
            focusable: false,
            deletable: false,
            // Links are drawn for the pointer (`docs/23` §7.8): not a tab stop, and nothing for a screen reader to list; the keyboard path
            // is the More menu.
            ariaRole: "presentation" as const,
            domAttributes: { "aria-hidden": true },
            zIndex: on ? 10 : 0,
          };
        }),
    [settings, selectedLink, draft],
  );

  const focusCard = useCallback((id: NodeId): void => {
    rootRef.current?.querySelector<HTMLElement>(`[data-node-card="${String(id)}"]`)?.focus();
  }, []);
  const select = useCallback(
    (id: NodeId): void => {
      onSelect(id);
      focusCard(id);
    },
    [onSelect, focusCard],
  );
  const finishRename = useCallback(
    (id: NodeId, name: string | null, restoreFocus: boolean): void => {
      onRename(id, name);
      // The box is gone with the name: the card gets the focus back, once it has been drawn again.
      if (restoreFocus) requestAnimationFrame(() => focusCard(id));
    },
    [onRename, focusCard],
  );
  const navigate = useCallback(
    (from: NodeId, direction: Direction): void => {
      const next = neighbourIn(layout, from, direction);
      if (next !== undefined) select(next);
    },
    [layout, select],
  );

  const onPortDown = useCallback(
    (event: ReactPointerEvent, end: NodeRef, side: "in" | "out"): void => {
      if (event.button !== 0) return;
      const link = side === "in" ? settings.nodes.links.find((l) => sameRef(l.to, end)) : undefined;
      if (side === "in" && !link) return;
      event.preventDefault();
      event.stopPropagation();
      const next: LinkDraft = link ? { from: link.from, lifted: link } : { from: end, lifted: null };
      setGesture({ draft: next, start: { x: event.clientX, y: event.clientY } });
    },
    [settings],
  );
  const endGesture = useCallback(
    (end: GestureEnd): void => {
      const was = gesture?.draft;
      setGesture(null);
      setHover(null);
      if (!was || end.kind !== "drop") return;
      if (was.lifted) {
        if (end.to) onReconnect(was.lifted, end.to);
        else onDisconnect(was.lifted);
      } else if (end.to) {
        onConnect(was.from, end.to);
      }
    },
    [gesture, onConnect, onReconnect, onDisconnect],
  );

  const ui = useMemo<GraphUi>(
    () => ({
      settings,
      states,
      selected,
      linkSelected: selectedLink !== null,
      renaming,
      dropLink,
      draft,
      lit,
      hover,
      thumbnailOf,
      onSelect: select,
      onNavigate: navigate,
      onStartRename,
      onRename: finishRename,
      onDeleteLink: onDisconnect,
      onPortDown,
    }),
    [settings, states, selected, selectedLink, renaming, dropLink, draft, lit, hover, thumbnailOf, select, navigate, onStartRename, finishRename, onDisconnect, onPortDown],
  );

  const onNodesChange = useCallback((changes: NodeChange[]): void => {
    for (const change of changes) {
      if (change.type !== "position" || !change.position) continue;
      const { id, position } = change;
      setMoved((was) => new Map(was).set(id === INPUT_ID || id === OUTPUT_ID ? id : Number(id), position));
    }
  }, []);
  const overLink = useCallback(
    (id: NodeId, position: XY): NodeLink | null =>
      linkUnder(layout, settings.nodes.links, id, { x: position.x + (layout.boxes.get(id)?.w ?? 0) / 2, y: position.y + (layout.boxes.get(id)?.h ?? 0) / 2 }) ?? null,
    [layout, settings],
  );
  const onNodeDrag = useCallback(
    (_event: unknown, node: Node): void => {
      const id = (node.data as { id: NodeId }).id;
      const target = overLink(id, node.position);
      setDropLink((was) => (was && target && sameLink(was, target) ? was : target));
    },
    [overLink],
  );
  const onNodeDragStop = useCallback(
    (_event: unknown, node: Node): void => {
      const id = (node.data as { id: NodeId }).id;
      const target = overLink(id, node.position);
      setDropLink(null);
      setMoved(new Map());
      if (target) onReorder(id, target);
      else {
        // A glyph card's `pos` is its middle line, a corrector's its top (`glyphOrigin`).
        const box = layout.boxes.get(id);
        onMove(id, isGlyphKind(nodeOf(settings, id)?.kind ?? "corrector") && box ? glyphPosOf(node.position, box.h) : posOf(node.position));
      }
    },
    [overLink, onMove, onReorder, layout, settings],
  );

  const only = numberedOf(settings).length === 1;
  const main = layout.boxes.get(MAIN_ID);
  const output = layout.boxes.get(OUTPUT_ID);
  return (
    <GraphUiContext.Provider value={ui}>
      <div ref={rootRef} className="ym-flow" data-linking={draft ? draft.from.port : undefined}>
        <ReactFlow
          aria-label="Graph"
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodesChange={onNodesChange}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
          onNodeDragStart={(_event, node) => onSelect((node.data as { id: NodeId }).id)}
          onNodeClick={(_event, node) => onSelect((node.data as { id: NodeId }).id)}
          onEdgeClick={(_event, edge) => onSelectLink((edge.data as LinkEdgeData).link)}
          onEdgeContextMenu={(_event, edge) => onSelectLink((edge.data as LinkEdgeData).link)}
          onPaneClick={() => onSelectLink(null)}
          minZoom={MIN_ZOOM}
          maxZoom={MAX_ZOOM}
          fitView
          fitViewOptions={{ padding: "10px", minZoom: MIN_ZOOM, maxZoom: 1, duration: 0 }}
          nodesConnectable={false}
          nodesFocusable={false}
          edgesFocusable={false}
          selectNodesOnDrag={false}
          disableKeyboardA11y
          deleteKeyCode={null}
          selectionKeyCode={null}
          multiSelectionKeyCode={null}
          panActivationKeyCode={null}
          zoomActivationKeyCode="Meta"
          panOnScroll
          zoomOnScroll={false}
          zoomOnPinch
          zoomOnDoubleClick={false}
          panOnDrag
          nodeDragThreshold={4}
          autoPanOnNodeDrag={false}
          preventScrolling
        >
          <LinkMarkers />
          <ZoomControl />
          {only && main && output ? (
            <ViewportPortal>
              <div
                className="ym-graph-hint"
                style={{ left: (portPoint(layout, { node: MAIN_ID, port: "picture" }, "out").x + portPoint(layout, { node: OUTPUT_ID, port: "picture" }, "in").x) / 2, top: main.y + CARD_H + 22 }}
              >
                {EMPTY_GRAPH_HINT}
              </div>
            </ViewportPortal>
          ) : null}
          {gesture ? (
            <GestureLayer
              key={`${linkId({ from: gesture.draft.from, to: gesture.draft.from })}:${gesture.draft.lifted ? linkId(gesture.draft.lifted) : ""}`}
              draft={gesture.draft}
              start={gesture.start}
              layout={layout}
              root={rootRef.current}
              onHover={setHover}
              immovable={gesture.draft.lifted?.to.node === MAIN_ID && gesture.draft.lifted.to.port === "picture" ? GRAPH_OP_TEXT.mainInput : undefined}
              onRefuse={onRefuse}
              onEnd={endGesture}
            />
          ) : null}
        </ReactFlow>
      </div>
    </GraphUiContext.Provider>
  );
}

/** The graph of the node view (`docs/23` §7.2): the cards, the markers, the ports and the links on `@xyflow/react`'s canvas. */
export function NodeGraph(props: NodeGraphProps) {
  return (
    <ReactFlowProvider key={props.photoKey ?? "none"}>
      <Canvas {...props} />
    </ReactFlowProvider>
  );
}
