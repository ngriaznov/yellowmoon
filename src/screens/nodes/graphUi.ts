/**
 * What the graph's nodes and links need from the canvas that draws them, through one context: the settings, the plan's numbers, the node
 * the panel is on, the link being dragged and the inputs that light up for it, the thumbnails, and the callbacks. The nodes are xyflow
 * nodes whose `data` is only their id, so the pointer moving during a link drag never rebuilds the list of nodes.
 */
import { createContext, useContext } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { DevelopSettings, NodeId, NodeLink, NodeRef } from "../../api/types";
import type { CardState, InputState } from "../../state/graphOps";

/** An input port's key in the maps below. */
export const portKey = (ref: NodeRef): string => `${ref.node}:${ref.port}`;

/** A link in the one place a gesture may be: being drawn from an output, or picked up from an input. */
export interface LinkDraft {
  /** The output end that stays where it is. */
  from: NodeRef;
  /** The link whose input end was picked up, if one was. */
  lifted: NodeLink | null;
}

/** What the canvas offers every node and link it draws. */
export interface GraphUi {
  settings: DevelopSettings;
  /** Every corrector's number and whether it is live. */
  states: ReadonlyMap<number, CardState>;
  /** The node the graph and the panel are on. */
  selected: NodeId;
  /** A link is selected: the cards' rings go quiet so that the link is the one thing that is. */
  linkSelected: boolean;
  /** The card being renamed. */
  renaming: NodeId | null;
  /** The link a card being dragged is over, lit up as the place it will go. */
  dropLink: NodeLink | null;
  /** The link being drawn or carried, and which inputs take it. */
  draft: LinkDraft | null;
  lit: ReadonlyMap<string, InputState>;
  /** The input port under the pointer while a link is dragged. */
  hover: string | null;
  /** The picture of a node (`docs/23` §5.7): the one the core made for it, else the photo's own thumbnail until that arrives; none without either. */
  thumbnailOf: (id: NodeId) => string | undefined;
  onSelect: (id: NodeId) => void;
  /** An arrow key on a card: the node next to it in that direction, selected and focused. */
  onNavigate: (from: NodeId, direction: "left" | "right" | "up" | "down") => void;
  onStartRename: (id: NodeId) => void;
  /** The name typed, or null for Esc; `restoreFocus` is true when a key ended it (the focus goes back to the card). */
  onRename: (id: NodeId, name: string | null, restoreFocus: boolean) => void;
  /** Delete link, from a link's own menu. */
  onDeleteLink: (link: NodeLink) => void;
  /** A press on a port: an output starts a link, an input that has one picks it up. */
  onPortDown: (event: ReactPointerEvent, end: NodeRef, side: "in" | "out") => void;
}

export const GraphUiContext = createContext<GraphUi | null>(null);

/** The canvas's context; only the graph's own nodes and edges call it. */
export function useGraphUi(): GraphUi {
  const ui = useContext(GraphUiContext);
  if (!ui) throw new Error("A graph node is drawn outside the graph.");
  return ui;
}
