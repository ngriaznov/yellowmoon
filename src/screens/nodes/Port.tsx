import { Handle, Position } from "@xyflow/react";
import type { CSSProperties } from "react";
import type { NodeId, Port as PortName } from "../../api/types";
import { KEY_PORT_Y, MARKER, MARKER_ARROW_OUT, MARKER_SQUARE_OUT, PICTURE_PORT_Y, PORT_OUT } from "./graphLayout";
import { portKey, useGraphUi } from "./graphUi";

/**
 * One port of a node (`docs/23` §7.2): a triangle for an input, a square for an output; light grey for a picture, the mask accent for a key.
 * It is an xyflow `Handle` so that the links find their ends, but xyflow's own connecting is off: a press on it starts the graph's own
 * gesture (`LinkGesture`), which does what Resolve does (connect, replace, pick up an end, drop it on the canvas to remove it). The press
 * lands on a larger invisible target around the glyph, and nothing about a port is a Tab stop: the keyboard path is the More menu.
 */

/** Where a port's box is on its node, from the node's top left: the card's picture and key ports, and the markers' own. */
function boxOf(kind: "card" | "input" | "output", side: "in" | "out", port: PortName): CSSProperties {
  if (kind === "input") return { left: MARKER + 0, top: MARKER / 2 - 4, width: MARKER_SQUARE_OUT, height: 8 };
  if (kind === "output") return { left: -MARKER_ARROW_OUT, top: MARKER / 2 - 5, width: 8, height: 10 };
  const y = port === "key" ? KEY_PORT_Y : PICTURE_PORT_Y;
  // An output square is 8 × 8 and an input arrowhead 8 × 10; both centred on the port's line.
  return side === "out" ? { left: 146, top: y - 4, width: 8, height: 8 } : { left: -PORT_OUT, top: y - 5, width: 8, height: 10 };
}

export interface PortProps {
  node: NodeId;
  port: PortName;
  side: "in" | "out";
  /** The node's own kind: a marker's ports sit on its disc. */
  on: "card" | "input" | "output";
}

export function Port({ node, port, side, on }: PortProps) {
  const ui = useGraphUi();
  const key = portKey({ node, port });
  const linked = ui.settings.nodes.links.some((l) => (side === "in" ? l.to : l.from).node === node && (side === "in" ? l.to : l.from).port === port);
  const lit = side === "in" && ui.draft ? ui.lit.get(key) : undefined;
  const state = !ui.draft || side === "out" ? undefined : lit?.ok ? "lit" : "dim";
  return (
    <Handle
      type={side === "out" ? "source" : "target"}
      position={side === "out" ? Position.Right : Position.Left}
      id={port}
      isConnectable={false}
      isConnectableStart={false}
      isConnectableEnd={false}
      className="ym-port"
      style={{ ...boxOf(on, side, port), right: "auto", bottom: "auto", transform: "none" }}
      data-port={port}
      data-side={side}
      data-linked={linked || undefined}
      data-state={state}
      data-hover={(side === "in" && ui.hover === key) || undefined}
    >
      {side === "in" ? (
        <svg className="ym-port__glyph" aria-hidden="true" width="8" height="10" viewBox="0 0 8 10">
          <path d="M0 0 L8 5 L0 10 Z" fill="currentColor" />
        </svg>
      ) : (
        <span className="ym-port__glyph ym-port__square" aria-hidden="true" />
      )}
      <span
        className="ym-port__hit"
        data-port-node={String(node)}
        data-port-name={port}
        data-port-side={side}
        // A press on an output starts a link; a press on an input that has one picks its end up. Nothing else on a port does anything.
        onPointerDown={(event) => ui.onPortDown(event, { node, port }, side)}
      />
    </Handle>
  );
}
