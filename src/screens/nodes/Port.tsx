import { Handle, Position } from "@xyflow/react";
import type { CSSProperties } from "react";
import type { NodeId, Port as PortName } from "../../api/types";
import { CHANNEL_TINT } from "../../state/ports";
import { GLYPH_W, KEY_PORT_Y, MARKER, MARKER_ARROW_OUT, MARKER_SQUARE_OUT, PICTURE_PORT_Y, PORT_OUT } from "./graphLayout";
import { portKey, useGraphUi } from "./graphUi";

/**
 * One port of a node (`docs/23` §7.2): a triangle for an input, a square for an output; light grey for a picture, the mask accent for a key.
 * It is an xyflow `Handle` so that the links find their ends, but xyflow's own connecting is off: a press on it starts the graph's own
 * gesture (`LinkGesture`), which does what Resolve does (connect, replace, pick up an end, drop it on the canvas to remove it). The press
 * lands on a larger invisible target around the glyph, and nothing about a port is a Tab stop: the keyboard path is the More menu.
 */

/** Where a port's box is on its node, from the node's top left: the card's picture and key ports, and the markers' own. */
function boxOf(kind: "card" | "glyph" | "input" | "output", side: "in" | "out", port: PortName, glyphY?: number, plus?: boolean): CSSProperties {
  if (kind === "glyph") {
    // A glyph card's ports are stacked 24 px apart (`glyphGeometry`); the "+" is a larger triangle that ends 1 px from the card.
    const at = glyphY ?? 0;
    if (plus) return { left: -16, top: at - 8, width: 15, height: 16 };
    return side === "out" ? { left: GLYPH_W - 4, top: at - 4, width: 8, height: 8 } : { left: -PORT_OUT, top: at - 5, width: 8, height: 10 };
  }
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
  /** The node's own kind: a marker's ports sit on its disc, a glyph card's are stacked at `y`. */
  on: "card" | "glyph" | "input" | "output";
  /** A glyph card's port: the distance from the card's top to the port's centre. */
  y?: number;
  /** The "+" of a mixer: an empty triangle that adds the next numbered input when a link is dropped on it. `port` is that next input. */
  plus?: boolean;
}

/** The "+" port: an empty triangle with a plus, a little larger than an arrowhead (`design/mixers`). */
function PlusGlyph() {
  return (
    <svg className="ym-port__glyph" aria-hidden="true" width="15" height="16" viewBox="0 0 15 16">
      <path d="M1 1 L14 8 L1 15 Z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      <path d="M3.4 8h4.2M5.5 5.9v4.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none" />
    </svg>
  );
}

export function Port({ node, port, side, on, y, plus = false }: PortProps) {
  const ui = useGraphUi();
  const key = portKey({ node, port });
  const linked = ui.settings.nodes.links.some((l) => (side === "in" ? l.to : l.from).node === node && (side === "in" ? l.to : l.from).port === port);
  const lit = side === "in" && ui.draft ? ui.lit.get(key) : undefined;
  const state = !ui.draft || side === "out" ? undefined : lit?.ok ? "lit" : "dim";
  return (
    <Handle
      type={side === "out" ? "source" : "target"}
      position={side === "out" ? Position.Right : Position.Left}
      id={plus ? "plus" : port}
      isConnectable={false}
      isConnectableStart={false}
      isConnectableEnd={false}
      className={`ym-port${plus ? " ym-port--plus" : ""}${port in CHANNEL_TINT ? " ym-port--ch" : ""}`}
      style={{ ...boxOf(on, side, port, y, plus), right: "auto", bottom: "auto", transform: "none", ...(port in CHANNEL_TINT ? { "--ch": CHANNEL_TINT[port] } : {}) } as CSSProperties}
      data-port={port}
      data-plus={plus || undefined}
      data-side={side}
      data-linked={linked || undefined}
      data-state={state}
      data-hover={(side === "in" && ui.hover === key) || undefined}
    >
      {plus ? (
        <PlusGlyph />
      ) : side === "in" ? (
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
