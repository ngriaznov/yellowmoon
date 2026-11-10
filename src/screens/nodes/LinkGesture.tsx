import { ViewportPortal, useReactFlow } from "@xyflow/react";
import { useEffect, useRef, useState } from "react";
import type { NodeRef, Port } from "../../api/types";
import { bezierOf, portPoint } from "./graphLayout";
import type { GraphLayout, XY } from "./graphLayout";
import { portKey } from "./graphUi";
import type { LinkDraft } from "./graphUi";

/** How far, in screen pixels, the pointer must go from where it was pressed before a press is a drag (and not a click on a port). */
export const DRAG_START_PX = 4;
/** How far outside an input port's target a release still counts as on it, in screen pixels. */
export const DROP_REACH_PX = 14;

/** The input port under or near a point (the nearest within `reach` pixels of its target), or null. */
export function inputPortAt(root: ParentNode, x: number, y: number, reach = DROP_REACH_PX): NodeRef | null {
  let best: { ref: NodeRef; d: number; c: number } | null = null;
  for (const el of root.querySelectorAll<HTMLElement>('[data-port-side="in"]')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    const d = Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom));
    // Targets of ports 24 px apart overlap a little: the nearer centre wins.
    const c = Math.hypot((r.left + r.right) / 2 - x, (r.top + r.bottom) / 2 - y);
    if (d > reach || (best && (d > best.d || (d === best.d && c >= best.c)))) continue;
    const raw = el.dataset.portNode ?? "";
    best = { ref: { node: raw === "input" || raw === "output" ? raw : Number(raw), port: (el.dataset.portName ?? "picture") as Port }, d, c };
  }
  return best?.ref ?? null;
}

/** How a gesture ends: on an input (or on none, `to` null), or abandoned (Esc, or a press that never became a drag). */
export type GestureEnd = { kind: "drop"; to: NodeRef | null } | { kind: "cancel" };

export interface GestureLayerProps {
  draft: LinkDraft;
  /** Where the press landed, in screen pixels. */
  start: XY;
  layout: GraphLayout;
  /** The canvas, whose input ports the pointer is tested against. */
  root: HTMLElement | null;
  /** The input port under the pointer changed (`portKey`, or null). */
  onHover: (key: string | null) => void;
  /** A link that is carried and cannot be moved at all (Input → Main): said at the first move, and the gesture is over. */
  immovable?: string;
  onRefuse: (reason: string) => void;
  onEnd: (end: GestureEnd) => void;
}

/**
 * The link being drawn or carried (`docs/23` §7.2 *Links by hand*): from the output port that stays, a line follows the pointer; the
 * inputs that can take it are lit by the nodes (`graphUi`), the others dim. It listens on the window for the pointer, so the line follows
 * it over any part of the page, and ends on release (a drop on an input, or off one) or on Esc. It draws nothing until the pointer has
 * moved a few pixels, so a click on a port is not a link.
 */
export function GestureLayer({ draft, start, layout, root, onHover, immovable, onRefuse, onEnd }: GestureLayerProps) {
  const flow = useReactFlow();
  const [pointer, setPointer] = useState<XY | null>(null);
  const moved = useRef(false);
  const hover = useRef<string | null>(null);
  const latest = useRef({ onHover, onRefuse, onEnd, immovable, root });
  latest.current = { onHover, onRefuse, onEnd, immovable, root };

  useEffect(() => {
    const target = (event: PointerEvent): NodeRef | null => (latest.current.root ? inputPortAt(latest.current.root, event.clientX, event.clientY) : null);
    const onMove = (event: PointerEvent): void => {
      if (!moved.current && Math.hypot(event.clientX - start.x, event.clientY - start.y) < DRAG_START_PX) return;
      if (!moved.current && latest.current.immovable) {
        // Input → Main is fixed: picking it up says so, and nothing comes away.
        latest.current.onRefuse(latest.current.immovable);
        latest.current.onEnd({ kind: "cancel" });
        return;
      }
      moved.current = true;
      setPointer(flow.screenToFlowPosition({ x: event.clientX, y: event.clientY }));
      const over = target(event);
      const key = over ? portKey(over) : null;
      if (key !== hover.current) {
        hover.current = key;
        latest.current.onHover(key);
      }
    };
    const onUp = (event: PointerEvent): void => {
      if (event.button !== 0 && event.type !== "pointercancel") return;
      latest.current.onEnd(moved.current && event.type === "pointerup" ? { kind: "drop", to: target(event) } : { kind: "cancel" });
    };
    const onKey = (event: KeyboardEvent): void => {
      if (event.key !== "Escape") return;
      // Esc ends the link and nothing else: no other shortcut of the screen sees it.
      event.preventDefault();
      event.stopPropagation();
      latest.current.onEnd({ kind: "cancel" });
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      window.removeEventListener("keydown", onKey, true);
      latest.current.onHover(null);
    };
  }, [flow, start.x, start.y]);

  if (!pointer) return null;
  if (!layout.boxes.has(draft.from.node)) return null;
  const [p0, p1, p2, p3] = bezierOf(portPoint(layout, draft.from, "out"), pointer);
  return (
    <ViewportPortal>
      <svg className="ym-draft" data-kind={draft.from.port} width="1" height="1" aria-hidden="true">
        <path d={`M ${p0.x} ${p0.y} C ${p1.x} ${p1.y} ${p2.x} ${p2.y} ${p3.x} ${p3.y}`} />
      </svg>
    </ViewportPortal>
  );
}
