import { useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { usePointerDrag } from "../../components";
import { fitInCell, handlePoint, HANDLES, moveRect, resizeRect } from "./cropMath";
import type { Handle, Rect } from "./cropMath";
import type { CropSession } from "./useCropSession";

const HANDLE_NAMES: Record<Handle, string> = {
  nw: "top left corner",
  n: "top edge",
  ne: "top right corner",
  e: "right edge",
  se: "bottom right corner",
  s: "bottom edge",
  sw: "bottom left corner",
  w: "left edge",
};

const HANDLE_CURSORS: Record<Handle, string> = {
  nw: "nwse-resize",
  se: "nwse-resize",
  ne: "nesw-resize",
  sw: "nesw-resize",
  n: "ns-resize",
  s: "ns-resize",
  e: "ew-resize",
  w: "ew-resize",
};

const pct = (n: number): string => `${Math.round(n * 10000) / 100}%`;

/** One arrow-key step of a handle or the frame, as a share of the photo; Shift takes ten. */
const KEY_STEP = 0.005;

/**
 * The crop frame over the photo (`docs/11-ui.md` §Crop): a rectangle with a 2 px ink border and
 * a 2 px paper outer stroke, the rest of the photo dimmed, eight 10 × 10 handles and thirds lines
 * while it is dragged. Drag a handle to resize (locked to the aspect when one is chosen), drag
 * inside to move. The arrow keys do the same from the keyboard: on a handle they move it, on the
 * frame they move the frame (Shift: ten times as far).
 */
export function CropOverlay({ session, cellAspect = 1.5 }: { session: CropSession; cellAspect?: number }) {
  const { draft, imageAspect, lockedRatio, setRect } = session;
  const picture = fitInCell(cellAspect, imageAspect);
  const image = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const gesture = useRef<{ handle: Handle | "move"; start: Rect; x: number; y: number } | null>(null);

  const pointInImage = (clientX: number, clientY: number): { x: number; y: number } => {
    const box = image.current?.getBoundingClientRect();
    if (!box || box.width <= 0 || box.height <= 0) return { x: 0, y: 0 };
    return { x: (clientX - box.left) / box.width, y: (clientY - box.top) / box.height };
  };

  const onPointerDown = usePointerDrag({
    start: (event: ReactPointerEvent<Element>) => {
      const target = (event.target as HTMLElement).closest<HTMLElement>("[data-handle]");
      if (!target) return false;
      const handle = target.dataset.handle as Handle | "move";
      const at = pointInImage(event.clientX, event.clientY);
      gesture.current = { handle, start: draft.rect, x: at.x, y: at.y };
      setDragging(true);
      event.preventDefault();
      return true;
    },
    move: (event: PointerEvent) => {
      const g = gesture.current;
      if (!g) return;
      const at = pointInImage(event.clientX, event.clientY);
      if (g.handle === "move") setRect(moveRect(g.start, at.x - g.x, at.y - g.y));
      else setRect(resizeRect(g.start, g.handle, at, lockedRatio, imageAspect));
    },
    end: () => {
      gesture.current = null;
      setDragging(false);
    },
  });

  const onKey = (event: KeyboardEvent<HTMLElement>, handle: Handle | "move"): void => {
    const unit = KEY_STEP * (event.shiftKey ? 10 : 1);
    const dx = event.key === "ArrowRight" ? unit : event.key === "ArrowLeft" ? -unit : 0;
    const dy = event.key === "ArrowDown" ? unit : event.key === "ArrowUp" ? -unit : 0;
    if (dx === 0 && dy === 0) return;
    event.preventDefault();
    event.stopPropagation();
    if (handle === "move") {
      setRect(moveRect(draft.rect, dx, dy));
      return;
    }
    const from = handlePoint(draft.rect, handle);
    setRect(resizeRect(draft.rect, handle, { x: from.x + dx, y: from.y + dy }, lockedRatio, imageAspect));
  };

  const { rect } = draft;
  // A handle on a frame edge that sits on the edge of the cell would be half cut off; draw it just inside instead. A handle on the
  // left or top edge starts at the edge (no margin); one on the right or bottom edge ends there, so it moves back by its own size, which
  // the stylesheet owns (`--crop-handle`: 10 px, 14 px in the glass interface).
  const EDGE = 0.0005;
  const atCell = {
    left: picture.x <= EDGE && rect.x <= EDGE,
    right: picture.x + picture.width >= 1 - EDGE && rect.x + rect.width >= 1 - EDGE,
    top: picture.y <= EDGE && rect.y <= EDGE,
    bottom: picture.y + picture.height >= 1 - EDGE && rect.y + rect.height >= 1 - EDGE,
  };
  const BACK = "calc(var(--crop-handle) * -1)";
  const inset = (handle: Handle): { marginLeft?: number | string; marginTop?: number | string } => ({
    marginLeft: handle.includes("w") && atCell.left ? 0 : handle.includes("e") && atCell.right ? BACK : undefined,
    marginTop: handle.includes("n") && atCell.top ? 0 : handle.includes("s") && atCell.bottom ? BACK : undefined,
  });
  /** The handle stands on the edge of the cell, where the cell clips: its focus ring is drawn inside it (see `.ym-crop__handle[data-at-cell-edge]`). */
  const onCellEdge = (handle: Handle): boolean =>
    (handle.includes("w") && atCell.left) || (handle.includes("e") && atCell.right) || (handle.includes("n") && atCell.top) || (handle.includes("s") && atCell.bottom);
  return (
    <div className="ym-crop" data-dragging={dragging || undefined} onPointerDown={onPointerDown}>
      <div
        ref={image}
        className="ym-crop__image"
        style={{ left: pct(picture.x), top: pct(picture.y), width: pct(picture.width), height: pct(picture.height) }}
      >
        <div
          className="ym-crop__frame"
          data-handle="move"
          role="group"
          tabIndex={0}
          aria-label="Crop frame. The arrow keys move it."
          style={{ left: pct(rect.x), top: pct(rect.y), width: pct(rect.width), height: pct(rect.height) }}
          onKeyDown={(event) => {
            if (event.target === event.currentTarget) onKey(event, "move");
          }}
        >
          <div className="ym-crop__thirds" aria-hidden="true">
            <i style={{ left: "33.333%", top: 0, bottom: 0, width: 1 }} />
            <i style={{ left: "66.666%", top: 0, bottom: 0, width: 1 }} />
            <i style={{ top: "33.333%", left: 0, right: 0, height: 1 }} />
            <i style={{ top: "66.666%", left: 0, right: 0, height: 1 }} />
          </div>
          {HANDLES.map((handle) => {
            const at = handlePoint({ x: 0, y: 0, width: 1, height: 1 }, handle);
            return (
              <button
                key={handle}
                type="button"
                className="ym-crop__handle"
                data-handle={handle}
                data-at-cell-edge={onCellEdge(handle) || undefined}
                aria-label={`Crop ${HANDLE_NAMES[handle]}. The arrow keys move it.`}
                style={{ left: pct(at.x), top: pct(at.y), cursor: HANDLE_CURSORS[handle], ...inset(handle) }}
                onKeyDown={(event) => onKey(event, handle)}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
