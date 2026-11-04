import { useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { usePointerDrag } from "../../components";
import { NUDGE_BIG_PX, NUDGE_PX } from "./layout";

export interface DividerProps {
  /** The editor's height now, and the height under the header it is a share of. */
  editor: number;
  area: number;
  /** The least and the most the editor may be (`layout.ts`). */
  min: number;
  max: number;
  /** A drag: the editor's new height (the divider is under the photo, so a pointer moving up makes it taller). */
  onResize: (editorPx: number) => void;
  /** An arrow key: how much the editor grows (positive: the divider goes up). */
  onNudge: (deltaPx: number) => void;
  /** A double click: back to 45 %. */
  onReset: () => void;
  /** The node view is closed but kept built: the divider takes no room and no focus (U-02). */
  hidden?: boolean;
}

/**
 * The divider between the photo and the graph (`docs/23` §7.1): 10 px, a grab handle in the middle, `role="separator"` named "Resize the
 * graph", a window splitter. Drag it, or press ↑ and ↓ with it focused: 10 px a press, 50 with ⇧. A double click puts it back at 45 %.
 * Its value is the editor's share of the height under the header, in per cent; the bounds are 25 % and what leaves the photo 200 px.
 * Ours, with no counterpart in glasscn (its Separator is a static hairline).
 */
export function Divider({ editor, area, min, max, onResize, onNudge, onReset, hidden = false }: DividerProps) {
  const [dragging, setDragging] = useState(false);
  const start = useRef<{ y: number; editor: number } | null>(null);
  const onPointerDown = usePointerDrag({
    start: (event) => {
      start.current = { y: event.clientY, editor };
      setDragging(true);
    },
    move: (event) => {
      const from = start.current;
      if (from) onResize(from.editor + (from.y - event.clientY));
    },
    end: () => {
      start.current = null;
      setDragging(false);
    },
  });

  const onKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const step = event.shiftKey ? NUDGE_BIG_PX : NUDGE_PX;
    if (event.key === "ArrowUp") onNudge(step);
    else if (event.key === "ArrowDown") onNudge(-step);
    else return;
    event.preventDefault();
  };

  const percent = (px: number): number => Math.round((px / area) * 100);
  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      aria-label="Resize the graph"
      aria-valuenow={percent(editor)}
      aria-valuemin={percent(min)}
      aria-valuemax={percent(max)}
      tabIndex={0}
      hidden={hidden || undefined}
      className="ym-divider"
      data-dragging={dragging || undefined}
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
      onDoubleClick={onReset}
    >
      <span className="ym-divider__grip" aria-hidden="true" />
    </div>
  );
}
