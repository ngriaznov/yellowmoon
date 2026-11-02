import { RotateCwIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import type { PhotoSummary, ResolvedSettings } from "../../api/types";
import { usePointerDrag } from "../../components";
import { useAppState } from "../../state/context";
import { patchOf } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { dragPatch, frameUnits, handlePlan } from "./maskGeometry";
import type { HandleDrawing, HandleId, Pt, ShapeRow } from "./maskGeometry";

/** What the page needs to draw the handles of the selected mask row. */
export interface MaskHandlesTarget {
  /** The corrector the row is in, and its index in the node's rows. */
  node: number;
  index: number;
  row: ShapeRow;
}

export interface MaskHandlesProps extends MaskHandlesTarget {
  store: AppStore;
  photo: PhotoSummary;
  rs: ResolvedSettings;
  /** The cell's size in CSS pixels: a new size, like a new frame, asks the core where the points are drawn. */
  cell: { width: number; height: number };
}

/** An arrow key moves a handle this many CSS pixels, ten times as far with Shift (`docs/23` §7.6). */
const NUDGE_PX = 1;

/** The kind of a handle, for its look. */
const KIND: Readonly<Record<HandleId, "move" | "rotate" | "radius" | "feather" | "line">> = {
  move: "move",
  line: "line",
  rotate: "rotate",
  "radius-x": "radius",
  "radius-y": "radius",
  "feather-start": "feather",
  "feather-end": "feather",
};

const points = (line: Pt[]): string => line.map(([x, y]) => `${Math.round(x * 100) / 100},${Math.round(y * 100) / 100}`).join(" ");

/**
 * The handles of a Linear or a Radial on the photo (`docs/23` §7.6 *On the photo*), drawn by the page over the cell as the crop frame is, in the
 * mask accent: for a Linear the solid centre line, the dashed start and end lines (the feather) and the rotate handle; for a Radial the outline,
 * the inner (feather) ring, the centre handle, a handle for each radius and the rotate handle. They are the selected row's, and the page draws
 * only what the core maps: `viewport_map_points` takes the shape's points in the photo frame to CSS pixels through the turn, the crop, the
 * straightening, the lens's bend and the view's zoom and pan (`maskHandles.ts` says which points, and what a drag of each writes).
 *
 * A handle is a focusable button; the arrow keys move it 1 px, Shift 10 px. A drag is a `masks.edit` row patch per frame that folds into one
 * and is one undo step; the pointer is mapped back to the photo by the core as well. The lines follow the next frame the core draws.
 */
export function MaskHandles({ store, photo, rs, node, index, row, cell }: MaskHandlesProps) {
  const units = useMemo(() => frameUnits(photo.width_px / Math.max(1, photo.height_px)), [photo.width_px, photo.height_px]);
  const plan = useMemo(() => handlePlan(row, units), [row, units]);
  // A frame drawn by the core (the zoom or the pan moved, the photo's geometry changed) moves what the points map to.
  const frame = useAppState((s) => s.stats);
  const zoom = useAppState((s) => s.zoom);
  const [drawing, setDrawing] = useState<HandleDrawing | null>(null);
  // While a handle is dragged it is drawn where the pointer is; the lines catch up with the next answer of the core.
  const [held, setHeld] = useState<{ id: HandleId; at: Pt } | null>(null);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    void store.mapPoints(plan.points, "photo").then((mapped) => {
      if (live && mapped) setDrawing(plan.draw(mapped));
    });
    return () => {
      live = false;
    };
  }, [store, plan, cell.width, cell.height, frame, zoom, photo.id]);

  // What a gesture reads when it moves: the row as it is now, the node and the row it began on.
  const latest = useRef({ row, node, index, photoId: rs.photo_id, units });
  latest.current = { row, node, index, photoId: rs.photo_id, units };

  const write = (patch: ReturnType<typeof dragPatch>, final: boolean): void => {
    if (!patch) return;
    const { node: n, index: i, photoId } = latest.current;
    void store.update(patchOf(n, { masks: { edit: { [String(i)]: patch } } }), final, photoId);
  };

  /** The pointer's place in the cell, in CSS pixels from its corner. */
  const cellPoint = (event: { clientX: number; clientY: number }): Pt => {
    const box = root.current?.getBoundingClientRect();
    return box ? [event.clientX - box.left, event.clientY - box.top] : [event.clientX, event.clientY];
  };

  // The gesture: one handle, the moves not yet answered (the core maps one pointer position at a time, and the latest is the one that counts) and the
  // patch of the last answer, which the release sends as the final one.
  const gesture = useRef<{ id: HandleId; next: Pt | null; busy: boolean; patch: ReturnType<typeof dragPatch>; done: Array<() => void> } | null>(null);

  const drive = async (g: NonNullable<typeof gesture.current>): Promise<void> => {
    if (g.busy) return;
    g.busy = true;
    try {
      while (g.next) {
        const at = g.next;
        g.next = null;
        const mapped = await store.mapPoints([at], "cell");
        const place = mapped?.[0];
        // On the paper around the photo the pointer has no place in it: the handle stays where it was.
        if (!place) continue;
        const patch = dragPatch(g.id, latest.current.row, place, latest.current.units);
        if (patch) {
          g.patch = patch;
          write(patch, false);
        }
      }
    } finally {
      g.busy = false;
      for (const wake of g.done.splice(0)) wake();
    }
  };

  const onPointerDown = usePointerDrag({
    start: (event: ReactPointerEvent<Element>) => {
      const target = (event.target as HTMLElement).closest<HTMLElement>("[data-handle]");
      if (!target) return false;
      event.stopPropagation();
      event.preventDefault();
      const id = target.dataset.handle as HandleId;
      gesture.current = { id, next: null, busy: false, patch: null, done: [] };
      setHeld({ id, at: cellPoint(event) });
      return true;
    },
    move: (event: PointerEvent) => {
      const g = gesture.current;
      if (!g) return;
      const at = cellPoint(event);
      setHeld({ id: g.id, at });
      g.next = at;
      void drive(g);
    },
    end: () => {
      const g = gesture.current;
      gesture.current = null;
      setHeld(null);
      if (!g) return;
      // The release (or a cancelled gesture) waits for the move still being answered, then sends the last patch as the gesture's one undo step. A
      // press that never moved changes nothing.
      const finish = (): void => {
        if (g.patch) write(g.patch, true);
      };
      if (g.busy) g.done.push(finish);
      else finish();
    },
  });

  /** An arrow key: the handle goes 1 px (Shift 10) the way the key says, as a drag to there would, in one step. */
  const onKeyDown = async (event: KeyboardEvent<HTMLButtonElement>, id: HandleId, at: Pt): Promise<void> => {
    const unit = NUDGE_PX * (event.shiftKey ? 10 : 1);
    const dx = event.key === "ArrowRight" ? unit : event.key === "ArrowLeft" ? -unit : 0;
    const dy = event.key === "ArrowDown" ? unit : event.key === "ArrowUp" ? -unit : 0;
    if (dx === 0 && dy === 0) return;
    // The arrows belong to the handle: they switch no photo and move no selection.
    event.preventDefault();
    event.stopPropagation();
    const mapped = await store.mapPoints([[at[0] + dx, at[1] + dy]], "cell");
    const place = mapped?.[0];
    if (place) write(dragPatch(id, latest.current.row, place, latest.current.units), true);
  };

  if (!drawing) return <div ref={root} className="ym-mask-handles" data-mask-handles="" />;
  const handles = drawing.handles.map((h) => (held && held.id === h.id ? { ...h, at: held.at } : h));
  return (
    <div ref={root} className="ym-mask-handles" data-mask-handles="" data-kind={row.kind} data-dragging={held ? "true" : undefined} onPointerDown={onPointerDown} onDoubleClick={(e) => e.stopPropagation()}>
      <svg className="ym-mask-handles__svg" width={cell.width} height={cell.height} viewBox={`0 0 ${cell.width} ${cell.height}`} aria-hidden="true" focusable="false">
        {drawing.lines.map((line) => (
          <g key={line.id} className="ym-mask-line" data-line={line.id} data-dashed={line.dashed || undefined}>
            <polyline className="ym-mask-line__under" points={points(line.points)} />
            <polyline className="ym-mask-line__ink" points={points(line.points)} />
            {line.id === "centre" ? <polyline className="ym-mask-line__hit" points={points(line.points)} data-handle="line" /> : null}
          </g>
        ))}
      </svg>
      {handles.map((h) => (
        <button
          key={h.id}
          type="button"
          className="ym-mask-handle"
          data-handle={h.id}
          data-kind={KIND[h.id]}
          aria-label={h.label}
          style={{ left: h.at[0], top: h.at[1] }}
          onKeyDown={(event) => void onKeyDown(event, h.id, h.at)}
        >
          {h.id === "rotate" ? <RotateCwIcon aria-hidden="true" /> : null}
        </button>
      ))}
    </div>
  );
}
