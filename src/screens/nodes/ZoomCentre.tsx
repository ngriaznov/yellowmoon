import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import type { PhotoSummary, ResolvedSettings } from "../../api/types";
import { usePointerDrag } from "../../components";
import { useAppState } from "../../state/context";
import { patchOf } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import type { Pt } from "./maskGeometry";

/** What the page needs to draw the Zoom centre of the selected Blur node. */
export interface ZoomCentreTarget {
  /** The Blur node. */
  node: number;
  /** Its Zoom centre, a point of the photo frame (0 to 1 each way). */
  center: [number, number];
  /** Pick centre is on: a click on the photo sets the centre. */
  picking: boolean;
  /** A click on the photo with Pick centre on, in CSS pixels from the cell's corner. The screen ends the pick when the centre was set. */
  onPick: (point: { x: number; y: number }) => void;
}

export interface ZoomCentreProps extends ZoomCentreTarget {
  store: AppStore;
  photo: PhotoSummary;
  rs: ResolvedSettings;
  /** The cell's size in CSS pixels: a new size, like a new frame, asks the core where the centre is drawn. */
  cell: { width: number; height: number };
}

/** An arrow key moves the handle this many CSS pixels, ten times as far with Shift (`docs/23` §7.6, as the mask handles). */
const NUDGE_PX = 1;

/** The eight directions of the dashed rays (`design/blur/html/build.py` `centre_handle`). */
const RAYS: ReadonlyArray<Pt> = [
  [1, 0],
  [0.7071, 0.7071],
  [0, 1],
  [-0.7071, 0.7071],
  [-1, 0],
  [-0.7071, -0.7071],
  [0, -1],
  [0.7071, -0.7071],
];

/** Long enough to leave any cell; the layer clips what is outside. */
const RAY_PX = 4000;

const hold = (v: number): number => Math.round(Math.min(1, Math.max(0, v)) * 1e4) / 1e4;

/**
 * The Zoom centre on the photo (`docs/33` §2.2, BL-14): a handle with eight dashed rays, drawn by the page over the cell as the Radial mask's centre is,
 * at the place the core maps (`viewport_map_points`: the turn, the crop, the straightening, the lens's bend and the view's zoom and pan are the core's; the
 * page never converts). A drag writes `zoom.center` where the pointer is, a patch per move that folds into one undo step; the arrow keys move it 1 px, Shift
 * 10 px, each press one step. With Pick centre on, a click anywhere on the photo sets the centre (`onPick`, the store's `pickCentre`) and the handle stays
 * draggable. The layer is not there while Crop or Export shows, in the Before view, and while the Mask tab's own handles are the photo's.
 */
export function ZoomCentre({ store, photo, rs, node, center, picking, onPick, cell }: ZoomCentreProps) {
  const frame = useAppState((s) => s.stats);
  const zoom = useAppState((s) => s.zoom);
  const root = useRef<HTMLDivElement>(null);
  const [drawn, setDrawn] = useState<Pt | null>(null);
  // While the handle is dragged it is drawn where the pointer is; the rays catch up with the next answer of the core.
  const [held, setHeld] = useState<Pt | null>(null);
  const x = center[0];
  const y = center[1];

  useEffect(() => {
    let live = true;
    void store.mapPoints([[x, y]], "photo").then((mapped) => {
      if (live && mapped) setDrawn(mapped[0] ?? null);
    });
    return () => {
      live = false;
    };
  }, [store, x, y, cell.width, cell.height, frame, zoom, photo.id]);

  const latest = useRef({ node, photoId: rs.photo_id });
  latest.current = { node, photoId: rs.photo_id };

  /** The pointer's place in the cell, in CSS pixels from its corner. */
  const cellPoint = (event: { clientX: number; clientY: number }): Pt => {
    const box = root.current?.getBoundingClientRect();
    return box ? [event.clientX - box.left, event.clientY - box.top] : [event.clientX, event.clientY];
  };

  const write = (at: Pt, final: boolean): void => {
    const { node: n, photoId } = latest.current;
    void store.update(patchOf(n, { zoom: { center: [hold(at[0]), hold(at[1])] } }), final, photoId);
  };

  // The gesture: the moves not yet answered (the core maps one pointer position at a time and the latest counts) and the place of the last answer, which
  // the release sends as the final patch.
  const gesture = useRef<{ next: Pt | null; busy: boolean; last: Pt | null; done: Array<() => void> } | null>(null);

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
        g.last = place;
        write(place, false);
      }
    } finally {
      g.busy = false;
      for (const wake of g.done.splice(0)) wake();
    }
  };

  const onPointerDown = usePointerDrag({
    start: (event: ReactPointerEvent<Element>) => {
      if (!(event.target as HTMLElement).closest("[data-zoom-handle]")) return false;
      event.stopPropagation();
      event.preventDefault();
      gesture.current = { next: null, busy: false, last: null, done: [] };
      setHeld(cellPoint(event));
      return true;
    },
    move: (event: PointerEvent) => {
      const g = gesture.current;
      if (!g) return;
      const at = cellPoint(event);
      setHeld(at);
      g.next = at;
      void drive(g);
    },
    end: () => {
      const g = gesture.current;
      gesture.current = null;
      setHeld(null);
      if (!g) return;
      // The release waits for the move still being answered, then sends the last place as the gesture's one undo step. A press that never moved changes nothing.
      const finish = (): void => {
        if (g.last) write(g.last, true);
      };
      if (g.busy) g.done.push(finish);
      else finish();
    },
  });

  const onKeyDown = async (event: KeyboardEvent<HTMLButtonElement>, at: Pt): Promise<void> => {
    const unit = NUDGE_PX * (event.shiftKey ? 10 : 1);
    const dx = event.key === "ArrowRight" ? unit : event.key === "ArrowLeft" ? -unit : 0;
    const dy = event.key === "ArrowDown" ? unit : event.key === "ArrowUp" ? -unit : 0;
    if (dx === 0 && dy === 0) return;
    // The arrows belong to the handle: they switch no photo.
    event.preventDefault();
    event.stopPropagation();
    const mapped = await store.mapPoints([[at[0] + dx, at[1] + dy]], "cell");
    const place = mapped?.[0];
    if (place) write(place, true);
  };

  const at = held ?? drawn;
  return (
    <div
      ref={root}
      className="ym-mask-handles ym-zoom-centre"
      data-zoom-centre=""
      data-picking={picking || undefined}
      data-dragging={held ? "true" : undefined}
      onPointerDown={onPointerDown}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      {picking ? (
        <div
          className="ym-zoom-centre__catch"
          data-zoom-catch=""
          onClick={(event) => {
            const [px, py] = cellPoint(event);
            onPick({ x: px, y: py });
          }}
        />
      ) : null}
      {at ? (
        <>
          <svg className="ym-mask-handles__svg" width={cell.width} height={cell.height} viewBox={`0 0 ${cell.width} ${cell.height}`} aria-hidden="true" focusable="false">
            {RAYS.map(([dx, dy], i) => (
              <path key={i} className="ym-zoom-centre__ray" d={`M${at[0]} ${at[1]}L${at[0] + dx * RAY_PX} ${at[1] + dy * RAY_PX}`} />
            ))}
          </svg>
          <button
            type="button"
            className="ym-mask-handle"
            data-kind="move"
            data-zoom-handle=""
            aria-label="Zoom centre. The arrow keys move it."
            style={{ left: at[0], top: at[1] }}
            onKeyDown={(event) => void onKeyDown(event, at)}
          />
        </>
      ) : null}
    </div>
  );
}
