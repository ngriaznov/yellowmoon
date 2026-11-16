import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { PhotoSummary, ResolvedSettings } from "../../api/types";
import { usePointerDrag } from "../../components";
import { useAppState } from "../../state/context";
import { BOX_MIN_PX, boxPrompt } from "../../state/objectMask";
import type { ObjectRow } from "../../state/objectMask";
import type { AppStore } from "../../state/store";
import type { Pt } from "./maskGeometry";

/** What the page needs to draw the selected Object row: the corrector it is in, its index in the node's rows and the row. */
export interface ObjectTarget {
  node: number;
  index: number;
  row: ObjectRow;
}

export interface ObjectLayerProps extends ObjectTarget {
  store: AppStore;
  photo: PhotoSummary;
  rs: ResolvedSettings;
  /** The cell's size in CSS pixels: a new size, like a new frame, asks the core where the points are drawn. */
  cell: { width: number; height: number };
  /** The Object tool is on: a click finds an object, a drag with the Box tool draws a box. Off (after Esc) the points are drawn and the photo is the person's again. */
  tool: boolean;
}

const POINT_RADIUS = 9;

const poly = (line: Pt[]): string => line.map(([x, y]) => `${Math.round(x * 100) / 100},${Math.round(y * 100) / 100}`).join(" ");

/** What the layer draws: the row's points and the corners of its box, in CSS pixels, as the core mapped them (null for a point it could not place). */
interface Drawing {
  points: Array<{ at: Pt; positive: boolean }>;
  box: Pt[] | null;
}

/**
 * The Object row on the photo (`docs/23` §7.6 *On the photo*): the row's **points** as circles with a "+" (on the object) or a "−" (a part taken away) and
 * its **box**, in the mask accent, drawn by the page over the cell as the handles are, at the places the core maps (`viewport_map_points`: the turn, the crop, the
 * straightening, the lens's bend and the view's zoom and pan are the core's). While the tool is on the layer takes the press: a **click** is a `mask_find`
 * with the sign of the tool (Add point +, Remove point −), **⌥-click** is a negative point whatever the tool is, and a **drag** with the Box tool draws a box
 * (a rubber band on the page while it goes, one `mask_find` on release). The layer is a surface to click on, not a control: nothing in it is a Tab stop, and it
 * is not there while Crop, Export, Before or a pick shows (`PhotoView`). Esc ends the tool and leaves the points drawn. Every answer is the store's: a found mask
 * is one row patch and one undo step, and nothing found changes nothing (`Store.findObject`).
 */
export function ObjectLayer({ store, photo, rs, node, index, row, cell, tool }: ObjectLayerProps) {
  const mode = useAppState((s) => s.objectMode);
  // A frame drawn by the core (the zoom or the pan moved, the photo's geometry changed) moves what the points map to.
  const frame = useAppState((s) => s.stats);
  const zoom = useAppState((s) => s.zoom);
  const root = useRef<HTMLDivElement>(null);
  const [drawing, setDrawing] = useState<Drawing | null>(null);
  const [band, setBand] = useState<{ from: Pt; to: Pt } | null>(null);

  const photoPoints = useMemo(() => {
    const flat: Array<[number, number]> = row.points.map(([x, y]) => [x, y]);
    if (row.box) {
      const [x0, y0, x1, y1] = row.box;
      // The box's outline is four corners, so a turn or a bend shows in it.
      flat.push([x0, y0], [x1, y0], [x1, y1], [x0, y1]);
    }
    return flat;
  }, [row.points, row.box]);

  useEffect(() => {
    let live = true;
    if (photoPoints.length === 0) {
      setDrawing({ points: [], box: null });
      return;
    }
    void store.mapPoints(photoPoints, "photo").then((mapped) => {
      if (!live || !mapped) return;
      const n = row.points.length;
      const points = row.points.flatMap(([, , positive], i) => {
        const at = mapped[i];
        return at ? [{ at, positive }] : [];
      });
      const corners = row.box ? mapped.slice(n, n + 4) : [];
      setDrawing({ points, box: corners.length === 4 && corners.every((c): c is [number, number] => c !== null) ? corners : null });
    });
    return () => {
      live = false;
    };
  }, [store, photoPoints, row.points, row.box, cell.width, cell.height, frame, zoom, photo.id]);

  /** The pointer's place in the cell, in CSS pixels from its corner. */
  const place = (event: { clientX: number; clientY: number }, box: DOMRect | undefined = root.current?.getBoundingClientRect()): Pt =>
    box ? [event.clientX - box.left, event.clientY - box.top] : [event.clientX, event.clientY];

  // What a gesture reads when it ends: the row and the tool as they are now.
  const latest = useRef({ node, index, mode, photoId: rs.photo_id });
  latest.current = { node, index, mode, photoId: rs.photo_id };
  const gesture = useRef<{ from: Pt; last: Pt; box: DOMRect | undefined; alt: boolean; moved: boolean } | null>(null);

  const finish = async (from: Pt, to: Pt, alt: boolean, moved: boolean): Promise<void> => {
    const { node: n, index: i, mode: m, photoId } = latest.current;
    if (m === "box" && !alt && moved && Math.abs(to[0] - from[0]) >= BOX_MIN_PX && Math.abs(to[1] - from[1]) >= BOX_MIN_PX) {
      const mapped = await store.mapPoints([from, to], "cell");
      const a = mapped?.[0];
      const b = mapped?.[1];
      // A corner on the paper around the photo is not a place in it: nothing is asked.
      if (a && b) await store.findObject(n, i, boxPrompt(a, b), photoId);
      return;
    }
    if (moved && Math.hypot(to[0] - from[0], to[1] - from[1]) >= BOX_MIN_PX) return;
    // The same road as Return of "Place points" (`docs/28` A-11).
    await store.clickObject(n, i, m, alt, from, photoId);
  };

  const onPointerDown = usePointerDrag({
    start: (event: ReactPointerEvent<Element>) => {
      // The press is the tool's, not the photo's: a drag does not pan the photo while the tool is on (Esc ends the tool).
      event.stopPropagation();
      event.preventDefault();
      const box = root.current?.getBoundingClientRect();
      const from = place(event, box);
      gesture.current = { from, last: from, box, alt: event.altKey, moved: false };
      return true;
    },
    move: (event: PointerEvent) => {
      const g = gesture.current;
      if (!g) return;
      const to = place(event, g.box);
      g.last = to;
      if (Math.hypot(to[0] - g.from[0], to[1] - g.from[1]) >= 3) g.moved = true;
      if (latest.current.mode === "box" && !g.alt && g.moved) setBand({ from: g.from, to });
    },
    end: (event, cancelled) => {
      const g = gesture.current;
      gesture.current = null;
      setBand(null);
      if (!g || cancelled) return;
      void finish(g.from, event ? place(event, g.box) : g.last, g.alt, g.moved);
    },
  });

  return (
    <div
      ref={root}
      className="ym-object-layer"
      data-object-layer=""
      data-tool={tool ? mode : undefined}
      aria-hidden="true"
      onPointerDown={tool ? onPointerDown : undefined}
      // A double click finds twice; it does not zoom the photo.
      onDoubleClick={tool ? (event) => event.stopPropagation() : undefined}
    >
      <svg className="ym-object-layer__svg" width={cell.width} height={cell.height} viewBox={`0 0 ${cell.width} ${cell.height}`} focusable="false">
        {drawing?.box ? (
          <g className="ym-object-box" data-object-box="">
            <polygon className="ym-object-box__under" points={poly(drawing.box)} />
            <polygon className="ym-object-box__ink" points={poly(drawing.box)} />
          </g>
        ) : null}
        {band ? (
          <rect
            className="ym-object-box__band"
            data-object-band=""
            x={Math.min(band.from[0], band.to[0])}
            y={Math.min(band.from[1], band.to[1])}
            width={Math.abs(band.to[0] - band.from[0])}
            height={Math.abs(band.to[1] - band.from[1])}
          />
        ) : null}
        {drawing?.points.map(({ at, positive }, i) => (
          <g key={i} className="ym-object-point" data-object-point={positive ? "add" : "remove"} transform={`translate(${at[0]} ${at[1]})`}>
            <circle r={POINT_RADIUS} />
            <path d={positive ? "M-4 0H4M0 -4V4" : "M-4 0H4"} />
          </g>
        ))}
      </svg>
    </div>
  );
}
