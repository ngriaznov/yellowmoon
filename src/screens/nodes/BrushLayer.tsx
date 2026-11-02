import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { PhotoSummary, ResolvedSettings } from "../../api/types";
import { usePointerDrag } from "../../components";
import { useAppState } from "../../state/context";
import { buildStroke } from "../../state/brushStrokes";
import type { BrushSettings } from "../../state/brushStrokes";
import type { AppStore } from "../../state/store";
import { frameUnits } from "./maskGeometry";
import type { FrameUnits, Pt } from "./maskGeometry";

/** What the page needs to draw on the selected Brush row: the corrector it is in and its index in the node's rows. */
export interface BrushTarget {
  node: number;
  index: number;
}

export interface BrushLayerProps extends BrushTarget {
  store: AppStore;
  photo: PhotoSummary;
  rs: ResolvedSettings;
  /** The cell's size in CSS pixels: a new size, like a new frame, asks the core for the brush's size on screen again. */
  cell: { width: number; height: number };
}

/** A pointer position is added to the stroke when it is at least this far (CSS pixels) from the last one added. */
const MIN_STEP_PX = 0.75;
/** The cursor is measured again (the core is asked) when the pointer has gone this far since it was measured last. */
const REMEASURE_PX = 48;
/** A stroke that was sent stays drawn on the page this long after the core has taken it, so that the picture it makes has time to arrive (CSS pixels, not data). */
const HOLD_MS = 600;

/** A stroke being drawn, or sent and not yet in the picture: the page draws it in CSS pixels. */
interface Drawn {
  id: number;
  /** The `points` of an SVG polyline. */
  points: string;
  /** The stroke's width: the brush's diameter on screen. */
  width: number;
  erase: boolean;
  hardness: number;
}

const coords = (p: Pt): string => `${Math.round(p[0] * 100) / 100},${Math.round(p[1] * 100) / 100}`;

/**
 * The Brush tool on the photo (`docs/23` §7.6 *On the photo*): a layer over the cell, like the handles, that is there while the tool is on for the
 * selected Brush row. A **cursor circle** at the brush's size follows the pointer, in the mask accent (an inner ring marks where the brush starts
 * to soften). **Pointer down** starts a stroke, which is drawn on the page as it goes, in CSS pixels, so the drag never waits on the core.
 * **Pointer up** sends it: one `viewport_map_points("cell")` for all its places, the ones on the paper dropped, the rest simplified in the photo
 * frame (`buildStroke`), and one `masks.edit` row patch with `strokes_append` (`store.addStroke`): one undo step, whose inverse is
 * `strokes_keep`. A single click is a stroke of one point. A gesture the system cancels, and a tool that ends mid-stroke, send nothing.
 *
 * The brush's size on screen is the core's too (the page never converts): the circle is measured at the pointer by mapping the photo-frame
 * points a brush radius to either side and above and below it, so a lens's bend or the zoom is in it. The layer takes the press (the photo is not
 * panned by a stroke), lets the wheel and the zoom keys through, and is not a Tab stop: it is a surface to draw on, not a control.
 */
export function BrushLayer({ store, photo, rs, node, index, cell }: BrushLayerProps) {
  const brush = useAppState((s) => s.brush);
  const units = useMemo(() => frameUnits(photo.width_px / Math.max(1, photo.height_px)), [photo.width_px, photo.height_px]);
  // A frame drawn by the core (the zoom or the pan moved, the photo's geometry changed) changes what the brush measures on screen.
  const frame = useAppState((s) => s.stats);
  const zoom = useAppState((s) => s.zoom);
  const root = useRef<HTMLDivElement>(null);

  const [at, setAt] = useState<Pt | null>(null);
  const [radius, setRadius] = useState<number>(() => Math.max(4, (brush.size * Math.max(cell.width, cell.height)) / 2));
  const [paper, setPaper] = useState(false);
  const [live, setLive] = useState<Drawn | null>(null);
  const [held, setHeld] = useState<Drawn[]>([]);

  // What a gesture and the measuring read when they run: the latest props, not the ones of the render that started them.
  const latest = useRef({ brush, units, node, index, photoId: rs.photo_id, radius });
  latest.current = { brush, units, node, index, photoId: rs.photo_id, radius };

  // The measuring of the cursor: one question at a time to the core, the latest pointer position the one that counts.
  const measuring = useRef<{ busy: boolean; next: Pt | null; last: Pt | null; stale: boolean }>({ busy: false, next: null, last: null, stale: true });
  const stroking = useRef(false);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const measure = async (): Promise<void> => {
    const m = measuring.current;
    if (m.busy) return;
    m.busy = true;
    try {
      while (m.next) {
        const place = m.next;
        m.next = null;
        const photoAt = (await store.mapPoints([place], "cell"))?.[0];
        if (!alive.current) return;
        if (!photoAt) {
          // The paper around the photo (or an empty corner): nothing is painted there, so the circle goes.
          setPaper(true);
          m.last = place;
          m.stale = false;
          continue;
        }
        setPaper(false);
        const { brush: b, units: u } = latest.current;
        // The brush's radius is a fraction of the long edge; the frame's x and y are fractions of the sides.
        const dx = b.size / 2 / u.w;
        const dy = b.size / 2 / u.h;
        const ring = await store.mapPoints(
          [photoAt, [photoAt[0] + dx, photoAt[1]], [photoAt[0] - dx, photoAt[1]], [photoAt[0], photoAt[1] + dy], [photoAt[0], photoAt[1] - dy]],
          "photo",
        );
        if (!alive.current) return;
        const centre = ring?.[0];
        if (ring && centre) {
          const spans = ring.slice(1).flatMap((p) => (p ? [Math.hypot(p[0] - centre[0], p[1] - centre[1])] : []));
          if (spans.length > 0) setRadius(Math.max(1, spans.reduce((sum, span) => sum + span, 0) / spans.length));
        }
        m.last = place;
        m.stale = false;
      }
    } finally {
      m.busy = false;
    }
  };
  const ask = (place: Pt): void => {
    const m = measuring.current;
    if (!m.stale && m.last && Math.hypot(place[0] - m.last[0], place[1] - m.last[1]) < REMEASURE_PX) return;
    m.next = place;
    void measure();
  };

  // The brush's size, the zoom, a frame of the core, the cell or the photo changed: what a radius is on screen is asked again at the next move
  // (and now, if the pointer is on the photo).
  useEffect(() => {
    measuring.current.stale = true;
    const place = measuring.current.last;
    if (place && !stroking.current) ask(place);
    // `ask` reads refs and the store only.
  }, [brush.size, frame, zoom, cell.width, cell.height, photo.id, units]);

  /** The pointer's place in the cell, in CSS pixels from its corner. */
  const place = (event: { clientX: number; clientY: number }, box: DOMRect | undefined = root.current?.getBoundingClientRect()): Pt =>
    box ? [event.clientX - box.left, event.clientY - box.top] : [event.clientX, event.clientY];

  // The gesture: the places the pointer went through (CSS pixels), the polyline that draws them, and what the brush and the row were when it began.
  const gesture = useRef<{
    id: number;
    places: Pt[];
    points: string;
    box: DOMRect | undefined;
    width: number;
    brush: BrushSettings;
    units: FrameUnits;
    node: number;
    index: number;
    photoId: ResolvedSettings["photo_id"];
  } | null>(null);
  const strokes = useRef(0);

  const send = async (g: NonNullable<typeof gesture.current>): Promise<void> => {
    // One question to the core for the whole stroke.
    const mapped = await store.mapPoints(g.places, "cell");
    const stroke = mapped ? buildStroke(mapped, g.brush, g.units) : null;
    if (stroke) await store.addStroke(g.node, g.index, stroke, g.photoId);
    // The page's drawing stays until the picture the core makes of the stroke has had time to arrive.
    window.setTimeout(() => {
      if (alive.current) setHeld((list) => list.filter((d) => d.id !== g.id));
    }, HOLD_MS);
  };

  const onPointerDown = usePointerDrag({
    start: (event: ReactPointerEvent<Element>) => {
      // The press is the stroke's, not the photo's (a drag does not pan the photo while the tool is on: Esc ends the tool).
      event.stopPropagation();
      event.preventDefault();
      const box = root.current?.getBoundingClientRect();
      const first = place(event, box);
      strokes.current += 1;
      const current = latest.current;
      const g = {
        id: strokes.current,
        places: [first],
        // A stroke of one point is a dot: the same point twice, which a round cap draws.
        points: `${coords(first)} ${coords(first)}`,
        box,
        width: current.radius * 2,
        brush: { ...current.brush },
        units: current.units,
        node: current.node,
        index: current.index,
        photoId: current.photoId,
      };
      gesture.current = g;
      stroking.current = true;
      setAt(first);
      setLive({ id: g.id, points: g.points, width: g.width, erase: g.brush.erase, hardness: g.brush.hardness });
      return true;
    },
    move: (event: PointerEvent) => {
      const g = gesture.current;
      if (!g) return;
      const here = place(event, g.box);
      setAt(here);
      const last = g.places[g.places.length - 1] as Pt;
      if (Math.hypot(here[0] - last[0], here[1] - last[1]) < MIN_STEP_PX) return;
      g.places.push(here);
      g.points = g.places.length === 2 ? `${coords(last)} ${coords(here)}` : `${g.points} ${coords(here)}`;
      setLive({ id: g.id, points: g.points, width: g.width, erase: g.brush.erase, hardness: g.brush.hardness });
    },
    end: (_event, cancelled) => {
      const g = gesture.current;
      gesture.current = null;
      stroking.current = false;
      setLive(null);
      if (!g || cancelled) return;
      setHeld((list) => [...list, { id: g.id, points: g.points, width: g.width, erase: g.brush.erase, hardness: g.brush.hardness }]);
      void send(g);
    },
  });

  const drawn = live ? [...held, live] : held;
  return (
    <div
      ref={root}
      className="ym-brush-layer"
      data-brush-layer=""
      data-stroking={live ? "true" : undefined}
      data-paper={paper || undefined}
      aria-hidden="true"
      onPointerDown={onPointerDown}
      onPointerMove={(event) => {
        if (stroking.current) return;
        const here = place(event);
        setAt(here);
        ask(here);
      }}
      onPointerLeave={() => {
        if (!stroking.current) setAt(null);
      }}
      // A double click paints two dabs; it does not zoom the photo (the zoom keys and the wheel work as they do).
      onDoubleClick={(event) => event.stopPropagation()}
    >
      <svg className="ym-brush-layer__svg" width={cell.width} height={cell.height} viewBox={`0 0 ${cell.width} ${cell.height}`} focusable="false">
        {drawn.map((d) => (
          <polyline
            key={d.id}
            className="ym-brush-stroke"
            points={d.points}
            strokeWidth={d.width}
            data-erase={d.erase || undefined}
            data-live={d === live || undefined}
            style={d.hardness < 1 ? { filter: `blur(${Math.round((1 - d.hardness) * d.width * 0.18 * 10) / 10}px)` } : undefined}
          />
        ))}
      </svg>
      {at && !paper ? (
        <div
          className="ym-brush-cursor"
          data-brush-cursor=""
          data-erase={brush.erase || undefined}
          data-radius={Math.round(radius * 100) / 100}
          style={{ left: at[0], top: at[1], width: radius * 2, height: radius * 2 }}
        >
          {brush.hardness < 1 ? <span className="ym-brush-cursor__core" style={{ width: `${brush.hardness * 100}%`, height: `${brush.hardness * 100}%` }} /> : null}
          <span className="ym-brush-cursor__dot" />
        </div>
      ) : null}
    </div>
  );
}
