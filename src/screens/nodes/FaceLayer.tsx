import { useEffect, useMemo, useState } from "react";
import type { FaceBox, PhotoSummary } from "../../api/types";
import { useAppState } from "../../state/context";
import type { AppStore } from "../../state/store";
import { selectedFace } from "./faceModel";
import type { Pt } from "./maskGeometry";

export interface FaceLayerProps {
  store: AppStore;
  photo: PhotoSummary;
  /** The faces the core found, in the photo's frame (shares of its width and height, turned and cropped as the picture is). */
  faces: readonly FaceBox[];
  /** The node's Same for all faces: every face is drawn as selected, because the tools act on every face. */
  all: boolean;
  /** The cell's size in CSS pixels: a new size, like a new frame, asks the core where the faces are. */
  cell: { width: number; height: number };
}

/** A face as the page draws it: the oval in CSS pixels from the cell's corner. */
interface Oval {
  number: number;
  cx: number;
  cy: number;
  rx: number;
  ry: number;
}

/**
 * The faces on the photo while the Face tab of a Face Refinement node is shown (`docs/27` S-11c, `design/skin/03-skin-per-face.jpg`): a soft yellow oval round the
 * selected face and a dashed white one round each of the others, every one with its number, and a click on a face selects it. The boxes come from `face_state`;
 * the page does not convert them, the core maps their corners to the cell (`viewport_map_points`, which knows the turn, the crop, the lens and the zoom), so the
 * ovals follow the picture. The layer takes no press itself: the photo is panned and zoomed as usual, and only the ovals are clickable.
 */
export function FaceLayer({ store, photo, faces, all, cell }: FaceLayerProps) {
  const frame = useAppState((s) => s.stats);
  const zoom = useAppState((s) => s.zoom);
  const selected = selectedFace(useAppState((s) => s.faceSelected), faces.length);
  const [ovals, setOvals] = useState<Oval[]>([]);
  // Two opposite corners of each box: the oval is the ellipse in the box they make.
  const corners = useMemo(() => faces.flatMap((f): Array<[number, number]> => [[f.x, f.y], [f.x + f.width, f.y + f.height]]), [faces]);
  useEffect(() => {
    let live = true;
    if (corners.length === 0) {
      setOvals([]);
      return;
    }
    void store.mapPoints(corners, "photo").then((mapped) => {
      if (!live || !mapped) return;
      const next: Oval[] = [];
      faces.forEach((face, i) => {
        const a: Pt | null | undefined = mapped[2 * i];
        const b: Pt | null | undefined = mapped[2 * i + 1];
        if (!a || !b) return;
        next.push({ number: face.number, cx: (a[0] + b[0]) / 2, cy: (a[1] + b[1]) / 2, rx: Math.abs(b[0] - a[0]) / 2, ry: Math.abs(b[1] - a[1]) / 2 });
      });
      setOvals(next);
    });
    return () => {
      live = false;
    };
  }, [store, corners, faces, cell.width, cell.height, frame, zoom, photo.id]);

  return (
    <div className="ym-face-layer" data-face-layer="" aria-hidden="true">
      <svg className="ym-face-layer__svg" width={cell.width} height={cell.height} viewBox={`0 0 ${cell.width} ${cell.height}`} focusable="false">
        {ovals.map((o) => {
          const on = all || o.number === selected;
          return (
            <g key={o.number} className="ym-face-oval" data-selected={on || undefined} data-face-number={o.number}>
              <ellipse className="ym-face-oval__ring" cx={o.cx} cy={o.cy} rx={o.rx} ry={o.ry} />
              <ellipse
                className="ym-face-oval__hit"
                data-face-hit=""
                cx={o.cx}
                cy={o.cy}
                rx={o.rx}
                ry={o.ry}
                onPointerDown={(event) => event.stopPropagation()}
                onClick={() => store.selectFace(o.number)}
              />
              <g transform={`translate(${o.cx + o.rx * 0.72 - 9} ${o.cy - o.ry * 0.72 - 9})`}>
                <rect className="ym-face-oval__badge" width="18" height="18" rx="6" />
                <text className="ym-face-oval__num" x="9" y="13">
                  {o.number}
                </text>
              </g>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
