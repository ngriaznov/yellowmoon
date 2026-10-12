import { useCallback, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, WheelEvent as ReactWheelEvent } from "react";
import type { PhotoSummary, ResolvedSettings, ViewportRect } from "../api/types";
import { PhotoCell, usePointerDrag } from "../components";
import type { PhotoProbe } from "../components";
import { isTauri } from "../env";
import { cameraLine, photoName } from "../state/photo";
import type { AppStore } from "../state/store";
import { imageAspectOf, isFullRect } from "./crop/cropMath";
import { CropOverlay } from "./crop/CropOverlay";
import type { CropSession } from "./crop/useCropSession";
import { PickWhiteLayer } from "./PickWhiteLayer";
import { previewUrl } from "./preview";

export interface PhotoViewProps {
  store: AppStore;
  photo: PhotoSummary | null;
  rs: ResolvedSettings | null;
  /** Names of the looks by id, for the chip ("Neutral · 100%"). */
  lookName: (id: string) => string;
  before: boolean;
  /** Pick white is on: a crosshair layer covers the photo. */
  picking: boolean;
  onPicked: (point: [number, number]) => void;
  crop: CropSession | null;
  /** Reports the cell's rectangle, for the "Fit 34%" label. */
  onRect?: (rect: ViewportRect) => void;
  /** Leave the camera body out of the line under the photo, as the Export screen does. */
  compactCaption?: boolean;
}

/** The note drawn over a photo that is not ready: still developing, or a file that can't be opened. */
function notReadyNote(photo: PhotoSummary): { title: string; text: string } | null {
  switch (photo.status.kind) {
    case "ready":
      return null;
    case "skipped":
      return { title: "This file can't be opened.", text: photo.status.reason };
    default:
      return { title: "Still developing.", text: "This photo will show here when it is ready." };
  }
}

/**
 * The photo cell of the Photos screen: the 3:2 viewport with its overlays (look chip, camera
 * line, Before label, ⌥ pixel readout) and, while they are on, the pick-white crosshair or the
 * crop frame. In the browser it draws the mock's picture; in the app the native layer draws
 * behind it. Double click toggles Fit and 100 %, the wheel zooms and pans, and a drag pans.
 */
export function PhotoView({ store, photo, rs, lookName, before, picking, onPicked, crop, onRect, compactCaption = false }: PhotoViewProps) {
  const native = isTauri();
  const [readout, setReadout] = useState<string | null>(null);
  const probing = useRef(false);

  const picture = useMemo(() => (photo && !native ? previewUrl(photo, rs, before) : undefined), [photo, rs, before, native]);

  const handleRect = useCallback(
    (next: ViewportRect) => {
      store.setViewportRect(next);
      onRect?.(next);
    },
    [store, onRect],
  );

  const onProbe = useCallback(
    async (probe: PhotoProbe | null) => {
      if (!probe) {
        setReadout(null);
        return;
      }
      if (probing.current) return;
      probing.current = true;
      try {
        const pixel = await store.samplePixelAt(probe.x, probe.y);
        if (pixel) setReadout(`R ${pixel.r} G ${pixel.g} B ${pixel.b}`);
      } finally {
        probing.current = false;
      }
    },
    [store],
  );

  const cellPoint = (event: { clientX: number; clientY: number; currentTarget: HTMLElement }): { x: number; y: number } => {
    const box = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - box.left, y: event.clientY - box.top };
  };

  const panning = useRef<{ x: number; y: number } | null>(null);
  const onPointerDown = usePointerDrag({
    start: (event) => {
      if (store.getState().zoom.kind === "fit" || crop || picking) return false;
      panning.current = { x: event.clientX, y: event.clientY };
      return true;
    },
    move: (event) => {
      const last = panning.current;
      if (!last) return;
      panning.current = { x: event.clientX, y: event.clientY };
      void store.viewportInput({ kind: "pan", dx: event.clientX - last.x, dy: event.clientY - last.y });
    },
    end: () => {
      panning.current = null;
    },
  });

  const note = photo ? notReadyNote(photo) : null;
  const level = rs ? `${lookName(rs.values.look)} · ${Math.round(rs.values.look_intensity)}%` : undefined;
  const orientation = rs?.values.orientation ?? 0;
  const appliedCrop = !crop && rs?.values.crop && !isFullRect(rs.values.crop.rect) ? rs.values.crop.rect : null;
  const displayAspect = photo
    ? appliedCrop
      ? (appliedCrop.width / appliedCrop.height) * imageAspectOf(photo.width_px, photo.height_px, orientation)
      : imageAspectOf(photo.width_px, photo.height_px, orientation)
    : 1.5;

  return (
    <PhotoCell
      imageSrc={note ? undefined : picture}
      imageLabel={photo ? `Photo ${photoName(photo.file_name)}${level ? `, ${level}` : ""}` : "Photo"}
      lookChip={note || !level ? undefined : level}
      cameraLine={photo && !note ? cameraLine(photo, { body: !compactCaption }) : undefined}
      before={before && !note}
      pixelReadout={readout}
      onRect={handleRect}
      onPixelProbe={onProbe}
      onDoubleClick={(event: ReactMouseEvent<HTMLElement>) => {
        if (crop || picking) return;
        const at = cellPoint(event);
        void store.viewportInput({ kind: "toggle_zoom", ...at });
      }}
      onWheel={(event: ReactWheelEvent<HTMLElement>) => {
        if (crop || picking) return;
        if (event.ctrlKey || event.metaKey) {
          const at = cellPoint(event);
          void store.viewportInput({ kind: "zoom", factor: Math.exp(-event.deltaY * 0.01), ...at });
        } else if (store.getState().zoom.kind !== "fit") {
          void store.viewportInput({ kind: "pan", dx: -event.deltaX, dy: -event.deltaY });
        }
      }}
      onPointerDown={onPointerDown}
    >
      {note ? (
        <div className="ym-photo-note" role="status">
          <span className="ym-photo-note__title">{note.title}</span>
          <span className="ym-photo-note__text">{note.text}</span>
        </div>
      ) : crop ? (
        <CropOverlay session={crop} />
      ) : picking ? (
        <PickWhiteLayer displayAspect={displayAspect} crop={appliedCrop} onPick={onPicked} />
      ) : null}
    </PhotoCell>
  );
}

