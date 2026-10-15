import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, WheelEvent as ReactWheelEvent } from "react";
import type { PhotoSummary, ResolvedSettings, ViewportRect } from "../api/types";
import { PhotoCell, usePointerDrag } from "../components";
import type { PhotoProbe } from "../components";
import { isTauri } from "../env";
import { useAppState } from "../state/context";
import { cameraLine, photoName } from "../state/photo";
import { showsWhole } from "../state/zoom";
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
  /** A click in Pick white mode, in CSS pixels from the corner of the cell: the core finds the spot of the photo. */
  onPicked: (point: { x: number; y: number }) => void;
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
 * crop frame. Who draws the photo depends on the viewport backend (`viewport_info`): the native
 * layer behind a transparent cell (`native`), the shell's pictures of the cell shown as images
 * (`fallback`, from `viewport_frame`), or, with no backend (a browser build on the mock), the
 * mock's own picture. Double click toggles Fit and 100 %, the wheel zooms and pans, and a drag pans.
 */
export function PhotoView({ store, photo, rs, lookName, before, picking, onPicked, crop, onRect, compactCaption = false }: PhotoViewProps) {
  const backend = useAppState((s) => s.viewportBackend);
  const sentFrame = useAppState((s) => s.viewportFrame);
  const zoom = useAppState((s) => s.zoom);
  const [readout, setReadout] = useState<string | null>(null);
  const probing = useRef(false);

  // With a backend the picture is the backend's; the mock's fixture is for a build that has none (never inside the app: a real
  // photo called DSC04204 is not the mock's DSC04204, and the answer of `viewport_info` may not have arrived yet).
  const drawsFixture = backend === null && !isTauri();

  // The native layer draws the photo behind the webview: the cell alone being transparent is not enough, every box under it (the
  // screen, the body, the root) must let it through too. The root says so while this view is on screen (`src/styles/screens.css`); the
  // window behind the page is paper, so the page looks the same wherever it draws nothing.
  const native = backend === "native";
  useEffect(() => {
    if (!native) return;
    const root = document.documentElement;
    root.dataset.nativeViewport = "";
    return () => {
      delete root.dataset.nativeViewport;
    };
  }, [native]);
  const picture = useMemo(() => (photo && drawsFixture ? previewUrl(photo, rs, before) : undefined), [photo, rs, before, drawsFixture]);

  const handleRect = useCallback(
    (next: ViewportRect) => {
      store.setViewportRect(next);
      onRect?.(next);
    },
    [store, onRect],
  );

  // The pointer's latest place while ⌥ is held (null once it is released or has left). The core answers one reading at a time and
  // takes a moment for each, so a reading is asked for the place the pointer is at when the last one comes back, and a reading that
  // comes back after ⌥ was released is not shown.
  const latest = useRef<PhotoProbe | null>(null);
  const onProbe = useCallback(
    async (probe: PhotoProbe | null) => {
      latest.current = probe;
      if (!probe) {
        setReadout(null);
        return;
      }
      if (probing.current) return;
      probing.current = true;
      try {
        let read: PhotoProbe | null = null;
        while (latest.current && latest.current !== read) {
          read = latest.current;
          const pixel = await store.samplePixelAt(read.x, read.y);
          if (!latest.current) break;
          // On the paper around a letterboxed photo there is nothing to read: the readout goes.
          setReadout(pixel ? `R ${pixel.r} G ${pixel.g} B ${pixel.b}` : null);
        }
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
      if (store.showsWholePhoto() || crop || picking) return false;
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
  const frame = backend === "fallback" && sentFrame && !note ? { src: sentFrame.url, widthPx: sentFrame.width, heightPx: sentFrame.height } : undefined;
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
      native={native}
      data-viewport-backend={backend ?? undefined}
      imageSrc={note ? undefined : picture}
      frame={frame}
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
        } else if (!store.showsWholePhoto()) {
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
        <PickWhiteLayer displayAspect={displayAspect} letterboxed={showsWhole(zoom, store.fitPercentExact())} onPick={onPicked} />
      ) : null}
    </PhotoCell>
  );
}

