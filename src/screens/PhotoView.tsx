import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { PhotoSummary, ResolvedSettings, ViewportRect } from "../api/types";
import { PhotoCell, usePointerDrag } from "../components";
import type { PhotoGestureHandlers, PhotoProbe } from "../components";
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
 * mock's own picture. Double click toggles Fit and 100 %, the wheel pans and (with ⌘) zooms, a pinch zooms, and a drag pans: the
 * wheel, the pinch and the drag go through `store.queueViewportInput`, which sends one command per frame (`docs/11` §Photos input).
 *
 * The screen draws it again only when what it shows changes (`memo`): a `render_stats` tick or a zoom step elsewhere does not.
 */
export const PhotoView = memo(function PhotoView({ store, photo, rs, lookName, before, picking, onPicked, crop, onRect, compactCaption = false }: PhotoViewProps) {
  const backend = useAppState((s) => s.viewportBackend);
  const sentFrame = useAppState((s) => s.viewportFrame);
  // Only Pick white needs to know whether the picture is letterboxed; the zoom changes on every wheel step and is not read otherwise.
  const letterboxed = useAppState((s) => picking && showsWhole(s.zoom, store.fitPercentExact()));
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
    // The stylesheet's rule is not enough on its own: a WebKit build on the Mac CI runner (2026-10-16, the 0.1.4 viewport job) kept
    // the root and the screen paper with the attribute set and the rule loaded, so the photo was hidden again. The same boxes are also
    // made transparent inline, which no rule can override, and put back as they were when the view goes.
    const screen = document.querySelector<HTMLElement>('.ym-screen[data-screen="photos"]');
    const boxes = [root, document.body, document.getElementById("root"), screen].filter((el): el is HTMLElement => el !== null);
    const before = boxes.map((el) => [el.style.getPropertyValue("background-color"), el.style.getPropertyPriority("background-color")] as const);
    // No transition either: a transition is above !important, and one that WebKit does not run on leaves the old paper in place.
    for (const el of boxes) {
      el.style.setProperty("transition", "none", "important");
      el.style.setProperty("background-color", "transparent", "important");
    }
    return () => {
      delete root.dataset.nativeViewport;
      boxes.forEach((el, i) => {
        const [value, priority] = before[i] ?? ["", ""];
        if (value) el.style.setProperty("background-color", value, priority);
        else el.style.removeProperty("background-color");
        el.style.removeProperty("transition");
      });
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
      store.queueViewportInput({ kind: "pan", dx: event.clientX - last.x, dy: event.clientY - last.y });
    },
    end: () => {
      panning.current = null;
      // The last moves go out now, not with the next frame: the drag is over.
      void store.flushViewportInput();
    },
  });

  // The wheel and the pinch (native listeners, `usePhotoGestures`): claimed over the whole cell, acted on only when nothing else has
  // the cell (Crop and Pick white do).
  const gestures: PhotoGestureHandlers = {
    enabled: () => !(crop || picking),
    canPan: () => !store.showsWholePhoto(),
    pan: (dx, dy) => store.queueViewportInput({ kind: "pan", dx, dy }),
    zoom: (factor, x, y) => store.queueViewportInput({ kind: "zoom", factor, x, y }),
  };

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
      gestures={gestures}
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
        <PickWhiteLayer displayAspect={displayAspect} letterboxed={letterboxed} onPick={onPicked} />
      ) : null}
    </PhotoCell>
  );
});

