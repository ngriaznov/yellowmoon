import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { PhotoSummary, ResolvedSettings, ViewportRect } from "../api/types";
import { PhotoCell, useLatest, usePointerDrag } from "../components";
import type { PhotoGestureHandlers, PhotoKeyHandlers, PhotoProbe } from "../components";
import { isTauri } from "../env";
import { useAppState } from "../state/context";
import { cameraLine, photoName, pixelText } from "../state/photo";
import { showsWhole } from "../state/zoom";
import type { AppStore } from "../state/store";
import { imageAspectOf, isFullRect } from "./crop/cropMath";
import { CropOverlay } from "./crop/CropOverlay";
import type { CropSession } from "./crop/useCropSession";
import { BrushLayer } from "./nodes/BrushLayer";
import type { BrushTarget } from "./nodes/BrushLayer";
import { MaskHandles } from "./nodes/MaskHandles";
import { ObjectLayer } from "./nodes/ObjectLayer";
import type { ObjectTarget } from "./nodes/ObjectLayer";
import { ObjectPill } from "./nodes/ObjectPill";
import type { MaskHandlesTarget } from "./nodes/MaskHandles";
import { PickLayer } from "./PickLayer";
import type { PickMode } from "./PickLayer";
import { PickDepthHint } from "./PickDepthHint";
import { PickWhiteHint } from "./PickWhiteHint";
import { previewUrl } from "./preview";
import { RenderingNote } from "./RenderingNote";
import { useSingleKeys } from "../state/useShortcuts";

export interface PhotoViewProps {
  store: AppStore;
  photo: PhotoSummary | null;
  rs: ResolvedSettings | null;
  /** Names of the looks by id, for the chip ("Neutral · 100%"). */
  lookName: (id: string) => string;
  before: boolean;
  /** Pick white or Pick colour is on: a crosshair layer covers the photo. Null while neither is. */
  picking: PickMode | null;
  /** A click in either mode, in CSS pixels from the corner of the cell: the core finds the spot of the photo. */
  onPicked: (point: { x: number; y: number }) => void;
  crop: CropSession | null;
  /** Reports the cell's rectangle, for the "Fit 34%" label. */
  onRect?: (rect: ViewportRect) => void;
  /** Leave the camera body out of the line under the photo, as the Export screen does. */
  compactCaption?: boolean;
  /** The handles of the selected mask row (`docs/23` §7.6 *On the photo*): drawn over the cell, except while Crop or a pick shows. */
  maskHandles?: MaskHandlesTarget | null;
  /** The Brush tool for the selected Brush row (`docs/23` §7.6): a layer over the cell with the cursor circle, where a drag paints. Hidden, like the handles, while Crop or a pick shows and in the Before view. */
  brush?: BrushTarget | null;
  /**
   * The selected Object row (`docs/23` §7.6): its points and box are drawn over the cell, and with `objectTool` a click finds an object and a drag with the
   * Box tool draws a box. Hidden, like the handles, while Crop or a pick shows and in the Before view.
   */
  object?: ObjectTarget | null;
  /** The Object tool is on (`store.objectTool`): the layer takes the press. */
  objectTool?: boolean;
  /**
   * "Place points" of the selected Object row is on (`docs/28` A-11): the crosshair layer of Pick white is over the photo, with the row's points drawn under
   * it. `onPlace` adds a point at a place in the cell (negative with ⇧Return or ⌥-click) and `onBack` takes the last one out.
   */
  placing?: { onPlace: (point: { x: number; y: number }, negative: boolean) => void; onBack: () => void } | null;
  /** The chip that stands in for the look's while the selected node has a selected mask row ("MASK · LINEAR", `docs/23` §7.1). */
  maskChip?: string;
  /** The I key of the zoomed photo read the pixel at the middle of the view: its text ("R 127 G 166 B 199"), or null when there was nothing to read. The screen says it. */
  onReadPixel?: (pixel: string | null) => void;
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
 * line, Before label, ⌥ pixel readout, "Rendering on the CPU" while the core renders the photo on the CPU) and, while they are on, the crosshair of Pick white or Pick colour (with
 * Pick white's hint pill) or the crop frame. Who draws the photo depends on the viewport backend (`viewport_info`): the native
 * layer behind a transparent cell (`native`), the shell's pictures of the cell shown as images
 * (`fallback`, from `viewport_frame`), or, with no backend (a browser build on the mock), the
 * mock's own picture. Double click toggles Fit and 100 %, the wheel pans and (with ⌘) zooms, a pinch zooms, and a drag pans: the
 * wheel, the pinch and the drag go through `store.queueViewportInput`, which sends one command per frame (`docs/11` §Photos input).
 *
 * The screen draws it again only when what it shows changes (`memo`): a `render_stats` tick or a zoom step elsewhere does not.
 */
export const PhotoView = memo(function PhotoView({ store, photo, rs, lookName, before, picking, onPicked, crop, onRect, compactCaption = false, maskHandles = null, brush = null, object = null, objectTool = false, placing = null, maskChip, onReadPixel }: PhotoViewProps) {
  const backend = useAppState((s) => s.viewportBackend);
  const sentFrame = useAppState((s) => s.viewportFrame);
  // Only the pick layer needs to know whether the picture is letterboxed; the zoom changes on every wheel step and is not read otherwise.
  const letterboxed = useAppState((s) => (picking !== null || placing !== null) && showsWhole(s.zoom, store.fitPercentExact()));
  // The picture is a Tab stop while it is larger than the cell and nothing else has it (`docs/28` A-10).
  const zoomed = useAppState((s) => crop === null && picking === null && placing === null && photo !== null && photo.status.kind === "ready" && !showsWhole(s.zoom, store.fitPercentExact()));
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
  // The browser build's fixture carries the mask view too (the core draws it in the app: `docs/23` §5.8).
  const maskShown = useAppState((s) => (drawsFixture ? s.maskShown : null));
  // Subject and People: the kinds this computer makes, and whether the core is making them now (`docs/23` §4.5).
  const maskKinds = useAppState((s) => s.maskKinds);
  const finding = useAppState((s) => s.viewportBusy.busy && s.viewportBusy.finding);
  const picture = useMemo(
    () => (photo && drawsFixture ? previewUrl(photo, rs, before, maskShown, { kinds: maskKinds, finding }) : undefined),
    [photo, rs, before, drawsFixture, maskShown, maskKinds, finding],
  );

  // The cell's size, for the handles: a new size is a new place for every point they draw.
  const [cellSize, setCellSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const handleRect = useCallback(
    (next: ViewportRect) => {
      store.setViewportRect(next);
      setCellSize((was) => (was.width === next.width && was.height === next.height ? was : { width: next.width, height: next.height }));
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
          setReadout(pixel ? pixelText(pixel) : null);
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
      // A handle of a mask is dragged, not the photo (`MaskHandles` takes the press), and a stroke of the Brush tool is drawn, not panned.
      if ((event.target as HTMLElement).closest?.("[data-mask-handles], [data-brush-layer]")) return false;
      if (store.showsWholePhoto() || crop || picking || placing) return false;
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
    enabled: () => !(crop || picking || placing),
    canPan: () => !store.showsWholePhoto(),
    pan: (dx, dy) => store.queueViewportInput({ kind: "pan", dx, dy }),
    zoom: (factor, x, y) => store.queueViewportInput({ kind: "zoom", factor, x, y }),
  };

  // The arrows, Home and I. The pan is the wheel's and the drag's: one input, queued, sent once per frame, clamped by the core. The pixel is the
  // ⌥ readout's call (`samplePixelAt`) at the middle of the cell.
  const onReadPixelRef = useLatest(onReadPixel);
  const singleKeys = useSingleKeys();
  const keys = useMemo<PhotoKeyHandlers | undefined>(
    () =>
      zoomed
        ? {
            pan: (dx, dy) => store.queueViewportInput({ kind: "pan", dx, dy }),
            centre: () => void store.recentreView(),
            // I is a single key: it is not given while the switch in Preferences is off (`docs/28` A-P11).
            readPixel: singleKeys
              ? () => {
                  void store.samplePixelAt(0.5, 0.5).then((pixel) => onReadPixelRef.current?.(pixel ? pixelText(pixel) : null));
                }
              : undefined,
          }
        : undefined,
    [zoomed, store, onReadPixelRef, singleKeys],
  );

  const note = photo ? notReadyNote(photo) : null;
  const frame = backend === "fallback" && sentFrame && !note ? { src: sentFrame.url, widthPx: sentFrame.width, heightPx: sentFrame.height } : undefined;
  const level = rs ? `${lookName(rs.values.output.look)} · ${Math.round(rs.values.output.look_intensity)}%` : undefined;
  const orientation = rs?.values.input.orientation ?? 0;
  const appliedCrop = !crop && rs?.values.input.crop && !isFullRect(rs.values.input.crop.rect) ? rs.values.input.crop.rect : null;
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
      lookChip={note || !level ? undefined : (maskChip ?? level)}
      cameraLine={photo && !note ? cameraLine(photo, { body: !compactCaption }) : undefined}
      before={before && !note}
      pixelReadout={readout}
      notice={
        <>
          {picking === "white" ? <PickWhiteHint /> : null}
          {picking === "depth" ? <PickDepthHint /> : null}
          <ObjectPill />
          <RenderingNote />
        </>
      }
      onRect={handleRect}
      onPixelProbe={onProbe}
      onDoubleClick={(event: ReactMouseEvent<HTMLElement>) => {
        if (crop || picking || placing) return;
        const at = cellPoint(event);
        void store.viewportInput({ kind: "toggle_zoom", ...at });
      }}
      gestures={gestures}
      keys={keys}
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
        <PickLayer mode={picking} displayAspect={displayAspect} letterboxed={letterboxed} onPick={onPicked} />
      ) : placing && object && photo && rs && !before && cellSize.width > 0 ? (
        <>
          <ObjectLayer store={store} photo={photo} rs={rs} cell={cellSize} tool={false} {...object} />
          <PickLayer mode="points" displayAspect={displayAspect} letterboxed={letterboxed} onPick={(point, negative) => placing.onPlace(point, negative === true)} onBack={placing.onBack} />
        </>
      ) : brush && photo && rs && !before && cellSize.width > 0 ? (
        <BrushLayer store={store} photo={photo} rs={rs} cell={cellSize} {...brush} />
      ) : object && photo && rs && !before && cellSize.width > 0 ? (
        <ObjectLayer store={store} photo={photo} rs={rs} cell={cellSize} tool={objectTool} {...object} />
      ) : maskHandles && photo && rs && !before && cellSize.width > 0 ? (
        <MaskHandles store={store} photo={photo} rs={rs} cell={cellSize} {...maskHandles} />
      ) : null}
    </PhotoCell>
  );
});

