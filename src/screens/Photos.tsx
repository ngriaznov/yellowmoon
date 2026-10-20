import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { Aurora } from "@/components/glass/aurora";
import { ChevronDownIcon } from "lucide-react";
import type { ColorSpace, PhotoSummary, RenderStats, ResolvedSettings, ViewportMode } from "../api/types";
import { Button, Filmstrip, filmstripCaption, TabBar, tabPanelProps, TitleBlock, useLatest, useTabShortcuts, useWindowDrop } from "../components";
import { exportModel } from "../export/model";
import { cellLabel, summarise } from "../export/run";
import type { FilmstripPhoto, TabId } from "../components";
import { perfCount } from "../perf";
import { useAppState, useCurrentPhoto, useCurrentSettings, useStore } from "../state/context";
import { lookLabeller, looksInDisplayOrder } from "../state/looks";
import { folderSubline, isReady, photoName } from "../state/photo";
import type { AppStore } from "../state/store";
import { useShortcuts } from "../state/useShortcuts";
import { fitZoomExact, fitZoomOf, showsWhole, ZOOMS, zoomKey, zoomStep } from "../state/zoom";
import { CropPanel } from "./crop/CropPanel";
import { useCropSession } from "./crop/useCropSession";
import { ExportPanel } from "./export/ExportPanel";
import { ExportCell, Header } from "./Header";
import { Menu } from "./Menu";
import type { PickMode } from "./PickLayer";
import { PhotoView } from "./PhotoView";
import { lookSlugOf } from "./preview";
import { ColorTab } from "./tools/ColorTab";
import { CurveTab } from "./tools/CurveTab";
import { DetailTab } from "./tools/DetailTab";
import { LightTab } from "./tools/LightTab";
import { LookTab } from "./tools/LookTab";
import { RawTab } from "./tools/RawTab";

const SPACE_LABELS: Record<ColorSpace, string> = { srgb: "sRGB", display_p3: "Display P3", rec2020: "Rec.2020" };

/** "Neural · 42 ms · Display P3": the demosaic path, the last render time and the output space. */
export function statusText(stats: RenderStats): string {
  return `${stats.path === "neural" ? "Neural" : "Classical"} · ${Math.round(stats.ms)} ms · ${SPACE_LABELS[stats.space]}`;
}

/** "Neural · Display P3": the export header's status, without a render time (nothing is being rendered there). */
export function exportStatusText(stats: RenderStats): string {
  return `${stats.path === "neural" ? "Neural" : "Classical"} · ${SPACE_LABELS[stats.space]}`;
}

/** The filmstrip only draws again when its photos or the selection change, not on every slider frame. */
const PhotoStrip = memo(Filmstrip);

/**
 * Where "4 of 12 · 3 picked" is shown. The shipped answer is the filmstrip's own caption. Variant (a) puts it in the header
 * beside the status text instead (windows 1440 px and wider; narrower ones keep the caption, there is no room); `?count=header` turns
 * it on in the dev page so that both can be looked at.
 */
const COUNT_IN_HEADER = import.meta.env.DEV && typeof window !== "undefined" && new URLSearchParams(window.location.search).get("count") === "header";

/** Variant (a): the count as a second piece of the header's status text, in the same type. */
function HeaderCount({ photos, currentId, mode }: { photos: readonly FilmstripPhoto[]; currentId: number | string | null; mode: "browse" | "export" }) {
  return <span className="ym-status ym-status--count">{filmstripCaption(photos, currentId, mode)}</span>;
}

/** The header's status text. It reads `render_stats` itself, so a frame drawn by the viewport redraws this span and not the screen. */
function StatusText({ forExport = false }: { forExport?: boolean }) {
  const text = useAppState((s) => (s.stats ? (forExport ? exportStatusText(s.stats) : statusText(s.stats)) : null));
  return text === null ? null : <span className="ym-status">{text}</span>;
}

/** What the zoom button says and which entry of its menu is the current one, from the zoom, the core's Fit figure and an estimate of it. */
function zoomView(zoom: ViewportMode, stats: RenderStats | null, estimate: number | null): { label: string; key: string } {
  // The Fit zoom, not rounded: a fixed zoom no larger than it is Fit to the core (50 % where Fit is 62 %). The core's own figure
  // (`render_stats.fit_percent`, which counts the lens crop, the straighten angle and the crop) once it has drawn a frame, else the
  // estimate from the photo and the cell, which is all the mock has.
  const fit = fitZoomOf(stats, estimate);
  const whole = showsWhole(zoom, fit);
  const fitPercent = fit === null ? null : Math.round(fit);
  const label = whole || zoom.kind === "fit" ? (fitPercent === null ? "Fit" : `Fit ${fitPercent}%`) : `${Math.round(zoom.value)}%`;
  return { label, key: zoomKey(whole ? { kind: "fit" } : zoom) };
}

interface ZoomMenuProps {
  store: AppStore;
  photo: PhotoSummary | null;
  rs: ResolvedSettings | null;
  /** The photo cell's size, for the estimate of the Fit zoom. */
  cell: { width: number; height: number; scale: number } | null;
}

/** The header's zoom button and menu. It reads the zoom and `render_stats` itself: a wheel zoom or a frame redraws this button only. */
const ZoomMenu = memo(function ZoomMenu({ store, photo, rs, cell }: ZoomMenuProps) {
  const estimate = cell && photo ? fitZoomExact(cell, photo, rs?.values ?? null) : null;
  const label = useAppState((s) => zoomView(s.zoom, s.stats, estimate).label);
  const key = useAppState((s) => zoomView(s.zoom, s.stats, estimate).key);
  return (
    <Menu
      label="Zoom"
      options={ZOOMS.map((z) => ({ value: z.key, label: z.label }))}
      value={key}
      onSelect={(next) => {
        const z = ZOOMS.find((item) => item.key === next);
        if (z) void store.setZoom(z.mode);
      }}
      trigger={({ ref, ...props }) => (
        <Button variant="header" className="ym-fit" ref={ref} title="Zoom (Z toggles Fit and 100%)" {...props}>
          {label}
          <ChevronDownIcon data-icon="inline-end" aria-hidden="true" />
        </Button>
      )}
    />
  );
});

/**
 * The header's "Add photos" button: the system's file dialog for ARW files, and what is chosen goes into the session after the
 * photos that are open (⌘O does the same; ⇧⌘O or a drop adds a folder). It is off while a dialog or an open is under way, so a second
 * click cannot ask twice.
 */
const AddPhotosButton = memo(function AddPhotosButton({ store }: { store: AppStore }) {
  const opening = useAppState((s) => s.opening);
  return (
    <Button variant="header" disabled={opening} onClick={() => void store.openDialog("files")} title="Add photos (⌘O)">
      Add photos
    </Button>
  );
});

/**
 * "Copy settings" and "Paste" (⇧⌘C, ⇧⌘V; the Edit menu has them and "Paste Settings to Picked" too), in the row at the foot of the tool
 * panel: the header has no room for them at its narrowest (`docs/11` §Copy and paste settings). Copy needs a photo that is developed;
 * Paste needs something copied as well, and its title names the photo the settings came from.
 */
const CopyPasteButtons = memo(function CopyPasteButtons({ store, canCopy }: { store: AppStore; canCopy: boolean }) {
  const source = useAppState((s) => s.copied?.source ?? null);
  return (
    <>
      <Button variant="header" disabled={!canCopy} onClick={() => void store.copySettings()} title="Copy the settings of this photo (⇧⌘C)">
        Copy settings
      </Button>
      <Button
        variant="header"
        disabled={!canCopy || source === null}
        onClick={() => void store.pasteSettings("this")}
        title={source === null ? "Copy the settings of a photo first (⇧⌘C)" : `Paste the settings of ${source} (⇧⌘V)`}
      >
        Paste
      </Button>
    </>
  );
});

/**
 * The Export cell of the header. It reads the run and the picks itself, so a tile finishing in an export redraws this button, not the
 * screen.
 */
function ExportCellContainer({ store }: { store: AppStore }) {
  const exporting = useAppState((s) => s.panel === "export");
  // While the panel is open it says where the export is; the cell keeps its name, as in the mockup.
  const label = useAppState((s) => {
    if (s.panel === "export") return "Export";
    if (s.exportRun) return cellLabel(s.exportRun);
    const nPicked = s.session?.picked.length ?? 0;
    return nPicked > 1 ? `Export ${nPicked}` : "Export";
  });
  const progress = useAppState((s) => (s.panel !== "export" && s.exportRun?.phase === "running" ? summarise(s.exportRun).percent : undefined));
  return <ExportCell label={label} current={exporting} progress={progress} onClick={() => (store.getState().panel === "export" ? store.closeExport() : store.openExport())} />;
}

/**
 * The Photos screen (`design/mockups/photos.html`, `docs/11-ui.md` §Photos): the header with
 * Before, Crop, Pick white, Add photos and the zoom menu, the photo, the filmstrip and the tool panel with its
 * six tabs. The Crop panel and the Export panel replace the tool panel.
 */
export function Photos() {
  perfCount("Photos");
  const store = useStore();
  const session = useAppState((s) => s.session);
  const looks = useAppState((s) => s.looks);
  const before = useAppState((s) => s.before);
  const thumbnails = useAppState((s) => s.thumbnails);
  const photo = useCurrentPhoto();
  const rs = useCurrentSettings();

  // What changes often is read where it is shown, not here: the zoom and `render_stats` by the zoom button and the status text, the
  // export run by the Export cell. A frame drawn by the viewport or a tile finished by an export does not redraw this screen.
  const panel = useAppState((s) => s.panel);
  // What only the Export panel needs is not subscribed to while it is closed: the settings of every photo change with each slider move.
  const exportScope = useAppState((s) => (s.panel === "export" ? s.exportScope : null));
  const exportOptions = useAppState((s) => (s.panel === "export" ? s.exportOptions : null));
  const settings = useAppState((s) => (s.panel === "export" ? s.settings : null));
  const preferences = useAppState((s) => (s.panel === "export" ? s.preferences : null));
  const overlay = useAppState((s) => s.overlay);
  const [tab, setTab] = useState<TabId>("look");
  // Pick white (the header) or Pick colour (Colour › Colour warp): one crosshair layer over the photo, so the two are exclusive.
  const [picking, setPicking] = useState<PickMode | null>(null);
  // Files dropped anywhere on the window are added to the session, as on Start (`docs/11` §Photos); a dialog takes the drop away.
  const { dragging } = useWindowDrop({ onDrop: (paths) => void store.openPaths(paths), enabled: overlay === null });
  const [cell, setCell] = useState<{ width: number; height: number; scale: number } | null>(null);
  const crop = useCropSession(store, photo, rs);

  const ordered = useMemo(() => looksInDisplayOrder(looks ?? []), [looks]);
  // The look chip names the look alone, so a title that two libraries share says which one it is ("Film · LogC3").
  const lookName = useMemo(() => {
    const label = lookLabeller(looks ?? []);
    const byId = new Map((looks ?? []).map((l) => [l.id, l]));
    return (id: string): string => {
      const look = byId.get(id);
      return look ? label(look) : lookSlugOf(id);
    };
  }, [looks]);

  const photos = session?.photos ?? [];
  const cropping = crop.active;

  // In the Export panel the filmstrip dims the photos that won't be exported.
  const included = useMemo(
    () =>
      panel === "export" && settings && exportScope && exportOptions
        ? exportModel({ session, settings, looks, preferences }, exportScope, exportOptions).included
        : null,
    [panel, session, settings, looks, preferences, exportScope, exportOptions],
  );

  const strip: FilmstripPhoto[] = useMemo(
    () =>
      photos.map((p) => ({
        id: p.id,
        name: photoName(p.file_name),
        src: thumbnails[p.id],
        status: p.status.kind === "ready" ? "ready" : p.status.kind === "skipped" ? "failed" : "developing",
        picked: p.picked,
        included: included?.has(p.id) ?? false,
      })),
    [photos, thumbnails, included],
  );

  /** Leaving a photo ends a crop in progress without keeping it. */
  const cropSession = useLatest(crop.session);
  const select = useCallback(
    (id: number | string) => {
      cropSession.current?.cancel();
      setPicking(null);
      void store.setCurrent(Number(id));
    },
    [store, cropSession],
  );
  const togglePickOf = useCallback((id: number | string) => void store.togglePick(Number(id)), [store]);
  /**
   * Remove from filmstrip (the thumbnail's menu, Delete): a crop or a Pick white in progress ends first, as when another photo is chosen,
   * so nothing is left of a mode on a photo that is gone.
   */
  const removePhoto = useCallback(
    (id: number | string) => {
      cropSession.current?.cancel();
      setPicking(null);
      void store.removePhotos([Number(id)]);
    },
    [store, cropSession],
  );
  const removePicked = useCallback(() => {
    cropSession.current?.cancel();
    setPicking(null);
    void store.removePickedPhotos();
  }, [store, cropSession]);

  /** ← and →. A crop in progress stays: leaving it is a click on another photo or on Cancel. */
  const stepPhoto = (direction: -1 | 1): void => {
    if (crop.session) return;
    setPicking(null);
    void store.step(direction);
  };

  const showTab = (next: TabId): void => {
    store.setPanel("tools");
    setTab(next);
  };
  useTabShortcuts(showTab, !cropping && overlay === null);

  // The zoom and the Fit zoom are read when the key is pressed, not at each render: this screen does not follow them.
  const zoomBy = (delta: number): void => void store.setZoom(zoomStep(store.getState().zoom, delta, store.fitPercentExact()));
  const toggleZoom = (): void => void store.toggleZoom();

  const startCrop = (): void => {
    if (!rs || !photo || !isReady(photo)) return;
    setPicking(null);
    store.setPanel("tools");
    crop.start();
  };
  const togglePicking = (): void => {
    if (!rs || !photo || !isReady(photo) || cropping) return;
    setPicking((mode) => (mode === "white" ? null : "white"));
  };
  // Pick colour, from the Colour warp card's toggle. It turns Pick white off, and stays on for several clicks until Esc or the toggle.
  const setPickingColor = useCallback((on: boolean): void => {
    setPicking((mode) => (on ? "color" : mode === "color" ? null : mode));
  }, []);
  const onPicked = useCallback(
    (point: { x: number; y: number }): void => {
      setTab("color");
      if (picking === "color") {
        void store.pickColor(point.x, point.y);
        return;
      }
      setPicking(null);
      void store.pickWhite(point.x, point.y);
    },
    [store, picking],
  );
  const onRect = useCallback((rect: { x: number; y: number; width: number; height: number; scale: number }): void => {
    setCell((was) => (was && was.width === rect.width && was.height === rect.height && was.scale === rect.scale ? was : { width: rect.width, height: rect.height, scale: rect.scale }));
    // The aurora behind the glass panes must leave the photo's rectangle clear: with the native viewport the photo is a layer behind
    // the transparent page, and anything the page paints there would cover it (`.ym-aurora` masks this rectangle out).
    const root = document.documentElement.style;
    root.setProperty("--cell-x", `${rect.x}px`);
    root.setProperty("--cell-y", `${rect.y}px`);
    root.setProperty("--cell-w", `${rect.width}px`);
    root.setProperty("--cell-h", `${rect.height}px`);
  }, []);
  useEffect(
    () => () => {
      for (const name of ["--cell-x", "--cell-y", "--cell-w", "--cell-h"]) document.documentElement.style.removeProperty(name);
    },
    [],
  );

  useShortcuts(
    (shortcut) => {
      switch (shortcut.action) {
        case "open-files":
          void store.openDialog("files");
          return true;
        case "open-folder":
          void store.openDialog("folder");
          return true;
        case "export":
          store.openExport();
          return true;
        case "close-session":
          void store.requestCloseSession();
          return true;
        case "undo":
          void store.undo();
          return true;
        case "redo":
          void store.redo();
          return true;
        case "previous-photo":
          stepPhoto(-1);
          return true;
        case "next-photo":
          stepPhoto(1);
          return true;
        case "toggle-pick":
          void store.togglePick();
          return true;
        case "remove-photo":
          // The Export panel works on the photos as they are; Back to photos, then remove.
          if (panel === "export") return false;
          void store.removeCurrentPhoto();
          return true;
        case "pick-all":
          void store.pickAll();
          return true;
        case "pick-look": {
          const look = ordered[(shortcut.look ?? 0) - 1];
          if (look) void store.chooseLook(look.id);
          return true;
        }
        case "before":
          void store.toggleBefore();
          return true;
        case "toggle-zoom":
          toggleZoom();
          return true;
        case "zoom-in":
          zoomBy(1);
          return true;
        case "zoom-out":
          zoomBy(-1);
          return true;
        case "crop":
          if (!cropping) startCrop();
          return true;
        case "pick-white":
          togglePicking();
          return true;
        case "copy-settings":
          void store.copySettings();
          return true;
        case "paste-settings":
          void store.pasteSettings("this");
          return true;
        case "paste-settings-picked":
          void store.pasteSettings("picked");
          return true;
        case "escape":
          if (picking) setPicking(null);
          else if (crop.session) crop.session.cancel();
          else if (panel === "export") store.closeExport();
          else return false;
          return true;
        case "confirm":
          crop.session?.done();
          return true;
        default:
          return false;
      }
    },
    { tab: panel === "tools" ? tab : undefined, mode: cropping ? "crop" : picking === "white" ? "pick-white" : picking === "color" ? "pick-color" : null },
    overlay === null,
  );

  let panelNode;
  if (crop.session) {
    panelNode = <CropPanel session={crop.session} />;
  } else if (panel === "export") {
    panelNode = <ExportPanel store={store} />;
  } else {
    panelNode = (
      <aside aria-label="Tools" className="ym-tools">
        <TabBar value={tab} onChange={setTab} />
        <div className="ym-tools__scroll" {...tabPanelProps(tab)}>
          <TitleBlock tab={tab} />
          {photo && rs && isReady(photo) ? (
            tab === "look" ? (
              <LookTab store={store} photo={photo} rs={rs} onOpenTab={setTab} />
            ) : tab === "light" ? (
              <LightTab store={store} photo={photo} rs={rs} />
            ) : tab === "color" ? (
              <ColorTab store={store} photo={photo} rs={rs} pickingColor={picking === "color"} onPickColor={setPickingColor} />
            ) : tab === "detail" ? (
              <DetailTab store={store} photo={photo} rs={rs} />
            ) : tab === "curve" ? (
              <CurveTab store={store} photo={photo} rs={rs} />
            ) : (
              <RawTab store={store} photo={photo} rs={rs} />
            )
          ) : photo && !isReady(photo) ? (
            <p className="ym-tools__notice">
              {photo.status.kind === "skipped" ? "There is nothing to adjust: this file can't be opened." : "This photo can be adjusted when it is developed."}
            </p>
          ) : null}
        </div>
        <div className="ym-tools__foot" role="group" aria-label="Copy and paste settings">
          <CopyPasteButtons store={store} canCopy={rs !== null && photo !== null && isReady(photo)} />
        </div>
      </aside>
    );
  }

  const exporting = panel === "export";

  return (
    <div className="ym-screen" data-screen="photos" data-panel={panel} data-count={COUNT_IN_HEADER ? "header" : undefined} data-dragging={dragging || undefined}>
      <Aurora className="ym-aurora" animate={false} />
      <Header titleAs="h1" title={session?.folder_name ?? "Yellowmoon"} subline={session ? folderSubline(session) : ""} export={<ExportCellContainer store={store} />}>
        {exporting ? (
          <>
            <Button variant="header" onClick={() => store.closeExport()}>
              ← Back to photos
            </Button>
            {COUNT_IN_HEADER ? (
              <span className="ym-status-group">
                <StatusText forExport />
                <HeaderCount photos={strip} currentId={photo?.id ?? null} mode="export" />
              </span>
            ) : (
              <StatusText forExport />
            )}
          </>
        ) : (
          <>
            <Button variant="header" pressed={before} onClick={() => void store.toggleBefore()} title="Before and after (\)">
              Before
            </Button>
            <Button variant="header" pressed={cropping} disabled={!rs} onClick={() => (cropping ? crop.session?.cancel() : startCrop())} title="Crop (C)">
              Crop
            </Button>
            <Button variant="header" pressed={picking === "white"} disabled={!rs || cropping} onClick={togglePicking} title="Pick white (W)">
              Pick white
            </Button>
            <AddPhotosButton store={store} />
            <ZoomMenu store={store} photo={photo} rs={rs} cell={cell} />
            {picking ? (
              <span className="ym-status" data-hint="true" role="status">
                {picking === "color" ? "Click a colour in the photo." : "Click something neutral: grey or white."}
              </span>
            ) : COUNT_IN_HEADER ? (
              <span className="ym-status-group">
                <StatusText />
                <HeaderCount photos={strip} currentId={photo?.id ?? null} mode="browse" />
              </span>
            ) : (
              <StatusText />
            )}
          </>
        )}
      </Header>

      <PhotoView
        store={store}
        photo={photo}
        rs={rs}
        lookName={lookName}
        before={before}
        compactCaption={exporting}
        picking={picking}
        onPicked={onPicked}
        crop={crop.session}
        onRect={onRect}
      />

      <PhotoStrip
        mode={exporting ? "export" : "browse"}
        photos={strip}
        currentId={photo?.id ?? null}
        tab={exporting ? undefined : tab}
        onSelect={select}
        onTogglePick={togglePickOf}
        onRemove={exporting ? undefined : removePhoto}
        onRemovePicked={exporting ? undefined : removePicked}
      />

      {panelNode}
    </div>
  );
}
