import { memo, useCallback, useMemo, useState } from "react";
import type { ColorSpace, RenderStats } from "../api/types";
import { Button, Filmstrip, TabBar, tabPanelProps, TitleBlock, useLatest, useTabShortcuts } from "../components";
import { exportModel } from "../export/model";
import { cellLabel, summarise } from "../export/run";
import type { FilmstripPhoto, TabId } from "../components";
import { useAppState, useCurrentPhoto, useCurrentSettings, useStore } from "../state/context";
import { folderSubline, isReady, photoName } from "../state/photo";
import { useShortcuts } from "../state/useShortcuts";
import { fitZoomExact, showsWhole, ZOOMS, zoomKey, zoomStep } from "../state/zoom";
import { CropPanel } from "./crop/CropPanel";
import { useCropSession } from "./crop/useCropSession";
import { ExportPanel } from "./export/ExportPanel";
import { ExportCell, Header } from "./Header";
import { Menu } from "./Menu";
import { PhotoView } from "./PhotoView";
import { lookSlugOf } from "./preview";
import { ColorTab } from "./tools/ColorTab";
import { CurveTab } from "./tools/CurveTab";
import { DetailTab } from "./tools/DetailTab";
import { LightTab } from "./tools/LightTab";
import { LookTab, looksInDisplayOrder } from "./tools/LookTab";
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
 * The Photos screen (`design/mockups/photos.html`, `docs/11-ui.md` §Photos): the header with
 * Before, Crop, Pick white and the zoom menu, the photo, the filmstrip and the tool panel with its
 * six tabs. The Crop panel and the Export panel replace the tool panel.
 */
export function Photos() {
  const store = useStore();
  const session = useAppState((s) => s.session);
  const looks = useAppState((s) => s.looks);
  const stats = useAppState((s) => s.stats);
  const before = useAppState((s) => s.before);
  const zoom = useAppState((s) => s.zoom);
  const thumbnails = useAppState((s) => s.thumbnails);
  const photo = useCurrentPhoto();
  const rs = useCurrentSettings();

  const panel = useAppState((s) => s.panel);
  const exportRun = useAppState((s) => s.exportRun);
  const exportScope = useAppState((s) => s.exportScope);
  const exportOptions = useAppState((s) => s.exportOptions);
  const settings = useAppState((s) => s.settings);
  const preferences = useAppState((s) => s.preferences);
  const overlay = useAppState((s) => s.overlay);
  const [tab, setTab] = useState<TabId>("look");
  const [picking, setPicking] = useState(false);
  const [cell, setCell] = useState<{ width: number; height: number; scale: number } | null>(null);
  const crop = useCropSession(store, photo, rs);

  const ordered = useMemo(() => looksInDisplayOrder(looks ?? []), [looks]);
  const lookName = useCallback((id: string) => looks?.find((l) => l.id === id)?.name ?? lookSlugOf(id), [looks]);

  const photos = session?.photos ?? [];
  const nPicked = session?.picked.length ?? 0;
  const cropping = crop.active;

  // In the Export panel the filmstrip dims the photos that won't be exported.
  const included = useMemo(
    () => (panel === "export" ? exportModel({ session, settings, looks, preferences }, exportScope, exportOptions).included : null),
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
      setPicking(false);
      void store.setCurrent(Number(id));
    },
    [store, cropSession],
  );
  const togglePickOf = useCallback((id: number | string) => void store.togglePick(Number(id)), [store]);

  /** ← and →. A crop in progress stays: leaving it is a click on another photo or on Cancel. */
  const stepPhoto = (direction: -1 | 1): void => {
    if (crop.session) return;
    setPicking(false);
    void store.step(direction);
  };

  const showTab = (next: TabId): void => {
    store.setPanel("tools");
    setTab(next);
  };
  useTabShortcuts(showTab, !cropping);

  // The Fit zoom, not rounded: a fixed zoom no larger than it is Fit to the core (50 % where Fit is 62 %).
  const fitExact = cell && photo ? fitZoomExact(cell, photo, rs?.values ?? null) : null;
  const zoomBy = (delta: number): void => void store.setZoom(zoomStep(zoom, delta, fitExact));
  const toggleZoom = (): void => void store.toggleZoom();

  const startCrop = (): void => {
    if (!rs || !photo || !isReady(photo)) return;
    setPicking(false);
    store.setPanel("tools");
    crop.start();
  };
  const togglePicking = (): void => {
    if (!rs || !photo || !isReady(photo) || cropping) return;
    setPicking((on) => !on);
  };
  const onPicked = (point: { x: number; y: number }): void => {
    setPicking(false);
    setTab("color");
    void store.pickWhite(point.x, point.y);
  };

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
        case "escape":
          if (picking) setPicking(false);
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
    { tab: panel === "tools" ? tab : undefined, mode: cropping ? "crop" : picking ? "pick-white" : null },
    overlay === null,
  );

  const whole = showsWhole(zoom, fitExact);
  const fitPercent = fitExact === null ? null : Math.round(fitExact);
  const zoomLabel = whole || zoom.kind === "fit" ? (fitPercent === null ? "Fit" : `Fit ${fitPercent}%`) : `${Math.round(zoom.value)}%`;

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
              <ColorTab store={store} photo={photo} rs={rs} />
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
      </aside>
    );
  }

  const exporting = panel === "export";
  const exportCell = (
    <ExportCell
      // While the panel is open it says where the export is; the cell keeps its name, as in the mockup.
      label={exporting ? "Export" : exportRun ? cellLabel(exportRun) : nPicked > 1 ? `Export ${nPicked}` : "Export"}
      current={exporting}
      progress={!exporting && exportRun?.phase === "running" ? summarise(exportRun).percent : undefined}
      onClick={() => (exporting ? store.closeExport() : store.openExport())}
    />
  );

  return (
    <div className="ym-screen" data-screen="photos" data-panel={panel}>
      <Header titleAs="h1" title={session?.folder_name ?? "Yellowmoon"} subline={session ? folderSubline(session) : ""} export={exportCell}>
        {exporting ? (
          <>
            <Button variant="header" onClick={() => store.closeExport()}>
              ← Back to photos
            </Button>
            {stats ? <span className="ym-status">{exportStatusText(stats)}</span> : null}
          </>
        ) : (
          <>
            <Button variant="header" pressed={before} onClick={() => void store.toggleBefore()} title="Before and after (\)">
              Before
            </Button>
            <Button variant="header" pressed={cropping} disabled={!rs} onClick={() => (cropping ? crop.session?.cancel() : startCrop())} title="Crop (C)">
              Crop
            </Button>
            <Button variant="header" pressed={picking} disabled={!rs || cropping} onClick={togglePicking} title="Pick white (W)">
              Pick white
            </Button>
            <Menu
              label="Zoom"
              options={ZOOMS.map((z) => ({ value: z.key, label: z.label }))}
              value={zoomKey(whole ? { kind: "fit" } : zoom)}
              onSelect={(key) => {
                const z = ZOOMS.find((item) => item.key === key);
                if (z) void store.setZoom(z.mode);
              }}
              trigger={({ ref, ...props }) => (
                <Button variant="header" className="ym-fit" ref={ref} title="Zoom (Z toggles Fit and 100%)" {...props}>
                  {zoomLabel}
                  <span className="ym-caret" aria-hidden="true" />
                </Button>
              )}
            />
            {picking ? (
              <span className="ym-status" data-hint="true" role="status">
                Click something neutral: grey or white.
              </span>
            ) : stats ? (
              <span className="ym-status">{statusText(stats)}</span>
            ) : null}
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
        onRect={(rect) => setCell({ width: rect.width, height: rect.height, scale: rect.scale })}
      />

      <PhotoStrip mode={exporting ? "export" : "browse"} photos={strip} currentId={photo?.id ?? null} onSelect={select} onTogglePick={togglePickOf} />

      {panelNode}
    </div>
  );
}
