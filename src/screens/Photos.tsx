import { memo, useCallback, useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { Aurora } from "@/components/glass/aurora";
import { ChevronDownIcon } from "lucide-react";
import type { ColorSpace, NodeId, PhotoSummary, RenderStats, ResolvedSettings, ViewportMode } from "../api/types";
import { Button, FIRST_NODE_TAB, Filmstrip, filmstripCaption, NODE_FOR_SIMPLE_TAB, NODE_TABS, TabBar, tabPanelProps, TitleBlock, useLatest, useTabShortcuts, useWindowDrop } from "../components";
import { exportModel } from "../export/model";
import { cellLabel, summarise } from "../export/run";
import type { FilmstripPhoto, PanelKind, TabId } from "../components";
import { perfCount } from "../perf";
import { useAppState, useCurrentPhoto, useCurrentSettings, useStore } from "../state/context";
import { lookLabeller, looksInDisplayOrder } from "../state/looks";
import { folderSubline, isReady, photoName } from "../state/photo";
import { MASK_KIND_NAME } from "../state/graphOps";
import { hiddenByPhoto, hiddenFromSimpleView, INPUT_ID, MAIN_ID, nodeOf, OUTPUT_ID } from "../state/settingsGraph";
import type { AppStore } from "../state/store";
import { useShortcuts } from "../state/useShortcuts";
import { fitZoomExact, fitZoomOf, showsWhole, ZOOMS, zoomKey, zoomStep } from "../state/zoom";
import { CropPanel } from "./crop/CropPanel";
import { useCropSession } from "./crop/useCropSession";
import { ExportPanel } from "./export/ExportPanel";
import { ExportCell, Header } from "./Header";
import { Menu } from "./Menu";
import { Divider } from "./nodes/Divider";
import { GraphArea } from "./nodes/GraphArea";
import { hasHandles } from "./nodes/maskGeometry";
import { NodePanel } from "./nodes/NodePanel";
import { nodesGlyphText, NodesNotice } from "./nodes/NodesNotice";
import { useNodeGraph } from "./nodes/useNodeGraph";
import { useNodeLayout } from "./nodes/useNodeLayout";
import type { PickMode } from "./PickLayer";
import { PhotoView } from "./PhotoView";
import { lookSlugOf } from "./preview";
import { useFullScreen } from "./useFullScreen";
import { ColorTab } from "./tools/ColorTab";
import { CurveTab } from "./tools/CurveTab";
import { DetailTab } from "./tools/DetailTab";
import { LensTab } from "./tools/LensTab";
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
  const estimate = cell && photo ? fitZoomExact(cell, photo, rs?.values.input ?? null) : null;
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
        <Button variant="header" className="ym-fit" ref={ref} title="Zoom (Z toggles Fit and 100%, F full screen)" {...props}>
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
 * Before, Crop, Nodes, Add photos and the zoom menu, the photo, the filmstrip and the tool panel with its
 * seven tabs. The Crop panel and the Export panel replace the tool panel. F shows the photo alone, filling the window
 * (`docs/11` §Full screen). Nodes (N) is the node view (`docs/23` §7.1, layout A): the photo on top, a divider, the graph under it in
 * place of the filmstrip (its toolbar holds the photo switcher), the panel as before. Which view is on, the divider and Expand belong to the
 * window, as the tab does: nothing about them is saved.
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
  // The simple view (today's Photos) or the node view (`docs/23`), and the node the graph is on. Both are the window's: nothing is kept.
  const [view, setView] = useState<"simple" | "nodes">("simple");
  const [selectedNode, setSelectedNode] = useState<NodeId>(MAIN_ID);
  // The tab each kind of node shows in the node view; nodes of one kind share it (`docs/23` §7.5 *Tab memory*).
  const [nodeTabs, setNodeTabs] = useState<Record<PanelKind, TabId>>(FIRST_NODE_TAB);
  // Pick white (Colour › White balance, or W) or Pick colour (Colour › Colour warp): one crosshair layer over the photo, so the two are
  // exclusive. `asked` is what was asked for, and on which node (a node change ends a pick: `docs/23` §8); `picking` (below, once the crop,
  // the tab and the node are known) is what is on.
  const [asked, setAsked] = useState<{ mode: PickMode; node: NodeId } | null>(null);
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
  // The node the graph and the panel are on: the one that was chosen while the photo has it (an undo may take it away), else Main.
  const graphSettings = rs?.settings ?? null;
  const selectedItem = graphSettings ? nodeOf(graphSettings, selectedNode) : undefined;
  const onNode = selectedItem ? selectedNode : MAIN_ID;
  const onKind: PanelKind = selectedItem?.kind ?? "corrector";
  const nodeTab = nodeTabs[onKind];
  const nodeTabList = NODE_TABS[onKind];
  const onNodeNow = useLatest(onNode);
  // Pick white and Pick colour belong to the Colour tab (`docs/11` §Pick white, §Colour warp; in the node view the Input's Colour tab has
  // Pick white and a corrector's has Pick colour): a mode is on only while that tab is the one shown, in the tool panel (the Export and Crop
  // panels replace it), and only on the node it was asked on. It is read from the tab and the node, not switched off at each way of
  // leaving, so a click on a tab, ⌘1 to ⌘7, the Look tab's links, a click on another card and the panels all end it in the same render that
  // hides the tab: the layer is gone at once and takes no click. Leaving the screen (Start, the wizard) unmounts this one, and with it the mode.
  const colourShown = panel === "tools" && (view === "nodes" ? nodeTab : tab) === "color" && !cropping;
  // The Mask tab's pick (Pick colour of a Colour range, `docs/23` §7.6) belongs to the Mask tab of the node view and to the Colour range row
  // that is selected: it ends with the tab, the node or the photo, as the others do, and with the row.
  const maskRow = useAppState((s) => s.maskRow);
  const maskRows = selectedItem?.kind === "corrector" ? selectedItem.masks.items : [];
  const rowIndex = maskRow !== null && maskRow < maskRows.length ? maskRow : null;
  const selectedMask = rowIndex === null ? undefined : maskRows[rowIndex];
  const rangeShown = panel === "tools" && view === "nodes" && nodeTab === "mask" && onKind === "corrector" && !cropping && selectedMask?.kind === "color_range" && selectedMask.enabled;
  const picking: PickMode | null =
    asked !== null && asked.node === onNode && (asked.mode === "range" ? rangeShown : colourShown) ? asked.mode : null;
  /** Starts a pick on the selected node, or ends it (null). */
  const setPicking = useCallback((mode: PickMode | null, node?: NodeId): void => {
    setAsked(mode === null ? null : { mode, node: node ?? onNodeNow.current });
  }, [onNodeNow]);
  // What was asked for is forgotten with it, so coming back to Colour finds the mode off. A photo that changes, by whatever road (the
  // filmstrip, the arrows, Delete, an undo of a removal), ends it too, if it is not ended already.
  useEffect(() => {
    if (asked !== null && picking === null) setAsked(null);
  }, [asked, picking]);
  const photoId = photo?.id ?? null;
  useEffect(() => {
    setPicking(null);
    // A node of the last photo may not be on this one.
    setSelectedNode(MAIN_ID);
  }, [photoId, setPicking]);
  // The selected pin of the colour warp is an index into one node's pins: another node starts with none.
  useEffect(() => {
    store.selectWarpPin(null);
    // A mask row is an index into one node's rows: another node starts with none selected.
    store.selectMaskRow(null);
  }, [store, onNode]);
  // Full screen (F): the photo alone. The Export panel and Crop need the panel, so neither shares the window with it.
  const fullScreen = useFullScreen(!cropping && panel !== "export");
  // The node view's layout is on while the view is and the window is not full screen (the photo alone is the whole window then).
  const nodesOn = view === "nodes" && !fullScreen.on;
  const nodeLayout = useNodeLayout(nodesOn);
  // Show mask (`docs/23` §5.8): the core draws the mask of the corrector the node view has selected. It is on only while the node view is (not
  // in the simple view or full screen), not while Crop or Export shows, and only on a corrector: the Input and the Output have no mask. The
  // view's on and style are the store's and come back with the node; the core is told when any of them changes, and for a new photo.
  const maskOn = useAppState((s) => s.maskView.on);
  const maskNode = nodesOn && !cropping && panel !== "export" && graphSettings !== null && onKind === "corrector" && typeof onNode === "number" ? onNode : null;
  const photoOnScreen = photo?.id ?? null;
  useEffect(() => {
    store.setMaskNode(maskNode);
  }, [store, maskNode, photoOnScreen]);
  // Leaving the screen ends it.
  useEffect(() => () => store.setMaskNode(null), [store]);
  // The photos whose graph has more than the simple view shows (a glyph on their thumbnails): a Map that is the same while the answer is.
  const hiddenNodes = useAppState((s) => hiddenByPhoto(s.settings));
  const graph = useNodeGraph({ store, settings: graphSettings, photoId: photo?.id ?? null, selected: onNode, select: setSelectedNode });
  // A card shows the picture after its node (`docs/23` §5.7), which the store asks the core for while the node view is on (when a change
  // has landed, never during a drag). Until a node's first picture arrives its card shows the photo's thumbnail, the filmstrip's.
  useEffect(() => (nodesOn ? store.watchNodeThumbnails() : undefined), [store, nodesOn]);
  const nodePictures = useAppState((s) => (photo ? s.nodeThumbnails[photo.id] : undefined));
  const photoThumbnail = photo ? thumbnails[photo.id] : undefined;
  const thumbnailOf = useCallback(
    (id: NodeId): string | undefined => (typeof id === "number" ? nodePictures?.[id] : undefined) ?? photoThumbnail,
    [nodePictures, photoThumbnail],
  );

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
      photos.map((p) => {
        const nodes = hiddenNodes.get(p.id);
        return {
          id: p.id,
          name: photoName(p.file_name),
          src: thumbnails[p.id],
          status: p.status.kind === "ready" ? "ready" : p.status.kind === "skipped" ? "failed" : "developing",
          picked: p.picked,
          included: included?.has(p.id) ?? false,
          nodes: nodes ? nodesGlyphText(nodes) : undefined,
        };
      }),
    [photos, thumbnails, included, hiddenNodes],
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
   * Remove from filmstrip (the thumbnail's menu, Delete): a crop or a pick in progress ends first, as when another photo is chosen,
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

  /**
   * The Nodes button and N: the other view. Picking ends, as it does on any change of what the cell is for (`docs/11` §Pick white). Entering
   * the node view selects the node and the tab that match the tab the simple view stands on: Look → Output › Look, Lens → Input › Lens,
   * Light → Main › Light, Colour → Main › Colour, Detail → Input › Detail, Curve → Main › Curve, RAW → Input › RAW (`docs/23` §7.5, §14
   * decision 21).
   */
  const toggleNodes = (): void => {
    setPicking(null);
    if (view === "simple") {
      const target = NODE_FOR_SIMPLE_TAB[tab === "mask" ? "light" : tab];
      setSelectedNode(target.kind === "input" ? INPUT_ID : target.kind === "output" ? OUTPUT_ID : MAIN_ID);
      setNodeTabs((was) => ({ ...was, [target.kind]: target.tab }));
    }
    setView((was) => (was === "nodes" ? "simple" : "nodes"));
  };
  /** "Open nodes" of the simple view's notice: the node view, with Main selected (on the simple tab's tab when Main has it). */
  const openNodes = (): void => {
    setPicking(null);
    setSelectedNode(MAIN_ID);
    const target = NODE_FOR_SIMPLE_TAB[tab === "mask" ? "light" : tab];
    if (target.kind === "corrector") setNodeTabs((was) => ({ ...was, corrector: target.tab }));
    setView("nodes");
  };

  const showTab = (next: TabId): void => {
    fullScreen.leave();
    store.setPanel("tools");
    setTab(next);
  };
  /** A tab of the selected node (a click, or ⌘1 to ⌘5): the node's kind remembers it. */
  const showNodeTab = (next: TabId): void => {
    store.setPanel("tools");
    setNodeTabs((was) => ({ ...was, [onKind]: next }));
  };
  // ⌘1 to ⌘7 are the simple view's tabs; in the node view ⌘1 to ⌘5 are the selected node's own (a corrector's four, the Output's two).
  useTabShortcuts(
    (next) => {
      fullScreen.leave();
      if (view === "nodes") showNodeTab(next);
      else showTab(next);
    },
    !cropping && overlay === null,
    view === "nodes" ? nodeTabList : undefined,
  );

  // The zoom and the Fit zoom are read when the key is pressed, not at each render: this screen does not follow them.
  const zoomBy = (delta: number): void => void store.setZoom(zoomStep(store.getState().zoom, delta, store.fitPercentExact()));
  const toggleZoom = (): void => void store.toggleZoom();

  const startCrop = (): void => {
    if (!rs || !photo || !isReady(photo)) return;
    setPicking(null);
    store.setPanel("tools");
    crop.start();
  };
  // Pick white, from the White balance card's toggle. It turns Pick colour off, and ends itself after one click.
  const setPickingWhite = useCallback((on: boolean): void => {
    setAsked((was) => (on ? { mode: "white", node: onNodeNow.current } : was?.mode === "white" ? null : was));
  }, [onNodeNow]);
  // W: opens the Colour tab and starts Pick white; W again ends it. Leaving the Export panel for the tab is what ⌘4 does. In the node view
  // the Input is where the photo's white is set: W selects it first and opens its Colour tab (`docs/23` §8).
  const togglePickWhite = (): void => {
    if (!rs || !photo || !isReady(photo) || cropping) return;
    if (picking === "white") {
      setPicking(null);
      return;
    }
    store.setPanel("tools");
    if (view === "nodes") {
      setSelectedNode(INPUT_ID);
      setNodeTabs((was) => ({ ...was, input: "color" }));
      setPicking("white", INPUT_ID);
    } else {
      setTab("color");
      setPicking("white");
    }
  };
  // Pick colour, from the Colour warp card's toggle. It turns Pick white off, and stays on for several clicks until Esc or the toggle.
  const setPickingColor = useCallback((on: boolean): void => {
    setAsked((was) => (on ? { mode: "color", node: onNodeNow.current } : was?.mode === "color" ? null : was));
  }, [onNodeNow]);
  // Pick colour of a Colour range, from its card's toggle (or when the row is added). Exclusive with the other two; stays on for several clicks.
  const setPickingRange = useCallback((on: boolean): void => {
    setAsked((was) => (on ? { mode: "range", node: onNodeNow.current } : was?.mode === "range" ? null : was));
  }, [onNodeNow]);
  // A click with Pick colour on reads the chroma at the input of the selected node's warp and puts the pin in that node (`docs/23` §7.5).
  const pickedNode = view === "nodes" && typeof onNode === "number" ? onNode : MAIN_ID;
  const onPicked = useCallback(
    (point: { x: number; y: number }): void => {
      if (picking === "color") {
        void store.pickColor(point.x, point.y, pickedNode);
        return;
      }
      if (picking === "range") {
        // The chroma at the node, set as the centre of the selected Colour range row (`docs/23` §7.6).
        if (rowIndex !== null && typeof onNode === "number") void store.pickRange(point.x, point.y, onNode, rowIndex);
        return;
      }
      setPicking(null);
      void store.pickWhite(point.x, point.y);
    },
    [store, picking, pickedNode, setPicking, rowIndex, onNode],
  );
  // The handles of the selected row of the selected node (`docs/23` §7.6 *On the photo*): in the node view, whatever tab the panel shows; none
  // while Crop or Export shows (the graph is read only), none while a pick is on (PhotoView), none for a row that is off or a kind with no handles.
  const maskHandles = useMemo(
    () =>
      nodesOn && !cropping && panel !== "export" && onKind === "corrector" && typeof onNode === "number" && rowIndex !== null && hasHandles(selectedMask)
        ? { node: onNode, index: rowIndex, row: selectedMask }
        : null,
    [nodesOn, cropping, panel, onKind, onNode, rowIndex, selectedMask],
  );
  // The Brush tool (`docs/23` §7.6): on for the selected Brush row while the store says so (adding a Brush row, a click on it or on Paint or Erase starts it;
  // another row, another node or photo and Esc end it), in the node view with the row on. Crop, Export, Before and a pick hide it (PhotoView draws the pick
  // layer, the crop frame or the brush, never two).
  const brushAsked = useAppState((s) => s.brushTool);
  const brush = useMemo(
    () =>
      brushAsked && nodesOn && !cropping && panel !== "export" && onKind === "corrector" && typeof onNode === "number" && rowIndex !== null && selectedMask?.kind === "brush" && selectedMask.enabled
        ? { node: onNode, index: rowIndex }
        : null,
    [brushAsked, nodesOn, cropping, panel, onKind, onNode, rowIndex, selectedMask],
  );
  // The look chip says "MASK · LINEAR" while the selected node has a selected mask row (`docs/23` §7.1).
  const maskChip = nodesOn && !cropping && panel !== "export" && selectedMask ? `MASK · ${(MASK_KIND_NAME[selectedMask.kind] ?? selectedMask.kind).toUpperCase()}` : undefined;
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
          fullScreen.leave();
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
          // The Export panel works on the photos as they are; Back to photos, then remove. The node view has no filmstrip: its Delete is
          // the graph's (`delete-node`), never a photo's.
          if (panel === "export" || view === "nodes") return false;
          void store.removeCurrentPhoto();
          return true;
        // The graph's keys (`docs/23` §7.7): Delete (the selected link, else the selected node), B, and ⌥← and ⌥→. Not while the graph is
        // read only (Crop, Export).
        case "delete-node":
        case "bypass-node":
        case "move-node-earlier":
        case "move-node-later":
          if (view !== "nodes" || cropping || panel === "export") return false;
          if (shortcut.action === "delete-node") graph.deleteSelection();
          else if (shortcut.action === "bypass-node") graph.toggleBypass();
          else if (shortcut.action === "move-node-earlier") graph.moveEarlier();
          else graph.moveLater();
          return true;
        // Show or hide the mask of the selected corrector (`docs/23` §7.7): not on the Input or the Output, and not while the graph is read only.
        case "mask-view":
          if (view !== "nodes" || cropping || panel === "export" || maskNode === null) return false;
          store.toggleMaskView();
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
        // Crop and Pick white need the panel (the Colour tab with the result) and the header: full screen ends first.
        case "crop":
          fullScreen.leave();
          if (!cropping) startCrop();
          return true;
        case "auto-crop":
          if (!crop.session) return false;
          crop.session.toggleAuto();
          return true;
        case "pick-white":
          fullScreen.leave();
          togglePickWhite();
          return true;
        // The node view needs the header and the panel, as Crop does: full screen ends first. Not from the Export panel (the graph is
        // read only there) or while cropping.
        case "nodes":
          if (cropping || panel === "export") return false;
          fullScreen.leave();
          toggleNodes();
          return true;
        case "full-screen":
          // Not from the Export panel or Crop, which are the panel.
          if (cropping || panel === "export") return false;
          fullScreen.toggle();
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
          else if (brush && !before) store.setBrushTool(false);
          else if (crop.session) crop.session.cancel();
          else if (panel === "export") store.closeExport();
          else if (view === "nodes" && graph.escape()) return true;
          else if (fullScreen.on) fullScreen.leave();
          else return false;
          return true;
        case "confirm":
          crop.session?.done();
          return true;
        default:
          return false;
      }
    },
    {
      // The number keys pick looks on the Look tab, which in the node view is the Output's.
      tab: panel === "tools" ? (view === "nodes" ? nodeTab : tab) : undefined,
      mode: cropping ? "crop" : picking === "white" ? "pick-white" : picking === "color" || picking === "range" ? "pick-color" : null,
      nodeView: view === "nodes",
    },
    overlay === null,
  );

  // What the simple view cannot show of this photo (`docs/23` §6); the node view's panel is the node's, so it has no notice.
  const hidden = view === "simple" && photo && rs && isReady(photo) ? hiddenFromSimpleView(rs.settings) : null;

  const copyPaste = <CopyPasteButtons store={store} canCopy={rs !== null && photo !== null && isReady(photo)} />;
  let panelNode;
  if (crop.session) {
    panelNode = <CropPanel session={crop.session} />;
  } else if (panel === "export") {
    panelNode = <ExportPanel store={store} />;
  } else if (view === "nodes") {
    // The node view's panel edits the selected node: its head, its kind's tabs and its cards (`docs/23` §7.5).
    panelNode = (
      <NodePanel
        store={store}
        photo={photo}
        rs={rs}
        node={onNode}
        tab={nodeTab}
        onTab={showNodeTab}
        graph={graph}
        pickingWhite={picking === "white"}
        onPickWhite={setPickingWhite}
        pickingColor={picking === "color"}
        onPickColor={setPickingColor}
        pickingRange={picking === "range"}
        onPickRange={setPickingRange}
        foot={copyPaste}
        hidden={fullScreen.on}
      />
    );
  } else {
    panelNode = (
      <aside aria-label="Tools" className="ym-tools" hidden={fullScreen.on || undefined}>
        <TabBar value={tab} onChange={setTab} />
        {hidden ? <NodesNotice hidden={hidden} onOpen={openNodes} /> : null}
        <div className="ym-tools__scroll" {...tabPanelProps(tab)}>
          <TitleBlock tab={tab} />
          {photo && rs && isReady(photo) ? (
            tab === "look" ? (
              <LookTab store={store} photo={photo} rs={rs} onOpenTab={setTab} />
            ) : tab === "lens" ? (
              <LensTab store={store} photo={photo} rs={rs} />
            ) : tab === "light" ? (
              <LightTab store={store} photo={photo} rs={rs} />
            ) : tab === "color" ? (
              <ColorTab
                store={store}
                photo={photo}
                rs={rs}
                pickingWhite={picking === "white"}
                onPickWhite={setPickingWhite}
                pickingColor={picking === "color"}
                onPickColor={setPickingColor}
              />
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
          {copyPaste}
        </div>
      </aside>
    );
  }

  const exporting = panel === "export";

  return (
    <div
      className="ym-screen"
      data-screen="photos"
      data-panel={panel}
      data-count={COUNT_IN_HEADER ? "header" : undefined}
      data-dragging={dragging || undefined}
      data-fullscreen={fullScreen.on || undefined}
      data-nodes={nodesOn || undefined}
      style={nodesOn ? ({ "--editor-h": `${nodeLayout.editor}px`, "--viewer-h": `${nodeLayout.viewer}px` } as CSSProperties) : undefined}
    >
      <Aurora className="ym-aurora" animate={false} />
      <Header
        titleAs="h1"
        title={session?.folder_name ?? "Yellowmoon"}
        subline={session ? folderSubline(session) : ""}
        export={<ExportCellContainer store={store} />}
        hidden={fullScreen.on}
      >
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
            <Button variant="header" pressed={view === "nodes"} disabled={cropping} onClick={toggleNodes} title="Nodes (N)">
              Nodes
            </Button>
            <AddPhotosButton store={store} />
            <ZoomMenu store={store} photo={photo} rs={rs} cell={cell} />
            {picking === "color" || picking === "range" ? (
              // Pick white's hint is a pill on the photo (`PickWhiteHint`); Pick colour's stays in the header (the Colour range's too).
              <span className="ym-status" data-hint="true" role="status">
                Click a colour in the photo.
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

      <div className="ym-viewer">
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
          maskHandles={maskHandles}
          brush={brush}
          maskChip={maskChip}
        />
      </div>

      {nodesOn ? (
        <>
          <Divider
            editor={nodeLayout.editor}
            area={nodeLayout.area}
            min={nodeLayout.min}
            max={nodeLayout.max}
            onResize={nodeLayout.setEditor}
            onNudge={nodeLayout.nudge}
            onReset={nodeLayout.reset}
          />
          <GraphArea
            position={session && session.current_index !== null ? session.current_index + 1 : null}
            count={photos.length}
            onPrevious={() => stepPhoto(-1)}
            onNext={() => stepPhoto(1)}
            settings={graphSettings}
            photoKey={photo?.id ?? null}
            selected={onNode}
            graph={graph}
            thumbnailOf={thumbnailOf}
            expanded={nodeLayout.expanded}
            onToggleExpanded={nodeLayout.toggleExpanded}
            dimmed={cropping || exporting}
            maskOn={maskOn}
            onToggleMask={() => store.toggleMaskView()}
          />
        </>
      ) : null}

      <PhotoStrip
        mode={exporting ? "export" : "browse"}
        photos={strip}
        currentId={photo?.id ?? null}
        tab={exporting ? undefined : tab}
        onSelect={select}
        onTogglePick={togglePickOf}
        onRemove={exporting ? undefined : removePhoto}
        onRemovePicked={exporting ? undefined : removePicked}
        hidden={fullScreen.on || nodesOn}
      />

      {panelNode}
      {/* Said once on entering; the hidden header and panel are gone for a screen reader too. */}
      <div className="sr-only" aria-live="polite">
        {fullScreen.on ? "Full screen. Press F or Esc to leave." : ""}
      </div>
    </div>
  );
}
