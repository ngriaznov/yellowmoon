/**
 * The frontend's mirror of the core's state, and the one place that talks to `Api`
 * (`docs/11-ui.md` §Frontend structure: "State lives in Rust; the frontend keeps a mirror for
 * rendering and sends patches").
 *
 * The store holds what the screens draw: the session (photos, current, picks), each photo's
 * resolved settings, the look library, the status line from `render_stats`, and the few things
 * that belong to the window rather than the core (the screen, the "Apply to" scope, Before,
 * zoom). It is plain TypeScript with a `subscribe`, so it is tested without React; `context.tsx`
 * makes it available to components.
 *
 * Settings are a graph of nodes (`docs/23`), and changes go out as node patches (`patchOf` in
 * `settingsGraph.ts`), one at a time per photo: a slider drag sends one node's partial. While one
 * is in flight, newer drag patches fold into a single waiting patch (`foldPatches`), so a slow core
 * is never flooded and the last value always arrives. The reply of a patch replaces the mirrored
 * settings, unless a newer patch is waiting (its reply will carry everything). Final patches (a
 * released drag, a key, a typed value) are recorded for undo as the difference, per node and for
 * the links, between the settings before the gesture and the core's reply (`diffSettings`).
 */
import type { Api, Unsubscribe } from "../api";
import { toAppError } from "../api";
import { isUnreadableMessage } from "../api/openMessages";
import type {
  ApplyScope,
  AutoProvider,
  AutoStatus,
  AppError,
  BalanceSetting,
  BrushStroke,
  ChromaSample,
  ChromaScatter,
  CliLinkStatus,
  ColorBalance,
  ColorBalancePoint,
  CropAspect,
  DepthFar,
  CropSuggestion,
  Curves,
  DevelopSettings,
  ExportJob,
  ExportOptions,
  ExportProgress,
  FilmCatalogue,
  FocusPlaneResult,
  GradePatch,
  GradeValues,
  HistogramData,
  LensInfo,
  Look,
  LookId,
  LookInput,
  LookLibrary,
  MainAutoFlags,
  MainEstimates,
  MapFrom,
  MaskFindPrompt,
  MaskItem,
  MaskKind,
  MasksPatch,
  MaskStyle,
  MaskView,
  MenuAction,
  MenuState,
  NodeGrade,
  NodeId,
  NodeItem,
  NodeThumbnailsReady,
  ObjectState,
  DepthState,
  FaceState,
  VisionState,
  OpenProgress,
  OpenSummary,
  Orientation,
  PasteReport,
  PercentSetting,
  PhotoId,
  PhotoProgress,
  PhotoStatus,
  PhotoSummary,
  PixelSample,
  Preferences,
  PreferencesPatch,
  RenderStats,
  ResolvedCorrector,
  ResolvedInput,
  ResolvedOutput,
  ResolvedSettings,
  SessionChange,
  SessionState,
  SettingsPatch,
  SourceRecognition,
  ThumbnailReady,
  ViewportBackendKind,
  ViewportBusy,
  ViewportFrame,
  ViewportInput,
  ViewportMode,
  ViewportRect,
} from "../api/types";
import { filterArwPaths } from "../components/useWindowDrop";
import { clampToDisc, movingPin, WARP_COORD_LIMIT, WARP_MAX_PINS } from "../components/warpMath";
import { BRUSH_FULL, brushPointCount, DEFAULT_BRUSH, MAX_BRUSH_POINTS } from "./brushStrokes";
import type { BrushSettings } from "./brushStrokes";
import { COLOR_RANGE_REACH } from "./maskRows";
import { DEFAULT_OBJECT_MODE, foundText, FOUND_PILL_MS, NOTHING_FOUND } from "./objectMask";
import type { ObjectMode } from "./objectMask";
import { isTauri } from "../env";
import { defaultExportOptions, defaultScope, sameOptions } from "../export/options";
import { applyProgress, startRun } from "../export/run";
import { templateProblem } from "../export/template";
import type { ExportRun } from "../export/run";
import { diffSettings, PhotoHistory } from "./history";
import type { HistoryEntry } from "./history";
import { perfCount } from "../perf";
import { stampTick } from "../perfTimeline";
import { isReady, mergeSession, photoName, releaseUrl, removedNotice, restoredNotice, withoutPhotos, withRestoredPhotos } from "./photo";
import { RequestLimiter } from "./requestLimiter";
import { gestureHeld, onGestureEnd } from "./gestureHold";
import { share } from "./share";
import {
  applyMasks,
  applyPatch,
  cardsOf,
  foldPatches,
  INPUT_ID,
  inputOf,
  gradeOf,
  isCardItem,
  isEmptyPatch,
  MAIN_ID,
  mainOf,
  nodeIdOfKey,
  nodeOf,
  OUTPUT_ID,
  outputOf,
  outputPatch,
  patchOf,
} from "./settingsGraph";
import type { CorrectorItem, InputItem, OutputItem } from "./settingsGraph";
import { ViewportInputQueue } from "./viewportInputQueue";
import type { StreamInput } from "./viewportInputQueue";
import { fitZoomExact, fitZoomOf, showsWhole, zoomAfter } from "./zoom";

/** The screens of the app (`docs/11-ui.md` §Screens that have a window of their own). */
export type Screen = "start" | "photos" | "import-looks";

/** What Photos shows in the right-hand panel: the seven tool tabs, or Export (`docs/11-ui.md` §Export). */
export type Panel = "tools" | "export";

/** What the notice bar offers next to Dismiss: Undo for a removal (`docs/11` §Removing photos). Only the notice of a removal has one. */
export type NoticeAction = "undo-removal";

/** A dialog laid over the screen: Preferences (⌘,), About, the question before closing a session during an export, the `.cube` import. */
export type Overlay = "preferences" | "about" | "confirm-close" | "import-cube";

/** One file of the `.cube` import dialog (`docs/08` §Any other .cube). */
export interface CubeItem {
  found: SourceRecognition;
  /** What the LUT expects as input; null until answered (not asked for a recognised package). */
  answer: LookInput | null;
}

/** The `.cube` import dialog's state. */
export interface CubeImport {
  items: CubeItem[];
  /** The import is running. */
  busy: boolean;
}

/** Said when a drop holds nothing the app can open. */
export const NO_ARW_NOTICE = "No ARW files in what you dropped.";

/** Said when files are added to a session and every one of them is open already. */
export const ALREADY_OPEN_NOTICE = "These photos are already open.";

/** Whether two mask views are the same one (a node and a style, or none). */
function sameMaskView(a: MaskView | null, b: MaskView | null): boolean {
  return a === b || (a !== null && b !== null && a.node === b.node && a.style === b.style && (a.row ?? null) === (b.row ?? null) && (a.depth ?? false) === (b.depth ?? false) && (a.face ?? null) === (b.face ?? null));
}

/** How long an open reads before Start and Photos say how far it has got (`openProgress`). */
export const OPEN_PROGRESS_DELAY_MS = 300;

/** What Start and Photos say after Cancel stopped an open: how many of the files were read and kept (`docs/26` H-10). */
export function stoppedNotice(opened: number): string {
  if (opened === 0) return "Stopped. No files were opened.";
  return `Stopped. ${opened} ${opened === 1 ? "file was" : "files were"} opened.`;
}

/** What Photos says after files were added to the session: "Added 3 photos. 2 were already open." */
export function openedNotice(opened: OpenSummary): string {
  if (opened.added === 0 && opened.already_open === 0) return NO_ARW_NOTICE;
  if (opened.added === 0) return ALREADY_OPEN_NOTICE;
  const added = `Added ${opened.added} ${opened.added === 1 ? "photo" : "photos"}.`;
  if (opened.already_open === 0) return added;
  return `${added} ${opened.already_open} ${opened.already_open === 1 ? "was" : "were"} already open.`;
}

/**
 * What Photos says after Paste settings: "Pasted the settings of DSC04213." for one photo, "Pasted the settings of DSC04213 to 5
 * photos." for more, and how many were left out when a photo was still developing or could not be opened. `pasted` counts the photos
 * that now have the settings, whether or not anything changed in them.
 */
export function pastedNotice(source: string, pasted: number, skipped: number): string {
  const left = (n: number): string => `${n} ${n === 1 ? "photo" : "photos"}`;
  if (pasted === 0) return `Nothing was pasted: ${left(skipped)} ${skipped === 1 ? "isn't" : "aren't"} developed.`;
  const done = pasted === 1 ? `Pasted the settings of ${source}.` : `Pasted the settings of ${source} to ${pasted} photos.`;
  if (skipped === 0) return done;
  return `${done} ${left(skipped)} ${skipped === 1 ? "was" : "were"} skipped because ${skipped === 1 ? "it isn't" : "they aren't"} developed.`;
}

/**
 * The settings Copy settings took from a photo, kept for pasting onto others. It lives in the store, for the running app only: never
 * written to disk (`docs/01` §Persistence, AGENTS rule 4) and never put on the system clipboard. A session closing does not clear it,
 * so settings can go from one folder to another.
 */
export interface CopiedSettings {
  /** The photo they came from, as the notices name it: "DSC04213". */
  source: string;
  /**
   * The id of that photo while its session is open; null once the session is closed (the next session numbers its photos again, so the
   * id would name another photo). Paste Settings to Picked uses it to leave the source's own picked white alone.
   */
  sourceId: PhotoId | null;
  /** What `copy_settings` answered: the whole graph, but the Input's crop, orientation and Pick white point. */
  patch: SettingsPatch;
}

/** Everything the screens draw. Replaced, never changed in place. */
/** Show mask: on or off, and how it looks when it is on. The style starts as Overlay. */
export interface MaskViewState {
  on: boolean;
  style: MaskStyle;
}

export interface AppState {
  screen: Screen;
  /** Where "Back" goes from a full-window detour such as the import wizard. */
  returnTo: Screen;
  /** The core's session; null before anything is open. */
  session: SessionState | null;
  /** Thumbnail URLs by photo id, from `thumbnail_ready`. */
  thumbnails: Readonly<Record<PhotoId, string>>;
  /**
   * The pictures of the node view's cards (`docs/23` §5.7), per photo and corrector: the address of the photo after that node and the
   * Output, from `node_thumbnails_ready`. A node has none until its first arrives (its card shows the photo's thumbnail meanwhile).
   * Kept for the running app only, like the thumbnails.
   */
  nodeThumbnails: Readonly<Record<PhotoId, Readonly<Record<number, string>>>>;
  /** Resolved settings by photo id, for the photos that were shown. */
  settings: Readonly<Record<PhotoId, ResolvedSettings>>;
  /** The look library; null until it has been asked for. */
  looks: readonly Look[] | null;
  /**
   * The lenses of the Lens card's list (`lens_list`, `docs/21` §10), asked for once; null until the first answer, empty when the
   * lens data are missing.
   */
  lenses: readonly LensInfo[] | null;
  /** The last `render_stats`: "Neural · 42 ms · Display P3". */
  stats: RenderStats | null;
  /** Showing the photo before the look and display transform. */
  before: boolean;
  zoom: ViewportMode;
  /**
   * Show mask (`docs/23` §5.8): whether the node view shows the selected corrector's mask, and in which style. Neither is saved or
   * per photo; it stays across nodes and photos for the running app. The core draws it; it is sent whenever it, the style or the
   * selected node changes ({@link Store.setMaskNode}).
   */
  maskView: MaskViewState;
  /**
   * The mask row whose eye is on (`docs/23` §5.8, §7.6): the index of a row of the node on show, or null. View state like {@link AppState.maskView}: not
   * saved, not undone, not copied. At most one; it and Show mask are never on together; the style is the View card's.
   */
  maskEye: number | null;
  /**
   * The Depth row whose **Show depth** is on (`docs/23` §4.8, `docs/11`): the index of a row of the node on show, or null. View state like the eye: not saved, not
   * undone, not copied, and exclusive with Show mask and the eyes (turning one on turns the others off). The core draws the photo's depth map as grey for it.
   */
  showDepth: number | null;
  /**
   * The face whose **Show skin areas** eye is on (`docs/27` S-11c): its number (from 1) for the Face Refinement node on show, or null. View state like the mask
   * eye: not saved, not undone, not copied, exclusive with Show mask, the row eyes and Show depth, and it ends where they do (another node, the photo, Before,
   * Esc) and when the Face tab goes away. The core draws the skin areas tinted over the photo, whatever the style says.
   */
  skinAreas: number | null;
  /** The mask view the core was last told to draw: what {@link AppState.maskView} comes to for the node on show (the browser build's fixture draws it). */
  maskShown: MaskView | null;
  /** "Apply to" on the Look card. */
  applyScope: ApplyScope;
  /**
   * The Look tab's sections the person folded away. Kept for the running app only (`docs/01` §Persistence: nothing about the look
   * list goes to disk); it outlives a session and starts empty (every section open) the next time the app starts.
   */
  collapsedLookLibraries: readonly LookLibrary[];
  /** A problem with the last drop, in words ("No ARW files in what you dropped."). */
  dropNotice: string | null;
  /** The last error from the core, until dismissed. */
  error: AppError | null;
  /** Open files or Open folder is waiting for the core or the dialog. */
  opening: boolean;
  /**
   * How far the core has read the files of the open that is under way, once it has taken `OPEN_PROGRESS_DELAY_MS` (a small open
   * finishes before anyone could read the line); null otherwise. Start and Photos say "Reading {done} of {total} files…" with Cancel.
   */
  openProgress: OpenProgress | null;
  /** The current photo's histogram, when asked for. */
  histogram: HistogramData | null;
  /** The display values at the middle of the photo cell, while the Light tab watches them (`watchHistogram`). */
  middleReadout: PixelSample | null;
  /** The current photo's colours on the colour warp's plane, while the Colour tab shows it (`watchChromaScatter`). */
  chromaScatter: ChromaScatter | null;
  /**
   * The selected pin of Colour › Colour warp: an index into the current photo's `color_warp.pins`, or null. It belongs to the window,
   * not to the settings (nothing about it is undone or copied); a new photo starts with none, and an index past the end counts as none.
   */
  warpPin: number | null;
  /**
   * The selected mask row of the node the node view is on (`docs/23` §7.6): an index into that node's `masks.items`, or null. It belongs to
   * the window, like `warpPin`: nothing about it is undone or copied, another node or photo starts with none, and an index past the end
   * counts as none. Its handles show on the photo, whatever tab the panel is on.
   */
  maskRow: number | null;
  /**
   * The brush the next strokes of a Brush row are drawn with (`docs/23` §7.6): the BRUSH card's Paint · Erase, Size and Hardness. Window
   * state like `maskRow`: not saved, not undone, and a stroke that is drawn keeps what it was drawn with.
   */
  brush: BrushSettings;
  /**
   * The Brush tool is on for the selected row (`docs/23` §7.6): a drag on the photo paints. Adding a Brush row, a click on a Brush row and a
   * click on Paint or Erase turn it on; Esc, another row, another node and another photo turn it off. Whether the cursor and the drawing
   * surface show also depends on the screen (Crop, Export, Before and a pick hide them: `Photos`).
   */
  brushTool: boolean;
  /** The kinds of mask this machine can make, once asked (`mask_kinds`, `ensureMaskKinds`): what the Add mask menu lists. */
  maskKinds: readonly MaskKind[] | null;
  /** The Object card's tool (`docs/23` §7.6): what a plain click on the photo does. A window setting; it is no patch and no undo step. */
  objectMode: ObjectMode;
  /**
   * The Object tool is on for the selected row (`docs/23` §7.6): a click on the photo finds an object, a drag with the Box tool draws a box. Adding an Object
   * row, a click on one and a click on a tool turn it on; Esc, another row, another node and another photo turn it off.
   */
  objectTool: boolean;
  /** The photos whose SAM encoder is working (`object_state`): their Object rows say "Getting ready…". */
  objectPreparing: readonly PhotoId[];
  /** A `mask_find` is under way for this photo: its Object rows say "Finding…". */
  objectFinding: PhotoId | null;
  /**
   * The kinds whose newest Subject or People mask found nothing, by photo (`vision_state`, `docs/23` §7.6): their rows say "Nothing found". The core
   * sends the list each time its worker ends; a photo it has not sent one for has none.
   */
  visionNothing: Readonly<Record<number, readonly MaskKind[]>>;
  /**
   * The photos whose depth map failed, with the sentence (`depth_state`, `docs/23` §4.8): their Depth rows say "Can't be made" and the card says why, with Try
   * again. A photo whose map was made (or is being made again) has none.
   */
  depthFailed: Readonly<Record<number, string>>;
  /**
   * Where the faces of each photo are (`face_state`, `docs/27` S-11c): the Face Refinement node's panel and the boxes drawn on the photo read it. A photo the
   * core has said nothing about has no entry ({@link Store.refreshFaces} asks).
   */
  faces: Readonly<Record<number, FaceState>>;
  /** The face the Face tab is on, counted from 1 (window state: not saved, not undone). A number past the faces found reads as the last one. */
  faceSelected: number;
  /** The photos whose file has no focus point (`focus_plane`'s `no_focus_point`): Focus plane is off for them, with the reason. */
  noFocusPoint: Readonly<Record<number, true>>;
  /**
   * What Focus plane last set on a photo, when the camera's point is exactly the frame's centre (which Sony also writes when it does not know): the card says "The
   * camera may not have recorded where it focused. Check the result." while the row still has this far limit.
   */
  focusCentre: Readonly<Record<number, number>>;
  /** The pill at the top of the photo, "Object found · 0.4 s", for three seconds after a find. */
  objectPill: { seq: number; text: string } | null;
  /** Undo and redo are available for the current photo. */
  canUndo: boolean;
  canRedo: boolean;
  /** The preferences, once loaded (`docs/01` §Persistence). */
  preferences: Preferences | null;
  /** Whether the command-line tool is linked onto the PATH, once asked for (Preferences). */
  cliLink: CliLinkStatus | null;
  /**
   * The providers of Auto and which one grades (`auto_status`, `docs/22` section 12.7), asked for each time Preferences opens (the one
   * read that makes the programs check) and read again from the cache after a change of the choice, the consent or the instructions and
   * when a provider changes; null before the first answer. Preferences shows "Auto grading" while `visible` or while the saved choice is
   * a program.
   */
  auto: AutoStatus | null;
  /**
   * The program whose consent dialog is open over Preferences (`docs/22` section 12.3): chosen in the dropdown before the person agreed to
   * send photos through it. The dropdown keeps showing the previous provider until they say yes. It lives only while Preferences is the
   * dialog: every other overlay, and none, clears it (`set`).
   */
  consentFor: "claude" | "agy" | null;
  /** The operating system the app runs on (`app_info`: "macos", "windows", "linux"), asked for when Preferences opens; null before the answer. */
  os: string | null;
  /** Which panel Photos shows. */
  panel: Panel;
  /** The Export panel's options: the remembered defaults, changed by the person (`docs/09` §Options). */
  exportOptions: ExportOptions;
  /** Which photos the Export panel is set to export. */
  exportScope: ApplyScope;
  /** The export that is running or has just finished; null when none. It outlives the panel and the session. */
  exportRun: ExportRun | null;
  /** Words about an export that was stopped, for the options view ("Export stopped. 1 photo was saved."). */
  exportStopped: string | null;
  /** A dialog laid over the screen. */
  overlay: Overlay | null;
  /** The `.cube` import dialog. */
  cubeImport: CubeImport | null;
  /** A short message that is not an error ("Yellowmoon is up to date."), until dismissed. */
  notice: string | null;
  /** The button the notice has besides Dismiss: Undo, after photos were removed from the filmstrip. Null for every other notice. */
  noticeAction: NoticeAction | null;
  /**
   * Which viewport backend draws the photo (`viewport_info`): `native` leaves the photo cell transparent, `fallback` shows the
   * frames below. Null while unknown, and for an API that has none (the browser mock), where the screen draws its own picture.
   */
  viewportBackend: ViewportBackendKind | null;
  /** The fallback viewport's latest frame (`viewport_frame`); null when it has none or took the picture away. */
  viewportFrame: ViewportFrame | null;
  /** An export is queued or running in the core (`export_busy`), also one this window did not start. */
  exportBusy: boolean;
  /**
   * The viewport is working on a frame of the photo on screen or on its full-resolution tiles (`busy`), and the raw stages of that
   * photo run on the CPU (`cpu`), and that the round makes the photo's first blur map (`finding`: the Background rows say "Finding…"), as the
   * core last said (`viewport_busy`). The photo cell says "Rendering on the CPU" from it.
   */
  viewportBusy: ViewportBusy;
  /** The settings Copy settings took, or null before the first copy; it outlives a session and is never saved. */
  copied: CopiedSettings | null;
}

/** The state of an app with nothing open. */
export function initialState(): AppState {
  return {
    screen: "start",
    returnTo: "start",
    session: null,
    thumbnails: {},
    nodeThumbnails: {},
    settings: {},
    looks: null,
    lenses: null,
    stats: null,
    before: false,
    zoom: { kind: "fit" },
    maskView: { on: false, style: "overlay" },
    maskEye: null,
    showDepth: null,
    skinAreas: null,
    maskShown: null,
    applyScope: "this",
    collapsedLookLibraries: [],
    dropNotice: null,
    error: null,
    opening: false,
    openProgress: null,
    histogram: null,
    middleReadout: null,
    chromaScatter: null,
    warpPin: null,
    maskRow: null,
    brush: DEFAULT_BRUSH,
    brushTool: false,
    maskKinds: null,
    objectMode: DEFAULT_OBJECT_MODE,
    objectTool: false,
    objectPreparing: [],
    objectFinding: null,
    visionNothing: {},
    depthFailed: {},
    faces: {},
    faceSelected: 1,
    noFocusPoint: {},
    focusCentre: {},
    objectPill: null,
    canUndo: false,
    canRedo: false,
    preferences: null,
    cliLink: null,
    auto: null,
    consentFor: null,
    os: null,
    panel: "tools",
    exportOptions: defaultExportOptions(),
    exportScope: "this",
    exportRun: null,
    exportStopped: null,
    overlay: null,
    cubeImport: null,
    notice: null,
    noticeAction: null,
    viewportBackend: null,
    viewportFrame: null,
    exportBusy: false,
    viewportBusy: { busy: false, cpu: false, finding: false },
    copied: null,
  };
}

/** Fields of the Input whose resolved value is the stored value, so a patch can be shown at once (the others wait for the core's reply). */
const DIRECT_INPUT: ReadonlySet<string> = new Set(["pos", "light_source", "lens", "rolloff", "decode", "crop", "orientation", "white_pick"]);

/** The same for the Output: all of its fields. */
const DIRECT_OUTPUT: ReadonlySet<string> = new Set(["pos", "look", "look_intensity", "output_sharpening"]);

/** The same for Main's own fields (its grade and its masks are below). */
const DIRECT_MAIN: ReadonlySet<string> = new Set(["pos", "name", "bypass", "trim"]);

/** The grade fields of Main whose resolved value is the stored value. */
const DIRECT_GRADE: ReadonlySet<string> = new Set(["brightness", "white", "color_warp", "curve"]);

/** The settings that are Auto or a value from −100 to 100 (`docs/19` section 5), each resolved against the estimate of the same name. */
type PercentField = "contrast" | "saturation" | "highlights" | "shadows" | "black";
const PERCENT_KEYS: ReadonlySet<string> = new Set(["contrast", "saturation", "highlights", "shadows", "black"]);

/** Fields of a node patch that are partial objects: they merge into the field they name, as the core merges them. */
const PARTS: ReadonlySet<string> = new Set(["lens", "rolloff", "decode", "trim"]);

/** Puts the fields of `patch` that are in `direct` into the node's stored and resolved fields. */
function directFields(stored: object, resolved: object, patch: object, direct: ReadonlySet<string>): { stored: object; resolved: object } {
  const s: Record<string, unknown> = { ...stored };
  const r: Record<string, unknown> = { ...resolved };
  for (const [key, value] of Object.entries(patch)) {
    if (value === undefined || !direct.has(key)) continue;
    const part = PARTS.has(key) && value !== null && typeof value === "object";
    s[key] = part ? { ...(s[key] as object), ...value } : value;
    r[key] = part ? { ...(r[key] as object), ...value } : value;
  }
  return { stored: s, resolved: r };
}

/**
 * The mirrored settings with a patch applied to the fields that need no resolving (`docs/23` section 3.7). A node patch is read node by
 * node: the Input's light preset, lens, range, switches, crop, turn and white pick; Main's brightness, white, colour warp, curve, name,
 * bypass, trim and masks (a row patch, the node's Invert and the list of rows: the Mask tab shows its edit at once); the Output's look, intensity and sharpening. Contrast, saturation, highlights, shadows, black and the shadows
 * and highlights wheels of Main are Auto or a value (`docs/19` section 5) and need no resolving either: Auto is the photo's Auto value,
 * which the mirror already has (`estimates`). What waits for the core's reply: the Input's exposure, white balance, sharpening and noise
 * reduction (their Auto resolves in the core), and a patch that adds or removes a node or gives links. Another corrector's place, name, bypass, trim and grade are direct, as Main's are (the node view's cards and panel): it has no
 * Auto, so nothing in its grade waits for the core.
 */
export function applyDirect(rs: ResolvedSettings, patch: SettingsPatch): ResolvedSettings {
  let settings = rs.settings;
  let values = rs.values;
  let auto = rs.auto;
  const change = (id: NodeId, edit: (item: NodeItem) => NodeItem): void => {
    settings = { nodes: { ...settings.nodes, items: settings.nodes.items.map((item) => (item.id === id ? edit(item) : item)) } };
  };
  for (const [key, node] of Object.entries(patch.nodes ?? {})) {
    const id = nodeIdOfKey(key);
    if (id === INPUT_ID) {
      const { stored, resolved } = directFields(inputOf(settings), values.input, node, DIRECT_INPUT);
      change(id, () => stored as InputItem);
      values = { ...values, input: resolved as ResolvedInput };
      if (node.light_source !== undefined) auto = { ...auto, input: { ...auto.input, light_source: node.light_source === "auto" } };
    } else if (id === OUTPUT_ID) {
      const { stored, resolved } = directFields(outputOf(settings), values.output, node, DIRECT_OUTPUT);
      change(id, () => stored as OutputItem);
      values = { ...values, output: resolved as ResolvedOutput };
    } else if (id === MAIN_ID) {
      const direct = directFields(mainOf(settings), values.main, node, DIRECT_MAIN);
      const { stored, resolved } = withMasks(direct.stored as CorrectorItem, direct.resolved as ResolvedCorrector, node.masks);
      const grade = node.grade;
      const next = grade ? directGrade(stored as CorrectorItem, resolved as ResolvedCorrector, grade, rs.estimates.main, auto.main) : null;
      change(id, () => (next?.stored ?? stored) as CorrectorItem);
      values = { ...values, main: (next?.resolved ?? resolved) as ResolvedCorrector };
      if (next) auto = { ...auto, main: next.auto };
    } else if (typeof id === "number" && nodeOf(settings, id)?.kind === "corrector") {
      // Another corrector (the node view's cards and panel): the fields that need no resolving, so that a card dropped where it was
      // moved, a name typed, a bypass pressed and a slider of its panel show at once. Its grade is plain numbers, with no Auto to
      // resolve (`docs/23` §3.2: Auto in a corrector other than Main is a manual 0), so it is direct too, and so are its masks (a row
      // patch is a partial of the row). A node added or removed waits for the core's reply.
      const have = values.nodes.find((n) => n.id === id);
      const direct = directFields(nodeOf(settings, id) as CorrectorItem, have ?? {}, node, DIRECT_MAIN);
      const { stored, resolved } = withMasks(direct.stored as CorrectorItem, direct.resolved as ResolvedCorrector, node.masks);
      const next = node.grade && have ? directGrade(stored as CorrectorItem, resolved as ResolvedCorrector, withoutAuto(node.grade), NO_ESTIMATES, NO_FLAGS) : null;
      change(id, () => (next?.stored ?? stored) as CorrectorItem);
      if (have) values = { ...values, nodes: values.nodes.map((n) => (n.id === id ? ((next?.resolved ?? resolved) as ResolvedCorrector) : n)) };
    } else if (typeof id === "number" && (nodeOf(settings, id)?.kind === "film" || nodeOf(settings, id)?.kind === "face_refinement")) {
      // A Film node (`docs/24`) or a Face Refinement node (`docs/27`): its fields are plain numbers and ids, so the page folds the patch in at once, as the core will.
      settings = applyPatch(settings, { nodes: { [key]: node } });
    }
  }
  return settings === rs.settings && values === rs.values && auto === rs.auto ? rs : { ...rs, settings, values, auto };
}

/**
 * A corrector's stored and resolved masks with a masks patch applied (the node's Invert, the list of rows, row patches): a row patch is a
 * partial of the row's own fields, so the mirror shows it at once, as it shows a slider (the core's reply, held to the field's range,
 * replaces it). Nothing changes without a masks part.
 */
function withMasks(stored: CorrectorItem, resolved: ResolvedCorrector, masks: MasksPatch | undefined): { stored: object; resolved: object } {
  if (!masks) return { stored, resolved };
  return { stored: { ...stored, masks: applyMasks(stored.masks, masks) }, resolved: { ...resolved, masks: applyMasks(resolved.masks, masks) } };
}

/** What Auto resolves to in a corrector other than Main: nothing, a manual 0 (`docs/23` §3.2). */
const NO_ESTIMATES: MainEstimates = { contrast: 0, saturation: 0, highlights: 0, shadows: 0, black: 0, balance_shadows: [0, 0], balance_highlights: [0, 0] };
const NO_FLAGS: MainAutoFlags = { contrast: false, highlights: false, shadows: false, black: false, balance_shadows: false, balance_highlights: false, saturation: false };

/** A grade patch for a corrector other than Main, as the core reads it: every Auto is a manual 0 (and an Auto wheel a manual point at the centre). */
function withoutAuto(patch: GradePatch): GradePatch {
  const out: Record<string, unknown> = { ...patch };
  for (const key of PERCENT_KEYS) if ((out[key] as PercentSetting | undefined)?.kind === "auto") out[key] = { kind: "manual", value: 0 };
  const balance = patch.color_balance;
  if (balance) {
    const wheel = (w: BalanceSetting | undefined): BalanceSetting | undefined => (w?.kind === "auto" ? { kind: "manual", point: [0, 0] } : w);
    out.color_balance = { ...balance, ...(balance.shadows ? { shadows: wheel(balance.shadows) } : {}), ...(balance.highlights ? { highlights: wheel(balance.highlights) } : {}) };
  }
  return out as GradePatch;
}

/** Main's grade patch applied to its stored and resolved grade, with the Auto flags that follow. */
function directGrade(
  stored: CorrectorItem,
  resolved: ResolvedCorrector,
  patch: GradePatch,
  estimates: MainEstimates,
  flags: MainAutoFlags,
): { stored: CorrectorItem; resolved: ResolvedCorrector; auto: MainAutoFlags } {
  const grade: Record<string, unknown> = { ...stored.grade };
  const values: Record<string, unknown> = { ...resolved.grade };
  let auto = flags;
  for (const [key, value] of Object.entries(patch)) {
    if (value === undefined) continue;
    if (PERCENT_KEYS.has(key)) {
      const field = key as PercentField;
      const setting = value as PercentSetting;
      grade[field] = setting;
      values[field] = setting.kind === "manual" ? setting.value : estimates[field];
      auto = { ...auto, [field]: setting.kind === "auto" };
    } else if (key === "color_balance") {
      const cb = { ...stored.grade.color_balance, ...(value as Partial<ColorBalance>) };
      const point = (w: BalanceSetting, est: ColorBalancePoint): ColorBalancePoint => (w.kind === "manual" ? w.point : est);
      grade.color_balance = cb;
      values.color_balance = {
        shadows: point(cb.shadows, estimates.balance_shadows),
        midtones: cb.midtones,
        highlights: point(cb.highlights, estimates.balance_highlights),
      };
      auto = { ...auto, balance_shadows: cb.shadows.kind === "auto", balance_highlights: cb.highlights.kind === "auto" };
    } else if (DIRECT_GRADE.has(key)) {
      const part = key === "curve" ? { ...stored.grade.curve, ...(value as Partial<Curves>) } : value;
      grade[key] = part;
      values[key] = part;
    }
  }
  return {
    stored: { ...stored, grade: grade as unknown as NodeGrade },
    resolved: { ...resolved, grade: values as unknown as GradeValues },
    auto,
  };
}

interface Queued {
  patch: SettingsPatch;
  /** Set on a final patch: the settings before its gesture, for the undo entry. */
  record: { base: DevelopSettings } | null;
  done: Array<() => void>;
}

interface Sender {
  busy: boolean;
  queue: Queued[];
  idle: Array<() => void>;
}

/** A frame of the fallback viewport smaller than this, in device pixels on either side, is not a picture of the cell. */
const MIN_FRAME_PX = 16;

/** Milliseconds the filmstrip waits after its range changed before it tells the shell which photos it shows (`thumbnailsInView`). */
export const THUMBNAILS_IN_VIEW_DELAY_MS = 120;

/** Milliseconds a replaced look thumbnail's URL is kept, so a tile never shows a broken picture. */
const URL_GRACE_MS = 3000;

/**
 * The longest a shown histogram goes without being asked for again while frames keep coming: five a second during a drag, and one
 * after the last frame. A throttle, not a debounce: the plot follows the drag, a little behind.
 */
export const HISTOGRAM_THROTTLE_MS = 200;

/**
 * How many look tiles the core is asked for at once. It has one foreground worker and takes no job off it, so every tile asked for
 * at once is a job the next frame of the viewport may wait behind: with two, it waits behind one.
 */
export const LOOK_THUMBNAIL_CONCURRENCY = 2;

/** A look tile's request: the address it will have (or has), so a tile that fails to load it can be recognised. */
interface LookThumb {
  promise: Promise<string>;
  url: string | null;
}

/** The photos of the last removal, as far as the page keeps them (the core keeps the rest: `Session::restore_removed`). */
interface PendingRemoval {
  /** Their undo histories (⌘Z steps of their settings), put back with them. */
  histories: Map<PhotoId, PhotoHistory>;
  /**
   * How many undo steps the photos that stayed had when the photos left. While the histories still hold that many, nothing was done
   * since and ⌘Z takes the removal back; after a new change ⌘Z undoes the change first, as in any undo stack.
   */
  depth: number;
}

/** The store. Create one per window with the `Api` to use. */
export class AppStore {
  private state: AppState = initialState();
  private readonly listeners = new Set<() => void>();
  private stops: Unsubscribe[] = [];
  private batch: Array<(s: AppState) => AppState> = [];
  private batchScheduled = false;
  private readonly pendingStatus = new Map<PhotoId, PhotoStatus>();
  private readonly histories = new Map<PhotoId, PhotoHistory>();
  private readonly gestures = new Map<PhotoId, { base: DevelopSettings }>();
  private readonly senders = new Map<PhotoId, Sender>();
  private readonly settingsRequests = new Map<PhotoId, Promise<void>>();
  /**
   * Counts the times the core changed photos behind the mirror's back ("Apply to" copies, pastes). A read of a photo's settings that
   * began before the latest one may carry the old settings, so it is asked again instead of being kept.
   */
  private bulkEpoch = 0;
  /**
   * How many times the store has written each photo's settings or sent a patch for it (a patch made, its reply, settings read in): a
   * `getSettings` reply asked for before the latest of these may predate it, so {@link resyncSettings} and {@link refreshSettings} do not
   * show it (a slider being dragged would jump back).
   */
  private readonly writes = new Map<PhotoId, number>();
  /** The end of the "Apply to" copies asked for so far: the next one waits for it, so they reach the core in the order they were made. */
  private applyTail: Promise<void> = Promise.resolve();
  /** The number of the newest `viewport_frame` heard, whether it showed a picture or took it away; null before the first. */
  private newestFrame: number | null = null;
  /** Photos with a Pick white click on its way to the core: their settings are read when the click is answered. */
  private readonly picks = new Set<PhotoId>();
  /** Photos a paste is changing: the core's `settings_changed` for them is the paste's own news, which the store handles itself. */
  private readonly pasting = new Set<PhotoId>();
  /**
   * The last removal, which Undo takes back until the next removal or the end of the session: the removed photos' undo histories (the
   * core keeps their settings, picks and place). In memory only, like everything about a photo (hard rule 4).
   */
  private removal: PendingRemoval | null = null;
  /**
   * Photos being removed: from the moment {@link removePhotos} begins until the core has answered. A read of a leaving photo's settings
   * that the core's news started ({@link refreshSettings}) neither starts nor reports its failure, and shows nothing: the photo is gone.
   */
  private readonly leaving = new Set<PhotoId>();
  /** The state the native menu was last told, which is the one it starts with: the settings items off. */
  private menuState: MenuState = { copy_settings: false, paste_settings: false, paste_settings_to_picked: false };
  private readonly lookThumbs = new Map<PhotoId, { key: string; urls: Map<LookId, LookThumb> }>();
  /** Holds the look tiles' requests to {@link LOOK_THUMBNAIL_CONCURRENCY} at a time; what waits is dropped when its photo or settings change. */
  private readonly thumbLimiter = new RequestLimiter<{ photoId: PhotoId; key: string }>(LOOK_THUMBNAIL_CONCURRENCY);
  /** The Film node's stock tiles (`docs/24` section 4.4): by photo, the Film node and the settings up to it, each stock's picture. In memory only. */
  private readonly filmTiles = new Map<PhotoId, { key: string; urls: Map<string, LookThumb> }>();
  /** One stock tile at a time, in the order they are asked for (the list's order): the core makes each at the thumbnails' priority. */
  private readonly filmTileLimiter = new RequestLimiter<{ photoId: PhotoId; key: string }>(1);
  /** The Film node's list of stocks and papers, read once. */
  private filmCatalogueRead: Promise<FilmCatalogue> | null = null;
  private rect: ViewportRect | null = null;
  /** The corrector the node view has selected and shows (`setMaskNode`); null in the simple view, while Crop or Export shows and on the Input and the Output. */
  private maskNode: number | null = null;
  /** The mask view the core was last told about, with the photo it was told on (a new photo starts without one in the core). */
  private maskSent: { photo: PhotoId | null; view: MaskView | null } | null = { photo: null, view: null };
  /** `mask_kinds` has been asked for (or answered): the Add mask menu's list is read once. */
  private maskKindsAsked = false;
  /**
   * Wheel, pinch and drag input for the photo, collected and sent once per animation frame (`queueViewportInput`). The zoom mirror
   * follows once per frame too, when a zoom is closed.
   */
  private readonly viewportQueue = new ViewportInputQueue({
    schedule: (callback) => {
      if (typeof requestAnimationFrame === "function") requestAnimationFrame(callback);
      else setTimeout(callback, 16);
    },
    send: (event) => this.api.viewportInput(event),
    mirrorZoom: (factor) => this.set({ zoom: zoomAfter(this.state.zoom, factor, this.fitPercentExact()) }),
    onError: (error) => this.fail(error),
  });
  private histogramBusy = false;
  private histogramAgain = false;
  /** The node the histogram on show is of (`at`, `docs/01`): undefined is Main's, as the simple view reads it. */
  private histogramAt: NodeId | undefined = undefined;
  /** Who has the histogram on show (the Light and Curve tabs): while there is one, `render_stats` keeps it fresh. */
  private histogramWatchers = 0;
  /** Of those, who also show the middle readout (the Light tab). */
  private readoutWatchers = 0;
  private histogramTimer: ReturnType<typeof setTimeout> | null = null;
  private readoutBusy = false;
  private readoutAgain = false;
  private scatterBusy = false;
  private scatterAgain = false;
  /** The node whose warp the scatter on show is at the input of (`at`): undefined is Main's. */
  private scatterAt: NodeId | undefined = undefined;
  /** Who shows the colour warp's plane (the Colour tab): while there is one, `render_stats` keeps its scatter fresh. */
  private scatterWatchers = 0;
  private scatterTimer: ReturnType<typeof setTimeout> | null = null;
  /** Who has the node view on show (`watchNodeThumbnails`): while there is one, the store asks for the node thumbnails of the current photo. */
  private nodeThumbWatchers = 0;
  /** Photos whose node thumbnails are about to be asked for (one microtask collects the calls of a burst). */
  private readonly nodeThumbAsked = new Set<PhotoId>();
  /** Export events that arrived before `export_start` answered, by job. */
  private readonly earlyExport: ExportProgress[] = [];
  private starting = false;
  /** Runs a menu item that needs the webview (`menu_action`); set by `App`, which knows the screens. */
  menuHandler: ((action: MenuAction) => void) | null = null;
  /**
   * The View menu's Nodes item: the Photos screen, which owns the node view, sets it while it is on screen. It does what N does and
   * answers whether it did anything (not in Crop or the Export panel); with no Photos screen it is `null`.
   */
  nodesCommand: (() => boolean) | null = null;

  /**
   * `batchMs` is how long event-driven changes may wait to be drawn together (about two frames);
   * 0 applies them at the end of the current task, which tests use.
   */
  constructor(
    readonly api: Api,
    private readonly batchMs = 32,
  ) {}

  // --- Subscription -------------------------------------------------------------------------

  /** For `useSyncExternalStore`. */
  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  /** The current state. The same object until something changes. */
  readonly getState = (): AppState => this.state;

  private set(patch: Partial<AppState>): void {
    // The consent question (`docs/22` section 12.3) belongs to Preferences: whatever dialog takes its place (About from the native menu, the
    // close question, the .cube import, or no dialog at all) drops it, so the next ⌘, never opens Preferences with it already showing.
    const dropsConsent = patch.overlay !== undefined && patch.overlay !== "preferences" && patch.consentFor === undefined;
    this.state = { ...this.state, ...patch, ...(dropsConsent ? { consentFor: null } : {}) };
    this.notify();
  }

  /** Tells every listener that `state` is a new object. */
  private notify(): void {
    perfCount("store.notify");
    for (const listener of [...this.listeners]) listener();
    this.syncMenu();
  }

  /**
   * Tells the native menu which of the Edit menu's settings items are available: Copy Settings needs a developed current photo, Paste
   * Settings something copied as well, Paste Settings to Picked something copied and a picked photo. All three are off while a dialog
   * (Preferences, About, the close question, the .cube import) is open, as the Photos keys are. Sent when it changes only.
   */
  private syncMenu(): void {
    const { session, copied, overlay } = this.state;
    const current = session && session.current_index !== null ? session.photos[session.current_index] : undefined;
    const ready = current !== undefined && isReady(current) && overlay === null;
    const next: MenuState = {
      copy_settings: ready,
      paste_settings: ready && copied !== null,
      paste_settings_to_picked: overlay === null && copied !== null && (session?.picked.length ?? 0) > 0,
    };
    const was = this.menuState;
    if (next.copy_settings === was.copy_settings && next.paste_settings === was.paste_settings && next.paste_settings_to_picked === was.paste_settings_to_picked) return;
    this.menuState = next;
    void this.api.setMenuState(next).catch(() => {
      // The menu keeps its items as they were; the keys and the buttons work without it.
    });
  }

  /**
   * Queues a change that came from an event and applies the queue once per `batchMs`, so a burst
   * of events (500 photos developing, each with four stages and a thumbnail) draws a few times
   * instead of thousands.
   */
  private later(change: (s: AppState) => AppState): void {
    this.batch.push(change);
    if (this.batchScheduled) return;
    this.batchScheduled = true;
    const flush = (): void => {
      this.batchScheduled = false;
      const changes = this.batch;
      this.batch = [];
      let next = this.state;
      for (const apply of changes) next = apply(next);
      if (next === this.state) return;
      this.state = next;
      this.notify();
    };
    if (this.batchMs > 0) setTimeout(flush, this.batchMs);
    else queueMicrotask(flush);
  }

  // --- Connection ---------------------------------------------------------------------------

  /** Listens to the core's events and loads the looks and the session. Returns the disconnect function. */
  connect(): () => void {
    this.disconnect();
    const api = this.api;
    this.stops = [
      api.on("photo_progress", (e) => this.onProgress(e)),
      api.on("open_progress", (e) => this.onOpenProgress(e)),
      api.on("thumbnail_ready", (e) => this.onThumbnail(e)),
      api.on("node_thumbnails_ready", (e) => this.onNodeThumbnails(e)),
      api.on("render_stats", (stats) => {
        this.later((s) => (s.stats && share(s.stats, stats) === s.stats ? s : { ...s, stats }));
        this.scheduleHistogram();
        this.scheduleChromaScatter();
      }),
      api.on("looks_changed", () => {
        void this.refreshLooks();
        void this.refreshPreferences();
      }),
      // A provider was found, signed in or out, or a check ended. The cached read asks no program to check anything (`docs/22` section
      // 12.6), so it is safe at any time; only Preferences opening makes them check (`refreshAuto`, `agy`'s sign-in check contacts its
      // service, D7).
      api.on("auto_status_changed", () => void this.refreshAutoCached()),
      api.on("error", (error) => this.set({ error })),
      api.on("export_progress", (e) => this.onExportProgress(e)),
      api.on("menu_action", (e) => this.menuHandler?.(e.action)),
      api.on("viewport_frame", (frame) => this.onFrame(frame)),
      api.on("settings_changed", (e) => void this.onSettingsChanged(e.photo_id)),
      api.on("export_busy", (e) => this.set({ exportBusy: e.busy })),
      api.on("viewport_busy", (e) => this.onViewportBusy(e)),
      api.on("object_state", (e) => this.onObjectState(e)),
      api.on("vision_state", (e) => this.onVisionState(e)),
      api.on("depth_state", (e) => this.onDepthState(e)),
      api.on("face_state", (e) => this.onFaceState(e)),
      onGestureEnd(() => this.afterGesture()),
    ];
    void this.load();
    return () => this.disconnect();
  }

  private disconnect(): void {
    for (const stop of this.stops) stop();
    this.stops = [];
  }

  /** Loads the look library and any session the core already has (after a reload of the window). */
  async load(): Promise<void> {
    await Promise.all([
      this.refreshLooks(),
      this.refreshLenses(),
      this.refreshSession(),
      this.refreshPreferences(),
      this.refreshViewportInfo(),
      this.refreshExportBusy(),
    ]);
  }

  /** Asks which viewport backend draws the photo. A core that can't say leaves the screen drawing its own picture. */
  private async refreshViewportInfo(): Promise<void> {
    try {
      const info = await this.api.viewportInfo();
      this.set({ viewportBackend: info?.backend ?? null });
    } catch {
      this.set({ viewportBackend: null });
    }
  }

  /**
   * Asks the core whether an export is queued or running, which it knows after a reload of the window and for exports this window
   * did not start. Resolves to the answer, false when the core can't say.
   */
  async refreshExportBusy(): Promise<boolean> {
    try {
      const busy = await this.api.exportBusy();
      if (busy !== this.state.exportBusy) this.set({ exportBusy: busy });
      return busy;
    } catch {
      return this.state.exportBusy;
    }
  }

  private async refreshSession(): Promise<void> {
    try {
      const session = await this.api.sessionState();
      if (session.photos.length > 0) {
        this.adoptSession(session, true);
        await this.refreshThumbnails();
      }
    } catch (error) {
      this.fail(error);
    }
  }

  async refreshLooks(): Promise<void> {
    try {
      const looks = await this.api.looksList();
      this.set({ looks });
    } catch (error) {
      this.fail(error);
    }
  }

  /**
   * Asks for the lenses of the Lens card's list, once per run (`docs/21` §10). A core that has no lens data answers with none; one that
   * fails to answer leaves the card with none too, and the error bar says what went wrong.
   */
  async refreshLenses(): Promise<void> {
    try {
      this.set({ lenses: await this.api.lensList() });
    } catch (error) {
      this.set({ lenses: [] });
      this.fail(error);
    }
  }

  /** How many commands have failed so far; a caller compares it before and after a step to know that the step failed. */
  private failures = 0;

  private fail(error: unknown): void {
    this.failures += 1;
    this.set({ error: toAppError(error) });
  }

  /** Shows an error from a command the screens call themselves (a folder chooser, the Trash). */
  reportError(error: unknown): void {
    this.fail(error);
  }

  /** Hides the error bar. */
  dismissError(): void {
    this.set({ error: null });
  }

  // --- Events -------------------------------------------------------------------------------

  private onProgress(e: PhotoProgress): void {
    // Apple Auto's answer is not a development stage the status shows: the photo is ready before it (`docs/19` section 6).
    if (e.stage === "apple") {
      const pending = !e.done;
      this.later((s) => this.withApplePending(s, e.id, pending));
      return;
    }
    let status: PhotoStatus | null = null;
    if (e.error) status = { kind: "skipped", reason: e.error };
    else if (e.stage === "thumbnail" && e.done) status = { kind: "ready" };
    else if (!e.done) status = { kind: "developing", stage: e.stage };
    if (!status) return;
    const next = status;
    this.later((s) => this.withStatus(s, e.id, next));
  }

  /** The state with one photo's status changed; remembered when the photo isn't in the session yet. */
  private withStatus(s: AppState, id: PhotoId, status: PhotoStatus): AppState {
    const session = s.session;
    const index = session ? session.photos.findIndex((p) => p.id === id) : -1;
    if (!session || index < 0) {
      this.pendingStatus.set(id, status);
      return s;
    }
    // Copy on write: one copy of the array and a new object for the one photo that changed; the other 1,999 keep their identity, so the
    // lists that draw them compare equal and do not draw them again.
    const photos = session.photos.slice();
    photos[index] = { ...photos[index]!, status };
    if (status.kind === "ready" && photos[session.current_index ?? -1]?.id === id) {
      queueMicrotask(() => void this.ensureSettings(id));
    }
    return { ...s, session: { ...session, photos } };
  }

  /** The state with one photo's `apple_pending` changed. A photo not in the session yet is left alone: its summary will say. */
  private withApplePending(s: AppState, id: PhotoId, pending: boolean): AppState {
    const session = s.session;
    const index = session ? session.photos.findIndex((p) => p.id === id) : -1;
    const photo = session?.photos[index];
    if (!session || !photo || photo.apple_pending === pending) return s;
    const photos = session.photos.slice();
    photos[index] = { ...photo, apple_pending: pending };
    return { ...s, session: { ...session, photos } };
  }

  private onThumbnail(e: ThumbnailReady): void {
    this.later((s) => {
      const old = s.thumbnails[e.id];
      if (old === e.url) return s;
      // The shell's addresses are not ours to free; the mock's `blob:` ones are.
      if (old) releaseUrl(old);
      return { ...s, thumbnails: { ...s.thumbnails, [e.id]: e.url } };
    });
  }

  /** The pictures of some of a photo's nodes (only the ones that changed): kept per photo and node, the old address of a node let go. */
  private onNodeThumbnails(e: NodeThumbnailsReady): void {
    this.later((s) => {
      // A photo that is not in the session (it was removed while its pictures were made) has no node view; its pictures are let go.
      if (!s.session?.photos.some((p) => p.id === e.photo_id)) {
        for (const t of e.thumbnails) releaseUrl(t.url);
        return s;
      }
      const had = s.nodeThumbnails[e.photo_id] ?? {};
      const next: Record<number, string> = { ...had };
      let changed = false;
      for (const t of e.thumbnails) {
        const old = next[t.node];
        if (old === t.url) continue;
        // The shell's addresses are not ours to free; the mock's `blob:` ones are.
        if (old) releaseUrl(old);
        next[t.node] = t.url;
        changed = true;
      }
      return changed ? { ...s, nodeThumbnails: { ...s.nodeThumbnails, [e.photo_id]: next } } : s;
    });
  }

  /**
   * The node view is on show (the Photos screen calls this while the graph is up and calls the function it returns when it goes). While
   * somebody watches, the store asks the core for the current photo's node thumbnails (`node_thumbnails`): at once, when the photo on
   * screen changes, and whenever a change to its settings has landed (a released drag, a key, a typed value, a node added or removed,
   * an undo, a paste, the core's own news), never during a drag. Nothing is asked for while nobody watches.
   */
  watchNodeThumbnails(): () => void {
    this.nodeThumbWatchers += 1;
    this.wantNodeThumbnails(this.currentId());
    let watching = true;
    return () => {
      if (!watching) return;
      watching = false;
      this.nodeThumbWatchers -= 1;
    };
  }

  /**
   * A photo's settings changed or the photo was just shown: asks for its node thumbnails if the node view is on, the photo is the one
   * on screen, no drag is on and no patch is on its way (the reply of the last one comes here again). A burst of calls is one
   * request. The pictures of nodes the photo no longer has are let go first.
   */
  private wantNodeThumbnails(id: PhotoId | null): void {
    if (id === null || this.nodeThumbWatchers === 0 || this.nodeThumbAsked.has(id)) return;
    this.nodeThumbAsked.add(id);
    queueMicrotask(() => {
      this.nodeThumbAsked.delete(id);
      if (this.nodeThumbWatchers === 0 || id !== this.currentId()) return;
      if (this.gestures.has(id) || this.isSending(id)) return;
      const rs = this.state.settings[id];
      if (!rs) return;
      this.forgetGoneNodeThumbnails(id, rs.settings);
      void this.api.nodeThumbnails(id).catch(() => {
        // The cards keep the pictures they have; the next change asks again.
      });
    });
  }

  /** Lets go of the pictures of the nodes a photo no longer has (a node that is added again later gets its own). */
  private forgetGoneNodeThumbnails(id: PhotoId, settings: DevelopSettings): void {
    const kept = this.state.nodeThumbnails[id];
    if (!kept) return;
    const present = new Set(cardsOf(settings).map((n) => n.id as number));
    const gone = Object.keys(kept)
      .map(Number)
      .filter((node) => !present.has(node));
    if (gone.length === 0) return;
    const next: Record<number, string> = { ...kept };
    for (const node of gone) {
      releaseUrl(next[node] as string);
      delete next[node];
    }
    this.set({ nodeThumbnails: { ...this.state.nodeThumbnails, [id]: next } });
  }

  /** Lets go of the node thumbnails of these photos (the shell's addresses are not ours to free; the mock's `blob:` ones are). */
  private releaseNodeThumbnails(ids: Iterable<PhotoId>): Record<PhotoId, Readonly<Record<number, string>>> {
    const next = { ...this.state.nodeThumbnails };
    for (const id of ids) {
      for (const url of Object.values(next[id] ?? {})) releaseUrl(url);
      delete next[id];
    }
    return next;
  }

  /**
   * Asks the core for the thumbnails it already announced. It announces each one once (`thumbnail_ready`), so a webview that
   * reloaded would show an empty filmstrip; only the addresses come back. A thumbnail an event brought meanwhile is newer and stays.
   */
  async refreshThumbnails(): Promise<void> {
    try {
      const known = await this.api.thumbnails();
      if (known.length === 0) return;
      this.later((s) => {
        const fresh = known.filter((t) => !s.thumbnails[t.id]);
        return fresh.length === 0 ? s : { ...s, thumbnails: { ...s.thumbnails, ...Object.fromEntries(fresh.map((t) => [t.id, t.url])) } };
      });
    } catch {
      // The filmstrip waits for the next thumbnail the core announces.
    }
  }

  /** The photos the filmstrip draws, waiting to be sent: one ask goes out per pause in the scrolling. */
  private inView: PhotoId[] | null = null;
  private inViewTimer: ReturnType<typeof setTimeout> | null = null;

  /**
   * The filmstrip draws these photos (`docs/26` H-11). The shell keeps the thumbnails under a byte budget and lets go of the ones used
   * least recently, so a photo that was scrolled out of sight may have none when it comes back. The ask goes out once the scrolling
   * pauses; the shell counts the photos as used and makes the missing thumbnails again (`thumbnail_ready` brings them).
   */
  thumbnailsInView(ids: readonly PhotoId[]): void {
    this.inView = ids.slice();
    if (this.inViewTimer !== null) return;
    this.inViewTimer = setTimeout(() => {
      this.inViewTimer = null;
      const wanted = this.inView;
      this.inView = null;
      if (wanted && wanted.length > 0) void this.askThumbnailsInView(wanted);
    }, THUMBNAILS_IN_VIEW_DELAY_MS);
  }

  private async askThumbnailsInView(ids: PhotoId[]): Promise<void> {
    const sent = this.state.thumbnails;
    try {
      const held = new Map((await this.api.thumbnailsInView(ids)).map((t) => [t.id, t.url]));
      this.later((s) => {
        let next: Record<PhotoId, string> | null = null;
        for (const id of ids) {
          const was = sent[id];
          const now = s.thumbnails[id];
          // An address an event brought while the ask was out is newer than the answer, and stays.
          if (now !== was) continue;
          const url = held.get(id);
          if (url === now) continue;
          next ??= { ...s.thumbnails };
          // The shell let go of this photo's thumbnail: the old address would answer 404, so the strip shows no picture until the new
          // one is announced.
          if (url === undefined) delete next[id];
          else next[id] = url;
        }
        return next === null ? s : { ...s, thumbnails: next };
      });
    } catch {
      // The strip keeps what it has; the next pause in the scrolling asks again.
    }
  }

  /**
   * A frame of the fallback viewport, or (an empty address) the news that the picture is gone. Drawn at once, never batched. The shell
   * can draw a frame before the webview has said where the cell is (it uses a 1 × 1 cell then); a frame that small is no picture.
   */
  /** `viewport_busy`: kept as the core said it (a repeat changes nothing). */
  private onViewportBusy(e: ViewportBusy): void {
    const now = this.state.viewportBusy;
    if (now.busy === e.busy && now.cpu === e.cpu && now.finding === e.finding) return;
    this.set({ viewportBusy: { busy: e.busy, cpu: e.cpu, finding: e.finding } });
  }

  /** `object_state`: SAM's encoder started or stopped working for a photo. When it stops the core draws the photo again, and the nodes' thumbnails follow. */
  private onObjectState(e: ObjectState): void {
    const was = this.state.objectPreparing.includes(e.photo_id);
    if (was === e.preparing) return;
    this.set({ objectPreparing: e.preparing ? [...this.state.objectPreparing, e.photo_id] : this.state.objectPreparing.filter((id) => id !== e.photo_id) });
    if (!e.preparing) this.wantNodeThumbnails(e.photo_id);
  }

  /** `vision_state`: the Subject and People worker ended its work for a photo. The rows whose request found nothing say so; the core draws the photo again and the nodes' thumbnails follow. */
  private onVisionState(e: VisionState): void {
    const was = this.state.visionNothing[e.photo_id] ?? [];
    const same = was.length === e.nothing_found.length && was.every((k, i) => k === e.nothing_found[i]);
    if (!same) this.set({ visionNothing: { ...this.state.visionNothing, [e.photo_id]: e.nothing_found } });
    this.wantNodeThumbnails(e.photo_id);
  }

  /**
   * `depth_state`: the depth worker ended its work for a photo. `failed` is the sentence of the failure, or null when the map is ready. The core draws the photo
   * again, and the nodes' thumbnails follow.
   */
  private onDepthState(e: DepthState): void {
    const was = this.state.depthFailed[e.photo_id] ?? null;
    if (was !== e.failed) {
      const next = { ...this.state.depthFailed };
      if (e.failed === null) delete next[e.photo_id];
      else next[e.photo_id] = e.failed;
      this.set({ depthFailed: next });
    }
    this.wantNodeThumbnails(e.photo_id);
  }

  /** `face_state`: the core's face worker started or ended for a photo. The node's panel and the boxes on the photo follow; the thumbnails are asked again when it ends. */
  private onFaceState(e: FaceState): void {
    this.set({ faces: { ...this.state.faces, [e.photo_id]: e } });
    if (e.phase !== "finding") this.wantNodeThumbnails(e.photo_id);
  }

  /**
   * Asks the core where the faces of a photo are (`face_state` command) and keeps the answer, unless an event of the same worker has already said something
   * newer. The Face tab asks when it opens, so a photo whose worker ended before the page listened is not left on "Finding faces".
   */
  async refreshFaces(photoId: PhotoId): Promise<void> {
    try {
      const state = await this.api.faceState(photoId);
      const had = this.state.faces[photoId];
      if (!had || had.phase === "finding" || had.phase !== state.phase || had.faces.length !== state.faces.length) this.set({ faces: { ...this.state.faces, [photoId]: state } });
    } catch (error) {
      this.fail(error);
    }
  }

  /** Selects the face the Face tab is on (counted from 1). Nothing is sent. */
  selectFace(number: number): void {
    const next = Math.max(1, Math.round(number));
    if (next !== this.state.faceSelected) this.set({ faceSelected: next });
  }

  private onFrame(frame: ViewportFrame): void {
    // Frames can arrive out of order (the shell numbers a frame before it encodes it, and the news that the picture is gone is sent from
    // another thread): an older one never replaces a newer one, and never brings back a picture that was taken away.
    if (this.newestFrame !== null && frame.seq < this.newestFrame) return;
    this.newestFrame = frame.seq;
    const none = frame.url === "" || frame.width < MIN_FRAME_PX || frame.height < MIN_FRAME_PX;
    this.set({ viewportFrame: none ? null : frame });
  }

  /** The core changed a photo's settings or estimates by itself (an Apple Auto answer, a burst's exposure): ask for them again. */
  private async onSettingsChanged(id: PhotoId): Promise<void> {
    if (this.picks.has(id) || this.pasting.has(id) || this.leaving.has(id) || !this.state.settings[id]) return;
    await this.refreshSettings(id);
  }

  // --- Session ------------------------------------------------------------------------------

  private adoptSession(session: SessionState, restoring = false): void {
    let next: AppState = { ...this.state, session };
    for (const [id, status] of this.pendingStatus) next = this.withStatus(next, id, status);
    this.pendingStatus.clear();
    const photos = next.session?.photos ?? [];
    if (restoring && photos.some(isReady)) next = { ...next, screen: "photos" };
    this.state = next;
    this.notify();
    const current = this.currentPhoto();
    if (current && isReady(current)) void this.ensureSettings(current.id);
    this.syncHistoryFlags();
  }

  /**
   * Opens dropped or chosen paths. Anything that isn't an ARW file or a folder is ignored.
   *
   * In the app the paths are real and the core sorts them out (it opens a folder and keeps the ARW files in it, whatever the folder is
   * called) and answers "No ARW files in what you dropped." itself, which shows in the drop zone. In a browser the names stand in for
   * paths, so a name that looks like a file of another kind is left out here.
   *
   * With a session open this adds to it: the new photos go after the others, the current photo stays and the others are not touched.
   * On Photos the answer is said in a notice ("Added 3 photos. 2 were already open.", `openedNotice`), and the "No ARW files" sentence
   * goes there too, since only Start has a drop zone to show it in.
   */
  async openPaths(paths: string[]): Promise<void> {
    const addingToPhotos = this.addingToPhotos();
    const toOpen = isTauri() ? paths : filterArwPaths(paths).accepted;
    if (toOpen.length === 0) {
      this.sayNoArw(addingToPhotos);
      return;
    }
    this.set({ dropNotice: null, opening: true });
    try {
      const session = await this.api.openPaths(toOpen);
      this.adoptOpened(session);
      this.sayOpened(session, addingToPhotos);
    } catch (error) {
      const failure = toAppError(error);
      if (failure.scope === "open" && (failure.message === NO_ARW_NOTICE || isUnreadableMessage(failure.message))) this.sayDropProblem(failure.message, addingToPhotos);
      else this.fail(failure);
    } finally {
      this.endOpening();
    }
  }

  /** Asks for files or a folder with the system dialog. A cancelled dialog changes nothing. */
  async openDialog(kind: "files" | "folder"): Promise<void> {
    const addingToPhotos = this.addingToPhotos();
    this.set({ dropNotice: null, opening: true });
    try {
      const session = await this.api.openDialog(kind);
      if (session) {
        this.adoptOpened(session);
        this.sayOpened(session, addingToPhotos);
      }
    } catch (error) {
      const failure = toAppError(error);
      if (failure.scope === "open" && isUnreadableMessage(failure.message)) this.sayDropProblem(failure.message, addingToPhotos);
      else this.fail(failure);
    } finally {
      this.endOpening();
    }
  }

  /** The core has read `e.done` of `e.total` files of the open under way. Shown once the open has taken a moment. */
  private onOpenProgress(e: OpenProgress): void {
    if (!this.state.opening) return;
    this.openProgressLatest = e;
    if (this.state.openProgress !== null) {
      this.set({ openProgress: e });
      return;
    }
    this.openProgressTimer ??= setTimeout(() => {
      this.openProgressTimer = null;
      if (this.state.opening && this.openProgressLatest !== null) this.set({ openProgress: this.openProgressLatest });
    }, OPEN_PROGRESS_DELAY_MS);
  }

  /** An open ended, however it ended: the line "Reading … files…" goes. */
  private endOpening(): void {
    if (this.openProgressTimer !== null) clearTimeout(this.openProgressTimer);
    this.openProgressTimer = null;
    this.openProgressLatest = null;
    this.set({ opening: false, openProgress: null });
  }

  /** Stops the open that is reading files (the Cancel button). The files read so far stay; the open then ends with `opened.cancelled`. */
  async cancelOpen(): Promise<void> {
    try {
      await this.api.cancelOpen();
    } catch (error) {
      this.fail(error);
    }
  }

  /** Says what an open did, where the person is looking: "Stopped. 40 files were opened." after Cancel, "Added 3 photos." on Photos. */
  private sayOpened(session: SessionState, onPhotos: boolean): void {
    const opened = session.opened;
    if (!opened) return;
    if (opened.cancelled) this.sayDropProblem(stoppedNotice(opened.added), onPhotos);
    else if (onPhotos) this.showNotice(openedNotice(opened));
  }

  /**
   * Takes the shell's answer to an open. With a session already here the answer is merged into it (`mergeSession`) and not put in its
   * place: it is a snapshot from before the page handled the events and clicks that came meanwhile, and replacing the page's own state
   * with it would bring back a status that has since changed, a pick that was toggled or the photo that was current.
   */
  private adoptOpened(snapshot: SessionState): void {
    const old = this.state.session;
    // Files opened on the Start screen start a new session: the removal of the last one is not offered any more.
    if (!old) this.forgetRemoval();
    this.adoptSession(old ? mergeSession(old, snapshot) : snapshot);
  }

  /** The last removal can no longer be taken back (a new session has begun): its histories and the notice that offered Undo go. */
  private forgetRemoval(): void {
    this.removal = null;
    if (this.state.noticeAction !== null) this.set({ notice: null, noticeAction: null });
  }

  /** True while Photos is on screen: files opened now are added to the session, and Photos says what was added. */
  private addingToPhotos(): boolean {
    return this.state.screen === "photos" && this.state.session !== null;
  }

  /** "No ARW files in what you dropped.": in the drop zone on Start, in the notice bar where there is no drop zone. */
  private sayNoArw(onPhotos: boolean): void {
    this.sayDropProblem(NO_ARW_NOTICE, onPhotos);
  }

  /** What was wrong with a drop or a dialog's choice ("No ARW files…", "Yellowmoon can't read the folder …"), where the person is looking. */
  private sayDropProblem(message: string, onPhotos: boolean): void {
    if (onPhotos) this.set({ dropNotice: null, notice: message, noticeAction: null });
    else this.set({ dropNotice: message });
  }

  /** Goes to the Photos screen, on the first photo that is ready when the current one isn't. */
  async openPhotos(): Promise<void> {
    const session = this.state.session;
    if (!session) return;
    const current = this.currentPhoto();
    if (!current || !isReady(current)) {
      const first = session.photos.find(isReady);
      if (first) await this.setCurrent(first.id);
    }
    this.set({ screen: "photos" });
  }

  /** Forgets the session and returns to Start (⌘W, "Start over"). */
  async closeSession(): Promise<void> {
    this.viewportQueue.clear();
    try {
      await this.api.closeSession();
    } catch (error) {
      this.fail(error);
      return;
    }
    // What a removal kept goes with the session, and so does the notice that offered to take it back.
    this.forgetRemoval();
    this.endSessionView();
  }

  /**
   * The page forgets the session and shows Start: the thumbnails, settings and histories of its photos, the Look tab's tiles and
   * what depends on the photo on screen. Not the core: `closeSession` asks it to close first, and a session whose last photo was
   * removed is empty in the core already (and keeps what Undo needs).
   */
  private endSessionView(): void {
    for (const url of Object.values(this.state.thumbnails)) releaseUrl(url);
    for (const urls of Object.values(this.state.nodeThumbnails)) for (const url of Object.values(urls)) releaseUrl(url);
    this.nodeThumbAsked.clear();
    this.releaseLookThumbs();
    this.pendingStatus.clear();
    this.histories.clear();
    this.gestures.clear();
    this.senders.clear();
    this.writes.clear();
    this.settingsRequests.clear();
    this.batch = [];
    this.state = {
      ...initialState(),
      looks: this.state.looks,
      lenses: this.state.lenses,
      applyScope: this.state.applyScope,
      maskView: this.state.maskView,
      collapsedLookLibraries: this.state.collapsedLookLibraries,
      stats: null,
      // What belongs to the app rather than the session stays: the preferences, the export in flight, the dialogs.
      preferences: this.state.preferences,
      cliLink: this.state.cliLink,
      auto: this.state.auto,
      os: this.state.os,
      exportOptions: this.state.exportOptions,
      // A run still writing files outlives the session; a finished one has been seen.
      exportRun: this.state.exportRun?.phase === "running" ? this.state.exportRun : null,
      overlay: this.state.overlay === "confirm-close" ? null : this.state.overlay,
      cubeImport: this.state.cubeImport,
      notice: this.state.notice,
      noticeAction: this.state.noticeAction,
      viewportBackend: this.state.viewportBackend,
      exportBusy: this.state.exportBusy,
      // What was copied is for pasting in another folder too; the id of its photo belongs to the closed session.
      copied: this.state.copied && { ...this.state.copied, sourceId: null },
    };
    this.notify();
    // A fresh session starts at Fit with the camera view off and no mask view (the page asks again for the one the node view has).
    this.maskSent = { photo: null, view: null };
    void this.api.viewportSetMode({ kind: "fit" }, null).catch(() => {});
    void this.api.setBefore(false).catch(() => {});
  }

  /** The photo on screen, or null. */
  currentPhoto(): PhotoSummary | null {
    const session = this.state.session;
    if (!session || session.current_index === null) return null;
    return session.photos[session.current_index] ?? null;
  }

  private currentId(): PhotoId | null {
    return this.currentPhoto()?.id ?? null;
  }

  async setCurrent(id: PhotoId): Promise<void> {
    const session = this.state.session;
    if (!session) return;
    const index = session.photos.findIndex((p) => p.id === id);
    if (index < 0 || index === session.current_index) return;
    // A new photo opens at Fit with the camera view off, in the core and here. The zoom figures of the last frame (`render_stats`)
    // belong to the photo that is left: a portrait after a landscape has another Fit zoom, so they are dropped until the first
    // frame of the new photo reports its own (the status text stays, and the screen estimates the Fit zoom meanwhile).
    const stats = this.state.stats && { ...this.state.stats, zoom_percent: null, fit_percent: null };
    // Wheel and drag input that has not been sent was meant for the photo that is left, and so were the tiles not yet asked for.
    this.viewportQueue.clear();
    this.thumbLimiter.supersede((t) => t.photoId !== id);
    this.filmTileLimiter.supersede((t) => t.photoId !== id);
    this.set({
      session: { ...session, current_index: index },
      histogram: null,
      middleReadout: null,
      chromaScatter: null,
      warpPin: null,
      maskRow: null,
      maskEye: null,
      showDepth: null,
      skinAreas: null,
      brushTool: false,
      objectTool: false,
      objectPill: null,
      zoom: { kind: "fit" },
      before: false,
      stats,
    });
    this.syncHistoryFlags();
    if (isReady(session.photos[index] as PhotoSummary)) {
      void this.ensureSettings(id);
      this.wantNodeThumbnails(id);
    }
    try {
      await this.api.setCurrent(id);
      // The core has dropped the last photo's mask view with the photo; whatever raced it is not known, so tell it again.
      this.maskSent = { photo: id, view: null };
      this.syncMaskView();
    } catch (error) {
      this.fail(error);
    }
  }

  /** Moves to the previous (-1) or next (+1) photo, skipping nothing: a photo that can't open is still shown. */
  step(direction: -1 | 1): Promise<void> {
    const session = this.state.session;
    if (!session || session.current_index === null) return Promise.resolve();
    const target = session.photos[session.current_index + direction];
    return target ? this.setCurrent(target.id) : Promise.resolve();
  }

  async setPicked(id: PhotoId, picked: boolean): Promise<void> {
    const session = this.state.session;
    const photo = session?.photos.find((p) => p.id === id);
    if (!session || !photo || photo.picked === picked) return;
    // Only a photo that can be exported can be picked.
    if (picked && !isReady(photo)) return;
    this.set({ session: this.withPicked(session, id, picked) });
    try {
      await this.api.setPicked(id, picked);
    } catch (error) {
      this.fail(error);
      void this.refreshSession();
    }
  }

  private withPicked(session: SessionState, id: PhotoId, picked: boolean): SessionState {
    const photos = session.photos.map((p) => (p.id === id ? { ...p, picked } : p));
    return { ...session, photos, picked: photos.filter((p) => p.picked).map((p) => p.id) };
  }

  /** P or ⌘-click: picks the photo, or unpicks it. */
  togglePick(id: PhotoId | null = this.currentId()): Promise<void> {
    const photo = this.state.session?.photos.find((p) => p.id === id);
    return photo ? this.setPicked(photo.id, !photo.picked) : Promise.resolve();
  }

  /** ⌘A: picks every photo that is ready. */
  async pickAll(): Promise<void> {
    const photos = this.state.session?.photos ?? [];
    await Promise.all(photos.filter((p) => isReady(p) && !p.picked).map((p) => this.setPicked(p.id, true)));
  }

  /** Edit › Clear Picks. */
  async clearPicks(): Promise<void> {
    const photos = this.state.session?.photos ?? [];
    await Promise.all(photos.filter((p) => p.picked).map((p) => this.setPicked(p.id, false)));
  }

  // --- Removing photos ----------------------------------------------------------------------

  /**
   * Remove from filmstrip (`docs/11` §Removing photos): takes photos out of the session. The files on disk are not touched (hard
   * rule 5); the photos' edits go with them, because nothing is saved per photo (hard rule 4), and are kept in memory for Undo until
   * the next removal or the end of the session. The photo on screen hands over to the next one (the previous one when the last was
   * removed); with no photo left the Start screen shows. Says what happened, with an Undo button ({@link undoRemoval}). Ids that are
   * not in the session are ignored.
   */
  async removePhotos(ids: readonly PhotoId[]): Promise<void> {
    const open = this.state.session;
    if (!open) return;
    const present = new Set(open.photos.map((p) => p.id));
    const wanted = [...new Set(ids)].filter((id) => present.has(id));
    if (wanted.length === 0) return;
    // From now on the core's news about these photos is not read again (an Apple Auto answer that arrives now is for a photo that is
    // leaving), and a read that is already on its way shows nothing.
    for (const id of wanted) this.leaving.add(id);
    try {
      await this.takeOut(wanted);
    } finally {
      for (const id of wanted) this.leaving.delete(id);
    }
    // A removal the core refused leaves the photos here, and news of theirs that was passed over meanwhile: read their settings again.
    for (const id of wanted) if (this.state.settings[id]) void this.refreshSettings(id);
  }

  /** The body of {@link removePhotos}, for the photos that are in the session. */
  private async takeOut(wanted: PhotoId[]): Promise<void> {
    const asked = new Set(wanted);
    // Patches and the first read of the settings still on their way to these photos land first: none may reach a photo that has left
    // (it would come back as an error from the core, and as a message on screen).
    await Promise.all(wanted.map((id) => Promise.all([this.whenIdle(id), this.settingsRequests.get(id)])));
    const before = this.state.session;
    if (!before) return;
    const names = before.photos.filter((p) => asked.has(p.id)).map((p) => photoName(p.file_name));
    const wasCurrent = this.currentId();
    // Wheel and drag input that has not been sent was meant for the photo that leaves.
    if (wasCurrent !== null && asked.has(wasCurrent)) this.viewportQueue.clear();
    let change: SessionChange;
    try {
      change = await this.api.removePhotos(wanted);
    } catch (error) {
      this.fail(error);
      return;
    }
    const gone = new Set(change.photo_ids);
    if (gone.size === 0) return;
    const session = this.state.session;
    if (!session) return;

    // Their undo histories are kept with them; the rest of what the page knows about them goes now.
    const histories = new Map<PhotoId, PhotoHistory>();
    for (const id of gone) {
      const history = this.histories.get(id);
      if (history) histories.set(id, history);
      this.histories.delete(id);
      this.gestures.delete(id);
      this.senders.delete(id);
      this.settingsRequests.delete(id);
      this.pendingStatus.delete(id);
    }
    this.forgetLookThumbsOf(gone);
    const thumbnails = { ...this.state.thumbnails };
    for (const id of gone) {
      const url = thumbnails[id];
      if (url !== undefined) releaseUrl(url);
      delete thumbnails[id];
    }
    const settings = { ...this.state.settings };
    for (const id of gone) delete settings[id];
    const nodeThumbnails = this.releaseNodeThumbnails(gone);
    this.removal = { histories, depth: this.undoDepth() };

    // The photo on screen stays unless it left; then it is the shell's choice, the next photo or the previous one.
    const shell = change.session.current_index === null ? null : (change.session.photos[change.session.current_index]?.id ?? null);
    const stays = wasCurrent !== null && !gone.has(wasCurrent);
    const current = stays ? wasCurrent : shell;
    const next = withoutPhotos(session, gone, current);
    const notice = { notice: removedNotice(names), noticeAction: "undo-removal" as const };
    if (next.photos.length === 0) {
      this.endSessionView();
      this.set(notice);
      return;
    }
    this.set({ session: next, thumbnails, nodeThumbnails, settings, ...notice, ...(stays ? {} : this.currentChanged()) });
    this.syncHistoryFlags();
    const shown = next.photos[next.current_index ?? 0];
    if (!stays && shown && isReady(shown)) {
      void this.ensureSettings(shown.id);
      this.wantNodeThumbnails(shown.id);
    }
  }

  /** Delete or Backspace: removes the photo on screen. */
  removeCurrentPhoto(): Promise<void> {
    const id = this.currentId();
    return id === null ? Promise.resolve() : this.removePhotos([id]);
  }

  /** "Remove picked photos" in the filmstrip's menu. */
  removePickedPhotos(): Promise<void> {
    return this.removePhotos((this.state.session?.photos ?? []).filter((p) => p.picked).map((p) => p.id));
  }

  /**
   * Undo of the last removal: the photos come back where they were in the filmstrip, with their settings, picks and undo steps; the
   * photo that was on screen is on screen again unless the person has moved on since. Used by the notice's Undo button, and by ⌘Z while
   * nothing was changed after the removal ({@link undo}). A photo whose file was added again meanwhile stays out. Nothing happens
   * when there is nothing to put back.
   */
  async undoRemoval(): Promise<void> {
    const pending = this.removal;
    if (!pending) return;
    let change: SessionChange;
    try {
      change = await this.api.restorePhotos();
    } catch (error) {
      this.fail(error);
      return;
    }
    this.removal = null;
    const back = new Set(change.photo_ids);
    if (back.size === 0) {
      this.set({ notice: "There is nothing to put back.", noticeAction: null });
      this.syncHistoryFlags();
      return;
    }
    for (const [id, history] of pending.histories) if (back.has(id)) this.histories.set(id, history);
    const names = change.session.photos.filter((p) => back.has(p.id)).map((p) => photoName(p.file_name));
    const notice = { notice: restoredNotice(names, change.already_open), noticeAction: null };
    const old = this.state.session;
    if (!old) {
      // The last photo had been removed: the session starts again from what the core has.
      this.adoptSession(change.session);
      this.set({ screen: "photos", ...notice });
      return;
    }
    // The photo on screen stays, unless the core put the person back on the photo the removal had taken away.
    const wasCurrent = this.currentId();
    const shell = change.session.current_index === null ? null : (change.session.photos[change.session.current_index]?.id ?? null);
    const current = shell !== null && back.has(shell) ? shell : wasCurrent;
    const session = withRestoredPhotos(old, change.session, back, current);
    const moved = current !== wasCurrent;
    this.set({ session, ...notice, ...(moved ? this.currentChanged() : {}) });
    this.syncHistoryFlags();
    const shown = session.photos[session.current_index ?? 0];
    if (moved && shown && isReady(shown)) void this.ensureSettings(shown.id);
  }

  /** True while a removal can be taken back: the notice's Undo button, and ⌘Z on the Start screen after the last photo left. */
  hasRemovalToUndo(): boolean {
    return this.removal !== null;
  }

  /**
   * The state that depends on the photo on screen, reset because it is another one now (as {@link setCurrent} does): Fit, the camera
   * view off, no histogram, readout or scatter, no selected colour-warp pin, and the zoom figures of the last frame dropped (they
   * belong to the photo that left). The core has done the same for its viewport.
   */
  private currentChanged(): Partial<AppState> {
    this.viewportQueue.clear();
    this.thumbLimiter.supersede(() => true);
    this.filmTileLimiter.supersede(() => true);
    return {
      histogram: null,
      middleReadout: null,
      chromaScatter: null,
      warpPin: null,
      maskRow: null,
      maskEye: null,
      showDepth: null,
      skinAreas: null,
      brushTool: false,
      objectTool: false,
      objectPill: null,
      zoom: { kind: "fit" },
      before: false,
      stats: this.state.stats && { ...this.state.stats, zoom_percent: null, fit_percent: null },
    };
  }

  /** The Look tab's tiles of photos that left are let go (the shell's addresses are not ours to free; the mock's `blob:` ones are). */
  private forgetLookThumbsOf(ids: ReadonlySet<PhotoId>): void {
    this.thumbLimiter.supersede((t) => ids.has(t.photoId));
    this.filmTileLimiter.supersede((t) => ids.has(t.photoId));
    for (const id of ids) {
      const tiles = this.filmTiles.get(id);
      if (tiles) {
        for (const thumb of tiles.urls.values()) void thumb.promise.then(releaseUrl, () => {});
        this.filmTiles.delete(id);
      }
      const entry = this.lookThumbs.get(id);
      if (!entry) continue;
      for (const thumb of entry.urls.values()) void thumb.promise.then(releaseUrl, () => {});
      this.lookThumbs.delete(id);
    }
  }

  /** The undo steps of all photos together: what ⌘Z would take back, photo by photo. */
  private undoDepth(): number {
    let depth = 0;
    for (const history of this.histories.values()) depth += history.undoDepth;
    return depth;
  }

  /** The removal is the last thing the person did that ⌘Z can take back: nothing was changed since (or what was has been undone). */
  private removalIsLatest(): boolean {
    return this.removal !== null && this.removal.depth === this.undoDepth();
  }

  // --- Navigation ---------------------------------------------------------------------------

  go(screen: Screen): void {
    if (screen === this.state.screen) return;
    this.set({ screen, returnTo: screen === "import-looks" ? this.state.screen : this.state.returnTo });
  }

  /** Back from a detour (the import wizard) to where it started. */
  back(): void {
    const target = this.state.returnTo;
    this.set({ screen: target === "import-looks" ? "start" : target });
  }

  setApplyScope(scope: ApplyScope): void {
    this.set({ applyScope: scope });
  }

  /** Folds a section of the Look tab away, or opens it again. Remembered until the app quits, never saved. */
  toggleLookLibrary(library: LookLibrary): void {
    const collapsed = this.state.collapsedLookLibraries;
    this.set({ collapsedLookLibraries: collapsed.includes(library) ? collapsed.filter((l) => l !== library) : [...collapsed, library] });
  }

  // --- Preferences and dialogs ----------------------------------------------------------------

  /** Loads the preferences; the Export panel starts from the remembered defaults. */
  async refreshPreferences(): Promise<void> {
    try {
      const preferences = await this.api.preferencesGet();
      const untouched = sameOptions(this.state.exportOptions, defaultExportOptions()) || this.state.preferences === null;
      this.set({ preferences, ...(untouched ? { exportOptions: preferences.export_defaults } : {}) });
    } catch (error) {
      this.fail(error);
    }
  }

  /**
   * Changes a preference. The screen shows the change at once; the core's answer replaces it, and a refusal
   * puts the old values back and shows the reason.
   */
  async setPreferences(patch: PreferencesPatch): Promise<void> {
    const before = this.state.preferences;
    if (before) this.set({ preferences: { ...before, ...patch } });
    try {
      const preferences = await this.api.preferencesSet(patch);
      this.set({ preferences });
      // The provider acts, or stops, with the choice and the consent, and every photo is asked again with new instructions; the photos' own
      // news comes as `settings_changed`. The cached read: typing instructions saves a preference every few seconds, and none of those
      // may make a program check its sign-in (`docs/22` D7).
      if (patch.auto_provider !== undefined || patch.auto_consent !== undefined || patch.apple_auto_instructions !== undefined) {
        void this.refreshAutoCached();
      }
    } catch (error) {
      if (before) this.set({ preferences: before });
      this.fail(error);
    }
  }

  /**
   * Asks which providers this computer has and which one grades, and makes the core check every one of them again (Apple Intelligence may
   * have been switched on, a program installed or signed in; `agy`'s sign-in check contacts its service). Only Preferences opening calls
   * it (`docs/22` section 3.3 and D7); everything else reads {@link AppStore.refreshAutoCached}. The core never waits for a program, so
   * this answers at once with what it knows; `auto_status_changed` says when that moved.
   */
  async refreshAuto(): Promise<void> {
    try {
      this.set({ auto: await this.api.autoStatus() });
    } catch (error) {
      this.fail(error);
    }
  }

  /**
   * The same status as each provider last said it, and no program is asked to check anything (`auto_status_cached`, `docs/22` section
   * 12.6): what the store reads after a change of the choice, the consent or the instructions and on `auto_status_changed`.
   */
  async refreshAutoCached(): Promise<void> {
    try {
      this.set({ auto: await this.api.autoStatusCached() });
    } catch (error) {
      this.fail(error);
    }
  }

  /** Asks which operating system the app runs on, for the sentences that differ on Windows. */
  async refreshOs(): Promise<void> {
    try {
      this.set({ os: (await this.api.appInfo()).os });
    } catch (error) {
      this.fail(error);
    }
  }

  /**
   * The dropdown's choice of a provider (`docs/22` section 12.1). Default Auto, Apple Intelligence and a program the person
   * already agreed to are saved at once; a program they have not agreed to opens the consent dialog and changes nothing until they say yes
   * ({@link AppStore.answerConsent}).
   */
  async chooseProvider(provider: AutoProvider): Promise<void> {
    if ((provider === "claude" || provider === "agy") && this.state.preferences?.auto_consent[provider] !== true) {
      this.set({ consentFor: provider });
      return;
    }
    await this.setPreferences({ auto_provider: provider });
  }

  /**
   * The consent dialog's answer: yes saves the choice and the consent in one patch (so a program is never saved without its consent) and
   * closes the dialog; no closes it and keeps the previous choice.
   */
  async answerConsent(agree: boolean): Promise<void> {
    const provider = this.state.consentFor;
    if (provider === null) return;
    this.set({ consentFor: null });
    if (!agree) return;
    const saved = this.state.preferences?.auto_consent ?? { claude: false, agy: false };
    await this.setPreferences({ auto_provider: provider, auto_consent: { ...saved, [provider]: true } });
  }

  async refreshCliLink(): Promise<void> {
    try {
      this.set({ cliLink: await this.api.cliLinkStatus() });
    } catch (error) {
      this.fail(error);
    }
  }

  /** "Install Command-Line Tool…": links `ym` onto the PATH and says where. */
  async installCli(): Promise<void> {
    try {
      const result = await this.api.installCliLink();
      this.set({ cliLink: { installed: true, path: result.path } });
      this.showNotice(`The command-line tool is installed at ${result.path}.`);
    } catch (error) {
      this.fail(error);
    }
  }

  openOverlay(overlay: Overlay): void {
    this.set({ overlay });
    if (overlay === "preferences") {
      void this.refreshCliLink();
      void this.refreshAuto();
      void this.refreshOs();
    }
  }

  closeOverlay(): void {
    // Closing any dialog drops the consent question too (`set`).
    this.set(this.state.overlay === "import-cube" ? { overlay: null, cubeImport: null } : { overlay: null });
  }

  /** A short message that is not an error. */
  showNotice(notice: string): void {
    this.set({ notice, noticeAction: null });
  }

  dismissNotice(): void {
    this.set({ notice: null, noticeAction: null });
  }

  /** ⌘W: closes the session, after asking when an export is running (`docs/11` §Keyboard shortcuts). */
  requestCloseSession(): Promise<void> {
    if (!this.state.session) return Promise.resolve();
    if (this.exportRunning()) {
      this.set({ overlay: "confirm-close" });
      return Promise.resolve();
    }
    return this.closeSession();
  }

  /** The answer "Stop the export and close". */
  async stopExportAndClose(): Promise<void> {
    this.set({ overlay: null });
    await this.cancelExport();
    await this.closeSession();
  }

  // --- Looks import: a loose .cube ----------------------------------------------------------------

  /**
   * Looks › Import .cube…: asks for files, finds out what each is, and opens the dialog that asks what a LUT expects as
   * input (`docs/08` §Any other .cube); ARRI Look Files (`.alf4`, `.alf4c`, `.aml`) say what they expect and are not asked.
   * A cancelled chooser changes nothing.
   */
  async startCubeImport(): Promise<void> {
    let paths: string[] | null;
    try {
      paths = await this.api.chooseLookFiles();
    } catch (error) {
      this.fail(error);
      return;
    }
    if (!paths || paths.length === 0) return;
    try {
      const found = await Promise.all(paths.map((path) => this.api.looksRecognise(path)));
      const items = found.map((f): CubeItem => ({
        found: f,
        answer: f.recognition.kind === "needs_answer" ? (f.recognition.suggestion ?? null) : null,
      }));
      this.set({ cubeImport: { items, busy: false }, overlay: "import-cube" });
    } catch (error) {
      this.fail(error);
    }
  }

  answerCube(path: string, input: LookInput): void {
    const dialog = this.state.cubeImport;
    if (!dialog) return;
    this.set({ cubeImport: { ...dialog, items: dialog.items.map((i) => (i.found.path === path ? { ...i, answer: input } : i)) } });
  }

  /** Imports what the dialog lists and can import; says what happened. */
  async confirmCubeImport(): Promise<void> {
    const dialog = this.state.cubeImport;
    if (!dialog || dialog.busy) return;
    const todo = dialog.items.filter(
      (i) => i.found.recognition.kind === "package" || i.found.recognition.kind === "look_files" || (i.found.recognition.kind === "needs_answer" && i.answer),
    );
    if (todo.length === 0) return;
    this.set({ cubeImport: { ...dialog, busy: true } });
    let added = 0;
    let failed = false;
    for (const item of todo) {
      try {
        const result = await this.api.looksImport(item.found.path, item.answer ?? undefined);
        added += result.added.length;
      } catch (error) {
        failed = true;
        this.fail(error);
      }
    }
    this.set({ overlay: null, cubeImport: null });
    await this.refreshLooks();
    if (!failed) this.showNotice(added === 0 ? "Those LUTs were already imported." : `${added} ${added === 1 ? "LUT" : "LUTs"} added to Your LUTs.`);
  }

  // --- Export -----------------------------------------------------------------------------------

  /** True while a run is still writing files. Used to hold back an update and to ask before closing the session. */
  exportRunning(): boolean {
    return this.state.exportRun?.phase === "running" || this.starting;
  }

  setPanel(panel: Panel): void {
    if (panel === this.state.panel) return;
    this.set({ panel });
  }

  /** ⌘E, the Export cell: shows the panel. A new visit starts from picked photos, or the current one. */
  openExport(): void {
    if (!this.state.session) return;
    if (this.state.panel === "export") return;
    const fresh = this.state.exportRun === null;
    this.set({
      panel: "export",
      ...(fresh ? { exportScope: defaultScope(this.state.session), exportStopped: null } : {}),
    });
  }

  closeExport(): void {
    this.setPanel("tools");
  }

  setExportScope(scope: ApplyScope): void {
    this.set({ exportScope: scope, exportStopped: null });
  }

  /**
   * Changes the options and remembers them as the defaults (`docs/09`: "Defaults are remembered as preferences").
   * While a slider is dragged (`final = false`) only the screen changes; the release saves.
   */
  setExportOptions(patch: Partial<ExportOptions>, final = true): void {
    const options = { ...this.state.exportOptions, ...patch };
    this.set({ exportOptions: options, exportStopped: null });
    // A template that can't be used is not remembered, and nothing is sent when nothing changed.
    const saved = this.state.preferences?.export_defaults;
    if (final && templateProblem(options.name_template) === null && !(saved && sameOptions(saved, options))) {
      void this.setPreferences({ export_defaults: options });
    }
  }

  /** The job for the current scope and options. */
  exportJob(): ExportJob {
    return { scope: this.state.exportScope, options: this.state.exportOptions };
  }

  /**
   * Queues the job. `names` are the file names the panel expects, for the waiting rows; `destination` is the folder
   * they go to. Resolves true when the core accepted it.
   */
  async startExport(names: readonly string[], destination: string): Promise<boolean> {
    if (this.exportRunning()) return false;
    this.starting = true;
    this.earlyExport.length = 0;
    try {
      const jobId = await this.api.exportStart(this.exportJob());
      let run = startRun(jobId, names, destination);
      for (const e of this.earlyExport.splice(0)) run = applyProgress(run, e);
      this.set({ exportRun: run, exportStopped: null });
      return true;
    } catch (error) {
      this.fail(error);
      return false;
    } finally {
      this.starting = false;
    }
  }

  /**
   * A tile of the export finished (`export_progress`: one event per tile and photo). Events that arrive before `export_start` has
   * answered are held for it; the rest are applied with the other batched events, so a long export draws a few times a second and not
   * once per event.
   */
  private onExportProgress(e: ExportProgress): void {
    if (!this.state.exportRun) {
      if (this.starting) this.earlyExport.push(e);
      return;
    }
    this.later((s) => {
      const run = s.exportRun;
      if (!run) return s;
      const next = applyProgress(run, e);
      return next === run ? s : { ...s, exportRun: next };
    });
  }

  /** Cancel: the core stops after the current tile and deletes the partial file; the panel goes back to the options. */
  async cancelExport(): Promise<void> {
    const run = this.state.exportRun;
    if (!run) return;
    const saved = run.rows.filter((r) => r.state === "done").length;
    try {
      await this.api.exportCancel(run.jobId);
    } catch (error) {
      this.fail(error);
    }
    this.set({
      exportRun: null,
      exportStopped: saved > 0 ? `Export stopped. ${saved} ${saved === 1 ? "photo was" : "photos were"} saved.` : "Export stopped. Nothing was saved.",
    });
  }

  /** Done: forget the finished run and go back to the tools. */
  finishExport(): void {
    if (this.state.exportRun?.phase === "running") return;
    this.set({ exportRun: null, panel: "tools" });
  }

  /** Show in Finder. */
  async revealExport(): Promise<void> {
    const run = this.state.exportRun;
    if (!run) return;
    try {
      await this.api.revealInFinder(run.destination);
    } catch (error) {
      this.fail(error);
    }
  }

  // --- Viewport -----------------------------------------------------------------------------

  /** Tells the core where the photo cell is. Failures are not shown: the cell reports again on the next resize. */
  setViewportRect(rect: ViewportRect): void {
    this.rect = rect;
    void this.api.viewportSetRect(rect).catch(() => {});
  }

  async setBefore(before: boolean): Promise<void> {
    if (before === this.state.before) return;
    // The Before view has no mask (`docs/23` §5.8): a row's eye ends there.
    this.set(before && (this.state.maskEye !== null || this.state.showDepth !== null || this.state.skinAreas !== null) ? { before, maskEye: null, showDepth: null, skinAreas: null } : { before });
    if (before) this.syncMaskView();
    try {
      await this.api.setBefore(before);
    } catch (error) {
      this.set({ before: !before });
      this.fail(error);
    }
  }

  toggleBefore(): Promise<void> {
    return this.setBefore(!this.state.before);
  }

  /**
   * Show mask (`docs/23` §5.8): turns the view on or off, or chooses its style (a style chosen turns it on when `on` is not given). The core
   * draws the mask, so this tells it ({@link Store.syncMaskView}); the choice outlives nodes and photos.
   */
  setMaskView(change: Partial<MaskViewState>): void {
    const next = { ...this.state.maskView, ...change };
    const eye = this.state.maskEye;
    // With a row's eye on the style is the eye's: choosing one changes how that row shows and leaves Show mask off.
    if (eye !== null && change.on === undefined) {
      if (next.style === this.state.maskView.style) return;
      this.set({ maskView: next });
      this.syncMaskView();
      return;
    }
    if (change.style !== undefined && change.on === undefined) next.on = true;
    if (next.on === this.state.maskView.on && next.style === this.state.maskView.style && eye === null && (!next.on || (this.state.showDepth === null && this.state.skinAreas === null))) return;
    // Show mask on ends the eye: one mask at a time.
    this.set(next.on && (eye !== null || this.state.showDepth !== null || this.state.skinAreas !== null) ? { maskView: next, maskEye: null, showDepth: null, skinAreas: null } : { maskView: next });
    this.syncMaskView();
  }

  /** M and the toolbar toggle: the view on or off, with the style it last had. With a row's eye on it turns the eye off and Show mask on. */
  toggleMaskView(): void {
    this.setMaskView({ on: this.state.maskEye !== null ? true : !this.state.maskView.on });
  }

  /**
   * A mask row's eye (`docs/23` §5.8, §7.6): shows that row's own mask on the photo, in the style of the View card, or ends it (`null`). One row at a
   * time: turning one on turns the others off, and Show mask off. It is view state, not settings: no patch, no undo step. Ignored while there is no
   * corrector on show. It ends with the node (another node, the Input or the Output, the simple view, full screen, Crop, Export), with the photo, in the
   * Before view, in a pick and on Esc; the page ends it when the node's rows change place.
   */
  setMaskEye(row: number | null): void {
    if (row !== null && this.maskNode === null) return;
    // Whatever ends the eye ends Show depth with it (they are one view at a time), and an eye on ends Show depth.
    if (row === this.state.maskEye && (row !== null || (this.state.showDepth === null && this.state.skinAreas === null))) return;
    this.set(row === null ? { maskEye: null, showDepth: null, skinAreas: null } : { maskEye: row, showDepth: null, skinAreas: null, maskView: { ...this.state.maskView, on: false } });
    this.syncMaskView();
  }

  /**
   * Show depth (`docs/23` §4.8): shows the photo's depth map as grey, white for near, for the Depth row `row` of the node on show, or ends it (`null`). View
   * state like the eye, no patch and no undo step, and exclusive with it and with Show mask. It ends where the eye does (another node, the photo, Before, a
   * pick, Esc) and when the Depth card goes away. Ignored while there is no corrector on show.
   */
  setShowDepth(row: number | null): void {
    if (row !== null && this.maskNode === null) return;
    if (row === this.state.showDepth) return;
    this.set(row === null ? { showDepth: null } : { showDepth: row, maskEye: null, skinAreas: null, maskView: { ...this.state.maskView, on: false } });
    this.syncMaskView();
  }

  /**
   * Show skin areas (`docs/27` S-11c): shows where the Face Refinement node on show works on face `face` (from 1), tinted over the photo, or ends it (`null`).
   * View state like the eye, no patch and no undo step, and exclusive with Show mask, the row eyes and Show depth. Choosing another face while it is on shows that
   * face. Ignored while there is no node on show.
   */
  setSkinAreas(face: number | null): void {
    if (face !== null && this.maskNode === null) return;
    if (face === this.state.skinAreas) return;
    this.set(face === null ? { skinAreas: null } : { skinAreas: face, maskEye: null, showDepth: null, maskView: { ...this.state.maskView, on: false } });
    this.syncMaskView();
  }

  /**
   * The corrector the node view shows the mask of: the selected node when it is a corrector and the node view is up (not in the simple view,
   * not while Crop or Export shows, not on the Input or the Output), else null. The view's on and style are kept meanwhile, and apply
   * again when there is a node. The core is told whenever this, the view or the style changes, and again for a new photo.
   */
  setMaskNode(node: number | null): void {
    // A row's eye is for one node: another node, or none, ends it.
    if (node !== this.maskNode && (this.state.maskEye !== null || this.state.showDepth !== null || this.state.skinAreas !== null)) this.set({ maskEye: null, showDepth: null, skinAreas: null });
    this.maskNode = node;
    this.syncMaskView();
  }

  /**
   * Sends the mask view the core should have now, when it is not the one the core has. The core starts a photo without one, so a photo
   * that is not the one it was last told about counts as having none; after a failed command what it has is not known.
   */
  private syncMaskView(): void {
    const { on, style } = this.state.maskView;
    const eye = this.state.maskEye;
    const photo = this.currentId();
    const node = this.maskNode;
    let wanted: MaskView | null = null;
    if (node !== null && photo !== null) {
      if (this.state.skinAreas !== null) wanted = { node, style: "overlay", face: this.state.skinAreas };
      else if (this.state.showDepth !== null) wanted = { node, style: "mask_only", row: this.state.showDepth, depth: true };
      else if (eye !== null) wanted = { node, style, row: eye };
      else if (on) wanted = { node, style };
    }
    if (!sameMaskView(this.state.maskShown, wanted)) this.set({ maskShown: wanted });
    const sent = this.maskSent;
    const has = sent === null ? undefined : sent.photo === photo ? sent.view : null;
    if (has !== undefined && sameMaskView(has, wanted)) return;
    this.maskSent = { photo, view: wanted };
    void this.api.viewportSetMode(this.state.zoom, wanted).catch((error: unknown) => {
      this.maskSent = null;
      this.fail(error);
    });
  }

  async setZoom(mode: ViewportMode): Promise<void> {
    // A wheel zoom still on its way lands first, so it cannot arrive after the mode and zoom the new view.
    const queued = this.viewportQueue.busy ? this.viewportQueue.flush() : null;
    const previous = this.state.zoom;
    this.set({ zoom: mode });
    if (queued) await queued;
    try {
      await this.api.viewportSetMode(mode);
    } catch (error) {
      this.set({ zoom: previous });
      this.fail(error);
    }
  }

  /**
   * Collects wheel, pinch and drag input for the photo (a pan, or a zoom about a point of the cell). It is sent once per animation
   * frame, in order, one command at a time (`ViewportInputQueue`), and the mirrored zoom follows once per frame; a burst of events
   * costs one command. A drag calls {@link flushViewportInput} when it ends.
   */
  queueViewportInput(event: StreamInput): void {
    this.viewportQueue.push(event);
  }

  /** Sends what {@link queueViewportInput} has collected, now. Resolves when the core has taken it. */
  flushViewportInput(): Promise<void> {
    return this.viewportQueue.flush();
  }

  /**
   * Forwards one input over the photo to the core now, after whatever wheel and drag input is waiting (the order of inputs is the
   * order of events). A double click flips the mirrored zoom between Fit and 100 %, and a zoom moves it by the same factor as the core
   * does, at once (the core's own zoom and Fit zoom come back with the next `render_stats`: the Fit zoom is used from there,
   * `fitPercentExact`), so the header and the drag to pan follow what the photo does. The hot path of the wheel is
   * {@link queueViewportInput}.
   */
  async viewportInput(event: ViewportInput): Promise<void> {
    if (event.kind === "zoom" || event.kind === "pan") {
      this.viewportQueue.push(event);
      await this.viewportQueue.flush();
      return;
    }
    // A click lands after the input before it: the spot it names is on the picture as that input left it.
    const queued = this.viewportQueue.busy ? this.viewportQueue.flush() : null;
    if (event.kind === "toggle_zoom") this.set({ zoom: this.toggledZoom() });
    if (queued) await queued;
    try {
      await this.api.viewportInput(event);
    } catch (error) {
      this.fail(error);
    }
  }

  /** The zoom that fits the whole picture in the cell, in percent of the photo's pixels; null before the cell has reported. */
  fitPercent(): number | null {
    const exact = this.fitPercentExact();
    return exact === null ? null : Math.round(exact);
  }

  /**
   * `fitPercent` not rounded: what the core compares a zoom with to tell whether it shows the whole picture. The core's own figure
   * once a viewport frame has reported it (`render_stats.fit_percent`: it counts the lens crop, the straighten angle and the crop), else
   * this screen's estimate from the photo and the cell, which is what the mock has.
   */
  fitPercentExact(): number | null {
    const photo = this.currentPhoto();
    const estimate = this.rect && photo ? fitZoomExact(this.rect, photo, this.currentSettings()?.values.input ?? null) : null;
    return fitZoomOf(this.state.stats, estimate);
  }

  /**
   * True when the photo cell shows the whole picture: at Fit, or at a fixed zoom no larger than the Fit zoom, which the core draws as Fit
   * (50 % on a big display where Fit is 62 %). The picture is letterboxed then, there is nothing to pan, and a double click zooms in.
   */
  showsWholePhoto(): boolean {
    return showsWhole(this.state.zoom, this.fitPercentExact());
  }

  /** Fit becomes 100 %, any other zoom becomes Fit (a double click, Z). Where 100 % is no more than the Fit zoom, it stays Fit. */
  private toggledZoom(): ViewportMode {
    const hundred: ViewportMode = { kind: "percent", value: 100 };
    if (!this.showsWholePhoto()) return { kind: "fit" };
    return showsWhole(hundred, this.fitPercentExact()) ? { kind: "fit" } : hundred;
  }

  /** Z: Fit and 100 % in turn. */
  toggleZoom(): Promise<void> {
    return this.setZoom(this.toggledZoom());
  }

  /**
   * Pick white: sends a click on the photo cell (CSS pixels from its corner) to the core, which finds the spot of the photo under it
   * through the straightening, the turn and the lens maps (only the core knows them), sets `white_pick` and the white balance that
   * follows from it, and says so with `settings_changed`. The settings are read again here and the change is one undo step. A click on
   * the paper around the photo picks nothing; a spot that can't be a white (too bright, too dark) is an error from the core.
   */
  async pickWhite(x: number, y: number): Promise<void> {
    const id = this.currentId();
    if (id === null) return;
    // Wheel and drag input still on its way lands first: the click names a spot of the picture as that input left it.
    if (this.viewportQueue.busy) await this.viewportQueue.flush();
    // Patches in flight land first, so the undo step starts from the settings the core has.
    await this.whenIdle(id);
    const base = this.state.settings[id]?.settings;
    this.picks.add(id);
    try {
      await this.api.viewportInput({ kind: "pick_white", x, y });
      const reply = await this.api.getSettings(id);
      this.setSettings(id, reply);
      const entry = base ? diffSettings(base, reply.settings) : null;
      if (entry) this.historyOf(id).push(entry);
      this.syncHistoryFlags();
    } catch (error) {
      this.fail(error);
    } finally {
      this.picks.delete(id);
    }
  }

  /** Display values under a point of the photo cell, in CSS pixels from its corner; null when the core can't say. */
  async samplePixel(x: number, y: number): Promise<PixelSample | null> {
    try {
      return await this.api.samplePixel(x, y);
    } catch {
      return null;
    }
  }

  /** The same for a point given as a share of the cell (0–1 across and down), such as its middle. */
  samplePixelAt(u: number, v: number): Promise<PixelSample | null> {
    const rect = this.rect;
    return this.samplePixel(rect ? u * rect.width : u, rect ? v * rect.height : v);
  }

  /**
   * Auto crop (`docs/20-auto-crop.md` §1, §3): the core's proposal for a photo at the crop draft's aspect and orientation, or null when
   * it can't make one. The Crop panel says so in its own note, so a failure does not go to the error bar.
   */
  async suggestCrop(photoId: PhotoId, aspect: CropAspect, orientation: Orientation): Promise<CropSuggestion | null> {
    try {
      return await this.api.suggestCrop(photoId, aspect, orientation);
    } catch {
      return null;
    }
  }

  /**
   * Pick colour (Colour › Colour warp, `docs/11`): asks the core for the chroma under a click on the photo cell (CSS pixels from its
   * corner, as Pick white) at the warp's input, and adds a moving pin there with its target on its source, selected, so the next drag
   * on the plane moves it. One undo step. A click on the paper around the photo adds nothing, and nor does one at sixteen pins (the
   * selection goes, so the card says why). In the node view `node` is the corrector whose Colour tab is shown: the chroma is read at the
   * input of its warp (`sample_chroma`'s `at`) and the pin goes into its pins.
   */
  async pickColor(x: number, y: number, node: number = MAIN_ID): Promise<void> {
    const id = this.currentId();
    if (id === null) return;
    // Wheel and drag input still on its way lands first: the click names a spot of the picture as that input left it.
    if (this.viewportQueue.busy) await this.viewportQueue.flush();
    // Patches in flight land first, so the new pin joins the pins the core has.
    await this.whenIdle(id);
    let sample: ChromaSample | null;
    try {
      sample = node === MAIN_ID ? await this.api.sampleChroma(x, y) : await this.api.sampleChroma(x, y, node);
    } catch (error) {
      this.fail(error);
      return;
    }
    const rs = this.state.settings[id];
    if (!sample || !rs || this.currentId() !== id) return;
    const pins = gradeOf(rs.settings, node).color_warp.pins;
    if (pins.length >= WARP_MAX_PINS) {
      this.selectWarpPin(null);
      return;
    }
    const next = [...pins, movingPin(clampToDisc([sample.a, sample.b], WARP_COORD_LIMIT))];
    this.selectWarpPin(next.length - 1);
    await this.update(patchOf(node, { grade: { color_warp: { pins: next } } }), true);
  }

  /** Selects a pin of the colour warp (an index into the current photo's pins), or none. */
  selectWarpPin(index: number | null): void {
    if (index !== this.state.warpPin) this.set({ warpPin: index });
  }

  /**
   * Pick colour of a Colour range row (Mask tab, `docs/23` §7.6): asks the core for the chroma under a click on the photo cell (CSS pixels
   * from its corner, as Pick white) **at the node's input** (`sample_chroma`'s `at` and `tap: "node_input"`: the picture the node reads, before
   * its own trim and grade, which is what a Colour range selects on) and sets the row's centre to it, held to the plane (at most 0.5 from the
   * middle). One undo step. A click on the paper around the photo changes nothing, and nor does a row that is no longer a Colour
   * range (taken away, or moved, while the reading was on its way).
   */
  async pickRange(x: number, y: number, node: number, row: number): Promise<void> {
    const id = this.currentId();
    if (id === null) return;
    if (this.viewportQueue.busy) await this.viewportQueue.flush();
    await this.whenIdle(id);
    let sample: ChromaSample | null;
    try {
      sample = await this.api.sampleChroma(x, y, node, "node_input");
    } catch (error) {
      this.fail(error);
      return;
    }
    if (!sample || this.currentId() !== id) return;
    const item = this.state.settings[id] ? nodeOf(this.state.settings[id].settings, node) : undefined;
    const target = isCardItem(item) ? item.masks.items[row] : undefined;
    if (target?.kind !== "color_range") return;
    const center = clampToDisc([sample.a, sample.b], COLOR_RANGE_REACH);
    await this.update(patchOf(node, { masks: { edit: { [String(row)]: { center } } } }), true);
  }

  /**
   * Selects a row of the selected node's masks (an index into its `masks.items`), or none. A row change ends the Brush tool of the row that
   * was selected, and so does selecting a row again: the caller that wants the tool turns it on afterwards ({@link Store.setBrushTool}).
   */
  selectMaskRow(index: number | null): void {
    if (index !== this.state.maskRow || this.state.brushTool || this.state.objectTool) this.set({ maskRow: index, brushTool: false, objectTool: false });
  }

  /** Turns the Object tool of the selected row on or off (`docs/23` §7.6). */
  setObjectTool(on: boolean): void {
    if (on !== this.state.objectTool) this.set({ objectTool: on });
  }

  /** Chooses what a plain click does on the Object card: Add point, Remove point or Box. Nothing is sent. */
  setObjectMode(mode: ObjectMode): void {
    if (mode !== this.state.objectMode) this.set({ objectMode: mode });
  }

  /**
   * Pick depth (`docs/23` §4.8, §7.6): a click at `(x, y)` of the cell (CSS pixels from its corner) sets `far` of the Depth row `row` of `node` a little behind the
   * depth under it. The click is mapped to the photo frame (`viewport_map_points`; a click on the paper around the photo does nothing), the core answers the limit
   * (`sample_depth`, waiting for the depth map if it is still being made) and the number is written as a `masks.edit` row patch: one undo step. A
   * refusal goes to the error bar; no map to read (it failed) changes nothing. Nothing is written for a row that is not a Depth row any more.
   */
  async pickDepth(x: number, y: number, node: number, row: number): Promise<void> {
    const id = this.currentId();
    if (id === null) return;
    if (this.viewportQueue.busy) await this.viewportQueue.flush();
    await this.whenIdle(id);
    const mapped = await this.mapPoints([[x, y]], "cell");
    const at = mapped?.[0];
    if (!at) return;
    let picked: DepthFar | null;
    try {
      picked = await this.api.sampleDepth(id, at[0], at[1]);
    } catch (error) {
      this.fail(error);
      return;
    }
    if (!picked || this.currentId() !== id || this.depthRowAt(id, node, row) === undefined) return;
    await this.update(patchOf(node, { masks: { edit: { [String(row)]: { far: picked.far } } } }), true, id);
  }

  /** The Depth row `row` of `node` of a photo, as the page has it. */
  private depthRowAt(id: PhotoId, node: number, row: number): Extract<MaskItem, { kind: "depth" }> | undefined {
    const rs = this.state.settings[id];
    const item = rs ? nodeOf(rs.settings, node) : undefined;
    const target = isCardItem(item) ? item.masks.items[row] : undefined;
    return target?.kind === "depth" ? target : undefined;
  }

  /**
   * Focus plane (`docs/23` §4.8, §7.6): sets `far` of the Depth row `row` of `node` a little behind the plane the camera focused on (`focus_plane`), as one
   * row patch and one undo step. A photo with no focus point is remembered (`noFocusPoint`: the button goes off, the card says why) and nothing is written. When
   * the camera's point is exactly the frame's centre the limit is made and the card says to check it (`focusCentre`). A depth map that could not be made changes
   * nothing: its failure is the row's own (`depth_state`).
   */
  async focusPlane(node: number, row: number): Promise<void> {
    const id = this.currentId();
    if (id === null) return;
    let answer: FocusPlaneResult;
    try {
      answer = await this.api.focusPlane(id);
    } catch (error) {
      this.fail(error);
      return;
    }
    if (answer.reason === "no_focus_point") {
      this.set({ noFocusPoint: { ...this.state.noFocusPoint, [id]: true } });
      return;
    }
    if (answer.far === null || this.currentId() !== id || this.depthRowAt(id, node, row) === undefined) return;
    const { far } = answer;
    await this.update(patchOf(node, { masks: { edit: { [String(row)]: { far } } } }), true, id);
    const next = { ...this.state.focusCentre };
    if (answer.exact_centre) next[id] = far;
    else delete next[id];
    this.set({ focusCentre: next });
  }

  /** Asks once whether the photo has a focus point, so that the card can say so before Focus plane is pressed (the core answers at once for a file with none). */
  ensureFocusPoint(photo: PhotoId): void {
    if (this.state.noFocusPoint[photo] || this.focusAsked.has(photo)) return;
    this.focusAsked.add(photo);
    void this.api.focusPlane(photo).then(
      (answer) => {
        if (answer.reason === "no_focus_point") this.set({ noFocusPoint: { ...this.state.noFocusPoint, [photo]: true } });
      },
      () => this.focusAsked.delete(photo),
    );
  }
  private readonly focusAsked = new Set<PhotoId>();

  /**
   * "Try again" (`docs/23` §4.8): starts a photo's depth map again after it failed. The failure is forgotten at once (the row says "Finding…" while the core works
   * and the card is the normal one); if it fails again `depth_state` brings the sentence back.
   */
  async retryDepth(photo: PhotoId): Promise<void> {
    if (this.state.depthFailed[photo] === undefined) return;
    const next = { ...this.state.depthFailed };
    delete next[photo];
    this.set({ depthFailed: next });
    try {
      await this.api.retryDepth(photo);
    } catch (error) {
      this.fail(error);
    }
  }

  /** One find at a time: a click while another is being answered waits, and reads the row as that one left it. */
  private finds: Promise<unknown> = Promise.resolve();
  private pillSeq = 0;
  private pillTimer: ReturnType<typeof setTimeout> | null = null;
  /** The newest `open_progress`, and the timer that shows it once the open has taken a moment (`onOpenProgress`). */
  private openProgressLatest: OpenProgress | null = null;
  private openProgressTimer: ReturnType<typeof setTimeout> | null = null;

  /**
   * Finds an object (`mask_find`, `docs/23` §4.4 and §9): asks the core for the Object row `row` of `node` with the person's click or box added. A **found**
   * answer carries what the row's `points` and `box` become and is sent as **one** `masks.edit` row patch (one undo step), and the pill "Object found ·
   * 0.4 s" shows for three seconds. **Nothing found** adds no point and changes nothing: the error bar says "Nothing found there. Click another part of the
   * object, or draw a box." A refusal of the core (no Object model, 256 points) goes to the error bar with its own sentence. The row says "Finding…" while
   * the core works. Resolves with whether a mask was found.
   */
  findObject(node: number, row: number, prompt: MaskFindPrompt, photoId: PhotoId | null = this.currentId()): Promise<boolean> {
    if (photoId === null) return Promise.resolve(false);
    const id = photoId;
    const run = async (): Promise<boolean> => {
      const rs = this.state.settings[id];
      const item = rs ? nodeOf(rs.settings, node) : undefined;
      const target = isCardItem(item) ? item.masks.items[row] : undefined;
      if (target?.kind !== "object") return false;
      this.set({ objectFinding: id });
      try {
        const found = await this.api.maskFind(id, node, row, prompt);
        if (found.kind === "nothing") {
          this.set({ error: { scope: "mask", message: NOTHING_FOUND } });
          return false;
        }
        await this.update(patchOf(node, { masks: { edit: { [String(row)]: { points: found.points, box: found.box } } } }), true, id);
        this.showPill(foundText(found.took_ms));
        return true;
      } catch (error) {
        this.fail(error);
        return false;
      } finally {
        if (this.state.objectFinding === id) this.set({ objectFinding: null });
      }
    };
    const next = this.finds.then(run, run);
    this.finds = next;
    return next;
  }

  /** Shows the pill "Object found · 0.4 s" for {@link FOUND_PILL_MS}; a newer one replaces it. */
  private showPill(text: string): void {
    if (this.pillTimer !== null) clearTimeout(this.pillTimer);
    this.pillSeq += 1;
    const seq = this.pillSeq;
    this.set({ objectPill: { seq, text } });
    this.pillTimer = setTimeout(() => {
      this.pillTimer = null;
      if (this.state.objectPill?.seq === seq) this.set({ objectPill: null });
    }, FOUND_PILL_MS);
  }

  /** Turns the Brush tool of the selected row on or off (`docs/23` §7.6). */
  setBrushTool(on: boolean): void {
    if (on !== this.state.brushTool) this.set({ brushTool: on });
  }

  /** Sets the brush the next strokes are drawn with (size, hardness, paint or erase). The strokes already drawn keep theirs. */
  setBrush(change: Partial<BrushSettings>): void {
    const next = { ...this.state.brush, ...change };
    if (next.size !== this.state.brush.size || next.hardness !== this.state.brush.hardness || next.erase !== this.state.brush.erase) this.set({ brush: next });
  }

  /**
   * Appends a stroke to a Brush row (`docs/23` §3.7): one `masks.edit` row patch with `strokes_append`, sent when the pointer is released. One
   * undo step, whose inverse is `strokes_keep` with the old count (`diffSettings`). A row that is not a Brush row any more (taken away or
   * moved while the stroke was being mapped) takes nothing. A stroke that would take the row over 100,000 points is refused with the core's
   * sentence in the error bar, before anything is sent, and changes nothing. Resolves with whether the stroke was sent.
   */
  async addStroke(node: number, row: number, stroke: BrushStroke, photoId: PhotoId | null = this.currentId()): Promise<boolean> {
    if (photoId === null) return false;
    const rs = this.state.settings[photoId];
    const item = rs ? nodeOf(rs.settings, node) : undefined;
    const target = isCardItem(item) ? item.masks.items[row] : undefined;
    if (target?.kind !== "brush" || stroke.points.length === 0) return false;
    if (brushPointCount(target) + stroke.points.length > MAX_BRUSH_POINTS) {
      this.fail({ scope: "render", message: BRUSH_FULL });
      return false;
    }
    await this.update(patchOf(node, { masks: { edit: { [String(row)]: { strokes_append: [stroke] } } } }), true, photoId);
    return true;
  }

  /**
   * Appends a stroke to a Face Refinement node's protection (`docs/27` S-11c): one `protect.strokes_append` patch, sent when the pointer is released, one undo
   * step. A stroke that would take the plane over 100,000 points is refused with the core's sentence in the error bar and changes nothing. Resolves with whether
   * the stroke was sent.
   */
  async addProtectStroke(node: number, stroke: BrushStroke, photoId: PhotoId | null = this.currentId()): Promise<boolean> {
    if (photoId === null) return false;
    const rs = this.state.settings[photoId];
    const item = rs ? nodeOf(rs.settings, node) : undefined;
    if (item?.kind !== "face_refinement" || stroke.points.length === 0) return false;
    const have = item.protect.strokes.reduce((n, s) => n + s.points.length, 0);
    if (have + stroke.points.length > MAX_BRUSH_POINTS) {
      this.fail({ scope: "render", message: BRUSH_FULL });
      return false;
    }
    await this.update(patchOf(node, { protect: { strokes_append: [stroke] } }), true, photoId);
    return true;
  }

  /** Asks which kinds of mask this machine can make, once (`mask_kinds`): the Add mask menu lists exactly these. A failure leaves the menu without items. */
  ensureMaskKinds(): void {
    if (this.state.maskKinds !== null || this.maskKindsAsked) return;
    this.maskKindsAsked = true;
    void this.api.maskKinds().then(
      (kinds) => this.set({ maskKinds: kinds }),
      (error: unknown) => {
        this.maskKindsAsked = false;
        this.fail(error);
      },
    );
  }

  /**
   * Maps points between the photo frame and the cell (`viewport_map_points`, `docs/23` §9): the handles and outlines the page draws over the
   * photo. The page never converts: the core knows the straightening, the turn, the crop, the lens's bend and the view's zoom and pan. `null`
   * when the core could not answer (the handles are then left as they were, with no error: the next frame asks again).
   */
  async mapPoints(points: Array<[number, number]>, from: MapFrom): Promise<Array<[number, number] | null> | null> {
    try {
      return await this.api.viewportMapPoints(points, from);
    } catch {
      return null;
    }
  }

  // --- Settings -----------------------------------------------------------------------------

  /** Settings of the current photo, when they have arrived. */
  currentSettings(): ResolvedSettings | null {
    const id = this.currentId();
    return id === null ? null : (this.state.settings[id] ?? null);
  }

  private ensureSettings(id: PhotoId): Promise<void> {
    if (this.state.settings[id]) return Promise.resolve();
    const existing = this.settingsRequests.get(id);
    if (existing) return existing;
    const request = this.readSettings(id).finally(() => {
      this.settingsRequests.delete(id);
    });
    this.settingsRequests.set(id, request);
    return request;
  }

  /** Reads a photo's settings into the mirror unless it already holds them; asks again when a copy to other photos landed meanwhile. */
  private async readSettings(id: PhotoId): Promise<void> {
    for (;;) {
      const epoch = this.bulkEpoch;
      try {
        const rs = await this.api.getSettings(id);
        if (epoch !== this.bulkEpoch) continue;
        if (!this.state.settings[id]) {
          this.set({ settings: { ...this.state.settings, [id]: rs } });
          this.wantNodeThumbnails(id);
        }
        return;
      } catch (error) {
        this.fail(error);
        return;
      }
    }
  }

  /**
   * Throws away what is known about a photo's settings and asks again. What comes back is dropped when a patch was made for the photo
   * or answered (or a copy to other photos landed) while the question was on its way: the answer may be older than that, and the
   * patch's own reply puts the settings right.
   */
  private async resyncSettings(id: PhotoId): Promise<void> {
    for (;;) {
      const epoch = this.bulkEpoch;
      const written = this.writes.get(id) ?? 0;
      try {
        const rs = await this.api.getSettings(id);
        if (epoch !== this.bulkEpoch) continue;
        if ((this.writes.get(id) ?? 0) === written) this.setSettings(id, rs);
        return;
      } catch (error) {
        this.fail(error);
        return;
      }
    }
  }

  /**
   * Reads a photo's settings again after news from the core, without taking back what the person is doing. Patches on their way land
   * first; then the settings are asked for, and a reply that a newer write overtook while it was out (a patch sent or answered, Re-run,
   * a Pick white, a paste: anything {@link writes} counts, or a copy to other photos) is not shown, because it may predate that write
   * and a slider being dragged would jump back. The store asks again once that write has landed: its own reply may predate the news. A
   * photo whose settings the mirror no longer holds, or that is being removed ({@link leaving}), is not asked for, and a reply for one
   * that left meanwhile is not shown.
   */
  private async refreshSettings(id: PhotoId): Promise<void> {
    for (;;) {
      await this.whenIdle(id);
      if (!this.state.settings[id] || this.leaving.has(id)) return;
      if (this.isSending(id)) continue;
      const epoch = this.bulkEpoch;
      const asked = this.writes.get(id) ?? 0;
      let rs: ResolvedSettings;
      try {
        rs = await this.api.getSettings(id);
      } catch (error) {
        // The core says the photo is unknown when it was removed while the question was on its way: nothing to tell.
        if (!this.leaving.has(id) && this.state.settings[id]) this.fail(error);
        return;
      }
      if (this.leaving.has(id) || !this.state.settings[id]) return;
      if (epoch === this.bulkEpoch && (this.writes.get(id) ?? 0) === asked) {
        this.setSettings(id, rs);
        return;
      }
    }
  }

  /** Counts a write of a photo's settings, or a patch sent for it ({@link writes}). */
  private wrote(id: PhotoId): void {
    this.writes.set(id, (this.writes.get(id) ?? 0) + 1);
  }

  private setSettings(id: PhotoId, next: ResolvedSettings): void {
    this.wrote(id);
    // The reply to a patch usually says what the page already shows (it folded the patch in at once): then nothing is stored and nobody
    // redraws. What did change comes in as a copy that shares every part that did not (`share`), so a control that reads one field
    // sees the same field (U-02).
    const had = this.state.settings[id];
    const rs = had ? share(had, next) : next;
    if (rs !== had) this.set({ settings: { ...this.state.settings, [id]: rs } });
    // The node thumbnails follow every change that has landed: this asks, and the ask waits while a drag or a patch is still out.
    this.wantNodeThumbnails(id);
  }

  /**
   * Sends a patch for a photo. While a drag is on, call it with `final = false` (at most once per frame, which the controls do); the
   * release, a key, a typed value or a reset calls it with `final = true`, which also records one undo step for the whole gesture.
   * Resolves when the core has answered.
   *
   * A patch belongs to the photo the person made the input on. Without `photoId` that is the photo on screen when this is called,
   * which is right for an input that is handled at once (a key, a click, a typed value). A control that collects its input and sends it
   * later (a drag, one patch per frame) must name the photo it began on, so that moving to another photo before the frame or the release
   * cannot send it there: the controls pass the `photo_id` of the settings they were drawn from.
   */
  update(patch: SettingsPatch, final = true, photoId: PhotoId | null = this.currentId()): Promise<void> {
    const id = photoId;
    if (id === null) return Promise.resolve();
    const rs = this.state.settings[id];
    if (!rs) return Promise.resolve();
    let gesture = this.gestures.get(id);
    if (!gesture) {
      gesture = { base: rs.settings };
      this.gestures.set(id, gesture);
    }
    const record = final ? { base: gesture.base } : null;
    if (final) this.gestures.delete(id);
    this.setSettings(id, applyDirect(rs, patch));
    return this.enqueue(id, patch, record);
  }

  private senderFor(id: PhotoId): Sender {
    let sender = this.senders.get(id);
    if (!sender) {
      sender = { busy: false, queue: [], idle: [] };
      this.senders.set(id, sender);
    }
    return sender;
  }

  private enqueue(id: PhotoId, patch: SettingsPatch, record: Queued["record"]): Promise<void> {
    this.wrote(id);
    const sender = this.senderFor(id);
    return new Promise<void>((resolve) => {
      const last = sender.queue[sender.queue.length - 1];
      if (last && last.record === null) {
        // The waiting patch belongs to a drag that hasn't ended: fold this one into it.
        last.patch = foldPatches(last.patch, patch);
        last.record = record;
        last.done.push(resolve);
      } else {
        sender.queue.push({ patch, record, done: [resolve] });
      }
      void this.pump(id, sender);
    });
  }

  private async pump(id: PhotoId, sender: Sender): Promise<void> {
    if (sender.busy) return;
    sender.busy = true;
    try {
      for (let item = sender.queue.shift(); item; item = sender.queue.shift()) {
        // For the timeline of a slider tick (`src/perfTimeline.ts`): a function that does nothing unless the shell was started with YM_PERF.
        const tick = stampTick(item.record !== null);
        try {
          const reply = await this.api.updateSettings(id, item.patch);
          if (item.record) {
            const entry = diffSettings(item.record.base, reply.settings);
            if (entry) this.historyOf(id).push(entry);
          }
          // A newer patch is waiting: its reply will carry everything, so this one is skipped.
          if (sender.queue.length === 0 && this.state.settings[id]) this.setSettings(id, reply);
          tick.replied();
        } catch (error) {
          tick.replied(false);
          for (const waiting of sender.queue.splice(0)) for (const done of waiting.done) done();
          this.fail(error);
          await this.resyncSettings(id);
        } finally {
          for (const done of item.done) done();
          this.syncHistoryFlags();
        }
      }
    } finally {
      sender.busy = false;
      for (const wake of sender.idle.splice(0)) wake();
    }
  }

  /** Whether a patch for a photo is on its way to the core or waiting to be sent. */
  private isSending(id: PhotoId): boolean {
    const sender = this.senders.get(id);
    return sender !== undefined && (sender.busy || sender.queue.length > 0);
  }

  /** Resolves when every patch for a photo has been answered. */
  private whenIdle(id: PhotoId): Promise<void> {
    const sender = this.senders.get(id);
    if (!sender || (!sender.busy && sender.queue.length === 0)) return Promise.resolve();
    return new Promise((resolve) => sender.idle.push(resolve));
  }

  /** "Re-run" on the Automatic card: recompute the estimates and put exposure, white balance and noise back to Auto. */
  async rerunAuto(): Promise<void> {
    const id = this.currentId();
    if (id === null) return;
    await this.whenIdle(id);
    const before = this.state.settings[id]?.settings;
    try {
      const reply = await this.api.rerunAuto(id);
      if (before) {
        const entry = diffSettings(before, reply.settings);
        if (entry) this.historyOf(id).push(entry);
      }
      this.setSettings(id, reply);
      this.syncHistoryFlags();
    } catch (error) {
      this.fail(error);
    }
  }

  // --- Copy and paste settings --------------------------------------------------------------

  /**
   * Copy settings (⇧⌘C): keeps the current photo's settings for pasting: the whole graph (every node and every link) but the Input's
   * crop, orientation and Pick white point, Auto kept as Auto, a white balance that came from a picked white as the manual Kelvin and tint it resolved to. The core
   * makes the patch (`copy_settings`), after the patches still on their way to it have landed. Nothing is saved or put on the system
   * clipboard. Does nothing for a photo that is not developed.
   */
  async copySettings(): Promise<void> {
    const photo = this.currentPhoto();
    if (!photo || !isReady(photo)) return;
    await this.whenIdle(photo.id);
    try {
      const patch = await this.api.copySettings(photo.id);
      const source = photoName(photo.file_name);
      this.set({ copied: { source, sourceId: photo.id, patch } });
      this.showNotice(`Copied the settings of ${source}.`);
    } catch (error) {
      this.fail(error);
    }
  }

  /**
   * Paste settings (⇧⌘V) onto the current photo, or Paste settings to Picked (⌥⇧⌘V) onto every picked photo, the current one too when
   * it is picked. One `paste_settings` for all of them, so five hundred photos are one command. Each photo's change is one undo step
   * in that photo's own history. A photo that is still developing or could not be opened is skipped, and the notice says how many.
   * The settings of the photo on screen when the answer comes are read again, the settings of the others are dropped and asked for
   * when they are shown, and the filmstrip thumbnails follow from the core.
   *
   * Paste to Picked leaves the source out when its white came from a picked white and still does: the paste would turn that Auto
   * white balance into the manual one that stands for it on other photos, and change the photo it came from.
   */
  async pasteSettings(scope: "this" | "picked"): Promise<void> {
    if (!this.state.session) return;
    if (!this.state.copied) {
      this.showNotice("Copy the settings of a photo first.");
      return;
    }
    // The photos the paste is for are the ones the person has now, picked or on screen: fixed before anything is waited for.
    const { session, copied } = this.state;
    const current = this.currentPhoto();
    const chosen = scope === "this" ? (current ? [current] : []) : session.photos.filter((p) => p.picked);
    const own = scope === "picked" ? await this.pickedWhiteSource(copied) : null;
    if (!this.state.session) return;
    if (chosen.length === 0) {
      this.showNotice(scope === "this" ? "There is no photo to paste onto." : "No photos are picked.");
      return;
    }
    const wanted = own === null ? chosen : chosen.filter((p) => p.id !== own);
    if (wanted.length === 0) {
      this.showNotice("No other photos are picked.");
      return;
    }
    const ids = wanted.filter(isReady).map((p) => p.id);
    // Patches still on their way to a photo land first, so its undo step starts from the settings the core has.
    await Promise.all(ids.map((id) => this.whenIdle(id)));
    let report: PasteReport = { pasted: [], skipped: [] };
    if (ids.length > 0) {
      for (const id of ids) this.pasting.add(id);
      try {
        report = await this.api.pasteSettings(ids, copied.patch);
        // Reads of settings that were on their way when the paste landed may hold the old ones.
        this.bulkEpoch += 1;
      } catch (error) {
        this.fail(error);
        return;
      } finally {
        // The core's `settings_changed` for these photos arrives about now; let it pass before the guard is lifted.
        setTimeout(() => {
          for (const id of ids) this.pasting.delete(id);
        }, 0);
      }
    }
    const changed = report.pasted.filter((p) => !isEmptyPatch(p.before) || !isEmptyPatch(p.after));
    for (const step of changed) {
      this.historyOf(step.id).push({ before: step.before, after: step.after });
      // A crop or a slider drag that was open on the photo when the paste landed records its own step when it ends, from the settings it
      // began with. Those must include the paste, or the step would carry it too and one Undo would take back both.
      const gesture = this.gestures.get(step.id);
      if (gesture) gesture.base = applyPatch(gesture.base, step.after);
    }
    // What the mirror knew about the other photos is out of date. The photo on screen now (it may not be the one that was when the paste
    // began) and the one that was are read again at once, so neither is left without settings.
    const shown = new Set<PhotoId>();
    if (current) shown.add(current.id);
    const now = this.currentId();
    if (now !== null) shown.add(now);
    const stale = new Set(changed.map((p) => p.id));
    if (stale.size > 0) {
      const kept: Record<PhotoId, ResolvedSettings> = {};
      for (const [key, value] of Object.entries(this.state.settings)) if (!stale.has(Number(key)) || shown.has(Number(key))) kept[Number(key)] = value;
      this.set({ settings: kept });
    }
    await Promise.all([...shown].filter((id) => stale.has(id)).map((id) => this.resyncSettings(id)));
    this.syncHistoryFlags();
    const skipped = wanted.length - ids.length + report.skipped.length;
    this.showNotice(pastedNotice(copied.source, report.pasted.length, skipped));
  }

  /**
   * The id of the photo the copied settings came from when Paste to Picked must leave it out: the copied white balance is the manual
   * one that stands for a picked white, and that photo's own white is still a picked one (a Pick white point, the light preset and the
   * white balance both Auto). Pasting there would swap its Auto for a manual white that renders a little differently and mark it edited.
   * Otherwise null: a source whose white was not picked, or that was changed since, takes the paste like any photo.
   */
  private async pickedWhiteSource(copied: CopiedSettings): Promise<PhotoId | null> {
    const id = copied.sourceId;
    if (id === null || copied.patch.nodes?.[INPUT_ID]?.white_balance?.kind !== "manual") return null;
    const photo = this.state.session?.photos.find((p) => p.id === id);
    if (!photo || !isReady(photo)) return null;
    await this.ensureSettings(id);
    const settings = this.state.settings[id]?.settings;
    const input = settings && inputOf(settings);
    const pickedWhite = input !== undefined && input.white_pick !== null && input.light_source === "auto" && input.white_balance.kind === "auto";
    return pickedWhite ? id : null;
  }

  // --- Looks --------------------------------------------------------------------------------

  /**
   * Picks a look for a photo (the one on screen unless `photoId` names another) and, when "Apply to" is Picked or All, for those photos
   * too. The photo goes first through `update_settings`, so the change is one undo step and the viewport answers at once; the others
   * follow through `apply_look`.
   *
   * Everything the pick means is fixed when it is made: the photo, the scope, the look and its strength. The person may move to
   * another photo, or pick again there, before the core has answered; this pick still goes to the photo it was made on, and its copy
   * to the others reaches the core before the copy of any pick made after it ({@link applyAfter}).
   */
  async chooseLook(lookId: LookId, intensity?: number, photoId: PhotoId | null = this.currentId()): Promise<void> {
    if (photoId === null) return;
    const rs = this.state.settings[photoId];
    const look = this.state.looks?.find((l) => l.id === lookId);
    if (!rs || !look || look.missing) return;
    const level = intensity ?? rs.values.output.look_intensity;
    await this.changeLook(photoId, this.state.applyScope, outputPatch({ look: lookId, look_intensity: level }), true, lookId, level);
  }

  /**
   * Sets the look strength of a photo (the one on screen unless `photoId` names another); the other photos of the scope get it, with
   * the look the photo has, when the drag ends. A drag names the photo it began on, as {@link update} asks of every control.
   */
  async setLookIntensity(level: number, final: boolean, photoId: PhotoId | null = this.currentId()): Promise<void> {
    if (photoId === null) return;
    const rs = this.state.settings[photoId];
    if (!rs) return;
    await this.changeLook(photoId, this.state.applyScope, outputPatch({ look_intensity: level }), final, rs.values.output.look, level);
  }

  /** The patch of one photo's look, then (at the end of a gesture) the copy to the photos of the scope. */
  private async changeLook(id: PhotoId, scope: ApplyScope, patch: SettingsPatch, final: boolean, lookId: LookId, level: number): Promise<void> {
    const failed = this.failures;
    const sent = this.update(patch, final, id);
    if (!final || scope === "this") {
      await sent;
      return;
    }
    await this.applyAfter(sent, async () => {
      // If the photo could not take the look, the others must not get it either.
      if (this.failures === failed) await this.copyLook(id, scope, lookId, level);
    });
  }

  /**
   * Runs a copy to other photos once the photo's own patch has been answered and every copy asked for before it has finished. Each
   * photo's patch is answered when it is (a photo's patches go one at a time, different photos' at once), and the core takes commands
   * as they come: left alone, a pick on photo 1 that is slow would have its copy land after the copy of a pick made on photo 2
   * afterwards, and leave every photo with the older look. The copies go in the order of the picks.
   */
  private async applyAfter(own: Promise<void>, copy: () => Promise<void>): Promise<void> {
    const before = this.applyTail;
    const turn = (async () => {
      await own;
      await before;
      await copy();
    })();
    this.applyTail = turn.catch(() => {});
    await turn;
  }

  /**
   * Copies a look and its strength to the photos of `scope`, which the person chose when they picked it. The photo they picked it on
   * (`source`) already has them, from its own reply. The other photos' look changed in the core behind the mirror's back: what the
   * mirror holds for them is dropped, and asked for again when they are shown. The photo on screen now (the person may have moved
   * since the pick) is asked for at once, so it is never left showing the look it had.
   */
  private async copyLook(source: PhotoId, scope: ApplyScope, lookId: LookId, level: number): Promise<void> {
    if (scope === "this") return;
    try {
      await this.api.applyLook(scope, lookId, level);
    } catch (error) {
      this.fail(error);
      return;
    }
    await this.forgetOthers(source);
  }

  /**
   * The core changed photos other than `keep` without the mirror knowing: forgets their settings, and reads the settings of the photo
   * on screen again if it is one of them. Its old settings stay in the meantime (a control being dragged there has to find them) and are
   * replaced when the answer comes. Reads that were already on their way are asked again when they arrive (`bulkEpoch`).
   */
  private async forgetOthers(keep: PhotoId | null): Promise<void> {
    this.bulkEpoch += 1;
    const shown = this.currentPhoto();
    const kept: Record<PhotoId, ResolvedSettings> = {};
    for (const [key, value] of Object.entries(this.state.settings)) if (Number(key) === keep || Number(key) === shown?.id) kept[Number(key)] = value;
    this.set({ settings: kept });
    if (!shown || shown.id === keep || !isReady(shown)) return;
    if (kept[shown.id]) await this.resyncSettings(shown.id);
    else await this.ensureSettings(shown.id);
  }

  /** "Make default": the look every new photo opens with (a preference). */
  async makeDefault(lookId: LookId): Promise<void> {
    try {
      await this.api.looksSetDefault(lookId);
      await this.refreshLooks();
    } catch (error) {
      this.fail(error);
    }
  }

  /**
   * The photo through one look, as an image URL, for a tile of the Look card. `key` stands for
   * everything about the photo except its look (its other settings): tiles are drawn again when
   * it changes. Thumbnails are kept for the session, except that a request that failed is not kept
   * (the core cancels a tile when the person leaves its photo, and a tile that asks again after
   * coming back is handed the same request, which then fails), and a tile whose address stopped
   * answering is dropped by {@link forgetLookThumbnail}.
   *
   * At most {@link LOOK_THUMBNAIL_CONCURRENCY} requests are out at once, the newest first. A request that has not started when
   * its photo's settings change (a new `key`) or the person moves to another photo is dropped and rejects with
   * `Superseded` (`isSuperseded`): that is not a failure, and whoever still wants the picture asks again.
   */
  lookThumbnail(photoId: PhotoId, lookId: LookId, key: string): Promise<string> {
    let entry = this.lookThumbs.get(photoId);
    if (!entry || entry.key !== key) {
      if (entry) {
        this.releaseLater([...entry.urls.values()]);
        this.thumbLimiter.supersede((t) => t.photoId === photoId && t.key !== key);
      }
      entry = { key, urls: new Map() };
      this.lookThumbs.set(photoId, entry);
    }
    const cache = entry.urls;
    let thumb = cache.get(lookId);
    if (!thumb) {
      const request = this.thumbLimiter.run({ photoId, key }, () => this.api.lookThumbnail(photoId, lookId));
      const made: LookThumb = { promise: request, url: null };
      thumb = made;
      cache.set(lookId, made);
      // Registered before anyone else sees the promise, so it runs first: a caller that hears the failure and asks again gets a new request.
      made.promise.then(
        (url) => {
          made.url = url;
        },
        () => {
          if (cache.get(lookId) === made) cache.delete(lookId);
        },
      );
    }
    return thumb.promise;
  }

  /**
   * Drops the address a tile was given because its picture did not load: the shell keeps a bounded
   * amount of tiles and answers 404 for the addresses of the photos used least recently. The next
   * {@link lookThumbnail} for it asks for a new picture. Does nothing when the address held now is a
   * different one (another tile of the same look has asked again already).
   */
  forgetLookThumbnail(photoId: PhotoId, lookId: LookId, key: string, url: string): void {
    const entry = this.lookThumbs.get(photoId);
    if (entry?.key !== key) return;
    const thumb = entry.urls.get(lookId);
    if (thumb?.url !== url) return;
    entry.urls.delete(lookId);
    this.releaseLater([thumb]);
  }

  private releaseLater(thumbs: LookThumb[]): void {
    const pending = thumbs.map((t) => t.promise);
    setTimeout(() => {
      for (const url of pending) void url.then(releaseUrl, () => {});
    }, URL_GRACE_MS);
  }

  /**
   * The list of stocks and papers of the Film node (`film_catalogue`), read once. A failure is not kept: the next ask reads again, so
   * the picker that said "Yellowmoon's film data is missing" recovers when the data is back.
   */
  filmCatalogue(): Promise<FilmCatalogue> {
    if (!this.filmCatalogueRead) {
      const read = this.api.filmCatalogue();
      this.filmCatalogueRead = read;
      read.catch(() => {
        if (this.filmCatalogueRead === read) this.filmCatalogueRead = null;
      });
    }
    return this.filmCatalogueRead;
  }

  /**
   * The photo through one stock in Film node `node`, as an image URL, for a tile of the stock picker (`docs/24` section 4.4). `key`
   * stands for everything about the photo up to and including the node except its stock: the tiles are made again when it changes.
   * Kept per photo for the session, in memory; a request that failed is not kept. One request is out at a time, in the order they
   * are asked for; what waits is dropped when the photo's settings change or the person moves to another photo
   * (`Superseded`, as for {@link lookThumbnail}).
   */
  filmTile(photoId: PhotoId, node: number, stock: string, key: string): Promise<string> {
    const picture = `${node} ${key}`;
    let entry = this.filmTiles.get(photoId);
    if (!entry || entry.key !== picture) {
      if (entry) {
        this.releaseLater([...entry.urls.values()]);
        this.filmTileLimiter.supersede((t) => t.photoId === photoId && t.key !== picture);
      }
      entry = { key: picture, urls: new Map() };
      this.filmTiles.set(photoId, entry);
    }
    const cache = entry.urls;
    let tile = cache.get(stock);
    if (!tile) {
      const request = this.filmTileLimiter.run({ photoId, key: picture }, () => this.api.filmTile(photoId, node, stock));
      const made: LookThumb = { promise: request, url: null };
      tile = made;
      cache.set(stock, made);
      made.promise.then(
        (url) => {
          made.url = url;
        },
        () => {
          if (cache.get(stock) === made) cache.delete(stock);
        },
      );
    }
    return tile.promise;
  }

  /** Drops the address of a stock tile whose picture did not load (the shell keeps a bounded amount of tiles); the next {@link filmTile} asks again. */
  forgetFilmTile(photoId: PhotoId, node: number, stock: string, key: string, url: string): void {
    const entry = this.filmTiles.get(photoId);
    if (entry?.key !== `${node} ${key}`) return;
    const tile = entry.urls.get(stock);
    if (tile?.url !== url) return;
    entry.urls.delete(stock);
    this.releaseLater([tile]);
  }

  private releaseLookThumbs(): void {
    this.thumbLimiter.supersede(() => true);
    this.filmTileLimiter.supersede(() => true);
    for (const entry of this.filmTiles.values()) for (const tile of entry.urls.values()) void tile.promise.then(releaseUrl, () => {});
    this.filmTiles.clear();
    for (const entry of this.lookThumbs.values()) for (const thumb of entry.urls.values()) void thumb.promise.then(releaseUrl, () => {});
    this.lookThumbs.clear();
  }

  // --- Histogram ----------------------------------------------------------------------------

  /** Asks for the current photo's histogram. Calls made while one is in flight collapse into one more. */
  async requestHistogram(): Promise<void> {
    const id = this.currentId();
    if (id === null) return;
    if (this.histogramBusy) {
      this.histogramAgain = true;
      return;
    }
    this.histogramBusy = true;
    try {
      do {
        this.histogramAgain = false;
        const asked = this.currentId();
        if (asked === null) break;
        // The simple view names no node (Main's); the node view names the selected one (`docs/01`).
        const at = this.histogramAt;
        const data = await (at === undefined ? this.api.histogram(asked) : this.api.histogram(asked, at));
        // An answer for a node the panel has left is not shown; the request for the new one is already waiting.
        if (this.histogramAt !== at) this.histogramAgain = true;
        else if (this.currentId() === asked) this.set({ histogram: data });
      } while (this.histogramAgain);
    } catch {
      // The plot stays as it was; the next render asks again.
    } finally {
      this.histogramBusy = false;
    }
  }

  /**
   * The histogram (and, with `readout`, the display values at the middle of the cell) is on show: the Light and Curve tabs call this
   * while they are mounted and call the function it returns when they go. It asks once at once, then after the frames the viewport
   * reports (`render_stats`), at most every {@link HISTOGRAM_THROTTLE_MS}: every request is a job in the one queue of the core that
   * also draws the viewport, so asking per frame made each frame wait for the plot. Nothing is asked for while nobody watches. `at` is
   * the node whose signal it is (`docs/01`: a corrector's curve input, or `"input"`); left out, Main's. A tab that changes node calls
   * the function this returns and watches again, which asks at once for the new node.
   */
  watchHistogram(options: { readout?: boolean; at?: NodeId } = {}): () => void {
    const readout = options.readout === true;
    this.histogramWatchers += 1;
    this.histogramAt = options.at;
    if (readout) this.readoutWatchers += 1;
    void this.requestHistogram();
    if (readout) void this.refreshMiddleReadout();
    let watching = true;
    return () => {
      if (!watching) return;
      watching = false;
      this.histogramWatchers -= 1;
      if (readout) this.readoutWatchers -= 1;
      if (this.histogramWatchers === 0 && this.histogramTimer !== null) {
        clearTimeout(this.histogramTimer);
        this.histogramTimer = null;
      }
    };
  }

  /** A frame was drawn: asks for the histogram once the throttle allows, if anybody watches it. */
  private scheduleHistogram(): void {
    if (this.histogramWatchers === 0 || this.histogramTimer !== null || gestureHeld()) return;
    this.histogramTimer = setTimeout(() => {
      this.histogramTimer = null;
      if (this.histogramWatchers === 0 || gestureHeld()) return;
      void this.requestHistogram();
      if (this.readoutWatchers > 0) void this.refreshMiddleReadout();
    }, HISTOGRAM_THROTTLE_MS);
  }

  /** Reads the middle of the cell for the Light tab's caption. One at a time; calls made while one is out collapse into one more. */
  private async refreshMiddleReadout(): Promise<void> {
    if (this.readoutBusy) {
      this.readoutAgain = true;
      return;
    }
    this.readoutBusy = true;
    try {
      do {
        this.readoutAgain = false;
        const asked = this.currentId();
        if (asked === null) break;
        const pixel = await this.samplePixelAt(0.5, 0.5);
        const same = this.state.middleReadout;
        if (pixel && this.currentId() === asked && !(same && same.r === pixel.r && same.g === pixel.g && same.b === pixel.b)) {
          this.set({ middleReadout: pixel });
        }
      } while (this.readoutAgain);
    } finally {
      this.readoutBusy = false;
    }
  }

  // --- The colour warp's scatter --------------------------------------------------------------

  /** Asks for the current photo's colours on the warp's plane. Calls made while one is in flight collapse into one more. */
  async requestChromaScatter(): Promise<void> {
    if (this.currentId() === null) return;
    if (this.scatterBusy) {
      this.scatterAgain = true;
      return;
    }
    this.scatterBusy = true;
    try {
      do {
        this.scatterAgain = false;
        const asked = this.currentId();
        if (asked === null) break;
        const at = this.scatterAt;
        const data = await (at === undefined ? this.api.chromaScatter(asked) : this.api.chromaScatter(asked, at));
        if (this.scatterAt !== at) this.scatterAgain = true;
        else if (this.currentId() === asked) this.set({ chromaScatter: data });
      } while (this.scatterAgain);
    } catch {
      // The dots stay as they were; the next render asks again.
    } finally {
      this.scatterBusy = false;
    }
  }

  /**
   * The colour warp's plane is on show (the Colour tab calls this while it is mounted and calls the function it returns when it goes):
   * the photo's colours are asked for at once, then after the frames the viewport reports, at most every {@link HISTOGRAM_THROTTLE_MS},
   * as the histogram is (`watchHistogram`). Nothing is asked for while nobody watches. `at` is the node whose warp it is (`docs/01`);
   * left out, Main's.
   */
  watchChromaScatter(options: { at?: NodeId } = {}): () => void {
    this.scatterWatchers += 1;
    this.scatterAt = options.at;
    void this.requestChromaScatter();
    let watching = true;
    return () => {
      if (!watching) return;
      watching = false;
      this.scatterWatchers -= 1;
      if (this.scatterWatchers === 0 && this.scatterTimer !== null) {
        clearTimeout(this.scatterTimer);
        this.scatterTimer = null;
      }
    };
  }

  /** A frame was drawn: asks for the scatter once the throttle allows, if anybody watches it. */
  private scheduleChromaScatter(): void {
    if (this.scatterWatchers === 0 || this.scatterTimer !== null || gestureHeld()) return;
    this.scatterTimer = setTimeout(() => {
      this.scatterTimer = null;
      if (this.scatterWatchers > 0 && !gestureHeld()) void this.requestChromaScatter();
    }, HISTOGRAM_THROTTLE_MS);
  }

  /** The last gesture (a drag of a knob, a wheel, a pin) ended: the extras that were held back are asked for once. */
  private afterGesture(): void {
    if (this.histogramWatchers > 0) void this.requestHistogram();
    if (this.readoutWatchers > 0) void this.refreshMiddleReadout();
    if (this.scatterWatchers > 0) void this.requestChromaScatter();
  }

  // --- Undo and redo ------------------------------------------------------------------------

  private historyOf(id: PhotoId): PhotoHistory {
    let history = this.histories.get(id);
    if (!history) {
      history = new PhotoHistory();
      this.histories.set(id, history);
    }
    return history;
  }

  private syncHistoryFlags(): void {
    const id = this.currentId();
    const history = id === null ? undefined : this.histories.get(id);
    // A removal that is the latest thing done can be undone too (⌘Z restores the photos).
    const canUndo = (history?.canUndo ?? false) || this.removalIsLatest();
    const canRedo = history?.canRedo ?? false;
    if (canUndo !== this.state.canUndo || canRedo !== this.state.canRedo) this.set({ canUndo, canRedo });
  }

  /**
   * ⌘Z: takes back the photos of the last removal while nothing was changed after it ({@link undoRemoval}); otherwise undoes the
   * newest setting change of the current photo.
   */
  async undo(): Promise<void> {
    if (this.removalIsLatest()) {
      await this.undoRemoval();
      return;
    }
    await this.stepHistory((h) => h.undo(), "before");
  }

  /** ⇧⌘Z for the current photo. */
  async redo(): Promise<void> {
    await this.stepHistory((h) => h.redo(), "after");
  }

  private async stepHistory(take: (h: PhotoHistory) => HistoryEntry | null, side: "before" | "after"): Promise<void> {
    const id = this.currentId();
    if (id === null) return;
    // Let a drag that is still in flight land first, so its step is on the stack.
    await this.whenIdle(id);
    const history = this.histories.get(id);
    const entry = history ? take(history) : null;
    this.syncHistoryFlags();
    if (!entry) return;
    const rs = this.state.settings[id];
    if (rs) this.setSettings(id, applyDirect(rs, entry[side]));
    await this.enqueue(id, entry[side], null);
  }
}

/** Makes a store for an `Api`. */
export function createStore(api: Api, batchMs?: number): AppStore {
  return new AppStore(api, batchMs);
}
