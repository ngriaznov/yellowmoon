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
 * Settings changes go out as patches, one at a time per photo. While one is in flight, newer
 * drag patches fold into a single waiting patch, so a slow core is never flooded and the last
 * value always arrives. The reply of a patch replaces the mirrored settings, unless a newer
 * patch is waiting (its reply will carry everything). Final patches (a released drag, a key, a
 * typed value) are recorded for undo as the difference between the settings before the gesture
 * and the core's reply.
 */
import type { Api, Unsubscribe } from "../api";
import { toAppError } from "../api";
import type {
  ApplyScope,
  AppError,
  AppleAutoStatus,
  BalanceSetting,
  ChromaSample,
  ChromaScatter,
  CliLinkStatus,
  ColorBalance,
  ColorBalancePoint,
  DevelopSettings,
  DisplayLut,
  ExportJob,
  ExportOptions,
  ExportProgress,
  HistogramData,
  Look,
  LookId,
  LookInput,
  LookLibrary,
  MenuAction,
  MenuState,
  OpenSummary,
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
  ResolvedSettings,
  SessionChange,
  SessionState,
  SettingsPatch,
  SourceRecognition,
  ThumbnailReady,
  ViewportBackendKind,
  ViewportFrame,
  ViewportInput,
  ViewportMode,
  ViewportRect,
} from "../api/types";
import { filterArwPaths } from "../components/useWindowDrop";
import { clampToDisc, movingPin, WARP_COORD_LIMIT, WARP_MAX_PINS } from "../components/warpMath";
import { isTauri } from "../env";
import { defaultExportOptions, defaultScope, sameOptions } from "../export/options";
import { applyProgress, startRun } from "../export/run";
import { templateProblem } from "../export/template";
import type { ExportRun } from "../export/run";
import { diffSettings, PhotoHistory } from "./history";
import type { HistoryEntry } from "./history";
import { perfCount } from "../perf";
import { isReady, mergeSession, photoName, releaseUrl, removedNotice, restoredNotice, withoutPhotos, withRestoredPhotos } from "./photo";
import { RequestLimiter } from "./requestLimiter";
import { ViewportInputQueue } from "./viewportInputQueue";
import type { StreamInput } from "./viewportInputQueue";
import { fitZoomExact, fitZoomOf, showsWhole, zoomAfter } from "./zoom";

/** The screens of the app (`docs/11-ui.md` §Screens that have a window of their own). */
export type Screen = "start" | "photos" | "import-looks";

/** What Photos shows in the right-hand panel: the six tool tabs, or Export (`docs/11-ui.md` §Export). */
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
  /** What `copy_settings` answered: everything but the crop, the orientation and the Pick white point. */
  patch: SettingsPatch;
}

/** Everything the screens draw. Replaced, never changed in place. */
export interface AppState {
  screen: Screen;
  /** Where "Back" goes from a full-window detour such as the import wizard. */
  returnTo: Screen;
  /** The core's session; null before anything is open. */
  session: SessionState | null;
  /** Thumbnail URLs by photo id, from `thumbnail_ready`. */
  thumbnails: Readonly<Record<PhotoId, string>>;
  /** Resolved settings by photo id, for the photos that were shown. */
  settings: Readonly<Record<PhotoId, ResolvedSettings>>;
  /** The look library; null until it has been asked for. */
  looks: readonly Look[] | null;
  /** The last `render_stats`: "Neural · 42 ms · Display P3". */
  stats: RenderStats | null;
  /** Showing the photo before the look and display transform. */
  before: boolean;
  zoom: ViewportMode;
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
  /** Undo and redo are available for the current photo. */
  canUndo: boolean;
  canRedo: boolean;
  /** The preferences, once loaded (`docs/01` §Persistence). */
  preferences: Preferences | null;
  /** Imported display LUTs, once asked for. */
  displayLuts: readonly DisplayLut[] | null;
  /** Whether the command-line tool is linked onto the PATH, once asked for (Preferences). */
  cliLink: CliLinkStatus | null;
  /**
   * Whether Apple Auto is available on this computer and acts (`apple_auto_status`, `docs/19`), asked for each time Preferences opens;
   * null before the first answer. Preferences shows "Auto with Apple Intelligence" only while it is available.
   */
  appleAuto: AppleAutoStatus | null;
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
    settings: {},
    looks: null,
    stats: null,
    before: false,
    zoom: { kind: "fit" },
    applyScope: "this",
    collapsedLookLibraries: [],
    dropNotice: null,
    error: null,
    opening: false,
    histogram: null,
    middleReadout: null,
    chromaScatter: null,
    warpPin: null,
    canUndo: false,
    canRedo: false,
    preferences: null,
    displayLuts: null,
    cliLink: null,
    appleAuto: null,
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
    copied: null,
  };
}

/** Settings whose resolved value is the stored value, so a patch can be shown at once. */
const DIRECT_KEYS: ReadonlySet<string> = new Set([
  "look",
  "look_intensity",
  "brightness",
  "white",
  "white_offset_ev",
  "light_source",
  "color_warp",
  "output_sharpening",
  "curve",
  "decode",
  "rolloff",
  "crop",
  "orientation",
  "white_pick",
]);

/** The settings that are Auto or a value from −100 to 100 (`docs/19` section 5), each resolved against the estimate of the same name. */
const PERCENT_KEYS: ReadonlySet<string> = new Set(["contrast", "saturation", "highlights", "shadows", "black"]);

/**
 * The mirrored settings with a patch applied to the fields that need no resolving. Contrast, saturation, highlights, shadows, black
 * and the shadows and highlights wheels are Auto or a value (`docs/19` section 5) and need none either: Auto is the photo's Auto value,
 * which the mirror already has (`estimates`).
 */
export function applyDirect(rs: ResolvedSettings, patch: SettingsPatch): ResolvedSettings {
  const settings = { ...rs.settings } as Record<string, unknown>;
  const values = { ...rs.values } as Record<string, unknown>;
  let auto = rs.auto;
  for (const [key, value] of Object.entries(patch)) {
    if (value !== undefined && PERCENT_KEYS.has(key)) {
      const field = key as "contrast" | "saturation" | "highlights" | "shadows" | "black";
      const setting = value as PercentSetting;
      settings[field] = setting;
      values[field] = setting.kind === "manual" ? setting.value : rs.estimates[field];
      auto = { ...auto, [field]: setting.kind === "auto" };
      continue;
    }
    if (value !== undefined && key === "color_balance") {
      const cb = { ...rs.settings.color_balance, ...(value as Partial<ColorBalance>) };
      const point = (w: BalanceSetting, est: ColorBalancePoint): ColorBalancePoint => (w.kind === "manual" ? w.point : est);
      settings.color_balance = cb;
      values.color_balance = {
        shadows: point(cb.shadows, rs.estimates.balance_shadows),
        midtones: cb.midtones,
        highlights: point(cb.highlights, rs.estimates.balance_highlights),
      };
      auto = { ...auto, balance_shadows: cb.shadows.kind === "auto", balance_highlights: cb.highlights.kind === "auto" };
      continue;
    }
    if (value === undefined || !DIRECT_KEYS.has(key)) continue;
    settings[key] = value;
    values[key] = value;
    if (key === "light_source") auto = { ...auto, light_source: value === "auto" };
    // The lift in effect follows the offset at once: the automatic amount plus the offset, never below 0 (the core's rule).
    if (key === "white_offset_ev") values.white_ev = Math.min(3, Math.max(0, rs.estimates.white_ev + (value as number)));
  }
  return { ...rs, settings: settings as unknown as DevelopSettings, values: values as unknown as ResolvedSettings["values"], auto };
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
  private rect: ViewportRect | null = null;
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
  /** Who has the histogram on show (the Light and Curve tabs): while there is one, `render_stats` keeps it fresh. */
  private histogramWatchers = 0;
  /** Of those, who also show the middle readout (the Light tab). */
  private readoutWatchers = 0;
  private histogramTimer: ReturnType<typeof setTimeout> | null = null;
  private readoutBusy = false;
  private readoutAgain = false;
  private scatterBusy = false;
  private scatterAgain = false;
  /** Who shows the colour warp's plane (the Colour tab): while there is one, `render_stats` keeps its scatter fresh. */
  private scatterWatchers = 0;
  private scatterTimer: ReturnType<typeof setTimeout> | null = null;
  /** Export events that arrived before `export_start` answered, by job. */
  private readonly earlyExport: ExportProgress[] = [];
  private starting = false;
  /** Runs a menu item that needs the webview (`menu_action`); set by `App`, which knows the screens. */
  menuHandler: ((action: MenuAction) => void) | null = null;

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
    this.state = { ...this.state, ...patch };
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
      api.on("thumbnail_ready", (e) => this.onThumbnail(e)),
      api.on("render_stats", (stats) => {
        this.later((s) => ({ ...s, stats }));
        this.scheduleHistogram();
        this.scheduleChromaScatter();
      }),
      api.on("looks_changed", () => {
        void this.refreshLooks();
        void this.refreshDisplayLuts();
        void this.refreshPreferences();
      }),
      api.on("error", (error) => this.set({ error })),
      api.on("export_progress", (e) => this.onExportProgress(e)),
      api.on("menu_action", (e) => this.menuHandler?.(e.action)),
      api.on("viewport_frame", (frame) => this.onFrame(frame)),
      api.on("settings_changed", (e) => void this.onSettingsChanged(e.photo_id)),
      api.on("export_busy", (e) => this.set({ exportBusy: e.busy })),
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
      this.refreshSession(),
      this.refreshPreferences(),
      this.refreshDisplayLuts(),
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
    if (!session || !session.photos.some((p) => p.id === id)) {
      this.pendingStatus.set(id, status);
      return s;
    }
    const photos = session.photos.map((p) => (p.id === id ? { ...p, status } : p));
    if (status.kind === "ready" && photos[session.current_index ?? -1]?.id === id) {
      queueMicrotask(() => void this.ensureSettings(id));
    }
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

  /**
   * A frame of the fallback viewport, or (an empty address) the news that the picture is gone. Drawn at once, never batched. The shell
   * can draw a frame before the webview has said where the cell is (it uses a 1 × 1 cell then); a frame that small is no picture.
   */
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
      if (addingToPhotos && session.opened) this.showNotice(openedNotice(session.opened));
    } catch (error) {
      const failure = toAppError(error);
      if (failure.scope === "open" && failure.message === NO_ARW_NOTICE) this.sayNoArw(addingToPhotos);
      else this.fail(failure);
    } finally {
      this.set({ opening: false });
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
        if (addingToPhotos && session.opened) this.showNotice(openedNotice(session.opened));
      }
    } catch (error) {
      this.fail(error);
    } finally {
      this.set({ opening: false });
    }
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
    if (onPhotos) this.set({ dropNotice: null, notice: NO_ARW_NOTICE, noticeAction: null });
    else this.set({ dropNotice: NO_ARW_NOTICE });
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
      applyScope: this.state.applyScope,
      collapsedLookLibraries: this.state.collapsedLookLibraries,
      stats: null,
      // What belongs to the app rather than the session stays: the preferences, the export in flight, the dialogs.
      preferences: this.state.preferences,
      displayLuts: this.state.displayLuts,
      cliLink: this.state.cliLink,
      appleAuto: this.state.appleAuto,
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
    // A fresh session starts at Fit with the camera view off.
    void this.api.viewportSetMode({ kind: "fit" }).catch(() => {});
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
    this.set({
      session: { ...session, current_index: index },
      histogram: null,
      middleReadout: null,
      chromaScatter: null,
      warpPin: null,
      zoom: { kind: "fit" },
      before: false,
      stats,
    });
    this.syncHistoryFlags();
    if (isReady(session.photos[index] as PhotoSummary)) void this.ensureSettings(id);
    try {
      await this.api.setCurrent(id);
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
    this.set({ session: next, thumbnails, settings, ...notice, ...(stays ? {} : this.currentChanged()) });
    this.syncHistoryFlags();
    const shown = next.photos[next.current_index ?? 0];
    if (!stays && shown && isReady(shown)) void this.ensureSettings(shown.id);
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
    return {
      histogram: null,
      middleReadout: null,
      chromaScatter: null,
      warpPin: null,
      zoom: { kind: "fit" },
      before: false,
      stats: this.state.stats && { ...this.state.stats, zoom_percent: null, fit_percent: null },
    };
  }

  /** The Look tab's tiles of photos that left are let go (the shell's addresses are not ours to free; the mock's `blob:` ones are). */
  private forgetLookThumbsOf(ids: ReadonlySet<PhotoId>): void {
    this.thumbLimiter.supersede((t) => ids.has(t.photoId));
    for (const id of ids) {
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

  async refreshDisplayLuts(): Promise<void> {
    try {
      this.set({ displayLuts: await this.api.looksDisplayLuts() });
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
      // Apple Auto acts, or stops, with its preference, and asks every photo again with new instructions; the photos' own news comes
      // as `settings_changed`.
      if (patch.apple_auto !== undefined || patch.apple_auto_instructions !== undefined) void this.refreshAppleAuto();
    } catch (error) {
      if (before) this.set({ preferences: before });
      this.fail(error);
    }
  }

  /**
   * Asks whether Apple Auto is available on this computer (the core reads it again: Apple Intelligence may have been switched on, or its
   * model finished downloading) and whether it acts.
   */
  async refreshAppleAuto(): Promise<void> {
    try {
      this.set({ appleAuto: await this.api.appleAutoStatus() });
    } catch (error) {
      this.fail(error);
    }
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
      void this.refreshDisplayLuts();
      void this.refreshAppleAuto();
    }
  }

  closeOverlay(): void {
    if (this.state.overlay === "import-cube") this.set({ overlay: null, cubeImport: null });
    else this.set({ overlay: null });
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
    this.set({ before });
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
    const estimate = this.rect && photo ? fitZoomExact(this.rect, photo, this.currentSettings()?.values ?? null) : null;
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
   * Pick colour (Colour › Colour warp, `docs/11`): asks the core for the chroma under a click on the photo cell (CSS pixels from its
   * corner, as Pick white) at the warp's input, and adds a moving pin there with its target on its source, selected, so the next drag
   * on the plane moves it. One undo step. A click on the paper around the photo adds nothing, and nor does one at sixteen pins (the
   * selection goes, so the card says why).
   */
  async pickColor(x: number, y: number): Promise<void> {
    const id = this.currentId();
    if (id === null) return;
    // Wheel and drag input still on its way lands first: the click names a spot of the picture as that input left it.
    if (this.viewportQueue.busy) await this.viewportQueue.flush();
    // Patches in flight land first, so the new pin joins the pins the core has.
    await this.whenIdle(id);
    let sample: ChromaSample | null;
    try {
      sample = await this.api.sampleChroma(x, y);
    } catch (error) {
      this.fail(error);
      return;
    }
    const rs = this.state.settings[id];
    if (!sample || !rs || this.currentId() !== id) return;
    const pins = rs.settings.color_warp.pins;
    if (pins.length >= WARP_MAX_PINS) {
      this.selectWarpPin(null);
      return;
    }
    const next = [...pins, movingPin(clampToDisc([sample.a, sample.b], WARP_COORD_LIMIT))];
    this.selectWarpPin(next.length - 1);
    await this.update({ color_warp: { pins: next } }, true);
  }

  /** Selects a pin of the colour warp (an index into the current photo's pins), or none. */
  selectWarpPin(index: number | null): void {
    if (index !== this.state.warpPin) this.set({ warpPin: index });
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
        if (!this.state.settings[id]) this.set({ settings: { ...this.state.settings, [id]: rs } });
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

  private setSettings(id: PhotoId, rs: ResolvedSettings): void {
    this.wrote(id);
    this.set({ settings: { ...this.state.settings, [id]: rs } });
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
        last.patch = { ...last.patch, ...patch };
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
        try {
          const reply = await this.api.updateSettings(id, item.patch);
          if (item.record) {
            const entry = diffSettings(item.record.base, reply.settings);
            if (entry) this.historyOf(id).push(entry);
          }
          // A newer patch is waiting: its reply will carry everything, so this one is skipped.
          if (sender.queue.length === 0 && this.state.settings[id]) this.setSettings(id, reply);
        } catch (error) {
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
   * Copy settings (⇧⌘C): keeps the current photo's settings for pasting: everything but the crop, the orientation and the Pick white
   * point, Auto kept as Auto, a white balance that came from a picked white as the manual Kelvin and tint it resolved to. The core
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
    const changed = report.pasted.filter((p) => Object.keys(p.before).length > 0 || Object.keys(p.after).length > 0);
    for (const step of changed) {
      this.historyOf(step.id).push({ before: step.before, after: step.after });
      // A crop or a slider drag that was open on the photo when the paste landed records its own step when it ends, from the settings it
      // began with. Those must include the paste, or the step would carry it too and one Undo would take back both.
      const gesture = this.gestures.get(step.id);
      if (gesture) gesture.base = { ...gesture.base, ...step.after };
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
    if (id === null || copied.patch.white_balance?.kind !== "manual") return null;
    const photo = this.state.session?.photos.find((p) => p.id === id);
    if (!photo || !isReady(photo)) return null;
    await this.ensureSettings(id);
    const settings = this.state.settings[id]?.settings;
    const pickedWhite = settings !== undefined && settings.white_pick !== null && settings.light_source === "auto" && settings.white_balance.kind === "auto";
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
    const level = intensity ?? rs.values.look_intensity;
    await this.changeLook(photoId, this.state.applyScope, { look: lookId, look_intensity: level }, true, lookId, level);
  }

  /**
   * Sets the look strength of a photo (the one on screen unless `photoId` names another); the other photos of the scope get it, with
   * the look the photo has, when the drag ends. A drag names the photo it began on, as {@link update} asks of every control.
   */
  async setLookIntensity(level: number, final: boolean, photoId: PhotoId | null = this.currentId()): Promise<void> {
    if (photoId === null) return;
    const rs = this.state.settings[photoId];
    if (!rs) return;
    await this.changeLook(photoId, this.state.applyScope, { look_intensity: level }, final, rs.values.look, level);
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

  private releaseLookThumbs(): void {
    this.thumbLimiter.supersede(() => true);
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
        const data = await this.api.histogram(asked);
        if (this.currentId() === asked) this.set({ histogram: data });
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
   * also draws the viewport, so asking per frame made each frame wait for the plot. Nothing is asked for while nobody watches.
   */
  watchHistogram(options: { readout?: boolean } = {}): () => void {
    const readout = options.readout === true;
    this.histogramWatchers += 1;
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
    if (this.histogramWatchers === 0 || this.histogramTimer !== null) return;
    this.histogramTimer = setTimeout(() => {
      this.histogramTimer = null;
      if (this.histogramWatchers === 0) return;
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
        const data = await this.api.chromaScatter(asked);
        if (this.currentId() === asked) this.set({ chromaScatter: data });
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
   * as the histogram is (`watchHistogram`). Nothing is asked for while nobody watches.
   */
  watchChromaScatter(): () => void {
    this.scatterWatchers += 1;
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
    if (this.scatterWatchers === 0 || this.scatterTimer !== null) return;
    this.scatterTimer = setTimeout(() => {
      this.scatterTimer = null;
      if (this.scatterWatchers > 0) void this.requestChromaScatter();
    }, HISTOGRAM_THROTTLE_MS);
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
