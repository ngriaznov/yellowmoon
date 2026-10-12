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
  CliLinkStatus,
  DevelopSettings,
  DisplayLut,
  ExportJob,
  ExportOptions,
  ExportProgress,
  HistogramData,
  Look,
  LookId,
  LookInput,
  MenuAction,
  PhotoId,
  PhotoProgress,
  PhotoStatus,
  PhotoSummary,
  PixelSample,
  Preferences,
  PreferencesPatch,
  RenderStats,
  ResolvedSettings,
  SessionState,
  SettingsPatch,
  SourceRecognition,
  ThumbnailReady,
  ViewportInput,
  ViewportMode,
  ViewportRect,
} from "../api/types";
import { filterArwPaths } from "../components/useWindowDrop";
import { defaultExportOptions, defaultScope, sameOptions } from "../export/options";
import { applyProgress, startRun } from "../export/run";
import { templateProblem } from "../export/template";
import type { ExportRun } from "../export/run";
import { diffSettings, PhotoHistory } from "./history";
import type { HistoryEntry } from "./history";
import { bytesToUrl, isReady, releaseUrl } from "./photo";

/** The screens of the app (`docs/11-ui.md` §Screens that have a window of their own). */
export type Screen = "start" | "photos" | "import-looks";

/** What Photos shows in the right-hand panel: the six tool tabs, or Export (`docs/11-ui.md` §Export). */
export type Panel = "tools" | "export";

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
  /** A problem with the last drop, in words ("No ARW files in what you dropped."). */
  dropNotice: string | null;
  /** The last error from the core, until dismissed. */
  error: AppError | null;
  /** Open files or Open folder is waiting for the core or the dialog. */
  opening: boolean;
  /** The current photo's histogram, when asked for. */
  histogram: HistogramData | null;
  /** Undo and redo are available for the current photo. */
  canUndo: boolean;
  canRedo: boolean;
  /** The preferences, once loaded (`docs/01` §Persistence). */
  preferences: Preferences | null;
  /** Imported display LUTs, once asked for. */
  displayLuts: readonly DisplayLut[] | null;
  /** Whether the command-line tool is linked onto the PATH, once asked for (Preferences). */
  cliLink: CliLinkStatus | null;
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
    applyScope: "all",
    dropNotice: null,
    error: null,
    opening: false,
    histogram: null,
    canUndo: false,
    canRedo: false,
    preferences: null,
    displayLuts: null,
    cliLink: null,
    panel: "tools",
    exportOptions: defaultExportOptions(),
    exportScope: "this",
    exportRun: null,
    exportStopped: null,
    overlay: null,
    cubeImport: null,
    notice: null,
  };
}

/** Settings whose resolved value is the stored value, so a patch can be shown at once. */
const DIRECT_KEYS: ReadonlySet<string> = new Set([
  "look",
  "look_intensity",
  "contrast",
  "brightness",
  "highlights",
  "shadows",
  "white",
  "black",
  "light_source",
  "color_balance",
  "saturation",
  "curve",
  "decode",
  "rolloff",
  "crop",
  "orientation",
  "white_pick",
]);

/** The mirrored settings with a patch applied to the fields that need no resolving. */
export function applyDirect(rs: ResolvedSettings, patch: SettingsPatch): ResolvedSettings {
  const settings = { ...rs.settings } as Record<string, unknown>;
  const values = { ...rs.values } as Record<string, unknown>;
  let auto = rs.auto;
  for (const [key, value] of Object.entries(patch)) {
    if (value === undefined || !DIRECT_KEYS.has(key)) continue;
    settings[key] = value;
    values[key] = value;
    if (key === "light_source") auto = { ...auto, light_source: value === "auto" };
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

/** Milliseconds a replaced look thumbnail's URL is kept, so a tile never shows a broken picture. */
const URL_GRACE_MS = 3000;

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
  private readonly lookThumbs = new Map<PhotoId, { key: string; urls: Map<LookId, Promise<string>> }>();
  private rect: ViewportRect | null = null;
  private histogramBusy = false;
  private histogramAgain = false;
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
    for (const listener of [...this.listeners]) listener();
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
      for (const listener of [...this.listeners]) listener();
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
      api.on("render_stats", (stats) => this.later((s) => ({ ...s, stats }))),
      api.on("looks_changed", () => {
        void this.refreshLooks();
        void this.refreshDisplayLuts();
        void this.refreshPreferences();
      }),
      api.on("error", (error) => this.set({ error })),
      api.on("export_progress", (e) => this.onExportProgress(e)),
      api.on("menu_action", (e) => this.menuHandler?.(e.action)),
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
    await Promise.all([this.refreshLooks(), this.refreshSession(), this.refreshPreferences(), this.refreshDisplayLuts()]);
  }

  private async refreshSession(): Promise<void> {
    try {
      const session = await this.api.sessionState();
      if (session.photos.length > 0) this.adoptSession(session, true);
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
    const url = bytesToUrl(e.png_bytes);
    this.later((s) => {
      const old = s.thumbnails[e.id];
      if (old) releaseUrl(old);
      return { ...s, thumbnails: { ...s.thumbnails, [e.id]: url } };
    });
  }

  // --- Session ------------------------------------------------------------------------------

  private adoptSession(session: SessionState, restoring = false): void {
    let next: AppState = { ...this.state, session };
    for (const [id, status] of this.pendingStatus) next = this.withStatus(next, id, status);
    this.pendingStatus.clear();
    const photos = next.session?.photos ?? [];
    if (restoring && photos.some(isReady)) next = { ...next, screen: "photos" };
    this.state = next;
    for (const listener of [...this.listeners]) listener();
    const current = this.currentPhoto();
    if (current && isReady(current)) void this.ensureSettings(current.id);
    this.syncHistoryFlags();
  }

  /** Opens dropped or chosen paths. Anything that isn't an ARW file or a folder is ignored. */
  async openPaths(paths: string[]): Promise<void> {
    const { accepted } = filterArwPaths(paths);
    if (accepted.length === 0) {
      this.set({ dropNotice: NO_ARW_NOTICE });
      return;
    }
    this.set({ dropNotice: null, opening: true });
    try {
      const session = await this.api.openPaths(accepted);
      this.adoptSession(session);
    } catch (error) {
      this.fail(error);
    } finally {
      this.set({ opening: false });
    }
  }

  /** Asks for files or a folder with the system dialog. A cancelled dialog changes nothing. */
  async openDialog(kind: "files" | "folder"): Promise<void> {
    this.set({ dropNotice: null, opening: true });
    try {
      const session = await this.api.openDialog(kind);
      if (session) this.adoptSession(session);
    } catch (error) {
      this.fail(error);
    } finally {
      this.set({ opening: false });
    }
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
    try {
      await this.api.closeSession();
    } catch (error) {
      this.fail(error);
      return;
    }
    for (const url of Object.values(this.state.thumbnails)) releaseUrl(url);
    this.releaseLookThumbs();
    this.pendingStatus.clear();
    this.histories.clear();
    this.gestures.clear();
    this.senders.clear();
    this.settingsRequests.clear();
    this.batch = [];
    this.state = {
      ...initialState(),
      looks: this.state.looks,
      applyScope: this.state.applyScope,
      stats: null,
      // What belongs to the app rather than the session stays: the preferences, the export in flight, the dialogs.
      preferences: this.state.preferences,
      displayLuts: this.state.displayLuts,
      cliLink: this.state.cliLink,
      exportOptions: this.state.exportOptions,
      // A run still writing files outlives the session; a finished one has been seen.
      exportRun: this.state.exportRun?.phase === "running" ? this.state.exportRun : null,
      overlay: this.state.overlay === "confirm-close" ? null : this.state.overlay,
      cubeImport: this.state.cubeImport,
      notice: this.state.notice,
    };
    for (const listener of [...this.listeners]) listener();
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
    this.set({ session: { ...session, current_index: index }, histogram: null });
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
    } catch (error) {
      if (before) this.set({ preferences: before });
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

  /** "Install Command-Line Tool…": links `yellowmoon` onto the PATH and says where. */
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
    }
  }

  closeOverlay(): void {
    if (this.state.overlay === "import-cube") this.set({ overlay: null, cubeImport: null });
    else this.set({ overlay: null });
  }

  /** A short message that is not an error. */
  showNotice(notice: string): void {
    this.set({ notice });
  }

  dismissNotice(): void {
    this.set({ notice: null });
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
   * input (`docs/08` §Any other .cube). A cancelled chooser changes nothing.
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
    const todo = dialog.items.filter((i) => i.found.recognition.kind === "package" || (i.found.recognition.kind === "needs_answer" && i.answer));
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

  private onExportProgress(e: ExportProgress): void {
    const run = this.state.exportRun;
    if (!run) {
      if (this.starting) this.earlyExport.push(e);
      return;
    }
    const next = applyProgress(run, e);
    if (next !== run) this.set({ exportRun: next });
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
    const previous = this.state.zoom;
    this.set({ zoom: mode });
    try {
      await this.api.viewportSetMode(mode);
    } catch (error) {
      this.set({ zoom: previous });
      this.fail(error);
    }
  }

  /** Forwards pointer input over the photo. A double click also flips the mirrored zoom between Fit and 100 %. */
  async viewportInput(event: ViewportInput): Promise<void> {
    if (event.kind === "toggle_zoom") {
      const zoom: ViewportMode = this.state.zoom.kind === "fit" ? { kind: "percent", value: 100 } : { kind: "fit" };
      this.set({ zoom });
    }
    try {
      await this.api.viewportInput(event);
    } catch (error) {
      this.fail(error);
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
    const request = this.api
      .getSettings(id)
      .then((rs) => {
        if (!this.state.settings[id]) this.set({ settings: { ...this.state.settings, [id]: rs } });
      })
      .catch((error) => this.fail(error))
      .finally(() => {
        this.settingsRequests.delete(id);
      });
    this.settingsRequests.set(id, request);
    return request;
  }

  /** Throws away what is known about a photo's settings and asks again. */
  private async resyncSettings(id: PhotoId): Promise<void> {
    try {
      const rs = await this.api.getSettings(id);
      this.set({ settings: { ...this.state.settings, [id]: rs } });
    } catch (error) {
      this.fail(error);
    }
  }

  private setSettings(id: PhotoId, rs: ResolvedSettings): void {
    this.set({ settings: { ...this.state.settings, [id]: rs } });
  }

  /**
   * Sends a patch for the current photo. While a drag is on, call it with `final = false` (at
   * most once per frame, which the controls do); the release, a key, a typed value or a reset
   * calls it with `final = true`, which also records one undo step for the whole gesture.
   * Resolves when the core has answered.
   */
  update(patch: SettingsPatch, final = true): Promise<void> {
    const id = this.currentId();
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

  // --- Looks --------------------------------------------------------------------------------

  /**
   * Picks a look for the current photo and, when "Apply to" is Picked or All, for those photos
   * too. The current photo goes first through `update_settings`, so the change is one undo step
   * and the viewport answers at once; the others follow through `apply_look`.
   */
  async chooseLook(lookId: LookId, intensity?: number): Promise<void> {
    const rs = this.currentSettings();
    const look = this.state.looks?.find((l) => l.id === lookId);
    if (!rs || !look || look.missing) return;
    const level = intensity ?? rs.values.look_intensity;
    const failed = this.failures;
    await this.update({ look: lookId, look_intensity: level }, true);
    // If the current photo could not take the look, the others must not get it either.
    if (this.failures === failed) await this.copyLook(lookId, level);
  }

  /** Sets the look strength; the other photos of the scope get it when the drag ends. */
  async setLookIntensity(level: number, final: boolean): Promise<void> {
    const rs = this.currentSettings();
    if (!rs) return;
    const failed = this.failures;
    await this.update({ look_intensity: level }, final);
    if (final && this.failures === failed) await this.copyLook(rs.values.look, level);
  }

  private async copyLook(lookId: LookId, level: number): Promise<void> {
    const scope = this.state.applyScope;
    if (scope === "this") return;
    const current = this.currentId();
    try {
      await this.api.applyLook(scope, lookId, level);
    } catch (error) {
      this.fail(error);
      return;
    }
    // The other photos' looks changed behind the mirror's back.
    const kept: Record<PhotoId, ResolvedSettings> = {};
    for (const [key, value] of Object.entries(this.state.settings)) if (Number(key) === current) kept[Number(key)] = value;
    this.set({ settings: kept });
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
   * it changes. Thumbnails are kept for the session.
   */
  lookThumbnail(photoId: PhotoId, lookId: LookId, key: string): Promise<string> {
    let entry = this.lookThumbs.get(photoId);
    if (!entry || entry.key !== key) {
      if (entry) this.releaseLater(entry.urls);
      entry = { key, urls: new Map() };
      this.lookThumbs.set(photoId, entry);
    }
    const cache = entry.urls;
    let url = cache.get(lookId);
    if (!url) {
      url = this.api.lookThumbnail(photoId, lookId).then(bytesToUrl);
      cache.set(lookId, url);
      url.catch(() => cache.delete(lookId));
    }
    return url;
  }

  private releaseLater(urls: Map<LookId, Promise<string>>): void {
    const pending = [...urls.values()];
    setTimeout(() => {
      for (const url of pending) void url.then(releaseUrl, () => {});
    }, URL_GRACE_MS);
  }

  private releaseLookThumbs(): void {
    for (const entry of this.lookThumbs.values()) for (const url of entry.urls.values()) void url.then(releaseUrl, () => {});
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
    const canUndo = history?.canUndo ?? false;
    const canRedo = history?.canRedo ?? false;
    if (canUndo !== this.state.canUndo || canRedo !== this.state.canRedo) this.set({ canUndo, canRedo });
  }

  /** ⌘Z for the current photo. */
  async undo(): Promise<void> {
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
