/**
 * The typed bridge to the Rust core (`docs/01-architecture.md` §IPC between webview and core).
 *
 * `Api` has one method per Tauri command and a typed `on()` for the events. Two
 * implementations exist:
 *  - `tauriApi`, which calls the commands `tauri-specta` generated from the Rust side
 *    (`api/bindings.ts`: snake_case command names, camelCase argument keys, rejections are the
 *    core's `AppError`) and listens to events with `listen`;
 *  - the in-memory mock in `api.mock.ts`, which lets the app run in a plain browser, in Vitest
 *    and in Playwright without Tauri.
 *
 * `getApi()` picks one. Every payload type lives in `api/types.ts`; `api/contract.ts` makes `tsc`
 * fail when those types and the generated ones disagree.
 */
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { mockApi } from "./api.mock";
import { commands } from "./api/bindings";
import type * as B from "./api/bindings";
import { isTauri } from "./env";
import { perf } from "./perf";
import type {
  ApiEventName,
  ApiEvents,
  AppError,
  AppleAutoStatus,
  ApplyScope,
  AppInfo,
  ChromaSample,
  ChromaScatter,
  CliLinkResult,
  CliLinkStatus,
  DisplayLut,
  ExportEstimate,
  ExportJob,
  ExportJobId,
  HistogramData,
  LookId,
  Look,
  LookInput,
  LooksImportResult,
  MenuState,
  OpenKind,
  PasteReport,
  PhotoId,
  PixelSample,
  Preferences,
  PreferencesPatch,
  ResolvedSettings,
  SessionChange,
  SessionState,
  SettingsPatch,
  SourceRecognition,
  ThumbnailReady,
  ViewportInfo,
  ViewportInput,
  ViewportMode,
  ViewportRect,
  WatchStatus,
} from "./api/types";

export { isTauri } from "./env";

/** Stops listening to an event. Safe to call more than once. */
export type Unsubscribe = () => void;

/** One method per command in `docs/01`, one `on` for the events. */
export interface Api {
  // --- Session -------------------------------------------------------------

  /** `open_paths`: start a session or add photos. Resolves once the files are probed; development continues in the background (`photo_progress`). */
  openPaths(paths: string[]): Promise<SessionState>;
  /** `open_dialog`: ask the user for files or a folder, then open them. Resolves `null` when the dialog is cancelled. */
  openDialog(kind: OpenKind): Promise<SessionState | null>;
  /** `session_state`: photos, current index, picked set and per-photo summaries. */
  sessionState(): Promise<SessionState>;
  /**
   * `close_session`: forget the session (⌘W, "Start over"). Cancels development that is still
   * running; running exports keep going. Addition to `docs/01`: the Start screen is shown "after
   * closing a session" (`docs/11`), which needs a command.
   */
  closeSession(): Promise<void>;
  /** `set_current`. */
  setCurrent(photoId: PhotoId): Promise<void>;
  /** `set_picked`. */
  setPicked(photoId: PhotoId, picked: boolean): Promise<void>;
  /**
   * `remove_photos` (addition, "Remove from filmstrip"): take photos out of the filmstrip. The files on disk are not touched; the
   * photos' edits go with them (nothing is saved per photo). Queued development and renders for them stop, and the burst frames that
   * stay share their median again (`settings_changed` for those whose estimates moved). The photo on screen hands over to the next
   * one (the previous one when the last was removed). An id that is not in the session is ignored. The last removal is kept in
   * memory for `restorePhotos` until the next removal or the end of the session; exports that were queued keep their photos.
   */
  removePhotos(photoIds: PhotoId[]): Promise<SessionChange>;
  /**
   * `restore_photos` (addition): Undo of the last `removePhotos`. The photos come back where they were in the filmstrip, with their
   * settings, picks and thumbnails (`thumbnail_ready` again). Answers an empty change when there is nothing to put back.
   */
  restorePhotos(): Promise<SessionChange>;

  // --- Settings ------------------------------------------------------------

  /** `get_settings`: the resolved settings with Auto flags and estimates. */
  getSettings(photoId: PhotoId): Promise<ResolvedSettings>;
  /** `update_settings`: apply a patch to one photo and return the resolved settings. */
  updateSettings(photoId: PhotoId, patch: SettingsPatch): Promise<ResolvedSettings>;
  /** `apply_look`: copy a look and its intensity to this photo, the picked photos, or all. */
  applyLook(scope: ApplyScope, lookId: LookId, intensity: number): Promise<void>;
  /** `rerun_auto`: recompute the automatic estimates and reset exposure, white lift, white balance and noise to Auto. */
  rerunAuto(photoId: PhotoId): Promise<ResolvedSettings>;
  /**
   * `copy_settings` (addition, 2026-10-16): the settings of a photo as a patch to paste onto other photos: everything but the crop,
   * the orientation and the Pick white point; Auto stays Auto; a white balance that came from a picked white is the manual Kelvin
   * and tint it resolved to. The core stores nothing: the caller keeps the patch (`docs/11` §Copy and paste settings). Rejects for a
   * photo that is unknown, failed or not developed yet.
   */
  copySettings(photoId: PhotoId): Promise<SettingsPatch>;
  /**
   * `paste_settings` (addition, 2026-10-16): apply a copied patch to every photo in `photoIds` in one call. A photo that is still
   * developing, failed or unknown is skipped. The answer holds each changed photo's before and after patches (one undo step). A
   * look that is no longer in the library is refused, and nothing changes.
   */
  pasteSettings(photoIds: PhotoId[], patch: SettingsPatch): Promise<PasteReport>;
  /** `set_menu_state` (addition, 2026-10-16): which of the Edit menu's settings items are available. */
  setMenuState(state: MenuState): Promise<void>;
  /**
   * `histogram`: the RGB histogram of the photo as it is rendered now (Light tab, Curve plot).
   * Addition to `docs/01`; ask again after every `render_stats` event.
   */
  histogram(photoId: PhotoId): Promise<HistogramData>;
  /**
   * `chroma_scatter` (addition, the colour warp of 2026-10-17): the photo's colours on the colour warp's plane, at the warp's input, at
   * most 2048 points, for the faint scatter behind Colour › Colour warp. Ask again after `render_stats`, as for the histogram.
   */
  chromaScatter(photoId: PhotoId): Promise<ChromaScatter>;

  // --- Native viewport -----------------------------------------------------

  /**
   * `viewport_info` (addition): which backend draws the photo, `native` (a wgpu surface behind the transparent
   * webview) or `fallback` (PNG frames sent as `viewport_frame` events). Resolves `null` when the API has no viewport
   * of its own, as the browser mock does: the screen then draws the mock's fixture picture itself.
   */
  viewportInfo(): Promise<ViewportInfo | null>;
  /** `viewport_set_rect`: where the photo cell is, on mount, resize and layout change. */
  viewportSetRect(rect: ViewportRect): Promise<void>;
  /** `viewport_input`: pointer input over the photo. */
  viewportInput(event: ViewportInput): Promise<void>;
  /** `viewport_set_mode`: fit, or a fixed zoom. */
  viewportSetMode(mode: ViewportMode): Promise<void>;
  /** `set_before`: show the photo before the look and display transform. */
  setBefore(before: boolean): Promise<void>;
  /** `sample_pixel`: display values under a point, for the readout and Pick white. */
  samplePixel(x: number, y: number): Promise<PixelSample>;
  /**
   * `sample_chroma` (addition, the colour warp's Pick colour): the chroma of the pixel under a point of the photo cell (CSS pixels from
   * its corner, as Pick white) at the warp's input. `null` on the paper around a letterboxed photo.
   */
  sampleChroma(x: number, y: number): Promise<ChromaSample | null>;

  // --- Looks ---------------------------------------------------------------

  /** `looks_list`. */
  looksList(): Promise<Look[]>;
  /**
   * `look_thumbnail`: the photo as it is developed now (its exposure, white balance and so on)
   * through one look at 100 %, for the Look tab's tiles (`docs/08` §Look tab behaviour: rendered
   * lazily, cached for the session). The answer is the address of a PNG of at most 240 px on the
   * long edge, served on the `ym-frame` scheme like `thumbnail_ready`'s (`docs/01` IPC); a new
   * request is a new address. Addition to `docs/01`. Rejects with scope `looks` when the look's
   * file is missing.
   */
  lookThumbnail(photoId: PhotoId, lookId: LookId): Promise<string>;
  /**
   * `looks_import`: import a zip, a folder or a `.cube` file. `input` answers the generic question for
   * `.cube` files that match no ARRI package (`docs/08` §Any other .cube); leave it out for a recognised
   * package. Progress arrives as `looks_import_progress` events. Same importer as `ym looks import`.
   */
  looksImport(path: string, input?: LookInput): Promise<LooksImportResult>;
  /**
   * `looks_recognise`: what a zip, a folder or a `.cube` file is, before anything is copied
   * (`docs/08` §Recognition rules). Never rejects for a bad file: it answers with `unusable` or `missing`.
   * Addition to `docs/01`.
   */
  looksRecognise(path: string): Promise<SourceRecognition>;
  /** `looks_display_luts`: the display LUTs imported so far (Preferences › Display rendering, the wizard's Finish step). Addition. */
  looksDisplayLuts(): Promise<DisplayLut[]>;
  /**
   * `looks_watch_downloads`: start or stop the Downloads watcher (FSEvents, `docs/08` §Wizard step 2). Asks
   * macOS for access the first time; `problem` says so when it is refused. Found zips arrive as `download_found`.
   * Addition.
   */
  looksWatchDownloads(enabled: boolean): Promise<WatchStatus>;
  /** `looks_set_default`. */
  looksSetDefault(id: LookId): Promise<void>;
  /** `looks_remove`. */
  looksRemove(id: LookId): Promise<void>;

  // --- Export --------------------------------------------------------------

  /** `export_estimate`: bytes and seconds for the Export panel. */
  exportEstimate(job: ExportJob): Promise<ExportEstimate>;
  /** `export_start`: queue a job; progress arrives as `export_progress`. */
  exportStart(job: ExportJob): Promise<ExportJobId>;
  /** `export_cancel`: stop after the current tile and delete the partial file. */
  exportCancel(jobId: ExportJobId): Promise<void>;
  /** `cli_preview`: the equivalent `ym render …` command. */
  cliPreview(job: ExportJob): Promise<string>;
  /**
   * `export_busy` (addition): an export is queued or running in the core, also one this window did not start
   * (after a reload). "Update ready" waits while it is; the `export_busy` event says when it changes.
   */
  exportBusy(): Promise<boolean>;
  /**
   * `restart_app` (addition): restarts into the update that was installed. Refused with a plain sentence while an export is queued or
   * running (the guard is the core's; the webview's own check, `restartAvailability`, is a courtesy). Resolves only if the app is not
   * replaced.
   */
  restartApp(): Promise<void>;
  /**
   * `thumbnails` (addition): the address of the thumbnail of every photo that has one. The core announces each thumbnail once
   * (`thumbnail_ready`), so a webview that reloaded asks for them again. Only addresses cross the bridge.
   */
  thumbnails(): Promise<ThumbnailReady[]>;

  // --- The system: dialogs, Finder, the browser ------------------------------

  /** `choose_folder`: the system's folder chooser, starting at `start`. Resolves `null` when cancelled. Addition (Save to › Choose…). */
  chooseFolder(start: string | null): Promise<string | null>;
  /** `choose_look_files`: the system's file chooser for zips and `.cube` files, several allowed. Resolves `null` when cancelled. Addition (wizard › Choose file…). */
  chooseLookFiles(): Promise<string[] | null>;
  /** `open_url`: open a page in the system browser (ARRI's page, GitHub). Addition. */
  openUrl(url: string): Promise<void>;
  /** `reveal_in_finder`: show a file or folder in Finder (Export › Show in Finder). Addition. */
  revealInFinder(path: string): Promise<void>;
  /** `trash_path`: move a file to the Trash, recoverable (the wizard's "Move to Trash"; the `trash` crate). Addition. */
  trashPath(path: string): Promise<void>;

  // --- Preferences, About and CLI --------------------------------------------

  /** `preferences_get`. */
  preferencesGet(): Promise<Preferences>;
  /** `preferences_set`: apply a patch and return the new preferences. */
  preferencesSet(patch: PreferencesPatch): Promise<Preferences>;
  /**
   * `apple_auto_status`: whether Apple Auto is available on this computer (read again for the answer) and whether it acts
   * (`docs/19` sections 1 and 2). Preferences asks when it opens and shows "Auto with Apple Intelligence" only while it is available.
   * Addition.
   */
  appleAutoStatus(): Promise<AppleAutoStatus>;
  /** `install_cli_link`: the "Install Command-Line Tool…" menu item. */
  installCliLink(): Promise<CliLinkResult>;
  /** `cli_link_status`: whether the link exists and where (Preferences › Command-line tool). Addition. */
  cliLinkStatus(): Promise<CliLinkStatus>;
  /** `app_info`: version numbers for About. Addition. */
  appInfo(): Promise<AppInfo>;

  // --- Events --------------------------------------------------------------

  /** Listen to a core event. Returns a function that stops listening. */
  on<E extends ApiEventName>(event: E, handler: (payload: ApiEvents[E]) => void): Unsubscribe;
}

/** Narrows a rejected command (or any thrown value) to the core's `AppError`. */
export function isAppError(value: unknown): value is AppError {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as AppError).scope === "string" &&
    typeof (value as AppError).message === "string"
  );
}

/** Turns anything a command can reject with into an `AppError` the UI can show. */
export function toAppError(value: unknown): AppError {
  if (isAppError(value)) return value;
  const message = value instanceof Error ? value.message : typeof value === "string" ? value : "Something went wrong. Try again.";
  return { scope: "app", message };
}

/**
 * An answer that carries patches (`copy_settings`, the steps of `paste_settings`). The generated `SettingsPatch` allows `null` in
 * every field, because the bridge reads an optional Rust field as `T | null`; the core leaves a field out and writes a `null` only
 * for `crop` and `white_pick`, which `Partial<DevelopSettings>` allows too. The screens read the narrower type: this changes the
 * type and nothing at run time.
 */
const narrowed = <T extends B.SettingsPatch | B.PasteReport>(answer: Promise<T>): Promise<T extends B.PasteReport ? PasteReport : SettingsPatch> =>
  answer as unknown as Promise<T extends B.PasteReport ? PasteReport : SettingsPatch>;

/** The unit commands answer `null` on the wire; the interface says `void`. */
const unit = (call: Promise<null>): Promise<void> => call.then(() => undefined);

/**
 * Calls the Rust commands through the generated `commands` (`api/bindings.ts`), so a command that is renamed, or whose arguments or
 * answer change, is a compile error here (the types of `api/types.ts` are checked against the generated ones in `api/contract.ts`).
 */
export const tauriApi: Api = {
  openPaths: (paths) => commands.openPaths(paths),
  openDialog: (kind) => commands.openDialog(kind),
  sessionState: () => commands.sessionState(),
  closeSession: () => unit(commands.closeSession()),
  setCurrent: (photoId) => unit(commands.setCurrent(photoId)),
  setPicked: (photoId, picked) => unit(commands.setPicked(photoId, picked)),
  removePhotos: (photoIds) => commands.removePhotos(photoIds),
  restorePhotos: () => commands.restorePhotos(),

  getSettings: (photoId) => commands.getSettings(photoId),
  updateSettings: (photoId, patch) => commands.updateSettings(photoId, patch),
  applyLook: (scope, lookId, intensity) => unit(commands.applyLook(scope, lookId, intensity)),
  rerunAuto: (photoId) => commands.rerunAuto(photoId),
  copySettings: (photoId) => narrowed(commands.copySettings(photoId)),
  pasteSettings: (photoIds, patch) => narrowed(commands.pasteSettings(photoIds, patch)),
  setMenuState: (state) => unit(commands.setMenuState(state)),
  histogram: (photoId) => commands.histogram(photoId),
  chromaScatter: (photoId) => commands.chromaScatter(photoId),

  viewportInfo: () => commands.viewportInfo(),
  viewportSetRect: (rect) => unit(commands.viewportSetRect(rect)),
  viewportInput: (event) => unit(commands.viewportInput(event)),
  viewportSetMode: (mode) => unit(commands.viewportSetMode(mode)),
  setBefore: (before) => unit(commands.setBefore(before)),
  samplePixel: (x, y) => commands.samplePixel(x, y),
  sampleChroma: (x, y) => commands.sampleChroma(x, y),

  looksList: () => commands.looksList(),
  lookThumbnail: (photoId, lookId) => commands.lookThumbnail(photoId, lookId),
  looksImport: (path, input) => commands.looksImport(path, input ?? null),
  looksRecognise: (path) => commands.looksRecognise(path),
  looksDisplayLuts: () => commands.looksDisplayLuts(),
  looksWatchDownloads: (enabled) => commands.looksWatchDownloads(enabled),
  looksSetDefault: (id) => unit(commands.looksSetDefault(id)),
  looksRemove: (id) => unit(commands.looksRemove(id)),

  exportEstimate: (job) => commands.exportEstimate(job),
  exportStart: (job) => commands.exportStart(job),
  exportCancel: (jobId) => unit(commands.exportCancel(jobId)),
  cliPreview: (job) => commands.cliPreview(job),
  exportBusy: () => commands.exportBusy(),
  restartApp: () => unit(commands.restartApp()),
  thumbnails: () => commands.thumbnails(),

  chooseFolder: (start) => commands.chooseFolder(start),
  chooseLookFiles: () => commands.chooseLookFiles(),
  openUrl: (url) => unit(commands.openUrl(url)),
  revealInFinder: (path) => unit(commands.revealInFinder(path)),
  trashPath: (path) => unit(commands.trashPath(path)),

  preferencesGet: () => commands.preferencesGet(),
  preferencesSet: (patch) => commands.preferencesSet(patch),
  appleAutoStatus: () => commands.appleAutoStatus(),
  installCliLink: () => commands.installCliLink(),
  cliLinkStatus: () => commands.cliLinkStatus(),
  appInfo: () => commands.appInfo(),

  on(event, handler) {
    // `listen` is asynchronous but `on` hands back its unsubscribe at once, so a call to
    // unsubscribe that arrives before the listener is registered still takes effect.
    let unlisten: UnlistenFn | undefined;
    let cancelled = false;
    listen<ApiEvents[typeof event]>(event, (e) => handler(e.payload)).then(
      (fn) => {
        if (cancelled) fn();
        else unlisten = fn;
      },
      () => {
        // Registration failed (for example the window is closing); there is nothing to undo.
      },
    );
    return () => {
      cancelled = true;
      unlisten?.();
      unlisten = undefined;
    };
  },
};

/** The API for this environment: Tauri in the app, the in-memory mock in a browser. */
export function getApi(): Api {
  // With `?perf` the api counts its calls, round trips and events (`src/perf.ts`); otherwise this is the plain api.
  return perf.wrapApi(isTauri() ? tauriApi : mockApi);
}
