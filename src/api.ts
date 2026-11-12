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
  ApplyScope,
  AppInfo,
  AutoStatus,
  ChromaSample,
  ChromaScatter,
  ChromaTap,
  CliLinkResult,
  CliLinkStatus,
  CropAspect,
  CropSuggestion,
  DepthFar,
  ExportEstimate,
  ExportJob,
  ExportJobId,
  FaceState,
  FocusPlaneResult,
  HistogramData,
  LensInfo,
  FilmCatalogue,
  LookId,
  Look,
  LookInput,
  LooksImportResult,
  MapFrom,
  MaskFindPrompt,
  MaskFindResult,
  MaskKind,
  MaskView,
  MenuState,
  NodeId,
  OpenKind,
  Orientation,
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
  /**
   * `cancel_open` (`docs/26` H-10): stop the `openPaths` or `openDialog` call that is reading files. It resolves with the files read so
   * far, which stay (`opened.cancelled` is true); the others are dropped. Photos already open are untouched.
   */
  cancelOpen(): Promise<void>;
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
   * `copy_settings` (addition, 2026-10-16): the settings of a photo as a patch to paste onto other photos: the whole graph, every node
   * and every link (`docs/23` §3.8), but the Input's crop, orientation and Pick white point; Auto stays Auto; a white balance that
   * came from a picked white is the manual Kelvin and tint it resolved to. The core stores nothing: the caller keeps the patch (`docs/11` §Copy and paste settings). Rejects for a
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
   * `perf_report` (addition, U-02): the page's stamps of the slider ticks and clicks of the last second, as a string of JSON
   * (`src/perfTimeline.ts`). Sent only when the shell was started with `YM_PERF`; the shell joins it with its own stamps and writes the
   * local log. Nothing leaves the machine.
   */
  perfReport(report: string): Promise<void>;
  /**
   * `histogram`: the RGB histogram of the photo as it is rendered now (Light tab, Curve plot).
   * Addition to `docs/01`; ask again after every `render_stats` event. `at` (`docs/01`, `docs/23` §9) is the node whose signal it
   * is: `"input"` (the Input's output) or a corrector (the input of its curve); left out it is Main's, which is what the simple view
   * reads and never names.
   */
  histogram(photoId: PhotoId, at?: NodeId): Promise<HistogramData>;
  /**
   * `chroma_scatter` (addition, the colour warp of 2026-10-17): the photo's colours on the colour warp's plane, at the warp's input, at
   * most 2048 points, for the faint scatter behind Colour › Colour warp. Ask again after `render_stats`, as for the histogram.
   * `at` is the node whose warp the colours are at the input of (left out: Main's), as for `histogram`.
   */
  chromaScatter(photoId: PhotoId, at?: NodeId): Promise<ChromaScatter>;
  /**
   * `suggest_crop` (Auto crop, 2026-10-22): the straighten angle and up to three frames for the photo at an aspect and orientation,
   * best first (`docs/20-auto-crop.md` §3). The first call for a photo renders and analyses its picture; later ones are quick.
   */
  suggestCrop(photoId: PhotoId, aspect: CropAspect, orientation: Orientation): Promise<CropSuggestion>;

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
  /**
   * `viewport_set_mode`: fit, or a fixed zoom (the one the page mirrors), and the mask view of the node view: a corrector's mask drawn by
   * the core over the photo or alone, `null` (or left out) for the photo (`docs/23` §5.9).
   */
  viewportSetMode(mode: ViewportMode, maskView?: MaskView | null): Promise<void>;
  /** `set_before`: show the photo before the look and display transform. */
  setBefore(before: boolean): Promise<void>;
  /** `sample_pixel`: display values under a point, for the readout and Pick white. */
  samplePixel(x: number, y: number): Promise<PixelSample>;
  /**
   * `sample_chroma` (addition, Pick colour): the chroma of the pixel under a point of the photo cell (CSS pixels from its corner, as Pick
   * white). `null` on the paper around a letterboxed photo. `at` is the node it is read at (left out: Main's), as for `histogram`. `tap` is
   * where in that node: `"warp_input"` (left out) the input of its colour warp, after its own trim and grade, for the warp's Pick colour;
   * `"node_input"` the picture the node reads, before its trim and grade, for a Colour range's (`docs/23` §7.6, §9).
   */
  sampleChroma(x: number, y: number, at?: NodeId, tap?: ChromaTap): Promise<ChromaSample | null>;
  /**
   * `viewport_map_points` (`docs/23` §3.6, §9): maps points between the photo frame and the cell, for the handles and outlines the page draws over
   * the photo. `from` is `"photo"` (points `[x, y]` in the photo frame, answered as CSS pixels from the cell's corner, which may lie outside the
   * cell) or `"cell"` (CSS pixels, answered as photo-frame fractions, `null` for a point on the paper). One answer per point, in order. The page
   * never converts: the straightening, the turn, the crop, the lens's bend and the view's zoom and pan are the core's.
   */
  viewportMapPoints(points: Array<[number, number]>, from: MapFrom): Promise<Array<[number, number] | null>>;
  /**
   * `mask_kinds` (`docs/23` §9): the kinds of mask this machine can make now, in the order of the Add mask menu. The menu lists exactly these.
   */
  maskKinds(): Promise<MaskKind[]>;
  /**
   * `mask_find` (`docs/23` §9): runs SAM for the Object row `mask` of `node` (an index into its `masks.items`) with the person's click or box added.
   * Answers **found**, with what the row's `points` and `box` become and the decoder's `took_ms`, or **nothing found**. It changes no settings: the page
   * sends a found answer as one row patch (one undo step). It waits for the encoder if it is still running. A model that is missing, a mask that is not
   * an Object row and a row with 256 points reject with the scope `mask`.
   */
  maskFind(photoId: PhotoId, node: number, mask: number, prompt: MaskFindPrompt): Promise<MaskFindResult>;
  /**
   * `sample_depth` (`docs/23` §9, Pick depth): the far limit for the depth under a click at `(x, y)` of the photo frame (shares of its width and height), as
   * `{ far }` (the depth plus 0.1, at most 1), or `null` when no depth map can be made. It changes no settings: the page writes the answer as one `masks.edit` row
   * patch (`far`): one undo step. It waits for the depth map if the worker is still making it (the page shows the same "Finding…"). A model that is
   * missing rejects with the scope `mask`.
   */
  sampleDepth(photoId: PhotoId, x: number, y: number): Promise<DepthFar | null>;
  /**
   * `focus_plane` (`docs/23` §9, Focus plane): the far limit for the plane the camera focused on, from the maker note's focus point and the median depth in
   * its box; without a limit, `far` is `null` and `reason` says why (`no_focus_point` is answered at once). It changes no settings: the page writes the limit
   * as one `masks.edit` row patch. A model that is missing rejects with the scope `mask`.
   */
  focusPlane(photoId: PhotoId): Promise<FocusPlaneResult>;
  /**
   * `retry_depth` (`docs/23` §9, "Try again"): starts the depth map of a photo again after it failed. Returns at once: the viewport is busy and finding while
   * the worker runs, and `depth_state` says how it ended. It changes no settings. Nothing happens for a photo whose map did not fail.
   */
  retryDepth(photoId: PhotoId): Promise<void>;
  /**
   * `face_state` (`docs/27` §10, S-11b): where the faces of a photo are, for the panel of a Face Refinement node: `finding`, `ready` with the numbered faces and
   * their boxes in the oriented photo frame (none is a valid answer), `failed` with the sentence, or `models_missing`. It changes nothing and starts nothing
   * (the core finds the faces in the background when a photo with the node is the one looked at); `face_state` events say when it changes.
   */
  faceState(photoId: PhotoId): Promise<FaceState>;
  /**
   * `face_crop` (`docs/27` S-11c): the picture of face `number` (1-based) for the Faces card: the address of a PNG of 104 x 104 px on the `ym-frame` scheme,
   * a square around the face's box (1.4 times its larger side, kept inside the frame) from the picture the faces were found on, which shows the Input's
   * colour without the retouch. Made at the thumbnails' priority and cancelled when the person moves to another photo. Rejects with a plain sentence
   * before the faces are known and for a number the photo does not have.
   */
  faceCrop(photoId: PhotoId, number: number): Promise<string>;

  // --- Lens ----------------------------------------------------------------

  /**
   * `lens_list`: the lenses of the Lens card's list in list order, each with its name, its line and the Flares and Edges the card moves to
   * when it is chosen (`docs/21` §1 and §10). Read once. Empty when the lens data could not be loaded.
   */
  lensList(): Promise<LensInfo[]>;

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
   * `film_catalogue` (`docs/24` §4.1, F-09): the camera films and the still papers the Film node's pickers list, with the credit,
   * disclaimer and trademark lines. Read once. Rejects with scope `render` ("Yellowmoon's film data is missing. Reinstall the app.")
   * when the data cannot be read.
   */
  filmCatalogue(): Promise<FilmCatalogue>;
  /**
   * `film_tile` (`docs/24` §4.4, F-09): the photo as it is developed now with the stock of Film node `node` replaced by `stock`, for
   * the stock picker's tiles. Like `lookThumbnail`: the address of a PNG of at most 240 px on the `ym-frame` scheme, made at the
   * thumbnails' priority and cancelled when the person moves to another photo. A new request is a new address.
   */
  filmTile(photoId: PhotoId, node: number, stock: string): Promise<string>;
  /**
   * `node_thumbnails` (`docs/23` §5.7, §9): asks for the thumbnails of every node of the photo, dead ones included: each is the photo
   * after that node and then through the Output, about 160 px on the long edge. Resolves at once; the pictures that changed since the
   * page was told arrive as a `node_thumbnails_ready` event (an address each, on the `ym-frame` scheme). Ask after a change has landed,
   * never during a drag: the core keeps the pictures by the plan up to each node, so only the nodes a change reaches are made again.
   */
  nodeThumbnails(photoId: PhotoId): Promise<void>;
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
  /**
   * `thumbnails_in_view` (`docs/26` H-11): the filmstrip draws these photos. The shell answers with the address of the thumbnail it holds
   * for each (a photo it has none for is left out) and counts them as used. A photo whose thumbnail the shell let go of to keep its byte
   * budget gets a new one, announced with `thumbnail_ready` like any other; the page keeps no address for it until then.
   */
  thumbnailsInView(ids: PhotoId[]): Promise<ThumbnailReady[]>;

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
   * `auto_status` (`docs/22` sections 11 and 12.6; it replaces `apple_auto_status`): the four providers, which one grades now and whether
   * Preferences shows its "Auto grading" row. The core asks every provider to check its availability again (in the background: the call
   * never waits for a program, and `agy`'s sign-in check contacts its service) and answers what it has cached. Only Preferences opening
   * asks for it (`docs/22` D7); everything else reads `autoStatusCached`. Addition.
   */
  autoStatus(): Promise<AutoStatus>;
  /**
   * `auto_status_cached` (`docs/22` sections 3.3 and 12.6): the same status as each provider last said it, and no program is asked to
   * check anything. The store reads it after a change of the choice, the consent or the instructions and on `auto_status_changed`, so
   * typing instructions never makes a program contact its service. Addition.
   */
  autoStatusCached(): Promise<AutoStatus>;
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
 * An answer that carries patches (`copy_settings`, the steps of `paste_settings`). The generated `NodePatch` allows `null` in
 * every field, because the bridge reads an optional Rust field as `T | null`; the core leaves a field out and writes a `null` only
 * for the Input's `crop` and `white_pick`, which the hand-written `NodePatch` allows too. The screens read the narrower type: this changes the
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
  cancelOpen: () => unit(commands.cancelOpen()),
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
  perfReport: (report) => unit(commands.perfReport(report)),
  histogram: (photoId, at) => commands.histogram(photoId, at ?? null),
  chromaScatter: (photoId, at) => commands.chromaScatter(photoId, at ?? null),
  suggestCrop: (photoId, aspect, orientation) => commands.suggestCrop(photoId, aspect, orientation),

  viewportInfo: () => commands.viewportInfo(),
  viewportSetRect: (rect) => unit(commands.viewportSetRect(rect)),
  viewportInput: (event) => unit(commands.viewportInput(event)),
  viewportSetMode: (mode, maskView) => unit(commands.viewportSetMode(mode, maskView ?? null)),
  setBefore: (before) => unit(commands.setBefore(before)),
  samplePixel: (x, y) => commands.samplePixel(x, y),
  sampleChroma: (x, y, at, tap) => commands.sampleChroma(x, y, at ?? null, tap ?? null),
  viewportMapPoints: (points, from) => commands.viewportMapPoints(points, from),
  maskKinds: () => commands.maskKinds(),
  maskFind: (photoId, node, mask, prompt) => commands.maskFind(photoId, node, mask, prompt),
  sampleDepth: (photoId, x, y) => commands.sampleDepth(photoId, x, y),
  focusPlane: (photoId) => commands.focusPlane(photoId),
  retryDepth: (photoId) => unit(commands.retryDepth(photoId)),
  faceState: (photoId) => commands.faceState(photoId),
  faceCrop: (photoId, number) => commands.faceCrop(photoId, number),

  lensList: () => commands.lensList(),
  looksList: () => commands.looksList(),
  lookThumbnail: (photoId, lookId) => commands.lookThumbnail(photoId, lookId),
  filmCatalogue: () => commands.filmCatalogue(),
  filmTile: (photoId, node, stock) => commands.filmTile(photoId, node, stock),
  nodeThumbnails: (photoId) => unit(commands.nodeThumbnails(photoId)),
  looksImport: (path, input) => commands.looksImport(path, input ?? null),
  looksRecognise: (path) => commands.looksRecognise(path),
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
  thumbnailsInView: (ids) => commands.thumbnailsInView(ids),

  chooseFolder: (start) => commands.chooseFolder(start),
  chooseLookFiles: () => commands.chooseLookFiles(),
  openUrl: (url) => unit(commands.openUrl(url)),
  revealInFinder: (path) => unit(commands.revealInFinder(path)),
  trashPath: (path) => unit(commands.trashPath(path)),

  preferencesGet: () => commands.preferencesGet(),
  preferencesSet: (patch) => commands.preferencesSet(patch),
  autoStatus: () => commands.autoStatus(),
  autoStatusCached: () => commands.autoStatusCached(),
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
