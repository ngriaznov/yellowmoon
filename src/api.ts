/**
 * The typed bridge to the Rust core (`docs/01-architecture.md` §IPC between webview and core).
 *
 * `Api` has one method per Tauri command and a typed `on()` for the events. Two
 * implementations exist:
 *  - `tauriApi`, which calls `invoke` / `listen` (snake_case command names, camelCase args);
 *  - the in-memory mock in `api.mock.ts`, which lets the app run in a plain browser, in Vitest
 *    and in Playwright without Tauri.
 *
 * `getApi()` picks one. Every payload type lives in `api/types.ts`, so generated bindings
 * (tauri-specta, task M3-02) can replace that file without touching the code here.
 */
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { mockApi } from "./api.mock";
import { isTauri } from "./env";
import type {
  ApiEventName,
  ApiEvents,
  AppError,
  ApplyScope,
  CliLinkResult,
  ExportEstimate,
  ExportJob,
  ExportJobId,
  LookId,
  Look,
  LooksImportResult,
  OpenKind,
  PhotoId,
  PixelSample,
  Preferences,
  PreferencesPatch,
  ResolvedSettings,
  SessionState,
  SettingsPatch,
  ViewportInput,
  ViewportMode,
  ViewportRect,
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
  /** `set_current`. */
  setCurrent(photoId: PhotoId): Promise<void>;
  /** `set_picked`. */
  setPicked(photoId: PhotoId, picked: boolean): Promise<void>;

  // --- Settings ------------------------------------------------------------

  /** `get_settings`: the resolved settings with Auto flags and estimates. */
  getSettings(photoId: PhotoId): Promise<ResolvedSettings>;
  /** `update_settings`: apply a patch to one photo and return the resolved settings. */
  updateSettings(photoId: PhotoId, patch: SettingsPatch): Promise<ResolvedSettings>;
  /** `apply_look`: copy a look and its intensity to this photo, the picked photos, or all. */
  applyLook(scope: ApplyScope, lookId: LookId, intensity: number): Promise<void>;
  /** `rerun_auto`: recompute the automatic estimates and reset exposure, white balance and noise to Auto. */
  rerunAuto(photoId: PhotoId): Promise<ResolvedSettings>;

  // --- Native viewport -----------------------------------------------------

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

  // --- Looks ---------------------------------------------------------------

  /** `looks_list`. */
  looksList(): Promise<Look[]>;
  /** `looks_import`: import a zip, a folder or a `.cube` file. */
  looksImport(path: string): Promise<LooksImportResult>;
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
  /** `cli_preview`: the equivalent `yellowmoon render …` command. */
  cliPreview(job: ExportJob): Promise<string>;

  // --- Preferences and CLI -------------------------------------------------

  /** `preferences_get`. */
  preferencesGet(): Promise<Preferences>;
  /** `preferences_set`: apply a patch and return the new preferences. */
  preferencesSet(patch: PreferencesPatch): Promise<Preferences>;
  /** `install_cli_link`: the "Install Command-Line Tool…" menu item. */
  installCliLink(): Promise<CliLinkResult>;

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
 * Calls the Rust commands. Command names are snake_case, argument keys camelCase: Tauri 2
 * maps `photoId` to the Rust parameter `photo_id`.
 */
export const tauriApi: Api = {
  openPaths: (paths) => invoke("open_paths", { paths }),
  openDialog: (kind) => invoke("open_dialog", { kind }),
  sessionState: () => invoke("session_state"),
  setCurrent: (photoId) => invoke("set_current", { photoId }),
  setPicked: (photoId, picked) => invoke("set_picked", { photoId, picked }),

  getSettings: (photoId) => invoke("get_settings", { photoId }),
  updateSettings: (photoId, patch) => invoke("update_settings", { photoId, patch }),
  applyLook: (scope, lookId, intensity) => invoke("apply_look", { scope, lookId, intensity }),
  rerunAuto: (photoId) => invoke("rerun_auto", { photoId }),

  viewportSetRect: (rect) => invoke("viewport_set_rect", { rect }),
  viewportInput: (event) => invoke("viewport_input", { event }),
  viewportSetMode: (mode) => invoke("viewport_set_mode", { mode }),
  setBefore: (before) => invoke("set_before", { before }),
  samplePixel: (x, y) => invoke("sample_pixel", { x, y }),

  looksList: () => invoke("looks_list"),
  looksImport: (path) => invoke("looks_import", { path }),
  looksSetDefault: (id) => invoke("looks_set_default", { id }),
  looksRemove: (id) => invoke("looks_remove", { id }),

  exportEstimate: (job) => invoke("export_estimate", { job }),
  exportStart: (job) => invoke("export_start", { job }),
  exportCancel: (jobId) => invoke("export_cancel", { jobId }),
  cliPreview: (job) => invoke("cli_preview", { job }),

  preferencesGet: () => invoke("preferences_get"),
  preferencesSet: (patch) => invoke("preferences_set", { patch }),
  installCliLink: () => invoke("install_cli_link"),

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
  return isTauri() ? tauriApi : mockApi;
}
