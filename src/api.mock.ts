/**
 * In-memory implementation of `Api` with fixture data from the design mockups. It lets the
 * frontend run in a plain browser, in Vitest and in Playwright without Tauri or the Rust core
 * (`docs/01-architecture.md` §IPC, `docs/11-ui.md` §Tests).
 *
 * What it simulates (the same fakes as `design/mockups/`):
 *  - twelve "Lisbon" photos from a Sony A1 (2026-10-04, ISO 400, 1/125, f/2.8), drawn as flat
 *    shapes; opening any ARW file or a folder opens all twelve;
 *  - development that takes about 4 s for 12 photos (`photo_progress`, then `thumbnail_ready`
 *    with a real PNG), and an export that takes about 1.25 s per photo (`export_progress`);
 *  - the built-in looks (Neutral, Signature) and the imported look names of `photos.html`;
 *  - settings that resolve against fixed automatic estimates (+0.30 EV, 5450 K, tint 0).
 *
 * What it does not do: render anything. Looks and settings change stored values, not the
 * fixture images.
 *
 * Tests control it through the extra methods of `MockApi`: `speed` scales the timers,
 * `flush()` finishes running work at once, `failNext()` makes a command reject, and `reset()`
 * returns to a fresh app.
 */
import type { Api, Unsubscribe } from "./api";
import type {
  ApiEventName,
  ApiEvents,
  AppError,
  ApplyScope,
  AutoEstimates,
  CliLinkResult,
  DevelopSettings,
  ExportEstimate,
  ExportFormat,
  ExportJob,
  ExportJobId,
  ExportOptions,
  Look,
  LookId,
  LooksImportResult,
  OpenKind,
  PhotoId,
  PhotoStage,
  PhotoStatus,
  PhotoSummary,
  PixelSample,
  Preferences,
  PreferencesPatch,
  RenderStats,
  ResolvedSettings,
  ResolvedValues,
  SessionState,
  SettingsPatch,
  ViewportInput,
  ViewportMode,
  ViewportRect,
} from "./api/types";
import { fixtureUrl } from "./fixtures";
import { scenePng, hexToRgb } from "./fixtures/png";
import { layerAt, SCENES, type Scene } from "./fixtures/scenes";

// ---------------------------------------------------------------------------
// Fixture constants
// ---------------------------------------------------------------------------

/** The folder the twelve photos "come from". */
const FOLDER_NAME = "Lisbon";
const FOLDER_PATH = "~/Pictures/Lisbon";
/** Capture time of the first photo; the rest follow every 25 minutes. */
const FIRST_CAPTURE = Date.UTC(2026, 9, 4, 15, 20, 0);
const CAPTURE_STEP_MS = 25 * 60 * 1000;

/** Milliseconds per development tick and photos advanced per tick (≈ 4.1 s for 12 photos). */
const DEV_TICK_MS = 120;
const DEV_STEP = 0.35;
/** Milliseconds per export tick and photos advanced per tick (≈ 1.25 s per photo). */
const EXPORT_TICK_MS = 100;
const EXPORT_STEP = 0.08;
/** Fraction of a photo's development spent decoding before the estimates start. */
const DECODE_SHARE = 0.4;

const RENDER_STATS: RenderStats = { ms: 42, path: "neural", space: "display_p3" };

/** What "Auto" resolves to for every photo in the mock. */
export const MOCK_ESTIMATES: AutoEstimates = {
  exposure_ev: 0.3,
  kelvin: 5450,
  tint: 0,
  light_class: "daylight",
  light_confidence: 0.82,
  noise: { a: 1.9e-5, b: 3.1e-7 },
  sharpen: { amount: 120, radius_px: 0.8 },
  noise_reduction: { luminance: 30, color: 50, detail: 50 },
};

const NEUTRAL: LookId = "builtin:neutral";

const BUILT_IN_LOOKS: ReadonlyArray<Pick<Look, "id" | "name">> = [
  { id: NEUTRAL, name: "Neutral" },
  { id: "builtin:signature", name: "Signature" },
];

/** The imported look names shown in `photos.html`. The ids and group are placeholders. */
const IMPORTED_LOOKS: ReadonlyArray<Pick<Look, "id" | "name">> = [
  ["film-warm", "Film warm"],
  ["film-cool", "Film cool"],
  ["bleach-bypass", "Bleach bypass"],
  ["teal-and-orange", "Teal and orange"],
  ["soft-print", "Soft print"],
  ["night-blue", "Night blue"],
  ["monochrome", "Monochrome"],
].map(([slug, name]) => ({ id: `arri:3000:${slug}`, name: name as string }));

// ---------------------------------------------------------------------------
// Options and the mock-only surface
// ---------------------------------------------------------------------------

/** Options for `createMockApi`, `configure` and `reset`. */
export interface MockOptions {
  /**
   * Speed of simulated work. 1 matches the mockups (about 4 s to develop 12 photos);
   * 10 is ten times faster; `Infinity` runs every timer at 0 ms. Default 1.
   */
  speed?: number;
  /** Files that can't be developed, keyed by file name ("DSC04207.ARW" or "DSC04207") with the reason. */
  skipped?: Record<string, string>;
  /** Indexes of the photos that are picked when the session opens. Default none. */
  picked?: number[];
  /** Index of the current photo when the session opens. Default 0. */
  current?: number;
  /** Whether the imported looks of the mockups are in the library. Default true. */
  looksImported?: boolean;
}

/** What the viewport commands have recorded, for tests. */
export interface MockViewportState {
  rect: ViewportRect | null;
  mode: ViewportMode;
  before: boolean;
  inputs: ViewportInput[];
}

/** `Api` plus the controls tests and demos need. */
export interface MockApi extends Api {
  /** The options in force. */
  readonly options: Readonly<Required<MockOptions>>;
  /** Change options without touching state. */
  configure(options: MockOptions): void;
  /** Stop all work, forget the session and restore looks and preferences. `options` replaces the current ones. */
  reset(options?: MockOptions): void;
  /** Finish development and every running export now, emitting the remaining events. */
  flush(): void;
  /** Make the next call of `command` reject with `error`. */
  failNext(command: keyof Api, error: AppError): void;
  /** Make the next `openDialog` resolve `null`, as if the user cancelled. */
  cancelNextDialog(): void;
  /** URL of a photo's full-size SVG fixture (what the native viewport would draw). */
  fixtureUrl(photoId: PhotoId): string | undefined;
  /** What `viewport_*` and `set_before` have been called with. */
  readonly viewport: Readonly<MockViewportState>;
}

// ---------------------------------------------------------------------------
// Pure helpers (exported for tests)
// ---------------------------------------------------------------------------

/** The settings of a photo nobody has touched: Auto everywhere, the given look at 100 %. */
export function defaultSettings(look: LookId = NEUTRAL): DevelopSettings {
  return {
    look,
    look_intensity: 100,
    exposure: { kind: "auto" },
    contrast: 0,
    brightness: 0,
    highlights: 0,
    shadows: 0,
    white: 0,
    black: 0,
    light_source: "auto",
    white_balance: { kind: "auto" },
    color_balance: { shadows: [0, 0], midtones: [0, 0], highlights: [0, 0] },
    saturation: 0,
    sharpen: { kind: "auto" },
    noise_reduction: { kind: "auto" },
    curve: {
      luma: [[0, 0], [1, 1]],
      red: [[0, 0], [1, 1]],
      green: [[0, 0], [1, 1]],
      blue: [[0, 0], [1, 1]],
    },
    decode: { neural_demosaic: true, highlight_recovery: true, lens_corrections: true, chromatic_aberration: true },
    rolloff: { highlights: 0.5, low_light: 0.5 },
    crop: null,
    orientation: 0,
    white_pick: null,
  };
}

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

/** Clamps every value to the ranges of `docs/01` §DevelopSettings, as the core does. */
export function clampSettings(s: DevelopSettings): DevelopSettings {
  const pct = (v: number) => clamp(v, -100, 100);
  const out: DevelopSettings = {
    ...s,
    look_intensity: clamp(s.look_intensity, 0, 100),
    contrast: pct(s.contrast),
    brightness: pct(s.brightness),
    highlights: pct(s.highlights),
    shadows: pct(s.shadows),
    white: pct(s.white),
    black: pct(s.black),
    saturation: pct(s.saturation),
    rolloff: { highlights: clamp(s.rolloff.highlights, 0, 1), low_light: clamp(s.rolloff.low_light, 0, 1) },
  };
  if (s.exposure.kind === "manual") out.exposure = { kind: "manual", ev: clamp(s.exposure.ev, -4, 4) };
  if (s.white_balance.kind === "manual") {
    out.white_balance = {
      kind: "manual",
      kelvin: clamp(s.white_balance.kelvin, 2000, 11000),
      tint: clamp(s.white_balance.tint, -50, 50),
    };
  }
  if (s.sharpen.kind === "manual") {
    out.sharpen = {
      kind: "manual",
      amount: clamp(s.sharpen.amount, 0, 300),
      radius_px: clamp(s.sharpen.radius_px, 0.3, 2.5),
    };
  }
  if (s.noise_reduction.kind === "manual") {
    out.noise_reduction = {
      kind: "manual",
      luminance: clamp(s.noise_reduction.luminance, 0, 100),
      color: clamp(s.noise_reduction.color, 0, 100),
      detail: clamp(s.noise_reduction.detail, 0, 100),
    };
  }
  return out;
}

/** Resolves every `Auto` against the estimates and reports which fields are still automatic. */
export function resolveSettings(photoId: PhotoId, settings: DevelopSettings, estimates: AutoEstimates): ResolvedSettings {
  const { exposure, white_balance, sharpen, noise_reduction, ...rest } = settings;
  const values: ResolvedValues = {
    ...rest,
    exposure_ev: exposure.kind === "manual" ? exposure.ev : estimates.exposure_ev,
    kelvin: white_balance.kind === "manual" ? white_balance.kelvin : estimates.kelvin,
    tint: white_balance.kind === "manual" ? white_balance.tint : estimates.tint,
    sharpen:
      sharpen.kind === "manual"
        ? { amount: sharpen.amount, radius_px: sharpen.radius_px }
        : { ...estimates.sharpen },
    noise_reduction:
      noise_reduction.kind === "manual"
        ? { luminance: noise_reduction.luminance, color: noise_reduction.color, detail: noise_reduction.detail }
        : { ...estimates.noise_reduction },
  };
  return {
    photo_id: photoId,
    settings,
    values,
    auto: {
      exposure: exposure.kind === "auto",
      light_source: settings.light_source === "auto",
      white_balance: white_balance.kind === "auto",
      sharpen: sharpen.kind === "auto",
      noise_reduction: noise_reduction.kind === "auto",
    },
    estimates,
  };
}

/** Slug of a look id: "arri:3000:film-warm" gives "film-warm". */
export function lookSlug(id: LookId): string {
  return id.slice(id.lastIndexOf(":") + 1);
}

const FILE_EXTENSION: Record<ExportFormat, string> = { jpeg: "jpg", tiff: "tif", png: "png", dng: "dng" };

/** Expands `{name}`, `{look}`, `{date}`, `{seq}` and `{camera}`; unknown tokens stay as typed. */
export function expandNameTemplate(
  template: string,
  parts: { name: string; look: string; date: string; seq: number; camera: string },
): string {
  return template.replace(/\{(name|look|date|seq|camera)\}/g, (_, key: keyof typeof parts) =>
    key === "seq" ? String(parts.seq).padStart(3, "0") : String(parts[key]),
  );
}

/** Size estimate with the arithmetic of the export mockup, in megabytes per photo. */
function megabytesPerPhoto(o: ExportOptions): number {
  const scale = o.format === "dng" || o.size.kind === "full" ? 1 : (o.size.px / 8640) ** 2;
  const base = {
    jpeg: 6 + (o.jpeg_quality - 50) * 0.45,
    tiff: (o.bit_depth === 16 ? 298 : 149) * (o.tiff_compression === "none" ? 1 : 0.68),
    png: o.bit_depth === 16 ? 210 : 105,
    dng: 300,
  }[o.format];
  return base * scale;
}

const clone = <T>(value: T): T => (value === undefined ? value : (JSON.parse(JSON.stringify(value)) as T));

function defaultExportOptions(): ExportOptions {
  return {
    format: "jpeg",
    jpeg_quality: 92,
    bit_depth: 16,
    tiff_compression: "lzw",
    color_space: "srgb",
    size: { kind: "full" },
    destination: null,
    name_template: "{name}_{look}",
    if_exists: "keep_both",
  };
}

function defaultPreferences(): Preferences {
  return {
    default_look: NEUTRAL,
    display_rendering: { kind: "aces2" },
    export_defaults: defaultExportOptions(),
    remove_location_from_exports: false,
    check_for_updates_automatically: true,
    watch_downloads_during_import: false,
    last_used_folders: { open: null, export: null, looks_import: null },
  };
}

const DEFAULT_OPTIONS: Required<MockOptions> = { speed: 1, skipped: {}, picked: [], current: 0, looksImported: true };

/** Fills in defaults, ignoring `undefined` values and a speed that isn't a positive number. */
function withDefaults(...layers: MockOptions[]): Required<MockOptions> {
  const out: Required<MockOptions> = { ...DEFAULT_OPTIONS };
  for (const layer of layers) {
    for (const key of Object.keys(layer) as Array<keyof MockOptions>) {
      if (layer[key] !== undefined) (out as unknown as Record<string, unknown>)[key] = layer[key];
    }
  }
  if (!(out.speed > 0)) out.speed = 1;
  return out;
}

/** Reads `?mockSpeed=20` from the page URL so Playwright can speed the timers up. */
function speedFromLocation(): number | undefined {
  if (typeof location === "undefined") return undefined;
  const raw = new URLSearchParams(location.search).get("mockSpeed");
  if (raw === null) return undefined;
  const value = raw === "Infinity" ? Infinity : Number(raw);
  return value > 0 ? value : undefined;
}

// ---------------------------------------------------------------------------
// The mock
// ---------------------------------------------------------------------------

interface PhotoRecord {
  scene: Scene;
  summary: PhotoSummary;
  settings: DevelopSettings;
  /** 0 not started, 1 decoding, 2 estimating, 3 finished. */
  phase: 0 | 1 | 2 | 3;
  skipReason: string | null;
}

interface ExportRun {
  id: ExportJobId;
  files: string[];
  progress: number;
  timer: ReturnType<typeof setInterval> | null;
}

type Handler<E extends ApiEventName> = (payload: ApiEvents[E]) => void;

/** Creates an independent mock backend. Most code uses the shared `mockApi` through `getApi()`. */
export function createMockApi(initial: MockOptions = {}): MockApi {
  let options = withDefaults(initial);

  const listeners = new Map<ApiEventName, Set<Handler<ApiEventName>>>();
  const failures = new Map<keyof Api, AppError>();
  let cancelDialog = false;

  let photos: PhotoRecord[] = [];
  let current = 0;
  let developProgress = 0;
  let developTimer: ReturnType<typeof setInterval> | null = null;
  const exportRuns = new Map<ExportJobId, ExportRun>();
  let nextJobId = 1;

  let preferences = defaultPreferences();
  let looks: Look[] = [];
  const viewport: MockViewportState = { rect: null, mode: { kind: "fit" }, before: false, inputs: [] };

  // --- Events ---------------------------------------------------------------

  function emit<E extends ApiEventName>(event: E, payload: ApiEvents[E]): void {
    const set = listeners.get(event);
    if (!set) return;
    for (const handler of [...set]) {
      try {
        (handler as Handler<E>)(clone(payload));
      } catch (error) {
        // One failing listener must not stop the others or the simulated work, as with real events.
        console.error(`api.mock: a "${event}" listener threw`, error);
      }
    }
  }

  /** Events sent from inside a command arrive after the command's own result, like the bridge. */
  function emitSoon<E extends ApiEventName>(event: E, payload: ApiEvents[E]): void {
    queueMicrotask(() => emit(event, payload));
  }

  // --- Looks and preferences ---------------------------------------------------

  function buildLooks(): Look[] {
    const list: Look[] = BUILT_IN_LOOKS.map((l) => ({
      ...l,
      group: { kind: "built_in" },
      input: "logc4",
      is_default: false,
      missing: false,
    }));
    if (options.looksImported) list.push(...importedLooks());
    return list.map((l) => ({ ...l, is_default: l.id === preferences.default_look }));
  }

  function importedLooks(): Look[] {
    return IMPORTED_LOOKS.map((l) => ({
      ...l,
      group: { kind: "arri", no: 3000, name: "Film" },
      input: "logc4",
      is_default: false,
      missing: false,
    }));
  }

  function syncDefaultLook(): void {
    looks = looks.map((l) => ({ ...l, is_default: l.id === preferences.default_look }));
  }

  function findLook(id: LookId): Look | undefined {
    return looks.find((l) => l.id === id);
  }

  // --- Session ----------------------------------------------------------------

  function snapshot(): SessionState {
    if (photos.length === 0) {
      return { folder_name: null, photos: [], current_index: null, picked: [], default_export_folder: null };
    }
    return {
      folder_name: FOLDER_NAME,
      photos: photos.map((p) => p.summary),
      current_index: current,
      picked: photos.filter((p) => p.summary.picked).map((p) => p.summary.id),
      default_export_folder: `${FOLDER_PATH} export`,
    };
  }

  function skipReasonFor(scene: Scene): string | null {
    const skipped = options.skipped;
    return skipped[`${scene.name}.ARW`] ?? skipped[scene.name] ?? null;
  }

  function openFixtureSession(): void {
    const pickedSet = new Set(options.picked);
    photos = SCENES.map((scene, index): PhotoRecord => {
      const summary: PhotoSummary = {
        id: index + 1,
        file_name: `${scene.name}.ARW`,
        camera: {
          body: "Sony A1",
          model_id: 384,
          profile: "A7R IV",
          profile_id: "sony-ilce-7rm4",
          profile_version: "1.0.0",
          match_kind: { kind: "default" },
        },
        iso: 400,
        shutter_s: 1 / 125,
        aperture_f: 2.8,
        captured_at: new Date(FIRST_CAPTURE + index * CAPTURE_STEP_MS).toISOString().slice(0, 19),
        width_px: 8640,
        height_px: 5760,
        status: { kind: "waiting" },
        picked: pickedSet.has(index),
        reduced: false,
      };
      return {
        scene,
        summary,
        settings: defaultSettings(preferences.default_look),
        phase: 0,
        skipReason: skipReasonFor(scene),
      };
    });
    current = clamp(options.current, 0, photos.length - 1);
    developProgress = 0;
    startDevelopment();
  }

  function setStatus(rec: PhotoRecord, status: PhotoStatus): void {
    rec.summary.status = status;
  }

  function progress(rec: PhotoRecord, stage: PhotoStage, done: boolean, error: string | null = null): void {
    emit("photo_progress", { id: rec.summary.id, stage, done, error });
  }

  function enterPhase(rec: PhotoRecord, phase: 1 | 2 | 3): void {
    rec.phase = phase;
    if (phase === 1) {
      setStatus(rec, { kind: "developing", stage: "decode" });
      progress(rec, "decode", false);
    } else if (phase === 2) {
      if (rec.skipReason) return;
      setStatus(rec, { kind: "developing", stage: "estimate" });
      progress(rec, "decode", true);
      progress(rec, "estimate", false);
    } else if (rec.skipReason) {
      setStatus(rec, { kind: "skipped", reason: rec.skipReason });
      progress(rec, "decode", true, rec.skipReason);
    } else {
      setStatus(rec, { kind: "ready" });
      progress(rec, "estimate", true);
      emit("thumbnail_ready", { id: rec.summary.id, png_bytes: Array.from(scenePng(rec.scene)) });
      progress(rec, "thumbnail", true);
    }
  }

  function applyDevelopProgress(): void {
    photos.forEach((rec, i) => {
      const target = developProgress >= i + 1 ? 3 : developProgress >= i + DECODE_SHARE ? 2 : developProgress >= i ? 1 : 0;
      while (rec.phase < target) enterPhase(rec, (rec.phase + 1) as 1 | 2 | 3);
    });
  }

  function stopDevelopment(): void {
    if (developTimer !== null) clearInterval(developTimer);
    developTimer = null;
  }

  function startDevelopment(): void {
    stopDevelopment();
    queueMicrotask(applyDevelopProgress);
    developTimer = setInterval(() => {
      developProgress = Math.min(photos.length, developProgress + DEV_STEP);
      applyDevelopProgress();
      if (developProgress >= photos.length) stopDevelopment();
    }, DEV_TICK_MS / options.speed);
  }

  function isArwPath(path: string): boolean {
    return /\.arw$/i.test(path);
  }

  /** A path whose last segment has no dot is taken for a folder. */
  function isFolderPath(path: string): boolean {
    const last = path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
    return last.length > 0 && !last.includes(".");
  }

  function openPathsNow(paths: string[]): SessionState {
    const hasPhotos = paths.some((p) => isArwPath(p) || isFolderPath(p));
    // Non-ARW files are ignored; a session that is already open already has these photos.
    if (hasPhotos && photos.length === 0) openFixtureSession();
    return snapshot();
  }

  // --- Command plumbing -----------------------------------------------------------

  async function run<T>(command: keyof Api, fn: () => T): Promise<T> {
    await Promise.resolve();
    const failure = failures.get(command);
    if (failure) {
      failures.delete(command);
      throw clone(failure);
    }
    return clone(fn());
  }

  function fail(scope: AppError["scope"], message: string): never {
    throw { scope, message } satisfies AppError;
  }

  function photo(id: PhotoId): PhotoRecord {
    const rec = photos.find((p) => p.summary.id === id);
    if (!rec) fail("app", "That photo is no longer open. Open it again.");
    return rec;
  }

  function targets(scope: ApplyScope): PhotoRecord[] {
    if (scope === "all") return photos;
    if (scope === "picked") return photos.filter((p) => p.summary.picked);
    const rec = photos[current];
    return rec ? [rec] : [];
  }

  // --- Export -----------------------------------------------------------------------

  function exportFiles(job: ExportJob): string[] {
    const recs = targets(job.scope);
    const ext = FILE_EXTENSION[job.options.format];
    return recs.map((rec, i) =>
      `${expandNameTemplate(job.options.name_template, {
        name: rec.scene.name,
        look: lookSlug(rec.settings.look),
        date: rec.summary.captured_at.slice(0, 10),
        seq: i + 1,
        camera: "sony-a1",
      })}.${ext}`,
    );
  }

  function emitExportProgress(run: ExportRun): void {
    const total = run.files.length;
    const finished = run.progress >= total;
    const index = Math.min(Math.floor(run.progress), total - 1);
    emit("export_progress", {
      job_id: run.id,
      file: run.files[index] ?? "",
      done: Math.min(total, Math.floor(run.progress)),
      total,
      file_percent: finished ? 100 : Math.round((run.progress - index) * 100),
      error: null,
    });
  }

  function stopExport(run: ExportRun): void {
    if (run.timer !== null) clearInterval(run.timer);
    run.timer = null;
  }

  function cliCommand(job: ExportJob): string {
    const o = job.options;
    const recs = targets(job.scope);
    const first = recs[0];
    const look = lookSlug(first?.settings.look ?? preferences.default_look);
    const flags = [`--look ${look}`, `--format ${o.format}`];
    if (o.format === "jpeg") flags.push(`--quality ${o.jpeg_quality}`);
    if (o.format === "tiff" || o.format === "png") flags.push(`--bits ${o.bit_depth}`);
    if (o.format === "tiff") flags.push(`--compression ${o.tiff_compression}`);
    if (o.format !== "dng") flags.push(`--colour-space ${o.color_space === "display_p3" ? "p3" : o.color_space}`);
    if (o.format !== "dng" && o.size.kind === "long_edge") flags.push(`--long-edge ${o.size.px}`);
    if (o.name_template !== "{name}_{look}") flags.push(`--name "${o.name_template}"`);
    if (o.if_exists === "skip") flags.push("--skip-existing");
    if (o.if_exists === "replace") flags.push("--overwrite");
    flags.push(`--out "${o.destination ?? `${FOLDER_PATH} export`}"`);
    const inputs = job.scope === "all" ? FOLDER_PATH : recs.map((r) => r.summary.file_name).join(" ");
    return `yellowmoon render ${inputs} \\\n  ${flags.join(" \\\n  ")}`;
  }

  // --- Lifecycle ------------------------------------------------------------------------

  function stopAll(): void {
    stopDevelopment();
    for (const run of exportRuns.values()) stopExport(run);
    exportRuns.clear();
  }

  function resetState(): void {
    stopAll();
    photos = [];
    current = 0;
    developProgress = 0;
    preferences = defaultPreferences();
    looks = buildLooks();
    failures.clear();
    cancelDialog = false;
    viewport.rect = null;
    viewport.mode = { kind: "fit" };
    viewport.before = false;
    viewport.inputs = [];
    nextJobId = 1;
  }

  resetState();

  // --- The Api ------------------------------------------------------------------------------

  const api: MockApi = {
    get options() {
      return options;
    },
    get viewport() {
      return viewport;
    },

    configure(patch) {
      options = withDefaults(options, patch);
    },
    reset(next) {
      if (next) options = withDefaults(next);
      resetState();
    },
    flush() {
      if (developTimer !== null) {
        developProgress = photos.length;
        applyDevelopProgress();
        stopDevelopment();
      }
      for (const run of [...exportRuns.values()]) {
        if (run.timer === null) continue;
        stopExport(run);
        run.progress = run.files.length;
        emitExportProgress(run);
      }
    },
    failNext(command, error) {
      failures.set(command, error);
    },
    cancelNextDialog() {
      cancelDialog = true;
    },
    fixtureUrl(photoId) {
      const rec = photos.find((p) => p.summary.id === photoId);
      return rec ? fixtureUrl(rec.scene.name) : undefined;
    },

    openPaths: (paths) => run("openPaths", () => openPathsNow(paths)),
    openDialog: (kind: OpenKind) =>
      run("openDialog", () => {
        if (cancelDialog) {
          cancelDialog = false;
          return null;
        }
        return openPathsNow(kind === "folder" ? [FOLDER_PATH] : SCENES.map((s) => `${FOLDER_PATH}/${s.name}.ARW`));
      }),
    sessionState: () => run("sessionState", snapshot),
    setCurrent: (photoId) =>
      run("setCurrent", () => {
        const index = photos.findIndex((p) => p.summary.id === photoId);
        if (index < 0) fail("app", "That photo is no longer open. Open it again.");
        current = index;
        emitSoon("render_stats", RENDER_STATS);
      }),
    setPicked: (photoId, picked) =>
      run("setPicked", () => {
        photo(photoId).summary.picked = picked;
      }),

    getSettings: (photoId) =>
      run("getSettings", () => resolveSettings(photoId, photo(photoId).settings, MOCK_ESTIMATES)),
    updateSettings: (photoId, patch: SettingsPatch) =>
      run("updateSettings", () => {
        const rec = photo(photoId);
        const merged = { ...rec.settings } as Record<string, unknown>;
        for (const [key, value] of Object.entries(patch)) if (value !== undefined) merged[key] = value;
        rec.settings = clampSettings(merged as unknown as DevelopSettings);
        emitSoon("render_stats", RENDER_STATS);
        return resolveSettings(photoId, rec.settings, MOCK_ESTIMATES);
      }),
    applyLook: (scope, lookId, intensity) =>
      run("applyLook", () => {
        const look = findLook(lookId);
        if (!look || look.missing) fail("looks", "That look isn't in the library any more. Import it again.");
        for (const rec of targets(scope)) {
          rec.settings = { ...rec.settings, look: lookId, look_intensity: clamp(intensity, 0, 100) };
        }
        emitSoon("render_stats", RENDER_STATS);
      }),
    rerunAuto: (photoId) =>
      run("rerunAuto", () => {
        const rec = photo(photoId);
        // Exposure, white balance (including the light preset and a picked white) and noise.
        rec.settings = {
          ...rec.settings,
          exposure: { kind: "auto" },
          light_source: "auto",
          white_balance: { kind: "auto" },
          white_pick: null,
          noise_reduction: { kind: "auto" },
        };
        emitSoon("render_stats", RENDER_STATS);
        return resolveSettings(photoId, rec.settings, MOCK_ESTIMATES);
      }),

    viewportSetRect: (rect) =>
      run("viewportSetRect", () => {
        viewport.rect = rect;
      }),
    viewportInput: (event) =>
      run("viewportInput", () => {
        viewport.inputs.push(event);
        const rec = photos[current];
        if (event.kind === "toggle_zoom") {
          viewport.mode = viewport.mode.kind === "fit" ? { kind: "percent", value: 100 } : { kind: "fit" };
        } else if (event.kind === "pick_white" && rec) {
          const rect = viewport.rect ?? { x: 0, y: 0, width: 1200, height: 800, scale: 2 };
          rec.settings = {
            ...rec.settings,
            white_pick: [clamp(event.x / rect.width, 0, 1), clamp(event.y / rect.height, 0, 1)],
          };
        }
      }),
    viewportSetMode: (mode) =>
      run("viewportSetMode", () => {
        viewport.mode = mode;
      }),
    setBefore: (before) =>
      run("setBefore", () => {
        viewport.before = before;
      }),
    samplePixel: (x, y) =>
      run("samplePixel", (): PixelSample => {
        const rec = photos[current];
        if (!rec) fail("viewport", "Open a photo first.");
        const rect = viewport.rect ?? { x: 0, y: 0, width: 1200, height: 800, scale: 2 };
        const layer = layerAt(rec.scene, clamp(x / rect.width, 0, 1), clamp(y / rect.height, 0, 1));
        const [r, g, b] = hexToRgb(rec.scene[layer]);
        return { r, g, b };
      }),

    looksList: () => run("looksList", () => looks),
    looksImport: (path) =>
      run("looksImport", (): LooksImportResult => {
        void path;
        const have = new Set(looks.map((l) => l.id));
        const added = importedLooks().filter((l) => !have.has(l.id));
        looks = [...looks, ...added].map((l) => ({ ...l, is_default: l.id === preferences.default_look }));
        if (added.length > 0) emitSoon("looks_changed", null);
        return { added, display_luts_added: 0 };
      }),
    looksSetDefault: (id) =>
      run("looksSetDefault", () => {
        const look = findLook(id);
        if (!look || look.missing) fail("looks", "That look isn't in the library any more. Import it again.");
        preferences = { ...preferences, default_look: id };
        syncDefaultLook();
        emitSoon("looks_changed", null);
      }),
    looksRemove: (id) =>
      run("looksRemove", () => {
        const look = findLook(id);
        if (!look) return;
        if (look.group.kind === "built_in") fail("looks", "Built-in looks can't be removed.");
        looks = looks.filter((l) => l.id !== id);
        if (preferences.default_look === id) preferences = { ...preferences, default_look: NEUTRAL };
        for (const rec of photos) if (rec.settings.look === id) rec.settings = { ...rec.settings, look: NEUTRAL };
        syncDefaultLook();
        emitSoon("looks_changed", null);
      }),

    exportEstimate: (job) =>
      run("exportEstimate", (): ExportEstimate => {
        const count = targets(job.scope).length;
        return {
          bytes: Math.round(megabytesPerPhoto(job.options) * count * 1_000_000),
          seconds: Math.round(count * ((EXPORT_TICK_MS / EXPORT_STEP) / 1000) * 10) / 10,
        };
      }),
    exportStart: (job) =>
      run("exportStart", () => {
        const files = exportFiles(job);
        if (files.length === 0) {
          fail("export", "No photos are picked. Pick some photos, or choose All, then export again.");
        }
        const exportRun: ExportRun = { id: nextJobId++, files, progress: 0, timer: null };
        exportRuns.set(exportRun.id, exportRun);
        queueMicrotask(() => {
          if (exportRun.timer !== null) emitExportProgress(exportRun);
        });
        exportRun.timer = setInterval(() => {
          exportRun.progress = Math.min(exportRun.files.length, exportRun.progress + EXPORT_STEP);
          emitExportProgress(exportRun);
          if (exportRun.progress >= exportRun.files.length) stopExport(exportRun);
        }, EXPORT_TICK_MS / options.speed);
        return exportRun.id;
      }),
    exportCancel: (jobId) =>
      run("exportCancel", () => {
        const found = exportRuns.get(jobId);
        if (!found) return;
        stopExport(found);
        exportRuns.delete(jobId);
      }),
    cliPreview: (job) => run("cliPreview", () => cliCommand(job)),

    preferencesGet: () => run("preferencesGet", () => preferences),
    preferencesSet: (patch: PreferencesPatch) =>
      run("preferencesSet", () => {
        if (patch.default_look !== undefined) {
          const look = findLook(patch.default_look);
          if (!look || look.missing) fail("preferences", "That look isn't in the library. Pick another default look.");
        }
        const next = { ...preferences } as Record<string, unknown>;
        for (const [key, value] of Object.entries(patch)) if (value !== undefined) next[key] = value;
        preferences = next as unknown as Preferences;
        if (patch.default_look !== undefined) {
          syncDefaultLook();
          emitSoon("looks_changed", null);
        }
        return preferences;
      }),
    installCliLink: () => run("installCliLink", (): CliLinkResult => ({ path: "/usr/local/bin/yellowmoon" })),

    on<E extends ApiEventName>(event: E, handler: (payload: ApiEvents[E]) => void): Unsubscribe {
      let set = listeners.get(event);
      if (!set) {
        set = new Set();
        listeners.set(event, set);
      }
      const stored = handler as Handler<ApiEventName>;
      set.add(stored);
      return () => {
        set.delete(stored);
      };
    },
  };

  return api;
}

/**
 * The shared mock `getApi()` returns outside Tauri. Speed can be set from the page URL
 * (`/?mockSpeed=20`) so end-to-end tests don't wait for the mockup's timers.
 */
export const mockApi: MockApi = createMockApi({ speed: speedFromLocation() ?? 1 });
