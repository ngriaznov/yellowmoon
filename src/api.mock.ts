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
 *  - settings that resolve against fixed automatic estimates (+0.30 EV, 5450 K, tint 0), a
 *    light preset that resolves to a fixed kelvin, and a picked white that resolves to a kelvin
 *    and tint from the colour under the point;
 *  - look thumbnails, histograms and re-sent photo thumbnails drawn with the mockup's flat-shape
 *    grading (`fixtures/grade.ts`).
 *
 * What it does not do: render anything real. Settings change stored values; the pictures it
 * returns only show the effect the way the mockups fake it. It has no viewport of its own: the
 * Photos screen draws the fixture picture itself (`viewportInfo()` answers `null`), unless a test
 * asks it to behave like the app's two backends (`viewport: "native"` or `"fallback"`, which also
 * sends `viewport_frame` events with a cell-sized picture).
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
  AppInfo,
  ApplyScope,
  AutoEstimates,
  CliLinkResult,
  CliLinkStatus,
  DevelopSettings,
  DisplayLut,
  DownloadFound,
  ExportEstimate,
  ExportJob,
  ExportJobId,
  ExportOptions,
  HistogramData,
  LightSource,
  Look,
  LookId,
  LookInput,
  LooksImportResult,
  OpenKind,
  PackageId,
  PhotoId,
  PhotoStage,
  PhotoStatus,
  PhotoSummary,
  PixelSample,
  Preferences,
  PreferencesPatch,
  Recognition,
  RenderStats,
  ResolvedSettings,
  ResolvedValues,
  SessionState,
  SettingsPatch,
  SourceKind,
  SourceRecognition,
  ViewportBackendKind,
  ViewportInfo,
  ViewportInput,
  ViewportMode,
  ViewportRect,
  WatchStatus,
} from "./api/types";
import { expandNameTemplate, FILE_EXTENSION, sanitizeStem, slugify } from "./export/template";
import { version as APP_VERSION } from "../package.json";
import { fixtureUrl } from "./fixtures";
import { developScene, previewSvg } from "./fixtures/grade";
import type { Adjustment } from "./fixtures/grade";
import { sceneHistogram } from "./fixtures/histogram";
import { scenePng, hexToRgb } from "./fixtures/png";
import { layerAt, SCENES, type Scene } from "./fixtures/scenes";
import { fitInCell, imageAspectOf } from "./screens/crop/cropMath";
import { bytesToUrl } from "./state/photo";

// ---------------------------------------------------------------------------
// Fixture constants
// ---------------------------------------------------------------------------

/** The folder the twelve photos "come from". */
const DEFAULT_FOLDER_NAME = "Lisbon";
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

// The mock has no zoom of its own to report: the screen estimates it (`fit_percent: null`).
const RENDER_STATS: RenderStats = { ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null };

/** What "Auto" resolves to for every photo in the mock. */
export const MOCK_ESTIMATES: AutoEstimates = {
  exposure_ev: 0.3,
  white_ev: 0.8,
  kelvin: 5450,
  tint: 0,
  light_class: "daylight",
  light_confidence: 0.82,
  noise: { a: 1.9e-5, b: 3.1e-7 },
  sharpen: { amount: 120, radius_px: 0.8 },
  noise_reduction: { luminance: 30, color: 50, detail: 50 },
};

const NEUTRAL: LookId = "builtin:neutral";

/** What a light preset resolves to while White balance is Auto (kelvin and tint). */
export const LIGHT_PRESETS: Readonly<Record<Exclude<LightSource, "auto">, { kelvin: number; tint: number }>> = {
  day: { kelvin: 5500, tint: 0 },
  shade: { kelvin: 7500, tint: 4 },
  tungsten: { kelvin: 3200, tint: 0 },
  led: { kelvin: 4800, tint: -3 },
  fluorescent: { kelvin: 4100, tint: 12 },
};

/** Milliseconds before a photo's thumbnail is re-sent after its look or light changed. */
const THUMB_DEBOUNCE_MS = 120;

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
  /** The name of the folder the photos are in (the header, the export panel and the default destination use it). Default "Lisbon". */
  folderName?: string;
  /** How many photos the session has. The twelve fixtures repeat, with the names DSC04213 and on, so a long folder can be shown and timed. Default 12. */
  photoCount?: number;
  /** Whether the imported looks of the mockups are in the library. Default true. */
  looksImported?: boolean;
  /** Slugs of imported looks whose file is "missing or changed" ("film-warm"): listed with `missing: true`. */
  missingLooks?: string[];
  /** Development stops at this many photos (fractions allowed: 5.4 is five done, the sixth under way). Default: all. */
  holdAt?: number;
  /** Settings the photos open with, keyed by file name ("DSC04204.ARW" or "DSC04204"). */
  initialSettings?: Record<string, SettingsPatch>;
  /** Changes to a photo's summary (a different camera, a reduced decode), keyed like `initialSettings`. */
  photoOverrides?: Record<string, Partial<Omit<PhotoSummary, "camera">> & { camera?: Partial<PhotoSummary["camera"]> }>;
  /**
   * Zips that "land in Downloads" one after the other, 2.5 s apart, once the wizard starts watching
   * (`looks_watch_downloads(true)`). File names as in `SYNTHETIC_SOURCES`. Default none: tests call `simulateDownload`.
   */
  downloads?: string[];
  /** macOS refuses access to the Downloads folder: `looks_watch_downloads(true)` answers with a problem. Default false. */
  downloadsDenied?: boolean;
  /** Output files that already exist in the destination, by name ("DSC04204_neutral.jpg"): Skip skips them, Keep both numbers them. */
  existingExports?: string[];
  /** Photos whose export fails, by name ("DSC04204"), with the reason. */
  exportFailures?: Record<string, string>;
  /** An export stops at this progress, in photos (1.45 is one done and the second at 45 %) until `flush()`, so a screenshot can show the running state. Default: never. */
  exportHoldAt?: number;
  /** A look import stops after this many files and waits for `flush()`, so a screenshot can show it half way (the mockup's 40 of 87). Default: never. */
  importHoldAt?: number;
  /** Keep the preferences in `localStorage` so a reload of the page keeps them (`?mockPersist=1`). Default false. */
  persistPreferences?: boolean;
  /**
   * The viewport backend the mock claims to be (`?mockViewport=native|fallback`). `null` (the default): none, and the Photos screen
   * draws the fixture picture itself, as a browser build does. `native`: the screen leaves the cell transparent, as in the app on a
   * Mac. `fallback`: the mock sends `viewport_frame` events with a picture of the cell's size, which the screen shows.
   */
  viewport?: ViewportBackendKind | null;
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
  /** What the system commands were asked to do, in order: pages opened, folders shown in Finder, files moved to the Trash. */
  readonly system: Readonly<MockSystemLog>;
  /**
   * The answer of the next `chooseFolder` or `chooseLookFiles`: a path (or paths), or `null` for a cancelled
   * dialog. Answers queue up. Without one `chooseFolder` answers "~/Pictures/Chosen export" and `chooseLookFiles`
   * the Look Library · LogC4 zip.
   */
  queueChoice(kind: "folder", answer: string | null): void;
  queueChoice(kind: "files", answer: string[] | null): void;
  /** Add (or replace) a synthetic file the recogniser can find, by file name. */
  addSource(fileName: string, fixture: SourceFixture): void;
  /** Take a synthetic file away, as if it had been deleted or moved: it is `missing` from now on. */
  removeSource(fileName: string): void;
  /** Send an event to the listeners, as the core would: a `menu_action` from the native menu, an `update_state`, an `error`. */
  emitEvent<E extends ApiEventName>(event: E, payload: ApiEvents[E]): void;
  /** Make a zip "land in Downloads": a `download_found` event when the watcher is on; resolves to whether it was sent. */
  simulateDownload(fileName: string): boolean;
}

/** What `MockApi.system` has recorded. */
export interface MockSystemLog {
  opened: string[];
  revealed: string[];
  trashed: string[];
  /** How many times `restartApp` was allowed (it is refused while an export runs). */
  restarts: number;
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
    white_offset_ev: 0,
    black: 0,
    light_source: "auto",
    white_balance: { kind: "auto" },
    color_balance: { shadows: [0, 0], midtones: [0, 0], highlights: [0, 0] },
    saturation: 0,
    sharpen: { kind: "auto" },
    output_sharpening: 100,
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

/** Pulls a colour-balance point into the unit disc. */
function unitDisc([a, b]: [number, number]): [number, number] {
  const length = Math.hypot(a, b);
  return length > 1 ? [a / length, b / length] : [a, b];
}

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
    white_offset_ev: clamp(s.white_offset_ev, -3, 3),
    black: pct(s.black),
    saturation: pct(s.saturation),
    output_sharpening: clamp(s.output_sharpening, 0, 200),
    rolloff: { highlights: clamp(s.rolloff.highlights, 0, 1), low_light: clamp(s.rolloff.low_light, 0, 1) },
    color_balance: {
      shadows: unitDisc(s.color_balance.shadows),
      midtones: unitDisc(s.color_balance.midtones),
      highlights: unitDisc(s.color_balance.highlights),
    },
    orientation: (((Math.round(s.orientation / 90) * 90) % 360 + 360) % 360) as DevelopSettings["orientation"],
    white_pick: s.white_pick ? [clamp(s.white_pick[0], 0, 1), clamp(s.white_pick[1], 0, 1)] : null,
    crop: s.crop
      ? {
          ...s.crop,
          angle_deg: clamp(s.crop.angle_deg, -45, 45),
          rect: {
            x: clamp(s.crop.rect.x, 0, 1),
            y: clamp(s.crop.rect.y, 0, 1),
            width: clamp(s.crop.rect.width, 0.01, 1 - clamp(s.crop.rect.x, 0, 1)),
            height: clamp(s.crop.rect.height, 0.01, 1 - clamp(s.crop.rect.y, 0, 1)),
          },
        }
      : null,
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
  // A light preset stands in for the estimated light while White balance is Auto.
  const preset = settings.light_source === "auto" ? null : LIGHT_PRESETS[settings.light_source];
  const values: ResolvedValues = {
    ...rest,
    exposure_ev: exposure.kind === "manual" ? exposure.ev : estimates.exposure_ev,
    white_ev: clamp(estimates.white_ev + settings.white_offset_ev, 0, 3),
    kelvin: white_balance.kind === "manual" ? white_balance.kelvin : (preset?.kelvin ?? estimates.kelvin),
    tint: white_balance.kind === "manual" ? white_balance.tint : (preset?.tint ?? estimates.tint),
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

export { expandNameTemplate };

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

/**
 * What a picked white resolves to: a bluer pixel means a cooler light than Automatic assumed, a
 * redder one a warmer light, and green or magenta in the pixel shows as tint. Deterministic and
 * only a stand-in for `docs/02` C1.
 */
export function whiteFromPick(scene: Scene, point: [number, number]): { kelvin: number; tint: number } {
  const [r, g, b] = hexToRgb(scene[layerAt(scene, clamp(point[0], 0, 1), clamp(point[1], 0, 1))]);
  const warmth = Math.log((r + 1) / (b + 1));
  const kelvin = clamp(Math.round((MOCK_ESTIMATES.kelvin * Math.exp(-warmth * 0.9)) / 50) * 50, 2000, 11000);
  const tint = clamp(Math.round(((g - (r + b) / 2) / 255) * -60), -50, 50);
  return { kelvin, tint };
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

const DEFAULT_OPTIONS: Required<MockOptions> = {
  speed: 1,
  skipped: {},
  picked: [],
  current: 0,
  photoCount: SCENES.length,
  folderName: DEFAULT_FOLDER_NAME,
  looksImported: true,
  missingLooks: [],
  holdAt: Number.POSITIVE_INFINITY,
  initialSettings: {},
  photoOverrides: {},
  downloads: [],
  downloadsDenied: false,
  existingExports: [],
  exportFailures: {},
  exportHoldAt: Number.POSITIVE_INFINITY,
  importHoldAt: Number.POSITIVE_INFINITY,
  persistPreferences: false,
  viewport: null,
};

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

/**
 * Reads the mock's options from the page URL so Playwright (and a person with a browser) can
 * set the scene without code: `?mockSpeed=20` (or `Infinity`), `mockSkipped=DSC04207,DSC04209`
 * (a reason after a colon), `mockPicked=2,3,4`, `mockCurrent=3`, `mockPhotos=500` (a folder of 500 photos, the twelve repeating), `mockFolder=Name` (the folder's name), `mockLooks=0` (no imported
 * looks), `mockMissing=film-warm,film-cool`, `mockHold=5.4`, `mockDownloads=ARRI_LookLibrary_LogC4.zip` (zips that
 * land in Downloads while the wizard watches), `mockDownloadsDenied=1`, `mockExisting=DSC04204_neutral.jpg`,
 * `mockExportFails=DSC04204:reason`, `mockExportHold=1.45` (an export waits there until `flush()`), `mockImportHold=40` (a look import waits at 40 files until `flush()`) and `mockPersist=1`
 * (preferences survive a reload) and `mockViewport=native` or `fallback` (the mock behaves like that viewport backend of the app).
 */
export function optionsFromLocation(search: string | undefined = typeof location === "undefined" ? undefined : location.search): MockOptions {
  if (search === undefined) return {};
  const params = new URLSearchParams(search);
  const out: MockOptions = {};
  const speed = params.get("mockSpeed");
  if (speed !== null) {
    const value = speed === "Infinity" ? Number.POSITIVE_INFINITY : Number(speed);
    if (value > 0) out.speed = value;
  }
  const list = (name: string): string[] | undefined =>
    params.has(name) ? (params.get(name) ?? "").split(",").map((x) => x.trim()).filter(Boolean) : undefined;
  const skipped = list("mockSkipped");
  if (skipped) {
    out.skipped = Object.fromEntries(
      skipped.map((entry) => {
        const at = entry.indexOf(":");
        return at < 0 ? [entry, "This file is from a camera mode we can't read yet."] : [entry.slice(0, at), entry.slice(at + 1)];
      }),
    );
  }
  const picked = list("mockPicked");
  if (picked) out.picked = picked.map(Number).filter((n) => Number.isInteger(n) && n >= 0);
  const current = params.get("mockCurrent");
  if (current !== null && Number.isInteger(Number(current))) out.current = Number(current);
  const folder = params.get("mockFolder");
  if (folder !== null && folder.trim() !== "") out.folderName = folder;
  const count = params.get("mockPhotos");
  if (count !== null && Number.isInteger(Number(count)) && Number(count) >= 1) out.photoCount = Number(count);
  if (params.get("mockLooks") === "0") out.looksImported = false;
  const missing = list("mockMissing");
  if (missing) out.missingLooks = missing;
  const hold = params.get("mockHold");
  if (hold !== null && Number(hold) >= 0) out.holdAt = Number(hold);
  const downloads = list("mockDownloads");
  if (downloads) out.downloads = downloads;
  if (params.get("mockDownloadsDenied") === "1") out.downloadsDenied = true;
  const existing = list("mockExisting");
  if (existing) out.existingExports = existing;
  const failing = list("mockExportFails");
  if (failing) {
    out.exportFailures = Object.fromEntries(
      failing.map((entry) => {
        const at = entry.indexOf(":");
        return at < 0 ? [entry, "The disk is full. Free some space, then export again."] : [entry.slice(0, at), entry.slice(at + 1)];
      }),
    );
  }
  const exportHold = params.get("mockExportHold");
  if (exportHold !== null && Number(exportHold) >= 0) out.exportHoldAt = Number(exportHold);
  const importHold = params.get("mockImportHold");
  if (importHold !== null && Number(importHold) >= 0) out.importHoldAt = Number(importHold);
  if (params.get("mockPersist") === "1") out.persistPreferences = true;
  const viewportBackend = params.get("mockViewport");
  if (viewportBackend === "native" || viewportBackend === "fallback") out.viewport = viewportBackend;
  return out;
}

// ---------------------------------------------------------------------------
// Synthetic ARRI packages for the wizard (`docs/08` §Wizard, §Recognition rules)
// ---------------------------------------------------------------------------

/** What a synthetic file, folder or zip is, as the recogniser would find it. */
export interface SourceFixture {
  source: SourceKind;
  size_bytes: number | null;
  recognition: Recognition;
}

interface PackageFact {
  /** The zip's file name. */
  zip: string;
  title: string;
  what: string;
  contents: string;
  fileCount: number;
  megabytes: number;
  input: LookInput;
  role: "looks" | "display";
}

/** The five packages as the mock "downloads" them: file names, sizes and counts of `docs/08`. */
export const PACKAGE_FACTS: Readonly<Record<PackageId, PackageFact>> = {
  look_library_logc4: {
    zip: "ARRI_LookLibrary_LogC4.zip",
    title: "Look Library · LogC4",
    what: "Look Library · LogC4 · log to log",
    contents: "87 .cube · 65-point",
    fileCount: 87,
    megabytes: 181,
    input: "logc4",
    role: "looks",
  },
  display_luts_logc4: {
    zip: "ARRI_DisplayLUTs_LogC4.zip",
    title: "Display LUTs · LogC4",
    what: "Display LUTs · LogC4",
    contents: "22 .cube",
    fileCount: 22,
    megabytes: 32,
    input: "logc4",
    role: "display",
  },
  look_library_logc3: {
    zip: "ARRI_LookLibrary_LogC3.zip",
    title: "Look Library · LogC3",
    what: "Look Library · LogC3 · log to log",
    contents: "87 .cube · 33-point",
    fileCount: 87,
    megabytes: 30,
    input: "logc3_ei800",
    role: "looks",
  },
  look_library_logc3_to_rec709: {
    zip: "ARRI_LookLibrary_LogC3_to_Rec709.zip",
    title: "Look Library · LogC3 to Rec.709",
    what: "Look Library · LogC3 to Rec.709",
    contents: "87 .cube",
    fileCount: 87,
    megabytes: 32,
    input: "logc3_to_rec709",
    role: "looks",
  },
  display_luts_logc3: {
    zip: "ARRI_DisplayLUTs_LogC3.zip",
    title: "Display LUTs · LogC3",
    what: "Display LUTs · LogC3",
    contents: "48 .cube",
    fileCount: 48,
    megabytes: 46,
    input: "logc3_ei800",
    role: "display",
  },
};

/** The zip file name of each package, for tests and the Playwright flows. */
export const PACKAGE_ZIPS = Object.fromEntries(Object.entries(PACKAGE_FACTS).map(([id, f]) => [id, f.zip])) as Record<PackageId, string>;

/**
 * ARRI's two look-file downloads as the mock "downloads" them: the same looks as a 3D-LUT package, as camera look files. They
 * are recognised as that package (`docs/08` §ARRI Look Files).
 */
export const LOOK_FILE_ZIPS = {
  /** Look Library for LogC4 (ALF4 Look Files): the looks of `look_library_logc4`, 87 `.alf4` files. */
  alf4Library: "arri-look-library-logc4-log-to-log-alf4-data.zip",
  /** ARRI Look Library LogC3 (Look Files): the looks of `look_library_logc3_to_rec709`, 261 `.aml` files (three strengths each). */
  amlLibrary: "ARRI Look Library LogC3 (log-to-Rec709) ALF2_data.zip",
} as const;

/** Zips the mock knows that are not one of the five packages: the failure cases of the wizard. */
export const FAILURE_ZIPS = {
  /** One ARRI Look File that is damaged (its checksum does not match). */
  damagedLookFile: "Damaged_Look.alf4",
  /** One loose `.alf4` of the person's own: it says what it expects, so nothing is asked. */
  ownAlf4: "Warm_Pull.alf4",
  /** One loose `.alf4c`: its CMT is the look, its display transforms are not used. */
  ownAlf4c: "Custom_Look.alf4c",
  /** A folder of three of the person's own `.alf4` files. */
  ownLookFolder: "My_Looks",
  /** ARRI `.cube` files from a package Yellowmoon doesn't know (another camera). */
  otherCamera: "ARRI_ALEXA35_CameraLUTs.zip",
  /** A zip with no LUTs at all. */
  noLuts: "Holiday_Notes.zip",
  /** A known package with files missing. */
  incomplete: "ARRI_LookLibrary_LogC4_incomplete.zip",
  /** One loose `.cube` of the person's own. */
  ownCube: "Warm_Film.cube",
} as const;

/** The source fixtures every mock starts with, keyed by lower-case file name. */
function defaultSources(): Map<string, SourceFixture> {
  const map = new Map<string, SourceFixture>();
  for (const [id, f] of Object.entries(PACKAGE_FACTS) as Array<[PackageId, PackageFact]>) {
    map.set(f.zip.toLowerCase(), {
      source: "zip",
      size_bytes: f.megabytes * 1_000_000,
      recognition: { kind: "package", package_id: id, title: f.title, what: f.what, contents: f.contents, file_count: f.fileCount, note: null },
    });
  }
  // The same package unpacked into a folder.
  map.set("arri_looklibrary_logc4", {
    source: "folder",
    size_bytes: null,
    recognition: {
      kind: "package",
      package_id: "look_library_logc4",
      title: PACKAGE_FACTS.look_library_logc4.title,
      what: PACKAGE_FACTS.look_library_logc4.what,
      contents: PACKAGE_FACTS.look_library_logc4.contents,
      file_count: 87,
      note: null,
    },
  });
  // ARRI's look-file downloads are the looks of their 3D-LUT package.
  map.set(LOOK_FILE_ZIPS.alf4Library.toLowerCase(), {
    source: "zip",
    size_bytes: 16_800_000,
    recognition: {
      kind: "package",
      package_id: "look_library_logc4",
      title: PACKAGE_FACTS.look_library_logc4.title,
      what: PACKAGE_FACTS.look_library_logc4.what,
      contents: "87 .alf4 · 33-point",
      file_count: 87,
      note: null,
    },
  });
  map.set(LOOK_FILE_ZIPS.amlLibrary.toLowerCase(), {
    source: "zip",
    size_bytes: 38_700_000,
    recognition: {
      kind: "package",
      package_id: "look_library_logc3_to_rec709",
      title: PACKAGE_FACTS.look_library_logc3_to_rec709.title,
      what: PACKAGE_FACTS.look_library_logc3_to_rec709.what,
      contents: "261 .aml · 33-point",
      file_count: 261,
      note: null,
    },
  });
  const ownNote = "Imported as your own looks, with the input read from each file.";
  map.set(FAILURE_ZIPS.ownAlf4.toLowerCase(), {
    source: "file",
    size_bytes: 217_246,
    recognition: { kind: "look_files", what: "ARRI Look Files · LogC4", contents: "1 .alf4 · 33-point", file_count: 1, input: "logc4", note: ownNote },
  });
  map.set(FAILURE_ZIPS.ownAlf4c.toLowerCase(), {
    source: "file",
    size_bytes: 432_100,
    recognition: {
      kind: "look_files",
      what: "ARRI Look Files · LogC4",
      contents: "1 .alf4c · 33-point",
      file_count: 1,
      input: "logc4",
      note: `${ownNote} Custom_Look.alf4c: its own display transforms are not used; Yellowmoon's display rendering is.`,
    },
  });
  map.set(FAILURE_ZIPS.ownLookFolder.toLowerCase(), {
    source: "folder",
    size_bytes: null,
    recognition: { kind: "look_files", what: "ARRI Look Files · LogC4", contents: "3 .alf4 · 33-point", file_count: 3, input: "logc4", note: ownNote },
  });
  map.set(FAILURE_ZIPS.damagedLookFile.toLowerCase(), {
    source: "file",
    size_bytes: 217_246,
    recognition: {
      kind: "unusable",
      package_id: null,
      reason: "Damaged_Look.alf4 cannot be used. Its checksum does not match, so the file is damaged. Download it again.",
    },
  });
  map.set(FAILURE_ZIPS.otherCamera.toLowerCase(), {
    source: "zip",
    size_bytes: 12_000_000,
    recognition: {
      kind: "needs_answer",
      file_count: 24,
      lut_size: 33,
      suggestion: null,
      note: "These files look like they come from ARRI, but not from a package Yellowmoon knows. They may be for another camera. You can still import them as your own LUTs if you say what they expect as input.",
    },
  });
  map.set(FAILURE_ZIPS.noLuts.toLowerCase(), {
    source: "zip",
    size_bytes: 2_000_000,
    recognition: {
      kind: "unusable",
      package_id: null,
      reason: "This zip has no .cube, .alf4, .alf4c or .aml files, so there is nothing to import. Choose a look file or a LUT, or a folder or zip that holds them.",
    },
  });
  map.set(FAILURE_ZIPS.incomplete.toLowerCase(), {
    source: "zip",
    size_bytes: 160_000_000,
    recognition: {
      kind: "unusable",
      package_id: "look_library_logc4",
      reason:
        "Look Library · LogC4 should have 87 files, but this zip has 79. The download may be incomplete. Download it again from ARRI's page. Missing: Film_08.cube, Film_09.cube, Film_10.cube, Film_11.cube, Film_12.cube and 3 more.",
    },
  });
  map.set(FAILURE_ZIPS.ownCube.toLowerCase(), {
    source: "file",
    size_bytes: 1_200_000,
    recognition: {
      kind: "needs_answer",
      file_count: 1,
      lut_size: 33,
      suggestion: null,
      note: "1 file is not from an ARRI package we know. Say what it expects as input: LogC4, LogC3, or LogC3 with Rec.709 baked in.",
    },
  });
  return map;
}

/** The look names the mockup's Look tab and wizard show, in order. */
const NAMED_LOOKS = ["Film warm", "Film cool", "Bleach bypass", "Teal and orange", "Soft print", "Night blue", "Monochrome"];
/** Names for the rest of a package's 87 looks. */
const LOOK_FAMILIES = ["Print", "Fade", "Noir", "Dusk", "Dawn", "Bloom", "Haze", "Ember", "Frost"];

/** The looks a package adds: the mockup's seven, then generated ones up to the package's count. LogC3 packages get their own ids. */
function packageLooks(id: PackageId): Look[] {
  const fact = PACKAGE_FACTS[id];
  if (fact.role !== "looks") return [];
  const suffix = id === "look_library_logc4" ? "" : id === "look_library_logc3" ? "-logc3" : "-rec709";
  const names = [...NAMED_LOOKS];
  for (let i = 0; names.length < fact.fileCount; i++) names.push(`${LOOK_FAMILIES[i % LOOK_FAMILIES.length]} ${Math.floor(i / LOOK_FAMILIES.length) + 1}`);
  return names.map((name) => ({
    id: `arri:3000:${slugify(name)}${suffix}`,
    name: suffix ? `${name} · ${suffix === "-logc3" ? "LogC3" : "Rec.709"}` : name,
    group: { kind: "arri", no: 3000, name: "Film" },
    input: fact.input,
    is_default: false,
    missing: false,
  }));
}

/** The display LUTs a package adds. The first is ARRI's Rec.709 rendering, `arri-display:logc4:rec709-g24`. */
function packageDisplayLuts(id: PackageId): DisplayLut[] {
  const fact = PACKAGE_FACTS[id];
  if (fact.role !== "display") return [];
  const generation = id === "display_luts_logc4" ? "logc4" : "logc3";
  const targets: Array<[string, string]> = [
    ["rec709-g24", "Rec.709 · Gamma 2.4"],
    ["rec709-g22", "Rec.709 · Gamma 2.2"],
    ["p3-d65", "P3 · D65"],
    ["rec2020", "Rec.2020"],
    ["hlg", "HLG"],
    ["pq", "PQ"],
  ];
  const out: DisplayLut[] = [];
  for (let i = 0; out.length < fact.fileCount; i++) {
    const [target, label] = targets[i % targets.length] as [string, string];
    const round = Math.floor(i / targets.length);
    out.push({
      id: `arri-display:${generation}:${target}${round === 0 ? "" : `-${round + 1}`}`,
      name: round === 0 ? label : `${label} · ${round + 1}`,
      input: fact.input === "logc4" ? "logc4" : "logc3_ei800",
      target,
    });
  }
  return out;
}

/** "Warm_Film.cube" gives "Warm Film". */
const cubeName = (fileName: string): string => fileName.replace(/\.(cube|zip|alf4c?|aml)$/i, "").replace(/[_-]+/g, " ").trim();

const MISSING_SOURCE = "That file is not there any more. Download it again, or choose it again.";

/** The last part of a path. */
const fileNameOf = (path: string): string => path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? path;

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
  /** Files already reported finished (so a failed or skipped file is reported once). */
  reported: number;
  /** Per file: the reason it fails, or null. */
  errors: Array<string | null>;
  /** Per file: it already exists and is skipped. */
  skips: boolean[];
  timer: ReturnType<typeof setInterval> | null;
}

/** A look import in flight: the simulated copy of one source. */
interface ImportRun {
  path: string;
  total: number;
  done: number;
  timer: ReturnType<typeof setInterval> | null;
  /** Reports the rest of the files as copied and finishes the run. */
  complete: () => void;
  resolve: () => void;
}

const PREFERENCES_KEY = "yellowmoon.mock.state";

/** What the mock keeps in `localStorage` when `persistPreferences` is on. */
interface StoredState {
  preferences: Preferences;
  importedPackages: PackageId[];
  userLooks: Look[];
  cliLink: string | null;
}

function readStored(): StoredState | null {
  try {
    const text = typeof localStorage === "undefined" ? null : localStorage.getItem(PREFERENCES_KEY);
    return text ? (JSON.parse(text) as StoredState) : null;
  } catch {
    return null;
  }
}

function writeStored(state: StoredState | null): void {
  try {
    if (typeof localStorage === "undefined") return;
    if (state) localStorage.setItem(PREFERENCES_KEY, JSON.stringify(state));
    else localStorage.removeItem(PREFERENCES_KEY);
  } catch {
    // Private windows and blocked storage: the mock simply forgets on reload.
  }
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
  const thumbTimers = new Map<PhotoId, ReturnType<typeof setTimeout>>();
  /** The address of each photo's newest thumbnail (the shell keeps these for `thumbnails()`). */
  const thumbnailUrls = new Map<PhotoId, string>();
  const exportRuns = new Map<ExportJobId, ExportRun>();
  let nextJobId = 1;

  let preferences = defaultPreferences();
  let looks: Look[] = [];
  let importedPackages: PackageId[] = [];
  let userLooks: Look[] = [];
  let displayLuts: DisplayLut[] = [];
  let sources = defaultSources();
  let watching = false;
  let downloadTimers: Array<ReturnType<typeof setTimeout>> = [];
  const importRuns = new Set<ImportRun>();
  const choices: { folder: Array<string | null>; files: Array<string[] | null> } = { folder: [], files: [] };
  const system: MockSystemLog = { opened: [], revealed: [], trashed: [], restarts: 0 };
  let cliLink: string | null = null;
  const viewport: MockViewportState = { rect: null, mode: { kind: "fit" }, before: false, inputs: [] };
  /** The fallback viewport's frame counter, and whether it has a picture on show. */
  let frameSeq = 0;
  let frameShown = false;
  /** Whether an `export_busy` event has said the queue is busy. */
  let busyAnnounced = false;

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

  // --- The viewport ------------------------------------------------------------

  /** The photo cell as the page reported it; a cell with no size (a test with no layout) is taken to be 1200 × 800 at 2×. */
  function cellRect(): ViewportRect {
    const rect = viewport.rect;
    return rect && rect.width > 0 && rect.height > 0 ? rect : { x: 0, y: 0, width: 1200, height: 800, scale: 2 };
  }

  /**
   * The spot of the photo (as a share of the whole photo as the camera had it, 0–1 across and down) under a point of the cell in CSS
   * pixels, or null on the paper around a letterboxed picture: what the core's `pick_point_at` and `sample_pixel` find. The picture is
   * the photo turned by `orientation` (clockwise) and cut to its crop, fitted in the cell: a point of the picture goes through the crop
   * (a rectangle of the turned photo), then back through the turn. Straightening is not modelled.
   */
  function photoPointAt(rec: PhotoRecord, x: number, y: number): [number, number] | null {
    const rect = cellRect();
    const { orientation, crop } = rec.settings;
    const crossed = crop ? crop.rect.width / crop.rect.height : 1;
    const aspect = imageAspectOf(rec.summary.width_px, rec.summary.height_px, orientation) * crossed;
    const picture = fitInCell(rect.width / rect.height, aspect);
    const u = (x / rect.width - picture.x) / picture.width;
    const v = (y / rect.height - picture.y) / picture.height;
    if (!(u >= 0 && u <= 1 && v >= 0 && v <= 1)) return null;
    // The spot of the turned photo.
    const cx = crop ? crop.rect.x + u * crop.rect.width : u;
    const cy = crop ? crop.rect.y + v * crop.rect.height : v;
    // A quarter turn clockwise takes the photo's (x, y) to (1 − y, x); undo it.
    switch (((orientation % 360) + 360) % 360) {
      case 90:
        return [cy, 1 - cx];
      case 180:
        return [1 - cx, 1 - cy];
      case 270:
        return [1 - cy, cx];
      default:
        return [cx, cy];
    }
  }

  /** The picture of the cell the fallback viewport would send: the developed scene letterboxed on paper, as many pixels as the cell has. */
  function frameImage(rec: PhotoRecord): { url: string; width: number; height: number } {
    const rect = cellRect();
    const width = Math.round(rect.width * rect.scale);
    const height = Math.round(rect.height * rect.scale);
    const { values } = resolveSettings(rec.summary.id, rec.settings, MOCK_ESTIMATES);
    const picture = previewSvg(developed(rec), {
      orientation: values.orientation,
      angle_deg: values.crop?.angle_deg ?? 0,
      crop: values.crop?.rect ?? null,
    }).replace(/^<svg ([^>]*?) width="[^"]*" height="[^"]*"/, '<svg $1 width="100%" height="100%"');
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="100%" fill="#FFFFFF"/>${picture}</svg>`;
    return { url: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`, width, height };
  }

  /**
   * With `viewport: "fallback"`, tells the page what the cell shows now, as the shell does after every change: a frame of the current
   * photo, or no picture while it is not ready or the session is closed.
   */
  function sendFrame(): void {
    if (options.viewport !== "fallback") return;
    const rec = photos[current];
    if (!rec || rec.phase < 3 || rec.skipReason || viewport.rect === null) {
      if (!frameShown) return;
      frameShown = false;
      emitSoon("viewport_frame", { seq: ++frameSeq, width: 0, height: 0, sharp: true, url: "" });
      return;
    }
    frameShown = true;
    const frame = frameImage(rec);
    emitSoon("viewport_frame", { seq: ++frameSeq, width: frame.width, height: frame.height, sharp: true, url: frame.url });
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
    const seen = new Set(list.map((l) => l.id));
    const add = (more: Look[]): void => {
      for (const look of more) {
        if (seen.has(look.id)) continue;
        seen.add(look.id);
        list.push(look);
      }
    };
    if (options.looksImported) add(importedLooks());
    for (const id of importedPackages) add(packageLooks(id).map((l) => ({ ...l, missing: options.missingLooks.includes(lookSlug(l.id)) })));
    add(userLooks);
    return list.map((l) => ({ ...l, is_default: l.id === preferences.default_look }));
  }

  /** Library after something was imported or removed: rebuilt, keeping the default look when it still exists. */
  function rebuildLooks(): void {
    looks = buildLooks();
    if (!looks.some((l) => l.id === preferences.default_look)) {
      preferences = { ...preferences, default_look: NEUTRAL };
      looks = looks.map((l) => ({ ...l, is_default: l.id === NEUTRAL }));
    }
  }

  function importedLooks(): Look[] {
    return IMPORTED_LOOKS.map((l) => ({
      ...l,
      group: { kind: "arri", no: 3000, name: "Film" },
      input: "logc4",
      is_default: false,
      missing: options.missingLooks.includes(lookSlug(l.id)),
    }));
  }

  /** Writes what a real install would keep: the preferences and the imported looks. */
  function persist(): void {
    if (!options.persistPreferences) return;
    writeStored({ preferences, importedPackages, userLooks, cliLink });
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
      folder_name: options.folderName,
      photos: photos.map((p) => p.summary),
      current_index: current,
      picked: photos.filter((p) => p.summary.picked).map((p) => p.summary.id),
      default_export_folder: `${folderPath()} export`,
    };
  }

  /** Where the photos are, as the destination and the command line say it. */
  function folderPath(): string {
    return `~/Pictures/${options.folderName}`;
  }

  function skipReasonFor(scene: Scene): string | null {
    const skipped = options.skipped;
    return skipped[`${scene.name}.ARW`] ?? skipped[scene.name] ?? null;
  }

  function openFixtureSession(): void {
    const pickedSet = new Set(options.picked);
    photos = Array.from({ length: Math.max(1, Math.floor(options.photoCount)) }, (_, index): PhotoRecord => {
      const fixture = SCENES[index % SCENES.length] as Scene;
      // Past the twelve fixtures the scenes repeat under new names, so every file name is its own.
      const scene: Scene = index < SCENES.length ? fixture : { ...fixture, name: `DSC${String(4201 + index).padStart(5, "0")}` };
      const summary: PhotoSummary = {
        id: index + 1,
        file_name: `${scene.name}.ARW`,
        camera: {
          body: "Sony A1",
          model: "ILCE-1",
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
      const initial = options.initialSettings[`${scene.name}.ARW`] ?? options.initialSettings[scene.name] ?? {};
      const override = options.photoOverrides[`${scene.name}.ARW`] ?? options.photoOverrides[scene.name];
      if (override) Object.assign(summary, { ...override, camera: { ...summary.camera, ...override.camera } });
      return {
        scene,
        summary,
        settings: clampSettings({ ...defaultSettings(preferences.default_look), ...initial }),
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
      emitThumbnail(rec);
      progress(rec, "thumbnail", true);
      if (photos[current] === rec) sendFrame();
    }
  }

  /** The photo as it is developed now: adjusted by what differs from Automatic, then through its look. */
  function developed(rec: PhotoRecord, lookId: LookId = rec.settings.look, intensity = rec.settings.look_intensity): Scene {
    const { values } = resolveSettings(rec.summary.id, rec.settings, MOCK_ESTIMATES);
    const adjustment: Adjustment = {
      ev: values.exposure_ev,
      autoEv: MOCK_ESTIMATES.exposure_ev,
      kelvin: values.kelvin,
      autoKelvin: MOCK_ESTIMATES.kelvin,
      tint: values.tint,
      autoTint: MOCK_ESTIMATES.tint,
    };
    return developScene(rec.scene, { slug: lookSlug(lookId), intensity }, adjustment);
  }

  /**
   * A photo's own thumbnail: its look and its adjustments (the filmstrip shows it). The picture is announced by its address, as the
   * shell does (there, `ym-frame://…`; here a `blob:` URL), and the address is kept for `thumbnails()`.
   */
  function emitThumbnail(rec: PhotoRecord): void {
    const id = rec.summary.id;
    const url = bytesToUrl(Array.from(scenePng(developed(rec))));
    thumbnailUrls.set(id, url);
    emit("thumbnail_ready", { id, url });
  }

  /** Re-sends a photo's thumbnail shortly after its look or light changed, once for a burst of changes. */
  function scheduleThumbnail(rec: PhotoRecord): void {
    if (rec.phase < 3 || rec.skipReason) return;
    const id = rec.summary.id;
    const earlier = thumbTimers.get(id);
    if (earlier !== undefined) clearTimeout(earlier);
    thumbTimers.set(
      id,
      setTimeout(() => {
        thumbTimers.delete(id);
        if (photos.includes(rec)) emitThumbnail(rec);
      }, THUMB_DEBOUNCE_MS / options.speed),
    );
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
    const limit = Math.min(photos.length, options.holdAt);
    developTimer = setInterval(() => {
      developProgress = Math.min(limit, developProgress + DEV_STEP);
      applyDevelopProgress();
      if (developProgress >= limit) stopDevelopment();
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

  /** Like `run`, for a command that finishes later (a simulated import). */
  async function runAsync<T>(command: keyof Api, fn: () => Promise<T>): Promise<T> {
    await Promise.resolve();
    const failure = failures.get(command);
    if (failure) {
      failures.delete(command);
      throw clone(failure);
    }
    return clone(await fn());
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
    if (scope === "all") return photos.filter((p) => p.skipReason === null);
    if (scope === "picked") return photos.filter((p) => p.summary.picked);
    const rec = photos[current];
    return rec ? [rec] : [];
  }

  // --- Export -----------------------------------------------------------------------

  /** "DSC04204_neutral.jpg" gives "DSC04204_neutral 2.jpg", then " 3" (Keep both). */
  function numbered(name: string, taken: ReadonlySet<string>): string {
    const dot = name.lastIndexOf(".");
    const stem = dot < 0 ? name : name.slice(0, dot);
    const ext = dot < 0 ? "" : name.slice(dot);
    for (let n = 2; ; n++) {
      const candidate = `${stem} ${n}${ext}`;
      if (!taken.has(candidate)) return candidate;
    }
  }

  function exportFiles(job: ExportJob): { files: string[]; skips: boolean[]; errors: Array<string | null> } {
    const recs = targets(job.scope);
    const ext = FILE_EXTENSION[job.options.format];
    const existing = new Set(options.existingExports);
    const taken = new Set<string>(existing);
    const files: string[] = [];
    const skips: boolean[] = [];
    const errors: Array<string | null> = [];
    recs.forEach((rec, i) => {
      let name = `${sanitizeStem(
        expandNameTemplate(job.options.name_template, {
          name: rec.scene.name,
          look: lookSlug(rec.settings.look),
          date: rec.summary.captured_at.slice(0, 10),
          seq: i + 1,
          camera: "sony-a1",
        }),
      )}.${ext}`;
      let skip = false;
      if (taken.has(name)) {
        if (job.options.if_exists === "skip") skip = true;
        else if (job.options.if_exists === "keep_both") name = numbered(name, taken);
      }
      taken.add(name);
      files.push(name);
      skips.push(skip);
      errors.push(options.exportFailures[rec.scene.name] ?? null);
    });
    return { files, skips, errors };
  }

  /** A DNG of the job is close to, not the same as, the picture: a photo has a look with its own display rendering (at an intensity above 0), or ARRI's display rendering is chosen. */
  function dngApproximate(job: ExportJob): boolean {
    if (preferences.display_rendering.kind === "arri_display") return true;
    return targets(job.scope).some((rec) => rec.settings.look_intensity > 0 && findLook(rec.settings.look)?.input === "logc3_to_rec709");
  }

  /** Photos of a job with adjustments beyond their look: what a command line can't carry. */
  function adjustedPhotos(job: ExportJob): number {
    const plain = (s: DevelopSettings): string => JSON.stringify({ ...s, look: "", look_intensity: 0 });
    const base = plain(defaultSettings());
    return targets(job.scope).filter((rec) => plain(rec.settings) !== base).length;
  }

  function emitExportProgress(run: ExportRun): void {
    const total = run.files.length;
    const finished = run.progress >= total;
    const index = Math.min(Math.floor(run.progress), total - 1);
    const done = Math.min(total, Math.floor(run.progress));
    // A file that failed or was skipped is reported once, as it finishes.
    while (run.reported < done) {
      const i = run.reported++;
      const error = run.errors[i] ?? null;
      const skipped = !error && run.skips[i] === true;
      if (error || skipped) {
        emit("export_progress", { job_id: run.id, file: run.files[i] ?? "", done: i + 1, total, file_percent: 100, error, skipped });
      }
    }
    emit("export_progress", {
      job_id: run.id,
      file: run.files[index] ?? "",
      done,
      total,
      file_percent: finished ? 100 : Math.round((run.progress - index) * 100),
      error: null,
      skipped: false,
    });
    announceBusy();
  }

  /** Whether any export still has files to write. */
  function exportsBusy(): boolean {
    return [...exportRuns.values()].some((run) => run.progress < run.files.length);
  }

  /** Sends `export_busy` when the queue goes busy or quiet, as the core does. */
  function announceBusy(): void {
    const busy = exportsBusy();
    if (busy === busyAnnounced) return;
    busyAnnounced = busy;
    emitSoon("export_busy", { busy });
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
    // LZW is the default (docs/09), so the command only names another compression, as the mockup's does.
    if (o.format === "tiff" && o.tiff_compression !== "lzw") flags.push(`--compression ${o.tiff_compression}`);
    if (o.format !== "dng") flags.push(`--colour-space ${o.color_space === "display_p3" ? "p3" : o.color_space}`);
    if (o.format !== "dng" && o.size.kind === "long_edge") flags.push(`--long-edge ${o.size.px}`);
    if (o.name_template !== "{name}_{look}") flags.push(`--name "${o.name_template}"`);
    if (o.if_exists === "skip") flags.push("--skip-existing");
    if (o.if_exists === "replace") flags.push("--overwrite");
    flags.push(`--out "${o.destination ?? `${folderPath()} export`}"`);
    const inputs = job.scope === "all" ? folderPath() : recs.map((r) => r.summary.file_name).join(" ");
    return `ym render ${inputs} \\\n  ${flags.join(" \\\n  ")}`;
  }

  // --- The ARRI wizard ----------------------------------------------------------------------

  function recognise(path: string): SourceRecognition {
    const name = fileNameOf(path);
    const fixture = sources.get(name.toLowerCase());
    if (!fixture) {
      return {
        path,
        name,
        source: /\.zip$/i.test(name) ? "zip" : /\.(cube|alf4c?|aml)$/i.test(name) ? "file" : "folder",
        size_bytes: null,
        recognition: { kind: "missing", reason: MISSING_SOURCE },
      };
    }
    return { path, name, source: fixture.source, size_bytes: fixture.size_bytes, recognition: clone(fixture.recognition) };
  }

  /** The simulated copy of `total` files, about three seconds at speed 1 (the mockup's import). */
  function copyFiles(path: string, total: number): Promise<void> {
    return new Promise((resolve) => {
      const step = Math.max(1, Math.ceil(total / 30));
      const importRun: ImportRun = {
        path,
        total,
        done: 0,
        timer: null,
        resolve,
        complete: () => {
          importRun.done = total;
          emit("looks_import_progress", { path, done: total, total, file: `${total}.cube` });
          finish();
        },
      };
      const finish = (): void => {
        if (importRun.timer !== null) clearInterval(importRun.timer);
        importRun.timer = null;
        importRuns.delete(importRun);
        resolve();
      };
      importRuns.add(importRun);
      importRun.timer = setInterval(() => {
        importRun.done = Math.min(total, options.importHoldAt, importRun.done + step);
        emit("looks_import_progress", { path, done: importRun.done, total, file: `${importRun.done}.cube` });
        if (importRun.done >= total) finish();
        // Held: the copy waits where it is until `flush()`.
        else if (importRun.done >= options.importHoldAt && importRun.timer !== null) {
          clearInterval(importRun.timer);
          importRun.timer = null;
        }
      }, 100 / options.speed);
    });
  }

  // --- Lifecycle ------------------------------------------------------------------------

  function stopAll(): void {
    stopDevelopment();
    for (const timer of thumbTimers.values()) clearTimeout(timer);
    thumbTimers.clear();
    for (const run of exportRuns.values()) stopExport(run);
    exportRuns.clear();
    for (const run of importRuns) {
      if (run.timer !== null) clearInterval(run.timer);
      run.timer = null;
      run.resolve();
    }
    importRuns.clear();
    for (const timer of downloadTimers) clearTimeout(timer);
    downloadTimers = [];
    watching = false;
  }

  function resetState(): void {
    stopAll();
    photos = [];
    current = 0;
    developProgress = 0;
    const stored = options.persistPreferences ? readStored() : null;
    preferences = stored ? { ...defaultPreferences(), ...stored.preferences } : defaultPreferences();
    importedPackages = stored?.importedPackages ?? [];
    userLooks = stored?.userLooks ?? [];
    cliLink = stored?.cliLink ?? null;
    displayLuts = importedPackages.flatMap(packageDisplayLuts);
    sources = defaultSources();
    choices.folder = [];
    choices.files = [];
    system.opened = [];
    system.revealed = [];
    system.trashed = [];
    system.restarts = 0;
    looks = buildLooks();
    failures.clear();
    cancelDialog = false;
    viewport.rect = null;
    viewport.mode = { kind: "fit" };
    viewport.before = false;
    viewport.inputs = [];
    frameSeq = 0;
    frameShown = false;
    busyAnnounced = false;
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
      writeStored(null);
      resetState();
    },
    flush() {
      if (developTimer !== null || developProgress < photos.length) {
        developProgress = photos.length;
        applyDevelopProgress();
        stopDevelopment();
      }
      for (const run of [...importRuns]) run.complete();
      for (const run of [...exportRuns.values()]) {
        if (run.progress >= run.files.length) continue;
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
        return openPathsNow(kind === "folder" ? [folderPath()] : SCENES.map((s) => `${folderPath()}/${s.name}.ARW`));
      }),
    sessionState: () => run("sessionState", snapshot),
    closeSession: () =>
      run("closeSession", () => {
        // Development stops with the session; exports in flight carry on, as in the core.
        stopDevelopment();
        for (const timer of thumbTimers.values()) clearTimeout(timer);
        thumbTimers.clear();
        thumbnailUrls.clear();
        photos = [];
        current = 0;
        developProgress = 0;
        viewport.mode = { kind: "fit" };
        viewport.before = false;
        sendFrame();
      }),
    setCurrent: (photoId) =>
      run("setCurrent", () => {
        const index = photos.findIndex((p) => p.summary.id === photoId);
        if (index < 0) fail("app", "That photo is no longer open. Open it again.");
        current = index;
        viewport.mode = { kind: "fit" };
        viewport.before = false;
        emitSoon("render_stats", RENDER_STATS);
        sendFrame();
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
        if (patch.white_pick) {
          // Picking white sets the white balance by hand from the colour under the point.
          merged.white_balance = { kind: "manual", ...whiteFromPick(rec.scene, patch.white_pick) };
        }
        rec.settings = clampSettings(merged as unknown as DevelopSettings);
        emitSoon("render_stats", RENDER_STATS);
        scheduleThumbnail(rec);
        if (rec === photos[current]) sendFrame();
        return resolveSettings(photoId, rec.settings, MOCK_ESTIMATES);
      }),
    applyLook: (scope, lookId, intensity) =>
      run("applyLook", () => {
        const look = findLook(lookId);
        if (!look || look.missing) fail("looks", "That look isn't in the library any more. Import it again.");
        for (const rec of targets(scope)) {
          rec.settings = { ...rec.settings, look: lookId, look_intensity: clamp(intensity, 0, 100) };
          scheduleThumbnail(rec);
        }
        emitSoon("render_stats", RENDER_STATS);
        sendFrame();
      }),
    rerunAuto: (photoId) =>
      run("rerunAuto", () => {
        const rec = photo(photoId);
        // Exposure, white (the lift's offset), white balance (including the light preset and a picked white) and noise.
        rec.settings = {
          ...rec.settings,
          exposure: { kind: "auto" },
          white_offset_ev: 0,
          light_source: "auto",
          white_balance: { kind: "auto" },
          white_pick: null,
          noise_reduction: { kind: "auto" },
        };
        emitSoon("render_stats", RENDER_STATS);
        scheduleThumbnail(rec);
        if (rec === photos[current]) sendFrame();
        return resolveSettings(photoId, rec.settings, MOCK_ESTIMATES);
      }),
    histogram: (photoId) =>
      run("histogram", (): HistogramData => {
        const rec = photo(photoId);
        const { values } = resolveSettings(photoId, rec.settings, MOCK_ESTIMATES);
        // Exposure moves the plot along the tonal scale; the camera-log view is flatter and lower.
        const shift = (values.exposure_ev - MOCK_ESTIMATES.exposure_ev) * 0.06 + (viewport.before ? -0.05 : 0);
        return sceneHistogram(developed(rec), shift);
      }),

    viewportInfo: () => run("viewportInfo", (): ViewportInfo | null => (options.viewport === null ? null : { backend: options.viewport })),
    viewportSetRect: (rect) =>
      run("viewportSetRect", () => {
        viewport.rect = rect;
        // The core draws a frame when the cell is placed or resized, and reports it.
        emitSoon("render_stats", RENDER_STATS);
        sendFrame();
      }),
    viewportInput: (event) =>
      run("viewportInput", () => {
        viewport.inputs.push(event);
        const rec = photos[current];
        if (event.kind === "toggle_zoom") {
          viewport.mode = viewport.mode.kind === "fit" ? { kind: "percent", value: 100 } : { kind: "fit" };
        } else if (event.kind === "pick_white" && rec) {
          // The core finds the spot of the photo under the click (straightening, turns and lens crop included) and sets the pick; a click
          // on the paper picks nothing. Like `update_settings` with a pick, the mock sets the white balance by hand from the colour there.
          const point = photoPointAt(rec, event.x, event.y);
          if (point) {
            rec.settings = clampSettings({
              ...rec.settings,
              white_pick: point,
              light_source: "auto",
              white_balance: { kind: "manual", ...whiteFromPick(rec.scene, point) },
            });
            emitSoon("settings_changed", { photo_id: rec.summary.id });
            emitSoon("render_stats", RENDER_STATS);
            scheduleThumbnail(rec);
            sendFrame();
          }
        }
      }),
    viewportSetMode: (mode) =>
      run("viewportSetMode", () => {
        viewport.mode = mode;
        sendFrame();
      }),
    setBefore: (before) =>
      run("setBefore", () => {
        viewport.before = before;
        sendFrame();
      }),
    samplePixel: (x, y) =>
      run("samplePixel", (): PixelSample => {
        const rec = photos[current];
        if (!rec) fail("viewport", "Open a photo first.");
        // On the paper around a letterboxed picture there is nothing to read.
        const point = photoPointAt(rec, x, y);
        if (!point) fail("viewport", "That point is outside the photo.");
        const layer = layerAt(rec.scene, clamp(point[0], 0, 1), clamp(point[1], 0, 1));
        const [r, g, b] = hexToRgb(rec.scene[layer]);
        return { r, g, b };
      }),

    looksList: () => run("looksList", () => looks),
    lookThumbnail: (photoId, lookId) =>
      run("lookThumbnail", (): string => {
        const rec = photo(photoId);
        const look = findLook(lookId);
        if (!look) fail("looks", "That look isn't in the library any more. Import it again.");
        if (look.missing) fail("looks", "File missing or changed. Import again.");
        // The shell answers with an address (`ym-frame://…/look-<photo>-<n>.png`); here a `blob:` URL stands for it, new for every request.
        return bytesToUrl(Array.from(scenePng(developed(rec, lookId, 100))));
      }),
    looksImport: (path, input) =>
      runAsync("looksImport", async (): Promise<LooksImportResult> => {
        const found = recognise(path);
        const r = found.recognition;
        if (r.kind === "missing" || r.kind === "unusable") fail("looks", r.kind === "missing" ? MISSING_SOURCE : r.reason);
        if (r.kind === "needs_answer" && !input) {
          fail("looks", "Say what the LUT expects as input: LogC4, LogC3, or LogC3 with Rec.709 baked in.");
        }
        await copyFiles(path, r.kind === "package" ? r.file_count : r.file_count);
        const before = new Set(looks.map((l) => l.id));
        const lutsBefore = displayLuts.length;
        if (r.kind === "package") {
          if (!importedPackages.includes(r.package_id)) importedPackages = [...importedPackages, r.package_id];
          displayLuts = importedPackages.flatMap(packageDisplayLuts);
        } else {
          const base = cubeName(found.name);
          const slug = slugify(base);
          const made: Look[] = Array.from({ length: r.file_count }, (_, i) => ({
            id: `user:${slug}${r.file_count > 1 ? `-${i + 1}` : ""}`,
            name: r.file_count > 1 ? `${base} ${i + 1}` : base,
            group: { kind: "user" },
            // A look file says what it expects; a `.cube` is what the person answered.
            input: (r.kind === "look_files" ? r.input : null) ?? input ?? "logc4",
            is_default: false,
            missing: false,
          }));
          userLooks = [...userLooks.filter((l) => !made.some((m) => m.id === l.id)), ...made];
        }
        rebuildLooks();
        persist();
        const added = looks.filter((l) => !before.has(l.id));
        const displayAdded = displayLuts.length - lutsBefore;
        emitSoon("looks_changed", null);
        return { added, display_luts_added: displayAdded, unchanged: Math.max(0, r.file_count - added.length - displayAdded) };
      }),
    looksRecognise: (path) => run("looksRecognise", () => recognise(path)),
    looksDisplayLuts: () => run("looksDisplayLuts", () => displayLuts),
    looksWatchDownloads: (enabled) =>
      run("looksWatchDownloads", (): WatchStatus => {
        for (const timer of downloadTimers) clearTimeout(timer);
        downloadTimers = [];
        if (!enabled) {
          watching = false;
          return { watching: false, problem: null };
        }
        if (options.downloadsDenied) {
          watching = false;
          return {
            watching: false,
            problem:
              "Yellowmoon can't open your Downloads folder. Allow it in System Settings, under Privacy & Security, or choose the file yourself.",
          };
        }
        watching = true;
        options.downloads.forEach((name, i) => {
          downloadTimers.push(setTimeout(() => api.simulateDownload(name), (2500 * (i + 1)) / options.speed));
        });
        return { watching: true, problem: null };
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
          adjusted_photos: adjustedPhotos(job),
          dng_approximate: dngApproximate(job),
        };
      }),
    exportStart: (job) =>
      run("exportStart", () => {
        const planned = exportFiles(job);
        if (planned.files.length === 0) {
          fail("export", "No photos are picked. Pick some photos, or choose All, then export again.");
        }
        const exportRun: ExportRun = {
          id: nextJobId++,
          files: planned.files,
          progress: 0,
          reported: 0,
          errors: planned.errors,
          skips: planned.skips,
          timer: null,
        };
        exportRuns.set(exportRun.id, exportRun);
        announceBusy();
        queueMicrotask(() => {
          if (exportRun.timer !== null) emitExportProgress(exportRun);
        });
        exportRun.timer = setInterval(() => {
          exportRun.progress = Math.min(exportRun.files.length, options.exportHoldAt, exportRun.progress + EXPORT_STEP);
          emitExportProgress(exportRun);
          // Finished, or held where a screenshot wants it: `flush()` carries a held run on.
          if (exportRun.progress >= exportRun.files.length || exportRun.progress >= options.exportHoldAt) stopExport(exportRun);
        }, EXPORT_TICK_MS / options.speed);
        return exportRun.id;
      }),
    exportCancel: (jobId) =>
      run("exportCancel", () => {
        const found = exportRuns.get(jobId);
        if (!found) return;
        stopExport(found);
        exportRuns.delete(jobId);
        announceBusy();
      }),
    cliPreview: (job) => run("cliPreview", () => cliCommand(job)),
    exportBusy: () => run("exportBusy", exportsBusy),
    restartApp: () =>
      run("restartApp", () => {
        // The shell's guard: no restart while an export is queued or running. (The mock has no process to replace.)
        if (exportsBusy()) fail("export", "An export is still running. Restart after it finishes.");
        system.restarts += 1;
      }),
    thumbnails: () => run("thumbnails", () => [...thumbnailUrls].map(([id, url]) => ({ id, url }))),

    preferencesGet: () => run("preferencesGet", () => preferences),
    preferencesSet: (patch: PreferencesPatch) =>
      run("preferencesSet", () => {
        if (patch.default_look !== undefined) {
          const look = findLook(patch.default_look);
          if (!look || look.missing) fail("preferences", "That look isn't in the library. Pick another default look.");
        }
        const next = { ...preferences } as Record<string, unknown>;
        for (const [key, value] of Object.entries(patch)) if (value !== undefined) next[key] = value;
        if (patch.display_rendering?.kind === "arri_display") {
          const wanted = patch.display_rendering.id;
          if (!displayLuts.some((l) => l.id === wanted)) {
            fail("preferences", "Those display LUTs aren't imported. Import ARRI's display LUTs first.");
          }
        }
        preferences = next as unknown as Preferences;
        if (patch.default_look !== undefined) {
          syncDefaultLook();
          emitSoon("looks_changed", null);
        }
        persist();
        return preferences;
      }),
    installCliLink: () =>
      run("installCliLink", (): CliLinkResult => {
        cliLink = "/usr/local/bin/ym";
        persist();
        return { path: cliLink };
      }),
    cliLinkStatus: () => run("cliLinkStatus", (): CliLinkStatus => ({ installed: cliLink !== null, path: cliLink })),
    appInfo: () => run("appInfo", (): AppInfo => ({ version: APP_VERSION, pipeline_version: "1.0.0" })),

    chooseFolder: (start) =>
      run("chooseFolder", () => {
        void start;
        return choices.folder.length > 0 ? (choices.folder.shift() ?? null) : "~/Pictures/Chosen export";
      }),
    chooseLookFiles: () =>
      run("chooseLookFiles", () =>
        choices.files.length > 0 ? (choices.files.shift() ?? null) : [`~/Downloads/${PACKAGE_FACTS.look_library_logc4.zip}`],
      ),
    openUrl: (url) =>
      run("openUrl", () => {
        system.opened.push(url);
      }),
    revealInFinder: (path) =>
      run("revealInFinder", () => {
        system.revealed.push(path);
      }),
    trashPath: (path) =>
      run("trashPath", () => {
        system.trashed.push(path);
        sources.delete(fileNameOf(path).toLowerCase());
      }),

    get system() {
      return system;
    },
    queueChoice(kind: "folder" | "files", answer: string | string[] | null) {
      if (kind === "folder") choices.folder.push(answer as string | null);
      else choices.files.push(answer as string[] | null);
    },
    emitEvent(event, payload) {
      emit(event, payload);
    },
    addSource(fileName, fixture) {
      sources.set(fileName.toLowerCase(), clone(fixture));
    },
    removeSource(fileName) {
      sources.delete(fileName.toLowerCase());
    },
    simulateDownload(fileName) {
      if (!watching) return false;
      const fixture = sources.get(fileName.toLowerCase());
      const hint = fixture?.recognition.kind === "package" ? fixture.recognition.package_id : null;
      const found: DownloadFound = {
        path: `~/Downloads/${fileName}`,
        file_name: fileName,
        size_bytes: fixture?.size_bytes ?? 0,
        package_hint: hint,
      };
      emit("download_found", found);
      return true;
    },

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
 * The shared mock `getApi()` returns outside Tauri. Its options can be set from the page URL
 * (`/?mockSpeed=20`, see `optionsFromLocation`) so end-to-end tests don't wait for the mockup's
 * timers and start in the state they need.
 */
export const mockApi: MockApi = createMockApi(optionsFromLocation());
