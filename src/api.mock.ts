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
 *    grading (`fixtures/grade.ts`);
 *  - node thumbnails (`node_thumbnails`, then `node_thumbnails_ready` with the pictures that changed): the flat scene graded
 *    through the nodes up to each card and the Output's look (`nodeScene`), as far as the flat grade can show.
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
import { APPLE_TABLE, MOCK_DEFAULT_INSTRUCTIONS, MOCK_INSTRUCTIONS_MAX_CHARS } from "./api/appleAutoMock";
import {
  consentRefusal,
  MOCK_AGY_INFO,
  MOCK_APPLE_INFO,
  MOCK_CLAUDE_INFO,
  mockActivity,
  mockAutoStatus,
  mockInstructionsId,
  mockPhotoState,
  parseMockConsent,
  parseMockProviders,
} from "./api/providersMock";
import { UNREADABLE_PREFIX } from "./api/openMessages";
import type { MockActivity, MockProviders } from "./api/providersMock";
import type {
  ApiEventName,
  ApiEvents,
  AppError,
  AppInfo,
  DisplayOptions,
  AppleAuto,
  AppleAutoAnswer,
  ApplyScope,
  AutoConsent,
  AutoEstimates,
  AutoProvider,
  AutoStatus,
  BalanceSetting,
  ChromaSample,
  ChromaScatter,
  CliLinkResult,
  CliLinkStatus,
  ColorBalancePoint,
  CropAspect,
  CropSuggestion,
  DepthFar,
  DevelopSettings,
  DownloadFound,
  ExportEstimate,
  ExportJob,
  ExportJobId,
  FocusPlaneResult,
  ExportOptions,
  GradeValues,
  HistogramData,
  LensInfo,
  LensKind,
  LensSettings,
  LightSource,
  Look,
  LookFormat,
  LookGroup,
  LookId,
  LookInput,
  LookLibrary,
  LooksImportResult,
  MainEstimates,
  MapFrom,
  MaskFindResult,
  MaskKind,
  MaskView,
  MenuState,
  OpenKind,
  NodeGrade,
  NodeId,
  NodeItem,
  NodeThumbnail,
  Orientation,
  PackageId,
  PasteReport,
  PercentSetting,
  PhotoId,
  PhotoStage,
  PhotoStatus,
  PhotoSummary,
  PixelSample,
  Preferences,
  PreferencesPatch,
  Recognition,
  RenderStats,
  ResolvedCorrector,
  ResolvedSettings,
  ResolvedValues,
  SessionChange,
  SessionState,
  SettingsPatch,
  SourceKind,
  SourceRecognition,
  ThumbnailReady,
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
import { clampPins } from "./components/warpMath";
import { adjustForNode, adjustScene, developScene, nodeScene, previewSvg, signalScene } from "./fixtures/grade";
import type { Adjustment, NodeLook } from "./fixtures/grade";
import { filmScene, mockFilmCatalogue, MOCK_FILM_STOCKS, sceneThroughFilms } from "./fixtures/film";
import { hasMixers, outputFeeder, sceneAfter, sceneAt } from "./fixtures/mixers";
import { hexChroma, sceneScatter } from "./fixtures/warp";
import { sceneHistogram } from "./fixtures/histogram";
import { scenePng, hexToRgb } from "./fixtures/png";
import { layerAt, SCENES, type Scene } from "./fixtures/scenes";
import { canvasOf, fitInCell, fitInside, imageAspectOf } from "./screens/crop/cropMath";
import { copiedPatch, pastedOnto, refusalOfPatch, validateGraph } from "./fixtures/graph";
import { hasBackgroundRow, hasObjectRow, maskViewSvg, masksOfView, MOCK_DEPTH, pickDepthFar, SCENES_WITH_PEOPLE, visionFoundIn, visionRowKinds } from "./fixtures/maskView";
import { diffSettings } from "./state/history";
import { cardStates, linkInto } from "./state/graphOps";
import {
  applyPatch,
  correctorsOf,
  defaultSettings,
  foldPatches,
  INPUT_ID,
  inputOf,
  inputPatch,
  isGlyphItem,
  MAIN_ID,
  MIXER_NAMES,
  mainGradePatch,
  mainOf,
  neutralGrade,
  nodeOf,
  OUTPUT_ID,
  outputOf,
  outputPatch,
  withoutLook,
} from "./state/settingsGraph";
import type { CorrectorItem, FilmItem, InputItem, MixerItem } from "./state/settingsGraph";
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

/** How long a frame takes the mock with `cpu` (divided by `speed`): long enough for "Rendering on the CPU" to show. */
const CPU_FRAME_MS = 600;
/** How long the mock takes to make a photo's blur map for a Background row the first time (divided by `speed`): long enough for "Finding…" to show. */
export const FINDING_MS = 400;
/** How long the mock's encoder takes to get a photo with an Object row ready (divided by `speed`): long enough for "Getting ready…" to show. */
export const ENCODER_MS = 900;
/** How long one `mask_find` takes the mock (divided by `speed`): long enough for "Finding…" to show. */
export const FIND_MS = 120;
/** How long the mock's Vision takes to make a photo's Subject and People masks, in the background (divided by `speed`): long enough for "Finding…" to show. */
export const VISION_MS = 500;
/** How long the mock takes to make a photo's depth map in the background (divided by `speed`): long enough for "Finding…" to show. */
export const DEPTH_MS = 500;
/** The flat scene whose focus point the mock puts at the exact centre of the frame. */
export const FOCUS_AT_CENTRE_SCENE = "DSC04212";
/** The sentence the mock's depth worker fails with (`?mockDepth=fail`): the core's "The Depth mask can't be made: <the model's text>." */
export const DEPTH_FAILED = "The Depth mask can't be made: the model stopped before it finished.";
/** What the decoder's time reads in the mock's "Object found" pill: 0.4 s. */
export const FOUND_MS = 412;

// The mock has no zoom of its own to report: the screen estimates it (`fit_percent: null`).
const RENDER_STATS: RenderStats = { ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null };

/** What "Auto" resolves to for every photo in the mock, with Apple Auto off (Default Auto, "the rule"). */
export const MOCK_ESTIMATES: AutoEstimates = {
  input: {
    exposure_ev: 0.3,
    kelvin: 5450,
    tint: 0,
    sharpen: { amount: 120, radius_px: 0.8 },
    noise_reduction: { luminance: 30, color: 50, detail: 50 },
  },
  main: {
    contrast: 0,
    saturation: 0,
    highlights: 0,
    shadows: 0,
    black: 0,
    balance_shadows: [0, 0],
    balance_highlights: [0, 0],
  },
  source: "rule",
  provider: "yellowmoon",
  apple: { state: "off" },
  apple_steps: null,
  light_class: "daylight",
  light_confidence: 0.82,
  noise: { a: 1.9e-5, b: 3.1e-7 },
};

/**
 * How the mock plays the photos' Auto states (`docs/19`; `?mockAppleAuto=…`), now of the provider that grades (`docs/22` section 12.8).
 * `none` (the default) is the browser build and every computer the app has no advisor on: no provider, no reason, every photo `off`, so
 * every screen is the one without a provider. `unavailable` is a Mac where Apple Intelligence is off. `pending`, `answered` and `failed`
 * are a Mac where Apple Intelligence is available: with it chosen, every developed photo is waiting for its answer, has the answer of
 * {@link MOCK_APPLE_ANSWER}, or failed (no answer within 30 seconds); with Default Auto chosen every photo is `off`. Where
 * `providers` names the computer's providers (`?mockProviders=`) the mode applies to the one that grades, and `none` or `unavailable`
 * then read as `answered`.
 */
export { APPLE_TABLE, MOCK_DEFAULT_INSTRUCTIONS, MOCK_INSTRUCTIONS_MAX_CHARS };
export { MOCK_AGY_INFO, MOCK_APPLE_INFO, MOCK_CLAUDE_INFO, mockInstructionsId };

export type MockAppleAuto = "none" | "unavailable" | "pending" | "answered" | "failed";

const APPLE_AUTO_MODES: readonly MockAppleAuto[] = ["none", "unavailable", "pending", "answered", "failed"];

/**
 * The answer every photo gets while the mock's provider answers: a little brighter and warmer, darker highlights, deeper blacks,
 * more contrast, less saturation, teal shadows and warm highlights; it leaves the shadows and the tint empty (nothing set).
 */
export const MOCK_APPLE_ANSWER: AppleAutoAnswer = {
  steps: {
    exposure: 1,
    highlights: -1,
    shadows: null,
    blacks: -1,
    contrast: 1,
    white: 1,
    tint: null,
    saturation: -1,
    shadow_color: "teal",
    shadow_color_strength: 2,
    highlight_color: "warm",
    highlight_color_strength: 1,
  },
  scene: "Street in late sun",
  plan: "Teal shadows and warm highlights for a cinematic late-sun street",
};

/** The wheel's point of a colour at a strength (`docs/19` section 5), `[0, 0]` for none or a colour without a strength. */
function colorPoint(direction: ColorBalancePoint | undefined, strength: number | null): ColorBalancePoint {
  if (!direction || strength === null) return [0, 0];
  const r = APPLE_TABLE.strength_radius[strength] ?? 0;
  return [r * direction[0], r * direction[1]];
}

/** The providers a mock computer has when `providers` is not set: what `appleAuto` says about Apple Intelligence, and nothing else. */
export function providersOfMode(mode: MockAppleAuto): MockProviders {
  if (mode === "none") return {};
  return { apple: mode === "unavailable" ? "unavailable" : "available" };
}

/**
 * Where a photo stands with the mock's Auto: the activity (who acts, `Activity::decide`) played in the given mode, with the
 * default instructions or the person's own (the core's `Activity::photo_state`).
 */
export function mockAppleState(mode: MockAppleAuto, activity: MockActivity, ownInstructions = ""): AppleAuto {
  return mockPhotoState(activity, mode, MOCK_APPLE_ANSWER, ownInstructions);
}

/**
 * What Auto resolves to for a photo of the mock: {@link MOCK_ESTIMATES}, with the answer's values while a provider has answered
 * (`docs/19` section 5: the exposure added to the rule's, the white made warmer along the locus and read to 50 K, the tint added to
 * the scene's, the sliders and the wheels from the table; a step left empty sets nothing).
 */
export function mockEstimates(apple: AppleAuto, provider: AutoProvider = "yellowmoon"): AutoEstimates {
  if (apple.state !== "answered") return { ...MOCK_ESTIMATES, apple, provider };
  const { steps } = apple.answer;
  const at = (table: Record<number, number>, step: number | null): number => (step === null ? 0 : (table[step] ?? 0));
  const mired = 1e6 / MOCK_ESTIMATES.input.kelvin - at(APPLE_TABLE.warmer_mired, steps.white);
  const answeredBy = apple.info.provider;
  return {
    ...MOCK_ESTIMATES,
    input: {
      ...MOCK_ESTIMATES.input,
      exposure_ev: Math.round((MOCK_ESTIMATES.input.exposure_ev + at(APPLE_TABLE.exposure_ev, steps.exposure)) * 100) / 100,
      kelvin: Math.round(1e6 / mired / 50) * 50,
      tint: clamp(MOCK_ESTIMATES.input.tint + at(APPLE_TABLE.tint, steps.tint), -50, 50),
    },
    main: {
      contrast: at(APPLE_TABLE.percent, steps.contrast),
      saturation: at(APPLE_TABLE.percent, steps.saturation),
      highlights: at(APPLE_TABLE.highlights, steps.highlights),
      shadows: at(APPLE_TABLE.percent, steps.shadows),
      black: at(APPLE_TABLE.percent, steps.blacks),
      balance_shadows: colorPoint(
        steps.shadow_color === null ? undefined : APPLE_TABLE.shadow_directions[steps.shadow_color],
        steps.shadow_color === "none" ? 0 : steps.shadow_color_strength,
      ),
      balance_highlights: colorPoint(
        steps.highlight_color === null ? undefined : APPLE_TABLE.highlight_directions[steps.highlight_color],
        steps.highlight_color === "none" ? 0 : steps.highlight_color_strength,
      ),
    },
    source: answeredBy === "yellowmoon" ? "rule" : answeredBy,
    provider: answeredBy,
    apple,
    apple_steps: steps,
  };
}

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
/** How long the mock takes to make the thumbnails of a photo's nodes after they are asked for (the core makes them behind the answer). */
const NODE_THUMB_MS = 60;
/** A node thumbnail is this wide and this tall, in pixels (the core's are about 160 px on the long edge). */
const NODE_THUMB_SIZE: readonly [number, number] = [160, 107];

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

/**
 * The library a look is listed under, by the core's rule (`LookLibrary::classify` in `crates/ym-looks/src/library/model.rs`): the
 * mock plays the core, so it has to say the same. Built-in group: built in; user group: Your LUTs; ARRI group: by input, with
 * an `.aml` of LogC3 to Rec.709 being the ALF2 library.
 */
export function libraryOf(group: LookGroup, input: LookInput, format: LookFormat | null): LookLibrary {
  if (group.kind === "built_in") return "built_in";
  if (group.kind === "user") return "user";
  if (input === "logc4") return "arri_logc4";
  if (input === "logc3_ei800") return "arri_logc3";
  return format === "aml" ? "arri_alf2" : "arri_logc3_to_rec709";
}

const ARRI_FILM: LookGroup = { kind: "arri", no: 3000, name: "Film" };

/** An imported look as the core lists it: group, input, kind of file and table size, with the library worked out from them. */
function importedLook(id: LookId, name: string, group: LookGroup, input: LookInput, format: LookFormat, lutSize: number): Look {
  return { id, name, group, input, library: libraryOf(group, input, format), format, lut_size: lutSize, is_default: false, missing: false };
}

/** A built-in look: no file, so no kind of file and no table size. */
function builtInLook(id: LookId, name: string): Look {
  return { id, name, group: { kind: "built_in" }, input: "logc4", library: "built_in", format: null, lut_size: null, is_default: false, missing: false };
}

/**
 * The looks `mockLibraries=all` adds so every section of the Look tab has some: ARRI's titles repeat from library to library (as they do
 * in the real ones), the ALF2 library has its three strengths, and the person's own LUTs hold one of each kind of file, one of them with
 * a title ARRI uses too.
 */
function sectionLooks(): Look[] {
  const titles = ["Film warm", "Film cool", "Bleach bypass", "Teal and orange", "Soft print"];
  const logc3 = titles.map((t) => importedLook(`arri:3000:${slugify(t)}-logc3`, t, ARRI_FILM, "logc3_ei800", "cube", 33));
  const rec709 = titles.slice(0, 4).map((t) => importedLook(`arri:3000:${slugify(t)}-logc3-rec709`, t, ARRI_FILM, "logc3_to_rec709", "cube", 33));
  const alf2 = ["Film warm", "Soft print"].flatMap((t) =>
    ["Low", "Medium", "High"].map((strength) =>
      importedLook(`arri:3000:${slugify(t)}-${strength.toLowerCase()}`, `${t} · ${strength}`, ARRI_FILM, "logc3_to_rec709", "aml", 33),
    ),
  );
  const own: Look[] = [
    importedLook("user:warm-film", "Warm Film", { kind: "user" }, "logc4", "cube", 33),
    importedLook("user:film-cool", "Film cool", { kind: "user" }, "logc4", "cube", 65),
    importedLook("user:warm-pull", "Warm Pull", { kind: "user" }, "logc4", "alf4", 33),
    importedLook("user:custom-look", "Custom Look", { kind: "user" }, "logc4", "alf4c", 33),
    importedLook("user:night", "Night", { kind: "user" }, "logc3_to_rec709", "aml", 33),
  ];
  return [...logc3, ...rec709, ...alf2, ...own];
}

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
  /**
   * The name of a folder that cannot be read (`?mockUnreadable=Name`, `docs/26` H-02): opening a folder with nothing else answers
   * "Yellowmoon can't read the folder Name. …" (`UNREADABLE_PREFIX`), as the core does, and no session starts. Default: none.
   */
  unreadable?: string;
  /**
   * `?mockSlowOpen=2000` (`docs/26` H-10): opening takes a while, as reading 2000 files does. `open_progress` reports 50 files at a
   * time (about 100 ms apart at speed 1) until `cancelOpen` or the end, so Start and Photos can show "Reading {done} of {total} files…"
   * and Cancel. Default 0: no delay.
   */
  slowOpen?: number;
  /**
   * `?mockThumbnailBudget=40` (`docs/26` H-11): the mock keeps at most this many photos' thumbnails, as the shell keeps a byte budget.
   * The photo whose thumbnail was made last stays; the photos used least recently lose theirs (a thumbnail made, or asked for by
   * `thumbnailsInView`, is a use) and get a new one when `thumbnailsInView` names them. Default 0: no budget.
   */
  thumbnailBudget?: number;
  /** How many photos the session has. The twelve fixtures repeat, with the names DSC04213 and on, so a long folder can be shown and timed. Default 12. */
  photoCount?: number;
  /** Whether the imported looks of the mockups are in the library. Default true. */
  looksImported?: boolean;
  /**
   * Looks in every section of the Look tab as well: ARRI's LogC3, LogC3 to Rec.709 and ALF2 libraries (the same titles as the LogC4
   * one, as in the real libraries) and the person's own LUTs of every kind of file (`?mockLibraries=all`). Default false.
   */
  allLibraries?: boolean;
  /** Slugs of imported looks whose file is "missing or changed" ("film-warm"): listed with `missing: true`. */
  missingLooks?: string[];
  /** The lens data could not be loaded (`?mockLenses=0`): `lens_list` answers with no lens, as the core does (`docs/21` §10). Default false. */
  lensesMissing?: boolean;
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
  /**
   * Photos whose export fails, by name ("DSC04204"), with the reason. A reason that is a kind of failure of the destination
   * (`disk_full`, `read_only`, `gone`, `permission`) is the core's sentence for it and **stops** the export at that photo: the
   * photos after it are not tried (`docs/26` P2-2).
   */
  exportFailures?: Record<string, string>;
  /**
   * Photos whose file goes away after they are Ready (`?mockVanish=DSC04207`, `docs/26` H-04), by name. The file is missed the next time
   * it is read: when the photo is shown (it becomes the current photo) or when an export reaches it. The photo then becomes `skipped`
   * with the core's sentence for a file that is gone (`FILE_GONE_MESSAGE`) and a `photo_progress` error event is sent, so the filmstrip
   * says CAN'T OPEN and the cell says why. In an export that file fails with the same sentence and the others go on.
   */
  vanish?: string[];
  /**
   * Bytes free in the destination folder (`?mockFreeBytes=200000000`). `export_estimate` says `low_space` when it is under 1.35 times the
   * estimate (`docs/26` P2-3). Default 500 GB.
   */
  freeBytes?: number;
  /** An export stops at this progress, in photos (1.45 is one done and the second at 45 %) until `flush()`, so a screenshot can show the running state. Default: never. */
  exportHoldAt?: number;
  /** A look import stops after this many files and waits for `flush()`, so a screenshot can show it half way (the mockup's 40 of 87). Default: never. */
  importHoldAt?: number;
  /** Keep the preferences in `localStorage` so a reload of the page keeps them (`?mockPersist=1`). Default false. */
  persistPreferences?: boolean;
  /**
   * How the mock plays the photos' Auto states (`?mockAppleAuto=none|unavailable|pending|answered|failed`). Default `none`: no provider.
   * Unless `providers` is set it also says whether Apple Intelligence is on this computer: anything but `none` is a Mac with it
   * (`unavailable`: switched off).
   */
  appleAuto?: MockAppleAuto;
  /**
   * The providers this computer has and in what state (`?mockProviders=apple,claude:signed_out,agy:too_old`, `docs/22` section 12.8): `null`
   * (the default) is what `appleAuto` says; `{}` is none (the "Auto grading" row is hidden). A provider that is not named is not installed.
   */
  providers?: MockProviders | null;
  /** The saved `auto_provider` a fresh mock starts with (`?mockAutoProvider=claude`). Default `apple`. */
  autoProvider?: AutoProvider;
  /** The saved `auto_consent` a fresh mock starts with (`?mockConsent=claude,agy`). Default: none. */
  consent?: AutoConsent;
  /** The operating system the mock says it runs on, for the words that differ on Windows (`?mockOs=windows`). Default `macos`. */
  os?: "macos" | "windows" | "linux";
  /**
   * The viewport backend the mock claims to be (`?mockViewport=native|fallback`). `null` (the default): none, and the Photos screen
   * draws the fixture picture itself, as a browser build does. `native`: the screen leaves the cell transparent, as in the app on a
   * Mac. `fallback`: the mock sends `viewport_frame` events with a picture of the cell's size, which the screen shows.
   */
  viewport?: ViewportBackendKind | null;
  /**
   * Render like a core whose raw stages run on the CPU (`?mockCpu=1`): every frame the mock "renders" (a photo shown, its settings,
   * a look, the zoom, the cell, Before) is preceded by `viewport_busy { busy: true, cpu: true }` and followed, `CPU_FRAME_MS` (600 ms)
   * divided by `speed` later, by `viewport_busy { busy: false, cpu: true }`; a frame asked for meanwhile makes the wait start again, as
   * back-to-back rounds of the core do. Default false: no `viewport_busy` at all, as on a graphics card, where a frame takes
   * milliseconds. Whatever `cpu` says, the first frame of a photo that has a Background row with the eye on takes `FINDING_MS` (400 ms) divided by `speed`
   * and is announced the same way with `finding: true` (the core is making the photo's blur map; the rows say "Finding…"), once for each photo.
   */
  cpu?: boolean;
  /**
   * The kinds of mask the mock says this machine can make (`?mockMasks=`, `docs/23` §9, answered by `mask_kinds`). The default is every kind
   * (`all`, also `built`), `noai` leaves out Object, Subject and People, `nomac` leaves out Subject and People (a Mac that cannot make them: the
   * menu hides them and a row of the kind says "Subject masks need a Mac that supports them"), and a comma list names the kinds.
   */
  maskKinds?: readonly MaskKind[];
  /**
   * What `mask_find` answers (`?mockMaskFind=`, `docs/23` §9): `scene` (the default) finds the flat scene's mountain, the far hill, when the click or the
   * box is on it and answers nothing found anywhere else; `none` always answers nothing found.
   */
  maskFind?: "scene" | "none";
  /**
   * What the mock's depth worker does (`?mockDepth=`, `docs/23` §4.8): `ok` (the default) makes the map; `fail` fails the first time for each photo, with the
   * sentence "The Depth mask can't be made: the model stopped before it finished.", and makes it when `retry_depth` asks again; `fail-always` fails every time.
   */
  depth?: "ok" | "fail" | "fail-always";
  /**
   * What `display_options` answers (`?mockDisplay=reduce-transparency,reduce-motion,increase-contrast`, any of the three, `docs/28` A-P12): the
   * system settings the shell reads on a Mac. The default is all off, as on a computer that has none of them on and off macOS. A later change
   * is played with `emitEvent("display_options", …)`.
   */
  displayOptions?: DisplayOptions;
}

/** What the viewport commands have recorded, for tests. */
export interface MockViewportState {
  rect: ViewportRect | null;
  mode: ViewportMode;
  before: boolean;
  /** The mask view the page asked for with `viewport_set_mode`: a corrector's mask over the photo or alone, or none. */
  maskView: MaskView | null;
  inputs: ViewportInput[];
  /**
   * How far the picture has been moved since it was last in the middle, in CSS pixels: the sum of the `pan` inputs while the zoom is not Fit.
   * Fit, a new photo and a fresh `viewport_set_mode` to Fit put it back to 0. The mock has no picture edges, so it is not clamped (the core's
   * clamp is tested in Rust).
   */
  offset: { x: number; y: number };
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
  /** The last state `set_menu_state` was given (which of the Edit menu's settings items are available); null before the first. */
  readonly menuState: Readonly<MenuState> | null;
  /** What the system commands were asked to do, in order: pages opened, folders shown in Finder, files moved to the Trash. */
  readonly system: Readonly<MockSystemLog>;
  /**
   * The answer of the next `chooseFolder` or `chooseLookFiles`: a path (or paths), or `null` for a cancelled
   * dialog. Answers queue up. Without one `chooseFolder` answers "~/Pictures/Chosen export" and `chooseLookFiles`
   * the Look Library · LogC4 zip.
   */
  queueChoice(kind: "folder", answer: string | null): void;
  queueChoice(kind: "files", answer: string[] | null): void;
  /**
   * The paths the next `openDialog` opens (instead of the twelve fixture files or the fixture folder), whichever kind it asks for.
   * A name the session does not have is a new photo; one it has is "already open". Answers queue up.
   */
  queueChoice(kind: "open", answer: string[]): void;
  /** Add (or replace) a synthetic file the recogniser can find, by file name. */
  addSource(fileName: string, fixture: SourceFixture): void;
  /** Take a synthetic file away, as if it had been deleted or moved: it is `missing` from now on. */
  removeSource(fileName: string): void;
  /** Send an event to the listeners, as the core would: a `menu_action` from the native menu, an `update_state`, an `error`. */
  emitEvent<E extends ApiEventName>(event: E, payload: ApiEvents[E]): void;
  /** Make a zip "land in Downloads": a `download_found` event when the watcher is on; resolves to whether it was sent. */
  simulateDownload(fileName: string): boolean;
  /**
   * Change how Auto behaves, as when its answers arrive (`pending` to `answered` or `failed`) or Apple Intelligence is switched on
   * or off: every developed photo whose Auto values or state change gets `settings_changed` and a new thumbnail, as the shell sends for
   * the core's `estimates_changed`.
   */
  setAppleAuto(mode: MockAppleAuto): void;
  /**
   * Change the providers the computer has, as when a program is installed, signed in or out, or removed (`docs/22` section 12.8): the
   * photos' states follow as with `setAppleAuto`, and `auto_status_changed` is sent. `null` goes back to what `appleAuto` says.
   */
  setProviders(providers: MockProviders | null): void;
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

/** The settings of a photo nobody has touched: the default graph, Auto wherever there is Auto, the given look at 100 % (`docs/23` section 3.1). */
export { defaultSettings };

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

/** Auto, in a patch. */
const AUTO = { kind: "auto" } as const;

/** The ids of the seven lenses, in list order (`LensKind`, `docs/21` §1). */
export const LENS_KINDS: readonly LensKind[] = ["default", "classic", "vintage", "petzval", "triplet", "sonnar", "detuned"];

/**
 * What `lens_list` answers in the browser build (`docs/21` §1 and §10): the seven lenses as `data/lenses/` (version 1) describes them,
 * with their names, lines, starting amounts and the designs they were computed from.
 */
export const MOCK_LENSES: readonly LensInfo[] = [
  ["default", "Default", "Clean and modern: even sharpness, faint glow.", 0, 0, "US 2,532,751 (J. G. Baker, Perkin-Elmer), Example 3"],
  ["classic", "Classic", "Warm and gentle: soft glow, amber flares, smooth falloff.", 35, 30, "US 2,019,985 (H. W. Lee, Kapella Ltd)"],
  ["vintage", "Vintage", "Glows wide open, lower contrast, soft edges, a little swirl.", 45, 40, "US 1,786,916 (W. Merte, Carl Zeiss), Fig. 2"],
  ["petzval", "Petzval", "Very sharp centre, strong swirl, fast falloff.", 30, 70, "US 2,500,046 (W. Schade, Eastman Kodak)"],
  ["triplet", "Triplet", "Soft corners, busy flares, a crisp centre.", 50, 35, "US 568,052 (H. D. Taylor), Series I"],
  ["sonnar", "Sonnar", "Smooth and glowing, with gentle edges.", 40, 35, "US 2,084,309 (L. Bertele, Zeiss Ikon)"],
  [
    "detuned",
    "Detuned",
    "Sharp only in the centre, with smeared flares.",
    45,
    65,
    "US 2,532,751 (J. G. Baker, Perkin-Elmer), Example 3; decentres and tilts seeded (seed 22), see README",
  ],
].map(([kind, name, line, start_flares, start_edges, design_source]) => ({
  kind: kind as LensKind,
  name: name as string,
  line: line as string,
  start_flares: start_flares as number,
  start_edges: start_edges as number,
  version: 1,
  design_source: design_source as string,
}));

/**
 * The `lens` setting as the core reads it (`docs/21` §2): a missing part keeps its default, an unknown lens reads as Default, and the
 * three amounts are clamped to 0…100.
 */
export function lensSettings(value: Partial<LensSettings> | undefined): LensSettings {
  const kind = LENS_KINDS.includes(value?.kind as LensKind) ? (value?.kind as LensKind) : "default";
  const amount = (v: number | undefined): number => clamp(Number.isFinite(v) ? (v as number) : 0, 0, 100);
  return { kind, flares: amount(value?.flares), edges: amount(value?.edges), anamorphic: amount(value?.anamorphic) };
}

/** Pulls a colour-balance point into the unit disc. */
function unitDisc([a, b]: [number, number]): [number, number] {
  const length = Math.hypot(a, b);
  return length > 1 ? [a / length, b / length] : [a, b];
}

/**
 * `contrast`, `highlights`, `shadows`, `black` or `saturation` as the core reads it (`docs/19` section 5): the `AutoOr` form, or a
 * plain number from an older document, which is Manual with that value except 0, which is Auto; a manual value is clamped to −100…100.
 */
export function percentSetting(value: PercentSetting | number | undefined): PercentSetting {
  if (value === undefined || value === 0) return { kind: "auto" };
  if (typeof value === "number") return { kind: "manual", value: clamp(value, -100, 100) };
  return value.kind === "manual" ? { kind: "manual", value: clamp(value.value, -100, 100) } : { kind: "auto" };
}

/**
 * The shadows or highlights wheel as the core reads it (`docs/19` section 5): the `AutoOr` form, or a plain point from an older
 * document, which is Manual at that point except `[0, 0]`, which is Auto; a manual point is pulled into the unit disc.
 */
export function balanceSetting(value: BalanceSetting | ColorBalancePoint | undefined): BalanceSetting {
  if (value === undefined) return { kind: "auto" };
  if (Array.isArray(value)) return value[0] === 0 && value[1] === 0 ? { kind: "auto" } : { kind: "manual", point: unitDisc(value) };
  return value.kind === "manual" ? { kind: "manual", point: unitDisc(value.point) } : { kind: "auto" };
}

/** The exposure, white balance, sharpening and noise reduction of the Input in the ranges of `docs/01` §DevelopSettings. */
function clampInput(s: InputItem): InputItem {
  const out: InputItem = {
    ...s,
    rolloff: { highlights: clamp(s.rolloff.highlights, 0, 1), low_light: clamp(s.rolloff.low_light, 0, 1) },
    // Settings from before the lens have none: they read as the default lens.
    lens: lensSettings(s.lens),
    orientation: (((Math.round(s.orientation / 90) * 90) % 360 + 360) % 360) as Orientation,
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

/** A grade in the ranges of `docs/01` §DevelopSettings; in a corrector other than Main, Auto reads as a manual 0 (`docs/23` section 3.2). */
function clampGrade(g: NodeGrade, isMain: boolean): NodeGrade {
  const pct = (v: number) => clamp(v, -100, 100);
  const read = (value: PercentSetting): PercentSetting => {
    const setting = percentSetting(value);
    return !isMain && setting.kind === "auto" ? { kind: "manual", value: 0 } : setting;
  };
  const wheel = (value: BalanceSetting): BalanceSetting => {
    const setting = balanceSetting(value);
    return !isMain && setting.kind === "auto" ? { kind: "manual", point: [0, 0] } : setting;
  };
  return {
    ...g,
    contrast: read(g.contrast),
    brightness: pct(g.brightness),
    highlights: read(g.highlights),
    shadows: read(g.shadows),
    white: pct(g.white),
    black: read(g.black),
    saturation: read(g.saturation),
    color_balance: { shadows: wheel(g.color_balance.shadows), midtones: unitDisc(g.color_balance.midtones), highlights: wheel(g.color_balance.highlights) },
    // The core's rule (`WarpPin::clamped`): the first sixteen pins, every value in its range. Settings from before the warp have none.
    color_warp: { pins: clampPins(g.color_warp?.pins ?? []) },
  };
}

/** A corrector in its ranges: the name trimmed and cut (an empty one reads as the default), the trim, the grade. */
function clampCorrector(s: CorrectorItem): CorrectorItem {
  const name = s.name.trim().slice(0, 32);
  return {
    ...s,
    name: name === "" ? (s.id === MAIN_ID ? "Main" : "Corrector") : name,
    trim: {
      exposure_ev: clamp(s.trim.exposure_ev, -4, 4),
      temperature: clamp(s.trim.temperature, -100, 100),
      tint: clamp(s.trim.tint, -100, 100),
    },
    grade: clampGrade(s.grade, s.id === MAIN_ID),
  };
}

/** A Film node in its ranges (`docs/24` section 4.1): the name trimmed and cut, every number clamped, the lights on their half-unit grid. */
function clampFilm(s: FilmItem): FilmItem {
  const name = s.name.trim().slice(0, 32);
  const half = (v: number): number => Math.round(clamp(v, -50, 50) * 2) / 2;
  return {
    ...s,
    name: name === "" ? "Film" : name,
    film: { ...s.film, exposure_ev: clamp(s.film.exposure_ev, -4, 4) },
    print: {
      ...s.print,
      // A slide has no paper, whatever the settings hold (`docs/24` section 3.9).
      exposure_ev: clamp(s.print.exposure_ev, -3, 3),
      lights: { c: half(s.print.lights.c), m: half(s.print.lights.m), y: half(s.print.lights.y) },
    },
    halation: { amount: clamp(s.halation.amount, 0, 300), size: clamp(s.halation.size, 50, 200) },
    grain: { ...s.grain, amount: clamp(s.grain.amount, 0, 200), size: clamp(s.grain.size, 50, 200) },
    strength: clamp(s.strength, 0, 100),
  };
}

/** A node of `docs/25` in its ranges: the name trimmed and cut (an empty one reads as the kind's name), a Layer's opacity 0 to 1. */
function clampMixer(s: MixerItem): MixerItem {
  const name = s.name.trim().slice(0, 32);
  const named = { ...s, name: name === "" ? MIXER_NAMES[s.kind] : name };
  return named.kind === "layer" ? { ...named, opacity: clamp(named.opacity, 0, 1) } : named;
}

/** Clamps every value to the ranges of `docs/01` §DevelopSettings, node by node, as the core does. */
export function clampSettings(s: DevelopSettings): DevelopSettings {
  const items = s.nodes.items.map((item): NodeItem => {
    if (item.kind === "input") return clampInput(item);
    if (item.kind === "output") return { ...item, look_intensity: clamp(item.look_intensity, 0, 100), output_sharpening: clamp(item.output_sharpening, 0, 200) };
    if (item.kind === "film") return clampFilm(item);
    if (item.kind === "corrector") return clampCorrector(item);
    return clampMixer(item);
  });
  return { nodes: { items, links: s.nodes.links } };
}

/** What a percent setting resolves to: the manual value, or the photo's Auto value. */
const percent = (setting: PercentSetting, auto: number): number => (setting.kind === "manual" ? setting.value : auto);
/** What a wheel resolves to: the manual point, or the photo's Auto point. */
const wheelPoint = (setting: BalanceSetting, auto: ColorBalancePoint): ColorBalancePoint =>
  setting.kind === "manual" ? setting.point : [auto[0], auto[1]];

/** A corrector's grade with every `Auto` resolved against Main's estimates (none for a corrector other than Main: it has no Auto). */
function gradeValues(grade: NodeGrade, estimates: MainEstimates | null): GradeValues {
  const est = estimates ?? { contrast: 0, saturation: 0, highlights: 0, shadows: 0, black: 0, balance_shadows: [0, 0], balance_highlights: [0, 0] };
  return {
    contrast: percent(grade.contrast, est.contrast),
    brightness: grade.brightness,
    highlights: percent(grade.highlights, est.highlights),
    shadows: percent(grade.shadows, est.shadows),
    white: grade.white,
    black: percent(grade.black, est.black),
    color_balance: {
      shadows: wheelPoint(grade.color_balance.shadows, est.balance_shadows),
      midtones: grade.color_balance.midtones,
      highlights: wheelPoint(grade.color_balance.highlights, est.balance_highlights),
    },
    saturation: percent(grade.saturation, est.saturation),
    color_warp: grade.color_warp,
    curve: grade.curve,
  };
}

function resolvedCorrector(item: CorrectorItem, estimates: MainEstimates | null): ResolvedCorrector {
  return {
    id: item.id as number,
    name: item.name,
    bypass: item.bypass,
    pos: item.pos,
    trim: item.trim,
    grade: gradeValues(item.grade, estimates),
    masks: item.masks,
  };
}

/** Resolves every `Auto` against the estimates and reports which fields are still automatic, node by node (`docs/23` section 3.9). */
export function resolveSettings(photoId: PhotoId, settings: DevelopSettings, estimates: AutoEstimates): ResolvedSettings {
  const input = inputOf(settings);
  const main = mainOf(settings);
  const output = outputOf(settings);
  const { exposure, white_balance, sharpen, noise_reduction } = input;
  // A light preset stands in for the estimated light while White balance is Auto.
  const preset = input.light_source === "auto" ? null : LIGHT_PRESETS[input.light_source];
  const values: ResolvedValues = {
    input: {
      pos: input.pos,
      exposure_ev: exposure.kind === "manual" ? exposure.ev : estimates.input.exposure_ev,
      light_source: input.light_source,
      kelvin: white_balance.kind === "manual" ? white_balance.kelvin : (preset?.kelvin ?? estimates.input.kelvin),
      tint: white_balance.kind === "manual" ? white_balance.tint : (preset?.tint ?? estimates.input.tint),
      sharpen: sharpen.kind === "manual" ? { amount: sharpen.amount, radius_px: sharpen.radius_px } : { ...estimates.input.sharpen },
      noise_reduction:
        noise_reduction.kind === "manual"
          ? { luminance: noise_reduction.luminance, color: noise_reduction.color, detail: noise_reduction.detail }
          : { ...estimates.input.noise_reduction },
      lens: input.lens,
      decode: input.decode,
      rolloff: input.rolloff,
      crop: input.crop,
      orientation: input.orientation,
      white_pick: input.white_pick,
    },
    main: resolvedCorrector(main, estimates.main),
    nodes: correctorsOf(settings)
      .filter((item) => item.id !== MAIN_ID)
      .map((item) => resolvedCorrector(item, null)),
    output: { pos: output.pos, look: output.look, look_intensity: output.look_intensity, output_sharpening: output.output_sharpening },
  };
  const grade = main.grade;
  return {
    photo_id: photoId,
    settings,
    values,
    auto: {
      input: {
        exposure: exposure.kind === "auto",
        light_source: input.light_source === "auto",
        white_balance: white_balance.kind === "auto",
        sharpen: sharpen.kind === "auto",
        noise_reduction: noise_reduction.kind === "auto",
      },
      main: {
        contrast: grade.contrast.kind === "auto",
        highlights: grade.highlights.kind === "auto",
        shadows: grade.shadows.kind === "auto",
        black: grade.black.kind === "auto",
        balance_shadows: grade.color_balance.shadows.kind === "auto",
        balance_highlights: grade.color_balance.highlights.kind === "auto",
        saturation: grade.saturation.kind === "auto",
      },
    },
    estimates,
  };
}

/**
 * What Copy settings takes from a photo (the core's `SettingsPatch::copied_from`): the whole graph, every node and every link, but the
 * Input's crop, orientation and Pick white point; Auto stays Auto and manual values stay as they are; a white balance that came from a
 * picked white is the manual Kelvin and tint it resolved to (`src/fixtures/graph.ts` has the rules).
 */
export { copiedPatch };

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
  const kelvin = clamp(Math.round((MOCK_ESTIMATES.input.kelvin * Math.exp(-warmth * 0.9)) / 50) * 50, 2000, 11000);
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

function defaultPreferences(options: Pick<Required<MockOptions>, "autoProvider" | "consent"> = DEFAULT_OPTIONS): Preferences {
  return {
    default_look: NEUTRAL,
    export_defaults: defaultExportOptions(),
    remove_location_from_exports: false,
    check_for_updates_automatically: true,
    watch_downloads_during_import: false,
    last_used_folders: { open: null, export: null, looks_import: null },
    auto_provider: options.autoProvider,
    auto_consent: { ...options.consent },
    apple_auto_instructions: "",
  };
}

/**
 * Stored preferences without `apple_auto`, which `auto_provider` replaced (`docs/22` D15), and without `display_rendering`, the
 * choice of ARRI's display LUT that Yellowmoon no longer offers: a page that kept the old file still loads, and the old choice is
 * read as nothing, without a word, as the core reads `preferences.json`.
 */
function withoutRetired(stored: Preferences): Preferences {
  const { apple_auto: retired, display_rendering: retiredDisplay, ...rest } = stored as Preferences & { apple_auto?: boolean; display_rendering?: unknown };
  void retired;
  void retiredDisplay;
  return rest;
}

/** Whether the photo's graph has a Depth row with the eye on: the mock then "makes" the photo's depth map in the background. */
const hasDepthRow = (settings: DevelopSettings): boolean =>
  settings.nodes.items.some((node) => (node.kind === "corrector" || node.kind === "film") && node.masks.items.some((row) => row.kind === "depth" && row.enabled));

/** The depth positions the mock's depth map gives the flat scenes' areas (`docs/algorithms/06` §5.8: 0 is the nearest part of the photo): the near ground, the far hill and the sky. */
const DEPTH_RANKS = MOCK_DEPTH;

/** The depth position of the point `(x, y)` of a flat scene: the near ground is near, the far hill far and the sky and the sun farthest. */
function depthRankIn(scene: Scene, x: number, y: number): number {
  const layer = layerAt(scene, x, y);
  return layer === "near" ? DEPTH_RANKS.near : layer === "far" ? DEPTH_RANKS.far : DEPTH_RANKS.sky;
}

/** Pick depth's far limit for a depth position (`docs/algorithms/06` §5.8, step 4): the depth plus 0.1, at most 1, not rounded. */
function pickDepthLimit(z: number): DepthFar {
  return { far: pickDepthFar(z) };
}

/** Every kind of mask, in the order of the Add mask menu. */
export const ALL_MASK_KINDS: readonly MaskKind[] = ["object", "subject", "people", "background", "depth", "linear", "radial", "brush", "color_range", "brightness_range"];
/** The kinds the core can make: Object (where the SAM weights are), Subject and People (where this Mac supports them), Background, the shapes, Brush and the ranges: every kind. */
export const BUILT_MASK_KINDS: readonly MaskKind[] = ALL_MASK_KINDS;

/** The kinds a `?mockMasks=` value names: `all`, `noai`, `nomac`, `nodepth`, `built`, or a comma list of kinds; null for anything else. */
export function parseMockMasks(value: string | null): readonly MaskKind[] | null {
  if (value === null) return null;
  const without = (...left: MaskKind[]): MaskKind[] => ALL_MASK_KINDS.filter((k) => !left.includes(k));
  switch (value.trim()) {
    case "all":
      return ALL_MASK_KINDS;
    case "noai":
      // Background needs no download and no Mac: the blur map comes from the photo (`docs/23` §4.6).
      return without("object", "subject", "people", "depth");
    case "nomac":
      return without("subject", "people");
    case "nodepth":
      // The depth weights are missing (`docs/23` §3.10a): Depth is not offered.
      return without("depth");
    case "built":
      return BUILT_MASK_KINDS;
    default: {
      const named = value.split(",").map((k) => k.trim());
      return named.length > 0 && named.every((k): k is MaskKind => (ALL_MASK_KINDS as readonly string[]).includes(k)) ? ALL_MASK_KINDS.filter((k) => named.includes(k)) : null;
    }
  }
}

const DEFAULT_OPTIONS: Required<MockOptions> = {
  speed: 1,
  skipped: {},
  picked: [],
  current: 0,
  photoCount: SCENES.length,
  unreadable: "",
  slowOpen: 0,
  thumbnailBudget: 0,
  folderName: DEFAULT_FOLDER_NAME,
  looksImported: true,
  allLibraries: false,
  missingLooks: [],
  lensesMissing: false,
  holdAt: Number.POSITIVE_INFINITY,
  initialSettings: {},
  photoOverrides: {},
  downloads: [],
  downloadsDenied: false,
  existingExports: [],
  exportFailures: {},
  vanish: [],
  freeBytes: 500_000_000_000,
  exportHoldAt: Number.POSITIVE_INFINITY,
  importHoldAt: Number.POSITIVE_INFINITY,
  persistPreferences: false,
  appleAuto: "none",
  providers: null,
  autoProvider: "apple",
  consent: { claude: false, agy: false },
  os: "macos",
  viewport: null,
  cpu: false,
  maskKinds: BUILT_MASK_KINDS,
  maskFind: "scene",
  depth: "ok",
  displayOptions: { reduce_transparency: false, reduce_motion: false, increase_contrast: false },
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
 * (a reason after a colon), `mockPicked=2,3,4`, `mockCurrent=3`, `mockPhotos=500` (a folder of 500 photos, the twelve repeating), `mockFolder=Name` (the folder's name), `mockUnreadable=Name` (a folder that cannot be read: opening a folder says so), `mockLooks=0` (no imported
 * looks), `mockLibraries=all` (looks in every section of the Look tab), `mockLenses=0` (no lens data: the Lens card's list is empty), `mockMissing=film-warm,film-cool`, `mockHold=5.4`, `mockDownloads=ARRI_LookLibrary_LogC4.zip` (zips that
 * land in Downloads while the wizard watches), `mockDownloadsDenied=1`, `mockExisting=DSC04204_neutral.jpg`,
 * `mockVanish=DSC04207` (the photo's file goes away once it is Ready: the next time it is shown or exported it is skipped), `mockExportFails=DSC04204:reason` (`DSC04204:disk_full` stops the export there, as a full disk does), `mockFreeBytes=200000000` (the free space of the destination folder; under 1.35 times the estimate the panel warns), `mockExportHold=1.45` (an export waits there until `flush()`), `mockImportHold=40` (a look import waits at 40 files until `flush()`) and `mockPersist=1`
 * (preferences survive a reload), `mockViewport=native` or `fallback` (the mock behaves like that viewport backend of the app),
 * `mockAppleAuto=unavailable|pending|answered|failed` (Apple Auto as on a Mac where it is not available, or where every photo is waiting
 * for its answer, has it, or failed; `MockAppleAuto`), `mockProviders=apple,claude,agy:signed_out` (the providers of the computer and their
 * states, `none` for no provider; `docs/22` section 12.8), `mockAutoProvider=claude` (the saved choice), `mockConsent=claude,agy` (the programs
 * agreed to), `mockOs=windows` and `mockCpu=1` (frames render on the CPU and take 600 ms divided by
 * `mockSpeed`, with `viewport_busy` around them: the photo cell says "Rendering on the CPU"; `MockOptions.cpu`) and `mockMasks=all|noai|nomac|built`
 * or a list of kinds (the kinds of mask `mask_kinds` says this machine can make; `MockOptions.maskKinds`), and `mockDisplay=reduce-transparency,reduce-motion,increase-contrast` (any of the three: what `display_options` answers, `MockOptions.displayOptions`).
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
  const unreadable = params.get("mockUnreadable");
  if (unreadable !== null && unreadable.trim() !== "") out.unreadable = unreadable;
  const slowOpen = params.get("mockSlowOpen");
  if (slowOpen !== null && Number.isInteger(Number(slowOpen)) && Number(slowOpen) >= 1) out.slowOpen = Number(slowOpen);
  const thumbnailBudget = params.get("mockThumbnailBudget");
  if (thumbnailBudget !== null && Number.isInteger(Number(thumbnailBudget)) && Number(thumbnailBudget) >= 1) out.thumbnailBudget = Number(thumbnailBudget);
  const count = params.get("mockPhotos");
  if (count !== null && Number.isInteger(Number(count)) && Number(count) >= 1) out.photoCount = Number(count);
  if (params.get("mockLooks") === "0") out.looksImported = false;
  if (params.get("mockLibraries") === "all") out.allLibraries = true;
  if (params.get("mockLenses") === "0") out.lensesMissing = true;
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
  const vanish = list("mockVanish");
  if (vanish) out.vanish = vanish.map((name) => name.replace(/\.arw$/i, ""));
  const freeBytes = params.get("mockFreeBytes");
  if (freeBytes !== null && Number(freeBytes) >= 0) out.freeBytes = Number(freeBytes);
  const exportHold = params.get("mockExportHold");
  if (exportHold !== null && Number(exportHold) >= 0) out.exportHoldAt = Number(exportHold);
  const importHold = params.get("mockImportHold");
  if (importHold !== null && Number(importHold) >= 0) out.importHoldAt = Number(importHold);
  if (params.get("mockPersist") === "1") out.persistPreferences = true;
  const viewportBackend = params.get("mockViewport");
  if (viewportBackend === "native" || viewportBackend === "fallback") out.viewport = viewportBackend;
  if (params.get("mockCpu") === "1") out.cpu = true;
  const appleAuto = params.get("mockAppleAuto");
  if (appleAuto !== null && (APPLE_AUTO_MODES as readonly string[]).includes(appleAuto)) out.appleAuto = appleAuto as MockAppleAuto;
  const providers = parseMockProviders(params.get("mockProviders"));
  if (providers) out.providers = providers;
  const autoProvider = params.get("mockAutoProvider");
  if (autoProvider === "yellowmoon" || autoProvider === "apple" || autoProvider === "claude" || autoProvider === "agy") out.autoProvider = autoProvider;
  const consent = parseMockConsent(params.get("mockConsent"));
  if (consent) out.consent = consent;
  const os = params.get("mockOs");
  if (os === "macos" || os === "windows" || os === "linux") out.os = os;
  const masks = parseMockMasks(params.get("mockMasks"));
  if (masks) out.maskKinds = masks;
  const maskFind = params.get("mockMaskFind");
  if (maskFind === "none" || maskFind === "scene") out.maskFind = maskFind;
  const depth = params.get("mockDepth");
  if (depth === "ok" || depth === "fail" || depth === "fail-always") out.depth = depth;
  const display = list("mockDisplay");
  if (display) {
    out.displayOptions = {
      reduce_transparency: display.includes("reduce-transparency"),
      reduce_motion: display.includes("reduce-motion"),
      increase_contrast: display.includes("increase-contrast"),
    };
  }
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
}

/** The three look packages as the mock "downloads" them: file names, sizes and counts of `docs/08`. */
export const PACKAGE_FACTS: Readonly<Record<PackageId, PackageFact>> = {
  look_library_logc4: {
    zip: "ARRI_LookLibrary_LogC4.zip",
    title: "Look Library · LogC4",
    what: "Look Library · LogC4 · log to log",
    contents: "87 .cube · 65-point",
    fileCount: 87,
    megabytes: 181,
    input: "logc4",
  },
  look_library_logc3: {
    zip: "ARRI_LookLibrary_LogC3.zip",
    title: "Look Library · LogC3",
    what: "Look Library · LogC3 · log to log",
    contents: "87 .cube · 33-point",
    fileCount: 87,
    megabytes: 30,
    input: "logc3_ei800",
  },
  look_library_logc3_to_rec709: {
    zip: "ARRI_LookLibrary_LogC3_to_Rec709.zip",
    title: "Look Library · LogC3 to Rec.709",
    what: "Look Library · LogC3 to Rec.709",
    contents: "87 .cube",
    fileCount: 87,
    megabytes: 32,
    input: "logc3_to_rec709",
  },
};

/** The words the core uses for a source that holds only ARRI's display LUTs (`ym-looks`, `explain.rs`). */
export const DISPLAY_ONLY_REASON = "This package has only display renderings. Yellowmoon uses its own rendering, so there is nothing to import.";

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

/** Zips the mock knows that are not one of the three packages: the failure cases of the wizard. */
export const FAILURE_ZIPS = {
  /** ARRI's LogC4 display LUT package (Technical Downloads page): recognised, and said to hold nothing to import. */
  displayLutsLogc4: "ARRI_DisplayLUTs_LogC4.zip",
  /** ARRI's LogC3 display LUT package. */
  displayLutsLogc3: "ARRI_DisplayLUTs_LogC3.zip",
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
  // ARRI's display LUT packages are recognised and not offered: Yellowmoon uses its own display rendering.
  for (const [name, mb] of [
    [FAILURE_ZIPS.displayLutsLogc4, 32],
    [FAILURE_ZIPS.displayLutsLogc3, 46],
  ] as const) {
    map.set(name.toLowerCase(), {
      source: "zip",
      size_bytes: mb * 1_000_000,
      recognition: { kind: "unusable", package_id: null, reason: DISPLAY_ONLY_REASON },
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

/**
 * The looks a package adds: the mockup's seven, then generated ones up to the package's count. A LogC3 package has its own ids and
 * its own library; the titles are the same as the LogC4 package's, as they are in ARRI's libraries.
 */
function packageLooks(id: PackageId): Look[] {
  const fact = PACKAGE_FACTS[id];
  const suffix = id === "look_library_logc4" ? "" : id === "look_library_logc3" ? "-logc3" : "-logc3-rec709";
  const lutSize = id === "look_library_logc4" ? 65 : 33;
  const names = [...NAMED_LOOKS];
  for (let i = 0; names.length < fact.fileCount; i++) names.push(`${LOOK_FAMILIES[i % LOOK_FAMILIES.length]} ${Math.floor(i / LOOK_FAMILIES.length) + 1}`);
  return names.map((name) => importedLook(`arri:3000:${slugify(name)}${suffix}`, name, ARRI_FILM, fact.input, "cube", lutSize));
}

/** "Warm_Film.cube" gives "Warm Film". */
const cubeName = (fileName: string): string => fileName.replace(/\.(cube|zip|alf4c?|aml)$/i, "").replace(/[_-]+/g, " ").trim();

const MISSING_SOURCE = "That file is not there any more. Download it again, or choose it again.";

/** The last part of a path. */
const fileNameOf = (path: string): string => path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? path;

// ---------------------------------------------------------------------------
// The mock
// ---------------------------------------------------------------------------

/**
 * The mock's Auto crop (`docs/20-auto-crop.md` §10): a photo whose file name has "keep" in it is level and keeps the largest frame of the
 * aspect; any other is levelled by 1.4° and cropped to 90 % of that frame, a little to the left of the centre.
 */
export function mockCropSuggestion(summary: PhotoSummary, aspect: CropAspect, orientation: Orientation): CropSuggestion {
  const turned = orientation === 90 || orientation === 270;
  const canvas = turned ? summary.height_px / summary.width_px : summary.width_px / summary.height_px;
  const landscape = canvas >= 1;
  const ratio =
    aspect === "square"
      ? 1
      : aspect === "four_by_five"
        ? landscape
          ? 5 / 4
          : 4 / 5
        : aspect === "sixteen_by_nine"
          ? landscape
            ? 16 / 9
            : 9 / 16
          : canvas;
  const keep = summary.file_name.toLowerCase().includes("keep");
  const angle = keep ? 0 : 1.4;
  const t = (angle * Math.PI) / 180;
  const [c, s] = [Math.abs(Math.cos(t)), Math.abs(Math.sin(t))];
  // The largest frame of the ratio with no empty corner, on a canvas `canvas` wide and 1 high.
  const h = Math.min(canvas / (ratio * c + s), 1 / (ratio * s + c));
  const w = (ratio * h) / canvas;
  const round = (v: number) => Math.round(v * 1e6) / 1e6;
  // Every frame is inside the straightened photo by the panel's rule (`fitInside`, the core's `fit_inside`: no empty corner, 2 px from
  // the photo's edge), as the core's frames are. The largest frame shrinks by that margin; the 90 % frame is 90 % of it.
  const canvasPx = canvasOf(summary.width_px, summary.height_px, orientation);
  const centred = (width: number, height: number, dx: number) =>
    fitInside(
      {
        x: round(Math.min(Math.max((1 - width) / 2 + dx, 0), 1 - width)),
        y: round((1 - height) / 2),
        width: round(width),
        height: round(height),
      },
      angle,
      canvasPx,
    );
  // Scores are the frames' overlap with the crop model's crop (`docs/20` §7).
  const whole = { rect: centred(w, h, 0), score: 1 };
  if (keep) return { straighten: "level", angle_deg: 0, frames: [whole], keep: true, subject: "picture" };
  return {
    straighten: "rotate",
    angle_deg: angle,
    frames: [
      { rect: centred(0.9 * whole.rect.width, 0.9 * whole.rect.height, -0.03), score: 0.93 },
      { ...whole, score: 0.78 },
    ],
    keep: false,
    subject: "picture",
  };
}

interface PhotoRecord {
  scene: Scene;
  summary: PhotoSummary;
  settings: DevelopSettings;
  /** 0 not started, 1 decoding, 2 estimating, 3 finished. */
  phase: 0 | 1 | 2 | 3;
  skipReason: string | null;
}

/** What a photo says when its file is not there any more (`MESSAGE_NOT_FOUND` of `ym_raw::error`, `docs/26` P1-2). */
export const FILE_GONE_MESSAGE =
  "Yellowmoon can't find this file any more. It may have been moved, renamed or its drive disconnected.";

/** The photos the last removal took out, with their places, for Undo. */
interface MockRemoval {
  taken: Array<{ place: number; rec: PhotoRecord }>;
  /** The current photo before the removal, and the one it handed over to. */
  wasCurrent: PhotoId | null;
  successor: PhotoId | null;
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
  /** The file at which the export stops (a failure of the destination), or null. */
  stopAt: number | null;
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

/** A look kept by an earlier version of the mock has no library, kind of file or size: it was one of the person's own `.cube` files. */
function withLibraryFields(look: Look): Look {
  const stored = look as Partial<Look> & Pick<Look, "group" | "input">;
  const format = stored.format ?? "cube";
  return { ...look, library: stored.library ?? libraryOf(look.group, look.input, format), format, lut_size: stored.lut_size ?? 33 };
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
  /** The highest photo id handed out in this session: ids are never used twice, also not by a photo that was removed. */
  let highestId = 0;
  /** The last removal, kept for `restorePhotos` until the next one or the end of the session (the core's `Removal`). */
  let removal: MockRemoval | null = null;
  let developProgress = 0;
  let developTimer: ReturnType<typeof setInterval> | null = null;
  const thumbTimers = new Map<PhotoId, ReturnType<typeof setTimeout>>();
  /** The address of each photo's newest thumbnail (the shell keeps these for `thumbnails()`). */
  const thumbnailUrls = new Map<PhotoId, string>();
  /** The photos whose thumbnail the budget let go of (`?mockThumbnailBudget`) and that have not got a new one. */
  const lostThumbnails = new Set<PhotoId>();
  /**
   * The node thumbnails the mock has told the page about, per photo and node: the address, and the key of everything the picture was
   * made from (the Input's adjustments, the path of nodes to it and the look), so that a node whose key did not change is not sent
   * again (`node_thumbnails_ready` has only the thumbnails that changed).
   */
  const nodeThumbs = new Map<PhotoId, Map<number, { key: string; url: string }>>();
  const nodeThumbTimers = new Map<PhotoId, ReturnType<typeof setTimeout>>();
  const exportRuns = new Map<ExportJobId, ExportRun>();
  let nextJobId = 1;

  let preferences = defaultPreferences();
  let looks: Look[] = [];
  let importedPackages: PackageId[] = [];
  let userLooks: Look[] = [];
    let sources = defaultSources();
  let watching = false;
  let downloadTimers: Array<ReturnType<typeof setTimeout>> = [];
  const importRuns = new Set<ImportRun>();
  const choices: { folder: Array<string | null>; files: Array<string[] | null>; open: string[][] } = { folder: [], files: [], open: [] };
  let menuState: MenuState | null = null;
  const system: MockSystemLog = { opened: [], revealed: [], trashed: [], restarts: 0 };
  let cliLink: string | null = null;
  const viewport: MockViewportState = { rect: null, mode: { kind: "fit" }, before: false, maskView: null, inputs: [], offset: { x: 0, y: 0 } };
  /** The fallback viewport's frame counter, and whether it has a picture on show. */
  let frameSeq = 0;
  let frameShown = false;
  /** Whether an `export_busy` event has said the queue is busy. */
  let busyAnnounced = false;
  /** With `cpu` or a Background row to find, the end of the frame being "rendered" (`viewport_busy` false); null while none is. */
  let cpuFrameTimer: ReturnType<typeof setTimeout> | null = null;
  /** The photos whose blur map the mock has "made" (`viewport_busy.finding` says so once for each photo). */
  const blurMaps = new Set<PhotoId>();
  /** The Subject and People masks the mock's Vision has made for each photo (`vision_state` says so once for each kind and photo), and the photos it is making them for. */
  const visionReady = new Map<PhotoId, Set<string>>();
  const visionTimers = new Map<PhotoId, ReturnType<typeof setTimeout>>();
  /** The photos whose depth map the mock has "made" (`depth_state` says so once for each photo), and the photos it is making it for. */
  const depthReady = new Set<PhotoId>();
  const depthTimers = new Map<PhotoId, ReturnType<typeof setTimeout>>();
  /** What waits for a photo's depth map (`sample_depth`). */
  const depthWaiting = new Map<PhotoId, Array<() => void>>();
  /** The sentence of the failure of each photo whose depth map failed (`?mockDepth=fail`), until `retry_depth` asks again. */
  const depthFailures = new Map<PhotoId, string>();
  /** The photos whose depth map has failed once already (`?mockDepth=fail` fails each photo once). */
  const depthFailedOnce = new Set<PhotoId>();
  const depthFailure = (id: PhotoId): string | null => depthFailures.get(id) ?? null;
  /** The photos whose Object features the mock has "made" (`object_state` says "preparing" for a moment, once for each photo). */
  const objectReady = new Set<PhotoId>();
  /** The encoder of the photos that are getting ready: when it ends, and what waits for it (`mask_find`). */
  const objectEncoders = new Map<PhotoId, { timer: ReturnType<typeof setTimeout>; waiting: Array<() => void> }>();

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
    const { orientation, crop } = inputOf(rec.settings);
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

  /**
   * Where a point of the photo frame (`docs/23` §3.6: the photo as the camera oriented it, before the turn, the straightening and the crop) is
   * drawn in the cell, as a share of the cell across and down; it may be outside the picture, and the cell. The inverse of `photoPointAt`
   * (the mock's picture is the photo turned and cut to its crop, fitted in the cell; it does not zoom or straighten).
   */
  function photoFrameInCell(rec: PhotoRecord, x: number, y: number, cell: { width: number; height: number }): [number, number] {
    const { orientation, crop } = inputOf(rec.settings);
    const crossed = crop ? crop.rect.width / crop.rect.height : 1;
    const aspect = imageAspectOf(rec.summary.width_px, rec.summary.height_px, orientation) * crossed;
    const picture = fitInCell(cell.width / cell.height, aspect);
    // The point of the turned photo: a quarter turn clockwise takes (x, y) to (1 − y, x), so undo it for the way there.
    let cx = x;
    let cy = y;
    switch (((orientation % 360) + 360) % 360) {
      case 90:
        [cx, cy] = [1 - y, x];
        break;
      case 180:
        [cx, cy] = [1 - x, 1 - y];
        break;
      case 270:
        [cx, cy] = [y, 1 - x];
        break;
      default:
    }
    const u = crop ? (cx - crop.rect.x) / crop.rect.width : cx;
    const v = crop ? (cy - crop.rect.y) / crop.rect.height : cy;
    return [picture.x + u * picture.width, picture.y + v * picture.height];
  }

  /**
   * The mask view over the picture, as the core would draw it (`docs/23` §5.8): for the corrector the page named, if the photo has it, and
   * never in the Before view. The picture's rectangle is the one the frame fits the photo into; the shapes are in the photo frame, so they go
   * through the picture's turn and crop (an affine map from the photo's own pixels, three of its points as the page's mapping gives them).
   */
  function maskLayer(rec: PhotoRecord, width: number, height: number): string {
    const view = viewport.maskView;
    if (!view || viewport.before) return "";
    const node = nodeOf(rec.settings, view.node);
    if (!node || node.kind !== "corrector") return "";
    const { orientation, crop } = inputOf(rec.settings);
    const crossed = crop ? crop.rect.width / crop.rect.height : 1;
    const aspect = imageAspectOf(rec.summary.width_px, rec.summary.height_px, orientation) * crossed;
    const fit = fitInCell(width / height, aspect);
    const at = (x: number, y: number): [number, number] => {
      const [u, v] = photoFrameInCell(rec, x, y, { width, height });
      return [u * width, v * height];
    };
    const [o, ex, ey] = [at(0, 0), at(1, 0), at(0, 1)];
    const [w, h] = [rec.summary.width_px, rec.summary.height_px];
    const frame = { width: w, height: h, matrix: [(ex[0] - o[0]) / w, (ex[1] - o[1]) / w, (ey[0] - o[0]) / h, (ey[1] - o[1]) / h, o[0], o[1]] as [number, number, number, number, number, number] };
    // Vision's masks are in the picture once the background worker has made them, for the kinds this machine can make (the rows select nothing before).
    const scene = visionFoundIn(rec.scene.name);
    const ready = visionReady.get(rec.summary.id);
    const found = {
      subject: scene.subject && options.maskKinds.includes("subject") && (ready?.has("subject") ?? false),
      people: scene.people && options.maskKinds.includes("people") && (ready?.has("people") ?? false),
    };
    const depth = { on: view.depth === true, ready: depthReady.has(rec.summary.id) && depthFailure(rec.summary.id) === null };
    return maskViewSvg(view.style, masksOfView(node.masks, view.row), { x: fit.x * width, y: fit.y * height, width: fit.width * width, height: fit.height * height }, frame, found, depth);
  }

  /** The picture of the cell the fallback viewport would send: the developed scene letterboxed on paper, as many pixels as the cell has. */
  function frameImage(rec: PhotoRecord): { url: string; width: number; height: number } {
    const rect = cellRect();
    const width = Math.round(rect.width * rect.scale);
    const height = Math.round(rect.height * rect.scale);
    const { values } = resolveSettings(rec.summary.id, rec.settings, estimates());
    const picture = previewSvg(developed(rec), {
      orientation: values.input.orientation,
      angle_deg: values.input.crop?.angle_deg ?? 0,
      crop: values.input.crop?.rect ?? null,
    }).replace(/^<svg ([^>]*?) width="[^"]*" height="[^"]*"/, '<svg $1 width="100%" height="100%"');
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="100%" fill="#FFFFFF"/>${picture}${maskLayer(rec, width, height)}</svg>`;
    return { url: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`, width, height };
  }

  /**
   * The core starts SAM's encoder in the background as soon as an Object row exists on the photo being looked at (`docs/23` §4.4): `object_state` says
   * `preparing` now and not any more `ENCODER_MS / speed` later, once for each photo (a click that comes meanwhile waits for it).
   */
  function prepareObject(rec: PhotoRecord): void {
    const id = rec.summary.id;
    if (!options.maskKinds.includes("object") || !hasObjectRow(rec.settings) || objectReady.has(id) || objectEncoders.has(id)) return;
    emitSoon("object_state", { photo_id: id, preparing: true });
    const entry = {
      waiting: [] as Array<() => void>,
      timer: setTimeout(() => {
        objectEncoders.delete(id);
        objectReady.add(id);
        emit("object_state", { photo_id: id, preparing: false });
        for (const wake of entry.waiting.splice(0)) wake();
      }, ENCODER_MS / options.speed),
    };
    objectEncoders.set(id, entry);
  }

  /**
   * The core makes the Subject and People masks of the photo being looked at in the background as soon as such a row exists (`docs/23` §4.5): the viewport
   * is busy and `finding` for `VISION_MS / speed` (a viewer never waits: the rows select nothing meanwhile), then `vision_state` says which kinds found
   * nothing and the photo is drawn again with the masks. Once for each kind and photo; nothing where `mask_kinds` leaves the kind out (a Mac that cannot).
   */
  function prepareVision(rec: PhotoRecord): void {
    const id = rec.summary.id;
    const wanted = visionRowKinds(rec.settings).filter((k) => options.maskKinds.includes(k));
    const ready = visionReady.get(id) ?? new Set<string>();
    const missing = wanted.filter((k) => !ready.has(k));
    if (missing.length === 0 || visionTimers.has(id)) return;
    emitSoon("viewport_busy", { busy: true, cpu: options.cpu, finding: true });
    visionTimers.set(
      id,
      setTimeout(() => {
        visionTimers.delete(id);
        const made = visionReady.get(id) ?? new Set<string>();
        for (const kind of missing) made.add(kind);
        visionReady.set(id, made);
        emit("viewport_busy", { busy: false, cpu: options.cpu, finding: false });
        const found = visionFoundIn(rec.scene.name);
        emit("vision_state", { photo_id: id, nothing_found: visionRowKinds(rec.settings).filter((k) => made.has(k) && !found[k]) });
        sendFrame();
      }, VISION_MS / options.speed),
    );
  }

  /**
   * The core makes the depth map of the photo being looked at in the background as soon as a Depth row exists (`docs/23` §4.8): the viewport is busy and
   * `finding` for `DEPTH_MS / speed` (a viewer never waits: the row selects nothing meanwhile), then `depth_state` says the map is ready (`failed: null`) and
   * the photo is drawn again. Once for each photo; nothing where `mask_kinds` leaves Depth out (the weights are missing).
   */
  function prepareDepth(rec: PhotoRecord): void {
    const id = rec.summary.id;
    if (!options.maskKinds.includes("depth") || !hasDepthRow(rec.settings) || depthReady.has(id) || depthTimers.has(id) || depthFailures.has(id)) return;
    emitSoon("viewport_busy", { busy: true, cpu: options.cpu, finding: true });
    depthTimers.set(
      id,
      setTimeout(() => {
        depthTimers.delete(id);
        const fails = options.depth === "fail-always" || (options.depth === "fail" && !depthFailedOnce.has(id));
        if (fails) {
          depthFailedOnce.add(id);
          depthFailures.set(id, DEPTH_FAILED);
        } else depthReady.add(id);
        for (const wake of depthWaiting.get(id)?.splice(0) ?? []) wake();
        emit("viewport_busy", { busy: false, cpu: options.cpu, finding: false });
        emit("depth_state", { photo_id: id, failed: fails ? DEPTH_FAILED : null });
        sendFrame();
      }, DEPTH_MS / options.speed),
    );
  }

  /** Resolves when the photo's depth map is made (at once when it is not being made). */
  function depthDone(id: PhotoId): Promise<void> {
    if (!depthTimers.has(id)) return Promise.resolve();
    return new Promise((resolve) => depthWaiting.set(id, [...(depthWaiting.get(id) ?? []), resolve]));
  }

  /** Resolves when the photo's encoder is done (at once when it is not running). */
  function encoderDone(id: PhotoId): Promise<void> {
    const entry = objectEncoders.get(id);
    return entry ? new Promise((resolve) => entry.waiting.push(resolve)) : Promise.resolve();
  }

  /**
   * With `cpu`, a frame of the photo on screen is "rendered" on the CPU, as the core says around a round of its viewport: `viewport_busy`
   * true now (unless a frame is still under way, which this one follows) and false `CPU_FRAME_MS / speed` later. The first frame of a photo
   * with a Background row (eye on) is the one that makes its blur map: `finding` is true for `FINDING_MS / speed`, once for each photo, with or
   * without `cpu`. Nothing without a developed photo on screen, which the core does not render either.
   */
  function renderOnCpu(): void {
    const rec = photos[current];
    if (!rec || rec.phase < 3 || rec.skipReason) return;
    prepareObject(rec);
    prepareVision(rec);
    prepareDepth(rec);
    const id = rec.summary.id;
    const finding = !blurMaps.has(id) && hasBackgroundRow(rec.settings);
    if (finding) blurMaps.add(id);
    if (!options.cpu && !finding) return;
    const cpu = options.cpu;
    if (cpuFrameTimer === null) emitSoon("viewport_busy", { busy: true, cpu, finding });
    else {
      clearTimeout(cpuFrameTimer);
      if (finding) emitSoon("viewport_busy", { busy: true, cpu, finding });
    }
    const ms = Math.max(finding ? FINDING_MS : 0, cpu ? CPU_FRAME_MS : 0);
    cpuFrameTimer = setTimeout(() => {
      cpuFrameTimer = null;
      // The core's Subject and People worker goes on after the round that asked for the masks: the viewport stays finding until it ends.
      const masks = visionTimers.size > 0 || depthTimers.size > 0;
      emit("viewport_busy", { busy: masks, cpu, finding: masks });
    }, ms / options.speed);
  }

  /**
   * Tells the page what the cell shows now, as the shell does after every change: with `viewport: "fallback"`, a frame of the current
   * photo, or no picture while it is not ready or the session is closed; with `cpu`, the frame is first "rendered" on the CPU.
   */
  function sendFrame(): void {
    renderOnCpu();
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
    const list: Look[] = BUILT_IN_LOOKS.map((l) => builtInLook(l.id, l.name));
    const seen = new Set(list.map((l) => l.id));
    const add = (more: Look[]): void => {
      for (const look of more) {
        if (seen.has(look.id)) continue;
        seen.add(look.id);
        list.push(look);
      }
    };
    if (options.looksImported) add(importedLooks());
    if (options.allLibraries) add(sectionLooks().map((l) => ({ ...l, missing: options.missingLooks.includes(lookSlug(l.id)) })));
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
    return IMPORTED_LOOKS.map((l) => ({ ...importedLook(l.id, l.name, ARRI_FILM, "logc4", "cube", 65), missing: options.missingLooks.includes(lookSlug(l.id)) }));
  }

  /** Writes what a real install would keep: the preferences and the imported looks. */
  function persist(): void {
    if (!options.persistPreferences) return;
    writeStored({ preferences, importedPackages, userLooks, cliLink });
  }

  // --- Apple Auto ---------------------------------------------------------------------------

  /** The providers this computer has: the option, or what `appleAuto` says about Apple Intelligence. */
  function providers(): MockProviders {
    return options.providers ?? providersOfMode(options.appleAuto);
  }

  /** Who grades now (`Activity::decide`): the saved choice, when it is there, available and agreed to. */
  function activity(): MockActivity {
    return mockActivity(providers(), preferences.auto_provider, preferences.auto_consent);
  }

  /** Where every photo stands: the same for all of them in the mock. */
  function appleState(): AppleAuto {
    return mockAppleState(options.appleAuto, activity(), preferences.apple_auto_instructions);
  }

  /** What Auto resolves to now: the same for every photo, with the answer of the provider that grades while it has one. */
  function estimates(): AutoEstimates {
    const activityNow = activity();
    // The provider the photo's state belongs to: the one that acts, or the chosen one it was asked about; Yellowmoon's own while off.
    const provider = activityNow.kind === "active" ? activityNow.provider : activityNow.kind === "unavailable" ? preferences.auto_provider : "yellowmoon";
    return mockEstimates(appleState(), provider);
  }

  /**
   * Whether the photo waits for its provider's answer (the summary's `apple_pending`): the mock's Auto is one state for every photo,
   * so a photo waits exactly while that state is `pending` (a file that can't be opened never does).
   */
  function applePending(rec: PhotoRecord): boolean {
    return !rec.skipReason && appleState().state === "pending";
  }

  /** The core's `auto_status`. */
  function autoStatus(): AutoStatus {
    return mockAutoStatus({
      providers: providers(),
      preference: preferences.auto_provider,
      consent: preferences.auto_consent,
      own: preferences.apple_auto_instructions,
    });
  }

  /**
   * Runs `change` and tells about every developed photo whose estimates moved, as the shell does for the core's `estimates_changed`
   * (`docs/19` section 6): `settings_changed`, a new thumbnail and, for the photo on screen, a new frame.
   */
  function changingEstimates(change: () => void): void {
    const before = JSON.stringify(estimates());
    const waited = photos.map(applePending);
    change();
    // The `apple` stage of the photos that started or stopped waiting for Apple Intelligence, as the core sends it.
    photos.forEach((rec, i) => {
      const waits = applePending(rec);
      if (waits !== waited[i]) progress(rec, "apple", !waits);
    });
    if (JSON.stringify(estimates()) === before) return;
    for (const rec of photos) {
      if (rec.phase < 3 || rec.skipReason) continue;
      emitSoon("settings_changed", { photo_id: rec.summary.id });
      scheduleThumbnail(rec);
    }
    sendFrame();
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
      photos: photos.map((p) => ({ ...p.summary, apple_pending: applePending(p) })),
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

  /**
   * A photo of the fixture folder: the scene for its place in the folder, under `name` when the file has one of its own. Past the
   * twelve fixtures the scenes repeat under new names, so every file name is its own.
   */
  function makePhoto(index: number, id: PhotoId, name?: string, picked = false): PhotoRecord {
    const fixture = SCENES[index % SCENES.length] as Scene;
    const scene: Scene =
      name !== undefined ? { ...fixture, name } : index < SCENES.length ? fixture : { ...fixture, name: `DSC${String(4201 + index).padStart(5, "0")}` };
    const summary: PhotoSummary = {
      id,
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
      picked,
      reduced: false,
      apple_pending: false,
    };
    const initial: SettingsPatch = options.initialSettings[`${scene.name}.ARW`] ?? options.initialSettings[scene.name] ?? {};
    const override = options.photoOverrides[`${scene.name}.ARW`] ?? options.photoOverrides[scene.name];
    if (override) Object.assign(summary, { ...override, camera: { ...summary.camera, ...override.camera } });
    return {
      scene,
      summary,
      settings: clampSettings(applyPatch(defaultSettings(preferences.default_look), initial)),
      phase: 0,
      skipReason: skipReasonFor(scene),
    };
  }

  function openFixtureSession(): void {
    const pickedSet = new Set(options.picked);
    photos = Array.from({ length: Math.max(1, Math.floor(options.photoCount)) }, (_, index) => makePhoto(index, index + 1, undefined, pickedSet.has(index)));
    current = clamp(options.current, 0, photos.length - 1);
    highestId = photos.length;
    removal = null;
    developProgress = 0;
    startDevelopment();
  }

  /** The file names of the folder the session came from, as the folder itself would list them. */
  function folderFileNames(): string[] {
    return Array.from({ length: Math.max(1, Math.floor(options.photoCount)) }, (_, index) =>
      index < SCENES.length ? (SCENES[index] as Scene).name : `DSC${String(4201 + index).padStart(5, "0")}`,
    );
  }

  /**
   * Adds to a session that is open (the core's `Session::open`): a file the session has is counted as already open, a new one becomes
   * a photo after the others, new files in name order, and only those develop. The current photo and the photos before stay as they are.
   * A folder is the fixture folder.
   */
  function addPhotos(paths: string[]): { added: number; already_open: number } {
    const known = new Set(photos.map((p) => p.scene.name.toLowerCase()));
    const fresh: string[] = [];
    let alreadyOpen = 0;
    const want = (name: string): void => {
      if (known.has(name.toLowerCase())) alreadyOpen += 1;
      else {
        known.add(name.toLowerCase());
        fresh.push(name);
      }
    };
    for (const path of paths) {
      if (isArwPath(path)) want(fileNameOf(path).replace(/\.arw$/i, ""));
      else if (isFolderPath(path)) folderFileNames().forEach(want);
    }
    fresh.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
    const before = photos.length;
    for (const name of fresh) photos.push(makePhoto(photos.length, ++highestId, name));
    if (fresh.length > 0) {
      // Only the new photos are left to develop; the others keep what they have.
      developProgress = Math.min(developProgress, before);
      startDevelopment();
    }
    return { added: fresh.length, already_open: alreadyOpen };
  }

  /**
   * The photo's file is read again (it is shown, or an export reaches it): a photo named by `?mockVanish` that is Ready has lost its file,
   * so it becomes skipped, with an event, as the core does (`Session::raw_image`, `docs/26` P1-4). True when it did.
   */
  function vanishIfGone(rec: PhotoRecord): boolean {
    if (rec.phase < 3 || rec.skipReason || !options.vanish.includes(rec.scene.name)) return false;
    rec.skipReason = FILE_GONE_MESSAGE;
    setStatus(rec, { kind: "skipped", reason: FILE_GONE_MESSAGE });
    progress(rec, "decode", true, FILE_GONE_MESSAGE);
    return true;
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
      const shown = photos[current] === rec;
      if (shown && vanishIfGone(rec)) {
        sendFrame();
        return;
      }
      if (shown) sendFrame();
    }
  }

  /** How a photo differs from Automatic (its exposure and white balance), with or without its colour warp. */
  function adjustmentOf(rec: PhotoRecord, warp: boolean): Adjustment {
    // Against the rule's values, so Apple Auto's answer shows in the picture as the change it is.
    const { values } = resolveSettings(rec.summary.id, rec.settings, estimates());
    return {
      ev: values.input.exposure_ev,
      autoEv: MOCK_ESTIMATES.input.exposure_ev,
      kelvin: values.input.kelvin,
      autoKelvin: MOCK_ESTIMATES.input.kelvin,
      tint: values.input.tint,
      autoTint: MOCK_ESTIMATES.input.tint,
      warp: warp ? values.main.grade.color_warp.pins : undefined,
    };
  }

  /** The photo as it is developed now: adjusted by what differs from Automatic, moved by its colour warp, then through its look. */
  function developed(rec: PhotoRecord, lookId: LookId = outputOf(rec.settings).look, intensity = outputOf(rec.settings).look_intensity): Scene {
    return developScene(throughFilms(rec), { slug: lookSlug(lookId), intensity }, adjustmentOf(rec, true));
  }

  /** What a Film node does in the mock: its stock's flat-shape grading at its Strength. */
  function filmLook(item: FilmItem): NodeLook {
    return {
      bypass: item.bypass,
      trim: { exposure_ev: 0, temperature: 0, tint: 0 },
      grade: neutralGrade(),
      apply: (scene) => filmScene(scene, item.film.stock, item.strength),
      key: [item.film.stock, item.strength],
    };
  }

  /** The photo's scene through every live Film node (the mock draws their stocks flat). */
  function throughFilms(rec: PhotoRecord, settings: DevelopSettings = rec.settings): Scene {
    // A graph with a mixer the Output needs is drawn by walking it (`fixtures/mixers.ts`): Main's own grade is the frame's adjustment, the other
    // correctors and the Film nodes grade the scene where they are.
    if (hasMixers(settings)) {
      return sceneAt(settings, rec.scene, outputFeeder(settings), (item, scene) =>
        item.kind === "film" ? adjustForNode(scene, filmLook(item)) : item.id === MAIN_ID ? scene : adjustForNode(scene, item),
      );
    }
    return sceneThroughFilms(rec.scene, settings);
  }

  /** The photo's colours where the colour warp takes them: adjusted, not yet warped, not yet through the look (`docs/02` C9b). */
  function warpInput(rec: PhotoRecord): Scene {
    return adjustScene(rec.scene, adjustmentOf(rec, false));
  }

  /**
   * A photo's own thumbnail: its look and its adjustments (the filmstrip shows it). The picture is announced by its address, as the
   * shell does (there, `ym-frame://…`; here a `blob:` URL), and the address is kept for `thumbnails()`.
   */
  function emitThumbnail(rec: PhotoRecord): void {
    const id = rec.summary.id;
    const url = bytesToUrl(Array.from(scenePng(developed(rec))));
    thumbnailUrls.delete(id);
    thumbnailUrls.set(id, url);
    lostThumbnails.delete(id);
    // `?mockThumbnailBudget`: the photos used least recently (the map is in order of use) lose theirs, this one never.
    while (options.thumbnailBudget > 0 && thumbnailUrls.size > options.thumbnailBudget) {
      const oldest = thumbnailUrls.keys().next().value;
      if (oldest === undefined || oldest === id) break;
      thumbnailUrls.delete(oldest);
      lostThumbnails.add(oldest);
    }
    emit("thumbnail_ready", { id, url });
  }

  /** `thumbnailsInView`: the addresses held for the photos in view, which count as used; the ones the budget let go are made again. */
  function thumbnailsInView(ids: PhotoId[]): ThumbnailReady[] {
    const held: ThumbnailReady[] = [];
    for (const id of ids) {
      const url = thumbnailUrls.get(id);
      if (url !== undefined) {
        thumbnailUrls.delete(id);
        thumbnailUrls.set(id, url);
        held.push({ id, url });
      } else if (lostThumbnails.has(id)) {
        const rec = photos.find((r) => r.summary.id === id);
        if (rec) scheduleThumbnail(rec);
      }
    }
    return held;
  }

  /** The correctors and Film nodes on the picture path from the Input to a node, in order (a node with no picture link reads the Input). */
  function pictureChain(settings: DevelopSettings, node: number): NodeLook[] {
    const chain: NodeLook[] = [];
    const seen = new Set<number>();
    let at: NodeId | undefined = node;
    while (typeof at === "number" && !seen.has(at)) {
      seen.add(at);
      const item = nodeOf(settings, at);
      if (!item || (item.kind !== "corrector" && item.kind !== "film")) break;
      chain.unshift(item.kind === "film" ? filmLook(item) : item);
      at = linkInto(settings.nodes.links, { node: at, port: "picture" })?.from.node;
    }
    return chain;
  }

  /**
   * The corrector a panel signal is read at (`at`, `docs/01`): null for Main (left out, or 1) and for the Input, which the callers read
   * as they always did; a corrector other than Main answers its id. A node the photo does not have and the Output are refused in the
   * core's words.
   */
  function signalNode(rec: PhotoRecord, at: NodeId | undefined): number | null {
    if (at === undefined || at === "input" || at === MAIN_ID) return null;
    if (at === "output") fail("render", "The output has no picture to read here.");
    const kind = nodeOf(rec.settings, at)?.kind;
    if (kind !== "corrector" && kind !== "parallel" && kind !== "layer" && kind !== "splitter" && kind !== "combiner") fail("render", `This photo has no node ${at}.`);
    return at;
  }

  /**
   * The scene a panel reads at a numbered node `node` (never Main): at a corrector as `signalScene` says; at a mixer, a Splitter or a
   * Combiner the picture after the node, whatever the tap (for a Splitter its input picture, `docs/25` §5.7).
   */
  function signalSceneAt(rec: PhotoRecord, node: number, stage: "curve" | "warp" | "input"): Scene {
    const adjustment = adjustmentOf(rec, false);
    if (nodeOf(rec.settings, node)?.kind === "corrector") return signalScene(rec.scene, adjustment, pictureChain(rec.settings, node), stage);
    const input = adjustScene(rec.scene, { ...adjustment, warp: undefined });
    const scene = sceneAfter(rec.settings, input, node, (item, scene) => (item.kind === "film" ? adjustForNode(scene, filmLook(item)) : adjustForNode(scene, item)));
    return scene ?? fail("render", `Node ${node} has no picture to read here.`);
  }

  /**
   * The thumbnails of every node of a photo: the flat scene graded through the nodes up to it and then the Output's look (the core
   * stops the plan at the node and runs the Output). Sends `node_thumbnails_ready` with the nodes whose picture changed since the page
   * was told, and nothing when none did.
   */
  function makeNodeThumbnails(rec: PhotoRecord): void {
    if (!photos.includes(rec) || rec.phase < 3 || rec.skipReason) return;
    const id = rec.summary.id;
    const { settings } = rec;
    const input = adjustmentOf(rec, false);
    const output = outputOf(settings);
    const look = { slug: lookSlug(output.look), intensity: output.look_intensity };
    const told = nodeThumbs.get(id) ?? new Map<number, { key: string; url: string }>();
    const changed: NodeThumbnail[] = [];
    const nodes = cardStates(settings);
    const mixed = settings.nodes.items.some(isGlyphItem);
    for (const node of nodes.keys()) {
      // A Parallel, a Layer, a Splitter and a Combiner have no thumbnail (`docs/25` §5.7); the nodes after them draw through them.
      if (isGlyphItem(nodeOf(settings, node))) continue;
      const chain = pictureChain(settings, node);
      const key = mixed
        ? JSON.stringify([input, look, node, settings.nodes.items.map((item) => ({ ...item, pos: null })), settings.nodes.links])
        : JSON.stringify([input, look, chain.map((n) => [n.bypass, n.trim, n.grade, n.key])]);
      if (told.get(node)?.key === key) continue;
      const scene = mixed
        ? developScene(
            sceneAt(settings, adjustScene(rec.scene, { ...input, warp: undefined }), { node, port: "picture" }, (item, s) =>
              item.kind === "film" ? adjustForNode(s, filmLook(item)) : adjustForNode(s, item),
            ),
            look,
            null,
          )
        : nodeScene(rec.scene, input, chain, look);
      const url = bytesToUrl(Array.from(scenePng(scene, ...NODE_THUMB_SIZE)));
      told.set(node, { key, url });
      changed.push({ node, url });
    }
    // A node that was taken out has no picture any more.
    for (const node of [...told.keys()]) if (!nodes.has(node)) told.delete(node);
    nodeThumbs.set(id, told);
    if (changed.length > 0) emit("node_thumbnails_ready", { photo_id: id, thumbnails: changed });
  }

  /** Forgets what the page was told about the node thumbnails of these photos (they left the filmstrip, or the session closed). */
  function forgetNodeThumbnails(ids: Iterable<PhotoId>): void {
    for (const id of ids) {
      const timer = nodeThumbTimers.get(id);
      if (timer !== undefined) clearTimeout(timer);
      nodeThumbTimers.delete(id);
      nodeThumbs.delete(id);
    }
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

  /**
   * Development goes through the photos in order, one `developProgress` step per photo. After photos left or came back, the progress is
   * set to where the first photo that is not finished stands, so that every photo goes on from the phase it has.
   */
  function resumeDevelopment(): void {
    const first = photos.findIndex((p) => p.phase < 3);
    if (first < 0) {
      developProgress = photos.length;
      stopDevelopment();
      return;
    }
    const rec = photos[first] as PhotoRecord;
    developProgress = first + (rec.phase >= 2 ? DECODE_SHARE : 0);
    startDevelopment();
  }

  /** The current photo after `doomed` leave: itself, else the next photo that stays, else the previous one (the core's rule). */
  function successorAfter(doomed: ReadonlySet<PhotoId>): PhotoId | null {
    const here = photos[current];
    if (!here) return null;
    if (!doomed.has(here.summary.id)) return here.summary.id;
    const stays = (rec: PhotoRecord): boolean => !doomed.has(rec.summary.id);
    return (photos.slice(current + 1).find(stays) ?? photos.slice(0, current).reverse().find(stays))?.summary.id ?? null;
  }

  /** The viewport shows the photo now current from the start: Fit, the camera view off. */
  function showCurrent(): void {
    viewport.mode = { kind: "fit" };
    viewport.offset = { x: 0, y: 0 };
    viewport.before = false;
    viewport.maskView = null;
    emitSoon("render_stats", RENDER_STATS);
    sendFrame();
  }

  /** `remove_photos`: the photos leave the filmstrip, their settings go with them, and the last removal is kept for Undo. */
  function removeNow(ids: readonly PhotoId[]): SessionChange {
    const doomed = new Set(ids);
    const taken = photos.map((rec, place) => ({ rec, place })).filter(({ rec }) => doomed.has(rec.summary.id));
    if (taken.length === 0) return { photo_ids: [], already_open: 0, session: snapshot() };
    const left = new Set(taken.map(({ rec }) => rec.summary.id));
    const wasCurrent = photos[current]?.summary.id ?? null;
    const successor = successorAfter(left);
    removal = { taken, wasCurrent, successor };
    photos = photos.filter((rec) => !left.has(rec.summary.id));
    for (const id of left) {
      const timer = thumbTimers.get(id);
      if (timer !== undefined) clearTimeout(timer);
      thumbTimers.delete(id);
      thumbnailUrls.delete(id);
      lostThumbnails.delete(id);
    }
    forgetNodeThumbnails(left);
    current = Math.max(0, photos.findIndex((rec) => rec.summary.id === successor));
    if (photos.length === 0) {
      stopDevelopment();
      developProgress = 0;
      showCurrent();
    } else {
      resumeDevelopment();
      if (successor !== wasCurrent) showCurrent();
    }
    return { photo_ids: taken.map(({ rec }) => rec.summary.id), already_open: 0, session: snapshot() };
  }

  /** `restore_photos`: the photos of the last removal come back where they were, the one on screen on screen again. */
  function restoreNow(): SessionChange {
    const last = removal;
    removal = null;
    if (!last) return { photo_ids: [], already_open: 0, session: snapshot() };
    const wasEmpty = photos.length === 0;
    const here = photos[current]?.summary.id ?? null;
    const known = new Set(photos.map((p) => p.scene.name.toLowerCase()));
    const restored: PhotoRecord[] = [];
    let alreadyOpen = 0;
    for (const { place, rec } of last.taken) {
      // The same file was added again meanwhile: the photo in the session is the one.
      if (known.has(rec.scene.name.toLowerCase())) {
        alreadyOpen += 1;
        continue;
      }
      known.add(rec.scene.name.toLowerCase());
      // Its development was cut short: it starts again.
      if (rec.phase < 3 && !rec.skipReason) {
        rec.phase = 0;
        setStatus(rec, { kind: "waiting" });
      }
      photos.splice(Math.min(place, photos.length), 0, rec);
      restored.push(rec);
    }
    const ids = restored.map((rec) => rec.summary.id);
    // The photo on screen stays, unless the removal had taken it away and the person has not moved since.
    current = Math.max(0, photos.findIndex((rec) => rec.summary.id === here));
    const back = last.wasCurrent !== null && ids.includes(last.wasCurrent) && here === last.successor ? last.wasCurrent : null;
    if (back !== null) current = photos.findIndex((rec) => rec.summary.id === back);
    if (photos.length > 0 && restored.length > 0) {
      resumeDevelopment();
      for (const rec of restored) if (rec.phase === 3 && !rec.skipReason) emitThumbnail(rec);
      if (wasEmpty || photos[current]?.summary.id !== here) showCurrent();
    }
    return { photo_ids: ids, already_open: alreadyOpen, session: snapshot() };
  }

  function isArwPath(path: string): boolean {
    return /\.arw$/i.test(path);
  }

  /** A path whose last segment has no dot is taken for a folder. */
  function isFolderPath(path: string): boolean {
    const last = path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
    return last.length > 0 && !last.includes(".");
  }

  /** What reading the files of an open came to: how many were read, and whether `cancelOpen` stopped it. */
  interface OpenReading {
    done: number;
    cancelled: boolean;
  }

  /** The open that is reading files now (`?mockSlowOpen`): `stop` ends it at once. */
  let openRun: { timer: ReturnType<typeof setInterval>; stop: (cancelled: boolean) => void } | null = null;

  /** `?mockSlowOpen`: reads the files 50 at a time and reports `open_progress`, until the end or `cancelOpen`. Instant without the option. */
  function readSlowly(paths: string[]): Promise<OpenReading> {
    const total = options.slowOpen;
    const reads = total > 0 && !options.unreadable && paths.some((p) => isArwPath(p) || isFolderPath(p));
    if (!reads) return Promise.resolve({ done: 0, cancelled: false });
    return new Promise((resolve) => {
      let done = 0;
      emit("open_progress", { done, total });
      const stop = (cancelled: boolean): void => {
        clearInterval(timer);
        openRun = null;
        resolve({ done, cancelled });
      };
      const timer = setInterval(() => {
        done = Math.min(total, done + 50);
        emit("open_progress", { done, total });
        if (done >= total) stop(false);
      }, 100 / options.speed);
      openRun = { timer, stop };
    });
  }

  /** The answer of an open once its files are read: all of them, or the ones read before `cancelOpen` (none when it came first). */
  function openAfterReading(paths: string[], read: OpenReading): SessionState {
    if (!read.cancelled) return openPathsNow(paths);
    if (read.done === 0) return { ...snapshot(), opened: { added: 0, already_open: 0, cancelled: true } };
    const state = openPathsNow(paths);
    return { ...state, opened: { added: 0, already_open: 0, ...state.opened, cancelled: true } };
  }

  function openPathsNow(paths: string[]): SessionState {
    // The core's sentence for a folder it could not list, when no file came with it (`docs/26` H-02).
    if (options.unreadable && !paths.some(isArwPath) && paths.some(isFolderPath)) {
      fail("open", `${UNREADABLE_PREFIX}${options.unreadable}. Check that it is still there and that you may open it, then try again.`);
    }
    const hasPhotos = paths.some((p) => isArwPath(p) || isFolderPath(p));
    // Non-ARW files are ignored. Any ARW file or folder opens the whole fixture folder when nothing is open; in an open session
    // the files are added (or already there).
    let opened = { added: 0, already_open: 0 };
    if (hasPhotos && photos.length === 0) {
      openFixtureSession();
      opened = { added: photos.length, already_open: 0 };
    } else if (hasPhotos) {
      opened = addPhotos(paths);
    }
    return { ...snapshot(), opened };
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

  /**
   * The photo's settings with a node patch applied, as the core applies it (`docs/23` section 3.7): the patch is checked first and the
   * document it makes after, a refusal says why in the core's words and changes nothing, and what is in range is clamped. A pick sets the
   * white balance by hand from the colour under the point, as the mock has always done.
   */
  function patched(rec: PhotoRecord, patch: SettingsPatch): DevelopSettings {
    const refusal = refusalOfPatch(rec.settings, patch);
    if (refusal) fail("render", refusal);
    let next = applyPatch(rec.settings, patch);
    const pick = patch.nodes?.[INPUT_ID]?.white_pick;
    if (pick) next = applyPatch(next, inputPatch({ white_balance: { kind: "manual", ...whiteFromPick(rec.scene, pick) } }));
    const settings = clampSettings(next);
    const invalid = validateGraph(settings);
    if (invalid) fail("render", invalid);
    return settings;
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

  /** The core's sentence for a failure of the destination (`DestinationProblem::app_message`, `docs/26` P2-1). */
  function destinationSentence(kind: string, folder: string): string | null {
    switch (kind) {
      case "disk_full":
        return "The disk is full, so the export stopped. Free some space, then export the photos that are left.";
      case "read_only":
        return `${folder} is read-only. Choose another folder.`;
      case "gone":
        return `Yellowmoon can't reach ${folder}. If it is on a drive, check that the drive is connected, then export again.`;
      case "permission":
        return `Yellowmoon isn't allowed to save to ${folder}. Choose another folder, or allow access in System Settings, under Privacy & Security.`;
      default:
        return null;
    }
  }

  function exportFiles(job: ExportJob): { files: string[]; skips: boolean[]; errors: Array<string | null>; stopAt: number | null } {
    const recs = targets(job.scope);
    const ext = FILE_EXTENSION[job.options.format];
    const existing = new Set(options.existingExports);
    const taken = new Set<string>(existing);
    const files: string[] = [];
    const skips: boolean[] = [];
    const errors: Array<string | null> = [];
    let stopAt: number | null = null;
    const folder = (job.options.destination ?? `${folderPath()} export`).split("/").filter(Boolean).pop() ?? "The folder";
    recs.forEach((rec, i) => {
      let name = `${sanitizeStem(
        expandNameTemplate(job.options.name_template, {
          name: rec.scene.name,
          look: lookSlug(outputOf(rec.settings).look),
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
      // A file that is gone when its turn comes fails that photo only, and the export goes on.
      const failure = options.exportFailures[rec.scene.name] ?? (vanishIfGone(rec) ? FILE_GONE_MESSAGE : null);
      const sentence = failure === null ? null : destinationSentence(failure, folder);
      if (sentence !== null && stopAt === null) stopAt = i;
      errors.push(sentence ?? failure);
    });
    return { files, skips, errors, stopAt };
  }

  /** A DNG of the job is close to, not the same as, the picture: a photo has a look with its own display rendering (at an intensity above 0). */
  function dngApproximate(job: ExportJob): boolean {
    return targets(job.scope).some((rec) => outputOf(rec.settings).look_intensity > 0 && findLook(outputOf(rec.settings).look)?.input === "logc3_to_rec709");
  }

  /** Photos of a job with adjustments beyond their look: what a command line can't carry. */
  function adjustedPhotos(job: ExportJob): number {
    const plain = (s: DevelopSettings): string => JSON.stringify(withoutLook(s));
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
      if (error && run.stopAt === i) {
        // A failure of the destination ends the job: this is its last event, and the files after it are not tried.
        emit("export_progress", { job_id: run.id, file: run.files[i] ?? "", done: i + 1, total, file_percent: null, error, skipped: false, stopped: true });
        run.progress = total;
        run.reported = total;
        stopExport(run);
        announceBusy();
        return;
      }
      if (error || skipped) {
        emit("export_progress", { job_id: run.id, file: run.files[i] ?? "", done: i + 1, total, file_percent: 100, error, skipped, stopped: false });
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
      stopped: false,
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
    const look = lookSlug(first ? outputOf(first.settings).look : preferences.default_look);
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
    if (cpuFrameTimer !== null) clearTimeout(cpuFrameTimer);
    cpuFrameTimer = null;
    for (const timer of thumbTimers.values()) clearTimeout(timer);
    thumbTimers.clear();
    for (const timer of nodeThumbTimers.values()) clearTimeout(timer);
    nodeThumbTimers.clear();
    nodeThumbs.clear();
    blurMaps.clear();
    visionReady.clear();
    for (const timer of visionTimers.values()) clearTimeout(timer);
    visionTimers.clear();
    depthReady.clear();
    depthFailures.clear();
    depthFailedOnce.clear();
    for (const timer of depthTimers.values()) clearTimeout(timer);
    depthTimers.clear();
    depthWaiting.clear();
    objectReady.clear();
    for (const entry of objectEncoders.values()) clearTimeout(entry.timer);
    objectEncoders.clear();
    for (const run of exportRuns.values()) stopExport(run);
    exportRuns.clear();
    for (const run of importRuns) {
      if (run.timer !== null) clearInterval(run.timer);
      run.timer = null;
      run.resolve();
    }
    importRuns.clear();
    openRun?.stop(true);
    for (const timer of downloadTimers) clearTimeout(timer);
    downloadTimers = [];
    watching = false;
  }

  function resetState(): void {
    stopAll();
    photos = [];
    current = 0;
    highestId = 0;
    removal = null;
    developProgress = 0;
    const stored = options.persistPreferences ? readStored() : null;
    preferences = stored ? { ...defaultPreferences(options), ...withoutRetired(stored.preferences) } : defaultPreferences(options);
    // A package id this version does not know (the display LUT packages an earlier mock kept) is ignored, as the core ignores a library's display LUTs.
    importedPackages = (stored?.importedPackages ?? []).filter((id): id is PackageId => id in PACKAGE_FACTS);
    userLooks = (stored?.userLooks ?? []).map(withLibraryFields);
    cliLink = stored?.cliLink ?? null;
    sources = defaultSources();
    choices.folder = [];
    choices.files = [];
    choices.open = [];
    menuState = null;
    system.opened = [];
    system.revealed = [];
    system.trashed = [];
    system.restarts = 0;
    looks = buildLooks();
    failures.clear();
    cancelDialog = false;
    viewport.rect = null;
    viewport.mode = { kind: "fit" };
    viewport.offset = { x: 0, y: 0 };
    viewport.before = false;
    viewport.maskView = null;
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
    get menuState() {
      return menuState;
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

    openPaths: async (paths) => {
      const read = await readSlowly(paths);
      return run("openPaths", () => openAfterReading(paths, read));
    },
    openDialog: async (kind: OpenKind) => {
      if (cancelDialog) {
        cancelDialog = false;
        return run("openDialog", () => null);
      }
      const queued = choices.open.shift();
      const paths = queued ?? (kind === "folder" ? [folderPath()] : SCENES.map((s) => `${folderPath()}/${s.name}.ARW`));
      const read = await readSlowly(paths);
      return run("openDialog", () => openAfterReading(paths, read));
    },
    cancelOpen: () =>
      run("cancelOpen", () => {
        openRun?.stop(true);
      }),
    sessionState: () => run("sessionState", snapshot),
    closeSession: () =>
      run("closeSession", () => {
        // Development stops with the session; exports in flight carry on, as in the core.
        stopDevelopment();
        for (const timer of thumbTimers.values()) clearTimeout(timer);
        thumbTimers.clear();
        thumbnailUrls.clear();
        lostThumbnails.clear();
        forgetNodeThumbnails([...nodeThumbs.keys(), ...nodeThumbTimers.keys()]);
        photos = [];
        current = 0;
        highestId = 0;
        removal = null;
        developProgress = 0;
        viewport.mode = { kind: "fit" };
        viewport.offset = { x: 0, y: 0 };
        viewport.before = false;
    viewport.maskView = null;
        sendFrame();
      }),
    setCurrent: (photoId) =>
      run("setCurrent", () => {
        const index = photos.findIndex((p) => p.summary.id === photoId);
        if (index < 0) fail("app", "That photo is no longer open. Open it again.");
        current = index;
        const shown = photos[index];
        if (shown) vanishIfGone(shown);
        viewport.mode = { kind: "fit" };
        viewport.offset = { x: 0, y: 0 };
        viewport.before = false;
    viewport.maskView = null;
        emitSoon("render_stats", RENDER_STATS);
        sendFrame();
      }),
    setPicked: (photoId, picked) =>
      run("setPicked", () => {
        photo(photoId).summary.picked = picked;
      }),
    removePhotos: (photoIds) => run("removePhotos", () => removeNow(photoIds)),
    restorePhotos: () => run("restorePhotos", restoreNow),

    getSettings: (photoId) =>
      run("getSettings", () => resolveSettings(photoId, photo(photoId).settings, estimates())),
    updateSettings: (photoId, patch: SettingsPatch) =>
      run("updateSettings", () => {
        const rec = photo(photoId);
        rec.settings = patched(rec, patch);
        emitSoon("render_stats", RENDER_STATS);
        scheduleThumbnail(rec);
        if (rec === photos[current]) sendFrame();
        return resolveSettings(photoId, rec.settings, estimates());
      }),
    applyLook: (scope, lookId, intensity) =>
      run("applyLook", () => {
        const look = findLook(lookId);
        if (!look || look.missing) fail("looks", "That look isn't in the library any more. Import it again.");
        for (const rec of targets(scope)) {
          rec.settings = applyPatch(rec.settings, outputPatch({ look: lookId, look_intensity: clamp(intensity, 0, 100) }));
          scheduleThumbnail(rec);
        }
        emitSoon("render_stats", RENDER_STATS);
        sendFrame();
      }),
    rerunAuto: (photoId) =>
      run("rerunAuto", () => {
        const rec = photo(photoId);
        // Exposure, white (the lift's offset), white balance (including the light preset and a picked white) and noise; contrast and
        // saturation too while Apple Auto acts on the photo (pending, answered or failed: the Automatic card shows their rows), and
        // otherwise they stay as they are (`docs/19` sections 5 and 10).
        const apple = estimates().apple.state;
        const appleActs = apple === "pending" || apple === "answered" || apple === "failed";
        const input = inputPatch({
          exposure: AUTO,
          light_source: "auto",
          white_balance: AUTO,
          white_pick: null,
          noise_reduction: AUTO,
        });
        const main = appleActs
          ? mainGradePatch({
              contrast: AUTO,
              saturation: AUTO,
              highlights: AUTO,
              shadows: AUTO,
              black: AUTO,
              color_balance: { shadows: AUTO, highlights: AUTO },
            })
          : {};
        rec.settings = clampSettings(applyPatch(rec.settings, foldPatches(input, main)));
        emitSoon("render_stats", RENDER_STATS);
        scheduleThumbnail(rec);
        if (rec === photos[current]) sendFrame();
        return resolveSettings(photoId, rec.settings, estimates());
      }),
    copySettings: (photoId) =>
      run("copySettings", () => {
        const rec = photo(photoId);
        if (rec.skipReason) fail("open", "This file can't be opened, so there are no settings to copy.");
        if (rec.phase < 3) fail("open", "That photo is still developing. Try again in a moment.");
        return copiedPatch(resolveSettings(photoId, rec.settings, estimates()));
      }),
    pasteSettings: (photoIds, patch: SettingsPatch) =>
      run("pasteSettings", (): PasteReport => {
        const wanted = patch.nodes?.[OUTPUT_ID]?.look;
        if (wanted !== undefined) {
          const look = findLook(wanted);
          if (!look || look.missing) fail("looks", "That look isn't in the library any more. Import it again.");
        }
        const report: PasteReport = { pasted: [], skipped: [] };
        for (const id of new Set(photoIds)) {
          const rec = photos.find((p) => p.summary.id === id);
          if (!rec || rec.skipReason || rec.phase < 3) {
            report.skipped.push(id);
            continue;
          }
          // The crop, the turn and the white pick are never pasted, whatever the patch says; a copy of the whole graph replaces the
          // target's graph (`pastedOnto`).
          const next = patched(rec, pastedOnto(rec.settings, patch));
          const step = diffSettings(rec.settings, next);
          report.pasted.push({ id, before: step?.before ?? {}, after: step?.after ?? {} });
          if (step) {
            rec.settings = next;
            scheduleThumbnail(rec);
            emitSoon("settings_changed", { photo_id: id });
            if (rec === photos[current]) sendFrame();
          }
        }
        if (report.pasted.length > 0) emitSoon("render_stats", RENDER_STATS);
        return report;
      }),
    setMenuState: (state) =>
      run("setMenuState", () => {
        menuState = { ...state };
      }),
    perfReport: () => run("perfReport", () => undefined),
    histogram: (photoId, at) =>
      run("histogram", (): HistogramData => {
        const rec = photo(photoId);
        const node = signalNode(rec, at);
        if (node !== null) return sceneHistogram(signalSceneAt(rec, node, "curve"));
        const { values } = resolveSettings(photoId, rec.settings, estimates());
        // Exposure moves the plot along the tonal scale. Before does not: the core's histogram is always that of the
        // finished picture (docs/02). At the Input it is the Input's picture, which Main's grade does not touch.
        const shift = (values.input.exposure_ev - MOCK_ESTIMATES.input.exposure_ev) * 0.06;
        return sceneHistogram(at === "input" ? adjustScene(rec.scene, { ...adjustmentOf(rec, false) }) : developed(rec), shift);
      }),

    chromaScatter: (photoId, at) =>
      run("chromaScatter", (): ChromaScatter => {
        const rec = photo(photoId);
        const node = signalNode(rec, at);
        // The photo's colours at the warp's input, seeded by the photo so the dots are the same on every run. At another corrector they
        // are the colours after the nodes before it and its own trims and grade (`docs/01`).
        const scene = node === null ? warpInput(rec) : signalSceneAt(rec, node, "warp");
        return { points: sceneScatter(scene, 7919 * photoId + 17) };
      }),

    suggestCrop: (photoId, aspect, orientation) =>
      runAsync("suggestCrop", async (): Promise<CropSuggestion> => {
        const rec = photo(photoId);
        if (rec.phase < 3) fail("open", "This photo is still being developed. Try again in a moment.");
        // The core renders and analyses the photo's picture first: answer after a moment.
        await new Promise((resolve) => setTimeout(resolve, 150 / options.speed));
        return mockCropSuggestion(rec.summary, aspect, orientation);
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
        // The core draws a frame for every zoom and pan (the mock's own picture does not move).
        if (event.kind !== "pick_white") renderOnCpu();
        if (event.kind === "pan" && viewport.mode.kind !== "fit") {
          viewport.offset = { x: viewport.offset.x + event.dx, y: viewport.offset.y + event.dy };
        } else if (event.kind === "toggle_zoom") {
          viewport.mode = viewport.mode.kind === "fit" ? { kind: "percent", value: 100 } : { kind: "fit" };
          if (viewport.mode.kind === "fit") viewport.offset = { x: 0, y: 0 };
        } else if (event.kind === "pick_white" && rec) {
          // The core finds the spot of the photo under the click (straightening, turns and lens crop included) and sets the pick; a click
          // on the paper picks nothing. Like `update_settings` with a pick, the mock sets the white balance by hand from the colour there.
          const point = photoPointAt(rec, event.x, event.y);
          if (point) {
            rec.settings = clampSettings(
              applyPatch(
                rec.settings,
                inputPatch({ white_pick: point, light_source: "auto", white_balance: { kind: "manual", ...whiteFromPick(rec.scene, point) } }),
              ),
            );
            emitSoon("settings_changed", { photo_id: rec.summary.id });
            emitSoon("render_stats", RENDER_STATS);
            scheduleThumbnail(rec);
            sendFrame();
          }
        }
      }),
    viewportSetMode: (mode, maskView) =>
      run("viewportSetMode", () => {
        viewport.mode = mode;
        if (mode.kind === "fit") viewport.offset = { x: 0, y: 0 };
        viewport.maskView = maskView ?? null;
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

    sampleChroma: (x, y, at, tap) =>
      run("sampleChroma", (): ChromaSample | null => {
        const rec = photos[current];
        if (!rec) fail("viewport", "Open a photo first.");
        const node = signalNode(rec, at);
        // On the paper around a letterboxed picture there is no colour to pick.
        const point = photoPointAt(rec, x, y);
        if (!point) return null;
        // The warp's input is after the node's trim and grade; the node's input (`tap` "node_input", a Colour range's Pick colour) is the
        // picture it reads, which Main and the Input take from the photo as adjusted at the Input.
        const stage = tap === "node_input" ? "input" : "warp";
        const scene = node === null ? warpInput(rec) : signalSceneAt(rec, node, stage);
        const [a, b] = hexChroma(scene[layerAt(scene, clamp(point[0], 0, 1), clamp(point[1], 0, 1))]);
        return { a, b };
      }),

    viewportMapPoints: (points, from: MapFrom) =>
      run("viewportMapPoints", (): Array<[number, number] | null> => {
        const rec = photos[current];
        if (!rec) return points.map(() => null);
        const rect = cellRect();
        return points.map(([x, y]) => {
          if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
          if (from === "cell") return photoPointAt(rec, x, y);
          const [u, v] = photoFrameInCell(rec, x, y, rect);
          return [u * rect.width, v * rect.height];
        });
      }),
    maskKinds: () => run("maskKinds", (): MaskKind[] => [...options.maskKinds]),
    maskFind: (photoId, node, mask, prompt) =>
      runAsync("maskFind", async (): Promise<MaskFindResult> => {
        if (!options.maskKinds.includes("object")) fail("mask", "This copy of Yellowmoon has no Object model. Reinstall the app.");
        const rec = photo(photoId);
        const item = nodeOf(rec.settings, node);
        const row = item?.kind === "corrector" ? item.masks.items[mask] : undefined;
        if (row?.kind !== "object") fail("mask", "That mask is not an Object mask. Select an Object mask and try again.");
        if (prompt.kind === "point" && row.points.length >= 256) fail("mask", "An Object mask can have up to 256 points.");
        // The click waits for the encoder if it is still running, then takes a moment of its own.
        await encoderDone(photoId);
        objectReady.add(photoId);
        await new Promise((resolve) => setTimeout(resolve, FIND_MS / options.speed));
        const round = (v: number): number => Math.round(clamp(v, 0, 1) * 10000) / 10000;
        // The scene's mountain is the far hill; a negative point only takes a part away from what the row has.
        const onHill = (x: number, y: number): boolean => layerAt(rec.scene, clamp(x, 0, 1), clamp(y, 0, 1)) === "far";
        if (options.maskFind === "none") return { kind: "nothing" };
        if (prompt.kind === "point") {
          const found = prompt.positive ? onHill(prompt.x, prompt.y) : row.points.length > 0 || row.box !== null;
          if (!found) return { kind: "nothing" };
          return { kind: "found", points: [...row.points, [round(prompt.x), round(prompt.y), prompt.positive]], box: row.box, took_ms: FOUND_MS };
        }
        const box: [number, number, number, number] = [round(Math.min(prompt.x0, prompt.x1)), round(Math.min(prompt.y0, prompt.y1)), round(Math.max(prompt.x0, prompt.x1)), round(Math.max(prompt.y0, prompt.y1))];
        if (!onHill((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)) return { kind: "nothing" };
        return { kind: "found", points: row.points.map((p) => [...p]), box, took_ms: FOUND_MS };
      }),

    sampleDepth: (photoId, x, y) =>
      runAsync("sampleDepth", async (): Promise<DepthFar | null> => {
        if (!options.maskKinds.includes("depth")) fail("mask", "This copy of Yellowmoon has no Depth model. Reinstall the app.");
        if (!Number.isFinite(x) || !Number.isFinite(y)) fail("mask", "That click is not on the photo. Click the photo again.");
        const rec = photo(photoId);
        // The click waits for the depth map if it is still being made, then reads the ranks of the scene's areas.
        await depthDone(photoId);
        if (depthFailures.has(photoId)) return null;
        depthReady.add(photoId);
        return pickDepthLimit(depthRankIn(rec.scene, clamp(x, 0, 1), clamp(y, 0, 1)));
      }),
    focusPlane: (photoId) =>
      run("focusPlane", (): FocusPlaneResult => {
        if (!options.maskKinds.includes("depth")) fail("mask", "This copy of Yellowmoon has no Depth model. Reinstall the app.");
        const rec = photo(photoId);
        // Some scenes have a focus point on the near ground (the scenes with a person), the others none (an older body).
        if (!SCENES_WITH_PEOPLE.has(rec.scene.name)) return { far: null, exact_centre: false, reason: "no_focus_point" };
        if (depthFailures.has(photoId)) return { far: null, exact_centre: false, reason: "map_failed" };
        const picked = pickDepthLimit(DEPTH_RANKS.near);
        // One scene's focus point is exactly the frame's centre, the value Sony writes when it does not know (the card says to check the result).
        return { far: picked.far, exact_centre: rec.scene.name === FOCUS_AT_CENTRE_SCENE, reason: null };
      }),

    retryDepth: (photoId) =>
      run("retryDepth", (): void => {
        const rec = photo(photoId);
        // Only a map that failed starts again; the worker runs as it did (the viewport is busy and finding, then `depth_state` says how it ended).
        if (!depthFailures.delete(photoId)) return;
        prepareDepth(rec);
      }),

    lensList: () => run("lensList", () => (options.lensesMissing ? [] : MOCK_LENSES.map((lens) => ({ ...lens })))),
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
    filmCatalogue: () => run("filmCatalogue", () => mockFilmCatalogue()),
    filmTile: (photoId, node, stock) =>
      run("filmTile", (): string => {
        const rec = photo(photoId);
        if (nodeOf(rec.settings, node)?.kind !== "film") fail("render", `This photo has no Film node ${node}.`);
        if (!MOCK_FILM_STOCKS.some((s) => s.id === stock)) fail("render", `Unknown film stock "${stock}". Run "ym film list" to see the stocks.`);
        // The photo as it is now with this node's stock replaced, through the Output's look (an address, as the shell answers).
        const settings = applyPatch(rec.settings, { nodes: { [String(node)]: { film: { stock } } } });
        const output = outputOf(settings);
        const scene = developScene(throughFilms(rec, settings), { slug: lookSlug(output.look), intensity: output.look_intensity }, adjustmentOf(rec, true));
        return bytesToUrl(Array.from(scenePng(scene)));
      }),
    nodeThumbnails: (photoId) =>
      run("nodeThumbnails", (): void => {
        // Answers at once; the pictures follow as an event, behind whatever the person is doing (a request while one waits is that one).
        const rec = photos.find((p) => p.summary.id === photoId);
        if (!rec || nodeThumbTimers.has(photoId)) return;
        nodeThumbTimers.set(
          photoId,
          setTimeout(() => {
            nodeThumbTimers.delete(photoId);
            makeNodeThumbnails(rec);
          }, NODE_THUMB_MS / options.speed),
        );
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
        if (r.kind === "package") {
          if (!importedPackages.includes(r.package_id)) importedPackages = [...importedPackages, r.package_id];
        } else {
          const base = cubeName(found.name);
          const slug = slugify(base);
          // The kind of file is the extension (a folder of look files is taken for `.alf4` files); a `.cube` has the size the question was asked with.
          const extension = /\.(cube|alf4c?|aml)$/i.exec(found.name)?.[1]?.toLowerCase();
          const format: LookFormat = extension === "alf4" || extension === "alf4c" || extension === "aml" ? extension : r.kind === "look_files" ? "alf4" : "cube";
          const lutSize = r.kind === "needs_answer" ? (r.lut_size ?? 33) : 33;
          const made: Look[] = Array.from({ length: r.file_count }, (_, i) =>
            importedLook(
              `user:${slug}${r.file_count > 1 ? `-${i + 1}` : ""}`,
              r.file_count > 1 ? `${base} ${i + 1}` : base,
              { kind: "user" },
              // A look file says what it expects; a `.cube` is what the person answered.
              (r.kind === "look_files" ? r.input : null) ?? input ?? "logc4",
              format,
              lutSize,
            ),
          );
          userLooks = [...userLooks.filter((l) => !made.some((m) => m.id === l.id)), ...made];
        }
        rebuildLooks();
        persist();
        const added = looks.filter((l) => !before.has(l.id));
        emitSoon("looks_changed", null);
        return { added, unchanged: Math.max(0, r.file_count - added.length) };
      }),
    looksRecognise: (path) => run("looksRecognise", () => recognise(path)),
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
        for (const rec of photos) if (outputOf(rec.settings).look === id) rec.settings = applyPatch(rec.settings, outputPatch({ look: NEUTRAL }));
        syncDefaultLook();
        emitSoon("looks_changed", null);
      }),

    exportEstimate: (job) =>
      run("exportEstimate", (): ExportEstimate => {
        const count = targets(job.scope).length;
        const bytes = Math.round(megabytesPerPhoto(job.options) * count * 1_000_000);
        return {
          bytes,
          free_bytes: options.freeBytes,
          folder: (job.options.destination ?? `${folderPath()} export`).split("/").filter(Boolean).pop() ?? null,
          low_space: options.freeBytes < 1.35 * bytes,
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
          stopAt: planned.stopAt,
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
    thumbnailsInView: (ids) => run("thumbnailsInView", () => thumbnailsInView(ids)),

    preferencesGet: () => run("preferencesGet", () => preferences),
    preferencesSet: (patch: PreferencesPatch) =>
      run("preferencesSet", () => {
        if (patch.default_look !== undefined) {
          const look = findLook(patch.default_look);
          if (!look || look.missing) fail("preferences", "That look isn't in the library. Pick another default look.");
        }
        if (patch.apple_auto_instructions !== undefined) {
          const n = [...patch.apple_auto_instructions].length;
          if (n > MOCK_INSTRUCTIONS_MAX_CHARS) {
            fail(
              "preferences",
              `The instructions are ${n} characters long. At most ${MOCK_INSTRUCTIONS_MAX_CHARS} fit, because the model must also see the photo.`,
            );
          }
        }
        // The consent is one patch with the choice: a program is never chosen without the person's yes (`docs/22` section 10.2).
        const consent: AutoConsent = patch.auto_consent
          ? { claude: patch.auto_consent.claude === true, agy: patch.auto_consent.agy === true }
          : preferences.auto_consent;
        if ((patch.auto_provider === "claude" || patch.auto_provider === "agy") && !consent[patch.auto_provider]) {
          fail("preferences", consentRefusal(patch.auto_provider));
        }
        const next = { ...preferences } as Record<string, unknown>;
        for (const [key, value] of Object.entries(patch)) if (value !== undefined) next[key] = value;
        next.auto_consent = consent;
        // Saved as "" when they are empty or the default (the core's `own_instructions`).
        if (patch.apple_auto_instructions !== undefined) {
          const own = patch.apple_auto_instructions;
          next.apple_auto_instructions = own.trim() === "" || own === MOCK_DEFAULT_INSTRUCTIONS ? "" : own;
        }
        // Auto follows its preference at once: Default Auto, every photo back to the rule; a provider, its answers (kept) come back.
        changingEstimates(() => {
          preferences = next as unknown as Preferences;
        });
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
    autoStatus: () => run("autoStatus", autoStatus),
    // The mock probes nothing, so the cached read is the same answer; the store's tests tell the two calls apart by spying on them.
    autoStatusCached: () => run("autoStatusCached", autoStatus),
    appInfo: () => run("appInfo", (): AppInfo => ({ version: APP_VERSION, pipeline_version: "1.0.0", os: options.os })),
    displayOptions: () => run("displayOptions", (): DisplayOptions => ({ ...options.displayOptions })),

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
    queueChoice(kind: "folder" | "files" | "open", answer: string | string[] | null) {
      if (kind === "folder") choices.folder.push(answer as string | null);
      else if (kind === "open") choices.open.push(answer as string[]);
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
    setAppleAuto(mode) {
      changingEstimates(() => {
        options = withDefaults(options, { appleAuto: mode });
      });
    },
    setProviders(next) {
      changingEstimates(() => {
        options = { ...options, providers: next };
      });
      emitSoon("auto_status_changed", null);
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
