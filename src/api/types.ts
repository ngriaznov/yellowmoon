/**
 * Types for the IPC contract between the webview and the Rust core (`docs/01-architecture.md`
 * §IPC between webview and core). Everything that crosses the bridge is typed here, and
 * nowhere else.
 *
 * This file is the hand-written contract the screens are written against. `bindings.ts` is what
 * `tauri-specta` generates from the Rust commands and types (`cargo test -p yellowmoon-app
 * bindings` keeps it current); `contract.ts` makes `tsc` fail when the two disagree, and the
 * Rust tests compare the field names and variants of this file with the generated ones, so a
 * drift on either side is an error in CI rather than a surprise at run time. The generated file
 * is looser in places (`SettingsPatch` allows partial nested objects, `UpdateState.percent` may
 * be null) and has one named enum per setting where this file has the generic `AutoOr`; the
 * JSON is the same.
 *
 * Conventions, matching serde's defaults on the Rust side:
 *  - Payload fields are snake_case, as in `docs/01` (`job_id`, `look_intensity`, `radius_px`).
 *  - Tagged unions use `kind` as the tag, with snake_case variant names, for example
 *    `{ kind: "auto" } | { kind: "manual", ev: number }`.
 *  - Command names are snake_case and their arguments camelCase (Tauri 2 converts them);
 *    see `src/api.ts`.
 *  - Units are in the names: `_ev`, `_k` / `kelvin`, `_px`, `_deg`, `_s`, `_ms`.
 *
 * Fields marked "(addition)" are not spelled out in `docs/01` but the screens in
 * `design/mockups/` need them; they are listed in the M0-01 report for the spec owners.
 */

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/** Identifies a photo for the lifetime of the session only (`docs/01` §Session). */
export type PhotoId = number;

/**
 * Identifies a look: `"builtin:neutral"`, `"builtin:signature"`, `"arri:<group>:<slug>"` or
 * `"user:<slug>"` (`docs/08` §Look model).
 */
export type LookId = string;

/** Identifies a queued export job. */
export type ExportJobId = number;

// ---------------------------------------------------------------------------
// Camera and photo summary
// ---------------------------------------------------------------------------

/**
 * Why a camera profile was chosen (`docs/03` §Detection, `CameraMatch.how`).
 * `exact`: the body's own measured profile. `sensor_family`: borrowed from a body with the
 * same sensor (`via` names it). `default`: the A7R IV stand-in.
 */
export type MatchKind =
  | { kind: "exact" }
  | { kind: "sensor_family"; via: string }
  | { kind: "default" };

/** The detected body and the profile used for it (`docs/03` §Detection). */
export interface CameraInfo {
  /** Body as shown in the UI, from the registry or the Model string: "Sony A1". */
  body: string;
  /** The camera's own model string, "ILCE-1", for the RAW tab's "Detected" row (addition). Empty when the file has none. */
  model: string;
  /** Sony's model ID (maker note 0xB001), when the file has one. */
  model_id: number | null;
  /** Display name of the profile used: "A7R IV". */
  profile: string;
  /** Profile id: "sony-ilce-7rm4". */
  profile_id: string;
  /** Profile version (semver). */
  profile_version: string;
  /** How the profile was chosen; anything but `exact` gets the STAND-IN tag. */
  match_kind: MatchKind;
}

/** A photo's place in the development pipeline (the Start screen's list, `docs/11` §Start). */
export type PhotoStatus =
  | { kind: "waiting" }
  /** `stage` is the stage in flight (see `PhotoStage`). */
  | { kind: "developing"; stage: PhotoStage }
  | { kind: "ready" }
  /** A file that can't be developed stays in the filmstrip with this plain-words reason. */
  | { kind: "skipped"; reason: string };

/** Stages of the background job that develops a freshly opened photo (`docs/01` §Session). */
export type PhotoStage = "probe" | "decode" | "estimate" | "thumbnail";

/** What the UI needs to know about one photo without asking for its settings. */
export interface PhotoSummary {
  id: PhotoId;
  /** File name with its extension: "DSC04204.ARW". */
  file_name: string;
  camera: CameraInfo;
  iso: number;
  /** Exposure time in seconds: 0.008 for 1/125. */
  shutter_s: number;
  /** F-number: 2.8. */
  aperture_f: number;
  /** Capture time as local ISO 8601 without a zone: "2026-10-04T17:42:10". */
  captured_at: string;
  /** Full-frame width in pixels, after the camera's orientation (addition: "Full · 8640 px"). */
  width_px: number;
  /** Full-frame height in pixels, after the camera's orientation (addition). */
  height_px: number;
  status: PhotoStatus;
  picked: boolean;
  /** Lossless M or S: the camera already processed it, so colour emulation is approximate. */
  reduced: boolean;
}

/** The session as the webview mirrors it (`session_state`). */
export interface SessionState {
  /** Name of the folder the photos came from, for the header cell: "Lisbon" (addition). */
  folder_name: string | null;
  photos: PhotoSummary[];
  /** Index into `photos` of the current photo; null when the session is empty. */
  current_index: number | null;
  /** Ids of the picked photos. The same information as `PhotoSummary.picked`. */
  picked: PhotoId[];
  /** Where Export saves by default: "<source folder> export" (addition, `docs/09` Save to). */
  default_export_folder: string | null;
  /**
   * What the `open_paths` or `open_dialog` call that answered with this state did; null (or absent) in every other answer. It lets
   * Photos say "Added 3 photos. 2 were already open." when files are added to a session (addition, 2026-10-16).
   */
  opened?: OpenSummary | null;
}

/**
 * The answer of `remove_photos` and `restore_photos` (addition, the owner's request "I also want to remove photos from the film
 * strip"): the photos that were taken out of the filmstrip or put back, and the session afterwards.
 */
export interface SessionChange {
  /** The photos taken out (those of the ids that were in the session) or put back, in filmstrip order. */
  photo_ids: PhotoId[];
  /** `restore_photos` only: photos of the removal that stayed out because their file had been added again meanwhile. 0 otherwise. */
  already_open: number;
  /** The session afterwards. A snapshot, as every answer with a session is (see `mergeSession`). */
  session: SessionState;
}

/** What one `open_paths` or `open_dialog` call did to the session. */
export interface OpenSummary {
  /** ARW files added to the filmstrip. */
  added: number;
  /** ARW files that were already in the session and were not added again. */
  already_open: number;
}

// ---------------------------------------------------------------------------
// Develop settings (docs/01 §DevelopSettings)
// ---------------------------------------------------------------------------

/**
 * A value that is either resolved from the automatic estimates or set by hand. Moving a
 * slider turns `auto` into `manual`; "Reset" turns it back (`docs/01` §DevelopSettings).
 */
export type AutoOr<Manual extends object> = { kind: "auto" } | ({ kind: "manual" } & Manual);

/**
 * `contrast`, `highlights`, `shadows`, `black` and `saturation` (`docs/19` section 5): Auto, which is 0 or Apple Auto's value for
 * the photo (`AutoEstimates.contrast`, `.highlights`, …), or a manual value in −100…100. A slider dropped on the Auto value sends
 * Auto (`manualOrAuto`).
 */
export type PercentSetting = AutoOr<{ value: number }>;

/** `light_source`: Colour › White balance, the segmented control. */
export type LightSource = "auto" | "day" | "shade" | "tungsten" | "led" | "fluorescent";

/** A point in the colour-balance disc: `[a, b]`, each within the unit disc. */
export type ColorBalancePoint = [a: number, b: number];

/**
 * The shadows or highlights colour-balance wheel (`docs/19` section 5): Auto, which is `[0, 0]` or the colour Apple Auto gave that
 * zone (`AutoEstimates.balance_shadows`, `.balance_highlights`), or a manual point. A puck dropped on the Auto point sends Auto.
 */
export type BalanceSetting = AutoOr<{ point: ColorBalancePoint }>;

/** Colour balance wheels: the shadows and highlights wheels are Auto (the default) or a point; the midtones wheel is a point, `[0, 0]` by default. */
export interface ColorBalance {
  shadows: BalanceSetting;
  midtones: ColorBalancePoint;
  highlights: BalanceSetting;
}

/** The three wheels' points in effect (`ResolvedValues.color_balance`): a manual point, or the photo's Auto point. */
export interface ColorBalancePoints {
  shadows: ColorBalancePoint;
  midtones: ColorBalancePoint;
  highlights: ColorBalancePoint;
}

/** A curve control point, `[x, y]` within [0, 1]². */
export type CurvePoint = [x: number, y: number];

/** Luma plus R, G and B curves, up to 16 control points each; identity is `[[0,0],[1,1]]`. */
export interface Curves {
  luma: CurvePoint[];
  red: CurvePoint[];
  green: CurvePoint[];
  blue: CurvePoint[];
}

/**
 * A point of the colour warp's chroma plane, `[a, b]` = `(e1·v, e2·v)` of a LogC4 pixel `v` (`docs/02` C9b): `a` red against cyan,
 * `b` green against blue. Scaled back to `|c| ≤ 0.5` by the core.
 */
export type ChromaPoint = [a: number, b: number];

/**
 * One pin of Colour › Colour warp (`docs/02` C9b, `docs/01` `color_warp`). A moving pin takes the colours near `source` to `target`; a
 * locked pin holds the colours within its range in place (its target is its source, its weights, pivot and exposure the defaults).
 */
export interface ColorWarpPin {
  source: ChromaPoint;
  /** Where the colour goes; equal to `source` for a pin that has not moved and for a locked pin. */
  target: ChromaPoint;
  locked: boolean;
  /** How far the pin reaches across the plane, 0.01–0.25, default 0.04 (the card shows it ×100). */
  range: number;
  /** How much of the move the shadows take, 0–1, default 1. */
  shadows: number;
  /** How much of the move the highlights take, 0–1, default 1. */
  highlights: number;
  /** Where the shadows end and the highlights begin, in stops from middle grey, −6…6, default 0. */
  pivot_ev: number;
  /** A change of brightness for the pin's colours, in stops, −2…2, default 0. */
  exposure_ev: number;
}

/** Colour › Colour warp: at most 16 pins; no pins is the identity. A patch replaces the whole list. */
export interface ColorWarp {
  pins: ColorWarpPin[];
}

/** RAW › Decode switches; all true by default. */
export interface DecodeOptions {
  neural_demosaic: boolean;
  highlight_recovery: boolean;
  lens_corrections: boolean;
  chromatic_aberration: boolean;
}

/** RAW › Range, each 0–1: Highlights 1 and Low light 0.5 by default (`docs/02` C7). */
export interface Rolloff {
  /**
   * The highlight range: where the camera's clipping point lands, from where it was recorded (0) to white (1, the clip of an
   * ALEXA at EI 800). Mid grey and everything up to a stop under the clip stay where they are.
   */
  highlights: number;
  /** The low-light toe: how much the noise near black is softened. */
  low_light: number;
}

/** A rectangle in image coordinates normalised to [0, 1]². */
export interface NormRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Crop aspect presets; the labels are Free, Original, 1:1, 4:5, 16:9. */
export type CropAspect = "free" | "original" | "square" | "four_by_five" | "sixteen_by_nine";

/** The crop (per photo, in memory only): a rectangle, a straighten angle and an aspect. */
export interface Crop {
  rect: NormRect;
  /** Straighten angle, −45…45. */
  angle_deg: number;
  aspect: CropAspect;
}

/** Orientation applied after the camera's EXIF orientation, in degrees. */
export type Orientation = 0 | 90 | 180 | 270;

/**
 * Every control in the UI and every CLI flag maps to one field. Defaults reproduce the
 * automatic development (`docs/01` §DevelopSettings). Field names, ranges and defaults follow
 * that table.
 */
export interface DevelopSettings {
  /** Look tab. Defaults to the user's default look (Neutral out of the box). */
  look: LookId;
  /** 0–100 %, default 100. */
  look_intensity: number;
  /** Light › Exposure: `Auto` or `Manual(ev)` in −4…+4, step 0.01. */
  exposure: AutoOr<{ ev: number }>;
  /** Light › Exposure › Contrast (and Look › Tone): `Auto` (the default) or `Manual(value)` in −100…100. */
  contrast: PercentSetting;
  /** −100…100, default 0. */
  brightness: number;
  /** Light › High dynamic range, each −100…100: Highlights and Shadows are `Auto` (the default: 0, or Apple Auto's value) or a value. */
  highlights: PercentSetting;
  shadows: PercentSetting;
  white: number;
  /**
   * Light › White › Offset: stops added to the automatic white lift (`AutoEstimates.white_ev`), the
   * per-photo gain on the tones above +1 stop over middle grey (`docs/02` C12 step 1b). −3…+3, step
   * 0.05, default 0 = the automatic amount; a negative offset of the automatic amount or more gives no
   * lift. A plain number, not Auto. Only the built-in ACES 2 display rendering has the lift.
   */
  white_offset_ev: number;
  /** Light › High dynamic range › Black: `Auto` (the default: 0, or Apple Auto's black level) or a value in −100…100. */
  black: PercentSetting;
  /** Colour › White balance, segmented control. */
  light_source: LightSource;
  /** Colour › White balance: `Auto` or `Manual { kelvin 2000–11000 step 50, tint −50…50 }`. */
  white_balance: AutoOr<{ kelvin: number; tint: number }>;
  color_balance: ColorBalance;
  /** Colour › Colour balance › Saturation (and Look › Tone): `Auto` (the default) or `Manual(value)` in −100…100. */
  saturation: PercentSetting;
  /** Colour › Colour warp: the pins of stage C9b, `{ pins: [] }` by default. */
  color_warp: ColorWarp;
  /** Detail › Sharpening: `Auto` or `{ amount 0–300, radius_px 0.3–2.5 }`. */
  sharpen: AutoOr<{ amount: number; radius_px: number }>;
  /**
   * Detail › Sharpening › Output: the strength in percent of the sharpening applied to the picture
   * at the size it is shown or written (every viewport frame, thumbnail and JPEG, TIFF or PNG
   * export; never a DNG). 0–200, step 5, default 100; a plain number, not Auto; 0 is off. The
   * owner's decision of 2026-10-13 (`docs/algorithms/04-output-sharpening.md`).
   */
  output_sharpening: number;
  /** Detail › Noise reduction: `Auto` or `{ luminance, color, detail }`, each 0–100. */
  noise_reduction: AutoOr<{ luminance: number; color: number; detail: number }>;
  curve: Curves;
  decode: DecodeOptions;
  rolloff: Rolloff;
  /** Crop (header); `null` when the photo isn't cropped. */
  crop: Crop | null;
  orientation: Orientation;
  /** Pick white (header): the sampled point as `[x, y]` in [0, 1]² image coordinates. */
  white_pick: [x: number, y: number] | null;
}

/** A partial update: only the fields present change. `update_settings(photo_id, patch)`. */
export type SettingsPatch = Partial<DevelopSettings>;

/**
 * What `paste_settings` did to one photo: the fields it changed, as two patches that make one undo step. `before` is what Undo
 * sends and `after` what Redo sends; both are empty when the photo already had the pasted settings (addition, 2026-10-16).
 */
export interface PastedPhoto {
  id: PhotoId;
  before: SettingsPatch;
  after: SettingsPatch;
}

/** The answer of `paste_settings` (addition, 2026-10-16). */
export interface PasteReport {
  /** The photos that took the settings, in the order they were asked for. */
  pasted: PastedPhoto[];
  /** The photos left out: still developing, failed or not in the session. */
  skipped: PhotoId[];
}

/** Which `Auto` fields are still automatic; the UI shows the AUTO tag for these. */
export interface AutoFlags {
  exposure: boolean;
  contrast: boolean;
  highlights: boolean;
  shadows: boolean;
  black: boolean;
  balance_shadows: boolean;
  balance_highlights: boolean;
  saturation: boolean;
  light_source: boolean;
  white_balance: boolean;
  sharpen: boolean;
  noise_reduction: boolean;
}

/** Light classes of the illuminant estimate (`docs/02`). */
export type LightClass = "daylight" | "tungsten" | "fluorescent" | "led";

/** Which Auto made a photo's automatic values: Yellowmoon's own (`rule`) or Apple Auto's answer (`apple`, `docs/19`). */
export type AutoSource = "rule" | "apple";

/** Why Apple Auto is not available on this computer (`docs/19` section 1), in the order the conditions are checked. */
export type AppleAutoReason =
  | "not_a_mac"
  | "intel_mac"
  | "built_without_support"
  | "macos_too_old"
  | "device_not_eligible"
  | "apple_intelligence_off"
  | "model_not_ready"
  | "unavailable"
  | "no_image_input";

/** Why one request to Apple's model failed (`docs/19` sections 6 and 8). A failed photo uses Yellowmoon's own Auto. */
export type AppleAutoFailure =
  | "unavailable"
  | "rate_limited"
  | "timeout"
  | "refused"
  | "guardrail"
  | "unsupported_language"
  | "context_too_large"
  | "assets_unavailable"
  | "concurrent_requests"
  | "bad_answer"
  | "other";

/** Whether Apple's model read the default instructions or the person's own (`docs/19` section 2). */
export type AppleAutoInstructions = "default" | "edited";

/** Which model answers: "AFM 3 Core" on macOS "27.0.1 (27A404)", with which instructions (the first 12 hex digits of their SHA-256). */
export interface AppleAutoInfo {
  model: string;
  os: string;
  instructions: AppleAutoInstructions;
  instructions_sha256: string;
}

/** The colour Apple Auto gives the shadows (`docs/19` section 4). */
export type AppleShadowColor = "none" | "teal" | "blue" | "green" | "amber" | "magenta";

/** The colour Apple Auto gives the highlights. */
export type AppleHighlightColor = "none" | "warm" | "gold" | "cool" | "pink";

/**
 * The steps of an answer (`docs/19` sections 4 and 5), each null when the model left the field empty (nothing is set for that
 * control): exposure −3…+3; highlights, shadows, blacks, contrast, white (the model's warmth), tint and saturation −2…+2, 0 for keep;
 * the colours of the shadows and the highlights with their strengths 1 (subtle) to 3 (strong), a strength only with a colour other
 * than none.
 */
export interface AppleAutoSteps {
  exposure: number | null;
  highlights: number | null;
  shadows: number | null;
  blacks: number | null;
  contrast: number | null;
  white: number | null;
  tint: number | null;
  saturation: number | null;
  shadow_color: AppleShadowColor | null;
  shadow_color_strength: number | null;
  highlight_color: AppleHighlightColor | null;
  highlight_color_strength: number | null;
}

/** One answer of Apple's model. `scene` is a few words about the photo, for reports only; `plan` the look it gives the photo (the Automatic card shows it). */
export interface AppleAutoAnswer {
  steps: AppleAutoSteps;
  scene: string;
  plan: string;
}

/**
 * Where a photo stands with Apple Auto (`docs/19` sections 6 and 12). `off`: the preference is off, or the app has no advisor (the
 * browser build); `unavailable`: not on this computer; `pending`: asked, or waiting for its turn (Auto is the rule until the answer);
 * `answered`; `failed` (Auto is the rule). The Automatic card shows its Contrast and Saturation rows and its line while the state is
 * `pending`, `answered` or `failed`.
 */
export type AppleAuto =
  | { state: "off" }
  | { state: "unavailable"; reason: AppleAutoReason }
  | { state: "pending" }
  | { state: "answered"; answer: AppleAutoAnswer; info: AppleAutoInfo }
  | { state: "failed"; kind: AppleAutoFailure; message: string; info: AppleAutoInfo };

/** How a run decides whether Apple Auto is used; the app always follows the preference (`from_preference`). */
export type AppleAutoMode = "from_preference" | "on" | "off";

/**
 * `apple_auto_status()` (addition, `docs/19` sections 1, 2 and 10): whether Apple Auto is available on this computer, read again for
 * the answer, and whether it acts. Preferences shows its row "Auto with Apple Intelligence" only while `available`.
 */
export interface AppleAutoStatus {
  available: boolean;
  /** Why it is not available; null when it is, and when the app has no advisor (the browser build). */
  reason: AppleAutoReason | null;
  info: AppleAutoInfo | null;
  /** The `apple_auto` preference. */
  preference: boolean;
  mode: AppleAutoMode;
  /** Apple Auto acts for the session now: available, and the preference on. */
  active: boolean;
  /** The prompt's and the table's versions (`docs/19` sections 4 and 5). */
  prompt: number;
  table: number;
  /** Which instructions the model reads: the default or the person's own, and the first 12 hex digits of their SHA-256. */
  instructions: AppleAutoInstructions;
  instructions_sha256: string;
  /** The person's own instructions (`Preferences.apple_auto_instructions`), "" for the default. */
  own_instructions: string;
  /** The default instructions, for Preferences' text field and its Reset to default. */
  default_instructions: string;
  /** The longest instructions the model is given, in characters (2000). */
  instructions_max_chars: number;
}

/**
 * Per-photo automatic estimates, computed once and recomputed by "Re-run" (`docs/01`
 * `AutoEstimates`; the shape of the `auto` object in `docs/10` `--json` output). With an answer
 * from Apple Auto, the exposure, the Kelvin, the contrast and the saturation are Apple's
 * (`docs/19` section 5).
 */
export interface AutoEstimates {
  exposure_ev: number;
  /** The automatic white lift in stops, 0 for a photo with nothing above +1 stop to lift (the Automatic card's White row). */
  white_ev: number;
  kelvin: number;
  tint: number;
  /** What `contrast: Auto` resolves to: 0, or Apple Auto's contrast (the Contrast slider's detent). */
  contrast: number;
  /** What `saturation: Auto` resolves to: 0, or Apple Auto's saturation (the Saturation slider's detent). */
  saturation: number;
  /** What `highlights: Auto` resolves to: 0, or Apple Auto's highlights (the Highlights slider's detent). */
  highlights: number;
  /** What `shadows: Auto` resolves to: 0, or Apple Auto's shadows. */
  shadows: number;
  /** What `black: Auto` resolves to: 0, or Apple Auto's black level. */
  black: number;
  /** What the shadows wheel's Auto resolves to: `[0, 0]`, or Apple Auto's shadow colour. */
  balance_shadows: ColorBalancePoint;
  /** What the highlights wheel's Auto resolves to: `[0, 0]`, or Apple Auto's highlight colour. */
  balance_highlights: ColorBalancePoint;
  /** Which Auto made these values. */
  source: AutoSource;
  /** Where the photo stands with Apple Auto. */
  apple: AppleAuto;
  /** Apple Auto's steps in effect (the photo's answer, or its burst's), null when Auto is the rule. */
  apple_steps: AppleAutoSteps | null;
  light_class: LightClass;
  /** 0–1. */
  light_confidence: number;
  /** Poisson–Gaussian noise model, `variance = a · signal + b`. */
  noise: { a: number; b: number };
  /** What `sharpen: Auto` resolves to. */
  sharpen: { amount: number; radius_px: number };
  /** What `noise_reduction: Auto` resolves to. */
  noise_reduction: { luminance: number; color: number; detail: number };
}

/** `DevelopSettings` with every `Auto` replaced by the number it resolves to. */
export type ResolvedValues = Omit<
  DevelopSettings,
  "exposure" | "contrast" | "highlights" | "shadows" | "black" | "white_balance" | "color_balance" | "saturation" | "sharpen" | "noise_reduction"
> & {
  exposure_ev: number;
  /** The contrast in effect: the manual value, or the photo's Auto contrast. */
  contrast: number;
  /** Highlights, shadows and black in effect: the manual value, or the photo's Auto value. */
  highlights: number;
  shadows: number;
  black: number;
  /** The wheels' points in effect. */
  color_balance: ColorBalancePoints;
  /** The saturation in effect: the manual value, or the photo's Auto saturation. */
  saturation: number;
  /** The white lift in effect in stops: the automatic amount plus `white_offset_ev`, never below 0. */
  white_ev: number;
  kelvin: number;
  tint: number;
  sharpen: { amount: number; radius_px: number };
  noise_reduction: { luminance: number; color: number; detail: number };
};

/** What `get_settings` and `update_settings` return: the panel's whole state for one photo. */
export interface ResolvedSettings {
  photo_id: PhotoId;
  /** The stored settings, with `Auto` kept as `Auto`. */
  settings: DevelopSettings;
  /** The same settings with every `Auto` resolved against `estimates`. */
  values: ResolvedValues;
  auto: AutoFlags;
  estimates: AutoEstimates;
}

/** "Apply to" on the Look card, and the scope of `apply_look`. */
export type ApplyScope = "this" | "picked" | "all";

// ---------------------------------------------------------------------------
// Looks (docs/08 §Look model)
// ---------------------------------------------------------------------------

/** Where a look comes from. ARRI looks keep their numbered group ("3000 Film"). */
export type LookGroup =
  | { kind: "built_in" }
  | { kind: "arri"; no: number; name: string }
  | { kind: "user" };

/** What a look's LUT expects as input (`docs/08`). */
export type LookInput = "logc4" | "logc3_ei800" | "logc3_to_rec709";

/**
 * The library a look belongs to: the Look tab shows one section per library, in this order (`docs/08` §Look model). The core works it
 * out from the group, the input and the file format, so the app and `ym looks list` agree.
 */
export type LookLibrary = "built_in" | "arri_logc4" | "arri_logc3" | "arri_logc3_to_rec709" | "arri_alf2" | "user";

/** The kind of file an imported look is kept as (`aml` is ARRI's ALF2). */
export type LookFormat = "cube" | "alf4" | "alf4c" | "aml";

/** One entry of the look library. */
export interface Look {
  id: LookId;
  name: string;
  group: LookGroup;
  input: LookInput;
  /** The section the look is listed in. */
  library: LookLibrary;
  /** The kind of file it is kept as; null for a built-in look. */
  format: LookFormat | null;
  /** Points per axis of its 3D table (65, 33, …); null for a built-in look. */
  lut_size: number | null;
  /** The user's default look (a preference, shown with the DEFAULT tag). */
  is_default: boolean;
  /** The look's file is gone or its hash changed: shown disabled, "File missing or changed". */
  missing: boolean;
}

/** What `looks_import` reports (addition: `docs/01` names the command, not the result). */
export interface LooksImportResult {
  /** Looks added by this import; empty when everything was already there. */
  added: Look[];
  /** Display LUTs added (ARRI's display rendering, not looks). */
  display_luts_added: number;
  /** Files whose hash matched a look or display LUT that was already imported (a re-import adds nothing, `docs/08`). */
  unchanged: number;
}

// ---------------------------------------------------------------------------
// The ARRI import wizard (docs/08 §Wizard, §Recognition rules, §Any other .cube). Additions to
// `docs/01`: the wizard needs to ask what a file is before anything is copied.
// ---------------------------------------------------------------------------

/**
 * The five free packages ARRI publishes (`docs/08` §What users can import). The ids are the ones
 * of `data/looks/arri-packages.json`, except that the two look libraries that also come as ARRI Look Files
 * (the LogC4 one as `.alf4`, the LogC3 to Rec.709 one as `.aml`) are the same package here.
 */
export type PackageId =
  | "look_library_logc4"
  | "display_luts_logc4"
  | "look_library_logc3"
  | "look_library_logc3_to_rec709"
  | "display_luts_logc3";

/** What a chosen source is: a zip, an unpacked folder, or a single `.cube` or look file. */
export type SourceKind = "zip" | "folder" | "file";

/**
 * What the recogniser made of one source (`looks_recognise`):
 *  - `package`: one of the five packages (an exact hash match, or by names and headers). `contents`
 *    reads "87 .cube · 65-point", or "87 .alf4 · 33-point" for the same looks as ARRI Look Files (the
 *    LogC4 library as `.alf4`, the LogC3 to Rec.709 one as `.aml`); `note` carries a remark that does not
 *    stop the import (more files than expected).
 *  - `look_files`: ARRI Look Files (`.alf4`, `.alf4c`, `.aml`) that are not part of a package: the person's
 *    own looks. Each file says what it expects, so nothing is asked. `input` is what they all expect, or
 *    null when they differ; `note` says how they are imported (and that an `.alf4c`'s display transforms are
 *    not used).
 *  - `needs_answer`: `.cube` files that match no package. The person says what they expect as input
 *    (`looks_import` takes it as `input`); `note` says why, and `suggestion` is what the names suggest.
 *  - `unusable`: nothing can be imported. `reason` is the plain explanation, for example "Its checksum does
 *    not match, so the file is damaged. Download it again." When the
 *    package is known but damaged (`package_id` set), the reason says what to download again.
 *  - `missing`: the file is not there (moved or deleted since it arrived).
 */
export type Recognition =
  | {
      kind: "package";
      package_id: PackageId;
      /** "Look Library · LogC4". */
      title: string;
      /** "Look Library · LogC4 · log to log". */
      what: string;
      /** "87 .cube · 65-point". */
      contents: string;
      /** Files that will be copied. */
      file_count: number;
      /** The person's own look files (`.alf4`, `.alf4c`, `.aml`) in the same source, copied too as their own looks; 0 or absent when there are none. */
      look_file_count?: number;
      note: string | null;
    }
  | {
      kind: "look_files";
      /** "ARRI Look Files · LogC4". */
      what: string;
      /** "3 .alf4 · 33-point". */
      contents: string;
      /** Files that will be copied. */
      file_count: number;
      /** What every file expects as input; null when they differ. */
      input: LookInput | null;
      note: string | null;
    }
  | {
      kind: "needs_answer";
      file_count: number;
      /** The LUT size when every file has the same one. */
      lut_size: number | null;
      suggestion: LookInput | null;
      /** The person's own look files in the same source, copied too; 0 or absent when there are none. */
      look_file_count?: number;
      note: string;
    }
  | { kind: "unusable"; reason: string; package_id: PackageId | null }
  | { kind: "missing"; reason: string };

/** `looks_recognise(path)`: one source and what it is. Nothing is copied. */
export interface SourceRecognition {
  path: string;
  /** File or folder name: "ARRI_LookLibrary_LogC4.zip". */
  name: string;
  source: SourceKind;
  /** Size on disk; null for a folder or a missing file. */
  size_bytes: number | null;
  recognition: Recognition;
}

/** `looks_import_progress`: files copied so far from one source. */
export interface LooksImportProgress {
  path: string;
  done: number;
  total: number;
  /** The file just copied, when known. */
  file: string | null;
}

/** An imported display LUT (ARRI's display rendering). Looks are `Look`s; these are chosen in Preferences. */
export interface DisplayLut {
  /** "arri-display:logc4:rec709-g24". */
  id: string;
  name: string;
  input: "logc4" | "logc3_ei800";
  /** "rec709-g24", "p3-d65", "rec2020", "hlg", "pq". */
  target: string;
}

/** `looks_watch_downloads(enabled)`: whether the watcher is running, and why not when it was asked to. */
export interface WatchStatus {
  watching: boolean;
  /** Plain words, when macOS did not allow access to the Downloads folder. */
  problem: string | null;
}

/**
 * `download_found`: a zip whose name matches one of the packages has landed in Downloads and its size has been
 * stable for 2 s (`docs/08` §Wizard step 2).
 */
export interface DownloadFound {
  path: string;
  file_name: string;
  size_bytes: number;
  /** The package the name suggests, if exactly one. */
  package_hint: PackageId | null;
}

// ---------------------------------------------------------------------------
// Viewport
// ---------------------------------------------------------------------------

/** The photo cell's rectangle in CSS pixels (`viewport_set_rect`, `docs/01` §Native viewport). */
export interface ViewportRect {
  x: number;
  y: number;
  width: number;
  height: number;
  /** Device pixels per CSS pixel. */
  scale: number;
}

/** Pointer input over the photo, forwarded to the core. Coordinates are CSS px in the cell. */
export type ViewportInput =
  | { kind: "zoom"; factor: number; x: number; y: number }
  | { kind: "pan"; dx: number; dy: number }
  | { kind: "toggle_zoom"; x: number; y: number }
  | { kind: "pick_white"; x: number; y: number };

/** Fit to the cell, or a fixed zoom (50, 100, 200 %). */
export type ViewportMode = { kind: "fit" } | { kind: "percent"; value: number };

/**
 * `chroma_scatter(photo_id)`: the photo's colours on the colour warp's plane, at the warp's input (after C9, before the pins), for the
 * faint scatter behind the plane. A flat list `[a0, b0, a1, b1, …]` of at most 2048 points.
 */
export interface ChromaScatter {
  points: number[];
}

/** `sample_chroma(x, y)`: the chroma of the pixel under a point of the photo cell, at the warp's input (Pick colour). */
export interface ChromaSample {
  a: number;
  b: number;
}

/** Display values under the pointer, 0–255 per channel (`sample_pixel`). */
export interface PixelSample {
  r: number;
  g: number;
  b: number;
}

/** Which backend draws the photo (`viewport_info`): see `ViewportInfo`. */
export type ViewportBackendKind = "native" | "fallback";

/**
 * `viewport_info` (addition): which backend draws the photo, so the photo cell knows whether to stay transparent
 * (`native`: a wgpu surface behind the webview) or to show the `viewport_frame` images (`fallback`).
 */
export interface ViewportInfo {
  backend: ViewportBackendKind;
}

/**
 * `viewport_frame` (addition, fallback viewport only): a new frame of the photo cell is ready. `url` is a PNG
 * (`ym-frame://localhost/<launch>/frame-<seq>.png`) of the cell's size in device pixels, letterbox included, in Display P3.
 * An empty `url` with a size of 0 × 0 says there is no picture any more: take the image away.
 */
export interface ViewportFrame {
  /** Counts up with every frame; also in `url`, so a changed frame is a new address. */
  seq: number;
  /** Width in device pixels. */
  width: number;
  /** Height in device pixels. */
  height: number;
  /** False when the frame was made from a coarser image while sharper tiles render; a sharp frame follows. */
  sharp: boolean;
  /** Where the PNG is; empty when there is no picture. */
  url: string;
}

/**
 * `settings_changed` (addition): the core changed a photo's settings by itself: after a `viewport_input` Pick white
 * click, and once for each photo a `paste_settings` changed. Ask `get_settings` again (the webview skips the photos it is
 * pasting onto: the paste's answer says what changed).
 */
export interface SettingsChanged {
  photo_id: PhotoId;
}

/** `export_busy` (addition): an export is queued or running. "Update ready" waits while this is true. */
export interface ExportBusy {
  busy: boolean;
}

/**
 * Per-channel histogram of the current render, for the Light tab and the Curve plot
 * (addition: `docs/01` lists no command for it; see `Api.histogram`). Each array holds the
 * same number of bins from black (first) to white (last), normalised so the tallest bin of
 * any channel is 1.
 */
export interface HistogramData {
  red: number[];
  green: number[];
  blue: number[];
}

/** Colour spaces the viewport and the exports can use. */
export type ColorSpace = "srgb" | "display_p3" | "rec2020";

/** Demosaic path of the last render (header status "Neural · 42 ms · Display P3"). */
export type RenderPath = "neural" | "classical";

/** `render_stats` event payload. */
export interface RenderStats {
  ms: number;
  path: RenderPath;
  space: ColorSpace;
  /**
   * The zoom the last viewport frame was drawn at, in percent (device pixels per pixel of the final image, times 100), not rounded.
   * The core's own, which knows the lens crop, the straighten angle and the crop that the screen can only guess. `null` when the
   * sender has none (the mock): the screen then estimates it from the photo and the cell (`fitZoomExact`).
   */
  zoom_percent: number | null;
  /** The zoom at which the whole final image fits the cell, in percent and not rounded; `null` as `zoom_percent`. */
  fit_percent: number | null;
}

// ---------------------------------------------------------------------------
// Export (docs/09 §Options)
// ---------------------------------------------------------------------------

export type ExportFormat = "jpeg" | "tiff" | "png" | "dng";
export type TiffCompression = "none" | "lzw" | "zip";
export type BitDepth = 8 | 16;
/** What to do when the target file exists. */
export type IfExists = "skip" | "replace" | "keep_both";
/** Export size: the full frame, or the long edge in pixels (1000 to full, 20 px steps). */
export type ExportSize = { kind: "full" } | { kind: "long_edge"; px: number };

/**
 * The options of `docs/09`. Defaults: JPEG, quality 92, 16-bit, LZW, sRGB, full size,
 * `{name}_{look}`, keep both. Colour space and size don't apply to DNG.
 */
export interface ExportOptions {
  format: ExportFormat;
  /** JPEG quality, 50–100. */
  jpeg_quality: number;
  /** TIFF and PNG share this preference. */
  bit_depth: BitDepth;
  tiff_compression: TiffCompression;
  color_space: ColorSpace;
  size: ExportSize;
  /** Destination folder; null means the default (`SessionState.default_export_folder`). */
  destination: string | null;
  /** Template with `{name}`, `{look}`, `{date}`, `{seq}`, `{camera}`. */
  name_template: string;
  if_exists: IfExists;
}

/** What to export and how. Settings are snapshotted by the core when the job is queued. */
export interface ExportJob {
  /** This photo, the picked photos, or all. */
  scope: ApplyScope;
  options: ExportOptions;
}

/** `export_estimate(job)`: the size and time shown in the Export panel ("≈ 120 MB"). */
export interface ExportEstimate {
  bytes: number;
  seconds: number;
  /**
   * Photos in the job with adjustments beyond the look. The command line preview cannot carry
   * them on one line, so the panel says "N photos have adjustments this command doesn't include."
   * (addition, `docs/09` §Options).
   */
  adjusted_photos: number;
  /**
   * A DNG of this job is close to, not the same as, the picture in the app: a photo in the job uses a look
   * with its own display rendering (LogC3 to Rec.709) at an intensity above 0, or ARRI's display rendering is
   * chosen. The panel then adds one sentence to the DNG note (addition, `docs/09` §What a DNG carries).
   */
  dng_approximate: boolean;
}

/** `export_progress` event payload. */
export interface ExportProgress {
  job_id: ExportJobId;
  /** Output file name now being written. */
  file: string;
  /** Files finished. */
  done: number;
  total: number;
  /** Progress of `file`, 0–100 (addition: the panel's "45%" per-file tag). */
  file_percent: number | null;
  /**
   * Plain-words error for `file`, when it failed. A failed file counts in `done`, so the event that
   * reports it names that file and carries the new count; the queue goes on with the next one.
   */
  error: string | null;
  /** `file` already existed and "If it exists" is Skip: it counts in `done` but was not written (addition). */
  skipped: boolean;
}

// ---------------------------------------------------------------------------
// Preferences (docs/01 §Persistence)
// ---------------------------------------------------------------------------

/** Display rendering: our ACES 2 transform, or an imported ARRI display LUT. */
export type DisplayRendering = { kind: "aces2" } | { kind: "arri_display"; id: string };

/** Folders remembered between runs. */
export interface LastUsedFolders {
  open: string | null;
  export: string | null;
  looks_import: string | null;
}

/** Everything that persists besides imported looks. The list is exhaustive (hard rule 4). */
export interface Preferences {
  default_look: LookId;
  display_rendering: DisplayRendering;
  /** Export defaults: format, quality, bits, compression, colour space, size, destination, name template, if-exists. */
  export_defaults: ExportOptions;
  remove_location_from_exports: boolean;
  check_for_updates_automatically: boolean;
  watch_downloads_during_import: boolean;
  last_used_folders: LastUsedFolders;
  /** Auto with Apple Intelligence (`docs/19` section 2): on by default; it acts only where Apple Auto is available. */
  apple_auto: boolean;
  /**
   * Instructions for Apple Intelligence (`docs/19` section 2): the person's own instructions for Apple Auto's model, at most 2000
   * characters; "" (the default) means `AppleAutoStatus.default_instructions`.
   */
  apple_auto_instructions: string;
}

/** A partial update of the preferences. */
export type PreferencesPatch = Partial<Preferences>;

// ---------------------------------------------------------------------------
// Events and errors
// ---------------------------------------------------------------------------

/** `photo_progress { id, stage, done }`: a stage of a photo's development started or finished. */
export interface PhotoProgress {
  id: PhotoId;
  stage: PhotoStage;
  /** True when `stage` has finished. A photo is ready when `thumbnail` is done. */
  done: boolean;
  /** Plain-words reason when the photo can't be developed (it becomes `skipped`). */
  error: string | null;
}

/**
 * `thumbnail_ready { id, url }`: the photo's thumbnail, a PNG of at most 240 px on its long edge, is at `url`. The picture does not
 * cross the bridge (as an array of byte values it was four times its size in JSON, 125 MB for 500 photos): the shell serves it on the
 * `ym-frame` scheme and an `<img>` loads it. A new thumbnail of the same photo (a look, a crop or a turn was applied) has a new
 * address, and so does the same thumbnail in the next launch of the app (`<launch>`). `url` is any address an `<img>` can load: the mock makes `blob:` ones.
 */
export interface ThumbnailReady {
  id: PhotoId;
  /** Where the PNG is (`ym-frame://localhost/<launch>/thumb-<id>-<version>.png`, `http://ym-frame.localhost/…` on Windows; `<launch>` is the app's
   * start time in hex milliseconds, a hyphen and a count). */
  url: string;
}

/** Updater phases (`docs/12` §In the app). */
export type UpdatePhase = UpdateState["phase"];

/**
 * `update_state { phase, version?, percent?, message? }`, as a union that says which fields each
 * phase carries. On the wire this is the flat object of `docs/01` with the optional fields
 * absent where they don't apply.
 */
export type UpdateState =
  | { phase: "idle" }
  | { phase: "available"; version: string }
  /** `percent` is 0–100; absent when the server sends no content length. */
  | { phase: "downloading"; version: string; percent?: number }
  | { phase: "ready"; version: string }
  /** `message` is the reason, shown on hover. */
  | { phase: "error"; message: string };

/** Where an error came from. */
export type ErrorScope =
  | "open"
  | "decode"
  | "render"
  | "export"
  | "looks"
  | "preferences"
  | "viewport"
  | "app";

/**
 * An error that reaches the UI: a plain message that also says what to do next
 * (`docs/01` §Errors and logging). Commands reject with it; the `error` event carries it.
 */
export interface AppError {
  scope: ErrorScope;
  message: string;
}

/** `install_cli_link` result: where the `ym` link was created (addition). */
export interface CliLinkResult {
  path: string;
}

/** What `open_dialog` offers. */
export type OpenKind = "files" | "folder";

/** What the app knows about itself, for About (addition). */
export interface AppInfo {
  /** The app and CLI version: "0.1.0". */
  version: string;
  /** The pipeline version written into exports (`docs/01` §Versions written into outputs). */
  pipeline_version: string;
}

/** `cli_link_status`: whether "Install Command-Line Tool…" has been done (addition, Preferences). */
export interface CliLinkStatus {
  installed: boolean;
  /** Where the `ym` link is, when installed. */
  path: string | null;
}

/**
 * The items of the native menu bar (`docs/11` §Menus) that need the webview. The core sends `menu_action` with one
 * of them; the items it handles itself (Show Looks Folder, the Help links) are not listed.
 */
export type MenuAction =
  | "about"
  | "preferences"
  | "check_for_updates"
  | "install_cli"
  | "open_files"
  | "open_folder"
  | "export"
  | "close_session"
  | "undo"
  | "redo"
  | "pick"
  | "pick_all"
  | "clear_picks"
  | "before_after"
  | "fit"
  | "zoom_100"
  | "zoom_in"
  | "zoom_out"
  | "next_photo"
  | "previous_photo"
  | "import_arri_looks"
  | "import_cube"
  | "make_default_look"
  | "copy_settings"
  | "paste_settings"
  | "paste_settings_to_picked";

/**
 * `set_menu_state(state)` (addition, 2026-10-16): which of the Edit menu's settings items are available. The webview knows (a developed
 * current photo, something copied, photos picked) and the native menu does not.
 */
export interface MenuState {
  /** Copy Settings: the current photo is developed. */
  copy_settings: boolean;
  /** Paste Settings: something is copied and the current photo is developed. */
  paste_settings: boolean;
  /** Paste Settings to Picked: something is copied and a photo is picked. */
  paste_settings_to_picked: boolean;
}

/** `menu_action` event payload. */
export interface MenuActionEvent {
  action: MenuAction;
}

/** Event name → payload type, for the typed `on()`. `looks_changed` carries no data. */
export interface ApiEvents {
  photo_progress: PhotoProgress;
  thumbnail_ready: ThumbnailReady;
  render_stats: RenderStats;
  export_progress: ExportProgress;
  looks_changed: null;
  update_state: UpdateState;
  error: AppError;
  /** Files of one source copied so far during `looks_import` (the wizard's Import step). */
  looks_import_progress: LooksImportProgress;
  /** The Downloads watcher found a package zip (`looks_watch_downloads`). */
  download_found: DownloadFound;
  /** A menu item that needs the webview was chosen. */
  menu_action: MenuActionEvent;
  /** The fallback viewport has a new frame of the photo cell, or none any more. */
  viewport_frame: ViewportFrame;
  /** The core changed a photo's settings by itself (a Pick white click, a paste); ask for them again. */
  settings_changed: SettingsChanged;
  /** An export is queued or running, or the queue has just gone quiet. */
  export_busy: ExportBusy;
}

/** Names of the events the core emits. */
export type ApiEventName = keyof ApiEvents;
