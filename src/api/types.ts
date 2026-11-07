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
 * is looser in places (`NodePatch` allows `null` in every field, `UpdateState.percent` may
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

/**
 * Stages of the background job that develops a freshly opened photo (`docs/01` §Session), and `apple`: the answer of the provider that
 * grades (Apple Intelligence, Claude Code or Antigravity; `docs/19` section 6, the name kept, `docs/22` D3), started while the photo
 * waits for it and done when it arrived, failed or the provider stopped acting on the photo. The
 * photo is ready (it can be edited) before `apple` is done; that stage changes `PhotoSummary.apple_pending`, never the status.
 */
export type PhotoStage = "probe" | "decode" | "estimate" | "thumbnail" | "apple";

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
  /** Apple Auto acts on the photo and has not answered for it yet: its Auto values are Yellowmoon's own until the answer. */
  apple_pending: boolean;
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
 * locked pin holds the colours within its range in place (its target is its source, its weights, pivot, exposure and saturation the defaults).
 */
export interface ColorWarpPin {
  source: ChromaPoint;
  /** Where the colour goes; equal to `source` for a pin that has not moved and for a locked pin. */
  target: ChromaPoint;
  locked: boolean;
  /** How far the pin reaches across the plane, 0.01–0.25, default 0.02 (the card shows it ×100). */
  range: number;
  /** How much of the move the shadows take, 0–1, default 1. */
  shadows: number;
  /** How much of the move the highlights take, 0–1, default 1. */
  highlights: number;
  /** Where the shadows end and the highlights begin, in stops from middle grey, −6…6, default 0. */
  pivot_ev: number;
  /** A change of brightness for the pin's colours, in stops, −2…2, default 0. */
  exposure_ev: number;
  /** A change of saturation for the pin's colours, −1…1 (shown as −100…+100 %), default 0: −1 turns them grey, +1 doubles their distance from neutral. */
  saturation: number;
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

/** RAW › Range, each 0–1: Highlights 0.6 and Low light 0.5 by default (`docs/02` C7). */
export interface Rolloff {
  /**
   * The highlight range: where the camera's clipping point lands, from where it was recorded (0) to white (1, the clip of an
   * ALEXA at EI 800). Mid grey and everything up to two stops under the clip stay where they are.
   */
  highlights: number;
  /** The low-light toe: how much the noise near black is softened. */
  low_light: number;
}

/** The Lens tab: which lens (`docs/21` §2). The ids are those of `data/lenses/` and of `ym render --lens`. */
export type LensKind = "default" | "classic" | "vintage" | "petzval" | "triplet" | "sonnar" | "detuned";

/**
 * The Lens tab (`docs/21` §2): a lens and three amounts, each 0–100, step 1. All three at 0 is no lens, whatever the lens is. Choosing a
 * lens does not move the amounts by itself: the card sends `kind` with the lens's `start_flares` and `start_edges` (`LensInfo`) in one
 * patch. A patch may carry only the parts it changes.
 */
export interface LensSettings {
  kind: LensKind;
  /** Glow, veiling glare, ghosts and starbursts. */
  flares: number;
  /** Edge softness and darkening, colour fringes and distortion. */
  edges: number;
  /** An anamorphic attachment: taller bokeh with the lens's own swirl and cat's-eye shapes, horizontal streaks, the anamorphic bend. */
  anamorphic: number;
}

/** One lens of the Lens card's list: `lens_list()` answers the seven, in list order (`docs/21` §1 and §10). */
export interface LensInfo {
  kind: LensKind;
  /** The name shown in the list. */
  name: string;
  /** The line under the name, and in the card's note. */
  line: string;
  /** Flares (0–100) the card moves to when this lens is chosen. */
  start_flares: number;
  /** Edges (0–100) the card moves to when this lens is chosen. */
  start_edges: number;
  /** The data version, raised whenever the lens's files change. */
  version: number;
  /** The design the lens was computed from: a patent number or a publication. */
  design_source: string;
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
  /** The rect and angle came from Auto crop (`docs/20-auto-crop.md` §2); absent is false. */
  auto?: boolean;
}

/** Auto crop's straightener: within 0.2° of level, levelled by `angle_deg`, or no answer (`docs/20-auto-crop.md` §3). */
export type StraightenKind = "level" | "rotate" | "unsure";

/** Where Auto crop's subject came from: the picture alone (the crop model), or the camera's focus point (no longer answered). */
export type SubjectSource = "camera" | "picture";

/** One frame Auto crop proposes: a rect of the canvas, as `Crop.rect`, and its score (its overlap with the crop model's crop, 0 to 1). */
export interface CropFrame {
  rect: NormRect;
  score: number;
}

/** `suggest_crop(photo_id, aspect, orientation)`: Auto crop's proposal (`docs/20-auto-crop.md` §3). */
export interface CropSuggestion {
  straighten: StraightenKind;
  /** The straighten angle the frames are made for: 0 unless `straighten` is `rotate`. */
  angle_deg: number;
  /** Best first, at most 3, never empty. */
  frames: CropFrame[];
  /** The first frame is the largest frame of the aspect at that angle: the whole photo is kept. */
  keep: boolean;
  subject: SubjectSource;
}

/** Orientation applied after the camera's EXIF orientation, in degrees. */
export type Orientation = 0 | 90 | 180 | 270;

// ---------------------------------------------------------------------------
// The settings are a graph of nodes (docs/23 §3, §9)
// ---------------------------------------------------------------------------

/**
 * A node's id: `"input"`, `"output"` or a whole number from 1 (Main is 1). Stable within a photo; links, selection, thumbnails and undo
 * use it. (The generated bindings give every node struct the whole union for `id`, as `specta` cannot narrow it, so the interfaces
 * below do too; `kind` is what tells the nodes apart.)
 */
export type NodeId = "input" | "output" | number;

/** What a node is. */
export type NodeKind = "input" | "output" | "corrector" | "film";

/** A port of a node: a picture (the LogC4 picture) or a key (a mask). */
export type Port = "picture" | "key";

/** A card's place on the canvas, in grid units (`docs/23` section 3.2). */
export type NodePos = [x: number, y: number];

/** One end of a link: a node and one of its ports. */
export interface NodeRef {
  node: NodeId;
  port: Port;
}

/** A link from an output port to an input port (`docs/23` section 3.4). */
export interface NodeLink {
  from: NodeRef;
  to: NodeRef;
}

/** A corrector's trims: an exposure trim in stops and a temperature and tint trim (`docs/23` section 1.3). */
export interface NodeTrim {
  /** Stops, −4…4. */
  exposure_ev: number;
  /** −100…100. */
  temperature: number;
  /** −100…100. */
  tint: number;
}

/**
 * A corrector's grade: the fields the settings had before they were nodes, with the same names and types (`docs/23` section 3.2). Auto
 * belongs to Main: in any other corrector Auto reads as a manual 0.
 */
export interface NodeGrade {
  /** Light › Exposure › Contrast (and Look › Tone): `Auto` (the default) or `Manual(value)` in −100…100. */
  contrast: PercentSetting;
  /** −100…100, default 0. */
  brightness: number;
  /** Light › High dynamic range, each −100…100: Highlights and Shadows are `Auto` (the default: 0, or Apple Auto's value) or a value. */
  highlights: PercentSetting;
  shadows: PercentSetting;
  white: number;
  /** Light › High dynamic range › Black: `Auto` (the default: 0, or Apple Auto's black level) or a value in −100…100. */
  black: PercentSetting;
  color_balance: ColorBalance;
  /** Colour › Colour balance › Saturation (and Look › Tone): `Auto` (the default) or `Manual(value)` in −100…100. */
  saturation: PercentSetting;
  /** Colour › Colour warp: the pins of stage C9b, `{ pins: [] }` by default. */
  color_warp: ColorWarp;
  curve: Curves;
}

/** Whether a mask row adds to the node's mask or takes away from it. */
export type MaskMode = "add" | "subtract";

/** One stroke of a Brush mask: its size (0.001…0.25 of the long edge), hardness (0…1), whether it erases, and its points in the photo frame. */
export interface BrushStroke {
  size: number;
  hardness: number;
  erase: boolean;
  points: Array<[x: number, y: number]>;
}

/** What every mask row has (`docs/23` section 3.3); the kind's own fields follow. */
type MaskBase = {
  mode: MaskMode;
  /** This mask alone is inverted. */
  invert: boolean;
  /** Off (the eye) counts as absent. */
  enabled: boolean;
  /** 0…1. */
  opacity: number;
};

/** The fields Subject, People, Background, Depth, Object and Brush rows share. */
type MaskEdge = {
  /** 0…1. */
  feather: number;
  /** −1…1. */
  grow: number;
  /** Refine edges, 0…1. */
  refine: number;
};

/** One row of a node's masks: `kind`, the fields every row has, then the kind's own (`docs/23` section 3.3). */
export type MaskItem =
  | ({ kind: "object" } & MaskBase & MaskEdge & { points: Array<[x: number, y: number, positive: boolean]>; box: [x0: number, y0: number, x1: number, y1: number] | null })
  | ({ kind: "subject" } & MaskBase & MaskEdge)
  | ({ kind: "people" } & MaskBase & MaskEdge)
  | ({ kind: "background" } & MaskBase & MaskEdge & { level: number; softness: number })
  | ({ kind: "depth" } & MaskBase & MaskEdge & { near: number; far: number; softness: number })
  | ({ kind: "linear" } & MaskBase & { center: [x: number, y: number]; angle_deg: number; feather: number })
  | ({ kind: "radial" } & MaskBase & { center: [x: number, y: number]; radii: [rx: number, ry: number]; angle_deg: number; feather: number })
  | ({ kind: "brush" } & MaskBase & MaskEdge & { strokes: BrushStroke[] })
  | ({ kind: "color_range" } & MaskBase & { center: [a: number, b: number]; width: number; softness: number })
  | ({ kind: "brightness_range" } & MaskBase & { low: number; high: number; softness: number });

/** A node's masks: its Invert (`m = 1 − m`) and the rows, in the order they are drawn and combined, at most 64. */
export interface MaskSettings {
  invert: boolean;
  items: MaskItem[];
}

/** The Input: the raw, light and framing settings (`docs/23` section 3.2), under the names and types the settings always had. */
export interface InputNode {
  /** `"input"`. */
  id: NodeId;
  pos: NodePos;
  /** Light › Exposure: `Auto` or `Manual(ev)` in −4…+4, step 0.01. */
  exposure: AutoOr<{ ev: number }>;
  /** Colour › White balance, segmented control. */
  light_source: LightSource;
  /** Colour › White balance: `Auto` or `Manual { kelvin 2000–11000 step 50, tint −50…50 }`. */
  white_balance: AutoOr<{ kelvin: number; tint: number }>;
  /** Pick white (Colour › White balance): the sampled point as `[x, y]` in [0, 1]² image coordinates. */
  white_pick: [x: number, y: number] | null;
  rolloff: Rolloff;
  decode: DecodeOptions;
  /** Detail › Noise reduction: `Auto` or `{ luminance, color, detail }`, each 0–100. */
  noise_reduction: AutoOr<{ luminance: number; color: number; detail: number }>;
  /** Detail › Sharpening: `Auto` or `{ amount 0–300, radius_px 0.3–2.5 }`. */
  sharpen: AutoOr<{ amount: number; radius_px: number }>;
  /** The Lens tab: the lens and its Flares, Edges and Anamorphic, `{ kind: "default", flares: 0, edges: 0, anamorphic: 0 }` (no lens) by default. */
  lens: LensSettings;
  /** Crop (header); `null` when the photo isn't cropped. */
  crop: Crop | null;
  orientation: Orientation;
}

/** The Output: the look and the output sharpening (`docs/23` section 3.2). */
export interface OutputNode {
  /** `"output"`. */
  id: NodeId;
  pos: NodePos;
  /** Look tab. Defaults to the user's default look (Neutral out of the box). */
  look: LookId;
  /** 0–100 %, default 100. */
  look_intensity: number;
  /**
   * Detail › Sharpening › Output: the strength in percent of the sharpening applied to the picture at the size it is shown or written
   * (every viewport frame, thumbnail and JPEG, TIFF or PNG export; never a DNG). 0–200, step 5, default 100; a plain number, not Auto;
   * 0 is off. The owner's decision of 2026-10-13 (`docs/algorithms/04-output-sharpening.md`).
   */
  output_sharpening: number;
}

/** A corrector node, Main (id 1) included (`docs/23` section 3.2). */
export interface CorrectorNode {
  /** 1 to 2³¹ − 1; Main is 1. */
  id: NodeId;
  /** The card's name, 1 to 32 characters. */
  name: string;
  /** A bypassed node passes its picture through. */
  bypass: boolean;
  pos: NodePos;
  trim: NodeTrim;
  grade: NodeGrade;
  masks: MaskSettings;
}

/** A Film node's negative (`docs/24` section 4.5). */
export interface FilmStockSettings {
  /** The profile id of a camera film (`ym film list`). */
  stock: string;
  /** Stops, -4 to 4; 0 is the Input's exposure. */
  exposure_ev: number;
}

/** The enlarger's filtration, shifts from neutral in Kodak CC units, -50 to 50. */
export interface PrintLights {
  c: number;
  m: number;
  y: number;
}

/** A Film node's enlarger and paper. */
export interface PrintSettings {
  /** The paper's profile id; null is the stock's matching paper (a slide takes none). */
  paper: string | null;
  /** Stops, -3 to 3; 0 keeps the app's brightness. */
  exposure_ev: number;
  lights: PrintLights;
}

/** A Film node's halation, in percent (100 is the stock's own). */
export interface HalationSettings {
  /** 0 to 300. */
  amount: number;
  /** 50 to 200. */
  size: number;
}

/** A Film node's grain, in percent (100 is the stock's own). */
export interface GrainSettings {
  /** 0 to 200. */
  amount: number;
  /** 50 to 200. */
  size: number;
  /** Not in the interface; it keeps the grain a function of the settings. */
  seed: number;
}

/** A Film node: a colour negative, an enlarger, a print and a scan (`docs/24` section 4.5). */
export interface FilmNode {
  /** 2 to 2³¹ − 1; correctors and Film nodes share the numbers. */
  id: NodeId;
  /** The card's name, 1 to 32 characters. */
  name: string;
  /** A bypassed node passes its picture through. */
  bypass: boolean;
  pos: NodePos;
  film: FilmStockSettings;
  print: PrintSettings;
  halation: HalationSettings;
  grain: GrainSettings;
  /** How much of the film look is applied, 0 to 100 percent. */
  strength: number;
  masks: MaskSettings;
}

/** A node of the graph, told by its `kind`. */
export type NodeItem =
  | ({ kind: "input" } & InputNode)
  | ({ kind: "output" } & OutputNode)
  | ({ kind: "corrector" } & CorrectorNode)
  | ({ kind: "film" } & FilmNode);

/** The graph of a photo: its nodes, the Input first, the correctors by id, the Output last, and the links between them. */
export interface NodesSettings {
  items: NodeItem[];
  links: NodeLink[];
}

/**
 * The settings of a photo are the graph of its nodes and nothing else (`docs/23` section 3.1, the owner, 2026-11-01). Every control in
 * the UI and every CLI flag is a field of one node; the simple view reads the Input, Main and the Output of it through `inputOf`,
 * `mainOf` and `outputOf` (`src/state/settingsGraph.ts`). Defaults reproduce the automatic development (`docs/01` §DevelopSettings).
 */
export interface DevelopSettings {
  nodes: NodesSettings;
}

/** `trim` in a patch: the parts to change. */
export interface TrimPatch {
  exposure_ev?: number;
  temperature?: number;
  tint?: number;
}

/** `grade` in a patch: the fields to change (each one whole; `color_balance` and `curve` may carry only the wheels or curves to change). */
export interface GradePatch {
  contrast?: PercentSetting;
  brightness?: number;
  highlights?: PercentSetting;
  shadows?: PercentSetting;
  white?: number;
  black?: PercentSetting;
  color_balance?: Partial<ColorBalance>;
  saturation?: PercentSetting;
  color_warp?: ColorWarp;
  curve?: Partial<Curves>;
}

/**
 * A partial of one mask row's fields (`masks.edit`, `docs/23` section 3.7); it cannot change the row's kind. A Brush row also takes
 * `strokes_append` (strokes added after the existing ones) and `strokes_keep` (keep only the first n strokes, the inverse of an append).
 */
export interface MaskPatch {
  /** The row's own kind, or nothing: a row patch cannot change it. */
  kind?: string;
  mode?: MaskMode;
  invert?: boolean;
  enabled?: boolean;
  opacity?: number;
  feather?: number;
  grow?: number;
  refine?: number;
  /** An Object row's clicks, replaced whole. */
  points?: Array<[x: number, y: number, positive: boolean]>;
  /** An Object row's box; `null` takes it away. */
  box?: [x0: number, y0: number, x1: number, y1: number] | null;
  level?: number;
  /** A Depth row's `near` and `far`: places in the photo's depth order, 0 the nearest part of the photo and 1 the farthest. */
  near?: number;
  far?: number;
  softness?: number;
  center?: [x: number, y: number];
  angle_deg?: number;
  radii?: [rx: number, ry: number];
  /** A Brush row's strokes, replaced whole. */
  strokes?: BrushStroke[];
  strokes_append?: BrushStroke[];
  strokes_keep?: number;
  width?: number;
  low?: number;
  high?: number;
}

/** `masks` in a patch: the node's Invert, the whole list of rows, and row patches by index (a string), applied after `items`. */
export interface MasksPatch {
  invert?: boolean;
  items?: MaskItem[];
  edit?: Record<string, MaskPatch>;
}

/**
 * A partial of one node's fields (`docs/23` section 3.7), the fields of every kind in one flat object: a field the node's kind does not
 * have is refused. Nested fields may carry only the parts to change (`decode`, `rolloff`, `lens`, `trim`, `grade`, `color_balance`,
 * `curve`); `crop: null` clears the crop and `white_pick: null` the picked white. A node the photo does not have is added when `kind`
 * is given.
 */
export interface NodePatch {
  /** The node's kind: needed to add a node. */
  kind?: NodeKind;
  pos?: NodePos;
  exposure?: AutoOr<{ ev: number }>;
  light_source?: LightSource;
  white_balance?: AutoOr<{ kelvin: number; tint: number }>;
  white_pick?: [x: number, y: number] | null;
  rolloff?: Partial<Rolloff>;
  decode?: Partial<DecodeOptions>;
  noise_reduction?: AutoOr<{ luminance: number; color: number; detail: number }>;
  sharpen?: AutoOr<{ amount: number; radius_px: number }>;
  lens?: Partial<LensSettings>;
  crop?: Crop | null;
  orientation?: Orientation;
  look?: LookId;
  look_intensity?: number;
  output_sharpening?: number;
  name?: string;
  bypass?: boolean;
  trim?: TrimPatch;
  grade?: GradePatch;
  masks?: MasksPatch;
  /** A Film node's negative. */
  film?: Partial<FilmStockSettings>;
  /** A Film node's enlarger and paper; `paper: null` is the matching paper. */
  print?: { paper?: string | null; exposure_ev?: number; lights?: Partial<PrintLights> };
  halation?: Partial<HalationSettings>;
  grain?: Partial<GrainSettings>;
  /** A Film node's Strength, percent. */
  strength?: number;
}

/**
 * A node-level partial update (`update_settings(photo_id, patch)`, `docs/23` section 3.7): `nodes` maps node ids as strings (`"input"`,
 * `"output"`, `"1"`, `"2"`) to what changes in each, `remove` lists the corrector nodes to delete, and `links`, when present, replaces
 * the links whole. A slider drag sends one node's partial, never the graph (`patchOf`, `src/state/settingsGraph.ts`).
 */
export interface SettingsPatch {
  nodes?: Record<string, NodePatch>;
  remove?: NodeId[];
  links?: NodeLink[];
}

/**
 * What `paste_settings` did to one photo: the nodes and fields it changed, as two patches that make one undo step. `before` is what Undo
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

/** Which of the Input's `Auto` fields are still automatic; the UI shows the AUTO tag for these. */
export interface InputAutoFlags {
  exposure: boolean;
  light_source: boolean;
  white_balance: boolean;
  sharpen: boolean;
  noise_reduction: boolean;
}

/** Which of Main's `Auto` fields are still automatic. */
export interface MainAutoFlags {
  contrast: boolean;
  highlights: boolean;
  shadows: boolean;
  black: boolean;
  balance_shadows: boolean;
  balance_highlights: boolean;
  saturation: boolean;
}

/** Which `Auto` fields are still automatic, by node: only the Input and Main can be Auto (`docs/23` section 3.2). */
export interface AutoFlags {
  input: InputAutoFlags;
  main: MainAutoFlags;
}

/** Light classes of the illuminant estimate (`docs/02`). */
export type LightClass = "daylight" | "tungsten" | "fluorescent" | "led";

/**
 * Which Auto made a photo's automatic values: Yellowmoon's own (`rule`), or the answer of Apple Intelligence (`apple`, `docs/19`), Claude
 * Code (`claude`) or Antigravity (`agy`, `docs/22` section 14.1).
 */
export type AutoSource = "rule" | "apple" | "claude" | "agy";

/**
 * Why a provider is not available on this computer: Apple's nine reasons in the order `docs/19` section 1 checks them, then the seven
 * of `docs/22` section 3 that the two programs (and the consent) add. `not_consented` is never a provider's own state: only a photo's
 * (`AppleAuto`) says it, when a cloud provider is chosen and available but the person has not agreed yet.
 */
export type AppleAutoReason =
  | "not_a_mac"
  | "intel_mac"
  | "built_without_support"
  | "macos_too_old"
  | "device_not_eligible"
  | "apple_intelligence_off"
  | "model_not_ready"
  | "unavailable"
  | "no_image_input"
  | "not_installed"
  | "too_old"
  | "signed_out"
  | "probe_failed"
  | "unsafe_launcher"
  | "checking"
  | "not_consented";

/** Why one request to Apple's model failed (`docs/19` sections 6 and 8). A failed photo uses Default Auto. */
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
  | "not_signed_in"
  | "cli_failed"
  | "other";

/** Whether Apple's model read the default instructions or the person's own (`docs/19` section 2). */
export type AppleAutoInstructions = "default" | "edited";

/**
 * Who grades: Default Auto (the rule, `docs/02`), Apple Intelligence (`docs/19`), or Claude Code or Antigravity (the `agy` program),
 * which the person installed and signed in to themselves (`docs/22` section 1). The order is the dropdown's.
 */
export type AutoProvider = "yellowmoon" | "apple" | "claude" | "agy";

/** Whether the person agreed to send photos through a program (`docs/22` section 10.2). Apple and Default Auto need none. */
export interface AutoConsent {
  claude: boolean;
  agy: boolean;
}

/**
 * Which model answers: "AFM 3 Core" on macOS "27.0.1 (27A404)", or Claude Code 2.1.310 (`tool_version`) with `claude-sonnet-5-5` (`os` is ""
 * for the programs), with which prompt and instructions (the first 12 hex digits of their SHA-256). `api` says where Claude Code's
 * pictures go (`firstParty`: Anthropic).
 */
export interface AppleAutoInfo {
  provider: AutoProvider;
  model: string;
  os: string;
  tool_version: string | null;
  api: string | null;
  prompt: number;
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
 * Where a photo stands with the provider that grades it (`docs/19` sections 6 and 12, `docs/22` section 10.4). `off`: Yellowmoon's own
 * Auto is chosen, or the app has no advisor (the browser build); `unavailable`: the chosen provider cannot answer (`reason`; a program
 * the person has not agreed to is `not_consented`); `pending`: asked, or waiting for its turn (Auto is the rule until the answer);
 * `answered` (`info.provider` says who); `failed` (Auto is the rule). The Automatic card shows its rows and its line while the state is
 * `pending`, `answered` or `failed`, and a line for a program that was chosen and `unavailable`.
 */
export type AppleAuto =
  | { state: "off" }
  | { state: "unavailable"; reason: AppleAutoReason }
  | { state: "pending" }
  | { state: "answered"; answer: AppleAutoAnswer; info: AppleAutoInfo }
  | { state: "failed"; kind: AppleAutoFailure; message: string; info: AppleAutoInfo };

/** Where one provider stands (`docs/22` section 3.1). */
export type ProviderState = "available" | "not_installed" | "too_old" | "signed_out" | "unavailable" | "checking";

/**
 * One provider of `auto_status()` (`docs/22` sections 3.1 and 11). `listed`: the dropdown shows it; `selectable`: it can be chosen
 * (`false` for a listed program that is signed out, too old or not working: shown disabled with a note). `consent` is whether the person
 * agreed to send photos through it; `experimental` marks Antigravity. `info` is there while it is available.
 */
export interface ProviderStatus {
  provider: AutoProvider;
  state: ProviderState;
  /** Why it is not available (`null` while it is). */
  reason: AppleAutoReason | null;
  info: AppleAutoInfo | null;
  /** The version the program printed, also while it is too old or signed out. */
  tool_version: string | null;
  consent: boolean;
  experimental: boolean;
  listed: boolean;
  selectable: boolean;
}

/**
 * `auto_status()` and `auto_status_cached()` (`docs/22` sections 11 and 12.6; they replace `apple_auto_status()`): the four providers in the
 * dropdown's order, the saved choice, the one that grades now (`effective`: the chosen one while it is available and agreed to, else
 * `yellowmoon`) and whether any provider calls for the "Auto grading" row and the instructions (`visible`: Apple Intelligence is available,
 * or one of the programs is listed). Preferences shows the row while `visible` or while the saved choice is a program (`autoRowShown`).
 */
export interface AutoStatus {
  providers: ProviderStatus[];
  /** The saved choice (`Preferences.auto_provider`). */
  preference: AutoProvider;
  /** The provider the session asks: the saved choice in the app. */
  chosen: AutoProvider;
  /** The provider that grades now. */
  effective: AutoProvider;
  /** `effective` is not `yellowmoon`. */
  active: boolean;
  /** A provider is there: Preferences shows its "Auto grading" row and the instructions (also while the saved choice is a program). */
  visible: boolean;
  consent: AutoConsent;
  /** The table's version (`docs/19` section 5). */
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

/** What Auto resolves to for the Input (`estimates.input`, `docs/23` section 3.9). */
export interface InputEstimates {
  exposure_ev: number;
  kelvin: number;
  tint: number;
  /** What `sharpen: Auto` resolves to. */
  sharpen: { amount: number; radius_px: number };
  /** What `noise_reduction: Auto` resolves to. */
  noise_reduction: { luminance: number; color: number; detail: number };
}

/** What Auto resolves to for Main (`estimates.main`). */
export interface MainEstimates {
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
}

/**
 * Per-photo automatic estimates, computed once and recomputed by "Re-run" (`docs/01` `AutoEstimates`; the shape of the `auto` object in
 * `docs/10` `--json` output), grouped by the node each number is for: `input` and `main` (`docs/23` section 3.9), with what belongs to the
 * photo beside them. With an answer from Apple Auto, the exposure, the Kelvin, the contrast and the saturation are Apple's (`docs/19`
 * section 5).
 */
export interface AutoEstimates {
  /** What Auto resolves to for the Input. */
  input: InputEstimates;
  /** What Auto resolves to for Main. */
  main: MainEstimates;
  /** Which Auto made these values. */
  source: AutoSource;
  /**
   * The provider `apple` and `apple_steps` belong to (`docs/22` section 14.1): the chosen provider while it acts or is asked about (so
   * the card can name it while the photo is `pending` or `unavailable`), `yellowmoon` while the state is `off`.
   */
  provider: AutoProvider;
  /** Where the photo stands with Apple Auto. */
  apple: AppleAuto;
  /** Apple Auto's steps in effect (the photo's answer, or its burst's), null when Auto is the rule. */
  apple_steps: AppleAutoSteps | null;
  light_class: LightClass;
  /** 0–1. */
  light_confidence: number;
  /** Poisson–Gaussian noise model, `variance = a · signal + b`. */
  noise: { a: number; b: number };
}

/** The Input with every `Auto` replaced by the number it resolves to (`values.input`). */
export interface ResolvedInput {
  pos: NodePos;
  exposure_ev: number;
  light_source: LightSource;
  kelvin: number;
  tint: number;
  sharpen: { amount: number; radius_px: number };
  noise_reduction: { luminance: number; color: number; detail: number };
  lens: LensSettings;
  decode: DecodeOptions;
  rolloff: Rolloff;
  crop: Crop | null;
  orientation: Orientation;
  white_pick: [x: number, y: number] | null;
}

/** A corrector's grade with every `Auto` replaced by the number it resolves to. */
export interface GradeValues {
  /** The contrast in effect: the manual value, or the photo's Auto contrast. */
  contrast: number;
  brightness: number;
  /** Highlights, shadows and black in effect: the manual value, or the photo's Auto value. */
  highlights: number;
  shadows: number;
  white: number;
  black: number;
  /** The wheels' points in effect. */
  color_balance: ColorBalancePoints;
  /** The saturation in effect: the manual value, or the photo's Auto saturation. */
  saturation: number;
  color_warp: ColorWarp;
  curve: Curves;
}

/** A corrector with its grade resolved (`values.main`, `values.nodes[…]`); only Main has Auto. */
export interface ResolvedCorrector {
  /** The node's id (Main is 1). */
  id: number;
  name: string;
  bypass: boolean;
  pos: NodePos;
  trim: NodeTrim;
  grade: GradeValues;
  masks: MaskSettings;
}

/** The Output (it has no Auto). */
export interface ResolvedOutput {
  pos: NodePos;
  look: LookId;
  look_intensity: number;
  output_sharpening: number;
}

/** The settings with every `Auto` replaced by the number it resolves to, node by node (`docs/23` section 3.9). */
export interface ResolvedValues {
  input: ResolvedInput;
  /** Main, node 1. */
  main: ResolvedCorrector;
  /** The other correctors, by id (their values are their fields: they have no Auto). */
  nodes: ResolvedCorrector[];
  output: ResolvedOutput;
}

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
  /** Files whose hash matched a look that was already imported (a re-import adds nothing, `docs/08`). */
  unchanged: number;
}

// ---------------------------------------------------------------------------
// The ARRI import wizard (docs/08 §Wizard, §Recognition rules, §Any other .cube). Additions to
// `docs/01`: the wizard needs to ask what a file is before anything is copied.
// ---------------------------------------------------------------------------

/**
 * The three free look packages ARRI publishes (`docs/08` §What users can import). The ids are the ones
 * of `data/looks/arri-packages.json`, except that the two look libraries that also come as ARRI Look Files
 * (the LogC4 one as `.alf4`, the LogC3 to Rec.709 one as `.aml`) are the same package here. ARRI's two display
 * LUT packages are not offered: Yellowmoon uses its own display rendering, and a display package that is chosen
 * anyway comes back `unusable` with a sentence that says so.
 */
export type PackageId = "look_library_logc4" | "look_library_logc3" | "look_library_logc3_to_rec709";

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

/** How the mask view shows a node's mask: tinted over the photo as it is, or alone as grey (`docs/23` §5.8). */
export type MaskStyle = "overlay" | "mask_only";

/**
 * The mask view of the viewport (`viewport_set_mode`'s second argument): the mask of a corrector shown in the frame by the core, over
 * the photo or alone. `null` on the wire is the photo.
 */
export interface MaskView {
  /** The corrector whose mask is shown. */
  node: number;
  style: MaskStyle;
  /**
   * The row of that corrector whose own mask is shown instead of the corrector's (its place in the list, rows that are off included): the row
   * as it enters the node's fold, with its own Invert and Opacity (`docs/23` §5.8). Left out, or `null`, for the corrector's mask.
   */
  row?: number | null;
}

/**
 * `chroma_scatter(photo_id)`: the photo's colours on the colour warp's plane, at the warp's input (after C9, before the pins), for the
 * faint scatter behind the plane. A flat list `[a0, b0, a1, b1, …]` of at most 2048 points.
 */
export interface ChromaScatter {
  points: number[];
}

/** A kind of mask row, as the row's `kind` is written (`docs/23` §3.3). */
export type MaskKind = MaskItem["kind"];

/**
 * Which side of `viewport_map_points` the points are given in (`docs/23` §9): `"photo"` (points `[x, y]` in the photo frame, answered as CSS
 * pixels from the cell's corner, which may lie outside the cell) or `"cell"` (CSS pixels, answered as photo-frame fractions, `null` for a
 * point on the paper).
 */
export type MapFrom = "photo" | "cell";

/**
 * What the person did on the photo to find an object: `mask_find`'s `prompt` (`docs/23` §9), in the photo frame. A click is `point` (`positive`
 * for a part of the object, not for a part to take away); a drag with the Box tool is `box`.
 */
export type MaskFindPrompt = { kind: "point"; x: number; y: number; positive: boolean } | { kind: "box"; x0: number; y0: number; x1: number; y1: number };

/**
 * The answer of `mask_find` (`docs/23` §9). `found` carries what the Object row's `points` and `box` become (the click or box added to what the
 * row had) and the time the decoder took; the page sends it as one `masks.edit` row patch. `nothing` is no object there: the page adds no point and
 * changes nothing, and the error bar says "Nothing found there. Click another part of the object, or draw a box."
 */
export type MaskFindResult = { kind: "found"; points: Array<[number, number, boolean]>; box: [number, number, number, number] | null; took_ms: number } | { kind: "nothing" };

/**
 * The answer of `sample_depth` (`docs/23` §9, Pick depth): the band around the depth under the click, in the photo's depth order (0 is the nearest part of the
 * photo, 1 the farthest), rounded to two decimals. The page writes the two numbers as one `masks.edit` row patch (`near`, `far`): one undo step.
 */
export interface DepthBand {
  near: number;
  far: number;
}

/**
 * Why `focus_plane` has no band (`docs/23` §9): `no_focus_point` (an older body, a panorama, a file that is not from a Sony camera; answered at once, without
 * waiting for the depth map: the card says "This photo has no focus point." and the button stays off) or `map_failed` (the depth map could not be made).
 */
export type FocusReason = "no_focus_point" | "map_failed";

/**
 * The answer of `focus_plane` (`docs/23` §9, Focus plane): with a band `near` and `far` are set as `sample_depth`'s and `reason` is `null`; without one they are
 * `null` and `reason` says why. The page writes a band as one `masks.edit` row patch: one undo step. `exact_centre` is true when the camera's point is exactly
 * the frame's centre, which Sony also writes when it does not know where it focused: the card says "The camera may not have recorded where it focused. Check
 * the result."
 */
export interface FocusPlaneResult {
  near: number | null;
  far: number | null;
  exact_centre: boolean;
  reason: FocusReason | null;
}

/**
 * `depth_state` (addition): the core's depth worker for a photo has ended (`docs/23` §4.8). `failed` is the plain sentence of the failure ("The Depth mask can't
 * be made: …") or `null` when the map is ready. The Depth rows with the eye on say "Finding…" while it runs (`viewport_busy.finding`); when it ends the photo is
 * drawn again with the mask, and a row whose map failed says "Can't be made".
 */
export interface DepthState {
  photo_id: PhotoId;
  failed: string | null;
}

/**
 * `object_state` (addition): SAM's encoder is working for a photo (`preparing`), or has stopped. The Object rows of that photo say "Getting ready…" while
 * it is (`docs/23` §7.6); when it stops with the features made the photo is drawn again.
 */
export interface ObjectState {
  photo_id: PhotoId;
  preparing: boolean;
}

/**
 * `vision_state` (addition): the core's worker for the Subject and People masks of a photo has ended its work (`docs/23` §4.5). `nothing_found` lists the
 * kinds whose newest mask for the photo found nothing: their rows say "Nothing found" (§7.6). The rows say "Finding…" while the worker runs
 * (`viewport_busy.finding`); when it ends the photo is drawn again with the masks. Sent each time the worker ends, with the list as it is then.
 */
export interface VisionState {
  photo_id: PhotoId;
  nothing_found: MaskKind[];
}

/**
 * Where in a node `sample_chroma` reads the colour (`docs/23` §9): `"warp_input"` (left out, or `null`) is the input of the node's colour warp,
 * after the node's own trim and grade (Pick colour of the Colour warp); `"node_input"` is the picture the node reads, before its trim and
 * grade, which is what a Colour range selects on (Pick colour of a Colour range).
 */
export type ChromaTap = "warp_input" | "node_input";

/** `sample_chroma(x, y, at, tap)`: the chroma of the pixel under a point of the photo cell, in the plane of the colour warp and of a Colour range's centre. */
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

/** One node's picture in `node_thumbnails_ready`: the corrector and the address of its thumbnail. */
export interface NodeThumbnail {
  /** The corrector (a card of the node view). */
  node: number;
  /**
   * Where the PNG is (`ym-frame://localhost/<launch>/node-<photo>-<node>-<version>.png`, `http://ym-frame.localhost/…` on Windows):
   * the picture after the node and then through the Output, about 160 px on its long edge. A new picture has a new address, and an
   * address names the same picture for as long as the app runs. The mock makes `blob:` ones.
   */
  url: string;
}

/**
 * `node_thumbnails_ready { photo_id, thumbnails }` (`docs/23` §5.7, §9): the answer to `node_thumbnails(photo_id)`, and only the
 * thumbnails that changed: a node that is not in the list keeps the address it has. The picture does not cross the bridge, as for
 * `thumbnail_ready`.
 */
export interface NodeThumbnailsReady {
  photo_id: PhotoId;
  thumbnails: NodeThumbnail[];
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
 * `viewport_busy` (addition): whether the viewport is working on a frame of the photo on screen, or on the full-resolution tiles a
 * frame needs, and whether the raw stages of that photo run on the CPU (no usable graphics card, a device error ended the route, the
 * device is lost, or a method the device does not run), which can take seconds. Sent when it changes (at most about 30 times a
 * second); `busy` goes false when the work is done or stopped, and when the cell shows no photo. The photo cell shows "Rendering on
 * the CPU" while both have held for 250 ms.
 */
export interface ViewportBusy {
  /** A frame, or the tiles a frame needs, is being made. */
  busy: boolean;
  /** The raw stages of the photo run on the CPU. */
  cpu: boolean;
  /**
   * The core makes the photo's first blur map for a Background mask in the round, or its Subject and People masks in the background (`docs/23`
   * §7.6): the Background, Subject and People rows say "Finding…" until `busy` goes false. While the Subject and People worker runs, `busy` and
   * `finding` stay true after the round that asked for the masks has ended. False otherwise, and once the photo has its map and its masks.
   */
  finding: boolean;
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
  /**
   * What the lens's bokeh reshaping did for the photo (`docs/21` §10), once the photo's planes exist: `reshaped`, and when it was not,
   * why (`not_asked`: no slider asks for it; `little_blur`: the photo has little out-of-focus background). `null` without a lens;
   * optional so that a sender with no lens to report (the mock) leaves it out.
   */
  lens?: { reshaped: boolean; reason: "not_asked" | "little_blur" | null } | null;
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
   * with its own display rendering (LogC3 to Rec.709) at an intensity above 0. The panel then adds one sentence
   * to the DNG note (addition, `docs/09` §What a DNG carries).
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

/** Folders remembered between runs. */
export interface LastUsedFolders {
  open: string | null;
  export: string | null;
  looks_import: string | null;
}

/** Everything that persists besides imported looks. The list is exhaustive (hard rule 4). */
export interface Preferences {
  default_look: LookId;
  /** Export defaults: format, quality, bits, compression, colour space, size, destination, name template, if-exists. */
  export_defaults: ExportOptions;
  remove_location_from_exports: boolean;
  check_for_updates_automatically: boolean;
  watch_downloads_during_import: boolean;
  last_used_folders: LastUsedFolders;
  /**
   * Who grades (`docs/22` section 10.1): Apple Intelligence by default. A provider that is not available, or a program the person has not
   * agreed to yet, acts as `yellowmoon`; the choice stays saved.
   */
  auto_provider: AutoProvider;
  /** Whether the person agreed to send photos through Claude Code and through Antigravity (`docs/22` section 10.2). */
  auto_consent: AutoConsent;
  /**
   * Instructions for the grading model (`docs/19` section 2, `docs/22` D1): the person's own instructions, at most 2000 characters, one
   * text for every provider; "" (the default) means `AutoStatus.default_instructions`.
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
  | "mask"
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

/** One camera film of the Film node's picker (`docs/24` section 4.1); the list is `names.json` through the core. */
export interface FilmStockInfo {
  /** The profile id ("kodak_portra_400"). */
  id: string;
  /** The name shown to the person ("Kodak Portra 400"). */
  name: string;
  /** The group of the picker: `colour_negative`, `cine_negative` or `slide` (see `FILM_GROUPS`). */
  kind: string;
  /** The maker. */
  group: string;
  iso: number | null;
  /** The profile id of the paper "Matching paper" means for it; null for a slide. */
  matching_paper: string | null;
}

/** One still paper of the Film node's paper picker. */
export interface FilmPaperInfo {
  id: string;
  name: string;
}

/** `film_catalogue`: what the Film node's pickers list, and the credit lines (`docs/24` sections 4.6, 6.2 and 6.5). */
export interface FilmCatalogue {
  /** The 20 camera films in the order of the picker. */
  stocks: FilmStockInfo[];
  /** The 6 still papers (the cine print films are not listed). */
  papers: FilmPaperInfo[];
  /** The version of the film data ("0.3.2"). */
  profile_version: string;
  /** The credit line of the film data (`docs/24` section 6.2); the Film tab and About show it with its link. */
  credit: string;
  /** "Modelled on published data. Not affiliated with Kodak or Fujifilm." */
  disclaimer: string;
  /** "Kodak, Portra, Vision3, Fujifilm and Velvia are trademarks of their owners." */
  trademarks: string;
}

/** What the app knows about itself, for About (addition). */
export interface AppInfo {
  /** The app and CLI version: "0.1.0". */
  version: string;
  /** The pipeline version written into exports (`docs/01` §Versions written into outputs). */
  pipeline_version: string;
  /** The operating system the app runs on, as Rust names it (`std::env::consts::OS`): "macos", "windows" or "linux" (`docs/22` section 12.3). */
  os: string;
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
  | "nodes"
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
  /** The thumbnails of a photo's nodes that changed (answers `node_thumbnails`; `docs/23` §5.7). */
  node_thumbnails_ready: NodeThumbnailsReady;
  render_stats: RenderStats;
  export_progress: ExportProgress;
  looks_changed: null;
  /** A provider's availability changed (a program found, signed in or out, a probe ended): read `auto_status_cached()` again (`docs/22` section 12.6). */
  auto_status_changed: null;
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
  /** The viewport is working on a frame (and whether on the CPU), or has stopped. */
  viewport_busy: ViewportBusy;
  /** SAM's encoder started or stopped working for a photo (`docs/23` §7.6: the Object rows say "Getting ready…"). */
  object_state: ObjectState;
  /** The Subject and People worker ended its work for a photo, and which kinds found nothing (`docs/23` §7.6: those rows say "Nothing found"). */
  vision_state: VisionState;
  /** The depth worker ended its work for a photo, and whether the map failed (`docs/23` §7.6: a failed row says "Can't be made"). */
  depth_state: DepthState;
}

/** Names of the events the core emits. */
export type ApiEventName = keyof ApiEvents;
