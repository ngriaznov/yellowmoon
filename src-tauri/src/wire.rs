//! The types that cross the bridge between the webview and the Rust core, as `specta` sees them.
//!
//! `src/api/types.ts` is the contract the screens are written against; this module is its Rust side,
//! and `src/api/bindings.ts` (generated from it by `tauri-specta`) is what the two have to agree
//! on. The core's own types (`ym_core::session::SessionState` and the rest) have the same serde
//! shape but do not derive `specta::Type`, so each one has a mirror here and
//! [`convert`] turns one into the other through serde. A test per type checks that the round trip
//! loses nothing (`tests::core_types_convert_without_loss`), and another that every field name of
//! `types.ts` is a field name here (`tests::field_names_match_the_typescript_contract`).
//!
//! # Where the generated TypeScript differs from `types.ts`
//!
//! * Ids are named numbers and strings (`PhotoId`, `LookId`, `ExportJobId`), as in `types.ts`.
//! * `Orientation` and `BitDepth` are the literal unions of `types.ts` (`0 | 90 | 180 | 270`,
//!   `8 | 16`).
//! * Settings are node-first (`docs/23` section 3): `DevelopSettings` is `{ nodes: { items, links } }`
//!   with the items told by `kind` (`NodeItem`), `NodeId` is `"input" | "output" | number`, and
//!   `SettingsPatch` is the core's node patch (`{ nodes, remove, links }`), a little more generous
//!   than a partial of the settings: nested objects (`decode`, `rolloff`, `lens`, `trim`, `grade`,
//!   `color_balance`, `curve`) and a manual `white_balance`, `sharpen` or `noise_reduction` may leave
//!   parts out, and `crop`, `white_pick` and a mask row's `box` may be `null` to clear them. Its
//!   `nodes` are keyed by the id as a string (`"input"`, `"1"`). `color_warp` is the whole list of
//!   pins. A flat patch of Yellowmoon 0.1 is an unknown field here (`deny_unknown_fields`) and is refused
//!   with the sentence of `SettingsError::OldForm` by the core's own patch type.
//! * A colour-warp pin on the bridge has every field (`ColorWarpPin` of `types.ts`); only a settings
//!   document (`ym render --settings`, `ym:Settings`) may leave out all but `source`.
//! * `MatchKind` has no `forced` variant (only the CLI's `--profile` makes one).
//! * `UpdateState` is the flat object of `docs/01` with `percent` and `message` optional, as
//!   `types.ts` has it.

use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use specta::Type;
use specta::datatype::DataType;
use tauri_specta::Event;

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Where an error came from (`ErrorScope` in `types.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ErrorScope {
    /// Opening files.
    Open,
    /// Decoding a raw file.
    Decode,
    /// Rendering.
    Render,
    /// Exporting.
    Export,
    /// The look library.
    Looks,
    /// The preferences.
    Preferences,
    /// The viewport.
    Viewport,
    /// A mask this computer cannot make or cannot find (`docs/23` section 9): the Object model is missing, the Object
    /// mask is not there, a row has all its points.
    Mask,
    /// Anything else.
    App,
}

impl ErrorScope {
    /// The scope for the name `ym_core::app::CoreError::scope` gives.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name {
            "open" => Self::Open,
            "decode" => Self::Decode,
            "render" => Self::Render,
            "export" => Self::Export,
            "looks" => Self::Looks,
            "preferences" => Self::Preferences,
            "viewport" => Self::Viewport,
            "mask" => Self::Mask,
            _ => Self::App,
        }
    }
}

/// An error that reaches the UI: a plain message that also says what to do next. Commands reject
/// with it and the `error` event carries it (`AppError` in `types.ts`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, thiserror::Error)]
#[error("{message}")]
pub struct AppError {
    /// Where it came from.
    pub scope: ErrorScope,
    /// What went wrong and what to do.
    pub message: String,
}

impl AppError {
    /// An error of `scope` with a plain message.
    pub fn new(scope: ErrorScope, message: impl Into<String>) -> Self {
        Self {
            scope,
            message: message.into(),
        }
    }

    /// A mistake the shell itself made (a type that did not convert, a thread that died).
    pub fn internal(what: impl std::fmt::Display) -> Self {
        tracing::error!("internal error: {what}");
        Self::new(
            ErrorScope::App,
            "Something unexpected went wrong inside Yellowmoon. Try again, or restart the app.",
        )
    }
}

impl From<ym_core::app::CoreError> for AppError {
    fn from(e: ym_core::app::CoreError) -> Self {
        Self::new(ErrorScope::from_name(e.scope()), e.to_string())
    }
}

impl From<ym_core::session::SessionError> for AppError {
    fn from(e: ym_core::session::SessionError) -> Self {
        use ym_core::session::SessionError as E;
        let scope = match &e {
            E::Decode(_) | E::Estimate(_) => ErrorScope::Decode,
            E::Settings(_) | E::AutoCrop(_) => ErrorScope::Render,
            E::Data(_) => ErrorScope::App,
            _ => ErrorScope::Open,
        };
        Self::new(scope, e.to_string())
    }
}

impl From<ym_core::looks::LooksError> for AppError {
    fn from(e: ym_core::looks::LooksError) -> Self {
        Self::new(ErrorScope::Looks, e.to_string())
    }
}

impl From<ym_core::pipeline::RenderError> for AppError {
    fn from(e: ym_core::pipeline::RenderError) -> Self {
        let scope = match &e {
            ym_core::pipeline::RenderError::Look(_) => ErrorScope::Looks,
            _ => ErrorScope::Render,
        };
        Self::new(scope, e.to_string())
    }
}

/// Converts between two types with the same serde shape (a core type and its mirror here).
///
/// # Errors
///
/// [`AppError::internal`] when the shapes do not match: a bug, caught by the tests of this module.
pub fn convert<A: Serialize, B: DeserializeOwned>(from: &A) -> Result<B, AppError> {
    let value = serde_json::to_value(from).map_err(AppError::internal)?;
    serde_json::from_value(value).map_err(AppError::internal)
}

/// A finite number on the bridge. TypeScript sees `number`.
///
/// `specta` describes a Rust `f64` as `number | null`, because `serde_json` writes a NaN or an
/// infinity as `null`. The core never sends one (every setting is clamped, every measurement is
/// finite), so the bridge says `number`, like `types.ts` does.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Num(pub f64);

impl Type for Num {
    fn definition(types: &mut specta::Types) -> DataType {
        <specta_typescript::Number as Type>::definition(types)
    }
}

// ---------------------------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------------------------

/// Identifies a photo for the lifetime of the session only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(transparent)]
pub struct PhotoId(pub u32);

/// Identifies a look: `builtin:neutral`, `arri:<group>:<slug>` or `user:<slug>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(transparent)]
pub struct LookId(pub String);

/// Identifies a queued export job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(transparent)]
pub struct ExportJobId(pub u32);

// ---------------------------------------------------------------------------------------------
// Camera and photo summary
// ---------------------------------------------------------------------------------------------

/// Why a camera profile was chosen. Anything but `exact` gets the STAND-IN tag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MatchKind {
    /// The body's own measured profile.
    Exact,
    /// Borrowed from a body with the same sensor; `via` names it.
    SensorFamily {
        /// The body whose profile is used.
        via: String,
    },
    /// The A7R IV stand-in.
    Default,
}

/// The detected body and the profile used for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct CameraInfo {
    /// Body as shown in the UI: "Sony A1".
    pub body: String,
    /// The camera's own model string, "ILCE-1". Empty when the file has none.
    pub model: String,
    /// Sony's model ID (maker note 0xB001), when the file has one.
    pub model_id: Option<u32>,
    /// Display name of the profile used: "A7R IV".
    pub profile: String,
    /// Profile id: "sony-ilce-7rm4".
    pub profile_id: String,
    /// Profile version (semver).
    pub profile_version: String,
    /// How the profile was chosen.
    pub match_kind: MatchKind,
}

/// Stages of the background job that develops a freshly opened photo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum PhotoStage {
    /// The metadata probe.
    Probe,
    /// Decoding the raw data.
    Decode,
    /// The automatic estimates.
    Estimate,
    /// The thumbnail render.
    Thumbnail,
    /// Apple Auto's answer (`docs/19` section 6): started while the photo waits for it, done when it
    /// arrived, failed, or Apple Auto stopped acting on the photo. Only while Apple Auto acts; the
    /// photo is ready before this stage is done.
    Apple,
}

/// A photo's place in the development pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhotoStatus {
    /// Probed, waiting for a decode thread.
    Waiting,
    /// A stage is running.
    Developing {
        /// The stage in flight.
        stage: PhotoStage,
    },
    /// Developed.
    Ready,
    /// Can't be developed; the plain-words reason.
    Skipped {
        /// What went wrong and what to do next.
        reason: String,
    },
}

/// What the UI needs to know about one photo without asking for its settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct PhotoSummary {
    /// The photo.
    pub id: PhotoId,
    /// File name with its extension: "DSC04204.ARW".
    pub file_name: String,
    /// The detected body and the profile used.
    pub camera: CameraInfo,
    /// ISO, 0 when the file does not say.
    pub iso: u32,
    /// Exposure time in seconds: 0.008 for 1/125.
    pub shutter_s: Num,
    /// F-number: 2.8.
    pub aperture_f: Num,
    /// Capture time as local ISO 8601 without a zone: "2026-10-04T17:42:10".
    pub captured_at: String,
    /// Width in pixels after the camera's orientation.
    pub width_px: u32,
    /// Height in pixels after the camera's orientation.
    pub height_px: u32,
    /// Where the photo is in development.
    pub status: PhotoStatus,
    /// Picked for the batch actions.
    pub picked: bool,
    /// Lossless M or S: the camera already processed it, so colour emulation is approximate.
    pub reduced: bool,
    /// Apple Auto acts on the photo and has not answered for it yet: its Auto values are Yellowmoon's
    /// own until the answer (`docs/19` section 6). The `apple` stage of `photo_progress` reports each
    /// change.
    pub apple_pending: bool,
}

/// The session as the webview mirrors it (`session_state`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct SessionState {
    /// Name of the folder the photos came from, for the header cell.
    pub folder_name: Option<String>,
    /// The photos in filmstrip order.
    pub photos: Vec<PhotoSummary>,
    /// Index into `photos` of the current photo; null when the session is empty.
    pub current_index: Option<u32>,
    /// Ids of the picked photos.
    pub picked: Vec<PhotoId>,
    /// Where Export saves by default: "<source folder> export".
    pub default_export_folder: Option<String>,
    /// What the `open_paths` or `open_dialog` call that answered with this state did; null in every
    /// other answer (`session_state`). It is what lets Photos say "Added 3 photos. 2 were already
    /// open." when files are added to a session.
    #[serde(default)]
    pub opened: Option<OpenSummary>,
}

/// The answer of `remove_photos` and `restore_photos`: the photos that were taken out of the filmstrip or put back, and the
/// session as it is now.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct SessionChange {
    /// The photos taken out (`remove_photos`: those of the ids that were in the session) or put back (`restore_photos`), in
    /// filmstrip order.
    pub photo_ids: Vec<PhotoId>,
    /// `restore_photos` only: photos of the removal that stayed out because their file had been added to the session again
    /// meanwhile. 0 for `remove_photos`.
    pub already_open: u32,
    /// The session afterwards. A snapshot, as every answer with a session is: about the photos that were open already it is a
    /// little older than the page's own state, which events and clicks have moved on since.
    pub session: SessionState,
}

/// What one `open_paths` or `open_dialog` call did to the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct OpenSummary {
    /// ARW files added to the filmstrip.
    pub added: u32,
    /// ARW files that were already in the session and were not added again.
    pub already_open: u32,
}

// ---------------------------------------------------------------------------------------------
// Develop settings
// ---------------------------------------------------------------------------------------------

/// `exposure`: Auto, or an absolute value in stops. The JSON is `{ "kind": "auto" }` or
/// `{ "kind": "manual", "ev": 0.3 }`.
///
/// (`types.ts` has one generic `AutoOr<M>`; specta cannot check that a generic payload merges with
/// the tag, so each setting has its own enum with the same JSON.)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExposureSetting {
    /// Resolve from the photo's automatic estimates.
    Auto,
    /// The user's value.
    Manual {
        /// Absolute exposure in stops, -4 to +4.
        ev: Num,
    },
}

/// `white_balance`: Auto, or Kelvin and tint.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WhiteBalanceSetting {
    /// Resolve from the photo's automatic estimates.
    Auto,
    /// The user's value.
    Manual {
        /// Colour temperature, 2000 to 11000 K.
        kelvin: Num,
        /// Tint, -50 to 50, positive towards magenta.
        tint: Num,
    },
}

/// `sharpen`: Auto, or amount and radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SharpenSetting {
    /// Resolve from the photo's automatic estimates.
    Auto,
    /// The user's value.
    Manual {
        /// Amount, 0 to 300.
        amount: Num,
        /// PSF radius in pixels, 0.3 to 2.5.
        radius_px: Num,
    },
}

/// `noise_reduction`: Auto, or the three strengths.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NoiseReductionSetting {
    /// Resolve from the photo's automatic estimates.
    Auto,
    /// The user's value.
    Manual {
        /// Luminance strength, 0 to 100.
        luminance: Num,
        /// Colour strength, 0 to 100.
        color: Num,
        /// Detail strength, 0 to 100.
        detail: Num,
    },
}

/// `contrast`, `highlights`, `shadows`, `black` and `saturation`: Auto (0, or Apple Auto's value for
/// the photo, `docs/19` section 5) or a manual value. The JSON is `{ "kind": "auto" }` or
/// `{ "kind": "manual", "value": 12 }`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PercentSetting {
    /// Resolve from the photo's automatic estimates.
    Auto,
    /// The user's value.
    Manual {
        /// -100 to 100.
        value: Num,
    },
}

/// `contrast`, `highlights`, `shadows`, `black` or `saturation` in a patch: either form of
/// [`PercentSetting`], or a plain number,
/// which the core reads as Manual with that value, except 0, which it reads as Auto (the form of a
/// settings document written before Apple Auto, `docs/19` section 5). The webview sends the tagged
/// form.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(untagged)]
pub enum PercentPatch {
    /// `{ "kind": "auto" }` or `{ "kind": "manual", "value": 12 }`.
    Setting(PercentSetting),
    /// A plain number: Manual, or Auto for 0.
    Number(Num),
}

/// The sharpening in effect.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct SharpenValue {
    /// Amount, 0 to 300.
    pub amount: Num,
    /// PSF radius in pixels, 0.3 to 2.5.
    pub radius_px: Num,
}

/// The noise reduction in effect.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct NoiseReductionValue {
    /// Luminance strength, 0 to 100.
    pub luminance: Num,
    /// Colour strength, 0 to 100.
    pub color: Num,
    /// Detail strength, 0 to 100.
    pub detail: Num,
}

/// `light_source`: Colour › White balance, the segmented control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LightSource {
    /// Estimated from the scene.
    Auto,
    /// Daylight, 5500 K.
    Day,
    /// Shade, 7500 K.
    Shade,
    /// Tungsten, 3200 K.
    Tungsten,
    /// LED.
    Led,
    /// Fluorescent.
    Fluorescent,
}

/// The shadows or highlights colour-balance wheel: Auto (`[0, 0]`, or the colour Apple Auto gave
/// that zone, `docs/19` section 5) or a manual point. The JSON is `{ "kind": "auto" }` or
/// `{ "kind": "manual", "point": [a, b] }`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BalanceSetting {
    /// Resolve from the photo's automatic estimates.
    Auto,
    /// The user's point.
    Manual {
        /// `[a, b]` within the unit disc.
        point: [Num; 2],
    },
}

/// A shadows or highlights wheel in a patch: either form of [`BalanceSetting`], or a plain point,
/// which the core reads as Manual at that point, except `[0, 0]`, which it reads as Auto (the form
/// of a settings document written before Apple Auto's prompt version 2). The webview sends the
/// tagged form.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(untagged)]
pub enum BalancePatch {
    /// `{ "kind": "auto" }` or `{ "kind": "manual", "point": [a, b] }`.
    Setting(BalanceSetting),
    /// A plain point: Manual, or Auto for `[0, 0]`.
    Point([Num; 2]),
}

/// Colour balance wheels: the shadows and highlights wheels are Auto or a point, the midtones wheel
/// a point `[a, b]` within the unit disc, `[0, 0]` by default.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct ColorBalance {
    /// The shadows wheel.
    pub shadows: BalanceSetting,
    /// The midtones wheel.
    pub midtones: [Num; 2],
    /// The highlights wheel.
    pub highlights: BalanceSetting,
}

/// The three wheels' points in effect (`ResolvedValues.color_balance`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct ColorBalancePoints {
    /// The shadows wheel.
    pub shadows: [Num; 2],
    /// The midtones wheel.
    pub midtones: [Num; 2],
    /// The highlights wheel.
    pub highlights: [Num; 2],
}

/// One pin of Colour › Colour warp (`docs/02` C9b): the colours near `source` move to `target`.
/// Points are `[a, b]` on the warp's chroma plane (`ChromaPoint` in `types.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct ColorWarpPin {
    /// The colour the pin takes hold of.
    pub source: [Num; 2],
    /// Where that colour moves to; a locked pin's target is its source.
    pub target: [Num; 2],
    /// A locked pin holds the colours within its range in place.
    pub locked: bool,
    /// How far the pin reaches across the plane, 0.01 to 0.25.
    pub range: Num,
    /// How much of the move applies to the shadows, 0 to 1.
    pub shadows: Num,
    /// How much of the move applies to the highlights, 0 to 1.
    pub highlights: Num,
    /// Where the shadows end and the highlights begin, stops from middle grey, -6 to +6.
    pub pivot_ev: Num,
    /// A change of brightness of the pin's colours, stops, -2 to +2.
    pub exposure_ev: Num,
    /// A change of saturation of the pin's colours, -1 to +1 (-1 grey, +1 twice as far from
    /// neutral); shown as -100 to +100 %.
    pub saturation: Num,
}

/// Colour › Colour warp: at most 16 pins, in order; none by default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct ColorWarp {
    /// The pins.
    pub pins: Vec<ColorWarpPin>,
}

/// Luma plus R, G and B curves, 2 to 16 control points `[x, y]` within [0, 1]² each.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct Curves {
    /// The luma curve.
    pub luma: Vec<[Num; 2]>,
    /// The red curve.
    pub red: Vec<[Num; 2]>,
    /// The green curve.
    pub green: Vec<[Num; 2]>,
    /// The blue curve.
    pub blue: Vec<[Num; 2]>,
}

/// RAW › Decode switches; all true by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct DecodeOptions {
    /// Neural demosaic and denoise.
    pub neural_demosaic: bool,
    /// Highlight reconstruction.
    pub highlight_recovery: bool,
    /// Vignetting and distortion corrections from the camera's tags.
    pub lens_corrections: bool,
    /// Lateral chromatic-aberration correction.
    pub chromatic_aberration: bool,
}

/// RAW › Range (`docs/02` C7), each 0 to 1: Highlights 0.6 and Low light 0.5 by default. The name
/// `rolloff` is kept from the earlier roll-off card.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct Rolloff {
    /// The highlight range: where the camera's clipping point lands, from where it was recorded (0)
    /// to white, the clip of an ALEXA at EI 800 (1).
    pub highlights: Num,
    /// Low-light toe strength.
    pub low_light: Num,
}

/// Look › Lens: which lens (`LensKind` in `types.ts`, `docs/21` section 2). The ids are those of
/// `data/lenses/` and of `ym render --lens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LensKind {
    /// Clean and modern: even sharpness, faint glow.
    Default,
    /// Warm and gentle: soft glow, amber flares, smooth falloff.
    Classic,
    /// Glows wide open, lower contrast, soft edges, a little swirl.
    Vintage,
    /// Very sharp centre, strong swirl, fast falloff.
    Petzval,
    /// Soft corners, busy flares, a crisp centre.
    Triplet,
    /// Smooth and glowing, with gentle edges.
    Sonnar,
    /// Sharp only in the centre, with smeared flares.
    Detuned,
}

impl From<ym_lens::LensKind> for LensKind {
    fn from(k: ym_lens::LensKind) -> Self {
        match k {
            ym_lens::LensKind::Default => Self::Default,
            ym_lens::LensKind::Classic => Self::Classic,
            ym_lens::LensKind::Vintage => Self::Vintage,
            ym_lens::LensKind::Petzval => Self::Petzval,
            ym_lens::LensKind::Triplet => Self::Triplet,
            ym_lens::LensKind::Sonnar => Self::Sonnar,
            ym_lens::LensKind::Detuned => Self::Detuned,
        }
    }
}

impl From<LensKind> for ym_lens::LensKind {
    fn from(k: LensKind) -> Self {
        match k {
            LensKind::Default => Self::Default,
            LensKind::Classic => Self::Classic,
            LensKind::Vintage => Self::Vintage,
            LensKind::Petzval => Self::Petzval,
            LensKind::Triplet => Self::Triplet,
            LensKind::Sonnar => Self::Sonnar,
            LensKind::Detuned => Self::Detuned,
        }
    }
}

/// Look › Lens: a lens and three amounts, each 0 to 100. All three at 0 is no lens, whatever the
/// lens is.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct LensSettings {
    /// The lens.
    pub kind: LensKind,
    /// Flares: glow, veiling glare, ghosts and starbursts, 0 to 100.
    pub flares: Num,
    /// Edges: softness, darkening, fringes and distortion, 0 to 100.
    pub edges: Num,
    /// Anamorphic: the attachment in front of the lens, with the lens's own swirl and cat's-eye shapes in the blur, 0 to 100.
    pub anamorphic: Num,
}

/// One lens of the Lens card's menu (`LensInfo` in `types.ts`, `lens_list`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct LensInfo {
    /// The lens.
    pub kind: LensKind,
    /// The name shown in the menu.
    pub name: String,
    /// The line shown under the name.
    pub line: String,
    /// Flares the card moves to when this lens is chosen, 0 to 100.
    pub start_flares: Num,
    /// Edges the card moves to when this lens is chosen, 0 to 100.
    pub start_edges: Num,
    /// The data version of the lens.
    pub version: u32,
    /// The design the lens was computed from (a patent or a publication).
    pub design_source: String,
}

/// A rectangle in image coordinates normalised to [0, 1]².
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct NormRect {
    /// Left edge.
    pub x: Num,
    /// Top edge.
    pub y: Num,
    /// Width.
    pub width: Num,
    /// Height.
    pub height: Num,
}

/// Crop aspect presets: Free, Original, 1:1, 4:5, 16:9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum CropAspect {
    /// No constraint.
    Free,
    /// The photo's own shape.
    Original,
    /// 1:1.
    Square,
    /// 4:5.
    FourByFive,
    /// 16:9.
    SixteenByNine,
}

/// The crop (per photo, in memory only): a rectangle, a straighten angle and an aspect.
///
/// Serialising leaves `auto` out when it is false, as the core does; the unified TypeScript export
/// refuses `skip_serializing_if`, hence the manual `Serialize`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Type)]
pub struct Crop {
    /// The frame.
    pub rect: NormRect,
    /// Straighten angle, -45 to 45 degrees.
    pub angle_deg: Num,
    /// The aspect preset the frame follows.
    pub aspect: CropAspect,
    /// The rect and angle came from Auto crop (`docs/20-auto-crop.md` §2); absent is false.
    #[serde(default)]
    pub auto: bool,
}

impl Serialize for Crop {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("rect", &self.rect)?;
        map.serialize_entry("angle_deg", &self.angle_deg)?;
        map.serialize_entry("aspect", &self.aspect)?;
        if self.auto {
            map.serialize_entry("auto", &true)?;
        }
        map.end()
    }
}

/// The straightener's answer (`docs/20-auto-crop.md` §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum StraightenKind {
    /// Within 0.2° of level.
    Level,
    /// Levelled by the suggestion's angle.
    Rotate,
    /// No horizon or verticals that agree, or a tilt beyond 10°.
    Unsure,
}

/// Where Auto crop's subject came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum SubjectSource {
    /// The camera's focus point.
    Camera,
    /// The picture alone.
    Picture,
}

/// One frame Auto crop proposes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct CropFrame {
    /// The frame, as `Crop.rect`.
    pub rect: NormRect,
    /// Its overlap with the crop model's crop (intersection over union, 0 to 1).
    pub score: Num,
}

/// What Auto crop proposes for a photo, an aspect and an orientation (`suggest_crop`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct CropSuggestion {
    /// The straightener's answer.
    pub straighten: StraightenKind,
    /// The straighten angle the frames are made for: 0 unless `straighten` is `rotate`.
    pub angle_deg: Num,
    /// Best first, at most 3, never empty.
    pub frames: Vec<CropFrame>,
    /// The first frame is the largest frame of the aspect at that angle.
    pub keep: bool,
    /// Where the subject came from.
    pub subject: SubjectSource,
}

/// Orientation applied after the camera's EXIF orientation, in degrees: `0 | 90 | 180 | 270`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Orientation(pub u16);

impl Type for Orientation {
    fn definition(_: &mut specta::Types) -> DataType {
        DataType::Reference(specta_typescript::define("0 | 90 | 180 | 270"))
    }
}

/// A node's id: `"input"`, `"output"` or a number (Main is 1). The core's [`ym_core::settings::NodeId`] with
/// the TypeScript type of `types.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub ym_core::settings::NodeId);

impl Type for NodeId {
    fn definition(_: &mut specta::Types) -> DataType {
        DataType::Reference(specta_typescript::define("\"input\" | \"output\" | number"))
    }
}

/// What a node is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// The fixed Input.
    Input,
    /// The fixed Output.
    Output,
    /// A corrector.
    Corrector,
    /// A Film node (`docs/24`).
    Film,
    /// A Face Refinement node (`docs/27`).
    FaceRefinement,
    /// A Parallel mixer (`docs/25`).
    Parallel,
    /// A Layer mixer (`docs/25`).
    Layer,
    /// A Splitter (`docs/25`).
    Splitter,
    /// A Combiner (`docs/25`).
    Combiner,
}

/// A port of a node: a picture or a key (a mask). A mixer's inputs are `picture.1` to `picture.8`; a Splitter's
/// outputs and a Combiner's inputs are `red`, `green` and `blue`, all pictures (`docs/25` section 2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub enum Port {
    /// The LogC4 picture.
    #[serde(rename = "picture")]
    Picture,
    /// A mask.
    #[serde(rename = "key")]
    Key,
    /// A mixer's first picture input.
    #[serde(rename = "picture.1")]
    Picture1,
    /// A mixer's second picture input.
    #[serde(rename = "picture.2")]
    Picture2,
    /// A mixer's third picture input.
    #[serde(rename = "picture.3")]
    Picture3,
    /// A mixer's fourth picture input.
    #[serde(rename = "picture.4")]
    Picture4,
    /// A mixer's fifth picture input.
    #[serde(rename = "picture.5")]
    Picture5,
    /// A mixer's sixth picture input.
    #[serde(rename = "picture.6")]
    Picture6,
    /// A mixer's seventh picture input.
    #[serde(rename = "picture.7")]
    Picture7,
    /// A mixer's eighth picture input.
    #[serde(rename = "picture.8")]
    Picture8,
    /// A Splitter's red output or a Combiner's red input.
    #[serde(rename = "red")]
    Red,
    /// A Splitter's green output or a Combiner's green input.
    #[serde(rename = "green")]
    Green,
    /// A Splitter's blue output or a Combiner's blue input.
    #[serde(rename = "blue")]
    Blue,
}

/// One end of a link: a node and one of its ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct NodeRef {
    /// The node.
    pub node: NodeId,
    /// The port.
    pub port: Port,
}

/// A link from an output port to an input port (`docs/23` section 3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct NodeLink {
    /// The output port the link starts at.
    pub from: NodeRef,
    /// The input port it ends at.
    pub to: NodeRef,
}

/// A corrector's trims: an exposure trim in stops and a temperature and tint trim.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct NodeTrim {
    /// Stops, -4 to 4.
    pub exposure_ev: Num,
    /// -100 to 100.
    pub temperature: Num,
    /// -100 to 100.
    pub tint: Num,
}

/// A corrector's grade: the fields the settings had before they were nodes, with the same names and
/// types. Auto belongs to Main: in another corrector Auto reads as a manual 0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct NodeGrade {
    /// Light › Exposure › Contrast: Auto or a value, -100 to 100.
    pub contrast: PercentSetting,
    /// -100 to 100.
    pub brightness: Num,
    /// Light › High dynamic range › Highlights: Auto or a value, -100 to 100.
    pub highlights: PercentSetting,
    /// Light › High dynamic range › Shadows: Auto or a value, -100 to 100.
    pub shadows: PercentSetting,
    /// -100 to 100.
    pub white: Num,
    /// Light › High dynamic range › Black: Auto or a value, -100 to 100.
    pub black: PercentSetting,
    /// The three colour-balance wheels.
    pub color_balance: ColorBalance,
    /// Colour › Colour balance › Saturation: Auto or a value, -100 to 100.
    pub saturation: PercentSetting,
    /// Colour › Colour warp: its pins.
    pub color_warp: ColorWarp,
    /// Luma, red, green and blue curves.
    pub curve: Curves,
}

/// Add or subtract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum MaskMode {
    /// The row adds to the mask.
    Add,
    /// The row takes away from the mask.
    Subtract,
}

/// One stroke of a Brush mask.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct BrushStroke {
    /// The brush's size, 0.001 to 0.25 of the long edge.
    pub size: Num,
    /// The brush's hardness, 0 to 1.
    pub hardness: Num,
    /// An eraser stroke.
    pub erase: bool,
    /// The stroke's points in the photo frame.
    pub points: Vec<[Num; 2]>,
}

/// One row of a node's masks: `kind` and the fields every row has (`mode`, `invert`, `enabled`,
/// `opacity`), then the kind's own (`docs/23` section 3.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaskItem {
    /// SAM 2.1 from clicks or a box.
    Object {
        /// Add or subtract.
        mode: MaskMode,
        /// This mask alone is inverted.
        invert: bool,
        /// Off (the eye) counts as absent.
        enabled: bool,
        /// 0 to 1.
        opacity: Num,
        /// Clicks `[x, y, positive]` in the photo frame, at most 256.
        points: Vec<(Num, Num, bool)>,
        /// A box `[x0, y0, x1, y1]` in the photo frame, or null.
        #[serde(rename = "box")]
        box_: Option<[Num; 4]>,
        /// Feather, 0 to 1.
        feather: Num,
        /// Grow, -1 to 1.
        grow: Num,
        /// Refine edges, 0 to 1.
        refine: Num,
    },
    /// Apple Vision's foreground.
    Subject {
        /// Add or subtract.
        mode: MaskMode,
        /// This mask alone is inverted.
        invert: bool,
        /// Off (the eye) counts as absent.
        enabled: bool,
        /// 0 to 1.
        opacity: Num,
        /// Feather, 0 to 1.
        feather: Num,
        /// Grow, -1 to 1.
        grow: Num,
        /// Refine edges, 0 to 1.
        refine: Num,
    },
    /// Apple Vision's people.
    People {
        /// Add or subtract.
        mode: MaskMode,
        /// This mask alone is inverted.
        invert: bool,
        /// Off (the eye) counts as absent.
        enabled: bool,
        /// 0 to 1.
        opacity: Num,
        /// Feather, 0 to 1.
        feather: Num,
        /// Grow, -1 to 1.
        grow: Num,
        /// Refine edges, 0 to 1.
        refine: Num,
    },
    /// The blur map with a level and a softness.
    Background {
        /// Add or subtract.
        mode: MaskMode,
        /// This mask alone is inverted.
        invert: bool,
        /// Off (the eye) counts as absent.
        enabled: bool,
        /// 0 to 1.
        opacity: Num,
        /// Where the blur map's own decision falls, 0 to 1.
        level: Num,
        /// Softness of the threshold, 0 to 1.
        softness: Num,
        /// Feather, 0 to 1.
        feather: Num,
        /// Grow, -1 to 1.
        grow: Num,
        /// Refine edges, 0 to 1.
        refine: Num,
    },
    /// A gradient over the depth map, full at the nearest part and none at a far limit (`docs/23` section 4.8).
    Depth {
        /// Add or subtract.
        mode: MaskMode,
        /// This mask alone is inverted.
        invert: bool,
        /// Off (the eye) counts as absent.
        enabled: bool,
        /// 0 to 1.
        opacity: Num,
        /// Where the mask fades out in the photo's depth order, 0.02 to 1: full at the nearest part of the photo, none at
        /// this depth and beyond. 1 is the whole depth.
        far: Num,
        /// Feather, 0 to 1.
        feather: Num,
        /// Grow, -1 to 1.
        grow: Num,
        /// Refine edges, 0 to 1.
        refine: Num,
    },
    /// A line, an angle and a feather.
    Linear {
        /// Add or subtract.
        mode: MaskMode,
        /// This mask alone is inverted.
        invert: bool,
        /// Off (the eye) counts as absent.
        enabled: bool,
        /// 0 to 1.
        opacity: Num,
        /// A point of the line in the photo frame.
        center: [Num; 2],
        /// The line's angle in degrees, -180 to 180.
        angle_deg: Num,
        /// Feather, 0 to 1.
        feather: Num,
    },
    /// An ellipse.
    Radial {
        /// Add or subtract.
        mode: MaskMode,
        /// This mask alone is inverted.
        invert: bool,
        /// Off (the eye) counts as absent.
        enabled: bool,
        /// 0 to 1.
        opacity: Num,
        /// The centre in the photo frame.
        center: [Num; 2],
        /// The two radii, 0.002 to 1 of the long edge.
        radii: [Num; 2],
        /// The angle in degrees, -180 to 180.
        angle_deg: Num,
        /// Feather, 0 to 1.
        feather: Num,
    },
    /// Strokes.
    Brush {
        /// Add or subtract.
        mode: MaskMode,
        /// This mask alone is inverted.
        invert: bool,
        /// Off (the eye) counts as absent.
        enabled: bool,
        /// 0 to 1.
        opacity: Num,
        /// The strokes in the order they were drawn, at most 100,000 points in all.
        strokes: Vec<BrushStroke>,
        /// Feather, 0 to 1.
        feather: Num,
        /// Grow, -1 to 1.
        grow: Num,
        /// Refine edges, 0 to 1.
        refine: Num,
    },
    /// A chroma range.
    ColorRange {
        /// Add or subtract.
        mode: MaskMode,
        /// This mask alone is inverted.
        invert: bool,
        /// Off (the eye) counts as absent.
        enabled: bool,
        /// 0 to 1.
        opacity: Num,
        /// The centre in the chroma plane of C9, at most 0.5 long.
        center: [Num; 2],
        /// The width, 0.005 to 0.3.
        width: Num,
        /// The softness, 0 to 1.
        softness: Num,
    },
    /// A brightness range.
    BrightnessRange {
        /// Add or subtract.
        mode: MaskMode,
        /// This mask alone is inverted.
        invert: bool,
        /// Off (the eye) counts as absent.
        enabled: bool,
        /// 0 to 1.
        opacity: Num,
        /// The low end, 0 to 1.
        low: Num,
        /// The high end, 0 to 1.
        high: Num,
        /// The softness, 0 to 1.
        softness: Num,
    },
}

/// A node's masks: its Invert and the rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct MaskSettings {
    /// The node's Invert: `m = 1 - m`.
    pub invert: bool,
    /// The rows, in the order they are drawn and combined, at most 64.
    pub items: Vec<MaskItem>,
}

/// The Input: the raw, light and framing settings (`docs/23` section 3.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct InputNode {
    /// `"input"`.
    pub id: NodeId,
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
    /// Light › Exposure: Auto or an absolute value, -4 to +4 EV.
    pub exposure: ExposureSetting,
    /// Colour › White balance, the light presets.
    pub light_source: LightSource,
    /// Colour › White balance: Auto or Kelvin and tint.
    pub white_balance: WhiteBalanceSetting,
    /// Pick white: the sampled point `[x, y]` in [0, 1]², measured on the photo as the camera
    /// oriented it.
    pub white_pick: Option<[Num; 2]>,
    /// RAW › Range: the highlight range and the low-light toe.
    pub rolloff: Rolloff,
    /// RAW › Decode switches.
    pub decode: DecodeOptions,
    /// Detail › Noise reduction: Auto or the three strengths.
    pub noise_reduction: NoiseReductionSetting,
    /// Detail › Sharpening: Auto or amount and radius.
    pub sharpen: SharpenSetting,
    /// Look › Lens: the lens and its three amounts.
    pub lens: LensSettings,
    /// Crop; null when the photo isn't cropped.
    pub crop: Option<Crop>,
    /// Orientation after the camera's EXIF orientation.
    pub orientation: Orientation,
}

/// The Output: the look and the output sharpening.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct OutputNode {
    /// `"output"`.
    pub id: NodeId,
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
    /// Look tab.
    pub look: LookId,
    /// 0 to 100 percent.
    pub look_intensity: Num,
    /// Detail › Sharpening › Output, in percent, 0 to 200, default 100.
    pub output_sharpening: Num,
}

/// A corrector node, Main (id 1) included.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct CorrectorNode {
    /// 1 to 2147483647 (Main is 1).
    pub id: NodeId,
    /// The card's name, 1 to 32 characters.
    pub name: String,
    /// A bypassed node passes its picture through.
    pub bypass: bool,
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
    /// The trims.
    pub trim: NodeTrim,
    /// The grade.
    pub grade: NodeGrade,
    /// The masks.
    pub masks: MaskSettings,
}

/// A Film node's `film`: the negative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct FilmStockSettings {
    /// The profile id of a camera film (`ym film list`).
    pub stock: String,
    /// Stops, -4 to 4; 0 is the Input's exposure.
    pub exposure_ev: Num,
}

/// The enlarger's filtration, as shifts from neutral in Kodak CC units, -50 to 50.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct PrintLights {
    /// Cyan.
    pub c: Num,
    /// Magenta.
    pub m: Num,
    /// Yellow.
    pub y: Num,
}

/// A Film node's `print`: the enlarger and the paper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct PrintSettings {
    /// The paper's profile id; null is the stock's matching paper (a slide takes none).
    pub paper: Option<String>,
    /// Stops, -3 to 3; 0 keeps the app's brightness.
    pub exposure_ev: Num,
    /// The enlarger's filtration.
    pub lights: PrintLights,
}

/// A Film node's `halation`, in percent (100 is the stock's own).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct HalationSettings {
    /// 0 to 300.
    pub amount: Num,
    /// 50 to 200.
    pub size: Num,
}

/// A Film node's `grain`, in percent (100 is the stock's own).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct GrainSettings {
    /// 0 to 200.
    pub amount: Num,
    /// 50 to 200.
    pub size: Num,
    /// Not in the interface; it keeps the grain a function of the settings.
    pub seed: u32,
}

/// A Film node: a colour negative, an enlarger, a print and a scan (`docs/24` section 4.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct FilmNode {
    /// 2 to 2147483647 (correctors and Film nodes share the numbers).
    pub id: NodeId,
    /// The card's name, 1 to 32 characters.
    pub name: String,
    /// A bypassed node passes its picture through.
    pub bypass: bool,
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
    /// The negative.
    pub film: FilmStockSettings,
    /// The enlarger and the paper.
    pub print: PrintSettings,
    /// Halation.
    pub halation: HalationSettings,
    /// Grain.
    pub grain: GrainSettings,
    /// How much of the film look is applied, 0 to 100 percent.
    pub strength: Num,
    /// The masks.
    pub masks: MaskSettings,
}

/// The amounts of the Face Refinement tools for one face (`docs/27` section 3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct FaceSettings {
    /// Blemish, 0 to 100 (default 30).
    pub blemish: Num,
    /// Dark circles, 0 to 100 (default 30).
    pub dark_circles: Num,
    /// Even skin, 0 to 100 (default 25).
    pub even_skin: Num,
    /// Texture, -100 to 100 (default 0).
    pub texture: Num,
    /// Skin grading › Contrast, -100 to 100 (default 0).
    pub skin_contrast: Num,
    /// Skin grading › Shadow desaturation, 0 to 100 (default 0).
    pub shadow_desaturation: Num,
    /// Skin grading › Shine removal, 0 to 100 (default 0).
    pub shine_removal: Num,
    /// Teeth › Whitening, 0 to 100 (default 0).
    pub teeth_whitening: Num,
}

/// A plane of brush strokes, in the format of a Brush mask row: a Face Refinement node's protection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct BrushMask {
    /// The strokes in the order they were drawn, at most 100,000 points in all.
    pub strokes: Vec<BrushStroke>,
    /// Feather, 0 to 1.
    pub feather: Num,
    /// Grow, -1 to 1.
    pub grow: Num,
    /// Refine edges, 0 to 1.
    pub refine: Num,
}

/// A Face Refinement node: retouches the faces it finds (`docs/27`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct FaceRefinementNode {
    /// 2 to 2147483647 (every numbered node shares the numbers).
    pub id: NodeId,
    /// The card's name, 1 to 32 characters.
    pub name: String,
    /// A bypassed node passes its picture through.
    pub bypass: bool,
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
    /// "Same for all faces": `all` serves every face and `faces` is kept but not used.
    pub same_for_all: bool,
    /// The amounts every face takes while `same_for_all` is on, and when `faces` is empty.
    pub all: FaceSettings,
    /// The amounts per face; index = face number - 1 (faces are numbered from 1, left to right), at most 32. A
    /// face past the end takes face 1's.
    pub faces: Vec<FaceSettings>,
    /// The protection plane: painted areas are left alone by every tool.
    pub protect: BrushMask,
    /// The masks.
    pub masks: MaskSettings,
}

/// How a Layer blends an input over the ones below it (`docs/25` section 3.3), in linear light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum BlendMode {
    /// The layer itself.
    Normal,
    /// Below plus the layer.
    Add,
    /// Below minus the layer.
    Subtract,
    /// Below times the layer.
    Multiply,
    /// The lighter of the two.
    Lighten,
    /// The darker of the two.
    Darken,
    /// The absolute difference.
    Difference,
}

/// A Parallel mixer: adds the change each of its inputs makes to the picture they started from (`docs/25`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct ParallelNode {
    /// 2 to 2147483647 (every numbered node shares the numbers).
    pub id: NodeId,
    /// The card's name, 1 to 32 characters.
    pub name: String,
    /// A bypassed Parallel outputs its base.
    pub bypass: bool,
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
}

/// A Layer mixer: stacks its inputs and blends each over the ones below it (`docs/25`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct LayerNode {
    /// 2 to 2147483647 (every numbered node shares the numbers).
    pub id: NodeId,
    /// The card's name, 1 to 32 characters.
    pub name: String,
    /// A bypassed Layer outputs its input 1.
    pub bypass: bool,
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
    /// How the layers blend.
    pub mode: BlendMode,
    /// 0 to 1.
    pub opacity: Num,
}

/// A Splitter: sends the red, green and blue of a picture to three outputs (`docs/25`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct SplitterNode {
    /// 2 to 2147483647 (every numbered node shares the numbers).
    pub id: NodeId,
    /// The card's name, 1 to 32 characters.
    pub name: String,
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
}

/// A Combiner: joins three pictures into one, a channel from each (`docs/25`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct CombinerNode {
    /// 2 to 2147483647 (every numbered node shares the numbers).
    pub id: NodeId,
    /// The card's name, 1 to 32 characters.
    pub name: String,
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
}

/// A node of the graph, told by its `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeItem {
    /// The Input.
    Input(InputNode),
    /// The Output.
    Output(OutputNode),
    /// A corrector.
    Corrector(CorrectorNode),
    /// A Film node.
    Film(FilmNode),
    /// A Face Refinement node.
    FaceRefinement(FaceRefinementNode),
    /// A Parallel mixer.
    Parallel(ParallelNode),
    /// A Layer mixer.
    Layer(LayerNode),
    /// A Splitter.
    Splitter(SplitterNode),
    /// A Combiner.
    Combiner(CombinerNode),
}

/// The graph of a photo: its nodes and the links between them (`docs/23` section 3.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct NodesSettings {
    /// The nodes: the Input, the correctors by id, the Output.
    pub items: Vec<NodeItem>,
    /// The links.
    pub links: Vec<NodeLink>,
}

/// The settings of a photo: the graph of its nodes (`docs/23` section 3). Every control in the UI and every
/// CLI flag is a field of one node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct DevelopSettings {
    /// The graph.
    pub nodes: NodesSettings,
}

#[cfg(test)]
impl DevelopSettings {
    /// The Input (tests).
    pub fn input(&self) -> &InputNode {
        self.nodes
            .items
            .iter()
            .find_map(|i| match i {
                NodeItem::Input(n) => Some(n),
                _ => None,
            })
            .expect("a graph has its Input")
    }

    /// Main, node 1 (tests).
    pub fn main(&self) -> &CorrectorNode {
        self.nodes
            .items
            .iter()
            .find_map(|i| match i {
                NodeItem::Corrector(n) if n.id == NodeId(ym_core::settings::NodeId::MAIN) => {
                    Some(n)
                }
                _ => None,
            })
            .expect("a graph has Main")
    }

    /// Main's grade (tests).
    pub fn main_grade(&self) -> &NodeGrade {
        &self.main().grade
    }

    /// The Output (tests).
    pub fn output(&self) -> &OutputNode {
        self.nodes
            .items
            .iter()
            .find_map(|i| match i {
                NodeItem::Output(n) => Some(n),
                _ => None,
            })
            .expect("a graph has its Output")
    }
}

/// Reads a field that may be absent (leave it), `null` (clear it) or a value (set it).
fn double_option<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

/// `white_balance` in a patch: Auto, or a manual value that may leave out Kelvin or tint (the
/// missing part keeps its current value).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WhiteBalancePatch {
    /// Back to Auto.
    Auto,
    /// Manual, with the parts to change.
    Manual {
        /// New Kelvin.
        #[serde(default)]
        kelvin: Option<Num>,
        /// New tint.
        #[serde(default)]
        tint: Option<Num>,
    },
}

/// `sharpen` in a patch: Auto, or a manual value that may leave out a part.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SharpenPatch {
    /// Back to Auto.
    Auto,
    /// Manual, with the parts to change.
    Manual {
        /// New amount.
        #[serde(default)]
        amount: Option<Num>,
        /// New radius in pixels.
        #[serde(default)]
        radius_px: Option<Num>,
    },
}

/// `noise_reduction` in a patch: Auto, or a manual value that may leave out a part.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NoiseReductionPatch {
    /// Back to Auto.
    Auto,
    /// Manual, with the parts to change.
    Manual {
        /// New luminance strength.
        #[serde(default)]
        luminance: Option<Num>,
        /// New colour strength.
        #[serde(default)]
        color: Option<Num>,
        /// New detail strength.
        #[serde(default)]
        detail: Option<Num>,
    },
}

/// Decode switches to change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct DecodePatch {
    /// New neural demosaic switch.
    #[serde(default)]
    pub neural_demosaic: Option<bool>,
    /// New highlight recovery switch.
    #[serde(default)]
    pub highlight_recovery: Option<bool>,
    /// New lens corrections switch.
    #[serde(default)]
    pub lens_corrections: Option<bool>,
    /// New chromatic aberration switch.
    #[serde(default)]
    pub chromatic_aberration: Option<bool>,
}

/// RAW › Range amounts to change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct RolloffPatch {
    /// New highlight range.
    #[serde(default)]
    pub highlights: Option<Num>,
    /// New low-light toe.
    #[serde(default)]
    pub low_light: Option<Num>,
}

/// Look › Lens parts to change; the others stay as they are.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct LensPatch {
    /// New lens. The amounts do not follow it: send the lens's starting amounts with it.
    #[serde(default)]
    pub kind: Option<LensKind>,
    /// New Flares.
    #[serde(default)]
    pub flares: Option<Num>,
    /// New Edges.
    #[serde(default)]
    pub edges: Option<Num>,
    /// New Anamorphic.
    #[serde(default)]
    pub anamorphic: Option<Num>,
}

/// Colour-balance wheels to change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct ColorBalancePatch {
    /// New shadows wheel: Auto, a manual point, or a plain point (`[0, 0]` is Auto).
    #[serde(default)]
    pub shadows: Option<BalancePatch>,
    /// New midtones wheel.
    #[serde(default)]
    pub midtones: Option<[Num; 2]>,
    /// New highlights wheel: Auto, a manual point, or a plain point (`[0, 0]` is Auto).
    #[serde(default)]
    pub highlights: Option<BalancePatch>,
}

/// Curves to replace.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct CurvesPatch {
    /// New luma curve.
    #[serde(default)]
    pub luma: Option<Vec<[Num; 2]>>,
    /// New red curve.
    #[serde(default)]
    pub red: Option<Vec<[Num; 2]>>,
    /// New green curve.
    #[serde(default)]
    pub green: Option<Vec<[Num; 2]>>,
    /// New blue curve.
    #[serde(default)]
    pub blue: Option<Vec<[Num; 2]>>,
}

/// A Film node's `film` to change. Serialising leaves absent fields out (see [`NodePatch`]).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct FilmStockPatch {
    /// New stock (a profile id).
    #[serde(default)]
    pub stock: Option<String>,
    /// New film exposure in stops.
    #[serde(default)]
    pub exposure_ev: Option<Num>,
}

/// A Film node's enlarger lights to change. Serialising leaves absent fields out (see [`NodePatch`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct LightsPatch {
    /// New cyan.
    #[serde(default)]
    pub c: Option<Num>,
    /// New magenta.
    #[serde(default)]
    pub m: Option<Num>,
    /// New yellow.
    #[serde(default)]
    pub y: Option<Num>,
}

/// A Film node's `print` to change. Serialising leaves absent fields out (see [`NodePatch`]).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct PrintPatch {
    /// New paper; `null` is the stock's matching paper.
    #[serde(default, deserialize_with = "double_option")]
    #[specta(type = Option<String>)]
    pub paper: Option<Option<String>>,
    /// New print exposure in stops.
    #[serde(default)]
    pub exposure_ev: Option<Num>,
    /// Lights to change.
    #[serde(default)]
    pub lights: Option<LightsPatch>,
}

/// A Film node's `halation` to change. Serialising leaves absent fields out (see [`NodePatch`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct HalationPatch {
    /// New amount, percent.
    #[serde(default)]
    pub amount: Option<Num>,
    /// New size, percent.
    #[serde(default)]
    pub size: Option<Num>,
}

/// A Film node's `grain` to change. Serialising leaves absent fields out (see [`NodePatch`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct GrainPatch {
    /// New amount, percent.
    #[serde(default)]
    pub amount: Option<Num>,
    /// New size, percent.
    #[serde(default)]
    pub size: Option<Num>,
    /// New seed.
    #[serde(default)]
    pub seed: Option<u32>,
}

/// A Face Refinement node's `all` to change. Serialising leaves absent fields out (see [`NodePatch`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct FaceSettingsPatch {
    /// New Blemish, 0 to 100.
    #[serde(default)]
    pub blemish: Option<Num>,
    /// New Dark circles, 0 to 100.
    #[serde(default)]
    pub dark_circles: Option<Num>,
    /// New Even skin, 0 to 100.
    #[serde(default)]
    pub even_skin: Option<Num>,
    /// New Texture, -100 to 100.
    #[serde(default)]
    pub texture: Option<Num>,
    /// New Skin contrast, -100 to 100.
    #[serde(default)]
    pub skin_contrast: Option<Num>,
    /// New Shadow desaturation, 0 to 100.
    #[serde(default)]
    pub shadow_desaturation: Option<Num>,
    /// New Shine removal, 0 to 100.
    #[serde(default)]
    pub shine_removal: Option<Num>,
    /// New Teeth whitening, 0 to 100.
    #[serde(default)]
    pub teeth_whitening: Option<Num>,
}

/// A Face Refinement node's `protect` to change: the strokes of a Brush row under the same names, and the three edge
/// amounts. Serialising leaves absent fields out (see [`NodePatch`]).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct ProtectPatch {
    /// The strokes, replaced whole.
    #[serde(default)]
    pub strokes: Option<Vec<BrushStroke>>,
    /// Strokes added after the existing ones.
    #[serde(default)]
    pub strokes_append: Option<Vec<BrushStroke>>,
    /// Keep only the first n strokes (the inverse of an append).
    #[serde(default)]
    pub strokes_keep: Option<u32>,
    /// Feather, 0 to 1.
    #[serde(default)]
    pub feather: Option<Num>,
    /// Grow, -1 to 1.
    #[serde(default)]
    pub grow: Option<Num>,
    /// Refine edges, 0 to 1.
    #[serde(default)]
    pub refine: Option<Num>,
}

/// A corrector's trims to change. Serialising leaves absent fields out (see [`NodePatch`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct TrimPatch {
    /// New exposure trim in stops.
    #[serde(default)]
    pub exposure_ev: Option<Num>,
    /// New temperature trim.
    #[serde(default)]
    pub temperature: Option<Num>,
    /// New tint trim.
    #[serde(default)]
    pub tint: Option<Num>,
}

/// A corrector's grade to change: the old patch's fields, under the same names. Serialising leaves
/// absent fields out (see [`NodePatch`]).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct GradePatch {
    /// New contrast: Auto, a manual value, or a plain number (0 is Auto).
    #[serde(default)]
    pub contrast: Option<PercentPatch>,
    /// New brightness.
    #[serde(default)]
    pub brightness: Option<Num>,
    /// New highlights: Auto, a manual value, or a plain number (0 is Auto).
    #[serde(default)]
    pub highlights: Option<PercentPatch>,
    /// New shadows: Auto, a manual value, or a plain number (0 is Auto).
    #[serde(default)]
    pub shadows: Option<PercentPatch>,
    /// New white.
    #[serde(default)]
    pub white: Option<Num>,
    /// New black: Auto, a manual value, or a plain number (0 is Auto).
    #[serde(default)]
    pub black: Option<PercentPatch>,
    /// Colour-balance wheels to change.
    #[serde(default)]
    pub color_balance: Option<ColorBalancePatch>,
    /// New saturation: Auto, a manual value, or a plain number (0 is Auto).
    #[serde(default)]
    pub saturation: Option<PercentPatch>,
    /// New colour warp: the whole list of pins (`{ pins: [] }` removes them all).
    #[serde(default)]
    pub color_warp: Option<ColorWarp>,
    /// Curves to replace.
    #[serde(default)]
    pub curve: Option<CurvesPatch>,
}

/// Serialises a patch struct as one entry per field that is present.
macro_rules! serialize_present {
    ($ty:ty, $($field:ident),* $(,)?) => {
        impl Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(None)?;
                $(if let Some(v) = &self.$field { map.serialize_entry(stringify!($field), v)?; })*
                map.end()
            }
        }
    };
}

serialize_present!(FilmStockPatch, stock, exposure_ev);
serialize_present!(LightsPatch, c, m, y);
serialize_present!(PrintPatch, paper, exposure_ev, lights);
serialize_present!(HalationPatch, amount, size);
serialize_present!(GrainPatch, amount, size, seed);
serialize_present!(
    FaceSettingsPatch,
    blemish,
    dark_circles,
    even_skin,
    texture,
    skin_contrast,
    shadow_desaturation,
    shine_removal,
    teeth_whitening
);
serialize_present!(
    ProtectPatch,
    strokes,
    strokes_append,
    strokes_keep,
    feather,
    grow,
    refine
);

impl Serialize for TrimPatch {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        macro_rules! put {
            ($($field:ident),* $(,)?) => {
                $(if let Some(v) = &self.$field { map.serialize_entry(stringify!($field), v)?; })*
            };
        }
        put!(exposure_ev, temperature, tint);
        map.end()
    }
}

impl Serialize for GradePatch {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        macro_rules! put {
            ($($field:ident),* $(,)?) => {
                $(if let Some(v) = &self.$field { map.serialize_entry(stringify!($field), v)?; })*
            };
        }
        put!(
            contrast,
            brightness,
            highlights,
            shadows,
            white,
            black,
            color_balance,
            saturation,
            color_warp,
            curve,
        );
        map.end()
    }
}

/// A partial of one mask row's fields (`masks.edit`, `docs/23` section 3.7). It cannot change the
/// row's kind.
///
/// Serialising leaves absent fields out, so that `box: null` (which takes the box away) is told apart from no `box`
/// at all; the derive cannot do that without `skip_serializing_if`, which the unified TypeScript export refuses,
/// hence the manual `Serialize`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct MaskPatch {
    /// The row's own kind, or nothing.
    #[serde(default)]
    pub kind: Option<String>,
    /// Add or subtract.
    #[serde(default)]
    pub mode: Option<MaskMode>,
    /// This mask alone is inverted.
    #[serde(default)]
    pub invert: Option<bool>,
    /// The eye.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// 0 to 1.
    #[serde(default)]
    pub opacity: Option<Num>,
    /// Feather.
    #[serde(default)]
    pub feather: Option<Num>,
    /// Grow.
    #[serde(default)]
    pub grow: Option<Num>,
    /// Refine edges.
    #[serde(default)]
    pub refine: Option<Num>,
    /// An Object row's clicks, replaced whole.
    #[serde(default)]
    pub points: Option<Vec<(Num, Num, bool)>>,
    /// An Object row's box; `null` takes it away.
    #[serde(default, rename = "box", deserialize_with = "double_option")]
    #[specta(type = Option<[Num; 4]>)]
    pub box_: Option<Option<[Num; 4]>>,
    /// A Background row's level.
    #[serde(default)]
    pub level: Option<Num>,
    /// A Depth row's `far`.
    #[serde(default)]
    pub far: Option<Num>,
    /// The softness of a Background, Colour range or Brightness range row.
    #[serde(default)]
    pub softness: Option<Num>,
    /// A Linear, Radial or Colour range row's centre.
    #[serde(default)]
    pub center: Option<[Num; 2]>,
    /// A Linear or Radial row's angle.
    #[serde(default)]
    pub angle_deg: Option<Num>,
    /// A Radial row's radii.
    #[serde(default)]
    pub radii: Option<[Num; 2]>,
    /// A Brush row's strokes, replaced whole.
    #[serde(default)]
    pub strokes: Option<Vec<BrushStroke>>,
    /// Strokes added after the existing ones (a Brush row).
    #[serde(default)]
    pub strokes_append: Option<Vec<BrushStroke>>,
    /// Keep only the first n strokes (a Brush row; the inverse of an append).
    #[serde(default)]
    pub strokes_keep: Option<u32>,
    /// A Colour range row's width.
    #[serde(default)]
    pub width: Option<Num>,
    /// A Brightness range row's low end.
    #[serde(default)]
    pub low: Option<Num>,
    /// A Brightness range row's high end.
    #[serde(default)]
    pub high: Option<Num>,
}

impl Serialize for MaskPatch {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        macro_rules! put {
            ($($field:ident => $key:expr),* $(,)?) => {
                $(if let Some(v) = &self.$field { map.serialize_entry($key, v)?; })*
            };
        }
        put!(
            kind => "kind", mode => "mode", invert => "invert", enabled => "enabled",
            opacity => "opacity", feather => "feather", grow => "grow", refine => "refine",
            points => "points", box_ => "box", level => "level", far => "far",
            softness => "softness", center => "center", angle_deg => "angle_deg", radii => "radii",
            strokes => "strokes",
            strokes_append => "strokes_append", strokes_keep => "strokes_keep", width => "width",
            low => "low", high => "high",
        );
        map.end()
    }
}

/// A node's masks to change: its Invert, the whole list of rows, and rows by index. Serialising
/// leaves absent fields out (see [`NodePatch`]).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct MasksPatch {
    /// The node's Invert.
    #[serde(default)]
    pub invert: Option<bool>,
    /// The whole list of rows: add, remove or reorder rows.
    #[serde(default)]
    pub items: Option<Vec<MaskItem>>,
    /// Row patches by the row's index (a string), applied after `items`.
    #[serde(default)]
    pub edit: Option<std::collections::BTreeMap<String, MaskPatch>>,
}

impl Serialize for MasksPatch {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        macro_rules! put {
            ($($field:ident),* $(,)?) => {
                $(if let Some(v) = &self.$field { map.serialize_entry(stringify!($field), v)?; })*
            };
        }
        put!(invert, items, edit);
        map.end()
    }
}

/// A partial of one node's fields (`docs/23` section 3.7), the fields of every kind in one flat
/// object: one the node's kind does not have is refused by the core. A node the photo does not
/// have is added when `kind` is given.
///
/// Serialising leaves absent fields out (so that `crop: null`, which clears the crop, is told apart from
/// no `crop` at all); the derive cannot do that without `skip_serializing_if`, which the unified
/// TypeScript export refuses, hence the manual `Serialize`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct NodePatch {
    /// The node's kind: needed to add a node.
    #[serde(default)]
    pub kind: Option<NodeKind>,
    /// The card's place.
    #[serde(default)]
    pub pos: Option<[Num; 2]>,
    /// New exposure (Manual to set it, Auto to reset it).
    #[serde(default)]
    pub exposure: Option<ExposureSetting>,
    /// New light preset.
    #[serde(default)]
    pub light_source: Option<LightSource>,
    /// New white balance; a manual one may leave out Kelvin or tint.
    #[serde(default)]
    pub white_balance: Option<WhiteBalancePatch>,
    /// New white pick; `null` clears it. A pick also sets `light_source` and `white_balance` to
    /// Auto unless the same patch says otherwise.
    #[serde(default, deserialize_with = "double_option")]
    #[specta(type = Option<[Num; 2]>)]
    pub white_pick: Option<Option<[Num; 2]>>,
    /// RAW › Range amounts to change.
    #[serde(default)]
    pub rolloff: Option<RolloffPatch>,
    /// Decode switches to change.
    #[serde(default)]
    pub decode: Option<DecodePatch>,
    /// New noise reduction; a manual one may leave out a part.
    #[serde(default)]
    pub noise_reduction: Option<NoiseReductionPatch>,
    /// New sharpening; a manual one may leave out a part.
    #[serde(default)]
    pub sharpen: Option<SharpenPatch>,
    /// Lens parts to change.
    #[serde(default)]
    pub lens: Option<LensPatch>,
    /// New crop; `null` removes it.
    #[serde(default, deserialize_with = "double_option")]
    #[specta(type = Option<Crop>)]
    pub crop: Option<Option<Crop>>,
    /// New orientation.
    #[serde(default)]
    pub orientation: Option<Orientation>,
    /// New look.
    #[serde(default)]
    pub look: Option<LookId>,
    /// New look intensity.
    #[serde(default)]
    pub look_intensity: Option<Num>,
    /// New output sharpening amount, percent.
    #[serde(default)]
    pub output_sharpening: Option<Num>,
    /// New name.
    #[serde(default)]
    pub name: Option<String>,
    /// Bypass on or off.
    #[serde(default)]
    pub bypass: Option<bool>,
    /// Trims to change.
    #[serde(default)]
    pub trim: Option<TrimPatch>,
    /// Grade fields to change.
    #[serde(default)]
    pub grade: Option<GradePatch>,
    /// Masks to change.
    #[serde(default)]
    pub masks: Option<MasksPatch>,
    /// A Film node's negative to change.
    #[serde(default)]
    pub film: Option<FilmStockPatch>,
    /// A Film node's enlarger and paper to change.
    #[serde(default)]
    pub print: Option<PrintPatch>,
    /// A Film node's halation to change.
    #[serde(default)]
    pub halation: Option<HalationPatch>,
    /// A Film node's grain to change.
    #[serde(default)]
    pub grain: Option<GrainPatch>,
    /// New Strength of a Film node, percent.
    #[serde(default)]
    pub strength: Option<Num>,
    /// A Face Refinement node's "Same for all faces" on or off.
    #[serde(default)]
    pub same_for_all: Option<bool>,
    /// A Face Refinement node's amounts for every face (while "Same for all faces" is on) to change.
    #[serde(default)]
    pub all: Option<FaceSettingsPatch>,
    /// A Face Refinement node's amounts per face, replaced whole (index = face number - 1).
    #[serde(default)]
    pub faces: Option<Vec<FaceSettings>>,
    /// A Face Refinement node's protection plane to change.
    #[serde(default)]
    pub protect: Option<ProtectPatch>,
    /// A Layer's new blend mode.
    #[serde(default)]
    pub mode: Option<BlendMode>,
    /// A Layer's new opacity, 0 to 1.
    #[serde(default)]
    pub opacity: Option<Num>,
}

impl Serialize for NodePatch {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        // One entry per field that is present; `Some(None)` is a present `null`.
        macro_rules! put {
            ($($field:ident),* $(,)?) => {
                $(if let Some(v) = &self.$field { map.serialize_entry(stringify!($field), v)?; })*
            };
        }
        put!(
            kind,
            pos,
            exposure,
            light_source,
            white_balance,
            white_pick,
            rolloff,
            decode,
            noise_reduction,
            sharpen,
            lens,
            crop,
            orientation,
            look,
            look_intensity,
            output_sharpening,
            name,
            bypass,
            trim,
            grade,
            masks,
            film,
            print,
            halation,
            grain,
            strength,
            same_for_all,
            all,
            faces,
            protect,
            mode,
            opacity,
        );
        map.end()
    }
}

/// A node-level partial update of a photo's settings (`update_settings(photo_id, patch)`, `docs/23`
/// section 3.7): `nodes` maps node ids (`"input"`, `"output"`, `"1"`, `"2"`, …) to what changes, `remove`
/// lists the corrector nodes to delete and `links`, when present, replaces the links whole.
///
/// Serialising leaves out what is absent; see [`NodePatch`].
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct SettingsPatch {
    /// Node patches by id.
    #[serde(default)]
    pub nodes: std::collections::BTreeMap<String, NodePatch>,
    /// The corrector nodes to delete.
    #[serde(default)]
    pub remove: Vec<NodeId>,
    /// The links, replacing the photo's whole.
    #[serde(default)]
    pub links: Option<Vec<NodeLink>>,
}

impl Serialize for SettingsPatch {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        if !self.nodes.is_empty() {
            map.serialize_entry("nodes", &self.nodes)?;
        }
        if !self.remove.is_empty() {
            map.serialize_entry("remove", &self.remove)?;
        }
        if let Some(links) = &self.links {
            map.serialize_entry("links", links)?;
        }
        map.end()
    }
}

/// Which of the Input's `Auto` fields are still automatic; the UI shows the AUTO tag for these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct InputAutoFlags {
    /// `exposure` is Auto.
    pub exposure: bool,
    /// `light_source` is Auto.
    pub light_source: bool,
    /// `white_balance` is Auto.
    pub white_balance: bool,
    /// `sharpen` is Auto.
    pub sharpen: bool,
    /// `noise_reduction` is Auto.
    pub noise_reduction: bool,
}

/// Which of Main's `Auto` fields are still automatic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct MainAutoFlags {
    /// `contrast` is Auto.
    pub contrast: bool,
    /// `highlights` is Auto.
    pub highlights: bool,
    /// `shadows` is Auto.
    pub shadows: bool,
    /// `black` is Auto.
    pub black: bool,
    /// The shadows colour-balance wheel is Auto.
    pub balance_shadows: bool,
    /// The highlights colour-balance wheel is Auto.
    pub balance_highlights: bool,
    /// `saturation` is Auto.
    pub saturation: bool,
}

/// Which `Auto` fields are still automatic, by node: only the Input and Main can be Auto.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AutoFlags {
    /// The Input's.
    pub input: InputAutoFlags,
    /// Main's.
    pub main: MainAutoFlags,
}

/// Light classes of the illuminant estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LightClass {
    /// Daylight.
    Daylight,
    /// Tungsten.
    Tungsten,
    /// Fluorescent.
    Fluorescent,
    /// LED.
    Led,
}

/// Poisson–Gaussian noise model, `variance = a · signal + b`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct NoiseModel {
    /// Shot-noise gain.
    pub a: Num,
    /// Read-noise variance.
    pub b: Num,
}

// ---------------------------------------------------------------------------------------------
// Auto providers (`docs/19`, `docs/22`)
// ---------------------------------------------------------------------------------------------

/// Which Auto made a photo's automatic values: Yellowmoon's own (`rule`), or the answer of Apple
/// Intelligence (`apple`), Claude Code (`claude`) or Antigravity (`agy`, `docs/22` section 14.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AutoSource {
    /// Default Auto.
    Rule,
    /// Apple Intelligence's answer (the photo's own or its burst's).
    Apple,
    /// Claude Code's answer.
    Claude,
    /// Antigravity's answer.
    Agy,
}

/// Who grades (`docs/22` section 1; the core's `AutoProvider`). The order is the dropdown's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AutoProvider {
    /// Default Auto, the rule.
    Yellowmoon,
    /// Apple Intelligence, the on-device model.
    Apple,
    /// Claude Code, a program the person installed and signed in to.
    Claude,
    /// Antigravity (`agy`), the same.
    Agy,
}

/// Whether the person agreed to send photos through each program (`docs/22` section 10.2; the
/// core's `AutoConsent`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AutoConsent {
    /// Claude Code.
    pub claude: bool,
    /// Antigravity.
    pub agy: bool,
}

/// Why a provider is not available on this computer (`docs/19` section 1 and `docs/22` section 3;
/// the core's `UnavailableReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AppleAutoReason {
    /// Not a Mac.
    NotAMac,
    /// A Mac with an Intel processor.
    IntelMac,
    /// This build has no bridge to Apple's model.
    BuiltWithoutSupport,
    /// macOS older than 27.
    MacosTooOld,
    /// The Mac cannot run Apple Intelligence.
    DeviceNotEligible,
    /// Apple Intelligence is off in System Settings.
    AppleIntelligenceOff,
    /// The model is still downloading.
    ModelNotReady,
    /// Any other unavailability Apple reports.
    Unavailable,
    /// The model on this Mac cannot take pictures.
    NoImageInput,
    /// No usable program was found.
    NotInstalled,
    /// The program runs but its version is below the minimum.
    TooOld,
    /// The program runs and is not signed in.
    SignedOut,
    /// The program was found but its check failed or timed out.
    ProbeFailed,
    /// Windows: only a script was found, which Yellowmoon does not run.
    UnsafeLauncher,
    /// The first check has not finished.
    Checking,
    /// A program is chosen and available but the person has not agreed to send photos through it.
    NotConsented,
}

/// Why one request to Apple's model failed (`docs/19` sections 6 and 8; the core's `FailureKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AppleAutoFailure {
    /// Apple Auto stopped being available.
    Unavailable,
    /// macOS asked the app to wait.
    RateLimited,
    /// No answer within 30 seconds.
    Timeout,
    /// The model refused.
    Refused,
    /// Apple's safety check stopped the request.
    Guardrail,
    /// The model does not support the language.
    UnsupportedLanguage,
    /// The request did not fit the model's context.
    ContextTooLarge,
    /// The model's assets were not there.
    AssetsUnavailable,
    /// Another request was running.
    ConcurrentRequests,
    /// The answer could not be read.
    BadAnswer,
    /// The program is not signed in.
    NotSignedIn,
    /// The program reported an error.
    CliFailed,
    /// Anything else; the message says what.
    Other,
}

/// Whether Apple's model read the default instructions or the person's own (`docs/19` section 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AppleAutoInstructions {
    /// The default instructions.
    Default,
    /// The person's own.
    Edited,
}

/// Which model answers, and with which instructions (the core's `AdvisorInfo`).
///
/// The core writes what a program says about itself as `tool: { version, api, auth }`; the webview
/// reads `tool_version` and `api` (`docs/22` section 12.6), so this type reads either shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Type)]
pub struct AppleAutoInfo {
    /// Whose grading it is.
    pub provider: AutoProvider,
    /// The model: Apple's variant display name ("AFM 3 Core"), or for the programs the id that
    /// answered, the alias asked for until one answered, or `default`.
    pub model: String,
    /// The macOS version and build: "27.0.1 (27A404)"; `""` for the programs.
    pub os: String,
    /// The version the program printed ("2.1.310"); null for Apple Intelligence.
    pub tool_version: Option<String>,
    /// Where Claude Code's requests go (`firstParty`: Anthropic); null for the others.
    pub api: Option<String>,
    /// The version of the prompt the provider is asked with.
    pub prompt: u32,
    /// The default instructions or the person's own.
    pub instructions: AppleAutoInstructions,
    /// The first 12 hex digits of the SHA-256 of the instructions.
    pub instructions_sha256: String,
}

impl<'de> Deserialize<'de> for AppleAutoInfo {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// What a program says about itself, as the core writes it.
        #[derive(Deserialize)]
        struct Tool {
            version: String,
            #[serde(default)]
            api: Option<String>,
        }
        #[derive(Deserialize)]
        struct Raw {
            provider: AutoProvider,
            model: String,
            os: String,
            #[serde(default)]
            tool: Option<Tool>,
            #[serde(default)]
            tool_version: Option<String>,
            #[serde(default)]
            api: Option<String>,
            prompt: u32,
            instructions: AppleAutoInstructions,
            instructions_sha256: String,
        }
        let raw = Raw::deserialize(deserializer)?;
        let (tool_version, api) = match raw.tool {
            Some(tool) => (Some(tool.version), tool.api),
            None => (raw.tool_version, raw.api),
        };
        Ok(Self {
            provider: raw.provider,
            model: raw.model,
            os: raw.os,
            tool_version,
            api,
            prompt: raw.prompt,
            instructions: raw.instructions,
            instructions_sha256: raw.instructions_sha256,
        })
    }
}

/// The colour Apple Auto gives the shadows (the core's `ShadowColor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AppleShadowColor {
    /// A neutral grade.
    None,
    /// Teal.
    Teal,
    /// Blue.
    Blue,
    /// Green.
    Green,
    /// Amber.
    Amber,
    /// Magenta.
    Magenta,
}

/// The colour Apple Auto gives the highlights (the core's `HighlightColor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AppleHighlightColor {
    /// A neutral grade.
    None,
    /// Warm.
    Warm,
    /// Gold.
    Gold,
    /// Cool.
    Cool,
    /// Pink.
    Pink,
}

/// The steps of an answer (the core's `AutoSteps`), each null when the model left the field empty:
/// exposure -3 to +3, the other numbers -2 to +2 (0 for keep), the colours of the shadows and the
/// highlights with their strengths 1 to 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AppleAutoSteps {
    /// Darker (negative) or brighter.
    pub exposure: Option<i8>,
    /// Darker (negative) or brighter highlights.
    pub highlights: Option<i8>,
    /// Darker (negative) or brighter shadows.
    pub shadows: Option<i8>,
    /// Deeper (negative) or lifted blacks.
    pub blacks: Option<i8>,
    /// Less (negative) or more contrast.
    pub contrast: Option<i8>,
    /// The model's warmth: cooler (negative) or warmer.
    pub white: Option<i8>,
    /// Greener (negative) or more magenta.
    pub tint: Option<i8>,
    /// Less (negative) or more saturation.
    pub saturation: Option<i8>,
    /// The colour of the shadows.
    pub shadow_color: Option<AppleShadowColor>,
    /// Its strength, 1 (subtle) to 3 (strong).
    pub shadow_color_strength: Option<i8>,
    /// The colour of the highlights.
    pub highlight_color: Option<AppleHighlightColor>,
    /// Its strength, 1 (subtle) to 3 (strong).
    pub highlight_color_strength: Option<i8>,
}

/// One answer of Apple's model (the core's `AdvisorAnswer`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AppleAutoAnswer {
    /// The steps.
    pub steps: AppleAutoSteps,
    /// What the photo shows, in a few words (for reports only).
    pub scene: String,
    /// The look the model gives the photo, in a few words (the Automatic card shows it).
    pub plan: String,
}

/// Where a photo stands with Apple Auto (the core's `AppleAuto`), tagged by `state`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AppleAuto {
    /// The preference is off, or the app has no advisor: Auto is the rule.
    Off,
    /// Apple Auto is not available on this computer: Auto is the rule.
    Unavailable {
        /// Why.
        reason: AppleAutoReason,
    },
    /// Asked, or waiting for its turn: Auto is the rule until the answer.
    Pending,
    /// Answered: Auto takes the answer's values.
    Answered {
        /// The answer.
        answer: AppleAutoAnswer,
        /// Which model answered.
        info: AppleAutoInfo,
    },
    /// The request failed: Auto is the rule.
    Failed {
        /// Why.
        kind: AppleAutoFailure,
        /// Apple's message, when there is one.
        message: String,
        /// Which model was asked.
        info: AppleAutoInfo,
    },
}

/// Where one provider stands (`docs/22` section 3.1; the core's `ProviderState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ProviderState {
    /// It can answer now.
    Available,
    /// No usable program was found.
    NotInstalled,
    /// The program runs but its version is below the minimum.
    TooOld,
    /// It runs and is not signed in.
    SignedOut,
    /// It cannot answer, for a reason.
    Unavailable,
    /// The first check has not finished.
    Checking,
}

/// One provider of [`AutoStatus`] (the core's `ProviderStatus`, without the program's path, which
/// never leaves the core).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct ProviderStatus {
    /// Which provider.
    pub provider: AutoProvider,
    /// Where it stands.
    pub state: ProviderState,
    /// Why it is not available; null when it is.
    pub reason: Option<AppleAutoReason>,
    /// Which model answers, when it is available.
    pub info: Option<AppleAutoInfo>,
    /// The program's version, also while it is too old or signed out.
    pub tool_version: Option<String>,
    /// Whether the person agreed to send photos through it.
    pub consent: bool,
    /// Whether the interface marks it Experimental (Antigravity).
    pub experimental: bool,
    /// Whether the dropdown lists it.
    pub listed: bool,
    /// Whether the dropdown lets the person choose it.
    pub selectable: bool,
}

/// `auto_status()` and `auto_status_cached()`: the four providers and which one grades now (`docs/22`
/// sections 11 and 12.6; the core's `AutoStatus`, which replaces `AppleAutoStatus`). Preferences
/// shows its "Auto grading" row while `visible`, or while the saved choice is a program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AutoStatus {
    /// Every provider, in the dropdown's order.
    pub providers: Vec<ProviderStatus>,
    /// The saved choice (`auto_provider`).
    pub preference: AutoProvider,
    /// The provider chosen for the session: the saved choice in the app.
    pub chosen: AutoProvider,
    /// The provider that grades now: the chosen one while it is available and agreed to, else
    /// `yellowmoon`.
    pub effective: AutoProvider,
    /// `effective` is not `yellowmoon`.
    pub active: bool,
    /// A provider is there: the dropdown and the instructions show (they also show while the saved
    /// choice is a program, so that it can be withdrawn).
    pub visible: bool,
    /// The saved consent.
    pub consent: AutoConsent,
    /// The table's version.
    pub table: u32,
    /// The default instructions or the person's own.
    pub instructions: AppleAutoInstructions,
    /// The first 12 hex digits of the SHA-256 of the instructions.
    pub instructions_sha256: String,
    /// The person's own instructions, `""` for the default.
    pub own_instructions: String,
    /// The default instructions, for Preferences' text field and its Reset to default.
    pub default_instructions: String,
    /// The longest instructions the models are given, in characters.
    pub instructions_max_chars: u32,
}

/// What Auto resolves to for the Input (`estimates.input`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct InputEstimates {
    /// `EV_auto` (with Apple Auto's change, when there is one).
    pub exposure_ev: Num,
    /// The Auto Kelvin (with Apple Auto's nudge, when there is one).
    pub kelvin: Num,
    /// The Auto tint.
    pub tint: Num,
    /// What `sharpen: Auto` resolves to.
    pub sharpen: SharpenValue,
    /// What `noise_reduction: Auto` resolves to.
    pub noise_reduction: NoiseReductionValue,
}

/// What Auto resolves to for Main (`estimates.main`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct MainEstimates {
    /// What `contrast: Auto` resolves to: 0, or Apple Auto's contrast.
    pub contrast: Num,
    /// What `saturation: Auto` resolves to: 0, or Apple Auto's saturation.
    pub saturation: Num,
    /// What `highlights: Auto` resolves to: 0, or Apple Auto's highlights.
    pub highlights: Num,
    /// What `shadows: Auto` resolves to: 0, or Apple Auto's shadows.
    pub shadows: Num,
    /// What `black: Auto` resolves to: 0, or Apple Auto's black level.
    pub black: Num,
    /// What the shadows wheel's Auto resolves to: `[0, 0]`, or Apple Auto's shadow colour.
    pub balance_shadows: [Num; 2],
    /// What the highlights wheel's Auto resolves to: `[0, 0]`, or Apple Auto's highlight colour.
    pub balance_highlights: [Num; 2],
}

/// Per-photo automatic estimates (`AutoEstimates` in `types.ts`; the core's `AutoSummary`), grouped by
/// the node each number is for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct AutoEstimates {
    /// What Auto resolves to for the Input.
    pub input: InputEstimates,
    /// What Auto resolves to for Main.
    pub main: MainEstimates,
    /// Which Auto made these values.
    pub source: AutoSource,
    /// The provider the photo's `apple` state belongs to (`docs/22` section 14.1): the chosen provider
    /// while it acts or is asked about, `yellowmoon` while the state is `off`.
    pub provider: AutoProvider,
    /// Where the photo stands with Apple Auto.
    pub apple: AppleAuto,
    /// Apple Auto's steps in effect (the photo's answer, or its burst's), null when Auto is the rule.
    pub apple_steps: Option<AppleAutoSteps>,
    /// The most probable light class.
    pub light_class: LightClass,
    /// Confidence of the light estimate, 0 to 1.
    pub light_confidence: Num,
    /// The noise model.
    pub noise: NoiseModel,
}

/// The Input with every `Auto` replaced by the number it resolves to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct ResolvedInput {
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
    /// The exposure in stops.
    pub exposure_ev: Num,
    /// `light_source`.
    pub light_source: LightSource,
    /// The Kelvin the White balance row shows.
    pub kelvin: Num,
    /// The tint the White balance row shows.
    pub tint: Num,
    /// The sharpening in effect.
    pub sharpen: SharpenValue,
    /// The noise reduction in effect.
    pub noise_reduction: NoiseReductionValue,
    /// `lens`.
    pub lens: LensSettings,
    /// `decode`.
    pub decode: DecodeOptions,
    /// `rolloff`.
    pub rolloff: Rolloff,
    /// `crop`.
    pub crop: Option<Crop>,
    /// `orientation`.
    pub orientation: Orientation,
    /// `white_pick`.
    pub white_pick: Option<[Num; 2]>,
}

/// The Output (it has no Auto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct ResolvedOutput {
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
    /// `look`.
    pub look: LookId,
    /// `look_intensity`.
    pub look_intensity: Num,
    /// `output_sharpening`.
    pub output_sharpening: Num,
}

/// A corrector's grade with every `Auto` replaced by the number it resolves to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct GradeValues {
    /// The contrast in effect: the manual value, or the photo's Auto contrast.
    pub contrast: Num,
    /// `brightness`.
    pub brightness: Num,
    /// The highlights in effect: the manual value, or the photo's Auto value.
    pub highlights: Num,
    /// The shadows in effect.
    pub shadows: Num,
    /// `white`.
    pub white: Num,
    /// The black in effect.
    pub black: Num,
    /// The wheels' points in effect.
    pub color_balance: ColorBalancePoints,
    /// The saturation in effect: the manual value, or the photo's Auto saturation.
    pub saturation: Num,
    /// `color_warp`.
    pub color_warp: ColorWarp,
    /// `curve`.
    pub curve: Curves,
}

/// A corrector with its grade resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct ResolvedCorrector {
    /// The node's id (Main is 1).
    pub id: u32,
    /// `name`.
    pub name: String,
    /// `bypass`.
    pub bypass: bool,
    /// The card's place on the canvas, in grid units.
    pub pos: [Num; 2],
    /// `trim`.
    pub trim: NodeTrim,
    /// The grade with its Auto resolved (only Main has Auto).
    pub grade: GradeValues,
    /// `masks`.
    pub masks: MaskSettings,
}

/// The settings with every `Auto` replaced by the number it resolves to, node by node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct ResolvedValues {
    /// The Input.
    pub input: ResolvedInput,
    /// Main, node 1.
    pub main: ResolvedCorrector,
    /// The other correctors, by id (their values are their fields: they have no Auto).
    pub nodes: Vec<ResolvedCorrector>,
    /// The Output.
    pub output: ResolvedOutput,
}

/// What `get_settings` and `update_settings` return: the panel's whole state for one photo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct ResolvedSettings {
    /// The photo.
    pub photo_id: PhotoId,
    /// The stored settings, with `Auto` kept as `Auto`.
    pub settings: DevelopSettings,
    /// The same settings with every `Auto` resolved.
    pub values: ResolvedValues,
    /// Which fields are still automatic.
    pub auto: AutoFlags,
    /// The automatic estimates the `Auto`s resolved against.
    pub estimates: AutoEstimates,
}

/// "Apply to" on the Look card, and the scope of `apply_look` and of an export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ApplyScope {
    /// The current photo.
    This,
    /// The picked photos.
    Picked,
    /// Every photo.
    All,
}

/// What `paste_settings` did to one photo: the fields it changed, as two patches that make one
/// undo step (`before` is what Undo sends, `after` what Redo sends). Both are empty when the photo
/// already had the pasted settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct PastedPhoto {
    /// The photo.
    pub id: PhotoId,
    /// The changed fields as they were.
    pub before: SettingsPatch,
    /// The same fields as they are now.
    pub after: SettingsPatch,
}

/// The answer of `paste_settings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct PasteReport {
    /// The photos that took the settings, in the order they were asked for.
    pub pasted: Vec<PastedPhoto>,
    /// The photos left out: still developing, failed or not in the session.
    pub skipped: Vec<PhotoId>,
}

// ---------------------------------------------------------------------------------------------
// Looks
// ---------------------------------------------------------------------------------------------

/// Where a look comes from. ARRI looks keep their numbered group ("3000 Film").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LookGroup {
    /// Shipped with the app.
    BuiltIn,
    /// One of ARRI's groups.
    Arri {
        /// The four-digit group number.
        no: u32,
        /// The group's name.
        name: String,
    },
    /// Imported by the user.
    User,
}

/// What a look's LUT expects as input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LookInput {
    /// LogC4 in ARRI Wide Gamut 4.
    Logc4,
    /// LogC3 (EI 800) in ARRI Wide Gamut 3.
    Logc3Ei800,
    /// LogC3 with Rec.709 output baked in.
    Logc3ToRec709,
}

/// The library a look is listed under: its section of the Look tab (`docs/08-looks.md`). In display
/// order; the core works it out (`ym_looks::library::Look::library`) so the app and `ym looks
/// list` agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub enum LookLibrary {
    /// Shipped with the app.
    #[serde(rename = "built_in")]
    BuiltIn,
    /// ARRI's LogC4 looks (the 3D-LUT library, or its `.alf4` version).
    #[serde(rename = "arri_logc4")]
    ArriLogC4,
    /// ARRI's LogC3 log-to-log 3D-LUT library.
    #[serde(rename = "arri_logc3")]
    ArriLogC3,
    /// ARRI's LogC3 to Rec.709 3D-LUT library.
    #[serde(rename = "arri_logc3_to_rec709")]
    ArriLogC3ToRec709,
    /// ARRI's ALF2 look files (`.aml`).
    #[serde(rename = "arri_alf2")]
    ArriAlf2,
    /// The person's own LUTs and look files.
    #[serde(rename = "user")]
    User,
}

/// The kind of file an imported look is kept as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum LookFormat {
    /// A `.cube` 3D LUT.
    Cube,
    /// An ARRI Look File for ALEXA 35 (`.alf4`).
    Alf4,
    /// An ARRI custom colour management file (`.alf4c`).
    Alf4c,
    /// An ARRI Look File for LogC3 cameras (`.aml`, "ALF2").
    Aml,
}

/// One entry of the look library.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct Look {
    /// The id.
    pub id: LookId,
    /// The display name.
    pub name: String,
    /// Where it is listed.
    pub group: LookGroup,
    /// What its LUT expects as input.
    pub input: LookInput,
    /// The library it belongs to: the Look tab shows one section per library.
    pub library: LookLibrary,
    /// The kind of file it is kept as; null for a built-in look.
    pub format: Option<LookFormat>,
    /// Points per axis of its 3D table; null for a built-in look.
    pub lut_size: Option<u32>,
    /// The user's default look.
    pub is_default: bool,
    /// The look's file is gone or its hash changed.
    pub missing: bool,
}

/// What `looks_import` reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct LooksImportResult {
    /// Looks added by this import; empty when everything was already there.
    pub added: Vec<Look>,
    /// Files whose hash matched a look that was already imported.
    pub unchanged: u32,
}

/// The three free look packages ARRI publishes (the ids of `data/looks/arri-packages.json`, with
/// underscores). The two look libraries that also come as ARRI Look Files (the LogC4 one as
/// `.alf4`, the LogC3 to Rec.709 one as `.aml`) are the same package here: a person who downloads
/// the look files has what they chose. ARRI's two display LUT packages are not offered: Yellowmoon
/// uses its own display rendering, and a display package that is chosen anyway is `unusable` with a
/// sentence that says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum PackageId {
    /// Look Library, LogC4 log-to-log.
    LookLibraryLogc4,
    /// Look Library, LogC3 log-to-log.
    LookLibraryLogc3,
    /// Look Library, LogC3 to Rec.709.
    LookLibraryLogc3ToRec709,
}

impl PackageId {
    /// The id the package has in `data/looks/arri-packages.json`.
    #[must_use]
    pub fn data_id(self) -> &'static str {
        match self {
            Self::LookLibraryLogc4 => "look-library-logc4",
            Self::LookLibraryLogc3 => "look-library-logc3",
            Self::LookLibraryLogc3ToRec709 => "look-library-logc3-rec709",
        }
    }

    /// The package for an id of `arri-packages.json`. The ids of the look-file libraries
    /// (`look-library-logc4-alf4`, `look-library-logc3-rec709-alf2`) give the package whose looks
    /// they are.
    #[must_use]
    pub fn from_data_id(id: &str) -> Option<Self> {
        let id = match id {
            "look-library-logc4-alf4" => "look-library-logc4",
            "look-library-logc3-rec709-alf2" => "look-library-logc3-rec709",
            other => other,
        };
        [
            Self::LookLibraryLogc4,
            Self::LookLibraryLogc3,
            Self::LookLibraryLogc3ToRec709,
        ]
        .into_iter()
        .find(|p| p.data_id() == id)
    }
}

/// What a chosen source is: a zip, an unpacked folder, or a single `.cube` or look file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// A zip archive.
    Zip,
    /// A folder.
    Folder,
    /// One `.cube` or look file.
    File,
}

/// What the recogniser made of one source (`looks_recognise`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Recognition {
    /// One of the five packages.
    Package {
        /// Which one.
        package_id: PackageId,
        /// "Look Library · LogC4".
        title: String,
        /// "Look Library · LogC4 · log to log".
        what: String,
        /// "87 .cube · 65-point".
        contents: String,
        /// Files that will be copied.
        file_count: u32,
        /// The person's own look files (`.alf4`, `.alf4c`, `.aml`) that sit in the same source
        /// and are copied too, as the person's own looks. Zero when there are none.
        #[serde(default)]
        look_file_count: u32,
        /// A remark that does not stop the import.
        note: Option<String>,
    },
    /// ARRI Look Files (`.alf4`, `.alf4c`, `.aml`) that are not part of a package: the person's own
    /// looks. Each file says what it expects, so nothing is asked.
    LookFiles {
        /// "ARRI Look Files · LogC4".
        what: String,
        /// "3 .alf4 · 33-point".
        contents: String,
        /// Files that will be copied.
        file_count: u32,
        /// What every file expects as input; `None` when they differ.
        input: Option<LookInput>,
        /// A remark that does not stop the import.
        note: Option<String>,
    },
    /// `.cube` files that match no package: the person says what they expect as input.
    NeedsAnswer {
        /// Files that need the answer.
        file_count: u32,
        /// The LUT size when every file has the same one.
        lut_size: Option<u32>,
        /// What the names suggest.
        suggestion: Option<LookInput>,
        /// The person's own look files that sit in the same source and are copied too. Zero when
        /// there are none.
        #[serde(default)]
        look_file_count: u32,
        /// Why the question is asked.
        note: String,
    },
    /// Nothing can be imported.
    Unusable {
        /// The plain explanation.
        reason: String,
        /// The package, when it is known but damaged.
        package_id: Option<PackageId>,
    },
    /// The file is not there.
    Missing {
        /// The plain explanation.
        reason: String,
    },
}

/// `looks_recognise(path)`: one source and what it is. Nothing is copied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct SourceRecognition {
    /// The path asked about.
    pub path: String,
    /// File or folder name.
    pub name: String,
    /// Zip, folder or file.
    pub source: SourceKind,
    /// Size on disk; null for a folder or a missing file.
    pub size_bytes: Option<Num>,
    /// What it is.
    pub recognition: Recognition,
}

/// `looks_import_progress`: files copied so far from one source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "looks_import_progress")]
pub struct LooksImportProgress {
    /// The source being imported.
    pub path: String,
    /// Files done.
    pub done: u32,
    /// Files in the source.
    pub total: u32,
    /// The file just copied, when known.
    pub file: Option<String>,
}

/// `looks_watch_downloads(enabled)`: whether the watcher is running, and why not when it was asked to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct WatchStatus {
    /// The watcher is running.
    pub watching: bool,
    /// Plain words, when macOS did not allow access to the Downloads folder.
    pub problem: Option<String>,
}

/// `download_found`: a zip whose name matches one of the packages has landed in Downloads and its
/// size has been stable for 2 s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "download_found")]
pub struct DownloadFound {
    /// Where the zip is.
    pub path: String,
    /// Its file name.
    pub file_name: String,
    /// Its size.
    pub size_bytes: Num,
    /// The package the name suggests, if exactly one.
    pub package_hint: Option<PackageId>,
}

// ---------------------------------------------------------------------------------------------
// Viewport
// ---------------------------------------------------------------------------------------------

/// The photo cell's rectangle in CSS pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct ViewportRect {
    /// Left edge in the window.
    pub x: Num,
    /// Top edge in the window.
    pub y: Num,
    /// Width in CSS pixels.
    pub width: Num,
    /// Height in CSS pixels.
    pub height: Num,
    /// Device pixels per CSS pixel.
    pub scale: Num,
}

/// Pointer input over the photo, forwarded to the core. Coordinates are CSS pixels in the cell.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ViewportInput {
    /// Wheel or pinch: multiply the zoom by `factor` about the point.
    Zoom {
        /// The factor, above 1 zooms in.
        factor: Num,
        /// Pointer x.
        x: Num,
        /// Pointer y.
        y: Num,
    },
    /// Drag: move the image with the pointer.
    Pan {
        /// Horizontal movement.
        dx: Num,
        /// Vertical movement.
        dy: Num,
    },
    /// Double click: toggle between Fit and 100 %.
    ToggleZoom {
        /// Pointer x.
        x: Num,
        /// Pointer y.
        y: Num,
    },
    /// Click in Pick white mode.
    PickWhite {
        /// Pointer x.
        x: Num,
        /// Pointer y.
        y: Num,
    },
}

/// Fit to the cell, or a fixed zoom (50, 100, 200 %).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ViewportMode {
    /// The whole image in the cell.
    Fit,
    /// A fixed zoom in percent.
    Percent {
        /// 50, 100, 200, ...
        value: Num,
    },
}

/// How the mask view shows a node's mask (`docs/23` section 5.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum MaskStyle {
    /// The mask tinted in the mask accent over the photo as it is.
    Overlay,
    /// The mask as grey: the code value after C13 is `m`.
    MaskOnly,
}

/// The mask view of the viewport (`viewport_set_mode`'s second argument): a corrector's mask shown in the
/// frame, over the photo or alone. `None` on the wire is no mask view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct MaskView {
    /// The corrector whose mask is shown.
    pub node: u32,
    /// Overlay or Mask only.
    pub style: MaskStyle,
    /// The row of that corrector whose own mask is shown instead of the corrector's (its place in the list, rows
    /// that are off included); `None` shows the corrector's mask (`docs/23` section 5.8).
    #[serde(default)]
    pub row: Option<u32>,
    /// Show depth (`docs/23` section 4.8): with `row` naming a Depth row, the frame shows the photo's depth map as grey, white
    /// for near, whole and not cut to the row's band, whatever `style` says. A row that is not a Depth row shows its own mask.
    #[serde(default)]
    pub depth: bool,
    /// Show skin areas (`docs/27` S-11c): with `node` naming a Face Refinement node, the frame shows face `face`'s skin areas tinted over the photo, whatever `style` says.
    #[serde(default)]
    pub face: Option<u32>,
}

/// A kind of mask row (`docs/23` section 3.3), as the row's `kind` is written: what `mask_kinds` lists and the Add
/// mask menu offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum MaskKind {
    /// SAM 2.1 from clicks or a box.
    Object,
    /// Apple Vision's foreground.
    Subject,
    /// Apple Vision's people.
    People,
    /// The blur map's out-of-focus area.
    Background,
    /// A gradient over the depth map: full at the nearest part, none at a far limit.
    Depth,
    /// A gradient across the photo.
    Linear,
    /// An oval with a soft edge.
    Radial,
    /// Strokes painted on the photo.
    Brush,
    /// A chroma centre, width and softness.
    ColorRange,
    /// A low, a high and a softness on the tones.
    BrightnessRange,
}

/// The answer of `sample_depth` (`docs/23` section 9, Pick depth): the far limit for the depth under the click, in the photo's
/// depth order (0 is the nearest part of the photo, 1 the farthest): the depth plus 0.1, at most 1. The page writes it as one
/// row patch (`masks.edit` with `far`), one undo step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct DepthFar {
    /// Where the mask fades out, 0.02 to 1.
    pub far: Num,
}

/// Why `focus_plane` has no far limit (`docs/23` section 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum FocusReason {
    /// The photo has no focus point (an older body, a panorama, a file that is not from a Sony camera). Answered at once,
    /// without waiting for the depth map: the card says "This photo has no focus point." and the button stays off.
    NoFocusPoint,
    /// The depth map could not be made.
    MapFailed,
}

/// The answer of `focus_plane` (`docs/23` section 9, Focus plane): with a limit, `far` as `sample_depth`'s and `reason` null;
/// without one, `far` null and the `reason`. The page writes the limit as one row patch, one undo step; `exact_centre` is true
/// when the camera's point is exactly the frame's centre, which Sony also writes when it does not know where it focused (the
/// card says so).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct FocusPlaneResult {
    /// Where the mask fades out, 0.02 to 1, or null when there is no limit.
    pub far: Option<Num>,
    /// The focus point is exactly the frame's centre.
    pub exact_centre: bool,
    /// Why there is no limit, or null when there is one.
    pub reason: Option<FocusReason>,
}

/// What the person did on the photo to find an object: `mask_find`'s `prompt` (`docs/23` section 9), in the photo frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaskFindPrompt {
    /// A click: on the object (`positive`), or on a part to take away.
    Point {
        /// Fraction of the photo's width.
        x: Num,
        /// Fraction of the photo's height.
        y: Num,
        /// On the object, or off it.
        positive: bool,
    },
    /// A box dragged around the object.
    Box {
        /// Left, as a fraction of the width.
        x0: Num,
        /// Top, as a fraction of the height.
        y0: Num,
        /// Right.
        x1: Num,
        /// Bottom.
        y1: Num,
    },
}

/// The answer of `mask_find` (`docs/23` section 9). Only a found answer is to change settings: the page sends it as one row
/// patch, one undo step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaskFindResult {
    /// A mask: what the row's `points` and `box` become, and the time the decoder took.
    Found {
        /// The row's clicks `[x, y, positive]` with the new one at the end (for a click).
        points: Vec<(Num, Num, bool)>,
        /// The row's box `[x0, y0, x1, y1]`, or null.
        #[serde(rename = "box")]
        box_: Option<[Num; 4]>,
        /// Milliseconds the decoder took.
        took_ms: Num,
    },
    /// No object there: the page adds no point and changes nothing, and says so in the error bar.
    Nothing,
}

/// `object_state { photo_id, preparing }` (addition): SAM's encoder is working for a photo, or has stopped. The rows of an Object
/// mask say "Getting ready…" while `preparing` is true (`docs/23` section 7.6). Sent when it changes; when it turns false
/// with features made, the photo is drawn again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "object_state")]
pub struct ObjectState {
    /// The photo.
    pub photo_id: PhotoId,
    /// The encoder is working for it, or will be in a moment.
    pub preparing: bool,
}

/// `depth_state { photo_id, failed }` (addition): the core's depth worker for a photo has ended (`docs/23` section 4.8).
/// `failed` is the plain sentence of the failure ("The Depth mask can't be made: …") or null when the map is ready. While the
/// worker runs the viewport is `viewport_busy { busy: true, finding: true }` and the Depth rows with the eye on say "Finding…";
/// when it ends the photo is drawn again with the mask, and a Depth row whose map failed says "Can't be made".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "depth_state")]
pub struct DepthState {
    /// The photo.
    pub photo_id: PhotoId,
    /// The sentence of the failure, or null when the map is ready.
    pub failed: Option<String>,
}

/// One face of a photo as the Face Refinement node's panel lists it (`docs/27` section 10, S-11b): its number (1 for the leftmost,
/// up to 32) and its box in the oriented photo frame, the picture as it is shown (crop and turn applied), as shares of the frame's
/// width and height.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct FaceBox {
    /// The face's number, from 1, left to right.
    pub number: u32,
    /// Left edge, 0 to 1.
    pub x: Num,
    /// Top edge, 0 to 1.
    pub y: Num,
    /// Width, 0 to 1.
    pub width: Num,
    /// Height, 0 to 1.
    pub height: Num,
}

/// Where the faces of a photo are (`docs/27` decision 9): `finding` (the node says "Finding faces…" and the photo is shown without
/// its effect), `ready` (the faces are known; there may be none), `failed` (the sentence says why; nothing asks again until the
/// picture changes) and `models_missing` (this copy of Yellowmoon has no face models: the node passes its picture through).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum FacePhase {
    /// The faces are being found.
    Finding,
    /// The faces are known.
    Ready,
    /// Finding them failed.
    Failed,
    /// This copy has no face models.
    ModelsMissing,
}

/// `face_state { photo_id, phase, faces, message }` (addition): where the faces of a photo are (`docs/27` section 10, S-11b). Sent
/// when the core's face worker starts on a photo (`finding`) and when it ends (`ready`, with the numbered faces, or `failed`), and
/// returned by the command of the same name. While the worker runs the viewport is `viewport_busy { busy: true, finding: true }`;
/// when it ends the photo is drawn again with the retouch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "face_state")]
pub struct FaceState {
    /// The photo.
    pub photo_id: PhotoId,
    /// Where the analysis is.
    pub phase: FacePhase,
    /// The faces, numbered from left to right; empty unless `phase` is `ready`.
    pub faces: Vec<FaceBox>,
    /// The plain sentence of the failure (`failed`) or of the missing models (`models_missing`); null otherwise.
    pub message: Option<String>,
}

/// `vision_state { photo_id, nothing_found }` (addition): the core's worker for the Subject and People masks of a photo has ended
/// its work, and `nothing_found` lists the kinds whose newest mask for the photo found nothing (`docs/23` section 7.6: their rows
/// say "Nothing found"). Sent each time the worker ends, with the list as it is then; the rows say "Finding…" while it works
/// (`viewport_busy.finding`), and the photo is drawn again with the masks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "vision_state")]
pub struct VisionState {
    /// The photo.
    pub photo_id: PhotoId,
    /// `subject` and/or `people`: the requests that found no instance.
    pub nothing_found: Vec<MaskKind>,
}

/// Which side of `viewport_map_points` the points are given in (`docs/23` section 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum MapFrom {
    /// Points `[x, y]` in the photo frame, answered as CSS pixels from the cell's corner (they may lie outside the
    /// cell).
    Photo,
    /// Points in CSS pixels from the cell's corner, answered as photo-frame fractions, `null` for a point on the
    /// paper.
    Cell,
}

/// Display values under the pointer, 0 to 255 per channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct PixelSample {
    /// Red.
    pub r: u8,
    /// Green.
    pub g: u8,
    /// Blue.
    pub b: u8,
}

/// Where in a node `sample_chroma` reads the colour (`docs/23` section 9). Left out (`null`) it is
/// `warp_input`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ChromaTap {
    /// At the input of the node's colour warp, after the node's own trim and grade: Pick colour of
    /// the Colour warp.
    WarpInput,
    /// At the node's own input, before its trim and its grade: Pick colour of a Colour range,
    /// which selects on the chroma of exactly this.
    NodeInput,
}

/// The chroma under a point of the photo (`sample_chroma`): a point of the plane the colour warp
/// draws and a Colour range's centre is in, read at the place [`ChromaTap`] names.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct ChromaSample {
    /// Along `e1`, red against cyan.
    pub a: Num,
    /// Along `e2`, green against blue.
    pub b: Num,
}

/// The photo's colours at the colour warp's input, for the faint scatter behind the pins
/// (`chroma_scatter`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct ChromaScatter {
    /// Flat `[a0, b0, a1, b1, ...]`, at most 2048 points, each coordinate rounded to 1e-4; empty
    /// while the photo is still being developed.
    pub points: Vec<Num>,
}

/// Per-channel histogram of the current render, for the Light tab and the Curve plot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct HistogramData {
    /// Red bins, black first.
    pub red: Vec<Num>,
    /// Green bins.
    pub green: Vec<Num>,
    /// Blue bins.
    pub blue: Vec<Num>,
}

/// Colour spaces the viewport and the exports can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    /// sRGB.
    Srgb,
    /// Display P3.
    DisplayP3,
    /// Rec.2020.
    Rec2020,
}

/// Demosaic path of the last render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum RenderPath {
    /// The learned demosaic.
    Neural,
    /// The classical demosaic.
    Classical,
}

// ---------------------------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------------------------

/// The four export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    /// JPEG.
    Jpeg,
    /// TIFF.
    Tiff,
    /// PNG.
    Png,
    /// DNG.
    Dng,
}

/// TIFF compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum TiffCompression {
    /// None.
    None,
    /// LZW.
    Lzw,
    /// Deflate, shown as "ZIP".
    Zip,
}

/// Bits per channel of a TIFF or PNG: `8 | 16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BitDepth(pub u8);

impl Type for BitDepth {
    fn definition(_: &mut specta::Types) -> DataType {
        DataType::Reference(specta_typescript::define("8 | 16"))
    }
}

/// What to do when the target file exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum IfExists {
    /// Leave the existing file alone.
    Skip,
    /// Replace it.
    Replace,
    /// Keep both: add " 2", " 3" to the new name.
    KeepBoth,
}

/// Export size: the full frame, or the long edge in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportSize {
    /// The full frame.
    Full,
    /// Resize so the long edge is `px` pixels.
    LongEdge {
        /// The long edge.
        px: u32,
    },
}

/// The export options of `docs/09`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct ExportOptions {
    /// The file format.
    pub format: ExportFormat,
    /// JPEG quality, 50 to 100.
    pub jpeg_quality: u8,
    /// TIFF and PNG bit depth.
    pub bit_depth: BitDepth,
    /// TIFF compression.
    pub tiff_compression: TiffCompression,
    /// The colour space (not for DNG).
    pub color_space: ColorSpace,
    /// The size (not for DNG).
    pub size: ExportSize,
    /// Destination folder; null means the default.
    pub destination: Option<String>,
    /// Template with `{name}`, `{look}`, `{date}`, `{seq}`, `{camera}`.
    pub name_template: String,
    /// What to do when the file exists.
    pub if_exists: IfExists,
}

/// What to export and how.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct ExportJob {
    /// This photo, the picked photos, or all.
    pub scope: ApplyScope,
    /// The options.
    pub options: ExportOptions,
}

/// `export_estimate(job)`: the size and time shown in the Export panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct ExportEstimate {
    /// Total size of the files in bytes.
    pub bytes: Num,
    /// Time the job will take, in seconds.
    pub seconds: Num,
    /// Photos with adjustments beyond the look.
    pub adjusted_photos: u32,
    /// A DNG of the job is close to, not the same as, the picture in the app: a photo uses a look with
    /// its own display rendering.
    pub dng_approximate: bool,
}

// ---------------------------------------------------------------------------------------------
// Preferences
// ---------------------------------------------------------------------------------------------

/// Folders remembered between runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct LastUsedFolders {
    /// Where files were last opened from.
    pub open: Option<String>,
    /// Where files were last exported to.
    pub export: Option<String>,
    /// Where looks were last imported from.
    pub looks_import: Option<String>,
}

/// Everything that persists besides imported looks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct Preferences {
    /// The look new photos get.
    pub default_look: LookId,
    /// Export defaults.
    pub export_defaults: ExportOptions,
    /// Leave the location out of exported files.
    pub remove_location_from_exports: bool,
    /// Check for updates in the background.
    pub check_for_updates_automatically: bool,
    /// Watch Downloads while the import wizard waits for a zip.
    pub watch_downloads_during_import: bool,
    /// Folders remembered between runs.
    pub last_used_folders: LastUsedFolders,
    /// Who grades (`docs/22` section 10.1): Apple Intelligence by default. A provider that is not
    /// available, or a program the person has not agreed to yet, acts as Default Auto.
    pub auto_provider: AutoProvider,
    /// Whether the person agreed to send photos through Claude Code and through Antigravity.
    pub auto_consent: AutoConsent,
    /// Instructions for the grading model (`docs/19` section 2, `docs/22` D1): the person's own, at
    /// most 2000 characters, one text for every provider; `""` is the default.
    pub apple_auto_instructions: String,
}

/// A partial update of the preferences.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct PreferencesPatch {
    /// New default look.
    #[serde(default)]
    pub default_look: Option<LookId>,
    /// New export defaults.
    #[serde(default)]
    pub export_defaults: Option<ExportOptions>,
    /// New value of "Remove location from exports".
    #[serde(default)]
    pub remove_location_from_exports: Option<bool>,
    /// New value of "Check for updates automatically".
    #[serde(default)]
    pub check_for_updates_automatically: Option<bool>,
    /// New value of "Watch Downloads during import".
    #[serde(default)]
    pub watch_downloads_during_import: Option<bool>,
    /// New remembered folders.
    #[serde(default)]
    pub last_used_folders: Option<LastUsedFolders>,
    /// New value of "Auto grading". A program is refused unless `auto_consent` (in the same patch
    /// or already saved) says the person agreed.
    #[serde(default)]
    pub auto_provider: Option<AutoProvider>,
    /// New consent for the two programs; replaces the saved one whole.
    #[serde(default)]
    pub auto_consent: Option<AutoConsent>,
    /// New instructions (`""` or the default text: the default); at most 2000 characters, longer
    /// text is refused.
    #[serde(default)]
    pub apple_auto_instructions: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------------------------

/// `install_cli_link` result: where the `ym` link was created.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct CliLinkResult {
    /// The link.
    pub path: String,
}

/// `cli_link_status`: whether "Install Command-Line Tool…" has been done.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct CliLinkStatus {
    /// A working `ym` link exists.
    pub installed: bool,
    /// Where the link is, when installed.
    pub path: Option<String>,
}

/// One camera film of the Film node's picker (`docs/24` section 4.1), from `names.json` through the core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct FilmStockInfo {
    /// The profile id.
    pub id: String,
    /// The name shown to the person ("Kodak Portra 400").
    pub name: String,
    /// `colour_negative`, `cine_negative` or `slide`.
    pub kind: String,
    /// The maker.
    pub group: String,
    /// The ISO number.
    pub iso: Option<u32>,
    /// The profile id of the paper "Matching paper" means for it; null for a slide.
    pub matching_paper: Option<String>,
}

/// One still paper of the Film node's paper picker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct FilmPaperInfo {
    /// The profile id.
    pub id: String,
    /// The name shown to the person.
    pub name: String,
}

/// `film_catalogue`: what the Film node's pickers list, and the lines About and the Film tab carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct FilmCatalogue {
    /// The camera films that are offered, in the order of the picker.
    pub stocks: Vec<FilmStockInfo>,
    /// The still papers that are offered.
    pub papers: Vec<FilmPaperInfo>,
    /// The version of the film data ("0.3.2").
    pub profile_version: String,
    /// The credit line of the film data: "Modelled on published data. Film profiles by …, CC BY-SA 4.0. Not affiliated with Kodak or Fujifilm."
    pub credit: String,
    /// "Modelled on published data. Not affiliated with Kodak or Fujifilm."
    pub disclaimer: String,
    /// "Kodak, Portra, Ektar, Vision3, Ektachrome, Kodachrome, Fujifilm, Velvia and the other film and paper names shown are trademarks of their owners."
    pub trademarks: String,
}

/// What the app knows about itself, for About.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AppInfo {
    /// The app and CLI version: "0.1.0".
    pub version: String,
    /// The pipeline version written into exports.
    pub pipeline_version: String,
    /// The operating system, as Rust names it (`std::env::consts::OS`): "macos", "windows" or
    /// "linux". Preferences words a few sentences differently on Windows (`docs/22` section 12.3).
    pub os: String,
}

/// What `open_dialog` offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum OpenKind {
    /// One or more ARW files.
    Files,
    /// A folder.
    Folder,
}

/// The items of the native menu bar that need the webview. The shell sends `menu_action` with one
/// of them; the items it handles itself (About, Show Looks Folder, the Help links, Quit) are not
/// listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum MenuAction {
    /// The in-window About dialog (Preferences' footer link; the menu's About item opens the
    /// system's About panel).
    About,
    /// Preferences.
    Preferences,
    /// Check for Updates.
    CheckForUpdates,
    /// Install Command-Line Tool.
    InstallCli,
    /// Open Files.
    OpenFiles,
    /// Open Folder.
    OpenFolder,
    /// Export.
    Export,
    /// Close Session.
    CloseSession,
    /// Undo.
    Undo,
    /// Redo.
    Redo,
    /// Toggle the pick of the current photo.
    Pick,
    /// Pick all photos.
    PickAll,
    /// Clear the picks.
    ClearPicks,
    /// Before / after.
    BeforeAfter,
    /// Open or close the node view (the Nodes button and N).
    Nodes,
    /// Fit.
    Fit,
    /// 100 %.
    #[serde(rename = "zoom_100")]
    Zoom100,
    /// Zoom in.
    ZoomIn,
    /// Zoom out.
    ZoomOut,
    /// Next photo.
    NextPhoto,
    /// Previous photo.
    PreviousPhoto,
    /// Import ARRI looks.
    ImportArriLooks,
    /// Import a .cube file.
    ImportCube,
    /// Make the current look the default.
    MakeDefaultLook,
    /// Copy the settings of the current photo.
    CopySettings,
    /// Paste the copied settings onto the current photo.
    PasteSettings,
    /// Paste the copied settings onto the picked photos.
    PasteSettingsToPicked,
}

/// `set_menu_state(state)`: which of the Edit menu's settings items are available. The webview
/// knows (a developed current photo, something copied, photos picked) and the native menu does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct MenuState {
    /// Copy Settings: the current photo is developed.
    pub copy_settings: bool,
    /// Paste Settings: something is copied and the current photo is developed.
    pub paste_settings: bool,
    /// Paste Settings to Picked: something is copied and a photo is picked.
    pub paste_settings_to_picked: bool,
}

// ---------------------------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------------------------

/// `photo_progress { id, stage, done, error }`: a stage of a photo's development started or finished.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "photo_progress")]
pub struct PhotoProgress {
    /// The photo.
    pub id: PhotoId,
    /// The stage that started or finished.
    pub stage: PhotoStage,
    /// True when the stage has finished. A photo is ready when `thumbnail` is done.
    pub done: bool,
    /// Plain-words reason when the photo can't be developed.
    pub error: Option<String>,
}

/// `thumbnail_ready { id, url }`: the photo's thumbnail, a PNG of at most 240 px on its long edge,
/// is at `url`.
///
/// The picture does not cross the bridge: the shell keeps it and serves it on the `ym-frame`
/// scheme (`ym-frame://localhost/<launch>/thumb-<id>-<version>.png`, `http://ym-frame.localhost/…`
/// on Windows; `<launch>` is made from the time the app started), so an `<img>` loads it. Every
/// new thumbnail of a photo (a look, a crop or a turn was applied) has a new address, and an
/// address names the same picture for as long as the app runs (it is served as immutable). Also
/// the answer of `thumbnails()` for every photo that has one, to get them again after the webview
/// reloaded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "thumbnail_ready")]
pub struct ThumbnailReady {
    /// The photo.
    pub id: PhotoId,
    /// Where the PNG is.
    pub url: String,
}

/// One node's thumbnail in `node_thumbnails_ready`: the corrector and the address of its picture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct NodeThumbnail {
    /// The corrector (a card of the node view; the Input and the Output have none).
    pub node: u32,
    /// Where the PNG is (the `ym-frame://` scheme, `http://ym-frame.localhost/` on Windows). Take
    /// it as it comes: the path holds an id of this launch of the app.
    pub url: String,
}

/// `node_thumbnails_ready { photo_id, thumbnails }` (`docs/23` sections 5.7 and 9): the answer to
/// `node_thumbnails(photo_id)`. Only the thumbnails that changed: each is a PNG of about 160 px on
/// its long edge, the photo after that node and then through the Output, at the address `url` on the
/// `ym-frame` scheme (`ym-frame://localhost/<launch>/node-<photo>-<node>-<version>.png`; the address
/// of a picture, never its bytes). A node whose picture did not change is not in the list: keep the
/// address you have. A new picture is a new address, and an address names the same picture for as
/// long as the app runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "node_thumbnails_ready")]
pub struct NodeThumbnailsReady {
    /// The photo.
    pub photo_id: PhotoId,
    /// The thumbnails that changed.
    pub thumbnails: Vec<NodeThumbnail>,
}

/// `render_stats { ms, path, space, zoom_percent, fit_percent, lens }`: the header's status text, the
/// zoom the last viewport frame was drawn at and what the lens did with the photo's bokeh.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "render_stats")]
pub struct RenderStats {
    /// Milliseconds the last render took.
    pub ms: Num,
    /// The demosaic path.
    pub path: RenderPath,
    /// The output colour space.
    pub space: ColorSpace,
    /// The zoom of the frame in percent (device pixels per pixel of the final image, times 100),
    /// not rounded; the core's own, which knows the lens crop, the straighten angle and the crop.
    pub zoom_percent: Option<Num>,
    /// The zoom at which the whole final image fits the cell, in percent and not rounded.
    pub fit_percent: Option<Num>,
    /// What the lens's bokeh reshaping did for the photo (`docs/21` section 10), once its planes
    /// exist; `None` without a lens. The Lens card's note reads it.
    pub lens: Option<LensStats>,
}

/// Why a photo's bokeh was left as it is (`render_stats.lens.reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LensReshapeReason {
    /// The sliders did not ask for it.
    NotAsked,
    /// The photo has little out-of-focus background.
    LittleBlur,
}

/// What the lens did with the photo's bokeh (`render_stats.lens`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct LensStats {
    /// True when the bokeh was reshaped.
    pub reshaped: bool,
    /// Why it was not, when it was not.
    pub reason: Option<LensReshapeReason>,
}

/// `export_progress { job_id, file, done, total, file_percent, error, skipped }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "export_progress")]
pub struct ExportProgress {
    /// The job.
    pub job_id: ExportJobId,
    /// Output file name now being written.
    pub file: String,
    /// Files finished.
    pub done: u32,
    /// Files in the job.
    pub total: u32,
    /// Progress of `file`, 0 to 100.
    pub file_percent: Option<Num>,
    /// Plain-words error for `file`, when it failed.
    pub error: Option<String>,
    /// `file` already existed and "If it exists" is Skip.
    pub skipped: bool,
}

/// `looks_changed`: the look list, the default look or a look's file changed. No data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "looks_changed")]
pub struct LooksChanged;

/// `auto_status_changed` (`docs/22` section 12.6): a provider's availability changed (a program was
/// found, signed in or out, or a check ended). No data; the webview reads `auto_status_cached()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "auto_status_changed")]
pub struct AutoStatusChanged;

/// `update_state { phase, version?, percent?, message? }` (the updater's phases, `docs/12`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[serde(tag = "phase", rename_all = "snake_case")]
#[tauri_specta(event_name = "update_state")]
pub enum UpdateState {
    /// Nothing to report.
    Idle,
    /// An update exists.
    Available {
        /// Its version.
        version: String,
    },
    /// An update is downloading.
    Downloading {
        /// Its version.
        version: String,
        /// 0 to 100; absent when the server sends no content length.
        #[serde(default)]
        percent: Option<u8>,
    },
    /// The update is installed; restarting finishes it.
    Ready {
        /// Its version.
        version: String,
    },
    /// The check or the download failed.
    Error {
        /// The reason, shown on hover.
        message: String,
    },
}

/// `error { scope, message }`: an error that did not come back from a command (a background job).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "error")]
pub struct ErrorEvent {
    /// Where it came from.
    pub scope: ErrorScope,
    /// What went wrong and what to do.
    pub message: String,
}

impl From<AppError> for ErrorEvent {
    fn from(e: AppError) -> Self {
        Self {
            scope: e.scope,
            message: e.message,
        }
    }
}

/// `menu_action { action }`: a menu item that needs the webview was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "menu_action")]
pub struct MenuActionEvent {
    /// Which item.
    pub action: MenuAction,
}

/// `settings_changed { photo_id }` (addition): the core changed a photo's settings by itself: after
/// a `viewport_input` Pick white click, once for each photo a `paste_settings` changed, and when the
/// estimates its Auto values resolve against changed (the core's `estimates_changed`: a burst's
/// median moved, Apple Auto answered or failed, or the `apple_auto` preference changed, `docs/19`
/// section 6). Ask `get_settings` again (the webview skips the photos it is pasting onto: the
/// paste's answer says what changed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "settings_changed")]
pub struct SettingsChanged {
    /// The photo.
    pub photo_id: PhotoId,
}

/// `export_busy { busy }` (addition): an export is queued or running. "Update ready" waits while
/// this is true (`docs/12`). Also available as the command `export_busy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "export_busy")]
pub struct ExportBusy {
    /// An export is queued or running.
    pub busy: bool,
}

/// `viewport_busy { busy, cpu, finding }` (addition): the viewport is working on a frame of the photo on
/// screen, or on the full-resolution tiles a frame needs; `cpu`: the raw stages of that photo run on
/// the CPU, which can take seconds; `finding`: the round makes the photo's first blur map for a
/// Background mask (`docs/23` section 7.6, the rows say "Finding…"). Sent when it changes, at most about 30 times a second (a round
/// that starts and ends within that time sends nothing new); `busy: false` when the work is done or
/// was stopped, and when the cell shows no photo. The photo cell says "Rendering on the CPU" once
/// both have held for a moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "viewport_busy")]
pub struct ViewportBusy {
    /// A frame, or the tiles a frame needs, is being made.
    pub busy: bool,
    /// The photo's frames run on the CPU (no usable graphics card, a device error ended the raw route or the
    /// view route, the device is lost, or a method the device does not run).
    pub cpu: bool,
    /// The round makes the photo's first blur map for a Background mask (`docs/23` section 7.6): the
    /// rows of that kind say "Finding…" until the round ends. False for every other round, and once the
    /// photo has its map.
    pub finding: bool,
}

/// `viewport_frame` (addition, fallback viewport only): a new frame of the photo cell is ready.
/// Show `url` in an `<img>` laid over the cell (the cell's rectangle, `width` and `height` are its
/// size in device pixels).
///
/// An empty `url` (with `width` and `height` 0) says there is no picture any more: the session was
/// closed, or the photo now selected can't be developed. Take the image away; the cell shows only
/// the paper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Event)]
#[tauri_specta(event_name = "viewport_frame")]
pub struct ViewportFrame {
    /// Counts up with every frame; also in `url`, so a changed frame is a new address.
    pub seq: u32,
    /// Width in device pixels.
    pub width: u32,
    /// Height in device pixels.
    pub height: u32,
    /// False when the frame was made from a coarser image while sharper tiles render; a sharp
    /// frame follows.
    pub sharp: bool,
    /// Where the PNG is (the `ym-frame://` scheme, `http://ym-frame.localhost/` on Windows). Take
    /// it as it comes: the path holds an id of this launch of the app.
    pub url: String,
}

/// Which viewport backend draws the photo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ViewportBackendKind {
    /// A wgpu surface behind the transparent webview (macOS).
    Native,
    /// Frames sent to the webview as images (`viewport_frame` events).
    Fallback,
}

/// `viewport_info` (addition): which backend draws the photo, so the photo cell knows whether to
/// stay transparent or to show `viewport_frame` images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct ViewportInfo {
    /// The backend in use.
    pub backend: ViewportBackendKind,
}

/// Paths are strings on the bridge; this is where they become paths.
#[must_use]
pub fn path_of(s: &str) -> PathBuf {
    PathBuf::from(s)
}
