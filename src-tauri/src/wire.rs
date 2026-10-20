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
//! * `SettingsPatch` is the core's patch, a little more generous than `Partial<DevelopSettings>`:
//!   nested objects (`decode`, `rolloff`, `color_balance`, `curve`) and a manual `white_balance`,
//!   `sharpen` or `noise_reduction` may leave parts out, and `crop` and `white_pick` may be `null`
//!   to clear them. Everything `Partial<DevelopSettings>` accepts is accepted. `color_warp` is the
//!   whole list of pins, as in `types.ts`.
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
            E::Settings(_) => ErrorScope::Render,
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

/// RAW › Range (`docs/02` C7), each 0 to 1: Highlights 1 and Low light 0.5 by default. The name
/// `rolloff` is kept from the earlier roll-off card.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct Rolloff {
    /// The highlight range: where the camera's clipping point lands, from where it was recorded (0)
    /// to white, the clip of an ALEXA at EI 800 (1).
    pub highlights: Num,
    /// Low-light toe strength.
    pub low_light: Num,
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
pub struct Crop {
    /// The frame.
    pub rect: NormRect,
    /// Straighten angle, -45 to 45 degrees.
    pub angle_deg: Num,
    /// The aspect preset the frame follows.
    pub aspect: CropAspect,
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

/// Every control in the UI and every CLI flag maps to one field. Defaults reproduce the
/// automatic development.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct DevelopSettings {
    /// Look tab.
    pub look: LookId,
    /// 0 to 100 percent.
    pub look_intensity: Num,
    /// Light › Exposure: Auto or an absolute value, -4 to +4 EV.
    pub exposure: ExposureSetting,
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
    /// Light › White › Offset: stops added to the automatic white lift, -3 to +3, default 0 (the
    /// automatic amount; a plain number, not Auto).
    pub white_offset_ev: Num,
    /// Light › High dynamic range › Black: Auto or a value, -100 to 100.
    pub black: PercentSetting,
    /// Colour › White balance, the light presets.
    pub light_source: LightSource,
    /// Colour › White balance: Auto or Kelvin and tint.
    pub white_balance: WhiteBalanceSetting,
    /// The three colour-balance wheels.
    pub color_balance: ColorBalance,
    /// Colour › Colour balance › Saturation: Auto or a value, -100 to 100.
    pub saturation: PercentSetting,
    /// Colour › Colour warp: its pins.
    pub color_warp: ColorWarp,
    /// Detail › Sharpening: Auto or amount and radius.
    pub sharpen: SharpenSetting,
    /// Detail › Sharpening › Output: the strength of the sharpening of the picture at the size it
    /// is shown and written, in percent, 0 to 200, default 100 (a plain number, not Auto).
    pub output_sharpening: Num,
    /// Detail › Noise reduction: Auto or the three strengths.
    pub noise_reduction: NoiseReductionSetting,
    /// Luma, red, green and blue curves.
    pub curve: Curves,
    /// RAW › Decode switches.
    pub decode: DecodeOptions,
    /// RAW › Range: the highlight range and the low-light toe.
    pub rolloff: Rolloff,
    /// Crop; null when the photo isn't cropped.
    pub crop: Option<Crop>,
    /// Orientation after the camera's EXIF orientation.
    pub orientation: Orientation,
    /// Pick white: the sampled point `[x, y]` in [0, 1]², measured on the photo as the camera
    /// oriented it.
    pub white_pick: Option<[Num; 2]>,
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

/// A partial update: only the fields present change (`update_settings(photo_id, patch)`).
///
/// Serialising leaves absent fields out (so that `crop: null`, which clears the crop, is told
/// apart from no `crop` at all); the derive cannot do that without `skip_serializing_if`, which
/// the unified TypeScript export refuses, hence the manual `Serialize`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct SettingsPatch {
    /// New look.
    #[serde(default)]
    pub look: Option<LookId>,
    /// New look intensity.
    #[serde(default)]
    pub look_intensity: Option<Num>,
    /// New exposure (Manual to set it, Auto to reset it).
    #[serde(default)]
    pub exposure: Option<ExposureSetting>,
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
    /// New white lift offset, in stops.
    #[serde(default)]
    pub white_offset_ev: Option<Num>,
    /// New black: Auto, a manual value, or a plain number (0 is Auto).
    #[serde(default)]
    pub black: Option<PercentPatch>,
    /// New light preset.
    #[serde(default)]
    pub light_source: Option<LightSource>,
    /// New white balance; a manual one may leave out Kelvin or tint.
    #[serde(default)]
    pub white_balance: Option<WhiteBalancePatch>,
    /// Colour-balance wheels to change.
    #[serde(default)]
    pub color_balance: Option<ColorBalancePatch>,
    /// New saturation: Auto, a manual value, or a plain number (0 is Auto).
    #[serde(default)]
    pub saturation: Option<PercentPatch>,
    /// New colour warp: the whole list of pins (`{ pins: [] }` removes them all).
    #[serde(default)]
    pub color_warp: Option<ColorWarp>,
    /// New sharpening; a manual one may leave out a part.
    #[serde(default)]
    pub sharpen: Option<SharpenPatch>,
    /// New output sharpening amount, percent.
    #[serde(default)]
    pub output_sharpening: Option<Num>,
    /// New noise reduction; a manual one may leave out a part.
    #[serde(default)]
    pub noise_reduction: Option<NoiseReductionPatch>,
    /// Curves to replace.
    #[serde(default)]
    pub curve: Option<CurvesPatch>,
    /// Decode switches to change.
    #[serde(default)]
    pub decode: Option<DecodePatch>,
    /// RAW › Range amounts to change.
    #[serde(default)]
    pub rolloff: Option<RolloffPatch>,
    /// New crop; `null` removes it.
    #[serde(default, deserialize_with = "double_option")]
    #[specta(type = Option<Crop>)]
    pub crop: Option<Option<Crop>>,
    /// New orientation.
    #[serde(default)]
    pub orientation: Option<Orientation>,
    /// New white pick; `null` clears it. A pick also sets `light_source` and `white_balance` to
    /// Auto unless the same patch says otherwise.
    #[serde(default, deserialize_with = "double_option")]
    #[specta(type = Option<[Num; 2]>)]
    pub white_pick: Option<Option<[Num; 2]>>,
}

impl Serialize for SettingsPatch {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        // One entry per field that is present; `Some(None)` is a present `null`.
        fn entry<M: SerializeMap, T: Serialize>(
            map: &mut M,
            key: &str,
            value: &Option<T>,
        ) -> Result<(), M::Error> {
            match value {
                Some(v) => map.serialize_entry(key, v),
                None => Ok(()),
            }
        }
        let mut map = serializer.serialize_map(None)?;
        entry(&mut map, "look", &self.look)?;
        entry(&mut map, "look_intensity", &self.look_intensity)?;
        entry(&mut map, "exposure", &self.exposure)?;
        entry(&mut map, "contrast", &self.contrast)?;
        entry(&mut map, "brightness", &self.brightness)?;
        entry(&mut map, "highlights", &self.highlights)?;
        entry(&mut map, "shadows", &self.shadows)?;
        entry(&mut map, "white", &self.white)?;
        entry(&mut map, "white_offset_ev", &self.white_offset_ev)?;
        entry(&mut map, "black", &self.black)?;
        entry(&mut map, "light_source", &self.light_source)?;
        entry(&mut map, "white_balance", &self.white_balance)?;
        entry(&mut map, "color_balance", &self.color_balance)?;
        entry(&mut map, "saturation", &self.saturation)?;
        entry(&mut map, "color_warp", &self.color_warp)?;
        entry(&mut map, "sharpen", &self.sharpen)?;
        entry(&mut map, "output_sharpening", &self.output_sharpening)?;
        entry(&mut map, "noise_reduction", &self.noise_reduction)?;
        entry(&mut map, "curve", &self.curve)?;
        entry(&mut map, "decode", &self.decode)?;
        entry(&mut map, "rolloff", &self.rolloff)?;
        entry(&mut map, "crop", &self.crop)?;
        entry(&mut map, "orientation", &self.orientation)?;
        entry(&mut map, "white_pick", &self.white_pick)?;
        map.end()
    }
}

/// Which `Auto` fields are still automatic; the UI shows the AUTO tag for these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AutoFlags {
    /// `exposure` is Auto.
    pub exposure: bool,
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
    /// `light_source` is Auto.
    pub light_source: bool,
    /// `white_balance` is Auto.
    pub white_balance: bool,
    /// `sharpen` is Auto.
    pub sharpen: bool,
    /// `noise_reduction` is Auto.
    pub noise_reduction: bool,
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
// Apple Auto (`docs/19`)
// ---------------------------------------------------------------------------------------------

/// Which Auto made a photo's automatic values: Yellowmoon's own (`rule`) or Apple Auto's answer
/// (`apple`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AutoSource {
    /// Yellowmoon's own Auto.
    Rule,
    /// Apple Auto's answer (the photo's own or its burst's).
    Apple,
}

/// Why Apple Auto is not available on this computer (`docs/19` section 1; the core's
/// `UnavailableReason`).
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AppleAutoInfo {
    /// The model variant's display name: "AFM 3 Core".
    pub model: String,
    /// The macOS version and build: "27.0.1 (27A404)".
    pub os: String,
    /// The default instructions or the person's own.
    pub instructions: AppleAutoInstructions,
    /// The first 12 hex digits of the SHA-256 of the instructions.
    pub instructions_sha256: String,
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

/// How a run decides whether Apple Auto is used (the core's `AppleAutoMode`); the app always
/// follows the preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AppleAutoMode {
    /// The `apple_auto` preference decides.
    FromPreference,
    /// On for this run.
    On,
    /// Off for this run.
    Off,
}

/// `apple_auto_status()`: whether Apple Auto is available on this computer (read again for the
/// answer), and whether it acts (`docs/19` sections 1 and 2; the core's `AppleAutoStatus`).
/// Preferences shows its row only while `available`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AppleAutoStatus {
    /// Apple Auto is available on this computer now.
    pub available: bool,
    /// Why it is not; null when it is, and when the app has no advisor.
    pub reason: Option<AppleAutoReason>,
    /// Which model answers, when it is available.
    pub info: Option<AppleAutoInfo>,
    /// The `apple_auto` preference.
    pub preference: bool,
    /// How the run decides (always `from_preference` in the app).
    pub mode: AppleAutoMode,
    /// Apple Auto acts for the session now.
    pub active: bool,
    /// The prompt's version.
    pub prompt: u32,
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
    /// The longest instructions the model is given, in characters.
    pub instructions_max_chars: u32,
}

/// Per-photo automatic estimates (`AutoEstimates` in `types.ts`; the core's `AutoSummary`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct AutoEstimates {
    /// `EV_auto` (with Apple Auto's change, when there is one).
    pub exposure_ev: Num,
    /// The automatic white lift in stops (0 when nothing is above +1 stop).
    pub white_ev: Num,
    /// The Auto Kelvin (with Apple Auto's nudge, when there is one).
    pub kelvin: Num,
    /// The Auto tint.
    pub tint: Num,
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
    /// Which Auto made these values.
    pub source: AutoSource,
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
    /// What `sharpen: Auto` resolves to.
    pub sharpen: SharpenValue,
    /// What `noise_reduction: Auto` resolves to.
    pub noise_reduction: NoiseReductionValue,
}

/// `DevelopSettings` with every `Auto` replaced by the number it resolves to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct ResolvedValues {
    /// `look`.
    pub look: LookId,
    /// `look_intensity`.
    pub look_intensity: Num,
    /// The exposure in stops.
    pub exposure_ev: Num,
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
    /// `white_offset_ev`.
    pub white_offset_ev: Num,
    /// The white lift in effect in stops: the automatic amount plus the offset, never below 0.
    pub white_ev: Num,
    /// The black in effect.
    pub black: Num,
    /// `light_source`.
    pub light_source: LightSource,
    /// The Kelvin the White balance row shows.
    pub kelvin: Num,
    /// The tint the White balance row shows.
    pub tint: Num,
    /// The wheels' points in effect.
    pub color_balance: ColorBalancePoints,
    /// The saturation in effect: the manual value, or the photo's Auto saturation.
    pub saturation: Num,
    /// `color_warp`.
    pub color_warp: ColorWarp,
    /// The sharpening in effect.
    pub sharpen: SharpenValue,
    /// `output_sharpening`.
    pub output_sharpening: Num,
    /// The noise reduction in effect.
    pub noise_reduction: NoiseReductionValue,
    /// `curve`.
    pub curve: Curves,
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
    /// Display LUTs added.
    pub display_luts_added: u32,
    /// Files whose hash matched a look or display LUT that was already imported.
    pub unchanged: u32,
}

/// The five free packages ARRI publishes (the ids of `data/looks/arri-packages.json`, with
/// underscores). The two look libraries that also come as ARRI Look Files (the LogC4 one as
/// `.alf4`, the LogC3 to Rec.709 one as `.aml`) are the same package here: a person who downloads
/// the look files has what they chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum PackageId {
    /// Look Library, LogC4 log-to-log.
    LookLibraryLogc4,
    /// Display LUTs, LogC4.
    DisplayLutsLogc4,
    /// Look Library, LogC3 log-to-log.
    LookLibraryLogc3,
    /// Look Library, LogC3 to Rec.709.
    LookLibraryLogc3ToRec709,
    /// Display LUTs, LogC3.
    DisplayLutsLogc3,
}

impl PackageId {
    /// The id the package has in `data/looks/arri-packages.json`.
    #[must_use]
    pub fn data_id(self) -> &'static str {
        match self {
            Self::LookLibraryLogc4 => "look-library-logc4",
            Self::DisplayLutsLogc4 => "display-luts-logc4",
            Self::LookLibraryLogc3 => "look-library-logc3",
            Self::LookLibraryLogc3ToRec709 => "look-library-logc3-rec709",
            Self::DisplayLutsLogc3 => "display-luts-logc3",
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
            Self::DisplayLutsLogc4,
            Self::LookLibraryLogc3,
            Self::LookLibraryLogc3ToRec709,
            Self::DisplayLutsLogc3,
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

/// What a display LUT expects as input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum DisplayLutInput {
    /// LogC4.
    Logc4,
    /// LogC3 (EI 800).
    Logc3Ei800,
}

/// An imported display LUT (ARRI's display rendering).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct DisplayLut {
    /// "arri-display:logc4:rec709-g24".
    pub id: String,
    /// "Rec.709 · Gamma 2.4".
    pub name: String,
    /// What it expects as input.
    pub input: DisplayLutInput,
    /// "rec709-g24", "p3-d65", "rec2020", "hlg", "pq".
    pub target: String,
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

/// The chroma under a point of the photo at the colour warp's input (`sample_chroma`): a point of
/// the warp's plane.
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
    /// its own display rendering, or ARRI's display rendering is chosen.
    pub dng_approximate: bool,
}

// ---------------------------------------------------------------------------------------------
// Preferences
// ---------------------------------------------------------------------------------------------

/// Display rendering: our ACES 2 transform, or an imported ARRI display LUT.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DisplayRendering {
    /// Our ACES 2 output transform.
    Aces2,
    /// An imported ARRI display LUT.
    ArriDisplay {
        /// The display LUT.
        id: String,
    },
}

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
    /// ACES 2, or an imported ARRI display LUT.
    pub display_rendering: DisplayRendering,
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
    /// Auto with Apple Intelligence (`docs/19` section 2); acts only where Apple Auto is available.
    pub apple_auto: bool,
    /// Instructions for Apple Intelligence (`docs/19` section 2): the person's own, at most 2000
    /// characters; `""` is the default.
    pub apple_auto_instructions: String,
}

/// A partial update of the preferences.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(deny_unknown_fields)]
pub struct PreferencesPatch {
    /// New default look.
    #[serde(default)]
    pub default_look: Option<LookId>,
    /// New display rendering.
    #[serde(default)]
    pub display_rendering: Option<DisplayRendering>,
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
    /// New value of "Auto with Apple Intelligence".
    #[serde(default)]
    pub apple_auto: Option<bool>,
    /// New instructions for Apple Intelligence (`""` or the default text: the default); at most
    /// 2000 characters, longer text is refused.
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

/// What the app knows about itself, for About.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AppInfo {
    /// The app and CLI version: "0.1.0".
    pub version: String,
    /// The pipeline version written into exports.
    pub pipeline_version: String,
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

/// `render_stats { ms, path, space, zoom_percent, fit_percent }`: the header's status text and the
/// zoom the last viewport frame was drawn at.
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
