//! The colour groups: C1 to C8 (the fused matrix, highlight range and toe) and C9 to C13 (grade, look,
//! curve, display transform, encode), built from a photo's settings and estimates
//! (`docs/02-colour-pipeline.md`).
//!
//! No colour maths lives here. The fused matrix is `ym_color::chain` (through
//! [`AutoEstimates::fused_chain`](crate::auto::AutoEstimates::fused_chain)), the per-pixel stages are
//! `ym_color::display::ColorChain`; this module maps the settings onto their parameters:
//!
//! * `rolloff.*`, the clip level and the noise floor become `ToneParams` (C7): `rolloff.highlights`
//!   is the highlight range, which places the camera's clip (the luma of the clipped neutral after
//!   C1 to C6, `ClipLevel::luma`, so it follows the exposure) between where it was recorded and the
//!   ALEXA's EI 800 clip; `rolloff.low_light` is the toe. The **noise floor** is the read noise of
//!   the photo's noise model carried through the fused matrix: the balanced channels have the
//!   noise `g0_c * sqrt(b + sigma_row^2 + q^2 / 12)`, the matrix (exposure included) mixes them
//!   into the three AWG4 channels, and the floor is the mean of those three sigmas (`docs/02` C7:
//!   "the noise model's read noise mapped to linear AWG4 at this exposure");
//! * `contrast` to `color_balance` become `GradeParams`, the pins of `color_warp` a `WarpParams`
//!   (stage C9b), the four curves a `CurveSet`;
//! * the photo's automatic white lift and `white_offset_ev` become the `WhiteLift` of C12 step 1b
//!   ([`white_lift`]), which acts after C11, alongside the highlight range of C7;
//! * the look comes from a [`LookProvider`] (the looks library in the app, [`NeutralOnly`] in tests)
//!   and the display transform from a [`DisplayChoice`] (ACES 2, or an imported ARRI display LUT).
//!
//! Every plan has a key made of the numbers that go into it, so the render graph can tell whether a
//! cached plan is still the plan the settings ask for.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use ym_color::camera_default::CameraDefault;
use ym_color::chain::FusedChain;
use ym_color::consts::Vec3;
use ym_color::curve::{Curve, CurveSet};
use ym_color::display::{
    ArriDisplayLut, BeforePoint, ChainParams, ColorChain, DisplayLutInput, DisplayLutOutput,
    DisplayTransform, LookSetting, OutputTarget,
};
use ym_color::exposure::white::WhiteLift;
use ym_color::grade::{ColorBalance, GradeParams};
use ym_color::look::LookTransform;
use ym_color::tone::ToneParams;
use ym_color::warp::{Warp, WarpParams};
use ym_looks::library::{DisplayInput, Library, LookId};
use ym_profiles::color_matrices::ColorMatrices;
use ym_profiles::profile::Profile;
use ym_profiles::registry::SensorGeometry;
use ym_raw::decode::RawImage;

use super::RenderError;
use super::raw::{CaptureMethod, DEFAULT_CAPTURE_METHOD};
use crate::auto::{AutoEstimates, as_shot_gains, as_shot_rggb};
use crate::session::{PhotoId, PhotoView, PickedWhite, ThumbnailInput};
use crate::settings::{AutoOr, CurvePoint, DevelopSettings};

// ---------------------------------------------------------------------------------------------
// Looks and the display transform
// ---------------------------------------------------------------------------------------------

/// A look that could not be used, in plain words ("File missing or changed. Import again.").
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct LookProblem(pub String);

/// A look ready to apply, with a fingerprint that changes when its contents do.
#[derive(Debug, Clone)]
pub struct ResolvedLook {
    /// The look.
    pub transform: Arc<LookTransform>,
    /// SHA-256 of the LUT file (imported looks) or the id (built-in ones).
    pub fingerprint: String,
}

/// Where the render graph gets looks and display LUTs from.
pub trait LookProvider: Send + Sync {
    /// The look with this id.
    ///
    /// # Errors
    ///
    /// [`LookProblem`] when the look is unknown or its file is missing or changed.
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem>;

    /// The imported display LUT with this id, as the C12 transform.
    ///
    /// # Errors
    ///
    /// [`LookProblem`] when the LUT is unknown, unusable or renders to a display the viewport cannot
    /// show.
    fn display_transform(&self, id: &LookId) -> Result<(DisplayTransform, String), LookProblem>;

    /// The SHA-256 (lower-case hex) of the look's file: the `ym:LookSHA256` an export records, so a
    /// file can be traced to the exact LUT that made it. `None` for a look that has no file of its
    /// own or when the provider does not know it (the property is then left out).
    fn look_sha256(&self, _id: &LookId) -> Option<String> {
        None
    }
}

/// Only the built-in Neutral look, no display LUTs (tests, and a core with no library).
#[derive(Debug, Clone, Copy, Default)]
pub struct NeutralOnly;

impl LookProvider for NeutralOnly {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        if id.as_str() == "builtin:neutral" {
            Ok(ResolvedLook {
                transform: Arc::new(LookTransform::neutral()),
                fingerprint: "builtin:neutral".to_owned(),
            })
        } else {
            Err(LookProblem(format!("The look {id} is not in the library.")))
        }
    }

    fn display_transform(&self, id: &LookId) -> Result<(DisplayTransform, String), LookProblem> {
        Err(LookProblem(format!(
            "The display LUT {id} is not in the library."
        )))
    }
}

impl LookProvider for Library {
    fn look_sha256(&self, id: &LookId) -> Option<String> {
        self.get(id)
            .and_then(|l| l.lut.as_ref().map(|r| r.sha256.clone()))
    }

    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        let transform = self.transform(id).map_err(|e| LookProblem(e.to_string()))?;
        let fingerprint = self
            .get(id)
            .and_then(|l| l.lut.as_ref().map(|r| r.sha256.clone()))
            .unwrap_or_else(|| format!("builtin:{id}"));
        Ok(ResolvedLook {
            transform,
            fingerprint,
        })
    }

    fn display_transform(&self, id: &LookId) -> Result<(DisplayTransform, String), LookProblem> {
        let info = self
            .display_lut_info(id)
            .ok_or_else(|| LookProblem(format!("The display LUT {id} is not in the library.")))?;
        let lut = self
            .display_lut(id)
            .map_err(|e| LookProblem(e.to_string()))?;
        let input = match info.input {
            DisplayInput::LogC4 => DisplayLutInput::LogC4,
            DisplayInput::LogC3 => DisplayLutInput::LogC3Ei800,
        };
        let output = DisplayLutOutput::from_target(&info.target)
            .map_err(|e| LookProblem(format!("This display LUT can't be used: {e}.")))?;
        let transform = ArriDisplayLut::new(lut, input, output)
            .map_err(|e| LookProblem(format!("This display LUT can't be used: {e}.")))?;
        Ok((DisplayTransform::arri(transform), info.lut.sha256.clone()))
    }
}

impl<T: LookProvider + ?Sized> LookProvider for Arc<T> {
    fn look_sha256(&self, id: &LookId) -> Option<String> {
        (**self).look_sha256(id)
    }

    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        (**self).look(id)
    }

    fn display_transform(&self, id: &LookId) -> Result<(DisplayTransform, String), LookProblem> {
        (**self).display_transform(id)
    }
}

impl LookProvider for RwLock<Library> {
    fn look_sha256(&self, id: &LookId) -> Option<String> {
        self.read()
            .unwrap_or_else(PoisonError::into_inner)
            .look_sha256(id)
    }

    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        self.read().unwrap_or_else(PoisonError::into_inner).look(id)
    }

    fn display_transform(&self, id: &LookId) -> Result<(DisplayTransform, String), LookProblem> {
        self.read()
            .unwrap_or_else(PoisonError::into_inner)
            .display_transform(id)
    }
}

impl LookProvider for Mutex<Library> {
    fn look_sha256(&self, id: &LookId) -> Option<String> {
        self.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .look_sha256(id)
    }

    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        self.lock().unwrap_or_else(PoisonError::into_inner).look(id)
    }

    fn display_transform(&self, id: &LookId) -> Result<(DisplayTransform, String), LookProblem> {
        self.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .display_transform(id)
    }
}

/// Which display rendering C12 uses (Preferences › Display rendering).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DisplayChoice {
    /// Ours: the ACES 2 output transform.
    #[default]
    Aces2,
    /// An imported ARRI display LUT.
    ArriLut(LookId),
}

// ---------------------------------------------------------------------------------------------
// The photo as the render sees it
// ---------------------------------------------------------------------------------------------

/// Everything the render graph needs to know about one photo at one moment: the decoded image,
/// the camera profile, the estimates and a snapshot of the settings.
#[derive(Debug, Clone)]
pub struct PhotoSource {
    /// The photo.
    pub id: PhotoId,
    /// The decoded raw image.
    pub raw: Arc<RawImage>,
    /// The camera profile.
    pub profile: Arc<Profile>,
    /// The automatic estimates (burst and pooled values applied).
    pub estimates: Arc<AutoEstimates>,
    /// The settings.
    pub settings: DevelopSettings,
    /// The Pick white sample that belongs to `settings.white_pick`.
    pub pick: Option<PickedWhite>,
    /// A fingerprint of the decoded pixels and levels: the graph drops its caches when the photo
    /// behind an id is not the one it cached (a second session, or a file that changed).
    pub fingerprint: u64,
    /// The published geometry of the camera's sensor, when the registry has it for this body
    /// (`CameraMatch::sensor`): the pixel pitch of R8 diffraction deconvolution.
    pub sensor: Option<SensorGeometry>,
    /// Which R8 capture-sharpening method the photo gets: an internal switch (no settings field),
    /// [`DEFAULT_CAPTURE_METHOD`] until the owner decides. Tests and the evaluation tools set it.
    pub capture_method: CaptureMethod,
    /// Whether this is the photo as the Before view reads it ([`PhotoSource::for_before_view`]): the raw
    /// stages run without R3, the lens vignetting correction, as the camera's own picture does. False for
    /// every other render. Set by `for_before_view` and nothing else, so a source made for the Before view
    /// can only be rendered as the Before view's raw front.
    pub before_view: bool,
}

/// A cheap fingerprint of a decoded image: its size, levels, white balance and a few thousand
/// samples spread over it (FNV-1a).
#[must_use]
pub fn raw_fingerprint(raw: &RawImage) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(raw.data.width as u64);
    mix(raw.data.height as u64);
    mix(u64::from(raw.is_reduced()));
    for v in raw.black.iter().chain(&raw.white) {
        mix(u64::from(v.to_bits()));
    }
    if let Some(wb) = raw.wb_as_shot {
        for v in wb {
            mix(u64::from(v.to_bits()));
        }
    }
    mix(u64::from(raw.meta.iso.unwrap_or(0)));
    mix(u64::from(raw.meta.orientation));
    let data = &raw.data.data;
    let step = (data.len() / 4096).max(1);
    for v in data.iter().step_by(step) {
        mix(u64::from(*v));
    }
    h
}

impl PhotoSource {
    /// From the session's view of a photo and its decoded image.
    #[must_use]
    pub fn from_view(view: &PhotoView, raw: Arc<RawImage>) -> Self {
        Self {
            id: view.id,
            profile: Arc::clone(&view.profile),
            estimates: Arc::clone(&view.estimates),
            settings: view.settings.clone(),
            pick: view.pick,
            fingerprint: raw_fingerprint(&raw),
            sensor: view.camera.matched.sensor,
            capture_method: DEFAULT_CAPTURE_METHOD,
            before_view: false,
            raw,
        }
    }

    /// From a session: the photo's current settings and estimates and its decoded image.
    ///
    /// # Errors
    ///
    /// [`SessionError`](crate::session::SessionError) for an unknown or skipped photo, one still
    /// being developed, or a file that cannot be decoded again.
    pub fn from_session(
        session: &crate::session::Session,
        id: PhotoId,
    ) -> Result<Self, crate::session::SessionError> {
        let view = session.view(id)?;
        let raw = session.raw_image(id)?;
        Ok(Self::from_view(&view, raw))
    }

    /// From the thumbnail hook's input (no Pick white yet: a photo is developed with the defaults).
    #[must_use]
    pub fn from_thumbnail_input(input: &ThumbnailInput<'_>) -> Self {
        Self {
            id: input.id,
            raw: Arc::clone(input.raw),
            profile: Arc::clone(input.profile),
            estimates: Arc::clone(input.estimates),
            settings: input.settings.clone(),
            pick: None,
            fingerprint: raw_fingerprint(input.raw),
            sensor: input.camera.matched.sensor,
            capture_method: DEFAULT_CAPTURE_METHOD,
            before_view: false,
        }
    }

    /// The same photo with other settings (look tiles, tests).
    #[must_use]
    pub fn with_settings(&self, settings: DevelopSettings) -> Self {
        Self {
            settings,
            ..self.clone()
        }
    }

    /// The photo as the Before view reads it: the same file, profile, estimates, decode switches, lens
    /// corrections (distortion and chromatic aberration, which move pixels, as for the final picture)
    /// and geometry, with the Detail controls that live in the raw group (noise reduction and capture
    /// sharpening) at Auto, and **without R3, the lens vignetting correction**. The Before view shows
    /// the camera default, and no setting reaches it; output sharpening is skipped by the view itself.
    /// The raw group is the final picture's except for R3, so the Before view has its own raw front,
    /// keyed apart from the final picture's ([`RawParams::vignetting`](super::raw::RawParams)).
    /// Owner's decisions of 2026-10-16 (`docs/STATUS.md`, question 29): no vignetting correction, and
    /// Detail stays at Auto whatever the Detail settings say.
    #[must_use]
    pub fn for_before_view(&self) -> Self {
        let mut settings = self.settings.clone();
        settings.noise_reduction = AutoOr::Auto;
        settings.sharpen = AutoOr::Auto;
        Self {
            before_view: true,
            ..self.with_settings(settings)
        }
    }

    /// The EXIF orientation of the file, 1 to 8.
    #[must_use]
    pub fn exif_orientation(&self) -> u16 {
        self.raw.meta.orientation
    }

    /// The Pick white sample to give the chain: the one that belongs to the current `white_pick`.
    #[must_use]
    pub fn picked_white(&self) -> Option<Vec3> {
        self.pick
            .filter(|p| Some(p.point) == self.settings.white_pick)
            .map(|p| p.sample)
    }
}

// ---------------------------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------------------------

/// Identifies the inputs of the colour group (C1 to C8): when it is equal, the fused matrix and the
/// tone parameters are the same.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColorKey(String);

/// Identifies the inputs of the grade group (C9 to C13) on top of the colour group.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GradeKey(String);

/// The key of the colour group of a photo.
#[must_use]
pub fn color_key(src: &PhotoSource) -> ColorKey {
    let s = &src.settings;
    let mut k = String::with_capacity(512);
    // `{:?}` of an f64 is its shortest round-trip form, so equal text is equal bits.
    let _ = write!(
        k,
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
        src.estimates.color,
        s.light_source,
        s.white_balance,
        s.exposure,
        s.white_pick,
        src.picked_white(),
        s.rolloff,
        (src.estimates.noise.model, src.profile.id.to_string()),
    );
    ColorKey(k)
}

// ---------------------------------------------------------------------------------------------
// The colour plan: C1 to C8
// ---------------------------------------------------------------------------------------------

/// C1 to C6 as one matrix, the parameters of C7 and the Before view's camera default.
#[derive(Debug, Clone)]
pub struct ColorPlan {
    /// The fused chain: matrix, factors, Kelvin and tint readout, clip level.
    pub fused: FusedChain,
    /// C7: the highlight range and the low-light toe.
    pub tone: ToneParams,
    /// The Before view: the camera's as-shot white balance, the body's colour matrix and the camera
    /// gain in one 3x3, and the camera tone curve. It depends on the file and the body only, never on
    /// a setting.
    pub camera_default: CameraDefault,
    /// What the plan was made from.
    pub key: ColorKey,
}

/// The Before view's camera default of a photo (`docs/02`, The Before view): the body's D65 colour matrix
/// (looked up by the file's Sony model ID, or by its Model string when the file has no ID; an ID that is not
/// listed gives the A7R IV entry, as in the profile registry), the
/// camera's as-shot white balance read from the file (our initial gains `g0` when the file has none),
/// and `g0`, which the raw stages have already applied to the camera RGB the chain receives.
///
/// Nothing here reads a setting. The numbers come from validated data and a validated estimate, so the
/// constructor does not fail for a photo that got this far; if it ever did, the placeholder
/// ([`CameraDefault::neutral`]) keeps the picture and the Before view working.
#[must_use]
pub fn camera_default_of(src: &PhotoSource, g0: Vec3) -> CameraDefault {
    let meta = &src.raw.meta;
    let body = ColorMatrices::shipped().lookup(&meta.make, &meta.model, meta.model_id);
    let gs = as_shot_gains(as_shot_rggb(&src.raw)).unwrap_or(g0);
    CameraDefault::new(&body.body.xyz_d65_to_camera, gs, g0)
        .unwrap_or_else(|_| CameraDefault::neutral())
}

/// Builds the colour plan of a photo.
///
/// # Errors
///
/// [`RenderError::Color`] for a Pick white sample the chain refuses ("That spot is too bright to
/// use ...") or values that cannot make a chain.
pub fn color_plan(src: &PhotoSource) -> Result<ColorPlan, RenderError> {
    let fused = src
        .estimates
        .fused_chain(&src.profile, &src.settings, src.picked_white())
        .map_err(|e| RenderError::Color(e.to_string()))?;
    let g0 = src
        .estimates
        .initial_gains()
        .map_err(|e| RenderError::Color(e.to_string()))?;
    let noise = &src.estimates.noise.model;
    let read_sigma = (noise.b + noise.sigma_row * noise.sigma_row + noise.q * noise.q / 12.0)
        .max(0.0)
        .sqrt();
    let m = fused.matrix();
    let floor = (0..3)
        .map(|j| {
            (0..3)
                .map(|i| (m[j][i] * g0[i] * read_sigma).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .sum::<f64>()
        / 3.0;
    let r = &src.settings.rolloff;
    let tone = ToneParams::new(r.highlights, r.low_light, fused.clip().luma(), floor)
        .map_err(|e| RenderError::Color(e.to_string()))?;
    Ok(ColorPlan {
        fused,
        tone,
        camera_default: camera_default_of(src, g0),
        key: color_key(src),
    })
}

// ---------------------------------------------------------------------------------------------
// The grade plan: C9 to C13
// ---------------------------------------------------------------------------------------------

/// The grade controls of the settings as `ym_color`'s parameters. An Auto contrast, highlights,
/// shadows, black, saturation or shadows or highlights wheel resolves against the photo's estimates
/// (0 or `[0, 0]`, or Apple Auto's value, `docs/19` section 5).
#[must_use]
pub fn grade_params(s: &DevelopSettings, estimates: &AutoEstimates) -> GradeParams {
    let b = |p: [f64; 2]| ColorBalance { a: p[0], b: p[1] };
    GradeParams {
        contrast: s.contrast.resolve(estimates.contrast),
        brightness: s.brightness,
        shadows: s.shadows.resolve(estimates.shadows),
        highlights: s.highlights.resolve(estimates.highlights),
        black: s.black.resolve(estimates.black),
        white: s.white,
        balance_shadows: b(s.color_balance.shadows.resolve(estimates.balance_shadows)),
        balance_midtones: b(s.color_balance.midtones),
        balance_highlights: b(s
            .color_balance
            .highlights
            .resolve(estimates.balance_highlights)),
        saturation: s.saturation.resolve(estimates.saturation),
    }
}

/// The colour warp's pins as `ym_color`'s parameters (stage C9b). No pins is the identity.
#[must_use]
pub fn warp_params(s: &DevelopSettings) -> WarpParams {
    s.color_warp.to_color()
}

fn curve_of(points: &[CurvePoint]) -> Result<Curve, RenderError> {
    let pts: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();
    Curve::new(&pts).map_err(|e| RenderError::Color(format!("The curve can't be used: {e}.")))
}

/// The four curves of the settings.
///
/// # Errors
///
/// [`RenderError::Color`] for a curve `ym_color` refuses (settings validation normally stops
/// these earlier).
pub fn curve_set(s: &DevelopSettings) -> Result<CurveSet, RenderError> {
    Ok(CurveSet {
        luma: curve_of(&s.curve.luma)?,
        r: curve_of(&s.curve.red)?,
        g: curve_of(&s.curve.green)?,
        b: curve_of(&s.curve.blue)?,
    })
}

/// What the grade group is made for.
#[derive(Clone)]
pub struct GradeRequest<'a> {
    /// The output the display values are encoded for.
    pub target: OutputTarget,
    /// Whether 8-bit outputs are dithered.
    pub dither: bool,
    /// ACES 2 or an imported display LUT.
    pub display: &'a DisplayChoice,
    /// Where looks come from.
    pub looks: &'a dyn LookProvider,
}

/// A prepared C7 to C13 chain and what to tell the user about it.
#[derive(Debug, Clone)]
pub struct GradePlan {
    /// The chain.
    pub chain: Arc<ColorChain>,
    /// What the plan was made from.
    pub key: GradeKey,
    /// Plain-words notes: a look or display LUT that could not be used and was replaced.
    pub warnings: Vec<String>,
}

/// The look and the display transform as the provider answers for them now.
struct Pieces {
    look: Option<ResolvedLook>,
    display: Option<(DisplayTransform, String)>,
    warnings: Vec<String>,
}

fn pieces(src: &PhotoSource, req: &GradeRequest<'_>) -> Pieces {
    let mut warnings = Vec::new();
    let look = match req.looks.look(&src.settings.look) {
        Ok(l) => Some(l),
        Err(e) => {
            warnings.push(e.to_string());
            None
        }
    };
    let display = match req.display {
        DisplayChoice::Aces2 => None,
        DisplayChoice::ArriLut(id) => match req.looks.display_transform(id) {
            Ok(d) => Some(d),
            Err(e) => {
                warnings.push(e.to_string());
                None
            }
        },
    };
    Pieces {
        look,
        display,
        warnings,
    }
}

/// The white lift a photo renders with: the automatic amount and anchor of its estimates plus the
/// user's offset (`white_offset_ev`), `docs/02` C12 step 1b. It acts after C11, alongside C7's
/// highlight range (which [`color_plan`] puts in the tone), and has an effect on the ACES 2 path only;
/// the chain ignores it for an imported ARRI display LUT.
#[must_use]
pub fn white_lift(src: &PhotoSource) -> WhiteLift {
    src.estimates.white.lift(src.settings.white_offset_ev)
}

fn key_of(src: &PhotoSource, color: &ColorPlan, req: &GradeRequest<'_>, p: &Pieces) -> GradeKey {
    let s = &src.settings;
    let mut key = String::with_capacity(768);
    let _ = write!(
        key,
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{:?}|",
        color.key,
        grade_params(s, &src.estimates),
        // The warp as the chain prepares it: pins that change nothing (locked pins with no pin that
        // moves, a pin whose target is its source) leave the key, and the frame, as without them.
        Warp::new(&warp_params(s)),
        s.curve,
        req.target,
        req.dither,
        s.look_intensity,
        white_lift(src),
        req.display,
        s.look,
        // The Before view's parameters: the as-shot white balance and the body's matrix (never a
        // setting), so that a frame of the Before view is not served for another body or white.
        color.camera_default,
    );
    match &p.look {
        Some(l) => key.push_str(&l.fingerprint),
        None => key.push_str("neutral-fallback"),
    }
    key.push('|');
    match (&p.display, req.display) {
        (Some((_, fp)), _) => key.push_str(fp),
        (None, DisplayChoice::ArriLut(_)) => key.push_str("aces2-fallback"),
        (None, DisplayChoice::Aces2) => key.push_str("aces2"),
    }
    GradeKey(key)
}

/// The key a grade plan for these inputs has, without building the chain.
#[must_use]
pub fn grade_key(src: &PhotoSource, color: &ColorPlan, req: &GradeRequest<'_>) -> GradeKey {
    key_of(src, color, req, &pieces(src, req))
}

/// Builds the grade plan: the look and the display transform are fetched from the provider (a
/// missing look renders as Neutral and says so in [`GradePlan::warnings`]).
///
/// # Errors
///
/// [`RenderError::Color`] when the chain cannot be prepared.
pub fn grade_plan(
    src: &PhotoSource,
    color: &ColorPlan,
    req: &GradeRequest<'_>,
) -> Result<GradePlan, RenderError> {
    let s = &src.settings;
    let p = pieces(src, req);
    let key = key_of(src, color, req, &p);
    let mut params = ChainParams::new(*color.fused.matrix(), color.tone);
    params.grade = grade_params(s, &src.estimates);
    params.warp = warp_params(s);
    params.curves = curve_set(s)?;
    params.output = req.target;
    params.dither = req.dither;
    params.before = BeforePoint::CameraDefault;
    params.camera_default = color.camera_default;
    params.white_lift = white_lift(src);
    if let Some(look) = p.look {
        params.look = Some(LookSetting::new(look.transform, s.look_intensity));
    }
    if let Some((transform, _)) = p.display {
        params.display = transform;
    }
    let chain = ColorChain::new(&params).map_err(|e| RenderError::Color(e.to_string()))?;
    Ok(GradePlan {
        chain: Arc::new(chain),
        key,
        warnings: p.warnings,
    })
}

/// A look applied to the same photo: the settings with another look at 100 % (look tiles).
#[must_use]
pub fn with_look(settings: &DevelopSettings, look: &LookId) -> DevelopSettings {
    let mut s = settings.clone();
    s.look = look.clone();
    s.look_intensity = 100.0;
    s
}
