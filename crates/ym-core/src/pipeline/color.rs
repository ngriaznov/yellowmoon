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
//!   (stage C9b), the four curves a `CurveSet`: Main's from its grade, every other live corrector's from
//!   its own (`NodeSet`), with its trim and its mask rows ([`node_masks`]: the shapes and ranges, the raster
//!   kinds as rows that select nothing until their tasks); the graph's links are compiled into a `Plan`
//!   (`ym_color::plan`, [`node_plan`]) that says which nodes run and in what order, and which node's key
//!   output feeds which key input (`docs/23` section 15);
//! * the look comes from a [`LookProvider`] (the looks library in the app, [`NeutralOnly`] in tests);
//!   the display transform is always Yellowmoon's own, the cine rendering (C12), with nothing in front of it.
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
    BeforePoint, ChainParams, ColorChain, FilmStep, JoinParams, LookSetting, MainNode, NodeParams,
    NodeSet, OutputTarget, Trim,
};
use ym_color::grade::{ColorBalance, GradeParams};
use ym_color::look::LookTransform;
use ym_color::mask::{MaskKind, MaskRow, NodeMasks};
use ym_color::mask_depth::DepthSource;
use ym_color::mask_raster::{
    BackgroundSource, BlurMap, BrushStrokes, MAX_RASTER_PLANES, RasterRow, RasterSet, RasterSource,
    Stroke, strokes_key,
};
use ym_color::plan::{
    self, JoinInput, JoinKind, PictureSource, Plan, PlanGraph, PlanJoin, PlanNode, Stop,
};

use ym_color::tone::ToneParams;
use ym_color::warp::{Warp, WarpParams};
use ym_looks::library::{Library, LookId};
use ym_profiles::color_matrices::ColorMatrices;
use ym_profiles::profile::Profile;
use ym_profiles::registry::SensorGeometry;
use ym_raw::decode::RawImage;

use super::RenderError;
use super::object::ObjectPlanes;
use super::raw::{CaptureMethod, DEFAULT_CAPTURE_METHOD};
use crate::auto::{AutoEstimates, as_shot_gains, as_shot_rggb};
use crate::session::{PhotoId, PhotoView, PickedWhite, ThumbnailInput};
use crate::settings::{
    AutoOr, BlendMode, ColorWarp, CurvePoint, Curves, DevelopSettings, LensSettings, MIXER_INPUTS,
    MaskMode, MaskSettings, MaskShape, MaskSupport, MixerNode, NodeGrade, NodeGraph, NodeId,
    NodeKind, NodeTrim, ObjectMask, Port,
};
use crate::vision::VisionKind;

// ---------------------------------------------------------------------------------------------
// Looks
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

/// Where the render graph gets looks from.
pub trait LookProvider: Send + Sync {
    /// The look with this id.
    ///
    /// # Errors
    ///
    /// [`LookProblem`] when the look is unknown or its file is missing or changed.
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem>;

    /// The SHA-256 (lower-case hex) of the look's file: the `ym:LookSHA256` an export records, so a
    /// file can be traced to the exact LUT that made it. `None` for a look that has no file of its
    /// own or when the provider does not know it (the property is then left out).
    fn look_sha256(&self, _id: &LookId) -> Option<String> {
        None
    }
}

/// Only the built-in Neutral look (tests, and a core with no library).
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
}

impl<T: LookProvider + ?Sized> LookProvider for Arc<T> {
    fn look_sha256(&self, id: &LookId) -> Option<String> {
        (**self).look_sha256(id)
    }

    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        (**self).look(id)
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
        settings.input_mut().noise_reduction = AutoOr::Auto;
        settings.input_mut().sharpen = AutoOr::Auto;
        // The camera default has no lens effect at all (`docs/21` section 1).
        settings.input_mut().lens = LensSettings::default();
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
            .filter(|p| Some(p.point) == self.settings.input().white_pick)
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
        s.input().light_source,
        s.input().white_balance,
        s.input().exposure,
        s.input().white_pick,
        src.picked_white(),
        s.input().rolloff,
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
    let r = &src.settings.input().rolloff;
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

/// A grade's controls as `ym_color`'s parameters. In Main an Auto contrast, highlights, shadows,
/// black, saturation or shadows or highlights wheel resolves against the photo's `estimates` (0 or
/// `[0, 0]`, or Apple Auto's value, `docs/19` section 5); with no estimates, as for every other
/// corrector, Auto would read as 0 (a corrector other than Main has none: `docs/23` section 3.2).
fn grade_params_of(g: &NodeGrade, estimates: Option<&AutoEstimates>) -> GradeParams {
    let b = |p: [f64; 2]| ColorBalance { a: p[0], b: p[1] };
    let num = |f: fn(&AutoEstimates) -> f64| estimates.map_or(0.0, f);
    let point = |f: fn(&AutoEstimates) -> [f64; 2]| estimates.map_or([0.0, 0.0], f);
    GradeParams {
        contrast: g.contrast.resolve(num(|e| e.contrast)),
        brightness: g.brightness,
        shadows: g.shadows.resolve(num(|e| e.shadows)),
        highlights: g.highlights.resolve(num(|e| e.highlights)),
        black: g.black.resolve(num(|e| e.black)),
        white: g.white,
        balance_shadows: b(g
            .color_balance
            .shadows
            .resolve(point(|e| e.balance_shadows))),
        balance_midtones: b(g.color_balance.midtones),
        balance_highlights: b(g
            .color_balance
            .highlights
            .resolve(point(|e| e.balance_highlights))),
        saturation: g.saturation.resolve(num(|e| e.saturation)),
    }
}

/// Main's grade controls as `ym_color`'s parameters. An Auto contrast, highlights, shadows, black,
/// saturation or shadows or highlights wheel resolves against the photo's estimates (0 or `[0, 0]`, or
/// Apple Auto's value, `docs/19` section 5).
#[must_use]
pub fn grade_params(s: &DevelopSettings, estimates: &AutoEstimates) -> GradeParams {
    grade_params_of(s.main_grade(), Some(estimates))
}

/// Main's colour warp's pins as `ym_color`'s parameters (stage C9b). No pins is the identity.
#[must_use]
pub fn warp_params(s: &DevelopSettings) -> WarpParams {
    s.main_grade().color_warp.to_color()
}

fn curve_of(points: &[CurvePoint]) -> Result<Curve, RenderError> {
    let pts: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();
    Curve::new(&pts).map_err(|e| RenderError::Color(format!("The curve can't be used: {e}.")))
}

fn curves_of(c: &Curves) -> Result<CurveSet, RenderError> {
    Ok(CurveSet {
        luma: curve_of(&c.luma)?,
        r: curve_of(&c.red)?,
        g: curve_of(&c.green)?,
        b: curve_of(&c.blue)?,
    })
}

/// Main's four curves.
///
/// # Errors
///
/// [`RenderError::Color`] for a curve `ym_color` refuses (settings validation normally stops
/// these earlier).
pub fn curve_set(s: &DevelopSettings) -> Result<CurveSet, RenderError> {
    curves_of(&s.main_grade().curve)
}

fn trim_of(t: &NodeTrim) -> Trim {
    Trim {
        exposure_ev: t.exposure_ev,
        temperature: t.temperature,
        tint: t.tint,
    }
}

/// The plane index of a raster row that has no plane: a kind this machine cannot make (`docs/23` section 3.10).
/// The row is in the fold and selects nothing, in a view; an export of the photo is refused
/// ([`unmakeable_mask`]).
pub const NO_PLANE: usize = usize::MAX;

/// Whether a row of this shape is made into a raster plane by this code: Brush (drawn from its strokes), Background (cut
/// from the photo's blur map), Depth (cut from the photo's depth map), Object (SAM's decoder on its points and box, N-18) and Subject and People (the union of
/// Vision's instances on the Input's picture, N-19). Whether the machine *can* make it is [`unmakeable_mask_with`].
fn makes_raster(shape: &MaskShape) -> bool {
    matches!(
        shape,
        MaskShape::Brush(_)
            | MaskShape::Background(_)
            | MaskShape::Depth(_)
            | MaskShape::Object(_)
            | MaskShape::Subject(_)
            | MaskShape::People(_)
    )
}

/// The raster row a mask row stands for, when this machine can make its `S0`: a Brush row's strokes with its
/// Grow, Feather and Refine edges, or a Background row's blur map with its Level, Softness and the same three
/// (`background` is the photo's map, which only the renderer can make: without it a Background row has no
/// plane and selects nothing), or an Object row's plane from SAM's decoder (`objects`, made by the renderer from the
/// photo's features: a row with no point and no box, or whose features are not there, has no plane and selects nothing),
/// or a Subject or People row's plane from Vision (`objects`, the same planes: a row on a computer that cannot make it,
/// whose mask is not made yet, or that found nothing, has no plane and selects nothing).
fn raster_row_of(
    shape: &MaskShape,
    background: Option<&Arc<BlurMap>>,
    objects: Option<&Arc<ObjectPlanes>>,
    node: u32,
) -> Option<RasterRow> {
    match shape {
        MaskShape::Object(o) => objects.and_then(|p| p.plane_for(o)).map(|plane| RasterRow {
            source: RasterSource::Plane(Arc::clone(plane)),
            grow: o.grow,
            feather: o.feather,
            refine: o.refine,
        }),
        MaskShape::Subject(f) | MaskShape::People(f) => {
            let kind = VisionKind::of(crate::settings::MaskKind::of(shape))?;
            objects
                .and_then(|p| p.vision_plane(kind))
                .map(|plane| RasterRow {
                    source: RasterSource::Plane(Arc::clone(plane)),
                    grow: f.grow,
                    feather: f.feather,
                    refine: f.refine,
                })
        }
        MaskShape::Background(b) => background.map(|map| RasterRow {
            source: RasterSource::Background(Arc::new(BackgroundSource {
                map: Arc::clone(map),
                level: b.level,
                softness: b.softness,
            })),
            grow: b.grow,
            feather: b.feather,
            refine: b.refine,
        }),
        // Show depth: the whole map as grey, with none of the row's steps (the view shows what the band is cut from).
        MaskShape::Depth(_) if objects.is_some_and(|p| p.depth_show_for(node).is_some()) => objects
            .and_then(|p| p.depth_show_for(node))
            .map(|plane| RasterRow {
                source: RasterSource::Plane(Arc::clone(plane)),
                grow: 0.0,
                feather: 0.0,
                refine: 0.0,
            }),
        MaskShape::Depth(d) => objects.and_then(|p| p.depth_map()).map(|map| RasterRow {
            source: RasterSource::Depth(Arc::new(DepthSource {
                map: Arc::clone(map),
                far: d.far,
            })),
            grow: d.grow,
            feather: d.feather,
            refine: d.refine,
        }),
        MaskShape::Brush(b) => Some(RasterRow {
            source: RasterSource::Brush(Arc::new(BrushStrokes::new(
                b.strokes
                    .iter()
                    .map(|s| Stroke {
                        size: s.size,
                        hardness: s.hardness,
                        erase: s.erase,
                        points: s.points.clone(),
                    })
                    .collect(),
            ))),
            grow: b.grow,
            feather: b.feather,
            refine: b.refine,
        }),
        _ => None,
    }
}

/// The raster row of a Face Refinement node's protection plane (a Brush row's strokes with its Grow, Feather and Refine), or
/// `None` when nothing is painted.
pub(crate) fn protect_row(node: &crate::settings::FaceRefinementNode) -> Option<RasterRow> {
    if node.protect.strokes.is_empty() {
        return None;
    }
    raster_row_of(&MaskShape::Brush(node.protect.clone()), None, None, 0)
}

fn mask_rows(
    m: &MaskSettings,
    rasters: &mut RasterSet,
    background: Option<&Arc<BlurMap>>,
    objects: Option<&Arc<ObjectPlanes>>,
    node: u32,
) -> NodeMasks {
    let rows = m
        .items
        .iter()
        .filter(|row| row.enabled)
        .map(|row| MaskRow {
            kind: match &row.shape {
                MaskShape::Linear(l) => MaskKind::Linear {
                    center: l.center,
                    angle_deg: l.angle_deg,
                    feather: l.feather,
                },
                MaskShape::Radial(r) => MaskKind::Radial {
                    center: r.center,
                    radii: r.radii,
                    angle_deg: r.angle_deg,
                    feather: r.feather,
                },
                MaskShape::ColorRange(c) => MaskKind::ColorRange {
                    center: c.center,
                    width: c.width,
                    softness: c.softness,
                },
                MaskShape::BrightnessRange(b) => MaskKind::BrightnessRange {
                    low: b.low,
                    high: b.high,
                    softness: b.softness,
                },
                MaskShape::Object(_)
                | MaskShape::Subject(_)
                | MaskShape::People(_)
                | MaskShape::Background(_)
                | MaskShape::Depth(_)
                | MaskShape::Brush(_) => match raster_row_of(&row.shape, background, objects, node)
                {
                    Some(r) if rasters.rows.len() < MAX_RASTER_PLANES => {
                        rasters.rows.push(r);
                        MaskKind::Raster {
                            plane: rasters.rows.len() - 1,
                        }
                    }
                    _ => MaskKind::Raster { plane: NO_PLANE },
                },
            },
            subtract: row.mode == MaskMode::Subtract,
            invert: row.invert,
            opacity: row.opacity,
        })
        .collect();
    NodeMasks {
        rows,
        invert: m.invert,
    }
}

/// The settings as the mask view of one row reads them (`docs/23` section 5.8): corrector `node` has only row `row`, on, added,
/// with its own Invert and Opacity kept, no Invert of the node's own and no key link, so that the node's mask `m` is that row's
/// `a`: its source, its raster steps, its own Invert and its Opacity, as it enters the node's fold. A Subtract row is shown as
/// the positive mask it takes away, and a row with the eye off can be shown. `None` when the node has no such row.
#[must_use]
pub fn only_row(s: &DevelopSettings, node: u32, row: usize) -> Option<DevelopSettings> {
    let mut only = s.clone();
    let masks = only.nodes.masks_of_mut(node)?;
    let mut item = masks.items.get(row)?.clone();
    item.mode = crate::settings::MaskMode::Add;
    item.enabled = true;
    masks.items = vec![item];
    masks.invert = false;
    let to = NodeId::Corrector(node);
    only.nodes
        .links
        .retain(|l| !(l.to.node == to && l.to.port == Port::Key));
    Some(only)
}

/// A node's mask rows as `ym_color`'s: the enabled rows in order (a row with the eye off is absent), each
/// with its kind, mode, Invert and Opacity, and the node's Invert. A raster row (Brush and Background now; Object,
/// Subject and People when they are built) is a [`MaskKind::Raster`] whose plane is its place among the
/// raster rows of a chain; this function numbers them from 0 for itself alone (a Background row has no plane here:
/// it needs the photo's blur map), and the chain's builder numbers them across the nodes of a plan.
#[must_use]
pub fn node_masks(m: &MaskSettings) -> NodeMasks {
    mask_rows(m, &mut RasterSet::default(), None, None, 0)
}

/// Whether the photo's plan that ends at `stop` has a Background row with the eye on: the render then needs the
/// photo's blur map ([`GradeRequest::background`]), made once per photo.
#[must_use]
pub fn uses_background(s: &DevelopSettings, stop: Stop) -> bool {
    let Ok(plan) = node_plan(&s.nodes, stop) else {
        return false;
    };
    plan.node_ids().into_iter().any(|id| {
        s.nodes.masks_of(id).is_some_and(|m| {
            m.items
                .iter()
                .any(|r| r.enabled && matches!(r.shape, MaskShape::Background(_)))
        })
    })
}

/// Whether the photo's plan that ends at `stop` has a Depth row with the eye on: the render then needs the photo's depth
/// map ([`ObjectPlanes::depth_map`]), made once per photo by the background worker.
#[must_use]
pub fn uses_depth(s: &DevelopSettings, stop: Stop) -> bool {
    let Ok(plan) = node_plan(&s.nodes, stop) else {
        return false;
    };
    plan.node_ids().into_iter().any(|id| {
        s.nodes.masks_of(id).is_some_and(|m| {
            m.items
                .iter()
                .any(|r| r.enabled && matches!(r.shape, MaskShape::Depth(_)))
        })
    })
}

/// The Face Refinement nodes of the plan that ends at `stop` that do something (not bypassed, and some tool of some face above
/// 0, [`FaceRefinementNode::is_inert`](crate::settings::FaceRefinementNode::is_inert)), in the plan's order. The render then
/// needs the photo's faces ([`ObjectPlanes::faces`]) and runs on the CPU.
#[must_use]
pub fn live_face_nodes(s: &DevelopSettings, stop: Stop) -> Vec<u32> {
    let Ok(plan) = node_plan(&s.nodes, stop) else {
        return Vec::new();
    };
    plan.node_ids()
        .into_iter()
        .filter(|&id| s.nodes.face_refinement(id).is_some_and(|f| !f.is_inert()))
        .collect()
}

/// Whether the photo has a Face Refinement node anywhere in its graph, live or not, in a node the Output needs or not: the
/// core starts the face worker in the background for the photo being looked at (`docs/27` decision 9), and the node's panel
/// says "Finding faces…" while it runs.
#[must_use]
pub fn has_face_node(s: &DevelopSettings) -> bool {
    s.nodes
        .numbered()
        .into_iter()
        .any(|(_, kind)| kind == crate::settings::NodeKind::FaceRefinement)
}

/// Whether the photo has a Depth row with the eye on anywhere in its graph, in a node the Output needs or not: the core
/// starts the depth worker in the background for the photo being looked at (`docs/23` section 4.8), and the app says
/// "Finding…" while it runs.
#[must_use]
pub fn has_depth_row(s: &DevelopSettings) -> bool {
    s.nodes.all_masks().any(|(_, m)| {
        m.items
            .iter()
            .any(|r| r.enabled && matches!(r.shape, MaskShape::Depth(_)))
    })
}

/// Whether the photo has a Background row with the eye on anywhere in its graph, in a node the Output needs or not
/// (the mask view can show a dead node's mask): the app says "Finding…" while the first blur map is made.
#[must_use]
pub fn has_background_row(s: &DevelopSettings) -> bool {
    s.nodes.all_masks().any(|(_, m)| {
        m.items
            .iter()
            .any(|r| r.enabled && matches!(r.shape, MaskShape::Background(_)))
    })
}

/// Whether the photo's plan has a node with a mask row that reads the pixel's place in the photo (a Linear
/// or a Radial): the render then needs the position of every pixel (`ColorChain::uses_photo_position`).
#[must_use]
pub fn uses_photo_position(s: &DevelopSettings) -> bool {
    uses_photo_position_to(s, Stop::Output)
}

/// [`uses_photo_position`] for the plan that ends at `stop`: the mask view's plan ends at a node's mask and
/// includes the node even when the Output does not need it.
#[must_use]
pub fn uses_photo_position_to(s: &DevelopSettings, stop: Stop) -> bool {
    let Ok(plan) = node_plan(&s.nodes, stop) else {
        return false;
    };
    plan.node_ids().into_iter().any(|id| {
        s.nodes
            .masks_of(id)
            .is_some_and(|m| node_masks(m).prepare().needs_position())
    })
}

/// The sentence for the first mask of the photo that a machine with nothing beyond Yellowmoon's own code cannot make:
/// [`unmakeable_mask_with`] of [`MaskSupport::bare`]. A renderer asks with its own support
/// ([`Renderer::unmakeable_mask`](super::Renderer)).
#[must_use]
pub fn unmakeable_mask(s: &DevelopSettings) -> Option<String> {
    unmakeable_mask_with(s, MaskSupport::bare())
}

/// The sentence for the first mask of the photo that this computer cannot make (`docs/23` section 3.10): a raster
/// row of a kind that is not in [`mask_kinds_with`](crate::settings::mask_kinds_with) (`support`: whether the SAM weights
/// are there, and what the Vision probe said for Subject and People) in a node the Output needs, with the eye on; or more raster
/// masks than a render holds ([`MAX_RASTER_PLANES`]). `None` when the photo can be rendered with all its masks. A dead
/// node's rows and a row with the eye off do not count: they are not in the picture. An Object row with no point and no box
/// counts: it selects nothing, but a copy of Yellowmoon without the model says so rather than export the photo as if the
/// row were fine.
#[must_use]
pub fn unmakeable_mask_with(s: &DevelopSettings, support: MaskSupport) -> Option<String> {
    let plan = node_plan(&s.nodes, Stop::Output).ok()?;
    let mut planes = 0usize;
    for id in plan.node_ids() {
        let Some(masks) = s.nodes.masks_of(id) else {
            continue;
        };
        for row in masks.items.iter().filter(|r| r.enabled) {
            if let Some(sentence) = crate::settings::unmakeable_mask_sentence_with(
                support,
                id,
                crate::settings::MaskKind::of(&row.shape),
            ) {
                return Some(sentence);
            }
            planes += usize::from(makes_raster(&row.shape));
        }
    }
    (planes > MAX_RASTER_PLANES).then(|| {
        format!(
            "This photo has more than {MAX_RASTER_PLANES} masks that are drawn or found. Remove some of them."
        )
    })
}

/// The Object rows with the eye on of the nodes of the plan that ends at `stop` (the Output, or a node's mask for the mask
/// view), in the plan's order and the list's.
#[must_use]
pub fn object_rows(s: &DevelopSettings, stop: Stop) -> Vec<&ObjectMask> {
    let Ok(plan) = node_plan(&s.nodes, stop) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for id in plan.node_ids() {
        if let Some(masks) = s.nodes.masks_of(id) {
            rows.extend(
                masks
                    .items
                    .iter()
                    .filter(|r| r.enabled)
                    .filter_map(|r| match &r.shape {
                        MaskShape::Object(o) => Some(o),
                        _ => None,
                    }),
            );
        }
    }
    rows
}

/// Whether the photo has an Object row with the eye on anywhere in its graph, in a node the Output needs or not: the core
/// starts SAM's encoder in the background for the photo being looked at (`docs/23` section 4.4).
#[must_use]
pub fn has_object_row(s: &DevelopSettings) -> bool {
    s.nodes.all_masks().any(|(_, m)| {
        m.items
            .iter()
            .any(|r| r.enabled && matches!(r.shape, MaskShape::Object(_)))
    })
}

/// The Vision requests that the Subject and People rows with the eye on need, in the nodes of the plan that ends at
/// `stop` (the Output, or a node's mask for the mask view), Subject first.
#[must_use]
pub fn vision_kinds_to(s: &DevelopSettings, stop: Stop) -> Vec<VisionKind> {
    let Ok(plan) = node_plan(&s.nodes, stop) else {
        return Vec::new();
    };
    let mut kinds = Vec::new();
    for id in plan.node_ids() {
        if let Some(masks) = s.nodes.masks_of(id) {
            kinds.extend(
                masks
                    .items
                    .iter()
                    .filter(|r| r.enabled)
                    .filter_map(|r| VisionKind::of(crate::settings::MaskKind::of(&r.shape))),
            );
        }
    }
    kinds.sort_unstable();
    kinds.dedup();
    kinds
}

/// The Vision requests that the Subject and People rows with the eye on need anywhere in the photo's graph, in a node
/// the Output needs or not: the core starts them in the background for the photo being looked at (`docs/23`
/// section 4.5), and the app says "Finding…" while they run.
#[must_use]
pub fn vision_kinds(s: &DevelopSettings) -> Vec<VisionKind> {
    let mut kinds: Vec<VisionKind> = s
        .nodes
        .all_masks()
        .flat_map(|(_, m)| {
            m.items
                .iter()
                .filter(|r| r.enabled)
                .filter_map(|r| VisionKind::of(crate::settings::MaskKind::of(&r.shape)))
        })
        .collect();
    kinds.sort_unstable();
    kinds.dedup();
    kinds
}

/// Whether the photo's plan has a node with a raster row: the render then makes the raster planes over its window
/// before the colour pass, and a view too big for one device window is made on the CPU.
#[must_use]
pub fn uses_rasters(s: &DevelopSettings) -> bool {
    let Ok(plan) = node_plan(&s.nodes, Stop::Output) else {
        return false;
    };
    plan.node_ids().into_iter().any(|id| {
        s.nodes
            .masks_of(id)
            .is_some_and(|m| node_masks(m).prepare().needs_rasters())
    })
}

/// Whether the photo's plan has a Film node that does something (not bypassed, Strength above 0): the picture then has
/// the Film passes (`docs/24` section 3.9; on the device `docs/24` section 3.9c, on the CPU `frame::film_pass`), and a DNG
/// says so.
#[must_use]
pub fn has_film(s: &DevelopSettings) -> bool {
    let Ok(plan) = node_plan(&s.nodes, Stop::Output) else {
        return false;
    };
    plan.node_ids()
        .into_iter()
        .any(|id| s.nodes.film(id).is_some_and(|f| !f.is_inert()))
}

/// The picture a link into `to`'s `port` carries: `None` when the port has no link (the node then reads the Input's
/// picture), else where it comes from. A link from a Splitter's `red`, `green` or `blue` is that channel.
fn picture_into(g: &NodeGraph, to: NodeId, port: Port) -> Option<PictureSource> {
    g.links
        .iter()
        .find(|l| l.to.node == to && l.to.port == port)
        .map(|l| match l.from.node {
            NodeId::Corrector(m) => match l.from.port {
                Port::Red => PictureSource::Split(m, 0),
                Port::Green => PictureSource::Split(m, 1),
                Port::Blue => PictureSource::Split(m, 2),
                _ => PictureSource::Node(m),
            },
            NodeId::Input | NodeId::Output => PictureSource::Input,
        })
}

/// The graph as the plan compiler reads it: every corrector and Film node with its place, its picture source
/// (the link into its picture input) and its key source, and the picture the Output reads. An input with no link
/// is `None` (the compiler reads the Input's picture there). The mixers, Splitters and Combiners are
/// [`plan_joins`].
#[must_use]
pub fn plan_graph(g: &NodeGraph) -> PlanGraph {
    let nodes = g
        .numbered()
        .into_iter()
        .filter_map(|(id, kind)| {
            match kind {
                // A Face Refinement node is a step of the plan like a Film node's (`node_set`), with its place, its picture
                // source and its key like a corrector's.
                NodeKind::Corrector | NodeKind::Film | NodeKind::FaceRefinement => {}
                NodeKind::Parallel
                | NodeKind::Layer
                | NodeKind::Splitter
                | NodeKind::Combiner
                | NodeKind::Input
                | NodeKind::Output => return None,
            }
            let pos = g.pos(NodeId::Corrector(id))?;
            Some(PlanNode {
                id,
                pos,
                picture: picture_into(g, NodeId::Corrector(id), Port::Picture),
                key: g
                    .links
                    .iter()
                    .find(|l| l.to.node == NodeId::Corrector(id) && l.to.port == Port::Key)
                    .and_then(|l| match l.from.node {
                        NodeId::Corrector(m) => Some(m),
                        _ => None,
                    }),
            })
        })
        .collect();
    let output = picture_into(g, NodeId::Output, Port::Picture).unwrap_or(PictureSource::Input);
    PlanGraph { nodes, output }
}

/// The mixers, Splitters and Combiners of the graph as the plan compiler reads them (`docs/25` section 4.1): their
/// picture inputs in port order, and for a Layer's input above the first whether it is linked from the picture output
/// of a corrector or Film node that is not bypassed (the Layer then reads that node's picture before its own mask
/// blend, with its `m` as the alpha: `docs/25` section 3.3).
#[must_use]
pub fn plan_joins(g: &NodeGraph) -> Vec<PlanJoin> {
    // The picture output of a node that has a mask `m` to give: a corrector or Film node that is not bypassed.
    let masked = |src: PictureSource| match src {
        PictureSource::Node(n) => match g.kind_of(NodeId::Corrector(n)) {
            Some(NodeKind::Corrector) => g.corrector(n).is_some_and(|c| !c.bypass),
            Some(NodeKind::Film) => g.film(n).is_some_and(|f| !f.bypass),
            // A Face Refinement node's picture is the retouched one where it has faces and its input elsewhere (`run_film`'s
            // rule), and its `m` is its mask: a Layer reads what it would read from a corrector at its identity.
            Some(NodeKind::FaceRefinement) => g.face_refinement(n).is_some_and(|f| !f.bypass),
            _ => false,
        },
        PictureSource::Input | PictureSource::Split(..) => false,
    };
    g.numbered()
        .into_iter()
        .filter_map(|(id, kind)| {
            let node = NodeId::Corrector(id);
            let (kind, ports): (JoinKind, Vec<Port>) = match kind {
                NodeKind::Parallel => (JoinKind::Parallel, mixer_ports(g, id)),
                NodeKind::Layer => (JoinKind::Layer, mixer_ports(g, id)),
                NodeKind::Splitter => (JoinKind::Splitter, vec![Port::Picture]),
                NodeKind::Combiner => {
                    (JoinKind::Combiner, vec![Port::Red, Port::Green, Port::Blue])
                }
                NodeKind::Corrector
                | NodeKind::Film
                | NodeKind::FaceRefinement
                | NodeKind::Input
                | NodeKind::Output => {
                    return None;
                }
            };
            let inputs = ports
                .into_iter()
                .enumerate()
                .map(|(i, port)| {
                    let source = picture_into(g, node, port);
                    JoinInput {
                        source,
                        graded: kind == JoinKind::Layer && i > 0 && source.is_some_and(masked),
                    }
                })
                .collect();
            Some(PlanJoin {
                id,
                pos: g.pos(node)?,
                kind,
                inputs,
            })
        })
        .collect()
}

/// A mixer's linked inputs: `picture.1` to `picture.N`, with no gap (the validator refuses one).
fn mixer_ports(g: &NodeGraph, id: u32) -> Vec<Port> {
    (1..=MIXER_INPUTS)
        .map(Port::PictureN)
        .take_while(|p| {
            g.links
                .iter()
                .any(|l| l.to.node == NodeId::Corrector(id) && l.to.port == *p)
        })
        .collect()
}

/// The plan of a photo's graph, ending at `stop`.
///
/// # Errors
///
/// [`RenderError::Color`] when the graph cannot be compiled (a validated graph can, unless `stop`
/// names a node it does not have, or a mask stop names a mixer, a Splitter or a Combiner, which have no mask).
pub fn node_plan(g: &NodeGraph, stop: Stop) -> Result<Plan, RenderError> {
    plan::compile_with(&plan_graph(g), &plan_joins(g), stop)
        .map_err(|e| RenderError::Color(e.to_string()))
}

/// The mixers of a photo for the chain: each Parallel's bypass and each Layer's bypass, mode and opacity, for the
/// mixers the plan runs.
fn join_params(g: &NodeGraph, plan: &Plan) -> std::collections::BTreeMap<u32, JoinParams> {
    plan.node_ids()
        .into_iter()
        .filter_map(|id| match g.mixer(id)? {
            MixerNode::Parallel(p) => Some((id, JoinParams::parallel(p.bypass))),
            MixerNode::Layer(l) => Some((
                id,
                JoinParams::layer(l.bypass, blend_mode_of(l.mode), l.opacity),
            )),
            MixerNode::Splitter(_) | MixerNode::Combiner(_) => None,
        })
        .collect()
}

/// A Layer's mode as the chain has it.
fn blend_mode_of(mode: BlendMode) -> ym_color::display::BlendMode {
    use ym_color::display::BlendMode as B;
    match mode {
        BlendMode::Normal => B::Normal,
        BlendMode::Add => B::Add,
        BlendMode::Subtract => B::Subtract,
        BlendMode::Multiply => B::Multiply,
        BlendMode::Lighten => B::Lighten,
        BlendMode::Darken => B::Darken,
        BlendMode::Difference => B::Difference,
    }
}

/// What [`node_set`] gives: the nodes, the raster rows of their masks, the Film steps and the mixers' parameters.
type NodeParts = (
    NodeSet,
    RasterSet,
    std::collections::BTreeMap<u32, FilmStep>,
    std::collections::BTreeMap<u32, JoinParams>,
);

/// The nodes of a photo for the chain: the plan to `stop` (the Output, or a node's mask for the mask view),
/// Main's trim and bypass, and every other corrector the plan runs, with its own grade (no Auto), warp, curves
/// and trim; and the raster rows of their masks, numbered across the nodes in the plan's order (a node's rows
/// in list order), which the chain's [`MaskKind::Raster`] rows index.
fn node_set(
    s: &DevelopSettings,
    stop: Stop,
    background: Option<&Arc<BlurMap>>,
    objects: Option<&Arc<ObjectPlanes>>,
) -> Result<NodeParts, RenderError> {
    let plan = node_plan(&s.nodes, stop)?;
    let mut rasters = RasterSet::default();
    let mut masks = std::collections::BTreeMap::new();
    for id in plan.node_ids() {
        if let Some(m) = s.nodes.masks_of(id) {
            masks.insert(id, mask_rows(m, &mut rasters, background, objects, id));
        }
    }
    let mut others = std::collections::BTreeMap::new();
    let mut films = std::collections::BTreeMap::new();
    for id in plan.node_ids() {
        if id == plan::MAIN_NODE {
            continue;
        }
        match s.nodes.kind_of(NodeId::Corrector(id)) {
            Some(NodeKind::Film) => {
                let Some(f) = s.nodes.film(id) else {
                    continue;
                };
                // A Film node: the identity for the corrector machinery, with its masks; the chain reads its
                // picture from the render's planes.
                others.insert(
                    id,
                    NodeParams {
                        masks: masks.remove(&id).unwrap_or_default(),
                        ..NodeParams::default()
                    },
                );
                films.insert(
                    id,
                    FilmStep {
                        strength: f.strength / 100.0,
                        bypass: f.bypass,
                    },
                );
            }
            Some(NodeKind::FaceRefinement) => {
                let Some(f) = s.nodes.face_refinement(id) else {
                    continue;
                };
                // A node with faces to retouch is a Film-like step: the render makes its picture before the colour pass
                // (`film_pass`, on the CPU) and the chain blends it by the node's mask. Otherwise (bypassed, every amount 0,
                // no faces found yet or none, no model) the node passes its picture through: the corrector machinery runs it
                // bypassed, so its picture is its input's, and its key output is still the mask of its Mask tab.
                let live = !f.is_inert()
                    && objects
                        .and_then(|o| o.faces())
                        .is_some_and(|e| !e.faces.is_empty());
                if live {
                    others.insert(
                        id,
                        NodeParams {
                            masks: masks.remove(&id).unwrap_or_default(),
                            ..NodeParams::default()
                        },
                    );
                    films.insert(
                        id,
                        FilmStep {
                            strength: 1.0,
                            bypass: false,
                        },
                    );
                } else {
                    others.insert(
                        id,
                        NodeParams {
                            bypass: true,
                            masks: masks.remove(&id).unwrap_or_default(),
                            ..NodeParams::default()
                        },
                    );
                }
            }
            Some(NodeKind::Corrector) => {
                let Some(c) = s.nodes.corrector(id) else {
                    continue;
                };
                others.insert(
                    id,
                    NodeParams {
                        bypass: c.bypass,
                        trim: trim_of(&c.trim),
                        grade: grade_params_of(&c.grade, None),
                        warp: ColorWarp::to_color(&c.grade.color_warp),
                        curves: curves_of(&c.grade.curve)?,
                        masks: masks.remove(&id).unwrap_or_default(),
                    },
                );
            }
            // The four kinds of `docs/25` have no node parameters: the chain joins their registers
            // (`join_params` gives it the mode and the opacity).
            Some(
                NodeKind::Parallel
                | NodeKind::Layer
                | NodeKind::Splitter
                | NodeKind::Combiner
                | NodeKind::Input
                | NodeKind::Output,
            )
            | None => {}
        }
    }
    // Main's rows: numbered with the rest when Main is in the plan, else (a plan that does not run it) its
    // own masks are not evaluated and need no planes.
    let main_masks = masks
        .remove(&plan::MAIN_NODE)
        .unwrap_or_else(|| node_masks(&s.main().masks));
    let joins = join_params(&s.nodes, &plan);
    Ok((
        NodeSet {
            plan,
            main: MainNode {
                bypass: s.main().bypass,
                trim: trim_of(&s.main().trim),
                masks: main_masks,
            },
            others,
        },
        rasters,
        films,
        joins,
    ))
}

/// What the grade group is made for.
#[derive(Clone)]
pub struct GradeRequest<'a> {
    /// The output the display values are encoded for.
    pub target: OutputTarget,
    /// Whether 8-bit outputs are dithered.
    pub dither: bool,
    /// Where looks come from.
    pub looks: &'a dyn LookProvider,
    /// The photo's blur map, for the plan's Background rows (`docs/23` section 4.6): made once per photo by the
    /// renderer (`Renderer::blur_map` is the same map) and `None` for a plan with no Background row (or
    /// a caller that has no photo to read, whose Background rows select nothing).
    pub background: Option<&'a Arc<BlurMap>>,
    /// The planes of the plan's Object rows (`docs/23` section 4.4), made by the renderer from the photo's features
    /// ([`Job::objects_for`](super::graph)): `None` for a plan with no Object row that asks for a mask, or a caller that
    /// has no photo to read, whose Object rows select nothing.
    pub objects: Option<&'a Arc<ObjectPlanes>>,
}

/// A prepared C7 to C13 chain and what to tell the user about it.
#[derive(Debug, Clone)]
pub struct GradePlan {
    /// The chain.
    pub chain: Arc<ColorChain>,
    /// What the plan was made from.
    pub key: GradeKey,
    /// Plain-words notes: a look that could not be used and was replaced.
    pub warnings: Vec<String>,
}

/// The look as the provider answers for it now.
struct Pieces {
    look: Option<ResolvedLook>,
    warnings: Vec<String>,
}

fn pieces(src: &PhotoSource, req: &GradeRequest<'_>) -> Pieces {
    let mut warnings = Vec::new();
    let look = match req.looks.look(&src.settings.output().look) {
        Ok(l) => Some(l),
        Err(e) => {
            warnings.push(e.to_string());
            None
        }
    };
    Pieces { look, warnings }
}

/// The text of everything the nodes put into the chain beyond Main's grade, warp and curves: the plan
/// the compiler made (which nodes run, in what order, in which registers and key slots), Main's trim,
/// bypass and mask rows and the parameters of every other corrector the plan runs, mask rows included
/// (the enabled rows as the chain gets them, so a row switched off leaves the key as a row never added).
/// A dead node is not in it, so a dead branch leaves the key as it was.
fn nodes_key(
    s: &DevelopSettings,
    stop: Stop,
    background: Option<&Arc<BlurMap>>,
    objects: Option<&Arc<ObjectPlanes>>,
) -> String {
    let plan = match node_plan(&s.nodes, stop) {
        Ok(plan) => plan,
        Err(e) => return format!("no-plan:{e}"),
    };
    let mut k = String::with_capacity(256);
    let _ = write!(
        k,
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|",
        plan.stop,
        plan.passes,
        plan.buffers,
        plan.result,
        s.main().bypass,
        trim_of(&s.main().trim),
        node_masks(&s.main().masks),
    );
    for id in plan.node_ids() {
        if id == plan::MAIN_NODE {
            continue;
        }
        if let Some(f) = s.nodes.film(id) {
            // A Film node, by everything that makes its picture (not its name or place).
            let _ = write!(
                k,
                "film{id}:{}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?};",
                f.bypass,
                f.film,
                f.print,
                f.halation,
                f.grain,
                f.strength.to_bits(),
                node_masks(&f.masks),
            );
        } else if let Some(f) = s.nodes.face_refinement(id) {
            // A Face Refinement node by everything that makes its picture (not its name or place): Bypass, the amounts (the
            // one set for every face, whether it is used, and the sets of the faces), the protection plane (a hash of the
            // strokes, never the strokes, and its Feather, Grow and Refine), the masks, and the faces it retouches (the
            // analysis' own key: a new picture or a new frame is a new key; none while the photo has none yet).
            let protect = strokes_key(
                f.protect
                    .strokes
                    .iter()
                    .map(|s| (s.size, s.hardness, s.erase, s.points.as_slice())),
            );
            let _ = write!(
                k,
                "face{id}:{}:{}:{:?}:{:?}:protect{protect:x}:{}:{}:{}:{:?}:faces{};",
                f.bypass,
                f.same_for_all,
                f.all,
                f.faces,
                f.protect.grow,
                f.protect.feather,
                f.protect.refine,
                node_masks(&f.masks),
                objects
                    .and_then(|o| o.faces())
                    .map_or_else(|| "none".to_owned(), |e| format!("{:x}", e.hash)),
            );
        } else if let Some(m) = s.nodes.mixer(id) {
            // A mixer, a Splitter or a Combiner: its kind, bypass and, for a Layer, mode and opacity. Its links are in
            // the plan above.
            match m {
                MixerNode::Parallel(p) => {
                    let _ = write!(k, "parallel{id}:{};", p.bypass);
                }
                MixerNode::Layer(l) => {
                    let _ = write!(
                        k,
                        "layer{id}:{}:{}:{};",
                        l.bypass,
                        l.mode.name(),
                        l.opacity.to_bits()
                    );
                }
                MixerNode::Splitter(_) => {
                    let _ = write!(k, "splitter{id};");
                }
                MixerNode::Combiner(_) => {
                    let _ = write!(k, "combiner{id};");
                }
            }
        } else if let Some(c) = s.nodes.corrector(id) {
            let _ = write!(
                k,
                "{id}:{}:{:?}:{:?}:{:?}:{:?}:{:?};",
                c.bypass,
                trim_of(&c.trim),
                grade_params_of(&c.grade, None),
                Warp::new(&c.grade.color_warp.to_color()),
                c.grade.curve,
                node_masks(&c.masks),
            );
        }
    }
    // The raster rows by what makes their planes (a hash of the strokes, never the strokes) and their sizes:
    // the numbers above only say that a node has a raster row.
    for id in plan.node_ids() {
        let Some(masks) = s.nodes.masks_of(id) else {
            continue;
        };
        for row in masks.items.iter().filter(|r| r.enabled) {
            match &row.shape {
                MaskShape::Brush(b) => {
                    let key = strokes_key(
                        b.strokes
                            .iter()
                            .map(|s| (s.size, s.hardness, s.erase, s.points.as_slice())),
                    );
                    let _ = write!(k, "raster:{key:x}:{}:{}:{};", b.grow, b.feather, b.refine);
                }
                // The map is named by the renderer's own key for it (a new map is a new key); a row without a map
                // (no renderer asked for one) selects nothing and says so.
                MaskShape::Background(b) => {
                    let _ = write!(
                        k,
                        "background:{}:{}:{}:{}:{}:{};",
                        background.map_or_else(|| "none".to_owned(), |m| format!("{:x}", m.key)),
                        b.level,
                        b.softness,
                        b.grow,
                        b.feather,
                        b.refine
                    );
                }
                // The map is named by its own key (the picture, the weights and the rules); a row without a map (the worker
                // has not made it, or it failed) selects nothing and says so.
                MaskShape::Depth(d) => {
                    let _ = write!(
                        k,
                        "depth{}:{}:{}:{}:{}:{};",
                        if objects.is_some_and(|p| p.depth_show_for(id).is_some()) {
                            "-show"
                        } else {
                            ""
                        },
                        objects
                            .and_then(|p| p.depth_map())
                            .map_or_else(|| "none".to_owned(), |m| format!("{:x}", m.key)),
                        d.far,
                        d.grow,
                        d.feather,
                        d.refine
                    );
                }
                // The plane is named by its own key (the features, the points and the box); a row whose features are not
                // there selects nothing and says so.
                MaskShape::Object(o) => {
                    let _ = write!(
                        k,
                        "object:{}:{}:{}:{};",
                        super::graph::plane_text(objects, o),
                        o.grow,
                        o.feather,
                        o.refine
                    );
                }
                // Likewise Vision's plane, named by its own key (the picture and the request); a row whose mask is not
                // made yet, found nothing or cannot be made says "none".
                MaskShape::Subject(f) | MaskShape::People(f) => {
                    if let Some(kind) = VisionKind::of(crate::settings::MaskKind::of(&row.shape)) {
                        let _ = write!(
                            k,
                            "vision:{}:{}:{}:{}:{};",
                            kind.label(),
                            objects
                                .and_then(|p| p.vision_plane(kind))
                                .map_or_else(|| "none".to_owned(), |p| format!("{:x}", p.key)),
                            f.grow,
                            f.feather,
                            f.refine
                        );
                    }
                }
                _ => {}
            }
        }
    }
    k
}

fn key_of(
    src: &PhotoSource,
    color: &ColorPlan,
    req: &GradeRequest<'_>,
    p: &Pieces,
    stop: Stop,
) -> GradeKey {
    let s = &src.settings;
    let mut key = String::with_capacity(768);
    let _ = write!(
        key,
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{:?}|",
        color.key,
        grade_params(s, &src.estimates),
        // The warp as the chain prepares it: pins that change nothing (locked pins with no pin that
        // acts, a pin whose target is its source and that changes neither exposure nor saturation)
        // leave the key, and the frame, as without them.
        Warp::new(&warp_params(s)),
        s.main_grade().curve,
        nodes_key(s, stop, req.background, req.objects),
        req.target,
        req.dither,
        s.output().look_intensity,
        s.output().look,
        // The Before view's parameters: the as-shot white balance and the body's matrix (never a
        // setting), so that a frame of the Before view is not served for another body or white.
        color.camera_default,
    );
    match &p.look {
        Some(l) => key.push_str(&l.fingerprint),
        None => key.push_str("neutral-fallback"),
    }
    GradeKey(key)
}

/// The key a grade plan for these inputs has, without building the chain.
#[must_use]
pub fn grade_key(src: &PhotoSource, color: &ColorPlan, req: &GradeRequest<'_>) -> GradeKey {
    grade_key_to(src, color, req, Stop::Output)
}

/// [`grade_key`] for the plan that ends at `stop`: the Output (the picture), or a node's mask (the mask view).
#[must_use]
pub fn grade_key_to(
    src: &PhotoSource,
    color: &ColorPlan,
    req: &GradeRequest<'_>,
    stop: Stop,
) -> GradeKey {
    key_of(src, color, req, &pieces(src, req), stop)
}

/// Builds the grade plan: the look is fetched from the provider (a missing look renders as Neutral
/// and says so in [`GradePlan::warnings`]).
///
/// # Errors
///
/// [`RenderError::Color`] when the chain cannot be prepared.
pub fn grade_plan(
    src: &PhotoSource,
    color: &ColorPlan,
    req: &GradeRequest<'_>,
) -> Result<GradePlan, RenderError> {
    grade_plan_to(src, color, req, Stop::Output)
}

/// [`grade_plan`] for the plan that ends at `stop`. With [`Stop::Mask`] the chain's picture is a node's mask
/// as grey code values (`ColorChain::shows_mask`): the mask view. The plan holds the node and what it needs
/// even when the Output does not (a dead node has a mask but no picture in the Output's plan).
///
/// # Errors
///
/// [`RenderError::Color`] when the chain cannot be prepared or `stop` names a node the graph does not have.
pub fn grade_plan_to(
    src: &PhotoSource,
    color: &ColorPlan,
    req: &GradeRequest<'_>,
    stop: Stop,
) -> Result<GradePlan, RenderError> {
    let s = &src.settings;
    let p = pieces(src, req);
    let key = key_of(src, color, req, &p, stop);
    let mut params = ChainParams::new(*color.fused.matrix(), color.tone);
    params.grade = grade_params(s, &src.estimates);
    params.warp = warp_params(s);
    params.curves = curve_set(s)?;
    let (nodes, rasters, films, joins) = node_set(s, stop, req.background, req.objects)?;
    params.nodes = nodes;
    params.rasters = rasters;
    params.films = films;
    params.joins = joins;
    params.output = req.target;
    params.dither = req.dither;
    params.before = BeforePoint::CameraDefault;
    params.camera_default = color.camera_default;
    if let Some(look) = p.look {
        params.look = Some(LookSetting::new(look.transform, s.output().look_intensity));
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
    s.output_mut().look = look.clone();
    s.output_mut().look_intensity = 100.0;
    s
}

#[cfg(test)]
mod schedule_tests {
    use super::*;
    use serde_json::json;
    use ym_color::mask_raster::Area;
    use ym_color::schedule::{Declared, Schedule, StepKind};

    /// Two Film nodes on parallel branches that join by a key link: Main -> Film 2 -> corrector 4, Input -> Film 3,
    /// and Film 3's key into corrector 4.
    fn two_branches() -> DevelopSettings {
        let doc = json!({ "nodes": { "items": [
            { "id": 1 },
            { "id": 2, "kind": "film", "pos": [2, 0.0], "film": { "exposure_ev": 0.3 } },
            { "id": 3, "kind": "film", "pos": [2, 1.0], "film": { "exposure_ev": -0.3 } },
            { "id": 4, "kind": "corrector", "pos": [3, 0.0], "trim": { "exposure_ev": 0.4 } } ],
          "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 3, "port": "picture" } },
            { "from": { "node": 2, "port": "picture" }, "to": { "node": 4, "port": "picture" } },
            { "from": { "node": 3, "port": "key" }, "to": { "node": 4, "port": "key" } },
            { "from": { "node": 4, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
        serde_json::from_value::<DevelopSettings>(doc)
            .expect("a document")
            .validated()
            .expect("a valid graph")
            .0
    }

    fn chain_of(s: &DevelopSettings) -> ColorChain {
        let (nodes, rasters, films, joins) = node_set(s, Stop::Output, None, None).expect("nodes");
        let mut params = ChainParams::new(ym_color::consts::linalg::IDENTITY, ToneParams::IDENTITY);
        params.nodes = nodes;
        params.rasters = rasters;
        params.films = films;
        params.joins = joins;
        ColorChain::new(&params).expect("a chain")
    }

    /// G-11 part 3, item 2: each Film node reads only the Film pictures that are its ancestors in the plan's graph.
    #[test]
    fn a_film_node_reads_only_the_pictures_of_its_ancestors() {
        let chain = chain_of(&two_branches());
        // Two parallel branches: neither reads the other's picture.
        let reads = chain.film_reads();
        assert_eq!(reads.get(&2), Some(&vec![]), "{reads:?}");
        assert_eq!(reads.get(&3), Some(&vec![]), "{reads:?}");
        // Areas for a 40 x 40 need in the middle of a 400 x 400 photo, with halos of 20 px and 60 px.
        let frame = Area::whole(400, 400);
        let need = Area {
            x: 180,
            y: 180,
            w: 40,
            h: 40,
        };
        let reach = |id: u32| Some(if id == 2 { 20 } else { 60 });
        let branches = chain.schedule(1.0, &reach).infer(need, &frame);
        let line = Schedule::new(
            (0..2).map(|i| -> Declared { StepKind::Film.declared(if i == 0 { 20 } else { 60 }) }),
        )
        .infer(need, &frame);
        println!(
            "two branches, before (a line): made {:?}, read {:?}",
            line.made, line.read
        );
        println!(
            "two branches, after (ancestors): made {:?}, read {:?}",
            branches.made, branches.read
        );
        // Before: node 2 was made over what node 3 reads (60 px around the need). After: over the need alone.
        assert_eq!(line.made[0], need.grown(60, &frame));
        assert_eq!(branches.made[0], need);
        assert_eq!(branches.read[0], need.grown(20, &frame));
        assert_eq!(branches.made[1], need);
        assert_eq!(branches.read[1], need.grown(60, &frame));
        // The camera picture is read over the longer branch (60), not the sum (80).
        assert_eq!(branches.source, need.grown(60, &frame));
        assert_eq!(line.source, need.grown(80, &frame));
        assert_eq!(chain.schedule(1.0, &reach).total_reach(), 60);
    }

    /// A line of Film nodes keeps the sum of its halos: node 3 after node 2 reads node 2's picture.
    #[test]
    fn a_line_of_film_nodes_reads_every_earlier_picture() {
        let doc = json!({ "nodes": { "items": [
            { "id": 1 },
            { "id": 2, "kind": "film", "pos": [2, 0.0], "film": { "exposure_ev": 0.3 } },
            { "id": 3, "kind": "film", "pos": [3, 0.0], "film": { "exposure_ev": -0.3 } } ],
          "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
            { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
            { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
        let s = serde_json::from_value::<DevelopSettings>(doc)
            .expect("a document")
            .validated()
            .expect("a valid graph")
            .0;
        let chain = chain_of(&s);
        assert_eq!(chain.film_reads().get(&3), Some(&vec![2]));
        assert_eq!(chain.film_reads().get(&2), Some(&vec![]));
        assert_eq!(chain.schedule(1.0, &|_| Some(25)).total_reach(), 50);
    }
}
