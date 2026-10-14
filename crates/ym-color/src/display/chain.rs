//! The prepared per-photo chain: C7 to C13 behind the fused C1 to C6 matrix (`docs/02-colour-pipeline.md`
//! section Stage order).
//!
//! [`ChainParams`] collects what the chain needs; [`ColorChain::new`] prepares it once (bakes the four
//! curves, multiplies out matrices, fetches the shared ACES 2 tables and classifies the look) and the
//! result is immutable, `Send` and `Sync`, so any number of threads can call it. To change one
//! setting, clone [`ColorChain::params`], edit it and build a new chain (the LUTs are shared, not
//! copied).
//!
//! # Look placement
//!
//! * A **log-to-log look** (LogC4 or LogC3 EI 800 input) runs at C10, between the grade and the
//!   curve: `C9 -> look -> blend by intensity -> C11`. Neutral, an identity CDL without a LUT, and
//!   an intensity of 0 skip the stage and are exactly the identity.
//! * A look that **includes display rendering** (LogC3 to Rec.709) replaces C12. The curve runs
//!   first, then both our C12 and the look's LUT are evaluated from the same C11 output and blended
//!   in display-linear by the intensity. At 0 % the result is our C12 exactly; at 100 % it is the
//!   look exactly (after the primaries matrix to the output).
//!
//! # The Before view
//!
//! `docs/11` says the Before mode "shows the LogC4 signal before look and display transform" and
//! `docs/16` records the choice as "LogC4 'camera log' (as mocked)". This implementation reads that
//! as the LogC4 signal the camera processing produced: the output of **C8**, before the primary grade
//! (the Light and Colour controls), the look, the curve and the display transform
//! ([`BeforePoint::CameraLog`], the default). [`BeforePoint::AfterGrade`] stops after C9 instead, for
//! a reading in which only the look, curve and display are "after". The log code values are shown as
//! if they were the output's display code values: each channel clamped to `[0, 1]`, no transfer
//! function, no primaries conversion, which is how a log signal looks on a monitor without a view
//! transform (flat and desaturated). Exposure and white balance live in the fused matrix and cannot
//! be undone here; the caller builds the Before chain from the same matrix (or from the Auto
//! matrix, if "before" should also drop manual exposure and white balance).

use std::fmt;
use std::sync::Arc;

use super::dng_inverse::DngDisplayInverse;
use super::transform::{DisplayTransform, PreparedDisplay};
use super::{DisplayError, OutputTarget};
use crate::consts::Vec3;
use crate::consts::linalg::{self, Mat3};
use crate::curve::{BakedCurves, CurveSet};
use crate::encode::{OutputSpace, dither_8bit, quantize};
use crate::exposure::white::WhiteLift;
use crate::gamut::LinearAwg4;
use crate::grade::{Grade, GradeParams, grade_luma};
use crate::logc::LogC4;
use crate::look::{LookTransform, blend_display_linear, intensity_fraction};
use crate::tone::ToneParams;

/// Largest magnitude a camera sample may have; NaN counts as 0 (see the module docs of [`super`]).
const INPUT_LIMIT: f64 = 1e30;

/// A look and how strongly to apply it (C10).
///
/// `Debug` names the look's input encoding and whether it has a CDL and a LUT; it does not print
/// the LUT.
#[derive(Clone)]
pub struct LookSetting {
    /// The look. Shared: a 65-point cube is several megabytes.
    pub look: Arc<LookTransform>,
    /// Intensity in percent, 0 to 100 (out-of-range values clamp, NaN counts as 0).
    pub intensity_pct: f64,
}

impl fmt::Debug for LookSetting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LookSetting")
            .field("input", &self.look.input())
            .field("has_cdl", &self.look.cdl().is_some())
            .field("lut_size", &self.look.lut().map(|l| l.cube.size()))
            .field("intensity_pct", &self.intensity_pct)
            .finish()
    }
}

impl LookSetting {
    /// A look at `intensity_pct` percent.
    #[must_use]
    pub fn new(look: Arc<LookTransform>, intensity_pct: f64) -> Self {
        Self {
            look,
            intensity_pct,
        }
    }
}

/// Where the Before view reads the LogC4 signal (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BeforePoint {
    /// The output of C8: LogC4 of the camera processing, before the grade, look, curve and display.
    #[default]
    CameraLog,
    /// The output of C9: LogC4 with the primary grade applied, before the look, curve and display.
    AfterGrade,
}

/// Which picture a buffer render produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum View {
    /// The full chain through C13.
    #[default]
    Final,
    /// The Before view ([`ColorChain::apply_before_pixel`]).
    Before,
}

/// Everything [`ColorChain::new`] needs.
#[derive(Debug, Clone)]
pub struct ChainParams {
    /// The fused C1 to C6 matrix: camera RGB (balanced with the initial gains) to linear AWG4,
    /// exposure included (`ym_color::chain`).
    pub matrix: Mat3,
    /// Stage C7: roll-off and toe, with the clip level and noise floor.
    pub tone: ToneParams,
    /// Stage C9: the primary grade. All zero is the identity.
    pub grade: GradeParams,
    /// Stage C10: the look and its intensity. `None` is Neutral.
    pub look: Option<LookSetting>,
    /// Stage C11: the four curves (baked into 1024-entry tables by [`ColorChain::new`]).
    pub curves: CurveSet,
    /// Stage C12: ACES 2 or an imported ARRI display LUT.
    pub display: DisplayTransform,
    /// Stage C12 step 1b: the white lift, a scene-linear gain on the tones above +1 stop over middle
    /// grey, before the display placement and ACES 2 (`docs/02`). Only the ACES 2 path has it: an
    /// imported ARRI display LUT keeps ARRI's highlights and ignores the lift. [`WhiteLift::NONE`] is
    /// the identity.
    pub white_lift: WhiteLift,
    /// Stage C13: which output the display-linear values are encoded for.
    pub output: OutputTarget,
    /// Whether 8-bit outputs are dithered (`docs/02`, C13). 10 and 16-bit outputs never are.
    pub dither: bool,
    /// Where the Before view reads the signal.
    pub before: BeforePoint,
}

impl ChainParams {
    /// Defaults for everything after C7: zero grade, Neutral look, identity curves, ACES 2, the
    /// viewport output, dither on, Before view at the camera log.
    #[must_use]
    pub fn new(matrix: Mat3, tone: ToneParams) -> Self {
        Self {
            matrix,
            tone,
            grade: GradeParams::default(),
            look: None,
            curves: CurveSet::default(),
            display: DisplayTransform::Aces2,
            white_lift: WhiteLift::NONE,
            output: OutputTarget::Viewport,
            dither: true,
            before: BeforePoint::CameraLog,
        }
    }
}

/// The value of every stage for one pixel, for tests and for debugging GPU parity stage by stage.
///
/// With a look that includes display rendering, `c10` equals `c9` (the look has not run yet) and the
/// look is inside `c12`, which is then the blend of our C12 with the look's LUT output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageTrace {
    /// The fused matrix applied: linear AWG4 (C1 to C6).
    pub c6_linear_awg4: Vec3,
    /// After the roll-off and toe (C7): linear AWG4.
    pub c7_linear_awg4: Vec3,
    /// LogC4 encode (C8).
    pub c8_logc4: LogC4,
    /// After the primary grade (C9).
    pub c9_logc4: LogC4,
    /// After the look (C10), log-to-log looks only.
    pub c10_logc4: LogC4,
    /// After the curve (C11): the DNG stop point.
    pub c11_logc4: LogC4,
    /// After the display transform (C12): display-linear in the output's primaries.
    pub c12_display_linear: Vec3,
    /// After the display encode (C13), clamped to `[0, 1]`, not quantised.
    pub c13_encoded: Vec3,
}

#[derive(Debug, Clone)]
enum LookStage {
    /// Neutral, an identity look, or an intensity of 0: nothing runs.
    None,
    /// A log-to-log look at C10.
    LogToLog {
        look: Arc<LookTransform>,
        intensity_pct: f64,
    },
    /// A look with display rendering inside C12.
    Baked {
        look: Arc<LookTransform>,
        intensity_pct: f64,
    },
}

/// Values of C1 to C11 for one pixel.
#[derive(Clone, Copy)]
struct UpToC11 {
    c6: Vec3,
    c7: Vec3,
    c8: LogC4,
    c9: LogC4,
    c10: LogC4,
    c11: LogC4,
}

/// The prepared, immutable chain for one photo (stages C7 to C13 behind the fused matrix).
///
/// Build it once per photo and settings change, then call it from as many threads as you like.
///
/// `Debug` prints the [`ChainParams`] it was built from (with the look and display LUT
/// summarised), not the baked tables.
#[derive(Clone)]
pub struct ColorChain {
    params: ChainParams,
    matrix: Mat3,
    tone: ToneParams,
    grade: Grade,
    look: LookStage,
    curves: BakedCurves,
    display: PreparedDisplay,
    output: OutputTarget,
    space: OutputSpace,
    dither: bool,
    before: BeforePoint,
    /// Set when C12 is not our ACES 2, so the DNG holds the scene values ACES 2 turns into this
    /// chain's picture (see [`ColorChain::dng_inverts_display`]).
    dng_inverse: Option<DngDisplayInverse>,
}

impl fmt::Debug for ColorChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColorChain")
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

fn sanitize(c: f64) -> f64 {
    if c.is_nan() {
        0.0
    } else {
        c.clamp(-INPUT_LIMIT, INPUT_LIMIT)
    }
}

fn unit(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

impl ColorChain {
    /// Prepares the chain.
    ///
    /// Bakes the four curves into their 1024-entry tables, multiplies out the display transform's
    /// matrices, fetches the shared ACES 2 tables for the output (10 to 35 ms the first time in a
    /// process) and classifies the look.
    ///
    /// # Errors
    ///
    /// [`DisplayError::NonFiniteMatrix`] when the fused matrix has a NaN or infinite entry.
    pub fn new(params: &ChainParams) -> Result<Self, DisplayError> {
        if params.matrix.iter().flatten().any(|v| !v.is_finite()) {
            return Err(DisplayError::NonFiniteMatrix);
        }
        let look = match &params.look {
            None => LookStage::None,
            Some(s) if s.look.includes_display_rendering() => LookStage::Baked {
                look: Arc::clone(&s.look),
                intensity_pct: s.intensity_pct,
            },
            Some(s) if s.look.is_identity() || intensity_fraction(s.intensity_pct) <= 0.0 => {
                LookStage::None
            }
            Some(s) => LookStage::LogToLog {
                look: Arc::clone(&s.look),
                intensity_pct: s.intensity_pct,
            },
        };
        // C12 is not our ACES 2 for an imported ARRI display LUT, and for a look that includes its
        // own display rendering at any intensity above 0 (at 0 the chain is our C12 exactly).
        let own_rendering = matches!(params.display, DisplayTransform::ArriDisplayLut(_))
            || matches!(&look, LookStage::Baked { intensity_pct, .. } if intensity_fraction(*intensity_pct) > 0.0);
        Ok(Self {
            params: params.clone(),
            matrix: params.matrix,
            tone: params.tone,
            grade: Grade::new(&params.grade),
            look,
            curves: params.curves.bake(),
            display: PreparedDisplay::new(&params.display, params.output, params.white_lift),
            output: params.output,
            space: params.output.space(),
            dither: params.dither,
            before: params.before,
            dng_inverse: own_rendering.then(|| DngDisplayInverse::new(params.output)),
        })
    }

    /// The settings this chain was built from (cheap to clone: LUTs are shared).
    #[must_use]
    pub fn params(&self) -> &ChainParams {
        &self.params
    }

    /// The output this chain encodes for.
    #[must_use]
    pub fn output(&self) -> OutputTarget {
        self.output
    }

    /// The C13 colour space of the output.
    #[must_use]
    pub fn space(&self) -> OutputSpace {
        self.space
    }

    /// Whether 8-bit outputs are dithered.
    #[must_use]
    pub fn dither(&self) -> bool {
        self.dither
    }

    /// Where the Before view reads the signal.
    #[must_use]
    pub fn before_point(&self) -> BeforePoint {
        self.before
    }

    /// The white lift this chain applies in front of ACES 2: the lift it was built with when C12 is
    /// our ACES 2, [`WhiteLift::NONE`] for an imported ARRI display LUT (which ignores it).
    #[must_use]
    pub fn white_lift(&self) -> WhiteLift {
        self.display.white_lift()
    }

    /// Whether the look includes its own display rendering (LogC3 to Rec.709), which replaces C12.
    #[must_use]
    pub fn look_includes_display_rendering(&self) -> bool {
        matches!(self.look, LookStage::Baked { .. })
    }

    /// Whether a DNG of this chain holds the picture through the inverse of our display transform
    /// instead of the C11 output itself: true when C12 is not our ACES 2 (an imported ARRI display LUT,
    /// or a look that includes its own display rendering at an intensity above 0). The DNG's fitted
    /// profile renders ACES 2, so the pixels are the scene values that ACES 2 turns into the picture
    /// this chain makes, and Adobe's renderer shows approximately that picture
    /// ([`ColorChain::apply_dng_pixel`], `docs/09`).
    #[must_use]
    pub fn dng_inverts_display(&self) -> bool {
        self.dng_inverse.is_some()
    }

    // ---- stages --------------------------------------------------------------------------------

    /// C1 to C6: the fused matrix on a sanitised camera triple.
    fn c1_to_c6(&self, camera_rgb: Vec3) -> Vec3 {
        let rgb = camera_rgb.map(sanitize);
        linalg::mul_vec(&self.matrix, rgb)
    }

    /// C10 for log-to-log looks.
    fn c10(&self, v: LogC4) -> LogC4 {
        match &self.look {
            // `apply_logc4` fails only for a look with display rendering, which `new` routes to
            // `Baked`; keeping the input is the right fallback if that ever changed.
            LookStage::LogToLog {
                look,
                intensity_pct,
            } => look.apply_logc4(v, *intensity_pct).unwrap_or(v),
            LookStage::None | LookStage::Baked { .. } => v,
        }
    }

    /// C12, including the display-linear blend with a look that includes display rendering.
    fn c12(&self, v: LogC4) -> Vec3 {
        let LookStage::Baked {
            look,
            intensity_pct,
        } = &self.look
        else {
            return self.display.display_linear(v);
        };
        if intensity_fraction(*intensity_pct) <= 0.0 {
            return self.display.display_linear(v);
        }
        // `eval_display_rec709` fails only for a log-to-log look, which `new` never puts here.
        let Ok(rec709) = look.eval_display_rec709(v) else {
            return self.display.display_linear(v);
        };
        let theirs = self.space.from_linear_rec709(rec709);
        if intensity_fraction(*intensity_pct) >= 1.0 {
            return theirs;
        }
        blend_display_linear(self.display.display_linear(v), theirs, *intensity_pct)
    }

    fn up_to_c11(&self, camera_rgb: Vec3) -> UpToC11 {
        let c6 = self.c1_to_c6(camera_rgb);
        let c7 = self.tone.apply_rgb(c6);
        let linear = LinearAwg4(c7);
        let c8 = linear.to_logc4();
        let c9 = self.grade.apply(c8, grade_luma(linear));
        let c10 = self.c10(c9);
        let c11 = self.curves.apply(c10);
        UpToC11 {
            c6,
            c7,
            c8,
            c9,
            c10,
            c11,
        }
    }

    // ---- single pixels -------------------------------------------------------------------------

    /// Every stage's value for one camera pixel.
    #[must_use]
    pub fn trace(&self, camera_rgb: Vec3) -> StageTrace {
        let s = self.up_to_c11(camera_rgb);
        let c12 = self.c12(s.c11);
        StageTrace {
            c6_linear_awg4: s.c6,
            c7_linear_awg4: s.c7,
            c8_logc4: s.c8,
            c9_logc4: s.c9,
            c10_logc4: s.c10,
            c11_logc4: s.c11,
            c12_display_linear: c12,
            c13_encoded: self.encode(c12),
        }
    }

    /// The whole chain: camera RGB to display code values in `[0, 1]` (not quantised, not dithered).
    #[must_use]
    pub fn apply_pixel(&self, camera_rgb: Vec3) -> Vec3 {
        self.encode(self.apply_to_display_linear(camera_rgb))
    }

    /// C12 alone: a LogC4 pixel (the C11 output) to display-linear RGB in the output's primaries,
    /// including the blend with a look that includes display rendering. The entry point for
    /// testing a display kernel without the stages before it.
    #[must_use]
    pub fn display_linear_from_logc4(&self, c11: LogC4) -> Vec3 {
        self.c12(c11)
    }

    /// C1 to C12: camera RGB to display-linear RGB in the output's primaries (1.0 is 100 nit white).
    ///
    /// This is the picture an exporter resizes in linear light before [`ColorChain::encode`] or
    /// [`ColorChain::encode_u8`] (`docs/09`, Pixel pipeline per format).
    #[must_use]
    pub fn apply_to_display_linear(&self, camera_rgb: Vec3) -> Vec3 {
        self.c12(self.up_to_c11(camera_rgb).c11)
    }

    /// C1 to C11: camera RGB to LogC4 with grade, look and curve applied (the DNG stop point before
    /// the decode). With a look that includes display rendering the look has not run at this
    /// point (the curve is the last LogC4 stage).
    #[must_use]
    pub fn apply_logc4_pixel(&self, camera_rgb: Vec3) -> LogC4 {
        self.up_to_c11(camera_rgb).c11
    }

    /// The DNG path (`docs/09`, Pixel pipeline per format): C1 to C11, then LogC4 decoded back to
    /// linear AWG4. Negative values are not clipped here (the writer clips and counts them).
    ///
    /// When C12 is not our ACES 2 ([`ColorChain::dng_inverts_display`]) the result is instead the
    /// scene-linear AWG4 triple that ACES 2 with the display placement turns into this chain's
    /// display-linear picture (look, intensity blend and all), limited to the range of the DNG
    /// profile ([`DNG_SCENE_LIMIT`](super::DNG_SCENE_LIMIT)): with the placement in `BaselineExposure`, Adobe's renderer shows
    /// approximately that picture. Build the chain for [`OutputTarget::Srgb`], the display the DNG
    /// profile was fitted to; for another output the display value is first converted to Rec.709
    /// primaries.
    #[must_use]
    pub fn apply_dng_pixel(&self, camera_rgb: Vec3) -> Vec3 {
        let c11 = self.up_to_c11(camera_rgb).c11;
        match &self.dng_inverse {
            // C12 is our ACES 2: the pixels are the scene values it takes in, with the white lift
            // (a scene-linear gain, in front of the placement that `BaselineExposure` carries).
            None => self.display.white_lift().apply(c11.to_linear().0),
            Some(inverse) => inverse.scene_awg4(self.c12(c11)),
        }
    }

    /// The Before view for one pixel: the LogC4 signal at the chain's [`BeforePoint`], each channel
    /// clamped to `[0, 1]` and used directly as a display code value (see the module docs).
    #[must_use]
    pub fn apply_before_pixel(&self, camera_rgb: Vec3) -> Vec3 {
        let s = self.up_to_c11(camera_rgb);
        let v = match self.before {
            BeforePoint::CameraLog => s.c8,
            BeforePoint::AfterGrade => s.c9,
        };
        v.0.map(unit)
    }

    /// One pixel of the chosen view as unquantised code values: [`ColorChain::apply_pixel`] for
    /// [`View::Final`], [`ColorChain::apply_before_pixel`] for [`View::Before`].
    #[must_use]
    pub fn apply_view_pixel(&self, view: View, camera_rgb: Vec3) -> Vec3 {
        match view {
            View::Final => self.apply_pixel(camera_rgb),
            View::Before => self.apply_before_pixel(camera_rgb),
        }
    }

    // ---- C13 -----------------------------------------------------------------------------------

    /// C13: encodes display-linear RGB (in the output's primaries) with the output's transfer
    /// function and clamps to `[0, 1]`. Not quantised.
    #[must_use]
    pub fn encode(&self, display_linear: Vec3) -> Vec3 {
        self.space.encode_clamped(display_linear)
    }

    /// Quantises encoded code values to 8 bits: blue-noise dither at absolute output pixel `(x, y)`
    /// when the chain dithers, plain rounding otherwise.
    #[must_use]
    pub fn quantize_u8(&self, encoded: Vec3, x: usize, y: usize) -> [u8; 3] {
        if self.dither {
            [
                dither_8bit(encoded[0], x, y),
                dither_8bit(encoded[1], x, y),
                dither_8bit(encoded[2], x, y),
            ]
        } else {
            encoded.map(|c| quantize(c, 8) as u8)
        }
    }

    /// C13 for an 8-bit output: encode, then [`ColorChain::quantize_u8`].
    #[must_use]
    pub fn encode_u8(&self, display_linear: Vec3, x: usize, y: usize) -> [u8; 3] {
        self.quantize_u8(self.encode(display_linear), x, y)
    }

    /// C13 for a 10 to 16-bit output (`bits` is clamped to 1 to 16): encode, then round. Never dithered.
    #[must_use]
    pub fn encode_u16(&self, display_linear: Vec3, bits: u32) -> [u16; 3] {
        self.encode(display_linear).map(|c| quantize(c, bits))
    }

    /// Quantises encoded code values to `bits` bits without dither.
    #[must_use]
    pub fn quantize_u16(&self, encoded: Vec3, bits: u32) -> [u16; 3] {
        encoded.map(|c| quantize(c, bits))
    }
}
