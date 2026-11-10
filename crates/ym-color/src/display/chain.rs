//! The prepared per-photo chain: C7 to C13 behind the fused C1 to C6 matrix (`docs/02-colour-pipeline.md`
//! section Stage order).
//!
//! [`ChainParams`] collects what the chain needs; [`ColorChain::new`] prepares it once (bakes the four
//! curves, multiplies out matrices, sets up the cine rendering and classifies the look) and the
//! result is immutable, `Send` and `Sync`, so any number of threads can call it. To change one
//! setting, clone [`ColorChain::params`], edit it and build a new chain (the LUTs are shared, not
//! copied).
//!
//! # Nodes and the look
//!
//! The chain runs the nodes of its plan ([`ChainParams::nodes`], `crate::plan`) between C8 and the look:
//! each corrector does its trims, C9, C9b and C11 (`display::node`), the default graph being Main alone.
//! The look belongs to the Output and acts after the last node (pipeline 0.9.0; until 0.8.2 the curve
//! followed the look).
//!
//! * A **log-to-log look** (LogC4 or LogC3 EI 800 input) runs at C10, after the nodes:
//!   `nodes -> look -> blend by intensity`. Neutral, an identity CDL without a LUT, and an intensity
//!   of 0 skip the stage and are exactly the identity.
//! * A look that **includes display rendering** (LogC3 to Rec.709) replaces C12. The nodes ran first,
//!   including every curve; then both our C12 and the look's LUT are evaluated from the same output
//!   and blended in display-linear by the intensity. At 0 % the result is our C12 exactly; at 100 % it
//!   is the look exactly (after the primaries matrix to the output).
//!
//! # The Before view
//!
//! The Before view shows the photo as the camera would render it by default
//! ([`BeforePoint::CameraDefault`], the default; `docs/02`, The Before view, and `docs/16`): the camera
//! RGB that enters the chain goes through one per-photo 3x3 ([`CameraDefault`]: the camera's as-shot
//! white balance, the body's D65 colour matrix to linear Rec.709 and the camera gain), then the camera
//! tone curve per channel, then the output primaries and C13 as for the final picture. It does not run
//! C7 to C12 and no setting reaches it: exposure, white balance, look, grade, curves and display are
//! all ignored. The raw stages in front of the chain (demosaic, denoise, lens corrections, the decode
//! switches) are the same as for the final picture, so they change it as they change the raw.
//!
//! Two older reading points of the LogC4 signal remain for tests and diagnostics (the app does not use
//! them): [`BeforePoint::CameraLog`], the output of C8, and [`BeforePoint::AfterGrade`], the output of
//! C9. Their log code values are shown as if they were display code values: each channel clamped to
//! `[0, 1]`, no transfer function, no primaries conversion.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use super::dng_inverse::DngDisplayInverse;
use super::mix::{self, JoinParams, Source};
use super::node::{FilmStep, MainStages, NodeParams, NodeSet, PreparedNode};
use super::transform::PreparedDisplay;
use super::trim::Trim;
use super::{DisplayError, OutputTarget};
use crate::camera_default::CameraDefault;
use crate::consts::Vec3;
use crate::consts::linalg::{self, Mat3};
use crate::curve::CurveSet;
use crate::encode::{OutputSpace, dither_8bit, quantize};
use crate::gamut::{LinearAwg4, LinearRec709};
use crate::grade::GradeParams;
use crate::logc::LogC4;
use crate::look::{LookTransform, blend_display_linear, intensity_fraction};
use crate::mask::{PixelAt, PreparedMasks};
use crate::mask_raster::RasterSet;
use crate::plan::{
    JoinKind, JoinStep, KeyRef, PLAN_INPUTS, PLAN_KEYS, PLAN_REGISTERS, PictureOut, PictureRef,
    PlanResult, Step,
};
use crate::tone::{PreparedTone, ToneParams};
use crate::warp::WarpParams;

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

/// What the Before view shows (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BeforePoint {
    /// The photo as the camera would render it by default: [`ChainParams::camera_default`] on the
    /// camera RGB, then the output primaries and C13. No colour stage of the chain runs.
    #[default]
    CameraDefault,
    /// The output of C8: LogC4 of the camera processing, before the grade, look, curve and display.
    /// Diagnostic; the app does not use it.
    CameraLog,
    /// The output of C9: LogC4 with the primary grade applied, before the look, curve and display.
    /// Diagnostic; the app does not use it.
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
    /// Stage C7: the highlight range and the low-light toe, with the clip level and noise floor.
    /// [`ToneParams::IDENTITY`] turns the stage off.
    pub tone: ToneParams,
    /// Stage C9 of Main, the corrector every photo has: the primary grade. All zero is the identity.
    pub grade: GradeParams,
    /// Stage C9b of Main: the colour warp's pins. No pins is the identity.
    pub warp: WarpParams,
    /// Stage C10, in the Output after the last node: the look and its intensity. `None` is Neutral.
    pub look: Option<LookSetting>,
    /// Stage C11 of Main: the four curves (baked into 1024-entry tables by [`ColorChain::new`]).
    pub curves: CurveSet,
    /// The nodes: the compiled plan, Main's trim and bypass and the other correctors. The default is
    /// the plan of the default graph, Main alone with no trim (`grade`, `warp` and `curves` above are
    /// then everything the nodes do).
    pub nodes: NodeSet,
    /// Stage C13: which output the display-linear values are encoded for.
    pub output: OutputTarget,
    /// Whether 8-bit outputs are dithered (`docs/02`, C13). 10 and 16-bit outputs never are.
    pub dither: bool,
    /// What the Before view shows.
    pub before: BeforePoint,
    /// The Before view's per-photo camera default ([`BeforePoint::CameraDefault`]): the as-shot white
    /// balance, the body's colour matrix and the camera gain folded into one 3x3, and the camera tone
    /// curve. [`CameraDefault::neutral`] when the caller does not know the body.
    pub camera_default: CameraDefault,
    /// The raster rows of the plan's nodes in plane order (`MaskKind::Raster { plane }` is a place in it):
    /// where each row's `S0` comes from and the sizes of its steps. Empty when no node has one.
    pub rasters: RasterSet,
    /// The Film nodes of the plan (`docs/24`), by id. Each has its [`NodeParams`] in
    /// [`NodeSet::others`] (the identity, with its masks); the chain reads the picture the node made from
    /// the render's planes ([`RasterPlanes::film`](crate::mask::RasterPlanes::film)). Empty for a photo without a
    /// Film node, which then runs exactly as it did.
    pub films: BTreeMap<u32, FilmStep>,
    /// The mixers of the plan (`docs/25`), by id: a Parallel's bypass, a Layer's bypass, mode and opacity. A Splitter or
    /// Combiner needs no entry, and a mixer with none is a Parallel or Layer at its defaults (not bypassed, Normal,
    /// opacity 0 - set it). Empty for a photo without a mixer, which then runs exactly as it did.
    pub joins: BTreeMap<u32, JoinParams>,
}

impl ChainParams {
    /// The parameters of the corrector `id` as the plan runs it: Main's are the chain's own `grade`,
    /// `warp` and `curves` with the trim and bypass of [`NodeSet::main`], every other corrector's are in
    /// [`NodeSet::others`]. `None` for a corrector the chain has no parameters for. The CPU chain and the
    /// device kernel read their nodes through this, so they cannot disagree about which fields are Main's.
    #[must_use]
    pub fn node_params(&self, id: u32) -> Option<NodeParams> {
        if id == crate::plan::MAIN_NODE {
            Some(NodeParams {
                bypass: self.nodes.main.bypass,
                trim: self.nodes.main.trim,
                grade: self.grade,
                warp: self.warp.clone(),
                curves: self.curves.clone(),
                masks: self.nodes.main.masks.clone(),
            })
        } else {
            self.nodes.others.get(&id).cloned()
        }
    }

    /// Defaults for everything after C7: zero grade, Neutral look, identity curves, the cine rendering,
    /// the viewport output, dither on, the Before view showing the camera default (with the neutral
    /// placeholder for the body: set [`ChainParams::camera_default`]).
    #[must_use]
    pub fn new(matrix: Mat3, tone: ToneParams) -> Self {
        Self {
            matrix,
            tone,
            grade: GradeParams::default(),
            warp: WarpParams::default(),
            look: None,
            curves: CurveSet::default(),
            nodes: NodeSet::default(),
            output: OutputTarget::Viewport,
            dither: true,
            before: BeforePoint::CameraDefault,
            camera_default: CameraDefault::neutral(),
            rasters: RasterSet::default(),
            films: BTreeMap::new(),
            joins: BTreeMap::new(),
        }
    }
}

/// The value of every stage for one pixel, for tests and for debugging GPU parity stage by stage.
///
/// C9, C9b and C11 are those of **Main** (its trim included in C9): the values the node loop has after
/// each of Main's steps, equal to the value before when the step is skipped. C10 is the look, which
/// acts on the output of the last node, so `c10_logc4` is the last LogC4 value of the chain (the
/// input of C12 and the DNG stop) and equals `c11_logc4` for the default graph with no look. With a
/// look that includes display rendering, `c10` equals the output of the nodes (the look has not run
/// yet) and the look is inside `c12`, which is then the blend of our C12 with the look's LUT output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageTrace {
    /// The fused matrix applied: linear AWG4 (C1 to C6).
    pub c6_linear_awg4: Vec3,
    /// After the highlight range and the toe (C7): linear AWG4.
    pub c7_linear_awg4: Vec3,
    /// LogC4 encode (C8).
    pub c8_logc4: LogC4,
    /// After Main's primary grade (C9).
    pub c9_logc4: LogC4,
    /// After Main's colour warp (C9b).
    pub c9b_logc4: LogC4,
    /// After the look (C10), log-to-log looks only: the output of the last node otherwise. The DNG
    /// stop point.
    pub c10_logc4: LogC4,
    /// After Main's curve (C11), before the look.
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

/// Values of C1 to C10 for one pixel.
#[derive(Clone, Copy)]
struct Staged {
    c6: Vec3,
    c7: Vec3,
    c8: LogC4,
    /// Main's values after C9, C9b and C11.
    main: MainStages,
    /// After the look: the last LogC4 value.
    c10: LogC4,
}

/// The prepared, immutable chain for one photo (stages C7 to C13 behind the fused matrix).
///
/// Build it once per photo and settings change, then call it from as many threads as you like.
///
/// `Debug` prints the [`ChainParams`] it was built from (with the look summarised), not the baked
/// tables.
#[derive(Clone)]
pub struct ColorChain {
    params: ChainParams,
    matrix: Mat3,
    tone: PreparedTone,
    /// The plan's steps in order, each with its node ready to run.
    steps: Vec<(Step, PreparedNode)>,
    /// The Film nodes the plan runs.
    films: BTreeMap<u32, FilmStep>,
    /// The mixers the plan runs.
    joins: BTreeMap<u32, JoinParams>,
    /// How many buffers a pixel's plan uses (zero for a plan of one pass).
    buffers: usize,
    look: LookStage,
    display: PreparedDisplay,
    output: OutputTarget,
    space: OutputSpace,
    dither: bool,
    before: BeforePoint,
    camera_default: CameraDefault,
    /// Set when C12 is not our cine rendering, so the DNG holds the scene values the cine rendering turns
    /// into this chain's picture (see [`ColorChain::dng_inverts_display`]).
    dng_inverse: Option<DngDisplayInverse>,
}

/// What a join step gives back.
enum JoinOut {
    /// The picture output.
    Picture(Vec3),
    /// A Splitter's red, green and blue outputs.
    Channels([Vec3; 3]),
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
    /// Prepares the nodes of the plan (bakes their curves, leaves out the steps at their identity), sets
    /// up the cine rendering for the output and classifies the look.
    ///
    /// # Errors
    ///
    /// [`DisplayError::NonFiniteMatrix`] when the fused matrix has a NaN or infinite entry;
    /// [`DisplayError::MissingNode`] when the plan runs a corrector that `params.nodes` has no
    /// parameters for.
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
        // C12 is not our cine rendering for a look that includes its own display rendering at any
        // intensity above 0 (at 0 the chain is our C12 exactly).
        let own_rendering = matches!(&look, LookStage::Baked { intensity_pct, .. } if intensity_fraction(*intensity_pct) > 0.0);
        let plan = &params.nodes.plan;
        let steps = plan
            .steps()
            .map(|step| {
                // A mixer, a Splitter or a Combiner has no node parameters: it runs as an identity corrector would
                // (nothing to grade, no mask rows) and the chain does its join.
                if step.join.is_some() {
                    let node = PreparedNode::new(
                        true,
                        &Trim::default(),
                        &GradeParams::default(),
                        &WarpParams::default(),
                        &CurveSet::default(),
                        &crate::mask::NodeMasks::default(),
                    );
                    return Ok((*step, node));
                }
                let n = params
                    .node_params(step.node)
                    .ok_or(DisplayError::MissingNode(step.node))?;
                let node =
                    PreparedNode::new(n.bypass, &n.trim, &n.grade, &n.warp, &n.curves, &n.masks);
                Ok((*step, node))
            })
            .collect::<Result<Vec<_>, DisplayError>>()?;
        Ok(Self {
            params: params.clone(),
            matrix: params.matrix,
            tone: params.tone.prepare(),
            steps,
            films: params
                .films
                .iter()
                .filter(|(id, _)| plan.steps().any(|s| s.node == **id))
                .map(|(id, f)| (*id, *f))
                .collect(),
            joins: params
                .joins
                .iter()
                .filter(|(id, _)| plan.steps().any(|s| s.node == **id))
                .map(|(id, j)| (*id, *j))
                .collect(),
            buffers: plan.buffers.len(),
            look,
            display: PreparedDisplay::new(params.output),
            output: params.output,
            space: params.output.space(),
            dither: params.dither,
            before: params.before,
            camera_default: params.camera_default,
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

    /// What the Before view shows.
    #[must_use]
    pub fn before_point(&self) -> BeforePoint {
        self.before
    }

    /// The camera default the Before view uses at [`BeforePoint::CameraDefault`].
    #[must_use]
    pub fn camera_default(&self) -> &CameraDefault {
        &self.camera_default
    }

    /// Whether the look includes its own display rendering (LogC3 to Rec.709), which replaces C12.
    #[must_use]
    pub fn look_includes_display_rendering(&self) -> bool {
        matches!(self.look, LookStage::Baked { .. })
    }

    /// Whether a DNG of this chain holds the picture through the inverse of our display transform
    /// instead of the C11 output itself: true when C12 is not our cine rendering (a look that includes its
    /// own display rendering at an intensity above 0). The DNG's fitted profile renders the cine
    /// rendering, so the pixels are the scene values that the cine rendering turns into the picture
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

    /// The nodes of the plan on one pixel: `c8` is the Input's picture and `c7` the linear values it
    /// was encoded from (the luma of C9 is taken from them for a node that reads the Input). Main's
    /// values after each of its steps go into `main`. `at` is the pixel's place in the photo frame, for
    /// the mask shapes.
    ///
    /// Each step is the node's four steps (`PreparedNode::apply`) and then the mask blend
    /// `out = in + m (node(in) - in)` (`docs/algorithms/06` section 1.1): `m` is 1 for a node with no
    /// mask rows, no Invert and no key, which then keeps its own bits; the blend branches on `m` being
    /// exactly 1 or exactly 0. A node's key output is its `m` whether or not the node is bypassed. A node
    /// that feeds a Layer above its first input also writes its graded value, the picture before the blend.
    /// A mixer, a Splitter or a Combiner is a join of the values its registers hold ([`ColorChain::run_join`]).
    fn run_nodes(
        &self,
        c8: LogC4,
        c7: Vec3,
        at: PixelAt<'_>,
        mut main: Option<&mut MainStages>,
    ) -> LogC4 {
        let mut registers = [[0.0f64; 3]; PLAN_REGISTERS];
        let mut slots = [0.0f64; PLAN_KEYS];
        let mut buffers = vec![[0.0f64; 3]; self.buffers];
        let put = |o: Option<PictureOut>,
                   v: [f64; 3],
                   registers: &mut [[f64; 3]; PLAN_REGISTERS],
                   buffers: &mut [[f64; 3]]| {
            if let Some(o) = o {
                if let Some(r) = o.register {
                    registers[usize::from(r)] = v;
                }
                if let Some(b) = o.buffer {
                    buffers[usize::from(b)] = v;
                }
            }
        };
        for (step, node) in &self.steps {
            if let Some(join) = &step.join {
                let outs = self.run_join(step.node, join, (c8, c7), (&registers, &slots, &buffers));
                match outs {
                    JoinOut::Picture(v) => put(step.picture_out, v, &mut registers, &mut buffers),
                    JoinOut::Channels(c) => {
                        for (o, v) in join.channels.iter().zip(c) {
                            put(*o, v, &mut registers, &mut buffers);
                        }
                    }
                }
                continue;
            }
            let (v_in, linear) = match step.picture {
                PictureRef::Input => (c8, Some(c7)),
                PictureRef::Register(r) => (LogC4(registers[usize::from(r)]), None),
                PictureRef::Buffer(b) => (LogC4(buffers[usize::from(b)]), None),
            };
            let key = step.key.map(|k| match k {
                KeyRef::Slot(s) => slots[usize::from(s)],
                KeyRef::Buffer(b) => buffers[usize::from(b)][0],
            });
            let stages = if step.node == crate::plan::MAIN_NODE {
                main.as_deref_mut()
            } else {
                None
            };
            let (out, m, graded) = if let Some(film) = self.films.get(&step.node) {
                let need_mask = node.masks().blends(key.is_some()) || step.key_out.is_some();
                Self::run_film(*film, node, v_in, linear, (key, need_mask), at, step.node)
            } else {
                let made = node.apply(v_in, linear, stages).0;
                // The mask: needed for the blend, and for the key output (also of a node that is skipped).
                let blends = node.masks().blends(key.is_some());
                let m = if (blends && !node.skips()) || step.key_out.is_some() {
                    node.mask(key, v_in, linear, at)
                } else {
                    1.0
                };
                let out = if !blends || node.skips() || m == 1.0 {
                    made
                } else if m == 0.0 {
                    v_in.0
                } else {
                    [
                        v_in.0[0] + m * (made[0] - v_in.0[0]),
                        v_in.0[1] + m * (made[1] - v_in.0[1]),
                        v_in.0[2] + m * (made[2] - v_in.0[2]),
                    ]
                };
                (out, m, made)
            };
            put(step.picture_out, out, &mut registers, &mut buffers);
            put(step.graded_out, graded, &mut registers, &mut buffers);
            if let Some(o) = step.key_out {
                if let Some(s) = o.slot {
                    slots[usize::from(s)] = m;
                }
                if let Some(b) = o.buffer {
                    buffers[usize::from(b)] = [m; 3];
                }
            }
        }
        match self.params.nodes.plan.result {
            PlanResult::Picture(PictureRef::Input) => c8,
            PlanResult::Picture(PictureRef::Register(r)) => LogC4(registers[usize::from(r)]),
            PlanResult::Picture(PictureRef::Buffer(b)) => LogC4(buffers[usize::from(b)]),
            PlanResult::Mask(KeyRef::Slot(s)) => LogC4([slots[usize::from(s)]; 3]),
            PlanResult::Mask(KeyRef::Buffer(b)) => LogC4(buffers[usize::from(b)]),
        }
    }

    /// A mixer, a Splitter or a Combiner on one pixel (`docs/25` section 3): reads every input before anything is
    /// written, so a step may write into a register its inputs leave.
    fn run_join(
        &self,
        id: u32,
        join: &JoinStep,
        (c8, c7): (LogC4, Vec3),
        (registers, slots, buffers): (&[[f64; 3]; PLAN_REGISTERS], &[f64; PLAN_KEYS], &[[f64; 3]]),
    ) -> JoinOut {
        let read = |r: PictureRef| match r {
            PictureRef::Input => Source {
                logc4: c8.0,
                linear: Some(c7),
            },
            PictureRef::Register(r) => Source {
                logc4: registers[usize::from(r)],
                linear: None,
            },
            PictureRef::Buffer(b) => Source {
                logc4: buffers[usize::from(b)],
                linear: None,
            },
        };
        let blank = Source {
            logc4: [0.0; 3],
            linear: None,
        };
        let mut all = [blank; PLAN_INPUTS];
        for (slot, &r) in all.iter_mut().zip(join.inputs()) {
            *slot = read(r);
        }
        let inputs = &all[..usize::from(join.count)];
        let params = self.joins.get(&id).copied().unwrap_or_default();
        match join.kind {
            JoinKind::Parallel => {
                let base = join.base.map_or(c8.0, |b| read(b).logc4);
                if params.bypass {
                    return JoinOut::Picture(base);
                }
                let mut logs = [[0.0; 3]; PLAN_INPUTS];
                for (slot, s) in logs.iter_mut().zip(inputs) {
                    *slot = s.logc4;
                }
                JoinOut::Picture(mix::parallel(base, &logs[..inputs.len()], |i| {
                    join.input_is_base(i)
                }))
            }
            JoinKind::Layer => {
                if params.bypass {
                    return JoinOut::Picture(inputs[0].logc4);
                }
                let mut alphas = [None; PLAN_INPUTS];
                for (slot, a) in alphas.iter_mut().zip(&join.alphas) {
                    *slot = a.map(|k| match k {
                        KeyRef::Slot(s) => slots[usize::from(s)],
                        KeyRef::Buffer(b) => buffers[usize::from(b)][0],
                    });
                }
                JoinOut::Picture(mix::layer(
                    params.mode,
                    params.opacity,
                    inputs,
                    &alphas[..inputs.len()],
                ))
            }
            JoinKind::Splitter => JoinOut::Channels(mix::split(inputs[0].logc4)),
            JoinKind::Combiner => JoinOut::Picture(mix::combine(
                inputs[0].logc4,
                inputs[1].logc4,
                inputs[2].logc4,
            )),
        }
    }

    /// A Film node on one pixel (`docs/24` section 2.6): `out = in + m (film - in)` in LogC4, with `film` the
    /// picture the render made for the node at this pixel and `m` the node's mask times its Strength. Returns the
    /// picture, `m` (the key output; it does not depend on Bypass) and the graded value (the film picture before the
    /// blend, the input when there is none). Without a picture for the pixel, or
    /// bypassed, the node passes its input through.
    fn run_film(
        film: FilmStep,
        node: &PreparedNode,
        v_in: LogC4,
        linear: Option<Vec3>,
        (key, need_mask): (Option<f64>, bool),
        at: PixelAt<'_>,
        id: u32,
    ) -> ([f64; 3], f64, [f64; 3]) {
        let mask = if need_mask {
            node.mask(key, v_in, linear, at)
        } else {
            1.0
        };
        let m = mask * film.strength;
        if film.bypass || m == 0.0 {
            return (v_in.0, m, v_in.0);
        }
        let made = at
            .and_then(|spot| spot.raster)
            .and_then(|r| r.planes.film(id, r.x, r.y));
        let Some(made) = made else {
            return (v_in.0, m, v_in.0);
        };
        let made = made.map(f64::from);
        if m == 1.0 {
            return (made, m, made);
        }
        (
            [
                v_in.0[0] + m * (made[0] - v_in.0[0]),
                v_in.0[1] + m * (made[1] - v_in.0[1]),
                v_in.0[2] + m * (made[2] - v_in.0[2]),
            ],
            m,
            made,
        )
    }

    /// The scheduler's schedule of this chain (`crate::schedule`): the footprints of the plan's steps and of their mask
    /// rows, in the plan's order, at a render with `u_px` output pixels along the photo's long edge. `film_reach_px`
    /// gives a Film node's reach (its halo); `None` for a node that does nothing (bypassed, Strength 0, or not built),
    /// which is then a corrector. Only footprints are consulted here, never the kind of a mask or node beyond what
    /// [`StepKind::declared`](crate::schedule::StepKind::declared) and [`OpKind::declared`](crate::mask::OpKind::declared) say.
    #[must_use]
    pub fn schedule(
        &self,
        u_px: f64,
        film_reach_px: &dyn Fn(u32) -> Option<usize>,
    ) -> crate::schedule::Schedule {
        use crate::schedule::StepKind;
        let raster_reach = if self.params.rasters.is_empty() {
            0
        } else {
            self.params.rasters.halo(u_px)
        };
        let ancestors = self.params.nodes.plan.picture_ancestors();
        let mut declared = Vec::new();
        // The picture steps (Film nodes that do something) by plan step, and what each reads.
        let mut picture_of: Vec<Option<usize>> = Vec::with_capacity(self.steps.len());
        let mut pictures = 0;
        for (step, node) in &self.steps {
            let film = self
                .films
                .contains_key(&step.node)
                .then(|| film_reach_px(step.node))
                .flatten();
            picture_of.push(film.map(|_| {
                pictures += 1;
                pictures - 1
            }));
            declared.push(match (&step.join, film) {
                (Some(j), _) => match j.kind {
                    JoinKind::Parallel => StepKind::Parallel,
                    JoinKind::Layer => StepKind::Layer,
                    JoinKind::Splitter => StepKind::Splitter,
                    JoinKind::Combiner => StepKind::Combiner,
                }
                .declared(0),
                (None, Some(reach)) => StepKind::Film.declared(reach),
                (None, None) => StepKind::Corrector.declared(0),
            });
            if self.params.rasters.is_empty() {
                continue;
            }
            declared.extend(
                node.masks()
                    .ops()
                    .iter()
                    .map(|o| o.kind.declared(raster_reach)),
            );
        }
        // A Film node reads the pictures of the Film nodes among its ancestors in the plan's graph, not every earlier one.
        let deps = picture_of
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_some())
            .map(|(i, _)| {
                ancestors[i]
                    .iter()
                    .filter_map(|&a| picture_of.get(a).copied().flatten())
                    .collect()
            })
            .collect();
        crate::schedule::Schedule::new(declared).with_picture_deps(deps)
    }

    /// For each Film node the plan runs, the ids of the Film nodes whose pictures it reads: its ancestors in the plan's
    /// graph (`Plan::picture_ancestors`), in the plan's order. Nodes on parallel branches are not in each other's lists.
    #[must_use]
    pub fn film_reads(&self) -> std::collections::BTreeMap<u32, Vec<u32>> {
        let ancestors = self.params.nodes.plan.picture_ancestors();
        self.steps
            .iter()
            .enumerate()
            .filter(|(_, (s, _))| self.films.contains_key(&s.node))
            .map(|(i, (s, _))| {
                let reads = ancestors[i]
                    .iter()
                    .filter_map(|&a| self.steps.get(a))
                    .map(|(a, _)| a.node)
                    .filter(|id| self.films.contains_key(id))
                    .collect();
                (s.node, reads)
            })
            .collect()
    }

    /// The ids of the Film nodes the plan runs, in the plan's order.
    #[must_use]
    pub fn film_nodes(&self) -> Vec<u32> {
        self.steps
            .iter()
            .filter(|(s, _)| self.films.contains_key(&s.node))
            .map(|(s, _)| s.node)
            .collect()
    }

    /// Whether the plan runs a Film node that does something (not bypassed, Strength above 0; `docs/24`). The render makes the node's picture before the colour pass
    /// (on the CPU in `ym-core`'s `film_pass`, on the device in `ym-gpu`'s `film_stack`) and the chain blends it.
    #[must_use]
    pub fn has_films(&self) -> bool {
        self.films.values().any(|f| !f.bypass && f.strength > 0.0)
    }

    fn stages(&self, camera_rgb: Vec3, at: PixelAt<'_>) -> Staged {
        let c6 = self.c1_to_c6(camera_rgb);
        let c7 = self.tone.apply_rgb(c6);
        let c8 = LinearAwg4(c7).to_logc4();
        let mut main = MainStages {
            c9: c8,
            c9b: c8,
            c11: c8,
        };
        let nodes = self.run_nodes(c8, c7, at, Some(&mut main));
        let c10 = self.c10(nodes);
        Staged {
            c6,
            c7,
            c8,
            main,
            c10,
        }
    }

    /// The last LogC4 value of the chain, behind the look: what C12 reads and the DNG holds.
    fn last_logc4(&self, camera_rgb: Vec3, at: PixelAt<'_>) -> LogC4 {
        let c6 = self.c1_to_c6(camera_rgb);
        let c7 = self.tone.apply_rgb(c6);
        let c8 = LinearAwg4(c7).to_logc4();
        self.c10(self.run_nodes(c8, c7, at, None))
    }

    // ---- single pixels -------------------------------------------------------------------------

    /// Whether any mask row of the plan's nodes reads the pixel's place in the photo (a Linear or a
    /// Radial), or needs it to make its plane (a raster row): a render then needs the photo-frame position of
    /// every pixel ([`Frame::with_photo`](super::Frame::with_photo)).
    #[must_use]
    pub fn uses_photo_position(&self) -> bool {
        self.has_films() || self.steps.iter().any(|(_, n)| n.masks().needs_position())
    }

    /// Whether any mask row of the plan's nodes reads a raster plane that exists: the render then makes the planes of
    /// [`ColorChain::rasters`] over its window before the colour pass ([`crate::mask_raster`]).
    #[must_use]
    pub fn uses_rasters(&self) -> bool {
        // A raster row with no plane (a Background row before its map exists, an Object row with no point) is
        // in the nodes but not in the set: it selects nothing, and a chain with only such rows has no planes to make.
        !self.params.rasters.is_empty() && self.steps.iter().any(|(_, n)| n.masks().needs_rasters())
    }

    /// The raster rows of the chain, in plane order.
    #[must_use]
    pub fn rasters(&self) -> &RasterSet {
        &self.params.rasters
    }

    /// The Input's luma of one camera pixel, the guide of Refine edges (`docs/algorithms/06` section 5.1):
    /// `encode(Y(lin))` of the linear values C7 made, held to [0, 1]. It is the luma a Brightness range of a
    /// node that reads the Input sees, with no node's grade in it.
    #[must_use]
    pub fn guide_pixel(&self, camera_rgb: Vec3) -> f64 {
        let c6 = self.c1_to_c6(camera_rgb);
        let c7 = self.tone.apply_rgb(c6);
        unit(PreparedMasks::luma_of(LogC4([0.0; 3]), Some(c7)))
    }

    /// Every stage's value for one camera pixel whose place in the photo frame is not known: the mask
    /// shapes select nothing there ([`ColorChain::trace_at`]).
    #[must_use]
    pub fn trace(&self, camera_rgb: Vec3) -> StageTrace {
        self.trace_at(camera_rgb, None)
    }

    /// Every stage's value for one camera pixel at `at` in the photo frame.
    #[must_use]
    pub fn trace_at(&self, camera_rgb: Vec3, at: PixelAt<'_>) -> StageTrace {
        let s = self.stages(camera_rgb, at);
        let c12 = self.c12(s.c10);
        StageTrace {
            c6_linear_awg4: s.c6,
            c7_linear_awg4: s.c7,
            c8_logc4: s.c8,
            c9_logc4: s.main.c9,
            c9b_logc4: s.main.c9b,
            c10_logc4: s.c10,
            c11_logc4: s.main.c11,
            c12_display_linear: c12,
            c13_encoded: self.encode(c12),
        }
    }

    /// The whole chain: camera RGB to display code values in `[0, 1]` (not quantised, not dithered).
    #[must_use]
    pub fn apply_pixel(&self, camera_rgb: Vec3) -> Vec3 {
        self.apply_pixel_at(camera_rgb, None)
    }

    /// [`ColorChain::apply_pixel`] for a pixel at `at` in the photo frame. For a chain whose plan ends at a
    /// node's mask ([`ColorChain::shows_mask`]) the code values are that mask, `m` in all three channels.
    #[must_use]
    pub fn apply_pixel_at(&self, camera_rgb: Vec3, at: PixelAt<'_>) -> Vec3 {
        if self.shows_mask() {
            return self.mask_pixel_at(camera_rgb, at);
        }
        self.encode(self.apply_to_display_linear_at(camera_rgb, at))
    }

    /// Whether the plan ends at a node's mask (`Stop::Mask`, the mask view): the picture the chain makes is the
    /// mask as grey code values, after the look, C12 and C13 have been left out (`docs/23` section 5.8).
    #[must_use]
    pub fn shows_mask(&self) -> bool {
        matches!(self.params.nodes.plan.result, PlanResult::Mask(_))
    }

    /// The mask view for one pixel: the mask `m` of the node the plan ends at as `[m, m, m]` code values in
    /// `[0, 1]` (not quantised, not dithered). A chain whose plan does not end at a mask gives 1, the mask of
    /// a node that has no rows.
    #[must_use]
    pub fn mask_pixel_at(&self, camera_rgb: Vec3, at: PixelAt<'_>) -> Vec3 {
        if !self.shows_mask() {
            return [1.0; 3];
        }
        let m = unit(self.last_logc4_unlooked(camera_rgb, at).0[0]);
        [m; 3]
    }

    /// The plan's value on a pixel without the look: what the mask view reads.
    fn last_logc4_unlooked(&self, camera_rgb: Vec3, at: PixelAt<'_>) -> LogC4 {
        let c6 = self.c1_to_c6(camera_rgb);
        let c7 = self.tone.apply_rgb(c6);
        let c8 = LinearAwg4(c7).to_logc4();
        self.run_nodes(c8, c7, at, None)
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
        self.apply_to_display_linear_at(camera_rgb, None)
    }

    /// [`ColorChain::apply_to_display_linear`] for a pixel at `at` in the photo frame.
    #[must_use]
    pub fn apply_to_display_linear_at(&self, camera_rgb: Vec3, at: PixelAt<'_>) -> Vec3 {
        self.c12(self.last_logc4(camera_rgb, at))
    }

    /// C1 to C10: camera RGB to LogC4 with the nodes (trims, grade, warp, curves) and then the look
    /// applied (the DNG stop point before the decode). With a look that includes display rendering the
    /// look has not run at this point (the curve of the last node is the last LogC4 stage).
    #[must_use]
    pub fn apply_logc4_pixel(&self, camera_rgb: Vec3) -> LogC4 {
        self.apply_logc4_pixel_at(camera_rgb, None)
    }

    /// [`ColorChain::apply_logc4_pixel`] for a pixel at `at` in the photo frame.
    #[must_use]
    pub fn apply_logc4_pixel_at(&self, camera_rgb: Vec3, at: PixelAt<'_>) -> LogC4 {
        self.last_logc4(camera_rgb, at)
    }

    /// The DNG path (`docs/09`, Pixel pipeline per format): the nodes and then the look, with LogC4
    /// decoded back to linear AWG4 when C12 is our cine rendering. Negative values are not clipped
    /// here (the writer clips and counts them).
    ///
    /// When C12 is not our cine rendering ([`ColorChain::dng_inverts_display`]) the result is instead the
    /// scene-linear AWG4 triple that the cine rendering turns into this chain's display-linear picture
    /// (look, intensity blend and all), limited to the range of the DNG profile
    /// ([`DNG_SCENE_LIMIT`](super::DNG_SCENE_LIMIT)): Adobe's renderer shows approximately that picture.
    /// Build the chain for [`OutputTarget::Srgb`], the display the DNG profile was fitted to; for another
    /// output the display value is first converted to Rec.709 primaries.
    #[must_use]
    pub fn apply_dng_pixel(&self, camera_rgb: Vec3) -> Vec3 {
        self.apply_dng_pixel_at(camera_rgb, None)
    }

    /// [`ColorChain::apply_dng_pixel`] for a pixel at `at` in the photo frame.
    #[must_use]
    pub fn apply_dng_pixel_at(&self, camera_rgb: Vec3, at: PixelAt<'_>) -> Vec3 {
        let last = self.last_logc4(camera_rgb, at);
        match &self.dng_inverse {
            // C12 is our cine rendering: the pixels are the scene values it takes in.
            None => last.to_linear().0,
            Some(inverse) => inverse.scene_awg4(self.c12(last)),
        }
    }

    /// The Before view for one pixel, as encoded display code values in `[0, 1]` (not quantised, not
    /// dithered).
    ///
    /// At [`BeforePoint::CameraDefault`]: the camera default's matrix and tone curve on the sanitised
    /// camera RGB give display-linear Rec.709, which is converted to the output primaries and encoded
    /// as C13 does for the final picture. At the two LogC4 points: the LogC4 signal at that point, each
    /// channel clamped to `[0, 1]` and used directly as a display code value (see the module docs).
    #[must_use]
    pub fn apply_before_pixel(&self, camera_rgb: Vec3) -> Vec3 {
        self.apply_before_pixel_at(camera_rgb, None)
    }

    /// [`ColorChain::apply_before_pixel`] for a pixel at `at` in the photo frame (only the two LogC4
    /// reading points run nodes, and only the nodes' masks read the place).
    #[must_use]
    pub fn apply_before_pixel_at(&self, camera_rgb: Vec3, at: PixelAt<'_>) -> Vec3 {
        match self.before {
            BeforePoint::CameraDefault => {
                let rec709 = self
                    .camera_default
                    .display_linear_rec709(camera_rgb.map(sanitize));
                self.encode(self.space.from_linear_rec709(LinearRec709(rec709)))
            }
            BeforePoint::CameraLog => self.stages(camera_rgb, at).c8.0.map(unit),
            BeforePoint::AfterGrade => self.stages(camera_rgb, at).main.c9.0.map(unit),
        }
    }

    /// One pixel of the chosen view as unquantised code values: [`ColorChain::apply_pixel`] for
    /// [`View::Final`], [`ColorChain::apply_before_pixel`] for [`View::Before`].
    #[must_use]
    pub fn apply_view_pixel(&self, view: View, camera_rgb: Vec3) -> Vec3 {
        self.apply_view_pixel_at(view, camera_rgb, None)
    }

    /// [`ColorChain::apply_view_pixel`] for a pixel at `at` in the photo frame.
    #[must_use]
    pub fn apply_view_pixel_at(&self, view: View, camera_rgb: Vec3, at: PixelAt<'_>) -> Vec3 {
        match view {
            View::Final => self.apply_pixel_at(camera_rgb, at),
            View::Before => self.apply_before_pixel_at(camera_rgb, at),
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
