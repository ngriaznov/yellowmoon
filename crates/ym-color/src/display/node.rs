//! A corrector node on the CPU reference (`docs/23-nodes.md` section 1.3, `docs/algorithms/06-masks.md`
//! section 1.1): the trims, C9, C9b and C11 of one node, with each step that is at its identity left
//! out.
//!
//! [`NodeParams`] is what the settings give a node; [`NodeSet`] is what a chain needs beyond Main's
//! fields of [`ChainParams`](super::ChainParams): the compiled [`Plan`], Main's trim, bypass and masks and
//! the other correctors. `PreparedNode` is a node ready to run on a pixel, with its mask rows
//! ([`crate::mask`]) prepared: the chain blends by the node's mask `m` after the four steps.

use std::collections::BTreeMap;

use super::trim::Trim;
use crate::consts::Vec3;
use crate::curve::{BakedCurves, CurveSet};
use crate::gamut::LinearAwg4;
use crate::grade::{Grade, GradeParams, grade_luma};
use crate::logc::LogC4;
use crate::mask::{NodeMasks, PixelAt, PreparedMasks};
use crate::plan::Plan;
use crate::warp::{Warp, WarpParams};

/// One corrector's parameters: its trims, its grade (C9), its colour warp (C9b) and its curves (C11).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NodeParams {
    /// A bypassed node passes its picture through.
    pub bypass: bool,
    /// The trims, in linear AWG4, before the grade.
    pub trim: Trim,
    /// C9: the primary grade. All zero is the identity.
    pub grade: GradeParams,
    /// C9b: the colour warp's pins. No pins is the identity.
    pub warp: WarpParams,
    /// C11: the four curves. The identity by default.
    pub curves: CurveSet,
    /// The mask rows (the enabled ones) and the node's Invert. None: `m = 1`.
    pub masks: NodeMasks,
}

/// A Film node in the plan (`docs/24-film.md`). Its spatial passes cannot run per pixel, so the render makes them
/// before the colour pass and hands the result to the chain ([`RasterPlanes::film`](crate::mask::RasterPlanes)); the
/// node's [`NodeParams`] hold its masks and are otherwise the identity. The chain blends
/// `out = in + m * (film - in)` with `m` the mask times `strength`, like a corrector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmStep {
    /// The node's Strength, 0 to 1.
    pub strength: f64,
    /// A bypassed node passes its picture through (its key output still carries `m`).
    pub bypass: bool,
    /// A Face Refinement node's step (`docs/27` S-13): its picture covers only the regions around the faces, and a pixel it
    /// holds as all zeros is "no picture here" and keeps the node's input, as the CPU's `NaN` marker does. A Film node's
    /// picture covers everything it is asked for.
    pub face: bool,
}

/// What Main has beyond the grade, warp and curves that [`ChainParams`](super::ChainParams) holds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MainNode {
    /// Main passes its picture through.
    pub bypass: bool,
    /// Main's trims.
    pub trim: Trim,
    /// Main's mask rows and Invert.
    pub masks: NodeMasks,
}

/// The nodes of a chain: the plan and the parameters of the nodes it runs.
///
/// Main's grade, warp and curves are [`ChainParams::grade`](super::ChainParams), `warp` and `curves`
/// (the fields they have always been); its trim and bypass are [`NodeSet::main`]; every other
/// corrector is in [`NodeSet::others`].
#[derive(Debug, Clone, PartialEq)]
pub struct NodeSet {
    /// The compiled graph. [`Plan::main_only`] for the default graph.
    pub plan: Plan,
    /// Main's trim and bypass.
    pub main: MainNode,
    /// The correctors other than Main, by id.
    pub others: BTreeMap<u32, NodeParams>,
}

impl Default for NodeSet {
    /// The default graph: Main alone, no trim, not bypassed.
    fn default() -> Self {
        Self {
            plan: Plan::main_only(),
            main: MainNode::default(),
            others: BTreeMap::new(),
        }
    }
}

/// Main's values after each of its steps, for [`StageTrace`](super::StageTrace).
#[derive(Debug, Clone, Copy)]
pub(super) struct MainStages {
    pub c9: LogC4,
    pub c9b: LogC4,
    pub c11: LogC4,
}

/// A node ready to run: the gain, and the grade, warp and curves that are not the identity, and the mask
/// rows.
#[derive(Debug, Clone)]
pub(super) struct PreparedNode {
    /// Nothing to do: bypassed, or every step at its identity.
    skip: bool,
    gain: Option<Vec3>,
    grade: Option<Grade>,
    warp: Option<Warp>,
    curves: Option<BakedCurves>,
    masks: PreparedMasks,
}

impl PreparedNode {
    pub(super) fn new(
        bypass: bool,
        trim: &Trim,
        grade: &GradeParams,
        warp: &WarpParams,
        curves: &CurveSet,
        masks: &NodeMasks,
    ) -> Self {
        let gain = trim.gain();
        let grade = (!grade.is_identity()).then(|| Grade::new(grade));
        let warp = Some(Warp::new(warp)).filter(|w| !w.is_identity());
        let curves = (!curves.is_identity()).then(|| curves.bake());
        let skip =
            bypass || (gain.is_none() && grade.is_none() && warp.is_none() && curves.is_none());
        Self {
            skip,
            gain,
            grade,
            warp,
            curves,
            masks: masks.prepare(),
        }
    }

    /// Whether the node's picture is its input as it is: bypassed, or every step at its identity.
    pub(super) fn skips(&self) -> bool {
        self.skip
    }

    /// The node's prepared mask rows.
    pub(super) fn masks(&self) -> &PreparedMasks {
        &self.masks
    }

    /// The node's mask `m` at a pixel (`docs/algorithms/06` section 3): `key` is the linked key's value,
    /// `v_in` the picture the node reads and `linear` that picture as linear AWG4 when it is known
    /// exactly (the Input's). The mask reads the node's input, never what the node made of it.
    pub(super) fn mask(
        &self,
        key: Option<f64>,
        v_in: LogC4,
        linear: Option<Vec3>,
        at: PixelAt<'_>,
    ) -> f64 {
        let l_in = if self.masks.needs_luma() {
            PreparedMasks::luma_of(v_in, linear)
        } else {
            0.0
        };
        self.masks.mask(key, v_in, l_in, at)
    }

    /// The node on one pixel: `v_in` is the picture it reads and `linear` that picture as linear
    /// AWG4 when it is known exactly (the Input's, which C7 made: no decode of an encode); the
    /// result is the node's LogC4 value before the mask blend. `main` receives the values after C9,
    /// C9b and C11.
    pub(super) fn apply(
        &self,
        v_in: LogC4,
        linear: Option<Vec3>,
        main: Option<&mut MainStages>,
    ) -> LogC4 {
        if self.skip {
            return v_in;
        }
        // 1. The trims, in linear AWG4.
        let mut linear = linear;
        let mut v = v_in;
        if let Some(g) = self.gain {
            let lin = linear.unwrap_or_else(|| v_in.to_linear().0);
            let scaled = [lin[0] * g[0], lin[1] * g[1], lin[2] * g[2]];
            v = LinearAwg4(scaled).to_logc4();
            linear = Some(scaled);
        }
        // 2. C9, with the luma of the linear values entering it.
        if let Some(grade) = &self.grade {
            let lin = linear.unwrap_or_else(|| v.to_linear().0);
            v = grade.apply(v, grade_luma(LinearAwg4(lin)));
        }
        let c9 = v;
        // 3. C9b.
        if let Some(warp) = &self.warp {
            v = warp.apply(v);
        }
        let c9b = v;
        // 4. C11.
        if let Some(curves) = &self.curves {
            v = curves.apply(v);
        }
        if let Some(stages) = main {
            *stages = MainStages { c9, c9b, c11: v };
        }
        v
    }
}
