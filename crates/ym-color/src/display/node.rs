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
use crate::hue::{ELEMENT_COUNT, ElementStep, HueStep};
use crate::logc::LogC4;
use crate::mask::{NodeMasks, PixelAt, PreparedMasks};
use crate::plan::Plan;
use crate::uniform::{self, Amounts, Target};
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
    /// A Colour Uniform node's step (`docs/32`): `Some` for a node of that kind, whatever its values (a node with no
    /// target or no amount is the identity and is skipped), `None` for a corrector.
    pub uniform: Option<UniformStep>,
    /// A Reference Grade node's hue and element steps (`docs/34` section 7.1): `Some` for a node of that kind, whatever its
    /// values (a node with both steps at the identity runs only its corrector steps), `None` for any other kind.
    pub reference: Option<ReferenceStep>,
}

/// A Reference Grade node's own steps after the corrector's (`docs/34` sections 4 and 4a), in effective values (Strength
/// already applied): the hue curves, then the element grades.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReferenceStep {
    /// The hue curves (step 6 of the node).
    pub hue: HueStep,
    /// The element grades (step 7 of the node), each through its soft mask.
    pub elements: ElementStep,
}

impl ReferenceStep {
    /// Whether both steps do nothing.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.hue.is_identity() && self.elements.is_identity()
    }
}

/// A Colour Uniform node's target and amounts (`docs/32` section 3): what [`uniform::apply`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct UniformStep {
    /// The colour pulled toward; `None` does nothing.
    pub target: Option<Target>,
    /// How far each axis is pulled; all zero does nothing.
    pub amounts: Amounts,
}

impl UniformStep {
    /// Whether the step leaves its input as it is: no target, or no amount.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.target.is_none() || self.amounts.is_zero()
    }
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
    /// A Blur node's step (`docs/33` section 6): its picture is the node's finished output (the blend in linear light is
    /// done by the render's pass, with the node's mask), so the chain takes it as it is where the node's mask is above 0 and
    /// does not blend it again. `strength` is 1.
    pub blur: bool,
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
    /// A Colour Uniform node's live step; `None` for a corrector and for a node at its identity.
    uniform: Option<(Target, Amounts)>,
    /// Whether the node is a Colour Uniform node (live or not): its step kind is `ColorUniform`.
    is_uniform: bool,
    /// A Reference Grade node's live steps; `None` for any other kind and for a node whose hue and element steps are at the
    /// identity.
    reference: Option<ReferenceStep>,
    /// Whether the node is a Reference Grade node (live or not): its step kind is `ReferenceGrade`.
    is_reference: bool,
    masks: PreparedMasks,
}

impl PreparedNode {
    // One argument per part of a node's parameters, as `NodeParams` lists them; a struct would only move the list.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        bypass: bool,
        trim: &Trim,
        grade: &GradeParams,
        warp: &WarpParams,
        curves: &CurveSet,
        uniform: Option<&UniformStep>,
        reference: Option<&ReferenceStep>,
        masks: &NodeMasks,
    ) -> Self {
        let gain = trim.gain();
        let grade = (!grade.is_identity()).then(|| Grade::new(grade));
        let warp = Some(Warp::new(warp)).filter(|w| !w.is_identity());
        let curves = (!curves.is_identity()).then(|| curves.bake());
        let live = uniform
            .filter(|u| !u.is_identity())
            .and_then(|u| u.target.map(|t| (t, u.amounts)));
        let live_reference = reference.filter(|r| !r.is_identity()).cloned();
        let skip = bypass
            || (gain.is_none()
                && grade.is_none()
                && warp.is_none()
                && curves.is_none()
                && live.is_none()
                && live_reference.is_none());
        Self {
            skip,
            uniform: live,
            is_uniform: uniform.is_some(),
            reference: live_reference,
            is_reference: reference.is_some(),
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

    /// Whether the node is a Colour Uniform node (`StepKind::ColorUniform`).
    pub(super) fn is_uniform(&self) -> bool {
        self.is_uniform
    }

    /// Whether the node is a Reference Grade node (`StepKind::ReferenceGrade`).
    pub(super) fn is_reference(&self) -> bool {
        self.is_reference
    }

    /// Whether the node's element grades are live: the chain then asks the render's planes for the element masks.
    pub(super) fn has_elements(&self) -> bool {
        self.reference
            .as_ref()
            .is_some_and(|r| !r.elements.is_identity())
    }

    /// Whether element `e` of the node is live (its effective grade is not the identity).
    pub(super) fn element_is_live(&self, e: usize) -> bool {
        self.reference
            .as_ref()
            .is_some_and(|r| r.elements.is_live(e))
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
    /// C9b and C11. `elements` holds a Reference Grade node's seven soft masks at the pixel (`docs/34` section 4a.3); `None`
    /// is every mask 0, so the element step does nothing.
    pub(super) fn apply(
        &self,
        v_in: LogC4,
        linear: Option<Vec3>,
        main: Option<&mut MainStages>,
        elements: Option<&[f64; ELEMENT_COUNT]>,
    ) -> LogC4 {
        if self.skip {
            return v_in;
        }
        let mut linear = linear;
        let mut v = v_in;
        // 0. A Colour Uniform node's pull (`docs/32` sections 3 and 12a.1), in f64: first, on the node's input, so that the
        // target the person picked there is a fixed point of it. The corrector's steps then run on its result, whose exact
        // linear values are no longer known: they are decoded from it.
        if let Some((target, amounts)) = &self.uniform {
            v = LogC4(uniform::apply(v.0, Some(target), *amounts));
            linear = None;
        }
        // 1. The trims, in linear AWG4.
        if let Some(g) = self.gain {
            let lin = linear.unwrap_or_else(|| v.to_linear().0);
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
        // 5. A Reference Grade node's hue curves (`docs/34` section 4.2), then its element grades (section 4a.2): the
        // corrector's steps keep their order whole and these two come last.
        if let Some(r) = &self.reference {
            v = LogC4(r.hue.apply(v.0));
            if let Some(masks) = elements {
                v = LogC4(r.elements.apply(v.0, masks));
            }
        }
        v
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    use crate::curve::Curve;
    use crate::hue::{ElementGrade, SUBJECT};

    fn masks() -> NodeMasks {
        NodeMasks::default()
    }

    fn graded() -> (Trim, GradeParams, CurveSet) {
        let trim = Trim {
            exposure_ev: 0.3,
            temperature: 12.0,
            tint: -6.0,
        };
        let grade = GradeParams {
            contrast: 14.0,
            saturation: 18.0,
            ..GradeParams::default()
        };
        let curves = CurveSet {
            luma: Curve::new(&[(0.0, 0.0), (0.4, 0.45), (1.0, 1.0)]).unwrap(),
            ..CurveSet::default()
        };
        (trim, grade, curves)
    }

    fn step() -> ReferenceStep {
        let mut grades = [ElementGrade::IDENTITY; ELEMENT_COUNT];
        grades[0] = ElementGrade {
            exposure_ev: -0.4,
            hue_deg: 25.0,
            chroma: 1.3,
        };
        grades[SUBJECT].exposure_ev = 0.5;
        ReferenceStep {
            hue: HueStep::new(&[(20.0, 45.0), (200.0, -30.0)], &[(100.0, 1.6)]),
            elements: ElementStep::new(grades),
        }
    }

    fn node(reference: Option<&ReferenceStep>, with_grade: bool) -> PreparedNode {
        let (trim, grade, curves) = graded();
        let (trim, grade, curves) = if with_grade {
            (trim, grade, curves)
        } else {
            (Trim::default(), GradeParams::default(), CurveSet::default())
        };
        PreparedNode::new(
            false,
            &trim,
            &grade,
            &WarpParams::default(),
            &curves,
            None,
            reference,
            &masks(),
        )
    }

    fn samples() -> Vec<Vec3> {
        let mut v = Vec::new();
        for r in [0.1, 0.3, 0.5, 0.7] {
            for g in [0.05, 0.35, 0.6] {
                for b in [0.2, 0.4, 0.8] {
                    v.push([r, g, b]);
                }
            }
        }
        v
    }

    const M: [f64; ELEMENT_COUNT] = [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    #[test]
    fn a_node_whose_steps_are_all_at_the_identity_is_skipped_and_changes_no_bit() {
        let identity = ReferenceStep::default();
        let n = PreparedNode::new(
            false,
            &Trim::default(),
            &GradeParams::default(),
            &WarpParams::default(),
            &CurveSet::default(),
            None,
            Some(&identity),
            &masks(),
        );
        assert!(n.skips() && n.is_reference());
        for v in samples() {
            assert_eq!(n.apply(LogC4(v), None, None, Some(&M)).0, v);
        }
        // A bypassed node passes its picture through whatever its steps are.
        let live = step();
        let bypassed = PreparedNode::new(
            true,
            &Trim::default(),
            &GradeParams::default(),
            &WarpParams::default(),
            &CurveSet::default(),
            None,
            Some(&live),
            &masks(),
        );
        assert!(bypassed.skips());
        for v in samples() {
            assert_eq!(bypassed.apply(LogC4(v), None, None, Some(&M)).0, v);
        }
        // A mask of 0 everywhere, or no planes at all, leaves the element step out: the hue step alone remains.
        let hue_only = ReferenceStep {
            hue: live.hue.clone(),
            elements: ElementStep::default(),
        };
        let a = node(Some(&live), false);
        let b = node(Some(&hue_only), false);
        for v in samples() {
            assert_eq!(
                a.apply(LogC4(v), None, None, Some(&[0.0; ELEMENT_COUNT])).0,
                b.apply(LogC4(v), None, None, None).0
            );
            assert_eq!(
                a.apply(LogC4(v), None, None, None).0,
                b.apply(LogC4(v), None, None, None).0
            );
        }
    }

    /// The order of `docs/34` section 2.3: the corrector's steps, then the hue curves, then the element grades.
    #[test]
    fn the_steps_run_in_the_order_of_the_spec_and_another_order_differs() {
        let live = step();
        let full = node(Some(&live), true);
        let corrector = node(None, true);
        let plain = node(None, false);
        let mut worst_hue_first = 0.0f64;
        let mut worst_elements_first = 0.0f64;
        for v in samples() {
            let got = full.apply(LogC4(v), None, None, Some(&M)).0;
            let c11 = corrector.apply(LogC4(v), None, None, None).0;
            let want = live.elements.apply(live.hue.apply(c11), &M);
            for k in 0..3 {
                assert!((got[k] - want[k]).abs() <= 1e-12, "{v:?}");
            }
            // The hue step before the corrector's steps.
            let a = corrector
                .apply(LogC4(live.hue.apply(v)), None, None, None)
                .0;
            let a = live.elements.apply(a, &M);
            // The elements before the hue step.
            let b = live.hue.apply(live.elements.apply(c11, &M));
            for k in 0..3 {
                worst_hue_first = worst_hue_first.max((got[k] - a[k]).abs());
                worst_elements_first = worst_elements_first.max((got[k] - b[k]).abs());
            }
            // Without a Reference step the node is the corrector's.
            assert_eq!(plain.apply(LogC4(v), None, None, None).0, v);
        }
        assert!(worst_hue_first > 1e-3, "{worst_hue_first}");
        assert!(worst_elements_first > 1e-3, "{worst_elements_first}");
    }
}
