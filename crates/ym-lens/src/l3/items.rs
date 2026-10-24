//! The drawn items of L3: ghosts, starbursts and streaks (`docs/21` section 4, L3).
//!
//! Every item has **unit energy** before its light budget: its density integrates to 1 over the plane, so the
//! light a source sends into it is `a_i * tint_c * E_c` and nothing else. A pixel's value is that light times the
//! density at the pixel, in half-diagonal units (a density per half-diagonal squared), so the picture is the same
//! at every scale.
//!
//! * A **ghost** is the lens's aperture (or, for the attachment's own ghosts, a round one) of apothem `s`, drawn
//!   at `m` times the source's position, stretched along the axis, with the lens's rim profile inside and a
//!   smooth edge, cut by a clipping circle. Its density is `rim(g) Q((g - 1) / sigma) Q((d - R) / sigma)` over
//!   the shape's gauge `g` (1 on the edge) and the distance `d` to the clipping circle's centre, normalised by
//!   the exact integral of the first two factors. The edge is a Gaussian step whose standard deviation is the
//!   ghost's own softness, the source's extent and at least a pixel, so a pixel grid samples it without
//!   aliasing; the light the clipping circle blocks is lost.
//! * A **starburst** deposits the lens's spike table, projected on the camera's channels, at the source. Each
//!   pixel gets the exact mass of the piecewise-linear (tent) reconstruction of the table over its area.
//! * A **streak** is separable: a profile along the axis (a sharp core and a power-law tail, cut smoothly at its
//!   reach) times a Gaussian across it as tall as the source; each pixel gets the exact mass over its area.

use rayon::prelude::*;

use crate::consts::{
    COARSE_MIN_AREA_PX, COARSE_NODES_PER_FEATURE, COARSE_RIM_FEATURE, COMPACT_EXTENT_HD,
    EDGE_FLOOR_PX, EDGE_REACH_SIGMAS, GHOST_PRUNE_BUDGET, SPIKE_SIZE,
};
use crate::data::{Aperture, GhostData};
use crate::l3::geometry::{Frame, Span};
use crate::l3::profile::{NormalTail, RimCurve, cosine_window, gauss8, normal_tail};
use crate::l3::sources::Source;
use crate::l3::terms::Terms;
use crate::plan::LensPlan;
use crate::spectral::CameraResponse;

/// A shape as a gauge: the aperture polygon blended with its inscribed circle (`roundness`).
///
/// `gauge(x, y)` is 1 on the edge of the shape of apothem 1 and grows linearly with the distance, so the
/// shape of apothem `s` is `gauge <= s`, and its level sets are scaled copies of one another.
#[derive(Clone, Debug)]
pub(crate) struct Shape {
    /// The outward normals of the polygon's edges (empty for a round shape).
    normals: Vec<[f64; 2]>,
    /// The weight of the circle in the blend.
    rho: f64,
    /// The area of `gauge <= 1`.
    area: f64,
    /// The largest distance from the centre of a point of `gauge <= 1`.
    reach: f64,
}

impl Shape {
    /// A circle.
    pub(crate) fn round() -> Self {
        Shape {
            normals: Vec::new(),
            rho: 1.0,
            area: std::f64::consts::PI,
            reach: 1.0,
        }
    }

    /// The lens's aperture: `blades` straight blades turned by `rotation_deg`, rounded by `roundness`.
    pub(crate) fn aperture(ap: &Aperture) -> Self {
        if ap.blades < 3 {
            return Self::round();
        }
        let n = ap.blades as usize;
        let rot = ap.rotation_deg.to_radians();
        let step = 2.0 * std::f64::consts::PI / n as f64;
        let normals = (0..n)
            .map(|i| {
                let t = rot + step * i as f64;
                [libm::cos(t), libm::sin(t)]
            })
            .collect();
        let rho = ap.roundness.clamp(0.0, 1.0);
        let half = std::f64::consts::PI / n as f64;
        // Within a sector the edge is at distance 1 / ((1 - rho) cos(phi) + rho) from the centre.
        let radius = |phi: f64| 1.0 / ((1.0 - rho) * libm::cos(phi) + rho);
        let mut area = 0.0;
        let panels = 4;
        for p in 0..panels {
            let (a, b) = (
                half * p as f64 / panels as f64,
                half * (p + 1) as f64 / panels as f64,
            );
            area += gauss8(a, b, &mut |phi| radius(phi) * radius(phi));
        }
        Shape {
            normals,
            rho,
            area: area * n as f64,
            reach: radius(half),
        }
    }

    /// The gauge at `(x, y)` from the shape's centre.
    #[inline]
    pub(crate) fn gauge(&self, x: f64, y: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        if self.normals.is_empty() {
            return r;
        }
        let mut m = f64::NEG_INFINITY;
        for n in &self.normals {
            m = m.max(n[0] * x + n[1] * y);
        }
        (1.0 - self.rho) * m + self.rho * r
    }

    /// The area of the shape of apothem 1.
    #[cfg(test)]
    pub(crate) fn area(&self) -> f64 {
        self.area
    }

    /// The outward normals of the polygon's edges (empty for a round shape) and the weight of the circle in
    /// the blend, for the device.
    pub(crate) fn device_shape(&self) -> (&[[f64; 2]], f64) {
        (&self.normals, self.rho)
    }
}

/// A ghost's radial profile at the normalised radius `t` (1 on the edge): the rim curve, whose argument stops
/// rising smoothly as the edge begins, times the Gaussian step of the edge `Q((t - 1) / sigma)`.
///
/// The argument is `1 - sigma * ramp((1 - t) / sigma)`: `t` well inside the edge and 1 well outside, with every
/// derivative continuous, so the profile has no crease where the rim curve ends and a pixel grid samples it
/// exactly.
#[inline]
fn profile(rim: &RimCurve, tail: &NormalTail, t: f64, sigma_g: f64) -> f64 {
    let z = (1.0 - t) / sigma_g;
    let arg = if z >= 8.0 {
        t
    } else if z <= -8.0 {
        1.0
    } else {
        1.0 - sigma_g * tail.ramp(z)
    };
    rim.at(arg) * tail.q(-z)
}

/// `integral over t >= 0 of profile(t) t dt`: the radial integral that normalises a ghost (the area of
/// `gauge <= t` is `A t^2`, so the plane integral of the profile is `2 A integral(profile t dt)`).
pub(crate) fn profile_integral(rim: &RimCurve, sigma_g: f64) -> f64 {
    let q = normal_tail();
    let t_max = 1.0 + EDGE_REACH_SIGMAS * sigma_g;
    let mut breaks: Vec<f64> = (0..=15).map(|k| f64::from(k) / 15.0).collect();
    let lo = 1.0 - EDGE_REACH_SIGMAS * sigma_g;
    if lo > 0.0 {
        breaks.push(lo);
    }
    breaks.push(t_max);
    breaks.sort_by(f64::total_cmp);
    breaks.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let mut f = |t: f64| profile(rim, q, t, sigma_g) * t;
    let mut total = 0.0;
    for w in breaks.windows(2) {
        let (a, b) = (w[0], w[1]);
        let n = (((b - a) / (2.0 * sigma_g)).ceil() as usize).clamp(1, 64);
        for i in 0..n {
            let (x0, x1) = (
                a + (b - a) * i as f64 / n as f64,
                a + (b - a) * (i + 1) as f64 / n as f64,
            );
            total += gauss8(x0, x1, &mut f);
        }
    }
    total
}

/// One ghost of one source.
#[derive(Clone, Debug)]
pub(crate) struct Ghost {
    /// Index into [`Items::shapes`].
    shape: usize,
    centre: [f64; 2],
    clip_centre: [f64; 2],
    clip_radius: f64,
    /// The apothem.
    s: f64,
    /// The widening along the axis.
    stretch: f64,
    /// The edge's standard deviation in gauge units and in half-diagonals.
    sigma_g: f64,
    sigma_clip: f64,
    /// The gauge beyond which the density is 0.
    g_max: f64,
    /// Light per channel over the density's normaliser.
    amp: [f64; 3],
    /// The largest density the ghost reaches anywhere, per channel at most: the brightest channel's `amp`
    /// times the rim's peak.
    peak: f64,
    /// 1 when the ghost is drawn at every pixel, else the spacing in pixels of the grid of nodes it is drawn
    /// on ([`COARSE_LEVELS`]).
    level: usize,
    span: Span,
}

/// The spacings of the grids the slowly varying ghosts are drawn on, in pixels (even, so that a node sits on a
/// pixel corner).
pub(crate) const COARSE_LEVELS: [usize; 5] = [2, 4, 8, 16, 32];

/// How far past a tile an item can still matter to it: the nodes of the largest grid reach this far.
pub(crate) const COARSE_REACH_PX: usize = 2 * COARSE_LEVELS[4];

/// The starburst of one compact source.
#[derive(Clone, Debug)]
pub(crate) struct Burst {
    centre: [f64; 2],
    amp: [f64; 3],
    span: Span,
}

/// The streak of one source.
#[derive(Clone, Debug)]
pub(crate) struct Streak {
    centre: [f64; 2],
    /// The standard deviation across the axis.
    sigma: f64,
    amp: [f64; 3],
    span: Span,
}

/// One drawn item.
#[derive(Clone, Debug)]
pub(crate) enum Item {
    /// A ghost.
    Ghost(Ghost),
    /// A starburst.
    Burst(Burst),
    /// A streak.
    Streak(Streak),
}

impl Item {
    /// The pixels the item touches.
    pub(crate) fn span(&self) -> Span {
        match self {
            Item::Ghost(g) => g.span,
            Item::Burst(b) => b.span,
            Item::Streak(s) => s.span,
        }
    }
}

/// The profile of a streak along the axis, as a cumulative distribution (`docs/21` section 4, L3).
///
/// The profile is `(1 + |a| / core)^-power` out to `0.8 length`, then falls to 0 at `length` in a cosine. The
/// power law is integrated in closed form, the cut part by Gauss-Legendre panels, so the mass inside any
/// interval is exact (and the masses of neighbouring pixels telescope).
#[derive(Clone, Debug)]
pub(crate) struct AlongProfile {
    core: f64,
    power: f64,
    length: f64,
    flat: f64,
    /// The mass of the power law from 0 to `flat`.
    flat_mass: f64,
    /// Cumulative mass of the cut part at its panel boundaries (`CUT_PANELS + 1` values).
    cut: Vec<f64>,
    /// The mass of the whole half profile.
    half_mass: f64,
}

/// Panels of the cut part.
const CUT_PANELS: usize = 64;

/// The part of the reach where the streak's profile is still the plain power law.
const STREAK_FLAT: f64 = 0.8;

impl AlongProfile {
    /// The profile with a sharp core of width `core`, the exponent `power` and the reach `length` (all in
    /// half-diagonals).
    pub(crate) fn new(core: f64, power: f64, length: f64) -> Self {
        let flat = STREAK_FLAT * length;
        let mut p = AlongProfile {
            core,
            power,
            length,
            flat,
            flat_mass: 0.0,
            cut: Vec::new(),
            half_mass: 0.0,
        };
        p.flat_mass = p.power_law_mass(flat);
        let mut cut = Vec::with_capacity(CUT_PANELS + 1);
        cut.push(0.0);
        let mut acc = 0.0;
        let width = length - flat;
        for i in 0..CUT_PANELS {
            let (a, b) = (
                flat + width * i as f64 / CUT_PANELS as f64,
                flat + width * (i + 1) as f64 / CUT_PANELS as f64,
            );
            acc += gauss8(a, b, &mut |x| p.height(x));
            cut.push(acc);
        }
        p.half_mass = p.flat_mass + acc;
        p.cut = cut;
        p
    }

    /// The profile at distance `x >= 0` from the source.
    fn height(&self, x: f64) -> f64 {
        libm::pow(1.0 + x / self.core, -self.power) * cosine_window(x / self.length, STREAK_FLAT)
    }

    /// The closed-form mass of the plain power law from 0 to `x`.
    fn power_law_mass(&self, x: f64) -> f64 {
        let u = 1.0 + x / self.core;
        if (self.power - 1.0).abs() < 1e-9 {
            self.core * libm::log(u)
        } else {
            self.core * (1.0 - libm::pow(u, 1.0 - self.power)) / (self.power - 1.0)
        }
    }

    /// The mass of the half profile from 0 to `x >= 0`.
    fn half_cdf(&self, x: f64) -> f64 {
        if x <= self.flat {
            return self.power_law_mass(x);
        }
        if x >= self.length {
            return self.half_mass;
        }
        let width = self.length - self.flat;
        let t = (x - self.flat) / width * CUT_PANELS as f64;
        let i = (t as usize).min(CUT_PANELS - 1);
        let a = self.flat + width * i as f64 / CUT_PANELS as f64;
        let part = gauss8(a, x, &mut |y| self.height(y));
        self.flat_mass + self.cut[i] + part
    }

    /// The fraction of the whole (two-sided) mass at distances up to `a` from the source (`a` may be negative).
    pub(crate) fn cdf(&self, a: f64) -> f64 {
        let share = self.half_cdf(a.abs()) / self.half_mass;
        if a >= 0.0 {
            0.5 + 0.5 * share
        } else {
            0.5 - 0.5 * share
        }
    }

    /// The reach.
    pub(crate) fn length(&self) -> f64 {
        self.length
    }

    /// The profile's constants for the device: `core`, `power`, `length`, `flat`, the mass of the power law
    /// up to `flat`, the mass of the whole half profile, and the cumulative mass of the cut part at its
    /// panel boundaries.
    pub(crate) fn device_constants(&self) -> ([f64; 6], &[f64]) {
        (
            [
                self.core,
                self.power,
                self.length,
                self.flat,
                self.flat_mass,
                self.half_mass,
            ],
            &self.cut,
        )
    }
}

/// The tint of a coating or streak spectrum on `camera`, scaled to a mean of 1 over the channels (a flat spectrum
/// is neutral): the fraction of light a ghost has is its energy, the tint only gives it a colour.
pub(crate) fn normalised_tint(camera: &CameraResponse, spectrum: &[f64]) -> [f64; 3] {
    let mut s = [0.0; crate::consts::GRID_SAMPLES];
    for (d, v) in s.iter_mut().zip(spectrum) {
        *d = *v;
    }
    let t = camera.tint(&s);
    let sum = t[0] + t[1] + t[2];
    if sum > 0.0 {
        [3.0 * t[0] / sum, 3.0 * t[1] / sum, 3.0 * t[2] / sum]
    } else {
        [1.0; 3]
    }
}

/// Everything the items of one call share, and the items themselves.
pub(crate) struct Items<'a> {
    /// The lens's aperture and a circle.
    shapes: [Shape; 2],
    rim: RimCurve,
    along: Option<AlongProfile>,
    spikes: &'a [Vec<f64>; 3],
    spike_radius: f64,
    frame: Frame,
    tail: &'static NormalTail,
    /// True when slowly varying ghosts are drawn on grids of nodes.
    coarse: bool,
    /// The items, in list order.
    pub(crate) list: Vec<Item>,
}

/// Index of the lens's aperture in [`Items::shapes`].
const SHAPE_APERTURE: usize = 0;
/// Index of the round shape.
const SHAPE_ROUND: usize = 1;

impl<'a> Items<'a> {
    /// The items of `sources` for a frame at one scale.
    pub(crate) fn build(
        plan: &LensPlan,
        terms: &Terms,
        frame: Frame,
        sources: &[Source],
        spikes: &'a [Vec<f64>; 3],
        cull: &Span,
    ) -> Self {
        Self::build_with(plan, terms, frame, sources, spikes, cull, true)
    }

    /// [`Items::build`], with `coarse` false to draw every ghost at every pixel (the reference the coarse
    /// grids are measured against).
    pub(crate) fn build_with(
        plan: &LensPlan,
        terms: &Terms,
        frame: Frame,
        sources: &[Source],
        spikes: &'a [Vec<f64>; 3],
        cull: &Span,
        coarse: bool,
    ) -> Self {
        let (lens, att, camera) = (plan.lens(), plan.attachment(), plan.camera());
        let shapes = [Shape::aperture(&lens.flare.aperture), Shape::round()];
        let along = (terms.streak > 0.0 && terms.anam > 0.0).then(|| {
            AlongProfile::new(
                att.streak.core,
                att.streak.power,
                att.streak.reach * terms.anam,
            )
        });
        let mut items = Items {
            shapes,
            rim: RimCurve::new(&lens.rim),
            along,
            spikes,
            spike_radius: lens.flare.starburst.radius,
            frame,
            tail: normal_tail(),
            coarse,
            list: Vec::new(),
        };
        let lens_tints: Vec<[f64; 3]> = lens
            .flare
            .ghosts
            .iter()
            .map(|g| normalised_tint(camera, &g.coating))
            .collect();
        let att_tints: Vec<[f64; 3]> = att
            .ghosts
            .iter()
            .map(|g| normalised_tint(camera, &g.coating))
            .collect();
        let streak_tint = normalised_tint(camera, &att.streak.tint);
        let whole = Span {
            x0: 0,
            x1: frame.w,
            y0: 0,
            y1: frame.h,
        };
        let rim_peak = (0..=256)
            .map(|i| items.rim.at(f64::from(i) / 256.0))
            .fold(0.0f64, f64::max);
        let mut list = Vec::new();
        for src in sources {
            for (g, tint) in lens.flare.ghosts.iter().zip(&lens_tints) {
                if let Some(it) = items.ghost(
                    src,
                    g,
                    SHAPE_APERTURE,
                    terms.stretch,
                    g.fraction,
                    *tint,
                    rim_peak,
                    &whole,
                ) {
                    list.push(Item::Ghost(it));
                }
            }
            if terms.anam > 0.0 {
                for (g, tint) in att.ghosts.iter().zip(&att_tints) {
                    if let Some(it) = items.ghost(
                        src,
                        g,
                        SHAPE_ROUND,
                        terms.stretch,
                        g.fraction * terms.anam,
                        *tint,
                        rim_peak,
                        &whole,
                    ) {
                        list.push(Item::Ghost(it));
                    }
                }
            }
            if terms.burst > 0.0 && src.extent <= COMPACT_EXTENT_HD {
                list.push(Item::Burst(items.burst(src, terms.burst)));
            }
            if let Some(it) = items.streak(src, terms.streak, streak_tint) {
                list.push(Item::Streak(it));
            }
        }
        // The faintest ghosts are left out while their peaks add up to no more than the budget (a function of
        // the whole list, never of the tile, so any tiling draws the same items).
        let clip = f64::from(plan.frame().clip.iter().copied().fold(0.0f32, f32::max));
        let budget = GHOST_PRUNE_BUDGET * clip;
        let mut faint: Vec<(f64, usize)> = list
            .iter()
            .enumerate()
            .filter_map(|(i, it)| match it {
                Item::Ghost(g) => Some((g.peak, i)),
                _ => None,
            })
            .collect();
        faint.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut left_out = vec![false; list.len()];
        let mut sum = 0.0;
        for (peak, i) in faint {
            if sum + peak > budget {
                break;
            }
            sum += peak;
            left_out[i] = true;
        }
        let mut k = 0;
        list.retain(|_| {
            k += 1;
            !left_out[k - 1]
        });
        // Items within reach of the tile's nodes count as well: a node a few pixels outside the tile is the same
        // number whatever the tile is.
        let reach = Span {
            x0: cull.x0.saturating_sub(COARSE_REACH_PX),
            x1: cull.x1.saturating_add(COARSE_REACH_PX),
            y0: cull.y0.saturating_sub(COARSE_REACH_PX),
            y1: cull.y1.saturating_add(COARSE_REACH_PX),
        };
        list.retain(|it| !it.span().intersect(&reach).is_empty());
        items.list = list;
        items
    }

    /// A ghost of `src`, or `None` when nothing of it lies in `cull`.
    #[allow(clippy::too_many_arguments)]
    fn ghost(
        &self,
        src: &Source,
        g: &GhostData,
        shape: usize,
        stretch: f64,
        fraction: f64,
        tint: [f64; 3],
        rim_peak: f64,
        cull: &Span,
    ) -> Option<Ghost> {
        let h = self.frame.half_diag;
        let s = g.s;
        let var = 0.5 * (src.var_along + src.var_across);
        let sigma_edge = ((g.soft * s).powi(2) + var + (EDGE_FLOOR_PX / h).powi(2)).sqrt();
        let sigma_g = sigma_edge / s;
        let sh = &self.shapes[shape];
        let centre = [g.m * src.pos[0], g.m * src.pos[1]];
        let clip_centre = [g.n * src.pos[0], g.n * src.pos[1]];
        let g_max = 1.0 + EDGE_REACH_SIGMAS * sigma_g;
        let r_max = s * g_max * sh.reach;
        let r_clip = g.clip_radius + EDGE_REACH_SIGMAS * sigma_edge;
        let a_lo = (centre[0] - r_max * stretch).max(clip_centre[0] - r_clip);
        let a_hi = (centre[0] + r_max * stretch).min(clip_centre[0] + r_clip);
        let c_lo = (centre[1] - r_max).max(clip_centre[1] - r_clip);
        let c_hi = (centre[1] + r_max).min(clip_centre[1] + r_clip);
        if a_lo >= a_hi || c_lo >= c_hi {
            return None;
        }
        let span = self.frame.span_of(a_lo, a_hi, c_lo, c_hi);
        if span.intersect(cull).is_empty() {
            return None;
        }
        let norm = stretch * 2.0 * sh.area * s * s * profile_integral(&self.rim, sigma_g);
        if norm.is_nan() || norm <= 0.0 || !norm.is_finite() {
            return None;
        }
        let amp = [
            fraction * tint[0] * src.energy[0] / norm,
            fraction * tint[1] * src.energy[1] / norm,
            fraction * tint[2] * src.energy[2] / norm,
        ];
        let peak = amp[0].max(amp[1]).max(amp[2]) * rim_peak;
        let level = self.level_for(sigma_edge, s, &span);
        Some(Ghost {
            shape,
            centre,
            clip_centre,
            clip_radius: g.clip_radius,
            s,
            stretch,
            sigma_g,
            sigma_clip: sigma_edge,
            g_max,
            amp,
            peak,
            level,
            span,
        })
    }

    /// The grid spacing a ghost is drawn on: the largest of [`COARSE_LEVELS`] that keeps
    /// [`COARSE_NODES_PER_FEATURE`] nodes across its narrowest feature (the edge's standard deviation, and a
    /// tenth of the radius for the rim curve), or 1 (every pixel) for a ghost that is small or sharp.
    fn level_for(&self, sigma_edge: f64, s: f64, span: &Span) -> usize {
        let area = (span.x1.saturating_sub(span.x0)) * (span.y1.saturating_sub(span.y0));
        if !self.coarse || area < COARSE_MIN_AREA_PX {
            return 1;
        }
        let feature_px = sigma_edge.min(s * COARSE_RIM_FEATURE) * self.frame.half_diag;
        let nodes = feature_px / COARSE_NODES_PER_FEATURE;
        COARSE_LEVELS
            .iter()
            .rev()
            .copied()
            .find(|l| (*l as f64) <= nodes)
            .unwrap_or(1)
    }

    /// The starburst of `src` (the fraction already holds the f-number's strength).
    fn burst(&self, src: &Source, fraction: f64) -> Burst {
        let r = self.spike_radius;
        let delta = 2.0 * r / SPIKE_SIZE as f64;
        let reach = r + delta;
        let span = self.frame.span_of(
            src.pos[0] - reach,
            src.pos[0] + reach,
            src.pos[1] - reach,
            src.pos[1] + reach,
        );
        Burst {
            centre: src.pos,
            amp: [
                fraction * src.energy[0],
                fraction * src.energy[1],
                fraction * src.energy[2],
            ],
            span,
        }
    }

    /// The streak of `src`.
    fn streak(&self, src: &Source, fraction: f64, tint: [f64; 3]) -> Option<Streak> {
        let along = self.along.as_ref()?;
        if fraction <= 0.0 {
            return None;
        }
        let sigma = src.var_across.sqrt();
        let reach = EDGE_REACH_SIGMAS * sigma;
        let span = self.frame.span_of(
            src.pos[0] - along.length(),
            src.pos[0] + along.length(),
            src.pos[1] - reach,
            src.pos[1] + reach,
        );
        Some(Streak {
            centre: src.pos,
            sigma,
            amp: [
                fraction * tint[0] * src.energy[0],
                fraction * tint[1] * src.energy[1],
                fraction * tint[2] * src.energy[2],
            ],
            span,
        })
    }

    /// The streak's profile, when the plan has one.
    pub(crate) fn along(&self) -> Option<&AlongProfile> {
        self.along.as_ref()
    }

    /// The items as the device reads them: six four-lane vectors each (`l3::device`).
    pub(crate) fn device_records(&self) -> Vec<[f32; 4]> {
        let bits = |v: i32| f32::from_bits(v as u32);
        let mut out = Vec::with_capacity(6 * self.list.len());
        for item in &self.list {
            let span = item.span();
            let span_v = [
                bits(span.x0 as i32),
                bits(span.x1 as i32),
                bits(span.y0 as i32),
                bits(span.y1 as i32),
            ];
            match item {
                Item::Ghost(g) => {
                    out.push([
                        g.centre[0] as f32,
                        g.centre[1] as f32,
                        g.clip_centre[0] as f32,
                        g.clip_centre[1] as f32,
                    ]);
                    out.push([
                        g.clip_radius as f32,
                        g.s as f32,
                        g.stretch as f32,
                        g.sigma_g as f32,
                    ]);
                    out.push([
                        g.sigma_clip as f32,
                        g.g_max as f32,
                        g.amp[0] as f32,
                        g.amp[1] as f32,
                    ]);
                    out.push([g.amp[2] as f32, 0.0, 0.0, 0.0]);
                    out.push([bits(0), bits(g.shape as i32), bits(g.level as i32), 0.0]);
                }
                Item::Burst(b) => {
                    out.push([
                        b.centre[0] as f32,
                        b.centre[1] as f32,
                        b.amp[0] as f32,
                        b.amp[1] as f32,
                    ]);
                    out.push([b.amp[2] as f32, 0.0, 0.0, 0.0]);
                    out.push([0.0; 4]);
                    out.push([0.0; 4]);
                    out.push([bits(1), 0.0, 0.0, 0.0]);
                }
                Item::Streak(s) => {
                    out.push([
                        s.centre[0] as f32,
                        s.centre[1] as f32,
                        s.sigma as f32,
                        s.amp[0] as f32,
                    ]);
                    out.push([s.amp[1] as f32, s.amp[2] as f32, 0.0, 0.0]);
                    out.push([0.0; 4]);
                    out.push([0.0; 4]);
                    out.push([bits(2), 0.0, 0.0, 0.0]);
                }
            }
            out.push(span_v);
        }
        out
    }

    /// Which of the [`COARSE_LEVELS`] grids hold ghosts.
    pub(crate) fn coarse_levels(&self) -> [bool; 5] {
        let mut on = [false; 5];
        for it in &self.list {
            if let Item::Ghost(g) = it
                && let Some(i) = COARSE_LEVELS.iter().position(|l| *l == g.level)
            {
                on[i] = true;
            }
        }
        on
    }

    /// The pixel rectangle of every item, in list order.
    pub(crate) fn spans(&self) -> Vec<Span> {
        self.list.iter().map(Item::span).collect()
    }

    /// The spike radius of the starburst table, in half-diagonals.
    pub(crate) fn spike_radius(&self) -> f64 {
        self.spike_radius
    }

    /// Adds the item's light to `acc`, the pixels of `band` (row-major, `band` wide), for the pixels of the
    /// item inside `band`.
    pub(crate) fn add(&self, item: &Item, band: &Span, acc: &mut [[f64; 3]]) {
        let area = item.span().intersect(band);
        if area.is_empty() {
            return;
        }
        match item {
            // A coarse ghost is in the grids of `ghost_layer`.
            Item::Ghost(g) if g.level > 1 => {}
            Item::Ghost(g) => self.add_ghost(g, &area, band, acc),
            Item::Burst(b) => self.add_burst(b, &area, band, acc),
            Item::Streak(s) => self.add_streak(s, &area, band, acc),
        }
    }

    /// The ghost's density at the canvas position `(a, c)`: 0 outside its shape and its clipping circle.
    #[inline]
    fn ghost_density(&self, g: &Ghost, a: f64, c: f64) -> f64 {
        let shape = &self.shapes[g.shape];
        let x = (a - g.centre[0]) / g.stretch;
        let y = c - g.centre[1];
        let gauge = shape.gauge(x, y) / g.s;
        if gauge > g.g_max {
            return 0.0;
        }
        let prof = profile(&self.rim, self.tail, gauge, g.sigma_g);
        let (dx, dy) = (a - g.clip_centre[0], c - g.clip_centre[1]);
        let d = (dx * dx + dy * dy).sqrt();
        let dens = prof * self.tail.q((d - g.clip_radius) / g.sigma_clip);
        dens.max(0.0)
    }

    fn add_ghost(&self, g: &Ghost, area: &Span, band: &Span, acc: &mut [[f64; 3]]) {
        let frame = &self.frame;
        let bw = band.x1 - band.x0;
        for j in area.y0..area.y1 {
            let row = (j - band.y0) * bw;
            for i in area.x0..area.x1 {
                let (a, c) = frame.canvas_at(i as f64 + 0.5, j as f64 + 0.5);
                let dens = self.ghost_density(g, a, c);
                if dens > 0.0 {
                    let px = &mut acc[row + i - band.x0];
                    px[0] += g.amp[0] * dens;
                    px[1] += g.amp[1] * dens;
                    px[2] += g.amp[2] * dens;
                }
            }
        }
    }

    /// The grids of nodes the coarse ghosts are drawn on, over the nodes the pixels of `area` read.
    ///
    /// The node `(j, k)` of the grid of spacing `L` sits at the pixel position `(L j + L / 2, L k + L / 2)`
    /// (a corner between pixels) and holds the sum, in list order, of the densities of the grid's ghosts
    /// whose pixel rectangle grown by `L` holds it. A node is a function of its place and of the list, so any
    /// tiling reads the same numbers.
    pub(crate) fn ghost_layer(&self, area: &Span) -> GhostLayer {
        let mut grids = Vec::new();
        if area.is_empty() {
            return GhostLayer { grids };
        }
        for &l in &COARSE_LEVELS {
            let ghosts: Vec<&Ghost> = self
                .list
                .iter()
                .filter_map(|it| match it {
                    Item::Ghost(g) if g.level == l => Some(g),
                    _ => None,
                })
                .collect();
            if ghosts.is_empty() {
                continue;
            }
            let li = l as i64;
            let (j0, j1) = node_range(area.x0, area.x1, li);
            let (k0, k1) = node_range(area.y0, area.y1, li);
            let (nx, ny) = ((j1 - j0 + 1) as usize, (k1 - k0 + 1) as usize);
            let mut data = vec![[0.0f64; 3]; nx * ny];
            data.par_chunks_mut(nx).enumerate().for_each(|(row, out)| {
                let k = k0 + row as i64;
                let py = li * k + li / 2;
                for g in &ghosts {
                    let (sy0, sy1) = (g.span.y0 as i64 - li, g.span.y1 as i64 + li);
                    if py < sy0 || py >= sy1 {
                        continue;
                    }
                    let (sx0, sx1) = (g.span.x0 as i64 - li, g.span.x1 as i64 + li);
                    for (col, node) in out.iter_mut().enumerate() {
                        let px = li * (j0 + col as i64) + li / 2;
                        if px < sx0 || px >= sx1 {
                            continue;
                        }
                        let (a, c) = self.frame.canvas_at(px as f64, py as f64);
                        let dens = self.ghost_density(g, a, c);
                        if dens > 0.0 {
                            node[0] += g.amp[0] * dens;
                            node[1] += g.amp[1] * dens;
                            node[2] += g.amp[2] * dens;
                        }
                    }
                }
            });
            grids.push(NodeGrid {
                l: li,
                j0,
                k0,
                nx,
                ny,
                data,
            });
        }
        GhostLayer { grids }
    }

    fn add_streak(&self, s: &Streak, area: &Span, band: &Span, acc: &mut [[f64; 3]]) {
        let Some(along) = self.along.as_ref() else {
            return;
        };
        let frame = &self.frame;
        let (p0, p1) = frame.along_range(area);
        let (q0, q1) = frame.across_range(area);
        let mass_along: Vec<f64> = (p0..p1)
            .map(|p| {
                let (e0, e1) = frame.along_edges(p);
                along.cdf(e1 - s.centre[0]) - along.cdf(e0 - s.centre[0])
            })
            .collect();
        let mass_across: Vec<f64> = (q0..q1)
            .map(|q| {
                let (e0, e1) = frame.across_edges(q);
                self.tail.q((e0 - s.centre[1]) / s.sigma)
                    - self.tail.q((e1 - s.centre[1]) / s.sigma)
            })
            .collect();
        let per_area = frame.half_diag * frame.half_diag;
        let bw = band.x1 - band.x0;
        for (qi, mc) in mass_across.iter().enumerate() {
            if *mc <= 0.0 {
                continue;
            }
            for (pi, ma) in mass_along.iter().enumerate() {
                let (i, j) = frame.pixel_of(p0 + pi, q0 + qi);
                let m = ma * mc * per_area;
                let px = &mut acc[(j - band.y0) * bw + i - band.x0];
                px[0] += s.amp[0] * m;
                px[1] += s.amp[1] * m;
                px[2] += s.amp[2] * m;
            }
        }
    }

    fn add_burst(&self, b: &Burst, area: &Span, band: &Span, acc: &mut [[f64; 3]]) {
        let frame = &self.frame;
        let n = SPIKE_SIZE;
        let r = self.spike_radius;
        let delta = 2.0 * r / n as f64;
        let (p0, p1) = frame.along_range(area);
        let (q0, q1) = frame.across_range(area);
        // Tent weights of a pixel's interval over the table's cells along one direction.
        let weights = |e0: f64, e1: f64, centre: f64| -> (usize, Vec<f64>) {
            let x0 = (e0 - centre + r) / delta;
            let x1 = (e1 - centre + r) / delta;
            let lo = (libm::floor(x0 - 1.5) as i64 + 1).clamp(0, n as i64);
            let hi = (libm::ceil(x1 + 0.5) as i64).clamp(0, n as i64);
            let w: Vec<f64> = (lo..hi)
                .map(|m| {
                    let c = m as f64 + 0.5;
                    tent_integral(x1 - c) - tent_integral(x0 - c)
                })
                .collect();
            (lo as usize, w)
        };
        let along_w: Vec<(usize, Vec<f64>)> = (p0..p1)
            .map(|p| {
                let (e0, e1) = frame.along_edges(p);
                weights(e0, e1, b.centre[0])
            })
            .collect();
        let per_area = frame.half_diag * frame.half_diag;
        let bw = band.x1 - band.x0;
        let mut v: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; n]);
        for q in q0..q1 {
            let (e0, e1) = frame.across_edges(q);
            let (n0, wn) = weights(e0, e1, b.centre[1]);
            if wn.iter().all(|w| *w == 0.0) {
                continue;
            }
            // The table collapsed over the pixel's rows (across the axis).
            for (t, out) in self.spikes.iter().zip(v.iter_mut()) {
                out.fill(0.0);
                for (k, w) in wn.iter().enumerate() {
                    let row = &t[(n0 + k) * n..(n0 + k + 1) * n];
                    for (o, s) in out.iter_mut().zip(row) {
                        *o += w * s;
                    }
                }
            }
            for (pi, (m0, wm)) in along_w.iter().enumerate() {
                let mut mass = [0.0f64; 3];
                for (k, w) in wm.iter().enumerate() {
                    for (m, vc) in mass.iter_mut().zip(&v) {
                        *m += w * vc[m0 + k];
                    }
                }
                let (i, j) = frame.pixel_of(p0 + pi, q);
                let px = &mut acc[(j - band.y0) * bw + i - band.x0];
                for ((p, a), m) in px.iter_mut().zip(&b.amp).zip(&mass) {
                    *p += a * m * per_area;
                }
            }
        }
    }
}

/// The first and last node index (inclusive) of the grid of spacing `l` that the pixels `[p0, p1)` read: the pixel
/// `p` lies between the nodes `floor((2 p + 1 - l) / 2 l)` and the next.
fn node_range(p0: usize, p1: usize, l: i64) -> (i64, i64) {
    let first = (2 * p0 as i64 + 1 - l).div_euclid(2 * l);
    let last = (2 * (p1 as i64 - 1) + 1 - l).div_euclid(2 * l) + 1;
    (first, last)
}

/// One grid of nodes (see [`Items::ghost_layer`]).
struct NodeGrid {
    /// The spacing in pixels.
    l: i64,
    /// The index of the first node column and row.
    j0: i64,
    k0: i64,
    nx: usize,
    ny: usize,
    data: Vec<[f64; 3]>,
}

/// The coarse ghosts over a rectangle: their grids, read at every pixel by bilinear interpolation.
pub(crate) struct GhostLayer {
    grids: Vec<NodeGrid>,
}

impl GhostLayer {
    /// Adds the grids' light to `acc`, the pixels of `band` (row-major, `band` wide), grid by grid.
    pub(crate) fn add_band(&self, band: &Span, acc: &mut [[f64; 3]]) {
        let bw = band.x1 - band.x0;
        for grid in &self.grids {
            let l = grid.l;
            let cols: Vec<(usize, f64)> =
                (band.x0..band.x1).map(|x| split(x, l, grid.j0)).collect();
            for y in band.y0..band.y1 {
                let (k, ty) = split(y, l, grid.k0);
                debug_assert!(k + 1 < grid.ny, "the grid holds the nodes of the row");
                let row = (y - band.y0) * bw;
                let (r0, r1) = (k * grid.nx, (k + 1) * grid.nx);
                for (c, (j, tx)) in cols.iter().enumerate() {
                    let at = |r: usize, d: usize, ch: usize| grid.data[r + j + d][ch];
                    let px = &mut acc[row + c];
                    for ch in 0..3 {
                        let top = at(r0, 0, ch) + tx * (at(r0, 1, ch) - at(r0, 0, ch));
                        let bot = at(r1, 0, ch) + tx * (at(r1, 1, ch) - at(r1, 0, ch));
                        px[ch] += top + ty * (bot - top);
                    }
                }
            }
        }
    }
}

/// The node (as an index into the grid starting at `first`) left of the pixel `p`, and the fraction of the way
/// to the next one, for spacing `l`.
#[inline]
fn split(p: usize, l: i64, first: i64) -> (usize, f64) {
    let num = 2 * p as i64 + 1 - l;
    let node = num.div_euclid(2 * l);
    let frac = (num - 2 * l * node) as f64 / (2 * l) as f64;
    ((node - first) as usize, frac)
}

/// The integral of the tent `max(0, 1 - |t|)` from minus infinity to `t`.
#[inline]
fn tent_integral(t: f64) -> f64 {
    if t <= -1.0 {
        0.0
    } else if t <= 0.0 {
        0.5 * (t + 1.0) * (t + 1.0)
    } else if t <= 1.0 {
        1.0 - 0.5 * (1.0 - t) * (1.0 - t)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Axis;

    #[test]
    fn a_round_shape_has_the_area_of_a_circle_and_a_polygon_the_area_of_its_sectors() {
        assert!((Shape::round().area() - std::f64::consts::PI).abs() < 1e-15);
        let hex = Shape::aperture(&Aperture {
            blades: 6,
            rotation_deg: 0.0,
            roundness: 0.0,
        });
        // A hexagon of apothem 1: 6 tan(30 degrees).
        let want = 6.0 * (std::f64::consts::PI / 6.0).tan();
        assert!(
            (hex.area() - want).abs() < 1e-12,
            "{} vs {want}",
            hex.area()
        );
        assert!((hex.reach - 1.0 / (std::f64::consts::PI / 6.0).cos()).abs() < 1e-12);
        // Fully rounded is a circle again.
        let round = Shape::aperture(&Aperture {
            blades: 9,
            rotation_deg: 17.0,
            roundness: 1.0,
        });
        assert!((round.area() - std::f64::consts::PI).abs() < 1e-12);
        // The gauge is 1 on the hexagon's edge and scales linearly.
        assert!((hex.gauge(1.0, 0.0) - 1.0).abs() < 1e-15);
        assert!((hex.gauge(3.0, 0.0) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn the_profile_integral_matches_a_brute_force_sum() {
        let rim: Vec<f64> = (0..16)
            .map(|i| 0.8 + 0.05 * f64::from(i) * f64::from(i) / 15.0)
            .collect();
        let lut = RimCurve::new(&rim);
        for sigma in [0.004, 0.05, 0.2, 0.5] {
            let got = profile_integral(&lut, sigma);
            let q = normal_tail();
            let n = 400_000;
            let t_max = 1.0 + 8.0 * sigma;
            let mut sum = 0.0;
            for i in 0..n {
                let t = (f64::from(i) + 0.5) / f64::from(n) * t_max;
                sum += profile(&lut, q, t, sigma) * t;
            }
            sum *= t_max / f64::from(n);
            assert!(
                (got - sum).abs() < 1e-6 * sum,
                "sigma {sigma}: {got} vs {sum}"
            );
        }
    }

    #[test]
    fn the_streak_profile_is_a_distribution() {
        let p = AlongProfile::new(0.002, 1.5, 0.5);
        assert!((p.cdf(0.0) - 0.5).abs() < 1e-15);
        assert_eq!(p.cdf(-0.5), 0.0);
        assert!((p.cdf(0.5) - 1.0).abs() < 1e-15);
        assert_eq!(p.cdf(10.0), 1.0);
        let mut prev = 0.0;
        for i in 0..=2000 {
            let a = -0.6 + 1.2 * f64::from(i) / 2000.0;
            let c = p.cdf(a);
            assert!(c >= prev - 1e-15, "not monotone at {a}");
            prev = c;
        }
        // Symmetric.
        assert!((p.cdf(0.1) + p.cdf(-0.1) - 1.0).abs() < 1e-15);
        // A numerical check of the cut part against midpoint integration.
        let h = |x: f64| libm::pow(1.0 + x / 0.002, -1.5) * cosine_window(x / 0.5, STREAK_FLAT);
        let n = 2_000_000;
        let mut s = 0.0;
        for i in 0..n {
            s += h((f64::from(i) + 0.5) / f64::from(n) * 0.5);
        }
        s *= 0.5 / f64::from(n);
        assert!(
            (p.cdf(0.3)
                - (0.5
                    + 0.5 * {
                        let mut part = 0.0;
                        for i in 0..n {
                            part += h((f64::from(i) + 0.5) / f64::from(n) * 0.3);
                        }
                        part * 0.3 / f64::from(n) / s
                    }))
            .abs()
                < 1e-6
        );
        // The frame helpers agree with the axis.
        let _ = Frame::new(10, 10, Axis::X);
    }

    /// Items for a frame, with a lens aperture and a flat rim (for the raster tests).
    fn bare_items<'a>(frame: Frame, spikes: &'a [Vec<f64>; 3], rim: &[f64]) -> Items<'a> {
        Items {
            shapes: [Shape::round(), Shape::round()],
            rim: RimCurve::new(rim),
            along: None,
            spikes,
            spike_radius: 0.06,
            frame,
            tail: normal_tail(),
            coarse: false,
            list: Vec::new(),
        }
    }

    #[test]
    fn a_round_ghosts_raster_sums_to_its_light() {
        let (w, h) = (480usize, 320usize);
        let frame = Frame::new(w, h, Axis::X);
        let spikes: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        let rims: [Vec<f64>; 2] = [
            vec![1.0; 16],
            (0..16)
                .map(|i| 0.8 + 0.4 * (f64::from(i) / 15.0).powi(4))
                .collect(),
        ];
        let mut worst = 0.0f64;
        for rim in &rims {
            let items = bare_items(frame, &spikes, rim);
            for (s, soft) in [(0.03, 0.1), (0.05, 0.1), (0.08, 0.02), (0.03, 0.5)] {
                for (dx, dy) in [(0.0, 0.0), (0.37, 0.61), (0.5, 0.5)] {
                    let src = Source {
                        pos: [frame.u(300.0 + dx), frame.v(200.0 + dy)],
                        energy: [2.0, 1.0, 0.5],
                        var_along: 0.75 / (frame.half_diag * frame.half_diag),
                        var_across: 0.75 / (frame.half_diag * frame.half_diag),
                        extent: 0.01,
                    };
                    let g = GhostData {
                        m: -0.5,
                        n: 0.0,
                        s,
                        clip_radius: 3.0,
                        fraction: 0.1,
                        soft,
                        coating: vec![1.0; 31],
                    };
                    let cull = Span {
                        x0: 0,
                        x1: w,
                        y0: 0,
                        y1: h,
                    };
                    let ghost = items
                        .ghost(&src, &g, SHAPE_ROUND, 1.0, 0.1, [1.0; 3], 1.2, &cull)
                        .unwrap();
                    let mut acc = vec![[0.0f64; 3]; w * h];
                    items.add_ghost(&ghost, &cull, &cull, &mut acc);
                    let total: f64 = acc.iter().map(|p| p[0]).sum();
                    // fraction * tint * sum of H over pixels = fraction * E * H^2
                    let want = 0.1 * 2.0 * frame.half_diag * frame.half_diag;
                    let rel = (total / want - 1.0).abs();
                    worst = worst.max(rel);
                    // A very soft edge (the edge's deviation is half the ghost's radius) puts a cone at the
                    // ghost's centre, which costs a few parts in a million.
                    let allowed = if soft > 0.3 { 1e-5 } else { 1e-6 };
                    assert!(
                        rel < allowed,
                        "rim {:?} s {s} soft {soft} offset ({dx}, {dy}): {rel:e}",
                        rim[15]
                    );
                }
            }
        }
        eprintln!("worst relative error of a ghost's raster: {worst:e}");
    }

    #[test]
    fn a_polygon_ghosts_raster_sums_to_its_light_closely() {
        // The corners of a polygon aperture are creases of the gauge that a pixel grid samples with a small
        // error; the rounder the blades the smaller it is.
        let (w, h) = (480usize, 320usize);
        let frame = Frame::new(w, h, Axis::X);
        let spikes: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        let rim: Vec<f64> = (0..16)
            .map(|i| 0.8 + 0.4 * (f64::from(i) / 15.0).powi(4))
            .collect();
        let mut worst = [0.0f64; 3];
        for (k, roundness) in [0.0, 0.5, 0.8].into_iter().enumerate() {
            let mut items = bare_items(frame, &spikes, &rim);
            items.shapes[SHAPE_APERTURE] = Shape::aperture(&Aperture {
                blades: 6,
                rotation_deg: 10.0,
                roundness,
            });
            for (dx, dy) in [(0.0, 0.0), (0.37, 0.61)] {
                let src = Source {
                    pos: [frame.u(300.0 + dx), frame.v(200.0 + dy)],
                    energy: [1.0, 1.0, 1.0],
                    var_along: 0.75 / (frame.half_diag * frame.half_diag),
                    var_across: 0.75 / (frame.half_diag * frame.half_diag),
                    extent: 0.01,
                };
                let g = GhostData {
                    m: -0.5,
                    n: 0.0,
                    s: 0.05,
                    clip_radius: 3.0,
                    fraction: 0.1,
                    soft: 0.1,
                    coating: vec![1.0; 31],
                };
                let cull = Span {
                    x0: 0,
                    x1: w,
                    y0: 0,
                    y1: h,
                };
                let ghost = items
                    .ghost(&src, &g, SHAPE_APERTURE, 1.0, 0.1, [1.0; 3], 1.2, &cull)
                    .unwrap();
                let mut acc = vec![[0.0f64; 3]; w * h];
                items.add_ghost(&ghost, &cull, &cull, &mut acc);
                let total: f64 = acc.iter().map(|p| p[0]).sum();
                let want = 0.1 * frame.half_diag * frame.half_diag;
                worst[k] = worst[k].max((total / want - 1.0).abs());
            }
        }
        eprintln!("polygon ghosts, roundness 0 / 0.5 / 0.8: {worst:?}");
        assert!(worst.iter().all(|e| *e < 1e-3), "{worst:?}");
    }

    #[test]
    fn tint_is_neutral_for_a_flat_coating() {
        let cam = CameraResponse::neutral();
        let t = normalised_tint(&cam, &[0.3; 31]);
        for v in t {
            assert!((v - 1.0).abs() < 1e-15);
        }
        let blue: Vec<f64> = (0..31).map(|i| if i < 10 { 1.0 } else { 0.1 }).collect();
        let t = normalised_tint(&cam, &blue);
        assert!((t[0] + t[1] + t[2] - 3.0).abs() < 1e-12);
    }
}
