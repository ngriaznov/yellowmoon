//! The Reference Grade node's own colour steps (`docs/34-reference-grade.md` sections 4 and 4a): the **hue curves** and the
//! **element grades**, in f64 on the CPU reference path.
//!
//! * **Hue curves** ([`HueStep`]): `hue_vs_hue` (degrees the hue turns) and `hue_vs_sat` (a chroma multiplier) as functions
//!   of the hue on the LogC4 chroma-plane axis of the parametric look (`p = v . E1`, `q = v . E2`, hue `atan2(q, p)`). Each
//!   curve is a periodic Fritsch-Carlson spline through at most 16 points, **baked to a table of [`HUE_TABLE_LEN`] samples
//!   around the circle that is the definition**: the CPU reads it in f64, the device reads the same table as f32, so the
//!   baking error is not a parity term ([`HueTable::max_error_against_exact`] measures it against the exact spline). The step
//!   scales and rotates `(p, q)`, so the channel mean is unchanged, and **then restores the Rec.709 luma of the input** on the
//!   LogC4 codes, as the parametric look does (`ParametricLook::evaluate`; the owner's answer to question 48, 2026-11-25).
//! * **Element grades** ([`ElementStep`]): per element (sky, vegetation, water, ground, built, skin and the subject) an
//!   exposure offset, a hue rotation and a chroma ratio, each applied through the element's soft mask. No luma restoration.

use crate::consts::{REC709_LUMA_WEIGHTS, Vec3};
use crate::curve::{fritsch_carlson_tangents_periodic, hermite};
use crate::grade::{E1, E2};

/// The number of samples of a baked hue curve: one per `360 / 1024` degrees, the sample `i` at hue `360 i / 1024`.
pub const HUE_TABLE_LEN: usize = 1024;
/// The most points a hue curve has.
pub const MAX_HUE_POINTS: usize = 16;
/// One stop in LogC4 code (`docs/34` section 4a.1, the POC's `V5_STOP_CODE`): an element's `exposure_ev` adds
/// `exposure_ev * ELEMENT_STOP_CODE` to the three codes.
pub const ELEMENT_STOP_CODE: f64 = 0.0648;
/// The number of element grades: sky, vegetation, water, ground, built, skin and the subject.
pub const ELEMENT_COUNT: usize = 7;
/// The index of the subject among the elements.
pub const SUBJECT: usize = 6;

/// A periodic Fritsch-Carlson spline over the hue circle, the exact curve a [`HueTable`] is baked from.
#[derive(Debug, Clone, PartialEq)]
struct PeriodicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    tangents: Vec<f64>,
}

impl PeriodicSpline {
    /// A spline through `points` (hue in [0, 360), strictly increasing, at least one). One point is a constant.
    fn new(points: &[(f64, f64)]) -> Self {
        let mut xs: Vec<f64> = points.iter().map(|p| p.0).collect();
        let mut ys: Vec<f64> = points.iter().map(|p| p.1).collect();
        if xs.len() < 2 {
            return Self {
                tangents: vec![0.0; xs.len()],
                xs,
                ys,
            };
        }
        let mut tangents = fritsch_carlson_tangents_periodic(&xs, &ys, 360.0);
        // The closing segment runs to a copy of the first point a period later, with the first point's own tangent.
        xs.push(xs[0] + 360.0);
        ys.push(ys[0]);
        tangents.push(tangents[0]);
        Self { xs, ys, tangents }
    }

    /// The exact value at hue `hue_deg` (any finite number, read modulo 360).
    fn eval(&self, hue_deg: f64) -> f64 {
        let n = self.xs.len();
        if n == 1 {
            return self.ys[0];
        }
        let start = self.xs[0];
        let x = start + (hue_deg - start).rem_euclid(360.0);
        if x >= self.xs[n - 1] {
            return self.ys[n - 1];
        }
        let k = self
            .xs
            .partition_point(|&v| v <= x)
            .saturating_sub(1)
            .min(n - 2);
        hermite(
            self.xs[k],
            self.ys[k],
            self.tangents[k],
            self.xs[k + 1],
            self.ys[k + 1],
            self.tangents[k + 1],
            x,
        )
    }
}

/// The exact value of the periodic spline through `points` (`(hue_deg, value)`, hues in [0, 360) and increasing, at most
/// [`MAX_HUE_POINTS`]) at `hue_deg`. An empty list gives `neutral`. The reference [`HueTable`] is measured against.
#[must_use]
pub fn exact_value(points: &[(f64, f64)], hue_deg: f64, neutral: f64) -> f64 {
    if points.is_empty() {
        neutral
    } else {
        PeriodicSpline::new(points).eval(hue_deg)
    }
}

/// A hue curve baked to [`HUE_TABLE_LEN`] samples around the circle (the definition of the curve, `docs/34` section 4.2).
#[derive(Debug, Clone, PartialEq)]
pub struct HueTable {
    values: Vec<f64>,
}

impl HueTable {
    /// Bakes the periodic spline through `points`; `None` for an empty list (the identity).
    #[must_use]
    pub fn bake(points: &[(f64, f64)]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let spline = PeriodicSpline::new(points);
        let values = (0..HUE_TABLE_LEN)
            .map(|i| spline.eval(360.0 * i as f64 / HUE_TABLE_LEN as f64))
            .collect();
        Some(Self { values })
    }

    /// The samples, sample `i` at hue `360 i / 1024` degrees.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The samples as f32, the form the device reads.
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        self.values.iter().map(|&v| v as f32).collect()
    }

    /// Whether every sample is `neutral` (0 for `hue_vs_hue`, 1 for `hue_vs_sat`): the curve does nothing.
    #[must_use]
    pub fn is_neutral(&self, neutral: f64) -> bool {
        self.values.iter().all(|&v| v == neutral)
    }

    /// The value at `hue_deg` by linear interpolation between samples, wrapping round the circle.
    #[must_use]
    pub fn lookup(&self, hue_deg: f64) -> f64 {
        let t = hue_deg.rem_euclid(360.0) / 360.0 * HUE_TABLE_LEN as f64;
        let floor = t.floor();
        let frac = t - floor;
        // `rem_euclid` can round a tiny negative hue up to exactly 360; the index then wraps to 0.
        let i0 = (floor as usize) % HUE_TABLE_LEN;
        let i1 = (i0 + 1) % HUE_TABLE_LEN;
        self.values[i0] + frac * (self.values[i1] - self.values[i0])
    }

    /// The largest `|table - exact spline|` over `samples` hues spread evenly round the circle (the points the table does
    /// not sample are the ones that show the interpolation's error).
    #[must_use]
    pub fn max_error_against_exact(points: &[(f64, f64)], samples: usize) -> f64 {
        let Some(table) = Self::bake(points) else {
            return 0.0;
        };
        let spline = PeriodicSpline::new(points);
        (0..samples)
            .map(|i| {
                let h = 360.0 * (i as f64 + 0.37) / samples as f64;
                (table.lookup(h) - spline.eval(h)).abs()
            })
            .fold(0.0, f64::max)
    }
}

/// The hue step (`docs/34` section 4.2): both curves, each `None` when it does nothing.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HueStep {
    hue_vs_hue: Option<HueTable>,
    hue_vs_sat: Option<HueTable>,
}

impl HueStep {
    /// The step of two point lists (`(hue_deg, value)` with hues in [0, 360) and increasing; `hue_vs_hue` in degrees,
    /// `hue_vs_sat` a chroma multiplier). An empty list, or a curve whose every sample is neutral, is left out.
    #[must_use]
    pub fn new(hue_vs_hue: &[(f64, f64)], hue_vs_sat: &[(f64, f64)]) -> Self {
        Self {
            hue_vs_hue: HueTable::bake(hue_vs_hue).filter(|t| !t.is_neutral(0.0)),
            hue_vs_sat: HueTable::bake(hue_vs_sat).filter(|t| !t.is_neutral(1.0)),
        }
    }

    /// Whether the step does nothing: no curve is live.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.hue_vs_hue.is_none() && self.hue_vs_sat.is_none()
    }

    /// The `hue_vs_hue` table, when that curve is live.
    #[must_use]
    pub fn hue_vs_hue(&self) -> Option<&HueTable> {
        self.hue_vs_hue.as_ref()
    }

    /// The `hue_vs_sat` table, when that curve is live.
    #[must_use]
    pub fn hue_vs_sat(&self) -> Option<&HueTable> {
        self.hue_vs_sat.as_ref()
    }

    /// The step on a LogC4 triple: `(p, q)` is scaled by `hue_vs_sat(h)` and rotated by `hue_vs_hue(h)`, the channel mean
    /// stays, and the Rec.709 luma of `v` is then restored with one shift of the three codes. A pixel with no chroma is
    /// returned as it is.
    #[must_use]
    pub fn apply(&self, v: Vec3) -> Vec3 {
        if self.is_identity() {
            return v;
        }
        let p = v[0] * E1[0] + v[1] * E1[1] + v[2] * E1[2];
        let q = v[0] * E2[0] + v[1] * E2[1] + v[2] * E2[2];
        let r = p.hypot(q);
        if r == 0.0 || !r.is_finite() {
            return v;
        }
        let angle = q.atan2(p);
        let hue_deg = angle.to_degrees();
        let sigma = self.hue_vs_sat.as_ref().map_or(1.0, |t| t.lookup(hue_deg));
        let delta = self
            .hue_vs_hue
            .as_ref()
            .map_or(0.0, |t| t.lookup(hue_deg).to_radians());
        let turned = angle + delta;
        let (p2, q2) = (r * sigma * turned.cos(), r * sigma * turned.sin());
        let (dp, dq) = (p2 - p, q2 - q);
        let mut out = [
            v[0] + dp * E1[0] + dq * E2[0],
            v[1] + dp * E1[1] + dq * E2[1],
            v[2] + dp * E1[2] + dq * E2[2],
        ];
        let w = REC709_LUMA_WEIGHTS;
        let fix = (w[0] * v[0] + w[1] * v[1] + w[2] * v[2])
            - (w[0] * out[0] + w[1] * out[1] + w[2] * out[2]);
        for c in &mut out {
            *c += fix;
        }
        out
    }
}

/// One element's grade in effective values (`docs/34` section 4a.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementGrade {
    /// The exposure offset, in stops.
    pub exposure_ev: f64,
    /// The hue turn, in degrees.
    pub hue_deg: f64,
    /// The chroma ratio (1 is unchanged).
    pub chroma: f64,
}

impl Default for ElementGrade {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ElementGrade {
    /// No exposure change, no hue turn, chroma ratio 1.
    pub const IDENTITY: Self = Self {
        exposure_ev: 0.0,
        hue_deg: 0.0,
        chroma: 1.0,
    };

    /// Whether the grade does nothing.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.exposure_ev == 0.0 && self.hue_deg == 0.0 && self.chroma == 1.0
    }

    /// `local(v)` of `docs/34` section 4a.2: the mean moves by the exposure offset alone, `(p, q)` is turned by `hue_deg` and
    /// scaled by `chroma`. With no turn and no scaling (the subject) it is the exposure shift alone, with no round trip
    /// through the plane.
    #[must_use]
    pub fn local(&self, v: Vec3) -> Vec3 {
        let shift = self.exposure_ev * ELEMENT_STOP_CODE;
        if self.hue_deg == 0.0 && self.chroma == 1.0 {
            return [v[0] + shift, v[1] + shift, v[2] + shift];
        }
        let m = (v[0] + v[1] + v[2]) / 3.0;
        let p = v[0] * E1[0] + v[1] * E1[1] + v[2] * E1[2];
        let q = v[0] * E2[0] + v[1] * E2[1] + v[2] * E2[2];
        let (sin, cos) = self.hue_deg.to_radians().sin_cos();
        let p2 = self.chroma * (cos * p - sin * q);
        let q2 = self.chroma * (sin * p + cos * q);
        let mean = m + shift;
        [
            mean + p2 * E1[0] + q2 * E2[0],
            mean + p2 * E1[1] + q2 * E2[1],
            mean + p2 * E1[2] + q2 * E2[2],
        ]
    }
}

/// The seven element grades (`docs/34` section 4a): sky, vegetation, water, ground, built, skin, subject.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ElementStep {
    grades: [ElementGrade; ELEMENT_COUNT],
}

impl ElementStep {
    /// A step of the seven effective grades.
    #[must_use]
    pub fn new(grades: [ElementGrade; ELEMENT_COUNT]) -> Self {
        Self { grades }
    }

    /// The grades, in the order of `docs/34` section 2.3.
    #[must_use]
    pub fn grades(&self) -> &[ElementGrade; ELEMENT_COUNT] {
        &self.grades
    }

    /// Whether every grade does nothing.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.grades.iter().all(ElementGrade::is_identity)
    }

    /// Whether element `e` is live: its grade is not the identity (an element at the identity makes no mask).
    #[must_use]
    pub fn is_live(&self, e: usize) -> bool {
        self.grades.get(e).is_some_and(|g| !g.is_identity())
    }

    /// The step on a LogC4 triple: for each element in order, `w <- w + a_e (local_e(w) - w)` with `a_e` the element's soft
    /// mask at the pixel (`masks[e]`, 0 to 1). An element at the identity, or with a mask of 0 or below, is skipped.
    #[must_use]
    pub fn apply(&self, v: Vec3, masks: &[f64; ELEMENT_COUNT]) -> Vec3 {
        let mut w = v;
        for (g, &a) in self.grades.iter().zip(masks) {
            if g.is_identity() || a <= 0.0 {
                continue;
            }
            let l = g.local(w);
            w = [
                w[0] + a * (l[0] - w[0]),
                w[1] + a * (l[1] - w[1]),
                w[2] + a * (l[2] - w[2]),
            ];
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: Vec3 = REC709_LUMA_WEIGHTS;

    fn luma(v: Vec3) -> f64 {
        W[0] * v[0] + W[1] * v[1] + W[2] * v[2]
    }

    fn mean(v: Vec3) -> f64 {
        (v[0] + v[1] + v[2]) / 3.0
    }

    fn plane(v: Vec3) -> (f64, f64) {
        (
            v[0] * E1[0] + v[1] * E1[1] + v[2] * E1[2],
            v[0] * E2[0] + v[1] * E2[1] + v[2] * E2[2],
        )
    }

    fn close(a: Vec3, b: Vec3, tol: f64) {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() <= tol, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn one_point_is_a_constant_and_the_curve_is_periodic() {
        let c = [(120.0, 7.5)];
        for h in [0.0, 90.0, 120.0, 359.9, -10.0, 725.0] {
            assert_eq!(exact_value(&c, h, 0.0), 7.5);
        }
        let t = HueTable::bake(&c).unwrap();
        assert!(t.values().iter().all(|&v| v == 7.5));
        let pts = [(10.0, -20.0), (130.0, 30.0), (250.0, 5.0)];
        // 0 and 360 give one value; a fine sweep through the seam has no step.
        assert_eq!(exact_value(&pts, 0.0, 0.0), exact_value(&pts, 360.0, 0.0));
        let table = HueTable::bake(&pts).unwrap();
        assert_eq!(table.lookup(0.0), table.lookup(360.0));
        let mut worst = 0.0f64;
        let mut prev = table.lookup(340.0);
        for i in 1..=4000 {
            let h = 340.0 + 40.0 * f64::from(i) / 4000.0;
            let cur = table.lookup(h);
            worst = worst.max((cur - prev).abs());
            prev = cur;
        }
        // The curve spans 50 degrees over 360 degrees; a step of 0.01 degrees moves it by far less than 5e-3.
        assert!(worst < 5e-3, "{worst}");
        // The exact spline passes through its points and never overshoots them.
        for &(h, v) in &pts {
            assert!((exact_value(&pts, h, 0.0) - v).abs() < 1e-12);
        }
        for i in 0..3600 {
            let e = exact_value(&pts, f64::from(i) / 10.0, 0.0);
            assert!((-20.0 - 1e-9..=30.0 + 1e-9).contains(&e), "{e}");
        }
    }

    #[test]
    fn the_table_matches_the_exact_spline_closely() {
        let curves: [&[(f64, f64)]; 3] = [
            &[(10.0, -20.0), (130.0, 30.0), (250.0, 5.0)],
            &[(0.0, 1.0), (60.0, 2.5), (200.0, 0.2), (330.0, 1.4)],
            &[
                (0.0, -12.0),
                (60.0, 9.0),
                (120.0, -4.0),
                (180.0, 15.0),
                (240.0, -7.0),
                (300.0, 3.0),
            ],
        ];
        for pts in curves {
            let err = HueTable::max_error_against_exact(pts, 100_000);
            let span = pts.iter().map(|p| p.1).fold(f64::MIN, f64::max)
                - pts.iter().map(|p| p.1).fold(f64::MAX, f64::min);
            // Linear interpolation of a C1 cubic at 360/1024 degrees: the error is a small share of the value span.
            assert!(err <= 1e-3 * span.max(1.0), "{err} for {pts:?}");
        }
    }

    /// The error of the baked table against the exact spline (RG-02's gate: "the table against the exact spline, error stated").
    /// Fitted-style curves (6 to 16 points, any spacing of the hues) and a worst case, scanned at 20 000 hues each; the numbers are
    /// printed and written into `docs/STATUS.md`.
    #[test]
    fn the_baked_table_error_against_the_exact_spline_is_small_and_measured() {
        let mut seed = 0x5EED_u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut curve = |n: usize, lo: f64, hi: f64| -> Vec<(f64, f64)> {
            // Distinct hues at least 10 degrees apart, sorted.
            let mut hues: Vec<f64> = Vec::new();
            while hues.len() < n {
                let h = (next() * 360.0).floor();
                if hues
                    .iter()
                    .all(|&x| (x - h).abs() >= 10.0 && (x - h).abs() <= 350.0)
                {
                    hues.push(h);
                }
            }
            hues.sort_by(f64::total_cmp);
            hues.into_iter()
                .map(|h| (h, lo + (hi - lo) * next()))
                .collect()
        };
        let (mut worst_turn, mut worst_sat) = (0.0f64, 0.0f64);
        for i in 0..200 {
            let n = 6 + i % 11;
            // A fitted turn lies within about 20 degrees, a fitted chroma multiplier within 0.6 to 1.5.
            let turn = HueTable::max_error_against_exact(&curve(n, -20.0, 20.0), 20_000);
            let sat = HueTable::max_error_against_exact(&curve(n, 0.6, 1.5), 20_000);
            worst_turn = worst_turn.max(turn);
            worst_sat = worst_sat.max(sat);
        }
        // The widest the field allows: 16 points, the whole range, 10 degrees apart at least.
        let wide_turn = HueTable::max_error_against_exact(&curve(16, -180.0, 180.0), 20_000);
        let wide_sat = HueTable::max_error_against_exact(&curve(16, 0.0, 8.0), 20_000);
        eprintln!(
            "hue table error: fitted-style turns {worst_turn:.3e} deg, chroma multipliers {worst_sat:.3e}; widest curves {wide_turn:.3e} deg, {wide_sat:.3e}"
        );
        assert!(worst_turn < 5e-2, "{worst_turn}");
        assert!(worst_sat < 1.5e-3, "{worst_sat}");
        assert!(wide_turn < 0.4 && wide_sat < 1e-2, "{wide_turn} {wide_sat}");
    }

    #[test]
    fn an_empty_or_neutral_curve_is_the_identity_bit_for_bit() {
        let step = HueStep::new(&[], &[]);
        assert!(step.is_identity());
        let v = [0.31, 0.52, 0.44];
        assert_eq!(step.apply(v), v);
        let neutral = HueStep::new(&[(0.0, 0.0), (180.0, 0.0)], &[(90.0, 1.0)]);
        assert!(neutral.is_identity());
        assert_eq!(neutral.apply(v), v);
    }

    #[test]
    fn grey_and_zero_chroma_are_unchanged_and_never_nan() {
        let step = HueStep::new(&[(0.0, 40.0), (200.0, -90.0)], &[(0.0, 2.0), (180.0, 0.0)]);
        for g in [0.0, 0.2784, 1.0, -0.3, 1e-300] {
            let v = [g, g, g];
            assert_eq!(step.apply(v), v);
        }
        // Chroma tends to nothing: the move tends to nothing and stays finite.
        for r in [1e-300, 1e-200, 1e-100, 1e-20, 1e-12] {
            let v = [0.4 + r, 0.4, 0.4];
            let o = step.apply(v);
            assert!(o.iter().all(|c| c.is_finite()));
            close(o, v, 1e-11);
        }
    }

    #[test]
    fn the_mean_goes_through_the_plane_and_the_rec709_luma_is_restored() {
        let step = HueStep::new(
            &[(0.0, -35.0), (90.0, 60.0), (200.0, 10.0), (300.0, -80.0)],
            &[(0.0, 1.8), (120.0, 0.4), (240.0, 2.5)],
        );
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..2000 {
            let v = [next() * 1.2 - 0.1, next() * 1.2 - 0.1, next() * 1.2 - 0.1];
            let o = step.apply(v);
            assert!((luma(o) - luma(v)).abs() < 1e-12, "{v:?}");
            // The shift that restores luma is one number added to all three codes: the chroma plane is what the curves made.
            let (p, q) = plane(v);
            let (p2, q2) = plane(o);
            let r = p.hypot(q);
            let r2 = p2.hypot(q2);
            let a = q.atan2(p).to_degrees();
            let sigma = step.hue_vs_sat().unwrap().lookup(a);
            assert!((r2 - r * sigma).abs() < 1e-12, "{v:?}");
        }
    }

    #[test]
    fn a_half_turn_negates_the_plane_and_keeps_chroma_and_luma() {
        let step = HueStep::new(&[(40.0, 180.0)], &[]);
        let v = [0.62, 0.3, 0.41];
        let o = step.apply(v);
        let (p, q) = plane(v);
        let (p2, q2) = plane(o);
        assert!((p2 + p).abs() < 1e-12 && (q2 + q).abs() < 1e-12);
        assert!((luma(o) - luma(v)).abs() < 1e-12);
    }

    #[test]
    fn a_one_point_sat_curve_scales_the_chroma_everywhere() {
        let step = HueStep::new(&[], &[(10.0, 0.0)]);
        let v = [0.7, 0.2, 0.35];
        let o = step.apply(v);
        let (p, q) = plane(o);
        assert!(p.hypot(q) < 1e-12);
        assert!((luma(o) - luma(v)).abs() < 1e-12);
    }

    #[test]
    fn the_seam_has_no_step_in_the_pixel_result() {
        let step = HueStep::new(&[(20.0, 90.0), (200.0, -60.0)], &[(100.0, 1.7)]);
        let r = 0.1;
        let mut prev: Option<Vec3> = None;
        for i in 0..=3600 {
            let a = (350.0 + 20.0 * f64::from(i) / 3600.0f64).to_radians();
            let (p, q) = (r * a.cos(), r * a.sin());
            // The triple with mean 0.4 and plane (p, q).
            let v = [
                0.4 + p * E1[0] + q * E2[0],
                0.4 + p * E1[1] + q * E2[1],
                0.4 + p * E1[2] + q * E2[2],
            ];
            let o = step.apply(v);
            if let Some(pv) = prev {
                for k in 0..3 {
                    assert!((o[k] - pv[k]).abs() < 5e-3 * 0.0648, "{i}");
                }
            }
            prev = Some(o);
        }
    }

    #[test]
    fn an_element_step_is_local_by_hand() {
        // Exposure only: +1 stop adds 0.0648 to every code.
        let g = ElementGrade {
            exposure_ev: 1.0,
            ..ElementGrade::IDENTITY
        };
        let v = [0.5, 0.3, 0.2];
        close(g.local(v), [0.5648, 0.3648, 0.2648], 1e-15);
        // A half turn and a chroma of 2 on a pure red-axis triple: the plane vector is negated and doubled, the mean stays.
        let m = 0.3;
        let v = [m + 0.1 * E1[0], m + 0.1 * E1[1], m + 0.1 * E1[2]];
        let g = ElementGrade {
            exposure_ev: 0.0,
            hue_deg: 180.0,
            chroma: 2.0,
        };
        let o = g.local(v);
        close(
            o,
            [m - 0.2 * E1[0], m - 0.2 * E1[1], m - 0.2 * E1[2]],
            1e-12,
        );
        assert!((mean(o) - m).abs() < 1e-12);
        // The tone is kept by hue and chroma at any exposure of 0.
        let g = ElementGrade {
            exposure_ev: 0.0,
            hue_deg: 33.0,
            chroma: 0.7,
        };
        let v = [0.61, 0.22, 0.47];
        assert!((mean(g.local(v)) - mean(v)).abs() < 1e-12);
        // The exposure shifts the mean by its own amount whatever the hue and chroma.
        let g = ElementGrade {
            exposure_ev: -0.5,
            hue_deg: 33.0,
            chroma: 0.7,
        };
        assert!((mean(g.local(v)) - (mean(v) - 0.5 * 0.0648)).abs() < 1e-12);
    }

    #[test]
    fn the_element_step_blends_by_the_mask_in_order() {
        let mut grades = [ElementGrade::IDENTITY; ELEMENT_COUNT];
        grades[0] = ElementGrade {
            exposure_ev: 1.0,
            hue_deg: 20.0,
            chroma: 1.3,
        };
        grades[5] = ElementGrade {
            exposure_ev: -1.0,
            hue_deg: -10.0,
            chroma: 0.8,
        };
        let step = ElementStep::new(grades);
        let v = [0.52, 0.31, 0.27];
        // Mask 0 everywhere: bit for bit.
        assert_eq!(step.apply(v, &[0.0; ELEMENT_COUNT]), v);
        // Mask 1 on the sky alone is its local grade; 0.5 is halfway.
        let mut m = [0.0; ELEMENT_COUNT];
        m[0] = 1.0;
        close(step.apply(v, &m), grades[0].local(v), 1e-12);
        m[0] = 0.5;
        let half = step.apply(v, &m);
        let want: Vec<f64> = (0..3)
            .map(|k| v[k] + 0.5 * (grades[0].local(v)[k] - v[k]))
            .collect();
        close(half, [want[0], want[1], want[2]], 1e-12);
        // Two elements at once take the steps in order: the skin works on the sky's result.
        m[0] = 1.0;
        m[5] = 0.25;
        let a = grades[0].local(v);
        let b = grades[5].local(a);
        let want = [
            a[0] + 0.25 * (b[0] - a[0]),
            a[1] + 0.25 * (b[1] - a[1]),
            a[2] + 0.25 * (b[2] - a[2]),
        ];
        close(step.apply(v, &m), want, 1e-12);
        // The subject takes its exposure alone.
        let mut grades = [ElementGrade::IDENTITY; ELEMENT_COUNT];
        grades[SUBJECT].exposure_ev = 0.5;
        let step = ElementStep::new(grades);
        let mut m = [0.0; ELEMENT_COUNT];
        m[SUBJECT] = 1.0;
        close(
            step.apply(v, &m),
            [v[0] + 0.0324, v[1] + 0.0324, v[2] + 0.0324],
            1e-15,
        );
        assert!(!ElementStep::default().is_live(0));
        assert!(step.is_live(SUBJECT));
    }
}
