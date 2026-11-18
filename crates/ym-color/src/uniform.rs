//! Colour Uniform: the CPU reference of the node that pulls colours toward one target colour (`docs/32` §3).
//!
//! The node reads a LogC4 / AWG4 triple and works in the orthonormal basis of LogC4 code space that the colour warp uses
//! (`grade::E1`, `grade::E2` and the neutral axis): the chroma `c = (e1·v, e2·v)` with `r = |c|`, and the tone `m`, the mean
//! of the three codes. The target `(c_t, m_t)` and three amounts `κ_h`, `κ_s`, `κ_l` in [0, 1] shrink, for each pixel, the
//! angle to the target's hue, the distance to its chroma and the distance to its tone (§3.3). Two fixed guards keep the map
//! continuous: the neutral guard `g` on hue and saturation, and the antipode fade `a` on hue (§3.2).
//!
//! There is no selection weight here: the node's mask is the selection, and [`apply`] computes the pull for every pixel it is
//! given. The mask blend `in + m (node(in) − in)` belongs to the plan (CU-04).
//!
//! Everything is f64, with `f64::atan2`, `cos` and `sin`, no `fma`, and the order of §3.2 and §3.3, so the result is the same
//! for any number of threads. The device kernel (CU-05) is the same expressions in f32.

use crate::consts::{LOGC4_GRAY, LOGC4_STOP, Vec3};
use crate::grade::{E1, E2};
use crate::warp::WARP_NEUTRAL_GUARD;

/// Chroma below which a colour has no hue and the hue and saturation pulls fade out: the colour warp's guard (`g(r) = S(r / 0.02)`).
pub const UNIFORM_NEUTRAL_GUARD: f64 = WARP_NEUTRAL_GUARD;
/// The hue pull is full up to this angle (90°) from the target's hue.
pub const HUE_FULL: f64 = std::f64::consts::FRAC_PI_2;
/// The hue pull fades to nothing over this angle (60°) after [`HUE_FULL`], so it is zero from 150° on.
pub const HUE_FADE: f64 = std::f64::consts::FRAC_PI_3;

/// Smallest target chroma: the target is held off grey (Saturation 2), which keeps the saturation axis monotone (`docs/32` §3.7).
pub const TARGET_CHROMA_MIN: f64 = UNIFORM_NEUTRAL_GUARD;
/// Largest target chroma (Saturation 50).
pub const TARGET_CHROMA_MAX: f64 = 0.5;
/// Smallest target tone (a LogC4 code).
pub const TARGET_TONE_MIN: f64 = 0.0;
/// Largest target tone (a LogC4 code).
pub const TARGET_TONE_MAX: f64 = 1.0;

/// The target colour in the node's plane: the chroma `(c1, c2)` along `e1`, `e2` and the tone `m` (the mean of the three LogC4 codes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target {
    /// Chroma along `e1` (red against cyan).
    pub c1: f64,
    /// Chroma along `e2` (green against blue).
    pub c2: f64,
    /// Tone: the mean of the three LogC4 codes.
    pub tone: f64,
}

impl Target {
    /// A target from plane coordinates, held in range: chroma between [`TARGET_CHROMA_MIN`] and [`TARGET_CHROMA_MAX`] (a chroma
    /// of 0 is placed on `+e1`), tone between 0 and 1, a non-finite number read as 0.
    #[must_use]
    pub fn new(c1: f64, c2: f64, tone: f64) -> Self {
        let fin = |x: f64| if x.is_finite() { x } else { 0.0 };
        let (mut a, mut b) = (fin(c1), fin(c2));
        let r = a.hypot(b);
        if r < TARGET_CHROMA_MIN {
            if r == 0.0 {
                (a, b) = (TARGET_CHROMA_MIN, 0.0);
            } else {
                let k = TARGET_CHROMA_MIN / r;
                (a, b) = (a * k, b * k);
            }
        } else if r > TARGET_CHROMA_MAX {
            let k = TARGET_CHROMA_MAX / r;
            (a, b) = (a * k, b * k);
        }
        Self {
            c1: a,
            c2: b,
            tone: fin(tone).clamp(TARGET_TONE_MIN, TARGET_TONE_MAX),
        }
    }

    /// The target of the stored units (`docs/32` §3.2): `hue_deg` (read modulo 360), `saturation` (the chroma times 100, held to 2 … 50)
    /// and `lightness_ev` (stops from middle grey).
    #[must_use]
    pub fn from_stored(hue_deg: f64, saturation: f64, lightness_ev: f64) -> Self {
        let r = saturation / 100.0;
        let h = hue_deg.rem_euclid(360.0).to_radians();
        Self::new(
            r * h.cos(),
            r * h.sin(),
            LOGC4_GRAY + lightness_ev * LOGC4_STOP,
        )
    }

    /// The target that a LogC4 / AWG4 colour gives (what a pick sets), held in range.
    #[must_use]
    pub fn from_rgb(v: Vec3) -> Self {
        let c = chroma_of(v);
        Self::new(c[0], c[1], tone_of(v))
    }

    /// The stored units of this target: hue in degrees in [0, 360), Saturation (chroma times 100) and Lightness in stops from middle grey.
    #[must_use]
    pub fn to_stored(&self) -> (f64, f64, f64) {
        let h = self.c2.atan2(self.c1).to_degrees().rem_euclid(360.0);
        (
            h,
            100.0 * self.c1.hypot(self.c2),
            (self.tone - LOGC4_GRAY) / LOGC4_STOP,
        )
    }

    /// The target as a LogC4 / AWG4 triple on the grey axis plus its chroma: `m (1,1,1) + c1 e1 + c2 e2`.
    #[must_use]
    pub fn to_rgb(&self) -> Vec3 {
        let m = self.tone;
        [
            m + self.c1 * E1[0] + self.c2 * E2[0],
            m + self.c1 * E1[1] + self.c2 * E2[1],
            m + self.c1 * E1[2] + self.c2 * E2[2],
        ]
    }
}

/// The three amounts of the node, each in [0, 1] (the stored Compress values divided by 100).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Amounts {
    /// Hue: the share of the angle to the target's hue that is removed.
    pub hue: f64,
    /// Saturation: the share of the distance to the target's chroma that is removed.
    pub saturation: f64,
    /// Lightness: the share of the distance to the target's tone that is removed.
    pub lightness: f64,
}

impl Amounts {
    /// The amounts of the stored percentages (0 … 100 each), held to [0, 1]; a non-finite number reads as 0.
    #[must_use]
    pub fn from_percent(hue: f64, saturation: f64, lightness: f64) -> Self {
        let f = |p: f64| {
            if p.is_finite() {
                (p / 100.0).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        Self {
            hue: f(hue),
            saturation: f(saturation),
            lightness: f(lightness),
        }
    }

    /// True when every amount is 0: the node is the identity.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.hue == 0.0 && self.saturation == 0.0 && self.lightness == 0.0
    }
}

/// The smoothstep `S(x) = u²(3 − 2u)`, `u = clamp(x, 0, 1)` (a NaN reads as 0).
fn smooth(x: f64) -> f64 {
    let u = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
    u * u * (3.0 - 2.0 * u)
}

/// The chroma `(e1·v, e2·v)` of a LogC4 / AWG4 triple, a point of the warp's plane.
#[must_use]
pub fn chroma_of(v: Vec3) -> [f64; 2] {
    [
        v[0] * E1[0] + v[1] * E1[1] + v[2] * E1[2],
        v[0] * E2[0] + v[1] * E2[1] + v[2] * E2[2],
    ]
}

/// The tone of a LogC4 / AWG4 triple: the mean of its three codes.
#[must_use]
pub fn tone_of(v: Vec3) -> f64 {
    (v[0] + v[1] + v[2]) / 3.0
}

/// The neutral guard `g(r) = S(r / 0.02)`: 0 at `r = 0`, 1 from `r = 0.02`, non-decreasing.
#[must_use]
pub fn neutral_guard(r: f64) -> f64 {
    smooth(r / UNIFORM_NEUTRAL_GUARD)
}

/// The antipode fade `a(δ) = 1 − S((|δ| − π/2) / (π/3))`: 1 up to 90° from the target's hue, exactly 0 from 150° on, non-increasing in `|δ|`.
#[must_use]
pub fn antipode_fade(delta: f64) -> f64 {
    1.0 - smooth((delta.abs() - HUE_FULL) / HUE_FADE)
}

/// The signed angle from the target's hue to the hue of the chroma `(c1, c2)`, the short way round, in (−π, π]; 0 when `r = 0`.
#[must_use]
pub fn hue_delta(target: &Target, c1: f64, c2: f64) -> f64 {
    if c1.hypot(c2) > 0.0 {
        (target.c1 * c2 - target.c2 * c1).atan2(target.c1 * c1 + target.c2 * c2)
    } else {
        0.0
    }
}

/// Apply the node to one LogC4 / AWG4 triple, with its mask at 1 (`docs/32` §3.3 and §3.4).
///
/// No target, or all three amounts 0, returns the input's bits. An amount of 0 skips its axis: with Hue 0 the direction of `c` is
/// kept and only its length changes (no trigonometry), with Saturation 0 the length is kept, with Lightness 0 the tone is not touched.
/// A pixel with `r = 0` gets no chroma and no `atan2`. Nothing is clipped.
#[must_use]
pub fn apply(v: Vec3, target: Option<&Target>, amounts: Amounts) -> Vec3 {
    let Some(t) = target else { return v };
    if amounts.is_zero() {
        return v;
    }
    let (kh, ks, kl) = (amounts.hue, amounts.saturation, amounts.lightness);
    let rt = t.c1.hypot(t.c2);
    let [c1, c2] = chroma_of(v);
    let m = tone_of(v);
    let r = c1.hypot(c2);
    let g = neutral_guard(r);

    // Saturation: the distance to the target's chroma shrinks, not for neutrals.
    let r2 = if ks != 0.0 {
        rt + (r - rt) * (1.0 - ks * g)
    } else {
        r
    };
    // Hue: the angle to the target's hue shrinks, not for neutrals, not on the far side.
    let (n1, n2) = if r > 0.0 {
        if kh != 0.0 {
            let delta = hue_delta(t, c1, c2);
            let th = t.c2.atan2(t.c1) + delta * (1.0 - kh * g * antipode_fade(delta));
            (r2 * th.cos(), r2 * th.sin())
        } else {
            let k = r2 / r;
            (c1 * k, c2 * k)
        }
    } else {
        (0.0, 0.0)
    };
    let (d1, d2) = (n1 - c1, n2 - c2);
    let dm = if kl != 0.0 {
        (t.tone + (m - t.tone) * (1.0 - kl)) - m
    } else {
        0.0
    };
    let mut out = [0.0; 3];
    for (i, o) in out.iter_mut().enumerate() {
        let mut x = v[i] + d1 * E1[i] + d2 * E2[i];
        if kl != 0.0 {
            x += dm;
        }
        *o = x;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// The mock-ups' skin target of the sketch: hue 25.8°, Saturation 9.7, tone 0.2767.
    fn skin() -> Target {
        Target {
            c1: 0.0875,
            c2: 0.0423,
            tone: 0.2767,
        }
    }
    fn theta_t(t: &Target) -> f64 {
        t.c2.atan2(t.c1)
    }
    fn amt(h: f64, s: f64, l: f64) -> Amounts {
        Amounts {
            hue: h,
            saturation: s,
            lightness: l,
        }
    }
    /// `m (1,1,1) + c1 e1 + c2 e2`.
    fn rgb(m: f64, c1: f64, c2: f64) -> Vec3 {
        [
            m + c1 * E1[0] + c2 * E2[0],
            m + c1 * E1[1] + c2 * E2[1],
            m + c1 * E1[2] + c2 * E2[2],
        ]
    }
    fn polar(m: f64, r: f64, th: f64) -> Vec3 {
        rgb(m, r * th.cos(), r * th.sin())
    }
    /// (angle from the target, chroma, tone) of a triple.
    fn read(v: Vec3, t: &Target) -> (f64, f64, f64) {
        let c = chroma_of(v);
        (hue_delta(t, c[0], c[1]), c[0].hypot(c[1]), tone_of(v))
    }
    fn maxabs(a: Vec3, b: Vec3) -> f64 {
        (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max)
    }
    /// SplitMix64: a deterministic generator for the tests.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }
        fn rgb(&mut self, lo: f64, hi: f64) -> Vec3 {
            [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
        }
    }

    #[test]
    fn basis_reconstructs_the_triple() {
        let mut rs = Rng(1);
        for _ in 0..1000 {
            let v = rs.rgb(-0.3, 1.2);
            let c = chroma_of(v);
            assert!(maxabs(rgb(tone_of(v), c[0], c[1]), v) < 1e-15);
        }
    }

    // ---- identity
    #[test]
    fn identity_without_target_or_amounts_is_bit_exact() {
        let mut rs = Rng(2);
        let t = skin();
        for _ in 0..5000 {
            let v = rs.rgb(0.0, 0.9);
            assert_eq!(apply(v, Some(&t), amt(0.0, 0.0, 0.0)), v);
            assert_eq!(apply(v, None, amt(1.0, 1.0, 1.0)), v);
        }
    }

    #[test]
    fn identity_at_zero_on_each_axis() {
        let mut rs = Rng(3);
        let t = skin();
        for _ in 0..2000 {
            let v = rs.rgb(0.0, 0.9);
            // an axis at 0 leaves its coordinate alone while the others are on
            let (d0, r0, m0) = read(v, &t);
            let (d, _, m) = read(apply(v, Some(&t), amt(0.0, 1.0, 0.0)), &t);
            assert!((d - d0).abs() < 1e-12 && (m - m0).abs() < 1e-12);
            let (_, r, m) = read(apply(v, Some(&t), amt(1.0, 0.0, 0.0)), &t);
            assert!((r - r0).abs() < 1e-12 && (m - m0).abs() < 1e-12);
            let (d, r, _) = read(apply(v, Some(&t), amt(1.0, 1.0, 0.0)), &t);
            let (d1, r1, m1) = read(apply(v, Some(&t), amt(1.0, 1.0, 1.0)), &t);
            assert!((d - d1).abs() < 1e-9 && (r - r1).abs() < 1e-12 && (m1 - t.tone).abs() < 1e-12);
            let (d, r, m) = read(apply(v, Some(&t), amt(0.0, 0.0, 1.0)), &t);
            assert!((d - d0).abs() < 1e-12 && (r - r0).abs() < 1e-12 && (m - t.tone).abs() < 1e-12);
        }
    }

    #[test]
    fn saturation_only_changes_the_length_not_the_direction() {
        let mut rs = Rng(4);
        let t = skin();
        for _ in 0..2000 {
            let v = polar(rs.range(0.0, 1.0), rs.range(0.001, 0.3), rs.range(-PI, PI));
            let c = chroma_of(v);
            let o = chroma_of(apply(v, Some(&t), amt(0.0, 0.7, 0.0)));
            // the cross product of the two directions is zero and they point the same way
            assert!((c[0] * o[1] - c[1] * o[0]).abs() < 1e-15);
            assert!(c[0] * o[0] + c[1] * o[1] > 0.0);
        }
    }

    // ---- fixed point, 100 % and 50 %
    #[test]
    fn the_target_colour_is_a_fixed_point() {
        let t = skin();
        let vt = t.to_rgb();
        for k in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let o = apply(vt, Some(&t), amt(k, k, k));
            assert!(maxabs(o, vt) < 1e-12, "{k}");
        }
        let o = apply(vt, Some(&t), amt(1.0, 1.0, 1.0));
        assert!(maxabs(o, vt) < 1e-12);
    }

    fn within_full_zone(rs: &mut Rng, t: &Target, n: usize) -> Vec<(f64, f64, f64)> {
        (0..n)
            .map(|_| {
                (
                    rs.range(-HUE_FULL, HUE_FULL),
                    rs.range(UNIFORM_NEUTRAL_GUARD, 0.3),
                    rs.range(0.0, 1.0),
                )
            })
            .inspect(|_| {
                let _ = t;
            })
            .collect()
    }

    #[test]
    fn a_hundred_percent_lands_on_the_target() {
        let mut rs = Rng(5);
        let t = skin();
        let (rt, ta) = (t.c1.hypot(t.c2), theta_t(&t));
        for (dd, rr, mm) in within_full_zone(&mut rs, &t, 4000) {
            let v = polar(mm, rr, ta + dd);
            let (d, r, m) = read(apply(v, Some(&t), amt(1.0, 1.0, 1.0)), &t);
            assert!(d.abs() < 1e-9 && (r - rt).abs() < 1e-9 && (m - t.tone).abs() < 1e-9);
        }
    }

    #[test]
    fn fifty_percent_is_halfway_on_each_axis() {
        let mut rs = Rng(6);
        let t = skin();
        let (rt, ta) = (t.c1.hypot(t.c2), theta_t(&t));
        for (dd, rr, mm) in within_full_zone(&mut rs, &t, 4000) {
            let v = polar(mm, rr, ta + dd);
            let (d, r, m) = read(apply(v, Some(&t), amt(0.5, 0.5, 0.5)), &t);
            assert!((d - dd * 0.5).abs() < 1e-12);
            assert!((r - (rt + (rr - rt) * 0.5)).abs() < 1e-12);
            assert!((m - (t.tone + (mm - t.tone) * 0.5)).abs() < 1e-12);
        }
    }

    // ---- guards
    #[test]
    fn guards_are_monotone_and_in_range() {
        let mut last = -1.0;
        for i in 0..=2000 {
            let g = neutral_guard(f64::from(i) * 1e-4);
            assert!((0.0..=1.0).contains(&g) && g >= last);
            last = g;
        }
        assert_eq!(neutral_guard(0.0), 0.0);
        assert_eq!(neutral_guard(0.02), 1.0);
        assert_eq!(neutral_guard(0.5), 1.0);
        let mut last = 2.0;
        for i in 0..=3600 {
            let a = antipode_fade(f64::from(i) * PI / 3600.0);
            assert!((0.0..=1.0).contains(&a) && a <= last);
            last = a;
        }
        assert_eq!(antipode_fade(0.0), 1.0);
        assert_eq!(antipode_fade(HUE_FULL), 1.0);
        assert_eq!(antipode_fade(150f64.to_radians()), 0.0);
        assert_eq!(antipode_fade(-PI), 0.0);
        assert!((antipode_fade(120f64.to_radians()) - 0.5).abs() < 1e-12);
    }

    // ---- formulas, to 1e-12
    #[test]
    fn formulas_on_hand_computed_values() {
        let t = skin();
        let ta = theta_t(&t);
        let r = 0.1;
        // 60 degrees from the target: a = 1, so a hue pull of 1 (kappa 0.8) leaves 20 % of the angle
        let o = apply(
            polar(0.3, r, ta + 60f64.to_radians()),
            Some(&t),
            amt(0.8, 0.0, 0.0),
        );
        let (d, rr, m) = read(o, &t);
        assert!(
            (d - 60f64.to_radians() * 0.2).abs() < 1e-12
                && (rr - r).abs() < 1e-12
                && (m - 0.3).abs() < 1e-12
        );
        // 120 degrees: a = 0.5, so delta' = delta (1 - 0.5 kappa)
        let o = apply(
            polar(0.3, r, ta + 120f64.to_radians()),
            Some(&t),
            amt(1.0, 0.0, 0.0),
        );
        let (d, ..) = read(o, &t);
        assert!((d - 60f64.to_radians()).abs() < 1e-12);
        // 150 degrees and beyond: not moved
        for deg in [150.0, 170.0, -160.0] {
            let v = polar(0.3, r, ta + f64::to_radians(deg));
            assert!(maxabs(apply(v, Some(&t), amt(1.0, 0.0, 0.0)), v) < 1e-15);
        }
        // the target's exact opposite: a = 0 whichever side atan2 picks; no jump across it
        let opp = apply(polar(0.3, r, ta + PI), Some(&t), amt(1.0, 0.0, 0.0));
        assert!(maxabs(opp, polar(0.3, r, ta + PI)) < 1e-12);
        let eps = 1e-9;
        let (a, b) = (
            apply(polar(0.3, r, ta + PI - eps), Some(&t), amt(1.0, 0.0, 0.0)),
            apply(polar(0.3, r, ta + PI + eps), Some(&t), amt(1.0, 0.0, 0.0)),
        );
        assert!(maxabs(a, polar(0.3, r, ta + PI - eps)) < 1e-12);
        assert!(maxabs(b, polar(0.3, r, ta + PI + eps)) < 1e-12);
        // the short way round: target at 355 degrees, pixel at 5 degrees, 10 degrees apart
        let t2 = Target::from_stored(355.0, 8.0, 0.0);
        let v = polar(0.3, 0.1, 5f64.to_radians());
        let (d, ..) = read(v, &t2);
        assert!((d - 10f64.to_radians()).abs() < 1e-12);
        let (d, ..) = read(apply(v, Some(&t2), amt(1.0, 0.0, 0.0)), &t2);
        assert!(d.abs() < 1e-12);
        // saturation and lightness lines
        let rt = t.c1.hypot(t.c2);
        let v = polar(0.5, 0.2, ta + 0.3);
        let (_, rr, m) = read(apply(v, Some(&t), amt(0.0, 0.25, 0.4)), &t);
        assert!((rr - (rt + (0.2 - rt) * 0.75)).abs() < 1e-12);
        assert!((m - (t.tone + (0.5 - t.tone) * 0.6)).abs() < 1e-12);
        // inside the neutral guard the saturation pull is S(r/0.02) of the full one
        let r = 0.01;
        let g = neutral_guard(r);
        let (_, rr, _) = read(
            apply(polar(0.3, r, ta + 0.3), Some(&t), amt(0.0, 1.0, 0.0)),
            &t,
        );
        assert!((rr - (rt + (r - rt) * (1.0 - g))).abs() < 1e-12);
    }

    // ---- the worked example of docs/32 §3.3
    #[test]
    fn the_worked_example_holds_to_1e_9() {
        let t = skin();
        let (rt, ta) = (t.c1.hypot(t.c2), theta_t(&t));
        let (r, th) = (0.12, ta + 20f64.to_radians());
        let v = polar(t.tone + 0.5 * LOGC4_STOP, r, th);
        // 50 %: hue 10 degrees from the target, chroma halfway, tone halfway
        let o = apply(v, Some(&t), amt(0.5, 0.5, 0.5));
        let (d, rr, m) = read(o, &t);
        assert!((d - 10f64.to_radians()).abs() < 1e-9);
        assert!((rr - (rt + (r - rt) * 0.5)).abs() < 1e-9);
        assert!((m - (t.tone + 0.25 * LOGC4_STOP)).abs() < 1e-9);
        // the numbers of the text: Hue 35.8, Saturation 10.859, Lightness +0.224 EV (the sketch's values)
        let (hue, sat, ev) = (
            (ta + d).to_degrees(),
            100.0 * rr,
            (m - LOGC4_GRAY) / LOGC4_STOP,
        );
        assert!(
            (hue - 35.8).abs() < 0.05 && (sat - 10.859).abs() < 5e-4 && (ev - 0.224).abs() < 5e-4
        );
        // 100 %: the target
        let (d, rr, m) = read(apply(v, Some(&t), amt(1.0, 1.0, 1.0)), &t);
        assert!(d.abs() < 1e-9 && (rr - rt).abs() < 1e-9 && (m - t.tone).abs() < 1e-9);
        let (sat, ev) = (100.0 * rr, (m - LOGC4_GRAY) / LOGC4_STOP);
        assert!((sat - 9.719).abs() < 5e-4 && (ev + 0.026).abs() < 5e-4);
        // and with the stored values of the text: Hue 26, Saturation 9.7, Lightness -0.03 EV
        let t = Target::from_stored(26.0, 9.7, -0.03);
        let v = polar(
            t.tone + 0.5 * LOGC4_STOP,
            0.12,
            26f64.to_radians() + 20f64.to_radians(),
        );
        let (d, rr, m) = read(apply(v, Some(&t), amt(1.0, 1.0, 1.0)), &t);
        assert!(d.abs() < 1e-9 && (rr - 0.097).abs() < 1e-9 && (m - t.tone).abs() < 1e-9);
    }

    /// Outputs of `compress_ref.py` (f64, full precision) for pixels of the sketch's skin target: the cross-check of this port.
    #[test]
    fn matches_the_python_sketch() {
        let t = skin();
        let ta = theta_t(&t);
        let cases: &[(Vec3, (f64, f64, f64), Vec3)] = &PY_FIXTURE;
        for (i, (v, a, want)) in cases.iter().enumerate() {
            let o = apply(*v, Some(&t), amt(a.0, a.1, a.2));
            assert!(
                maxabs(o, *want) < 1e-12,
                "case {i}: {o:?} vs {want:?} (ta {ta})"
            );
        }
    }
    include!("uniform_py_fixture.rs");

    #[test]
    fn near_neutral_gain_matches_the_sketch() {
        let t = skin();
        let ta = theta_t(&t);
        // the sketch's added chroma at 100 % Hue and Saturation for pixels of chroma up to x (docs/32 §3.7)
        for (x, want) in [
            (0.001, 0.0006973642180299481),
            (0.005, 0.01440440125064604),
            (0.01, 0.04359408400206764),
            (0.02, 0.07762801395212879),
        ] {
            let mut worst = f64::MIN;
            for i in 0..=400 {
                let rr = x * f64::from(i) / 400.0;
                let v = polar(0.3, rr, ta + 1.0);
                let c = chroma_of(apply(v, Some(&t), amt(1.0, 1.0, 0.0)));
                worst = worst.max(c[0].hypot(c[1]) - chroma_of(v)[0].hypot(chroma_of(v)[1]));
            }
            // the sketch builds the line from tone 0.3 + r (cos, sin)(e1, e2): same pixels
            assert!((worst - want).abs() < 1e-9, "{x}: {worst} vs {want}");
        }
    }

    // ---- greys, black, NaN
    #[test]
    fn greys_keep_no_tint_and_nothing_is_nan() {
        let t = skin();
        for i in 0..50 {
            let m = -0.2 + 1.2 * f64::from(i) / 49.0;
            let g = [m, m, m];
            // Hue and Saturation on, Lightness 0: moves by rounding only
            assert!(maxabs(apply(g, Some(&t), amt(1.0, 1.0, 0.0)), g) < 1e-15);
            // with Lightness it stays grey
            let c = chroma_of(apply(g, Some(&t), amt(1.0, 1.0, 1.0)));
            assert!(c[0].abs() < 1e-15 && c[1].abs() < 1e-15);
        }
        let odd: [Vec3; 10] = [
            [0.0, 0.0, 0.0],
            [0.3, 0.3, 0.3],
            [1.0, 1.0, 1.0],
            [-0.5, -0.5, -0.5],
            [5.0, -5.0, 0.0],
            [1e-12, 0.0, 0.0],
            [1e-300, 0.0, 0.0],
            [5e-324, 0.0, 5e-324],
            [1e30, -1e30, 1e30],
            [-1e100, 1e100, 0.0],
        ];
        for v in odd {
            for a in [
                amt(1.0, 1.0, 1.0),
                amt(1.0, 0.0, 0.0),
                amt(0.0, 1.0, 0.0),
                amt(0.3, 0.6, 0.0),
            ] {
                assert!(apply(v, Some(&t), a).iter().all(|x| x.is_finite()), "{v:?}");
            }
        }
        // a pixel with r = 0 exactly has chroma 0 after
        let c = chroma_of(apply([0.0, 0.0, 0.0], Some(&t), amt(1.0, 1.0, 0.0)));
        assert_eq!(c, [0.0, 0.0]);
    }

    // ---- continuity across the hue wrap
    #[test]
    fn a_sweep_of_the_hue_circle_has_no_jump() {
        let degs = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 359.99];
        for deg in degs {
            let t = Target::from_stored(deg, 7.0, 0.5);
            for a in [amt(1.0, 0.5, 0.5), amt(0.7, 0.7, 0.0), amt(1.0, 1.0, 1.0)] {
                let mut prev: Option<Vec3> = None;
                let mut worst = 0.0f64;
                for i in 0..=36000 {
                    let th = -PI + 2.0 * PI * f64::from(i) / 36000.0;
                    let o = apply(polar(0.4, 0.07, th), Some(&t), a);
                    if let Some(p) = prev {
                        worst = worst.max(maxabs(o, p));
                    }
                    prev = Some(o);
                }
                eprintln!("sweep step {deg} {worst:e}");
                assert!(worst < 5e-3, "{deg}: {worst}");
                assert!(worst < 1e-4, "{deg}: {worst} (the sketch's order is 4e-5)");
            }
        }
    }

    #[test]
    fn the_seam_at_zero_and_360_changes_nothing() {
        let mut rs = Rng(7);
        let t1 = Target::from_stored(359.99, 7.0, 0.0);
        let t2 = Target::from_stored(-0.01, 7.0, 0.0);
        let t3 = Target::from_stored(719.99, 7.0, 0.0);
        for _ in 0..2000 {
            let th = rs.range(-PI, PI);
            let (r, m) = (rs.range(0.0, 0.3), rs.range(0.0, 1.0));
            let a = amt(0.8, 0.6, 0.3);
            let o1 = apply(polar(m, r, th), Some(&t1), a);
            assert!(maxabs(o1, apply(polar(m, r, th), Some(&t2), a)) < 1e-12);
            assert!(maxabs(o1, apply(polar(m, r, th), Some(&t3), a)) < 1e-12);
            // the same pixel a full turn on
            assert!(maxabs(o1, apply(polar(m, r, th + 2.0 * PI), Some(&t1), a)) < 1e-12);
        }
    }

    // ---- monotonicity
    const KS: [f64; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    #[test]
    fn monotone_along_the_hue_axis() {
        let t = skin();
        let ta = theta_t(&t);
        let mut rs = Rng(8);
        for _ in 0..100 {
            let (r, m) = (rs.range(0.001, 0.3), rs.range(0.0, 1.0));
            for k in KS {
                let mut last: Option<f64> = None;
                for i in 0..=2000 {
                    let d = -3.1 + 6.2 * f64::from(i) / 2000.0;
                    let (dd, ..) = read(apply(polar(m, r, ta + d), Some(&t), amt(k, 0.0, 0.0)), &t);
                    if let Some(p) = last {
                        assert!(dd - p >= -1e-12, "hue {k}");
                        if k < 1.0 {
                            assert!(dd - p > 0.0, "hue strict {k}");
                        }
                    }
                    last = Some(dd);
                }
            }
        }
    }

    #[test]
    fn monotone_along_the_saturation_axis() {
        let t = skin();
        let ta = theta_t(&t);
        let mut rs = Rng(9);
        for _ in 0..100 {
            let (d, m) = (rs.range(-3.0, 3.0), rs.range(0.0, 1.0));
            for k in KS {
                for hue in [0.0, k] {
                    let mut last: Option<f64> = None;
                    for i in 0..=2000 {
                        let r = 0.3 * f64::from(i) / 2000.0;
                        let c = chroma_of(apply(polar(m, r, ta + d), Some(&t), amt(hue, k, 0.0)));
                        let rr = c[0].hypot(c[1]);
                        if let Some(p) = last {
                            assert!(rr - p >= -1e-12, "sat {k} {hue}");
                            if k < 1.0 {
                                assert!(rr - p > 0.0, "sat strict {k} {hue}");
                            }
                        }
                        last = Some(rr);
                    }
                }
            }
        }
    }

    #[test]
    fn monotone_along_the_lightness_axis() {
        let t = skin();
        let ta = theta_t(&t);
        let mut rs = Rng(10);
        for _ in 0..100 {
            let (d, r) = (rs.range(-3.0, 3.0), rs.range(0.0, 0.3));
            for k in KS {
                let mut last: Option<f64> = None;
                for i in 0..=2000 {
                    let m = -0.2 + 1.3 * f64::from(i) / 2000.0;
                    let mo = tone_of(apply(polar(m, r, ta + d), Some(&t), amt(0.5, 0.5, k)));
                    if let Some(p) = last {
                        assert!(mo - p >= -1e-12, "light {k}");
                        if k < 1.0 {
                            assert!(mo - p > 0.0, "light strict {k}");
                        }
                    }
                    last = Some(mo);
                }
            }
        }
    }

    // ---- the Jacobian
    fn det3(j: [[f64; 3]; 3]) -> f64 {
        j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
    }
    fn jacobian_det(v: Vec3, t: &Target, a: Amounts) -> f64 {
        let h = 1e-6;
        let mut j = [[0.0; 3]; 3];
        for i in 0..3 {
            let (mut p, mut q) = (v, v);
            p[i] += h;
            q[i] -= h;
            let (op, oq) = (apply(p, Some(t), a), apply(q, Some(t), a));
            for r in 0..3 {
                j[r][i] = (op[r] - oq[r]) / (2.0 * h);
            }
        }
        det3(j)
    }

    #[test]
    fn the_jacobian_stays_above_its_bound() {
        let t = skin();
        let mut rs = Rng(11);
        let mut pts: Vec<Vec3> = Vec::new();
        for _ in 0..30000 {
            let m = rs.range(-0.2, 1.1);
            pts.push([
                m + rs.range(-0.3, 0.3),
                m + rs.range(-0.3, 0.3),
                m + rs.range(-0.3, 0.3),
            ]);
        }
        for _ in 0..30000 {
            let m = rs.range(0.0, 0.02);
            // near grey: a small normal-ish spread
            pts.push([
                m + rs.range(-0.006, 0.006),
                m + rs.range(-0.006, 0.006),
                m + rs.range(-0.006, 0.006),
            ]);
        }
        for (k, floor) in [(0.5, true), (0.9, true), (0.99, false), (1.0, false)] {
            let mut min = f64::MAX;
            for p in &pts {
                min = min.min(jacobian_det(*p, &t, amt(k, k, k)));
            }
            if floor {
                // (1 - k_s)^2 (1 - k_h)(1 - k_l), up to the noise of central differences
                let bound = (1.0 - k).powi(4);
                assert!(min >= bound - 1e-5, "{k}: {min} < {bound}");
                assert!(min > 0.0, "{k}: {min}");
            }
            if k == 1.0 {
                assert!(min >= -1e-6, "{min}");
            }
            eprintln!("jacobian min at {k}: {min:e}");
        }
    }

    // ---- determinism
    #[test]
    fn determinism_across_repeats_and_threads() {
        let t = skin();
        let mut rs = Rng(12);
        let px: Vec<Vec3> = (0..30000).map(|_| rs.rgb(-0.1, 1.0)).collect();
        let a = amt(0.8, 0.6, 0.4);
        let serial: Vec<Vec3> = px.iter().map(|v| apply(*v, Some(&t), a)).collect();
        let again: Vec<Vec3> = px.iter().map(|v| apply(*v, Some(&t), a)).collect();
        assert_eq!(serial, again);
        for threads in [1usize, 3, 4] {
            let chunk = px.len().div_ceil(threads);
            let mut out = vec![[0.0; 3]; px.len()];
            std::thread::scope(|s| {
                for (src, dst) in px.chunks(chunk).zip(out.chunks_mut(chunk)) {
                    s.spawn(move || {
                        for (v, o) in src.iter().zip(dst.iter_mut()) {
                            *o = apply(*v, Some(&t), a);
                        }
                    });
                }
            });
            let same = serial
                .iter()
                .zip(&out)
                .all(|(x, y)| (0..3).all(|i| x[i].to_bits() == y[i].to_bits()));
            assert!(same, "{threads} threads");
        }
    }

    // ---- units and the target
    #[test]
    fn stored_units_round_trip_and_clamp() {
        let t = Target::from_stored(26.0, 9.7, -0.03);
        let (h, s, ev) = t.to_stored();
        assert!((h - 26.0).abs() < 1e-9 && (s - 9.7).abs() < 1e-9 && (ev + 0.03).abs() < 1e-9);
        let t = Target::from_stored(-30.0, 9.7, 0.0);
        assert!((t.to_stored().0 - 330.0).abs() < 1e-9);
        let t = Target::from_stored(10.0, 0.5, 0.0);
        assert!((t.to_stored().1 - 2.0).abs() < 1e-9);
        let t = Target::from_stored(10.0, 90.0, 100.0);
        assert!((t.to_stored().1 - 50.0).abs() < 1e-9 && t.tone == 1.0);
        let t = Target::new(0.0, 0.0, f64::NAN);
        assert_eq!((t.c1, t.c2, t.tone), (0.02, 0.0, 0.0));
        // middle grey is zero stops
        let t = Target::from_stored(0.0, 5.0, 0.0);
        assert!((t.tone - 0.278_395_836_548_265_3).abs() < 1e-15);
        // a colour from the picture reads back as the target it was
        let t = skin();
        let t2 = Target::from_rgb(t.to_rgb());
        assert!(
            (t2.c1 - t.c1).abs() < 1e-15
                && (t2.c2 - t.c2).abs() < 1e-15
                && (t2.tone - t.tone).abs() < 1e-15
        );
        assert_eq!(
            Amounts::from_percent(150.0, -3.0, f64::NAN),
            amt(1.0, 0.0, 0.0)
        );
        assert!(Amounts::default().is_zero());
    }
}
