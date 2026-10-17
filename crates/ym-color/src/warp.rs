//! Colour warp in LogC4 (stage C9b) (`docs/02-colour-pipeline.md` section C9b).
//!
//! Pins move chosen colours across a chroma plane, in the manner of DaVinci Resolve's Color Warper
//! (the owner's request of 2026-10-17). Colours near a pin follow it smoothly, locked pins hold their
//! colours, and neutrals never move.
//!
//! The plane is the chroma of a LogC4 pixel `v` along the colour-balance axes of C9:
//! `c = (e1·v, e2·v)`, `e1 = (1, -1/2, -1/2) / sqrt(1.5)` (red against cyan) and
//! `e2 = (0, 1, -1) / sqrt(2)` (green against blue). Moving `c` by `d` adds `d1 e1 + d2 e2` to `v`, which
//! leaves the mean of the three channels unchanged. The tone of a pixel is `L = encode(Y(decode(v)))`.
//!
//! For a pixel `(c, L)` and the moving pins `i` (source `s`, target `t`, range `rho`, tonal weights
//! `w_lo`, `w_hi` around the pivot `P`, exposure `x` in stops) and the locked pins `k` (at `l`, range
//! `rho_k`):
//!
//! ```text
//! delta_i = t_i - s_i
//! rho_i'  = max(rho_i, |delta_i| / 1.5)
//! phi_i   = exp(-|c - s_i|^2 / (2 rho_i'^2))
//! w_i     = w_lo + (w_hi - w_lo) smoothstep(P_i - 2S, P_i + 2S, L)
//! n       = max(1, sum phi_i)
//! lambda  = prod_k (1 - exp(-|c - l_k|^2 / (2 rho_k^2)))
//! g       = smoothstep(0, 0.02, |c|)
//! D       = g lambda sum(w_i phi_i delta_i) / n
//! E       = g lambda sum(w_i phi_i x_i) / n
//! v'      = v + D1 e1 + D2 e2 + E S
//! ```
//!
//! # What the tests check
//!
//! * No pins, locked pins only, or pins that neither move nor change exposure: the identity, exactly.
//! * One pin, away from neutral and locks, with tonal weights of 1: its source lands on its target
//!   (to rounding), and the plane does not fold (the Jacobian determinant of `c -> c + D` stays
//!   positive on a fine grid).
//! * A locked pin's colour does not move at all, at any tone.
//! * Neutral pixels (`c = 0`) never move; the brightness along the neutral axis (the channel mean) is
//!   unchanged by `D`.
//! * Tonal weights: Shadows 0 leaves the shadows alone, Highlights 0 the highlights.
//! * Exposure moves all three channels by the same amount, so the chroma of the pin's own colour
//!   is unchanged.

use crate::consts::{LOGC4_STOP, Vec3};
use crate::grade::{E1, E2, logc4_at_stops};
use crate::logc::LogC4;

/// A point of the chroma plane, `(e1·v, e2·v)`.
pub type ChromaPoint = [f64; 2];

/// At most this many pins; later ones are ignored.
pub const WARP_MAX_PINS: usize = 16;
/// How far out the UI draws the plane (`|c|` at its edge). The colours of ordinary photos sit inside:
/// a red of linear (0.18, 0.009, 0.009) in AWG4 reads `|c| = 0.13`.
pub const WARP_PLANE_EXTENT: f64 = 0.30;
/// Sources and targets are scaled back to `|c| <= WARP_COORD_LIMIT`.
pub const WARP_COORD_LIMIT: f64 = 0.5;
/// The smallest range of a pin.
pub const WARP_RANGE_MIN: f64 = 0.01;
/// The largest range of a pin.
pub const WARP_RANGE_MAX: f64 = 0.25;
/// The range of a new pin.
pub const WARP_RANGE_DEFAULT: f64 = 0.04;
/// A move longer than this many ranges grows the range to `|delta| / WARP_FOLD_RATIO`, so one pin never
/// folds the plane (a Gaussian move folds from `|delta| = rho sqrt(e) ~ 1.65 rho`).
pub const WARP_FOLD_RATIO: f64 = 1.5;
/// Chroma below which the warp fades out: `g = smoothstep(0, WARP_NEUTRAL_GUARD, |c|)`.
pub const WARP_NEUTRAL_GUARD: f64 = 0.02;
/// Half the width of a pin's tonal crossover, in LogC4 code values (two stops).
pub const WARP_TONE_HALF_WIDTH: f64 = 2.0 * LOGC4_STOP;
/// A pin's pivot is limited to this many stops either side of middle grey.
pub const WARP_PIVOT_LIMIT_EV: f64 = 6.0;
/// A pin's exposure is limited to this many stops either way.
pub const WARP_EXPOSURE_LIMIT_EV: f64 = 2.0;

/// One pin of the warp, as the settings hold it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarpPin {
    /// The colour the pin takes hold of.
    pub source: ChromaPoint,
    /// Where it moves that colour to. A locked pin ignores it (its target is its source).
    pub target: ChromaPoint,
    /// A locked pin holds the colours within its range in place.
    pub locked: bool,
    /// How far the pin reaches across the plane: the `sigma` of its Gaussian, in chroma units.
    pub range: f64,
    /// How much of the move applies to the shadows, 0 to 1. A locked pin ignores it.
    pub shadows: f64,
    /// How much of the move applies to the highlights, 0 to 1. A locked pin ignores it.
    pub highlights: f64,
    /// Where the shadows end and the highlights begin, in stops from middle grey.
    pub pivot_ev: f64,
    /// A change of brightness for the pin's colours, in stops. A locked pin ignores it.
    pub exposure_ev: f64,
}

impl Default for WarpPin {
    fn default() -> Self {
        Self {
            source: [0.0, 0.0],
            target: [0.0, 0.0],
            locked: false,
            range: WARP_RANGE_DEFAULT,
            shadows: 1.0,
            highlights: 1.0,
            pivot_ev: 0.0,
            exposure_ev: 0.0,
        }
    }
}

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}

/// `p` scaled back onto the disc of radius `limit` when it lies outside; a non-finite coordinate
/// counts as 0.
#[must_use]
pub fn clamp_to_disc(p: ChromaPoint, limit: f64) -> ChromaPoint {
    let p = [finite_or(p[0], 0.0), finite_or(p[1], 0.0)];
    let r = p[0].hypot(p[1]);
    if r > limit {
        [p[0] * limit / r, p[1] * limit / r]
    } else {
        p
    }
}

impl WarpPin {
    /// A moving pin from `source` to `target` with the default range and weights.
    #[must_use]
    pub fn moving(source: ChromaPoint, target: ChromaPoint) -> Self {
        Self {
            source,
            target,
            ..Self::default()
        }
    }

    /// A locked pin at `at` with the default range.
    #[must_use]
    pub fn locked_at(at: ChromaPoint) -> Self {
        Self {
            source: at,
            target: at,
            locked: true,
            ..Self::default()
        }
    }

    /// Every value in its range: points scaled back to [`WARP_COORD_LIMIT`], the range, weights, pivot
    /// and exposure clamped, a non-finite value replaced by its default. A locked pin's target is its
    /// source and its weights, pivot and exposure are the defaults.
    #[must_use]
    pub fn clamped(&self) -> Self {
        let source = clamp_to_disc(self.source, WARP_COORD_LIMIT);
        let range = finite_or(self.range, WARP_RANGE_DEFAULT).clamp(WARP_RANGE_MIN, WARP_RANGE_MAX);
        if self.locked {
            return Self {
                source,
                target: source,
                locked: true,
                range,
                ..Self::default()
            };
        }
        Self {
            source,
            target: clamp_to_disc(self.target, WARP_COORD_LIMIT),
            locked: false,
            range,
            shadows: finite_or(self.shadows, 1.0).clamp(0.0, 1.0),
            highlights: finite_or(self.highlights, 1.0).clamp(0.0, 1.0),
            pivot_ev: finite_or(self.pivot_ev, 0.0)
                .clamp(-WARP_PIVOT_LIMIT_EV, WARP_PIVOT_LIMIT_EV),
            exposure_ev: finite_or(self.exposure_ev, 0.0)
                .clamp(-WARP_EXPOSURE_LIMIT_EV, WARP_EXPOSURE_LIMIT_EV),
        }
    }

    /// Whether the pin changes anything on its own: it moves its colour or changes its exposure.
    /// A locked pin does not (it only holds colours against other pins).
    #[must_use]
    pub fn acts(&self) -> bool {
        !self.locked && (self.target != self.source || self.exposure_ev != 0.0)
    }
}

/// The pins of stage C9b. No pins is the identity.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WarpParams {
    /// At most [`WARP_MAX_PINS`] are used, in order.
    pub pins: Vec<WarpPin>,
}

impl WarpParams {
    /// Whether the stage changes nothing: no pin, after clamping, moves a colour or changes its
    /// exposure (locked pins alone hold colours that nothing moves).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        !self
            .pins
            .iter()
            .take(WARP_MAX_PINS)
            .any(|p| p.clamped().acts())
    }
}

/// The chroma of a LogC4 pixel: `(e1·v, e2·v)`.
#[must_use]
pub fn chroma_of(v: LogC4) -> ChromaPoint {
    [dot(E1, v.0), dot(E2, v.0)]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The Hermite step of WGSL's `smoothstep`: 0 below `e0`, 1 above `e1`.
#[must_use]
pub fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A moving pin, ready to evaluate (also what the GPU kernel's uniform block carries).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreparedMove {
    /// The source.
    pub source: ChromaPoint,
    /// `target - source`.
    pub delta: ChromaPoint,
    /// `1 / (2 rho'^2)` with the range grown for a long move.
    pub inv_two_sigma_sq: f64,
    /// Weight in the shadows.
    pub shadows: f64,
    /// Weight in the highlights.
    pub highlights: f64,
    /// The pivot as a LogC4 code value.
    pub pivot_code: f64,
    /// Exposure in stops.
    pub exposure_ev: f64,
}

/// A locked pin, ready to evaluate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreparedLock {
    /// Where it holds.
    pub at: ChromaPoint,
    /// `1 / (2 rho^2)`.
    pub inv_two_sigma_sq: f64,
}

/// Stage C9b, prepared from [`WarpParams`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Warp {
    moves: Vec<PreparedMove>,
    locks: Vec<PreparedLock>,
}

impl Warp {
    /// Prepares the pins: the first [`WARP_MAX_PINS`], clamped. Pins that neither move nor change
    /// exposure are left out (they would only dilute their neighbours), and so are the locks when no
    /// pin acts.
    #[must_use]
    pub fn new(params: &WarpParams) -> Self {
        let mut moves = Vec::new();
        let mut locks = Vec::new();
        for pin in params.pins.iter().take(WARP_MAX_PINS).map(WarpPin::clamped) {
            if pin.locked {
                locks.push(PreparedLock {
                    at: pin.source,
                    inv_two_sigma_sq: 1.0 / (2.0 * pin.range * pin.range),
                });
            } else if pin.acts() {
                let delta = [pin.target[0] - pin.source[0], pin.target[1] - pin.source[1]];
                let sigma = pin.range.max(delta[0].hypot(delta[1]) / WARP_FOLD_RATIO);
                moves.push(PreparedMove {
                    source: pin.source,
                    delta,
                    inv_two_sigma_sq: 1.0 / (2.0 * sigma * sigma),
                    shadows: pin.shadows,
                    highlights: pin.highlights,
                    pivot_code: logc4_at_stops(pin.pivot_ev),
                    exposure_ev: pin.exposure_ev,
                });
            }
        }
        if moves.is_empty() {
            locks.clear();
        }
        Self { moves, locks }
    }

    /// Whether the stage is skipped (no pin acts).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.moves.is_empty()
    }

    /// The moving pins, in order.
    #[must_use]
    pub fn moves(&self) -> &[PreparedMove] {
        &self.moves
    }

    /// The locked pins, in order (empty when no pin acts).
    #[must_use]
    pub fn locks(&self) -> &[PreparedLock] {
        &self.locks
    }

    /// The displacement of the chroma `c` at the tone `l` (a LogC4 code value): the chroma move `D`
    /// and the exposure `E` in stops.
    #[must_use]
    pub fn displacement(&self, c: ChromaPoint, l: f64) -> (ChromaPoint, f64) {
        if self.moves.is_empty() {
            return ([0.0, 0.0], 0.0);
        }
        let guard = smoothstep(0.0, WARP_NEUTRAL_GUARD, c[0].hypot(c[1]));
        if guard == 0.0 {
            return ([0.0, 0.0], 0.0);
        }
        let mut hold = 1.0;
        for lock in &self.locks {
            let d2 = (c[0] - lock.at[0]).powi(2) + (c[1] - lock.at[1]).powi(2);
            hold *= 1.0 - (-d2 * lock.inv_two_sigma_sq).exp();
        }
        if hold == 0.0 {
            return ([0.0, 0.0], 0.0);
        }
        let (mut reach, mut d0, mut d1, mut exposure) = (0.0, 0.0, 0.0, 0.0);
        for m in &self.moves {
            let d2 = (c[0] - m.source[0]).powi(2) + (c[1] - m.source[1]).powi(2);
            let phi = (-d2 * m.inv_two_sigma_sq).exp();
            reach += phi;
            let t = smoothstep(
                m.pivot_code - WARP_TONE_HALF_WIDTH,
                m.pivot_code + WARP_TONE_HALF_WIDTH,
                l,
            );
            let w = (m.shadows + (m.highlights - m.shadows) * t) * phi;
            d0 += w * m.delta[0];
            d1 += w * m.delta[1];
            exposure += w * m.exposure_ev;
        }
        let k = guard * hold / reach.max(1.0);
        ([k * d0, k * d1], k * exposure)
    }

    /// Applies the warp to one LogC4 pixel.
    #[must_use]
    pub fn apply(&self, v: LogC4) -> LogC4 {
        if self.moves.is_empty() {
            return v;
        }
        let (d, e) = self.displacement(chroma_of(v), v.luma());
        if d == [0.0, 0.0] && e == 0.0 {
            return v;
        }
        let lift = e * LOGC4_STOP;
        LogC4(std::array::from_fn(|k| {
            v.0[k] + d[0] * E1[k] + d[1] * E2[k] + lift
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::LOGC4_GRAY;

    /// A LogC4 pixel with chroma `c` around the neutral `n`.
    fn pixel(n: f64, c: ChromaPoint) -> LogC4 {
        LogC4(std::array::from_fn(|k| n + c[0] * E1[k] + c[1] * E2[k]))
    }

    fn close(a: ChromaPoint, b: ChromaPoint, tol: f64) -> bool {
        (a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol
    }

    #[test]
    fn no_pins_locks_only_and_still_pins_are_the_identity() {
        let v = pixel(0.4, [0.1, -0.05]);
        let cases = [
            WarpParams::default(),
            WarpParams {
                pins: vec![WarpPin::locked_at([0.1, 0.0])],
            },
            WarpParams {
                pins: vec![WarpPin::moving([0.1, 0.0], [0.1, 0.0])],
            },
        ];
        for params in cases {
            assert!(params.is_identity());
            let warp = Warp::new(&params);
            assert!(warp.is_identity());
            assert_eq!(warp.apply(v), v);
        }
    }

    #[test]
    fn a_single_pin_lands_its_source_on_its_target() {
        let (s, t) = ([0.12, 0.03], [0.08, 0.10]);
        let warp = Warp::new(&WarpParams {
            pins: vec![WarpPin::moving(s, t)],
        });
        for n in [0.2, LOGC4_GRAY, 0.5, 0.7] {
            let out = warp.apply(pixel(n, s));
            assert!(close(chroma_of(out), t, 1e-12), "{:?}", chroma_of(out));
            // The channel mean, the neutral-axis brightness, is unchanged.
            let mean = |v: LogC4| (v.0[0] + v.0[1] + v.0[2]) / 3.0;
            assert!((mean(out) - n).abs() < 1e-12);
        }
    }

    #[test]
    fn neutrals_never_move() {
        let warp = Warp::new(&WarpParams {
            pins: vec![WarpPin {
                range: WARP_RANGE_MAX,
                exposure_ev: 1.0,
                ..WarpPin::moving([0.03, 0.0], [0.2, 0.1])
            }],
        });
        for n in [0.1, 0.3, 0.6, 0.9] {
            let v = LogC4([n; 3]);
            assert_eq!(warp.apply(v), v);
        }
    }

    #[test]
    fn a_locked_pin_holds_its_colour_at_every_tone() {
        let lock = [0.10, 0.05];
        let warp = Warp::new(&WarpParams {
            pins: vec![
                WarpPin {
                    range: 0.08,
                    exposure_ev: 0.5,
                    ..WarpPin::moving([0.12, 0.02], [0.02, 0.15])
                },
                WarpPin::locked_at(lock),
            ],
        });
        for n in [0.15, 0.3, 0.5, 0.8] {
            let v = pixel(n, lock);
            assert_eq!(warp.apply(v), v);
        }
        // And a colour away from the lock does move.
        let v = pixel(0.4, [0.12, 0.02]);
        assert_ne!(warp.apply(v), v);
    }

    #[test]
    fn one_pin_does_not_fold_the_plane_even_for_a_long_move() {
        // A move of 0.25 with the smallest range: the range grows to 0.25 / 1.5.
        let warp = Warp::new(&WarpParams {
            pins: vec![WarpPin {
                range: WARP_RANGE_MIN,
                ..WarpPin::moving([0.1, 0.0], [-0.1, 0.15])
            }],
        });
        let map = |c: ChromaPoint| {
            let (d, _) = warp.displacement(c, LOGC4_GRAY);
            [c[0] + d[0], c[1] + d[1]]
        };
        let h = 1e-4;
        let mut worst = f64::INFINITY;
        let n = 120;
        for i in 0..=n {
            for j in 0..=n {
                let c = [
                    -0.3 + 0.6 * f64::from(i) / f64::from(n),
                    -0.3 + 0.6 * f64::from(j) / f64::from(n),
                ];
                if c[0].hypot(c[1]) < WARP_NEUTRAL_GUARD + 1e-3 {
                    continue;
                }
                let (a, b) = (map([c[0] + h, c[1]]), map([c[0] - h, c[1]]));
                let (p, q) = (map([c[0], c[1] + h]), map([c[0], c[1] - h]));
                let j11 = (a[0] - b[0]) / (2.0 * h);
                let j21 = (a[1] - b[1]) / (2.0 * h);
                let j12 = (p[0] - q[0]) / (2.0 * h);
                let j22 = (p[1] - q[1]) / (2.0 * h);
                worst = worst.min(j11 * j22 - j12 * j21);
            }
        }
        assert!(worst > 0.05, "smallest Jacobian determinant {worst}");
    }

    #[test]
    fn tonal_weights_split_shadows_from_highlights() {
        let (s, t) = ([0.1, 0.0], [0.1, 0.1]);
        let warp = Warp::new(&WarpParams {
            pins: vec![WarpPin {
                shadows: 0.0,
                highlights: 1.0,
                ..WarpPin::moving(s, t)
            }],
        });
        // Deep shadows (four stops under grey, more than the two-stop crossover): untouched.
        let dark = pixel(logc4_at_stops(-4.0), s);
        assert!(close(chroma_of(warp.apply(dark)), s, 1e-3));
        // Bright highlights: the full move.
        let bright = pixel(logc4_at_stops(4.0), s);
        assert!(close(chroma_of(warp.apply(bright)), t, 2e-3));
    }

    #[test]
    fn exposure_moves_the_three_channels_together() {
        let s = [0.1, 0.02];
        let warp = Warp::new(&WarpParams {
            pins: vec![WarpPin {
                exposure_ev: 1.0,
                ..WarpPin::moving(s, s)
            }],
        });
        let v = pixel(0.4, s);
        let out = warp.apply(v);
        assert!(close(chroma_of(out), s, 1e-12));
        for k in 0..3 {
            assert!((out.0[k] - v.0[k] - LOGC4_STOP).abs() < 1e-12);
        }
    }

    #[test]
    fn overlapping_pins_share_rather_than_add() {
        let s = [0.1, 0.0];
        let one = Warp::new(&WarpParams {
            pins: vec![WarpPin::moving(s, [0.1, 0.06])],
        });
        let two = Warp::new(&WarpParams {
            pins: vec![
                WarpPin::moving(s, [0.1, 0.06]),
                WarpPin::moving(s, [0.1, 0.06]),
            ],
        });
        let v = pixel(0.4, s);
        assert!(close(
            chroma_of(one.apply(v)),
            chroma_of(two.apply(v)),
            1e-12
        ));
    }

    #[test]
    fn clamping_keeps_every_value_in_range() {
        let pin = WarpPin {
            source: [3.0, 4.0],
            target: [f64::NAN, 0.1],
            locked: false,
            range: 9.0,
            shadows: -1.0,
            highlights: 2.0,
            pivot_ev: 40.0,
            exposure_ev: -9.0,
        }
        .clamped();
        assert!((pin.source[0].hypot(pin.source[1]) - WARP_COORD_LIMIT).abs() < 1e-12);
        assert_eq!(pin.target, [0.0, 0.1]);
        assert_eq!(pin.range, WARP_RANGE_MAX);
        assert_eq!((pin.shadows, pin.highlights), (0.0, 1.0));
        assert_eq!(pin.pivot_ev, WARP_PIVOT_LIMIT_EV);
        assert_eq!(pin.exposure_ev, -WARP_EXPOSURE_LIMIT_EV);
        let locked = WarpPin {
            locked: true,
            exposure_ev: 1.0,
            ..WarpPin::moving([0.1, 0.0], [0.2, 0.0])
        }
        .clamped();
        assert_eq!(locked.target, locked.source);
        assert_eq!(locked.exposure_ev, 0.0);
    }

    #[test]
    fn only_the_first_sixteen_pins_count() {
        let mut pins = vec![WarpPin::locked_at([0.2, 0.2]); WARP_MAX_PINS];
        pins.push(WarpPin::moving([0.1, 0.0], [0.0, 0.1]));
        assert!(WarpParams { pins }.is_identity());
    }

    #[test]
    fn red_lies_along_plus_e1_and_green_and_blue_a_third_of_a_turn_either_side() {
        let hue = |v: LogC4| {
            let c = chroma_of(v);
            c[1].atan2(c[0]).to_degrees().rem_euclid(360.0)
        };
        assert!(hue(LogC4([0.5, 0.2, 0.2])).abs() < 1e-9);
        assert!((hue(LogC4([0.2, 0.5, 0.2])) - 120.0).abs() < 1e-9);
        assert!((hue(LogC4([0.2, 0.2, 0.5])) - 240.0).abs() < 1e-9);
    }
}
