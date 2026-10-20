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
//! A pixel does not jump to its new colour in one move: it follows a flow of `N` small steps while
//! every moving pin's Gaussian slides from its source toward its target, so a pin's reach is the
//! range the person set however far the pin is dragged, and one pin never folds the plane (the fix
//! of 2026-10-19; before it, a long move grew the range to `|delta| / 1.5`). A pixel that takes only
//! a share `gamma w` of a pin's move (its tonal weight, the neutral guard, the locks) sees that pin's
//! centre slide by that share of the move, so a colour on the source lands on `s + gamma w delta`, as
//! in one step. For a pixel of chroma `c0` and tone `L`, the moving pins `i` (source `s`, target `t`,
//! range `rho`, tonal weights `w_lo`, `w_hi` around the pivot `P`, exposure `x` in stops) and the
//! locked pins `k` (at `l`, range `rho_k`):
//!
//! ```text
//! delta_i = t_i - s_i
//! G(d2, rho) = exp(-min(d2 / (2 rho^2), 80))
//! w_i     = w_lo + (w_hi - w_lo) smoothstep(P_i - 2S, P_i + 2S, L)
//! lambda  = prod_k (1 - G(|c0 - l_k|^2, rho_k))
//! g       = smoothstep(0, 0.02, |c0|)
//! gamma   = g lambda                                     at the pixel's own chroma, for every step
//! N       = clamp(ceil(1.25 sum_i |delta_i| / (rho_i sqrt(e))), 1, 64)   one number for the frame
//! c_0     = c0, and for k = 0 .. N-1, tau_k = k / N:
//!   phi_i   = G(|c_k - (s_i + tau_k gamma w_i delta_i)|^2, rho_i)
//!   c_{k+1} = c_k + (gamma / N) sum(w_i phi_i delta_i) / max(1, sum phi_i)
//! D       = c_N - c0
//! E       = gamma sum(w_i G(|c0 - s_i|^2, rho_i) x_i) / max(1, sum G(|c0 - s_i|^2, rho_i))
//! v'      = v + D1 e1 + D2 e2 + E S
//! ```
//!
//! Why it does not fold: at one tone, away from neutral and locks (`gamma = 1`), every pixel follows
//! the same flow, that of one pin moving `w delta`. Its field `w delta phi(c)` has the Lipschitz
//! constant `w |delta| / (rho sqrt(e))` (the steepest slope of a Gaussian, at one range from its
//! centre), and an Euler step `c -> c + V(c) / N` is one to one with a positive Jacobian determinant
//! when `Lip(V) / N < 1`; a chain of such steps is one to one too. The rule for `N` keeps
//! `Lip(V) / N <= 1 / 1.25` for every pin. The cap of 64 never binds that bound for one pin (sources
//! and targets lie within 0.5 of neutral, so `|delta| <= 1`, and `rho >= 0.01` give at most 60.7), but
//! it can for many long, narrow moves at once; then the plane can fold.
//!
//! # What the tests check
//!
//! * No pins, locked pins only, or pins that neither move nor change exposure: the identity, exactly.
//! * One pin, away from neutral and locks, with tonal weights of 1: its source lands on its target
//!   (to rounding), however long the move; with a weight or a lock's hold below 1, on that share of
//!   the move.
//! * Range: a colour three ranges beside or behind the source moves by at most `exp(-4.5)` of the
//!   move, even for a long move (the grown range moved it by most of the move).
//! * One pin never folds the plane: on a dense grid around the move every mapped cell keeps its
//!   orientation and the Jacobian determinant stays positive, for random pins up to `|delta| = 1` with
//!   `rho = 0.01`.
//! * A short move (`N = 1`) is the one-step formula, bit for bit.
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
/// The safety factor of the number of flow steps: `N = ceil(WARP_STEP_SAFETY sum |delta| / (rho sqrt(e)))`,
/// so each step moves a pixel by at most `1 / WARP_STEP_SAFETY` of what would fold the plane.
pub const WARP_STEP_SAFETY: f64 = 1.25;
/// The most flow steps a frame takes. One pin never needs more than 61 (`|delta| <= 1`, `rho >= 0.01`).
pub const WARP_MAX_STEPS: u32 = 64;
/// Every Gaussian's exponent is limited to this, so an absurd chroma gives `exp(-80)`, a normal f32,
/// and never `exp(-inf)` (`docs/17` K8, Metal's fast-math).
pub const WARP_EXP_LIMIT: f64 = 80.0;
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
    /// `1 / (2 rho^2)` with the pin's own range.
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

impl PreparedMove {
    /// The pin's tonal weight at the tone `l` (a LogC4 code value): Shadows below the pivot,
    /// Highlights above it, a smooth crossover of two stops either side.
    fn tonal_weight(&self, l: f64) -> f64 {
        let t = smoothstep(
            self.pivot_code - WARP_TONE_HALF_WIDTH,
            self.pivot_code + WARP_TONE_HALF_WIDTH,
            l,
        );
        self.shadows + (self.highlights - self.shadows) * t
    }
}

/// A locked pin, ready to evaluate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreparedLock {
    /// Where it holds.
    pub at: ChromaPoint,
    /// `1 / (2 rho^2)`.
    pub inv_two_sigma_sq: f64,
}

/// A Gaussian `exp(-d2 inv_two_sigma_sq)` with its exponent limited to [`WARP_EXP_LIMIT`].
fn gauss(d2: f64, inv_two_sigma_sq: f64) -> f64 {
    (-(d2 * inv_two_sigma_sq).min(WARP_EXP_LIMIT)).exp()
}

fn dist2(a: ChromaPoint, b: ChromaPoint) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

/// The number of flow steps for moving pins of these ranges and moves:
/// `clamp(ceil(WARP_STEP_SAFETY sum |delta| / (rho sqrt(e))), 1, WARP_MAX_STEPS)` (module notes).
/// Pins that only change exposure (`delta = 0`) add nothing.
#[must_use]
pub fn flow_steps(moves: impl IntoIterator<Item = (ChromaPoint, f64)>) -> u32 {
    let sqrt_e = std::f64::consts::E.sqrt();
    let lipschitz: f64 = moves
        .into_iter()
        .map(|(delta, range)| (delta[0] * delta[0] + delta[1] * delta[1]).sqrt() / (range * sqrt_e))
        .sum();
    let n = (WARP_STEP_SAFETY * lipschitz).ceil();
    if n.is_nan() {
        // Unreachable for clamped pins (finite moves, ranges of at least WARP_RANGE_MIN).
        return WARP_MAX_STEPS;
    }
    // In [1, WARP_MAX_STEPS], so the cast is exact.
    n.clamp(1.0, f64::from(WARP_MAX_STEPS)) as u32
}

/// Stage C9b, prepared from [`WarpParams`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Warp {
    moves: Vec<PreparedMove>,
    locks: Vec<PreparedLock>,
    /// Flow steps (0 when no pin acts).
    steps: u32,
}

impl Warp {
    /// Prepares the pins: the first [`WARP_MAX_PINS`], clamped. Pins that neither move nor change
    /// exposure are left out (they would only dilute their neighbours), and so are the locks when no
    /// pin acts. The number of flow steps is [`flow_steps`] of the moving pins.
    #[must_use]
    pub fn new(params: &WarpParams) -> Self {
        let mut moves = Vec::new();
        let mut locks = Vec::new();
        let mut reach = Vec::new();
        for pin in params.pins.iter().take(WARP_MAX_PINS).map(WarpPin::clamped) {
            if pin.locked {
                locks.push(PreparedLock {
                    at: pin.source,
                    inv_two_sigma_sq: 1.0 / (2.0 * pin.range * pin.range),
                });
            } else if pin.acts() {
                let delta = [pin.target[0] - pin.source[0], pin.target[1] - pin.source[1]];
                reach.push((delta, pin.range));
                moves.push(PreparedMove {
                    source: pin.source,
                    delta,
                    inv_two_sigma_sq: 1.0 / (2.0 * pin.range * pin.range),
                    shadows: pin.shadows,
                    highlights: pin.highlights,
                    pivot_code: logc4_at_stops(pin.pivot_ev),
                    exposure_ev: pin.exposure_ev,
                });
            }
        }
        if moves.is_empty() {
            return Self::default();
        }
        Self {
            moves,
            locks,
            steps: flow_steps(reach),
        }
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

    /// The number of flow steps every pixel takes: 1 to [`WARP_MAX_STEPS`], 0 when no pin acts.
    #[must_use]
    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// The displacement of the chroma `c` at the tone `l` (a LogC4 code value): the chroma move `D`
    /// and the exposure `E` in stops (module notes).
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
            hold *= 1.0 - gauss(dist2(c, lock.at), lock.inv_two_sigma_sq);
        }
        if hold == 0.0 {
            return ([0.0, 0.0], 0.0);
        }
        let gamma = guard * hold;
        // Each pin's tonal weight `w`, and the share of its move this pixel takes, `gamma w`.
        let mut weights = [0.0; WARP_MAX_PINS];
        let mut shares = [0.0; WARP_MAX_PINS];
        for ((w, b), m) in weights.iter_mut().zip(&mut shares).zip(&self.moves) {
            *w = m.tonal_weight(l);
            *b = gamma * *w;
        }
        // The exposure, at the pixel's own chroma with the pins at their sources.
        let (mut reach, mut exposure) = (0.0, 0.0);
        for (m, w) in self.moves.iter().zip(weights) {
            let phi = gauss(dist2(c, m.source), m.inv_two_sigma_sq);
            reach += phi;
            exposure += w * phi * m.exposure_ev;
        }
        let e = gamma / reach.max(1.0) * exposure;
        // The flow: each pin's centre slides from its source by the pixel's share of its move,
        // `s + tau gamma w delta`, and the pixel follows in N steps (so a pixel on a source lands on
        // `s + gamma w delta`, as in one step). The move is gathered in `d`, so one step is the
        // one-step formula bit for bit.
        let n = f64::from(self.steps);
        let h = gamma / n;
        let mut d = [0.0, 0.0];
        for k in 0..self.steps {
            let tau = f64::from(k) / n;
            let p = [c[0] + d[0], c[1] + d[1]];
            let (mut reach, mut v0, mut v1) = (0.0, 0.0, 0.0);
            for ((m, w), b) in self.moves.iter().zip(weights).zip(shares) {
                let slide = tau * b;
                let centre = [
                    m.source[0] + slide * m.delta[0],
                    m.source[1] + slide * m.delta[1],
                ];
                let phi = gauss(dist2(p, centre), m.inv_two_sigma_sq);
                reach += phi;
                let wphi = w * phi;
                v0 += wphi * m.delta[0];
                v1 += wphi * m.delta[1];
            }
            let s = h / reach.max(1.0);
            d[0] += s * v0;
            d[1] += s * v1;
        }
        (d, e)
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
    fn a_single_pin_lands_its_source_on_its_target_however_long_the_move() {
        for (s, t, range) in [
            ([0.12, 0.03], [0.08, 0.10], WARP_RANGE_DEFAULT),
            ([0.1, 0.0], [0.1, 0.3], WARP_RANGE_MIN),
            ([0.3, 0.4], [-0.3, -0.4], WARP_RANGE_MIN),
            ([-0.2, 0.1], [0.25, -0.05], 0.12),
        ] {
            let warp = Warp::new(&WarpParams {
                pins: vec![WarpPin {
                    range,
                    ..WarpPin::moving(s, t)
                }],
            });
            for n in [0.2, LOGC4_GRAY, 0.5, 0.7] {
                let out = warp.apply(pixel(n, s));
                assert!(close(chroma_of(out), t, 1e-12), "{:?}", chroma_of(out));
                // The channel mean, the neutral-axis brightness, is unchanged.
                let mean = |v: LogC4| (v.0[0] + v.0[1] + v.0[2]) / 3.0;
                assert!((mean(out) - n).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn the_number_of_steps_follows_the_moves_and_ranges() {
        let steps = |pins: Vec<WarpPin>| Warp::new(&WarpParams { pins }).steps();
        let pin = |s: ChromaPoint, t: ChromaPoint, range: f64| WarpPin {
            range,
            ..WarpPin::moving(s, t)
        };
        // No pin that acts: no steps. A move of up to 1.32 ranges: one step, the one-step formula.
        assert_eq!(steps(vec![]), 0);
        assert_eq!(steps(vec![WarpPin::locked_at([0.1, 0.0])]), 0);
        assert_eq!(steps(vec![pin([0.1, 0.0], [0.1, 0.05], 0.04)]), 1);
        assert_eq!(
            steps(vec![WarpPin {
                exposure_ev: 1.0,
                ..WarpPin::moving([0.1, 0.0], [0.1, 0.0])
            }]),
            1
        );
        // A move of 0.1 at the default range: ceil(1.25 * 0.1 / (0.04 sqrt(e))) = ceil(1.90) = 2.
        assert_eq!(steps(vec![pin([0.1, 0.0], [0.1, 0.1], 0.04)]), 2);
        // A move of 0.2 at range 1: ceil(15.2) = 16.
        assert_eq!(steps(vec![pin([0.1, 0.0], [0.1, 0.2], 0.01)]), 16);
        // The longest move at the smallest range needs 61 for one-to-one, 76 with the safety factor:
        // the cap of 64 still keeps one pin one to one.
        assert_eq!(
            steps(vec![pin([0.5, 0.0], [-0.5, 0.0], 0.01)]),
            WARP_MAX_STEPS
        );
        assert!(f64::from(WARP_MAX_STEPS) > 1.0 / (WARP_RANGE_MIN * std::f64::consts::E.sqrt()));
        // Pins add up: four moves of 0.1 at the default range take ceil(4 * 1.52 * 1.25) = 8.
        let four = (0..4)
            .map(|i| {
                let a = f64::from(i);
                pin(
                    [0.1 * a.cos(), 0.1 * a.sin()],
                    [0.1 * a.cos(), 0.1 * a.sin() + 0.1],
                    0.04,
                )
            })
            .collect();
        assert_eq!(steps(four), 8);
    }

    #[test]
    fn a_short_move_is_the_one_step_formula_bit_for_bit() {
        // Moves of 0.022 and 0.02 at the default range take one step together: D = g lambda
        // sum(w phi delta) / n with phi at the sources, as the warp was before the flow.
        let pins = vec![
            WarpPin {
                shadows: 0.3,
                exposure_ev: 0.4,
                ..WarpPin::moving([0.1, 0.02], [0.12, 0.03])
            },
            WarpPin::moving([0.05, -0.1], [0.05, -0.08]),
            WarpPin::locked_at([-0.1, 0.05]),
        ];
        let warp = Warp::new(&WarpParams { pins });
        assert_eq!(warp.steps(), 1);
        let one_step = |c: ChromaPoint, l: f64| {
            let guard = smoothstep(0.0, WARP_NEUTRAL_GUARD, c[0].hypot(c[1]));
            let mut hold = 1.0;
            for lock in warp.locks() {
                hold *= 1.0 - gauss(dist2(c, lock.at), lock.inv_two_sigma_sq);
            }
            let (mut reach, mut d0, mut d1, mut x) = (0.0, 0.0, 0.0, 0.0);
            for m in warp.moves() {
                let phi = gauss(dist2(c, m.source), m.inv_two_sigma_sq);
                reach += phi;
                let w = m.tonal_weight(l) * phi;
                d0 += w * m.delta[0];
                d1 += w * m.delta[1];
                x += w * m.exposure_ev;
            }
            let k = guard * hold / reach.max(1.0);
            ([k * d0, k * d1], k * x)
        };
        for (i, c) in [
            [0.1, 0.02],
            [0.08, 0.0],
            [0.03, -0.05],
            [-0.07, 0.03],
            [0.011, 0.0],
        ]
        .into_iter()
        .enumerate()
        {
            let l = logc4_at_stops(f64::from(i as u8) - 2.0);
            assert_eq!(warp.displacement(c, l), one_step(c, l), "{c:?}");
        }
    }

    #[test]
    fn range_limits_the_reach_even_for_a_long_move() {
        // The owner's report of 2026-10-19: with Range 1 a long drag moved most of the picture,
        // because the range grew to |delta| / 1.5. Now a colour three ranges beside or behind the
        // source moves by at most exp(-4.5) of the move (by exactly that for a short move); before,
        // it moved by most of the move.
        let bound = (-4.5f64).exp();
        for range in [WARP_RANGE_MIN, WARP_RANGE_DEFAULT, 0.12] {
            for (s, t) in [
                ([0.1, 0.0], [0.1, 0.1]),
                ([0.1, 0.0], [0.1, 0.3]),
                ([0.08, 0.06], [-0.2, 0.2]),
                ([0.0, -0.15], [0.3, 0.2]),
            ] {
                let warp = Warp::new(&WarpParams {
                    pins: vec![WarpPin {
                        range,
                        ..WarpPin::moving(s, t)
                    }],
                });
                let delta: ChromaPoint = [t[0] - s[0], t[1] - s[1]];
                let len = delta[0].hypot(delta[1]);
                let u = [delta[0] / len, delta[1] / len];
                // Beside (both sides), behind, and the directions between.
                for angle in [90.0f64, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0] {
                    let (sin, cos) = angle.to_radians().sin_cos();
                    let dir = [u[0] * cos - u[1] * sin, u[0] * sin + u[1] * cos];
                    let c = [s[0] + 3.0 * range * dir[0], s[1] + 3.0 * range * dir[1]];
                    if c[0].hypot(c[1]) < WARP_NEUTRAL_GUARD {
                        continue;
                    }
                    let (d, _) = warp.displacement(c, LOGC4_GRAY);
                    let moved = d[0].hypot(d[1]);
                    assert!(
                        moved <= bound * len * (1.0 + 1e-9),
                        "range {range}, {s:?} -> {t:?}, {angle} deg: moved {moved:.2e}, bound {:.2e}",
                        bound * len
                    );
                    // What the grown range did: phi = exp(-9 rho^2 / (2 rho'^2)) of the move.
                    let grown = range.max(len / 1.5);
                    let before = (-9.0 * range * range / (2.0 * grown * grown)).exp() * len;
                    if len > 1.5 * range * 3.0 {
                        assert!(moved < 0.05 * before, "{moved:.2e} against {before:.2e}");
                    }
                }
            }
        }
    }

    /// splitmix64: a seeded stream for the random pins of the folding test.
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

        /// A point of the disc of radius `r`.
        fn in_disc(&mut self, r: f64) -> ChromaPoint {
            let (a, q) = (self.next() * std::f64::consts::TAU, r * self.next().sqrt());
            [q * a.cos(), q * a.sin()]
        }
    }

    /// The smallest Jacobian determinant of `c -> c + D(c)` and the smallest ratio of a mapped
    /// cell's signed area to its own, on a grid of `range / 4` along and across the move, out to six
    /// ranges around it (farther out every pixel moves by less than `exp(-18)` of the move). Pixels
    /// within the neutral guard are left out: their weight `g` is below 1, so they follow the flow
    /// more slowly than their neighbours (as before the flow, the guard trades the no-folding for
    /// neutrals that never move).
    fn worst_fold(warp: &Warp, s: ChromaPoint, t: ChromaPoint, range: f64) -> (f64, f64) {
        let map = |c: ChromaPoint| {
            let (d, _) = warp.displacement(c, LOGC4_GRAY);
            [c[0] + d[0], c[1] + d[1]]
        };
        let delta = [t[0] - s[0], t[1] - s[1]];
        let len = delta[0].hypot(delta[1]);
        let u = [delta[0] / len, delta[1] / len];
        let v = [-u[1], u[0]];
        let step = range / 4.0;
        let along = ((len + 12.0 * range) / step).ceil() as usize;
        let across = 48usize;
        let at = |i: usize, j: usize| {
            let a = -6.0 * range + step * i as f64;
            let b = -6.0 * range + step * j as f64;
            [s[0] + a * u[0] + b * v[0], s[1] + a * u[1] + b * v[1]]
        };
        let outside = |c: ChromaPoint| c[0].hypot(c[1]) > WARP_NEUTRAL_GUARD + 1e-3;
        let mapped: Vec<Vec<Option<ChromaPoint>>> = (0..=along)
            .map(|i| {
                (0..=across)
                    .map(|j| {
                        let c = at(i, j);
                        outside(c).then(|| map(c))
                    })
                    .collect()
            })
            .collect();
        let cross = |o: ChromaPoint, a: ChromaPoint, b: ChromaPoint| {
            (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        };
        let (mut det, mut area) = (f64::INFINITY, f64::INFINITY);
        let cell = step * step / 2.0;
        let h = range * 1e-5;
        for i in 0..along {
            for j in 0..across {
                let corners = [
                    mapped[i][j],
                    mapped[i + 1][j],
                    mapped[i + 1][j + 1],
                    mapped[i][j + 1],
                ];
                let [Some(a), Some(b), Some(c), Some(d)] = corners else {
                    continue;
                };
                // The cell's two triangles, counter-clockwise in (along, across) before the map.
                area = area.min(cross(a, b, c) / cell).min(cross(a, c, d) / cell);
                let p = at(i, j);
                let (e, w) = (map([p[0] + h, p[1]]), map([p[0] - h, p[1]]));
                let (n, so) = (map([p[0], p[1] + h]), map([p[0], p[1] - h]));
                let j11 = (e[0] - w[0]) / (2.0 * h);
                let j21 = (e[1] - w[1]) / (2.0 * h);
                let j12 = (n[0] - so[0]) / (2.0 * h);
                let j22 = (n[1] - so[1]) / (2.0 * h);
                det = det.min(j11 * j22 - j12 * j21);
            }
        }
        (det, area)
    }

    #[test]
    fn one_pin_never_folds_the_plane() {
        // Fixed extremes (the longest moves at the smallest range, a move through neutral, the
        // largest range), then seeded random pins: sources and targets anywhere on the plane
        // (|delta| up to 1), ranges spread evenly in log between 0.01 and 0.25, tonal weights below 1.
        let mut cases = vec![
            ([0.5, 0.0], [-0.5, 0.0], WARP_RANGE_MIN, 1.0),
            ([0.3, 0.4], [-0.3, -0.4], WARP_RANGE_MIN, 1.0),
            ([0.1, 0.0], [0.1, 0.3], WARP_RANGE_MIN, 1.0),
            ([0.1, 0.0], [0.45, 0.2], WARP_RANGE_MIN, 0.6),
            ([0.4, -0.3], [-0.4, 0.3], WARP_RANGE_MAX, 1.0),
        ];
        let mut rng = Rng(0xC9B0_2026_1019);
        for _ in 0..40 {
            let s = rng.in_disc(WARP_COORD_LIMIT);
            let t = rng.in_disc(WARP_COORD_LIMIT);
            let range = WARP_RANGE_MIN * (WARP_RANGE_MAX / WARP_RANGE_MIN).powf(rng.next());
            let weight = 0.2 + 0.8 * rng.next();
            cases.push((s, t, range, weight));
        }
        let (mut worst_det, mut worst_area) = (f64::INFINITY, f64::INFINITY);
        for (s, t, range, weight) in cases {
            let warp = Warp::new(&WarpParams {
                pins: vec![WarpPin {
                    range,
                    shadows: weight,
                    highlights: weight,
                    ..WarpPin::moving(s, t)
                }],
            });
            // Each step stays one to one: h Lip(V) = |delta| / (N rho sqrt(e)) < 1.
            let len = (t[0] - s[0]).hypot(t[1] - s[1]);
            let lip = len / (f64::from(warp.steps()) * range * std::f64::consts::E.sqrt());
            assert!(lip < 0.95, "{s:?} -> {t:?}, range {range}: {lip}");
            let (det, area) = worst_fold(&warp, s, t, range);
            assert!(
                det > 0.0 && area > 0.0,
                "{s:?} -> {t:?}, range {range}: Jacobian {det:.3e}, cell {area:.3e}"
            );
            worst_det = worst_det.min(det);
            worst_area = worst_area.min(area);
        }
        println!(
            "45 single pins: smallest Jacobian determinant {worst_det:.3e}, smallest mapped cell {worst_area:.3e} of its own area"
        );
        // Not vacuous: the same move in one step with the range as set (no flow, no grown range)
        // folds the plane, and the test sees it.
        let (s, t) = ([0.1, 0.0], [0.1, 0.1]);
        let mut one_jump = Warp::new(&WarpParams {
            pins: vec![WarpPin {
                range: WARP_RANGE_DEFAULT,
                ..WarpPin::moving(s, t)
            }],
        });
        one_jump.steps = 1;
        let (det, area) = worst_fold(&one_jump, s, t, WARP_RANGE_DEFAULT);
        assert!(det < 0.0 && area < 0.0, "{det}, {area}");
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
    fn a_partial_weight_moves_the_source_by_that_share_of_a_long_move() {
        // Shadows and Highlights at 0.4, and a lock whose hold at the source is 1 - exp(-1/2): the
        // colour on the source takes that share of the move, as in one step (a flow whose centres
        // slid the whole way would leave it behind after a few hundredths).
        let (s, t) = ([0.1, 0.0], [0.1, 0.3]);
        let weighted = Warp::new(&WarpParams {
            pins: vec![WarpPin {
                range: WARP_RANGE_MIN,
                shadows: 0.4,
                highlights: 0.4,
                ..WarpPin::moving(s, t)
            }],
        });
        assert_eq!(weighted.steps(), 23);
        let out = chroma_of(weighted.apply(pixel(0.4, s)));
        assert!(close(out, [0.1, 0.4 * 0.3], 1e-12), "{out:?}");
        let held = Warp::new(&WarpParams {
            pins: vec![
                WarpPin {
                    range: WARP_RANGE_MIN,
                    ..WarpPin::moving(s, t)
                },
                WarpPin {
                    range: 0.04,
                    ..WarpPin::locked_at([0.14, 0.0])
                },
            ],
        });
        let hold = 1.0 - (-0.5f64).exp();
        let out = chroma_of(held.apply(pixel(0.4, s)));
        assert!(close(out, [0.1, hold * 0.3], 1e-12), "{out:?}");
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
        // Two identical pins take their colour where one takes it (not twice as far), for a short
        // move (one step) and a long one (the two take twice the steps of one).
        let s = [0.1, 0.0];
        for (t, range) in [
            ([0.1, 0.06], WARP_RANGE_DEFAULT),
            ([0.1, 0.3], WARP_RANGE_MIN),
        ] {
            let pin = WarpPin {
                range,
                ..WarpPin::moving(s, t)
            };
            let one = Warp::new(&WarpParams { pins: vec![pin] });
            let two = Warp::new(&WarpParams {
                pins: vec![pin, pin],
            });
            let v = pixel(0.4, s);
            assert!(close(chroma_of(one.apply(v)), t, 1e-12));
            assert!(close(chroma_of(two.apply(v)), t, 1e-12));
        }
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
