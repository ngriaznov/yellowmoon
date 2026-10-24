//! The re-blur gradient ratio of a disc-defocused edge (`docs/21` section 7, step 1).
//!
//! Zhuo and Sim (2011) read the blur of an edge from the ratio `rho` of the gradient magnitudes of the image and
//! of the image blurred once more, for an edge modelled as a step blurred by a Gaussian. A lens blurs with a
//! disc, so this module computes the ratio the paper's idea gives for a step edge blurred by a *disc* of radius
//! `r`, whose line spread function is the half circle `(2 / (pi r^2)) sqrt(r^2 - x^2)`: the peak gradient of the
//! edge seen through a Gaussian of `s` pixels is `g(s; r) = integral LSF_r(x) N(x; s) dx`, and the ratio between
//! two levels `j` and `k` of the scale ladder is `g(s_j; r) / g(s_k; r)`. The tables here invert it:
//! `ratio -> r`. A constant native blur (the sensor's binned pixel, the gradient operator) is added to every
//! Gaussian in quadrature, so that a sharp edge of the photo reads as a disc of well under one pixel.
//!
//! Written from the paper's idea and the geometry of a disc; no code of any other implementation is used.

use std::f64::consts::PI;
use std::sync::OnceLock;

/// Number of levels of the scale ladder: level 0 is the plane itself, level `m >= 1` is blurred by `2^(m - 1)`
/// pixels (1, 2, 4, 8, 16).
pub(super) const LEVELS: usize = 6;

/// The Gaussian of each ladder level, in pixels (level 0 is the raw plane).
pub(super) fn level_sigma(m: usize) -> f64 {
    const SIGMAS: [f64; LEVELS] = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0];
    SIGMAS[m]
}

/// The pairs of ladder levels `(j, k)` whose gradient ratios give the blur, each one octave apart in
/// Gaussian width (0 to 1, 1 to 2, 2 to 4, 4 to 8, 8 to 16 pixels).
pub(super) const PAIRS: [(usize, usize); 5] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];

/// Blur of the photo's own optics and sampling, in pixels, added in quadrature to every Gaussian of the model.
///
/// A step edge on the binned lens plane, seen through the central-difference gradient, has an apparent Gaussian
/// width of about this much; it is measured on the synthetic scenes of `tests/detector.rs` (an ideal step with
/// pixel-area coverage).
pub(super) const NATIVE_SIGMA: f64 = 0.6;

/// Smallest and largest disc radius of the tables, in pixels.
const R_MIN: f64 = 0.15;
const R_MAX: f64 = 80.0;
/// Grid points of the tables (geometric in `r`).
const GRID: usize = 160;
/// Midpoint samples of the line spread integral.
const STEPS: usize = 512;

/// The peak gradient of an edge blurred by a disc of radius `r` and a Gaussian of `s` pixels.
pub(super) fn peak_gradient(s: f64, r: f64) -> f64 {
    let norm = 1.0 / (s * (2.0 * PI).sqrt());
    let mut acc = 0.0;
    for i in 0..STEPS {
        let theta = -PI / 2.0 + PI * (i as f64 + 0.5) / STEPS as f64;
        let (sin, cos) = (libm::sin(theta), libm::cos(theta));
        let x = r * sin;
        acc += cos * cos * libm::exp(-0.5 * x * x / (s * s));
    }
    (2.0 / PI) * acc * (PI / STEPS as f64) * norm
}

/// One pair's table: `rho` falls as `r` grows.
struct PairTable {
    ln_r: Vec<f64>,
    rho: Vec<f64>,
}

fn tables() -> &'static Vec<PairTable> {
    static TABLES: OnceLock<Vec<PairTable>> = OnceLock::new();
    TABLES.get_or_init(|| {
        PAIRS
            .iter()
            .map(|&(j, k)| {
                let sj = (level_sigma(j).powi(2) + NATIVE_SIGMA * NATIVE_SIGMA).sqrt();
                let sk = (level_sigma(k).powi(2) + NATIVE_SIGMA * NATIVE_SIGMA).sqrt();
                let mut ln_r = Vec::with_capacity(GRID);
                let mut rho = Vec::with_capacity(GRID);
                for i in 0..GRID {
                    let t = i as f64 / (GRID - 1) as f64;
                    let l = libm::log(R_MIN) + t * (libm::log(R_MAX) - libm::log(R_MIN));
                    let r = libm::exp(l);
                    ln_r.push(l);
                    rho.push(peak_gradient(sj, r) / peak_gradient(sk, r));
                }
                PairTable { ln_r, rho }
            })
            .collect()
    })
}

/// A disc radius read from a gradient ratio.
#[derive(Clone, Copy, Debug)]
pub(super) struct RadiusEstimate {
    /// The disc radius in pixels.
    pub r: f64,
    /// `|d ln r / d ln rho|`: how much a relative error of the ratio becomes in the radius.
    pub sensitivity: f64,
}

/// The disc radius whose gradient ratio between the ladder levels of `PAIRS[pair]` is `rho`.
///
/// `None` when the ratio is too close to 1 to tell (the edge is much wider than the pair can resolve); a ratio
/// above the table's range reads as the smallest radius.
pub(super) fn radius_from_ratio(pair: usize, rho: f64) -> Option<RadiusEstimate> {
    let t = &tables()[pair];
    let n = t.rho.len();
    if !(rho.is_finite()) || rho <= t.rho[n - 1] {
        return None;
    }
    if rho >= t.rho[0] {
        let s = ((t.ln_r[1] - t.ln_r[0]) / (libm::log(t.rho[0]) - libm::log(t.rho[1]))).abs();
        return Some(RadiusEstimate {
            r: libm::exp(t.ln_r[0]),
            sensitivity: s,
        });
    }
    // `rho` falls with the index: find i with rho[i] >= rho > rho[i + 1].
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if t.rho[mid] >= rho {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let f = (t.rho[lo] - rho) / (t.rho[lo] - t.rho[hi]);
    let ln_r = t.ln_r[lo] + f * (t.ln_r[hi] - t.ln_r[lo]);
    let dln_rho = (libm::log(t.rho[lo]) - libm::log(t.rho[hi])).abs().max(1e-12);
    Some(RadiusEstimate {
        r: libm::exp(ln_r),
        sensitivity: (t.ln_r[hi] - t.ln_r[lo]).abs() / dln_rho,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ratio_falls_with_the_radius_for_every_pair() {
        for t in tables() {
            for w in t.rho.windows(2) {
                assert!(w[1] < w[0] + 1e-12, "{w:?}");
            }
            assert!(t.rho[0] > 1.0 && *t.rho.last().unwrap() >= 1.0);
        }
    }

    #[test]
    fn a_ratio_round_trips_to_its_radius() {
        for (pair, &(j, k)) in PAIRS.iter().enumerate() {
            let sj = (level_sigma(j).powi(2) + NATIVE_SIGMA * NATIVE_SIGMA).sqrt();
            let sk = (level_sigma(k).powi(2) + NATIVE_SIGMA * NATIVE_SIGMA).sqrt();
            for r in [0.5, 1.5, 3.0, 6.0, 12.0, 24.0] {
                let rho = peak_gradient(sj, r) / peak_gradient(sk, r);
                if let Some(e) = radius_from_ratio(pair, rho) {
                    // Where the pair resolves the radius (a small sensitivity) the inversion is exact.
                    if e.sensitivity < 6.0 && r > 0.2 {
                        assert!((e.r / r - 1.0).abs() < 0.03, "pair {pair} r {r}: {e:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_peak_gradient_of_a_tiny_disc_is_the_gaussian_one() {
        let s = 2.0;
        let g = peak_gradient(s, 1e-3);
        assert!((g - 1.0 / (s * (2.0 * PI).sqrt())).abs() < 1e-6);
        // A wide disc with no Gaussian is the half circle's peak 2 / (pi r).
        let r = 20.0;
        let g = peak_gradient(0.05, r);
        assert!((g * PI * r / 2.0 - 1.0).abs() < 0.02, "{g}");
    }

    #[test]
    fn a_sharp_edge_reads_below_one_pixel_and_a_wide_one_is_not_resolved_by_the_first_pair() {
        // rho of the ideal sharp edge: the native blur alone.
        let rho = (1.0 + NATIVE_SIGMA * NATIVE_SIGMA).sqrt() / NATIVE_SIGMA;
        let e = radius_from_ratio(0, rho).unwrap();
        assert!(e.r < 0.5, "{e:?}");
        // A disc of 12 px seen with the first pair is nearly 1.
        let sj = NATIVE_SIGMA;
        let sk = (1.0 + NATIVE_SIGMA * NATIVE_SIGMA).sqrt();
        let rho = peak_gradient(sj, 12.0) / peak_gradient(sk, 12.0);
        assert!(rho < 1.03, "{rho}");
    }
}
