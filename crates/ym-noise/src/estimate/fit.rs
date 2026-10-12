//! Weighted least-squares MAP fit of `(a, b, delta_c)` to the bin statistics
//! (`docs/06-noise.md` step 5).
//!
//! Each data point is a variance estimate `v_i` of a group of Haar detail coefficients (or of a
//! patch ensemble) at mean intensity `mu_i`, for one CFA plane or for all planes pooled. In terms of
//! `beta_c = b - a delta_c` the model is linear:
//!
//! ```text
//! v_i = a * mu_i + beta_{c(i)}            (pooled points use the mean of the four beta)
//! ```
//!
//! The variance of `v_i` is `rel_var_i * v_i^2` (a MAD-based variance has a relative variance of
//! about 5.5 / n), so the data weights are `1 / (rel_var_i * pred_i^2)`, iterated with the model's
//! own predictions to avoid the low bias of weighting by the observed values. The priors enter as
//! further Gaussian rows:
//!
//! * `ln a ~ N(m, s^2)`, linearised around the current `a`;
//! * `b` (the mean of the `beta_c`, i.e. the convention `sum delta_c = 0`): Gaussian, or a soft
//!   wall outside a log-uniform range;
//! * `delta_c ~ N(0, delta_sigma^2)`, which also pins down the one direction (a common shift of all
//!   `delta_c` against `b`) that the data cannot see.
//!
//! Texture that survives the selection only ever raises a bin's variance, and many bins of a busy
//! scene can be off by tens of percent or by a large factor, so the data rows carry robust,
//! **one-sided** weights: positive residuals (variance above the line) are discounted much earlier
//! than negative ones. The first iterations use a heavy-tailed Cauchy-type weight on the positive
//! side and Huber on the negative side, which pulls the line down to the lower envelope of the
//! points; the later ones use Tukey's bisquare with a tighter constant on the positive side.
//! Residuals are normalised by their nominal standard error and by a robust scale estimated from the
//! negative side only (never below 1), because contamination cannot hide there.
//!
//! The solution of the 5 x 5 normal equations uses Jacobi (diagonal) scaling and a Cholesky
//! factorisation, because the entries span many orders of magnitude.

use crate::model::NUM_PLANES;
use crate::priors::{BPrior, NoisePrior};

/// One variance observation for the fit.
#[derive(Clone, Copy, Debug)]
pub struct FitPoint {
    /// CFA plane of the observation, or `None` for a point pooled over the four planes.
    pub plane: Option<u8>,
    /// Mean intensity of the group.
    pub mean: f64,
    /// Variance of one photosite in the group with the quantisation term removed.
    pub var: f64,
    /// Relative variance of `var` (`Var(var) / var^2`).
    pub rel_var: f64,
}

/// Result of [`fit_map`].
#[derive(Clone, Copy, Debug)]
pub struct FitOutput {
    /// Shot-noise gain.
    pub a: f64,
    /// Read-noise variance (never negative).
    pub b: f64,
    /// Plane offsets with `mean(delta) = 0` unless `b` was clamped at zero.
    pub delta: [f64; NUM_PLANES],
    /// Root-mean-square relative residual `(v - pred) / pred` over the points.
    pub residual_rms: f64,
    /// Number of IRLS iterations used.
    pub iterations: usize,
    /// Fraction of the points more than [`HIGH_SIGMAS`] nominal standard errors above the fit
    /// (texture leaves points above the line, never below it).
    pub high_fraction: f64,
}

/// Residual, in robust scales of the points below the line, above which a point counts as "high".
pub const HIGH_SIGMAS: f64 = 4.0;

/// Smallest `a` the fit returns; below this the image is noise-free for our purposes.
pub const A_MIN: f64 = 1.0e-9;

/// Largest plane offset the fit returns, normalised units.
pub const DELTA_MAX: f64 = 2.0e-3;

const PARAMS: usize = 5;

/// Width (in `ln b`) of the soft wall outside a log-uniform prior on `b`: about a factor 2.
const WALL_SIGMA_LOG: f64 = 0.7;

/// Smallest ratio `b / bound` the log-uniform wall's secant is evaluated at (so that a negative or
/// vanishing `b` gets a finite, strong pull towards the bound).
const WALL_MIN_RATIO: f64 = 1e-3;

/// Width of the wall that keeps the mean intercept representable, as a fraction of its limit
/// `DELTA_MAX * a` (a hundredth: far stiffer than any data point, so the overshoot is negligible).
const WALL_STIFFNESS: f64 = 0.01;

/// First phase, positive residuals: Cauchy-type decay starting at this many robust scales.
const START_C_POS: f64 = 1.0;

/// First phase, negative residuals: Huber constant.
const START_C_NEG: f64 = 2.0;

/// Second phase, Tukey bisquare constant on the positive side.
const TUKEY_C_POS: f64 = 3.0;

/// Second phase, Tukey bisquare constant on the negative side (95% efficiency on Gaussian data).
const TUKEY_C_NEG: f64 = 4.685;

/// Iterations of the first phase.
const START_ITERATIONS: usize = 6;

/// Solves the symmetric positive-definite system `a x = rhs` (5 x 5) with diagonal scaling.
// Small dense linear algebra: the index form is the clearest.
#[allow(clippy::needless_range_loop)]
fn solve(mut a: [[f64; PARAMS]; PARAMS], mut rhs: [f64; PARAMS]) -> Option<[f64; PARAMS]> {
    let mut scale = [1.0; PARAMS];
    for i in 0..PARAMS {
        if a[i][i] <= 0.0 || !a[i][i].is_finite() {
            return None;
        }
        scale[i] = 1.0 / a[i][i].sqrt();
    }
    for i in 0..PARAMS {
        for j in 0..PARAMS {
            a[i][j] *= scale[i] * scale[j];
        }
        rhs[i] *= scale[i];
        // Tiny ridge keeps a nearly singular system factorisable.
        a[i][i] += 1e-12;
    }
    // Cholesky factorisation a = L L^T.
    let mut l = [[0.0; PARAMS]; PARAMS];
    for i in 0..PARAMS {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = [0.0; PARAMS];
    for i in 0..PARAMS {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = [0.0; PARAMS];
    for i in (0..PARAMS).rev() {
        let mut s = y[i];
        for k in i + 1..PARAMS {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    for i in 0..PARAMS {
        x[i] *= scale[i];
    }
    Some(x)
}

fn add_row(
    ata: &mut [[f64; PARAMS]; PARAMS],
    atb: &mut [f64; PARAMS],
    row: &[f64; PARAMS],
    target: f64,
    weight: f64,
) {
    for i in 0..PARAMS {
        if row[i] == 0.0 {
            continue;
        }
        for j in 0..PARAMS {
            ata[i][j] += weight * row[i] * row[j];
        }
        atb[i] += weight * row[i] * target;
    }
}

fn point_row(p: &FitPoint) -> [f64; PARAMS] {
    let mut row = [0.0; PARAMS];
    row[0] = p.mean;
    match p.plane {
        Some(c) => row[1 + (c as usize & 3)] = 1.0,
        None => row[1..].fill(0.25),
    }
    row
}

/// Fits `(a, b, delta_c)` to `points` with the priors in `prior`. Returns `None` when there are no
/// points or the system is degenerate.
pub fn fit_map(points: &[FitPoint], prior: &NoisePrior) -> Option<FitOutput> {
    if points.is_empty() {
        return None;
    }
    // Initial values: the prior centre, else a crude two-point slope through the data.
    let mut a = prior.a_center().unwrap_or_else(|| {
        let (lo, hi) = points.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
            (lo.min(p.mean), hi.max(p.mean))
        });
        let vlo = points
            .iter()
            .filter(|p| p.mean == lo)
            .map(|p| p.var)
            .next()
            .unwrap_or(0.0);
        let vhi = points
            .iter()
            .filter(|p| p.mean == hi)
            .map(|p| p.var)
            .next()
            .unwrap_or(0.0);
        if hi > lo && vhi > vlo {
            (vhi - vlo) / (hi - lo)
        } else {
            1e-4
        }
    });
    a = a.max(A_MIN);
    let b0 = prior.b_center().unwrap_or_else(|| {
        points
            .iter()
            .map(|p| p.var)
            .fold(f64::MAX, f64::min)
            .max(1e-12)
    });
    let mut beta = [b0; NUM_PLANES];
    let mut iterations = 0;
    for it in 0..40 {
        iterations = it + 1;
        let mut ata = [[0.0; PARAMS]; PARAMS];
        let mut atb = [0.0; PARAMS];
        // Normalised residuals against the current model, for the robust weights.
        let resid: Vec<f64> = points
            .iter()
            .map(|p| {
                let row = point_row(p);
                let pred = (row[0] * a + (1..PARAMS).map(|k| row[k] * beta[k - 1]).sum::<f64>())
                    .max(1e-14);
                (p.var - pred) / (pred * p.rel_var.max(1e-9).sqrt())
            })
            .collect();
        let scale = if it == 0 {
            1.0
        } else {
            // Scale from the negative side only: contamination cannot hide below the line.
            let mut neg: Vec<f64> = resid.iter().filter(|&&r| r < 0.0).map(|r| -r).collect();
            neg.sort_by(f64::total_cmp);
            if neg.len() < 4 {
                1.0
            } else {
                (1.4826 * neg[neg.len() / 2]).max(1.0)
            }
        };
        for (p, &r) in points.iter().zip(&resid) {
            let row = point_row(p);
            let pred =
                (row[0] * a + (1..PARAMS).map(|k| row[k] * beta[k - 1]).sum::<f64>()).max(1e-14);
            // First iteration: weight by the observation; later ones by the model prediction.
            let ref_var = if it == 0 { p.var.max(1e-14) } else { pred };
            let mut w = 1.0 / (p.rel_var.max(1e-9) * ref_var * ref_var);
            if it > 0 {
                let u = r.abs() / scale;
                let positive = r > 0.0;
                w *= if it <= START_ITERATIONS {
                    if positive {
                        // Cauchy-type: 1 / (1 + (u / c)^2), strongly discounts large positive residuals.
                        1.0 / (1.0 + (u / START_C_POS).powi(2))
                    } else if u <= START_C_NEG {
                        1.0
                    } else {
                        START_C_NEG / u
                    }
                } else {
                    let c = if positive { TUKEY_C_POS } else { TUKEY_C_NEG };
                    let t = u / c;
                    if t >= 1.0 { 0.0 } else { (1.0 - t * t).powi(2) }
                };
            }
            add_row(&mut ata, &mut atb, &row, p.var, w);
        }
        let b_now = beta.iter().sum::<f64>() / NUM_PLANES as f64;
        // Prior on ln a, linearised: ln a ~ ln a_k + (a - a_k) / a_k.
        if let Some(l) = prior.log_a {
            let s = l.sigma.max(1e-6);
            let row = [1.0 / (a * s), 0.0, 0.0, 0.0, 0.0];
            let target = (l.mean - a.ln() + 1.0) / s;
            add_row(&mut ata, &mut atb, &row, target, 1.0);
        }
        // Prior on b = mean(beta).
        let brow = [0.0, 0.25, 0.25, 0.25, 0.25];
        match prior.b {
            BPrior::Flat => {}
            BPrior::Gaussian { mean, sigma } => {
                let s = sigma.max(mean.abs() * 1e-3).max(1e-14);
                add_row(&mut ata, &mut atb, &brow.map(|v| v / s), mean / s, 1.0);
            }
            BPrior::LogUniform { min, max } => {
                // A soft wall in log space (sigma = WALL_SIGMA_LOG) outside [min, max]. The residual
                // is ln(b / bound) / sigma; instead of linearising it at the current b (a
                // Gauss-Newton step b -> b (1 - ln(b / bound)) goes negative as soon as b exceeds e
                // times the bound, after which the iteration cycles with wild values of `a`), the
                // iteration reweights: the penalty is a quadratic in (b - bound) whose weight is the
                // secant of the log residual through the bound, which is positive for every b and
                // equals the log residual's own slope at the bound.
                let bound = if b_now < min {
                    Some(min)
                } else if b_now > max {
                    Some(max)
                } else {
                    None
                };
                if let Some(bound) = bound {
                    let ratio = (b_now / bound).max(WALL_MIN_RATIO);
                    let slope = if (ratio - 1.0).abs() < 1e-9 {
                        1.0
                    } else {
                        ratio.ln() / (ratio - 1.0)
                    } / (bound * WALL_SIGMA_LOG);
                    add_row(
                        &mut ata,
                        &mut atb,
                        &brow.map(|v| v * slope),
                        bound * slope,
                        1.0,
                    );
                }
            }
        }
        // The model can only represent a negative mean intercept `b = mean(beta) < 0` as a common
        // plane offset `delta = -b / a`, and only up to DELTA_MAX (a black level that wrong is not an
        // offset any more). Without this wall the least-squares solution keeps its unrepresentable
        // intercept, the conversion at the end clamps it away, and the returned model no longer
        // reproduces the variance the image measured (a prior that pins `a` above what the data
        // allow then inflates the variance at the image's own levels). The row is the linear
        // constraint `DELTA_MAX * a + mean(beta) >= 0`, stiff when violated.
        if DELTA_MAX * a + b_now < 0.0 {
            let s = (WALL_STIFFNESS * DELTA_MAX * a).max(1e-18);
            let row = [DELTA_MAX / s, 0.25 / s, 0.25 / s, 0.25 / s, 0.25 / s];
            add_row(&mut ata, &mut atb, &row, 0.0, 1.0);
        }
        // Prior on delta_c = (mean(beta) - beta_c) / a.
        let ds = prior.delta_sigma.max(1e-9) * a;
        for c in 0..NUM_PLANES {
            let mut row = [0.0; PARAMS];
            for (k, slot) in row.iter_mut().enumerate().skip(1) {
                *slot = 0.25 - if k - 1 == c { 1.0 } else { 0.0 };
            }
            add_row(&mut ata, &mut atb, &row.map(|v| v / ds), 0.0, 1.0);
        }
        let x = solve(ata, atb)?;
        let a_new = x[0].max(A_MIN);
        let converged = (a_new / a - 1.0).abs() < 1e-6
            && (0..NUM_PLANES)
                .all(|c| (x[1 + c] - beta[c]).abs() <= 1e-6 * (x[1 + c].abs() + 1e-12));
        a = a_new;
        beta.copy_from_slice(&x[1..]);
        if converged {
            break;
        }
    }
    if !a.is_finite() || beta.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let beta_mean = beta.iter().sum::<f64>() / NUM_PLANES as f64;
    let b = beta_mean.max(0.0);
    let mut delta = [0.0; NUM_PLANES];
    for c in 0..NUM_PLANES {
        delta[c] = ((b - beta[c]) / a).clamp(-DELTA_MAX, DELTA_MAX);
    }
    // Residuals against the fitted line.
    let mut ss = 0.0;
    let mut resid = Vec::with_capacity(points.len());
    for p in points {
        let row = point_row(p);
        let pred = (row[0] * a + (1..PARAMS).map(|k| row[k] * beta[k - 1]).sum::<f64>()).max(1e-14);
        ss += ((p.var - pred) / pred).powi(2);
        resid.push((p.var - pred) / (pred * p.rel_var.max(1e-9).sqrt()));
    }
    // "High" is measured against the scatter of the points below the line, which texture cannot
    // inflate but correlated noise (fewer independent coefficients than counted) does.
    let mut neg: Vec<f64> = resid.iter().filter(|&&r| r < 0.0).map(|r| -r).collect();
    neg.sort_by(f64::total_cmp);
    let scale = if neg.len() < 4 {
        1.0
    } else {
        (1.4826 * neg[neg.len() / 2]).max(1.0)
    };
    let high = resid.iter().filter(|&&r| r > HIGH_SIGMAS * scale).count();
    Some(FitOutput {
        a,
        b,
        delta,
        residual_rms: (ss / points.len() as f64).sqrt(),
        iterations,
        high_fraction: high as f64 / points.len() as f64,
    })
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::model::synth::SynthRng;
    use crate::priors::{LogNormal, PriorSource};

    fn points(a: f64, beta: [f64; 4], rel: f64, noise: f64, seed: u64) -> Vec<FitPoint> {
        let mut rng = SynthRng::new(seed);
        let mut out = Vec::new();
        for c in 0..4u8 {
            for k in 0..12 {
                let mean = 0.0005 * 1.7f64.powi(k);
                let v = a * mean + beta[c as usize];
                out.push(FitPoint {
                    plane: Some(c),
                    mean,
                    var: v * (1.0 + noise * rng.normal()),
                    rel_var: rel,
                });
            }
        }
        out
    }

    #[test]
    fn recovers_exact_points() {
        let a = 4e-4;
        let b = 2e-7;
        let delta = [3e-4, -1e-4, -1e-4, -1e-4];
        let beta: [f64; 4] = std::array::from_fn(|c| b - a * delta[c]);
        let fit = fit_map(&points(a, beta, 1e-3, 0.0, 1), &NoisePrior::none()).unwrap();
        assert!((fit.a / a - 1.0).abs() < 1e-4, "a {}", fit.a);
        assert!((fit.b / b - 1.0).abs() < 5e-3, "b {}", fit.b);
        for c in 0..4 {
            assert!(
                (fit.delta[c] - delta[c]).abs() < 3e-5,
                "delta {c}: {}",
                fit.delta[c]
            );
        }
        assert!(fit.residual_rms < 1e-3);
    }

    #[test]
    fn noisy_points_are_recovered_within_their_scatter() {
        let a = 1e-3;
        let beta = [1e-6; 4];
        let mut worst: f64 = 0.0;
        for seed in 0..20 {
            let fit = fit_map(&points(a, beta, 4e-4, 0.02, seed), &NoisePrior::none()).unwrap();
            worst = worst.max((fit.a / a - 1.0).abs());
        }
        assert!(worst < 0.03, "worst relative error of a {worst}");
    }

    #[test]
    fn prior_on_a_dominates_when_data_are_uninformative() {
        // Two points at nearly the same intensity: the slope is unconstrained by the data.
        let pts = [
            FitPoint {
                plane: None,
                mean: 0.30,
                var: 3.0e-4,
                rel_var: 0.05,
            },
            FitPoint {
                plane: None,
                mean: 0.31,
                var: 3.1e-4,
                rel_var: 0.05,
            },
        ];
        let prior = NoisePrior {
            log_a: Some(LogNormal {
                mean: (1e-3f64).ln(),
                sigma: 0.25,
            }),
            b: BPrior::Flat,
            delta_sigma: 3e-4,
            source: PriorSource::None,
            widened: false,
        };
        let fit = fit_map(&pts, &prior).unwrap();
        assert!((fit.a / 1e-3 - 1.0).abs() < 0.5, "a {}", fit.a);
        // Without the prior the same data give something else.
        let free = fit_map(&pts, &NoisePrior::none()).unwrap();
        assert!(free.a != fit.a);
    }

    #[test]
    fn log_uniform_b_is_a_soft_wall() {
        let a = 1e-3;
        // True b = 1e-4 (large); prior range [1e-9, 1e-7] pulls it down, data weakly.
        let beta = [1e-4; 4];
        let prior = NoisePrior {
            log_a: None,
            b: BPrior::LogUniform {
                min: 1e-9,
                max: 1e-7,
            },
            delta_sigma: 3e-4,
            source: PriorSource::None,
            widened: false,
        };
        let strong = fit_map(&points(a, beta, 1e-5, 0.0, 1), &prior).unwrap();
        // Strong data still win over a soft wall.
        assert!(strong.b > 5e-5);
        let weak = fit_map(&points(a, beta, 1e2, 0.0, 1), &prior).unwrap();
        assert!(
            weak.b < 1e-5,
            "weak data should be held by the wall: {}",
            weak.b
        );
        // The iteration settles. With the wall linearised at the current b it did not: the first
        // step went to a negative b, `a` jumped to 4e-2, and the 40 iterations cycled -- the
        // assertion on `b` above passed only because of where in the cycle the loop stopped.
        assert!(
            weak.iterations < 40,
            "no convergence in {} iterations",
            weak.iterations
        );
    }

    #[test]
    fn negative_intercept_moves_into_the_offsets() {
        let a = 1e-3;
        let beta = [-5e-7; 4];
        let fit = fit_map(&points(a, beta, 1e-3, 0.0, 1), &NoisePrior::none()).unwrap();
        assert_eq!(fit.b, 0.0);
        for c in 0..4 {
            assert!(fit.delta[c] > 0.0, "pedestal becomes a positive offset");
        }
    }

    #[test]
    fn a_prior_above_what_the_data_allow_gives_way_instead_of_losing_the_level() {
        // One intensity, var(0.25) = 1.5e-4. With b >= 0 and |delta| <= DELTA_MAX the model can
        // only reach that level with a <= 1.5e-4 / (0.25 - 0.002) = 6.05e-4. A tight prior at
        // 1.2e-3 would need a negative intercept of -1.5e-4, an offset of an eighth of white,
        // which the conversion to (b, delta) used to clamp away: the returned model then
        // predicted 2.0e-4 -- twice the measured variance -- at the image's own level.
        let pts: Vec<FitPoint> = (0..4)
            .map(|c| FitPoint {
                plane: Some(c),
                mean: 0.25,
                var: 1.5e-4,
                rel_var: 2.0e-4,
            })
            .collect();
        let prior = NoisePrior {
            log_a: Some(LogNormal {
                mean: (1.2e-3f64).ln(),
                sigma: 0.05,
            }),
            b: BPrior::Flat,
            delta_sigma: 3e-4,
            source: PriorSource::None,
            widened: false,
        };
        let fit = fit_map(&pts, &prior).unwrap();
        let level = (0..4)
            .map(|c| fit.a * (0.25 - fit.delta[c]).max(0.0) + fit.b)
            .sum::<f64>()
            / 4.0;
        assert!(
            (level / 1.5e-4 - 1.0).abs() < 0.03,
            "model variance {level} against the measured 1.5e-4"
        );
        assert!(fit.a < 0.7e-3 && fit.a > 0.5e-3, "a {}", fit.a);
        assert!(fit.delta.iter().all(|d| d.abs() <= DELTA_MAX + 1e-12));
    }

    #[test]
    fn empty_input_gives_none() {
        assert!(fit_map(&[], &NoisePrior::none()).is_none());
    }
}
