//! Stages 4 and 5: the characteristic curves and the DIR couplers.
//!
//! Written from Hunt, *The Reproduction of Colour*, 6th ed. (chapter 15: colour couplers and inter-layer
//! inhibition), Giorgianni and Madden, *Digital Color Management*, 2nd ed. (characteristic curves, density
//! against log exposure), the film data sheets the profiles are read from, and Fritsch and Carlson, "Monotone
//! piecewise cubic interpolation", SIAM J. Numer. Anal. 17 (1980), for the curve interpolation.
//!
//! **Curves** are monotone cubic (Fritsch-Carlson, with the Fritsch-Butland slopes), density against log10
//! exposure, held flat beyond the axis.
//!
//! **Couplers.** A developer inhibitor is released in proportion to the dye density formed, so each layer's
//! effective log exposure is lowered by `amount * sum_j M[i][j] D_j`. The profile's `density_curves` are taken to
//! be what a *neutral ramp* gives with the couplers acting. The curves the engine reads are therefore the
//! curves without couplers `C0_i`, found from the profile's `C_i` by `C0_i(x - sum_j M[i][j] C_j(x)) = C_i(x)`;
//! a neutral ramp with `amount = 1` gives `C_i` back, and a coloured exposure gets the interlayer inhibition on
//! top (more saturation). With `amount = 0` the couplers are off and the density is `C0_i(log E_i)`.

use crate::Vec3;
use crate::error::FilmError;
use crate::math::solve3;
use crate::profile::Profile;

/// A monotone piecewise-cubic curve through `(x, y)` points, held flat outside.
#[derive(Debug, Clone)]
pub struct MonotoneCubic {
    xs: Vec<f64>,
    ys: Vec<f64>,
    ds: Vec<f64>,
}

impl MonotoneCubic {
    /// Builds the curve. `xs` must be strictly increasing and at least two long.
    ///
    /// # Panics
    ///
    /// Never for valid input; with fewer than two points it would index out of range, so callers pass 256.
    #[must_use]
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        let n = xs.len();
        let h: Vec<f64> = (0..n - 1).map(|k| xs[k + 1] - xs[k]).collect();
        let del: Vec<f64> = (0..n - 1).map(|k| (ys[k + 1] - ys[k]) / h[k]).collect();
        let mut ds = vec![0.0; n];
        if n == 2 {
            ds = vec![del[0]; 2];
        } else {
            for k in 1..n - 1 {
                if del[k - 1] * del[k] > 0.0 {
                    let w1 = 2.0 * h[k] + h[k - 1];
                    let w2 = h[k] + 2.0 * h[k - 1];
                    ds[k] = (w1 + w2) / (w1 / del[k - 1] + w2 / del[k]);
                }
            }
            ds[0] = end_slope(h[0], h[1], del[0], del[1]);
            ds[n - 1] = end_slope(h[n - 2], h[n - 3], del[n - 2], del[n - 3]);
        }
        Self { xs, ys, ds }
    }

    fn segment(&self, x: f64) -> usize {
        let k = self.xs.partition_point(|v| *v <= x);
        k.saturating_sub(1).min(self.xs.len() - 2)
    }

    /// The value at `x`; flat beyond the ends.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x.is_nan() || x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= self.xs[n - 1] {
            return self.ys[n - 1];
        }
        let k = self.segment(x);
        let h = self.xs[k + 1] - self.xs[k];
        let t = (x - self.xs[k]) / h;
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * self.ys[k]
            + (t3 - 2.0 * t2 + t) * h * self.ds[k]
            + (-2.0 * t3 + 3.0 * t2) * self.ys[k + 1]
            + (t3 - t2) * h * self.ds[k + 1]
    }

    /// The slope at `x`; 0 beyond the ends.
    #[must_use]
    pub fn slope(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x.is_nan() || x < self.xs[0] || x > self.xs[n - 1] {
            return 0.0;
        }
        let k = self.segment(x);
        let h = self.xs[k + 1] - self.xs[k];
        let t = (x - self.xs[k]) / h;
        let t2 = t * t;
        ((6.0 * t2 - 6.0 * t) * self.ys[k]
            + (3.0 * t2 - 4.0 * t + 1.0) * h * self.ds[k]
            + (-6.0 * t2 + 6.0 * t) * self.ys[k + 1]
            + (3.0 * t2 - 2.0 * t) * h * self.ds[k + 1])
            / h
    }
}

/// The three-point end slope of Fritsch and Butland, kept monotone.
fn end_slope(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    let mut d = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if d * d0 <= 0.0 {
        d = 0.0;
    } else if d0 * d1 <= 0.0 && d.abs() > 3.0 * d0.abs() {
        d = 3.0 * d0;
    }
    d
}

/// Stages 4 and 5 for one film.
#[derive(Debug, Clone)]
pub struct Develop {
    /// The profile's curves (neutral ramp, couplers acting).
    curves: [MonotoneCubic; 3],
    /// The curves without couplers.
    curves0: [MonotoneCubic; 3],
    matrix: [[f64; 3]; 3],
}

impl Develop {
    /// Builds the curves and finds the curves without couplers.
    ///
    /// # Errors
    ///
    /// [`FilmError::CouplersNotInvertible`] when `x - sum_j M[i][j] C_j(x)` does not rise with `x` on the axis.
    pub fn new(profile: &Profile, matrix: [[f64; 3]; 3]) -> Result<Self, FilmError> {
        let mk = |c: usize| {
            let (x, y) = profile.curve(c);
            MonotoneCubic::new(x, y)
        };
        let curves = [mk(0), mk(1), mk(2)];
        let xs = &profile.log_exposure;
        let mut curves0: Vec<MonotoneCubic> = Vec::with_capacity(3);
        for i in 0..3 {
            let mut u = Vec::with_capacity(xs.len());
            let mut y = Vec::with_capacity(xs.len());
            for (k, x) in xs.iter().enumerate() {
                let inhibition: f64 = (0..3)
                    .map(|j| matrix[i][j] * profile.density_curves[k][j])
                    .sum();
                u.push(x - inhibition);
                y.push(profile.density_curves[k][i]);
            }
            for k in 1..u.len() {
                if u[k] <= u[k - 1] {
                    // Wiggles in the data's toe or shoulder can flatten u for a step; nudge by a hair so the
                    // curve stays defined, but refuse a real fold.
                    if u[k] < u[k - 1] - 1e-3 {
                        return Err(FilmError::CouplersNotInvertible {
                            stock: profile.id.clone(),
                            channel: i,
                        });
                    }
                    u[k] = u[k - 1] + 1e-9;
                }
            }
            curves0.push(MonotoneCubic::new(u, y));
        }
        let mut it = curves0.into_iter();
        let curves0 = [
            it.next().unwrap_or_else(|| curves[0].clone()),
            it.next().unwrap_or_else(|| curves[1].clone()),
            it.next().unwrap_or_else(|| curves[2].clone()),
        ];
        Ok(Self {
            curves,
            curves0,
            matrix,
        })
    }

    /// Stage 4: density from the profile's curves, `D1_c = C_c(log E_c)`.
    #[must_use]
    pub fn curves(&self, log_e: Vec3) -> Vec3 {
        [
            self.curves[0].eval(log_e[0]),
            self.curves[1].eval(log_e[1]),
            self.curves[2].eval(log_e[2]),
        ]
    }

    /// Stage 5: density with the couplers acting at `amount`: solves
    /// `D_i = C0_i(log E_i - amount sum_j M[i][j] D_j)` by Newton's method.
    #[must_use]
    pub fn couple(&self, log_e: Vec3, amount: f64) -> Vec3 {
        let eval = |d: &Vec3| -> Vec3 {
            let mut g = [0.0; 3];
            for i in 0..3 {
                let inh: f64 = (0..3).map(|j| self.matrix[i][j] * d[j]).sum();
                g[i] = self.curves0[i].eval(log_e[i] - amount * inh) - d[i];
            }
            g
        };
        let mut d = [
            self.curves0[0].eval(log_e[0]),
            self.curves0[1].eval(log_e[1]),
            self.curves0[2].eval(log_e[2]),
        ];
        if amount == 0.0 {
            return d;
        }
        let norm = |g: &Vec3| g[0].abs() + g[1].abs() + g[2].abs();
        let mut g = eval(&d);
        for _ in 0..60 {
            if norm(&g) < 1e-13 {
                break;
            }
            // Jacobian of g: d g_i / d D_j = -s_i * amount * M[i][j] - delta_ij, with s_i the slope of C0_i.
            let mut jac = [[0.0; 3]; 3];
            for i in 0..3 {
                let inh: f64 = (0..3).map(|j| self.matrix[i][j] * d[j]).sum();
                let s = self.curves0[i].slope(log_e[i] - amount * inh);
                for j in 0..3 {
                    jac[i][j] = -s * amount * self.matrix[i][j] - if i == j { 1.0 } else { 0.0 };
                }
            }
            let Some(step) = solve3(jac, [-g[0], -g[1], -g[2]]) else {
                break;
            };
            // Damped step: halve until the residual falls.
            let mut t = 1.0;
            let mut improved = false;
            for _ in 0..30 {
                let cand = [d[0] + t * step[0], d[1] + t * step[1], d[2] + t * step[2]];
                let gc = eval(&cand);
                if norm(&gc) < norm(&g) {
                    d = cand;
                    g = gc;
                    improved = true;
                    break;
                }
                t *= 0.5;
            }
            if !improved {
                break;
            }
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_interpolates_and_stays_monotone() {
        let xs: Vec<f64> = (0..20).map(f64::from).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|x| 1.0 / (1.0 + (-(x - 10.0) * 0.6).exp()))
            .collect();
        let c = MonotoneCubic::new(xs.clone(), ys.clone());
        for (x, y) in xs.iter().zip(&ys) {
            assert!((c.eval(*x) - y).abs() < 1e-12);
        }
        let mut prev = c.eval(0.0);
        for k in 1..400 {
            let v = c.eval(f64::from(k) * 0.05);
            assert!(v >= prev - 1e-12);
            prev = v;
        }
        assert_eq!(c.eval(-5.0), ys[0]);
        assert_eq!(c.eval(99.0), ys[19]);
        let eps = 1e-6;
        let num = (c.eval(7.3 + eps) - c.eval(7.3 - eps)) / (2.0 * eps);
        assert!((c.slope(7.3) - num).abs() < 1e-6);
    }
}
