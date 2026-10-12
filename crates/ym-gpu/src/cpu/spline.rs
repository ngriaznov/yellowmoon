//! Interpolation of the lens knots (`docs/05-raw-processing.md` sections R3 and R9).
//!
//! The vignetting gain `G(r)`, the distortion factor `F(r)` and the two CA factors `K(r)` are all
//! given as a handful of knots at evenly spaced radii.
//!
//! * [`LinearKnots`] is what the lens stages use: piecewise-linear interpolation with the knots at
//!   `(i + 0.5) / (n - 1)`, which is exactly how darktable 4.6.1 (the oracle of the lens gate in
//!   `docs/13`) reads Sony's tags; `tools/lens/` measures that against `darktable-cli`.
//! * [`MonotoneCubic`] is the monotone cubic Hermite spline of Fritsch and Butland with the
//!   shape-preserving three-point end rule, i.e. `scipy.interpolate.PchipInterpolator`. The lens
//!   stages used it before the oracle showed the knots are read linearly (it differs from the
//!   linear reading by up to 0.15 px at the corner for a quadratic profile, most of the 0.2 px
//!   gate); it stays for other radial profiles.
//!
//! Evaluation is `f64`. Outside the knot range the value is held at the end knot.

use super::CpuError;

/// A monotone cubic Hermite spline through `(x_i, y_i)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MonotoneCubic {
    xs: Vec<f64>,
    ys: Vec<f64>,
    slopes: Vec<f64>,
}

impl MonotoneCubic {
    /// Builds the spline. `xs` must be strictly increasing and at least two long, `xs.len() == ys.len()`.
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<MonotoneCubic, CpuError> {
        let n = xs.len();
        if n < 2
            || ys.len() != n
            || xs
                .windows(2)
                .any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
        {
            return Err(CpuError::Mismatch(format!(
                "monotone cubic needs >= 2 strictly increasing knots (got {} x, {} y)",
                n,
                ys.len()
            )));
        }
        let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
        let delta: Vec<f64> = (0..n - 1).map(|k| (ys[k + 1] - ys[k]) / h[k]).collect();
        let mut d = vec![0.0; n];
        if n == 2 {
            d[0] = delta[0];
            d[1] = delta[0];
        } else {
            for k in 1..n - 1 {
                if delta[k - 1] * delta[k] > 0.0 {
                    let w1 = 2.0 * h[k] + h[k - 1];
                    let w2 = h[k] + 2.0 * h[k - 1];
                    d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
                }
            }
            d[0] = end_slope(h[0], h[1], delta[0], delta[1]);
            d[n - 1] = end_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        }
        Ok(MonotoneCubic {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            slopes: d,
        })
    }

    /// Builds a spline through `ys` at the evenly spaced abscissae `i / (n - 1)`, the layout of
    /// every Sony lens tag.
    pub fn uniform_unit(ys: &[f64]) -> Result<MonotoneCubic, CpuError> {
        let n = ys.len();
        if n < 2 {
            return Err(CpuError::Mismatch(format!("need >= 2 knots, got {n}")));
        }
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        MonotoneCubic::new(&xs, ys)
    }

    /// The knot abscissae.
    pub fn xs(&self) -> &[f64] {
        &self.xs
    }

    /// The knot values.
    pub fn ys(&self) -> &[f64] {
        &self.ys
    }

    /// Evaluates the spline at `x` (held at the end values outside the knots).
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= self.xs[n - 1] {
            return self.ys[n - 1];
        }
        // Largest k with xs[k] <= x. Knots are few (<= 16), a linear scan is fine and branch-friendly.
        let mut k = 0;
        while k + 2 < n && self.xs[k + 1] <= x {
            k += 1;
        }
        let h = self.xs[k + 1] - self.xs[k];
        let t = (x - self.xs[k]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        h00 * self.ys[k]
            + h10 * h * self.slopes[k]
            + h01 * self.ys[k + 1]
            + h11 * h * self.slopes[k + 1]
    }

    /// Samples the spline at `count` evenly spaced points over `[lo, hi]` (inclusive) into a table
    /// for fast per-pixel lookup with [`Table::eval`].
    pub fn tabulate(&self, lo: f64, hi: f64, count: usize) -> Table {
        let count = count.max(2);
        let values = (0..count)
            .map(|i| self.eval(lo + (hi - lo) * i as f64 / (count - 1) as f64))
            .collect();
        Table { lo, hi, values }
    }
}

/// Shape-preserving three-point end slope (as in SciPy's PCHIP).
fn end_slope(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    let mut d = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if d.signum() != d0.signum() {
        d = 0.0;
    } else if d0.signum() != d1.signum() && d.abs() > 3.0 * d0.abs() {
        d = 3.0 * d0;
    }
    d
}

/// A function given at `n >= 2` knots with uniform spacing, read by linear interpolation and held
/// at the end values outside the first and last knot.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearKnots {
    first: f64,
    step: f64,
    ys: Vec<f64>,
}

impl LinearKnots {
    /// The layout of the Sony lens tags: knot `i` of `n` at the normalised radius
    /// `(i + 0.5) / (n - 1)` (the first knot at half a step, the last one half a step beyond the
    /// corner of the frame). Needs at least two values.
    pub fn sony(ys: &[f64]) -> Result<LinearKnots, CpuError> {
        let n = ys.len();
        if n < 2 {
            return Err(CpuError::Mismatch(format!("need >= 2 knots, got {n}")));
        }
        let step = 1.0 / (n - 1) as f64;
        Ok(LinearKnots {
            first: 0.5 * step,
            step,
            ys: ys.to_vec(),
        })
    }

    /// The knot abscissae.
    pub fn xs(&self) -> Vec<f64> {
        (0..self.ys.len())
            .map(|i| self.first + self.step * i as f64)
            .collect()
    }

    /// The knot values.
    pub fn ys(&self) -> &[f64] {
        &self.ys
    }

    /// Evaluates the interpolant at `x`; a NaN gives the first knot value.
    #[inline]
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.ys.len();
        let u = (x - self.first) / self.step;
        if u.is_nan() || u <= 0.0 {
            return self.ys[0];
        }
        if u >= (n - 1) as f64 {
            return self.ys[n - 1];
        }
        let i = u as usize;
        let t = u - i as f64;
        self.ys[i] + (self.ys[i + 1] - self.ys[i]) * t
    }
}

/// A uniformly sampled function with linear interpolation, for per-pixel use.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    lo: f64,
    hi: f64,
    values: Vec<f64>,
}

impl Table {
    /// Linear interpolation, clamped to the table's range.
    #[inline]
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.values.len();
        let u = ((x - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0) * (n - 1) as f64;
        let i = (u as usize).min(n - 2);
        let t = u - i as f64;
        self.values[i] * (1.0 - t) + self.values[i + 1] * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_knots_follow_the_sony_layout() {
        // Knot i of 4 sits at (i + 0.5) / 3: 1/6, 1/2, 5/6, 7/6.
        let k = LinearKnots::sony(&[1.0, 2.0, 4.0, 8.0]).unwrap();
        let xs = k.xs();
        for (i, want) in [1.0 / 6.0, 0.5, 5.0 / 6.0, 7.0 / 6.0].iter().enumerate() {
            assert!((xs[i] - want).abs() < 1e-15, "knot {i}");
            assert!((k.eval(xs[i]) - k.ys()[i]).abs() < 1e-12);
        }
        // Held below the first knot, linear in between, midpoints are the means.
        assert_eq!(k.eval(0.0), 1.0);
        assert_eq!(k.eval(1.0 / 6.0 - 1e-9), 1.0);
        assert!((k.eval(1.0 / 3.0) - 1.5).abs() < 1e-12);
        assert!((k.eval(2.0 / 3.0) - 3.0).abs() < 1e-12);
        assert!((k.eval(1.0) - (4.0 + 4.0 * (1.0 - 5.0 / 6.0) / (1.0 / 3.0))).abs() < 1e-12);
        assert_eq!(k.eval(5.0), 8.0);
        assert_eq!(k.eval(f64::NAN), 1.0);
        assert!(LinearKnots::sony(&[1.0]).is_err());
    }

    #[test]
    fn interpolates_knots_exactly() {
        let ys = [1.0, 1.2, 1.5, 1.6, 2.0];
        let s = MonotoneCubic::uniform_unit(&ys).unwrap();
        for (i, y) in ys.iter().enumerate() {
            assert!((s.eval(i as f64 / 4.0) - y).abs() < 1e-12);
        }
    }

    #[test]
    fn preserves_monotonicity() {
        let ys = [1.0, 1.0, 1.01, 1.5, 1.52, 2.0, 2.0];
        let s = MonotoneCubic::uniform_unit(&ys).unwrap();
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=600 {
            let v = s.eval(i as f64 / 600.0);
            assert!(v >= prev - 1e-12, "not monotone at {i}: {v} < {prev}");
            prev = v;
        }
    }

    #[test]
    fn matches_scipy_pchip_reference_values() {
        // Values from scipy.interpolate.PchipInterpolator(np.linspace(0, 1, 5), y)(x), computed offline.
        let ys = [1.0, 1.1, 1.5, 1.6, 2.0];
        let s = MonotoneCubic::uniform_unit(&ys).unwrap();
        let expected = [
            (0.1, 1.01984_f64),
            (0.37, 1.2896038399999998),
            (0.5, 1.5),
            (0.83, 1.6820428799999998),
            (0.25, 1.1),
        ];
        for (x, e) in expected {
            let v = s.eval(x);
            assert!((v - e).abs() < 1e-12, "x={x}: {v} vs scipy {e}");
        }
    }

    #[test]
    fn two_knots_is_linear_and_flat_outside() {
        let s = MonotoneCubic::uniform_unit(&[2.0, 4.0]).unwrap();
        assert!((s.eval(0.25) - 2.5).abs() < 1e-12);
        assert_eq!(s.eval(-1.0), 2.0);
        assert_eq!(s.eval(2.0), 4.0);
        assert!(MonotoneCubic::uniform_unit(&[1.0]).is_err());
    }

    #[test]
    fn table_tracks_spline() {
        let s = MonotoneCubic::uniform_unit(&[1.0, 1.3, 1.4, 2.0]).unwrap();
        let t = s.tabulate(0.0, 1.0, 4097);
        for i in 0..=100 {
            let x = i as f64 / 100.0;
            assert!((t.eval(x) - s.eval(x)).abs() < 1e-6);
        }
    }
}
