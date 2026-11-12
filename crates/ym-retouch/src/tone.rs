//! The relation between the analysis picture and the node's input (`docs/27` decision 13, S-12).
//!
//! The blemish gain map is made on the analysis picture, which has the Input's colour and nothing else
//! (no grade, no look). The node's input has the grade of everything upstream, and a tone curve there
//! changes the ratio a gain makes: through a power curve of 0.7 a papule of -0.205 became +0.111 with
//! the analysis picture's map. A [`ToneRelation`] is a per-channel, monotone, piecewise-linear
//! estimate of that curve in the log domain, `ln X = F(ln A)`, found by matching the distributions of the
//! analysis picture's values and the input's around a face. The gain is then applied in the domain
//! where the curve cancels: `X' = exp(F(F^-1(ln X) + ln g))`, which is `X * g` for an exposure change
//! (so that case is detected and left as the plain multiplication, exactly) and `X * g^p` for a power
//! curve of exponent `p`.
//!
//! No model runs here and nothing is stored per photo: the relation is made when the face step
//! makes a region, from the pixels of that region.

/// Values at or under this are left out of the estimate and are not changed by it.
pub const TONE_FLOOR: f64 = 1e-4;

/// Equal-count bins per channel.
const BINS: usize = 32;

/// Fewest pairs per channel for an estimate; fewer count as "no relation known" (the plain gain).
const MIN_PAIRS: usize = 256;

/// A relation whose slope `d ln X / d ln A` is within this of 1 is an exposure change (or none) and the gain is applied as a
/// plain multiplication, as before: the error of that is the slope's distance from 1 times the gain's `ln` (under 0.4 for
/// the gains a face needs), so 0.04 keeps it under 1.6 %. The offset of the relation does not matter, only its slope.
pub const SLOPE_TOLERANCE: f64 = 0.04;

/// Slopes at the ends are kept between these, so that the inverse stays finite and the curve monotone.
const END_SLOPE: (f64, f64) = (0.05, 4.0);

/// One channel: knots `(ln A, ln X)`, `ln A` strictly increasing, `ln X` non-decreasing, and the slopes the curve goes on with
/// beyond its first and last knot (the pixels of a lesion lie outside the bins' medians, and the curve there is the end
/// segment's, not the identity).
#[derive(Debug, Clone, PartialEq)]
struct Curve {
    la: Vec<f64>,
    lx: Vec<f64>,
    slope_lo: f64,
    slope_hi: f64,
}

impl Curve {
    fn new(la: Vec<f64>, lx: Vec<f64>) -> Self {
        let n = la.len();
        let end = |a: usize, b: usize| {
            ((lx[b] - lx[a]) / (la[b] - la[a])).clamp(END_SLOPE.0, END_SLOPE.1)
        };
        let (slope_lo, slope_hi) = (end(0, 2.min(n - 1)), end(n.saturating_sub(3), n - 1));
        Self {
            la,
            lx,
            slope_lo,
            slope_hi,
        }
    }

    /// `ln X` for `ln A`.
    fn forward(&self, la: f64) -> f64 {
        let n = self.la.len();
        if la <= self.la[0] {
            return self.lx[0] + self.slope_lo * (la - self.la[0]);
        }
        if la >= self.la[n - 1] {
            return self.lx[n - 1] + self.slope_hi * (la - self.la[n - 1]);
        }
        let k = self.la.partition_point(|&v| v <= la) - 1;
        let t = (la - self.la[k]) / (self.la[k + 1] - self.la[k]);
        self.lx[k] + t * (self.lx[k + 1] - self.lx[k])
    }

    /// `ln A` for `ln X`: the smallest `ln A` that reaches it (a flat part of the curve, a clipped
    /// highlight, answers with its lower end).
    fn inverse(&self, lx: f64) -> f64 {
        let n = self.la.len();
        if lx <= self.lx[0] {
            return self.la[0] + (lx - self.lx[0]) / self.slope_lo;
        }
        if lx >= self.lx[n - 1] {
            return self.la[n - 1] + (lx - self.lx[n - 1]) / self.slope_hi;
        }
        // First knot with ln X >= lx.
        let k = self.lx.partition_point(|&v| v < lx);
        let (x0, x1) = (self.lx[k - 1], self.lx[k]);
        let t = if x1 > x0 { (lx - x0) / (x1 - x0) } else { 1.0 };
        self.la[k - 1] + t * (self.la[k] - self.la[k - 1])
    }
}

/// The estimated tone relation of one region, or none (the plain gain).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToneRelation {
    curves: Option<[Curve; 3]>,
}

impl ToneRelation {
    /// No relation: [`ToneRelation::gain`] is the plain multiplication.
    pub fn none() -> Self {
        Self::default()
    }

    /// True when the gain is applied as a plain multiplication (no relation, or an exposure change).
    pub fn is_plain(&self) -> bool {
        self.curves.is_none()
    }

    /// Estimates the relation from the pixels of the analysis picture and the pixels of the node's input over the same part of
    /// the photo, both linear AWG4. The two sets are not paired, and need not have the same number of pixels or the same
    /// resolution: the relation is found by matching their distributions, channel by channel (the value at each of 32 equal-count
    /// quantiles of the analysis picture goes to the value at the same quantile of the input), which a difference of
    /// resolution, a sharpening or a little grain upstream hardly moves, where pairing pixels would meet the regression
    /// dilution of a noisy abscissa. Deterministic: the values are sorted with a total order.
    ///
    /// A channel with fewer than 256 usable values in either set (a value is usable above [`TONE_FLOOR`]), or a relation whose
    /// slope is within [`SLOPE_TOLERANCE`] of 1 on all three channels (an exposure change), gives [`ToneRelation::none`].
    pub fn estimate(analysis: &[[f32; 3]], input: &[[f32; 3]]) -> Self {
        let mut curves: Vec<Curve> = Vec::with_capacity(3);
        let mut plain = true;
        for c in 0..3 {
            let logs = |set: &[[f32; 3]]| -> Vec<f64> {
                let mut v: Vec<f64> = set
                    .iter()
                    .map(|p| p[c])
                    .filter(|&x| x.is_finite() && f64::from(x) > TONE_FLOOR)
                    .map(|x| f64::from(x).ln())
                    .collect();
                v.sort_by(f64::total_cmp);
                v
            };
            let (la_all, lx_all) = (logs(analysis), logs(input));
            if la_all.len() < MIN_PAIRS || lx_all.len() < MIN_PAIRS {
                return Self::none();
            }
            let at =
                |v: &[f64], k: usize| v[(((2 * k + 1) * v.len()) / (2 * BINS)).min(v.len() - 1)];
            let (mut la, mut lx): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
            for k in 0..BINS {
                let (a, x) = (at(&la_all, k), at(&lx_all, k));
                if la.last().is_some_and(|&l| a <= l + 1e-9) {
                    continue;
                }
                la.push(a);
                // A tone curve does not decrease.
                lx.push(lx.last().map_or(x, |&l: &f64| l.max(x)));
            }
            if la.len() < 3 {
                return Self::none();
            }
            let curve = Curve::new(la, lx);
            let n = curve.la.len() as f64;
            let (ma, mx) = (
                curve.la.iter().sum::<f64>() / n,
                curve.lx.iter().sum::<f64>() / n,
            );
            let (cov, var) = curve
                .la
                .iter()
                .zip(&curve.lx)
                .fold((0.0, 0.0), |(c, v), (a, x)| {
                    (c + (a - ma) * (x - mx), v + (a - ma) * (a - ma))
                });
            let slope = cov / var;
            if (slope - 1.0).abs() > SLOPE_TOLERANCE
                || (curve.slope_lo - 1.0).abs() > 2.0 * SLOPE_TOLERANCE
                || (curve.slope_hi - 1.0).abs() > 2.0 * SLOPE_TOLERANCE
            {
                plain = false;
            }
            curves.push(curve);
        }
        if plain {
            return Self::none();
        }
        let [a, b, c] = <[Curve; 3]>::try_from(curves).unwrap_or_else(|_| unreachable!());
        Self {
            curves: Some([a, b, c]),
        }
    }

    /// Applies the gain `g` (per channel, already mixed with the strength) to the input pixel `rgb`
    /// in the domain where the curve cancels. A channel at or under [`TONE_FLOOR`], or a gain that is
    /// not positive and finite, gets the plain multiplication.
    pub fn gain(&self, rgb: [f32; 3], g: [f64; 3]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let x = f64::from(rgb[c]);
            out[c] = match &self.curves {
                Some(curves) if x > TONE_FLOOR && g[c] > 0.0 && g[c].is_finite() => {
                    let curve = &curves[c];
                    let la = curve.inverse(x.ln()) + g[c].ln();
                    curve.forward(la).exp() as f32
                }
                _ => (x * g[c]) as f32,
            };
        }
        out
    }

    /// The slope `d ln X / d ln A` of channel `c` at the input value `x`, for tests and reports.
    pub fn slope(&self, c: usize, x: f64) -> f64 {
        match &self.curves {
            Some(curves) if x > TONE_FLOOR => {
                let curve = &curves[c];
                let la = curve.inverse(x.ln());
                let h = 0.05;
                (curve.forward(la + h) - curve.forward(la - h)) / (2.0 * h)
            }
            _ => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The analysis values and the input values: `f` of a spread of values from 0.02 to 1.2.
    fn sets(f: impl Fn(f64) -> f64) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        (0..4000)
            .map(|i| {
                let a = 0.02 * (1.0 + 1.0 * i as f64 / 4000.0 * 60.0);
                ([a as f32; 3], [f(a) as f32; 3])
            })
            .unzip()
    }

    fn estimate(f: impl Fn(f64) -> f64) -> ToneRelation {
        let (a, x) = sets(f);
        ToneRelation::estimate(&a, &x)
    }

    #[test]
    fn exposure_and_nothing_are_plain() {
        assert!(estimate(|a| a).is_plain());
        assert!(estimate(|a| 4.0 * a).is_plain());
        assert!(ToneRelation::estimate(&[], &[]).is_plain());
        let plain = ToneRelation::none();
        assert_eq!(
            plain.gain([0.5, 0.4, 0.3], [2.0, 1.0, 0.5]),
            [1.0, 0.4, 0.15]
        );
    }

    #[test]
    fn a_power_curve_scales_the_log_gain_by_its_exponent() {
        for p in [0.5f64, 0.7, 1.4, 2.0] {
            let rel = estimate(|a| a.powf(p));
            assert!(!rel.is_plain(), "{p}");
            let a = 0.35f64;
            let x = a.powf(p) as f32;
            let g = 1.4f64;
            let got = f64::from(rel.gain([x; 3], [g; 3])[1]);
            let want = (a * g).powf(p);
            assert!(
                (got / want - 1.0).abs() < 2e-3,
                "p {p}: {got} against {want}"
            );
            assert!((rel.slope(1, f64::from(x)) - p).abs() < 0.03 * p, "{p}");
        }
    }

    #[test]
    fn the_relation_is_monotone_and_deterministic() {
        let s = |a: f64| {
            let u = ((a / 0.18).log2() + 4.0).clamp(0.0, 8.0) / 8.0;
            0.18 * 2f64.powf(8.0 * (3.0 * u * u - 2.0 * u * u * u) - 4.0)
        };
        let (a, b) = (estimate(s), estimate(s));
        assert_eq!(a, b);
        let rel = a;
        let mut last = 0.0f32;
        for i in 0..200 {
            let x = 0.03 + 0.005 * i as f32;
            let y = rel.gain([x; 3], [1.3; 3])[0];
            assert!(y > x);
            assert!(y >= last * 0.999 || i == 0);
            last = y;
        }
    }

    #[test]
    fn dark_pixels_and_bad_gains_get_the_plain_multiplication() {
        let rel = estimate(|a| a.powf(0.7));
        assert_eq!(rel.gain([0.0, 1e-5, 0.3], [2.0, 2.0, -1.0])[0], 0.0);
        assert!((rel.gain([0.0, 1e-5, 0.3], [2.0, 2.0, -1.0])[1] - 2e-5).abs() < 1e-9);
        assert!(rel.gain([0.5; 3], [f64::NAN; 3])[0].is_nan());
    }
}
