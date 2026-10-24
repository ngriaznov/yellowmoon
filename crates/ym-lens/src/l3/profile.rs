//! Small functions the highlight optics share: the normal tail, the rim lookup, Gauss-Legendre panels and the
//! cosine windows.
//!
//! Only `+ - * /`, `sqrt` and the `libm` functions are used, so the values are the same on every platform
//! (`docs/21` section 9).

use std::sync::OnceLock;

use crate::consts::RIM_SAMPLES;

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// The table covers `z` from `-TAIL_Z_MAX` to `TAIL_Z_MAX`.
const TAIL_Z_MAX: f64 = 8.0;

/// Table nodes per unit of `z`.
const TAIL_PER_UNIT: f64 = 256.0;

/// The upper tail of the standard normal, `Q(z) = erfc(z / sqrt 2) / 2`, as a cubic Hermite table.
///
/// An edge drawn with it is a smooth (analytic) step, which a pixel grid samples without aliasing once its
/// standard deviation is a pixel or more; and the difference of two values is the exact mass a pixel of a
/// Gaussian band holds. The table's error is below `1e-11`.
pub(crate) struct NormalTail {
    /// `(Q, dQ/dz)` at the nodes.
    nodes: Vec<[f64; 2]>,
    /// `(R, dR/dz)` at the nodes, with the ramp `R(z) = z (1 - Q(z)) + phi(z)`.
    ramp: Vec<[f64; 2]>,
}

impl NormalTail {
    fn build() -> Self {
        let n = (2.0 * TAIL_Z_MAX * TAIL_PER_UNIT) as usize;
        let z_at = |i: usize| -TAIL_Z_MAX + i as f64 / TAIL_PER_UNIT;
        let nodes = (0..=n)
            .map(|i| {
                let z = z_at(i);
                [
                    0.5 * libm::erfc(z / SQRT_2),
                    -INV_SQRT_2PI * libm::exp(-0.5 * z * z),
                ]
            })
            .collect();
        let ramp = (0..=n)
            .map(|i| {
                let z = z_at(i);
                let cdf = 0.5 * libm::erfc(-z / SQRT_2);
                [z * cdf + INV_SQRT_2PI * libm::exp(-0.5 * z * z), cdf]
            })
            .collect();
        NormalTail { nodes, ramp }
    }

    /// The table for the device: node `i` is `(Q, dQ/dz, R, dR/dz)` at `z = -8 + i / 256`.
    pub(crate) fn device_nodes(&self) -> Vec<[f32; 4]> {
        self.nodes
            .iter()
            .zip(&self.ramp)
            .map(|(q, r)| [q[0] as f32, q[1] as f32, r[0] as f32, r[1] as f32])
            .collect()
    }

    /// `Q(z)`: 1 far below 0, 0 far above.
    #[inline]
    pub(crate) fn q(&self, z: f64) -> f64 {
        if z >= TAIL_Z_MAX {
            return 0.0;
        }
        if z <= -TAIL_Z_MAX {
            return 1.0;
        }
        hermite(&self.nodes, z)
    }

    /// The ramp `max(z, 0)` smoothed by the standard normal, `z (1 - Q(z)) + phi(z)`: 0 far below 0, `z` far
    /// above.
    #[inline]
    pub(crate) fn ramp(&self, z: f64) -> f64 {
        if z >= TAIL_Z_MAX {
            return z;
        }
        if z <= -TAIL_Z_MAX {
            return 0.0;
        }
        hermite(&self.ramp, z)
    }
}

/// The cubic Hermite interpolation of a table of `(value, derivative)` nodes at `z` inside the table.
#[inline]
fn hermite(nodes: &[[f64; 2]], z: f64) -> f64 {
    let p = (z + TAIL_Z_MAX) * TAIL_PER_UNIT;
    let i = p as usize;
    let t = p - i as f64;
    let (a, b) = (nodes[i], nodes[i + 1]);
    let h = 1.0 / TAIL_PER_UNIT;
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * a[0]
        + (t3 - 2.0 * t2 + t) * h * a[1]
        + (-2.0 * t3 + 3.0 * t2) * b[0]
        + (t3 - t2) * h * b[1]
}

/// The shared normal-tail table, built on first use.
pub(crate) fn normal_tail() -> &'static NormalTail {
    static TABLE: OnceLock<NormalTail> = OnceLock::new();
    TABLE.get_or_init(NormalTail::build)
}

/// A ball's (here a ghost's) brightness against its normalised radius: the 16 samples of `bokeh.rim` (at
/// `i / 15`) joined by a cubic spline with a flat slope at the centre (no cone there) and a free end at the rim.
/// It is smooth to the second derivative, so a pixel grid samples it without aliasing; it is held at its ends
/// and never negative.
#[derive(Clone, Debug)]
pub(crate) struct RimCurve {
    y: Vec<f64>,
    /// The spline's second derivatives at the samples.
    m: Vec<f64>,
}

impl RimCurve {
    /// The curve of a lens's `bokeh.rim` (any other length is read as a flat profile).
    pub(crate) fn new(rim: &[f64]) -> Self {
        if rim.len() != RIM_SAMPLES {
            return RimCurve {
                y: vec![1.0; RIM_SAMPLES],
                m: vec![0.0; RIM_SAMPLES],
            };
        }
        let n = RIM_SAMPLES;
        let h = 1.0 / (n - 1) as f64;
        // Tridiagonal system for the second derivatives: slope 0 at the first sample, M = 0 at the last.
        let mut diag = vec![2.0 * h / 3.0; n];
        let mut up = vec![h / 6.0; n];
        let mut low = vec![h / 6.0; n];
        let mut rhs = vec![0.0; n];
        diag[0] = h / 3.0;
        rhs[0] = (rim[1] - rim[0]) / h;
        for i in 1..n - 1 {
            rhs[i] = (rim[i + 1] - 2.0 * rim[i] + rim[i - 1]) / h;
        }
        diag[n - 1] = 1.0;
        low[n - 1] = 0.0;
        rhs[n - 1] = 0.0;
        up[n - 1] = 0.0;
        // Thomas algorithm.
        for i in 1..n {
            let w = low[i] / diag[i - 1];
            diag[i] -= w * up[i - 1];
            rhs[i] -= w * rhs[i - 1];
        }
        let mut m = vec![0.0; n];
        m[n - 1] = rhs[n - 1] / diag[n - 1];
        for i in (0..n - 1).rev() {
            m[i] = (rhs[i] - up[i] * m[i + 1]) / diag[i];
        }
        RimCurve { y: rim.to_vec(), m }
    }

    /// The samples and the spline's second derivatives, for the device.
    pub(crate) fn device_tables(&self) -> (&[f64], &[f64]) {
        (&self.y, &self.m)
    }

    /// The brightness at the normalised radius `t` (held at 0 and 1 beyond them).
    #[inline]
    pub(crate) fn at(&self, t: f64) -> f64 {
        let last = RIM_SAMPLES - 1;
        let h = 1.0 / last as f64;
        let x = t.clamp(0.0, 1.0) * last as f64;
        let i = (x as usize).min(last - 1);
        let u = x - i as f64;
        let (a, b) = (1.0 - u, u);
        let v = a * self.y[i]
            + b * self.y[i + 1]
            + ((a * a * a - a) * self.m[i] + (b * b * b - b) * self.m[i + 1]) * (h * h / 6.0);
        v.max(0.0)
    }
}

/// Nodes and weights of the 8-point Gauss-Legendre rule on `[-1, 1]` (positive half; the rule is symmetric).
const GL8: [(f64, f64); 4] = [
    (0.1834346424956498, 0.362683783378362),
    (0.525532409916329, 0.3137066458778873),
    (0.7966664774136267, 0.2223810344533745),
    (0.9602898564975363, 0.1012285362903763),
];

/// `integral of f over [a, b]` by the 8-point Gauss-Legendre rule.
pub(crate) fn gauss8(a: f64, b: f64, f: &mut impl FnMut(f64) -> f64) -> f64 {
    let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
    let mut s = 0.0;
    for (x, w) in GL8 {
        s += w * (f(mid - half * x) + f(mid + half * x));
    }
    s * half
}

/// `1` up to `start`, `0` from `1`, a cosine in between (`x` is already divided by the length).
#[inline]
pub(crate) fn cosine_window(x: f64, start: f64) -> f64 {
    if x <= start {
        1.0
    } else if x >= 1.0 {
        0.0
    } else {
        0.5 * (1.0 + libm::cos(std::f64::consts::PI * (x - start) / (1.0 - start)))
    }
}

/// `3 t^2 - 2 t^3` on `[0, 1]`, 0 below and 1 above.
#[inline]
pub(crate) fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tail_table_matches_erfc() {
        let t = normal_tail();
        let mut worst = 0.0f64;
        for i in 0..4000 {
            let z = -7.9 + 15.8 * f64::from(i) / 4000.0 + 1e-3;
            let want = 0.5 * libm::erfc(z / SQRT_2);
            worst = worst.max((t.q(z) - want).abs());
        }
        assert!(worst < 1e-11, "worst error {worst}");
        assert_eq!(t.q(-9.0), 1.0);
        assert_eq!(t.q(9.0), 0.0);
        assert!((t.q(0.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn gauss8_integrates_polynomials_exactly() {
        let v = gauss8(0.0, 2.0, &mut |x| x.powi(5) - 2.0 * x + 1.0);
        let want = 2.0f64.powi(6) / 6.0 - 4.0 + 2.0;
        assert!((v - want).abs() < 1e-13, "{v} vs {want}");
    }

    #[test]
    fn the_rim_curve_passes_through_its_samples_and_is_flat_at_the_centre() {
        let rim: Vec<f64> = (0..RIM_SAMPLES)
            .map(|i| 0.5 + (i as f64 * 0.1).powi(2))
            .collect();
        let curve = RimCurve::new(&rim);
        for (k, want) in rim.iter().enumerate() {
            let t = k as f64 / (RIM_SAMPLES - 1) as f64;
            assert!((curve.at(t) - want).abs() < 1e-12, "sample {k}");
        }
        // No slope at the centre, held beyond the ends.
        let e = 1e-6;
        assert!((curve.at(e) - curve.at(0.0)).abs() < 1e-9);
        assert_eq!(curve.at(-1.0), curve.at(0.0));
        assert_eq!(curve.at(2.0), curve.at(1.0));
        // A flat rim is flat.
        let flat = RimCurve::new(&[1.0; RIM_SAMPLES]);
        for i in 0..=100 {
            assert!((flat.at(f64::from(i) / 100.0) - 1.0).abs() < 1e-15);
        }
        // Smooth between the samples: the second difference of a smooth rim is small and has no jumps.
        let smooth: Vec<f64> = (0..RIM_SAMPLES)
            .map(|i| 1.0 + 0.3 * (i as f64 / 15.0).powi(4))
            .collect();
        let c = RimCurve::new(&smooth);
        let d2 = |t: f64| (c.at(t + 1e-3) - 2.0 * c.at(t) + c.at(t - 1e-3)) / 1e-6;
        let before = d2(1.0 / 15.0 - 1e-4);
        let after = d2(1.0 / 15.0 + 1e-4);
        assert!((before - after).abs() < 1e-3, "{before} vs {after}");
    }

    #[test]
    fn the_smoothed_ramp_matches_its_definition() {
        let t = normal_tail();
        for i in 0..2000 {
            let z = -7.9 + 15.8 * f64::from(i) / 2000.0 + 1e-3;
            let want = z * 0.5 * libm::erfc(-z / SQRT_2) + INV_SQRT_2PI * libm::exp(-0.5 * z * z);
            assert!((t.ramp(z) - want).abs() < 1e-10, "z = {z}");
        }
        assert_eq!(t.ramp(-9.0), 0.0);
        assert_eq!(t.ramp(9.0), 9.0);
    }

    #[test]
    fn windows_are_smooth_and_bounded() {
        assert_eq!(cosine_window(0.5, 0.7), 1.0);
        assert_eq!(cosine_window(1.0, 0.7), 0.0);
        let mid = cosine_window(0.85, 0.7);
        assert!((mid - 0.5).abs() < 1e-12);
        assert_eq!(smoothstep(-1.0), 0.0);
        assert_eq!(smoothstep(2.0), 1.0);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-15);
    }
}
