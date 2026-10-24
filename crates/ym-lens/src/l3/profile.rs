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
}

impl NormalTail {
    fn build() -> Self {
        let n = (2.0 * TAIL_Z_MAX * TAIL_PER_UNIT) as usize;
        let nodes = (0..=n)
            .map(|i| {
                let z = -TAIL_Z_MAX + i as f64 / TAIL_PER_UNIT;
                [
                    0.5 * libm::erfc(z / SQRT_2),
                    -INV_SQRT_2PI * libm::exp(-0.5 * z * z),
                ]
            })
            .collect();
        NormalTail { nodes }
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
        let p = (z + TAIL_Z_MAX) * TAIL_PER_UNIT;
        let i = p as usize;
        let t = p - i as f64;
        let (a, b) = (self.nodes[i], self.nodes[i + 1]);
        let h = 1.0 / TAIL_PER_UNIT;
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * a[0]
            + (t3 - 2.0 * t2 + t) * h * a[1]
            + (-2.0 * t3 + 3.0 * t2) * b[0]
            + (t3 - t2) * h * b[1]
    }
}

/// The shared normal-tail table, built on first use.
pub(crate) fn normal_tail() -> &'static NormalTail {
    static TABLE: OnceLock<NormalTail> = OnceLock::new();
    TABLE.get_or_init(NormalTail::build)
}

/// Segments of the rim lookup.
const RIM_LUT: usize = 256;

/// A ball's (here a ghost's) brightness against its normalised radius: the 16 samples of `bokeh.rim` joined by a
/// Catmull-Rom curve (so there are no kinks to alias) and tabulated at 256 segments.
#[derive(Clone, Debug)]
pub(crate) struct RimLut {
    v: Vec<f64>,
}

impl RimLut {
    /// The lookup of a lens's `bokeh.rim` (any other length is read as a flat profile).
    pub(crate) fn new(rim: &[f64]) -> Self {
        if rim.len() != RIM_SAMPLES {
            return RimLut {
                v: vec![1.0; RIM_LUT + 1],
            };
        }
        let last = RIM_SAMPLES - 1;
        let v = (0..=RIM_LUT)
            .map(|j| {
                let p = j as f64 / RIM_LUT as f64 * last as f64;
                let seg = (p as usize).min(last - 1);
                let u = p - seg as f64;
                // Mirrored at the centre (no cone there), held at the rim.
                let at = |k: isize| rim[k.abs().min(last as isize) as usize];
                let (p0, p1, p2, p3) = (
                    at(seg as isize - 1),
                    at(seg as isize),
                    at(seg as isize + 1),
                    at(seg as isize + 2),
                );
                let u2 = u * u;
                let u3 = u2 * u;
                let y = 0.5
                    * (2.0 * p1
                        + (p2 - p0) * u
                        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                        + (3.0 * p1 - p0 - 3.0 * p2 + p3) * u3);
                y.max(0.0)
            })
            .collect();
        RimLut { v }
    }

    /// The brightness at the normalised radius `t` (held at the ends beyond 0 and 1).
    #[inline]
    pub(crate) fn at(&self, t: f64) -> f64 {
        let x = t.clamp(0.0, 1.0) * RIM_LUT as f64;
        let i = (x as usize).min(RIM_LUT - 1);
        let f = x - i as f64;
        self.v[i] + f * (self.v[i + 1] - self.v[i])
    }
}

/// Nodes and weights of the 8-point Gauss-Legendre rule on `[-1, 1]` (positive half; the rule is symmetric).
const GL8: [(f64, f64); 4] = [
    (0.183_434_642_495_649_8, 0.362_683_783_378_362_0),
    (0.525_532_409_916_329_0, 0.313_706_645_877_887_3),
    (0.796_666_477_413_626_7, 0.222_381_034_453_374_5),
    (0.960_289_856_497_536_3, 0.101_228_536_290_376_3),
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
    fn the_rim_lookup_passes_through_its_samples() {
        let rim: Vec<f64> = (0..RIM_SAMPLES).map(|i| 0.5 + i as f64 * 0.1).collect();
        let lut = RimLut::new(&rim);
        for (k, want) in rim.iter().enumerate() {
            let t = k as f64 / (RIM_SAMPLES - 1) as f64;
            assert!((lut.at(t) - want).abs() < 1e-9, "sample {k}");
        }
        assert_eq!(lut.at(-1.0), lut.at(0.0));
        assert_eq!(lut.at(2.0), lut.at(1.0));
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
